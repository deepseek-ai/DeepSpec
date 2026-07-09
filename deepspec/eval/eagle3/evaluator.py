from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import DynamicCache

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    VerificationResult,
    assert_no_final_target_layer,
    generate_decoding_sample,
    has_stop_token,
)
from deepspec.modeling.eagle3 import extract_eagle3_context_feature
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model
from deepspec.modeling.eagle3.ministral3 import Ministral3Eagle3Model
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.modeling.target_utils import load_target_causal_lm, load_target_tokenizer
from deepspec.utils.sampling import logits_to_probs, sample_tokens


class Qwen3Eagle3Evaluator(BaseEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    draft_model_cls = Qwen3Eagle3Model

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        # _update extends the draft cache with multiple committed tokens
        # without a causal mask. That is only safe when the draft head has a
        # single layer, so cached K/V are projected directly from per-token
        # inputs instead of from the previous layer's bidirectionally-mixed
        # attention output.
        draft_num_hidden_layers = int(self.draft_model.config.draft_num_hidden_layers)
        assert draft_num_hidden_layers == 1, (
            f"{self.__class__.__name__} requires draft_num_hidden_layers == 1, "
            f"got {draft_num_hidden_layers}."
        )

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.ttt_length)

    def build_models(self) -> tuple[object, Qwen3Eagle3Model, object]:
        target_model = load_target_causal_lm(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        draft_model.target_layer_ids = [int(x) for x in draft_model.target_layer_ids]
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)

        tokenizer = load_target_tokenizer(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer

    def _init_context(
        self,
        *,
        initial_output,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_input_tokens: int,
    ) -> SimpleNamespace:
        # Training pairs target hidden state i with token i + 1, while the
        # draft RoPE position stays at i.  Keep the same convention here when
        # pre-filling the draft cache from prompt hidden states.
        target_hidden = extract_eagle3_context_feature(
            initial_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        shifted_prompt_ids = torch.cat(
            [
                output_ids[:, 1:num_input_tokens],
                output_ids[:, num_input_tokens : num_input_tokens + 1],
            ],
            dim=1,
        )
        draft_cache = DynamicCache()
        draft_hidden = self.draft_model.extend_draft_cache(
            hidden_states=target_hidden,
            input_ids=shifted_prompt_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=draft_cache,
        )
        return SimpleNamespace(
            draft_cache=draft_cache,
            draft_hidden=draft_hidden,
            position_ids=position_ids,
            current_pos=num_input_tokens,
            cache_len_before=0,
        )

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        # Adapted from SpecForge/eval/eval.py Eagle3 draft proposal loop.
        # cache_len_before is the draft-cache length *before* this proposal
        # extends it; _update needs it to crop back to that length.
        context.cache_len_before = context.draft_cache.get_seq_length()
        candidate_ids = [output_ids[:, start : start + 1]]
        draft_logits_list = []
        proposal_hidden = context.draft_hidden
        next_position = start

        for _ in range(self.max_proposal_tokens):
            draft_logits = self.draft_model.compute_logits(proposal_hidden)
            draft_logits_list.append(draft_logits)
            next_token = sample_tokens(
                draft_logits,
                temperature=float(self.args.temperature),
            )
            candidate_ids.append(next_token[:, -1:])
            if has_stop_token(next_token, stop_token_ids):
                break
            proposal_hidden = self.draft_model(
                hidden_states=proposal_hidden,
                input_ids=next_token[:, -1:],
                position_ids=context.position_ids[
                    :,
                    next_position : next_position + 1,
                ],
                past_key_values=context.draft_cache,
                use_cache=True,
            )
            next_position += 1

        draft_logits = torch.cat(draft_logits_list, dim=1)
        return DraftProposal(
            draft_token_count=draft_logits.shape[1],
            verify_input_ids=torch.cat(candidate_ids, dim=1),
            draft_probs=logits_to_probs(
                draft_logits,
                float(self.args.temperature),
            ),
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        # Adapted from SpecForge/eval/eval.py Eagle3 draft-cache crop/extend.
        assert verification.committed_tokens is not None
        committed_length = int(verification.committed_tokens.shape[1])
        context.draft_cache.crop(int(context.cache_len_before))
        committed_hidden = extract_eagle3_context_feature(
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )[:, :committed_length, :]
        context.draft_hidden = self.draft_model.extend_draft_cache(
            hidden_states=committed_hidden,
            input_ids=verification.committed_tokens,
            position_ids=context.position_ids[
                :,
                context.current_pos : context.current_pos + committed_length,
            ],
            past_key_values=context.draft_cache,
        )
        context.current_pos += committed_length

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
        )


class Gemma4Eagle3Evaluator(Qwen3Eagle3Evaluator):
    draft_model_cls = Gemma4Eagle3Model


class Ministral3Eagle3Evaluator(Qwen3Eagle3Evaluator):
    draft_model_cls = Ministral3Eagle3Model


__all__ = [
    "Gemma4Eagle3Evaluator",
    "Ministral3Eagle3Evaluator",
    "Qwen3Eagle3Evaluator",
]
