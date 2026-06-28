from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    VerificationResult,
    assert_no_final_target_layer,
    generate_decoding_sample,
)
from deepspec.eval.dspark.confidence_head import ConfidenceHeadRecorder
from deepspec.eval.dspark.draft_ops import (
    DSparkDraftProposal,
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.utils import jsonable


CONFIDENCE_NUM_BINS = 20
CONFIDENCE_NUM_FINE_BINS = 1000


class Qwen3DSparkEvaluator(BaseEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    draft_model_cls = Qwen3DSparkModel

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        self.confidence_head_recorder = self._build_confidence_head_recorder()

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.block_size)

    def _build_confidence_head_recorder(self) -> ConfidenceHeadRecorder | None:
        if self.draft_model.confidence_head is None:
            return None
        if float(self.args.confidence_threshold) != 0.0:
            return None

        artifact_root = None
        if self.args.tensorboard_dir is not None:
            artifact_root = (
                Path(self.args.tensorboard_dir)
                / "artifacts"
                / f"step_{self.args.step}"
            )
        return ConfidenceHeadRecorder(
            device=self.device,
            max_proposal_tokens=self.max_proposal_tokens,
            num_bins=CONFIDENCE_NUM_BINS,
            num_fine_bins=CONFIDENCE_NUM_FINE_BINS,
            draft_name_or_path=self.args.draft_name_or_path,
            tensorboard_dir=self.args.tensorboard_dir,
            step=self.args.step,
            artifact_root=artifact_root,
        )

    def build_models(self) -> tuple[object, Qwen3DSparkModel, AutoTokenizer]:
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
        assert 0.0 <= float(self.args.confidence_threshold) <= 1.0
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer

    def _init_context(
        self,
        *,
        initial_output,
        **kwargs,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            past_key_values_draft=DynamicCache(),
            target_hidden_states=extract_context_feature(
                initial_output.hidden_states,
                self.draft_model.target_layer_ids,
            ),
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
        model = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), self.max_proposal_tokens),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=self.max_proposal_tokens,
        )
        return build_dspark_proposal(
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        verified_target_hidden = extract_context_feature(
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        context.target_hidden_states = verified_target_hidden[
            :,
            : verification.accepted_draft_tokens + 1,
            :,
        ]

    def _post_verify(
        self,
        proposal: DraftProposal,
        verification: VerificationResult,
    ) -> None:
        if self.confidence_head_recorder is None:
            return
        assert isinstance(proposal, DSparkDraftProposal)
        self.confidence_head_recorder.observe(
            proposal=proposal,
            verification=verification,
        )

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
            post_verify=self._post_verify,
        )

    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            if self.confidence_head_recorder is not None:
                self.confidence_head_recorder.start()
            dist.barrier()
            start_time = time.perf_counter()
            responses = self.run_dataset(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            dist.barrier()
            elapsed_seconds = time.perf_counter() - start_time
            metric_summary = self.allreduce_response_metrics(responses)
            elapsed_tensor = torch.tensor(elapsed_seconds, device=self.device, dtype=torch.float32)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            metric_summary["elapsed_seconds"] = float(elapsed_tensor.item())
            confidence_row = (
                self.confidence_head_recorder.finish(
                    dataset_name=dataset_name,
                    metric_summary=metric_summary,
                )
                if self.confidence_head_recorder is not None
                else None
            )

            metrics_row = self.record_dataset_metrics(
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )
            if metrics_row is not None and confidence_row is not None:
                self.confidence_head_recorder.report_dataset(
                    metrics_row=metrics_row,
                    confidence_row=confidence_row,
                    args_payload=jsonable(vars(self.args)),
                    tasks=self.tasks,
                )

        self.report_results()

    def log_tensorboard(self) -> None:
        super().log_tensorboard()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.log_tensorboard()

    def print_results(self) -> None:
        super().print_results()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.print_results()


class Gemma4DSparkEvaluator(Qwen3DSparkEvaluator):
    draft_model_cls = Gemma4DSparkModel
