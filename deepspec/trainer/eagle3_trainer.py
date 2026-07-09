from transformers import AutoConfig

from deepspec.data import CacheCollator
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model
from deepspec.modeling.eagle3.gemma4.config import (
    build_draft_config as build_gemma4_eagle3_config,
)
from deepspec.modeling.eagle3.loss import compute_eagle3_loss
from deepspec.modeling.eagle3.ministral3 import Ministral3Eagle3Model
from deepspec.modeling.eagle3.ministral3.config import (
    build_draft_config as build_ministral3_eagle3_config,
)
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.modeling.eagle3.qwen3.config import (
    build_draft_config as build_qwen3_eagle3_config,
)
from deepspec.modeling.target_utils import (
    get_target_input_embeddings,
    get_target_output_embeddings,
    load_target_causal_lm,
    load_target_tokenizer,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3Eagle3Trainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def build_models(self):
        model_args = self.args.model

        tokenizer = load_target_tokenizer(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        target_model = load_target_causal_lm(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = get_target_input_embeddings(target_model)
        target_lm_head = get_target_output_embeddings(target_model)
        assert (target_lm_head is not None) and (target_embed_tokens is not None)

        # The draft head and norm stay frozen / target-independent to match
        # the DSpark setup: head is not trained and norm is not inherited.
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )

        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3Eagle3Model(draft_config)

    def run_batch(self, batch):
        return compute_eagle3_loss(
            model=self.model,
            batch=batch,
            ttt_length=int(self.draft_model.ttt_length),
            step_loss_decay=float(self.draft_model.step_loss_decay),
            loss_type=str(getattr(self.args.model, "loss_type", "soft_ce")),
        )


class Gemma4Eagle3Trainer(Qwen3Eagle3Trainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4Eagle3Model(draft_config)


class Ministral3Eagle3Trainer(Qwen3Eagle3Trainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_ministral3_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Ministral3Eagle3Model(draft_config)
