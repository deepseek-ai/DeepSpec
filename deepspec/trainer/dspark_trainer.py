from transformers import AutoConfig

from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.ministral3 import Ministral3DSparkModel
from deepspec.modeling.dspark.ministral3.config import (
    build_draft_config as build_ministral3_draft_config,
)
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.modeling.target_utils import (
    get_target_input_embeddings,
    get_target_output_embeddings,
    load_target_causal_lm,
    load_target_tokenizer,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class Ministral3DSparkTrainer(Qwen3DSparkTrainer):
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
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )

        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_ministral3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Ministral3DSparkModel(draft_config)
