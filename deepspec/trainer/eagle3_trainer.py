from transformers import AutoConfig, AutoTokenizer

from deepspec.data import CacheCollator
from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model
from deepspec.modeling.eagle3.gemma4.config import (
    build_draft_config as build_gemma4_eagle3_config,
)
from deepspec.modeling.eagle3.loss import compute_eagle3_loss
from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
from deepspec.modeling.eagle3.qwen3.config import (
    build_draft_config as build_qwen3_eagle3_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3Eagle3Trainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
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

        # The draft head and norm stay frozen / target-independent to match
        # the DSpark setup: head is not trained and norm is not inherited.
        self._initialize_embeddings_and_head_from_target(draft_model, model_args)

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
        )


class Gemma4Eagle3Trainer(Qwen3Eagle3Trainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_eagle3_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4Eagle3Model(draft_config)
