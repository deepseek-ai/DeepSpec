from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
)


def is_mistral3_target(target_model_or_config) -> bool:
    if hasattr(target_model_or_config, "config"):
        model_type = target_model_or_config.config.model_type
    else:
        model_type = target_model_or_config.model_type
    return str(model_type) == "mistral3"


def get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(
            target_model.model,
            "language_model",
        ):
            return target_model.model.language_model
        assert False, "Gemma4 target model must expose a text language_model."
    if model_type == "mistral3":
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(
            target_model.model,
            "language_model",
        ):
            return target_model.model.language_model
        assert False, "Mistral3 target model must expose a text language_model."
    return getattr(target_model, "model", target_model)


def get_target_hidden_size(target_model) -> int:
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified", "mistral3"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def get_target_input_embeddings(target_model):
    if str(target_model.config.model_type) == "mistral3":
        return get_target_backbone(target_model).embed_tokens
    return target_model.get_input_embeddings()


def get_target_output_embeddings(target_model):
    return target_model.get_output_embeddings()


def load_target_causal_lm(model_name_or_path: str, **kwargs):
    target_config = AutoConfig.from_pretrained(model_name_or_path)
    if str(target_config.model_type) == "mistral3":
        return Mistral3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def load_target_tokenizer(model_name_or_path: str, **kwargs):
    target_config = AutoConfig.from_pretrained(model_name_or_path)
    if str(target_config.model_type) == "mistral3":
        kwargs.setdefault("fix_mistral_regex", True)
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
