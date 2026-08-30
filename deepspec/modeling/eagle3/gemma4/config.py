import copy

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def get_gemma4_text_config(target_config):
    assert target_config.model_type in ("gemma4", "gemma4_unified"), (
        "Gemma4 Eagle3 expects a Gemma4 or Gemma4 Unified top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type in ("gemma4_text", "gemma4_unified_text"), (
        "Gemma4 Eagle3 expects target_config.text_config.model_type to be "
        f"'gemma4_text' or 'gemma4_unified_text', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)


def _validate_required_text_fields(text_config) -> None:
    required_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_global_key_value_heads",
        "global_head_dim",
        "attention_bias",
        "attention_dropout",
        "attention_k_eq_v",
        "enable_moe_block",
        "head_dim",
        "hidden_activation",
        "hidden_size_per_layer_input",
        "initializer_range",
        "max_position_embeddings",
        "num_key_value_heads",
        "num_kv_shared_layers",
        "rms_norm_eps",
        "rope_parameters",
        "use_double_wide_mlp",
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target_config.text_config.{field} must be provided."
        )


def build_draft_config(*, target_config, model_args):
    draft_config = get_gemma4_text_config(target_config)
    _validate_required_text_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    ttt_length = int(model_args.ttt_length)
    assert ttt_length >= 1, f"ttt_length must be >= 1, got {ttt_length}"
    step_loss_decay = float(model_args.step_loss_decay)
    assert step_loss_decay > 0.0, (
        "step_loss_decay must be > 0.0, "
        f"got {step_loss_decay}"
    )
    draft_num_hidden_layers = int(model_args.draft_num_hidden_layers)
    assert draft_num_hidden_layers >= 1, (
        "draft_num_hidden_layers must be >= 1, "
        f"got {draft_num_hidden_layers}"
    )

    draft_config.architectures = ["Gemma4Eagle3Model"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.target_text_model_type = str(draft_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = draft_num_hidden_layers
    draft_config.layer_types = ["full_attention"] * draft_num_hidden_layers
    draft_config.target_model_name_or_path = str(model_args.target_model_name_or_path)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.ttt_length = ttt_length
    draft_config.step_loss_decay = step_loss_decay
    draft_config.trim_loss_positions = bool(
        getattr(model_args, "trim_loss_positions", False)
    )
    draft_config.draft_num_hidden_layers = draft_num_hidden_layers
    draft_config.tie_word_embeddings = False
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    return draft_config


__all__ = [
    "build_draft_config",
    "get_gemma4_text_config",
]
