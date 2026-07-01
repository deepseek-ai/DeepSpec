import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def get_gemma4_text_config(target_config):
    assert target_config.model_type in ("gemma4", "gemma4_unified"), (
        "Gemma4 DSpark expects a Gemma4 or Gemma4 Unified top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type in ("gemma4_text", "gemma4_unified_text"), (
        "Gemma4 DSpark expects target_config.text_config.model_type to be "
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


def build_draft_config(target_config, model_args):
    draft_config = get_gemma4_text_config(target_config)
    _validate_required_text_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config.architectures = ["Gemma4DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.target_text_model_type = str(draft_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = (
        str(model_args["attn_implementation"])
        if "attn_implementation" in model_args
        else TRAIN_ATTN_IMPLEMENTATION
    )
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
    "get_gemma4_text_config",
]
