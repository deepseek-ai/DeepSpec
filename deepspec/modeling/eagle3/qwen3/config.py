import copy

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(*, target_config, model_args):
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,
        int(target_config.num_hidden_layers),
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

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3Eagle3Model"]
    draft_config.num_target_layers = int(target_config.num_hidden_layers)
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
]
