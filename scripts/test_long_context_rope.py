"""CPU test that the optional long-context RoPE recipe reaches the drafter's attention.

No GPU needed (a real Qwen3 config is fetched if the Hub is reachable, else a faithful
synthetic one is used). It checks:
  (1) the rope extension (merged into rope_parameters, transformers>=5) propagates through
      build_qwen3_draft_config into the draft config (via deepcopy of target_config);
  (2) the resulting Qwen3RotaryEmbedding applies YaRN: attention_scaling != 1, finite at
      128k positions, and rescaled vs the unscaled rotary.

Usage: python scripts/test_long_context_rope.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoConfig, Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding  # noqa: E402
from deepspec.modeling.dspark.qwen3.config import build_draft_config  # noqa: E402
from deepspec.utils.config import ConfigNode  # noqa: E402

ROPE = dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=32768)


def _model_args(rope_scaling=None, max_position_embeddings=None):
    d = dict(num_draft_layers=2, target_layer_ids=[0], confidence_head_alpha=0.0,
             markov_rank=0, block_size=4, mask_token_id=0, num_anchors=8)
    if rope_scaling is not None:
        d["rope_scaling"] = rope_scaling
    if max_position_embeddings is not None:
        d["max_position_embeddings"] = max_position_embeddings
    return ConfigNode(d)


def _target_config():
    try:
        return AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    except Exception:
        c = Qwen3Config(num_hidden_layers=4, hidden_size=256, num_attention_heads=8,
                        num_key_value_heads=4, head_dim=128, max_position_embeddings=40960,
                        vocab_size=1000)
        c.rope_parameters = {"rope_theta": 1000000, "rope_type": "default"}
        return c


def _apply_rope(cfg, model_args):
    # mirrors deepspec/trainer/base_trainer.build_models
    rs = getattr(model_args, "rope_scaling", None)
    if rs is not None:
        rope_parameters = dict(getattr(cfg, "rope_parameters", None) or {})
        rope_parameters.update(rs)
        cfg.rope_parameters = rope_parameters
    mpe = getattr(model_args, "max_position_embeddings", None)
    if mpe is not None:
        cfg.max_position_embeddings = int(mpe)
    return cfg


def main():
    tgt = _apply_rope(_target_config(), _model_args(ROPE, 131072))
    draft = build_draft_config(target_config=tgt, model_args=_model_args(ROPE, 131072))
    assert draft.rope_parameters.get("rope_type") == "yarn", draft.rope_parameters
    assert draft.rope_parameters.get("factor") == 4.0
    assert "rope_theta" in draft.rope_parameters, "base rope_theta must be preserved"
    assert draft.max_position_embeddings == 131072
    print("(1) rope extension propagates into the draft config:", draft.rope_parameters)

    base = build_draft_config(target_config=_target_config(), model_args=_model_args())
    rb_base = Qwen3RotaryEmbedding(base)
    rb_scaled = Qwen3RotaryEmbedding(draft)
    assert rb_scaled.attention_scaling != rb_base.attention_scaling, (
        rb_base.attention_scaling, rb_scaled.attention_scaling)
    pos = torch.arange(0, 131072, 8192).unsqueeze(0)
    x = torch.zeros(1, pos.shape[1], base.hidden_size)
    cos_b, _ = rb_base(x, pos)
    cos_s, sin_s = rb_scaled(x, pos)
    assert torch.isfinite(cos_s).all() and torch.isfinite(sin_s).all()
    assert not torch.allclose(cos_b, cos_s)
    print(f"(2) YaRN applied to the drafter rotary: attention_scaling "
          f"{rb_base.attention_scaling:.4f} -> {rb_scaled.attention_scaling:.4f}, "
          f"max rope delta at positions up to 128k = {(cos_b - cos_s).abs().max().item():.4f}")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
