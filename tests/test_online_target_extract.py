"""CPU correctness tests for online target hidden-state extraction.

These validate the core claim behind online training mode: that recomputing
target hidden states at train time with ``run_target_forward_with_hooks``
produces exactly the tensors the offline cache pipeline would have stored.

Runs on CPU with a tiny randomly-initialized Qwen3 model -- no GPU, no
downloads. Run with: ``pytest tests/test_online_target_extract.py``
"""

import importlib.util
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

# Load deepspec/modeling/target_extract.py directly. Importing it via the
# package would execute deepspec/modeling/__init__.py, which eagerly imports
# the Gemma4 modeling classes -- those require a newer transformers than some
# environments have installed. The module under test has no such dependency.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "deepspec",
    "modeling",
    "target_extract.py",
)
_spec = importlib.util.spec_from_file_location("target_extract", _MODULE_PATH)
_target_extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_target_extract)

get_target_backbone = _target_extract.get_target_backbone
run_target_forward_with_hooks = _target_extract.run_target_forward_with_hooks


def _tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    # The helper consumes a *backbone* (returns last_hidden_state), exactly as
    # cache generation loads it via AutoModel and as build_models extracts it
    # via get_target_backbone. Use the inner .model here.
    return Qwen3ForCausalLM(config).eval().model


def _reference_capture(model, input_ids, attention_mask, target_layer_ids):
    """Independent re-implementation of the capture, mirroring cache-gen."""
    backbone = get_target_backbone(model)
    captured = {}
    handles = []

    def hook(layer_id):
        def _hook(_m, _i, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_id] = t.detach()
        return _hook

    for lid in target_layer_ids:
        handles.append(backbone.layers[lid].register_forward_hook(hook(lid)))
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        last = out.last_hidden_state.detach()
        hidden = torch.cat([captured[lid] for lid in target_layer_ids], dim=-1)
    finally:
        for h in handles:
            h.remove()
    return hidden, last


def test_online_extraction_matches_reference():
    model = _tiny_model()
    target_layer_ids = [1, 3, 5]
    input_ids = torch.randint(0, 256, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    result = run_target_forward_with_hooks(
        target_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_layer_ids=target_layer_ids,
    )
    ref_hidden, ref_last = _reference_capture(
        model, input_ids, attention_mask, target_layer_ids
    )

    assert result.target_hidden_states.shape == (2, 16, 32 * len(target_layer_ids))
    assert result.target_last_hidden_states.shape == (2, 16, 32)
    torch.testing.assert_close(result.target_hidden_states, ref_hidden)
    torch.testing.assert_close(result.target_last_hidden_states, ref_last)


def test_padding_does_not_change_real_token_hidden_states():
    """A right-padded batch must yield the same hidden states on the real
    tokens as an unpadded single-sequence forward. This is the padding/masking
    risk flagged during design review."""
    model = _tiny_model()
    target_layer_ids = [1, 3, 5]

    real_len = 10
    seq = torch.randint(1, 256, (1, real_len))

    # Unpadded reference.
    unpadded = run_target_forward_with_hooks(
        target_model=model,
        input_ids=seq,
        attention_mask=torch.ones_like(seq),
        target_layer_ids=target_layer_ids,
    )

    # Right-padded to length 16 with a masked-out pad region.
    pad_len = 16
    padded_ids = torch.zeros((1, pad_len), dtype=torch.long)
    padded_ids[0, :real_len] = seq[0]
    mask = torch.zeros((1, pad_len), dtype=torch.long)
    mask[0, :real_len] = 1
    padded = run_target_forward_with_hooks(
        target_model=model,
        input_ids=padded_ids,
        attention_mask=mask,
        target_layer_ids=target_layer_ids,
    )

    torch.testing.assert_close(
        padded.target_hidden_states[:, :real_len],
        unpadded.target_hidden_states,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        padded.target_last_hidden_states[:, :real_len],
        unpadded.target_last_hidden_states,
        rtol=1e-4,
        atol=1e-4,
    )


def test_target_backbone_unwraps_causal_lm():
    # A ForCausalLM must unwrap to its .model backbone (the layer container).
    full = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()
    backbone = get_target_backbone(full)
    assert backbone is full.model
    assert hasattr(backbone, "layers")
    # And an already-unwrapped backbone is returned as-is.
    assert get_target_backbone(full.model) is full.model
