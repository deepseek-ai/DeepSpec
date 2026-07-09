from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from transformers.utils import is_torchdynamo_compiling


@dataclass
class Eagle3ForwardOutput:
    hidden_states: torch.Tensor
    draft_logits: torch.Tensor
    target_logits: Optional[torch.Tensor] = None


def validate_eagle3_target_layer_ids(layer_ids, num_target_layers: int):
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    assert len(layer_ids) == 5, (
        "Eagle3 v1 expects exactly 5 target layers, "
        f"got {len(layer_ids)}: {layer_ids}"
    )
    previous = None
    for layer_id in layer_ids:
        assert 0 <= layer_id < int(num_target_layers), (
            f"target_layer_id {layer_id} is out of range [0, {num_target_layers - 1}]"
        )
        assert previous is None or layer_id > previous, (
            "target_layer_ids must be strictly increasing."
        )
        previous = layer_id
    return layer_ids


def extract_eagle3_context_feature(hidden_states, layer_ids):
    # Eagle3 v1 only consumes target decoder layers. DSpark supports -1 for
    # embeddings, but that is intentionally out of scope here.
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def configure_eagle3_flex_compile():
    if dynamo.config.recompile_limit < 64:
        dynamo.config.recompile_limit = 64


_COMPILED_FLEX_ATTENTION = None
_COMPILED_CREATE_BLOCK_MASK = None


# Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
# compile_friendly_flex_attention and compile_friendly_create_block_mask.
@torch.compiler.disable(recursive=False)
def get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        configure_eagle3_flex_compile()
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
    return _COMPILED_FLEX_ATTENTION


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    flex_attention_func = (
        flex_attention if is_torchdynamo_compiling() else get_compiled_flex_attention()
    )
    return flex_attention_func(query, key, value, **kwargs)


@torch.compiler.disable(recursive=False)
def get_compiled_create_block_mask():
    # Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
    # WrappedCreateBlockMask.
    global _COMPILED_CREATE_BLOCK_MASK
    if _COMPILED_CREATE_BLOCK_MASK is None:
        configure_eagle3_flex_compile()
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    create_block_mask_func = (
        create_block_mask
        if is_torchdynamo_compiling()
        else get_compiled_create_block_mask()
    )
    return create_block_mask_func(mask_mod, B, H, Q_LEN, KV_LEN, device)


def create_eagle3_attention_mask(
    *,
    attention_mask: torch.Tensor,
    q_len: int,
    kv_len: int,
    lck: int,
    device: torch.device,
):
    # Adapted from SpecForge/specforge/modeling/draft/flex_attention.py:
    # generate_eagle3_mask.
    seq_lengths = attention_mask.to(device=device).sum(dim=-1).to(torch.long)
    seq_lengths = (seq_lengths - int(lck)).clamp_min(0)

    def eagle3_mask_mod(b, h, q_idx, kv_idx):
        del h
        seq_len = seq_lengths[b]
        in_valid_query = q_idx < seq_len
        causal_mask = (q_idx >= kv_idx) & (kv_idx < seq_len)
        suffix_mask = (
            (kv_idx >= q_len)
            & ((kv_idx % q_len) < seq_len)
            & (((kv_idx - q_idx) % q_len) == 0)
        )
        return in_valid_query & (causal_mask | suffix_mask)

    eagle3_mask_mod.__name__ = f"eagle3_mask_Q_{q_len}_KV_{kv_len}_lck_{lck}"
    create_block_mask_func = (
        create_block_mask if int(q_len) <= 128 else compile_friendly_create_block_mask
    )
    return create_block_mask_func(
        eagle3_mask_mod,
        B=attention_mask.shape[0],
        H=1,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )


def prepare_4d_causal_attention_mask(
    *,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    q_len: int,
    kv_len: int,
    past_seen_tokens: int,
    device: torch.device,
):
    min_value = torch.finfo(dtype).min
    query_positions = torch.arange(
        past_seen_tokens,
        past_seen_tokens + q_len,
        device=device,
    ).view(q_len, 1)
    key_positions = torch.arange(kv_len, device=device).view(1, kv_len)
    causal = torch.where(
        key_positions <= query_positions,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )
    causal = causal.view(1, 1, q_len, kv_len)

    mask_len = int(attention_mask.shape[-1])
    if mask_len < kv_len:
        repeat_count = (kv_len + mask_len - 1) // mask_len
        attention_mask = attention_mask.repeat(1, repeat_count)
    expanded_mask = attention_mask[:, None, None, :kv_len].to(device=device).bool()
    padding = torch.where(
        expanded_mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )
    return causal + padding


def eagle3_prepare_position_ids(
    *,
    input_ids: Optional[torch.Tensor] = None,
    input_embeds: Optional[torch.Tensor] = None,
    past_key_values_length: int = 0,
) -> torch.LongTensor:
    reference = input_ids if input_ids is not None else input_embeds
    assert reference is not None, "input_ids or input_embeds must be provided."
    batch_size, seq_len = reference.shape[:2]
    device = reference.device
    position_ids = torch.arange(
        int(past_key_values_length),
        int(past_key_values_length) + int(seq_len),
        dtype=torch.long,
        device=device,
    )
    return position_ids.unsqueeze(0).expand(batch_size, -1)


__all__ = [
    "Eagle3ForwardOutput",
    "validate_eagle3_target_layer_ids",
    "extract_eagle3_context_feature",
    "configure_eagle3_flex_compile",
    "get_compiled_flex_attention",
    "compile_friendly_flex_attention",
    "get_compiled_create_block_mask",
    "compile_friendly_create_block_mask",
    "create_eagle3_attention_mask",
    "prepare_4d_causal_attention_mask",
    "eagle3_prepare_position_ids",
]
