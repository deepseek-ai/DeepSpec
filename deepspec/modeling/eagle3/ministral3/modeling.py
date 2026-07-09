from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from transformers.cache_utils import Cache
from transformers.models.ministral3.modeling_ministral3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Ministral3MLP,
    Ministral3PreTrainedModel,
    Ministral3RMSNorm,
    Ministral3RotaryEmbedding,
    eager_attention_forward,
    get_llama_4_attn_scale,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from deepspec.modeling.eagle3.common import (
    Eagle3ForwardOutput,
    compile_friendly_flex_attention,
    create_eagle3_attention_mask,
    eagle3_prepare_position_ids,
    prepare_4d_causal_attention_mask,
)
from deepspec.utils.sampling import sample_tokens


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Ministral3Eagle3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        input_dim = int(config.hidden_size) * 2
        self.q_proj = nn.Linear(
            input_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            input_dim,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            input_dim,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_seen_tokens: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del cache_position
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q * get_llama_4_attn_scale(
            position_ids,
            self.config.rope_parameters.get("llama_4_scaling_beta"),
            self.config.rope_parameters.get("original_max_position_embeddings"),
        ).to(q.dtype)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.config._attn_implementation == "flex_attention":
            # Direct flex_attention dispatch follows
            # SpecForge/specforge/modeling/draft/llama3_eagle.py.
            assert attention_mask is not None, (
                "Eagle3 flex_attention expects a BlockMask attention_mask."
            )
            flex_attention_func = (
                flex_attention
                if int(q_len) <= 128
                else compile_friendly_flex_attention
            )
            attn_output = flex_attention_func(
                query=q,
                key=k.contiguous(),
                value=v.contiguous(),
                block_mask=attention_mask,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            return self.o_proj(attn_output), None

        attn_fn: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )

        attn_is_causal = bool(
            kwargs.get(
                "is_causal",
                attention_mask is None and q_len > 1 and int(past_seen_tokens) == 0,
            )
        )
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class Ministral3Eagle3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Ministral3Eagle3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config)
        self.hidden_norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Ministral3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_seen_tokens: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        del output_attentions, use_cache
        assert position_ids is not None, "position_ids must be provided."
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_seen_tokens=past_seen_tokens,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Ministral3Eagle3Model(Ministral3PreTrainedModel):
    # Architecture adapted from
    # SpecForge/specforge/modeling/draft/llama3_eagle.py and
    # SpecForge/eval/model/eagle3.py.
    _no_split_modules = ["Ministral3Eagle3DecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "ttt_length",
            "step_loss_decay",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.ttt_length = int(config.ttt_length)
        self.step_loss_decay = float(config.step_loss_decay)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                Ministral3Eagle3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == len(self.target_layer_ids) * self.config.hidden_size
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def draft_sample(self, logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        return sample_tokens(logits, temperature=temperature)

    def _prepare_attention_mask(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        q_len: int,
        past_seen_tokens: int,
    ):
        if attention_mask is None or attention_mask.ndim != 2:
            return attention_mask
        kv_len = int(past_seen_tokens) + int(q_len)
        if self.config._attn_implementation == "flex_attention":
            assert int(past_seen_tokens) % int(q_len) == 0, (
                "Eagle3 flex_attention expects fixed-size TTT chunks: "
                f"past_seen_tokens={past_seen_tokens}, q_len={q_len}"
            )
            lck = int(past_seen_tokens) // int(q_len)
            return create_eagle3_attention_mask(
                attention_mask=attention_mask,
                q_len=q_len,
                kv_len=kv_len,
                lck=lck,
                device=hidden_states.device,
            )
        return prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
            q_len=q_len,
            kv_len=kv_len,
            past_seen_tokens=int(past_seen_tokens),
            device=hidden_states.device,
        )

    def extend_draft_cache(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
    ) -> torch.Tensor:
        assert input_ids.shape[1] > 0, "input_ids must contain at least one token."
        output = self(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return output[:, -1:, :]

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        target_logits_only: bool = False,
        return_logits: bool = False,
        rope_cache_step_offset: bool = False,
        **kwargs,
    ) -> torch.Tensor | Eagle3ForwardOutput:
        if target_logits_only:
            assert target_last_hidden_states is not None
            with torch.no_grad():
                return self.lm_head(target_last_hidden_states)

        assert hidden_states is not None, "hidden_states must be provided."
        if hidden_states.size(-1) == len(self.target_layer_ids) * self.config.hidden_size:
            hidden_states = self.project_hidden_states(hidden_states)

        if input_embeds is None:
            assert input_ids is not None, "Either input_ids or input_embeds must be provided."
            input_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = eagle3_prepare_position_ids(
                input_ids=input_ids,
                input_embeds=input_embeds,
            )

        q_len = int(hidden_states.shape[1])
        past_seen_tokens = (
            int(past_key_values.get_seq_length())
            if past_key_values is not None
            else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_len,
            device=hidden_states.device,
        )
        prepared_attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            q_len=q_len,
            past_seen_tokens=past_seen_tokens,
        )
        rope_position_ids = position_ids
        if rope_cache_step_offset:
            assert int(past_seen_tokens) % int(q_len) == 0, (
                "SpecForge-style Eagle3 RoPE offset expects fixed-size TTT chunks: "
                f"past_seen_tokens={past_seen_tokens}, q_len={q_len}"
            )
            rope_position_ids = position_ids + int(past_seen_tokens) // int(q_len)
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        for layer in self.layers:
            hidden_states = layer(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                attention_mask=prepared_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                past_seen_tokens=past_seen_tokens,
                **kwargs,
            )
        if return_logits:
            draft_logits = self.compute_logits(hidden_states)
            target_logits = None
            if target_last_hidden_states is not None:
                with torch.no_grad():
                    target_logits = self.lm_head(target_last_hidden_states)
            return Eagle3ForwardOutput(
                hidden_states=hidden_states,
                draft_logits=draft_logits,
                target_logits=target_logits,
            )
        return hidden_states


__all__ = [
    "Ministral3Eagle3Model",
    "Ministral3Eagle3Attention",
    "Ministral3Eagle3DecoderLayer",
    "apply_rotary_pos_emb",
]
