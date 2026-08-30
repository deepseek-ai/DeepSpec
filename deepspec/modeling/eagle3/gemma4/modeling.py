from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb as apply_gemma4_rotary_pos_emb,
)

from deepspec.modeling.eagle3.common import (
    Eagle3ForwardOutput,
    compile_friendly_flex_attention,
    create_eagle3_attention_mask,
    eagle3_prepare_position_ids,
    prepare_4d_causal_attention_mask,
)
from deepspec.utils.sampling import sample_tokens


GEMMA4_EAGLE3_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_warps": 4,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
}


class Gemma4Eagle3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_attention_heads = int(config.num_attention_heads)
        self.head_dim = int(config.global_head_dim)
        self.use_alternative_attention = bool(config.attention_k_eq_v)
        if self.use_alternative_attention:
            self.num_key_value_heads = int(config.num_global_key_value_heads)
        else:
            self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by the Gemma4 key/value head count."
        )
        self.scaling = 1.0
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False

        input_dim = int(config.hidden_size) * 2
        self.q_proj = nn.Linear(
            input_dim,
            self.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.v_proj = None
        if not self.use_alternative_attention:
            self.v_proj = nn.Linear(
                input_dim,
                self.num_key_value_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
        )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_seen_tokens: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        )
        k = self.k_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        if self.use_alternative_attention:
            v = k
        else:
            v = self.v_proj(hidden_states).view(
                bsz,
                q_len,
                self.num_key_value_heads,
                self.head_dim,
            )

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_gemma4_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)
        k = apply_gemma4_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.config._attn_implementation == "flex_attention":
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
                scale=self.scaling,
                kernel_options=GEMMA4_EAGLE3_FLEX_KERNEL_OPTIONS,
            )
        else:
            attn_is_causal = bool(
                kwargs.get(
                    "is_causal",
                    attention_mask is None
                    and q_len > 1
                    and int(past_seen_tokens) == 0,
                )
            )
            self.is_causal = attn_is_causal
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=attn_is_causal,
                scale=self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), None


class Gemma4Eagle3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        assert not bool(config.enable_moe_block), (
            "Gemma4 Eagle3 prototype does not support Gemma4 MoE blocks yet."
        )
        assert int(config.hidden_size_per_layer_input) == 0, (
            "Gemma4 Eagle3 prototype does not support per-layer input gates yet."
        )
        self.self_attn = Gemma4Eagle3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.hidden_norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.register_buffer("layer_scalar", torch.ones(1))

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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_seen_tokens: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, output_attentions, use_cache
        assert position_embeddings is not None, "position_embeddings must be provided."
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_seen_tokens=past_seen_tokens,
            **kwargs,
        )[0]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar


class Gemma4Eagle3Model(Gemma4PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4Eagle3DecoderLayer"]
    _supports_flex_attn = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "ttt_length",
            "step_loss_decay",
            "num_global_key_value_heads",
            "global_head_dim",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        self.target_layer_ids = [int(x) for x in config.target_layer_ids]
        self.ttt_length = int(config.ttt_length)
        self.step_loss_decay = float(config.step_loss_decay)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
            embed_scale=float(config.hidden_size) ** 0.5,
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4Eagle3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(
            config,
            layer_type="full_attention",
        )
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
        assert hidden_states.size(-1) == (
            len(self.target_layer_ids) * self.config.hidden_size
        )
        return self.fc(hidden_states)

    def _softcap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is None:
            return logits
        softcap = float(softcap)
        assert softcap > 0.0, (
            "config.final_logit_softcapping must be positive when provided."
        )
        return torch.tanh(logits / softcap) * softcap

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._softcap_logits(self.lm_head(self.norm(hidden_states)))

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
        logit_indices: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor | Eagle3ForwardOutput:
        if target_logits_only:
            assert target_last_hidden_states is not None
            with torch.no_grad():
                logits = self.lm_head(target_last_hidden_states)
                return self._softcap_logits(logits)

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
        position_embeddings = self.rotary_emb(
            hidden_states,
            rope_position_ids,
            layer_type="full_attention",
        )

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
            logit_hidden_states = hidden_states
            if logit_indices is not None:
                hidden_size = int(hidden_states.shape[-1])
                logit_hidden_states = hidden_states.reshape(
                    -1, hidden_size
                ).index_select(0, logit_indices)
                logit_hidden_states = logit_hidden_states.unsqueeze(0)
            draft_logits = self.compute_logits(logit_hidden_states)
            target_logits = None
            if target_last_hidden_states is not None:
                with torch.no_grad():
                    target_logits = self._softcap_logits(
                        self.lm_head(target_last_hidden_states)
                    )
            return Eagle3ForwardOutput(
                hidden_states=hidden_states,
                draft_logits=draft_logits,
                target_logits=target_logits,
            )
        return hidden_states


__all__ = [
    "Gemma4Eagle3Model",
    "Gemma4Eagle3Attention",
    "Gemma4Eagle3DecoderLayer",
]
