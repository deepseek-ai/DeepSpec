"""Triton-fused soft cross-entropy for Eagle3 TTT distillation.

Ports SpecForge's ``LogSoftmaxLoss`` from
``SpecForge/specforge/core/loss.py`` (Apache-2.0, itself incorporating
Unsloth and Liger-Kernel ideas). See the NOTICE file at the repository
root for third-party attribution. Masked positions write zero loss, and
``compute_eagle3_loss`` supplies the configured normalizer for each TTT step.

The fused forward/backward keeps only ``logits.detach()`` + two fp32
per-row scalars (m, d) across autograd's save_for_backward, never
materialising the [B, T, V] fp32 log-probs tensor that the naive
PyTorch path retains across TTT steps.
"""

import torch
import triton
import triton.language as tl
from transformers import DynamicCache

from deepspec.utils.metrics import add_metric


def _calculate_settings(n: int):
    # BLOCK_SIZE is the per-iteration chunk; the kernel loops over V in
    # chunks of BLOCK_SIZE, so large vocabularies (e.g. Qwen3 151936) do
    # NOT need BLOCK_SIZE == next_pow2(V). Cap at MAX_FUSED_SIZE to keep
    # register / shared memory usage in bounds.
    MAX_FUSED_SIZE = 131072
    BLOCK_SIZE = min(triton.next_power_of_2(n), MAX_FUSED_SIZE)

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        num_warps //= 2

    return BLOCK_SIZE, num_warps


@torch.no_grad()
def _shift_with_zero_padding(
    tensor: torch.Tensor,
    *,
    left: bool = True,
) -> torch.Tensor:
    zero_padding = torch.zeros_like(tensor[:, -1:])
    if left:
        return torch.cat((zero_padding, tensor[:, :-1]), dim=1)
    return torch.cat((tensor[:, 1:], zero_padding), dim=1)


def _compute_loss_normalizers(
    *,
    position_masks: list[torch.Tensor],
    loss_normalization: str = "local_mean",
) -> list[float]:
    # Eagle3 only supports local_mean: each TTT step divides by the local
    # B*T (per-sequence at local_batch_size=1), which weights every sequence
    # equally. valid_token_mean is intentionally disabled because its global
    # token-count divisor weights all tokens equally, so long sequences
    # dominate the gradient and eval accept length degrades over training.
    assert loss_normalization != "valid_token_mean", (
        "valid_token_mean normalization is disabled; Eagle3 uses local_mean."
    )
    return [
        float(int(position_mask.shape[0]) * int(position_mask.shape[1]))
        for position_mask in position_masks
    ]


@torch.no_grad()
def _build_next_token_position_mask(
    *,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Last valid tokens have no next-token target inside the cached sequence.
    position_mask = loss_mask.to(torch.float32).clone()
    seq_lengths = attention_mask.to(torch.long).sum(dim=-1)
    batch_indices = torch.arange(position_mask.shape[0], device=position_mask.device)
    last_token_indices = (seq_lengths - 1).clamp_min(0)
    position_mask[batch_indices, last_token_indices] = 0
    return position_mask.unsqueeze(-1)


@torch.no_grad()
def _build_padded_next_token_target_probs(
    target_logits: torch.Tensor,
    ttt_length: int,
) -> torch.Tensor:
    # The extra uniform tail keeps each TTT slice the same length.
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    batch_size, _, vocab_size = target_probs.shape
    uniform_tail = target_probs.new_full(
        (batch_size, int(ttt_length) + 1, vocab_size),
        1.0 / float(vocab_size),
    )
    return torch.cat((target_probs[:, 1:, :], uniform_tail), dim=1).detach()


@torch.no_grad()
def _log_eagle3_step_metrics(
    *,
    step_idx: int,
    draft_logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = position_mask.squeeze(-1) > 0
    correct_mask = (draft_logits.argmax(-1) == target_probs.argmax(-1)) & valid_mask
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    accept_rate_mask = 1.0 - 0.5 * (
        draft_probs - target_probs.float()
    ).abs().sum(dim=-1)
    accept_rate_mask = accept_rate_mask.clamp_(0.0, 1.0)
    accept_rate_mask = accept_rate_mask * valid_mask.to(torch.float32)

    correct = correct_mask.to(torch.float32).sum()
    valid_count = valid_mask.to(torch.float32).sum()
    accept_sum = accept_rate_mask.sum()
    add_metric(f"accuracy@{step_idx}", correct, den=valid_count, tag="train")
    add_metric(f"accept_rate@{step_idx}", accept_sum, den=valid_count, tag="train")
    add_metric(
        f"valid_tokens@{step_idx}",
        valid_count,
        reduction="dp_sum",
        tag="train",
    )
    return correct_mask, accept_rate_mask, valid_mask


@torch.no_grad()
def _log_eagle3_prefix_metrics(
    *,
    correct_masks: list[torch.Tensor],
    accept_rate_masks: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
) -> None:
    correct_tensor = torch.stack(
        [correct_mask.to(torch.float32) for correct_mask in correct_masks],
        dim=0,
    )
    start_valid = valid_masks[0].to(torch.float32)
    tau_count = start_valid.sum()
    accepted_draft_tokens = correct_tensor.cumprod(dim=0).sum(dim=0)
    tau_greedy_sum = (accepted_draft_tokens * start_valid).sum() + tau_count

    accept_rate_tensor = torch.stack(accept_rate_masks, dim=0).to(torch.float32)
    expected_draft_tokens = accept_rate_tensor.cumprod(dim=0).sum(dim=0)
    tau_prob_sum = (expected_draft_tokens * start_valid).sum() + tau_count

    add_metric("tau_greedy", tau_greedy_sum, den=tau_count, tag="train")
    add_metric("tau_probabilistic", tau_prob_sum, den=tau_count, tag="train")


def _compute_tv_acceptance_mask(
    *,
    draft_logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    overlap = torch.minimum(draft_probs, target_probs.float()).sum(dim=-1)
    valid_mask = position_mask.squeeze(-1).to(torch.float32)
    return overlap * valid_mask


def _compute_e2e_tv_loss(
    *,
    accept_rate_masks: list[torch.Tensor],
    start_valid_mask: torch.Tensor,
) -> torch.Tensor:
    accept_rate_tensor = torch.stack(accept_rate_masks, dim=0).to(torch.float32)
    prefix_acceptance = accept_rate_tensor.cumprod(dim=0).sum(dim=0)
    normalized_acceptance = prefix_acceptance / float(len(accept_rate_masks))
    per_position_loss = (1.0 - normalized_acceptance) * start_valid_mask
    denominator = start_valid_mask.sum().clamp_min(1.0)
    return per_position_loss.sum() / denominator


@triton.jit
def _log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

    loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        normalized_logits = logits_block - m
        log_normalizer = tl.log(d)
        log_softmax_logits = normalized_logits - log_normalizer
        weighted_log_prob = target_block * log_softmax_logits
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, -loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def _log_softmax_backward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad_output = grad_output * scaling_factor

    target_grad_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_grad_sum += tl.sum(tl.where(mask, target_block * grad_output, 0.0))

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_block * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class FusedLogSoftmaxLoss(torch.autograd.Function):
    """Soft cross-entropy with fused Triton forward/backward.

    Returns ``sum_i(-sum_v target_p[i,v] * log_softmax(logits)[i,v]) /
    normalizer``, where masked positions contribute zero to the numerator.

    Backward writes the gradient in-place into ``logits`` storage via a
    detached view and returns that tensor as the gradient. This is the
    Liger/Unsloth memory-saving pattern: callers must not read ``logits``
    after this loss has run backward.
    """

    @staticmethod
    def forward(ctx, logits, target_p, position_mask, normalizer):
        assert logits.is_cuda, "FusedLogSoftmaxLoss requires CUDA tensors."
        assert logits.shape == target_p.shape
        assert position_mask.shape[:2] == logits.shape[:2]
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device, dtype=torch.float32)
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target_p.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        _log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits.detach(), target_p, position_mask, m, d)
        ctx.normalizer = float(normalizer)
        return loss.squeeze(1).sum() / ctx.normalizer

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / ctx.normalizer
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        _log_softmax_backward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            grad_output,
            scaling_factor,
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return logits_flat.view(B, T, V), None, None, None


def compute_eagle3_loss(
    *,
    model,
    batch: dict[str, torch.Tensor],
    ttt_length: int,
    step_loss_decay: float,
    loss_type: str = "soft_ce",
) -> torch.Tensor:
    input_ids = batch["input_ids"].long()
    attention_mask = batch["attention_mask"].long()
    hidden_states = batch["target_hidden_states"]
    target_last_hidden_states = batch["target_last_hidden_states"]
    seq_len = int(input_ids.shape[1])
    base_position_ids = torch.arange(
        seq_len,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0).expand(input_ids.shape[0], -1)
    current_input_ids = _shift_with_zero_padding(input_ids, left=False)
    shifted_position_mask = _build_next_token_position_mask(
        loss_mask=batch["loss_mask"],
        attention_mask=attention_mask,
    )
    past_key_values = DynamicCache()
    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    with torch.no_grad():
        target_logits = model(
            target_last_hidden_states=target_last_hidden_states,
            target_logits_only=True,
        )
    target_probs = _build_padded_next_token_target_probs(
        target_logits=target_logits,
        ttt_length=int(ttt_length),
    )
    del target_logits
    position_masks = []
    for _ in range(int(ttt_length)):
        position_masks.append(shifted_position_mask)
        shifted_position_mask = _shift_with_zero_padding(
            shifted_position_mask.squeeze(-1),
            left=False,
        ).unsqueeze(-1)
    loss_normalizers = _compute_loss_normalizers(
        position_masks=position_masks,
    )

    correct_masks = []
    accept_rate_masks = []
    tv_accept_rate_masks = []
    valid_masks = []
    loss_type = str(loss_type)
    assert loss_type in ("soft_ce", "e2e_tv"), (
        "loss_type must be one of {'soft_ce', 'e2e_tv'}, "
        f"got {loss_type!r}."
    )
    for step_idx in range(int(ttt_length)):
        # Keep this slice alignment in sync with the Eagle3 reference.
        target_step_probs = target_probs[
            :,
            step_idx : step_idx + seq_len,
            :,
        ].contiguous()
        position_mask_step = position_masks[step_idx]
        output = model(
            hidden_states=hidden_states,
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            position_ids=base_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_logits=True,
            rope_cache_step_offset=True,
        )
        hidden_states = output.hidden_states
        correct_mask, accept_rate_mask, valid_mask = _log_eagle3_step_metrics(
            step_idx=step_idx,
            draft_logits=output.draft_logits,
            target_probs=target_step_probs,
            position_mask=position_mask_step,
        )
        correct_masks.append(correct_mask)
        accept_rate_masks.append(accept_rate_mask)
        valid_masks.append(valid_mask)
        if loss_type == "soft_ce":
            step_loss = FusedLogSoftmaxLoss.apply(
                output.draft_logits,
                target_step_probs,
                position_mask_step,
                loss_normalizers[step_idx],
            )
            add_metric(
                f"ploss_{step_idx}",
                step_loss.detach(),
                reduction="dp_mean",
                tag="train",
            )
            step_weight = float(step_loss_decay) ** step_idx
            total_loss = total_loss + step_loss * step_weight
        else:
            tv_accept_rate_masks.append(
                _compute_tv_acceptance_mask(
                    draft_logits=output.draft_logits,
                    target_probs=target_step_probs,
                    position_mask=position_mask_step,
                )
            )
        current_input_ids = _shift_with_zero_padding(current_input_ids, left=False)

    if loss_type == "e2e_tv":
        start_valid_mask = valid_masks[0].to(torch.float32)
        total_loss = _compute_e2e_tv_loss(
            accept_rate_masks=tv_accept_rate_masks,
            start_valid_mask=start_valid_mask,
        )
        add_metric(
            "e2e_tv_loss",
            total_loss.detach(),
            reduction="dp_mean",
            tag="train",
        )

    _log_eagle3_prefix_metrics(
        correct_masks=correct_masks,
        accept_rate_masks=accept_rate_masks,
        valid_masks=valid_masks,
    )
    add_metric("loss", total_loss.detach(), reduction="dp_mean", tag="train")
    return total_loss


__all__ = [
    "FusedLogSoftmaxLoss",
    "compute_eagle3_loss",
]
