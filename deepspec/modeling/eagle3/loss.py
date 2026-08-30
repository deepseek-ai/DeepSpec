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

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from transformers import DynamicCache

from deepspec.utils.metrics import add_metric


@dataclass(frozen=True)
class Eagle3CompactStep:
    """Row mappings for one A-level-compacted TTT step.

    ``draft_indices`` indexes the full ``[B, T, H]`` draft hidden states after
    the backbone has run. ``teacher_table_indices`` indexes a table containing
    teacher probabilities for unique supervised target rows. Empty steps carry
    one zero-masked dummy row so every rank still executes the same modules.
    """

    draft_indices: torch.Tensor
    teacher_indices: torch.Tensor
    teacher_table_indices: torch.Tensor
    position_mask: torch.Tensor
    scatter_indices: torch.Tensor
    num_valid_rows: int


@dataclass(frozen=True)
class Eagle3CompactPlan:
    """A-level row plan that leaves teacher projection and backbone full-length."""

    teacher_unique_indices: torch.Tensor
    steps: tuple[Eagle3CompactStep, ...]
    batch_size: int
    seq_len: int


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
def _build_compact_plan(
    *,
    position_masks: list[torch.Tensor],
) -> Eagle3CompactPlan:
    """Build exact row mappings for A-level supervised-position compaction.

    At TTT step ``j``, a live draft row ``r`` is supervised by target-model
    logits at row ``r + j + 1``. The full path expresses this with a padded,
    shifted target-probability tensor. This plan represents the same mapping
    directly, without changing the full-length recurrent backbone.
    """

    assert position_masks, "position_masks must not be empty."
    batch_size, seq_len = position_masks[0].shape[:2]
    device = position_masks[0].device
    raw_steps = []
    all_teacher_indices = []

    for step_idx, position_mask in enumerate(position_masks):
        assert tuple(position_mask.shape[:2]) == (batch_size, seq_len)
        flat_mask = position_mask.reshape(-1)
        draft_indices = (flat_mask > 0).nonzero(as_tuple=False).squeeze(-1)
        batch_indices = torch.div(draft_indices, seq_len, rounding_mode="floor")
        row_indices = draft_indices.remainder(seq_len)
        teacher_rows = row_indices + int(step_idx) + 1
        # `_build_next_token_position_mask` clears every sequence's final valid
        # row, and each later mask is shifted left once. Therefore every live
        # row satisfies row + step_idx + 1 < seq_len. Keep this on-device;
        # converting an `.all()` result to bool would synchronize every step.
        teacher_indices = batch_indices * seq_len + teacher_rows
        raw_steps.append((draft_indices, teacher_indices, flat_mask))
        if teacher_indices.numel() > 0:
            all_teacher_indices.append(teacher_indices)

    if all_teacher_indices:
        teacher_unique_indices = torch.unique(
            torch.cat(all_teacher_indices), sorted=True
        )
    else:
        # Keep the full teacher projection unchanged, and select one dummy row
        # after it so empty-loss batches still have a valid compact tensor.
        teacher_unique_indices = torch.zeros(1, dtype=torch.long, device=device)

    steps = []
    dummy_index = torch.zeros(1, dtype=torch.long, device=device)
    for draft_indices, teacher_indices, flat_mask in raw_steps:
        num_valid_rows = int(draft_indices.numel())
        if num_valid_rows == 0:
            padded_draft_indices = dummy_index
            padded_teacher_indices = teacher_unique_indices[:1]
            teacher_table_indices = dummy_index
            compact_position_mask = flat_mask.new_zeros((1, 1, 1))
        else:
            padded_draft_indices = draft_indices
            padded_teacher_indices = teacher_indices
            teacher_table_indices = torch.searchsorted(
                teacher_unique_indices, teacher_indices
            )
            compact_position_mask = flat_mask.index_select(
                0, draft_indices
            ).reshape(1, -1, 1)
        steps.append(
            Eagle3CompactStep(
                draft_indices=padded_draft_indices,
                teacher_indices=padded_teacher_indices,
                teacher_table_indices=teacher_table_indices,
                position_mask=compact_position_mask,
                scatter_indices=draft_indices,
                num_valid_rows=num_valid_rows,
            )
        )

    return Eagle3CompactPlan(
        teacher_unique_indices=teacher_unique_indices,
        steps=tuple(steps),
        batch_size=int(batch_size),
        seq_len=int(seq_len),
    )


@torch.no_grad()
def _build_compact_teacher_probs(
    *,
    target_logits: torch.Tensor,
    compact_plan: Eagle3CompactPlan,
) -> torch.Tensor:
    """Convert only unique supervised teacher-logit rows to fp32 probabilities."""

    vocab_size = int(target_logits.shape[-1])
    target_logits_flat = target_logits.reshape(-1, vocab_size)
    compact_logits = target_logits_flat.index_select(
        0, compact_plan.teacher_unique_indices
    )
    return torch.softmax(compact_logits.float(), dim=-1).detach()


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
    full_shape: tuple[int, int] | None = None,
    scatter_indices: torch.Tensor | None = None,
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
    if scatter_indices is not None:
        assert full_shape is not None
        num_valid_rows = int(scatter_indices.numel())
        full_numel = int(full_shape[0]) * int(full_shape[1])

        def scatter(values: torch.Tensor) -> torch.Tensor:
            output = values.new_zeros(full_numel)
            if num_valid_rows > 0:
                output.index_copy_(
                    0,
                    scatter_indices,
                    values.reshape(-1)[:num_valid_rows],
                )
            return output.reshape(full_shape)

        correct_mask = scatter(correct_mask)
        accept_rate_mask = scatter(accept_rate_mask)
        valid_mask = scatter(valid_mask)
    return correct_mask, accept_rate_mask, valid_mask


@triton.jit
def _indexed_accept_rate_kernel(
    logits_ptr,
    logits_stride,
    target_table_ptr,
    target_stride,
    target_indices_ptr,
    position_mask_ptr,
    accept_rate_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_row = tl.load(target_indices_ptr + program_id).to(tl.int64)
    target_table_ptr += target_row * target_stride
    position_mask = tl.load(position_mask_ptr + program_id)
    if position_mask == 0:
        tl.store(accept_rate_ptr + program_id, 0.0)
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

    l1_distance = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(
            target_table_ptr + offsets, mask=mask, other=0.0
        ).cast(tl.float32)
        draft_prob = tl.exp(logits_block - m) / d
        l1_distance += tl.sum(
            tl.where(mask, tl.abs(draft_prob - target_block), 0.0)
        )

    accept_rate = tl.maximum(0.0, tl.minimum(1.0, 1.0 - 0.5 * l1_distance))
    tl.store(accept_rate_ptr + program_id, accept_rate.to(tl.float32))


@torch.no_grad()
def _log_eagle3_indexed_step_metrics(
    *,
    step_idx: int,
    draft_logits: torch.Tensor,
    target_table: torch.Tensor,
    target_token_ids_table: torch.Tensor,
    target_indices: torch.Tensor,
    position_mask: torch.Tensor,
    full_shape: tuple[int, int],
    scatter_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute compact metrics by reading shared teacher rows through indices."""

    assert draft_logits.is_cuda
    assert draft_logits.shape[0] == 1
    _, num_rows, vocab_size = draft_logits.shape
    assert target_table.ndim == 2 and target_table.shape[1] == vocab_size
    assert target_indices.numel() == num_rows
    valid_mask = position_mask.squeeze(-1) > 0
    target_token_ids = target_token_ids_table.index_select(0, target_indices).view(
        1, num_rows
    )
    correct_mask = (draft_logits.argmax(-1) == target_token_ids) & valid_mask

    accept_rate_mask = torch.empty(
        (num_rows,), device=draft_logits.device, dtype=torch.float32
    )
    logits_flat = draft_logits.contiguous().view(num_rows, vocab_size)
    target_table = target_table.contiguous()
    target_indices = target_indices.contiguous()
    position_mask_flat = position_mask.contiguous().view(num_rows)
    BLOCK_SIZE, num_warps = _calculate_settings(vocab_size)
    _indexed_accept_rate_kernel[(num_rows,)](
        logits_flat,
        logits_flat.stride(0),
        target_table,
        target_table.stride(0),
        target_indices,
        position_mask_flat,
        accept_rate_mask,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    accept_rate_mask = accept_rate_mask.view(1, num_rows)

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

    num_valid_rows = int(scatter_indices.numel())
    full_numel = int(full_shape[0]) * int(full_shape[1])

    def scatter(values: torch.Tensor) -> torch.Tensor:
        output = values.new_zeros(full_numel)
        if num_valid_rows > 0:
            output.index_copy_(
                0,
                scatter_indices,
                values.reshape(-1)[:num_valid_rows],
            )
        return output.reshape(full_shape)

    return scatter(correct_mask), scatter(accept_rate_mask), scatter(valid_mask)


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


@triton.jit
def _log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    target_indices_ptr,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    HAS_TARGET_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_row = program_id
    if HAS_TARGET_INDICES:
        target_row = tl.load(target_indices_ptr + program_id).to(tl.int64)
    target_ptr += target_row * target_stride
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
    target_indices_ptr,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    n_cols,
    HAS_TARGET_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_row = program_id
    if HAS_TARGET_INDICES:
        target_row = tl.load(target_indices_ptr + program_id).to(tl.int64)
    target_ptr += target_row * target_stride
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


def _fused_log_softmax_forward(
    logits,
    target_p,
    position_mask,
    normalizer,
    target_indices=None,
):
    assert logits.is_cuda, "FusedLogSoftmaxLoss requires CUDA tensors."
    assert position_mask.shape[:2] == logits.shape[:2]
    B, T, V = logits.shape
    if target_indices is None:
        assert logits.shape == target_p.shape
        target_indices_ptr = target_p
        has_target_indices = False
    else:
        assert B == 1
        assert target_p.ndim == 2 and target_p.shape[1] == V
        assert target_indices.numel() == B * T
        target_indices_ptr = target_indices.contiguous()
        has_target_indices = True
    loss = torch.zeros((B * T, 1), device=logits.device, dtype=torch.float32)
    logits_flat = logits.contiguous().view(B * T, V)
    target_flat = target_p.contiguous().view(-1, V)
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
        target_indices_ptr,
        position_mask_flat,
        position_mask_flat.stride(0),
        loss,
        loss.stride(0),
        m,
        d,
        V,
        HAS_TARGET_INDICES=has_target_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return loss.squeeze(1).sum() / float(normalizer), m, d


def _fused_log_softmax_backward(
    *,
    logits,
    target_p,
    position_mask,
    m,
    d,
    normalizer,
    grad_output,
    target_indices=None,
):
    B, T, V = logits.shape
    if target_indices is None:
        assert logits.shape == target_p.shape
        target_indices_ptr = target_p
        has_target_indices = False
    else:
        assert B == 1
        assert target_p.ndim == 2 and target_p.shape[1] == V
        assert target_indices.numel() == B * T
        target_indices_ptr = target_indices.contiguous()
        has_target_indices = True
    scaling_factor = 1.0 / float(normalizer)
    logits_flat = logits.contiguous().view(B * T, V)
    target_flat = target_p.contiguous().view(-1, V)
    position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
    grid = (B * T,)
    BLOCK_SIZE, num_warps = _calculate_settings(V)
    _log_softmax_backward_kernel[grid](
        logits_flat,
        logits_flat.stride(0),
        target_flat,
        target_flat.stride(0),
        target_indices_ptr,
        position_mask_flat,
        grad_output,
        scaling_factor,
        m,
        d,
        V,
        HAS_TARGET_INDICES=has_target_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return logits_flat.view(B, T, V)


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
        loss, m, d = _fused_log_softmax_forward(
            logits,
            target_p,
            position_mask,
            normalizer,
        )
        ctx.save_for_backward(logits.detach(), target_p, position_mask, m, d)
        ctx.normalizer = float(normalizer)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        grad_logits = _fused_log_softmax_backward(
            logits=logits,
            target_p=target,
            position_mask=position_mask,
            m=m,
            d=d,
            normalizer=ctx.normalizer,
            grad_output=grad_output,
        )
        return grad_logits, None, None, None


class FusedIndexedLogSoftmaxLoss(torch.autograd.Function):
    """Fused loss that reads target rows directly from one shared compact table.

    Forward and backward pass row indices to the Triton kernels, so neither
    phase materializes ``[N_step, V]`` target copies. The autograd context saves
    only the shared table plus row indices.
    """

    @staticmethod
    def forward(
        ctx,
        logits,
        target_table,
        target_indices,
        position_mask,
        normalizer,
    ):
        assert logits.shape[0] == 1
        loss, m, d = _fused_log_softmax_forward(
            logits,
            target_table,
            position_mask,
            normalizer,
            target_indices=target_indices,
        )
        ctx.save_for_backward(
            logits.detach(),
            target_table,
            target_indices,
            position_mask,
            m,
            d,
        )
        ctx.normalizer = float(normalizer)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            logits,
            target_table,
            target_indices,
            position_mask,
            m,
            d,
        ) = ctx.saved_tensors
        grad_logits = _fused_log_softmax_backward(
            logits=logits,
            target_p=target_table,
            position_mask=position_mask,
            m=m,
            d=d,
            normalizer=ctx.normalizer,
            grad_output=grad_output,
            target_indices=target_indices,
        )
        return grad_logits, None, None, None, None


def compute_eagle3_loss(
    *,
    model,
    batch: dict[str, torch.Tensor],
    ttt_length: int,
    step_loss_decay: float,
    trim_loss_positions: bool = False,
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
    compact_plan = (
        _build_compact_plan(position_masks=position_masks)
        if bool(trim_loss_positions)
        else None
    )

    past_key_values = DynamicCache()
    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    with torch.no_grad():
        # A-level deliberately leaves this full-length teacher LM-head
        # projection unchanged. Compaction starts only after target_logits.
        target_logits = model(
            target_last_hidden_states=target_last_hidden_states,
            target_logits_only=True,
        )
    if compact_plan is None:
        target_probs = _build_padded_next_token_target_probs(
            target_logits=target_logits,
            ttt_length=int(ttt_length),
        )
        compact_teacher_probs = None
        compact_teacher_token_ids = None
    else:
        target_probs = None
        compact_teacher_probs = _build_compact_teacher_probs(
            target_logits=target_logits,
            compact_plan=compact_plan,
        )
        compact_teacher_token_ids = compact_teacher_probs.argmax(dim=-1)
    del target_logits

    correct_masks = []
    accept_rate_masks = []
    valid_masks = []
    for step_idx in range(int(ttt_length)):
        if compact_plan is None:
            # Keep this slice alignment in sync with the Eagle3 reference.
            target_step_probs = target_probs[
                :,
                step_idx : step_idx + seq_len,
                :,
            ].contiguous()
            position_mask_step = position_masks[step_idx]
            logit_indices = None
            metric_scatter_indices = None
            metric_full_shape = None
        else:
            compact_step = compact_plan.steps[step_idx]
            target_step_probs = None
            position_mask_step = compact_step.position_mask
            logit_indices = compact_step.draft_indices
            metric_scatter_indices = compact_step.scatter_indices
            metric_full_shape = (compact_plan.batch_size, compact_plan.seq_len)
        output = model(
            hidden_states=hidden_states,
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            position_ids=base_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_logits=True,
            rope_cache_step_offset=True,
            logit_indices=logit_indices,
        )
        hidden_states = output.hidden_states
        if compact_plan is None:
            correct_mask, accept_rate_mask, valid_mask = _log_eagle3_step_metrics(
                step_idx=step_idx,
                draft_logits=output.draft_logits,
                target_probs=target_step_probs,
                position_mask=position_mask_step,
            )
        else:
            correct_mask, accept_rate_mask, valid_mask = (
                _log_eagle3_indexed_step_metrics(
                    step_idx=step_idx,
                    draft_logits=output.draft_logits,
                    target_table=compact_teacher_probs,
                    target_token_ids_table=compact_teacher_token_ids,
                    target_indices=compact_step.teacher_table_indices,
                    position_mask=position_mask_step,
                    full_shape=metric_full_shape,
                    scatter_indices=metric_scatter_indices,
                )
            )
        correct_masks.append(correct_mask)
        accept_rate_masks.append(accept_rate_mask)
        valid_masks.append(valid_mask)
        if compact_plan is None:
            step_loss = FusedLogSoftmaxLoss.apply(
                output.draft_logits,
                target_step_probs,
                position_mask_step,
                loss_normalizers[step_idx],
            )
        else:
            step_loss = FusedIndexedLogSoftmaxLoss.apply(
                output.draft_logits,
                compact_teacher_probs,
                compact_step.teacher_table_indices,
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
        current_input_ids = _shift_with_zero_padding(current_input_ids, left=False)

    _log_eagle3_prefix_metrics(
        correct_masks=correct_masks,
        accept_rate_masks=accept_rate_masks,
        valid_masks=valid_masks,
    )
    add_metric("loss", total_loss.detach(), reduction="dp_mean", tag="train")
    return total_loss


__all__ = [
    "Eagle3CompactPlan",
    "Eagle3CompactStep",
    "FusedIndexedLogSoftmaxLoss",
    "FusedLogSoftmaxLoss",
    "_build_compact_plan",
    "compute_eagle3_loss",
]
