from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.utils.metrics import add_metric
from .common import DSparkForwardOutput


def _all_reduce_loss_denominators(
    loss_terms: dict[str, torch.Tensor],
    *,
    world_size: int,
) -> dict[str, torch.Tensor]:
    denominators = {}
    for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den"):
        tensor = loss_terms[key].detach().clone()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        denominators[key] = tensor
    return denominators


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    loss_decay_gamma: Optional[float],
) -> torch.Tensor:
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def _compute_local_probabilistic_stats(
    *,
    outputs: DSparkForwardOutput,
    accept_rate_3d: Optional[torch.Tensor],
    valid_block_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, block_size, _ = outputs.draft_logits.shape
    device = outputs.draft_logits.device
    pos_accept_sums = torch.zeros(block_size, device=device, dtype=torch.float32)
    if accept_rate_3d is None:
        return outputs.draft_logits.new_zeros((), dtype=torch.float32), pos_accept_sums

    valid_accept_rate = accept_rate_3d * outputs.eval_mask.to(torch.float32)
    pos_accept_sums = valid_accept_rate.sum(dim=(0, 1))
    expected_draft_accepted = valid_accept_rate.cumprod(dim=-1).sum(dim=-1)
    tau_prob_per_block = expected_draft_accepted + 1.0
    tau_prob_sum = (tau_prob_per_block * valid_block_weights).sum()
    return tau_prob_sum, pos_accept_sums


def _compute_accept_rate_3d(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if aligned_target_logits is None:
        return None
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
    return accept_rate_3d.clamp_(0.0, 1.0)


def _compute_local_l1_term(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: Optional[torch.Tensor],
    loss_weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = outputs.draft_logits.new_zeros((), dtype=torch.float32)
    if aligned_target_logits is None:
        return zero, zero
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    l1_dist_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
    l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
    l1_loss_den = loss_weight_mask.sum()
    return l1_loss_num, l1_loss_den


def _collect_local_terms(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    l1_loss_alpha: float,
) -> tuple[dict[str, torch.Tensor], bool]:
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    block_keep_mask = outputs.block_keep_mask
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
    )
    flat_logits = draft_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()
    aligned_target_logits = outputs.aligned_target_logits
    accept_rate_3d = _compute_accept_rate_3d(
        outputs=outputs,
        aligned_target_logits=aligned_target_logits,
    )
    zero = ce_loss_num.new_zeros(())
    assert l1_loss_alpha <= 0 or aligned_target_logits is not None, (
        "aligned_target_logits is required when l1_loss_alpha > 0."
    )
    if l1_loss_alpha > 0:
        l1_loss_num, l1_loss_den = _compute_local_l1_term(
            outputs=outputs,
            aligned_target_logits=aligned_target_logits,
            loss_weight_mask=loss_weight_mask,
        )
    else:
        l1_loss_num = zero
        l1_loss_den = zero

    with torch.no_grad():
        pos_total_counts = eval_mask.to(torch.float32).sum(dim=(0, 1))
        valid_pred_tokens = eval_mask.any(dim=-1)
        valid_blocks = block_keep_mask & valid_pred_tokens
        valid_block_weights = valid_blocks.to(torch.float32)
        accept_block_count = valid_block_weights.sum()
        tau_prob_sum, pos_accept_sums = _compute_local_probabilistic_stats(
            outputs=outputs,
            accept_rate_3d=accept_rate_3d,
            valid_block_weights=valid_block_weights,
        )

    has_confidence = outputs.confidence_pred is not None
    confidence_loss_num = zero
    confidence_loss_den = zero
    confidence_abs_error_num = zero
    confidence_bias_num = zero
    confidence_cumprod_bias_num = zero
    if has_confidence:
        assert accept_rate_3d is not None, (
            "aligned_target_logits is required when confidence head is enabled."
        )
        confidence_targets = accept_rate_3d.detach()
        confidence_errors = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(),
            confidence_targets,
            reduction="none",
        ) * loss_weight_mask
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()
        with torch.no_grad():
            confidence_probs = outputs.confidence_pred.float().sigmoid()
            confidence_error = confidence_probs - accept_rate_3d
            confidence_abs_error_num = (
                confidence_error.abs() * loss_weight_mask
            ).sum()
            confidence_bias_num = (confidence_error * loss_weight_mask).sum()
            valid_mask = outputs.eval_mask.to(torch.float32)
            confidence_prefix_probs = (
                confidence_probs * valid_mask
            ).cumprod(dim=-1)
            confidence_prefix_targets = (
                accept_rate_3d * valid_mask
            ).cumprod(dim=-1)
            confidence_cumprod_bias_num = (
                (confidence_prefix_probs - confidence_prefix_targets)
                * loss_weight_mask
            ).sum()

    loss_terms = {
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_loss_num,
        "confidence_loss_den": confidence_loss_den,
    }

    for pos_idx in range(block_size):
        add_metric(
            f"accept_rate@{pos_idx}",
            pos_accept_sums[pos_idx],
            den=pos_total_counts[pos_idx],
            tag="train",
        )
    add_metric(
        "tau_probabilistic",
        tau_prob_sum,
        den=accept_block_count,
        tag="train",
    )
    if has_confidence:
        add_metric(
            "confidence_abs_error",
            confidence_abs_error_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_bias",
            confidence_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_cumprod_bias",
            confidence_cumprod_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
    return loss_terms, has_confidence


def _build_loss(
    *,
    loss_terms: dict[str, torch.Tensor],
    global_denominators: dict[str, torch.Tensor],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    has_confidence: bool,
    world_size: int,
) -> torch.Tensor:
    ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
    l1_loss = ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        l1_loss = loss_terms["l1_loss_num"] / (
            global_denominators["l1_loss_den"] + 1e-6
        )
    confidence_loss = ce_loss.new_zeros(())
    if has_confidence:
        confidence_loss = loss_terms["confidence_loss_num"] / (
            global_denominators["confidence_loss_den"] + 1e-6
        )
    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    ) * world_size


def compute_dspark_loss(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
):
    loss_terms, has_confidence = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        l1_loss_alpha=float(l1_loss_alpha),
    )
    world_size = dist.get_world_size()
    global_denominators = _all_reduce_loss_denominators(
        loss_terms,
        world_size=world_size,
    )
    ce_loss_alpha = float(ce_loss_alpha)
    l1_loss_alpha = float(l1_loss_alpha)
    confidence_head_alpha = float(confidence_head_alpha)

    add_metric(
        "ce_loss",
        loss_terms["ce_loss_num"],
        den=loss_terms["ce_loss_den"],
        tag="train",
    )
    if global_denominators["l1_loss_den"].item() > 0:
        add_metric(
            "l1_loss",
            loss_terms["l1_loss_num"],
            den=loss_terms["l1_loss_den"],
            tag="train",
        )
    if has_confidence:
        add_metric(
            "confidence_loss",
            loss_terms["confidence_loss_num"],
            den=loss_terms["confidence_loss_den"],
            tag="train",
        )
    backward_loss = _build_loss(
        loss_terms=loss_terms,
        global_denominators=global_denominators,
        ce_loss_alpha=ce_loss_alpha,
        l1_loss_alpha=l1_loss_alpha,
        confidence_head_alpha=confidence_head_alpha,
        has_confidence=has_confidence,
        world_size=world_size,
    )
    add_metric(
        "loss",
        backward_loss.detach(),
        reduction="dp_mean",
        tag="train",
    )
    return backward_loss


__all__ = [
    "compute_dspark_loss",
]
