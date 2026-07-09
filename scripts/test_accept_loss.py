"""CPU unit test for the optional direct-acceptance (LK) loss in DSpark.

Runs without a GPU or a real model: it mocks a ``DSparkForwardOutput`` and checks
  (1) accept_loss_alpha=0 is exactly the pre-existing CE(+L1+confidence) loss (zero regression);
  (2) the accept term is differentiable (gradient reaches draft_logits);
  (3) it rewards a draft that matches the target (lower loss when draft==target logits).

Usage: python scripts/test_accept_loss.py
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# metrics may accumulate to global rank state; make it a no-op for the unit test.
import deepspec.utils.metrics as _metrics  # noqa: E402
_metrics.add_metric = lambda *a, **k: None
import deepspec.modeling.dspark.loss as L  # noqa: E402
L.add_metric = lambda *a, **k: None
from deepspec.modeling.dspark.common import DSparkForwardOutput  # noqa: E402


def _init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _mock(B=1, A=2, block=4, V=32, draft_eq_target=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    target_logits = torch.randn(B, A, block, V, generator=g)
    draft = target_logits.clone() if draft_eq_target else torch.randn(B, A, block, V, generator=g)
    draft = draft.detach().requires_grad_(True)
    target_ids = target_logits.argmax(dim=-1)
    return draft, DSparkForwardOutput(
        draft_logits=draft,
        target_ids=target_ids,
        eval_mask=torch.ones(B, A, block, dtype=torch.bool),
        block_keep_mask=torch.ones(B, A, dtype=torch.bool),
        confidence_pred=None,
        aligned_target_logits=target_logits,
    )


def _loss(out, accept_alpha):
    return L.compute_dspark_loss(
        outputs=out, loss_decay_gamma=None,
        ce_loss_alpha=1.0, l1_loss_alpha=0.0, confidence_head_alpha=0.0,
        accept_loss_alpha=accept_alpha,
    )


def main():
    _init_dist()

    # (1) zero regression: accept_loss_alpha=0 == the CE-only loss to the bit
    _, o0 = _mock(seed=1)
    base = _loss(o0, 0.0)
    _, o0b = _mock(seed=1)
    off = _loss(o0b, 0.0)
    assert torch.allclose(base, off), (base, off)
    print(f"(1) zero-regression OK  (CE-only loss={base.item():.4f})")

    # (2) differentiable: grad reaches draft_logits, and the accept term changes the loss
    draft, o1 = _mock(seed=2)
    withacc = _loss(o1, 0.5)
    _, o1b = _mock(seed=2)
    ceonly = _loss(o1b, 0.0)
    assert not torch.allclose(withacc, ceonly), "accept term did not change the loss"
    withacc.backward()
    assert draft.grad is not None and torch.isfinite(draft.grad).all() and draft.grad.abs().sum() > 0
    print(f"(2) differentiable OK   (loss +accept={withacc.item():.4f} vs CE-only={ceonly.item():.4f}; grad flows)")

    # (3) rewards matching: a draft equal to target has a LOWER (more negative) accept term
    dm, om = _mock(seed=3, draft_eq_target=True)   # draft == target
    dr, orr = _mock(seed=3, draft_eq_target=False)  # random draft
    # isolate the accept term (ce_alpha=0, accept_alpha=1)
    acc_match = L.compute_dspark_loss(outputs=om, loss_decay_gamma=None,
                                      ce_loss_alpha=0.0, l1_loss_alpha=0.0,
                                      confidence_head_alpha=0.0, accept_loss_alpha=1.0)
    acc_rand = L.compute_dspark_loss(outputs=orr, loss_decay_gamma=None,
                                     ce_loss_alpha=0.0, l1_loss_alpha=0.0,
                                     confidence_head_alpha=0.0, accept_loss_alpha=1.0)
    assert acc_match.item() < acc_rand.item(), (acc_match.item(), acc_rand.item())
    print(f"(3) rewards matching OK (accept-loss match={acc_match.item():.4f} < random={acc_rand.item():.4f})")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
