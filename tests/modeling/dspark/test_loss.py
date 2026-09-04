from unittest.mock import patch

import pytest
import torch

from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import (
    _collect_local_terms,
    _compute_accept_rate_3d,
    _compute_l1_dist_per_token,
)


def _make_outputs(*, confidence: bool = False) -> DSparkForwardOutput:
    torch.manual_seed(0)
    shape = (1, 2, 3)
    vocab_size = 17
    return DSparkForwardOutput(
        draft_logits=torch.randn(*shape, vocab_size, requires_grad=True),
        target_ids=torch.randint(0, vocab_size, shape),
        eval_mask=torch.tensor([[[True, True, False], [True, True, True]]]),
        block_keep_mask=torch.tensor([[True, True]]),
        confidence_pred=(
            torch.randn(*shape, requires_grad=True) if confidence else None
        ),
        aligned_target_logits=torch.randn(*shape, vocab_size),
    )


def _reference_distance(outputs: DSparkForwardOutput) -> torch.Tensor:
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(outputs.aligned_target_logits.float(), dim=-1)
    return (draft_probs - target_probs).abs().sum(dim=-1)


def test_collect_local_terms_computes_probability_distance_once(monkeypatch):
    outputs = _make_outputs()
    monkeypatch.setattr(
        "deepspec.modeling.dspark.loss.add_metric",
        lambda *args, **kwargs: None,
    )

    with patch("torch.softmax", wraps=torch.softmax) as softmax:
        _collect_local_terms(
            outputs=outputs,
            loss_decay_gamma=None,
            l1_loss_alpha=1.0,
        )

    assert softmax.call_count == 2


@pytest.mark.parametrize("confidence", [False, True])
def test_shared_distance_matches_original_values_and_gradients(confidence):
    outputs = _make_outputs(confidence=confidence)
    actual = _compute_l1_dist_per_token(
        draft_logits=outputs.draft_logits,
        aligned_target_logits=outputs.aligned_target_logits,
    )
    expected = _reference_distance(outputs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        _compute_accept_rate_3d(actual),
        (1.0 - 0.5 * expected).clamp(0.0, 1.0),
        rtol=1e-5,
        atol=1e-6,
    )
    actual_grad = torch.autograd.grad(
        actual.sum(), outputs.draft_logits, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad(expected.sum(), outputs.draft_logits)[0]
    torch.testing.assert_close(
        actual_grad,
        expected_grad,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("l1_loss_alpha", [0.0, 1.0])
def test_collect_local_terms_matches_weighted_reference(
    monkeypatch,
    l1_loss_alpha,
):
    outputs = _make_outputs()
    monkeypatch.setattr(
        "deepspec.modeling.dspark.loss.add_metric",
        lambda *args, **kwargs: None,
    )
    terms, _ = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=None,
        l1_loss_alpha=l1_loss_alpha,
    )
    if l1_loss_alpha == 0.0:
        assert terms["l1_loss_num"].item() == 0.0
        assert terms["l1_loss_den"].item() == 0.0
    else:
        weights = outputs.eval_mask.to(torch.float32)
        expected_num = (_reference_distance(outputs) * weights).sum()
        torch.testing.assert_close(terms["l1_loss_num"], expected_num)
        torch.testing.assert_close(terms["l1_loss_den"], weights.sum())


def test_collect_local_terms_skips_probability_work_without_target(monkeypatch):
    outputs = _make_outputs()
    outputs.aligned_target_logits = None
    monkeypatch.setattr(
        "deepspec.modeling.dspark.loss.add_metric",
        lambda *args, **kwargs: None,
    )

    with patch("torch.softmax", wraps=torch.softmax) as softmax:
        terms, has_confidence = _collect_local_terms(
            outputs=outputs,
            loss_decay_gamma=None,
            l1_loss_alpha=0.0,
        )

    assert not has_confidence
    assert terms["l1_loss_num"].item() == 0.0
    assert terms["l1_loss_den"].item() == 0.0
    assert softmax.call_count == 0
