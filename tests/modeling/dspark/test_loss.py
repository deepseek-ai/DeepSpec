from unittest.mock import patch

import torch

from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import _collect_local_terms


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
