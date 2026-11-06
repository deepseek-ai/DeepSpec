import pytest
import torch

from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
    sample_tokens,
)


@pytest.mark.parametrize("shape", [(10,), (4, 10), (2, 3, 10), (2, 2, 3, 10)])
def test_sample_from_probs_shapes(shape):
    torch.manual_seed(42)
    raw = torch.rand(*shape)
    probs = raw / raw.sum(dim=-1, keepdim=True)
    sampled = sample_from_probs(probs)

    assert sampled.shape == shape[:-1]
    assert (sampled >= 0).all()
    assert (sampled < shape[-1]).all()


@pytest.mark.parametrize("shape", [(10,), (4, 10), (2, 3, 10), (2, 2, 3, 10)])
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_sample_tokens_multidimensional(shape, temperature):
    torch.manual_seed(42)
    logits = torch.randn(*shape)
    sampled = sample_tokens(logits, temperature=temperature)

    assert sampled.shape == shape[:-1]
    assert (sampled >= 0).all()
    assert (sampled < shape[-1]).all()


def test_sample_tokens_greedy_equivalence():
    logits = torch.tensor([[1.0, 5.0, 2.0], [4.0, 1.0, 0.0]])
    sampled = sample_tokens(logits, temperature=0.0)
    expected = torch.tensor([1, 0])
    assert torch.equal(sampled, expected)


def test_sample_residual_2d_and_3d():
    torch.manual_seed(42)
    # 2D test (batch, vocab)
    target_probs_2d = torch.softmax(torch.randn(4, 32), dim=-1)
    draft_probs_2d = torch.softmax(torch.randn(4, 32), dim=-1)
    sampled_2d = sample_residual(target_probs_2d, draft_probs_2d)
    assert sampled_2d.shape == (4,)
    assert (sampled_2d >= 0).all() and (sampled_2d < 32).all()

    # 3D test (batch, seq, vocab)
    target_probs_3d = torch.softmax(torch.randn(2, 5, 32), dim=-1)
    draft_probs_3d = torch.softmax(torch.randn(2, 5, 32), dim=-1)
    sampled_3d = sample_residual(target_probs_3d, draft_probs_3d)
    assert sampled_3d.shape == (2, 5)
    assert (sampled_3d >= 0).all() and (sampled_3d < 32).all()


def test_sample_residual_identical_fallback():
    # When target and draft probabilities are identical, residual_mass is 0
    # and fallback to target_probs must occur smoothly without NaNs.
    probs = torch.softmax(torch.randn(3, 16), dim=-1)
    sampled = sample_residual(probs, probs)
    assert sampled.shape == (3,)
    assert not torch.isnan(sampled.float()).any()


def test_gather_token_probs():
    probs = torch.tensor([
        [0.1, 0.7, 0.2],
        [0.8, 0.1, 0.1],
    ])
    token_ids = torch.tensor([1, 0])
    gathered = gather_token_probs(probs, token_ids)
    assert torch.allclose(gathered, torch.tensor([0.7, 0.8]))


def test_logits_to_probs():
    logits = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    # Low temperature (greedy one-hot)
    probs_zero = logits_to_probs(logits, temperature=0.0)
    assert torch.equal(probs_zero, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    # High temperature
    probs_temp = logits_to_probs(logits, temperature=1.0)
    assert torch.allclose(probs_temp.sum(dim=-1), torch.ones(2))
