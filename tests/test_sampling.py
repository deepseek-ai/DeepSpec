"""Tests for deepspec.utils.sampling.

Covers the deterministic surfaces of the sampling helpers: greedy/temperature-0
paths, the softmax path, probability gathering, and speculative-decoding residual
sampling (including the equal-distribution fallback). Sampling calls use one-hot
distributions so ``torch.multinomial`` is deterministic. CPU-only.
"""

from __future__ import annotations

import torch

from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
    sample_tokens,
)


def test_logits_to_probs_greedy_is_one_hot_at_argmax():
    logits = torch.tensor([[[1.0, 3.0, 2.0]]])
    probs = logits_to_probs(logits, temperature=0.0)
    assert torch.equal(probs, torch.tensor([[[0.0, 1.0, 0.0]]]))
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1))


def test_logits_to_probs_temperature_one_matches_softmax():
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    probs = logits_to_probs(logits, temperature=1.0)
    expected = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(probs, expected)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1))


def test_sample_tokens_greedy_returns_argmax():
    logits = torch.tensor([[[0.1, 0.9, 0.0], [5.0, 1.0, 2.0]]])
    out = sample_tokens(logits, temperature=0.0)
    assert torch.equal(out, logits.argmax(dim=-1))
    assert out.shape == (1, 2)


def test_gather_token_probs_selects_indexed_values():
    probs = torch.tensor([[[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]])
    token_ids = torch.tensor([[2, 0]])
    out = gather_token_probs(probs, token_ids)
    assert torch.allclose(out, torch.tensor([[0.5, 0.6]]))


def test_sample_from_probs_is_deterministic_for_one_hot():
    torch.manual_seed(0)
    probs = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    out = sample_from_probs(probs)
    assert torch.equal(out, torch.tensor([[1, 0]]))


def test_sample_residual_prefers_unmatched_target_mass():
    torch.manual_seed(0)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    draft = torch.tensor([[0.0, 0.0, 1.0]])
    # residual = clamp(target - draft, 0) = [0, 1, 0] -> token 1
    out = sample_residual(target, draft)
    assert torch.equal(out, torch.tensor([1]))


def test_sample_residual_falls_back_when_distributions_match():
    torch.manual_seed(0)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    draft = torch.tensor([[0.0, 1.0, 0.0]])
    # residual mass is ~0 -> fall back to target_probs -> token 1
    out = sample_residual(target, draft)
    assert torch.equal(out, torch.tensor([1]))
