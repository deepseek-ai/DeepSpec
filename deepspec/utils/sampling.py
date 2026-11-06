from __future__ import annotations

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    orig_shape = probs.shape[:-1]
    vocab_size = probs.shape[-1]
    flat = probs.reshape(-1, vocab_size)
    return torch.multinomial(flat, num_samples=1).reshape(orig_shape)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)

    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(orig_shape)


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return sample_from_probs(residual)


__all__ = [
    "gather_token_probs",
    "logits_to_probs",
    "sample_from_probs",
    "sample_residual",
    "sample_tokens",
]
