"""Distribution-exactness tests for speculative rejection sampling.

Covers the lossless guarantee of ``verify_draft_tokens`` and the sampling
helpers: committed tokens must follow the target model's distribution at
every position, for any declared draft distribution. A negative control
proves the chi-square thresholds can actually reject (the bug class of
issue #31), and deterministic regression tests pin the draft-side sampling
distribution to the verification-side distribution and the latter to an
independent float64 reference.

All tests are CPU-only. Statistical thresholds are chosen so the suite
passes for any RNG stream (per-assert false-fail rate ~1e-9, asymptotic
chi-square tail); the fixed seed only makes failures reproducible. Tests
that need ``deepspec.eval.base_evaluator`` skip cleanly in environments
without transformers installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from deepspec.utils.sampling import (
    logits_to_probs,
    sample_from_probs,
    sample_residual,
    sample_tokens,
)

_SEED = 20260702
_VOCAB = 4
_DRAFT_LEN = 2
_TRIALS = 20_000
_CONTROL_TRIALS = 10_000
_TEMPERATURE = 0.7

# Chi-square critical values for df = _VOCAB - 1 = 3, from the closed-form
# survival function S(x) = erfc(sqrt(x/2)) + sqrt(2x/pi) * exp(-x/2)
# (no scipy dependency): S(44.8413) = 1.0e-9 and S(64.0) = 8.2e-14.
# 44.8413 bounds each positive assert's false-fail rate at ~1e-9 for any
# seed; 64.0 is the floor the negative control must clear (its analytic
# non-centrality is ~356 at _CONTROL_TRIALS, so its expected chi-square is
# ~359, more than 7 standard deviations above the floor).
_CRIT_1E9 = 44.8413
_NEGATIVE_FLOOR = 64.0

# Target logits for three verification positions, and the draft
# distributions the tests declare. Reference target distributions are
# recomputed with float64 softmax, independent of deepspec code, so a bug
# in logits_to_probs cannot move the reference the tests compare against.
_TARGET_LOGITS = [
    [1.2, 0.4, -0.3, -0.9],
    [0.1, 1.0, -0.5, 0.3],
    [-0.2, 0.6, 0.9, -0.4],
]
_DRAFT_PROBS = [
    [0.40, 0.30, 0.20, 0.10],
    [0.25, 0.35, 0.15, 0.25],
]
# Deliberately mismatched proposal distribution for the negative control.
_MISMATCHED_PROBS = [0.55, 0.22, 0.13, 0.10]


@pytest.fixture(scope="module")
def evaluator_mod():
    """Import the evaluator module, skipping when transformers is absent.

    ``deepspec/eval/__init__.py`` imports the full modeling stack, which
    needs the pinned transformers from requirements.txt; environments
    without it skip these tests instead of failing at collection time.
    """
    return pytest.importorskip(
        "deepspec.eval.base_evaluator",
        reason="needs the pinned transformers from requirements.txt",
    )


def _reference_target_probs() -> torch.Tensor:
    """Float64 softmax of the target logits, independent of deepspec."""
    logits = torch.tensor(_TARGET_LOGITS, dtype=torch.float64)
    return torch.softmax(logits / _TEMPERATURE, dim=-1)


def _chi_square(counts: torch.Tensor, probs: torch.Tensor) -> float:
    expected = probs.to(torch.float64) * counts.sum()
    observed = counts.to(torch.float64)
    return float(((observed - expected) ** 2 / expected).sum())


def _mismatch_noncentrality() -> float:
    """Analytic chi-square non-centrality of the negative control.

    Closed form of the committed-token law when proposals come from ``r``
    but ``q`` is declared: committed(x) = r(x) * min(1, p(x) / q(x)) +
    P(reject) * residual(x). Guards the control's power against future
    edits to the probability vectors; the control's pass criterion itself
    never uses this model, so the two cannot be wrong together.
    """
    p = _reference_target_probs()[0]
    q = torch.tensor(_DRAFT_PROBS, dtype=torch.float64)[0]
    r = torch.tensor(_MISMATCHED_PROBS, dtype=torch.float64)
    accept = torch.minimum(torch.ones_like(p), p / q)
    reject_mass = 1.0 - (r * accept).sum()
    residual = torch.clamp(p - q, min=0.0)
    residual = residual / residual.sum()
    committed = r * accept + reject_mass * residual
    return float(_CONTROL_TRIALS * ((committed - p) ** 2 / p).sum())


def _make_fake_target(logits_rows: list[list[float]]):
    """A stand-in target model returning fixed rank-3 float32 logits.

    ``seen`` records the keyword arguments of the most recent call so
    tests can check what verification actually passed to the target.
    """
    full = torch.tensor(logits_rows, dtype=torch.float32).unsqueeze(0)
    seen: dict[str, torch.Tensor] = {}

    def fake_target(
        input_ids=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
    ):
        seen["input_ids"] = input_ids
        seen["position_ids"] = position_ids
        length = input_ids.shape[1]
        return SimpleNamespace(logits=full[:, :length, :].clone())

    return fake_target, seen


def _run_verify_once(
    evaluator_mod,
    *,
    fake_target,
    verify_input_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    draft_token_count: int,
    temperature: float = _TEMPERATURE,
    stop_token_ids: list[int] | None = None,
):
    """Single verification round; the only verify_draft_tokens call site."""
    proposal = evaluator_mod.DraftProposal(
        draft_token_count=draft_token_count,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
    )
    return evaluator_mod.verify_draft_tokens(
        target_model=fake_target,
        proposal=proposal,
        position_ids=torch.arange(_DRAFT_LEN + 1).unsqueeze(0),
        start=0,
        past_key_values_target=evaluator_mod.DynamicCache(),
        temperature=temperature,
        max_proposal_tokens=_DRAFT_LEN,
        current_token_ids=verify_input_ids[:, :1],
        stop_token_ids=stop_token_ids,
    )


def _committed_histograms(
    evaluator_mod,
    *,
    proposal_dist: torch.Tensor,
    declared_dist: torch.Tensor,
    trials: int,
    seed: int,
):
    """Histogram committed tokens per position over ``trials`` rounds.

    Proposals are drawn from ``proposal_dist`` while ``declared_dist`` is
    passed as draft_probs; the two match in the honest tests and differ in
    the negative control. stop_token_ids stays None here: after a
    stop-token truncation, committed_tokens carries a trailing correction
    token that is not distribution-guaranteed (the decode loop discards
    it), so stop handling gets deterministic coverage instead.
    """
    torch.manual_seed(seed)
    fake_target, _ = _make_fake_target(_TARGET_LOGITS)
    declared = declared_dist.to(torch.float32).unsqueeze(0)
    proposals = torch.stack(
        [
            torch.multinomial(
                proposal_dist[pos].to(torch.float32), trials, replacement=True
            )
            for pos in range(_DRAFT_LEN)
        ],
        dim=1,
    )
    counts = torch.zeros(_DRAFT_LEN + 1, _VOCAB)
    reached = torch.zeros(_DRAFT_LEN + 1)
    current = torch.tensor([[3]])
    with torch.inference_mode():
        for i in range(trials):
            result = _run_verify_once(
                evaluator_mod,
                fake_target=fake_target,
                verify_input_ids=torch.cat([current, proposals[i : i + 1]], dim=1),
                draft_probs=declared,
                draft_token_count=_DRAFT_LEN,
            )
            committed = result.committed_tokens[0]
            for pos in range(committed.shape[0]):
                counts[pos, committed[pos]] += 1
                reached[pos] += 1
    return counts, reached


def test_committed_tokens_match_target_distribution(evaluator_mod):
    """Lossless guarantee: with proposals drawn from the declared
    draft_probs, the committed token at every position follows the target
    distribution even though the draft distribution is deliberately
    different from the target's."""
    declared = torch.tensor(_DRAFT_PROBS, dtype=torch.float32)
    counts, reached = _committed_histograms(
        evaluator_mod,
        proposal_dist=declared,
        declared_dist=declared,
        trials=_TRIALS,
        seed=_SEED,
    )
    reference = _reference_target_probs()
    assert int(reached[1]) > 1_000 and int(reached[2]) > 1_000, (
        f"acceptance collapsed: reached={reached.tolist()} out of {_TRIALS} "
        f"trials (seed={_SEED})"
    )
    for pos in range(_DRAFT_LEN + 1):
        stat = _chi_square(counts[pos], reference[pos])
        expected = [round(x, 1) for x in (reference[pos] * reached[pos]).tolist()]
        assert stat < _CRIT_1E9, (
            f"position {pos}: chi2={stat:.2f} exceeds {_CRIT_1E9} "
            f"(false-fail rate 1e-9), n={int(reached[pos])}, seed={_SEED}, "
            f"observed={counts[pos].tolist()}, expected={expected}"
        )


def test_chi_square_rejects_mismatched_proposals(evaluator_mod):
    """Negative control: proposals drawn from a distribution other than
    the declared draft_probs must blow past the rejection floor. This is
    the bug class of issue #31 and proves the positive thresholds have
    power (a distribution test that has never rejected proves nothing)."""
    noncentrality = _mismatch_noncentrality()
    assert noncentrality > 300, (
        f"mismatch too weak to prove power: non-centrality={noncentrality:.0f}"
    )
    declared = torch.tensor(_DRAFT_PROBS, dtype=torch.float32)
    mismatched = declared.clone()
    mismatched[0] = torch.tensor(_MISMATCHED_PROBS)
    counts, _ = _committed_histograms(
        evaluator_mod,
        proposal_dist=mismatched,
        declared_dist=declared,
        trials=_CONTROL_TRIALS,
        seed=_SEED,
    )
    stat = _chi_square(counts[0], _reference_target_probs()[0])
    assert stat > _NEGATIVE_FLOOR, (
        f"mismatched proposals not detected: chi2={stat:.2f} is below "
        f"{_NEGATIVE_FLOOR} (expected ~{3 + noncentrality:.0f}, seed={_SEED})"
    )


@pytest.mark.xfail(
    strict=True,
    reason="issue #31: sample_tokens samples from the native-dtype softmax "
    "while verification uses the float32 distribution from logits_to_probs; "
    "delete this marker when the fix lands",
)
def test_sample_tokens_matches_verification_distribution():
    """Seed-matched equivalence: drawing tokens via sample_tokens must be
    bitwise-indistinguishable from sampling the distribution rejection
    sampling assumes (logits_to_probs). Both paths make exactly one
    torch.multinomial call whose internal noise inherits the probs dtype,
    so equality requires the same probability values in the same dtype:
    the float32 verification pipeline. The current bug compounds the
    softmax dtype gap with multinomial's dtype-dependent noise for a
    ~1e-3 mismatch rate over 100k draws, so this xfail is deterministic
    for any seed and strict=True is safe: fixing #31 turns this into a
    hard reminder to remove the marker and arm the regression guard."""
    draws = 100_000
    logits = (
        torch.tensor([2.5, 1.25, -0.6, 0.4], dtype=torch.bfloat16)
        .view(1, 1, _VOCAB)
        .expand(1, draws, _VOCAB)
    )
    torch.manual_seed(_SEED)
    via_sample_tokens = sample_tokens(logits, temperature=_TEMPERATURE)
    torch.manual_seed(_SEED)
    via_verify_dist = sample_from_probs(logits_to_probs(logits, _TEMPERATURE))
    mismatches = int((via_sample_tokens != via_verify_dist).sum())
    assert mismatches == 0, (
        f"{mismatches}/{draws} draft tokens were drawn from a different "
        f"distribution than rejection sampling assumes (seed={_SEED})"
    )


def test_logits_to_probs_matches_float64_reference():
    """The verification-side distribution must be computed in full
    precision from reduced-precision logits: softmax of bfloat16 logits
    must match an independent float64 reference within 1e-5, far tighter
    than bfloat16 arithmetic (~1e-3) could achieve."""
    torch.manual_seed(_SEED)
    for scale in (1.0, 8.0, 32.0):
        logits = (torch.randn(2, 5, _VOCAB) * scale).to(torch.bfloat16)
        got = logits_to_probs(logits, _TEMPERATURE)
        want = torch.softmax(logits.to(torch.float64) / _TEMPERATURE, dim=-1)
        torch.testing.assert_close(got.to(torch.float64), want, rtol=1e-5, atol=1e-5)


def test_greedy_verification_commits_argmax_chain(evaluator_mod):
    """temperature < 1e-5 turns verification deterministic: a draft that
    matches the target argmax chain is fully accepted and extended with
    the argmax bonus token; a mismatching first token is rejected and
    corrected to the argmax."""
    fake_target, seen = _make_fake_target(_TARGET_LOGITS)
    argmax_chain = [int(torch.tensor(row).argmax()) for row in _TARGET_LOGITS]
    uniform = torch.full((1, _DRAFT_LEN, _VOCAB), 1.0 / _VOCAB)
    current = torch.tensor([[3]])

    matched = torch.tensor([argmax_chain[:_DRAFT_LEN]])
    result = _run_verify_once(
        evaluator_mod,
        fake_target=fake_target,
        verify_input_ids=torch.cat([current, matched], dim=1),
        draft_probs=uniform,
        draft_token_count=_DRAFT_LEN,
        temperature=0.0,
    )
    assert result.accepted_draft_tokens == _DRAFT_LEN
    assert result.committed_tokens[0].tolist() == argmax_chain
    assert seen["position_ids"].tolist() == [[0, 1, 2]]
    expected_input = torch.cat([current, matched], dim=1)
    assert seen["input_ids"].tolist() == expected_input.tolist()

    wrong_first = matched.clone()
    wrong_first[0, 0] = (argmax_chain[0] + 1) % _VOCAB
    result = _run_verify_once(
        evaluator_mod,
        fake_target=fake_target,
        verify_input_ids=torch.cat([current, wrong_first], dim=1),
        draft_probs=uniform,
        draft_token_count=_DRAFT_LEN,
        temperature=0.0,
    )
    assert result.accepted_draft_tokens == 0
    assert result.committed_tokens[0].tolist() == argmax_chain[:1]


def test_greedy_stop_token_truncates_acceptance(evaluator_mod):
    """A stop token inside the accepted prefix truncates acceptance to the
    stop position and flags termination. Deterministic (greedy) coverage
    only: the trailing correction token in committed_tokens after a stop
    truncation is not distribution-guaranteed and the decode loop discards
    it, so no distributional assert belongs here."""
    fake_target, _ = _make_fake_target(_TARGET_LOGITS)
    argmax_chain = [int(torch.tensor(row).argmax()) for row in _TARGET_LOGITS]
    uniform = torch.full((1, _DRAFT_LEN, _VOCAB), 1.0 / _VOCAB)
    current = torch.tensor([[3]])
    matched = torch.tensor([argmax_chain[:_DRAFT_LEN]])
    result = _run_verify_once(
        evaluator_mod,
        fake_target=fake_target,
        verify_input_ids=torch.cat([current, matched], dim=1),
        draft_probs=uniform,
        draft_token_count=_DRAFT_LEN,
        temperature=0.0,
        stop_token_ids=[argmax_chain[0]],
    )
    assert result.terminated_by_stop_token is True
    assert result.accepted_draft_tokens == 1
    assert result.effective_proposal_length == 1
    assert result.committed_tokens[0, 0].item() == argmax_chain[0]


def test_sample_residual_near_zero_mass_falls_back():
    """sample_residual must fall back to sampling the target distribution
    when the residual mass is zero (target == draft) or positive but under
    the 1e-8 threshold. A one-hot target makes the fallback deterministic;
    seed-matching pins it to the target distribution rather than any
    degenerate renormalization of the tiny residual."""
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    token = sample_residual(one_hot, one_hot.clone())
    assert token.tolist() == [0]

    target = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    torch.manual_seed(_SEED)
    expect = sample_from_probs(target.unsqueeze(1)).squeeze(1)
    torch.manual_seed(_SEED)
    got = sample_residual(target, target.clone())
    assert torch.equal(got, expect)

    # One float32 ulp on the 0.05 entry (~3.7e-9) keeps the residual mass
    # positive but under the 1e-8 threshold; smaller perturbations (e.g.
    # +5e-9 on 0.5) round away entirely in float32.
    nearly = target.clone()
    nearly[0, 3] = torch.nextafter(nearly[0, 3], torch.tensor(1.0))
    assert not torch.equal(nearly, target)
    torch.manual_seed(_SEED)
    expect = sample_from_probs(nearly.unsqueeze(1)).squeeze(1)
    torch.manual_seed(_SEED)
    got = sample_residual(nearly, target)
    assert torch.equal(got, expect)


def test_zero_draft_proposal_commits_one_token(evaluator_mod):
    """draft_token_count=0 (no speculation): verification must still
    commit exactly one token without needing draft_probs; under greedy
    decoding that token is deterministically the target argmax."""
    fake_target, _ = _make_fake_target(_TARGET_LOGITS)
    current = torch.tensor([[3]])
    result = _run_verify_once(
        evaluator_mod,
        fake_target=fake_target,
        verify_input_ids=current,
        draft_probs=None,
        draft_token_count=0,
        temperature=0.0,
    )
    assert result.accepted_draft_tokens == 0
    assert result.committed_tokens.shape == (1, 1)
    argmax_first = int(torch.tensor(_TARGET_LOGITS[0]).argmax())
    assert result.committed_tokens[0, 0].item() == argmax_first
