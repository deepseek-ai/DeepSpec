from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import deepspec.modeling.eagle3.loss as loss_module
from deepspec.modeling.eagle3.loss import (
    _build_compact_plan,
    _build_next_token_position_mask,
    _shift_with_zero_padding,
    compute_eagle3_loss,
)
from deepspec.modeling.eagle3.qwen3.modeling import Qwen3Eagle3Model
from deepspec.modeling.eagle3.qwen3.config import build_draft_config
from deepspec.utils.config import load_config, parse_opts_to_config


def _position_masks(loss_mask: torch.Tensor, attention_mask: torch.Tensor, k: int):
    shifted = _build_next_token_position_mask(
        loss_mask=loss_mask,
        attention_mask=attention_mask,
    )
    masks = []
    for _ in range(k):
        masks.append(shifted)
        shifted = _shift_with_zero_padding(
            shifted.squeeze(-1), left=False
        ).unsqueeze(-1)
    return masks


def test_compact_plan_exact_indices():
    loss_mask = torch.tensor(
        [
            [1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    plan = _build_compact_plan(
        position_masks=_position_masks(loss_mask, attention_mask, k=4)
    )

    expected_draft = (
        [0, 2, 5, 8, 10],
        [1, 4, 7, 9],
        [0, 3, 8],
        [2, 7],
    )
    expected_teacher = (
        [1, 3, 6, 9, 11],
        [3, 6, 9, 11],
        [3, 6, 11],
        [6, 11],
    )
    assert [step.num_valid_rows for step in plan.steps] == [5, 4, 3, 2]
    for step, draft_indices, teacher_indices in zip(
        plan.steps, expected_draft, expected_teacher, strict=True
    ):
        assert step.draft_indices.tolist() == draft_indices
        assert step.teacher_indices.tolist() == teacher_indices


def test_trim_config_is_opt_in():
    target_config = _tiny_config()
    target_config.num_hidden_layers = 5
    base_args = dict(
        target_model_name_or_path="tiny-local-fixture",
        target_layer_ids=[0, 1, 2, 3, 4],
        ttt_length=3,
        step_loss_decay=0.8,
        draft_num_hidden_layers=1,
    )
    default_config = build_draft_config(
        target_config=target_config,
        model_args=SimpleNamespace(**base_args),
    )
    enabled_config = build_draft_config(
        target_config=target_config,
        model_args=SimpleNamespace(**base_args, trim_loss_positions=True),
    )
    assert default_config.trim_loss_positions is False
    assert enabled_config.trim_loss_positions is True


def test_public_config_accepts_trim_override():
    config = load_config("config/eagle3/eagle3_qwen3_8b.py")
    assert config.model.trim_loss_positions is False
    overridden = parse_opts_to_config(
        ["model.trim_loss_positions=true"],
        config,
    )
    assert overridden.model.trim_loss_positions is True


def _tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
    )
    config.target_layer_ids = [0, 1, 2, 3, 4]
    config.ttt_length = 3
    config.step_loss_decay = 0.8
    config.trim_loss_positions = False
    config._attn_implementation = "flex_attention"
    return config


def _tiny_model(device: torch.device) -> Qwen3Eagle3Model:
    model = Qwen3Eagle3Model(_tiny_config()).to(
        device=device, dtype=torch.bfloat16
    )
    model.embed_tokens.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.train()
    return model


def _tiny_batch(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    batch_size, seq_len, hidden_size = 1, 16, 32
    loss_mask = torch.zeros(
        batch_size, seq_len, dtype=torch.long, device=device
    )
    loss_mask[0, [0, 2, 5, 7, 13]] = 1
    return {
        "input_ids": torch.randint(
            0,
            97,
            (batch_size, seq_len),
            generator=generator,
            device=device,
        ),
        "attention_mask": torch.ones(
            batch_size, seq_len, dtype=torch.long, device=device
        ),
        "loss_mask": loss_mask,
        "target_hidden_states": torch.randn(
            batch_size,
            seq_len,
            5 * hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "target_last_hidden_states": torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
    }


def _capture_metrics():
    metrics = {}

    def capture(name, value, *, den=None, reduction="dp_sum", tag="train"):
        del reduction
        numerator = float(torch.as_tensor(value).detach().float().item())
        if den is None:
            resolved = numerator
        else:
            denominator = float(torch.as_tensor(den).detach().float().item())
            resolved = 0.0 if denominator == 0.0 else numerator / denominator
        metrics[f"{tag}/{name}"] = resolved

    return metrics, capture


def _run_tiny_arm(
    *,
    state_dict: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    trim_loss_positions: bool,
    wrong_compact_indices: bool = False,
):
    device = batch["input_ids"].device
    model = _tiny_model(device)
    model.load_state_dict(state_dict)
    metrics, capture_metric = _capture_metrics()

    original_build_plan = loss_module._build_compact_plan

    def build_wrong_plan(*args, **kwargs):
        plan = original_build_plan(*args, **kwargs)
        max_index = plan.batch_size * plan.seq_len
        wrong_steps = tuple(
            replace(
                step,
                draft_indices=(step.draft_indices + 1).remainder(max_index),
            )
            for step in plan.steps
        )
        return replace(plan, steps=wrong_steps)

    plan_patch = (
        mock.patch.object(loss_module, "_build_compact_plan", build_wrong_plan)
        if wrong_compact_indices
        else mock.patch.object(
            loss_module, "_build_compact_plan", original_build_plan
        )
    )
    with mock.patch.object(loss_module, "add_metric", capture_metric), plan_patch:
        loss = compute_eagle3_loss(
            model=model,
            batch=batch,
            ttt_length=3,
            step_loss_decay=0.8,
            trim_loss_positions=trim_loss_positions,
        )
        loss.backward()

    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    return float(loss.detach().float().item()), gradients, metrics


def _within_loss_tolerance(a: float, b: float) -> bool:
    return abs(a - b) <= 5e-3 * max(abs(a), abs(b)) + 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trim_matches_full_loss_grad_and_metrics_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    template = _tiny_model(device)
    # Make the row-shift negative control observably wrong under the same
    # bf16 tolerance used for the equivalence arm. The production head is
    # frozen too; scaling this tiny fixture changes sensitivity, not semantics.
    with torch.no_grad():
        template.lm_head.weight.mul_(16.0)
    state_dict = copy.deepcopy(template.state_dict())
    del template
    batch = _tiny_batch(device)

    full_loss, full_gradients, full_metrics = _run_tiny_arm(
        state_dict=state_dict,
        batch=batch,
        trim_loss_positions=False,
    )
    trim_loss, trim_gradients, trim_metrics = _run_tiny_arm(
        state_dict=state_dict,
        batch=batch,
        trim_loss_positions=True,
    )

    assert _within_loss_tolerance(full_loss, trim_loss)
    assert full_gradients.keys() == trim_gradients.keys()
    for name in full_gradients:
        torch.testing.assert_close(
            full_gradients[name],
            trim_gradients[name],
            rtol=5e-3,
            atol=5e-3,
            msg=lambda message, name=name: f"gradient {name}: {message}",
        )
    assert full_metrics.keys() == trim_metrics.keys()
    for name in full_metrics:
        assert _within_loss_tolerance(full_metrics[name], trim_metrics[name]), (
            name,
            full_metrics[name],
            trim_metrics[name],
        )

    wrong_loss, wrong_gradients, _ = _run_tiny_arm(
        state_dict=state_dict,
        batch=batch,
        trim_loss_positions=True,
        wrong_compact_indices=True,
    )
    negative_control_detected = not _within_loss_tolerance(full_loss, wrong_loss)
    if not negative_control_detected:
        for name in full_gradients:
            try:
                torch.testing.assert_close(
                    full_gradients[name],
                    wrong_gradients[name],
                    rtol=5e-3,
                    atol=5e-3,
                )
            except AssertionError:
                negative_control_detected = True
                break
    assert negative_control_detected


def _run_scope_trace(*, trim_loss_positions: bool):
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _tiny_model(device)
    batch = _tiny_batch(device)
    q_lengths = []
    lm_head_lengths = []
    cache_lengths = []

    q_hook = model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _module, args: q_lengths.append(int(args[0].shape[1]))
    )
    lm_head_hook = model.lm_head.register_forward_pre_hook(
        lambda _module, args: lm_head_lengths.append(int(args[0].shape[1]))
    )

    cache = DynamicCache()
    original_update = cache.update

    def recording_update(key_states, value_states, layer_idx, cache_kwargs=None):
        output = original_update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        cache_lengths.append(int(cache.get_seq_length(layer_idx)))
        return output

    cache.update = recording_update
    try:
        with mock.patch.object(loss_module, "DynamicCache", return_value=cache), mock.patch.object(
            loss_module, "add_metric", lambda *args, **kwargs: None
        ):
            compute_eagle3_loss(
                model=model,
                batch=batch,
                ttt_length=3,
                step_loss_decay=0.8,
                trim_loss_positions=trim_loss_positions,
            )
    finally:
        q_hook.remove()
        lm_head_hook.remove()
    return q_lengths, lm_head_lengths, cache_lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trim_scope_keeps_backbone_and_teacher_projection_full_cuda():
    q_lengths, lm_head_lengths, cache_lengths = _run_scope_trace(
        trim_loss_positions=True
    )
    assert q_lengths == [16, 16, 16]
    assert lm_head_lengths == [16, 5, 4, 4]
    assert cache_lengths == [16, 32, 48]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trim_default_false_keeps_full_logits():
    q_lengths, lm_head_lengths, cache_lengths = _run_scope_trace(
        trim_loss_positions=False
    )
    assert q_lengths == [16, 16, 16]
    assert lm_head_lengths == [16, 16, 16, 16]
    assert cache_lengths == [16, 32, 48]
