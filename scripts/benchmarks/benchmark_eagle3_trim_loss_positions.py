#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from types import SimpleNamespace

import torch
from transformers import AutoConfig

import deepspec.modeling.eagle3.loss as loss_module
from deepspec.modeling.eagle3.loss import compute_eagle3_loss
from deepspec.modeling.eagle3.qwen3.config import build_draft_config
from deepspec.modeling.eagle3.qwen3.modeling import Qwen3Eagle3Model


GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Eagle3 A-level supervised-position compaction."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--ttt-length", type=int, required=True)
    parser.add_argument("--densities", type=float, nargs="+", required=True)
    parser.add_argument("--layout", choices=("tail",), required=True)
    parser.add_argument("--warmup-pairs", type=int, required=True)
    parser.add_argument("--measure-pairs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_model(args, device):
    target_config = AutoConfig.from_pretrained(
        args.config_path,
        local_files_only=True,
    )
    model_args = SimpleNamespace(
        target_model_name_or_path=args.config_path,
        target_layer_ids=[1, 9, 17, 25, 33],
        ttt_length=args.ttt_length,
        step_loss_decay=0.8,
        draft_num_hidden_layers=1,
        trim_loss_positions=False,
    )
    draft_config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    model = Qwen3Eagle3Model(draft_config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    model.embed_tokens.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.train()
    return model


def build_inputs(args, model, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_size = int(model.config.hidden_size)
    num_target_layers = len(model.target_layer_ids)
    vocab_size = int(model.config.vocab_size)
    return {
        "input_ids": torch.randint(
            0,
            vocab_size,
            (1, args.seq_len),
            generator=generator,
            device=device,
        ),
        "attention_mask": torch.ones(
            1, args.seq_len, dtype=torch.long, device=device
        ),
        "target_hidden_states": torch.randn(
            1,
            args.seq_len,
            num_target_layers * hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "target_last_hidden_states": torch.randn(
            1,
            args.seq_len,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
    }


def build_loss_mask(seq_len: int, density: float, device):
    assert 0.0 < density <= 1.0
    # The final valid token has no in-sequence next-token teacher. Construct an
    # exact contiguous tail over the remaining L-1 eligible positions.
    eligible = seq_len - 1
    supervised = max(1, int(round(eligible * density)))
    start = eligible - supervised
    mask = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    mask[:, start:eligible] = 1
    return mask, supervised


def run_once(*, model, batch, ttt_length: int, trim: bool):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = compute_eagle3_loss(
        model=model,
        batch=batch,
        ttt_length=ttt_length,
        step_loss_decay=0.8,
        trim_loss_positions=trim,
    )
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    peak_bytes = int(torch.cuda.max_memory_allocated())
    loss_value = float(loss.detach().float().item())
    del loss
    model.zero_grad(set_to_none=True)
    return {
        "time_ms": elapsed_ms,
        "peak_allocated_bytes": peak_bytes,
        "loss": loss_value,
    }


def run_pairs(*, model, batch, ttt_length: int, count: int, offset: int):
    pairs = []
    for pair_idx in range(count):
        full_first = (pair_idx + offset) % 2 == 0
        order = (False, True) if full_first else (True, False)
        observations = {}
        for trim in order:
            observations["trim" if trim else "full"] = run_once(
                model=model,
                batch=batch,
                ttt_length=ttt_length,
                trim=trim,
            )
        pairs.append(
            {
                "order": ["full", "trim"] if full_first else ["trim", "full"],
                "full": observations["full"],
                "trim": observations["trim"],
                "speedup": observations["full"]["time_ms"]
                / observations["trim"]["time_ms"],
            }
        )
    return pairs


def summarize(pairs):
    full_times = [pair["full"]["time_ms"] for pair in pairs]
    trim_times = [pair["trim"]["time_ms"] for pair in pairs]
    speedups = [pair["speedup"] for pair in pairs]
    full_peaks = [pair["full"]["peak_allocated_bytes"] for pair in pairs]
    trim_peaks = [pair["trim"]["peak_allocated_bytes"] for pair in pairs]
    return {
        "paired_median_speedup": statistics.median(speedups),
        "median_full_time_ms": statistics.median(full_times),
        "median_trim_time_ms": statistics.median(trim_times),
        "median_full_peak_allocated_bytes": int(statistics.median(full_peaks)),
        "median_trim_peak_allocated_bytes": int(statistics.median(trim_peaks)),
        "median_peak_reduction_bytes": int(
            statistics.median(full_peaks) - statistics.median(trim_peaks)
        ),
    }


def validate_results(results):
    by_density = {entry["density"]: entry for entry in results}
    required = (0.3, 0.6, 0.9)
    assert all(density in by_density for density in required)
    for entry in results:
        for pair in entry["measured_pairs"]:
            assert math.isfinite(pair["full"]["loss"])
            assert math.isfinite(pair["trim"]["loss"])

    d30 = by_density[0.3]["summary"]
    d60 = by_density[0.6]["summary"]
    d90 = by_density[0.9]["summary"]
    assert d30["paired_median_speedup"] >= 1.05
    assert d30["median_peak_reduction_bytes"] >= GIB
    assert d60["paired_median_speedup"] >= 1.02
    assert d60["median_peak_reduction_bytes"] > 0
    assert d90["paired_median_speedup"] >= 0.98
    assert (
        d30["paired_median_speedup"]
        >= d60["paired_median_speedup"]
        >= d90["paired_median_speedup"]
    )


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loss_module.add_metric = lambda *unused_args, **unused_kwargs: None

    model = build_model(args, device)
    base_batch = build_inputs(args, model, device)
    results = []
    for density_idx, density in enumerate(args.densities):
        loss_mask, supervised = build_loss_mask(args.seq_len, density, device)
        batch = dict(base_batch)
        batch["loss_mask"] = loss_mask
        run_pairs(
            model=model,
            batch=batch,
            ttt_length=args.ttt_length,
            count=args.warmup_pairs,
            offset=density_idx,
        )
        measured_pairs = run_pairs(
            model=model,
            batch=batch,
            ttt_length=args.ttt_length,
            count=args.measure_pairs,
            offset=density_idx + args.warmup_pairs,
        )
        entry = {
            "density": float(density),
            "effective_supervised_rows": supervised,
            "measured_pairs": measured_pairs,
            "summary": summarize(measured_pairs),
        }
        results.append(entry)
        print(
            f"density={density:.2f} "
            f"speedup={entry['summary']['paired_median_speedup']:.4f}x "
            "peak_reduction_gib="
            f"{entry['summary']['median_peak_reduction_bytes'] / GIB:.3f}",
            flush=True,
        )

    payload = {
        "config": {
            "config_path": args.config_path,
            "seq_len": args.seq_len,
            "ttt_length": args.ttt_length,
            "densities": args.densities,
            "layout": args.layout,
            "warmup_pairs": args.warmup_pairs,
            "measure_pairs": args.measure_pairs,
            "seed": args.seed,
            "dtype": "bfloat16",
            "device_name": torch.cuda.get_device_name(device),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    validate_results(results)
    print(f"BENCHMARK_GATES_OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
