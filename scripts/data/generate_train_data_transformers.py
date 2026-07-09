from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["USE_HUB_KERNELS"] = "0"

import torch
from tqdm import tqdm

from deepspec.data.parser import encode_chat_messages
from deepspec.modeling.target_utils import load_target_causal_lm, load_target_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate DeepSpec training conversations with a local HF model."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file-path", type=Path, required=True)
    parser.add_argument("--output-file-path", type=Path, required=True)
    parser.add_argument("--error-file-path", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_shard_rows(path: Path, shard_index: int, num_shards: int):
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index % num_shards != shard_index:
                continue
            yield line_index, json.loads(line)


def generate_assistant(
    *,
    model,
    tokenizer,
    messages: list[dict],
    args: argparse.Namespace,
) -> str:
    input_ids = encode_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    do_sample = float(args.temperature) > 0.0
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": do_sample,
        "max_new_tokens": int(args.max_new_tokens),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            }
        )
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    generated_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def regenerate_sample(*, model, tokenizer, sample: dict, args: argparse.Namespace) -> dict:
    regenerated = []
    for message in sample["conversations"]:
        role = message["role"]
        if role == "system":
            regenerated.append(message)
            continue
        if role == "assistant":
            continue
        assert role == "user", f"Unsupported message role: {role}"
        regenerated.append(message)
        regenerated.append(
            {
                "role": "assistant",
                "content": generate_assistant(
                    model=model,
                    tokenizer=tokenizer,
                    messages=regenerated,
                    args=args,
                ),
            }
        )
    sample = dict(sample)
    sample["conversations"] = regenerated
    sample["status"] = "success"
    return sample


def main() -> None:
    args = parse_args()
    assert 0 <= args.shard_index < args.num_shards, (
        f"Expected shard_index in [0, {args.num_shards}), got {args.shard_index}"
    )
    assert args.max_new_tokens > 0, (
        f"max_new_tokens must be positive, got {args.max_new_tokens}"
    )
    args.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.error_file_path
    if error_path is None:
        error_path = args.output_file_path.with_name(
            args.output_file_path.stem + "_error.jsonl"
        )
    error_path.parent.mkdir(parents=True, exist_ok=True)

    completed = count_lines(args.output_file_path) + count_lines(error_path)
    if not args.resume:
        completed = 0

    tokenizer = load_target_tokenizer(args.model)
    model = load_target_causal_lm(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device="cuda").eval()

    total_seen = 0
    success = 0
    errors = 0
    start_time = perf_counter()
    output_mode = "a" if args.resume else "w"
    with (
        args.output_file_path.open(output_mode, encoding="utf-8") as output_handle,
        error_path.open(output_mode, encoding="utf-8") as error_handle,
    ):
        rows = iter_shard_rows(args.input_file_path, args.shard_index, args.num_shards)
        progress = tqdm(rows, desc=f"shard {args.shard_index}/{args.num_shards}")
        for _line_index, sample in progress:
            if total_seen < completed:
                total_seen += 1
                continue
            if args.max_samples is not None and success + errors >= args.max_samples:
                break
            try:
                regenerated = regenerate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    args=args,
                )
            except Exception as exc:
                sample = dict(sample)
                sample["status"] = "error"
                sample["error"] = str(exc)
                error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                error_handle.flush()
                errors += 1
            else:
                output_handle.write(json.dumps(regenerated, ensure_ascii=False) + "\n")
                output_handle.flush()
                success += 1
            total_seen += 1
            elapsed = max(perf_counter() - start_time, 1e-6)
            progress.set_postfix(success=success, errors=errors, samples_per_s=success / elapsed)

    print(f"success={success} errors={errors} output={args.output_file_path}")


if __name__ == "__main__":
    main()
