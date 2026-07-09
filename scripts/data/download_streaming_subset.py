from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the first N Open-PerfectBlend rows as DeepSpec JSONL."
    )
    parser.add_argument("--dataset-name", default="mlabonne/open-perfectblend")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def normalize_conversations(row: dict, idx: int) -> dict:
    conversations = []
    for message in row["conversations"]:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            continue
        conversations.append({"role": role, "content": message["value"]})
    assert conversations, f"row {idx} produced no conversations."
    assert conversations[0]["role"] == "user", (
        f"row {idx} does not start with user: {conversations[0]['role']}"
    )
    return {"id": idx, "conversations": conversations}


def main() -> None:
    args = parse_args()
    assert args.sample_size > 0, f"sample_size must be positive, got {args.sample_size}"
    if args.output_path.exists():
        if args.skip_existing:
            print(f"skip existing output: {args.output_path}")
            return
        raise FileExistsError(f"Output JSONL already exists: {args.output_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
    written = 0
    with args.output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            converted = normalize_conversations(row, idx)
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.sample_size:
                break
    print(f"wrote {written} rows -> {args.output_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
