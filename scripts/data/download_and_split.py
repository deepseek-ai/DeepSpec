from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_DATASET_NAME = "mlabonne/open-perfectblend"
DEFAULT_TRAIN_OUTPUT_PATH = Path("cache/dataset/perfectblend_train.jsonl")
DEFAULT_TEST_OUTPUT_DIR = Path("eval_datasets")
DEFAULT_TEST_OUTPUT_NAME = "perfectblend.jsonl"
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PerfectBlend and split it into train and eval JSONL files."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to load.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Process only the first N source rows before splitting.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="Fraction held out for eval, matching SpecForge's default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by train_test_split.",
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_PATH,
        help="Output path for training conversations JSONL.",
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_DIR,
        help="Directory for the eval turns JSONL file.",
    )
    parser.add_argument(
        "--test-output-name",
        default=DEFAULT_TEST_OUTPUT_NAME,
        help="Eval turns JSONL filename under --test-output-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only when both expected output files already exist.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be in the range (0, 1).")


def test_output_path(args: argparse.Namespace) -> Path:
    return args.test_output_dir / args.test_output_name


def check_output_paths(args: argparse.Namespace) -> bool:
    output_paths = [args.train_output_path, test_output_path(args)]
    existing_paths = [path for path in output_paths if path.exists()]
    if not existing_paths:
        return False
    if args.skip_existing and len(existing_paths) == len(output_paths):
        for path in output_paths:
            print(f"skip existing output: {path}")
        return True

    joined_paths = ", ".join(str(path) for path in existing_paths)
    raise FileExistsError(f"Output JSONL already exists: {joined_paths}")


def add_index(row: dict, idx: int) -> dict:
    row["id"] = idx
    return row


def load_source_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))
    dataset = dataset.map(add_index, with_indices=True)
    return dataset


def normalize_conversations(row: dict) -> dict:
    conversations = []
    for message in row["conversations"]:
        if message["from"] not in ROLE_MAPPING:
            continue
        conversations.append(
            {
                "role": ROLE_MAPPING[message["from"]],
                "content": message["value"],
            }
        )
    return {"id": row["id"], "conversations": conversations}


def validate_conversations(row: dict, row_number: int) -> None:
    conversations = row["conversations"]
    if not conversations:
        raise ValueError(f"row {row_number} produced no conversations.")
    if conversations[0]["role"] != "user":
        raise ValueError(f"row {row_number} does not start with a user message.")
    for message in conversations:
        if message["role"] not in {"user", "assistant"}:
            raise ValueError(f"row {row_number} has invalid role: {message['role']}")
        if not isinstance(message["content"], str) or not message["content"]:
            raise ValueError(f"row {row_number} has invalid message content.")


def user_turns(row: dict) -> list[str]:
    return [
        message["content"]
        for message in row["conversations"]
        if message["role"] == "user"
    ]


def write_train_jsonl(dataset, handle) -> int:
    count = 0
    for row_number, row in enumerate(dataset, start=1):
        converted = normalize_conversations(row)
        validate_conversations(converted, row_number)
        handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
        count += 1
    handle.flush()
    os.fsync(handle.fileno())
    return count


def write_eval_jsonl(dataset, handle) -> int:
    count = 0
    for row_number, row in enumerate(dataset, start=1):
        converted = normalize_conversations(row)
        validate_conversations(converted, row_number)
        turns = user_turns(converted)
        handle.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
        count += 1
    handle.flush()
    os.fsync(handle.fileno())
    return count


def create_temporary_output(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    current_umask = os.umask(0)
    os.umask(current_umask)
    os.fchmod(fd, 0o666 & ~current_umask)
    return Path(temporary_path), os.fdopen(fd, "w", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        if check_output_paths(args):
            return
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None

    dataset = load_source_dataset(args)
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    eval_output_path = test_output_path(args)
    train_temporary_path = None
    eval_temporary_path = None
    train_handle = None
    eval_handle = None
    try:
        train_temporary_path, train_handle = create_temporary_output(
            args.train_output_path
        )
        eval_temporary_path, eval_handle = create_temporary_output(eval_output_path)
        with train_handle:
            train_count = write_train_jsonl(split_dataset["train"], train_handle)
        with eval_handle:
            test_count = write_eval_jsonl(split_dataset["test"], eval_handle)
        os.replace(train_temporary_path, args.train_output_path)
        train_temporary_path = None
        os.replace(eval_temporary_path, eval_output_path)
        eval_temporary_path = None
    finally:
        for handle in (train_handle, eval_handle):
            if handle is not None and not handle.closed:
                handle.close()
        for temporary_path in (train_temporary_path, eval_temporary_path):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    print(f"wrote train split: {train_count} rows -> {args.train_output_path}")
    print(f"wrote eval split: {test_count} rows -> {eval_output_path}")


if __name__ == "__main__":
    main()
