import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_DATASETS = [
    "gsm8k",
    "math500",
    "aime25",
    "humaneval",
    "mbpp",
    "livecodebench",
    "mt-bench",
    "alpaca",
    "arena-hard-v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--extra-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    assert isinstance(data, dict), f"Expected JSON object in {path}"
    assert "rows" in data, f"Expected rows in {path}"
    return data


def row_dataset(row: dict[str, Any]) -> str:
    dataset = row["dataset"]
    assert isinstance(dataset, str), f"Expected string dataset, got {dataset!r}"
    return dataset


def merge_rows(*row_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            dataset = row_dataset(row)
            assert dataset not in by_dataset, f"Duplicate metrics for {dataset}"
            by_dataset[dataset] = row

    missing = [dataset for dataset in EXPECTED_DATASETS if dataset not in by_dataset]
    assert not missing, f"Missing datasets: {missing}"
    return [by_dataset[dataset] for dataset in EXPECTED_DATASETS]


def main() -> None:
    args = parse_args()
    base = load_json(args.base_json)
    extra = load_json(args.extra_json)

    for key in ["target_model", "draft_model", "step"]:
        assert base[key] == extra[key], (
            f"Mismatched {key}: base={base[key]!r}, extra={extra[key]!r}"
        )

    base_rows = base["rows"]
    extra_rows = extra["rows"]
    assert isinstance(base_rows, list), f"Expected list rows in {args.base_json}"
    assert isinstance(extra_rows, list), f"Expected list rows in {args.extra_json}"

    merged = dict(base)
    merged["rows"] = merge_rows(base_rows, extra_rows)
    merged["complete"] = True

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
