import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ScalarSeries = list[tuple[int, float]]

QWEN3_4B_EAGLE3_ACCEPTANCE_LENGTHS = {
    "gsm8k": 5.14,
    "math500": 4.62,
    "aime25": 3.92,
    "mbpp": 3.69,
    "humaneval": 4.16,
    "livecodebench": 3.77,
    "mt-bench": 2.39,
    "alpaca": 2.26,
    "arena-hard-v2": 2.55,
}

DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH",
    "aime25": "AIME25",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
    "livecodebench": "LCB",
    "mt-bench": "MT-Bench",
    "alpaca": "Alpaca",
    "arena-hard-v2": "Arena-Hard",
}


def load_scalars(tensorboard_dir: Path) -> dict[str, ScalarSeries]:
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        values_by_step = {
            int(event.step): float(event.value)
            for event in accumulator.Scalars(tag)
        }
        scalars[tag] = sorted(values_by_step.items())
    return scalars


def plot_series(
    *,
    scalars: dict[str, ScalarSeries],
    tags: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for tag in tags:
        series = scalars.get(tag, [])
        if not series:
            continue
        steps = [step for step, _ in series]
        values = [value for _, value in series]
        label = tag.removeprefix("train/")
        plt.plot(steps, values, label=label, linewidth=1.8)

    plt.title(title)
    plt.xlabel("Optimizer step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(scalars: dict[str, ScalarSeries], output_path: Path) -> None:
    summary = {}
    for tag, series in sorted(scalars.items()):
        if not series:
            continue
        step, value = series[-1]
        summary[tag] = {"step": step, "value": value}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def plot_eval_comparison(eval_json_path: Path, output_dir: Path) -> None:
    eval_payload = json.loads(eval_json_path.read_text())
    ministral_by_dataset = {
        row["dataset"]: float(row["acceptance_length"])
        for row in eval_payload["rows"]
    }
    datasets = [
        dataset
        for dataset in QWEN3_4B_EAGLE3_ACCEPTANCE_LENGTHS
        if dataset in ministral_by_dataset
    ]
    labels = [DATASET_LABELS[dataset] for dataset in datasets]
    ministral_values = [ministral_by_dataset[dataset] for dataset in datasets]
    qwen_values = [
        QWEN3_4B_EAGLE3_ACCEPTANCE_LENGTHS[dataset] for dataset in datasets
    ]
    x_positions = list(range(len(datasets)))
    bar_width = 0.38

    plt.figure(figsize=(10, 4.8))
    plt.bar(
        [position - bar_width / 2 for position in x_positions],
        ministral_values,
        width=bar_width,
        label="Ministral3-3B Eagle3",
    )
    plt.bar(
        [position + bar_width / 2 for position in x_positions],
        qwen_values,
        width=bar_width,
        label="Qwen3-4B Eagle3",
    )
    plt.xticks(x_positions, labels, rotation=30, ha="right")
    plt.ylabel("Accepted length")
    plt.title("Final accepted length by benchmark")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "eval_acceptance_comparison.png", dpi=180)
    plt.close()

    comparison_rows = []
    for dataset, ministral_value, qwen_value in zip(
        datasets, ministral_values, qwen_values, strict=True
    ):
        comparison_rows.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS[dataset],
                "ministral3_3b_eagle3": ministral_value,
                "qwen3_4b_eagle3": qwen_value,
                "delta": ministral_value - qwen_value,
            }
        )
    output_path = output_dir / "eval_acceptance_comparison.json"
    output_path.write_text(
        json.dumps(comparison_rows, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-json", type=Path, default=None)
    args = parser.parse_args()

    scalars = load_scalars(args.tensorboard_dir)

    plot_series(
        scalars=scalars,
        tags=["train/loss"],
        output_path=args.output_dir / "train_loss.png",
        title="Training loss",
        ylabel="Loss",
    )
    plot_series(
        scalars=scalars,
        tags=["train/tau_greedy", "train/tau_probabilistic"],
        output_path=args.output_dir / "train_tau.png",
        title="Expected accepted length proxies",
        ylabel="Tokens",
    )
    plot_series(
        scalars=scalars,
        tags=[f"train/accept_rate@{idx}" for idx in range(7)],
        output_path=args.output_dir / "train_accept_rates.png",
        title="Per-position acceptance rates",
        ylabel="Acceptance rate",
    )
    plot_series(
        scalars=scalars,
        tags=[f"train/accuracy@{idx}" for idx in range(7)],
        output_path=args.output_dir / "train_accuracies.png",
        title="Per-position token accuracies",
        ylabel="Accuracy",
    )
    write_summary(scalars, args.output_dir / "train_metric_summary.json")
    if args.eval_json is not None:
        plot_eval_comparison(args.eval_json, args.output_dir)


if __name__ == "__main__":
    main()
