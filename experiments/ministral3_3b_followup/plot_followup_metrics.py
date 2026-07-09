import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ScalarSeries = list[tuple[int, float]]

TRAIN_RUNS = {
    "kl_baseline": {
        "label": "Eagle3 KL baseline",
        "tensorboard_dir": Path(
            "/mnt/vast/home/andy/tensorboard/deepspec/"
            "eagle3_ttt7_ministral3_3b_10k_3000steps"
        ),
    },
    "e2e_tv_ttt7": {
        "label": "Eagle3 e2e-TV TTT-7",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "eagle3_ttt7_ministral3_3b_e2e_tv"
        ),
    },
    "e2e_tv_ttt5": {
        "label": "Eagle3 e2e-TV TTT-5",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "eagle3_ttt5_ministral3_3b_e2e_tv"
        ),
    },
    "e2e_tv_from_kl": {
        "label": "Eagle3 KL -> e2e-TV",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "eagle3_ttt7_ministral3_3b_e2e_tv_from_kl"
        ),
    },
    "e2e_tv_from_kl_lr1e4": {
        "label": "Eagle3 KL -> e2e-TV lr=1e-4",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "eagle3_ttt7_ministral3_3b_e2e_tv_from_kl_lr1e4"
        ),
    },
    "e2e_tv_from_kl_lr1e4_8gpu": {
        "label": "Eagle3 KL -> e2e-TV lr=1e-4 8GPU",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "eagle3_ttt7_ministral3_3b_e2e_tv_from_kl_lr1e4_8gpu"
        ),
    },
    "dspark": {
        "label": "DSpark block-7",
        "tensorboard_dir": Path(
            "/mnt/vast/runs/andy/deepspec_tensorboard/deepspec/"
            "dspark_block7_ministral3_3b_8gpu"
        ),
    },
}

EVAL_JSONS = {
    "kl_baseline": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_eagle3/"
        "eval_10k_step3000_metrics.json"
    ),
    "e2e_tv_ttt7": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_eagle3_e2e_tv_step3000_metrics.json"
    ),
    "e2e_tv_ttt5": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_eagle3_e2e_tv_ttt5_step3000_metrics.json"
    ),
    "e2e_tv_from_kl": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_eagle3_e2e_tv_from_kl_step3000_metrics.json"
    ),
    "e2e_tv_from_kl_lr1e4": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_eagle3_e2e_tv_from_kl_lr1e4_step3000_metrics.json"
    ),
    "e2e_tv_from_kl_lr1e4_8gpu": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_eagle3_e2e_tv_from_kl_lr1e4_8gpu_step3000_metrics.json"
    ),
    "dspark": Path(
        "/mnt/vast/runs/andy/deepspec_ministral3_3b_followup/"
        "eval_dspark_8gpu_step3000_metrics.json"
    ),
}


def load_scalars(tensorboard_dir: Path) -> dict[str, ScalarSeries]:
    if not tensorboard_dir.exists():
        return {}
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        values_by_step = {
            int(event.step): float(event.value) for event in accumulator.Scalars(tag)
        }
        scalars[tag] = sorted(values_by_step.items())
    return scalars


def load_all_scalars() -> dict[str, dict[str, ScalarSeries]]:
    return {
        run_name: load_scalars(run["tensorboard_dir"])
        for run_name, run in TRAIN_RUNS.items()
    }


def latest_value(series: ScalarSeries) -> dict[str, float | int] | None:
    if not series:
        return None
    step, value = series[-1]
    return {"step": int(step), "value": float(value)}


def write_summary(
    all_scalars: dict[str, dict[str, ScalarSeries]],
    output_path: Path,
) -> None:
    summary = {}
    for run_name, scalars in sorted(all_scalars.items()):
        run_summary = {}
        for tag, series in sorted(scalars.items()):
            value = latest_value(series)
            if value is not None:
                run_summary[tag] = value
        summary[run_name] = run_summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def plot_metric(
    *,
    all_scalars: dict[str, dict[str, ScalarSeries]],
    tag: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8.5, 4.8))
    for run_name, scalars in all_scalars.items():
        series = scalars.get(tag, [])
        if not series:
            continue
        steps = [step for step, _ in series]
        values = [value for _, value in series]
        plt.plot(
            steps,
            values,
            label=str(TRAIN_RUNS[run_name]["label"]),
            linewidth=1.8,
        )

    plt.title(title)
    plt.xlabel("Optimizer step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_accept_rates(
    *,
    all_scalars: dict[str, dict[str, ScalarSeries]],
    output_path: Path,
) -> None:
    tags = ["train/accept_rate@0", "train/accept_rate@3", "train/accept_rate@6"]
    fig, axes = plt.subplots(1, len(tags), figsize=(13, 4.2), sharey=True)
    for ax, tag in zip(axes, tags, strict=True):
        for run_name, scalars in all_scalars.items():
            series = scalars.get(tag, [])
            if not series:
                continue
            ax.plot(
                [step for step, _ in series],
                [value for _, value in series],
                label=str(TRAIN_RUNS[run_name]["label"]),
                linewidth=1.6,
            )
        ax.set_title(tag.removeprefix("train/"))
        ax.set_xlabel("Optimizer step")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Acceptance rate")
    axes[-1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_eval_acceptance() -> dict[str, dict[str, float]]:
    values = {}
    for run_name, path in EVAL_JSONS.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        values[run_name] = {
            str(row["dataset"]): float(row["acceptance_length"])
            for row in payload["rows"]
        }
    return values


def plot_eval_acceptance(output_path: Path) -> None:
    eval_values = load_eval_acceptance()
    if len(eval_values) < 2:
        return

    datasets = sorted(set().union(*(values.keys() for values in eval_values.values())))
    x_positions = list(range(len(datasets)))
    run_names = [run_name for run_name in TRAIN_RUNS if run_name in eval_values]
    bar_width = min(0.8 / len(run_names), 0.25)

    plt.figure(figsize=(11, 5.2))
    for idx, run_name in enumerate(run_names):
        offset = (idx - (len(run_names) - 1) / 2) * bar_width
        values = [
            eval_values[run_name].get(dataset, 0.0)
            for dataset in datasets
        ]
        plt.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width,
            label=str(TRAIN_RUNS[run_name]["label"]),
        )

    plt.xticks(x_positions, datasets, rotation=30, ha="right")
    plt.ylabel("Accepted length")
    plt.title("Final accepted length by benchmark")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ministral3_3b_followup/report_assets"),
    )
    args = parser.parse_args()

    all_scalars = load_all_scalars()
    write_summary(all_scalars, args.output_dir / "train_metric_summary.json")
    plot_metric(
        all_scalars=all_scalars,
        tag="train/loss",
        output_path=args.output_dir / "train_loss.png",
        title="Training Loss",
        ylabel="Loss",
    )
    plot_metric(
        all_scalars=all_scalars,
        tag="train/tau_probabilistic",
        output_path=args.output_dir / "train_tau_probabilistic.png",
        title="Train-Side Probabilistic Accepted Length Proxy",
        ylabel="Tokens",
    )
    plot_accept_rates(
        all_scalars=all_scalars,
        output_path=args.output_dir / "train_accept_rates.png",
    )
    plot_eval_acceptance(args.output_dir / "eval_acceptance_comparison.png")


if __name__ == "__main__":
    main()
