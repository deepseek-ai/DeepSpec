from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed as dist

from deepspec.eval.base_evaluator import VerificationResult
from deepspec.utils import jsonable


EPS_PROB = 1e-8
RELIABILITY_PLOT_FILENAME = "reliability_diagram.png"


class ConfidenceProposal(Protocol):
    draft_token_count: int
    confidence_logits: torch.Tensor | None


def model_display_name(path: str) -> str:
    normalized = path.rstrip("/")
    return os.path.basename(normalized) or normalized


def confidence_metric_dtype(device: torch.device) -> torch.dtype:
    device = torch.device(device)
    return torch.float32 if device.type == "npu" else torch.float64


class PerPositionConfidenceMetrics:
    """Per-position ECE + AUROC + Brier for cumprod predictions."""

    def __init__(
        self,
        *,
        block_size: int,
        num_coarse_bins: int,
        num_fine_bins: int,
        device: torch.device,
    ):
        self.block_size = int(block_size)
        self.num_coarse_bins = int(num_coarse_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.dtype = confidence_metric_dtype(device)
        self.coarse_count = torch.zeros(
            (self.block_size, self.num_coarse_bins),
            dtype=self.dtype,
            device=device,
        )
        self.coarse_pred = torch.zeros_like(self.coarse_count)
        self.coarse_target = torch.zeros_like(self.coarse_count)
        self.fine_pos = torch.zeros(
            (self.block_size, self.num_fine_bins),
            dtype=self.dtype,
            device=device,
        )
        self.fine_neg = torch.zeros_like(self.fine_pos)
        self.brier_num = torch.zeros(
            self.block_size,
            dtype=self.dtype,
            device=device,
        )

    def update(
        self,
        *,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        probs = probs.reshape(-1).to(self.dtype).clamp(EPS_PROB, 1.0 - EPS_PROB)
        targets = targets.reshape(-1).to(self.dtype)
        assert probs.shape == targets.shape
        pos_count = probs.numel()
        assert pos_count <= self.block_size
        weights = torch.ones_like(probs, dtype=self.dtype)
        pos_idx = torch.arange(pos_count, device=probs.device)

        coarse_idx = (
            probs * self.num_coarse_bins
        ).long().clamp_(0, self.num_coarse_bins - 1)
        flat_coarse = pos_idx * self.num_coarse_bins + coarse_idx
        self.coarse_count.view(-1).scatter_add_(0, flat_coarse, weights)
        self.coarse_pred.view(-1).scatter_add_(0, flat_coarse, probs * weights)
        self.coarse_target.view(-1).scatter_add_(0, flat_coarse, targets * weights)

        fine_idx = (
            probs * self.num_fine_bins
        ).long().clamp_(0, self.num_fine_bins - 1)
        flat_fine = pos_idx * self.num_fine_bins + fine_idx
        self.fine_pos.view(-1).scatter_add_(0, flat_fine, weights * targets)
        self.fine_neg.view(-1).scatter_add_(0, flat_fine, weights * (1.0 - targets))

        self.brier_num[:pos_count].add_(weights * (probs - targets).pow(2))

    def all_reduce(self) -> None:
        for tensor in (
            self.coarse_count,
            self.coarse_pred,
            self.coarse_target,
            self.fine_pos,
            self.fine_neg,
            self.brier_num,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    @staticmethod
    def _auroc_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
        total_pos = float(pos_hist.sum().item())
        total_neg = float(neg_hist.sum().item())
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan")
        cum_neg = torch.cumsum(neg_hist, dim=0)
        cum_neg_before = cum_neg - neg_hist
        pair = (pos_hist * cum_neg_before).sum() + 0.5 * (pos_hist * neg_hist).sum()
        return float(pair.item() / (total_pos * total_neg))

    def compute(self) -> list[dict]:
        out = []
        for pos in range(self.block_size):
            weights = self.coarse_count[pos]
            total = float(weights.sum().item())
            if total <= 1e-12:
                out.append(
                    {
                        "position": pos,
                        "total_weight": 0.0,
                        "ece": float("nan"),
                        "auc": float("nan"),
                        "brier": float("nan"),
                        "pred_mean": float("nan"),
                        "target_mean": float("nan"),
                        "reliability": [],
                    }
                )
                continue

            denom = weights.clamp_min(1e-12)
            avg_pred = self.coarse_pred[pos] / denom
            avg_target = self.coarse_target[pos] / denom
            bin_err = (avg_pred - avg_target).abs()
            ece = float((bin_err * weights).sum().item() / total)
            auc = self._auroc_from_hist(self.fine_pos[pos], self.fine_neg[pos])
            brier = float(self.brier_num[pos].item()) / total
            reliability = []
            for bin_idx in range(self.num_coarse_bins):
                weight = float(weights[bin_idx].item())
                if weight <= 0.0:
                    continue
                reliability.append(
                    {
                        "bin": bin_idx,
                        "range": [
                            bin_idx / self.num_coarse_bins,
                            (bin_idx + 1) / self.num_coarse_bins,
                        ],
                        "avg_pred": float(avg_pred[bin_idx].item()),
                        "avg_target": float(avg_target[bin_idx].item()),
                        "weight": weight,
                    }
                )
            out.append(
                {
                    "position": pos,
                    "total_weight": total,
                    "ece": ece,
                    "auc": auc,
                    "brier": brier,
                    "pred_mean": float(self.coarse_pred[pos].sum().item()) / total,
                    "target_mean": float(self.coarse_target[pos].sum().item()) / total,
                    "reliability": reliability,
                }
            )
        return out


def summarize_confidence_row(row: dict) -> dict:
    total_w = 0.0
    weighted_ece = 0.0
    weighted_brier = 0.0
    weighted_pred = 0.0
    weighted_target = 0.0
    auc_w = 0.0
    weighted_auc = 0.0

    for entry in row.get("per_position") or []:
        weight = float(entry["total_weight"])
        if weight <= 0.0:
            continue
        total_w += weight
        weighted_ece += float(entry["ece"]) * weight
        weighted_brier += float(entry["brier"]) * weight
        weighted_pred += float(entry["pred_mean"]) * weight
        weighted_target += float(entry["target_mean"]) * weight
        auc = float(entry["auc"])
        if not math.isnan(auc):
            auc_w += weight
            weighted_auc += auc * weight

    if total_w <= 0.0:
        return {
            "ece_mean": float("nan"),
            "auc_mean": float("nan"),
            "brier_mean": float("nan"),
            "pred_mean": float("nan"),
            "target_mean": float("nan"),
        }
    return {
        "ece_mean": weighted_ece / total_w,
        "auc_mean": weighted_auc / auc_w if auc_w > 0.0 else float("nan"),
        "brier_mean": weighted_brier / total_w,
        "pred_mean": weighted_pred / total_w,
        "target_mean": weighted_target / total_w,
    }


def format_float(value, digits=4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def compact_row(row: dict) -> dict:
    out = {key: value for key, value in row.items() if key != "per_position"}
    out.update(summarize_confidence_row(row))
    out["per_position"] = [
        {key: value for key, value in entry.items() if key != "reliability"}
        for entry in row.get("per_position") or []
    ]
    return out


def plot_reliability_diagram(dataset_dir: Path, row: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_position = row["per_position"]
    num_positions = max(len(per_position), 1)
    ncols = min(3, num_positions)
    nrows = (num_positions + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for idx, entry in enumerate(per_position):
        ax = axes[idx // ncols][idx % ncols]
        reliability = entry.get("reliability") or []
        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.7", linewidth=1.0)
        if reliability:
            centers = [
                0.5 * (float(bin_row["range"][0]) + float(bin_row["range"][1]))
                for bin_row in reliability
            ]
            widths = [
                float(bin_row["range"][1]) - float(bin_row["range"][0])
                for bin_row in reliability
            ]
            avg_pred = [float(bin_row["avg_pred"]) for bin_row in reliability]
            avg_target = [float(bin_row["avg_target"]) for bin_row in reliability]
            weights = [float(bin_row["weight"]) for bin_row in reliability]
            ax.plot(avg_pred, avg_target, "o-", color="C1", linewidth=1.5, markersize=4)
            ax2 = ax.twinx()
            ax2.bar(
                centers,
                weights,
                width=widths,
                color="C0",
                alpha=0.2,
                edgecolor="white",
                linewidth=0.5,
            )
            max_weight = max(weights) if weights else 0.0
            ax2.set_ylim(0.0, max_weight if max_weight > 0.0 else 1.0)
            ax2.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted prefix acceptance")
        ax.set_ylabel("Observed prefix acceptance")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_title(
            "pos={pos}  ECE={ece:.4f}  AUC={auc:.4f}\nmean_pred={pred:.4f}  "
            "mean_target={target:.4f}".format(
                pos=int(entry["position"]),
                ece=float(entry["ece"]),
                auc=float(entry["auc"]),
                pred=float(entry["pred_mean"]),
                target=float(entry["target_mean"]),
            ),
            fontsize=10,
        )

    for idx in range(len(per_position), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{row['dataset']} confidence-head reliability", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = dataset_dir / RELIABILITY_PLOT_FILENAME
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def add_tensorboard_scalar(writer, tag: str, value, step: int) -> None:
    value = float(value)
    if math.isfinite(value):
        writer.add_scalar(tag, value, step)


class ConfidenceHeadRecorder:
    def __init__(
        self,
        *,
        device: torch.device,
        max_proposal_tokens: int,
        num_bins: int,
        num_fine_bins: int,
        draft_name_or_path: str,
        tensorboard_dir: str | None,
        step: int | None,
        artifact_root: Path | None,
    ):
        self.device = device
        self.max_proposal_tokens = int(max_proposal_tokens)
        self.num_bins = int(num_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.draft_name_or_path = draft_name_or_path
        self.tensorboard_dir = tensorboard_dir
        self.step = step
        self.artifact_root = artifact_root
        self.dataset_metrics: PerPositionConfidenceMetrics | None = None
        self.rows: list[dict] = []

    def start(self) -> None:
        self.dataset_metrics = PerPositionConfidenceMetrics(
            block_size=self.max_proposal_tokens,
            num_coarse_bins=self.num_bins,
            num_fine_bins=self.num_fine_bins,
            device=self.device,
        )

    def observe(
        self,
        *,
        proposal: ConfidenceProposal,
        verification: VerificationResult,
    ) -> None:
        assert self.dataset_metrics is not None
        if int(proposal.draft_token_count) <= 0:
            return
        effective_length = int(verification.effective_proposal_length)
        if effective_length <= 0:
            return
        confidence_logits = proposal.confidence_logits
        assert confidence_logits is not None
        assert verification.accept_prefix_mask is not None
        # BaseEvaluator.generate_one_sample enforces bsz=1, so this removes
        # only the single-sequence batch dimension. Truncate to
        # effective_proposal_length to skip positions past an accepted EOS.
        step_probs = torch.sigmoid(
            confidence_logits[:, :effective_length]
        ).squeeze(0)
        cumprod_pred = step_probs.to(self.dataset_metrics.dtype).cumprod(dim=0)
        prefix_label = (
            verification.accept_prefix_mask[:, :effective_length]
            .squeeze(0)
            .to(self.dataset_metrics.dtype)
        )
        self.dataset_metrics.update(
            probs=cumprod_pred,
            targets=prefix_label,
        )

    def finish(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
    ) -> dict | None:
        assert self.dataset_metrics is not None
        dataset_metrics = self.dataset_metrics
        dataset_metrics.all_reduce()
        self.dataset_metrics = None

        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) == 0:
            return None

        row = self.build_dataset_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
            confidence_metrics=dataset_metrics,
        )
        self.rows.append(row)
        return row

    def build_dataset_row(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | list[int]],
        confidence_metrics: PerPositionConfidenceMetrics,
    ) -> dict:
        return {
            "dataset": dataset_name,
            "sample_count": int(metric_summary["sample_count"]),
            "proposal_count": int(metric_summary["proposal_count"]),
            "draft_name_or_path": self.draft_name_or_path,
            "draft_name": model_display_name(self.draft_name_or_path),
            "per_position": confidence_metrics.compute(),
        }

    def report_dataset(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_row: dict,
        args_payload: dict,
        tasks: list[tuple[str, int | None]],
    ) -> None:
        print(
            json.dumps(
                compact_row(confidence_row),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if self.artifact_root is not None:
            metrics_path, extra_paths = self.write_dataset_outputs(
                metrics_row=metrics_row,
                confidence_row=confidence_row,
                args_payload=args_payload,
                tasks=tasks,
            )
            print(f"Wrote dataset metrics to {metrics_path}", flush=True)
            for path in extra_paths:
                print(f"Wrote dataset artifact to {path}", flush=True)

    def build_dataset_payload(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_row: dict,
        args_payload: dict,
        tasks: list[tuple[str, int | None]],
    ) -> dict:
        return {
            "config": {
                "args": args_payload,
                "tasks": jsonable(tasks),
            },
            "spec": metrics_row,
            "confidence": confidence_row,
            "confidence_summary": summarize_confidence_row(confidence_row),
        }

    def write_dataset_outputs(
        self,
        *,
        metrics_row: dict[str, object],
        confidence_row: dict,
        args_payload: dict,
        tasks: list[tuple[str, int | None]],
    ) -> tuple[Path, list[Path]]:
        assert self.artifact_root is not None
        dataset_dir = self.artifact_root / str(metrics_row["dataset"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = dataset_dir / "metrics.json"
        payload = self.build_dataset_payload(
            metrics_row=metrics_row,
            confidence_row=confidence_row,
            args_payload=args_payload,
            tasks=tasks,
        )
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        extra_paths = [plot_reliability_diagram(dataset_dir, confidence_row)]
        return metrics_path, extra_paths

    def log_tensorboard(self) -> None:
        if not self.rows:
            return

        from torch.utils.tensorboard import SummaryWriter

        assert self.tensorboard_dir is not None
        assert self.step is not None
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for row in self.rows:
            dataset = row["dataset"]
            per_position = row.get("per_position") or []
            for entry in per_position:
                pos = int(entry["position"])
                weight = float(entry["total_weight"])
                if weight <= 0.0:
                    continue
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/ece@{pos}",
                    entry["ece"],
                    self.step,
                )
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/auc@{pos}",
                    entry["auc"],
                    self.step,
                )
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/brier@{pos}",
                    entry["brier"],
                    self.step,
                )
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/pred_mean@{pos}",
                    entry["pred_mean"],
                    self.step,
                )
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/target_mean@{pos}",
                    entry["target_mean"],
                    self.step,
                )

            for key, value in summarize_confidence_row(row).items():
                add_tensorboard_scalar(
                    writer,
                    f"confidence/{dataset}/{key}",
                    value,
                    self.step,
                )
        writer.close()

    def build_table(self) -> str:
        from prettytable import PrettyTable

        table = PrettyTable()
        max_position_count = max(
            (len(row.get("per_position") or []) for row in self.rows),
            default=0,
        )
        field_names = [
            "dataset",
            "draft_model",
            "samples",
            "proposals",
            "ece_mean",
            "auc_mean",
            "brier_mean",
            "pred_mean",
            "target_mean",
        ]
        field_names.extend(f"ece@{pos}" for pos in range(max_position_count))
        field_names.extend(f"auc@{pos}" for pos in range(max_position_count))
        table.field_names = field_names

        draft_name = model_display_name(self.draft_name_or_path)
        for row in self.rows:
            summary = summarize_confidence_row(row)
            per_position = {
                int(entry["position"]): entry
                for entry in row.get("per_position") or []
            }
            table_row = [
                row["dataset"],
                draft_name,
                row["sample_count"],
                row["proposal_count"],
                format_float(summary["ece_mean"]),
                format_float(summary["auc_mean"]),
                format_float(summary["brier_mean"]),
                format_float(summary["pred_mean"]),
                format_float(summary["target_mean"]),
            ] + [
                (
                    format_float(per_position[pos]["ece"])
                    if pos in per_position
                    and float(per_position[pos]["total_weight"]) > 0.0
                    else "-"
                )
                for pos in range(max_position_count)
            ] + [
                (
                    format_float(per_position[pos]["auc"])
                    if pos in per_position
                    and float(per_position[pos]["total_weight"]) > 0.0
                    else "-"
                )
                for pos in range(max_position_count)
            ]
            table.add_row(table_row)
        return table.get_string()

    def print_results(self) -> None:
        if dist.get_rank() == 0 and self.rows:
            print("Confidence head reliability metrics:", flush=True)
            print(self.build_table(), flush=True)
