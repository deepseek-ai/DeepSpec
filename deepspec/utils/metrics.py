import re

import torch
import torch.distributed as dist

from .device import make_device


_REDUCTION_PATTERN = re.compile(r"^(dp_)?(mean|sum|max|min|last)$")
_DEFAULT_RATIO_REDUCTION = "dp_sum"
_metrics = {}


def _detach_scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        assert value.numel() == 1, "metrics only support scalar values"
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32)


def _clone_to_reduce_device(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().clone().to(torch.float32)
    if dist.get_backend() in {"nccl", "hccl"} and tensor.device.type == "cpu":
        tensor = tensor.to(make_device())
    return tensor


def _reduce_dp_value(value: torch.Tensor, op_name: str) -> torch.Tensor:
    if op_name == "sum" or op_name == "mean":
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if op_name == "mean":
            value = value / dist.get_world_size()
        return value
    if op_name == "max":
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value
    if op_name == "min":
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return value
    if op_name == "last":
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        return gathered[-1]
    raise AssertionError(f"unsupported reduction: {op_name}")


def _local_reduce(values, reduction: str) -> torch.Tensor:
    assert values, "cannot reduce an empty metric buffer"
    if reduction == "last":
        return values[-1]
    stacked = torch.stack([_clone_to_reduce_device(value) for value in values])
    if reduction == "max":
        return stacked.max()
    if reduction == "min":
        return stacked.min()
    if reduction in ("mean", "sum"):
        # Sum-style metrics report the average per emit in the logging window.
        return stacked.mean()
    raise AssertionError(f"unsupported reduction: {reduction}")


def _schema():
    items = []
    for name, entry in sorted(_metrics.items()):
        if entry["kind"] == "ratio":
            count = len(entry["num"])
        else:
            count = len(entry["values"])
        items.append((name, entry["kind"], entry["reduction"], count))
    return items


def _assert_schema_consistent():
    local_schema = _schema()
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_schema)
    reference = gathered[0]
    for rank, schema in enumerate(gathered[1:], start=1):
        assert schema == reference, (
            "metric schema mismatch across ranks: "
            f"rank0={reference}, rank{rank}={schema}"
        )


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if denominator.item() == 0:
        return 0.0
    return (numerator / denominator).item()


@torch.compiler.disable(recursive=False)
def add_metric(
    name,
    value,
    *,
    den=None,
    reduction: str = _DEFAULT_RATIO_REDUCTION,
    tag: str = "train",
):
    """Record one scalar metric for the next logging-window flush.

    Tensor inputs are detached at the API boundary to prevent metric logging
    from retaining autograd graphs. Ratios must pass their denominator through
    ``den=``; callers should not pre-divide locally because ``flush`` computes
    the global ratio as ``sum(num) / sum(den)``.
    """

    assert _REDUCTION_PATTERN.match(reduction), f"unsupported reduction: {reduction}"
    metric_name = f"{tag}/{name}"
    if den is not None:
        assert reduction == _DEFAULT_RATIO_REDUCTION, (
            "ratio metrics must use the default dp_sum reduction"
        )
        value = _detach_scalar(value)
        den = _detach_scalar(den)
        entry = _metrics.setdefault(
            metric_name,
            {"kind": "ratio", "reduction": reduction, "num": [], "den": []},
        )
        assert entry["kind"] == "ratio", f"metric kind changed for {metric_name}"
        assert entry["reduction"] == reduction, (
            f"metric reduction changed for {metric_name}: "
            f"{entry['reduction']} != {reduction}"
        )
        entry["num"].append(value)
        entry["den"].append(den)
        return

    value = _detach_scalar(value)
    entry = _metrics.setdefault(
        metric_name,
        {"kind": "scalar", "reduction": reduction, "values": []},
    )
    assert entry["kind"] == "scalar", f"metric kind changed for {metric_name}"
    assert entry["reduction"] == reduction, (
        f"metric reduction changed for {metric_name}: "
        f"{entry['reduction']} != {reduction}"
    )
    entry["values"].append(value)


def flush() -> dict[str, float]:
    _assert_schema_consistent()
    try:
        summary = {}
        for name, entry in sorted(_metrics.items()):
            reduction = entry["reduction"]
            if entry["kind"] == "ratio":
                num = torch.stack(
                    [_clone_to_reduce_device(value) for value in entry["num"]]
                ).sum()
                den = torch.stack(
                    [_clone_to_reduce_device(value) for value in entry["den"]]
                ).sum()
                num = _reduce_dp_value(num, "sum")
                den = _reduce_dp_value(den, "sum")
                summary[name] = _safe_div(num, den)
                continue

            local_reduction = reduction[3:] if reduction.startswith("dp_") else reduction
            value = _local_reduce(entry["values"], local_reduction)
            if reduction.startswith("dp_"):
                value = _reduce_dp_value(value, local_reduction)
            summary[name] = value.item()
        return summary
    finally:
        reset()


def reset() -> None:
    _metrics.clear()


__all__ = ["add_metric", "flush", "reset"]
