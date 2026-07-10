import argparse
import ast
import json
import math
import struct


INDEX_RECORD_SIZE = struct.calcsize("<QIIQQQQQ")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the disk space needed by a DeepSpec target cache without "
            "loading the target model or writing cache shards."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional DeepSpec config path. When provided, max_length and "
            "target_layer_ids are read from config.data.max_length and "
            "config.model.target_layer_ids unless explicitly overridden."
        ),
    )
    parser.add_argument(
        "--opts",
        action="append",
        default=[],
        help="Optional config override in the same KEY=VALUE format used by training.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of valid samples expected in the target cache.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help=(
            "Token length to estimate per sample. Defaults to config.data.max_length "
            "when --config is provided."
        ),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        required=True,
        help=(
            "Target model hidden size. This is intentionally explicit so the "
            "estimator does not need to download model metadata."
        ),
    )
    parser.add_argument(
        "--num-target-layers",
        type=int,
        help=(
            "Number of captured target layers. Defaults to len(config.model.target_layer_ids) "
            "when --config is provided."
        ),
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=64 * 1024**3,
        help="Shard size used by prepare_target_cache.py; used only to estimate shard count.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the estimate as machine-readable JSON.",
    )
    return parser.parse_args()


def _format_bytes(num_bytes):
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _literal_or_string(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _literal_config_mapping(node):
    if isinstance(node, ast.Dict):
        return ast.literal_eval(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if node.args:
            raise ValueError("Only keyword-based dict(...) config sections are supported.")
        return {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
    raise ValueError(f"Unsupported config section syntax: {ast.dump(node)}")


def _load_config_defaults(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=config_path)
    defaults = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in {"data", "model"}:
            continue
        defaults[name] = _literal_config_mapping(node.value)
    return defaults


def _apply_opts(config_defaults, opts):
    for opt in opts:
        name, value = opt.split("=", 1)
        parts = name.split(".")
        if len(parts) != 2 or parts[0] not in config_defaults:
            continue
        section, key = parts
        if key in config_defaults[section]:
            config_defaults[section][key] = _literal_or_string(value)
    return config_defaults


def _resolve_config_defaults(args):
    config_defaults = None
    if args.config:
        config_defaults = _apply_opts(_load_config_defaults(args.config), args.opts)

    seq_len = args.seq_len
    if seq_len is None and config_defaults is not None:
        seq_len = int(config_defaults["data"]["max_length"])
    if seq_len is None:
        raise ValueError("Either --seq-len or --config must be provided.")

    num_target_layers = args.num_target_layers
    if num_target_layers is None and config_defaults is not None:
        num_target_layers = len(config_defaults["model"]["target_layer_ids"])
    if num_target_layers is None:
        raise ValueError("Either --num-target-layers or --config must be provided.")

    return seq_len, num_target_layers


def build_estimate(
    *,
    num_samples,
    seq_len,
    hidden_size,
    num_target_layers,
    max_shard_bytes,
):
    per_sample_tensors = {
        "input_ids": int(seq_len) * 4,
        "attention_mask": int(seq_len),
        "loss_mask": int(seq_len),
        "target_hidden_states": (
            int(seq_len) * int(num_target_layers) * int(hidden_size) * 2
        ),
        "target_last_hidden_states": int(seq_len) * int(hidden_size) * 2,
    }
    per_sample_data_bytes = sum(per_sample_tensors.values())
    per_sample_total_bytes = per_sample_data_bytes + INDEX_RECORD_SIZE
    data_bytes = per_sample_data_bytes * num_samples
    index_bytes = INDEX_RECORD_SIZE * num_samples
    total_bytes = data_bytes + index_bytes
    estimated_shards = max(1, math.ceil(data_bytes / max_shard_bytes))
    return {
        "num_samples": int(num_samples),
        "seq_len": int(seq_len),
        "hidden_size": int(hidden_size),
        "num_target_layers": int(num_target_layers),
        "bytes_per_sample": int(per_sample_total_bytes),
        "tensor_bytes_per_sample": {
            key: int(value) for key, value in per_sample_tensors.items()
        },
        "data_bytes": int(data_bytes),
        "index_bytes": int(index_bytes),
        "total_bytes": int(total_bytes),
        "estimated_shards": int(estimated_shards),
        "max_shard_bytes": int(max_shard_bytes),
    }


def _print_human_readable(estimate):
    print("Target cache size estimate")
    print(f"  samples:              {estimate['num_samples']:,}")
    print(f"  sequence length:      {estimate['seq_len']:,}")
    print(f"  hidden size:          {estimate['hidden_size']:,}")
    print(f"  target layers:        {estimate['num_target_layers']:,}")
    print(f"  bytes per sample:     {_format_bytes(estimate['bytes_per_sample'])}")
    print(f"  tensor data:          {_format_bytes(estimate['data_bytes'])}")
    print(f"  samples.idx:          {_format_bytes(estimate['index_bytes'])}")
    print(f"  total:                {_format_bytes(estimate['total_bytes'])}")
    print(f"  estimated shards:     {estimate['estimated_shards']:,}")


def main():
    args = _parse_args()
    seq_len, num_target_layers = _resolve_config_defaults(args)
    estimate = build_estimate(
        num_samples=args.num_samples,
        seq_len=seq_len,
        hidden_size=args.hidden_size,
        num_target_layers=num_target_layers,
        max_shard_bytes=args.max_shard_bytes,
    )
    if args.json:
        print(json.dumps(estimate, indent=2, sort_keys=True))
    else:
        _print_human_readable(estimate)


if __name__ == "__main__":
    main()
