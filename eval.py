from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.utils import CustomJSONEncoder, device_count

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]
TASK_DEFAULTS = dict(TASKS)


def get_evaluator_cls(architecture: str):
    if architecture == "Qwen3DSparkModel":
        from deepspec.eval.dspark import Qwen3DSparkEvaluator
        return Qwen3DSparkEvaluator
    if architecture == "Gemma4DSparkModel":
        from deepspec.eval.dspark import Gemma4DSparkEvaluator
        return Gemma4DSparkEvaluator
    if architecture in {"Qwen3Eagle3Model", "Eagle3DraftModel"}:
        from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
        return Qwen3Eagle3Evaluator
    if architecture == "Gemma4Eagle3Model":
        from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator
        return Gemma4Eagle3Evaluator
    raise ValueError(f"Unsupported draft architecture: {architecture}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Dataset name under eval_datasets without .jsonl. Repeat to run several tasks.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the sample count for every selected task.",
    )
    args = parser.parse_args()
    task_names = args.task if args.task else [name for name, _ in TASKS]
    args.tasks = [
        (name, args.num_samples if args.num_samples is not None else TASK_DEFAULTS[name])
        for name in task_names
    ]
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = get_evaluator_cls(draft_config.architectures[0])
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=device_count(),
    )
