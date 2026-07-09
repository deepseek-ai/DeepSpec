from __future__ import annotations
import argparse
import json
import os

os.environ["USE_HUB_KERNELS"] = "0"

import torch
from transformers import AutoConfig
from deepspec.eval.dspark import (
    Gemma4DSparkEvaluator,
    Ministral3DSparkEvaluator,
    Qwen3DSparkEvaluator,
)
from deepspec.eval.eagle3 import (
    Gemma4Eagle3Evaluator,
    Ministral3Eagle3Evaluator,
    Qwen3Eagle3Evaluator,
)
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Ministral3DSparkModel": Ministral3DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Ministral3Eagle3Model": Ministral3Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

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


def parse_task_spec(task_spec: str) -> tuple[str, int | None]:
    if ":" not in task_spec:
        return task_spec, None

    name, max_samples = task_spec.rsplit(":", 1)
    if max_samples.lower() == "none":
        return name, None
    return name, int(max_samples)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help=(
            "Evaluate only this task. Repeatable. Format is `name` to use all "
            "available samples or `name:max_samples` to cap the dataset."
        ),
    )
    args = parser.parse_args()
    if args.task is None:
        args.tasks = list(TASKS)
    else:
        args.tasks = [parse_task_spec(task_spec) for task_spec in args.task]
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
