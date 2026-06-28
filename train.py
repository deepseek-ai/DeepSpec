import argparse
import json
import os
import torch
from deepspec.utils import (
    CustomJSONEncoder,
    device_count,
    get_git_diff,
    load_config,
    parse_opts_to_config,
    seed_all,
    get_git_sha,
)

os.environ['USE_TORCH']='true'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    args = parser.parse_args()
    config = parse_opts_to_config(args.opts, load_config(args.config))
    config._origin_config_path = os.path.abspath(args.config)
    config._origin_opts = list(args.opts)
    return config


def main(local_rank):
    args = parse_args()
    seed_all(int(args.seed))
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    trainer = args.train.trainer_cls(local_rank, args)
    trainer.train()
    trainer.clean_up()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print(f"git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    torch.multiprocessing.spawn(main, nprocs=device_count())
