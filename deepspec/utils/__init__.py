import random

import numpy as np
import torch

from .config import CustomJSONEncoder, jsonable, load_config, parse_opts_to_config
from .device import (
    accelerator_backend,
    accelerator_module,
    current_device_index,
    device_count,
    device_type,
    empty_cache,
    get_rng_state,
    is_npu_available,
    make_device,
    manual_seed_all,
    set_device,
    set_rng_state,
)
from .distributed import (
    StatelessResumableDistributedSampler,
    init_dist,
    is_global_main_process,
    is_local_main_process,
    main_process_first,
    print_on_global_main,
    print_on_local_main,
)
from .io import ensure_dir, safe_symlink
from .metrics import add_metric, flush, reset
from .optim import BF16Optimizer

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    manual_seed_all(seed)

def get_git_sha(detail_info=False):
    import subprocess

    def run_git_text(cmd):
        try:
            return subprocess.check_output(
                cmd,
                encoding="utf-8",
                errors="replace",
            ).rstrip()
        except Exception:
            return "unknown"

    cmd = ["git", "rev-parse", "--short", "HEAD"]
    sha = run_git_text(cmd)
    if not detail_info:
        return sha
    # 获取一些 git 详细信息，帮助从训练日志中得到运行时的目录状态
    # 查看当前目录下没有提交的文件信息
    cmd = ["git", "status"]
    status = run_git_text(cmd)
    # 查看当前目录下上一次提交的信息
    cmd = ["git", "log", "-n", "1"]
    last_commit = run_git_text(cmd)
    return sha, status, last_commit


def get_git_diff(rev="HEAD"):
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "diff", rev],
            encoding="utf-8",
            errors="replace",
        ).rstrip()
    except Exception:
        return "unknown"


__all__ = [
    "BF16Optimizer",
    "CustomJSONEncoder",
    "StatelessResumableDistributedSampler",
    "accelerator_backend",
    "accelerator_module",
    "add_metric",
    "current_device_index",
    "device_count",
    "device_type",
    "empty_cache",
    "ensure_dir",
    "flush",
    "get_git_diff",
    "get_git_sha",
    "get_rng_state",
    "is_npu_available",
    "init_dist",
    "is_global_main_process",
    "is_local_main_process",
    "jsonable",
    "load_config",
    "make_device",
    "manual_seed_all",
    "main_process_first",
    "parse_opts_to_config",
    "print_on_global_main",
    "print_on_local_main",
    "reset",
    "safe_symlink",
    "set_device",
    "set_rng_state",
    "seed_all",
]
