import contextlib
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from .device import (
    accelerator_backend,
    current_device_index,
    device_count,
    make_device,
    set_device,
)


def init_dist(local_rank: int, timeout_minutes: int = 60):
    local_world_size = device_count()
    assert local_world_size > 0, "no accelerator devices are visible"
    node_rank = int(os.environ["RANK"])
    node_world_size = int(os.environ["WORLD_SIZE"])
    rank = node_rank * local_world_size + local_rank
    world_size = node_world_size * local_world_size
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    set_device(local_rank)
    device = make_device(local_rank)

    init_kwargs = dict(
        backend=accelerator_backend(),
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    if device.type == "cuda":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    return device, rank, world_size


def is_global_main_process():
    return dist.get_rank() == 0


def is_local_main_process():
    return current_device_index() == 0


def print_on_global_main(*args, **kwargs):
    if is_global_main_process():
        kwargs.setdefault("flush", True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)


def print_on_local_main(*args, **kwargs):
    if is_local_main_process():
        kwargs.setdefault("flush", True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)


@contextlib.contextmanager
def main_process_first():
    if dist.get_rank() == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield

class StatelessResumableDistributedSampler(Sampler):
    """Deterministic distributed sampler that streams across epoch boundaries.

    Each epoch uses the first ``total_size`` samples from a deterministic
    shuffle of the dataset.  The sampler can produce a fixed number of
    per-rank samples (``num_samples``) starting from an arbitrary per-rank
    offset, transparently crossing epoch boundaries with fresh shuffles.

    When ``num_samples`` is *None* (default) the sampler yields the remaining
    samples in the current epoch — this preserves backward compatibility with
    code that rebuilds the dataloader at every epoch boundary.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        total_size: int,
        seed: int = 42,
        start_global_offset_samples: int = 0,
        num_samples: int | None = None,
    ):
        assert start_global_offset_samples >= 0, "start_global_offset_samples must be >= 0"
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = int(total_size)
        self.seed = int(seed)
        self.dataset_size = len(self.dataset)
        assert self.dataset_size > 0, "dataset must have positive length"
        assert self.total_size > 0, "total_size must be > 0"
        assert self.total_size <= self.dataset_size, (
            f"total_size ({self.total_size}) cannot exceed dataset size ({self.dataset_size})"
        )
        assert self.total_size % self.num_replicas == 0, (
            f"total_size ({self.total_size}) must be divisible by num_replicas ({self.num_replicas})"
        )
        assert num_samples is None or num_samples >= 0, "num_samples must be >= 0"

        self.per_rank_len_per_epoch = self.total_size // self.num_replicas
        self._global_offset = int(start_global_offset_samples)
        self._num_samples = num_samples

    def __len__(self):
        if self._num_samples is not None:
            return self._num_samples
        mod = self._global_offset % self.per_rank_len_per_epoch
        return self.per_rank_len_per_epoch - mod if mod != 0 else self.per_rank_len_per_epoch

    def _epoch_perm(self, epoch_idx: int):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch_idx)
        return torch.randperm(self.dataset_size, generator=g).tolist()[: self.total_size]

    def _epoch_slice_for_rank(self, perm):
        return perm[self.rank : self.total_size : self.num_replicas]

    def _iter_stream(self):
        epoch_idx = self._global_offset // self.per_rank_len_per_epoch
        offset_in_epoch = self._global_offset % self.per_rank_len_per_epoch

        perm = self._epoch_perm(epoch_idx)
        my_seq = self._epoch_slice_for_rank(perm)
        for i in range(offset_in_epoch, len(my_seq)):
            yield my_seq[i]

        epoch_idx += 1
        while True:
            perm = self._epoch_perm(epoch_idx)
            my_seq = self._epoch_slice_for_rank(perm)
            for idx in my_seq:
                yield idx
            epoch_idx += 1

    def __iter__(self):
        it = self._iter_stream()
        for _ in range(len(self)):
            yield next(it)
