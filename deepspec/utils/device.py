import os

import torch


def _npu_module():
    module = getattr(torch, "npu", None)
    if module is None:
        return None
    try:
        return module if module.is_available() else None
    except Exception:
        return None


def device_type() -> str:
    requested = os.environ.get("DEEPSPEC_DEVICE", "").strip().lower()
    if requested in {"cuda", "npu"}:
        return requested
    if _npu_module() is not None:
        return "npu"
    return "cuda"


def accelerator_module():
    return getattr(torch, device_type())


def accelerator_backend() -> str:
    return "hccl" if device_type() == "npu" else "nccl"


def device_count() -> int:
    return int(accelerator_module().device_count())


def set_device(local_rank: int) -> None:
    accelerator_module().set_device(int(local_rank))


def current_device_index() -> int:
    return int(accelerator_module().current_device())


def make_device(local_rank: int | None = None) -> torch.device:
    if local_rank is None:
        local_rank = current_device_index()
    return torch.device(device_type(), int(local_rank))


def manual_seed_all(seed: int) -> None:
    if device_count() > 0:
        accelerator_module().manual_seed_all(int(seed))


def empty_cache() -> None:
    module = accelerator_module()
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def get_rng_state():
    return accelerator_module().get_rng_state()


def set_rng_state(state) -> None:
    accelerator_module().set_rng_state(state)
