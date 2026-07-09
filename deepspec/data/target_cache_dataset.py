"""Target-cache storage protocol, writers, dataset, collator, and validation."""

import json
import mmap
import os
import queue
import shutil
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from deepspec.data.parser import preprocess_record


TARGET_CACHE_VERSION = 2
INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQQQ")
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size

TARGET_CACHE_HIDDEN_DTYPE = "bfloat16"
TARGET_CACHE_TOKEN_DTYPE  = "int32"
TARGET_CACHE_MASK_DTYPE   = "uint8"


def atomic_json_dump(payload, path: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def build_target_cache_shard_path(cache_dir: str, file_name: str) -> str:
    return os.path.join(cache_dir, file_name)


def expected_target_cache_tensor_numel(
    *,
    seq_len: int,
    hidden_size: int,
    num_target_layers: int,
):
    return {
        "input_ids": int(seq_len),
        "attention_mask": int(seq_len),
        "loss_mask": int(seq_len),
        "target_hidden_states": int(seq_len) * int(num_target_layers) * int(hidden_size),
        "target_last_hidden_states": int(seq_len) * int(hidden_size),
    }


def expected_target_cache_tensor_nbytes(
    *,
    seq_len: int,
    hidden_size: int,
    num_target_layers: int,
    hidden_dtype: str = "bfloat16",
):
    numel = expected_target_cache_tensor_numel(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
    )
    hidden_element_size = 2 if hidden_dtype == "bfloat16" else 1
    return {
        "input_ids": numel["input_ids"] * 4,
        "attention_mask": numel["attention_mask"],
        "loss_mask": numel["loss_mask"],
        "target_hidden_states": numel["target_hidden_states"] * hidden_element_size,
        "target_last_hidden_states": numel["target_last_hidden_states"] * hidden_element_size,
    }


def pack_index_record(
    *,
    sample_id: int,
    shard_id: int,
    seq_len: int,
    input_ids_offset: int,
    attention_mask_offset: int,
    loss_mask_offset: int,
    target_hidden_states_offset: int,
    target_last_hidden_states_offset: int,
):
    return INDEX_RECORD_STRUCT.pack(
        int(sample_id),
        int(shard_id),
        int(seq_len),
        int(input_ids_offset),
        int(attention_mask_offset),
        int(loss_mask_offset),
        int(target_hidden_states_offset),
        int(target_last_hidden_states_offset),
    )


def unpack_index_record(buffer, offset: int = 0):
    (
        sample_id,
        shard_id,
        seq_len,
        input_ids_offset,
        attention_mask_offset,
        loss_mask_offset,
        target_hidden_states_offset,
        target_last_hidden_states_offset,
    ) = INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    return {
        "sample_id": sample_id,
        "shard_id": shard_id,
        "seq_len": seq_len,
        "input_ids_offset": input_ids_offset,
        "attention_mask_offset": attention_mask_offset,
        "loss_mask_offset": loss_mask_offset,
        "target_hidden_states_offset": target_hidden_states_offset,
        "target_last_hidden_states_offset": target_last_hidden_states_offset,
    }


def load_target_cache_manifest(cache_dir: str):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"Missing target cache manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_target_cache_manifest(cache_dir=cache_dir, manifest=manifest)
    return manifest


def validate_target_cache_manifest(*, cache_dir: str, manifest):
    required_fields = {
        "version",
        "num_samples",
        "num_shards",
        "target_layer_ids",
        "hidden_dtype",
        "token_dtype",
        "mask_dtype",
        "index_record_size",
        "hidden_size",
        "shards",
    }
    missing = sorted(required_fields - set(manifest))
    assert not missing, (
        f"Target cache manifest is missing required fields {missing}: "
        f"{os.path.join(cache_dir, 'manifest.json')}"
    )
    assert int(manifest["version"]) == TARGET_CACHE_VERSION, (
        "Unsupported target cache manifest version: "
        f"{manifest['version']} != {TARGET_CACHE_VERSION}"
    )
    supported_hidden_dtypes = ("bfloat16", "float8_e4m3fn", "float8_e5m2")
    assert manifest["hidden_dtype"] in supported_hidden_dtypes, (
        f"Unsupported hidden_dtype in target cache manifest: {manifest['hidden_dtype']}. "
        f"Supported dtypes are {supported_hidden_dtypes}."
    )
    assert manifest["token_dtype"] == TARGET_CACHE_TOKEN_DTYPE, (
        "Unsupported token_dtype in target cache manifest: "
        f"{manifest['token_dtype']}"
    )
    assert manifest["mask_dtype"] == TARGET_CACHE_MASK_DTYPE, (
        "Unsupported mask_dtype in target cache manifest: "
        f"{manifest['mask_dtype']}"
    )
    assert int(manifest["index_record_size"]) == INDEX_RECORD_SIZE, (
        "index_record_size does not match canonical target cache protocol: "
        f"{manifest['index_record_size']} != {INDEX_RECORD_SIZE}"
    )
    hidden_size = int(manifest["hidden_size"])
    assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
    target_layer_ids = [int(layer_id) for layer_id in manifest["target_layer_ids"]]
    assert target_layer_ids, "target_layer_ids must not be empty."
    assert target_layer_ids == sorted(target_layer_ids), (
        "target_layer_ids must be sorted in ascending order."
    )
    num_shards = int(manifest["num_shards"])
    assert num_shards == len(manifest["shards"]), (
        "num_shards does not match shard metadata count: "
        f"{num_shards} != {len(manifest['shards'])}"
    )
    for expected_shard_id, shard in enumerate(manifest["shards"]):
        assert int(shard["shard_id"]) == expected_shard_id, (
            "Target cache shard ids must be contiguous starting from 0: "
            f"expected {expected_shard_id}, got {shard['shard_id']}"
        )
        shard_path = build_target_cache_shard_path(cache_dir, shard["file_name"])
        assert os.path.exists(shard_path), f"Missing target cache shard file: {shard_path}"
    index_path = os.path.join(cache_dir, "samples.idx")
    assert os.path.exists(index_path), f"Missing target cache index file: {index_path}"
    index_size = os.path.getsize(index_path)
    assert index_size % INDEX_RECORD_SIZE == 0, (
        "samples.idx size is not a multiple of the canonical record size: "
        f"{index_size} % {INDEX_RECORD_SIZE} != 0"
    )
    num_samples = int(manifest["num_samples"])
    assert index_size == num_samples * INDEX_RECORD_SIZE, (
        "samples.idx size does not match manifest.num_samples: "
        f"{index_size} != {num_samples} * {INDEX_RECORD_SIZE}"
    )


def validate_train_cache(*, train_dataset, draft_model, target_model_name_or_path):
    manifest = train_dataset.manifest
    expected_layer_ids = [int(layer_id) for layer_id in draft_model.target_layer_ids]
    assert [int(layer_id) for layer_id in manifest["target_layer_ids"]] == expected_layer_ids, (
        "Target cache target_layer_ids do not match draft model configuration: "
        f"{manifest['target_layer_ids']} != {expected_layer_ids}"
    )
    assert int(manifest["hidden_size"]) == int(draft_model.config.hidden_size), (
        "Target cache hidden_size does not match draft model hidden size: "
        f"{manifest['hidden_size']} != {draft_model.config.hidden_size}"
    )
    cache_target_model_name = manifest["target_model_name_or_path"]
    assert str(cache_target_model_name) == str(target_model_name_or_path), (
        "Target cache target_model_name_or_path does not match training config: "
        f"{cache_target_model_name} != {target_model_name_or_path}"
    )


def _tensor_to_bytes(tensor: torch.Tensor, dtype: torch.dtype):
    cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu_tensor.numpy().tobytes()


def _tensor_to_bfloat16_bytes(tensor: torch.Tensor):
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return cpu_tensor.view(torch.uint16).numpy().tobytes()


def _tensor_to_float8_bytes(tensor: torch.Tensor, dtype: torch.dtype):
    cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu_tensor.view(torch.uint8).numpy().tobytes()


def compute_local_sample_range(*, num_samples: int, rank: int, world_size: int):
    base = int(num_samples) // int(world_size)
    remainder = int(num_samples) % int(world_size)
    start = rank * base + min(rank, remainder)
    local_count = base + (1 if rank < remainder else 0)
    return start, start + local_count


def prepare_target_cache_output_dir(output_dir: str):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        existing = sorted(os.listdir(output_dir))
        if existing:
            raise FileExistsError(
                f"Target cache output dir is not empty: {output_dir}. "
                "Use a new output directory."
            )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)


@dataclass
class LocalCacheWriteSummary:
    global_rank: int
    source_sample_start: int
    source_sample_end: int
    num_local_samples: int
    num_local_shards: int
    local_shard_files: list[str]

    def to_json(self):
        return {
            "global_rank": self.global_rank,
            "source_sample_start": self.source_sample_start,
            "source_sample_end": self.source_sample_end,
            "num_local_samples": self.num_local_samples,
            "num_local_shards": self.num_local_shards,
            "local_shard_files": list(self.local_shard_files),
        }


@dataclass(frozen=True)
class TargetCacheSampleBytes:
    sample_id: int
    seq_len: int
    input_ids: bytes
    attention_mask: bytes
    loss_mask: bytes
    target_hidden_states: bytes
    target_last_hidden_states: bytes


def build_target_cache_sample_bytes(
    *,
    sample_id: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    target_hidden_states: torch.Tensor,
    target_last_hidden_states: torch.Tensor,
    hidden_dtype: str = "bfloat16",
):
    if hidden_dtype == "bfloat16":
        th_bytes = _tensor_to_bfloat16_bytes(target_hidden_states)
        tlh_bytes = _tensor_to_bfloat16_bytes(target_last_hidden_states)
    elif hidden_dtype == "float8_e4m3fn":
        th_bytes = _tensor_to_float8_bytes(target_hidden_states, torch.float8_e4m3fn)
        tlh_bytes = _tensor_to_float8_bytes(target_last_hidden_states, torch.float8_e4m3fn)
    elif hidden_dtype == "float8_e5m2":
        th_bytes = _tensor_to_float8_bytes(target_hidden_states, torch.float8_e5m2)
        tlh_bytes = _tensor_to_float8_bytes(target_last_hidden_states, torch.float8_e5m2)
    else:
        raise ValueError(f"Unsupported hidden_dtype for writing bytes: {hidden_dtype}")

    return TargetCacheSampleBytes(
        sample_id=int(sample_id),
        seq_len=int(input_ids.shape[0]),
        input_ids=_tensor_to_bytes(input_ids, torch.int32),
        attention_mask=_tensor_to_bytes(attention_mask, torch.uint8),
        loss_mask=_tensor_to_bytes(loss_mask, torch.uint8),
        target_hidden_states=th_bytes,
        target_last_hidden_states=tlh_bytes,
    )


class LocalTargetCacheWriter:
    def __init__(self, *, rank_dir: str, max_shard_bytes: int, hidden_dtype: str = "bfloat16"):
        self.rank_dir = rank_dir
        self.max_shard_bytes = int(max_shard_bytes)
        self.hidden_dtype = hidden_dtype
        self.local_index_path = os.path.join(rank_dir, "samples.local.idx")
        self.index_handle = open(self.local_index_path, "wb")
        self.current_shard_id = -1
        self.current_shard_handle = None
        self.current_shard_size = 0
        self.local_shard_files = []
        self.num_local_samples = 0

    def close(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
            self.current_shard_handle = None
        if getattr(self, "index_handle", None) is not None:
            self.index_handle.flush()
            os.fsync(self.index_handle.fileno())
            self.index_handle.close()
            self.index_handle = None

    def _open_new_shard(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
        self.current_shard_id += 1
        file_name = f"shard-local-{self.current_shard_id:05d}.bin"
        shard_path = os.path.join(self.rank_dir, file_name)
        self.current_shard_handle = open(shard_path, "wb")
        self.current_shard_size = 0
        self.local_shard_files.append(file_name)

    def _ensure_shard(self, sample_nbytes: int):
        if self.current_shard_handle is None:
            self._open_new_shard()
            return
        if (
            self.current_shard_size > 0
            and self.current_shard_size + int(sample_nbytes) > self.max_shard_bytes
        ):
            self._open_new_shard()

    def write_sample_bytes(self, sample: TargetCacheSampleBytes):
        sample_nbytes = (
            len(sample.input_ids)
            + len(sample.attention_mask)
            + len(sample.loss_mask)
            + len(sample.target_hidden_states)
            + len(sample.target_last_hidden_states)
        )
        self._ensure_shard(sample_nbytes)
        input_ids_offset = self.current_shard_size
        self.current_shard_handle.write(sample.input_ids)
        self.current_shard_size += len(sample.input_ids)
        attention_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.attention_mask)
        self.current_shard_size += len(sample.attention_mask)
        loss_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.loss_mask)
        self.current_shard_size += len(sample.loss_mask)
        target_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_hidden_states)
        self.current_shard_size += len(sample.target_hidden_states)
        target_last_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_last_hidden_states)
        self.current_shard_size += len(sample.target_last_hidden_states)
        self.index_handle.write(
            pack_index_record(
                sample_id=sample.sample_id,
                shard_id=self.current_shard_id,
                seq_len=sample.seq_len,
                input_ids_offset=input_ids_offset,
                attention_mask_offset=attention_mask_offset,
                loss_mask_offset=loss_mask_offset,
                target_hidden_states_offset=target_hidden_states_offset,
                target_last_hidden_states_offset=target_last_hidden_states_offset,
            )
        )
        self.num_local_samples += 1

    def write_sample(
        self,
        *,
        sample_id: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
    ):
        sample = build_target_cache_sample_bytes(
            sample_id=sample_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
            hidden_dtype=self.hidden_dtype,
        )
        self.write_sample_bytes(sample)


class AsyncTargetCacheWriter:
    def __init__(
        self,
        *,
        rank_dir: str,
        max_shard_bytes: int,
        max_queue_size: int = 128,
        hidden_dtype: str = "bfloat16",
    ):
        self.hidden_dtype = hidden_dtype
        self.writer = LocalTargetCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=max_shard_bytes,
            hidden_dtype=hidden_dtype,
        )
        # Queue CPU byte records only; never hold CUDA tensor references here.
        self.queue = queue.Queue(maxsize=int(max_queue_size))
        self.sentinel = object()
        self.num_local_samples = 0
        self._closed = False
        self._exception = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"target-cache-writer-{os.path.basename(rank_dir)}",
        )
        self.thread.start()

    @property
    def local_shard_files(self):
        return self.writer.local_shard_files

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.sentinel:
                        break
                    self.writer.write_sample_bytes(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self._exception = exc
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                if self._exception is None:
                    self._exception = exc

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async target cache writer failed.") from self._exception

    def _put(self, item):
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def write_sample(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
    ):
        sample = build_target_cache_sample_bytes(
            sample_id=self.num_local_samples,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
            hidden_dtype=self.hidden_dtype,
        )
        self._put(sample)
        self.num_local_samples += 1

    def close(self):
        if self._closed:
            self._raise_if_failed()
            return
        if self._exception is None:
            self._put(self.sentinel)
        self.thread.join()
        self._closed = True
        self._raise_if_failed()
        assert self.writer.num_local_samples == self.num_local_samples, (
            "Async target cache writer lost samples: "
            f"{self.writer.num_local_samples} != {self.num_local_samples}"
        )


def load_local_cache_write_summary(rank_dir: str):
    with open(os.path.join(rank_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_global_target_cache_shard_map(summaries):
    shard_map = {}
    shards = []
    next_shard_id = 0
    for summary in sorted(
        summaries,
        key=lambda item: int(item["source_sample_start"]),
    ):
        local_map = []
        for _local_shard_id, _file_name in enumerate(summary["local_shard_files"]):
            local_map.append(next_shard_id)
            shards.append(
                {
                    "shard_id": next_shard_id,
                    "file_name": f"shard-{next_shard_id:05d}.bin",
                }
            )
            next_shard_id += 1
        shard_map[int(summary["global_rank"])] = local_map
    return shard_map, shards


def rename_local_target_cache_shards(*, output_dir: str, rank_dir: str, summary, shard_map):
    local_map = shard_map[int(summary["global_rank"])]
    for local_shard_id, file_name in enumerate(summary["local_shard_files"]):
        source = os.path.join(rank_dir, file_name)
        target = os.path.join(output_dir, f"shard-{local_map[local_shard_id]:05d}.bin")
        os.replace(source, target)


def finalize_target_cache_index(*, output_dir: str, summaries, shard_map):
    index_tmp_path = os.path.join(output_dir, "samples.idx.tmp")
    next_expected_sample_id = 0
    with open(index_tmp_path, "wb") as output_handle:
        for summary in sorted(
            summaries,
            key=lambda item: int(item["source_sample_start"]),
        ):
            rank_dir = os.path.join(
                output_dir,
                "_tmp",
                f"rank_{int(summary['global_rank'])}",
            )
            local_index_path = os.path.join(rank_dir, "samples.local.idx")
            with open(local_index_path, "rb") as local_handle:
                local_bytes = local_handle.read()
            assert len(local_bytes) % INDEX_RECORD_SIZE == 0, (
                "Local target cache index has invalid size: "
                f"{local_index_path}"
            )
            next_local_sample_id = 0
            for offset in range(0, len(local_bytes), INDEX_RECORD_SIZE):
                record = unpack_index_record(local_bytes, offset)
                assert int(record["sample_id"]) == next_local_sample_id, (
                    "Local target cache index is not ordered by local sample_id: "
                    f"got {record['sample_id']}, expected {next_local_sample_id}"
                )
                record["sample_id"] = next_expected_sample_id
                record["shard_id"] = shard_map[int(summary["global_rank"])][
                    int(record["shard_id"])
                ]
                output_handle.write(pack_index_record(**record))
                next_local_sample_id += 1
                next_expected_sample_id += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(index_tmp_path, os.path.join(output_dir, "samples.idx"))
    return next_expected_sample_id


def build_target_cache_manifest(
    *,
    num_samples: int,
    shards,
    target_layer_ids,
    hidden_size: int,
    hidden_dtype: str = "bfloat16",
    extra_fields=None,
):
    manifest = {
        "version": TARGET_CACHE_VERSION,
        "num_samples": int(num_samples),
        "num_shards": len(shards),
        "target_layer_ids": [int(layer_id) for layer_id in target_layer_ids],
        "hidden_dtype": hidden_dtype,
        "token_dtype": TARGET_CACHE_TOKEN_DTYPE,
        "mask_dtype": TARGET_CACHE_MASK_DTYPE,
        "index_record_size": INDEX_RECORD_SIZE,
        "hidden_size": int(hidden_size),
        "shards": shards,
    }
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def write_target_cache_manifest(*, output_dir: str, manifest):
    atomic_json_dump(manifest, os.path.join(output_dir, "manifest.json"))


def cleanup_target_cache_tmp_dir(output_dir: str):
    tmp_dir = os.path.join(output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.manifest = load_target_cache_manifest(self.cache_dir)
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.target_layer_ids = [int(layer_id) for layer_id in self.manifest["target_layer_ids"]]
        self.num_target_layers = len(self.target_layer_ids)
        self.hidden_dtype = self.manifest.get("hidden_dtype", "bfloat16")
        self.index_path = os.path.join(self.cache_dir, "samples.idx")
        self.index_file = None
        self.index_mmap = None
        self.max_open_shards = max_open_shards
        self.shard_handles = OrderedDict()
        self.shard_mmaps = OrderedDict()
        self.shard_paths = {
            int(shard["shard_id"]): build_target_cache_shard_path(
                self.cache_dir,
                shard["file_name"],
            )
            for shard in self.manifest["shards"]
        }

    def __len__(self):
        return self.num_samples

    def close(self):
        for shard_mmap in getattr(self, "shard_mmaps", {}).values():
            shard_mmap.close()
        for handle in getattr(self, "shard_handles", {}).values():
            handle.close()
        if hasattr(self, "shard_mmaps"):
            self.shard_mmaps.clear()
        if hasattr(self, "shard_handles"):
            self.shard_handles.clear()
        if getattr(self, "index_mmap", None) is not None:
            self.index_mmap.close()
            self.index_mmap = None
        if getattr(self, "index_file", None) is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        self.close()

    def __getstate__(self):  # pragma: no cover
        state = dict(self.__dict__)
        state["index_file"] = None
        state["index_mmap"] = None
        state["shard_handles"] = OrderedDict()
        state["shard_mmaps"] = OrderedDict()
        return state

    def _ensure_index_mmap(self):
        if self.index_mmap is None:
            self.index_file = open(self.index_path, "rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )

    def _get_shard_mmap(self, shard_id: int):
        shard_id = int(shard_id)
        if shard_id in self.shard_mmaps:
            self.shard_mmaps.move_to_end(shard_id)
            self.shard_handles.move_to_end(shard_id)
            return self.shard_mmaps[shard_id]
        shard_path = self.shard_paths[shard_id]
        handle = open(shard_path, "rb")
        self.shard_handles[shard_id] = handle
        self.shard_mmaps[shard_id] = mmap.mmap(
            handle.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )
        while len(self.shard_mmaps) > self.max_open_shards:
            evicted_id, evicted_mmap = self.shard_mmaps.popitem(last=False)
            evicted_mmap.close()
            self.shard_handles.pop(evicted_id).close()
        return self.shard_mmaps[shard_id]

    def _read_record(self, index: int):
        self._ensure_index_mmap()
        offset = int(index) * INDEX_RECORD_SIZE
        record = unpack_index_record(self.index_mmap, offset)
        assert int(record["sample_id"]) == int(index), (
            "Target cache index is not sorted by sample_id or sample ids are not dense: "
            f"record sample_id={record['sample_id']}, expected {index}"
        )
        return record

    def _read_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        np_dtype,
        torch_dtype,
        nbytes: int,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np_dtype,
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(*shape)
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(dtype=torch_dtype)
        return tensor

    def _read_bfloat16_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        nbytes: int,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np.uint16,
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(torch.bfloat16)
        return tensor.view(*shape)

    def _read_float8_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        nbytes: int,
        torch_dtype,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np.uint8,
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(torch_dtype)
        return tensor.view(*shape)

    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        record = self._read_record(int(index))
        seq_len = int(record["seq_len"])
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        shard_mmap = self._get_shard_mmap(int(record["shard_id"]))
        nbytes = expected_target_cache_tensor_nbytes(
            seq_len=seq_len,
            hidden_size=self.hidden_size,
            num_target_layers=self.num_target_layers,
            hidden_dtype=self.hidden_dtype,
        )
        input_ids = self._read_tensor_from_shard(
            shard_mmap=shard_mmap,
            offset=record["input_ids_offset"],
            shape=(seq_len,),
            np_dtype=np.int32,
            torch_dtype=torch.int32,
            nbytes=nbytes["input_ids"],
        )
        loss_mask = self._read_tensor_from_shard(
            shard_mmap=shard_mmap,
            offset=record["loss_mask_offset"],
            shape=(seq_len,),
            np_dtype=np.uint8,
            torch_dtype=torch.uint8,
            nbytes=nbytes["loss_mask"],
        )
        if self.hidden_dtype == "bfloat16":
            target_hidden_states = self._read_bfloat16_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_hidden_states_offset"],
                shape=(seq_len, self.num_target_layers * self.hidden_size),
                nbytes=nbytes["target_hidden_states"],
            )
            target_last_hidden_states = self._read_bfloat16_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_last_hidden_states_offset"],
                shape=(seq_len, self.hidden_size),
                nbytes=nbytes["target_last_hidden_states"],
            )
        elif self.hidden_dtype == "float8_e4m3fn":
            target_hidden_states = self._read_float8_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_hidden_states_offset"],
                shape=(seq_len, self.num_target_layers * self.hidden_size),
                nbytes=nbytes["target_hidden_states"],
                torch_dtype=torch.float8_e4m3fn,
            ).to(dtype=torch.bfloat16)
            target_last_hidden_states = self._read_float8_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_last_hidden_states_offset"],
                shape=(seq_len, self.hidden_size),
                nbytes=nbytes["target_last_hidden_states"],
                torch_dtype=torch.float8_e4m3fn,
            ).to(dtype=torch.bfloat16)
        elif self.hidden_dtype == "float8_e5m2":
            target_hidden_states = self._read_float8_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_hidden_states_offset"],
                shape=(seq_len, self.num_target_layers * self.hidden_size),
                nbytes=nbytes["target_hidden_states"],
                torch_dtype=torch.float8_e5m2,
            ).to(dtype=torch.bfloat16)
            target_last_hidden_states = self._read_float8_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_last_hidden_states_offset"],
                shape=(seq_len, self.hidden_size),
                nbytes=nbytes["target_last_hidden_states"],
                torch_dtype=torch.float8_e5m2,
            ).to(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported hidden_dtype: {self.hidden_dtype}")
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden_states,
            "target_last_hidden_states": target_last_hidden_states,
        }


def _pad_1d_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


def _pad_hidden_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    hidden_dim = features[0][key].shape[1]
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length, hidden_dim), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


class ConversationCollator:
    def __init__(
        self,
        tokenizer,
        chat_template,
        max_length,
        min_loss_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)

    def _process_feature(self, item):
        processed = preprocess_record(
            record=item,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
        )
        if int(processed["loss_mask"].sum().item()) < self.min_loss_tokens:
            return None
        return processed

    def __call__(self, features: List[Dict]):
        features = [self._process_feature(item) for item in features]
        features = [item for item in features if item is not None]
        if not features:
            return None
        batch = {}
        for key in ("input_ids", "attention_mask", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        return batch


class CacheCollator:
    def __call__(self, features: List[Dict]):
        batch = {}
        for key in ("input_ids", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        for i, item in enumerate(features):
            attention_mask[i, : item["input_ids"].shape[0]] = 1
        batch["attention_mask"] = attention_mask
        for key in ("target_hidden_states", "target_last_hidden_states"):
            batch[key] = _pad_hidden_batch(features, key)
        return batch
