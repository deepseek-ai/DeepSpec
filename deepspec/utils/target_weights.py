from __future__ import annotations

import json
import os
from collections.abc import Iterable

import torch
from torch import nn


_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_FILE = "model.safetensors"
_PYTORCH_INDEX = "pytorch_model.bin.index.json"
_PYTORCH_FILE = "pytorch_model.bin"

_EMBED_SUFFIXES = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "text_model.embed_tokens.weight",
    "embed_tokens.weight",
)
_LM_HEAD_SUFFIXES = (
    "lm_head.weight",
    "language_model.lm_head.weight",
    "text_model.lm_head.weight",
)


class _WeightOnlyModule(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


def load_target_embeddings_and_head(
    model_name_or_path: str,
    *,
    embed_shape: torch.Size,
    lm_head_shape: torch.Size,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module]:
    """Load only target input embeddings and lm_head from a HF checkpoint."""

    reader = _CheckpointReader(model_name_or_path)
    embed_weight = reader.find_tensor(
        suffixes=_EMBED_SUFFIXES,
        expected_shape=tuple(embed_shape),
        description="input embeddings",
    )
    lm_head_weight = reader.find_tensor(
        suffixes=_LM_HEAD_SUFFIXES,
        expected_shape=tuple(lm_head_shape),
        description="lm_head",
        required=False,
    )
    if lm_head_weight is None:
        assert tuple(embed_weight.shape) == tuple(lm_head_shape), (
            "Target checkpoint does not contain lm_head.weight and input "
            "embeddings cannot be reused as a tied lm_head because shapes differ: "
            f"embed={tuple(embed_weight.shape)}, lm_head={tuple(lm_head_shape)}"
        )
        lm_head_weight = embed_weight

    embed_weight = embed_weight.detach().cpu().to(dtype=dtype).contiguous()
    lm_head_weight = lm_head_weight.detach().cpu().to(dtype=dtype).contiguous()
    return _WeightOnlyModule(embed_weight), _WeightOnlyModule(lm_head_weight)


class _CheckpointReader:
    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.local_dir = (
            os.path.abspath(os.path.expanduser(self.model_name_or_path))
            if os.path.isdir(os.path.expanduser(self.model_name_or_path))
            else None
        )
        self.index_path = self._resolve_file(_SAFETENSORS_INDEX)
        self.is_safetensors = True
        if self.index_path is not None:
            self.weight_map = self._read_weight_map(self.index_path)
            return

        self.single_file = self._resolve_file(_SAFETENSORS_FILE)
        if self.single_file is not None:
            self.weight_map = None
            return

        self.index_path = self._resolve_file(_PYTORCH_INDEX)
        self.is_safetensors = False
        if self.index_path is not None:
            self.weight_map = self._read_weight_map(self.index_path)
            return

        self.single_file = self._resolve_file(_PYTORCH_FILE)
        if self.single_file is not None:
            self.weight_map = None
            self._state_dict_cache = {}
            return

        raise FileNotFoundError(
            "Could not find a Hugging Face checkpoint in "
            f"{self.model_name_or_path!r}. Expected one of: "
            f"{_SAFETENSORS_INDEX}, {_SAFETENSORS_FILE}, "
            f"{_PYTORCH_INDEX}, {_PYTORCH_FILE}."
        )

    def find_tensor(
        self,
        *,
        suffixes: Iterable[str],
        expected_shape: tuple[int, ...],
        description: str,
        required: bool = True,
    ) -> torch.Tensor | None:
        candidates = self._candidate_keys(suffixes)
        for key in candidates:
            tensor = self._get_tensor(key)
            if tuple(tensor.shape) == expected_shape:
                return tensor
        if not required:
            return None
        raise KeyError(
            f"Could not find target {description} with shape {expected_shape} in "
            f"{self.model_name_or_path!r}. Tried keys: {candidates}"
        )

    def _candidate_keys(self, suffixes: Iterable[str]) -> list[str]:
        suffixes = tuple(suffixes)
        keys = list(self.weight_map) if self.weight_map is not None else self._keys()
        candidates = [
            key
            for key in keys
            if any(key == suffix or key.endswith(f".{suffix}") for suffix in suffixes)
        ]
        return sorted(candidates, key=lambda key: self._key_priority(key, suffixes))

    @staticmethod
    def _key_priority(key: str, suffixes: tuple[str, ...]) -> tuple[int, int, str]:
        for index, suffix in enumerate(suffixes):
            if key == suffix or key.endswith(f".{suffix}"):
                return index, len(key), key
        return len(suffixes), len(key), key

    @staticmethod
    def _read_weight_map(index_path: str) -> dict[str, str]:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        return dict(index["weight_map"])

    def _resolve_file(self, filename: str) -> str | None:
        if self.local_dir is not None:
            path = os.path.join(self.local_dir, filename)
            return path if os.path.exists(path) else None

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

        try:
            return hf_hub_download(self.model_name_or_path, filename)
        except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
            return None

    def _resolve_shard(self, shard_name: str) -> str:
        if self.local_dir is not None:
            return os.path.join(self.local_dir, shard_name)

        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.model_name_or_path, shard_name)

    def _keys(self) -> list[str]:
        if self.is_safetensors:
            from safetensors import safe_open

            with safe_open(self.single_file, framework="pt", device="cpu") as handle:
                return list(handle.keys())
        return list(self._load_state_dict(self.single_file))

    def _get_tensor(self, key: str) -> torch.Tensor:
        checkpoint_file = (
            self._resolve_shard(self.weight_map[key])
            if self.weight_map is not None
            else self.single_file
        )
        if self.is_safetensors:
            from safetensors import safe_open

            with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        return self._load_state_dict(checkpoint_file)[key]

    def _load_state_dict(self, checkpoint_file: str) -> dict[str, torch.Tensor]:
        if not hasattr(self, "_state_dict_cache"):
            self._state_dict_cache = {}
        if checkpoint_file not in self._state_dict_cache:
            try:
                state_dict = torch.load(
                    checkpoint_file,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                state_dict = torch.load(checkpoint_file, map_location="cpu")
            if "state_dict" in state_dict and isinstance(
                state_dict["state_dict"], dict
            ):
                state_dict = state_dict["state_dict"]
            self._state_dict_cache[checkpoint_file] = state_dict
        return self._state_dict_cache[checkpoint_file]
