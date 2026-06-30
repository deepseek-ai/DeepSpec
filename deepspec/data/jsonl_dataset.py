import hashlib
import json
import mmap
import os
import pickle
from bisect import bisect_right

import torch
from tqdm import tqdm

CACHE_DIR = os.path.expanduser("~/.cache/deepspec")
LINE_INDEX_CACHE_VERSION = 2


class JsonLineDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        super().__init__()
        self.data_paths = sorted(data_paths)
        self.cache_dir = os.path.abspath(CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.num_data_per_file = []
        self.cum_counts = [0]
        self.files = [None] * len(self.data_paths)
        self.mmaps = [None] * len(self.data_paths)
        self.line_starts_per_file = [None] * len(self.data_paths)
        self._build_all_line_starts()
        for count in self.num_data_per_file:
            self.cum_counts.append(self.cum_counts[-1] + count)
        self.num_data = self.cum_counts[-1]
        self.close()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_data):
            raise IndexError(idx)
        file_idx, local_idx = self._map_global_to_local(idx)
        if self.files[file_idx] is None or self.mmaps[file_idx] is None:
            self._open_file_mmap(file_idx)
        mm = self.mmaps[file_idx]
        starts = self.line_starts_per_file[file_idx]
        mm.seek(starts[local_idx])
        line = mm.readline().decode("utf-8")
        return json.loads(line)

    def close(self):
        for idx, mm in enumerate(self.mmaps):
            if mm is not None:
                mm.close()
                self.mmaps[idx] = None
        for idx, handle in enumerate(self.files):
            if handle is not None:
                handle.close()
                self.files[idx] = None

    def __del__(self):  # pragma: no cover
        self.close()

    def _map_global_to_local(self, global_idx):
        file_idx = bisect_right(self.cum_counts, global_idx) - 1
        local_idx = global_idx - self.cum_counts[file_idx]
        return file_idx, local_idx

    def _open_file_mmap(self, index):
        if self.files[index] is None:
            self.files[index] = open(self.data_paths[index], "rb")
        if self.mmaps[index] is None:
            self.mmaps[index] = mmap.mmap(
                self.files[index].fileno(), 0, access=mmap.ACCESS_READ
            )

    def _file_key(self, path):
        abspath = os.path.abspath(path)
        st = os.stat(abspath)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return f"{os.path.normcase(abspath)}|{mtime_ns}"

    def _cache_path_from_key(self, file_key):
        key_hash = hashlib.blake2b(file_key.encode("utf-8"), digest_size=16).hexdigest()
        return os.path.join(self.cache_dir, f"jsonlindex-{key_hash}.pkl")

    def _atomic_pickle_dump(self, obj, dst_path):
        tmp_path = f"{dst_path}.tmp-{os.getpid()}"
        with open(tmp_path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dst_path)

    def _build_all_line_starts(self):
        for idx, path in tqdm(enumerate(self.data_paths), total=len(self.data_paths)):
            file_key = self._file_key(path)
            cache_path = self._cache_path_from_key(file_key)
            if cache_path is not None and os.path.exists(cache_path):
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if (
                    isinstance(cached, dict)
                    and cached.get("version") == LINE_INDEX_CACHE_VERSION
                    and cached.get("file_key") == file_key
                    and isinstance(cached.get("line_starts"), list)
                ):
                    starts = cached["line_starts"]
                    self.line_starts_per_file[idx] = starts
                    self.num_data_per_file.append(len(starts))
                    continue

            handle = open(path, "rb")
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.files[idx] = handle
            self.mmaps[idx] = mm
            starts = []
            mm.seek(0)
            while True:
                pos = mm.tell()
                line = mm.readline()
                if not line:
                    break
                if line.strip():
                    starts.append(pos)
            self.line_starts_per_file[idx] = starts
            self.num_data_per_file.append(len(starts))
            if cache_path is not None:
                self._atomic_pickle_dump(
                    {
                        "version": LINE_INDEX_CACHE_VERSION,
                        "file_key": file_key,
                        "file_path": os.path.abspath(path),
                        "line_starts": starts,
                    },
                    cache_path,
                )
