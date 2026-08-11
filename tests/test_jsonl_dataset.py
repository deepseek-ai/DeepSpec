"""Tests for deepspec.data.jsonl_dataset.JsonLineDataset.

Exercises the mmap-based line indexing end to end on temporary .jsonl files:
length counting (with and without a trailing newline), per-record decoding,
multi-file global indexing, bounds checking, and the on-disk line-start cache.
CPU-only; ``CACHE_DIR`` is redirected to a temp dir so tests are hermetic.
"""

from __future__ import annotations

import json
import pickle

import pytest

import deepspec.data.jsonl_dataset as jsonl_mod
from deepspec.data.jsonl_dataset import JsonLineDataset


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect the module-level CACHE_DIR to a temp dir for each test."""
    monkeypatch.setattr(jsonl_mod, "CACHE_DIR", str(tmp_path / "jsonl_cache"))
    return tmp_path


def _write_jsonl(path, records, trailing_newline=True):
    body = "\n".join(json.dumps(r) for r in records)
    if trailing_newline and records:
        body += "\n"
    path.write_text(body, encoding="utf-8")
    return path


def test_len_with_trailing_newline(tmp_path):
    records = [{"i": 0}, {"i": 1}, {"i": 2}]
    path = _write_jsonl(tmp_path / "a.jsonl", records, trailing_newline=True)
    ds = JsonLineDataset([str(path)])
    assert len(ds) == 3
    ds.close()


def test_len_without_trailing_newline(tmp_path):
    records = [{"i": 0}, {"i": 1}, {"i": 2}]
    path = _write_jsonl(tmp_path / "a.jsonl", records, trailing_newline=False)
    ds = JsonLineDataset([str(path)])
    assert len(ds) == 3
    ds.close()


def test_getitem_roundtrips_records(tmp_path):
    records = [{"i": 0, "text": "alpha"}, {"i": 1, "text": "beta"}]
    path = _write_jsonl(tmp_path / "a.jsonl", records)
    ds = JsonLineDataset([str(path)])
    assert ds[0] == records[0]
    assert ds[1] == records[1]
    ds.close()


def test_empty_file_has_zero_length(tmp_path):
    # A 0-byte .jsonl must be treated as 0 records rather than crashing on the
    # `mmap` of an empty file (regression test for the empty-shard guard).
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    ds = JsonLineDataset([str(path)])
    assert len(ds) == 0
    ds.close()


def test_empty_shard_among_populated_files(tmp_path):
    empty = tmp_path / "a_empty.jsonl"
    empty.write_text("", encoding="utf-8")
    populated = _write_jsonl(tmp_path / "b.jsonl", [{"i": 0}, {"i": 1}])
    ds = JsonLineDataset([str(empty), str(populated)])
    assert len(ds) == 2
    assert ds[0] == {"i": 0}
    assert ds[1] == {"i": 1}
    ds.close()


def test_out_of_range_index_raises(tmp_path):
    path = _write_jsonl(tmp_path / "a.jsonl", [{"i": 0}])
    ds = JsonLineDataset([str(path)])
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-1]
    ds.close()


def test_multi_file_global_indexing(tmp_path):
    # data_paths are sorted internally; name files so order is deterministic.
    f0 = _write_jsonl(tmp_path / "a.jsonl", [{"f": 0, "i": 0}, {"f": 0, "i": 1}])
    f1 = _write_jsonl(tmp_path / "b.jsonl", [{"f": 1, "i": 0}, {"f": 1, "i": 1}, {"f": 1, "i": 2}])
    ds = JsonLineDataset([str(f1), str(f0)])  # pass unsorted on purpose
    assert len(ds) == 5
    # First two records come from a.jsonl, the next three from b.jsonl.
    assert ds[0] == {"f": 0, "i": 0}
    assert ds[1] == {"f": 0, "i": 1}
    assert ds[2] == {"f": 1, "i": 0}
    assert ds[4] == {"f": 1, "i": 2}
    ds.close()


def test_line_start_cache_is_written_and_reused(tmp_path, monkeypatch):
    cache_dir = tmp_path / "jsonl_cache"
    monkeypatch.setattr(jsonl_mod, "CACHE_DIR", str(cache_dir))
    path = _write_jsonl(tmp_path / "a.jsonl", [{"i": 0}, {"i": 1}, {"i": 2}])

    ds1 = JsonLineDataset([str(path)])
    ds1.close()
    cache_files = list(cache_dir.glob("jsonlindex-*.pkl"))
    assert len(cache_files) == 1

    cached = pickle.loads(cache_files[0].read_bytes())
    assert cached["line_starts"] == [0, len('{"i": 0}\n'), 2 * len('{"i": 0}\n')]

    # Second instance must reuse the cache and report the same length.
    ds2 = JsonLineDataset([str(path)])
    assert len(ds2) == 3
    assert ds2[2] == {"i": 2}
    ds2.close()
