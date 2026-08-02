import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_jsonl_dataset_module(monkeypatch, cache_dir: Path):
    class Dataset:
        pass

    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=Dataset))
    constants = types.ModuleType("deepspec.utils.constant")
    constants.CACHE_DIR = str(cache_dir)
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "deepspec", types.ModuleType("deepspec"))
    monkeypatch.setitem(sys.modules, "deepspec.utils", types.ModuleType("deepspec.utils"))
    monkeypatch.setitem(sys.modules, "deepspec.utils.constant", constants)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    module_path = Path(__file__).parents[1] / "deepspec" / "data" / "jsonl_dataset.py"
    spec = importlib.util.spec_from_file_location("jsonl_dataset_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jsonl_dataset_skips_empty_shards(monkeypatch, tmp_path):
    module = _load_jsonl_dataset_module(monkeypatch, tmp_path / "cache")
    empty_path = tmp_path / "empty.jsonl"
    data_path = tmp_path / "data.jsonl"
    empty_path.write_bytes(b"")
    rows = [{"id": 1}, {"id": 2}]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dataset = module.JsonLineDataset([data_path, empty_path])

    assert len(dataset) == 2
    assert [dataset[index] for index in range(len(dataset))] == rows
    assert dataset.num_data_per_file == [2, 0]
    dataset.close()

    cached_dataset = module.JsonLineDataset([data_path, empty_path])

    assert len(cached_dataset) == 2
    assert cached_dataset.num_data_per_file == [2, 0]
    cached_dataset.close()


def test_jsonl_dataset_accepts_only_empty_shards(monkeypatch, tmp_path):
    module = _load_jsonl_dataset_module(monkeypatch, tmp_path / "cache")
    paths = [tmp_path / "empty-1.jsonl", tmp_path / "empty-2.jsonl"]
    for path in paths:
        path.write_bytes(b"")

    dataset = module.JsonLineDataset(paths)

    assert len(dataset) == 0
    assert dataset.num_data_per_file == [0, 0]
    dataset.close()
