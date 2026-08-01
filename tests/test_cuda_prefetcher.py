import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.dtype = "int64"
        self.recorded_streams = []

    def to(self, *args, **kwargs):
        return self

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class FakeStream:
    def wait_stream(self, stream):
        self.waited_for = stream


class ScriptedIterator:
    def __init__(self, events):
        self._events = iter(events)

    def __iter__(self):
        return self

    def __next__(self):
        event = next(self._events)
        if isinstance(event, BaseException):
            raise event
        return event


class ResettableLoader:
    def __init__(self, runs):
        self._runs = iter(runs)

    def __iter__(self):
        return ScriptedIterator(next(self._runs))

    def __len__(self):
        return 1


@pytest.fixture
def prefetcher_module(monkeypatch):
    current_stream = FakeStream()
    fake_cuda = types.SimpleNamespace(
        Stream=lambda device=None: FakeStream(),
        stream=lambda stream: contextlib.nullcontext(),
        current_stream=lambda device=None: current_stream,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda
    fake_torch.long = "int64"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_path = (
        Path(__file__).parents[1] / "deepspec" / "data" / "cuda_prefetcher.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_prefetcher_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch(value):
    return {"input_ids": FakeTensor(value)}


def test_normal_batches_are_returned_in_order(prefetcher_module):
    prefetcher = prefetcher_module.CUDAPrefetcher(
        ScriptedIterator([batch(1), batch(2)]),
        "cuda:0",
    )

    assert [item["input_ids"].value for item in prefetcher] == [1, 2]


def test_background_loader_error_is_reraised(prefetcher_module):
    error = RuntimeError("bad batch")
    prefetcher = prefetcher_module.CUDAPrefetcher(
        ScriptedIterator([batch(1), error]),
        "cuda:0",
    )
    iterator = iter(prefetcher)

    assert next(iterator)["input_ids"].value == 1
    with pytest.raises(RuntimeError, match="bad batch") as raised:
        next(iterator)
    assert raised.value is error


def test_reiteration_resets_background_error(prefetcher_module):
    loader = ResettableLoader(
        [
            [batch(1), RuntimeError("bad batch")],
            [batch(2)],
        ]
    )
    prefetcher = prefetcher_module.CUDAPrefetcher(loader, "cuda:0")
    first = iter(prefetcher)

    assert next(first)["input_ids"].value == 1
    with pytest.raises(RuntimeError, match="bad batch"):
        next(first)

    assert [item["input_ids"].value for item in prefetcher] == [2]
