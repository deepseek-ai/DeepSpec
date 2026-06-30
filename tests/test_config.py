"""Tests for deepspec.utils.config.

Characterization tests for the pure-Python config machinery: the attribute-dict
``ConfigNode``, the recursive (to|from)-config-node converters, ``jsonable``,
the ``CustomJSONEncoder``, and the ``--opts`` override parser with its error
paths. All CPU-only, no model or GPU required.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deepspec.utils.config import (
    ConfigNode,
    CustomJSONEncoder,
    config_to_plain_dict,
    finalize_config,
    jsonable,
    load_config,
    parse_opts_to_config,
    to_config_node,
)


class TestConfigNode:
    def test_attribute_access_reads_items(self):
        node = ConfigNode({"lr": 0.1})
        assert node.lr == 0.1
        assert node["lr"] == 0.1

    def test_attribute_assignment_sets_items(self):
        node = ConfigNode()
        node.epochs = 5
        assert node["epochs"] == 5

    def test_missing_attribute_raises_attribute_error(self):
        node = ConfigNode({"a": 1})
        with pytest.raises(AttributeError):
            _ = node.does_not_exist

    def test_copy_returns_config_node(self):
        node = ConfigNode({"a": 1})
        clone = node.copy()
        assert isinstance(clone, ConfigNode)
        assert clone == node
        clone.a = 2
        assert node.a == 1  # copy is shallow but independent at the top level


class TestToConfigNode:
    def test_nested_dict_becomes_config_node(self):
        out = to_config_node({"outer": {"inner": 1}})
        assert isinstance(out, ConfigNode)
        assert isinstance(out["outer"], ConfigNode)
        assert out.outer.inner == 1

    def test_list_is_preserved_with_converted_elements(self):
        out = to_config_node({"items": [{"k": 1}, 2]})
        assert isinstance(out["items"], list)
        assert isinstance(out["items"][0], ConfigNode)
        assert out["items"][1] == 2

    def test_tuple_is_preserved(self):
        out = to_config_node({"pair": ({"k": 1}, 2)})
        assert isinstance(out["pair"], tuple)
        assert isinstance(out["pair"][0], ConfigNode)

    def test_scalars_pass_through(self):
        assert to_config_node(7) == 7
        assert to_config_node("x") == "x"


class TestConfigToPlainDict:
    def test_config_node_becomes_plain_dict(self):
        node = to_config_node({"a": {"b": 1}})
        plain = config_to_plain_dict(node)
        assert type(plain) is dict
        assert type(plain["a"]) is dict
        assert plain == {"a": {"b": 1}}

    def test_tuple_becomes_list(self):
        node = to_config_node({"pair": (1, 2)})
        plain = config_to_plain_dict(node)
        assert plain["pair"] == [1, 2]


class TestJsonable:
    def test_path_becomes_string(self):
        assert jsonable(Path("/tmp/x")) == "/tmp/x"

    def test_nested_structures_are_converted(self):
        node = ConfigNode({"p": Path("/a"), "items": (1, Path("/b"))})
        out = jsonable(node)
        assert out == {"p": "/a", "items": [1, "/b"]}


class TestCustomJSONEncoder:
    def _dumps(self, obj):
        return json.loads(json.dumps(obj, cls=CustomJSONEncoder))

    def test_encodes_function(self):
        def my_fn():
            return None

        assert self._dumps({"f": my_fn}) == {"f": "<function my_fn>"}

    def test_encodes_type(self):
        assert self._dumps({"t": int}) == {"t": "<class 'int'>"}

    def test_encodes_torch_dtype(self):
        assert self._dumps({"dt": torch.float32}) == {"dt": "torch.float32"}

    def test_encodes_path(self):
        assert self._dumps({"p": Path("/a/b")}) == {"p": "/a/b"}

    def test_encodes_namespace(self):
        assert self._dumps(Namespace(a=1, b="x")) == {"a": 1, "b": "x"}
        assert self._dumps(SimpleNamespace(c=3)) == {"c": 3}

    def test_encodes_config_node(self):
        assert self._dumps(to_config_node({"a": {"b": 1}})) == {"a": {"b": 1}}


class TestParseOptsToConfig:
    def test_sets_nested_value(self):
        cfg = {"train": {"lr": 0.1}}
        out = parse_opts_to_config(["train.lr=0.5"], cfg)
        assert out.train.lr == 0.5

    def test_value_is_yaml_parsed(self):
        out = parse_opts_to_config(["train.steps=10"], {"train": {"steps": 1}})
        assert out.train.steps == 10
        assert isinstance(out.train.steps, int)

    def test_empty_opts_returns_finalized_config_node(self):
        out = parse_opts_to_config([], {"a": {"b": 1}})
        assert isinstance(out, ConfigNode)
        assert out.a.b == 1

    def test_unknown_top_level_key_raises_key_error(self):
        with pytest.raises(KeyError):
            parse_opts_to_config(["missing=1"], {"a": 1})

    def test_unknown_nested_key_raises_key_error(self):
        with pytest.raises(KeyError):
            parse_opts_to_config(["a.missing=1"], {"a": {"b": 1}})

    def test_non_mapping_intermediate_raises_type_error(self):
        with pytest.raises(TypeError):
            parse_opts_to_config(["a.b=1"], {"a": 1})


class TestFinalizeConfig:
    def test_returns_config_node_without_hook(self):
        out = finalize_config({"a": 1})
        assert isinstance(out, ConfigNode)
        assert out.a == 1

    def test_runs_finalize_cfg_hook(self):
        def finalize_cfg(cfg):
            cfg["added"] = cfg["base"] * 2
            return cfg

        out = finalize_config({"base": 3, "finalize_cfg": finalize_cfg})
        assert out.added == 6


class TestLoadConfig:
    def test_loads_module_level_vars(self, tmp_path):
        cfg_file = tmp_path / "cfg.py"
        cfg_file.write_text(
            "import os\n"  # modules must be skipped
            "lr = 0.01\n"
            "layers = [1, 2, 3]\n"
            "_private = 'kept'  # leading single underscore is NOT skipped\n"
            "__dunder__ = 'skipped'\n"
        )
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, ConfigNode)
        assert cfg.lr == 0.01
        assert cfg.layers == [1, 2, 3]
        assert "os" not in cfg  # ModuleType filtered out
        assert "__dunder__" not in cfg  # dunder filtered out
