from types import SimpleNamespace

import pytest

from scripts.data.generate_train_data import build_query_kwargs, validate_args


def make_args(**overrides):
    values = {
        "model": "test-model",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "repetition_penalty": None,
        "enable_thinking": False,
        "disable_thinking": False,
        "is_gpt_oss": False,
        "concurrency": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_query_kwargs_forwards_repetition_penalty_to_sglang():
    kwargs = build_query_kwargs(
        make_args(repetition_penalty=1.1),
        [{"role": "user", "content": "hello"}],
    )

    assert kwargs["extra_body"]["repetition_penalty"] == 1.1
    assert "presence_penalty" not in kwargs


@pytest.mark.parametrize("repetition_penalty", [0.0, -1.0])
def test_validate_args_rejects_non_positive_repetition_penalty(
    repetition_penalty,
):
    with pytest.raises(ValueError, match="repetition-penalty must be greater than 0"):
        validate_args(make_args(repetition_penalty=repetition_penalty))
