import pytest

from amlv.config import load_config, validate_config
from amlv.errors import ConfigError


def test_example():
    from pathlib import Path

    config = load_config(Path("config/validation.example.yaml"))
    assert config["models"] == ["mlp", "cnn", "attention"]


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": True},
        {"models": ["unknown"]},
        {"compare": {"atol": -1}},
        {"cases": {"distributions": ["unknown"]}},
        {"seed": True},
        {"cases": {"batch_sizes": [100000]}},
        {"compare": {"rtol": float("nan")}},
        {"compare": {"atol": 2**4096}},
        {"export": {"dynamic_batch": False}},
        {"cases": {"max_input_megabytes": 0.001}},
    ],
)
def test_invalid(raw):
    with pytest.raises(ConfigError):
        validate_config(raw)
