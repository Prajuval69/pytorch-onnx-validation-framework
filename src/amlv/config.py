"""Strict, bounded YAML configuration without implicit coercion."""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from amlv.errors import ConfigError

MODELS = ("mlp", "cnn", "attention")
DISTRIBUTIONS = ("normal", "uniform", "zeros", "ones", "near_zero", "large_bounded")
DEFAULTS = {
    "schema_version": 1,
    "seed": 2026,
    "models": list(MODELS),
    "providers": ["CPUExecutionProvider"],
    "export": {"dynamo": True, "dynamic_batch": True, "opset_version": None},
    "cases": {
        "batch_sizes": [1, 4],
        "distributions": ["normal", "zeros", "near_zero"],
        "dtype": "float32",
        "include_invalid_input_tests": True,
        "max_input_megabytes": 128,
    },
    "compare": {"atol": 1e-5, "rtol": 1e-4, "nan_policy": "fail"},
    "report": {"html": True, "csv": True, "json": True},
}


def _merge(raw: dict, defaults: dict, prefix: str = "") -> dict:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix or 'config'} must be a mapping")
    if unknown := set(raw) - set(defaults):
        raise ConfigError(f"Unknown keys in {prefix or 'config'}: {sorted(unknown, key=str)}")
    result = deepcopy(defaults)
    for key, value in raw.items():
        result[key] = (
            _merge(value, defaults[key], prefix + key + ".")
            if isinstance(defaults[key], dict)
            else value
        )
    return result


def _number(value: Any, name: str, minimum: float = 0) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigError(f"{name} must be a finite number >= {minimum}")


def validate_config(raw: dict) -> dict:
    config = _merge(raw, DEFAULTS)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ConfigError("seed must be an integer in [0, 2**32)")
    models = config["models"]
    if not isinstance(models, list) or not models or any(m not in MODELS for m in models):
        raise ConfigError(f"models must be a nonempty list from {MODELS}")
    if len(set(models)) != len(models):
        raise ConfigError("models must not repeat")
    if config["providers"] != ["CPUExecutionProvider"]:
        raise ConfigError("This CPU release requires providers: [CPUExecutionProvider]")
    for section, keys in {
        "export": ("dynamo", "dynamic_batch"),
        "cases": ("include_invalid_input_tests",),
        "report": ("html", "csv", "json"),
    }.items():
        for key in keys:
            if type(config[section][key]) is not bool:
                raise ConfigError(f"{section}.{key} must be a boolean")
    if not config["report"]["json"]:
        raise ConfigError("report.json must be true: immutable JSON is required for replay")
    opset = config["export"]["opset_version"]
    if opset is not None and (type(opset) is not int or not 18 <= opset <= 23):
        raise ConfigError("opset_version must be null or an integer from 18 to 23")
    cases = config["cases"]
    batches = cases["batch_sizes"]
    if (
        not isinstance(batches, list)
        or not batches
        or any(type(b) is not int or not 1 <= b <= 8 for b in batches)
        or len(set(batches)) != len(batches)
    ):
        raise ConfigError("batch_sizes must be distinct integers from 1 to 8")
    if not config["export"]["dynamic_batch"] and len(batches) != 1:
        raise ConfigError("Static export requires exactly one batch size")
    distributions = cases["distributions"]
    if (
        not isinstance(distributions, list)
        or not distributions
        or any(d not in DISTRIBUTIONS for d in distributions)
        or len(set(distributions)) != len(distributions)
    ):
        raise ConfigError(f"distributions must be distinct names from {DISTRIBUTIONS}")
    if cases["dtype"] != "float32":
        raise ConfigError("Ordinary parity inputs must use float32")
    _number(cases["max_input_megabytes"], "max_input_megabytes", minimum=0.000001)
    # Largest supported input family is CNN, 256 elements per batch.
    if max(batches) * 256 * 8 > cases["max_input_megabytes"] * 1024**2:
        raise ConfigError(
            "Configured cases exceed max_input_megabytes (including negative float64)"
        )
    for key in ("atol", "rtol"):
        _number(config["compare"][key], f"compare.{key}")
    if config["compare"]["nan_policy"] not in ("fail", "equal"):
        raise ConfigError("compare.nan_policy must be fail or equal")
    return config


def load_config(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load config {path}: {exc}") from exc
    return validate_config(raw)
