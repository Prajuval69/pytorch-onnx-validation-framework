"""Independent seeded generators with stable content fingerprints and negative cases."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from amlv.case_store import array_fingerprint, canonical
from amlv.config import DISTRIBUTIONS
from amlv.errors import ConfigError
from amlv.models import ModelSpec


@dataclass
class TestCase:
    __test__ = False

    case_id: str
    model_id: str
    seed: int
    distribution: str
    inputs: dict[str, np.ndarray]
    test_kind: str = "parity"
    negative_reason: str | None = None

    @property
    def fingerprint(self) -> str:
        return array_fingerprint(self.inputs)


def generate(
    spec: ModelSpec,
    batch: int,
    distribution: str,
    seed: int,
    max_bytes: int = 128 * 1024**2,
    negative_reason: str | None = None,
) -> TestCase:
    if distribution not in DISTRIBUTIONS:
        raise ConfigError(f"Unknown distribution: {distribution}")
    shape = spec.shape(batch)
    if np.prod(shape) * 8 > max_bytes:
        raise ConfigError("Input exceeds configured memory bound")
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        array = rng.normal(0, 1, shape)
    elif distribution == "uniform":
        array = rng.uniform(-1, 1, shape)
    elif distribution == "zeros":
        array = np.zeros(shape)
    elif distribution == "ones":
        array = np.ones(shape)
    elif distribution == "near_zero":
        array = rng.uniform(-1e-6, 1e-6, shape)
    else:
        array = rng.uniform(-100, 100, shape)
    inputs = {"input": array.astype(np.float32)}
    if negative_reason == "rank":
        inputs["input"] = inputs["input"].reshape(-1)
    elif negative_reason == "shape":
        # Channel/features must be invalid even though CNN supports internal pooling.
        axis = 1 if spec.model_id == "cnn" else len(shape) - 1
        inputs["input"] = np.take(inputs["input"], np.arange(shape[axis] - 1), axis=axis)
    elif negative_reason == "dtype":
        inputs["input"] = inputs["input"].astype(np.float64)
    elif negative_reason == "missing":
        inputs = {}
    elif negative_reason is not None:
        raise ConfigError(f"Unknown negative reason: {negative_reason}")
    identity = [
        spec.model_id,
        batch,
        distribution,
        seed,
        negative_reason,
        array_fingerprint(inputs),
    ]
    case_id = hashlib.sha256(canonical(identity)).hexdigest()[:24]
    return TestCase(
        case_id,
        spec.model_id,
        seed,
        distribution,
        inputs,
        "negative_input" if negative_reason else "parity",
        negative_reason,
    )


def generate_suite(spec: ModelSpec, config: dict) -> Iterator[TestCase]:
    max_bytes = int(config["cases"]["max_input_megabytes"] * 1024**2)
    for batch in config["cases"]["batch_sizes"]:
        for distribution in config["cases"]["distributions"]:
            # Stable seeds regardless of model/distribution selection or ordering.
            salt = int.from_bytes(
                hashlib.sha256(canonical([spec.model_id, batch, distribution])).digest()[:4],
                "little",
            )
            yield generate(spec, batch, distribution, (config["seed"] + salt) % 2**32, max_bytes)
    if config["cases"]["include_invalid_input_tests"]:
        for reason in ("rank", "shape", "dtype", "missing"):
            yield generate(
                spec,
                config["cases"]["batch_sizes"][0],
                "uniform",
                config["seed"],
                max_bytes,
                reason,
            )
