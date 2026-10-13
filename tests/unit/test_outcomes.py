import numpy as np
import pytest

from amlv.config import validate_config
from amlv.data_gen import generate
from amlv.errors import InvalidInput, RuntimeFailure
from amlv.models import ModelSpec, build_model
from amlv.pipeline import apply_fault, evaluate_case, exit_code


@pytest.mark.parametrize(
    "summary,expected",
    [
        ({"error": 0, "fail": 0, "skip": 1}, 0),
        ({"error": 0, "fail": 1}, 1),
        ({"error": 1, "fail": 1}, 3),
    ],
)
def test_exit_priority(summary, expected):
    assert exit_code(summary) == expected


class StubRunner:
    def __init__(self, exception=None):
        self.exception = exception

    def run(self, inputs):
        if self.exception:
            raise self.exception
        return {"output": np.zeros((1, 8), np.float32)}


@pytest.mark.parametrize(
    "exception,status",
    [
        (InvalidInput("wrong rank", "rank"), "PASS"),
        (None, "FAIL"),
        (RuntimeFailure("session defect"), "ERROR"),
        (InvalidInput("wrong reason", "dtype"), "ERROR"),
    ],
)
def test_negative_taxonomy(exception, status):
    spec = ModelSpec("mlp")
    case = generate(spec, 1, "zeros", 2026, negative_reason="rank")
    result = evaluate_case(
        case, build_model(spec), spec, StubRunner(exception), validate_config({})["compare"]
    )
    assert result["status"] == status
    assert result["metrics"] == {}


def test_unexpected_parity_runtime_error():
    spec = ModelSpec("mlp")
    case = generate(spec, 1, "zeros", 2026)
    result = evaluate_case(
        case,
        build_model(spec),
        spec,
        StubRunner(RuntimeFailure("broken")),
        validate_config({})["compare"],
    )
    assert result["status"] == "ERROR"
    assert result["error"]["category"] == "RuntimeFailure"


def test_fault_does_not_mutate_backend_output():
    source = {"output": np.zeros((1, 8), np.float32)}
    target = apply_fault(
        source, {"type": "add", "output": "output", "index": [0, 0], "amplitude": 1.0}
    )
    assert source["output"][0, 0] == 0
    assert target["output"][0, 0] == 1
