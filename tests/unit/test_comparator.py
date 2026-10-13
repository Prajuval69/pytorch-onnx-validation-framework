import json

import numpy as np
import pytest

from amlv.comparator import compare


def comparison(a, b, **kwargs):
    return compare(
        {"output": np.array(a, dtype=np.float64)},
        {"output": np.array(b, dtype=np.float64)},
        **kwargs,
    )


def test_exact_metrics_and_zero_cosine():
    result = comparison([0, 0], [0, 0])
    assert result["status"] == "PASS"
    assert result["metrics"]["output"]["cosine_similarity"] is None
    assert result["metrics"]["output"]["rmse"] == 0
    result = comparison([3, 4], [3, 4])
    assert result["metrics"]["output"]["cosine_similarity"] == 1


def test_tolerance_boundary_and_reference_asymmetry():
    # Binary-exact values demonstrate <= at the boundary and reference-scaled tolerance.
    assert comparison([1], [1.5], atol=0.25, rtol=0.25)["status"] == "PASS"
    assert comparison([1], [np.nextafter(1.5, 2)], atol=0.25, rtol=0.25)["status"] == "FAIL"
    assert comparison([1], [2], atol=0, rtol=0.5)["status"] == "FAIL"
    assert comparison([2], [1], atol=0, rtol=0.5)["status"] == "PASS"


def test_near_zero_and_worst_failure():
    result = comparison([[0, 1], [2, 3]], [[1e-6, 1], [2, 4]], atol=1e-5, rtol=1e-4)
    assert result["status"] == "FAIL"
    m = result["metrics"]["output"]
    assert m["mismatched_elements"] == 1
    assert m["total_elements"] == 4
    assert m["worst"]["index"] == [1, 1]
    assert m["worst"]["reference"] == 3
    assert m["max_rel"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["mean_abs"] == pytest.approx(0.25000025)


def test_cosine_does_not_determine_verdict():
    result = comparison([1, 2], [2, 4])
    assert result["status"] == "FAIL"
    assert result["metrics"]["output"]["cosine_similarity"] == pytest.approx(1)


@pytest.mark.parametrize(
    "a,b,policy,status",
    [
        ([np.nan], [np.nan], "fail", "FAIL"),
        ([np.inf], [np.inf], "fail", "FAIL"),
        ([np.nan, np.inf, -np.inf], [np.nan, np.inf, -np.inf], "equal", "PASS"),
        ([np.inf], [-np.inf], "equal", "FAIL"),
        ([np.nan], [0], "equal", "FAIL"),
    ],
)
def test_nonfinite(a, b, policy, status):
    result = comparison(a, b, nan_policy=policy)
    assert result["status"] == status
    assert result["metrics"]["output"]["max_abs"] is None
    assert result["metrics"]["output"]["cosine_similarity"] is None
    json.dumps(result, allow_nan=False)


def test_multi_output_shape_names_dtype():
    a = np.array([1], np.float32)
    assert compare({"a": a, "b": a}, {"b": a, "a": a})["status"] == "PASS"
    assert compare({"a": a}, {"b": a})["status"] == "FAIL"
    assert compare({"a": a}, {"a": a.reshape(1, 1)})["status"] == "FAIL"
    assert compare({"a": a}, {"a": a.astype(np.float64)})["status"] == "FAIL"
    assert compare({}, {})["status"] == "FAIL"


@pytest.mark.parametrize("options", [{"atol": -1}, {"rtol": np.inf}, {"nan_policy": "ignore"}])
def test_invalid_comparison_options(options):
    with pytest.raises(ValueError):
        comparison([0], [0], **options)


def test_extreme_finite_values_still_json_safe():
    result = comparison([1e308], [-1e308])
    assert result["status"] == "FAIL"
    assert result["metrics"]["output"]["mismatched_elements"] == 1
    json.dumps(result, allow_nan=False)
