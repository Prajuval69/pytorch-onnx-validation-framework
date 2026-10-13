import io
import zipfile

import numpy as np
import pytest
import torch

from amlv.case_store import (
    array_fingerprint,
    load_npz,
    read_json,
    safe_path,
    save_npz,
    sha256_file,
    write_json,
)
from amlv.config import DISTRIBUTIONS, validate_config
from amlv.data_gen import generate, generate_suite
from amlv.errors import ArtifactError, ConfigError
from amlv.exporter import model_arrays
from amlv.models import ModelSpec, build_model


@pytest.mark.parametrize("model_id", ["mlp", "cnn", "attention"])
def test_model_init_preserves_rng_and_repeats(model_id):
    state = torch.get_rng_state().clone()
    spec = ModelSpec(model_id)
    a, b = build_model(spec), build_model(spec)
    assert torch.equal(state, torch.get_rng_state())
    assert array_fingerprint(model_arrays(a)) == array_fingerprint(model_arrays(b))
    assert array_fingerprint(model_arrays(a)) != array_fingerprint(
        model_arrays(build_model(ModelSpec(model_id, 2027)))
    )
    assert not a.training
    assert all(not p.requires_grad for p in a.parameters())


@pytest.mark.parametrize("model_id", ["mlp", "cnn", "attention"])
def test_model_cpu_float32_independent_of_default_device_dtype(model_id):
    spec = ModelSpec(model_id)
    original = array_fingerprint(model_arrays(build_model(spec)))
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            model = build_model(spec)
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        assert array_fingerprint(model_arrays(model)) == original
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_generator_reproducible_bounded(distribution):
    spec = ModelSpec("mlp")
    a = generate(spec, 4, distribution, 2026)
    b = generate(spec, 4, distribution, 2026)
    assert a.case_id == b.case_id
    assert a.fingerprint == b.fingerprint
    assert a.inputs["input"].dtype == np.float32
    assert np.all(np.isfinite(a.inputs["input"]))
    if distribution not in ("zeros", "ones"):
        c = generate(spec, 4, distribution, 2027)
        assert c.fingerprint != a.fingerprint
    if distribution == "large_bounded":
        assert np.max(np.abs(a.inputs["input"])) <= 100
    if distribution == "near_zero":
        assert np.max(np.abs(a.inputs["input"])) <= 1e-6


def test_case_selection_independent_and_invalid_bounds():
    spec = ModelSpec("mlp")
    full = {c.case_id for c in generate_suite(spec, validate_config({}))}
    selected = generate_suite(spec, validate_config({"cases": {"distributions": ["normal"]}}))
    assert all(c.case_id in full for c in selected)
    with pytest.raises(ConfigError):
        generate(spec, 4, "normal", 1, max_bytes=1)
    with pytest.raises(ConfigError):
        generate(spec, 1, "invalid", 1)


def test_safe_roundtrip_integrity_and_no_pickle(tmp_path):
    path = tmp_path / "inputs.npz"
    arrays = {"input": np.arange(10, dtype=np.float32)}
    checksum = save_npz(path, arrays)
    restored = load_npz(path, checksum)
    np.testing.assert_array_equal(restored["input"], arrays["input"])
    with np.load(path, allow_pickle=False) as data:
        assert data["input"].dtype == np.float32
    with pytest.raises(ArtifactError, match="integrity"):
        load_npz(path, "bad")
    with pytest.raises(ArtifactError):
        load_npz(tmp_path / "missing.npz", checksum)
    path.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        load_npz(path, sha256_file(path))
    with pytest.raises(ArtifactError, match="Object"):
        save_npz(path, {"bad": np.array([object()], dtype=object)})


def test_npz_declared_size_guard(tmp_path):
    path = tmp_path / "giant.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": "<f4",
            "fortran_order": False,
            "shape": (2**40,),
        },
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("input.npy", header.getvalue())
    with pytest.raises(ArtifactError, match="dimensions"):
        load_npz(path, sha256_file(path))


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "nested/../../escape",
        "",
        "C:/escape",
        "..\\escape",
        "javascript:alert(1)",
        "//remote/escape",
    ],
)
def test_path_traversal(tmp_path, relative):
    with pytest.raises(ArtifactError):
        safe_path(tmp_path, relative)


def test_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "case.json"
    with pytest.raises(ValueError):
        write_json(path, {"value": np.nan})
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ArtifactError):
        read_json(path)
