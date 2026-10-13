from pathlib import Path

import numpy as np
import onnx
import pytest

from amlv.config import validate_config
from amlv.exporter import export_model
from amlv.models import ModelSpec, build_model
from amlv.runners import OnnxRunner, reference_run


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["mlp", "cnn", "attention"])
def test_real_export_dynamic_and_cache(tmp_path, model_id):
    spec = ModelSpec(model_id)
    model = build_model(spec)
    record = export_model(model, spec, tmp_path, validate_config({})["export"])
    path = tmp_path / record["onnx_path"]
    onnx.checker.check_model(str(path))
    runner = OnnxRunner(path, spec)
    for batch in (1, 4):
        inputs = {"input": np.ones(spec.shape(batch), np.float32)}
        ref, actual = reference_run(model, spec, inputs), runner.run(inputs)
        np.testing.assert_allclose(actual["output"], ref["output"], atol=1e-5, rtol=1e-4)
    second = export_model(model, spec, tmp_path, validate_config({})["export"])
    assert second["cache_hit"]
    assert second["onnx_sha256"] == record["onnx_sha256"]
    assert list(Path(tmp_path / "models").glob("*.onnx")) == [path]
