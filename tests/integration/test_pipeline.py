import json
import shutil

import numpy as np
import pytest

from amlv.case_store import read_json, safe_path
from amlv.config import validate_config
from amlv.errors import ArtifactError, ExportFailure
from amlv.exporter import export_model
from amlv.models import ModelSpec, build_model
from amlv.pipeline import reproduce, run_suite
from amlv.reporting import build_report, read_run
from amlv.runners import OnnxRunner

pytestmark = pytest.mark.integration


def test_second_demo_determinism(demo_run, tmp_path):
    first_manifest, first = read_run(demo_run)
    second_manifest = run_suite(validate_config({}), tmp_path / "second")
    _, second = read_run(tmp_path / "second")
    assert first_manifest["run_id"] != second_manifest["run_id"]
    assert [
        (r["case_id"], r["input_fingerprint"], r["input_sha256"], r["status"]) for r in first
    ] == [(r["case_id"], r["input_fingerprint"], r["input_sha256"], r["status"]) for r in second]


@pytest.mark.parametrize("kind", ["inputs", "onnx", "weights", "case_json"])
def test_replay_refuses_corruption(demo_run, tmp_path, kind):
    root = tmp_path / "source"
    shutil.copytree(demo_run, root)
    _, records = read_run(root)
    case = records[0]
    path = (
        root
        / {
            "inputs": case["inputs_path"],
            "onnx": case["export"]["onnx_path"],
            "weights": case["export"]["weights_path"],
            "case_json": case["case_path"],
        }[kind]
    )
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError):
        reproduce(root / case["case_path"], tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_report_refuses_jsonl_corruption(demo_run, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(demo_run, root)
    with (root / "cases.jsonl").open("a") as stream:
        stream.write("corruption")
    with pytest.raises(ArtifactError, match="integrity"):
        build_report(root)


def test_setup_error_has_reports_and_traceback(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ExportFailure("controlled exporter failure")

    monkeypatch.setattr("amlv.pipeline.export_model", broken)
    manifest = run_suite(validate_config({"models": ["mlp"]}), tmp_path / "error")
    assert manifest["exit_code"] == 3
    assert manifest["summary"]["error"] == 10
    assert (tmp_path / "error/report.html").is_file()
    assert "controlled exporter failure" in (tmp_path / "error/report.html").read_text()
    assert "Traceback" in (tmp_path / "error/logs/validation.log").read_text()
    _, records = read_run(tmp_path / "error")
    assert records[0]["error"]["category"] == "ExportFailure"


def test_explicit_legacy_static_and_dynamic_exports(tmp_path):
    spec = ModelSpec("mlp")
    model = build_model(spec)
    for dynamic, batches in ((True, [1, 4]), (False, [4])):
        options = validate_config(
            {
                "export": {"dynamo": False, "dynamic_batch": dynamic},
                "cases": {"batch_sizes": batches},
            }
        )["export"]
        record = export_model(model, spec, tmp_path, options, static_batch=4)
        assert record["exporter_mode"] == "legacy"
        assert record["fallback_reason"] is None
        runner = OnnxRunner(safe_path(tmp_path, record["onnx_path"]), spec)
        for batch in batches:
            assert runner.run({"input": np.zeros(spec.shape(batch), np.float32)})[
                "output"
            ].shape == (batch, 8)


def test_recorded_dynamo_fallback(monkeypatch, tmp_path):
    import torch

    original = torch.onnx.export

    def exporter(*args, **kwargs):
        if kwargs.get("dynamo"):
            raise RuntimeError("controlled dynamo capability failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", exporter)
    spec = ModelSpec("mlp")
    record = export_model(build_model(spec), spec, tmp_path, validate_config({})["export"])
    assert record["exporter_mode"] == "legacy"
    assert "controlled dynamo capability failure" in record["fallback_reason"]
    assert record["graph_check"] == "PASS"


def test_modern_api_without_removed_fallback_argument(monkeypatch, tmp_path):
    import torch

    original = torch.onnx.export
    calls = []

    def without_fallback(model, args, f=None, *, dynamo=False, **kwargs):
        # Simulate the current API signature on all supported library versions.
        assert "fallback" not in kwargs
        calls.append(dynamo)
        return original(model, args, f, dynamo=dynamo, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", without_fallback)
    spec = ModelSpec("mlp")
    record = export_model(build_model(spec), spec, tmp_path, validate_config({})["export"])
    assert calls[0] is True
    assert not record["exporter_flags"]["fallback_argument_supported"]
    assert "unexpected keyword argument 'fallback'" not in (record["fallback_reason"] or "")


def test_negative_case_replay(demo_run, tmp_path):
    _, records = read_run(demo_run)
    case = next(r for r in records if r["test_kind"] == "negative_input")
    manifest = reproduce(demo_run / case["case_path"], tmp_path / "replay")
    assert manifest["summary"]["pass"] == 1
    replay = read_json(tmp_path / "replay" / case["case_path"])
    assert replay["backend_outcomes"] == case["backend_outcomes"]


def test_report_options_are_respected(tmp_path):
    config = validate_config(
        {
            "models": ["mlp"],
            "report": {"html": False, "csv": False},
            "cases": {
                "batch_sizes": [1],
                "distributions": ["zeros"],
                "include_invalid_input_tests": False,
            },
        }
    )
    manifest = run_suite(config, tmp_path / "json_only")
    assert manifest["summary"]["pass"] == 1
    assert not (tmp_path / "json_only/report.html").exists()
    assert not (tmp_path / "json_only/metrics.csv").exists()
    assert json.loads((tmp_path / "json_only/manifest.json").read_text())["schema_version"] == 1
