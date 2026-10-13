import csv
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from amlv.case_store import sha256_file

pytestmark = pytest.mark.cli


@pytest.mark.parametrize("command", [None, "demo", "doctor", "validate", "report", "reproduce"])
def test_help(cli_run, command):
    args = [command, "--help"] if command else ["--help"]
    result = cli_run(*args)
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_doctor(cli_run, tmp_path):
    result = cli_run("doctor", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["healthy"] and data["artifact_root_writable"]
    assert data["selected_providers"] == ["CPUExecutionProvider"]


def test_demo_schemas_reports_and_actual_parity(demo_run):
    manifest = json.loads((demo_run / "manifest.json").read_text())
    case_schema = json.loads(Path("spec/schemas/case.schema.json").read_text())
    manifest_schema = json.loads(Path("spec/schemas/manifest.schema.json").read_text())
    jsonschema.validate(manifest, manifest_schema, format_checker=jsonschema.FormatChecker())
    cases = [json.loads(line) for line in (demo_run / "cases.jsonl").read_text().splitlines()]
    assert manifest["summary"] == {"total": 30, "pass": 30, "fail": 0, "error": 0, "skip": 0}
    for case in cases:
        jsonschema.validate(case, case_schema)
        assert (demo_run / case["inputs_path"]).is_file()
        if case["test_kind"] == "parity":
            assert case["metrics"]["output"]["total_elements"] > 0
            assert case["metrics"]["output"]["mismatched_elements"] == 0
        else:
            assert all(v["status"] == "PASS" for v in case["backend_outcomes"].values())
            assert case["expected_exception"]["category"] == "InvalidInput"
    for model_id in ("mlp", "cnn", "attention"):
        export = manifest["models"][model_id]
        assert export["graph_check"] == "PASS"
        assert export["selected_providers"] == ["CPUExecutionProvider"]
        assert export["dynamic_batch"]
        if export["exporter_mode"] == "legacy":
            assert export["fallback_reason"]
    rows = list(csv.DictReader((demo_run / "metrics.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == len(cases)
    assert all(row["max_abs"] == "" for row in rows if row["test_kind"] == "negative_input")
    report = (demo_run / "report.html").read_text(encoding="utf-8")
    assert "AMLV · Synthetic model validation" in report
    assert "synthetic inputs" in report
    assert "<script src=" not in report and "<link " not in report
    assert 'id="search"' in report
    assert "No observed mismatches" in report


def test_fault_cli_and_both_replay_modes(cli_run, fault_run, tmp_path):
    root, failed = fault_run
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["summary"]["fail"] == 1
    assert failed["intentional_fault_injection"] and failed["fault_replay_requested"]
    assert failed["metrics"]["output"]["mismatched_elements"] == 1
    assert "synthetic_fault_injection=true" in (root / "report.html").read_text()
    case_path = root / failed["case_path"]
    result = cli_run("reproduce", "--case", str(case_path), "--out", str(tmp_path / "fault"))
    assert result.returncode == 1, result.stdout + result.stderr
    replay = json.loads((tmp_path / "fault" / failed["case_path"]).read_text())
    assert replay["metrics"] == failed["metrics"]
    assert replay["input_sha256"] == failed["input_sha256"]
    assert replay["onnx_sha256"] == failed["onnx_sha256"]
    result = cli_run(
        "reproduce", "--case", str(case_path), "--out", str(tmp_path / "clean"), "--without-fault"
    )
    assert result.returncode == 0, result.stderr
    clean = json.loads((tmp_path / "clean" / failed["case_path"]).read_text())
    assert clean["status"] == "PASS"
    assert not clean["intentional_fault_injection"]
    result = cli_run(
        "reproduce", "--case", str(case_path), "--out", str(tmp_path / "tolerance"), "--atol", "2"
    )
    assert result.returncode == 0, result.stderr
    overridden = json.loads((tmp_path / "tolerance" / failed["case_path"]).read_text())
    assert overridden["reproduction"]["tolerance_override"]
    assert overridden["compare"]["atol"] == 2


def test_report_preserves_evidence_without_importing_torch(demo_run, cli_run):
    before = {p.name: sha256_file(p) for p in demo_run.glob("*.json*")}
    result = cli_run("report", "--run", str(demo_run))
    assert result.returncode == 0, result.stderr
    assert before == {p.name: sha256_file(p) for p in demo_run.glob("*.json*")}
    # ReportBuilder itself must never load models; the CLI keeps inference imports lazy.
    import subprocess
    import sys

    code = (
        "import sys; from amlv.cli import main; main(['report','--run',sys.argv[1]]); "
        "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(demo_run)], capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        {"models": ["unavailable"]},
        {"compare": {"rtol": -0.1}},
        {"extra": 1},
        {"cases": {"distributions": ["unrecognized"]}},
        {"export": {"dynamo": "yes"}},
    ],
)
def test_invalid_config_exit(cli_run, tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = cli_run("validate", "--config", str(path), "--out", str(tmp_path / "out"))
    assert result.returncode == 2, result.stderr
    assert "CONFIG:" in result.stderr
    assert not (tmp_path / "out").exists()


def test_missing_config_and_occupied_output(cli_run, demo_run, tmp_path):
    result = cli_run("validate", "--config", str(tmp_path / "missing.yaml"), "--out", "unused")
    assert result.returncode == 2
    result = cli_run("demo", "--out", str(demo_run))
    assert result.returncode == 3
    assert "refusing to overwrite" in result.stderr
    file = tmp_path / "file"
    file.write_text("preserve me")
    result = cli_run("demo", "--out", str(file))
    assert result.returncode == 3
    assert file.read_text() == "preserve me"


def test_full_example_config(cli_run, tmp_path):
    out = tmp_path / "example"
    result = cli_run("validate", "--config", "config/validation.example.yaml", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "manifest.json").read_text())["summary"]
    assert summary == {"total": 66, "pass": 66, "fail": 0, "error": 0, "skip": 0}
