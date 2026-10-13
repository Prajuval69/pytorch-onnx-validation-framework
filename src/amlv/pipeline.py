"""Model -> export -> cases -> both runtimes -> comparison -> durable run evidence."""

import json
import logging
import math
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from amlv.case_store import (
    array_fingerprint,
    atomic_text,
    load_npz,
    read_json,
    safe_path,
    save_npz,
    sha256_file,
    write_json,
)
from amlv.comparator import compare
from amlv.config import validate_config
from amlv.data_gen import TestCase, generate_suite
from amlv.doctor import environment, require_dependencies
from amlv.errors import ArtifactError, InvalidInput, RuntimeFailure
from amlv.exporter import export_model, model_arrays
from amlv.models import ModelSpec, build_model
from amlv.runners import OnnxRunner, reference_run

LOG = logging.getLogger("amlv")


def now() -> str:
    return datetime.now(UTC).isoformat()


def exit_code(summary: dict) -> int:
    return 3 if summary["error"] else 1 if summary["fail"] else 0


def prepare_output(root: Path) -> Path:
    root = root.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ArtifactError(f"Output must be new or empty; refusing to overwrite {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir()
    return root


def exception_record(exc: Exception) -> dict:
    return {"category": type(exc).__name__, "message": str(exc)}


def apply_fault(outputs: dict[str, np.ndarray], fault: dict | None) -> dict:
    # Always copy; original backend results and saved model are never mutated.
    result = {name: array.copy() for name, array in outputs.items()}
    if fault:
        if (
            fault.get("type") != "add"
            or fault.get("output") not in result
            or type(fault.get("amplitude")) not in (float, int)
            or not math.isfinite(fault["amplitude"])
        ):
            raise ArtifactError("Unsupported fault metadata")
        index = fault.get("index")
        output = result[fault["output"]]
        if (
            not isinstance(index, list)
            or len(index) != output.ndim
            or any(
                type(v) is not int or not 0 <= v < size
                for v, size in zip(index, output.shape, strict=True)
            )
        ):
            raise ArtifactError("Fault coordinate outside output")
        output[tuple(index)] += fault["amplitude"]
    return result


def evaluate_case(
    case: TestCase,
    model,
    spec: ModelSpec,
    runner: OnnxRunner,
    thresholds: dict,
    fault: dict | None = None,
) -> dict:
    if case.test_kind == "negative_input":
        outcomes = {}
        for backend, run in {
            "reference": lambda: reference_run(model, spec, case.inputs),
            "onnx": lambda: runner.run(case.inputs),
        }.items():
            try:
                run()
                outcomes[backend] = {"status": "FAIL", "message": "Unexpected acceptance"}
            except InvalidInput as exc:
                expected = exc.reason == case.negative_reason
                outcomes[backend] = {
                    "status": "PASS" if expected else "ERROR",
                    **exception_record(exc),
                    "reason": exc.reason,
                }
                if not expected:
                    LOG.exception("Wrong negative-input rejection category")
            except Exception as exc:
                LOG.exception("Unexpected %s error for negative case %s", backend, case.case_id)
                outcomes[backend] = {"status": "ERROR", **exception_record(exc)}
        counts = Counter(v["status"].lower() for v in outcomes.values())
        status = "ERROR" if counts["error"] else "FAIL" if counts["fail"] else "PASS"
        return {
            "status": status,
            "metrics": {},
            "backend_outcomes": outcomes,
            "reason": "Expected rejection in both backends"
            if status == "PASS"
            else "Unexpected negative-input behavior",
            "error": None
            if status == "PASS"
            else {
                "category": "RuntimeFailure" if status == "ERROR" else "UnexpectedAcceptance",
                "message": "Negative-input expectation failed",
            },
        }
    try:
        reference = reference_run(model, spec, case.inputs)
        actual = runner.run(case.inputs)
        result = compare(reference, apply_fault(actual, fault), **thresholds)
        result["error"] = (
            {"category": "NumericalMismatch", "message": result["reason"]}
            if result["status"] == "FAIL"
            else None
        )
        return result
    except Exception as exc:
        LOG.exception("Case inference/comparison failed: %s", case.case_id)
        return {
            "status": "ERROR",
            "metrics": {},
            "reason": str(exc),
            "error": exception_record(exc),
        }


def save_case(
    case: TestCase,
    root: Path,
    spec: ModelSpec,
    export: dict | None,
    config: dict,
    result: dict,
    fault: dict | None,
    duration: float,
) -> dict:
    inputs_path = f"inputs/{case.case_id}.npz"
    record = {
        "schema_version": 1,
        "case_id": case.case_id,
        "model_id": case.model_id,
        "model_spec": asdict(spec),
        "seed": case.seed,
        "distribution": case.distribution,
        "input_shapes": {k: list(v.shape) for k, v in case.inputs.items()},
        "input_dtypes": {k: str(v.dtype) for k, v in case.inputs.items()},
        "input_sha256": save_npz(safe_path(root, inputs_path), case.inputs),
        "input_fingerprint": case.fingerprint,
        "onnx_sha256": export["onnx_sha256"] if export else "",
        "inputs_path": inputs_path,
        "case_path": f"cases/{case.case_id}.json",
        "test_kind": case.test_kind,
        "negative_reason": case.negative_reason,
        "expected_exception": {"category": "InvalidInput", "reason": case.negative_reason}
        if case.test_kind == "negative_input"
        else None,
        "tags": ["synthetic", case.test_kind] + (["injected_fault"] if fault else []),
        "synthetic_data": True,
        "intentional_fault_injection": bool(fault),
        "synthetic_fault_injection": bool(fault),
        "fault_replay_requested": bool(fault),
        "fault": fault,
        "compare": config["compare"],
        "config": config,
        "export": export,
        "duration_seconds": duration,
        **result,
    }
    write_json(safe_path(root, record["case_path"]), record)
    return record


def finish_run(
    root: Path,
    config: dict,
    started: str,
    start_time: float,
    records: list,
    exports: dict,
    info: dict,
    inject: bool,
    source: dict | None = None,
) -> dict:
    counts = Counter(r["status"].lower() for r in records)
    summary = {"total": len(records), **{v: counts[v] for v in ("pass", "fail", "error", "skip")}}
    jsonl = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    atomic_text(root / "cases.jsonl", jsonl)
    manifest = {
        "schema_version": 1,
        "run_id": started[:19].replace(":", "") + "-" + uuid.uuid4().hex[:8],
        "started_at_utc": started,
        "finished_at_utc": now(),
        "duration_seconds": time.monotonic() - start_time,
        "synthetic_data": True,
        "synthetic_fault_injection": inject,
        "fault_injection": inject,
        "environment": info,
        "config": config,
        "summary": summary,
        "models": exports,
        "cases_file": "cases.jsonl",
        "cases_file_sha256": sha256_file(root / "cases.jsonl"),
        "case_json_sha256": {
            r["case_id"]: sha256_file(safe_path(root, r["case_path"])) for r in records
        },
        "html_report": "report.html",
        "csv_report": "metrics.csv",
        "source": source,
        "exit_code": exit_code(summary),
        "methodology": "abs(reference-candidate) <= atol + rtol*abs(reference); "
        "synthetic numerical equivalence, no predictive accuracy or board testing",
    }
    write_json(root / "manifest.json", manifest)
    from amlv.reporting import build_report

    build_report(root)
    return manifest


def run_suite(config: dict, root: Path, inject_mismatch: bool = False) -> dict:
    config = validate_config(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    info = environment()
    require_dependencies(info)
    root = prepare_output(root)
    handler = logging.FileHandler(root / "logs/validation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    started, start_time = now(), time.monotonic()
    records, exports = [], {}
    injected = False
    try:
        for model_id in config["models"]:
            spec = ModelSpec(model_id, config["seed"])
            model, export, runner, setup_error = None, None, None, None
            try:
                model = build_model(spec)
                export = export_model(
                    model, spec, root, config["export"], config["cases"]["batch_sizes"][0]
                )
                runner = OnnxRunner(safe_path(root, export["onnx_path"]), spec)
                export["selected_providers"] = runner.providers
                exports[model_id] = {"status": "PASS", **export}
            except Exception as exc:
                LOG.exception("Model setup failed: %s", model_id)
                setup_error = exception_record(exc)
                exports[model_id] = {
                    "status": "ERROR",
                    "error": setup_error,
                    "model_spec": asdict(spec),
                    "export_options": config["export"],
                    "requested_opset": config["export"]["opset_version"],
                }
            for case in generate_suite(spec, config):
                case_start = time.monotonic()
                fault = None
                if inject_mismatch and not injected and case.test_kind == "parity" and runner:
                    fault = {
                        "type": "add",
                        "output": "output",
                        "index": [0, 0, 0] if model_id == "attention" else [0, 0],
                        "amplitude": 1.0,
                        "seed": config["seed"],
                    }
                    injected = True
                result = (
                    {
                        "status": "ERROR",
                        "metrics": {},
                        "error": setup_error,
                        "reason": setup_error["message"],
                    }
                    if setup_error
                    else evaluate_case(case, model, spec, runner, config["compare"], fault)
                )
                record = save_case(
                    case, root, spec, export, config, result, fault, time.monotonic() - case_start
                )
                records.append(record)
                LOG.info(
                    "case=%s model=%s kind=%s status=%s",
                    case.case_id,
                    model_id,
                    case.test_kind,
                    record["status"],
                )
        return finish_run(root, config, started, start_time, records, exports, info, injected)
    finally:
        LOG.removeHandler(handler)
        handler.close()


def reproduce(
    case_path: Path,
    out: Path,
    without_fault: bool = False,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict:
    case_path = case_path.resolve()
    source_root = case_path.parent.parent
    original = read_json(case_path)
    manifest = read_json(source_root / "manifest.json")
    try:
        if original["schema_version"] != 1:
            raise ArtifactError("Unsupported case schema")
        if (
            case_path != safe_path(source_root, original["case_path"])
            or sha256_file(case_path) != manifest["case_json_sha256"][original["case_id"]]
        ):
            raise ArtifactError("Case manifest integrity failure")
        config = validate_config(original["config"])
        thresholds = dict(original["compare"])
        for name, value in (("atol", atol), ("rtol", rtol)):
            if value is not None:
                thresholds[name] = value
        config["compare"] = thresholds
        config = validate_config(config)
        spec = ModelSpec(**original["model_spec"])
        export = original["export"]
        if (
            not export
            or export["model_spec"] != original["model_spec"]
            or export["onnx_sha256"] != original["onnx_sha256"]
        ):
            raise ArtifactError("Case has no consistent, replayable model export")
        arrays = load_npz(
            safe_path(source_root, original["inputs_path"]),
            original["input_sha256"],
            int(config["cases"]["max_input_megabytes"] * 1024**2),
        )
        if array_fingerprint(arrays) != original["input_fingerprint"]:
            raise ArtifactError("Saved inputs differ from content fingerprint")
        onnx_path = safe_path(source_root, export["onnx_path"])
        if sha256_file(onnx_path) != export["onnx_sha256"]:
            raise ArtifactError("ONNX model integrity failure")
        weights = load_npz(
            safe_path(source_root, export["weights_path"]), export["weights_file_sha256"]
        )
        model = build_model(spec)
        expected_weights = model_arrays(model)
        if (
            set(weights) != set(expected_weights)
            or any(
                weights[k].shape != v.shape or weights[k].dtype != v.dtype
                for k, v in expected_weights.items()
            )
            or array_fingerprint(weights) != export["weights_sha256"]
        ):
            raise ArtifactError("Frozen weights differ from model contract/checksum")
        model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
        model.eval()
        fault = (
            original["fault"] if original["fault_replay_requested"] and not without_fault else None
        )
        if bool(original["intentional_fault_injection"]) != bool(original["fault"]):
            raise ArtifactError("Inconsistent fault replay metadata")
        case = TestCase(
            original["case_id"],
            original["model_id"],
            original["seed"],
            original["distribution"],
            arrays,
            original["test_kind"],
            original["negative_reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"Malformed reproduction metadata: {exc}") from exc
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    info = environment()
    require_dependencies(info)
    root = prepare_output(out)
    handler = logging.FileHandler(root / "logs/validation.log", encoding="utf-8")
    LOG.addHandler(handler)
    started, start_time = now(), time.monotonic()
    try:
        for key in ("onnx_path", "weights_path", "diagnostics_path"):
            target = safe_path(root, export[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(source_root, export[key]), target)
        try:
            runner = OnnxRunner(safe_path(root, export["onnx_path"]), spec)
            result = evaluate_case(case, model, spec, runner, thresholds, fault)
        except RuntimeFailure as exc:
            LOG.exception("Replay session setup failed")
            result = {
                "status": "ERROR",
                "metrics": {},
                "error": exception_record(exc),
                "reason": str(exc),
            }
        record = save_case(
            case, root, spec, export, config, result, fault, time.monotonic() - start_time
        )
        record["reproduction"] = {
            "original_run_id": manifest["run_id"],
            "original_status": original["status"],
            "without_fault": without_fault,
            "tolerance_override": atol is not None or rtol is not None,
        }
        write_json(safe_path(root, record["case_path"]), record)
        return finish_run(
            root,
            config,
            started,
            start_time,
            [record],
            {spec.model_id: {"status": "PASS", **export}},
            info,
            bool(fault),
            record["reproduction"],
        )
    finally:
        LOG.removeHandler(handler)
        handler.close()
