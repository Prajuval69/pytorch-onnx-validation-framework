"""Validate saved schemas, input integrity, and actual exported ONNX graphs offline."""

import argparse
import json
from pathlib import Path

import jsonschema
import onnx

from amlv.case_store import array_fingerprint, load_npz, safe_path, sha256_file
from amlv.reporting import read_run


def check(root: Path) -> dict:
    manifest, cases = read_run(root)
    schemas = Path(__file__).resolve().parents[1] / "spec/schemas"
    for name, records in (("manifest", [manifest]), ("case", cases)):
        schema = json.loads((schemas / f"{name}.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        for record in records:
            validator.validate(record)
    for case in cases:
        arrays = load_npz(safe_path(root, case["inputs_path"]), case["input_sha256"])
        assert array_fingerprint(arrays) == case["input_fingerprint"]
        assert {k: list(v.shape) for k, v in arrays.items()} == case["input_shapes"]
        assert {k: str(v.dtype) for k, v in arrays.items()} == case["input_dtypes"]
    for export in manifest["models"].values():
        if export["status"] != "PASS":
            continue
        path = safe_path(root, export["onnx_path"])
        assert sha256_file(path) == export["onnx_sha256"]
        onnx.checker.check_model(str(path))
        weights = load_npz(safe_path(root, export["weights_path"]), export["weights_file_sha256"])
        assert array_fingerprint(weights) == export["weights_sha256"]
    print(
        json.dumps(
            {"verified": True, "run_id": manifest["run_id"], "summary": manifest["summary"]},
            indent=2,
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    check(parser.parse_args().run)
