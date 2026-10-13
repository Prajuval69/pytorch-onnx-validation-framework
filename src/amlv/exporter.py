"""Modern dynamo export first; explicit, recorded compatibility fallback to legacy."""

import contextlib
import hashlib
import inspect
import io
import logging
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from amlv.case_store import (
    array_fingerprint,
    canonical,
    read_json,
    safe_path,
    save_npz,
    sha256_file,
    write_json,
)
from amlv.errors import ArtifactError, ExportFailure
from amlv.models import ModelSpec

LOG = logging.getLogger("amlv")


def model_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()
    }


def export_model(
    model: nn.Module, spec: ModelSpec, root: Path, options: dict, static_batch: int = 2
) -> dict:
    arrays = model_arrays(model)
    weights_hash = array_fingerprint(arrays)
    identity = {
        "export_contract_version": 2,
        "model_spec": asdict(spec),
        "weights_sha256": weights_hash,
        "export_options": options,
        "static_batch": static_batch,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
    }
    cache_key = hashlib.sha256(canonical(identity)).hexdigest()
    stem = f"models/{spec.model_id}-{cache_key[:16]}"
    path = safe_path(root, stem + ".onnx")
    metadata_path = safe_path(root, stem + ".json")
    if metadata_path.exists():
        record = read_json(metadata_path)
        if record["cache_key"] != cache_key or sha256_file(path) != record["onnx_sha256"]:
            raise ArtifactError("Export cache integrity failure")
        if sha256_file(safe_path(root, record["weights_path"])) != record["weights_file_sha256"]:
            raise ArtifactError("Cached weights integrity failure")
        onnx.checker.check_model(str(path))
        return {**record, "cache_hit": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    example_batch = 2 if options["dynamic_batch"] else static_batch
    example = torch.zeros(spec.shape(example_batch), dtype=torch.float32, device="cpu")
    # Opset 18 is supported by the selected minimum dependency versions and native ops.
    opset = options["opset_version"] or 18
    diagnostics = io.StringIO()
    fallback_reason = None
    mode = "dynamo" if options["dynamo"] else "legacy"
    common = {
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "opset_version": opset,
        "external_data": False,
    }
    caught_messages = []
    # PyTorch 2.9+ removed `fallback`; older modern APIs require explicitly disabling it
    # so that a library-managed fallback cannot silently masquerade as dynamo success.
    modern_extra = (
        {"fallback": False} if "fallback" in inspect.signature(torch.onnx.export).parameters else {}
    )
    try:
        with (
            warnings.catch_warnings(record=True) as caught,
            contextlib.redirect_stdout(diagnostics),
            contextlib.redirect_stderr(diagnostics),
        ):
            warnings.simplefilter("always")
            if mode == "dynamo":
                try:
                    dynamic = (
                        ({0: torch.export.Dim("batch", min=1, max=8)},)
                        if options["dynamic_batch"]
                        else None
                    )
                    program = torch.onnx.export(
                        model,
                        (example,),
                        dynamo=True,
                        dynamic_shapes=dynamic,
                        **modern_extra,
                        **common,
                    )
                    if program is None:
                        raise ExportFailure("Modern exporter did not return ONNXProgram")
                    program.save(str(path), external_data=False)
                    exported = onnx.load(str(path), load_external_data=False)
                    if (
                        options["dynamic_batch"]
                        and not exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param
                    ):
                        raise ExportFailure("Dynamo export specialized the required dynamic batch")
                except Exception as exc:
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                    LOG.warning("%s dynamo fallback: %s", spec.model_id, fallback_reason)
                    mode = "legacy"
            if mode == "legacy":
                axes = (
                    {"input": {0: "batch"}, "output": {0: "batch"}}
                    if options["dynamic_batch"]
                    else None
                )
                torch.onnx.export(
                    model, (example,), str(path), dynamo=False, dynamic_axes=axes, **common
                )
            caught_messages = [str(item.message) for item in caught]
        onnx.checker.check_model(str(path))
        graph = onnx.load(str(path), load_external_data=False)
        if [v.name for v in graph.graph.input] != list(spec.input_names) or [
            v.name for v in graph.graph.output
        ] != list(spec.output_names):
            raise ExportFailure("Exported input/output names do not match model contract")
        if (
            options["dynamic_batch"]
            and not graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        ):
            raise ExportFailure("Export has no dynamic batch dimension")
    except Exception as exc:
        raise ExportFailure(f"{spec.model_id} {mode} opset={opset}: {exc}") from exc
    finally:
        diagnostic_path = safe_path(root, stem + "-export.log")
        diagnostic_path.write_text(
            diagnostics.getvalue() + "\n" + "\n".join(caught_messages), encoding="utf-8"
        )
    weights_path = stem + "-weights.npz"
    weights_file_hash = save_npz(safe_path(root, weights_path), arrays)
    record = {
        **identity,
        "cache_key": cache_key,
        "cache_hit": False,
        "onnx_path": stem + ".onnx",
        "onnx_sha256": sha256_file(path),
        "weights_path": weights_path,
        "weights_file_sha256": weights_file_hash,
        "exporter_mode": mode,
        "exporter_flags": {
            "dynamo": mode == "dynamo",
            "external_data": False,
            "dynamic_batch": options["dynamic_batch"],
            "fallback_argument_supported": bool(modern_extra),
        },
        "fallback_reason": fallback_reason,
        "opset_version": next(v.version for v in graph.opset_import if v.domain == ""),
        "graph_check": "PASS",
        "dynamic_batch": options["dynamic_batch"],
        "node_count": len(graph.graph.node),
        "operators": dict(Counter(n.op_type for n in graph.graph.node)),
        "warnings": caught_messages,
        "diagnostics_path": stem + "-export.log",
    }
    write_json(metadata_path, record)
    return record
