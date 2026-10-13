"""CPU inference adapters, explicit contract rejection and narrow ORT error mapping."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from amlv.errors import InvalidInput, RuntimeFailure
from amlv.models import ModelSpec


def reference_run(model: nn.Module, spec: ModelSpec, inputs: dict[str, np.ndarray]) -> dict:
    if set(inputs) != set(spec.input_names):
        raise InvalidInput("Missing or unexpected named input", "missing")
    array = inputs["input"]
    expected = spec.shape(1)
    if array.ndim != len(expected):
        raise InvalidInput(f"Expected rank {len(expected)}, got {array.ndim}", "rank")
    if array.shape[1:] != expected[1:] or not 1 <= array.shape[0] <= 8:
        raise InvalidInput(f"Incompatible shape: {array.shape}", "shape")
    if array.dtype != np.float32:
        raise InvalidInput(f"Expected float32, got {array.dtype}", "dtype")
    try:
        with torch.inference_mode():
            result = model.eval()(torch.from_numpy(array.copy()))
        tensors = result if isinstance(result, tuple) else (result,)
        if len(tensors) != len(spec.output_names):
            raise RuntimeFailure("Reference output count differs from contract")
        outputs = {
            name: value.detach().cpu().numpy().copy()
            for name, value in zip(spec.output_names, tensors, strict=True)
        }
        if sum(a.nbytes for a in outputs.values()) > 128 * 1024**2:
            raise RuntimeFailure("Reference output exceeds memory bound")
        return outputs
    except RuntimeFailure:
        raise
    except Exception as exc:
        raise RuntimeFailure(f"PyTorch inference: {type(exc).__name__}: {exc}") from exc


class OnnxRunner:
    def __init__(self, path: Path, spec: ModelSpec) -> None:
        self.spec = spec
        try:
            onnx.checker.check_model(str(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.session.disable_fallback()
            self.providers = self.session.get_providers()
            if self.providers != ["CPUExecutionProvider"]:
                raise RuntimeFailure(f"Unexpected selected providers: {self.providers}")
            if [v.name for v in self.session.get_inputs()] != list(spec.input_names) or [
                v.name for v in self.session.get_outputs()
            ] != list(spec.output_names):
                raise RuntimeFailure("ONNX input/output contract differs from reference")
        except Exception as exc:
            raise RuntimeFailure(f"ONNX session setup: {type(exc).__name__}: {exc}") from exc

    def run(self, inputs: dict[str, np.ndarray]) -> dict:
        try:
            results = self.session.run(list(self.spec.output_names), inputs)
            if len(results) != len(self.spec.output_names):
                raise RuntimeFailure("ONNX output count differs from contract")
            if sum(a.nbytes for a in results) > 128 * 1024**2:
                raise RuntimeFailure("ONNX output exceeds memory bound")
            return dict(zip(self.spec.output_names, results, strict=True))
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument as exc:
            message = str(exc)
            lowered = message.lower()
            if "data type" in lowered:
                reason = "dtype"
            elif "invalid rank" in lowered:
                reason = "rank"
            elif "dimensions" in lowered or "dimension" in lowered:
                reason = "shape"
            elif "input name" in lowered:
                reason = "missing"
            else:
                raise RuntimeFailure(f"Unexpected ORT invalid argument: {message}") from exc
            raise InvalidInput(message, reason) from exc
        except ValueError as exc:
            if "required inputs" in str(exc).lower() and "missing" in str(exc).lower():
                raise InvalidInput(str(exc), "missing") from exc
            raise RuntimeFailure(f"Unexpected ORT ValueError: {exc}") from exc
        except (InvalidInput, RuntimeFailure):
            raise
        except Exception as exc:
            raise RuntimeFailure(f"ONNX inference: {type(exc).__name__}: {exc}") from exc
