"""Read-only dependency checks and an actual writable artifact-root probe."""

import importlib
import platform
import sys
import tempfile
from pathlib import Path

from amlv import __version__
from amlv.errors import RuntimeFailure


def environment() -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "amlv": __version__,
        "executable": sys.executable,
        "providers": [],
        "selected_providers": ["CPUExecutionProvider"],
        "dependency_errors": {},
    }
    for label, module_name in {
        "pytorch": "torch",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
        "numpy": "numpy",
        "onnxscript": "onnxscript",
    }.items():
        try:
            module = importlib.import_module(module_name)
            result[label] = str(module.__version__)
            if module_name == "onnxruntime":
                result["providers"] = module.get_available_providers()
            if module_name == "torch":
                result["torch_num_threads"] = module.get_num_threads()
                result["torch_deterministic_algorithms"] = (
                    module.are_deterministic_algorithms_enabled()
                )
        except Exception as exc:
            result[label] = "unavailable"
            result["dependency_errors"][label] = f"{type(exc).__name__}: {exc}"
    return result


def doctor(root: Path) -> dict:
    result = environment()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root) as stream:
            stream.write(b"amlv writable probe")
            stream.flush()
        result["artifact_root_writable"] = True
    except OSError as exc:
        result["artifact_root_writable"] = False
        result["artifact_root_error"] = str(exc)
    result["artifact_root"] = str(root.resolve())
    result["gpu_note"] = "GPU/board hardware is outside this CPU validation scope; not probed."
    result["healthy"] = (
        not result["dependency_errors"]
        and result["artifact_root_writable"]
        and "CPUExecutionProvider" in result["providers"]
    )
    return result


def require_dependencies(info: dict) -> None:
    if info["dependency_errors"] or "CPUExecutionProvider" not in info["providers"]:
        raise RuntimeFailure(f"Core CPU dependency failure: {info['dependency_errors']}")
