"""Atomic text and safe NPZ persistence, path containment, SHA256 integrity."""

import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from amlv.errors import ArtifactError

MAX_ARTIFACT_BYTES = 128 * 1024**2


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"Cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def array_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        if array.dtype.hasobject:
            raise ArtifactError("Object arrays are prohibited")
        digest.update(canonical([name, list(array.shape), array.dtype.str]))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
    ):
        raise ArtifactError("Artifact paths must be portable relative paths with forward slashes")
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ArtifactError(f"Artifact path escapes root: {relative}")
    return path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ArtifactError("JSON exceeds artifact size bound")
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
        if not isinstance(value, dict):
            raise ArtifactError("Expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read {path.name}: {exc}") from exc


def _invalid_constant(value: str) -> None:
    raise ArtifactError(f"Nonfinite JSON constant: {value}")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    array_fingerprint(arrays)  # enforce no objects before serialization
    if sum(a.nbytes for a in arrays.values()) > MAX_ARTIFACT_BYTES:
        raise ArtifactError("Arrays exceed artifact size bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(path)


def load_npz(path: Path, expected_sha256: str, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ArtifactError(f"SHA256 integrity failure: {path.name}")
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_bytes + 65536:
                raise ArtifactError("NPZ decompressed size exceeds bound")
            # Bound declared dimensions before NumPy attempts any large allocation.
            for item in archive.infolist():
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ArtifactError("Unsupported NPY version")
                    if dtype.hasobject or math.prod(shape) * dtype.itemsize > max_bytes:
                        raise ArtifactError("Unsafe NPZ dtype or dimensions")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        if sum(a.nbytes for a in arrays.values()) > max_bytes:
            raise ArtifactError("NPZ arrays exceed size bound")
        return arrays
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise ArtifactError(f"Cannot load NPZ {path.name}: {exc}") from exc
