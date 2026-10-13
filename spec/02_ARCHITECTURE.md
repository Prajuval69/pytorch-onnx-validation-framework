# Architecture and implementation contracts

## Component flow

`ModelRegistry -> ExportManager -> CaseGenerator -> ReferenceRunner/OnnxRunner -> Comparator -> RunRecorder -> ReportBuilder`

## Suggested package tree (Codex may refine)
```
src/amlv/__init__.py
src/amlv/cli.py
src/amlv/config.py
src/amlv/models/{mlp,cnn,attention,registry}.py
src/amlv/exporter.py
src/amlv/data_gen.py
src/amlv/runners.py
src/amlv/comparator.py
src/amlv/case_store.py
src/amlv/pipeline.py
src/amlv/reporting.py
src/amlv/doctor.py
src/amlv/errors.py
tests/{unit,integration,cli}/...
.github/workflows/ci.yml
```

## Contracts
- `ModelSpec`: id, deterministic initialization seed, input and output names, legal input shape family, supported dtypes, exporter options.
- `TestCase`: case ID (stable hash), model ID, seed, distribution, input shapes and dtypes, label/tags, `np.ndarray` inputs.
- `ExportRecord`: model config/weights checksum, ONNX file hash, opset version, exporter flags, graph-check outcome.
- `ComparisonRecord`: verdict, numeric metrics per output, mismatched element count, worst-case index/value, exception info.
- `RunManifest`: run ID, start/end UTC, versions, architecture metadata, counts and output-relative paths.

## Security and reproducibility
- Generate predictable model weights via isolated seeded PyTorch RNG; no internet or arbitrary code loading.
- Store only `.npz` arrays and text metadata; never unpickle untrusted data. Validate hashes for replay.
- Cache exports under controlled artifact dir with SHA256 hashes of model/config/export settings.
- Guard against unexpectedly giant tensors (e.g. 128 MB default input bound, configurable).
- Report output paths relative to artifact root. Write results atomically when practical.
- Use structured exceptions: `ExportFailure`, `InvalidInput`, `RuntimeFailure`, `NumericalMismatch`, `ArtifactError`.
- Exceptions thrown by ORT and PyTorch should be converted into explicit case outcome messages, not lost.

## Numerical alignment
- Build a single model per model spec, freeze it in eval mode, save input and weights hashes.
- Compare CPU outputs after converting PyTorch tensors to NumPy; ONNX Runtime uses same NumPy input arrays.
- Record selected ONNX provider `CPUExecutionProvider` and versions.
- Guard multi-output model ordering by comparing declared names and mapping outputs to names.
