# Implementation evidence

The original specification files are retained unchanged. This maps implemented behavior to
inspectable code and actual checks, without making claims about physical hardware.

| Requirement | Code | Verification |
|---|---|---|
| Python package and CLI | `pyproject.toml`, `src/amlv/cli.py` | Editable install, wheel/sdist build, all command helps |
| Native PyTorch models | `src/amlv/models/` | Frozen eval models, isolated seed and explicit CPU/float32 tests |
| Modern ONNX and dynamic batches | `src/amlv/exporter.py` | Real dynamo exports/checker and ORT batches 1/4 for all three models |
| Version-compatible fallback | `src/amlv/exporter.py` | Removed-argument regression, deliberate fallback, explicit legacy/static tests |
| Deterministic synthetic data | `src/amlv/data_gen.py` | All distributions, seed isolation, bounded arrays, stable selected cases |
| Numerical correctness | `src/amlv/comparator.py` | Boundary/asymmetric tolerance, near-zero, nonfinite, shape/name/dtype, multi-output tests |
| Input-rejection taxonomy | `src/amlv/runners.py`, `src/amlv/pipeline.py` | Actual four rejection types per model, acceptance FAIL/unrelated ERROR tests |
| Durable safe replay | `src/amlv/case_store.py`, `src/amlv/pipeline.py` | Exact fault/negative replay, frozen weights, corruption and path/dimension guards |
| Local engineering reports | `src/amlv/reporting.py` | Real HTML/CSV/JSON, both schemas, browser search/sort/mobile review |
| Failure diagnostics | `src/amlv/errors.py`, `src/amlv/pipeline.py` | Tracebacks and ERROR reports, one actual injected/replayed numerical FAIL |
| CPU operation | `src/amlv/doctor.py`, `src/amlv/runners.py` | Doctor healthy, CPUExecutionProvider explicitly selected, no GPU/keys |
| Linux tooling | `.github/workflows/ci.yml`, README | Ubuntu CPU workflow configured; local Linux execution unavailable |
| Board/CUDA/HIP/ROCm testing | Outside implemented scope | **Not performed; no claims made** |

The actual session results and environment are in `VERIFICATION.md`. For a portfolio demo,
show the real generated passing report, the explicitly labeled synthetic fault, and its replay.
The implementation supports discussing export fidelity and debugging; it does not establish
production accuracy or hardware-validation experience.
