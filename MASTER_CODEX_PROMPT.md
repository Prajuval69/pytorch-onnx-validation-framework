# Master Codex prompt — PyTorch–ONNX Neural Network Validation Framework

Act as the lead AI/ML validation engineer and implement a polished open-source Python repository named `pytorch-onnx-validation-framework`. You are authorized to autonomously design code, create files, install compatible public dependencies, run tests, diagnose defects and revise the implementation. This is an implementation task, not a request for a plan.

## Mission
Build an **offline-first, CPU-first, deterministic model-conformance system**. Create small PyTorch reference models, export them to ONNX, generate validation input datasets, execute each model with PyTorch and ONNX Runtime, compare outputs with correct numerical semantics, detect problems, and deliver reproducible failure artifacts and an attractive static HTML report.

## Mandatory capabilities
1. MLP, CNN and small attention model built with native PyTorch operations; no external dataset or pretrained weight download.
2. ONNX export via installed modern `torch.onnx` API, prioritizing `dynamo=True`, verified graph checks and dynamic batch support for at least MLP/CNN; version-compatible fallbacks documented explicitly.
3. Deterministic test data from independent seeded NumPy generators: random normal, uniform, zeros, ones, low-magnitude and large-magnitude (bounded) cases; varied legal batch sizes. Clear separate invalid input tests.
4. True cross-runtime comparison using absolute + relative tolerances, with max absolute, max robust relative, RMSE, cosine similarity and mismatch counts. NaN/Inf explicitly handled; cosine never solely determines pass/fail. Invalid-input negative tests PASS only if the expected rejection occurs; surprising acceptance FAILS; unexpected errors are ERROR.
5. Clear test outcome taxonomy: PASS/FAIL/ERROR/SKIP; meaningful error category, metadata and return codes.
6. Machine-readable `manifest.json`, `cases.jsonl`, serialized case inputs in compressed NPZ with `allow_pickle=False`, plus HTML summary, CSV table and reproducibility metadata.
7. `amlv doctor`, `amlv demo`, `amlv validate`, `amlv reproduce`, `amlv report` CLI commands as specified.
8. Unit, integration, negative-path, regression and CLI tests; Linux GitHub Actions CI.
9. A one-command end-to-end CPU demo and an explicit fault-injection demo that proves detection and returns a nonzero exit code.
10. Clear README with architecture, limitations, screenshots instructions, quickstart and commands that actually work.

## Required scope and order
Read all `spec/` docs and examples. Implement M0 scaffold; M1 models + export; M2 deterministic generator; M3 comparison engine; M4 reporting + replay; M5 tests/CI/docs. Run gates for each milestone. Do not postpone the CLI or reports until the end. Adapt APIs to installed versions rather than fabricating compatibility. If no network or dependencies, complete as much as the environment permits and honestly document verification blockers.

## Critical correctness rules
- Reuse identical model weights and exact input tensors for PyTorch and ONNX Runtime. Always switch model to `eval()` and use no-grad/inference-mode.
- Export **once per model configuration**, cache with content hashes, validate ONNX graph before inferencing. Never compare different models by accident.
- Do not confuse an invalid-input error test with a numerical mismatch.
- Dynamic dimensions: test legal batch sizes only; treat unsupported exporter features as explicit errors or optional-skips, never silently claim success.
- Use `numpy.isclose` semantics (`abs(a-b) <= atol + rtol*abs(reference)`) for the verdict; robust percentage errors need a denominator floor to avoid division by zero.
- Keep output tensors bounded and defensively check output names/count/shapes/dtypes.
- No GPU required and no external API keys, Docker or board hardware.

## Exit requirements
Attempt `python -m pip install -e '.[dev]'`, `ruff check .`, `pytest -q`, `amlv doctor`, `amlv demo --out artifacts/demo`, `amlv validate --config config/validation.example.yaml --out artifacts/example`, and one explicit mismatch injection test. Inspect report, JSON and CSV for actual values. If feasible run a second demo and demonstrate deterministic case fingerprints. Finish with a report of actual commands/results and unimplemented optional items.

Do not ask for permissions or design preferences for routine choices. Build the working repo.
