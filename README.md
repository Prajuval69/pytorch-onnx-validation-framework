# PyTorch–ONNX Validation Framework

[Live demo and reports](https://prajuval69.github.io/pytorch-onnx-validation-framework/) · [Source repository](https://github.com/Prajuval69/pytorch-onnx-validation-framework)

**Offline, CPU-first numerical conformance for synthetic models.** Export small frozen
PyTorch networks, execute identical inputs with PyTorch and ONNX Runtime, detect numerical
or input-contract failures, and retain local evidence you can replay.

This measures numerical equivalence, not predictive accuracy. No training, pretrained weights,
external datasets, GPU, board hardware, API keys, server, or Docker is required. Dependency
installation needs a package index or a prepared offline wheelhouse; execution is offline.

## Quickstart

Python 3.11+; Linux is the CI baseline. Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
# Select a CPU wheel first on Linux to avoid installing accelerator dependencies.
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
amlv doctor
amlv demo --out artifacts/demo
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e '.[dev]'
.venv/Scripts/amlv doctor
.venv/Scripts/amlv demo --out artifacts/demo
```

Open `artifacts/demo/report.html` directly in a browser. The report embeds its CSS and JavaScript
and has no remote dependencies. Run directories must be new or empty; an occupied directory
is rejected to preserve evidence. Use another `--out` for a second run.

## Commands

```bash
amlv --help
amlv doctor --out artifacts
amlv demo --out artifacts/demo
amlv validate --config config/validation.example.yaml --out artifacts/example
amlv report --run artifacts/example
amlv demo --out artifacts/intentional-failure --inject-mismatch
```

Every command supports `--help`; `python -m amlv` is equivalent to `amlv`.
The example config exercises all six distributions at batches 1, 2, and 4. The bounded demo
uses three distributions at batches 1 and 4. Both include four separate input-rejection tests
per model. Config rejects unknown keys, unsupported models/providers, negative or nonfinite
tolerances, invalid distributions, illegal batch sizes, and oversized inputs. This release
supports float32 parity on `CPUExecutionProvider` only.

### Fault detection and exact replay

`--inject-mismatch` adds 1.0 to the first element of one ONNX **output copy**, after inference.
Original weights, the ONNX model, and saved inputs remain intact. Reports and case metadata
label the simulation `synthetic_fault_injection=true`. A numerical FAIL returns exit code 1.
Find that case's `case_path` in `cases.jsonl` or follow its report link:

```bash
amlv reproduce --case artifacts/intentional-failure/cases/<case-id>.json --out artifacts/replay
# Replay underlying parity, explicitly removing the stored synthetic fault:
amlv reproduce --case artifacts/intentional-failure/cases/<case-id>.json \
  --out artifacts/replay-clean --without-fault
# An explicit tolerance override is recorded in the reproduction metadata:
amlv reproduce --case artifacts/demo/cases/<case-id>.json --out artifacts/replay-override \
  --atol 0.00002 --rtol 0.0002
```

Replay verifies the individual case JSON against its manifest, input-file SHA256 and array
fingerprint, ONNX SHA256, and frozen-weight file and content checksums. It uses the saved ONNX
graph and exact saved weight arrays, with strict architecture/name/shape/dtype checks; it does
not re-export. Keep the entire source run together. Stored perturbations replay only when case
metadata explicitly requests it. Environment changes may affect last-bit arithmetic, so each
replay has its own environment metadata. SHA256 detects corruption, not authenticity.

`report` rebuilds HTML/CSV from immutable JSON without importing the inference backends.
Its exit code indicates successful report generation, even if the saved run contains FAILs.

| Exit | Meaning |
|---|---|
| 0 | All required cases PASS (optional SKIP is allowed); successful doctor/report |
| 1 | At least one FAIL, including unexpected negative-test acceptance |
| 2 | Invalid CLI arguments or config |
| 3 | Export, runtime, dependency, integrity, or filesystem ERROR |

ERROR takes precedence over FAIL in mixed runs. Core CPU cases are never skipped. SKIP is a
reserved outcome for future optional capabilities, and is not used to hide an export failure.

## Models and inputs

| Model | Architecture | Float32 input | Output |
|---|---|---|---|
| `mlp` | Linear(32,64), ReLU, Linear(64,8) | `[B,32]` | `[B,8]` |
| `cnn` | Conv2d(1,4,3,pad=1), ReLU, spatial mean, Linear(4,6) | `[B,1,16,16]` | `[B,6]` |
| `attention` | Q/K/V projections, scaled matmul, softmax, output projection | `[B,4,16]` | `[B,4,16]` |

Default model seed is 2026. Initialization preserves caller RNG state. Models run on CPU in
eval/inference mode with frozen parameters. Suite execution uses one PyTorch and one ORT
inference thread, deterministic PyTorch algorithms, and isolated NumPy RNGs for each case.
Inputs are normal N(0,1), uniform [-1,1], zeros, ones, near-zero [-1e-6,1e-6], and bounded large
[-100,100]. Stable case IDs include the model, batch, distribution, seed, negative kind, and
array fingerprint; selecting fewer cases does not change the retained cases.

Legal batches are integers 1–8. Dynamic export uses example batch 2 and
`torch.export.Dim('batch', min=1, max=8)`. One export is reused across all batches for each model.
Static export requires exactly one configured batch size and uses that size as its example.
Negative tests deliberately introduce wrong rank, incompatible features/channels, float64,
or missing named input. A negative PASS requires expected rejection by **both** backends;
surprising acceptance is FAIL, unrelated exceptions or rejection categories are ERROR.
The PyTorch adapter enforces the declared reference input contract before inference; the ORT
adapter executes actual runtime validation and narrowly maps recognizable input errors.

### Export compatibility

The default exporter prioritizes `torch.onnx.export(..., dynamo=True)` (with `fallback=False`
only when the installed API accepts it) and
serializes the returned ONNXProgram without external weight files. The framework verifies
explicit input/output names, ONNX graph correctness, and a symbolic batch dimension. It chooses
opset 18 for this native-op baseline when config specifies null; the **actual** saved opset,
installed versions, warnings, node/operator inventory, and export diagnostics are recorded.
An explicitly configured opset must be 18–23 and must actually pass the installed exporter.

If dynamo fails or specializes the required batch dimension, an explicit legacy
`dynamo=False` / `dynamic_axes` fallback is attempted, with the original exception recorded
under `fallback_reason` and in `logs/validation.log`. `export.dynamo: false` selects legacy
directly. Checker/session failures become ERROR, not success or optional skip. Cache identity
includes model config, actual weight content, export options, versions, and example batch;
cache hits recheck file integrity and graph correctness. Cache reuse is available through
`export_model`; CLI runs intentionally require empty output directories.

## Numerical methodology

For each declared output, every element must satisfy:

```text
abs(reference - candidate) <= atol + rtol * abs(reference)
```

Defaults: `atol=1e-5`, `rtol=1e-4`. Output names, counts, shapes, and dtypes must agree.
Diagnostics include maximum absolute error, robust relative error, mean absolute error,
RMSE, cosine, mismatch count, and the worst failing coordinate with values and threshold.
The relative denominator is `max(abs(reference), atol, float64 epsilon)`; cosine never controls
the verdict, and zero norms yield null. Metrics are calculated in float64.

Default `nan_policy: fail` rejects all nonfinite output elements. `equal` permits matching NaNs
and matching same-sign infinities, while rejecting unequal nonfinite values. Outputs containing
nonfinite elements have nullable numerical summary metrics. JSON never contains NaN/Infinity
constants; nonfinite worst sample values are labeled strings. Ordinary generated parity cases
contain finite values. Explicit nonfinite diagnostic cases are covered by comparator tests.
Negative-input metrics are empty and displayed as undefined, rather than misleading zeros.

## Architecture and evidence

```text
ModelRegistry → ExportManager → CaseGenerator → ReferenceRunner + OnnxRunner
                                            → Comparator → RunRecorder → ReportBuilder
```

Modules live in `src/amlv/`. Structured exceptions distinguish config, export, invalid input,
runtime, numerical mismatch, and artifact failures. Unexpected failures retain stack traces
in the run log and concise messages in reports. UTC timestamps and monotonic durations describe
the observed run; timing is not a benchmark claim.

```text
artifacts/run/
  manifest.json                 # schema version, environment, config, counts, export metadata
  cases.jsonl                   # complete immutable case records
  cases/<case-id>.json           # replay entry points
  inputs/<case-id>.npz           # exact arrays, loaded with allow_pickle=False
  models/<model>-<hash>.onnx     # graph used by ORT
  models/*-weights.npz           # exact frozen reference state, no pickle
  models/*.json                  # export cache metadata
  models/*-export.log            # exporter output and warnings
  metrics.csv
  report.html
  logs/validation.log
```

Paths in manifests are relative to the artifact root. Path containment, object-dtype rejection,
compressed-file checksums, and decompressed/declared dimension bounds protect artifact loading.
Ordinary cases are bounded by configured `max_input_megabytes` (128 MiB default) and legal batches;
NPZ persistence also has a 128 MiB safety ceiling. Text and NPZ writes use atomic replacement.
Reports include environment, outcome KPIs, export matrix, searchable/sortable cases, worst
failure, replay command, methodology, and evidence links. HTML/CSV can be disabled in config;
immutable JSON is required.

## Tests and operations

```bash
python -m pip check
ruff check .
ruff format --check .
pytest -q
amlv demo --out artifacts/fresh-smoke
python scripts/check_artifacts.py artifacts/fresh-smoke
python -m build
```

Tests cover tolerance boundaries, nonfinite and near-zero semantics, multi-output comparison,
isolated seeds, stable cases, safe NPZ/JSON, corrupt inputs/models/weights, path traversal,
actual exports and ORT batches, negative taxonomy, fallback/static export, setup ERROR reporting,
all CLI commands, schema validation, deterministic second runs, and faithful fault replay.
The artifact checker independently validates both supplied JSON schemas, array metadata,
checksums, frozen weights, and ONNX graphs. Ubuntu CI installs CPU PyTorch, lints, tests, runs
the demo/full example/fault gates, checks artifacts, and builds distributions. It does not
publish artifacts or contact model services.

See [local verification notes](docs/VERIFICATION.md) for actual commands, versions, results,
and platform limits from the implementation session. Do not confuse configured CI with a
completed remote CI run.

To capture a portfolio screenshot, open an **actual generated** `report.html`, capture the
hero/export matrix/case table in your browser, and caption it “Synthetic CPU numerical equivalence.”
Also show the labeled fault report and its replay. Generated local artifacts are gitignored.

## Limitations and extension points

This release does not validate real boards, GPU node placement, production accuracy, training,
external user-supplied models, quantization, or performance speedups. Optional P1 items not
implemented: Hypothesis shape fuzzing, external ONNX CLI input, case filtering, and baseline
diffs. The comparator already supports multiple named outputs; the built-in models each
declare one output. Operator inventory is implemented. Extend the registry and runners for
new model families or hardware when actual access and an appropriate test contract exist.
Only locally exercised dependency combinations are verified; exporter diagnostics expose
compatibility fallback instead of claiming universal version support.

## References and license

Implementation follows the official [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html),
[ONNX checker](https://onnx.ai/onnx/api/checker.html), and
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
The original work order remains in `spec/`. Source is MIT licensed; upstream libraries retain
their own licenses. Dependencies are bounded in `pyproject.toml`; to reproduce an environment,
use a fresh virtualenv and save `python -m pip freeze` to an environment-specific constraints
file. A verified local constraints snapshot is retained in `config/constraints.windows-py312.txt`.
