# Local verification — 13 October 2026

These are actual local results from this implementation session. All models, inputs, and
faults are synthetic. No production accuracy, accelerator performance, or physical board
validation was performed.

## Environment

| Component | Observed version |
|---|---|
| OS / architecture | Windows 11, build 26300 / AMD64 |
| Python | 3.12.4 |
| PyTorch | 2.14.1+cpu |
| ONNX | 1.23.2 |
| ONNX Runtime | 1.31.0 |
| ONNX Script | 0.7.2 |
| NumPy | 2.5.4 |
| pytest | 9.1.1 |
| Ruff | 0.17.0 |
| Selected inference provider | CPUExecutionProvider |

`doctor` observed AzureExecutionProvider and CPUExecutionProvider as installed providers.
Only CPUExecutionProvider was selected and used for validation. No GPU availability is inferred.
The isolated `.venv` has system-site-packages disabled; `pip check` found no broken requirements.
The environment-specific dependency snapshot is `config/constraints.windows-py312.txt`.

## Executed gates

All commands below used `.venv/Scripts/python` or `.venv/Scripts/amlv` on PowerShell;
commands are abbreviated after the executable prefix for readability.

| Command | Observed result |
|---|---|
| `python -m pip install -e '.[dev]'` | Exit 0; editable project and dev dependencies installed |
| `python -m pip check` | Exit 0; no broken requirements |
| `python -m ruff check .` | Exit 0; all checks passed |
| `python -m ruff format --check .` | Exit 0 |
| `python -m pytest -q` | **89 passed**, no skips/failures; 61.64 seconds in the final full suite |
| `amlv --help` and all five subcommand helps | Exit 0 (also exercised by CLI tests) |
| `amlv doctor` | Exit 0; core dependencies and writable artifact probe healthy |
| `amlv demo --out artifacts/demo` | Exit 0; 30 PASS, 0 FAIL/ERROR/SKIP |
| `amlv validate --config config/validation.example.yaml --out artifacts/example` | Exit 0; 66 PASS, 0 FAIL/ERROR/SKIP |
| `amlv demo --out artifacts/intentional-failure --inject-mismatch` | **Expected exit 1**; 29 PASS, 1 FAIL, 0 ERROR/SKIP |
| `amlv report --run artifacts/example` | Exit 0; regenerated from saved JSON |
| `amlv reproduce --case artifacts/intentional-failure/cases/aa84c46ca752b9b2ad77e307.json --out artifacts/replay` | **Expected exit 1**; one faithfully reproduced FAIL |
| Same reproduction with `--out artifacts/replay-clean --without-fault` | Exit 0; one PASS |
| `python scripts/check_artifacts.py artifacts/demo` | Verified schemas, arrays, hashes, frozen weights, and ONNX graphs |
| Same checker for `artifacts/example`, `artifacts/intentional-failure`, and replay | Verified |
| `python -m build` | Exit 0; sdist and universal Python wheel built |

The demo contains 18 numerical parity cases (3 models × 3 distributions × 2 batches)
and 12 expected input rejections (4 per model). The full example contains 54 parity cases
and 12 rejections. CSV review found 66 rows in the example and empty numerical fields for
negative-input cases. JSON schemas were checked using jsonschema with date-time validation.

All three final exports used **dynamo**, actual opset 18, with graph checks PASS, no fallback,
and no captured warnings. The actual operator inventories contain 3 nodes for MLP,
4 for CNN, and 13 for attention. Real ORT inference tested batches 1 and 4 with one export
per model; the example also tested batch 2.

The installed PyTorch API had removed `fallback`. This was found during artifact review and
fixed by signature detection; a regression test checks that APIs without the argument receive
a genuine dynamo attempt. Earlier legacy-path debugging artifacts are preserved under
`artifacts/legacy-debug/`; they are not the final demo/example reports.

## Actual replayed fault

Run `2026-10-13T060347-35b31d22`, MLP case `aa84c46ca752b9b2ad77e307`:

| Field | Observed value |
|---|---|
| Failing coordinate | output `[0,0]` |
| Reference | -0.4839649200439453 |
| Perturbed candidate | 0.5160350203514099 |
| Absolute error | 0.9999999403953552 |
| Allowed threshold | 0.00005839649200439453 |
| Mismatched elements | 1 of 8 |
| RMSE | 0.3535533695198529 |

The replay retained identical input and ONNX hashes and exactly equal recorded numerical
metrics in this environment. `--without-fault` verified the underlying parity PASS.

## Determinism and report review

Two real runs, `artifacts/demo-modern` and `artifacts/demo`, had identical case IDs,
array fingerprints, and compressed input SHA256 values for all 30 cases. Their run IDs and
UTC timestamps differ as intended. The test suite independently checks two fresh runs.

The actual generated HTML was inspected in the local in-app browser. Searching `attention`
showed 10 cases; searching `FAIL` in the fault report showed exactly one failure. Numeric
sorting set the table's ascending state. A 390-pixel viewport had no page-wide overflow;
wide tables scroll within their containers. The report shows the actual fault coordinate,
values, threshold, simulation label, replay command, and dependency versions. These UI checks
were local and did not publish data. PNG evidence is retained under `artifacts/`.

## Platform and scope limits

Linux execution was **not available**: `wsl --list --quiet` reported that Windows Subsystem
for Linux is not installed. Ubuntu/Python 3.11 CPU installation and checks are configured
in `.github/workflows/ci.yml`, but no remote GitHub Actions execution is claimed.
No Docker, API key, GPU, or downloaded dataset was used. Only this installed dependency
combination was exercised locally.

Optional P1 work not implemented: Hypothesis fuzzing, external ONNX model CLI input, case
filtering, and baseline diffs. Multi-output comparison and operator inventory are implemented;
the built-in reference models expose one output each. Core P0 functionality was exercised
locally, with Linux verification explicitly pending the CI runner.
