# Product brief — ONNX cross-runtime functional validation

## Problem
AI engineers need to know if model conversion and deployment change model outputs, especially for dynamic input shapes and numerical corner cases. A reproducible validation system should automatically expose such discrepancies before release.

## Audience / job-relevance
Junior AI/ML validation engineers working on PyTorch/ONNX data generation, numerical testing, Linux tooling, feature verification and defect triage. This **does not** perform board-level hardware validation.

## In scope
- Offline small synthetic neural network models and data, no training needed.
- CPU PyTorch vs CPU ONNX Runtime inference parity for valid inputs.
- Legal dynamic batch scenarios; separate invalid-shape, invalid-dtype error behavior.
- Export checks, mismatch inspection, case replay, static report.
- Well-engineered package, GitHub Actions, CI smoke.

## Out of scope
- Real silicon/board test rigs, hardware protocol validation, driver/kernel development.
- Accuracy-on-real-datasets claims, production ML monitoring, neural network training benchmarks.
- Do not depend on paid APIs, network-sourced model weights or long-running workloads.

## Success demonstration
On a clean CPU Linux machine, one demo builds/exports at least 3 model families; executes multiple input categories; reports real parity results; and a purposely corrupted comparison is visibly FAIL with a reproducible case artifact.

## Nonfunctional targets
- Standard demo target ≤2 minutes on typical developer CPU, adjustable for slower environments; correctness above speed.
- Default files only under user-configured `--out` directory.
- All random seeds controlled, all results auditable; no fabricated counts.
