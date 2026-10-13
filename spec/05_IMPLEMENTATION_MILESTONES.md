# Autonomous build milestones (execute in order)

**M0: project & environment**: `pyproject.toml` src layout, CLI stub with help, config parser, formatting/tests/CI starter. Exit gate: import + CLI help + two parser tests.

**M1: reference models & exporters**: registry, 3 deterministic small PyTorch modules, ONNX export/cache/check. Exit gate: fresh exports, checker passes, one ORT run per model, dynamic batch check for MLP/CNN.

**M2: datasets & replay data**: seeded data generator, case ID and hash logic, NPZ safe persistence, negative tests. Gate: distinct seeds reproducible and stable, invalid inputs recognized, no pickle.

**M3: parity suite**: PyTorch/ORT runners, numerical metrics, outcome classification, traceable logs and exception mapping. Gate: passing real case, artificially perturbed FAIL case, deliberate shape ERROR, tolerance boundary tests.

**M4: CLI/report**: all commands, manifest/cases/metrics, self-contained HTML, reproduce functionality, doctor. Gate: CLI end-to-end on CPU with outputs; report works offline; reproduce consistent.

**M5: quality & demo**: CI, tests, README, license, build/packaging, `ruff` clean. Gate: fresh demo and test suite. Collect actual run facts for final handoff.

When blocked on advanced exporter capability, preserve working CPU baseline, isolate capability into a feature flag, update limitation doc, continue remaining milestones.
