# CLI + configuration contract

Entry-point executable: `amlv`, backed by `python -m amlv` if possible. Use `argparse` or `typer`; ensure all below are implemented and documented.

```bash
amlv --help
amlv doctor
amlv demo --out artifacts/demo
amlv validate --config config/validation.example.yaml --out artifacts/run1
amlv report --run artifacts/run1
amlv reproduce --case artifacts/run1/cases/<case-id>.json --out artifacts/replay
amlv demo --out artifacts/intentional-failure --inject-mismatch
```

Commands:
- `doctor`: Python/PyTorch/ONNX/ORT versions, installed providers, writable artifact root, dependency checks; never imply GPU present based on package alone.
- `demo`: generate/export all three reference models and execute a bounded representative suite; produce JSON, CSV and HTML.
- `validate`: parse provided YAML, validate schema and thresholds, generate fresh models/cases, run suite.
- `report`: regenerate HTML/CSV from immutable run JSON, no rerun of inference.
- `reproduce`: rerun exact model and saved input from case manifest; checksum validation; for an intentional injected mismatch, faithfully replay the stored deterministic perturbation only when the case metadata requests fault replay, and label it clearly. Without perturbation, the underlying parity may PASS. Optional tolerance override should be explicit.
- `--inject-mismatch`: controlled deterministic perturbation applied **only to comparison input/output copy**, never alter original saved model; mark every report clearly `synthetic_fault_injection=true`.

Config example is authoritative: `config/validation.example.yaml`. Schema implementation should reject unknown keys, invalid distributions, unsupported model names and negative tolerances cleanly.

Output per run:
```
artifacts/run1/
  manifest.json
  metrics.csv
  report.html
  models/*.onnx
  cases/*.json
  inputs/*.npz
  logs/validation.log
```

Use relative artifact paths in stored manifest and cases. Determine `run_id` from timestamp plus deterministic short suffix or UUID; case IDs stable independent of run ID.

Exit code contract: 0 all expected parity tests PASS with optional SKIPs; 1 at least one numerical FAIL; 2 config/CLI invalid; 3 export/runtime/system ERROR. Document if an invocation contains multiple categories and provide consistent precedence (ERROR > FAIL).
