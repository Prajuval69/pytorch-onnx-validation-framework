# Test strategy and QA matrix

## Unit tests
- Comparator: exact equality; within tolerance; outside tolerance; near zero; all-zero cosine undefined; NaN/Inf handling; multiple outputs; mismatched shapes/names.
- Data: same spec+seed -> equal arrays and fingerprints; different seed -> distinct arrays; bounded amplitudes; invalid config rejected.
- Artifacts: NPZ round-trip with `allow_pickle=False`; SHA integrity check; missing/corrupt file fails loudly; paths remain inside artifact root.
- Outcome priority: ERROR > FAIL; SKIP optional only; correct CLI exit status.

## Integration tests (real inference)
- Export the actual synthetic MLP/CNN/attention models; check ONNX graph; run actual ORT CPU inference with exactly the saved inputs.
- Test batches 1 and 4 at minimum for MLP/CNN dynamic batch.
- MLP normal/zeros; CNN uniform/ones; attention near-zero; tolerance based on calculated difference.
- Inject numerical mismatch in comparator inputs and assert failure with correct metadata (negative demonstration).
- Confirm `reproduce` matches original observed verdict/metrics within numeric precision.

## CLI/report tests
- `amlv demo --out <tmp>` exits zero in standard CPU installation; verifies schema, summary counts, HTML title and no remote dependencies.
- `--inject-mismatch` exits 1 and report marks an intentional injected error.
- `amlv validate` rejects nonexistent file, unknown model, negative tolerance, unrecognized distribution, unwriteable output.
- `amlv report` from saved JSON does not export or run models.
- Smoke duration should be bounded for CI. Conditional capability tests use clear pytest skip reasons; never skip core CPU functionality.

## CI
Ubuntu runner; Python 3.11 stable baseline; install CPU dependencies; `ruff check .`, `pytest -q`, one demo invocation and artifact-schema sanity check. Do not gate CI on absolute speed or cloud/GPU availability.
