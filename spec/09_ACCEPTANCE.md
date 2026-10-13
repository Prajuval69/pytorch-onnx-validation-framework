# Definition of done / release checklist

## P0 — required (all must be satisfied)
- [ ] Installation via `pip install -e '.[dev]'` on CPU Linux documented and exercised.
- [ ] MLP/CNN/simple-attention export successfully to ONNX; `onnx.checker.check_model` succeeds.
- [ ] At least MLP and CNN run at two legal dynamic batch sizes using one exported model each.
- [ ] Three model families execute with actual PyTorch + actual ONNX Runtime CPU; identical inputs and frozen weights.
- [ ] Deterministic synthetic data generation; safe case serialization + hash replay.
- [ ] Correct numerical comparator with tolerance boundary, nonfinite, and shape mismatch tests.
- [ ] `doctor`, `demo`, `validate`, `reproduce`, `report` CLI commands work.
- [ ] Real JSON/CSV/HTML reports generated with machine-readable schema.
- [ ] Injected mismatch leads to a clear FAIL, nonzero exit, replayable saved case.
- [ ] `pytest`, `ruff`, Linux CI + README/docs + MIT license.
- [ ] No fabricated validation results, no GPUs/boards/API keys required.

## P1 — valuable bonus only if P0 complete
- [ ] Export operator inventory and ONNX graph node/op summaries.
- [ ] Property-based randomized shape/distribution testing with Hypothesis.
- [ ] Multiple outputs, user-supplied ONNX model option, case filtering.
- [ ] Compare results against a stored baseline with stable model/export versions.

## Final Codex handoff
Show exact verification commands and outcomes; where tests weren't run, mark unverified. Include example observed run counts and paths only when they actually exist.
