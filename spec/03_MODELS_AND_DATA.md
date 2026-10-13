# Model and test-data specification

## Minimal required model registry

| Model ID | Architecture | Input shape | Output example | Notes |
|---|---|---|---|---|
| `mlp` | Linear(32,64) -> ReLU -> Linear(64,8) | `[B,32]` float32 | `[B,8]` | stable baseline |
| `cnn` | Conv2d(1,4,3,pad=1) -> ReLU -> global avg pool -> Linear(4,6) | `[B,1,16,16]` float32 | `[B,6]` | convolution path |
| `attention` | Small Q/K/V projections, `matmul`, scaled scores, softmax, output proj | `[B,4,16]` float32 | `[B,4,16]` | use simple ONNX-exportable ops |

B = 1,2,4 (and 8 optional). Initialize weights deterministically, not trained. Explicitly label results **numerical equivalence of synthetic models**, not model predictive accuracy.

## ONNX conversion
- Export with `dynamo=True` where supported, `model.eval()`, `torch.inference_mode()` for inference.
- For dynamic batch: `torch.export.Dim` and `dynamic_shapes` with a legal range. Verify B=1 and B=4 at minimum on MLP and CNN (may use B=2 as export example).
- Model inputs/outputs must use explicit names. Use `onnx.checker.check_model()`; shape inference may be attempted but should not invalidate a checker-passing model due to noncritical inference limitations.
- Ensure exporter result correctly serialized with current installed torch version. Capture warnings/reports.

## Data generation
- Default seed 2026, independent `numpy.random.default_rng(seed)` per test case; stable ID based on model/shape/distribution/seed.
- Distributions: normal N(0,1); uniform [-1,1]; zeros; ones; near-zero uniform [-1e-6,1e-6]; bounded large magnitude uniform [-100,100].
- Every tensor float32 unless a particular test explicitly targets another supported dtype.
- Each model gets >=3 distributions x >=2 batch sizes in smoke; full example config uses more.
- Negative tests: wrong rank, incompatible shape, dtype that runtime rejects, missing named input. These test expected-rejection handling, NOT parity metrics. Store `test_kind: negative_input` and `expected_exception` separately. Mark negative test PASS if and only if expected rejection occurs; unexpected successful inference is FAIL, unexpected exception category is ERROR.
- Do not use NaN/Inf as ordinary parity inputs by default. Add explicit tagged diagnostic cases and classify according to configured nonfinite policy.
- Save original test arrays in compressed NPZ, plus SHA256; `np.load(..., allow_pickle=False)` on replay.

## Numeric metrics
- Verdict per output from `abs(reference-candidate) <= atol + rtol*abs(reference)` using both `atol=1e-5` and `rtol=1e-4` by default for float32, configurable.
- `max_abs`, `max_rel` (denominator max(abs(ref), atol, epsilon)), `rmse`, `mean_abs`, `cosine_similarity` (undefined for zero norms and reported as null), total/mismatched element count.
- Return FAIL if at least one element exceeds tolerance, or output name/shape/dtype incompatible. Handle zero denominators rigorously; never average away a large error.
- Separate compare function receives NumPy arrays and can be tested without ONNX deps for mismatch injection.
- Provide worst mismatch tensor coordinate and both values without flooding reports.
