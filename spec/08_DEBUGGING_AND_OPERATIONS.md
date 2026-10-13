# Debugging, performance and failure handling

- `doctor` prints versions + providers + OS + architecture + warning if optional GPU not installed (GPU is out of scope, no error).
- Export issues: store exception type/message, model name, exporter mode, opset and installed versions. Do not mark as parity success.
- Inference issues: differentiate invalid test input expected rejection vs unexpected runtime failure. Log stack trace to local log; HTML shows concise summary.
- Memory limits: default small models and B<=8. Validate user-configured maximal tensor elements/bytes to avoid accidental CPU exhaustion.
- In a nonfinite output case, numeric metrics have explicit nullable fields; never serialize NaN/Infinity into JSON (use JSON `allow_nan=False`).
- Reproduction artifacts: include test input and hashes, deterministic model init seed/export config and model hash. Replay exact saved input and ONNX model where possible rather than regenerating with unknown weights. If fault injection produced a case FAIL, serialize perturbation type, target output/index, amplitude and seed; only replay the injected mismatch when explicitly marked, and display the simulated-fault notice.
- Underlying ORT/PyTorch differences can arise from floating-point reduction order. Be exact about default tolerances and record observed data.
