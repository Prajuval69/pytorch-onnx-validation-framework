# Verified official documentation / compatibility notes

Official project references to check at implementation time:
- PyTorch ONNX exporter: https://docs.pytorch.org/docs/stable/onnx
- ONNX Python tooling: https://onnx.ai/onnx/
- ONNX Runtime Python / API: https://onnxruntime.ai/docs/api/python/api_summary.html
- ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
- ONNX Runtime CUDA provider: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime MIGraphX provider (AMD): https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime profiling: https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html
- pytest: https://docs.pytest.org/en/stable/

## Implementation alerts (verified October 2026)
- The modern PyTorch ONNX exporter uses `torch.onnx.export(..., dynamo=True)`; `dynamic_shapes` is preferred over legacy `dynamic_axes` when using the new exporter. Verify the installed API before choosing a serialization approach.
- AMD ONNX Runtime `ROCMExecutionProvider` was removed in ONNX Runtime 1.23. For supported AMD hardware use MIGraphX EP instead, if the installed wheel supplies it. Do not advertise the deprecated EP as supported.
- An installed CUDA or MIGraphX execution provider may fall back to CPU for some nodes. Distinguish provider availability, actual selected providers, and node placement evidence, rather than assuming every operation ran on the GPU.
- Compare only like-for-like environments and use realistic tolerances. Microbenchmarks on shared CI are noisy.
