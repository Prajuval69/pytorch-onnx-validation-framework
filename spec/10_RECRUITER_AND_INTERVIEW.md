# Portfolio positioning: Quest Global — AI/ML Validation Engineer

## 30-second explanation
'I built a framework for checking whether exporting PyTorch networks to ONNX changes inference behavior. It generates reproducible tensor inputs, verifies ONNX graphs, tests multiple legal batch shapes, compares results numerically, and outputs failure cases that can be replayed under Linux.'

## Explain with a demonstration
1. Run `amlv doctor` and show CPU provider.
2. Run `amlv demo` and open the report.
3. Show 3 synthetically initialized models and distribution matrix.
4. Run `amlv demo --inject-mismatch`, show failure at a tensor coordinate and replay instructions.
5. Explain why `atol+rtol*|ref|` is better than a simple global average error.

## Interview questions to prepare
- Why might ONNX vs PyTorch differ even with the same network?
- How do dynamic-shape exporters work, and how do you test them?
- Why do near-zero denominators make relative error misleading?
- How do you distinguish model exporter bugs from unsupported ONNX operators?
- How would you extend the framework to actual board-level validation if supplied board access?

## Resume bullet drafts (edit after verifying real facts)
- Developed CPU-based PyTorch-to-ONNX numerical validation framework with deterministic synthetic data generation, batch-shape testing, and ONNX Runtime parity checks.
- Implemented replayable mismatch diagnostics, CI checks, and self-contained HTML/CSV/JSON validation reporting for three synthetic model families.

These are descriptions of planned deliverables, not proof that implementation/tests happened. Only use after finished.
