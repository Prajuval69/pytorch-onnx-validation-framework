# JD traceability matrix

| Quest Global requirement | Implemented evidence to build | Test/demo proof |
|---|---|---|
| Strong Python | typed CLI, package, tests | lint + pytest |
| Hands-on PyTorch | native reference models + inference | model unit & integration tests |
| ONNX | dynamo export, graph checks, runtime | ONNX checker + real ORT runs |
| Data generation workflows | seeded distributions, shapes, NPZ corpus | reproducibility test |
| Design validation | feature-level functional parity checks | pass/FAIL/error cases |
| Analytical/debugging | numeric metrics, worst offending output, replay | fault injection demonstration |
| Linux | CPU Linux CLI, CI | GitHub Actions and documented commands |
| Board-level testing | **Not performed**; discuss extensibility | no misleading hardware claims |
| CUDA/HIP/ROCm | nice-to-have, NOT essential in this first project | no unsupported claims |
