# START HERE — Codex spec pack

This ZIP is an **implementation specification**, not the completed application. It provides a complete work order for Codex.

## How to use
1. Extract this ZIP into a new empty project folder.
2. Open the folder in Codex (IDE extension, Codex app, or terminal Codex).
3. Send Codex the contents of `MASTER_CODEX_PROMPT.md` (or the one-line starter prompt below).
4. Allow Codex to edit files and run local installation, tests and demos.
5. Review its final report and actual generated artifacts. Push to GitHub only after validating its work.

**Starter message (copy/paste):**

> Read `MASTER_CODEX_PROMPT.md` and `AGENTS.md`, then implement the entire project from the `spec/` files. Do not stop after planning. Work autonomously through all milestones, run tests, generate the demo report, fix failures, and give me a factual final verification summary. Use CPU-only fallback; do not require Docker, hardware, or API keys.

## What is in this ZIP?
- `MASTER_CODEX_PROMPT.md`: the main end-to-end Codex execution prompt.
- `AGENTS.md`: agent working principles and test gates.
- `spec/`: product, architecture, CLI, dataset/model, build roadmap, QA, UX, and acceptance specs.
- `config/`: example YAML configuration that the implementation must support.
- `spec/schemas/`: authoritative machine-readable output contracts.

## Important
The packages contain no finished implementation. The generated code, benchmark measurements and sample reports must be produced by Codex through real execution.
