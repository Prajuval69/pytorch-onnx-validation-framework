# AGENTS.md — autonomous Codex instructions

You are implementing the software specified in this repository, not answering with a design proposal.

Read `MASTER_CODEX_PROMPT.md`, then `spec/01_PRODUCT_BRIEF.md`, `spec/02_ARCHITECTURE.md`, `spec/04_CLI_AND_CONFIG.md`, `spec/06_TEST_PLAN.md`, `spec/09_ACCEPTANCE.md`, and the remaining specs.

Your objective: produce a fully runnable, tested, CPU-first Python project. Do not request credentials, an API key, a GPU, or user approval for ordinary engineering decisions. No Docker is required. Work step-by-step, adjusting to installed library capabilities, and implement required acceptance criteria. Optional functionality must never break CPU basic paths.

Before claiming completion run (or document why you could not run): install, `pytest`, `ruff check .`, CLI help, a demo invocation, and a review of generated report artifacts. Give a truthful final implementation summary.

Do not create fictitious success metrics or claim real board-level testing. All example results must come from an actual local execution, be labeled synthetic, and contain environment metadata.
