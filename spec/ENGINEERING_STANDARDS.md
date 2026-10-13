# Engineering standards and Codex working rules

## Development principles
- Implement code, not just pseudocode. Complete all required vertical slices from CLI to reports.
- Python 3.11+ supported; prefer plain portable Python and `src/` packaging via `pyproject.toml`.
- CPU-only Linux is the required baseline. Windows/macOS compatibility is a bonus, not a blocker.
- No paid APIs, API keys, cloud account, proprietary data, Docker, or special boards required.
- Keep dependencies minimal. Do not download pretrained models or large datasets; synthesize small deterministic models and test inputs locally.
- Use type annotations, dataclasses or Pydantic where useful, structured logging, sane exceptions, and clear module boundaries.
- Do not train a huge model. Generated small random-weight models are enough for validation. They are not representative of production accuracy.
- Use safe file operations; do not overwrite user data or traverse outside configured artifact dirs; bound input sizes.
- Use a license-compatible dependency set; place upstream attribution and verified links in README.
- Supply `pyproject.toml`, `.gitignore`, `README.md`, `LICENSE` (MIT), CLI entry points, unit and integration tests.
- Use `ruff`, `pytest`; use `mypy` only if readily configured without delaying working functionality.
- Build a Linux GitHub Actions workflow running formatting/lint/test/smoke and publishing no artifacts externally.
- Run independent smoke test on a fresh temporary output folder. Never declare a check passed without running it.
- Record environment versions in generated artifacts; use local UTC ISO 8601 timestamps and monotonic timing for durations.

## Dependencies and environment
- Create a virtualenv if possible: `python -m venv .venv && . .venv/bin/activate`.
- Resolve package versions against the real environment and official documentation rather than pinning invented version pairs.
- Prefer compatible bounded version ranges with lock/reproducibility instructions. Generate a constraints/lock file if tooling supports it.
- Use `python -m pip install -e '.[dev]'` as primary setup. If optional dependency missing, skip only the optional feature with a documented reason.
- Never silently substitute another execution provider or precision mode; annotate requested vs actual.
- If installation/network is unavailable, implement as much as possible, run what works, and document precise blockers.

## Repo discipline
- Do not stop after writing a plan. Work through milestones and verify outputs in this session.
- Commit logically if Git is available; never require a remote GitHub repository.
- Existing specification files under `spec/` must be retained as development context, but the generated project README should describe the implemented software.
- Treat milestone checklists as work items, not as a reason to ask the user for approval.
- At finish report: changed files, runnable commands, tests actually run, output paths, known limitations and TODOs.
- If one advanced feature is unsupported by installed versions, implement a graceful capability probe, keep CPU critical path working, mark the feature unavailable and explain it.
- Under no circumstances fabricate benchmark values, performance uplift, accelerator usage, physical board validation or test coverage.

## Required user experience
1. `python -m pip install -e '.[dev]'` installs project + tests on a supported CPU Linux environment.
2. `--help` works for main CLI and each subcommand.
3. A single `demo` command generates a self-contained sample report using locally generated models/inputs.
4. Tests include positive, negative, determinism and artifact-schema cases.
5. Reports are local static HTML with no server/API requirement; raw JSON/CSV retained for auditing.
6. Meaningful failure types return nonzero exit codes (except explicitly classified optional skips).
