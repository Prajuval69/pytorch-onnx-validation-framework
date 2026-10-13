# Reports and UI standards

Create `report.html` self-contained (no external CSS/JS fonts/CDN). Make it look like an engineering validation report, not a toy.

## Page sections
1. Hero: run ID, synthetic-data notice, verdict banner, start date and environment.
2. KPI row: cases, pass/fail/error/skip, tested models, worst absolute error.
3. Model matrix: per-model case totals and export/graph-check status.
4. Sortable/searchable case table (vanilla JS allowed): model, shape, distribution, verdict, max abs, max rel, RMSE, cosine, error reason.
5. Worst-failure detail: expected and actual sample values, failing tensor index, threshold, reproducible case command.
6. Methodology: tolerance formula, reference backend, ONNX provider, artifact hash caveats and synthetic fault banner.
7. Evidence links to relative JSON/CSV/NPZ artifacts.

Charts optional (inline SVG desirable), but never use external scripts. Ensure accessible contrast, mobile readability and no misleading zero comparisons. Validate report even when some cases ERROR or SKIP.

Machine-readable outputs must conform to `spec/schemas/manifest.schema.json` and `spec/schemas/case.schema.json`. Include `schema_version: 1`.
