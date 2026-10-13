"""Self-contained HTML and CSV from immutable JSON; no model imports or inference."""

import csv
import html
import io
import json
from collections import Counter
from pathlib import Path

from amlv.case_store import atomic_text, read_json, safe_path, sha256_file
from amlv.errors import ArtifactError


def read_run(root: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(root / "manifest.json")
    try:
        if manifest["schema_version"] != 1:
            raise ArtifactError("Unsupported manifest schema")
        path = safe_path(root, manifest["cases_file"])
        if sha256_file(path) != manifest["cases_file_sha256"]:
            raise ArtifactError("Case JSONL integrity failure")
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        counts = Counter(r["status"].lower() for r in records)
        if manifest["summary"] != {
            "total": len(records),
            **{v: counts[v] for v in ("pass", "fail", "error", "skip")},
        }:
            raise ArtifactError("Manifest counts differ from case records")
        for record in records:
            if record["status"] not in ("PASS", "FAIL", "ERROR", "SKIP"):
                raise ArtifactError("Unknown case status")
            for key in ("inputs_path", "case_path"):
                safe_path(root, record[key])
            case_path = safe_path(root, record["case_path"])
            if sha256_file(case_path) != manifest["case_json_sha256"][record["case_id"]]:
                raise ArtifactError("Individual case JSON integrity failure")
            if read_json(case_path) != record:
                raise ArtifactError("Case JSONL and case file differ")
        return manifest, records
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ArtifactError(f"Malformed report metadata: {exc}") from exc


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.6g}"


def _link(path: str, label: str) -> str:
    return f'<a href="{_e(path)}">{_e(label)}</a>'


def _metrics_rows(record: dict):
    return record["metrics"].items() or [("—", {})]


def csv_text(records: list[dict]) -> str:
    stream = io.StringIO(newline="")
    columns = [
        "case_id",
        "model",
        "test_kind",
        "shape",
        "distribution",
        "status",
        "output",
        "max_abs",
        "max_rel",
        "mean_abs",
        "rmse",
        "cosine_similarity",
        "total_elements",
        "mismatched_elements",
        "nonfinite_elements",
        "reason",
        "synthetic_fault_injection",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for record in records:
        for output, metrics in _metrics_rows(record):
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "model": record["model_id"],
                    "test_kind": record["test_kind"],
                    "shape": json.dumps(record["input_shapes"]),
                    "distribution": record["distribution"],
                    "status": record["status"],
                    "output": output,
                    "reason": record.get("reason") or "",
                    "synthetic_fault_injection": record["intentional_fault_injection"],
                    **{k: metrics.get(k) for k in columns if k in metrics},
                }
            )
    return stream.getvalue()


STYLE = """
:root{color-scheme:dark;--bg:#0c1219;--surface:#151e29;--line:#293746;--ink:#edf3f8;
--muted:#a8bacb;--pass:#7fe1b4;--fail:#ffac93;--error:#ffd478;--accent:#84c9f2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1480px;margin:auto;
padding:48px 36px}a{color:var(--accent)}header{border-bottom:1px solid var(--line);
padding-bottom:30px}
.eyebrow{letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-size:12px}
h1{font-size:clamp(32px,4vw,58px);line-height:1.1;letter-spacing:-.035em;margin:16px 0}
h2{font-size:22px;margin:0 0 18px}.lead{font-size:18px;max-width:900px;color:var(--muted)}
.verdict{padding:8px 14px;border:1px solid currentColor;display:inline-block;font-weight:700}
.PASS{color:var(--pass)}.FAIL{color:var(--fail)}.ERROR{color:var(--error)}.SKIP{color:var(--muted)}
.notice{background:#34251d;border-left:4px solid var(--fail);padding:16px;margin:24px 0}
.kpis{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));border:1px solid var(--line);
margin:30px 0}.kpi{padding:20px;border-right:1px solid var(--line)}.kpi:last-child{border:0}
.kpi strong{display:block;font-size:29px;line-height:1.3}.kpi small{color:var(--muted)}
section{margin-top:44px}.table-wrap{overflow:auto;border:1px solid var(--line)}table{width:100%;
border-collapse:collapse;text-align:left;font-size:13px}th,td{padding:13px 15px;
border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--surface);
color:var(--muted);white-space:nowrap}tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface)}td.numeric{font-variant-numeric:tabular-nums;
white-space:nowrap}
button{border:0;background:transparent;color:inherit;font:inherit;cursor:pointer;text-align:left}
button:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
input{width:min(480px,100%);background:var(--surface);color:var(--ink);border:1px solid var(--line);
padding:12px;font:inherit;margin:8px 0 18px}code,pre{font-family:ui-monospace,Consolas,monospace;
font-size:13px}pre{overflow:auto;background:var(--surface);padding:18px;
border:1px solid var(--line)}
.muted{color:var(--muted)}.details{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.detail{background:var(--surface);padding:22px;border:1px solid var(--line)}dl{display:grid;
grid-template-columns:140px 1fr;gap:8px}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}
@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}.kpi{border-bottom:1px solid
var(--line)}.details{grid-template-columns:1fr}main{padding:28px 18px}
dl{grid-template-columns:110px 1fr}}
"""

SCRIPT = """
const search=document.getElementById('search');
search.addEventListener('input',()=>{
  const value=search.value.toLowerCase();
  document.querySelectorAll('#cases tbody tr').forEach(row=>{
    row.hidden=!row.textContent.toLowerCase().includes(value);
  });
});
document.querySelectorAll('#cases th button').forEach((button,index)=>{
  button.addEventListener('click',()=>{
    const body=document.querySelector('#cases tbody');
    const rows=Array.from(body.rows);const direction=button.dataset.asc==='true'?-1:1;
    button.dataset.asc=String(direction===1);
    rows.sort((a,b)=>{
      const av=a.cells[index].dataset.sort??a.cells[index].textContent;
      const bv=b.cells[index].dataset.sort??b.cells[index].textContent;
      return direction*((av!==''&&bv!==''&&isFinite(av)&&isFinite(bv))?
        Number(av)-Number(bv):av.localeCompare(bv));
    });
    rows.forEach(row=>body.appendChild(row));
    document.querySelectorAll('#cases th').forEach(th=>th.removeAttribute('aria-sort'));
    button.parentElement.setAttribute('aria-sort',direction===1?'ascending':'descending');
  });
});
"""


def html_text(manifest: dict, records: list[dict]) -> str:
    counts = manifest["summary"]
    verdict = "ERROR" if counts["error"] else "FAIL" if counts["fail"] else "PASS"
    env = manifest["environment"]
    values = [
        m["max_abs"] for r in records for m in r["metrics"].values() if m.get("max_abs") is not None
    ]
    worst_abs = max(values) if values else None
    labels = [
        ("Cases", counts["total"]),
        ("PASS", counts["pass"]),
        ("FAIL", counts["fail"]),
        ("ERROR", counts["error"]),
        ("SKIP", counts["skip"]),
        ("Models", len(manifest["models"])),
        ("Worst |Δ|", _fmt(worst_abs)),
    ]
    kpis = "".join(
        f'<div class="kpi"><strong>{_e(value)}</strong><small>{_e(label)}</small></div>'
        for label, value in labels
    )
    model_rows = []
    for model_id, export in manifest["models"].items():
        model_counts = Counter(r["status"] for r in records if r["model_id"] == model_id)
        status = export["status"]
        graph = export.get("graph_check", "unavailable")
        reason = export.get("fallback_reason") or (export.get("error") or {}).get("message") or "—"
        model_rows.append(
            f"<tr><td>{_e(model_id)}</td><td>{sum(model_counts.values())}</td>"
            f'<td>{_e(dict(model_counts))}</td><td class="{status}">{status}</td>'
            f"<td>{_e(graph)}</td><td>{_e(export.get('exporter_mode', '—'))}</td>"
            f"<td>{_e(export.get('opset_version', '—'))}</td>"
            f"<td>{_e(export.get('dynamic_batch', '—'))}</td>"
            f"<td>{_e(reason)}</td></tr>"
        )
    case_rows = []
    for record in records:
        for output, metrics in _metrics_rows(record):
            status = record["status"]
            shape = " / ".join(str(v) for v in record["input_shapes"].values()) or "missing input"
            cells = [
                f"<td>{_link(record['case_path'], record['model_id'])}<br>"
                f'<small class="muted">{_e(record["case_id"][:10])}</small></td>',
                f"<td>{_e(shape)}</td>",
                f"<td>{_e(record['distribution'])}<br><small>{_e(record['test_kind'])}"
                f"{': ' + _e(record['negative_reason']) if record.get('negative_reason') else ''}"
                "</small></td>",
                f'<td class="{status}">{status}</td>',
                f"<td>{_e(output)}</td>",
            ]
            for key in ("max_abs", "max_rel", "rmse", "cosine_similarity"):
                value = metrics.get(key)
                cells.append(
                    f'<td class="numeric" data-sort="{_e(value if value is not None else "")}">'
                    f"{_fmt(value)}</td>"
                )
            cells.append(
                f"<td>{_e(record.get('reason') or 'Within tolerance')}"
                f"{' • SYNTHETIC FAULT' if record['intentional_fault_injection'] else ''}"
                f"<br>{_link(record['inputs_path'], 'NPZ input')}</td>"
            )
            case_rows.append("<tr>" + "".join(cells) + "</tr>")
    failures = [(r, name, m) for r in records for name, m in r["metrics"].items() if m.get("worst")]
    if failures:
        record, name, metric = max(failures, key=lambda v: v[2].get("max_abs") or 0)
        fault_description = (
            "Intentional synthetic perturbation replayed from metadata."
            if record["fault"]
            else "Observed numerical mismatch."
        )
        detail = (
            f"<p><strong>{_e(record['model_id'])} / {_e(name)}</strong> · "
            f"{_e(record['case_id'])}</p>"
            f"<pre>{_e(json.dumps(metric['worst'], indent=2))}</pre>"
            f"<p>{fault_description}</p>"
            f"<pre>amlv reproduce --case &lt;run&gt;/{_e(record['case_path'])} "
            "--out artifacts/replay</pre>"
        )
    elif counts["error"] or counts["fail"]:
        detail = "<p>No numeric failure sample is available. See case error/rejection details.</p>"
    else:
        detail = "<p>No observed mismatches in this run.</p>"
    thresholds = manifest["config"]["compare"]
    environment_details = "".join(
        f"<dt>{_e(key)}</dt><dd>{_e(env.get(key, 'unavailable'))}</dd>"
        for key in (
            "python",
            "pytorch",
            "onnx",
            "onnxruntime",
            "numpy",
            "platform",
            "architecture",
            "selected_providers",
        )
    )
    fault_notice = (
        '<div class="notice"><strong>Intentional synthetic fault injection</strong><br>'
        "synthetic_fault_injection=true. A deterministic perturbation affects only a comparison "
        "output copy. This failure is a detection demonstration.</div>"
        if manifest["fault_injection"]
        else ""
    )
    title = "AMLV · Synthetic model validation"
    headers = (
        "Model / case",
        "Input shape",
        "Distribution / kind",
        "Verdict",
        "Output",
        "Max abs",
        "Max relative",
        "RMSE",
        "Cosine",
        "Reason / evidence",
    )
    table_headers = "".join(
        f'<th scope="col"><button type="button" aria-label="Sort by {_e(h)}">'
        f"{_e(h)} ↕</button></th>"
        for h in headers
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">AMLV / CPU conformance evidence</div>
<h1>Synthetic model validation.</h1><p class="lead">Numerical equivalence of frozen PyTorch models
and their ONNX exports using identical synthetic inputs.</p>
<span class="verdict {verdict}">{verdict}</span>
<p class="muted">Run {_e(manifest["run_id"])} · {_e(manifest["started_at_utc"])}<br>
Duration {_fmt(manifest["duration_seconds"])} seconds · CPUExecutionProvider</p></header>
{fault_notice}<div class="kpis">{kpis}</div>
<section><h2>Model export matrix</h2><div class="table-wrap"><table>
<thead><tr><th>Model</th><th>Cases</th><th>Outcomes</th><th>Setup</th><th>Graph check</th>
<th>Exporter</th><th>Opset</th><th>Dynamic batch</th><th>Compatibility / error</th></tr></thead>
<tbody>{"".join(model_rows)}</tbody></table></div></section>
<section><h2>Case evidence</h2>
<label for="search">Search model, verdict, distribution, or reason</label><br>
<input id="search" type="search" placeholder="Filter cases…"><div class="table-wrap">
<table id="cases"><thead><tr>{table_headers}</tr></thead><tbody>{"".join(case_rows)}</tbody></table>
</div><p class="muted">Negative-input PASS means both backends rejected the expected input defect.
Their numerical metrics are undefined (—), not zero.</p></section>
<section class="details"><div class="detail"><h2>Worst failure detail</h2>{detail}</div>
<div class="detail"><h2>Execution environment</h2><dl>{environment_details}</dl></div></section>
<section><h2>Methodology &amp; reproducibility</h2>
<p><code>abs(reference − candidate) &lt;= atol + rtol × abs(reference)</code></p>
<p>atol={_e(thresholds["atol"])} · rtol={_e(thresholds["rtol"])} ·
nonfinite policy={_e(thresholds["nan_policy"])}.
Robust relative error uses max(abs(reference), atol, float64 epsilon). Cosine is diagnostic only;
zero norms are null. Nonfinite metrics are null; matching NaN/Inf requires
the explicit equal policy.</p>
<p>Reference: CPU PyTorch eval/inference mode. Candidate: actual ONNX Runtime CPU inference.
Weights and original inputs are stored in safe NPZ; SHA256 checks detect corruption. Hashes do not
authenticate artifacts from an untrusted source. Environment changes can alter
last-bit arithmetic.</p>
<p>Evidence: {_link("manifest.json", "Run manifest")} ·
{_link(manifest["cases_file"], "Cases JSONL")} ·
{
        _link(manifest["csv_report"], "Metrics CSV")
        if manifest["config"]["report"]["csv"]
        else "CSV disabled"
    } ·
{_link("logs/validation.log", "Validation log")}</p></section>
<footer>Synthetic models and data. No training, predictive accuracy measurement, GPU validation,
or physical board testing. Offline report with no remote scripts, fonts, or services.</footer>
</main><script>{SCRIPT}</script></body></html>"""


def build_report(root: Path) -> dict:
    manifest, records = read_run(root)
    options = manifest["config"]["report"]
    if options["csv"]:
        atomic_text(safe_path(root, manifest["csv_report"]), csv_text(records))
    if options["html"]:
        atomic_text(safe_path(root, manifest["html_report"]), html_text(manifest, records))
    return manifest
