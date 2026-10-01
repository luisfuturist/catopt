"""Slidev renderer — assets the pitch deck can pull in.

Emits, under ``outdir``:

* ``<suite>.json`` — headline numbers + findings, for a Vue component to
  read; and
* ``<suite>.md`` — a ready-to-``src:``-import Markdown slide fragment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from bench.benchkit.stats import fmt_ms, param_keys, variant_names

if TYPE_CHECKING:
    from bench.benchkit.report import Report


def _headline_table(report: Report) -> list[str]:
    names = variant_names(report.cells)
    keys = param_keys(report.cells)
    cols = ["case", *keys, *names]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for cell in report.cells:
        row = [cell.case.name]
        row += [str(cell.case.params.get(k, "—")) for k in keys]
        row += [
            fmt_ms(cell.medians[n]) if n in cell.medians else "—"
            for n in names
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_slidev(
    report: Report, outdir: str | Path, figures: dict[str, str]
) -> dict[str, str]:
    """Write the Slidev asset pair; returns ``{kind: path}``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "suite": report.suite,
        "title": report.title or report.suite,
        "summary": report.summary,
        "findings": [
            {
                "claim": f.claim,
                "verdict": f.verdict.value,
                "headline": f.headline,
                "metric": f.metric,
                "value": f.value,
            }
            for f in report.findings
        ],
        "figures": figures,
    }
    json_path = outdir / f"{report.suite}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md = [
        "---",
        "layout: default",
        f"title: {report.title or report.suite}",
        "---",
        "",
        f"# {report.title or report.suite}",
        "",
    ]
    if report.summary:
        md += [report.summary, ""]
    for f in report.findings:
        md.append(f"- **{f.verdict.badge}** — {f.headline or f.claim}")
    if report.findings:
        md.append("")
    md += _headline_table(report)
    for name, path in figures.items():
        md += ["", f"![{name}]({path})"]
    md_path = outdir / f"{report.suite}.md"
    md_path.write_text("\n".join(md) + "\n")

    return {"json": str(json_path), "markdown": str(md_path)}
