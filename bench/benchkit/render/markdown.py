"""Markdown renderer — readable tables, findings, collapsible aux.

The old renderer inlined ``repr()`` of every nested aux dict, producing
multi-thousand-character truncated lines.  Here aux goes into a
``<details>`` block as pretty JSON, and the timing table is kept narrow
(timings and speedups are separate tables).
"""
# ruff: noqa: RUF001 -- the multiplication sign in tables is notation.

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from bench.benchkit.stats import (
    fmt_pm,
    param_keys,
    speedup,
    variant_names,
)

if TYPE_CHECKING:
    from bench.benchkit.report import Report


def _env_lines(env: dict) -> list[str]:
    dirty = " dirty" if env.get("git_dirty") else " clean"
    return [
        f"- **when**: {env.get('timestamp_utc', '?')}"
        f" (git `{env.get('git_sha', '?')}`{dirty})",
        f"- **device**: `{env.get('device', '?')}`"
        f" — {env.get('device_name', '?')}",
        f"- **torch**: {env.get('torch', '?')}"
        f" · **cuda**: {env.get('torch_cuda', '?')}"
        f" · **python**: {env.get('python', '?')}",
    ]


def _findings_section(report: Report) -> list[str]:
    if not report.findings:
        return []
    lines = ["", "## Findings", ""]
    lines.append("| verdict | claim | headline | metric |")
    lines.append("|---|---|---|---|")
    for f in report.findings:
        metric = (
            f"{f.metric} = {f.value:g}"
            if f.metric and f.value is not None
            else "—"
        )
        claim = f.claim.replace("|", "\\|")
        headline = f.headline.replace("|", "\\|")
        lines.append(
            f"| **{f.verdict.badge}** | {claim} | {headline} | {metric} |"
        )
    return lines


def _timing_table(report: Report) -> list[str]:
    cells = report.cells
    names = variant_names(cells)
    keys = param_keys(cells)
    cols = ["case", *keys, *[f"{n} ms" for n in names]]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for cell in cells:
        row = [cell.case.name]
        row += [str(cell.case.params.get(k, "—")) for k in keys]
        for n in names:
            if n in cell.medians:
                row.append(
                    fmt_pm(cell.medians[n], cell.iqr.get(n, 0.0))
                )
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _speedup_table(report: Report, baseline: str) -> list[str]:
    cells = report.cells
    names = [n for n in variant_names(cells) if n != baseline]
    keys = param_keys(cells)
    cols = ["case", *keys, *[f"{n}/vs-{baseline}" for n in names]]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for cell in cells:
        row = [cell.case.name]
        row += [str(cell.case.params.get(k, "—")) for k in keys]
        for n in names:
            s = speedup(cell, n, baseline)
            row.append(f"{s:.3f}×" if s is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _aux_section(report: Report) -> list[str]:
    aux_cells = [c for c in report.cells if c.aux]
    if not aux_cells:
        return []
    lines = ["", "## Metrics", ""]
    for cell in aux_cells:
        payload = json.dumps(
            cell.aux, indent=2, default=str, sort_keys=False
        )
        lines += [
            f"<details><summary><b>{cell.case.name}</b> — raw metrics"
            "</summary>",
            "",
            "```json",
            payload,
            "```",
            "",
            "</details>",
            "",
        ]
    return lines


def render_markdown(
    report: Report,
    speedup_vs: str | None = None,
    include_aux: bool = True,
) -> str:
    """Render a ``Report`` to GitHub-flavored Markdown."""
    title = report.title or report.suite
    lines = [f"# {title}", ""]
    if report.summary:
        lines += [report.summary, ""]
    lines += _env_lines(report.env)
    if report.provenance:
        items = " · ".join(
            f"`{k}`={v}" for k, v in report.provenance.items()
        )
        lines.append(f"- **config**: {items}")
    lines += _findings_section(report)
    lines += ["", "## Results", "", *_timing_table(report)]
    if speedup_vs:
        lines += [
            "",
            f"## Speedup vs `{speedup_vs}`",
            "",
            "> ratio > 1 means the variant is faster than the baseline",
            "",
            *_speedup_table(report, speedup_vs),
        ]
    if include_aux:
        lines += _aux_section(report)
    return "\n".join(lines) + "\n"
