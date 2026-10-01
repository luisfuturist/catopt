"""HTML renderer — a single-file, dark-themed report page.

Findings become verdict-badged cards; the timing table is rendered from
the same data as the Markdown table; plotly figures are embedded.  With
``self_contained`` the plotly.js bundle is inlined so the file can be
emailed or opened offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

from bench.benchkit.stats import fmt_pm, param_keys, variant_names

if TYPE_CHECKING:
    from bench.benchkit.report import Report

_ENV = Environment(
    loader=PackageLoader("bench.benchkit", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _columns_and_rows(
    report: Report,
) -> tuple[list[str], list[list[str]]]:
    names = variant_names(report.cells)
    keys = param_keys(report.cells)
    columns = ["case", *keys, *[f"{n} (ms)" for n in names]]
    rows: list[list[str]] = []
    for cell in report.cells:
        row = [cell.case.name]
        row += [str(cell.case.params.get(k, "—")) for k in keys]
        for n in names:
            row.append(
                fmt_pm(cell.medians[n], cell.iqr.get(n, 0.0))
                if n in cell.medians
                else "—"
            )
        rows.append(row)
    return columns, rows


def _figures_html(report: Report, self_contained: bool) -> list[str]:
    from bench.benchkit.render.plots import _bar_grid, _speedup_fig

    figures = []
    grids: dict[tuple, list] = {}
    for cell in report.cells:
        grids.setdefault(tuple(cell.case.params.keys()), []).append(
            cell
        )
    for key, cells in grids.items():
        if not variant_names(cells):
            continue
        suffix = f" ({', '.join(key)})" if key else ""
        figures.append(
            _bar_grid(cells, f"{report.title or report.suite}{suffix}")
        )

    x_param = report.provenance.get("x_param")
    baseline = report.provenance.get("speedup_vs")
    if x_param and baseline:
        figures.append(
            _speedup_fig(
                report.cells,
                x_param,
                baseline,
                f"{report.title or report.suite}: speedup vs {baseline}",
            )
        )

    divs = []
    for i, fig in enumerate(figures):
        include = (
            True
            if (self_contained and i == 0)
            else ("cdn" if not self_contained else False)
        )
        divs.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=include,
                config={"displaylogo": False},
            )
        )
    return divs


def render_page(report: Report, self_contained: bool = True) -> str:
    """Render the report to a standalone HTML document."""
    columns, rows = _columns_and_rows(report)
    findings = [
        {
            "verdict": f.verdict.value,
            "badge": f.verdict.badge,
            "claim": f.claim,
            "headline": f.headline,
            "metric": f.metric,
            "value": f"{f.value:g}" if f.value is not None else "",
        }
        for f in report.findings
    ]
    template = _ENV.get_template("report.html.j2")
    return template.render(
        title=report.title or report.suite,
        summary=report.summary,
        env=report.env,
        findings=findings,
        columns=columns,
        rows=rows,
        figures=_figures_html(report, self_contained),
        generated=datetime.now(UTC).isoformat(timespec="seconds"),
        ncells=len(report.cells),
    )
