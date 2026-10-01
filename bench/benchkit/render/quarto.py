"""Quarto renderer — a ``.qmd`` source for report-quality HTML/PDF.

The Markdown body is reused verbatim (so the tables can never drift),
prefixed with Quarto YAML front matter and figure includes for the
static SVGs emitted by the plot renderer.  Rendering itself is the
Quarto CLI's job:

    quarto render bench/results/<suite>.qmd --to pdf
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bench.benchkit.render.markdown import render_markdown

if TYPE_CHECKING:
    from bench.benchkit.report import Report


def render_quarto(report: Report, figures: dict[str, str]) -> str:
    """Render a ``.qmd`` document referencing ``figures`` (name -> path)."""
    title = report.title or report.suite
    front = [
        "---",
        f'title: "{title}"',
        f'subtitle: "catopt benchmark — git {report.env.get("git_sha", "?")}"',
        "format:",
        "  html: { toc: true, embed-resources: true }",
        "  pdf: { toc: false, geometry: margin=2cm }",
        "execute: { enabled: false }",
        "---",
        "",
    ]
    body = render_markdown(
        report, speedup_vs=report.provenance.get("speedup_vs")
    )
    lines = [*front, body]
    if figures:
        lines += ["", "## Figures", ""]
        for name, path in figures.items():
            lines += [f"![{name}]({path})", ""]
    return "\n".join(lines)
