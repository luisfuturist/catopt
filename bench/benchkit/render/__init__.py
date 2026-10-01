"""Report renderers — one canonical ``Report``, several surfaces."""

from __future__ import annotations

__all__ = [
    "render_markdown",
    "render_page",
    "render_plots",
    "render_quarto",
    "render_slidev",
]


def __getattr__(name: str):
    if name == "render_markdown":
        from bench.benchkit.render.markdown import render_markdown

        return render_markdown
    if name == "render_page":
        from bench.benchkit.render.html import render_page

        return render_page
    if name == "render_plots":
        from bench.benchkit.render.plots import render_plots

        return render_plots
    if name == "render_quarto":
        from bench.benchkit.render.quarto import render_quarto

        return render_quarto
    if name == "render_slidev":
        from bench.benchkit.render.slidev import render_slidev

        return render_slidev
    raise AttributeError(name)
