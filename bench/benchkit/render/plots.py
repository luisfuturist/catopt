"""Plot renderer — interactive plotly figures, optional static SVG.

One grouped bar chart per sweep-coordinate grid (median ms per variant,
IQR error bars, log-y when the spread exceeds 50x) and, when given an
``x_param`` axis and a ``speedup_vs`` baseline, a speedup curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bench.benchkit.stats import slug, speedup, variant_names

if TYPE_CHECKING:
    from bench.benchkit.report import Report


def _bar_grid(cells, title: str):
    import plotly.graph_objects as go

    names = variant_names(cells)
    labels = []
    for c in cells:
        coords = ",".join(f"{k}={v}" for k, v in c.case.params.items())
        labels.append(
            f"{c.case.name}\n{coords}" if coords else c.case.name
        )
    fig = go.Figure()
    for name in names:
        xs, ys, es = [], [], []
        for i, c in enumerate(cells):
            if name not in c.medians:
                continue
            xs.append(labels[i])
            ys.append(c.medians[name] * 1e3)
            es.append(c.iqr.get(name, 0.0) * 1e3)
        fig.add_bar(
            name=name,
            x=xs,
            y=ys,
            error_y={"type": "data", "array": es, "visible": True},
        )
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="median ms/call (IQR whiskers)",
        legend_title_text="variant",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 60, "b": 80},
    )
    meds = [m for c in cells for m in c.medians.values() if m > 0]
    if meds and max(meds) / min(meds) > 50:
        fig.update_yaxes(type="log")
    return fig


def _speedup_fig(cells, x_param: str, baseline: str, title: str):
    import plotly.graph_objects as go

    usable = [c for c in cells if x_param in c.case.params]

    def xval(c):
        v = c.case.params[x_param]
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    usable.sort(key=xval)
    fig = go.Figure()
    for name in variant_names(usable):
        if name == baseline:
            continue
        xs, ys = [], []
        for c in usable:
            s = speedup(c, name, baseline)
            if s is None:
                continue
            xs.append(str(c.case.params[x_param]))
            ys.append(s)
        if xs:
            fig.add_scatter(name=name, x=xs, y=ys, mode="lines+markers")
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=title,
        xaxis_title=x_param,
        yaxis_title=f"speedup vs {baseline} (>1 = faster)",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )
    return fig


def _emit(fig, outdir: Path, stem: str, static: bool) -> list[str]:
    paths = [str(outdir / f"{stem}.html")]
    fig.write_html(paths[0], include_plotlyjs="cdn")
    if static:
        try:
            svg = outdir / f"{stem}.svg"
            fig.write_image(str(svg))
            paths.append(str(svg))
        except Exception:
            # kaleido/chromium unavailable — interactive HTML still stands.
            pass
    return paths


def render_plots(
    report: Report,
    outdir: str | Path,
    x_param: str | None = None,
    speedup_vs: str | None = None,
    stem: str | None = None,
    static: bool = True,
) -> list[str]:
    """Write plotly figures; returns the paths written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = slug(stem or report.suite)
    paths: list[str] = []

    grids: dict[tuple, list] = {}
    for cell in report.cells:
        grids.setdefault(tuple(cell.case.params.keys()), []).append(
            cell
        )

    for gi, (key, cells) in enumerate(grids.items()):
        if not variant_names(cells):
            continue
        suffix = f" ({', '.join(key)})" if key else ""
        fig = _bar_grid(
            cells, f"{report.title or report.suite}{suffix}"
        )
        paths += _emit(fig, outdir, f"{stem}_grid{gi}", static)

    if (
        x_param
        and speedup_vs
        and any(
            x_param in c.case.params and speedup_vs in c.medians
            for c in report.cells
        )
    ):
        fig = _speedup_fig(
            report.cells,
            x_param,
            speedup_vs,
            f"{report.title or report.suite}: speedup vs {speedup_vs}",
        )
        paths += _emit(
            fig, outdir, f"{stem}_speedup_vs_{slug(speedup_vs)}", static
        )
    return paths
