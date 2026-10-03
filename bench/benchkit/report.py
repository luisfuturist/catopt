"""``Report`` — a suite's cells + findings + provenance, and its renderers.

A ``Report`` is the single canonical object a suite produces.  Every
presentation surface (JSON, Markdown, the HTML dashboard, the Quarto
document, the Slidev assets) is rendered *from* it, so the numbers can
never disagree between formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bench.benchkit.env import collect_env
from bench.benchkit.model import Cell, Finding
from bench.benchkit.stats import jsonable, to_dataframe

SCHEMA_VERSION = 1


@dataclass
class Report:
    """A named suite's results + conclusions + environment provenance."""

    suite: str
    cells: list[Cell]
    env: dict = field(default_factory=collect_env)
    title: str = ""
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # -- serialization -----------------------------------------------------

    def _cell_record(self, cell: Cell) -> dict:
        return {
            "name": cell.case.name,
            "params": dict(cell.case.params),
            "variants": [
                {"name": v.name, "flops": v.flops, "note": v.note}
                for v in cell.case.variants
            ],
            "median_s": dict(cell.medians),
            "iqr_s": dict(cell.iqr),
            "aux": jsonable(cell.aux),
        }

    def to_dict(self) -> dict:
        """The canonical, schema-versioned payload."""
        return {
            "schema": SCHEMA_VERSION,
            "suite": self.suite,
            "title": self.title,
            "summary": self.summary,
            "env": jsonable(self.env),
            "provenance": jsonable(self.provenance),
            "units": {"median_s": "seconds", "iqr_s": "seconds"},
            "findings": [
                {
                    "claim": f.claim,
                    "verdict": f.verdict.value,
                    "headline": f.headline,
                    "metric": f.metric,
                    "value": f.value,
                    "evidence": jsonable(f.evidence),
                }
                for f in self.findings
            ],
            "cells": [self._cell_record(c) for c in self.cells],
        }

    def to_json(self, path: str | Path) -> None:
        """Write the canonical JSON payload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n"
        )

    def to_baseline_dict(self) -> dict:
        """The baseline projection — the payload minus the live spread.

        A pinned baseline is a *regression reference*: ``compare`` /
        ``gate`` read ``median_s`` only, so the recorded spread
        (``iqr_s``) buys the reference nothing.  It is also the one
        field the timing contract can move under a frozen baseline —
        :mod:`catopt_core.timing` defines the IQR as
        :func:`statistics.quantiles` (exclusive), while the oldest
        baselines were recorded under torch's inclusive quantiles, so
        their stored ``iqr_s`` no longer matches what the harness
        reports.  Re-pinning would fix the spread but move the medians
        too (a real re-baseline); instead a baseline records only the
        quantity it gates on.  The spread stays in every *live* report
        (:meth:`to_dict`), where the renderers show ``median ± IQR``.
        """
        payload = self.to_dict()
        payload["units"] = {"median_s": "seconds"}
        for cell in payload["cells"]:
            cell.pop("iqr_s", None)
        return payload

    def to_baseline(self, path: str | Path) -> None:
        """Write the baseline projection (see ``to_baseline_dict``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_baseline_dict(), indent=2, default=str)
            + "\n"
        )

    def to_dataframe(self):
        """Flatten cells to a ``polars.DataFrame``."""
        return to_dataframe(self.cells)

    @classmethod
    def from_json(cls, path: str | Path) -> Report:
        """Reconstruct a report from its canonical JSON payload.

        Variants carry no callables after timing (the runner releases
        them), so the reconstructed ``Variant`` stmts are no-ops — the
        renderers only read names, notes and FLOPs.
        """
        from bench.benchkit.model import (
            Case,
            Cell,
            Finding,
            Variant,
            Verdict,
        )

        payload = json.loads(Path(path).read_text())
        cells = []
        for rec in payload["cells"]:
            variants = [
                Variant(
                    v["name"],
                    lambda: None,
                    v.get("flops"),
                    v.get("note", ""),
                )
                for v in rec["variants"]
            ]
            case = Case(
                rec["name"], rec["params"], variants, rec.get("aux", {})
            )
            cells.append(
                Cell(
                    case,
                    rec["median_s"],
                    rec.get("iqr_s", {}),
                    rec.get("aux", {}),
                )
            )
        return cls(
            suite=payload["suite"],
            cells=cells,
            env=payload.get("env", {}),
            title=payload.get("title", ""),
            summary=payload.get("summary", ""),
            findings=[
                Finding(
                    claim=f["claim"],
                    verdict=Verdict(f["verdict"]),
                    headline=f.get("headline", ""),
                    metric=f.get("metric"),
                    value=f.get("value"),
                    evidence=f.get("evidence", {}),
                )
                for f in payload.get("findings", [])
            ],
            provenance=payload.get("provenance", {}),
        )

    # -- renderers (lazy imports keep the framework import-light) ----------

    def to_markdown(
        self,
        path: str | Path,
        speedup_vs: str | None = None,
        include_aux: bool = True,
    ) -> None:
        """Readable GitHub-flavored Markdown (narrow tables, aux in
        collapsible ``<details>``)."""
        from bench.benchkit.render.markdown import render_markdown

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_markdown(
                self, speedup_vs=speedup_vs, include_aux=include_aux
            )
        )

    def to_plots(
        self,
        outdir: str | Path,
        x_param: str | None = None,
        speedup_vs: str | None = None,
        stem: str | None = None,
        static: bool = True,
    ) -> list[str]:
        """Bar grids (+ a speedup curve when given) as plotly figures.

        Interactive ``.html`` is always written; static ``.svg`` is
        written too when ``static`` (used by Quarto / Slidev).
        """
        from bench.benchkit.render.plots import render_plots

        return render_plots(
            self,
            outdir,
            x_param=x_param,
            speedup_vs=speedup_vs,
            stem=stem,
            static=static,
        )

    def to_html(
        self, path: str | Path, self_contained: bool = True
    ) -> None:
        """A single-file HTML report page."""
        from bench.benchkit.render.html import render_page

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_page(self, self_contained=self_contained)
        )

    def to_quarto(
        self, path: str | Path, figures: dict | None = None
    ) -> None:
        """A ``.qmd`` source (rendered by the Quarto CLI)."""
        from bench.benchkit.render.quarto import render_quarto

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_quarto(self, figures or {}))

    def to_slidev(
        self, outdir: str | Path, figures: dict | None = None
    ) -> dict[str, str]:
        """Emit Slidev assets (figures + a headline JSON) under ``outdir``."""
        from bench.benchkit.render.slidev import render_slidev

        return render_slidev(self, outdir, figures or {})
