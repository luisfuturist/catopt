"""benchkit — the catopt benchmark harness.

Suites build ``Case`` objects (one per sweep cell) with ``Variant``
callables, time them through ``Runner`` (torch ``Timer`` medians + IQR),
state their conclusions as ``Finding``s, and pack everything into a
``Report``.  A ``Report`` renders to JSON, Markdown, a single-file HTML
dashboard, a Quarto ``.qmd``, and Slidev assets — all from the same
canonical data.

    from bench.benchkit import Case, Finding, Report, Runner, Verdict

    runner = Runner(device="cpu", warmup=5, min_run_time=0.2)
    cells = runner.run(cases)
    report = Report(
        suite="my_bench",
        title="My bench",
        summary="Does X beat Y?",
        findings=[Finding(claim="X beats Y", verdict=Verdict.WIN)],
        cells=cells,
    )
    report.to_markdown("results/my_bench.md", speedup_vs="eager")
    report.to_html("results/my_bench.html")
"""

from __future__ import annotations

from bench.benchkit.env import collect_env
from bench.benchkit.model import (
    Case,
    Cell,
    Finding,
    Variant,
    Verdict,
)
from bench.benchkit.report import SCHEMA_VERSION, Report
from bench.benchkit.runner import Runner

__all__ = [
    "SCHEMA_VERSION",
    "Case",
    "Cell",
    "Finding",
    "Report",
    "Runner",
    "Variant",
    "Verdict",
    "collect_env",
]
