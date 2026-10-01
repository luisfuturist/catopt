"""Append-only run ledger + baseline comparison.

Every run appends one record per suite (env + cells + findings) to
``bench/results/ledger.jsonl``.  ``load`` reads it back as a polars
frame; ``latest``/``baseline`` pick records for the ``compare`` and
``gate`` commands.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.benchkit.report import Report

DEFAULT_LEDGER = Path("bench/results/ledger.jsonl")


def append(report: Report, path: str | Path = DEFAULT_LEDGER) -> Path:
    """Append a report to the ledger; returns the ledger path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = report.to_dict()
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path


def load(path: str | Path = DEFAULT_LEDGER):
    """Read the ledger as a ``polars.DataFrame`` (one row per run)."""
    import polars as pl

    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def latest(report_suite: str, path: str | Path = DEFAULT_LEDGER):
    """The most recent ledger record for ``report_suite`` (or ``None``)."""
    df = load(path)
    if df.is_empty() or "suite" not in df.columns:
        return None
    hits = df.filter(df["suite"] == report_suite)
    return None if hits.is_empty() else hits.tail(1).to_dicts()[0]


def write_baseline(report: Report, outdir: str | Path) -> Path:
    """Pin a report as the golden baseline for its suite."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{report.suite}.json"
    report.to_json(path)
    return path
