"""Baseline regression comparison for the ``compare`` / ``gate`` commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Regression:
    """One variant that got slower than its baseline beyond threshold."""

    case: str
    variant: str
    base_ms: float
    now_ms: float
    ratio: float


def _cells_by_name(payload: dict) -> dict[str, dict]:
    return {c["name"]: c for c in payload.get("cells", [])}


def compare_baseline(
    record: dict, baseline_path: str | Path, threshold: float = 0.05
) -> list[Regression]:
    """Return variants whose median rose by more than ``threshold``.

    Cells are matched by ``case`` name; variants by name.  A cell
    present in the baseline but missing from the run is ignored (a
    suite may legitimately shrink its sweep under ``--quick``).
    """
    baseline = json.loads(Path(baseline_path).read_text())
    base_cells = _cells_by_name(baseline)
    regressions: list[Regression] = []
    for cell in record.get("cells", []):
        base = base_cells.get(cell["name"])
        if base is None:
            continue
        for variant, now_s in cell["median_s"].items():
            base_s = base["median_s"].get(variant)
            if not base_s:
                continue
            ratio = now_s / base_s
            if ratio > 1 + threshold:
                regressions.append(
                    Regression(
                        case=cell["name"],
                        variant=variant,
                        base_ms=base_s * 1e3,
                        now_ms=now_s * 1e3,
                        ratio=ratio,
                    )
                )
    return regressions
