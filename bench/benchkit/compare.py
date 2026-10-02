"""Baseline regression comparison for the ``compare`` / ``gate`` commands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

#: Verdict keywords the registry's ``expects`` prose may advertise.
_VERDICT_WORDS = (
    "WIN",
    "PARITY",
    "NEGATIVE",
    "REGRESSION",
    "INCONCLUSIVE",
)


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
            # A ledger record read back through polars carries ``None``
            # for variants a cell did not time (e.g. under ``--quick``).
            if not base_s or not now_s:
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


def expected_verdict(expects: str) -> str | None:
    """The verdict keyword a registry ``expects`` string advertises.

    ``None`` when the prose names no verdict (e.g. a coverage map or an
    exploratory suite), in which case no consistency check applies.
    """
    for word in _VERDICT_WORDS:
        if re.search(rf"\b{word}\b", expects):
            return word
    return None


def expectation_gaps(baselines_dir: str | Path) -> list[str]:
    """Baselines whose findings contradict the registry's stated verdict.

    Advisory, not a hard gate: a suite's ``expects`` prose states the
    intended outcome, while a pinned baseline is one measurement.  A gap
    means the two disagree — the drift this harness exists to surface.
    """
    from bench import registry

    gaps: list[str] = []
    for path in sorted(Path(baselines_dir).glob("*.json")):
        try:
            spec = registry.get(path.stem)
        except KeyError:
            gaps.append(f"{path.stem}: not in the registry")
            continue
        payload = json.loads(path.read_text())
        present = {
            f["verdict"].upper() for f in payload.get("findings", [])
        }
        if not present:
            gaps.append(f"{path.stem}: baseline states no finding")
            continue
        want = expected_verdict(spec.expects)
        if want and want not in present:
            gaps.append(
                f"{path.stem}: expects {want}, baseline has "
                f"{', '.join(sorted(present))}"
            )
    return gaps
