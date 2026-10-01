"""Bench data model — variants, cells, findings, verdicts.

A suite builds ``Case`` objects (one per sweep cell); the ``Runner``
times their ``Variant`` callables into ``Cell``s; the suite states its
conclusions as ``Finding``s; everything is packed into a ``Report``
(``bench.benchkit.report``) that the renderers consume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    """The honest conclusion a suite draws from its cells."""

    WIN = "win"
    PARITY = "parity"
    REGRESSION = "regression"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"

    @property
    def badge(self) -> str:
        """Short uppercase tag for tables and dashboards."""
        return _BADGES[self]


_BADGES = {
    Verdict.WIN: "WIN",
    Verdict.PARITY: "PARITY",
    Verdict.REGRESSION: "REGRESSION",
    Verdict.NEGATIVE: "NEGATIVE",
    Verdict.INCONCLUSIVE: "INCONCLUSIVE",
}


@dataclass
class Variant:
    """One timed implementation inside a ``Case`` cell."""

    name: str  # "eager" | "inductor" | "catopt" | ...
    stmt: Callable[[], object]  # one forward call, CPU/GPU-synced
    flops: float | None = None  # analytic runtime-FLOPs, for report
    note: str = ""  # provenance notes (e.g. "weight-folded")


@dataclass
class Case:
    """One sweep cell: display name, coordinates, variants to time."""

    name: str  # display name
    params: dict  # sweep coords — ordered cols in report
    variants: list[Variant]
    aux: dict = field(default_factory=dict)  # verify diffs, proof notes


@dataclass
class Cell:
    """Timing outcome for one ``Case``."""

    case: Case
    medians: dict[str, float]  # variant -> median seconds
    iqr: dict[str, float]  # interquartile range seconds
    aux: dict = field(default_factory=dict)


@dataclass
class Finding:
    """A suite's headline conclusion, renderable without the console.

    ``claim`` states what was tested; ``verdict`` is the honest
    outcome; ``headline`` is the one-line human summary; ``metric`` /
    ``value`` / ``evidence`` carry the supporting numbers so a renderer
    can build a KPI card or a proof excerpt.
    """

    claim: str
    verdict: Verdict
    headline: str = ""
    metric: str | None = None
    value: float | None = None
    evidence: dict = field(default_factory=dict)
