"""Typed report layer and the uniform verification gate.

Plan 0001 phase 3a/3b: the optimizer's stats dicts stay plain dicts at
the public boundary — callers and tests index them — but every schema
is now code.  ``OptReport`` / ``CompositionalReport`` / ``BlockReport``
give typed access with an exact ``to_dict`` round-trip, and the three
near-duplicate diff+tolerance checks (optimize_model's verbose verify,
optimize_compositional's per-block and end-to-end gates, the bench
files' ``rel_diff`` helpers) collapse into ``verify_equiv`` /
``verify_module``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BlockReport",
    "CompositionalReport",
    "OptReport",
    "SaturationStats",
    "VerifyReport",
    "rel_diff",
    "verify_equiv",
    "verify_module",
]

#: Denominator floor of the historical rel-diff metric — every call
#: site this module replaces used ``+ 1e-8``; it is part of the gate's
#: bit-for-bit semantics, not a tunable.
_REL_FLOOR = 1e-8


# ---------------------------------------------------------------------------
#  Uniform verify gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """Result of one equivalence check between two output tensors."""

    max_abs: float  # max|ref - out|
    max_rel: float  # max_abs / (max|ref| + _REL_FLOOR)
    passed: bool


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """The historical relative-diff metric, identical everywhere it was
    inlined: ``max|a - b| / (max|a| + 1e-8)``."""
    return (a - b).abs().max().item() / (
        a.abs().max().item() + _REL_FLOOR
    )


def verify_equiv(
    ref_out: torch.Tensor,
    out: torch.Tensor,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> VerifyReport:
    """Compare two already-computed outputs.

    Gate semantics (deliberately NOT ``torch.allclose``'s elementwise
    ``atol + rtol * |ref|`` — this is the metric every historical call
    site used):

    * ``max_rel = max|ref - out| / (max|ref| + 1e-8)``
    * ``passed = max_rel < rtol`` and, when ``atol`` is given,
      additionally ``max_abs <= atol``.
    """
    max_abs = (ref_out - out).abs().max().item()
    max_rel = max_abs / (ref_out.abs().max().item() + _REL_FLOOR)
    passed = bool(max_rel < rtol) and (
        atol is None or bool(max_abs <= atol)
    )
    return VerifyReport(max_abs=max_abs, max_rel=max_rel, passed=passed)


def verify_module(
    ref_mod: torch.nn.Module,
    opt_mod: torch.nn.Module,
    inputs: torch.Tensor | tuple,
    rtol: float = 1e-4,
    atol: float | None = None,
) -> VerifyReport:
    """Run ``ref_mod`` and ``opt_mod`` on ``inputs`` (a tensor or an
    args tuple) under ``no_grad`` and compare with
    :func:`verify_equiv`.  Tensor inputs are cloned so a forward that
    mutates its arguments cannot corrupt the comparison."""
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    ref_mod.eval()
    opt_mod.eval()
    with torch.no_grad():
        ref = ref_mod(
            *[
                a.clone() if isinstance(a, torch.Tensor) else a
                for a in args
            ]
        )
        out = opt_mod(
            *[
                a.clone() if isinstance(a, torch.Tensor) else a
                for a in args
            ]
        )
    return verify_equiv(ref, out, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
#  Typed stats reports — exact dict round-trip
# ---------------------------------------------------------------------------


def _plain(v: Any) -> Any:
    """Serialize a field value: typed reports back to dicts, anything
    else (dicts, lists, offer objects) passes through untouched."""
    if isinstance(v, (OptReport, BlockReport)):
        return v.to_dict()
    return v


@dataclass
class SaturationStats:
    """The ``EGraph.run`` payload — always present on a stats dict that
    came out of equality saturation.  Exposed on ``OptReport`` as the
    ``saturation_stats`` view; stored flat so ``to_dict`` reproduces the
    original keys exactly."""

    iterations: int | None = None
    n_enodes: int | None = None
    n_classes: int | None = None
    n_proof_edges: int | None = None
    truncation_level: int | None = None
    rule_budgets: dict[str, int] | None = None
    budget_suspended: list[str] | None = None


@dataclass
class OptReport:
    """Typed view of the ``stats`` dict ``optimize_model`` returns.

    Field names equal the dict keys.  Saturation keys are always
    populated on a real stats dict; the path-dependent extras
    (``pairing_groups`` only when the pairing pass found groups,
    ``nonlocal_lifts`` only when a lift fired, ``eps_offers`` only with
    ``eps_rtol``, ``paired_extract``/``causal_specialized`` only when
    the corresponding specialization won) stay ``None`` when absent —
    and ``to_dict`` omits them, so
    ``OptReport.from_stats(stats).to_dict() == stats`` exactly.
    ``extra`` captures keys this schema does not model (forward-compat)
    and re-emits them verbatim.
    """

    # -- EGraph.run payload ------------------------------------------
    iterations: int | None = None
    n_enodes: int | None = None
    n_classes: int | None = None
    n_proof_edges: int | None = None
    truncation_level: int | None = None
    rule_budgets: dict[str, int] | None = None
    budget_suspended: list[str] | None = None
    # -- optimize_model additions ------------------------------------
    rule_fires: dict[str, int] | None = None
    pairing_groups: int | None = None
    nonlocal_lifts: int | None = None
    eps_offers: list | None = None
    paired_extract: bool | None = None
    causal_specialized: bool | None = None
    # -- unmodelled keys, re-emitted verbatim -------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def saturation_stats(self) -> SaturationStats:
        """The ``EGraph.run`` keys grouped as one typed view."""
        return SaturationStats(
            iterations=self.iterations,
            n_enodes=self.n_enodes,
            n_classes=self.n_classes,
            n_proof_edges=self.n_proof_edges,
            truncation_level=self.truncation_level,
            rule_budgets=self.rule_budgets,
            budget_suspended=self.budget_suspended,
        )

    @classmethod
    def from_stats(cls, stats: dict[str, Any]) -> OptReport:
        """Build the typed report from a stats dict.  Field names equal
        dict keys, so extraction is mechanical; ``blocks`` values (only
        on :class:`CompositionalReport`) are lifted to
        :class:`BlockReport`."""
        names = {f.name for f in dataclasses.fields(cls)} - {"extra"}
        kwargs = {n: stats.get(n) for n in names}
        if kwargs.get("blocks") is not None:
            kwargs["blocks"] = {
                n: BlockReport.from_dict(e, name=n)
                for n, e in kwargs["blocks"].items()
            }
        kwargs["extra"] = {
            k: v for k, v in stats.items() if k not in names
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the loose stats dict.  ``None`` fields are
        omitted, so the round-trip reproduces the input dict's exact
        key set."""
        out = dict(self.extra)
        for f in dataclasses.fields(self):
            if f.name == "extra":
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            if f.name == "blocks" and isinstance(v, dict):
                out[f.name] = {n: _plain(b) for n, b in v.items()}
            else:
                out[f.name] = _plain(v)
        return out


@dataclass
class BlockReport:
    """Typed view of one ``stats["blocks"][name]`` entry.

    ``status`` is one of ``"optimized"`` / ``"failed"`` / ``"skipped"``
    / ``"not_executed"``; which of the remaining keys an entry carries
    depends on it (today: ``not_executed`` has only ``status``;
    ``skipped`` adds ``reason``; ``failed`` adds ``error`` — plus
    ``reason == "resource_limit"`` and a ``rel_diff`` when it failed at
    verification — and ``time_s``; ``optimized`` adds ``rel_diff``,
    ``stats``, ``param_report``, ``time_s``).  ``None`` fields are
    omitted by :meth:`to_dict`, reproducing each shape exactly.
    ``name`` is the dotted block path — the dict key, not an entry
    key — so it is typed-only and never serialized into the entry.
    """

    name: str
    status: str
    rel_diff: float | None = None
    time_s: float | None = None
    reason: str | None = None
    error: str | None = None
    stats: OptReport | dict | None = None
    param_report: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, entry: dict[str, Any], name: str = ""
    ) -> BlockReport:
        names = {f.name for f in dataclasses.fields(cls)} - {
            "name",
            "extra",
        }
        kwargs = {n: entry.get(n) for n in names}
        if isinstance(kwargs.get("stats"), dict):
            kwargs["stats"] = OptReport.from_stats(kwargs["stats"])
        kwargs["extra"] = {
            k: v for k, v in entry.items() if k not in names
        }
        return cls(name=name, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        for f in dataclasses.fields(self):
            if f.name in ("name", "status", "extra"):
                continue
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = _plain(v)
        out.update(self.extra)
        return out


@dataclass
class CompositionalReport(OptReport):
    """Typed view of the ``stats`` dict ``optimize_compositional``
    returns — an ``OptReport`` extended with the driver keys.  The
    saturation fields stay ``None`` at top level (per-block saturation
    lives inside ``blocks[name].stats``) and are omitted on
    ``to_dict``, so the round-trip is exact."""

    compositional: bool | None = None
    n_blocks: int | None = None
    n_optimized: int | None = None
    n_failed: int | None = None
    n_skipped: int | None = None
    blocks: dict[str, BlockReport] | None = None
    in_place: bool | None = None
    param_report: dict | None = None
    end_to_end: dict | None = None
    wall_time_s: float | None = None
