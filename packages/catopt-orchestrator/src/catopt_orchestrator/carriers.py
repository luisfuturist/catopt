"""Carrier machinery — the orchestrator's carrier registration seam.

The orchestrator's carrier-aware passes — the batched plan builders,
the non-local lifts, the carrier rule sets and the carrier-root probes
— are supplied by whichever backend provides carriers, through
:func:`register_carriers`.  Nothing here imports a backend: the
registry holds the thunks the carrier package pushes in, so
``import catopt_orchestrator`` stays torch-free and the orchestrator
never imports ``catopt_carriers``.  Without a registration the passes
degrade to their carrier-free defaults — exactly the partial-install
behaviour the lazy imports used to provide.

The edge inversion matters for the dependency graph.  Previously the
orchestrator reached *down* into the torch-coupled carrier package,
which — with ``catopt_torch`` importing the orchestrator and the
carriers importing ``catopt_torch`` — closed a three-package cycle
``torch -> orchestrator -> carriers -> torch``.  Registration replaces
that with ``carriers -> orchestrator``, leaving the graph acyclic:
the orchestrator sits above the ``torch``/``carriers`` backend family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CarrierMachinery", "get_carriers", "register_carriers"]


@dataclass(frozen=True)
class CarrierMachinery:
    """The carrier backend's contribution to the neutral passes.

    Every field is a thunk so the carrier package's torch-coupled
    modules resolve only when a pass actually runs — importing
    ``catopt_carriers`` registers this value without loading a tensor
    library.

    * ``plans`` — ``() -> {root_op: plan_builder}``; the batched plan
      builders keyed by the carrier-apply root op.
    * ``lifts`` — ``(eg) -> offers``; the non-local carrier lifts the
      search pipeline runs.
    * ``regime_lifts`` — ``(eg) -> offers``; the frontier's lift set,
      which historically omits ``lift_scan_to_applyd`` (the pipeline
      runs it) — kept distinct so the two paths stay byte-identical.
    * ``rules`` — ``() -> RuleSet``; the carrier law families the
      carrier-search preset adds on top of ``core.CARRIER_SEARCH``.
    * ``xc_rules`` — ``() -> RuleSet``; the cross-carrier seam set.
    * ``preset`` — ``() -> RuleSet``; the composed ``CARRIERS`` preset
      the pipeline default adds on top of ``core.laws.DEFAULT``.
    * ``is_scan_root`` / ``is_om_root`` — ``(term) -> bool``; the
      carrier-root probes ``"auto"`` executor resolution consults.
    * ``scan_plan`` / ``om_plan`` — ``(term) -> plan | None``; the
      single-carrier plan builders the regime frontier prices with.
    """

    plans: Callable[[], dict[str, Callable]]
    lifts: Callable[[Any], list]
    regime_lifts: Callable[[Any], list]
    rules: Callable[[], Any]
    xc_rules: Callable[[], Any]
    preset: Callable[[], Any]
    is_scan_root: Callable[[Any], bool]
    is_om_root: Callable[[Any], bool]
    scan_plan: Callable[[Any], Any]
    om_plan: Callable[[Any], Any]


_MACHINERY: CarrierMachinery | None = None


def register_carriers(machinery: CarrierMachinery) -> None:
    """Install *machinery* — called by the carrier package at import."""
    global _MACHINERY
    _MACHINERY = machinery


def get_carriers() -> CarrierMachinery | None:
    """Return the registered machinery, or ``None`` without carriers."""
    return _MACHINERY
