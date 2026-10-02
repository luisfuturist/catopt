"""The carrier package's registration with the neutral orchestrator.

:func:`build_machinery` returns the
:class:`~catopt_orchestrator.carriers.CarrierMachinery` the
orchestrator's carrier-aware passes consult.  Every callable defers its
imports, so registration is import-safe — ``import catopt_carriers``
still loads no tensor library, and the orchestrator never imports this
package.

Each thunk is guarded by :func:`_optional`: when a carrier family's
module is unavailable (a partial install, or a torch-free process that
imported the package), the thunk contributes its carrier-free default
rather than raising — the partial-install accommodation the
orchestrator's old lazy imports used to provide, now owned by the side
that actually knows about carriers.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from catopt_core.laws import RuleSet
from catopt_orchestrator.carriers import CarrierMachinery

__all__ = ["build_machinery"]

_NO_RULES = RuleSet("carriers", ())


def _optional(default: Any) -> Callable:
    """Guard a thunk: yield *default* when its carrier module is absent."""

    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except ModuleNotFoundError:
                return default

        return inner

    return wrap


@_optional({})
def _plans() -> dict[str, Any]:
    """Return the batched plan builders, keyed by carrier-apply root op."""
    from catopt_carriers.om_lower import build_om_plan
    from catopt_carriers.omd_lower import build_omd_plan
    from catopt_carriers.scan_lower import build_scan_plan

    return {
        "apply": build_scan_plan,
        "applyd": build_scan_plan,
        "om_apply": build_om_plan,
        "omd_apply": build_omd_plan,
        "omd_applym": build_omd_plan,
    }


@_optional([])
def _lifts(eg: Any) -> list:
    """Return the non-local carrier lifts over *eg*."""
    from catopt_carriers.trace_lift import (
        lift_scan_to_applyd,
        lift_scan_to_trace,
    )
    from catopt_carriers.xcarrier import (
        gather_apply_stack,
        gather_applyd_stack,
        omd_tree_lift,
    )

    return (
        lift_scan_to_applyd(eg)
        + lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )


@_optional([])
def _regime_lifts(eg: Any) -> list:
    """Return the frontier's lift set — as ``_lifts`` but without applyd.

    The regime frontier historically omits ``lift_scan_to_applyd``
    (only the search pipeline runs it); the split is preserved so the
    two paths stay byte-identical.
    """
    from catopt_carriers.trace_lift import lift_scan_to_trace
    from catopt_carriers.xcarrier import (
        gather_apply_stack,
        gather_applyd_stack,
        omd_tree_lift,
    )

    return (
        lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )


@_optional(_NO_RULES)
def _rules() -> Any:
    """Return the carrier law families added to ``core.CARRIER_SEARCH``."""
    from catopt_carriers.decode_geom import DECODE_GEOM_RULES
    from catopt_carriers.decode_laws import DECODE_RULES
    from catopt_carriers.om import OM_RULES
    from catopt_carriers.trace import TRACE_RULES

    return DECODE_RULES + DECODE_GEOM_RULES + OM_RULES + TRACE_RULES


@_optional(_NO_RULES)
def _xc_rules() -> Any:
    """Return the cross-carrier seam rule set."""
    from catopt_carriers.xcarrier import XC_RULES

    return XC_RULES


@_optional(_NO_RULES)
def _preset() -> Any:
    """Return the composed ``CARRIERS`` preset."""
    import catopt_carriers

    return catopt_carriers.CARRIERS


@_optional(False)
def _is_scan_root(term: Any) -> bool:
    """Probe the scan carrier's root."""
    from catopt_carriers.scan_lower import is_scan_apply_term

    return is_scan_apply_term(term)


@_optional(False)
def _is_om_root(term: Any) -> bool:
    """Probe the om carrier's root."""
    from catopt_carriers.om_lower import is_om_apply_term

    return is_om_apply_term(term)


@_optional(None)
def _scan_plan(term: Any) -> Any:
    """Build the batched-scan plan."""
    from catopt_carriers.scan_lower import build_scan_plan

    return build_scan_plan(term)


@_optional(None)
def _om_plan(term: Any) -> Any:
    """Build the batched-om plan."""
    from catopt_carriers.om_lower import build_om_plan

    return build_om_plan(term)


def build_machinery() -> CarrierMachinery:
    """Assemble the carrier machinery value (import-safe)."""
    return CarrierMachinery(
        plans=_plans,
        lifts=_lifts,
        regime_lifts=_regime_lifts,
        rules=_rules,
        xc_rules=_xc_rules,
        preset=_preset,
        is_scan_root=_is_scan_root,
        is_om_root=_is_om_root,
        scan_plan=_scan_plan,
        om_plan=_om_plan,
    )
