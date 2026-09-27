"""Criterion-based cost blends — first-class selection axes.

``optimize_model`` extracts the member of the semantic equivalence
class [G] that minimizes a ``CostFn``.  Users have always been able
to hand-compose weighted objectives as lambdas over the cost models
in :mod:`catopt_core.cost`; :func:`criteria_cost` makes the standard
axes first-class:

    optimize_model(m, x, criteria={"latency": 1.0, "memory": 0.25})

is a normalised, inspectable spelling of the lambda it replaces —
and ``stats["criteria"]`` records the blend extraction priced.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from catopt_core.cost import (
    depth_cost_for,
    executor_cost_for,
    flops_cost,
    fused_cost_for,
    param_bytes_cost_for,
)

if TYPE_CHECKING:
    from catopt_core.ports import CostFn

#: The named selection axes :func:`criteria_cost` can blend, in
#: documentation order.  Each maps to one cost model from
#: :mod:`catopt_core.cost` — see :func:`_axis_fns`.
AXES: tuple[str, ...] = (
    "latency",
    "memory",
    "flops",
    "depth",
    "compiled",
)

#: ``criteria=None`` reproduces the shipped default model
#: (:func:`executor_cost_for` with ``lowering="generic"``).
_DEFAULT: dict[str, float] = {"latency": 1.0}


def _axis_fns(profile: Any) -> dict[str, CostFn]:
    """The cost model behind each axis name.

    * ``"latency"`` — the generic executor's delivered price:
      :func:`executor_cost_for` ``(lowering="generic")`` — per-op
      roofline + per-node dispatch overhead, nanoseconds.
    * ``"memory"`` — :func:`param_bytes_cost_for`: stored parameter
      values — the storage axis that prefers weight sharing,
      deduplicated heads and factorised weights.
    * ``"flops"`` — :func:`flops_cost`: estimated FLOP count.
    * ``"depth"`` — :func:`depth_cost_for`: critical-path roofline
      latency; rewards parallel structure a sequential chain hides.
    * ``"compiled"`` — :func:`fused_cost_for`: Inductor-style
      fusion-region pricing — the axis that sees what
      ``compile=True`` buys.
    """
    return {
        "latency": executor_cost_for(profile, lowering="generic"),
        "memory": param_bytes_cost_for(),
        "flops": flops_cost,
        "depth": depth_cost_for(profile),
        "compiled": fused_cost_for(profile),
    }


def criteria_cost(
    criteria: dict[str, float] | None = None,
    profile: Any = None,
) -> CostFn:
    """Blend named cost axes into one extraction ``CostFn``.

    ``criteria`` maps an axis name to a non-negative weight; the axes
    are :data:`AXES` (see :func:`_axis_fns` for what each prices).
    Weights are NORMALISED to sum 1 — the blend is convex, so only
    relative weights matter: ``{"latency": 2}`` is exactly
    ``{"latency": 1}``.  The axes' units are heterogeneous
    (nanoseconds, FLOPs, stored values), so each weight is both a
    preference and a unit-conversion factor — the blend's own unit is
    the weighted mix.  ``criteria=None`` is ``{"latency": 1.0}``,
    i.e. the default extraction model unchanged.  Unknown axes and
    negative / non-finite / non-numeric weights raise
    ``ValueError``; so does a blend with no positive weight.

    Additivity — safe inside ``extract_best``: a float sum of
    additive-per-node cost models is still additive, so the
    subtractive local-cost recovery ``local = c(t) - sum(children)``
    is exact for the ``latency`` / ``memory`` / ``flops`` axes
    (``memory`` contributes the same DAG-indexed storage billing
    ``param_bytes_cost`` applies inside extraction).  ``depth`` is
    max-composed and ``compiled`` a whole-DAG region partition —
    neither decomposes per node, so weighted in their contribution is
    a non-negative clamped approximation, the same caveat
    :func:`fused_cost_for` documents.  Prefer them for frontier
    reporting or small blend weights.

    ``profile`` calibrates the time-priced axes — a
    ``catopt_optimize.calibrate.TargetProfile`` or any object/dict
    with ``tflops`` / ``gbps`` / ``launch_us`` (and optionally
    ``dispatch_us``); ``None`` uses the built-in profile.

    Returns a :class:`_CriteriaBlend` — a ``CostFn``-conforming
    callable carrying the ``criteria`` / ``profile`` /
    ``charges_param_only`` markers extraction and reporting read
    (see its docstring).
    """
    blend = dict(_DEFAULT if criteria is None else criteria)
    unknown = sorted(set(blend) - set(AXES))
    if unknown:
        raise ValueError(
            f"unknown criteria axes {unknown} — "
            f"expected a subset of {sorted(AXES)}"
        )
    parts: dict[str, float] = {}
    total_w = 0.0
    for axis, w in blend.items():
        if (
            not isinstance(w, (int, float))
            or not math.isfinite(w)
            or w < 0
        ):
            raise ValueError(
                f"criteria[{axis!r}] must be a finite, "
                f"non-negative weight — got {w!r}"
            )
        if w > 0:
            parts[axis] = float(w)
            total_w += float(w)
    if not parts:
        raise ValueError(
            "criteria must give at least one axis a positive weight"
        )
    for axis in parts:
        parts[axis] /= total_w
    return _CriteriaBlend(_axis_fns(profile), parts, profile)


class _CriteriaBlend:
    """A normalised weighted blend of per-axis cost models.

    Conforms to the :class:`~catopt_core.ports.CostFn` port through
    ``__call__(term, memo=None)`` — a class rather than a closure so
    the reporting/billing markers are real typed attributes (and
    ``inspect.signature`` still finds ``memo`` to thread in).

    Attributes read off the callable (mirroring the
    ``param_bytes_cost`` / ``executor_cost_for`` conventions):

    * ``criteria`` — the normalised ``{axis: weight}`` blend actually
      priced; :func:`optimize_model` records it into
      ``stats["criteria"]``.
    * ``profile`` — the bound target profile; its presence also
      enables extraction's roofline-memo pre-seeding.
    * ``charges_param_only`` — True when a ``memory`` axis is priced:
      compile-time-folded subtrees still store their leaves' values,
      so the param-only discount must not zero them.  The marker is
      per cost-fn — a memory blend bills param-only subtrees on EVERY
      axis, not just storage (documented approximation).
    """

    def __init__(
        self,
        fns: dict[str, CostFn],
        weights: dict[str, float],
        profile: Any,
    ) -> None:
        self._fns = fns
        self.criteria = dict(weights)
        self.profile = profile
        self.charges_param_only = "memory" in weights
        self.__name__ = "criteria_cost"

    def __call__(self, term: Any, memo: dict | None = None) -> float:
        m = {} if memo is None else memo
        out = 0.0
        for axis, w in self.criteria.items():
            out += w * self._fns[axis](term, m)
        return out
