"""Criterion-based cost blends — composable, pluggable selection axes.

``optimize_model`` extracts the member of the semantic equivalence
class [G] that minimizes a ``CostFn``.  Users have always been able
to hand-compose weighted objectives as lambdas over the cost models
in :mod:`catopt_core.cost`; the criterion machinery makes the
standard axes first-class AND user axes pluggable::

    optimize_model(m, x, criteria={"latency": 1.0, "memory": 0.25})
    optimize_model(
        m,
        x,
        criteria=LatencyCriterion() * 0.7
        + MemoryCriterion(mode="peak") * 0.3,
    )
    optimize_model(m, x, criteria=MyCriterion())

A *criterion* is any object satisfying the :class:`Criterion`
protocol — ``cost_fn(profile) -> CostFn`` plus a ``name`` label —
so a project-specific axis (a calibrated kernel table, an energy
model, a bespoke schedule estimate) drops straight into the blend
with no registration step.  :func:`criteria_cost` accepts a
``{axis: weight}`` dict (the shipped axes, :data:`AXES`), a single
criterion, a :class:`Criteria`/:class:`Blend` container, or a list
mixing criteria and ``(criterion, weight)`` pairs; ``*``/``+``
operator overloads compose weighted blends directly.
``stats["criteria"]`` records the normalised blend extraction priced.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from catopt_core.cost import (
    _INVALID_COST,
    _VIEW_OPS,
    _folds_to_param,
    depth_cost_for,
    executor_cost_for,
    flops_cost,
    fused_cost_for,
    param_bytes_cost_for,
)
from catopt_core.ir import Op, Var
from catopt_core.typing import _INVALID, _numel, _shape_of

if TYPE_CHECKING:
    from catopt_core.ports import CostFn

# ---------------------------------------------------------------------------
#  The Criterion port — pluggable selection axes
# ---------------------------------------------------------------------------
#
# The canonical protocol lives in :mod:`catopt_core.ports` (promoted
# in plan 0007 — the orchestrator's neutral contract); it is imported
# here so ``catopt_optimize.criteria.Criterion`` stays the same
# object users subclass and ``isinstance``-check against.
from catopt_core.ports import Criterion

__all__ = [
    "AXES",
    "Blend",
    "CompiledCriterion",
    "Criteria",
    "Criterion",
    "DepthCriterion",
    "FlopsCriterion",
    "LatencyCriterion",
    "MemoryCriterion",
    "criteria_cost",
    "peak_bytes_cost",
]


# ---------------------------------------------------------------------------
#  Duck-typing + weight helpers
# ---------------------------------------------------------------------------


def _looks_like_criterion(x: Any) -> bool:
    """Duck test — the one member the port needs at call time."""
    return callable(getattr(x, "cost_fn", None))


def _crit_name(crit: Any) -> str:
    """Return the axis label for a criterion.

    ``crit.name`` when it is a string, else the class name (a
    nameless duck-typed criterion still records).
    """
    n = getattr(crit, "name", None)
    return n if isinstance(n, str) else type(crit).__name__


def _check_weight(label: str, w: Any) -> float:
    """Validate and return a finite, non-negative blend weight.

    One validation shared by dict specs, ``(criterion, weight)``
    members and ``crit * w``.
    """
    if not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
        raise ValueError(
            f"criteria[{label!r}] must be a finite, "
            f"non-negative weight — got {w!r}"
        )
    return float(w)


def _member_term(m: Any) -> tuple[Any, float]:
    """Normalise one container member to ``(crit, weight)``.

    A criterion → ``(crit, 1.0)``; a ``(criterion, weight)`` pair →
    the pair, weight validated.  Anything else is a spec error.
    """
    if isinstance(m, (tuple, list)):
        if len(m) != 2:
            raise TypeError(
                f"criteria member {m!r} is not a "
                "(criterion, weight) pair — expected a criterion or a"
                " 2-element pair"
            )
        crit, w = m[0], m[1]
    else:
        crit, w = m, 1.0
    if not _looks_like_criterion(crit):
        raise TypeError(
            f"criteria member {crit!r} is not a Criterion — "
            "expected an object with a callable cost_fn(profile)"
        )
    return crit, _check_weight(_crit_name(crit), w)


def _flatten(crit: Any, w: float, out: list[tuple[Any, float]]) -> None:
    """Append a member's leaf terms to *out*.

    A nested container distributes its weight over its own members
    (blending is linear, so flattening preserves the priced sum).
    """
    if isinstance(crit, Criteria):
        for c2, w2 in crit.terms:
            _flatten(c2, w * w2, out)
    else:
        out.append((crit, w))


def _accepts_memo(fn: Any) -> bool:
    """Check whether *fn* declares a ``memo`` parameter.

    The blend threads the shared extraction memo only into members
    that take it (the ``_memo_dispatch`` convention); a bare
    ``fn(term)`` works too.
    """
    try:
        return "memo" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
#  Composition algebra — `crit * w`, `a + b`
# ---------------------------------------------------------------------------


class _Composable:
    """The ``*``/``+`` algebra mixed into criteria and containers.

    ``crit * w`` (or ``w * crit``) → a one-term :class:`Blend`;
    ``a + b`` merges both sides' weighted terms; ``sum(...)`` works
    through the ``0 + x`` identity.
    """

    def __mul__(self, w: Any) -> Blend:
        return Blend([(self, _check_weight(_crit_name(self), w))])

    __rmul__ = __mul__

    def __add__(self, other: Any) -> Blend:
        return Blend([(self, 1.0), (other, 1.0)])

    def __radd__(self, other: Any) -> Any:
        if other == 0:
            return self  # sum()'s start token
        return Blend([(other, 1.0), (self, 1.0)])


# ---------------------------------------------------------------------------
#  Built-in criteria — the shipped selection axes
# ---------------------------------------------------------------------------


class LatencyCriterion(_Composable):
    """Delivered latency — per-op roofline + dispatch, ns.

    The generic executor's per-op roofline plus per-node dispatch
    overhead, nanoseconds
    (``executor_cost_for(lowering="generic")`` — the shipped default
    extraction model).  Profile-calibrated.
    """

    name = "latency"
    charges_shape = True

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the delivered-latency cost model."""
        return executor_cost_for(profile, lowering="generic")


class FlopsCriterion(_Composable):
    """Estimated FLOP count (:func:`flops_cost`) — profile-free."""

    name = "flops"
    charges_shape = True

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the profile-free FLOP cost model."""
        return flops_cost


class DepthCriterion(_Composable):
    """Critical-path roofline latency (:func:`depth_cost_for`).

    The axis that rewards parallel structure a sequential chain
    hides.  Max-composed, so inside extraction its contribution is
    the same clamped non-additive approximation ``compiled``
    carries.
    """

    name = "depth"
    charges_shape = True

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the critical-path depth cost model."""
        return depth_cost_for(profile)


class CompiledCriterion(_Composable):
    """Inductor-style fusion-region pricing (:func:`fused_cost_for`).

    The axis that sees what ``runner=TorchCompileRunner()`` buys.  A
    whole-DAG region partition: non-additive inside extraction;
    prefer it for frontier reporting or small blend weights.
    """

    name = "compiled"
    charges_shape = True

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the fusion-region cost model."""
        return fused_cost_for(profile)


# ---------------------------------------------------------------------------
#  The memory axis — weight storage AND transient-activation depth
# ---------------------------------------------------------------------------


class _PeakBytesCost:
    """Estimated peak TRANSIENT-activation footprint, bytes (fp32).

    The cost model behind ``MemoryCriterion(mode="peak")``: a
    liveness sweep over the term's op-DAG under a sequential
    post-order schedule — the order the generic executor evaluates
    it in (its eval memoizes shared subtrees; interned terms make
    the DAG dedup free).  A value is LIVE from the step that
    produces it until the step of its last consumer: ``Var`` inputs
    are live from entry, the root output stays live to the end, and
    the peak is the max over steps of the summed live bytes — the
    max-concurrency working set of the forward.

    Documented approximations:

    * sequential schedule — a real executor may overlap or reorder
      kernels; the estimate is the generic evaluator's footprint;
    * weight residency is out of scope: ``Param`` leaves and
      param-only subtrees the lowerer folds into materialised
      weights contribute ZERO activation — the ``"weights"`` mode
      prices their storage, ``"combined"`` adds both;
    * view ops (:data:`~catopt_core.cost._VIEW_OPS`) own no storage —
      a view aliases its base allocation, whose lifetime extends to
      the view's last consumer; every materialising op's output
      counts ``numel(shape) * 4`` bytes;
    * non-additive, like ``depth``/``compiled``: under
      ``extract_best``'s subtractive ``local = c(t) - sum children``
      each node bills its clamped marginal contribution — a
      whole-DAG property approximated per node.  The fn's own value
      IS the DAG price (``dag_exact``), so :func:`dag_cost` returns
      it verbatim.
    """

    __name__ = "peak_bytes_cost"
    charges_shape = True
    dag_exact = True

    @staticmethod
    def _bytes(v: Any, memo: dict) -> float:
        """Bytes of one live allocation, fp32 or poison.

        fp32 numel, or the near-infinite poison price when the value
        is provably ill-typed (the ``_INVALID_COST`` convention:
        such a member must never win extraction).
        """
        if _shape_of(v, memo) is _INVALID:
            return _INVALID_COST
        return float(_numel(_shape_of(v, memo))) * 4.0

    def __call__(self, term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = ("pk", term)
        hit = memo.get(ck)
        if hit is not None:
            return hit
        if isinstance(term, Var):
            memo[ck] = self._bytes(term, memo)
            return memo[ck]
        if not isinstance(term, Op):
            memo[ck] = 0.0
            return 0.0
        # Sequential post-order over the op-DAG, deduplicated — the
        # generic executor's own eval order.  Param-only subtrees the
        # lowerer folds into materialised weights are storage: not
        # scheduled, contributing no activation bytes.
        order: list[Op] = []
        seen: set = set()
        stack: list[tuple[Any, bool]] = [(term, False)]
        while stack:
            t, done = stack.pop()
            if done:
                order.append(t)
                continue
            if (
                not isinstance(t, Op)
                or t in seen
                or _folds_to_param(t, None, memo)
            ):
                continue
            seen.add(t)
            stack.append((t, True))
            for a in t.args:
                stack.append((a, False))
        n = len(order)

        def resolve(t: Any) -> Any:
            """Return the allocation a value lives in.

            View ops alias their input's storage, so forward through
            view chains.
            """
            while isinstance(t, Op) and t.op in _VIEW_OPS and t.args:
                t = t.args[0]
            return t

        # birth: the step producing the allocation (Var inputs live
        # at entry).  death: the last step consuming it.
        birth: dict[Any, int] = {}
        death: dict[Any, int] = {}
        for i, t in enumerate(order):
            if resolve(t) is t:  # a view owns no buffer — the base's
                birth[t] = i
                death.setdefault(t, i)
            for a in t.args:
                ra = resolve(a)
                if ra not in death or i > death[ra]:
                    death[ra] = i
                if isinstance(ra, Var):
                    birth.setdefault(ra, 0)
        # The root result stays live to the end of the forward.
        rroot = resolve(term)
        death[rroot] = max(death.get(rroot, 0), n)

        # Live-set sweep over steps 0..n — bytes(v) while
        # birth(v) <= step <= death(v).
        span = n + 1
        diff = [0.0] * (span + 1)
        for v, b in birth.items():
            w = self._bytes(v, memo)
            diff[b] += w
            diff[min(death.get(v, b), span - 1) + 1] -= w
        peak = cur = 0.0
        for delta in diff[:span]:
            cur += delta
            if cur > peak:
                peak = cur
        memo[ck] = peak
        return peak


#: The shared peak-activation model instance (stateless).
peak_bytes_cost = _PeakBytesCost()


class _CombinedMemoryCost:
    """``weights + peak`` — the whole memory footprint in one number.

    Resident parameter storage plus the transient-activation
    liveness peak.  Both components are already whole-DAG prices at
    the root (``dag_exact``); ``charges_param_only`` comes from the
    weights component — storage survives compile-time folding.
    """

    __name__ = "combined_memory_cost"
    charges_param_only = True
    charges_shape = True
    dag_exact = True

    def __init__(self, profile: Any = None) -> None:
        self.profile = profile
        self._weights = param_bytes_cost_for()

    def __call__(self, term: Any, memo: dict | None = None) -> float:
        m = {} if memo is None else memo
        return self._weights(term, m) + peak_bytes_cost(term, m)


#: ``MemoryCriterion`` modes — which memory the axis prices.
MEMORY_MODES: tuple[str, ...] = ("weights", "peak", "combined")


class MemoryCriterion(_Composable):
    """The memory axis — three pricing modes.

    * ``"weights"`` (default) — :func:`param_bytes_cost_for`: the
      LOWERED module's parameter storage.  Prefers weight-shared,
      deduplicated and factorised members.
    * ``"peak"`` — :data:`peak_bytes_cost`: the sequential-schedule
      liveness peak over transient activations (see its docstring
      for the approximation).  Prefers members that keep few
      intermediates live — a member materialising a huge
      intermediate pays its bytes even when it runs fast.
    * ``"combined"`` — weights + peak: the whole footprint axis.

    ``charges_param_only`` holds for ``weights``/``combined`` (they
    bill folded subtrees — storage survives compile-time folding);
    ``charges_shape`` always (every mode reads shapes).
    """

    def __init__(self, mode: str = "weights") -> None:
        """Initialise the axis in one of ``MEMORY_MODES``."""
        if mode not in MEMORY_MODES:
            raise ValueError(
                f"unknown memory mode {mode!r} — "
                f"expected one of {MEMORY_MODES}"
            )
        self.mode = mode

    @property
    def name(self) -> str:
        """``"memory"``, or the mode-qualified label otherwise.

        ``"memory"`` for the back-compat weights mode —
        mode-qualified otherwise, so a blend over two modes records
        both members.
        """
        return (
            "memory"
            if self.mode == "weights"
            else f"memory:{self.mode}"
        )

    @property
    def charges_param_only(self) -> bool:
        """True when the mode bills folded subtrees."""
        return self.mode in ("weights", "combined")

    @property
    def charges_shape(self) -> bool:
        """True — every memory mode reads shapes."""
        return True

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the cost model for the selected mode."""
        if self.mode == "weights":
            return param_bytes_cost_for()
        if self.mode == "peak":
            return peak_bytes_cost
        return _CombinedMemoryCost(profile)


# ---------------------------------------------------------------------------
#  Containers — Criteria (spec) and Blend (composition result)
# ---------------------------------------------------------------------------


class Criteria(_Composable):
    """An ordered, weighted container of criteria.

    A container that is itself a :class:`Criterion`.  Members are
    criterion objects (weight 1.0) or ``(criterion, weight)`` pairs;
    nested containers flatten — blending is linear, so a weighted
    sub-blend distributes onto its own members::

        Criteria(LatencyCriterion(), (MemoryCriterion("peak"), 0.5))

    ``.blend(profile)`` prices the container into the extraction
    ``CostFn``; as a criterion the port member ``cost_fn`` does the
    same.  :func:`criteria_cost` accepts a ``Criteria`` directly.
    """

    name = "criteria"

    def __init__(self, *members: Any) -> None:
        """Flatten *members* into ``(criterion, weight)`` terms."""
        flat: list[tuple[Any, float]] = []
        for m in members:
            crit, w = _member_term(m)
            _flatten(crit, w, flat)
        self._terms = tuple(flat)

    @property
    def terms(self) -> tuple[tuple[Any, float], ...]:
        """The flattened ``(criterion, weight)`` members."""
        return self._terms

    def __iter__(self):
        """Iterate the flattened terms."""
        return iter(self._terms)

    def __len__(self) -> int:
        """Return the number of flattened terms."""
        return len(self._terms)

    def blend(self, profile: Any = None) -> CostFn:
        """Price the container — the extraction ``CostFn``."""
        return _build_blend(self._terms, profile)

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the blend — the :class:`Criterion` port member."""
        return self.blend(profile)


class Blend(Criteria):
    """The ``criterion * weight`` / ``a + b`` composition result.

    A weighted sum of criteria that IS a :class:`Criterion` (and a
    :class:`Criteria` container).

    ``LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3``
    builds a two-term ``Blend``; ``.cost_fn(profile)`` /
    ``.blend(profile)`` prices it, and :func:`criteria_cost` /
    ``optimize_model(criteria=...)`` accept it directly.
    """

    name = "blend"

    def __init__(self, terms: Iterable[Any] = ()) -> None:
        """Build a blend from ``(criterion, weight)`` *terms*."""
        super().__init__(*terms)


# ---------------------------------------------------------------------------
#  The priced callable — a normalised weighted blend of CostFns
# ---------------------------------------------------------------------------


class _CriteriaBlend:
    """A normalised weighted blend of per-criterion cost models.

    The callable :func:`criteria_cost` / ``Criteria.blend`` return.

    Conforms to the :class:`~catopt_core.ports.CostFn` port through
    ``__call__(term, memo=None)`` — a class rather than a closure so
    the reporting/billing markers are real typed attributes (and
    ``inspect.signature`` still finds ``memo`` to thread in; members
    that do not declare ``memo`` are called bare).

    Attributes read off the callable (mirroring the
    ``param_bytes_cost`` / ``executor_cost_for`` conventions):

    * ``criteria`` — the normalised ``{axis_name: weight}`` blend
      actually priced (same-named members merge);
      :func:`optimize_model` records it into ``stats["criteria"]``.
    * ``profile`` — the bound target profile; its presence also
      enables extraction's roofline-memo pre-seeding.
    * ``charges_param_only`` — True when any member axis prices
      storage (e.g. ``MemoryCriterion`` weights/combined): folded
      subtrees still store values, so the param-only discount must
      not zero them.  The marker is per cost-fn — a memory blend
      bills param-only subtrees on EVERY axis, not just storage
      (documented approximation).
    * ``charges_shape`` — True when any member's prices are
      shape-dependent (all shipped axes are).
    """

    def __init__(
        self,
        parts: Iterable[tuple[str, float, CostFn]],
        weights: dict[str, float],
        profile: Any,
        charges_param_only: bool = False,
        charges_shape: bool = False,
    ) -> None:
        self._parts = tuple(parts)
        self._memo_flags = tuple(
            _accepts_memo(fn) for _, _, fn in self._parts
        )
        self.criteria = dict(weights)
        self.profile = profile
        self.charges_param_only = charges_param_only
        self.charges_shape = charges_shape
        self.__name__ = "criteria_cost"

    def __call__(self, term: Any, memo: dict | None = None) -> float:
        m = {} if memo is None else memo
        out = 0.0
        for (_, w, fn), am in zip(
            self._parts, self._memo_flags, strict=True
        ):
            out += w * (fn(term, m) if am else fn(term))
        return out


def _build_blend(
    terms: Iterable[tuple[Any, float]], profile: Any
) -> _CriteriaBlend:
    """Price a flat ``(criterion, weight)`` list into one CostFn.

    Weights NORMALISE to sum 1 — only relative weights matter; a
    zero-weight member is dropped from the priced blend and the
    record.  Billing markers aggregate off each member's built fn
    (falling back to the criterion's own declaration): the blend
    ``charges_param_only`` when any axis bills folded subtrees and
    ``charges_shape`` when any price is shape-dependent.
    """
    raw: list[tuple[Any, float]] = []
    total_w = 0.0
    for crit, w0 in terms:
        w = _check_weight(_crit_name(crit), w0)
        if w > 0:
            raw.append((crit, w))
            total_w += w
    if not raw:
        raise ValueError(
            "criteria must give at least one axis a positive weight"
        )
    parts: list[tuple[str, float, CostFn]] = []
    weights: dict[str, float] = {}
    cpo = csh = False
    for crit, w in raw:
        name = _crit_name(crit)
        fn = crit.cost_fn(profile)
        wn = w / total_w
        parts.append((name, wn, fn))
        weights[name] = weights.get(name, 0.0) + wn
        cpo = cpo or bool(
            getattr(fn, "charges_param_only", False)
            or getattr(crit, "charges_param_only", False)
        )
        csh = csh or bool(
            getattr(fn, "charges_shape", False)
            or getattr(crit, "charges_shape", False)
        )
    return _CriteriaBlend(parts, weights, profile, cpo, csh)


# ---------------------------------------------------------------------------
#  The named axes a dict spec can blend
# ---------------------------------------------------------------------------

#: Axis name → criterion constructor, for the dict spec.  The three
#: ``memory:*`` spellings select :class:`MemoryCriterion` modes.
_AXES: dict[str, Any] = {
    "latency": LatencyCriterion,
    "memory": MemoryCriterion,
    "memory:weights": lambda: MemoryCriterion("weights"),
    "memory:peak": lambda: MemoryCriterion("peak"),
    "memory:combined": lambda: MemoryCriterion("combined"),
    "flops": FlopsCriterion,
    "depth": DepthCriterion,
    "compiled": CompiledCriterion,
}

#: The named selection axes a ``{axis: weight}`` dict can blend, in
#: documentation order — ``memory`` plus its mode-qualified
#: spellings map to :class:`MemoryCriterion`; the rest to the
#: eponymous built-in criteria.
AXES: tuple[str, ...] = tuple(_AXES)

#: ``criteria=None`` reproduces the shipped default model
#: (:func:`executor_cost_for` with ``lowering="generic"``).
_DEFAULT: dict[str, float] = {"latency": 1.0}


def _terms_of(criteria: Any) -> list[tuple[Any, float]]:
    """Flatten a user ``criteria=`` spec into ``(criterion, weight)``.

    Accepted forms: a ``{axis: weight}`` dict (the named axes in
    :data:`AXES`); a :class:`Criteria`/:class:`Blend` container (its
    terms); a single criterion — any object with a callable
    ``cost_fn(profile)`` member, the pluggability contract; or a
    list/tuple mixing criteria and ``(criterion, weight)`` pairs.
    """
    if isinstance(criteria, dict):
        unknown = sorted(set(criteria) - set(AXES))
        if unknown:
            raise ValueError(
                f"unknown criteria axes {unknown} — "
                f"expected a subset of {sorted(AXES)}"
            )
        return [(_AXES[axis](), w) for axis, w in criteria.items()]
    if isinstance(criteria, Criteria):
        return list(criteria.terms)
    if isinstance(criteria, (list, tuple)):
        return [_member_term(m) for m in criteria]
    if _looks_like_criterion(criteria):
        return [(criteria, 1.0)]
    raise TypeError(
        f"unsupported criteria spec {criteria!r} — expected a "
        "{axis: weight} dict, a Criterion, a Criteria/Blend "
        "container, or a list of criteria / (criterion, weight) pairs"
    )


def criteria_cost(criteria: Any = None, profile: Any = None) -> CostFn:
    """Blend selection criteria into one extraction ``CostFn``.

    ``criteria`` accepts:

    * a ``{axis: weight}`` dict — the shipped axes in :data:`AXES`
      (``latency``, ``memory`` + ``memory:weights`` /
      ``memory:peak`` / ``memory:combined``, ``flops``, ``depth``,
      ``compiled`` — see the built-in criterion classes for what
      each prices);
    * a single :class:`Criterion` — ANY object with a callable
      ``cost_fn(profile)`` member: a user-defined axis needs no
      registration step, that IS the pluggability;
    * a :class:`Criteria` / :class:`Blend` container — e.g. the
      ``LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3``
      composition;
    * a list/tuple mixing criteria and ``(criterion, weight)``
      pairs (unweighted members default to 1.0).

    Weights NORMALISE to sum 1 — the blend is convex, so only
    relative weights matter: ``{"latency": 2}`` is exactly
    ``{"latency": 1}``.  The axes' units are heterogeneous
    (nanoseconds, FLOPs, bytes), so each weight is both a
    preference and a unit-conversion factor.  ``criteria=None`` is
    ``{"latency": 1.0}`` — the default extraction model unchanged.
    Unknown dict axes and negative / non-finite / non-numeric
    weights raise ``ValueError`` (members that aren't criteria and
    unsupported spec types raise ``TypeError``); so does a blend
    with no positive weight.

    Additivity — safe inside ``extract_best``: a float sum of
    additive-per-node cost models is still additive, so the
    subtractive local-cost recovery ``local = c(t) - sum(children)``
    is exact for ``latency`` / ``flops`` / ``memory:weights`` (the
    last via its DAG-indexed storage billing).  ``depth``,
    ``compiled`` and ``memory:peak`` / ``memory:combined`` are
    whole-DAG properties — their weighted contribution is a clamped
    marginal, the same documented approximation
    :func:`fused_cost_for` carries; prefer them for frontier
    reporting or small blend weights.

    ``profile`` calibrates the time-priced axes (``latency`` /
    ``depth`` / ``compiled``) — a ``calibrate.TargetProfile`` or any
    object/dict with ``tflops`` / ``gbps`` / ``launch_us`` (and
    optionally ``dispatch_us``); ``None`` uses the built-in
    profile.  Memory axes ignore it — bytes are hardware-independent.

    Returns a :class:`_CriteriaBlend` — a ``CostFn``-conforming
    callable carrying the ``criteria`` / ``profile`` /
    ``charges_param_only`` / ``charges_shape`` markers extraction
    and reporting read (see its docstring).
    """
    spec = _DEFAULT if criteria is None else criteria
    return _build_blend(_terms_of(spec), profile)
