# ruff: noqa: RUF002
"""Regime-adaptive architecture selection.

A *regime* pairs a cost model with an executor.  ``regime_frontier``
extracts, for each regime, the best e-graph member under that regime's
cost function and records which executor can serve it.  A
``RegimeDispatch`` wraps the resulting ``{regime_name: (term, module)}``
mapping into a single callable module: one set of weights, several
certified-equivalent architectures, dispatched by regime name.

Two honesty rules govern the whole module:

* Every served term is a member of the saturated e-graph, hence provably
  equivalent to the source (``frontier.certificate(name)`` produces the
  level-2 certificate).
* If a regime's preferred (executor-native) form is not reachable from
  the root e-class, the frontier does not pretend: it serves the closest
  reachable member and flags the choice ``degraded=True``.  When the
  preferred carrier *is* reachable but not cost-best, the frontier
  nominates it anyway and flags ``forced=True`` with the cost premium
  recorded.

Typical usage::

    from catopt_optimize.regime import Regime, regime_dispatch, default_regimes
    from catopt_core.cost import flops_cost, launch_aware_cost

    disp = regime_dispatch(model, x, regimes=[
        Regime("sequential", cost_fn=flops_cost, executor="generic"),
        Regime("parallel", extract_fn=EGraph.extract_min_depth,
               executor="scan"),
        Regime("decode", cost_fn=launch_aware_cost,
               executor="om_streaming"),
    ])
    y = disp(x, regime="parallel")
    print(disp.frontier.report())

Or directly on a saturated e-graph::

    frontier = regime_frontier(eg, root_eid, regimes, ir=ir)
    disp = frontier.build(param_values=state)

Target profiles
---------------
A regime can carry a *target profile* — measured hardware constants
from :mod:`catopt_optimize.calibrate` — which defaults its cost model to
``roofline_cost_for(profile)``::

    from catopt_optimize.calibrate import calibrate, load_profile

    prof = calibrate()                       # or load_profile("rtx2050")
    disp = regime_dispatch(model, x, regimes=[
        Regime("prefill", profile=prof, executor="om_batched"),
    ])

    # or measure the current device once and fill every
    # profile-less regime with it:
    disp = regime_dispatch(model, x, calibrate=True)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import (
    Any,
)

import torch
import torch.nn as nn
from catopt_carriers.om import OM_LAWS
from catopt_carriers.om_lower import (
    build_om_plan,
    is_om_apply_term,
    to_batched_om_module,
    to_streaming_om_module,
)
from catopt_carriers.scan_lower import (
    build_scan_plan,
    is_scan_apply_term,
    to_batched_scan_module,
)
from catopt_carriers.trace import TRACE_LAWS
from catopt_carriers.xcarrier import XC_LAWS
from catopt_core.cost import (
    _INVALID_COST,
    _VIEW_OPS,
    dag_cost,
    flops_cost,
    launch_aware_cost,
    roofline_cost,
    roofline_cost_for,
)
from catopt_core.egraph import EGraph, ENode
from catopt_core.ir import IR, Const, Op, Param, Var, op_repr
from catopt_core.rules import SCAN_DIAG_LAWS, SCAN_LAWS
from catopt_core.typing import _INVALID, _numel, _shape_of
from catopt_torch.torch_bridge import export_to_ir, ir_to_torch_module

from catopt_optimize.calibrate import TargetProfile, load_profile

__all__ = [
    "CARRIER_LAWS",
    "EXECUTORS",
    "ExecutorSpec",
    "Regime",
    "RegimeChoice",
    "RegimeDispatch",
    "RegimeFrontier",
    "architecture_label",
    "architecture_signature",
    "build_egraph",
    "default_regimes",
    "default_rules",
    "footprint_cost",
    "is_trace_rooted_term",
    "regime_dispatch",
    "regime_frontier",
]

# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def footprint_cost(term: Any, memo: dict | None = None) -> float:
    """Materialised-elements cost — a working-set proxy.

    Every non-view op is charged ``numel(output)``; leaves (Vars, Params,
    Consts) are charged their own size since they must be read.  Unlike
    :func:`flops_cost` this punishes forms that materialise large
    intermediates (e.g. a dense ``T×K`` score matrix), which is the
    quantity a bounded-memory regime actually optimises.  It is an
    additive proxy, not an exact peak-memory model — a genuinely
    streaming schedule's peak is smaller than this sum.
    """
    if memo is None:
        memo = {}
    key = id(term)
    if key in memo:
        return memo[key]
    if isinstance(term, (Var, Param, Const)):
        out = float(_numel(_shape_of(term)))
    elif not isinstance(term, Op):
        out = 0.0
    else:
        shape = _shape_of(term)
        if shape == _INVALID:
            out = _INVALID_COST
        else:
            own = 0.0 if term.op in _VIEW_OPS else float(_numel(shape))
            out = (
                sum(footprint_cost(ch, memo) for ch in term.args) + own
            )
    memo[key] = out
    return out


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def _always_true(_: Any) -> bool:
    return True


def is_trace_rooted_term(term: Any) -> bool:
    """Does the term's root carry the traced-monoidal carrier?

    Native forms: a bare ``trace`` fixpoint at the root, or the
    superposed (channel-split) ``bdiag(trace, trace, …)`` layout the
    ``tr_superpose`` law produces — every channel's fixpoint then
    solves independently, in parallel.
    """
    if not isinstance(term, Op):
        return False
    if term.op == "trace":
        return True
    if term.op == "bdiag":
        return any(
            isinstance(a, Op) and a.op == "trace" for a in term.args
        )
    return False


@dataclass(frozen=True)
class ExecutorSpec:
    """An executor pairing for a regime.

    ``lower`` builds the serving ``nn.Module`` from an ``IR``.
    ``accepts(term)`` is the *term-level* probe: does the term have the
    executor's native shape at the root, so the executor's specialised
    schedule actually fires (not just its serial fallback)?
    ``engaged(module)`` is the *module-level* probe after building.
    ``carrier`` is ``(root_ops, inner_ops, leaf_ops)`` describing the
    carrier family the executor accelerates, or ``None`` for generic
    executors that accept any term.
    """

    name: str
    lower: Callable[[IR, dict | None], nn.Module]
    accepts: Callable[[Any], bool]
    engaged: Callable[[nn.Module], bool]
    carrier: tuple[frozenset, frozenset, frozenset] | None = None


EXECUTORS: dict[str, ExecutorSpec] = {
    "generic": ExecutorSpec(
        "generic",
        ir_to_torch_module,
        _always_true,
        _always_true,
        None,
    ),
    "scan": ExecutorSpec(
        "scan",
        to_batched_scan_module,
        is_scan_apply_term,
        lambda m: bool(getattr(m, "is_batched", False)),
        (
            frozenset({"apply", "applyd"}),
            frozenset({"aff_compose", "affd_compose"}),
            frozenset({"aff", "aff_diag"}),
        ),
    ),
    "om_batched": ExecutorSpec(
        "om_batched",
        to_batched_om_module,
        is_om_apply_term,
        lambda m: bool(getattr(m, "is_batched", False)),
        (
            frozenset({"om_apply"}),
            frozenset({"om_compose"}),
            frozenset({"om", "om_elem"}),
        ),
    ),
    "om_streaming": ExecutorSpec(
        "om_streaming",
        to_streaming_om_module,
        is_om_apply_term,
        lambda m: bool(getattr(m, "is_streaming", False)),
        (
            frozenset({"om_apply"}),
            frozenset({"om_compose"}),
            frozenset({"om", "om_elem"}),
        ),
    ),
    "trace": ExecutorSpec(
        "trace",
        ir_to_torch_module,
        is_trace_rooted_term,
        _always_true,
        (
            frozenset({"trace"}),
            frozenset({"parl", "bdiag"}),
            frozenset({"eye", "cswap"}),
        ),
    ),
}


def _auto_executor(term: Any) -> str:
    """Resolve ``"auto"`` to the executor matching the term's root."""
    if is_scan_apply_term(term):
        return "scan"
    if is_om_apply_term(term):
        return "om_batched"
    if is_trace_rooted_term(term):
        return "trace"
    return "generic"


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """A cost model paired with an executor.

    ``cost_fn`` must be an additive per-op cost of signature
    ``fn(term, memo=None) -> float`` (e.g. ``flops_cost``,
    ``launch_aware_cost``, ``roofline_cost``, ``footprint_cost``); it is
    fed to ``EGraph.extract_best``.  For non-additive objectives such as
    critical-path depth, pass ``extract_fn(eg, root_eid) -> term``
    instead (e.g. ``EGraph.extract_min_depth``).

    ``executor`` is a key of :data:`EXECUTORS` or ``"auto"`` (resolve to
    whatever specialised executor the extracted term's root supports,
    else generic).

    ``prefer_executor``: when the extracted term is not native to the
    executor but the executor's carrier *is* reachable from the root
    e-class, serve the carrier form anyway (flagged ``forced=True``,
    with the cost premium recorded).  When the carrier is unreachable,
    the cost-best term is served under the declared executor — which
    runs its serial fallback — and the choice is flagged
    ``degraded=True``.

    ``profile``: a :class:`catopt_optimize.calibrate.TargetProfile`, a dict with
    ``tflops``/``gbps``/``launch_us`` keys, or a profile name loadable
    via ``catopt_optimize.calibrate.load_profile`` (resolved eagerly at
    construction).  When set and ``cost_fn`` is ``None``, the cost
    model defaults to ``roofline_cost_for(profile)`` — an explicit
    ``cost_fn`` always wins, so ``profile`` then only records which
    target the regime prices against (and still feeds cost accounting
    when ``extract_fn`` drives extraction).  ``profile=None`` is the
    old behaviour.
    """

    name: str
    cost_fn: Callable | None = None
    extract_fn: Callable | None = None
    executor: str = "auto"
    prefer_executor: bool = True
    note: str = ""
    profile: TargetProfile | dict | str | None = None

    def __post_init__(self) -> None:
        if self.profile is None:
            return
        prof = self.profile
        if isinstance(prof, str):
            prof = load_profile(prof)
            object.__setattr__(self, "profile", prof)
        if self.cost_fn is None:
            object.__setattr__(self, "cost_fn", roofline_cost_for(prof))


def default_regimes() -> list[Regime]:
    """The standard frontier: one regime per deployment point.

    * ``sequential``     — min elementwise work, serial evaluation.
    * ``parallel``       — min critical-path depth, auto executor
      (resolves to the batched scan on a lifted recurrence).
    * ``decode``         — min kernel-launch overhead, streaming fold.
    * ``prefill``        — roofline-bound, batched om tree.
    * ``bounded_memory`` — min materialised elements, streaming fold.
    """
    return [
        Regime(
            "sequential",
            cost_fn=flops_cost,
            executor="generic",
            note="min work; serial evaluation",
        ),
        Regime(
            "parallel",
            extract_fn=EGraph.extract_min_depth,
            executor="auto",
            note="min critical path; auto-resolved executor",
        ),
        Regime(
            "decode",
            cost_fn=launch_aware_cost,
            executor="om_streaming",
            note="min kernel launches; streaming OM fold",
        ),
        Regime(
            "prefill",
            cost_fn=roofline_cost,
            executor="om_batched",
            note="roofline-bound; batched OM tree",
        ),
        Regime(
            "bounded_memory",
            cost_fn=footprint_cost,
            executor="om_streaming",
            note="min materialised elements; streaming OM fold",
        ),
    ]


def _as_regime(name: str, spec: Any) -> Regime:
    """Normalise a regimes-mapping entry to a :class:`Regime`."""
    if isinstance(spec, Regime):
        return spec
    if callable(spec):
        return Regime(name, cost_fn=spec)
    if isinstance(spec, dict):
        return Regime(name, **spec)
    if isinstance(spec, (tuple, list)):
        # (cost_fn, executor) or (cost_fn, executor, prefer_executor)
        cost_fn = spec[0] if len(spec) > 0 else None
        executor = spec[1] if len(spec) > 1 else "auto"
        prefer = spec[2] if len(spec) > 2 else True
        return Regime(
            name,
            cost_fn=cost_fn,
            executor=executor,
            prefer_executor=prefer,
        )
    raise TypeError(
        f"cannot interpret regime spec for {name!r}: {spec!r}"
    )


def _normalise_regimes(regimes: Any) -> list[Regime]:
    """Normalise the ``regimes`` argument to a list of :class:`Regime`."""
    if regimes is None:
        return default_regimes()
    if isinstance(regimes, dict):
        return [_as_regime(n, s) for n, s in regimes.items()]
    return list(regimes)


def _attach_profiles(
    regime_list: list[Regime], profiles: dict[str, Any] | None
) -> list[Regime]:
    """Attach ``profiles[regime_name]`` to regimes not carrying one.

    ``profiles`` is keyed by *regime* name; each value is a profile spec
    (``TargetProfile``, dict, or persisted profile name).  A regime with
    an explicit ``profile=`` keeps it — the mapping only fills gaps.
    """
    if not profiles:
        return regime_list
    out: list[Regime] = []
    for r in regime_list:
        if r.profile is None and r.name in profiles:
            r = replace(r, profile=profiles[r.name])
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Term introspection
# ---------------------------------------------------------------------------

_SCAN_OPS = (
    "apply",
    "applyd",
    "aff",
    "aff_diag",
    "aff_compose",
    "affd_compose",
)
_OM_OPS = ("om", "om_elem", "om_compose", "om_apply")
_TRACE_OPS = ("trace", "bdiag", "parl", "eye", "cswap", "inv")


def _census(term: Any) -> dict[str, int]:
    """Op-name census of an extracted term."""
    counts: dict[str, int] = {}

    def rec(t: Any) -> None:
        if isinstance(t, Op):
            counts[t.op] = counts.get(t.op, 0) + 1
            for ch in t.args:
                rec(ch)

    rec(term)
    return counts


def _term_depth(term: Any) -> int:
    memo: dict[int, int] = {}

    def rec(t: Any) -> int:
        k = id(t)
        if k in memo:
            return memo[k]
        if isinstance(t, Op):
            d = 1 + max((rec(ch) for ch in t.args), default=0)
        else:
            d = 0
        memo[k] = d
        return d

    return rec(term)


def _has_carrier(
    term: Any, carrier: tuple[frozenset, frozenset, frozenset]
) -> bool:
    ops = set(carrier[0]) | set(carrier[1]) | set(carrier[2])
    c = _census(term)
    return any(o in c for o in ops)


def _nested_carriers(census: dict[str, int]) -> list[str]:
    return [
        o for o in _SCAN_OPS + _OM_OPS + _TRACE_OPS if census.get(o)
    ]


def architecture_signature(term: Any) -> tuple:
    """A hashable signature identifying the *architecture* of a term.

    Distinct signatures ⇒ genuinely different executable forms (dense
    tensor spine vs sequential ``applyd`` chain vs balanced
    ``affd_compose`` scan vs chunked ``om_apply``).  Equal-cost regimes
    may still collapse to the same signature — that is reported by
    :meth:`RegimeFrontier.collapsed`.
    """
    c = _census(term)
    if is_scan_apply_term(term):
        return (
            "scan",
            "diag" if term.op == "applyd" else "dense",
            c.get("aff", 0) + c.get("aff_diag", 0),
            c.get("aff_compose", 0) + c.get("affd_compose", 0),
        )
    if is_om_apply_term(term):
        return (
            "om",
            c.get("om", 0) + c.get("om_elem", 0),
            c.get("om_compose", 0),
        )
    if is_trace_rooted_term(term):
        return (
            "trace",
            "split" if term.op == "bdiag" else "joint",
            c.get("trace", 0),
        )
    nested = _nested_carriers(c)
    root = term.op if isinstance(term, Op) else "leaf"
    if nested:
        return ("tensor+nested", root, frozenset(nested))
    return ("tensor", root)


def architecture_label(term: Any) -> str:
    """Short human description of a term's architecture."""
    c = _census(term)
    if is_scan_apply_term(term):
        plan = build_scan_plan(term)
        kind = "applyd/aff_diag" if term.op == "applyd" else "apply/aff"
        n_comp = c.get("affd_compose", 0) + c.get("aff_compose", 0)
        if plan is not None:
            sched = (
                f"{len(plan['leaves'])} leaves / "
                f"{len(plan['levels'])} batched levels"
                if n_comp
                else f"{len(plan['leaves'])} leaves / sequential apply "
                "chain"
            )
        else:
            sched = "unplannable"
        return f"scan[{kind}] {sched}"
    if is_om_apply_term(term):
        plan = build_om_plan(term)
        sched = (
            f"{len(plan['leaves'])} blocks / "
            f"{len(plan['levels'])} batched levels"
            if plan is not None
            else "unplannable"
        )
        return f"om_apply {sched}"
    if is_trace_rooted_term(term):
        n_tr = c.get("trace", 0)
        kind = (
            f"bdiag of {n_tr} channel traces"
            if term.op == "bdiag"
            else "joint fixpoint"
        )
        return f"trace[{kind}]"
    nested = _nested_carriers(c)
    root = term.op if isinstance(term, Op) else "leaf"
    if nested:
        return f"tensor[{root}] + nested carrier ({','.join(nested)})"
    return f"tensor[{root}]"


# ---------------------------------------------------------------------------
# Carrier forcing (coordinated extraction)
# ---------------------------------------------------------------------------


def _force_carrier(
    eg: EGraph,
    root_eid: int,
    cost_fn: Callable,
    carrier: tuple[frozenset, frozenset, frozenset],
    accepts: Callable | None = None,
) -> Any | None:
    """Extract the carrier-preferred member via extraction overrides.

    Pins every e-class holding a carrier enode to that enode
    (inner/compose enodes take priority over root/apply enodes within a
    class) and re-extracts — the *coordinated extraction* pattern used
    in ``tests/test_hybrid.py``.  Among the root class's carrier enode
    candidates, the first whose extraction satisfies ``accepts`` (if
    given) wins; otherwise the first extractable candidate is returned.

    Returns the term, or ``None`` when the carrier is unreachable from
    the root class.
    """
    canon = eg.find(root_eid)
    root_ops, inner_ops, _leaf_ops = carrier
    interior: dict[int, ENode] = {}
    root_cands: list[ENode] = []
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:  # pragma: no cover — _classes self-consistent under find()
            continue
        inner = sorted(
            (n for n in ec.nodes if n.op in inner_ops), key=repr
        )
        roots = sorted(
            (n for n in ec.nodes if n.op in root_ops), key=repr
        )
        pick = inner[0] if inner else (roots[0] if roots else None)
        if c == canon:
            root_cands = roots
        elif pick is not None:
            interior[c] = pick
    if not root_cands:
        return None
    for with_interior in (True, False):
        best = None
        for n in root_cands:
            ov = dict(interior) if with_interior else {}
            ov[canon] = n
            t = eg.extract_best(canon, cost_fn, overrides=ov)
            if t is None:
                continue
            if accepts is None or accepts(t):
                return t
            if best is None:
                best = t
        if best is not None:
            return best
    return None


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


@dataclass
class RegimeChoice:
    """What a frontier extracted for one regime."""

    regime: str
    executor: str  # resolved executor key
    term: Any  # the served member of [G]
    cost_term: Any  # pure cost/extract extraction
    cost: float | None  # dag_cost(term, cost_fn)
    cost_best: float | None  # dag_cost(cost_term, cost_fn)
    depth: int  # op-tree depth of served term
    forced: bool  # term obtained by carrier pinning
    native: bool  # executor's schedule fires at root
    carrier_present: bool  # carrier ops present in term
    degraded: bool  # preferred form unreachable at all
    signature: tuple
    label: str
    alternatives: list[tuple[float, Any]] = field(default_factory=list)
    rank: int | None = None  # rank among alternatives by cost
    engaged: bool | None = None  # module probe; set at build() time
    note: str = ""


@dataclass
class RegimeFrontier:
    """The result of :func:`regime_frontier`."""

    eg: EGraph
    root_eid: int
    src_term: Any | None
    ir: IR | None
    regimes: list[Regime]
    choices: dict[str, RegimeChoice]

    # -- access ------------------------------------------------------
    def __getitem__(self, name: str) -> RegimeChoice:
        return self.choices[name]

    def __iter__(self) -> Iterable[RegimeChoice]:
        return iter(self.choices.values())

    def __len__(self) -> int:
        return len(self.choices)

    @property
    def names(self) -> list[str]:
        return list(self.choices)

    # -- analysis ----------------------------------------------------
    def architecture_groups(self) -> dict[tuple, list[str]]:
        """Group regime names by architecture signature."""
        groups: dict[tuple, list[str]] = {}
        for name, ch in self.choices.items():
            groups.setdefault(ch.signature, []).append(name)
        return groups

    @property
    def n_architectures(self) -> int:
        return len(self.architecture_groups())

    def collapsed(self) -> list[tuple[str, list[str]]]:
        """Groups of regimes that extracted the *same member*."""
        by_term: dict[str, list[str]] = {}
        for name, ch in self.choices.items():
            if ch.term is None:
                continue
            by_term.setdefault(op_repr(ch.term), []).append(name)
        return [(r, ns) for r, ns in by_term.items() if len(ns) > 1]

    def certificate(self, name: str):
        """Level-2 certificate for the regime's served member."""
        if self.src_term is None:
            raise ValueError("frontier has no src_term; cannot certify")
        return self.eg.certificate(
            self.src_term,
            self.choices[name].term,
            root_eid=self.root_eid,
        )

    # -- building ----------------------------------------------------
    def build(
        self,
        param_values: dict | None = None,
        *,
        default: str | None = None,
    ) -> RegimeDispatch:
        """Lower every choice with its executor into a RegimeDispatch."""
        return RegimeDispatch(self, param_values, default=default)

    # -- reporting ---------------------------------------------------
    def report(self) -> str:
        lines = [
            f"regime frontier: {len(self)} regimes, "
            f"{self.n_architectures} distinct architectures",
            f"{'regime':<16} {'executor':<14} {'cost':>12} "
            f"{'depth':>5}  architecture / flags",
        ]
        for name, ch in self.choices.items():
            flags = []
            flags.append("forced" if ch.forced else "cost-best")
            if ch.native:
                flags.append("native")
            elif ch.degraded:
                flags.append("DEGRADED")
            elif ch.carrier_present:
                flags.append("carrier-nested")
            else:  # pragma: no cover — degraded == (carrier and not native and not present)
                flags.append("non-native")
            if ch.engaged is not None:
                flags.append(
                    "engaged" if ch.engaged else "serial-fallback"
                )
            if ch.rank is not None:
                flags.append(f"alt#{ch.rank + 1}")
            cost = "-" if ch.cost is None else f"{ch.cost:.4g}"
            lines.append(
                f"{name:<16} {ch.executor:<14} {cost:>12} "
                f"{ch.depth:>5}  {ch.label}  [{', '.join(flags)}]"
            )
            if ch.note:
                lines.append(
                    f"{'':<16} {'':<14} {'':>12} {'':>5}  "
                    f"note: {ch.note}"
                )
        for _repr, names in self.collapsed():
            lines.append(f"collapsed: {names} extract the same member")
        return "\n".join(lines)


def regime_frontier(
    eg: EGraph,
    root_eid: int,
    regimes: Any = None,
    *,
    ir: IR | None = None,
    src_term: Any | None = None,
    top_k: int = 4,
    profiles: dict[str, Any] | None = None,
) -> RegimeFrontier:
    """Extract the best member per regime and pair it with an executor.

    ``regimes`` may be a list of :class:`Regime`, or a dict whose values
    are ``Regime``, a bare ``cost_fn``, ``(cost_fn, executor)`` tuples,
    or kwargs dicts.  ``None`` uses :func:`default_regimes`.

    ``profiles`` is an optional ``{regime_name: profile_spec}`` map:
    each named regime that doesn't already carry a ``profile=`` gets it
    attached (a ``TargetProfile``, a constants dict, or a persisted
    profile name).  Profiles otherwise ride on the ``Regime`` itself.

    ``ir`` (the exported source ``IR``) is needed to :meth:`build` a
    dispatch; ``src_term`` enables :meth:`certificate` — both default to
    ``ir.root`` when ``ir`` is given.
    """
    regime_list = _attach_profiles(
        _normalise_regimes(regimes), profiles
    )
    if src_term is None and ir is not None:
        src_term = ir.root

    alt_cache: dict[int, list[tuple[float, Any]]] = {}
    choices: dict[str, RegimeChoice] = {}
    for spec in regime_list:
        if spec.executor != "auto" and spec.executor not in EXECUTORS:
            raise KeyError(
                f"regime {spec.name!r}: unknown executor "
                f"{spec.executor!r} (have {sorted(EXECUTORS)} + 'auto')"
            )

        # 1. pure extraction under the regime's objective
        if spec.extract_fn is not None:
            cost_term = spec.extract_fn(eg, root_eid)
        else:
            cf = spec.cost_fn or flops_cost
            cost_term = eg.extract_best(root_eid, cf)
        if cost_term is None:
            choices[spec.name] = RegimeChoice(
                regime=spec.name,
                executor=spec.executor,
                term=None,
                cost_term=None,
                cost=None,
                cost_best=None,
                depth=0,
                forced=False,
                native=False,
                carrier_present=False,
                degraded=True,
                signature=("none",),
                label="extraction failed",
                note="extract_fn/cost_fn returned no member",
            )
            continue

        # 2. resolve executor
        ex_name = (
            spec.executor
            if spec.executor != "auto"
            else _auto_executor(cost_term)
        )
        ex = EXECUTORS[ex_name]

        # 3. serve the cost-best term, or force the executor's carrier
        served = cost_term
        forced = False
        native = ex.accepts(served)
        if (
            not native
            and spec.prefer_executor
            and ex.carrier is not None
        ):
            t = _force_carrier(
                eg,
                root_eid,
                spec.cost_fn or flops_cost,
                ex.carrier,
                accepts=ex.accepts,
            )
            if t is not None:
                served, forced = t, True
                native = ex.accepts(t)

        carrier_present = ex.carrier is not None and _has_carrier(
            served, ex.carrier
        )
        degraded = (
            ex.carrier is not None
            and not native
            and not carrier_present
        )

        # 4. cost accounting under the regime's own model
        if spec.cost_fn is not None:
            cost = dag_cost(served, spec.cost_fn)
            cost_best = dag_cost(cost_term, spec.cost_fn)
        else:
            cost = cost_best = None

        # 5. alternatives / rank under this cost model
        alternatives: list[tuple[float, Any]] = []
        rank = None
        if spec.cost_fn is not None and top_k > 0:
            key = id(spec.cost_fn)
            if key not in alt_cache:
                alt_cache[key] = eg.extract_alternatives(
                    root_eid, spec.cost_fn, top_k
                )
            alternatives = alt_cache[key]
            rep = op_repr(served)
            for i, (_c, t) in enumerate(alternatives):
                if op_repr(t) == rep:
                    rank = i
                    break

        note = spec.note
        if degraded:
            note = (
                f"{ex_name} carrier unreachable from this e-class; "
                "executor runs its serial fallback"
                + (f" ({note})" if note else "")
            )
        elif forced and not native:
            note = (
                "carrier reachable only nested below the term root; "
                "executor's specialised schedule cannot fire"
                + (f" ({note})" if note else "")
            )
        elif forced and cost is not None and cost_best is not None:
            note = (
                f"preferred form forced; premium "
                f"{cost - cost_best:+.4g} under this cost model"
                + (f" ({note})" if note else "")
            )

        choices[spec.name] = RegimeChoice(
            regime=spec.name,
            executor=ex_name,
            term=served,
            cost_term=cost_term,
            cost=cost,
            cost_best=cost_best,
            depth=_term_depth(served),
            forced=forced,
            native=native,
            carrier_present=carrier_present,
            degraded=degraded,
            signature=architecture_signature(served),
            label=architecture_label(served),
            alternatives=alternatives,
            rank=rank,
            note=note,
        )

    return RegimeFrontier(
        eg=eg,
        root_eid=root_eid,
        src_term=src_term,
        ir=ir,
        regimes=regime_list,
        choices=choices,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _safe_key(name: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in name
    )


class RegimeDispatch(nn.Module):
    """One set of weights, one architecture per regime.

    Holds ``{regime_name: (extracted_term, executor_module)}``.  Source
    parameters are shared objects across all executor modules — every
    form reads the same weights.  ``forward(*xs, regime=None)`` routes
    to the named regime (or the default).
    """

    def __init__(
        self,
        frontier: RegimeFrontier,
        param_values: dict | None = None,
        *,
        default: str | None = None,
    ):
        super().__init__()
        if frontier.ir is None:
            raise ValueError(
                "frontier has no IR; pass ir= to regime_frontier to "
                "build a dispatch"
            )
        self.frontier = frontier
        ir = frontier.ir

        self._name_map: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        forms: dict[str, nn.Module] = {}
        for name, ch in frontier.choices.items():
            if ch.term is None:
                continue
            spec = EXECUTORS[ch.executor]
            ir_i = IR(
                root=ch.term,
                inputs=ir.inputs,
                input_names=ir.input_names,
                params=ir.params,
            )
            mod = spec.lower(ir_i, param_values)
            key = _safe_key(name)
            self._name_map[name] = key
            forms[key] = mod
            ch.engaged = spec.engaged(mod)
            self._meta[name] = {
                "term": ch.term,
                "module": mod,
                "choice": ch,
            }
        self.forms = nn.ModuleDict(forms)
        self._share_params()

        names = [n for n in frontier.choices if n in self._meta]
        if not names:
            raise ValueError("no regime produced an executable form")
        self._regime = default if default is not None else names[0]
        if self._regime not in self._meta:
            raise KeyError(f"unknown default regime {default!r}")
        self.verification: dict[str, dict] | None = None

    # -- one set of weights ------------------------------------------
    @staticmethod
    def _param_map_of(mod: nn.Module) -> dict | None:
        """The ``{ir_name: Parameter}`` map of a module — on the module
        itself for ``IRModule``, on ``.eval_mod`` for the specialised
        wrappers (scan/om executors)."""
        pm = getattr(mod, "_param_map", None)
        if pm is None:
            inner = getattr(mod, "eval_mod", None)
            pm = (
                getattr(inner, "_param_map", None)
                if inner is not None
                else None
            )
        return pm

    def _share_params(self) -> None:
        """Rebind source parameters to a single shared Parameter object
        per name across all executor modules."""
        shared: dict[str, nn.Parameter] = {}
        for mod in self.forms.values():
            pm = self._param_map_of(mod)
            if pm is None:
                continue
            host = (
                mod
                if getattr(mod, "_param_map", None) is pm
                else mod.eval_mod
            )
            for pname in pm:
                if pname.startswith("fused_"):
                    continue  # per-module fold intermediates
                cur = pm[pname]
                if not isinstance(cur, nn.Parameter):
                    continue
                if pname in shared:
                    setattr(host, pname, shared[pname])
                    pm[pname] = shared[pname]
                else:
                    shared[pname] = cur

    # -- API ---------------------------------------------------------
    @property
    def regimes(self) -> list[str]:
        return list(self._meta)

    @property
    def regime(self) -> str:
        return self._regime

    def set_regime(self, name: str) -> None:
        if name not in self._meta:
            raise KeyError(
                f"unknown regime {name!r}; available: "
                f"{sorted(self._meta)}"
            )
        self._regime = name

    @property
    def entries(self) -> dict[str, tuple[Any, nn.Module]]:
        """``{regime_name: (extracted_term, executor_module)}``."""
        return {
            n: (m["term"], m["module"]) for n, m in self._meta.items()
        }

    def executor_module(self, name: str) -> nn.Module:
        return self.forms[self._name_map[name]]

    def forward(
        self, *xs: torch.Tensor, regime: str | None = None
    ) -> torch.Tensor:
        name = regime if regime is not None else self._regime
        if name not in self._meta:
            raise KeyError(
                f"unknown regime {name!r}; available: "
                f"{sorted(self._meta)}"
            )
        return self.forms[self._name_map[name]](*xs)

    # -- equivalence -------------------------------------------------
    def certificate(self, name: str):
        """Level-2 certificate: source term → this regime's member."""
        return self.frontier.certificate(name)

    def max_diff(
        self,
        reference: Any,
        *xs: torch.Tensor,
        regime: str | None = None,
    ) -> dict[str, float]:
        """Max |form(x) − reference| per regime (or one named regime)."""
        ref = reference(*xs) if callable(reference) else reference
        names = [regime] if regime is not None else self.regimes
        out: dict[str, float] = {}
        with torch.no_grad():
            for n in names:
                y = self.forward(*xs, regime=n)
                out[n] = (y - ref).abs().max().item()
        return out

    def verify(
        self, reference: Any, *xs: torch.Tensor, atol: float = 1e-9
    ) -> dict[str, dict]:
        """Check every dispatched form against a reference output.

        ``reference`` is a tensor or a callable producing it from
        ``*xs``.  Returns ``{regime: {"max_abs_diff": d, "ok": bool}}``
        and caches it on ``self.verification``.
        """
        diffs = self.max_diff(reference, *xs)
        self.verification = {
            n: {"max_abs_diff": d, "ok": d <= atol}
            for n, d in diffs.items()
        }
        return self.verification

    def report(self) -> str:
        lines = [self.frontier.report(), "", "built modules:"]
        for name, m in self._meta.items():
            ch = m["choice"]
            lines.append(
                f"  {name:<16} {type(m['module']).__name__:<22} "
                f"engaged={ch.engaged}"
            )
        if self.verification:
            lines.append("equivalence vs reference:")
            for n, v in self.verification.items():
                lines.append(
                    f"  {n:<16} max|Δ|={v['max_abs_diff']:.3e} "
                    f"ok={v['ok']}"
                )
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return f"regime={self._regime!r}, forms={list(self._meta)}"


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

#: All carrier rewrite families — one saturation serves every regime.
#: TRACE_LAWS rides along: importing catopt_carriers.trace also registers the
#: trace/bdiag/parl/eye/cswap/inv torch bindings, so any trace-bearing
#: member the JSV laws reach is executable by the generic executor.
#: trace enodes enter via the trace_lift non-local pass in optimize.py.
#: XC_LAWS crosses the carrier seam: linear readouts exit the scan
#: carriers, the om numerator is such a readout, and the deferred omd
#: carrier keeps chunked attention affine in the scan's initial state.
#: ``build_egraph`` keeps the cross-carrier seam laws (XC_LAWS) in a
#: bounded second tier (see its ``xc`` flag): the set is mostly
#: bidirectional pairs minting fresh enodes, so it gets its own
#: iteration budget after the carriers are established.
CARRIER_LAWS = SCAN_LAWS + SCAN_DIAG_LAWS + OM_LAWS + TRACE_LAWS


def default_rules() -> list:
    return list(CARRIER_LAWS)


def build_egraph(
    model: nn.Module,
    example_input: Any,
    *,
    rules: list | None = None,
    xc: bool = True,
    max_iterations: int = 14,
    max_nodes: int = 400_000,
):
    """Export ``model`` and saturate an e-graph with carrier laws.

    Returns ``(eg, root_eid, ir, source_tensors, stats)``.

    ``rules`` selects the core saturating set (default
    ``CARRIER_LAWS``).  ``xc`` (default on) adds a bounded second tier:
    after the non-local lifts have established the carriers, the
    cross-carrier seam laws (``XC_LAWS``) run with their own small
    iteration budget, then a short core pass integrates the seam
    members — looped at most twice.  Keeping XC out of the saturating
    tier bounds the blast radius of its bidirectional promotion pairs
    on carrier-heavy graphs while still letting om_elem_affd / the
    readout and omd lift rules fire.
    """
    ir, source = export_to_ir(model, example_input)
    eg = EGraph()
    root = eg.add_term(ir.root)
    core = default_rules() if rules is None else rules
    stats = eg.run(
        core, root, max_iterations=max_iterations, max_nodes=max_nodes
    )
    # Non-local lifts: recurrences -> trace(F), stacks of same-state
    # carrier applications -> one application, om trees over scanned
    # values -> the deferred omd carrier.  Witnessed, replayable.
    from catopt_carriers.trace_lift import lift_scan_to_trace
    from catopt_carriers.xcarrier import (
        gather_apply_stack,
        gather_applyd_stack,
        omd_tree_lift,
    )

    def _lifts():
        return (
            lift_scan_to_trace(eg)
            + gather_applyd_stack(eg)
            + gather_apply_stack(eg)
            + omd_tree_lift(eg)
        )

    lifts = _lifts()
    if lifts:
        eg.rebuild()
        stats["nonlocal_lifts"] = len(lifts)
        eg.run(core, root, max_iterations=5, max_nodes=max_nodes)
    if xc:
        xc_rounds = 0
        for _ in range(2):
            before = eg.n_enodes
            eg.run(XC_LAWS, root, max_iterations=4, max_nodes=max_nodes)
            grew = eg.n_enodes != before
            # XC-minted members can enable new non-local offers (e.g.
            # chunk_apply putting affine maps into om leaf values, which
            # omd_tree_lift then lifts whole) — re-run the passes.
            more = _lifts()
            if more:
                eg.rebuild()
                stats["nonlocal_lifts"] = stats.get(
                    "nonlocal_lifts", 0
                ) + len(more)
            if not grew and not more:
                break
            xc_rounds += 1
            # let the core laws absorb the seam members
            eg.run(core, root, max_iterations=2, max_nodes=max_nodes)
        stats["xc_rounds"] = xc_rounds
        stats["xc_fires"] = sum(
            eg.rule_fires.get(r.name, 0) for r in XC_LAWS
        )
    return eg, root, ir, source, stats


def regime_dispatch(
    model: nn.Module,
    example_input: Any,
    regimes: Any = None,
    *,
    rules: list | None = None,
    xc: bool = True,
    max_iterations: int = 14,
    max_nodes: int = 400_000,
    default: str | None = None,
    verify: bool = True,
    atol: float = 1e-9,
    profiles: dict[str, Any] | None = None,
    calibrate: Any = None,
) -> RegimeDispatch:
    """End-to-end: export → saturate → frontier → build → verify.

    Returns a :class:`RegimeDispatch` whose ``.frontier`` records every
    regime's choice.  With ``verify=True`` each form is checked against
    the model's own output on ``example_input`` (fp64 recommended).

    ``profiles`` is a ``{regime_name: profile_spec}`` map forwarded to
    :func:`regime_frontier` — it fills ``profile`` on named regimes
    that don't carry one.  ``calibrate`` is a convenience for "price
    this model against the current device": ``calibrate=True`` calls
    ``catopt_optimize.calibrate.calibrate()`` once and attaches the measured
    profile to every regime still lacking one; a ``TargetProfile`` /
    dict / persisted name does the same without measuring.  Since an
    explicit ``cost_fn`` always wins over a profile, ``calibrate``
    changes *extraction* only for regimes that declare no cost model —
    elsewhere it is recorded for provenance.  ``calibrate=None`` (the
    default) is the old behaviour.
    """
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    eg, root, ir, source, _stats = build_egraph(
        model,
        example_input,
        rules=rules,
        xc=xc,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
    )
    regime_list = _attach_profiles(
        _normalise_regimes(regimes), profiles
    )
    if calibrate:
        pending = [r.name for r in regime_list if r.profile is None]
        if pending:
            if calibrate is True:
                from catopt_optimize.calibrate import (
                    calibrate as _measure,
                )

                prof: Any = _measure()
            else:
                prof = calibrate
            regime_list = _attach_profiles(
                regime_list, {n: prof for n in pending}
            )
    frontier = regime_frontier(
        eg, root, regime_list, ir=ir, src_term=ir.root
    )
    disp = frontier.build(param_values=source, default=default)
    if verify:
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                ref = model(*[a.clone() for a in args])
                disp.verify(ref, *args, atol=atol)
        finally:
            model.train(was_training)
    return disp
