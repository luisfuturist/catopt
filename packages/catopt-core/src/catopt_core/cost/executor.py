"""Executor-aware pricing — dispatch overhead and per-lowering costs.

Counts the work each lowering performs (:func:`executor_overhead`),
prices it on top of a base model (:func:`executor_cost_for`) and takes
the min over lowerings (:func:`lowering_aware_cost_for`).
"""
# ruff: noqa: RUF002 — math notation in comments

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from catopt_core.ir import Op, Param
from catopt_core.typing import _numel, _shape_of

from .basic import _CostMarkers, flops_cost
from .fusion import (
    _SOLVER_FACTOR,
    _SOLVER_OPS,
    fused_cost_for,
    fusion_regions,
)
from .params import _folds_to_param
from .roofline import (
    _depth_cost,
    _kernel_lookup,
    _local_roofline,
    _profile_constants,
    _profile_dispatch_s,
    _profile_kernel_table,
    _profile_leaf_eval_s,
    _roofline_cost,
)

if TYPE_CHECKING:
    from catopt_core.ports import CostFn


# ---------------------------------------------------------------------------
#  Lowering-aware pricing — price the lowering, not just the term
# ---------------------------------------------------------------------------
#
# The fidelity study (bench/suites/core/cost_fidelity.py) showed term-level cost is
# blind to the lowering: the same term prices identically whether the
# generic IRModule dispatches it node-by-node, a batched carrier
# executor level-schedules it, or a compiled kernel fuses it — while
# measured latency differs 6-30x.  The models below put the executor
# in the price: executor_overhead counts the dispatched units a
# lowering performs, executor_cost_for charges them at the profile's
# dispatch rate on top of a base model, fused_cost_for approximates
# Inductor-style pointwise fusion, and lowering_aware_cost_for prices
# each term at its cheapest lowering — extraction then picks the term
# whose best lowering is cheapest.

#: Executor kinds a term can be lowered through.
LOWERINGS: tuple = ("generic", "batched_scan", "compiled")

#: Root ops whose first argument is a carrier map tree the
#: level-batched executors lower as a scan plan: ``scan_lower``'s
#: ``apply``/``applyd``, ``om_lower``'s ``om_apply``, ``omd_lower``'s
#: ``omd_apply``/``omd_applym``.  Anything else falls back to the
#: serial per-node evaluator inside those modules too.
_SCAN_ROOT_OPS = frozenset(
    {"apply", "applyd", "om_apply", "omd_apply", "omd_applym"}
)

#: Carrier compose ops forming the balanced tree the batched
#: executors level-schedule — one batched op per tree *level* rather
#: than one dispatch per node.
_SCAN_COMPOSE_OPS = frozenset(
    {"aff_compose", "affd_compose", "om_compose", "omd_compose"}
)


def _generic_overhead(term: Any, memo: dict) -> float:
    """Count the per-node dispatches the generic evaluator performs.

    One unit per op occurrence in *term* (+``_SOLVER_FACTOR`` for solver
    ops).

    Deliberately NOT DAG-deduplicated: ``EGraph.extract_best`` recovers
    a node's local cost as ``f(t) − Σf(children)``, which is exact
    only for additive functions — a shared child is already billed
    once at the e-class level, so deduplicating here would double-
    subtract and collapse locals to zero (non-additive cost fns are
    what made a ``trace`` resolvent term price like ~3 dispatches and
    win extraction — fidelity bench, ``cost_fidelity.py``).

    A param-only subtree ``_folds_to_param`` materialises at lowering
    contributes zero units — ``IRModule._fold_weight_chains`` rewrites
    it to a bound parameter before the first forward, so there is no
    runtime dispatch to price (a param-only subtree that does NOT fold
    — e.g. a ``trace`` resolvent — still evaluates every call and is
    counted).
    """
    ck = ("eo", "generic", term)
    if ck in memo:
        return memo[ck]
    # eo(t) = w(t) + Σ eo(children) — memoised PER NODE so dag_cost's
    # one-call-per-node pricing is linear in the DAG, not O(N²): the
    # previous flat stack walk repriced every node's whole subtree.
    # Iterative post-order — spine-depth chains blow the recursion
    # limit (T=2048).
    stack = [(term, 0)]
    while stack:
        t, phase = stack.pop()
        c2 = ("eo", "generic", t)
        if c2 in memo:
            continue
        if phase == 0:
            if not isinstance(t, Op) or _folds_to_param(t, None, memo):
                memo[c2] = 0.0
                continue
            stack.append((t, 1))
            for a in t.args:
                stack.append((a, 0))
        else:
            memo[c2] = (
                _SOLVER_FACTOR if t.op in _SOLVER_OPS else 1.0
            ) + sum(memo[("eo", "generic", a)] for a in t.args)
    return memo[ck]


def _outside_overhead(term: Op, spine: Op, memo: dict) -> float:
    """Distinct op nodes of *term*'s DAG outside the ``spine`` subtree.

    The batched executor still runs everything around the compose
    spine generically — the apply root itself, the carried-state
    argument, surrounding tensor terms.  Param-only foldable subtrees
    materialise at lowering (``_generic_overhead`` convention) and are
    not counted; nodes are deduplicated because the executor's eval
    memoizes shared subtrees.
    """
    n = 0.0
    seen: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if (
            not isinstance(t, Op)
            or t in seen
            or t == spine
            or _folds_to_param(t, None, memo)
        ):
            continue
        seen.add(t)
        n += 1.0
        stack.extend(t.args)
    return n


def _batched_scan_overhead(term: Op, memo: dict) -> float:
    """Dispatched units under the level-batched carrier lowering.

    The compose spine under an apply-family root collapses to one
    batched op per balanced-tree *level* — ``ceil(log2 n_leaves)``
    dispatches — plus the leaf materialisation: the executors stack
    uniform leaves into ONE batched evaluation per leaf *kind*
    (``leaf_a_shared``/``leaf_b_gather`` — a (T,·) gather, not T
    sequential evals), so distinct leaf op-families bill once each,
    weighted by their generic size.  Ops outside the spine — the
    apply root, the state argument — still dispatch generically and
    are counted as such.
    """
    spine = term.args[0]
    seen: set = set()
    leaves: list = []
    stack = [spine]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op) and t.op in _SCAN_COMPOSE_OPS:
            stack.extend(t.args)
        else:
            leaves.append(t)
    levels = math.ceil(math.log2(max(1, len(leaves))))
    total = float(levels) + _outside_overhead(term, spine, memo)
    # One batched leaf evaluation per leaf kind — the executor stacks
    # same-op leaves into a single gathered batch.
    for kind in {getattr(leaf, "op", "") for leaf in leaves}:
        rep = next(
            leaf for leaf in leaves if getattr(leaf, "op", "") == kind
        )
        total += max(1.0, _generic_overhead(rep, memo))
    return total


def executor_overhead(
    term: Any, lowering: str, memo: dict | None = None
) -> float:
    """Structural count of the executor work a *lowering* performs.

    Counts, not seconds — multiply by a per-dispatch cost (as
    :func:`executor_cost_for` does) to price it.  ``lowering`` is one
    of :data:`LOWERINGS`:

    * ``"generic"`` — the per-node IRModule dispatcher: every op
      occurrence is one dispatched evaluation (counted per-node, not
      deduplicated — additive, which ``extract_best``'s local-cost
      decomposition requires).  View ops count too — the generic
      evaluator still dispatches them even though they launch no
      kernel.  Solver ops (``trace``/``inv``) bill
      ``_SOLVER_FACTOR`` each.  Param-folded subtrees simply contain
      no ops to count: their params are bound at lowering, not
      evaluated.
    * ``"batched_scan"`` — the level-batched carrier executors
      (``BatchedScanModule`` / ``BatchedOMModule`` /
      ``BatchedOmdModule``): for an ``apply``/``applyd``/``om_apply``/
      ``omd_apply[m]``-rooted term, ``ceil(log2 n_leaves)`` batched
      compose levels + one evaluation per distinct compose-tree leaf;
      ops outside the spine dispatch generically.  Non-scan roots get
      the generic count — the modules' serial fallback.
    * ``"compiled"`` — ``len(fusion_regions(term))``: the compiled
      executor's unit of work is the KERNEL, so the count is the
      region count — a pointwise chain of any depth fuses to one
      region, not O(nodes).  Priced by :func:`fused_cost_for`; the
      count is a whole-DAG property (regions merge across siblings),
      i.e. NON-additive — fine for overhead reporting/min-over-
      lowerings, not for ``extract_best``'s subtractive local-cost
      decomposition (see ``_generic_overhead``).
    """
    memo = {} if memo is None else memo
    ck = ("eo", lowering, term)
    hit = memo.get(ck)
    if hit is not None:
        return hit
    if lowering == "compiled":
        out = float(len(fusion_regions(term, memo)))
    elif lowering == "generic":
        out = _generic_overhead(term, memo)
    elif lowering == "batched_scan":
        if (
            isinstance(term, Op)
            and term.op in _SCAN_ROOT_OPS
            and term.args
        ):
            out = _batched_scan_overhead(term, memo)
        else:
            # Not a scan shape — the batched modules run their serial
            # fallback, i.e. generic dispatch.
            out = _generic_overhead(term, memo)
    else:
        raise ValueError(
            f"unknown lowering {lowering!r} — "
            f"expected one of {LOWERINGS}"
        )
    memo[ck] = float(out)
    return memo[ck]


def _generic_latency_ns(
    term: Any,
    memo: dict,
    pf: float,
    bw: float,
    ls: float,
    dsp: float,
    kernel_ns=None,
) -> float:
    """Whole-subtree latency of the serial per-node evaluator, ns.

    The ``executor_cost_for(lowering="generic")`` composition:
    per-op roofline (kernel work + launch, floored at measured kernel
    times when ``kernel_ns`` is bound) plus one ``dispatch_s``
    machinery overhead per dispatched op.  Used inside the batched
    model for the pieces the level executors still run generically —
    leaf operand evals and everything outside the compose spine.
    """
    return _roofline_cost(term, memo, pf, bw, ls, kernel_ns) + (
        _generic_overhead(term, memo) * dsp * 1e9
    )


def _leaf_shared_a(leaves: list) -> bool:
    """``leaf_a_shared``: every leaf's first arg is the same term.

    Mirrors ``catopt_carriers.scan_lower.build_scan_plan`` — same
    object, or identically-named ``Param`` leaves (LTI recurrence):
    the executor evaluates the transition once and ``expand``s a
    stride-0 batch view instead of stacking T copies.
    """
    if not leaves or not getattr(leaves[0], "args", None):
        return False
    a0 = leaves[0].args[0]
    for leaf in leaves[1:]:
        if not getattr(leaf, "args", None) or not leaf.args:
            return False
        a = leaf.args[0]
        if a is a0 or (
            isinstance(a, Param)
            and isinstance(a0, Param)
            and a.name == a0.name
        ):
            continue
        return False
    return True


def _leaf_gather_base(leaves: list) -> Any | None:
    """``leaf_b_gather``'s shared base term, or ``None``.

    Mirrors ``catopt_carriers.scan_lower._leaf_b_gather``: every
    leaf's *second* argument is ``select(base, dim, index)`` over the
    SAME base term and dim — one ``index_select`` (or the base itself
    for contiguous indices) replaces n tiny indexing evals.
    """
    base = None
    dim = None
    for leaf in leaves:
        args = getattr(leaf, "args", None)
        if not args or len(args) < 2:
            return None
        b = args[1]
        if not (isinstance(b, Op) and b.op == "select" and b.args):
            return None
        d = b.attrs.get("dim", 0)
        i = b.attrs.get("index")
        if not isinstance(d, int) or not isinstance(i, int):
            return None
        if base is None:
            base, dim = b.args[0], d
        elif b.args[0] is not base or d != dim:
            return None
    return base


def _batched_scan_latency(
    term: Op,
    memo: dict,
    pf: float,
    bw: float,
    ls: float,
    dsp: float,
    leaf_eval_s: float,
    kernel_ns=None,
) -> float:
    """Whole-term nanoseconds under the level-batched carrier lowering.

    The level executors (``BatchedScanModule`` and the om-family
    variants) do NOT walk the term node-by-node — the compose spine
    collapses to ~log₂(n) batched levels, so the price decomposes as
    the executor's own work:

    * **leaf materialisation** — ``leaf_a_shared`` (every leaf's
      transition is the same term) evaluates the a-part ONCE;
      ``leaf_b_gather`` (every leaf's input is ``base[i]``) evaluates
      the base once and pays one ``index_select``.  Otherwise each
      leaf operand evals through the eval machinery — ``leaf_eval_s``
      per leaf plus the arg subtrees' kernel time — and one
      ``torch.stack`` per side.
    * **compose levels** — ``ceil(log2 n)`` levels; level *l* emits
      ~``n/2^{l+1}`` carried elements.  The diagonal/elementwise path
      costs ~9 dispatched calls per level (4 slot gathers, mul, mul+add,
      2 cat); the dense affine path ~4 (2 gathers, bmm, cat).  Each
      level also moves the running carried state: gather reads,
      arithmetic writes, and the cat copies —
      ``s·(4·m_l + done)·4`` bytes where ``s`` is the carried
      element's scalar count.
    * **outside the spine** — the apply root, the carried-state
      argument and surrounding terms still evaluate generically:
      per-node roofline + dispatch, deduplicated (the executor's eval
      memoizes shared subtrees).

    Whole-spine and non-additive — the same reporting/frontier caveat
    as ``fused_cost_for`` (see :func:`fusion_regions`).  Callers
    restrict it to ``_SCAN_ROOT_OPS`` terms; anything else keeps the
    serial-fallback (generic) price.
    """
    spine = term.args[0]
    seen: set = set()
    leaves: list = []
    stack = [spine]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op) and t.op in _SCAN_COMPOSE_OPS:
            stack.extend(t.args)
        else:
            leaves.append(t)
    n = max(1, len(leaves))
    dispatch_ns = dsp * 1e9
    call_ns = (ls + dsp) * 1e9
    total = 0.0

    # ---- leaf materialisation -------------------------------------
    # one leaf-operand eval = leaf_eval_s eval-machinery overhead +
    # the arg subtree's kernel time (roofline only — dispatch is what
    # leaf_eval_us measures, so charging it again would double-count).
    def leaf_eval(subtree: Any) -> float:
        return leaf_eval_s * 1e9 + _roofline_cost(
            subtree, memo, pf, bw, ls, kernel_ns
        )

    two_part = bool(leaves) and all(
        isinstance(lf, Op) and len(lf.args) >= 2 for lf in leaves
    )
    stack_ns = call_ns  # one torch.stack per materialised side
    if two_part and _leaf_shared_a(leaves):
        # LTI: one eval + a stride-0 expand view (free).
        total += leaf_eval(leaves[0].args[0])
    elif two_part:
        a_terms = list({lf.args[0] for lf in leaves})
        total += n * leaf_eval_s * 1e9 + stack_ns
        total += sum(
            _roofline_cost(a, memo, pf, bw, ls, kernel_ns)
            for a in a_terms
        )
    if two_part:
        base = _leaf_gather_base(leaves)
        if base is not None:
            # One base eval + one index_select — not n indexing evals.
            total += leaf_eval(base)
            total += (
                call_ns + _numel(_shape_of(base, memo)) * 4.0 / bw * 1e9
            )
        else:
            b_terms = list({lf.args[1] for lf in leaves})
            total += n * leaf_eval_s * 1e9 + stack_ns
            total += sum(
                _roofline_cost(b, memo, pf, bw, ls, kernel_ns)
                for b in b_terms
            )
    else:
        # Non pair-carriers (om triples and friends): per-leaf evals.
        arg_terms = list(
            {a for lf in leaves for a in getattr(lf, "args", ())}
        )
        total += n * leaf_eval_s * 1e9 + stack_ns
        total += sum(
            _roofline_cost(a, memo, pf, bw, ls, kernel_ns)
            for a in arg_terms
        )

    # ---- compose levels ---------------------------------------------
    # carried element scalars: the leaf packaging's summed arg numel.
    s = 1.0
    if leaves and isinstance(leaves[0], Op):
        s = float(
            sum(_numel(_shape_of(a, memo)) for a in leaves[0].args)
        )
        s = max(s, 1.0)
    dense = term.op == "apply"  # dense affine: (d+1)x(d+1) bmm compose
    calls = 4.0 if dense else 9.0
    levels = math.ceil(math.log2(n))
    done = float(n)
    for lv in range(levels):
        m_l = max(1.0, math.ceil(n / 2 ** (lv + 1)))
        flops_lv = 2.0 * m_l * s**1.5 if dense else 3.0 * m_l * s
        # 2·m_l·s gather reads + m_l·s arith writes + (done+m_l)·s cat
        # copies of the running carried state.
        bytes_lv = (4.0 * m_l + done) * s * 4.0
        total += (
            calls * call_ns + max(flops_lv / pf, bytes_lv / bw) * 1e9
        )
        done += m_l

    # ---- outside the spine: generic eval, DAG-deduplicated ----------
    seen2: set = set()
    stack2 = [term]
    while stack2:
        t = stack2.pop()
        if (
            not isinstance(t, Op)
            or t in seen2
            or t == spine
            or _folds_to_param(t, None, memo)
        ):
            continue
        seen2.add(t)
        total += (
            _local_roofline(
                t,
                memo,
                peak_flops=pf,
                peak_bw=bw,
                launch_s=ls,
                kernel_ns=kernel_ns,
            )
            + dispatch_ns
        )
        stack2.extend(t.args)
    return float(total)


def executor_cost_for(
    profile: Any = None,
    *,
    lowering: str = "generic",
    base: str = "roofline",
) -> CostFn:
    """Cost fn = a base model's term cost + per-dispatch overhead.

    ``base`` picks the underlying term price — ``"roofline"`` (per-op
    roofline sum, like :func:`roofline_cost`), ``"depth"``
    (critical-path roofline, like :func:`depth_cost`), or ``"flops"``
    (:func:`flops_cost`).  On top the closure adds
    ``executor_overhead(term, lowering) * dispatch_s`` where
    ``dispatch_s`` is the profile's ``dispatch_us`` (seconds;
    :func:`_profile_dispatch_s` fallback applies).  When the profile
    carries an ``op_kernel_ns`` table, per-op kernel times inside the
    roofline/depth/batched pieces floor at the measured values
    (``_kernel_lookup`` — see ``roofline_cost_for``).

    Units follow the base model, matching this file's conventions:
    roofline and depth are nanoseconds (``_local_roofline`` returns
    ns), so the overhead term is ``overhead * dispatch_s * 1e9``.
    For ``"flops"`` each dispatched unit is billed ``dispatch_us``
    flop-equivalents — the ``launch_aware_cost`` convention scaled to
    microseconds.  That is an approximation, documented as such: a
    dispatch is time, not arithmetic; the honest conversion would be
    ``dispatch_s * peak_flops``, which swamps every real term.
    µs-as-flops keeps the penalty commensurate with the model's
    magnitudes.

    ``lowering="compiled"`` ignores ``base`` and delegates to
    :func:`fused_cost_for` — the compiled executor's price is its
    fusion structure, not a per-node dispatch count
    (``executor_overhead`` is 0 there).  ``lowering="batched_scan"``
    likewise ignores ``base`` on scan-apply roots: the level-batched
    executors don't run the term's nodes at all — they run
    :func:`_batched_scan_latency`'s leaf-gather / level-compose
    schedule, whose work is priced directly (a whole-spine property,
    so non-additive like ``fused_cost_for``; serial-fallback terms
    keep the additive generic price).  The returned closure has the
    standard ``fn(term, memo=None)`` signature.
    """
    if lowering not in LOWERINGS:
        raise ValueError(
            f"unknown lowering {lowering!r} — "
            f"expected one of {LOWERINGS}"
        )
    if base not in ("roofline", "depth", "flops"):
        raise ValueError(
            f"unknown base {base!r} — "
            "expected 'roofline', 'depth' or 'flops'"
        )
    if lowering == "compiled":
        return fused_cost_for(profile)
    pf, bw, ls = _profile_constants(profile)
    dispatch_s = _profile_dispatch_s(profile)
    leaf_eval_s = _profile_leaf_eval_s(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))
    # ns for the roofline/depth bases; flop-equivalents for flops.
    per = dispatch_s * (1e9 if base != "flops" else 1e6)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = (
            "ec",
            lowering,
            base,
            pf,
            bw,
            ls,
            dispatch_s,
            leaf_eval_s,
            id(kns),
            term,
        )
        if ck in memo:
            return memo[ck]
        if (
            lowering == "batched_scan"
            and isinstance(term, Op)
            and term.op in _SCAN_ROOT_OPS
            and term.args
        ):
            out = _batched_scan_latency(
                term, memo, pf, bw, ls, dispatch_s, leaf_eval_s, kns
            )
        else:
            if base == "roofline":
                out = _generic_latency_ns(
                    term, memo, pf, bw, ls, dispatch_s, kns
                )
            else:
                if base == "depth":
                    b = _depth_cost(term, memo, pf, bw, ls, kns)
                else:  # flops
                    b = flops_cost(term, memo)
                out = b + executor_overhead(term, lowering, memo) * per
        memo[ck] = float(out)
        return out

    cost.__name__ = "executor_cost_for"
    cast(_CostMarkers, cost).profile = profile
    cast(_CostMarkers, cost).lowering = lowering
    return cost


def lowering_aware_cost_for(
    profile: Any = None,
    *,
    lowerings: tuple = LOWERINGS,
    base: str = "roofline",
) -> CostFn:
    """Min-over-lowerings cost: price each term at its cheapest executor.

    For each ``l`` in ``lowerings`` the term is priced by
    :func:`executor_cost_for` ``(lowering=l, base=base)`` —
    ``"compiled"`` routes to :func:`fused_cost_for` instead — and the
    term's cost is the minimum.  Extraction under this model
    implicitly picks the term whose *best* lowering is cheapest,
    rather than pricing every term as if the generic per-node
    dispatcher would run it: a balanced carrier tree credits its
    level-batched plan, a pointwise chain credits fusion.

    The ``"compiled"`` arm is non-additive (the region partition is a
    whole-DAG property — see :func:`fusion_regions`), so the minimum
    is too: as an ``extract_best`` cost_fn the model is approximate —
    a sibling merge can hide inside a clamped local.  Reporting and
    frontier comparison are its sound uses.

    The returned closure carries ``cost.best_lowering(term,
    memo=None) -> str`` — the argmin lowering (first in ``lowerings``
    order on ties), for reporting which executor the price
    corresponds to.
    """
    fns: dict[str, CostFn] = {}
    for lw in lowerings:
        if lw == "compiled":
            fns[lw] = fused_cost_for(profile)
        elif lw in LOWERINGS:
            fns[lw] = executor_cost_for(profile, lowering=lw, base=base)
        else:
            raise ValueError(
                f"unknown lowering {lw!r} — expected one of {LOWERINGS}"
            )
    if not fns:
        raise ValueError("lowerings must be non-empty")
    pf, bw, ls = _profile_constants(profile)
    dsp = _profile_dispatch_s(profile)
    kt = _profile_kernel_table(profile)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = (
            "lw",
            base,
            pf,
            bw,
            ls,
            dsp,
            id(kt),
            tuple(lowerings),
            term,
        )
        if ck in memo:
            return memo[ck]
        out = min(f(term, memo) for f in fns.values())
        memo[ck] = float(out)
        return out

    def best_lowering(term: Any, memo: dict | None = None) -> str:
        memo = {} if memo is None else memo
        return min(fns, key=lambda lw: fns[lw](term, memo))

    cost.__name__ = "lowering_aware_cost_for"
    cast(_CostMarkers, cost).profile = profile
    cast(_CostMarkers, cost).best_lowering = best_lowering
    return cost
