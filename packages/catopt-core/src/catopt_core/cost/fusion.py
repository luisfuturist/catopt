"""Compiled-lowering (Inductor-style) fusion-region cost model.

Partitions a term's op-DAG into the kernels a ``torch.compile``
lowering emits (:func:`fusion_regions`), prices each region's summed
member FLOPs against its external/boundary traffic
(:func:`_fused_cost`) and exposes the per-member tie-break
(:func:`fusion_member_key`) extraction uses among near-cost members.
"""
# ruff: noqa: RUF002, RUF003 — math notation in comments

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from catopt_core.ir import Op
from catopt_core.typing import _numel, _shape_of

from .basic import _CostMarkers, _flops_of
from .params import _FUSION_POINTWISE_OPS, _folds_to_param
from .roofline import (
    _kernel_lookup,
    _profile_constants,
    _profile_dispatch_s,
    _profile_graph_overhead_s,
    _profile_kernel_table,
)

if TYPE_CHECKING:
    from catopt_core.ports import CostFn


#: Ops emitting NO kernel under a compiled lowering — fusion-
#: TRANSPARENT plumbing.  Pure views (Inductor folds their index
#: arithmetic into the consumer's kernel — the _VIEW_OPS convention;
#: the set adds the aten spellings torch.export emits: unbind/getitem/
#: select/slice/squeeze/unsqueeze/expand/flatten/alias/dropout) and
#: carrier *packaging*: ``aff``/``aff_diag``/``om``/``omd`` assemble
#: the carried pair/triple and ``affd_a``/``affd_b``/``aff_A``/``aff_b``
#: project a component — under dynamo tracing they are Python-level
#: plumbing that never reaches the graph.  A pointwise consumer unions
#: with a transparent node's pointwise DESCENDANTS: the plumbing does
#: not split the region.  Constant morphisms (``eye``/``cswap``)
#: materialise at compile time — free, and they take no args so
#: nothing forwards through them.
_FUSION_TRANSPARENT_OPS: frozenset = frozenset(
    {
        "transpose",
        "reshape",
        "broadcast",
        "chunk",
        "split",
        "unsqueeze",
        "squeeze",
        "select",
        "slice",
        "expand",
        "flatten",
        "getitem",
        "unbind",
        "alias",
        "dropout",
        "leaf",
        "aff",
        "aff_diag",
        "om",
        "omd",
        "affd_a",
        "affd_b",
        "aff_A",
        "aff_b",
        "eye",
        "cswap",
    }
)

#: Everything that can ride inside a fused region — pointwise members
#: plus transparent plumbing.  (Previously ``_VIEW_OPS |
#: _FOLDABLE_ELEMWISE``; the compiled-lowering model now also knows
#: the pointwise carrier bodies and the remaining aten views.)
_FUSIBLE_OPS = _FUSION_POINTWISE_OPS | _FUSION_TRANSPARENT_OPS

#: Ops whose binding hides a direct solver call (``linalg.solve`` /
#: inverse) — measured orders of magnitude beyond a dispatch
#: (the fidelity sweep caught a ``trace`` term priced like ~3
#: dispatches that measured 14.5 s: the resolvent's solve, not the
#: graph).  FLOP models already bill ``2·du³``; this surcharge carries
#: the dispatch-side constant until a measured ``solve_us`` profile
#: field lands.
_SOLVER_OPS = frozenset({"trace", "inv"})

#: A solver call ≈ this many generic dispatches — a conservative
#: floor (a 128×128 solve measures ~ms vs ~µs per dispatch).
_SOLVER_FACTOR = 10_000.0


def fusion_regions(
    term: Any, memo: dict | None = None
) -> tuple[frozenset, ...]:
    """Partition *term*'s op-DAG into Inductor-style fusion regions.

    One entry per kernel the ``"compiled"`` lowering emits.

    Each frozenset is one kernel's member ops:

    * a maximal connected cluster of pointwise ops
      (``_FUSION_POINTWISE_OPS``) — Inductor streams the whole cluster
      in one kernel; or
    * a singleton ``{op}`` for every fusion BOUNDARY: contractions
      (``matmul``/``linear``/``conv2d``/``sdpa``), reductions
      (``sum``/``mean``/``max``/``min``/``softmax``/``*_norm``),
      materialising layout ops (``concat``/``stack``/``contiguous``,
      gathers ``index_select``/``embedding``, ``bdiag``/``parl``),
      solver ops (``trace``/``inv``), carrier ops whose binding hides
      a contraction (``apply``/``aff_compose``/``omd_applym``/
      ``om_elem``/``omd_elem``/``om_elem_aff*``) — and any unknown op,
      conservatively.

    ``_FUSION_TRANSPARENT_OPS`` plumbing (views and carrier packaging:
    ``aff``/``aff_diag``/``om``/``omd``/``affd_a``/...) emits no kernel
    and appears in no region — a pointwise consumer unions with a
    transparent node's pointwise DESCENDANTS, so the tuple/view
    plumbing does not split a region.  Param-only subtrees that
    ``_folds_to_param`` materialises at lowering contribute nothing:
    the compiled graph reads them as inputs (the same compile-time
    fold the extract_best param-only discount prices at 0).  Leaves
    are never members.

    ``len(fusion_regions(t))`` is the predicted kernel count — what
    :func:`executor_overhead` reports for ``lowering="compiled"``.
    The partition is profile-independent (shapes don't enter) and a
    WHOLE-DAG property, not additive per node: regions merge across
    siblings at a shared pointwise parent and a shared subterm fuses
    once.  Cost fns built on it (``fused_cost_for``) are therefore
    reporting/frontier models — under ``extract_best``/``dag_cost``'s
    subtractive ``local = c(t) − Σc(children)`` a sibling merge
    clamps to 0 and the merge is billed nowhere; fine for
    ``lowering_aware``/frontier comparisons, approximate inside
    extraction (the ``_generic_overhead`` docstring has the
    additivity contract).
    """
    memo = {} if memo is None else memo
    ck = ("fr", term)
    hit = memo.get(ck)
    if hit is not None:
        return hit
    out: list[frozenset] = []
    if isinstance(term, Op):
        # Collect the DAG's op nodes once.  A param-only subtree the
        # lowerer folds into a materialised Param is a kernel INPUT,
        # not a kernel — skip it without descending.
        nodes: list[Op] = []
        seen: set = set()
        stack = [term]
        while stack:
            t = stack.pop()
            if not isinstance(t, Op) or t in seen:
                continue
            seen.add(t)
            if _folds_to_param(t, None, memo):
                continue
            nodes.append(t)
            stack.extend(t.args)
        node_set = set(nodes)

        # Union-find: one region per connected pointwise cluster.
        rep = {t: t for t in nodes}

        def find(t: Op) -> Op:
            while rep[t] is not t:
                rep[t] = rep[rep[t]]
                t = rep[t]
            return t

        for t in nodes:
            if t.op not in _FUSION_POINTWISE_OPS:
                continue
            # Transparent args forward their own args (transparent ops
            # never fold, so every transparent arg was collected).
            eff: list[Any] = list(t.args)
            i = 0
            while i < len(eff):
                a = eff[i]
                if (
                    isinstance(a, Op)
                    and a.op in _FUSION_TRANSPARENT_OPS
                ):
                    eff[i : i + 1] = a.args
                else:
                    i += 1
            for a in eff:
                if (
                    isinstance(a, Op)
                    and a.op in _FUSION_POINTWISE_OPS
                    and a in node_set
                ):
                    ra, rb = find(t), find(a)
                    if ra is not rb:
                        rep[ra] = rb
        grouped: dict[Op, set] = {}
        for t in nodes:
            if t.op in _FUSION_POINTWISE_OPS:
                grouped.setdefault(find(t), set()).add(t)
        out = [frozenset(m) for m in grouped.values()]
        out += [
            frozenset({t}) for t in nodes if t.op not in _FUSIBLE_OPS
        ]
    memo[ck] = tuple(out)
    return memo[ck]


def _exposes_pointwise(t: Any, memo: dict) -> bool:
    """Whether *t*'s output can merge into a pointwise consumer's region.

    Mirrors :func:`fusion_regions`' effective-arg forwarding: a
    pointwise root exposes itself; transparent plumbing (views,
    carrier packaging) forwards the question to its own args; a
    param-only fold, a leaf, or a boundary op is a kernel *input* —
    there is no member op for a consumer to union with.
    """
    if not isinstance(t, Op) or _folds_to_param(t, None, memo):
        return False
    if t.op in _FUSION_POINTWISE_OPS:
        return True
    if t.op in _FUSION_TRANSPARENT_OPS:
        return any(_exposes_pointwise(a, memo) for a in t.args)
    return False


def fusion_member_key(
    term: Any, memo: dict | None = None
) -> tuple[int, int]:
    """Fusion tie-break key for near-cost-equal e-class members.

    ``(n_regions, boundary)``, compared lexicographically — smaller
    wins — by :meth:`EGraph.extract_best` when ``fusion_epsilon``
    puts several members inside one cost band:

    * ``n_regions`` — ``len(fusion_regions(term))``: the member
      subtree's predicted kernel count under the compiled lowering.
      A member whose pointwise internals collapse into one region
      beats a member that must launch several kernels.
    * ``boundary`` — 0 when the term's root can itself join a
      pointwise consumer's region (:func:`_exposes_pointwise`), 1
      otherwise.  Region count alone cannot see this: a pointwise
      member and a ``matmul`` member can both occupy one region, yet
      only the pointwise one lets a *parent* kernel absorb it.

    ``memo`` is the shared extraction cost memo —
    :func:`fusion_regions`, ``_folds_to_param`` and ``_shape_of`` all
    key into it, so the probe prices each distinct term once across
    every extraction pass.
    """
    memo = {} if memo is None else memo
    return (
        len(fusion_regions(term, memo)),
        0 if _exposes_pointwise(term, memo) else 1,
    )


def _region_traffic(
    region: frozenset,
    parents: dict,
    root: Op,
    memo: dict,
) -> tuple[float, float]:
    """Bytes a fused kernel actually moves: (in, out).

    *in* — every effective input of a member that is NOT itself a
    member, deduplicated: leaves, other kernels' outputs, and
    materialised param folds, each read once.  Transparent plumbing
    forwards its own args, so a view never surfaces as an input
    (Inductor folds its index arithmetic into the kernel).

    *out* — every member value consumed by a non-member op (through
    any number of transparent forwards) or returned as the term root:
    the tensors the kernel must write.  Interior member→member values
    stay in registers and move nothing — the structural fix over
    dominant-member pricing, where a region whose members read
    DIFFERENT large externals was billed only the largest op's
    traffic.
    """
    in_b = 0.0
    out_b = 0.0
    ext: set = set()
    for m in region:
        # inputs: forward through transparent args until an opaque
        # producer (member → interior; anything else → an external
        # read).
        pending = list(m.args)
        while pending:
            a = pending.pop()
            if isinstance(a, Op) and a.op in _FUSION_TRANSPARENT_OPS:
                pending.extend(a.args)
                continue
            if a in region or a in ext:
                continue
            ext.add(a)
            in_b += _numel(_shape_of(a, memo)) * 4.0
        # outputs: m crosses the region boundary if a consumer chain
        # (forwarding through transparent parents) reaches an op
        # outside the region — or m is the term root.
        if m is root:
            out_b += _numel(_shape_of(m, memo)) * 4.0
            continue
        reach = [m]
        seen: set = {m}
        boundary = False
        while reach and not boundary:
            u = reach.pop()
            for p in parents.get(u, ()):
                if p in region:
                    continue
                if p.op in _FUSION_TRANSPARENT_OPS:
                    if p not in seen:
                        seen.add(p)
                        reach.append(p)
                    continue
                boundary = True
                break
        if boundary:
            out_b += _numel(_shape_of(m, memo)) * 4.0
    return in_b, out_b


def _fused_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    dispatch_s: float,
    kernel_ns=None,
    graph_overhead_s: float = 0.0,
) -> float:
    """Inductor-approximation price of *term* (see fused_cost_for).

    One kernel per :func:`fusion_regions` entry.  A region costs
    ``max(total member FLOPs / peak, region traffic / bandwidth)`` +
    one ``launch_s``: the fused kernel runs the SUM of its members'
    arithmetic (fusion removes launches, not FLOPs) and streams
    :func:`_region_traffic`'s external reads + boundary writes, so
    intermediates never touch memory.  When the profile carries an
    ``op_kernel_ns`` table, the region's kernel time is additionally
    floored at the sum of its members' measured kernel work (each
    member's measured wall minus its solo launch — the region
    launches once).  The whole graph pays a single
    ``dispatch_s`` — compilation removes interior dispatch too; the
    surviving boundary is the compiled module's own call.  A region
    containing a solver op additionally bills ``_SOLVER_FACTOR``
    dispatches — the extern ``linalg.solve``/``inv`` call dominates
    any kernel math, the same floor the generic count carries.
    """
    if not isinstance(term, Op):
        return 0.0
    regions = fusion_regions(term, memo)
    if not regions:
        return 0.0
    # parent edges over the op-DAG (skipping folded param subtrees —
    # they are kernel inputs, not consumers) — used to find each
    # member's boundary-crossing outputs.
    parents: dict = {}
    seen_dag: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or t in seen_dag:
            continue
        seen_dag.add(t)
        if _folds_to_param(t, None, memo):
            continue
        for a in t.args:
            if isinstance(a, Op):
                parents.setdefault(a, []).append(t)
                stack.append(a)
    total = 0.0
    for region in regions:
        flops = 0.0
        solver = False
        for t in region:
            flops += _flops_of(t, memo)
            solver = solver or t.op in _SOLVER_OPS
        in_b, out_b = _region_traffic(region, parents, term, memo)
        kernel_s = max(flops / peak_flops, (in_b + out_b) / peak_bw)
        if kernel_ns is not None:
            # Measured kernel times floor the region: the fused kernel
            # cannot run faster than the sum of its members' measured
            # kernel work (each member's measured wall stripped of its
            # solo launch — the region launches once, priced below).
            work_ns = 0.0
            for t in region:
                m = kernel_ns(t, memo)
                if m is not None:
                    work_ns += max(0.0, m - launch_s * 1e9)
            kernel_s = max(kernel_s, work_ns / 1e9)
        total += (
            kernel_s * 1e9
            + launch_s * 1e9
            + (_SOLVER_FACTOR if solver else 0.0) * dispatch_s * 1e9
        )
    # One per-graph charge — the bigger of the serial dispatch and the
    # measured compiled-graph call overhead (guards + cudagraph-safe
    # entry): the per-kernel table cannot see it, and calibration
    # showed it dominates the compiled price's residual.
    total += max(dispatch_s, graph_overhead_s) * 1e9
    return float(total)


def fused_cost_for(profile: Any = None) -> CostFn:
    """Compiled-lowering cost — the fusion-region (Inductor) model.

    One predicted kernel per :func:`fusion_regions` region: a maximal
    connected pointwise cluster (``_FUSION_POINTWISE_OPS`` — elementwise
    bindings plus the pointwise carrier bodies ``affd_compose``/
    ``applyd``/``om[d]_compose``/``om[d]_apply``) priced at
    ``max(Σ member FLOPs / peak, region traffic / bandwidth)`` — the
    kernel runs every member's arithmetic (fusion removes launches,
    not FLOPs) and streams its external reads + boundary writes once,
    so intermediates never touch memory and interior launches vanish
    (:func:`_region_traffic` — the fix for dominant-member pricing
    undercharging a region whose members read different externals).
    Non-fusible ops (matmul/conv/sdpa, reductions, materialising
    layout ops, the solver ops ``trace``/``inv``, the
    contraction-bearing carrier ops) are region singletons priced the
    same way; solver singletons additionally bill ``_SOLVER_FACTOR``
    dispatches (the extern solve dwarfs launch overheads — the same
    floor the generic count carries).  Transparent plumbing (views,
    carrier packaging) emits no kernel; param-only folds are kernel
    inputs.  Each region pays one ``launch_s`` and the whole graph one
    ``dispatch_s`` (the profile's ``dispatch_us``, else the launch
    constant): compilation removes launches and interior dispatch —
    the surviving boundary is the compiled module's own call.  Without
    that floor a lone GEMM would always look cheaper compiled than
    generic, and it isn't; fusion cannot shrink one kernel.

    Non-additive: the region partition is a whole-DAG property (see
    :func:`fusion_regions`), so under ``extract_best``/``dag_cost``'s
    subtractive local-cost decomposition the model is approximate —
    prefer it for ``lowering_aware``/frontier reporting.

    Deliberately approximate: real fusion decisions are
    scheduler-dependent (rematerialise vs reuse, reduction splits,
    layout constraints), and a big fused kernel may also spill
    intermediates the register file can't hold — the traffic model
    bills boundary crossings only.  That is the honest part of the
    model — pointwise fusion is where the measured win lives.

    Units are nanoseconds, matching :func:`roofline_cost`; the closure
    has the standard ``fn(term, memo=None)`` signature.
    """
    pf, bw, ls = _profile_constants(profile)
    dispatch_s = _profile_dispatch_s(profile)
    goh = _profile_graph_overhead_s(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = ("fc", pf, bw, ls, dispatch_s, goh, id(kns), term)
        if ck in memo:
            return memo[ck]
        out = _fused_cost(term, memo, pf, bw, ls, dispatch_s, kns, goh)
        memo[ck] = float(out)
        return out

    cost.__name__ = "fused_cost_for"
    cast(_CostMarkers, cost).profile = profile
    return cost
