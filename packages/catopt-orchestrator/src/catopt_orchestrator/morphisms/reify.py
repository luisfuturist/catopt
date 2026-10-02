"""Reify + the strategy entry point.

Maps a morphism-level :class:`MorphismMatch` back to concrete, verified
IR: build the joint term per the boundary mode, saturate the named
recipe, cost-gate, verify through the sink, and emit per-slot
replacements.  Also holds :class:`MorphismSearch` and the
``optimize_morphisms`` entry point.  Split out of
:mod:`catopt_orchestrator.morphisms` (plan 0011); the whole surface is
re-exported from that package.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from catopt_core import laws
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, Param, Var, op_repr_dag
from catopt_core.laws import tags as _law_tags
from catopt_core.laws.pairing import share_duplicate_params
from catopt_core.pipeline import LowerResult
from catopt_core.ports import Composer, CostFn, Meter, Sink, Source

from catopt_orchestrator.runners import IdentityRunner

from .graph import MorphismGraph, _BlockRecord, lift_graph
from .laws import (
    MorphismLaw,
    MorphismMatch,
    NormCascade,
    OutInCompose,
    ReifySpec,
    ResidualAbsorb,
    ResidualReassoc,
    WeightTie,
    WindowCompose,
)
from .signature import BlockSig, _act_index, _iter_ops, _param_only

log = logging.getLogger("catopt_orchestrator.morphisms")


#: The default morphism law family, in application order — widest
#: spans first (a window claims its nodes before any pair law sees
#: them; a declined window leaves its pairs to the pair laws), then
#: pair-level transforms, then node-level and tying.
DEFAULT_MORPHISM_LAWS: tuple[MorphismLaw, ...] = (
    WindowCompose(),
    ResidualReassoc(),
    OutInCompose(),
    ResidualAbsorb(),
    NormCascade(),
    WeightTie(),
)


# ---------------------------------------------------------------------------
#  Reify — morphism rewrite -> concrete, verified IR
# ---------------------------------------------------------------------------


def _ns_prefix(name: str) -> str:
    """Namespace prefix for one block's params inside a joint term."""
    return name.replace(".", "_") + "__"


def _prefix_params(term: Any, pre: str) -> Any:
    """Rename every ``Param`` leaf in *term* with the block prefix."""
    if isinstance(term, Param):
        return Param(pre + term.name, term.typ)
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_prefix_params(a, pre) for a in term.args),
            **term.attrs,
        )
    return term


def _subst(term: Any, var: Var, repl: Any) -> Any:
    """Substitute *repl* for every occurrence of *var* in *term*."""
    if term == var:
        return repl
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_subst(a, var, repl) for a in term.args),
            **term.attrs,
        )
    return term


def _ctx_hit(
    host: _BlockRecord, member: _BlockRecord, j: int
) -> Var | None:
    """Return the host input var carrying member's context arg ``j``.

    Identity, not value: the captured live objects must be the same
    tensor, confirmed on the perturbed probe when it ran.
    """
    obj = member.in_objs[j]
    host_ir = host.ir
    if host_ir is None:
        return None
    for k, va in enumerate(host_ir.inputs):
        if k >= len(host.in_objs) or host.in_objs[k] is not obj:
            continue
        if (
            host.in_objs2
            and member.in_objs2
            and not (
                k < len(host.in_objs2)
                and j < len(member.in_objs2)
                and host.in_objs2[k] is member.in_objs2[j]
            )
        ):
            return None
        return va
    return None


def _ctx_var_map(
    host: _BlockRecord,
    member: _BlockRecord,
    member_act: int,
) -> tuple[dict | None, str | None]:
    """Map member's context vars onto host's input vars — by identity.

    The fused joint is delivered at the host's slot: it only ever
    sees the host's call args.  A member context input (a rope
    ``cos``/``sin`` table, a read-only cache) is therefore expressible
    only when the captured evidence shows the host received literally
    the same object — the live ``in_objs`` identity on the first
    capture, confirmed on the perturbed probe when it ran.  Returns
    ``(mapping, None)`` or ``(None, reason)`` — an unshared context
    input is an honest decline, never a guessed binding.
    """
    host_ir, mem_ir = host.ir, member.ir
    if host_ir is None or mem_ir is None:
        return None, "opaque node"
    mapping: dict = {}
    for j, vb in enumerate(mem_ir.inputs):
        if j == member_act:
            continue
        if j >= len(member.in_objs):
            return None, "context inputs not captured"
        hit = _ctx_hit(host, member, j)
        if hit is None:
            return (
                None,
                f"context input {j} of {member.name} not shared "
                f"by {host.name}",
            )
        mapping[vb] = hit
    return mapping, None


def _subst_ctx(term: Any, ctx_map: dict | None) -> Any:
    """Substitute every context var of the member per *ctx_map*."""
    for vb, va in (ctx_map or {}).items():
        term = _subst(term, vb, va)
    return term


def _joint_parts(
    ira: IR,
    irb: IR,
    rec_a: _BlockRecord,
    rec_b: _BlockRecord,
    mode: str,
    *,
    act_a: int = 0,
    act_b: int = 0,
    ctx_map: dict | None = None,
) -> tuple[Any, Var, Any, dict, dict]:
    """Compose the pair's IRs per the boundary mode — terms, not modules.

    Returns ``(joint, x, mid, params, leaves)``: the joint term over
    the shared input ``x`` (A's activation variable), ``mid`` — the
    value B reads (``A(x)`` for chains, ``x + A(x)`` for residuals) —
    and the namespaced param/value tables (per-block ``p_*`` leaf
    names never collide).  The ``_wrapped`` modes add the outer wrap
    the parent's ``y + ·`` performs.

    Multi-input: ``act_a``/``act_b`` select the activation var each
    side composes over; ``ctx_map`` (from :func:`_ctx_var_map`)
    rebinds B's context vars to the A-input vars carrying the same
    captured objects, so the joint is a pure function of A's inputs
    and B's const/state inputs pass through unchanged.
    """
    pa, pb = _ns_prefix(rec_a.name), _ns_prefix(rec_b.name)
    x, vb = ira.inputs[act_a], irb.inputs[act_b]
    y_a = _prefix_params(ira.root, pa)
    mid = Op.make("add", x, y_a) if mode.startswith("residual") else y_a
    body = _subst(
        _subst_ctx(_prefix_params(irb.root, pb), ctx_map), vb, mid
    )
    joint = (
        Op.make("add", mid, body) if mode.endswith("_wrapped") else body
    )
    params = {
        **{pa + k: Param(pa + k, p.typ) for k, p in ira.params.items()},
        **{pb + k: Param(pb + k, p.typ) for k, p in irb.params.items()},
    }
    leaves = {
        **{pa + k: v for k, v in rec_a.leaves.items()},
        **{pb + k: v for k, v in rec_b.leaves.items()},
    }
    return joint, x, mid, params, leaves


def _joint_parts_window(
    irs: list[IR],
    recs: list[_BlockRecord],
    kinds: tuple[str, ...],
    *,
    acts: tuple[int, ...] | None = None,
    ctx_maps: tuple[dict | None, ...] | None = None,
) -> tuple[Any, Var, tuple, dict, dict]:
    """Compose a ≥3-block window's IRs — the n-ary :func:`_joint_parts`.

    Returns ``(joint, x, mids, params, leaves)``: ``joint`` is the
    whole window's function of the first block's activation input
    ``x``; ``mids`` is the tuple of residual-*stream* nodes the later
    blocks read (in creation order ``s_0..s_{k-2}`` — the distribute
    offer expands them outermost-first); empty for the chain family,
    which has no additive structure to distribute over.

    Construction is driven by ``kinds`` (one per interior boundary):
    a residual-family window accumulates the stream
    ``s_j = s_{j-1} + f_j(s_{j-1})`` every later block reads; a
    chain-family window nests ``f_j(f_{j-1}(·))``.  In both, the last
    wire's ``_wrapped`` mark decides whether the segment's value is
    the raw last body or ``in + body``.  ``acts`` selects each
    block's activation input position; ``ctx_maps`` rebinds each
    member's context vars to the first block's input vars
    (:func:`_ctx_var_map`), so the window stays a pure function of
    the host slot's args.
    """
    pres = [_ns_prefix(r.name) for r in recs]
    acts = acts or (0,) * len(irs)
    ctx_maps = ctx_maps or (None,) * len(irs)
    roots = [
        _subst_ctx(_prefix_params(ir.root, pre), cm)
        for ir, pre, cm in zip(irs, pres, ctx_maps, strict=True)
    ]
    params: dict[str, Param] = {}
    leaves: dict[str, Any] = {}
    for ir, rec, pre in zip(irs, recs, pres, strict=True):
        params.update(
            {
                pre + k: Param(pre + k, p.typ)
                for k, p in ir.params.items()
            }
        )
        leaves.update({pre + k: v for k, v in rec.leaves.items()})
    x = irs[0].inputs[acts[0]]
    first = _subst(roots[0], irs[0].inputs[acts[0]], x)
    if kinds[0].startswith("residual"):
        joint, mids = _window_residual(
            roots, irs, acts, kinds, x, first
        )
    else:
        joint, mids = _window_chain(roots, irs, acts, kinds, first)
    return joint, x, tuple(mids), params, leaves


def _window_residual(
    roots: list,
    irs: list[IR],
    acts: tuple[int, ...],
    kinds: tuple[str, ...],
    x: Var,
    first: Any,
) -> tuple[Any, list]:
    """Accumulate the residual stream — ``s_j = s_{j-1} + f_j(s_{j-1})``.

    Block ``j >= 1`` reads the running in+out sum; interior wires are
    ``residual_wrapped`` by the law's own grammar (the stream must
    flow on).
    """
    stream = Op.make("add", x, first)
    mids: list[Any] = [stream]
    for j in range(1, len(irs) - 1):
        fj = _subst(roots[j], irs[j].inputs[acts[j]], stream)
        stream = Op.make("add", stream, fj)
        mids.append(stream)
    body = _subst(roots[-1], irs[-1].inputs[acts[-1]], stream)
    joint = (
        Op.make("add", stream, body)
        if kinds[-1].endswith("_wrapped")
        else body
    )
    return joint, mids


def _window_chain(
    roots: list,
    irs: list[IR],
    acts: tuple[int, ...],
    kinds: tuple[str, ...],
    first: Any,
) -> tuple[Any, list]:
    """Nest ``f_j(f_{j-1}(·))`` — the chain-family window body."""
    cur = first
    for j in range(1, len(irs) - 1):
        cur = _subst(roots[j], irs[j].inputs[acts[j]], cur)
    body = _subst(roots[-1], irs[-1].inputs[acts[-1]], cur)
    joint = (
        Op.make("add", cur, body)
        if kinds[-1].endswith("_wrapped")
        else body
    )
    return joint, []


def _distribute_over(term: Any, mid: Any) -> Any:
    """Distribute projections whose data operand IS the residual sum.

    ``linear(x + A(x), W[, b]) → linear(x, W) + linear(A(x), W[, b])``
    — bilinearity of the data slot (the bias rides one branch — exact).
    ``matmul`` gets the same treatment on whichever side carries the
    sum.  A *constructed* equality: the rule set has no ``linear``-slot
    distribute law, so this step is offered into the joint e-graph
    under a pointwise witness and gated by the pair's fp64 verify.
    ``mid`` is always an additive node by construction — the joint
    builders only ever pass residual-stream ``add`` nodes.
    """
    if not isinstance(term, Op):
        return term
    args = tuple(_distribute_over(a, mid) for a in term.args)
    if term.op == "linear" and args and args[0] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("linear", lhs, args[1]),
            Op.make("linear", rhs, *args[1:]),
        )
    if term.op == "matmul" and len(args) >= 2 and args[0] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("matmul", lhs, args[1]),
            Op.make("matmul", rhs, args[1]),
        )
    if term.op == "matmul" and len(args) >= 2 and args[1] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("matmul", args[0], lhs),
            Op.make("matmul", args[0], rhs),
        )
    return Op.make(term.op, *args, **term.attrs)


#: Term-level rule recipes the reify specs name.  Both are subsets of
#: the core ``FULL`` preset — the bilinearity / associativity / factor
#: family that composes projections (``compose``), and the diagonal
#: naturality family that folds scale gains into weights (``scale``).
_RECIPE_NAMES: dict[str, tuple[str, ...]] = {
    "compose": (
        "weight_factor_linear",
        "weight_distribute_linear",
        "right_factor_linear",
        "assoc_linear",
        "assoc_linear_bias",
        "assoc_linear_bias_rev",
        "weight_factor_matmul",
        "weight_distribute_matmul",
        "right_factor_matmul",
        "assoc_matmul",
        "assoc_matmul_rev",
        "linear_channel_scale",
        "linear_channel_scale_rev",
        "linear_row_scale",
        "linear_row_scale_rev",
        "naturality_scalar",
        "naturality_scalar_rev",
        "distribute_matmul_over_add",
        "factor_matmul",
        "right_distribute_matmul",
        "comm_add",
        "assoc_add",
        "comm_mul",
        "id_add",
        "id_mul",
        "sub_to_add",
        "double_neg",
    ),
    "scale": (
        "linear_channel_scale",
        "linear_channel_scale_rev",
        "linear_row_scale",
        "linear_row_scale_rev",
        "naturality_scalar",
        "naturality_scalar_rev",
        "weight_factor_linear",
        "assoc_linear",
        "comm_mul",
        "assoc_mul",
        "id_mul",
    ),
    "tie": (),
}

_RECIPE_RULESETS: dict[str, Any] = {}


def _recipe_rules(name: str) -> Any:
    """Materialise the named term-level recipe as a ``RuleSet``."""
    rs = _RECIPE_RULESETS.get(name)
    if rs is None:
        rs = laws.FULL.named(*_RECIPE_NAMES[name])
        _RECIPE_RULESETS[name] = rs
    return rs


def _witness_offer(
    eg: EGraph, eid: int, term: Any, offered: tuple
) -> None:
    """Merge one constructed equality into the joint's e-class.

    ``offered`` is ``(term, law_text)`` plus an optional kwargs dict
    (``note`` / ``error_bound`` / ``bound_norm`` — the KV-latent
    offer certifies its factorisation residual); identity offers are
    skipped.
    """
    term_o, law_text = offered[0], offered[1]
    if term_o is term:
        return
    kw = dict(offered[2]) if len(offered) > 2 else {}
    kw.setdefault(
        "note",
        "residual_absorb: data-slot bilinearity over the residual add",
    )
    eg._offer_witness(
        eid,
        rhs_term=term_o,
        lhs_term=term,
        provenance="morphism_reify",
        law=law_text,
        **kw,
    )


def _saturate(
    term: Any,
    rules: Any,
    *,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
    offers: list | None = None,
    node_offers: list | None = None,
) -> tuple[EGraph, int]:
    """One joint e-graph: intern, offer constructed members, saturate.

    ``offers`` are ``(term, law_text)`` constructed equalities (the
    residual-distribute step) merged into the joint's e-class under a
    pointwise witness — the documented non-local-pass ritual (the
    pairing/tying passes use ``EGraph._offer_witness`` the same way);
    they replay in certificates like every other witnessed merge and
    the pair verify gates the assertion.  A third element, when
    present, is a kwargs dict for the witness (``note`` /
    ``error_bound`` / ``bound_norm`` — the KV-latent offer certifies
    its factorisation residual).  ``node_offers`` are
    ``(node, expanded, law_text)`` subterm equalities merged into the
    node's own e-class — the fused-norm unfold that lets the
    diagonal-naturality laws see a gain the fused op carries as an
    operand.  Returns ``(eg, root_eid)``.
    """
    eg = EGraph()
    eid = eg.add_term(term)
    for offered in offers or ():
        _witness_offer(eg, eid, term, offered)
    for node, expanded, law_text in node_offers or ():
        eg._offer_witness(
            eg.add_term(node),
            rhs_term=expanded,
            lhs_term=node,
            provenance="morphism_reify",
            law=law_text,
            note="morphism-level subterm expansion",
        )
    if len(rules):
        budgets = (
            {
                r.name: symmetry_budget
                for r in rules.tagged(_law_tags.EXPANSIVE)
            }
            if symmetry_budget is not None
            else None
        )
        eg.run(
            rules,
            eid,
            max_iterations=max_iterations,
            max_nodes=max_enodes,
            rule_budgets=budgets,
        )
    return eg, eid


def _ins_list(inputs: Iterable[Var] | None, var: Var) -> list:
    """Return the lowering's input list — ``[var]`` in the single-input case."""
    return list(inputs) if inputs is not None else [var]


def _lower_term(
    term: Any,
    var: Var,
    params: dict,
    leaves: dict,
    sink: Sink,
    inputs: Iterable[Var] | None = None,
) -> Any:
    """Lower one term through the sink — over ``inputs`` vars.

    ``inputs`` defaults to ``[var]`` (the historical single-input
    spelling); a multi-input joint passes the full var list so the
    lowered module's positional binding matches the block's call.
    """
    ins = _ins_list(inputs, var)
    ir = IR(
        root=term,
        inputs=ins,
        input_names={v.name for v in ins},
        params=params,
    )
    return sink.lower(ir, leaves)


def _verify_pair(
    sink: Sink,
    ref_term: Any,
    opt_term: Any,
    var: Var,
    params: dict,
    leaves: dict,
    args: tuple,
    rtol: float,
    inputs: Iterable[Var] | None = None,
) -> Any:
    """Verify the reified term against the un-rewritten joint."""
    ref = _lower_term(ref_term, var, params, leaves, sink, inputs)
    opt = _lower_term(opt_term, var, params, leaves, sink, inputs)
    return sink.verify(ref, opt, args, rtol=rtol)


def _slot_filler(
    sink: Sink,
    var: Var,
    mode: str,
    inputs: Iterable[Var] | None = None,
) -> Any:
    """Build the B-slot filler for a consumed pair: id or exact zero.

    Both are lowered terms — backend-neutral: ``x`` evaluates to its
    input; ``x * 0`` evaluates to a zero of the input's shape (the
    composer's ``_Zero`` uses ``zeros_like`` — equivalent on finite
    inputs, which is what the verify gate runs).  ``inputs`` widens
    the filler's call signature for multi-input slots — the context
    args are simply ignored.
    """
    root = (
        Op.make("mul", var, Const(0))
        if mode.endswith("_wrapped")
        else var
    )
    ins = _ins_list(inputs, var)
    ir = IR(
        root=root,
        inputs=ins,
        input_names={v.name for v in ins},
        params={},
    )
    return sink.lower(ir, {})


def _rec_act(graph: MorphismGraph, name: str) -> int:
    """Return the record's activation input index (0 when unlifted)."""
    sig = graph.sig(name)
    return _act_index(sig.inputs) if sig is not None else 0


def _intra_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve an intra (single-node) match — its own IR as joint."""
    rec = graph.record(match.nodes[0])
    if rec.ir is None:
        return None, "opaque node"
    return (
        (
            rec.ir.root,
            rec.ir.inputs[_rec_act(graph, match.nodes[0])],
            (),
            rec.ir.params,
            rec.leaves,
            rec.args,
            tuple(rec.ir.inputs),
        ),
        None,
    )


def _window_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a ≥3-block window match — the n-ary joint term."""
    spec = match.reify
    recs = [graph.record(n) for n in match.nodes]
    if len(spec.kinds) != len(recs) - 1 or len(recs) < 2:
        return None, "malformed window spec"
    if any(r.ir is None for r in recs):
        return None, "opaque node"
    acts = tuple(_rec_act(graph, n) for n in match.nodes)
    ctx_maps: list[dict | None] = []
    for r, act_j in zip(recs[1:], acts[1:], strict=True):
        cmap, why = _ctx_var_map(recs[0], r, act_j)
        if cmap is None:
            return None, why
        ctx_maps.append(cmap)
    joint, var, mids, params, leaves = _joint_parts_window(
        [r.ir for r in recs if r.ir is not None],
        recs,
        spec.kinds,
        acts=acts,
        ctx_maps=(None, *ctx_maps),
    )
    return (
        (
            joint,
            var,
            mids if spec.distribute else (),
            params,
            leaves,
            recs[0].args,
            tuple(recs[0].ir.inputs) if recs[0].ir else (),
        ),
        None,
    )


def _pair_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a boundary-pair match — the two-block joint term."""
    spec = match.reify
    rec_a, rec_b = (graph.record(n) for n in match.nodes)
    if rec_a.ir is None or rec_b.ir is None:
        return None, "opaque node"
    act_a = _rec_act(graph, match.nodes[0])
    act_b = _rec_act(graph, match.nodes[1])
    ctx_map, why = _ctx_var_map(rec_a, rec_b, act_b)
    if ctx_map is None:
        return None, why
    term, var, mid, params, leaves = _joint_parts(
        rec_a.ir,
        rec_b.ir,
        rec_a,
        rec_b,
        spec.mode,
        act_a=act_a,
        act_b=act_b,
        ctx_map=ctx_map,
    )
    return (
        (
            term,
            var,
            (mid,) if spec.distribute else (),
            params,
            leaves,
            rec_a.args,
            tuple(rec_a.ir.inputs),
        ),
        None,
    )


def _resolve_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a match to its joint term plus lowering context.

    Returns ``((term, var, mids, params, leaves, args, inputs),
    None)`` — the joint program over the first block's activation
    variable, the additive nodes to distribute over (already gated by
    ``spec.distribute``, so empty unless the spec asks), the
    namespaced param/leaf tables, the first block's captured args,
    and its full input-var list for lowering — or
    ``(None, reason)`` for an honest decline.
    """
    spec = match.reify
    if spec.mode == "intra":
        return _intra_joint(match, graph)
    if spec.kinds:
        return _window_joint(match, graph)
    return _pair_joint(match, graph)


def _norm_unfolds(term: Any) -> list[tuple[Op, Any, str]]:
    """``(node, expanded, law)`` offers unfusing weighted norm ops.

    ``rms_norm(x, w)`` ≡ ``rms_norm(x) ∘ w`` — the fused op carries the
    affine gain as an operand, so no ``mul`` node exists for the
    diagonal-naturality laws (``linear_channel_scale``) to see.  The
    unfused member is the fused kernel's internal gain pass spelled
    pointwise: an exact equality asserted at morphism level, offered
    into the norm node's own e-class, and gated by the pair verify —
    the same witness ritual as :func:`_distribute_offers`, addressed
    at the subterm rather than the joint root.
    """
    out: list[tuple[Op, Any, str]] = []
    for n in _iter_ops(term):
        if (
            n.op == "rms_norm"
            and len(n.args) >= 2
            and _param_only(n.args[1])
        ):
            out.append(
                (
                    n,
                    Op.make(
                        "mul",
                        Op.make("rms_norm", n.args[0], **n.attrs),
                        n.args[1],
                    ),
                    "rms_norm(x, w) = rms_norm(x) ∘ w — the fused "
                    "op's gain pass spelled pointwise; morphism-level "
                    "assertion, gated by the pair verify",
                )
            )
    return out


def _distribute_offers(term: Any, mids: tuple) -> list:
    """Build the constructed bilinear steps, progressive over each mid.

    Expands every projection whose data operand IS one of the
    additive mid nodes — outermost first, so a window's nested
    stream fully unfolds.  Each progressively-distributed form is a
    separate exact-equality offer merged at the joint's e-class;
    the pair verify gates the whole assertion once.
    """
    dist = term
    offers = []
    for m_node in reversed(mids):
        nxt = _distribute_over(dist, m_node)
        if nxt is not dist:
            offers.append(
                (
                    nxt,
                    "bilinearity of the projection's data slot "
                    "over the residual add — morphism-level "
                    "assertion, gated by the pair verify",
                )
            )
            dist = nxt
    return offers


def _window_reps(
    best: Any,
    match: MorphismMatch,
    var: Var,
    inputs: Iterable[Var],
    graph: MorphismGraph,
    params: dict,
    leaves: dict,
    sink: Sink,
) -> dict[str, Any]:
    """Per-slot replacements for a grafted ≥3-block window.

    The fused term lands at the first slot — a ``best - x`` delta
    when the family wraps the first block's output in a residual
    add.  Each later slot's filler follows the wire INTO it: a
    wrapped consumption takes the exact-zero addend, a plain one
    the identity passthrough — same convention as the pair slots.
    Every slot is lowered over *its own* block's input vars, so a
    multi-input member's filler keeps its call signature.
    """
    spec = match.reify
    a_term = (
        Op.make("sub", best, var)
        if spec.mode.startswith("residual")
        else best
    )
    reps = {
        match.nodes[0]: _lower_term(
            a_term, var, params, leaves, sink, inputs
        )
    }
    for j, name in enumerate(match.nodes[1:], start=1):
        ir_j = graph.record(name).ir
        ins_j = tuple(ir_j.inputs) if ir_j is not None else (var,)
        act_j = _rec_act(graph, name)
        var_j = ir_j.inputs[act_j] if ir_j is not None else var
        reps[name] = _slot_filler(sink, var_j, spec.kinds[j - 1], ins_j)
    return reps


def _res_stats(res: dict[str, Any]) -> dict[str, Any]:
    """Public fields of a reify result — minus ``reps``/``_terms``.

    The graft record's private ``_``-prefixed payload (the per-slot
    reified bodies the diagram-state search writes back) is machinery,
    not stats.
    """
    return {
        k: v
        for k, v in res.items()
        if k != "reps" and not k.startswith("_")
    }


def _reify_terms(
    match: MorphismMatch,
    spec: ReifySpec,
    best: Any,
    var: Var,
    graph: MorphismGraph,
    params: dict,
    leaves: dict,
) -> dict[str, tuple]:
    """Per-slot reified bodies for a grafted match.

    Parallel to the ``reps`` construction: the first slot carries the
    reified term (``best``, or ``best - x`` when a residual wrap puts
    the first block's output in an additive slot); each consumed
    member carries the filler its rep computes — the exact-zero
    ``mul(var, 0)`` under a wrapped wire, the identity ``var`` under a
    plain one.  The diagram-state search (plan 0013 stage 3) writes
    these back as the members' current bodies when composing moves.
    """
    if spec.mode == "intra":
        return {match.nodes[0]: (best, params, leaves)}
    a_term = (
        Op.make("sub", best, var)
        if spec.mode.startswith("residual")
        else best
    )
    terms: dict[str, tuple] = {match.nodes[0]: (a_term, params, leaves)}
    for j, name in enumerate(match.nodes[1:], start=1):
        ir_j = graph.record(name).ir
        var_j = (
            ir_j.inputs[_rec_act(graph, name)]
            if ir_j is not None
            else var
        )
        kind = spec.kinds[j - 1] if spec.kinds else spec.mode
        fill = (
            Op.make("mul", var_j, Const(0))
            if kind.endswith("_wrapped")
            else var_j
        )
        terms[name] = (fill, {}, {})
    return terms


def _reify(
    match: MorphismMatch,
    graph: MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
) -> dict[str, Any]:
    """Execute one morphism rewrite; return the graft record or a decline.

    Term-level reification: build the joint IR per the boundary mode,
    run the recipe (distribute offer → rule saturation → extraction),
    cost-gate against the un-rewritten joint, verify through the sink,
    and produce the per-slot replacement modules.
    """
    spec = match.reify
    if spec.mode == "tie" or spec.share:
        return _reify_tie(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
        )
    if spec.mode == "family":
        # The KV-latent rewrite lives in the sibling module (its
        # factorisation machinery is substantial); resolved lazily
        # to keep the modules acyclic.
        from catopt_orchestrator.morphisms_kv import _reify_family

        return _reify_family(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            symmetry_budget=symmetry_budget,
        )
    if spec.mode == "cse":
        # The cross-block shared-subterm rewrite lives in the
        # sibling module; resolved lazily like ``family``.
        from catopt_orchestrator.crossblock_cse import _reify_cse

        return _reify_cse(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            symmetry_budget=symmetry_budget,
        )
    resolved, decline = _resolve_joint(match, graph)
    if resolved is None:
        return {"status": "declined", "reason": decline}
    term, var, mids, params, leaves, args, inputs = resolved

    offers = _distribute_offers(term, mids)
    eg, eid = _saturate(
        term,
        _recipe_rules(spec.rules),
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        symmetry_budget=symmetry_budget,
        offers=offers,
        # The scale recipe folds diagonal gains into weights — unfuse
        # the fused-norm spellings so their gain operand is reachable.
        node_offers=(
            _norm_unfolds(term) if spec.rules == "scale" else None
        ),
    )
    best = eg.extract_best(eid, cost_fn)
    base_cost = dag_cost(term, cost_fn)
    best_cost = dag_cost(best, cost_fn)
    info: dict[str, Any] = {
        "cost_before": base_cost,
        "cost_after": best_cost,
        # DAG-aware reprs: a window joint shares its stream nodes
        # across every later block body, and ``best`` can share
        # extracted subterms — ``op_repr``'s tree expansion explodes
        # exponentially on exactly the stats strings.
        "joint": op_repr_dag(term),
        "reified": op_repr_dag(best),
    }
    if not best_cost < base_cost:
        return {
            "status": "declined",
            "reason": "no_improvement",
            **info,
        }
    vr = _verify_pair(
        sink,
        term,
        best,
        var,
        params,
        leaves,
        args,
        verify_tol,
        inputs,
    )
    info["rel_diff"] = vr.max_rel
    if not vr.passed:
        return {
            "status": "declined",
            "reason": f"reify verify failed: {vr.max_rel:.3e}",
            **info,
        }
    if spec.mode == "intra":
        reps = {
            match.nodes[0]: _lower_term(
                best, var, params, leaves, sink, inputs
            )
        }
    elif spec.kinds:
        reps = _window_reps(
            best, match, var, inputs, graph, params, leaves, sink
        )
    else:
        a_name, b_name = match.nodes
        a_term = (
            Op.make("sub", best, var)
            if spec.mode.startswith("residual")
            else best
        )
        ir_b = graph.record(b_name).ir
        ins_b = tuple(ir_b.inputs) if ir_b is not None else (var,)
        var_b = (
            ir_b.inputs[_rec_act(graph, b_name)]
            if ir_b is not None
            else var
        )
        reps = {
            a_name: _lower_term(
                a_term, var, params, leaves, sink, inputs
            ),
            b_name: _slot_filler(sink, var_b, spec.mode, ins_b),
        }
    terms = _reify_terms(match, spec, best, var, graph, params, leaves)
    return {
        "status": "grafted",
        "reps": reps,
        "_terms": terms,
        **info,
    }


def _reify_tie(
    match: MorphismMatch,
    graph: MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
) -> dict[str, Any]:
    """Reify a weight-tying match: joint e-graph, exact share, verify.

    Interns the involved blocks' (namespaced) terms into one e-graph
    and runs the value-exact :func:`share_duplicate_params` pass.  The
    candidate tie the signature detected is real only when the pass
    merges names — a shape+stem coincidence that does not share
    *values* declines before lowering.  Each surviving block re-lowers
    on its extracted term and verifies against its original module on
    the captured input.
    """
    eg = EGraph()
    recs = [graph.record(n) for n in match.nodes]
    leaves: dict[str, Any] = {}
    params: dict[str, Param] = {}
    vars_: dict[str, Var] = {}
    ins_: dict[str, tuple] = {}
    eids: dict[str, int] = {}
    for r in recs:
        if r.ir is None:
            return {"status": "declined", "reason": "opaque node"}
        pre = _ns_prefix(r.name)
        eids[r.name] = eg.add_term(_prefix_params(r.ir.root, pre))
        params.update(
            {
                pre + k: Param(pre + k, p.typ)
                for k, p in r.ir.params.items()
            }
        )
        leaves.update({pre + k: v for k, v in r.leaves.items()})
        vars_[r.name] = r.ir.inputs[_rec_act(graph, r.name)]
        ins_[r.name] = tuple(r.ir.inputs)
    groups = share_duplicate_params(eg, leaves)
    if not groups:
        return {"status": "declined", "reason": "no_tied_values"}
    reps: dict[str, Any] = {}
    rels: dict[str, float] = {}
    terms: dict[str, tuple] = {}
    for r in recs:
        extracted = eg.extract_best(eids[r.name], cost_fn)
        # The merged param may be the *other* block's name — lower with
        # the joint tables so the shared canonical name resolves.
        opt = _lower_term(
            extracted,
            vars_[r.name],
            params,
            leaves,
            sink,
            ins_[r.name],
        )
        vr = sink.verify(r.module, opt, r.args, rtol=verify_tol)
        rels[r.name] = vr.max_rel
        if not vr.passed:
            return {
                "status": "declined",
                "reason": f"tie verify failed on {r.name}: "
                f"{vr.max_rel:.3e}",
            }
        reps[r.name] = opt
        terms[r.name] = (extracted, params, leaves)
    return {
        "status": "grafted",
        "reps": reps,
        "_terms": terms,
        "tied": groups,
        "rel_diff": max(rels.values()),
    }


# ---------------------------------------------------------------------------
#  The strategy + entry point
# ---------------------------------------------------------------------------


class MorphismSearch:
    """The morphism-level :class:`~catopt_core.ports.Strategy`.

    ``lift → match → reify → graft``: lift the model to a
    :class:`MorphismGraph`, fire the signature laws, reify each
    surviving match to a verified IR program, and graft the results
    into a parameter-sharing clone.  Blocks untouched by a grafted
    rewrite get the ordinary per-block search+lower pass
    (``optimize_rest``) — morphism transforms compose *over* the
    per-block pipeline, not instead of it.

    Configuration: ``laws`` (the morphism law family),
    ``block_pred`` (block selection), ``verify_tol`` (the fp gate),
    ``joint_max_iterations`` / ``joint_max_enodes`` /
    ``symmetry_budget`` (the joint e-graph bounds), and
    ``optimize_rest`` (per-block fallback for untouched blocks).
    ``optimize`` kwargs (``rules``, ``cost_fn``, ``verbose``, …)
    forward to the per-block searches — including ``error_budget`` /
    ``detect_specials`` / ``detect_factors``, where ``error_budget``
    additionally arms budget-aware morphism laws left at
    ``budget=None`` (e.g. ``KVLatentShare``).
    """

    name = "morphism"

    def __init__(
        self,
        *,
        laws: Iterable[MorphismLaw] | None = None,
        block_pred: Any = None,
        optimize_rest: bool = True,
        verify_tol: float = 1e-4,
        joint_max_iterations: int = 8,
        joint_max_enodes: int = 50_000,
        symmetry_budget: int | None = 512,
    ) -> None:
        """Store the strategy configuration."""
        self.laws = (
            tuple(laws) if laws is not None else DEFAULT_MORPHISM_LAWS
        )
        self.block_pred = block_pred
        self.optimize_rest = optimize_rest
        self.verify_tol = verify_tol
        self.joint_max_iterations = joint_max_iterations
        self.joint_max_enodes = joint_max_enodes
        self.symmetry_budget = symmetry_budget

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Lift, match, reify, graft — end to end through the ports."""
        mod, stats = _optimize_morphisms(
            model,
            x,
            optimizer=optimizer,
            laws=self.laws,
            block_pred=self.block_pred,
            optimize_rest=self.optimize_rest,
            verify_tol=self.verify_tol,
            joint_max_iterations=self.joint_max_iterations,
            joint_max_enodes=self.joint_max_enodes,
            symmetry_budget=self.symmetry_budget,
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


def _arm_law_budgets(
    laws: Iterable[MorphismLaw], error_budget: float | None
) -> tuple[MorphismLaw, ...]:
    """Wire ``error_budget`` into budget-aware morphism laws.

    A law carrying a ``budget`` attribute left at ``None`` (e.g.
    :class:`~catopt_orchestrator.morphisms_kv.KVLatentShare`) inherits
    the call-time ``error_budget`` — its certified-approximate mode
    arms; a law configured with its own budget keeps it.  Armed laws
    are shallow copies: the caller's configured objects are never
    mutated.  ``error_budget=None`` returns the laws untouched.
    """
    import copy

    if error_budget is None:
        return tuple(laws)
    out: list[MorphismLaw] = []
    for law in laws:
        if getattr(law, "budget", 0.0) is None:
            armed: Any = copy.copy(law)
            armed.budget = error_budget
            out.append(armed)
        else:
            out.append(law)
    return tuple(out)


def _rest_example(rec: _BlockRecord) -> Any:
    """Return the per-block search input — the full args tuple when multi-input."""
    return rec.args[0] if len(rec.args) == 1 else rec.args


def _sig_dict(sig: BlockSig) -> dict[str, Any]:
    """Serialize a signature for the stats report."""
    return {
        "in_projs": [w.name for w in sig.in_projs],
        "out_proj": [w.name for w in sig.out_proj],
        "tables": [w.name for w in sig.tables],
        "norm": sig.norm.kind,
        "norm_affine": sig.norm.affine,
        "norm_pre": sig.norm.pre,
        "act": list(sig.act),
        "residual": sig.residual,
        "shape": sig.shape,
        "inputs": [
            {
                "index": i.index,
                "name": i.name,
                "kind": i.kind,
                "role": i.role,
            }
            for i in sig.inputs
        ],
    }


def _fallback_cost_kw(cost_fn: CostFn | None) -> dict[str, Any]:
    """Per-block search extras — forward an explicit ``cost_fn``.

    An unset one stays unset so the block search keeps the
    optimizer's own cost resolution (criteria / executor-aware
    default); ``cost_fn`` itself only defaults to ``flops_cost`` for
    the joint reify pricing.
    """
    return {} if cost_fn is None else {"cost_fn": cost_fn}


def _optimize_morphisms(
    model: Any,
    x: Any,
    *,
    optimizer: Any,
    laws: Iterable[MorphismLaw],
    block_pred: Any,
    optimize_rest: bool,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
    error_budget: float | None = None,
    detect_specials: bool = False,
    detect_factors: bool = False,
    cost_fn: CostFn | None = None,
    rules: Any = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verbose: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Lift, match, reify, recompose — the morphism pipeline.

    ``error_budget`` / ``detect_specials`` / ``detect_factors``
    forward to each untouched block's per-block
    :meth:`Optimizer.search` (same contract as
    :class:`~catopt_orchestrator.optimize.Compositional`), and
    ``error_budget`` additionally arms budget-aware morphism laws
    whose own ``budget`` is unset (:func:`_arm_law_budgets` —
    ``KVLatentShare(budget=)``'s certified-approximate mode).  A
    bounded per-block delivery verifies against the propagated
    *output* bound (:func:`_propagate_bounds` — the certificate's
    weight-space bound amplified by the measured site-input norm),
    identical units to the measured ``max|Δy|``.

    Returns ``(model, stats)`` — ``stats["matches"]`` carries the
    per-match verdicts (``grafted`` / ``declined`` / ``skipped`` plus
    the cost/rel-diff record), ``stats["blocks"]`` the per-block
    fallback reports, ``stats["wires"]`` the boundary classification,
    and ``stats["end_to_end"]`` the whole-model equivalence check.
    """
    # Local import, same convention as ``optimize_morphisms`` below:
    # the bound helpers live in the pipeline module.
    from catopt_orchestrator.optimize import (
        _block_bound_gate,
        _bounded_gate,
        _rel_bound,
    )

    t_start = time.time()
    composer = optimizer.composer
    if composer is None:
        raise TypeError(
            "MorphismSearch needs a Composer port — pass composer= "
            "(or backend=) to Optimizer"
        )
    source = optimizer.source
    sink = optimizer.sink
    # ``cost_fn`` prices the joint reify; the per-block fallback only
    # inherits it when the caller supplied one — otherwise the block
    # search keeps the optimizer's own cost resolution.
    fallback_cost = _fallback_cost_kw(cost_fn)
    if cost_fn is None:
        cost_fn = flops_cost
    laws = _arm_law_budgets(laws, error_budget)

    graph = lift_graph(
        model,
        x,
        source=source,
        composer=composer,
        block_pred=block_pred,
    )
    matches = [m for law in laws for m in law.match(graph)]
    if verbose:
        log.info(
            "[Morphism] %s; matches: %s",
            graph,
            [(m.law, m.nodes) for m in matches],
        )

    replacements: dict[str, Any] = {}
    match_stats: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    consumed_by: dict[str, str] = {}
    morphism_fires: dict[str, int] = {}
    for m in matches:
        key = f"{m.law}:{'+'.join(m.nodes)}"
        if any(n in consumed for n in m.nodes):
            match_stats[key] = {
                "status": "skipped",
                "reason": "node already rewritten",
            }
            continue
        try:
            res = _reify(
                m,
                graph,
                sink=sink,
                cost_fn=cost_fn,
                verify_tol=verify_tol,
                max_iterations=joint_max_iterations,
                max_enodes=joint_max_enodes,
                symmetry_budget=symmetry_budget,
            )
        except Exception as e:
            match_stats[key] = {
                "status": "declined",
                "reason": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            continue
        match_stats[key] = {
            "boundary": m.boundary,
            "detail": m.detail,
            **_res_stats(res),
        }
        if res["status"] == "grafted":
            replacements.update(res["reps"])
            for n in m.nodes:
                consumed.add(n)
                consumed_by[n] = m.law
            morphism_fires[m.law] = morphism_fires.get(m.law, 0) + 1
            if verbose:
                log.info("[Morphism] %s: grafted (%s)", key, m.detail)

    block_reports: dict[str, dict[str, Any]] = {}
    if optimize_rest:
        for node in graph.nodes:
            name = node.name
            if name in consumed:
                block_reports[name] = {
                    "status": "rewritten",
                    "law": consumed_by[name],
                }
                continue
            rec = graph.record(name)
            rep: dict[str, Any] = {}
            block_reports[name] = rep
            if not rec.args or rec.ir is None:
                rep["status"] = "skipped"
                rep["reason"] = rec.note
                continue
            try:
                res_s = optimizer.search(
                    rec.module,
                    _rest_example(rec),
                    rules=rules,
                    max_iterations=max_iterations,
                    max_enodes=max_enodes,
                    max_memory_mb=max_memory_mb,
                    error_budget=error_budget,
                    detect_specials=detect_specials,
                    detect_factors=detect_factors,
                    verbose=verbose,
                    **fallback_cost,
                )
                lr = optimizer.lower(
                    res_s,
                    _rest_example(rec),
                    runner=IdentityRunner(),
                    verify=False,
                    verbose=verbose,
                )
                # Under a bounded search the certificate's bound is in
                # weight space: ``_block_bound_gate`` propagates it to
                # output units on the captured input and the gate
                # compares measured ``max|Δy|`` against that — the same
                # contract as the compositional driver's verify.
                vr, out_s = _block_bound_gate(
                    sink,
                    rec.module,
                    lr.module,
                    rec.args,
                    verify_tol,
                    lr.stats,
                    res_s,
                )
                rep["rel_diff"] = vr.max_rel
                passed = _bounded_gate(vr, out_s)
                if not passed:
                    rep["status"] = "failed"
                    rep["reason"] = (
                        f"block verify failed: {vr.max_rel:.3e} "
                        f"(accepted bound "
                        f"{_rel_bound(out_s, vr):.3e})"
                    )
                    continue
                replacements[name] = lr.module
                rep["status"] = "optimized"
                rep["stats"] = lr.stats
            except Exception as e:
                rep["status"] = "failed"
                rep["error"] = f"{type(e).__name__}: {e}"

    in_place = False
    try:
        new_model = composer.clone_sharing(model)
    except Exception:
        new_model = model
        in_place = True
        replacements = {}
    composer.graft(new_model, replacements)

    stats: dict[str, Any] = {
        "morphism": True,
        "n_blocks": len(graph.nodes),
        "n_lifted": sum(1 for n in graph.nodes if not n.opaque),
        "sigs": {
            n.name: (_sig_dict(n.sig) if n.sig is not None else None)
            for n in graph.nodes
        },
        "wires": [(w.src, w.dst, w.kind) for w in graph.wires],
        "matches": match_stats,
        "morphism_fires": morphism_fires,
        "n_rewritten": len(consumed),
        "blocks": block_reports,
        "in_place": in_place,
        "shared_params": not in_place,
    }
    if in_place:
        stats["end_to_end"] = {
            "skipped": "in_place",
            "reason": "param-sharing clone failed — model unmodified",
        }
    else:
        args2 = x if isinstance(x, tuple) else (x,)
        try:
            vr = sink.verify(model, new_model, args2)
            stats["end_to_end"] = {
                "max_abs_diff": vr.max_abs,
                "max_rel_diff": vr.max_rel,
            }
        except Exception as e:
            stats["end_to_end"] = {"error": f"{type(e).__name__}: {e}"}
    stats["wall_time_s"] = time.time() - t_start
    return new_model, stats


def optimize_morphisms(
    model: Any,
    x: Any,
    *,
    backend: Any = None,
    source: Source | None = None,
    sink: Sink | None = None,
    composer: Composer | None = None,
    meter: Meter | None = None,
    strategy: Any = None,
    **kw: Any,
) -> LowerResult:
    """Run the morphism pipeline on *model* — the function entry point.

    Assembles an :class:`~catopt_orchestrator.optimize.Optimizer` from
    the given ports (``backend`` bundle or explicit
    ``source``/``sink``/``composer``/``meter`` — the composer port is
    required for block lifting) and runs ``optimize`` under
    :class:`MorphismSearch`.  ``strategy`` overrides the search
    configuration entirely (``MorphismSearch(laws=..., ...)`` selects
    the morphism law family); remaining ``**kw`` (``rules``,
    ``cost_fn``, ``verbose``, …) reach the per-block fallback
    searches.
    """
    from catopt_orchestrator.optimize import Optimizer

    opt = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        composer=composer,
        meter=meter,
    )
    strat = strategy if strategy is not None else MorphismSearch()
    return opt.optimize(model, x, strategy=strat, **kw)
