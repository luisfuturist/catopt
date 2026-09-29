"""Routed-MoE pairing regression tests.

Pins the e2e_models2 ``moe_stack`` finding: stacked expert parameters
``W (E, out, in)`` export per expert as ``select(W, dim=0, index=e)``,
so the pairing pass's fused weight was spelled
``concat(select(W,0), …, select(W,E-1))`` — a RUNTIME concat over the
view/extraction family that ``IRModule._fold_weight_chains``
deliberately declines to fold (dedup re-materialisation).  The honest
cost model billed that concat every call, forced extraction lost by
~1.7M on one block, and all 40 offered groups were declined
(``paired_extract=0``).

The pass now recognises tiles that jointly cover a stacked base and
spells the fused piece ``reshape(base)`` — a free view that folds to
one fused ``Param`` at lowering — while ``cost._folds_to_param``
prices the pure-view family the way the lowerer actually folds it.
Per-member ``select`` tiles that only PARTIALLY cover a base keep the
runtime concat (and the decline is recorded in stats).

The residual after pairing — the per-expert down projections
``Σ_e g_e ⊙ linear(h_e, W[e])`` over DISTINCT gated inputs — is not a
pairing pattern (no shared domain) but it IS a grouped GEMM:
``batch_tiled_expert_sums`` offers each expert-sum class the batched
member ``reshape(sum(mul(bmm(stack h, W.mT), stack g), 0))`` (the
stack flattens to rank-3 because ``torch.matmul`` right-aligns batch
dims).  ``transpose(W)`` folds to a fused param at lowering, and
extraction picks the member only when honestly cheaper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, Param, TensorType, Var
from catopt_core.laws.pairing import (
    batch_tiled_expert_sums,
    pair_shared_input_linears,
)

from catopt_orchestrator import Optimizer

from catopt_torch.backend import TorchBackend


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _ops(term, seen=None):
    """All distinct Op nodes in a term DAG."""
    if seen is None:
        seen = set()
    if isinstance(term, Op) and term not in seen:
        seen.add(term)
        for a in term.args:
            _ops(a, seen)
    return seen


class RoutedMoEFFN(nn.Module):
    """Mixtral-lite routed MoE FFN in dense-eval form.

    rms → softmax router → top-k scatter mask → renormalised gates;
    ALL experts run (stacked ``(E, out, in)`` Parameters indexed
    ``w[e]`` — the ``select``-tile pattern) and gate-sum.  The gate
    multiply is data-dependent, so the expert sum must NOT fold; the
    same-input up/gate projections are the pairing targets.
    """

    def __init__(self, dim=64, dff=128, n_experts=4, top_k=2):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.rms = nn.Parameter(torch.ones(dim))
        self.router = nn.Linear(dim, n_experts, bias=False)
        g1, g2 = dim**-0.5, dff**-0.5
        self.w1 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w3 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w2 = nn.Parameter(torch.randn(n_experts, dim, dff) * g2)

    def forward(self, x):
        n = F.rms_norm(x, (x.shape[-1],), self.rms, 1e-5)
        w = F.softmax(self.router(n), dim=-1)
        tv, ti = w.topk(self.k, dim=-1)
        g = torch.zeros_like(w).scatter(-1, ti, tv)
        g = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
        out = None
        for e in range(self.E):
            he = F.silu(F.linear(n, self.w1[e])) * F.linear(
                n, self.w3[e]
            )
            ye = F.linear(he, self.w2[e]) * g[..., e].unsqueeze(-1)
            out = ye if out is None else out + ye
        return x + out


class PartialTile(nn.Module):
    """Two linears sharing one input, weighted by ``select`` slices 0
    and 2 of a ``(4, o, d)`` stacked parameter — NOT a full tile, so
    the fused weight keeps the runtime concat and cost decides."""

    def __init__(self, dim=64, out=64, stack=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(stack, out, dim) * dim**-0.5)

    def forward(self, x):
        return F.linear(x, self.W[0]) + F.linear(x, self.W[2])


class SumExperts(nn.Module):
    """Every expert of a ``(E, o, d)`` stack applied to x and summed:
    ``weight_factor_matmul``'s ``linear(x, ΣWₑ)`` is strictly cheaper
    than the E-wide fused GEMM — pairing must decline honestly."""

    def __init__(self, dim=64, out=64, stack=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(stack, out, dim) * dim**-0.5)

    def forward(self, x):
        acc = None
        for e in range(self.W.shape[0]):
            y = F.linear(x, self.W[e])
            acc = y if acc is None else acc + y
        return acc


def test_pairing_select_tiles_emit_reshape():
    """Unit level: members weighted by dim-0 ``select`` tiles covering
    a shared stacked base fuse as ``reshape(base)``, not a runtime
    concat over non-folding views."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))
    base = _p("W", 4, 16, 8)
    for e in range(4):
        we = eg.add_term(Op.make("select", base, dim=0, index=e))
        eg.add_enode("linear", (x, we))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc = eg._classes[eg.find(lin.children[1])]
    reshapes = [n for n in wc.nodes if n.op == "reshape"]
    assert len(reshapes) == 1
    assert dict(reshapes[0].attrs)["shape"] == (4 * 16, 8)
    assert not any(n.op == "concat" for n in wc.nodes)
    # split sizes ride in fused order — select-index order here.
    assert dict(split.attrs)["sizes"] == (16, 16, 16, 16)


def test_select_tile_guard_paths():
    """``_select_tile`` declines malformed tiles — each guard arm maps
    to a select enode that cannot be a stacked-parameter slice.

    The member classes keep a plain Param leaf (so the cluster forms
    and the pass inspects them); the degenerate select enode sits
    beside it and must be skipped:
      * a select with a non-integer ``index`` attr;
      * a select over a class with no resolvable term (cyclic);
      * a select on ``dim != 0``;
      * a select over a base with unknown dims;
      * a select enode with the wrong arity.
    """
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))

    def member(param_name):
        """A (16,8) weight class that also carries a weird select."""
        return eg.add_term(_p(param_name, 16, 8))

    # non-integer index
    wa = member("wa")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("b1", 4, 16, 8)),),
        {"dim": 0, "index": "e"},
    )
    eg.union(sel, wa)
    # unresolvable base (cyclic-only e-class)
    wb = member("wb")
    leaf = eg.add_term(_p("cyc", 4, 16, 8))
    neg = eg.add_enode("neg", (leaf,))
    eg.union(neg, leaf)
    cyc = eg.find(leaf)
    for n in list(eg._classes[cyc].nodes):
        if n.op == "leaf":
            eg._classes[cyc].nodes.discard(n)
    sel = eg.add_enode("select", (cyc,), {"dim": 0, "index": 0})
    eg.union(sel, wb)
    # dim != 0
    wc = member("wc")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("b3", 4, 16, 8)),),
        {"dim": 1, "index": 0},
    )
    eg.union(sel, wc)
    # base with an unknown extent
    wd = member("wd")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("bn", None, 16, 8)),),
        {"dim": 0, "index": 0},
    )
    eg.union(sel, wd)
    # wrong arity (a select enode is unary)
    we = member("we")
    sel = eg.add_enode(
        "select",
        (
            eg.add_term(_p("b5", 4, 16, 8)),
            eg.add_term(_p("b6", 4, 16, 8)),
        ),
        {"dim": 0, "index": 0},
    )
    eg.union(sel, we)
    # one ordinary member so a real group still forms
    wn = eg.add_term(_p("wn", 16, 8))
    for w in (wa, wb, wc, wd, we, wn):
        eg.add_enode("linear", (x, w))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc_class = eg._classes[eg.find(lin.children[1])]
    # No tile survived a guard — the fused weight is the runtime concat.
    assert any(n.op == "concat" for n in wc_class.nodes)
    assert not any(n.op == "reshape" for n in wc_class.nodes)


def test_pairing_partial_tile_keeps_runtime_concat():
    """Selects covering only SOME indices of a base can't reshape —
    the fused weight stays the honest runtime concat."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))
    base = _p("W", 4, 16, 8)
    for e in (0, 2):
        we = eg.add_term(Op.make("select", base, dim=0, index=e))
        eg.add_enode("linear", (x, we))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc = eg._classes[eg.find(lin.children[1])]
    assert any(n.op == "concat" for n in wc.nodes)
    assert not any(n.op == "reshape" for n in wc.nodes)


def test_moe_stacked_experts_pair_and_verify():
    """Routed MoE FFN: the w1/w3/router projections on the shared
    norm output pair into ONE fused GEMM (weights fold through
    ``reshape`` of the stacked params), and the gated expert sum
    batches into ONE grouped GEMM — ``sum_0(bmm(stack h, W2.mT)
    ⊙ stack g)`` — whose transposed weight folds to a fused Param.
    No serial per-expert ``linear`` survives.  The extracted module
    verifies against eager."""
    torch.manual_seed(0)
    m = RoutedMoEFFN().eval()
    x = torch.randn(2, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, max_iterations=12, verify=False, verbose=False)

    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is True
    with torch.no_grad():
        ref = m(x.clone())
        diff = (ref - opt(x.clone())).abs().max().item()
    assert diff < 1e-4

    nodes = _ops(opt._root)
    # One fused GEMM replaces the 9 same-input projections
    # (4*w1 + 4*w3 + router) and one grouped GEMM replaces the 4
    # select-weighted w2 linears — none survive serially.
    sel_lins = [
        t
        for t in nodes
        if t.op == "linear"
        and getattr(t.args[1], "op", None) == "select"
    ]
    assert not sel_lins
    # The grouped GEMM: matmul(reshape(stack(h_e)), w2.mT).
    mms = [t for t in nodes if t.op == "matmul"]
    assert len(mms) == 1
    assert mms[0].args[0].op == "reshape"
    assert mms[0].args[0].args[0].op == "stack"
    # The E per-expert gate muls collapse to ONE broadcast multiply;
    # the E activation muls (silu ⊙ gate-projection) are unchanged.
    assert sum(1 for t in nodes if t.op == "mul") == m.E + 1

    names = {n for n, _ in opt.named_parameters()}
    # All expert weights fold: up/gate/router into the shared GEMM,
    # the w2 stack through its transpose into a second fused param.
    assert "p_w1" not in names
    assert "p_w3" not in names
    assert "p_router_weight" not in names
    assert "p_w2" not in names
    assert sum(1 for n in names if n.startswith("fused_")) >= 2


def test_partial_tile_decline_is_recorded():
    """When forcing the shared GEMM honestly loses, the stats record
    the decline and the cost delta instead of silence."""
    torch.manual_seed(0)
    m = PartialTile().eval()
    x = torch.randn(4, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert stats.get("pairing_groups") == 1
    assert stats.get("paired_extract") is False
    assert stats.get("paired_delta") > 0
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4


def test_weight_factor_beats_pairing_decline():
    """A plain expert SUM admits ``linear(x, ΣWₑ)`` — strictly cheaper
    than the E-wide paired GEMM, so extraction declines the group and
    the verdict lands in stats."""
    torch.manual_seed(0)
    m = SumExperts().eval()
    x = torch.randn(4, 16, 64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, verify=False, verbose=False)

    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is False
    assert stats.get("paired_delta") > 0
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4


# ---------------------------------------------------------------------------
#  Expert-sum batching — unit-level offers and decline paths.
# ---------------------------------------------------------------------------


def _expert_sum_eg(gated=True, e=4, b=2, i=16, o=8, cover=None):
    """Build ``Σ_e [g_e ⊙] linear(h_e, select(W,0,e))`` in an e-graph.

    ``cover`` overrides the tile indices actually summed (default: all
    ``e`` tiles).  Returns ``(eg, root_eid, base_param)``.
    """
    eg = EGraph()
    base = _p("W", e, o, i)
    leaves = []
    for idx in range(e) if cover is None else cover:
        h = eg.add_term(_v(f"h{idx}", b, i))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        if gated:
            g = eg.add_term(_v(f"g{idx}", b, 1))
            lin = eg.add_enode("mul", (lin, g), {})
        leaves.append(lin)
    acc = leaves[0]
    for lin in leaves[1:]:
        acc = eg.add_enode("add", (acc, lin), {})
    return eg, acc, base


def _root_members(eg, root, op):
    return [n for n in eg._classes[eg.find(root)].nodes if n.op == op]


def test_batch_ungated_expert_sum_offered():
    """``Σ_e linear(h_e, W[e])`` gains the grouped-GEMM member
    ``reshape(sum(matmul(reshape(stack h), W.mT), 0))`` — no gate."""
    eg, root, _ = _expert_sum_eg(gated=False)
    n = batch_tiled_expert_sums(eg)
    assert n >= 1
    resh = _root_members(eg, root, "reshape")
    assert len(resh) == 1
    # reshape -> sum -> matmul -> reshape -> stack
    summ = eg._classes[eg.find(resh[0].children[0])]
    assert any(n.op == "sum" for n in summ.nodes)
    mm = next(
        n
        for cid in [
            eg.find(s.children[0]) for s in summ.nodes if s.op == "sum"
        ]
        for n in eg._classes[cid].nodes
        if n.op == "matmul"
    )
    flat = eg._classes[eg.find(mm.children[0])]
    stk = next(n for n in flat.nodes if n.op == "reshape")
    assert dict(stk.attrs)["shape"] == (4, 2, 16)
    st = eg._classes[eg.find(stk.children[0])]
    assert any(
        n.op == "stack" and len(n.children) == 4 for n in st.nodes
    )


def test_batch_gated_expert_sum_offered():
    """``Σ_e g_e ⊙ linear(h_e, W[e])`` gains the member with one
    broadcast gate multiply over the batched bmm output."""
    eg, root, _ = _expert_sum_eg(gated=True)
    assert batch_tiled_expert_sums(eg) >= 1
    resh = _root_members(eg, root, "reshape")
    assert len(resh) == 1
    summ = eg._classes[eg.find(resh[0].children[0])]
    muls = [
        n
        for cid in [
            eg.find(s.children[0]) for s in summ.nodes if s.op == "sum"
        ]
        for n in eg._classes[cid].nodes
        if n.op == "mul"
    ]
    assert len(muls) == 1
    # one mul operand is the bmm output, the other the stacked gates
    for ch in muls[0].children:
        ops = {n.op for n in eg._classes[eg.find(ch)].nodes}
        assert ops & {"matmul", "reshape"}


def test_batch_residual_leaf_rejoins_add():
    """``x + Σ_e leaf`` offers ``add(x, batched)`` — the non-expert
    residual folds back in through a rebuilt add."""
    eg, inner, _ = _expert_sum_eg(gated=False)
    x = eg.add_term(_v("x", 2, 8))
    root = eg.add_enode("add", (x, inner), {})
    assert batch_tiled_expert_sums(eg) >= 1
    # root class keeps its original add AND gains add(batched, x)
    adds = _root_members(eg, root, "add")
    resh_children = 0
    for a in adds:
        for c in a.children:
            if any(
                n.op == "reshape" for n in eg._classes[eg.find(c)].nodes
            ):
                resh_children += 1
    assert resh_children >= 1


def test_batch_partial_cover_declines():
    """Tiles {0,2} of a 4-expert base can't bmm off the stacked param —
    no offer (a runtime stack of selects would cost, not save)."""
    eg, root, _ = _expert_sum_eg(gated=False, cover=[0, 2])
    assert batch_tiled_expert_sums(eg) == 0
    assert not _root_members(eg, root, "reshape")


def test_batch_duplicate_tile_declines():
    """Two leaves on the same tile index can't form a full cover."""
    eg = EGraph()
    base = _p("W", 4, 8, 16)
    h0 = eg.add_term(_v("h0", 2, 16))
    h1 = eg.add_term(_v("h1", 2, 16))
    h2 = eg.add_term(_v("h2", 2, 16))
    w0 = eg.add_term(Op.make("select", base, dim=0, index=0))
    w0b = eg.add_term(Op.make("select", base, dim=0, index=0))
    w1 = eg.add_term(Op.make("select", base, dim=0, index=1))
    a = eg.add_enode("linear", (h0, w0), {})
    b = eg.add_enode("linear", (h1, w0b), {})
    c = eg.add_enode("linear", (h2, w1), {})
    eg.add_enode("add", (a, eg.add_enode("add", (b, c), {})), {})
    assert batch_tiled_expert_sums(eg) == 0


def test_batch_shape_mismatch_declines():
    """Heterogeneous expert inputs can't stack — no offer."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    hs = [eg.add_term(_v("h0", 2, 16)), eg.add_term(_v("h1", 4, 16))]
    ls = []
    for idx in range(2):
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        ls.append(eg.add_enode("linear", (hs[idx], we), {}))
    eg.add_enode("add", tuple(ls), {})
    assert batch_tiled_expert_sums(eg) == 0


def test_batch_unknown_or_low_rank_declines():
    """Unknown dims (``None``) and rank-1 inputs both decline — the
    flattened bmm needs a fully-known, rank >= 2 operand."""
    for shape in ((None, 16), (16,)):
        eg = EGraph()
        base = _p("W", 2, 8, 16)
        hs = [
            eg.add_term(_v("h0", *shape)),
            eg.add_term(_v("h1", *shape)),
        ]
        ls = []
        for idx in range(2):
            we = eg.add_term(Op.make("select", base, dim=0, index=idx))
            ls.append(eg.add_enode("linear", (hs[idx], we), {}))
        eg.add_enode("add", tuple(ls), {})
        assert batch_tiled_expert_sums(eg) == 0


def test_batch_mixed_gating_declines():
    """A sum of gated AND ungated leaves has no uniform batched form."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        if idx == 0:
            g = eg.add_term(_v("g0", 2, 1))
            lin = eg.add_enode("mul", (lin, g), {})
        leaves.append(lin)
    eg.add_enode("add", tuple(leaves), {})
    assert batch_tiled_expert_sums(eg) == 0


def test_batch_wrong_gate_shape_declines():
    """Gates that aren't per-token scalars (…, 1) can't broadcast
    against the flattened (E, N, o) bmm output — decline."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        g = eg.add_term(_v(f"g{idx}", 2, 8))  # full-width gate
        leaves.append(eg.add_enode("mul", (lin, g), {}))
    eg.add_enode("add", tuple(leaves), {})
    assert batch_tiled_expert_sums(eg) == 0


def test_batch_second_base_leftovers_rejoin():
    """A sum mixing one fully-covered base with another base's leaf
    batches the covered group and re-adds the stranger serially."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    other = _p("V", 4, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        leaves.append(eg.add_enode("linear", (h, we), {}))
    h3 = eg.add_term(_v("h3", 2, 16))
    w3 = eg.add_term(Op.make("select", other, dim=0, index=1))
    stranger = eg.add_enode("linear", (h3, w3), {})
    root = eg.add_enode(
        "add",
        (leaves[0], eg.add_enode("add", (leaves[1], stranger), {})),
        {},
    )
    assert batch_tiled_expert_sums(eg) >= 1
    # the offered member re-adds the un-covered leaf's class
    adds = _root_members(eg, root, "add")
    assert any(
        eg.find(ch) == eg.find(stranger)
        for a in adds
        for ch in a.children
    )


def test_batch_cyclic_sum_declines():
    """An add class referencing itself on the decomposition path is a
    residual cycle — folding it into the offer would build a
    self-referential member, so the offer is declined even though two
    real expert leaves are present."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        leaves.append(eg.add_enode("linear", (h, we), {}))
    v = eg.add_term(_v("v", 2, 8))
    inner = eg.add_enode("add", (v, v), {})
    outer = eg.add_enode("add", (inner, v), {})
    eg.union(inner, outer)
    cyc = eg.find(inner)
    # Drop the acyclic add(v,v) member so the class's only add enode
    # is the self-loop — a deterministic on-path cycle.
    for n in list(eg._classes[cyc].nodes):
        if n.op == "add" and all(eg.find(c) != cyc for c in n.children):
            eg._classes[cyc].nodes.discard(n)
    eg.add_enode(
        "add",
        (leaves[0], eg.add_enode("add", (leaves[1], cyc), {})),
        {},
    )
    assert batch_tiled_expert_sums(eg) == 0


def test_batch_leaf_classification_edges():
    """``_expert_leaf`` declines non-expert classes; ``_linear_tile``
    declines biased linears, non-tile weights, and non-rank-3 bases."""
    from catopt_core.laws.pairing import _expert_leaf, _linear_tile

    eg = EGraph()
    x = eg.add_term(_v("x", 2, 16))
    y = eg.add_term(_v("y", 2, 16))
    # plain mul/leaf classes are neither gated nor ungated experts
    plain_mul = eg.add_enode("mul", (x, y), {})
    assert _expert_leaf(eg, plain_mul) is None
    assert _linear_tile(eg, x) is None
    # bias (3-child) linear is skipped by the arity guard
    bias = eg.add_term(_p("b", 8))
    w = eg.add_term(_p("Wp", 8, 16))
    biased = eg.add_enode("linear", (x, w, bias), {})
    assert _linear_tile(eg, biased) is None
    # select tile of a rank-2 base is not a rank-3 expert stack
    w2d = _p("W2d", 4, 16)
    sel = eg.add_term(Op.make("select", w2d, dim=0, index=0))
    l2d = eg.add_enode("linear", (x, sel), {})
    assert _linear_tile(eg, l2d) is None
    # gate-first argument order still classifies as a gated leaf
    base = _p("W", 2, 8, 16)
    h = eg.add_term(_v("h", 2, 16))
    we = eg.add_term(Op.make("select", base, dim=0, index=0))
    lin = eg.add_enode("linear", (h, we), {})
    g = eg.add_term(_v("g", 2, 1))
    mul = eg.add_enode("mul", (g, lin), {})  # gate first
    leaf = _expert_leaf(eg, mul)
    assert leaf is not None and leaf[4] is not None
    # a mul enode with the wrong arity is skipped
    weird = eg.add_enode("mul", (x, y, g), {})
    assert _expert_leaf(eg, weird) is None


def test_batch_single_expert_and_addless_graphs_skip():
    """Sums with <2 expert leaves and classes with no add enode are
    skipped — nothing to batch."""
    eg = EGraph()
    base = _p("W", 1, 8, 16)
    h = eg.add_term(_v("h", 2, 16))
    we = eg.add_term(Op.make("select", base, dim=0, index=0))
    lin = eg.add_enode("linear", (h, we), {})
    z = eg.add_term(_v("z", 2, 8))
    eg.add_enode("add", (lin, z), {})  # 1 leaf + residual -> skip
    assert batch_tiled_expert_sums(eg) == 0
