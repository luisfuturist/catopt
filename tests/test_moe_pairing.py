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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt.egraph import EGraph
from catopt.ir import Op, Param, TensorType, Var
from catopt.laws.pairing import pair_shared_input_linears
from catopt.optimize import optimize_model


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
    ``reshape`` of the stacked params); the gate-weighted expert sum
    and the per-expert down projections — distinct inputs — do NOT
    fold.  The extracted module verifies against eager."""
    torch.manual_seed(0)
    m = RoutedMoEFFN().eval()
    x = torch.randn(2, 16, 64)
    opt, stats = optimize_model(m, x, verbose=False, max_iterations=12)
    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is True
    with torch.no_grad():
        ref = m(x.clone())
        diff = (ref - opt(x.clone())).abs().max().item()
    assert diff < 1e-4

    nodes = _ops(opt._root)
    # One fused GEMM replaces the 9 same-input projections
    # (4*w1 + 4*w3 + router); exactly E select-weighted linears
    # survive — the w2 down projections over distinct h_e inputs.
    sel_lins = [
        t
        for t in nodes
        if t.op == "linear"
        and getattr(t.args[1], "op", None) == "select"
    ]
    assert len(sel_lins) == m.E
    # The data-dependent gate multiplies are still runtime work.
    assert sum(1 for t in nodes if t.op == "mul") >= m.E

    names = {n for n, _ in opt.named_parameters()}
    # Stacked up/gate/router weights fold into fused GEMM params;
    # the per-expert down-projection stack survives untouched.
    assert "p_w1" not in names
    assert "p_w3" not in names
    assert "p_router_weight" not in names
    assert "p_w2" in names
    assert any(n.startswith("fused_") for n in names)


def test_partial_tile_decline_is_recorded():
    """When forcing the shared GEMM honestly loses, the stats record
    the decline and the cost delta instead of silence."""
    torch.manual_seed(0)
    m = PartialTile().eval()
    x = torch.randn(4, 16, 64)
    opt, stats = optimize_model(m, x, verbose=False)
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
    opt, stats = optimize_model(m, x, verbose=False)
    assert stats.get("pairing_groups", 0) >= 1
    assert stats.get("paired_extract") is False
    assert stats.get("paired_delta") > 0
    with torch.no_grad():
        diff = (m(x) - opt(x)).abs().max().item()
    assert diff < 1e-4
