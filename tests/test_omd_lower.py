"""Batched lowering for the deferred-affine (omd) carrier.

The cross-carrier lifts (``xcarrier.omd_tree_lift``) produce
``omd_apply(omd_compose-tree of omd_elem(s_i, a_i, b_i), h0)`` members —
attention over scanned values as ONE map affine in the initial state.
The coefficient maps inside the leaves are ``stack(affd_a f_j)`` /
``stack(affd_b f_j)`` over per-step compose chains: O(T²) unrolled
nodes that the generic IRModule dispatches one torch call at a time.

:class:`catopt.omd_lower.BatchedOmdModule` gathers all per-step
``(a, b)`` pairs once, computes every prefix map with a blocked
associative scan (chain shape) or a level-batched forest (arbitrary
bracketing), batches the omd_compose levels, then applies the root —
fp64-equivalent to both the eager model and the serial lowering.
"""

import pytest
import torch
from catopt.cost import flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.omd_lower import (
    BatchedOmdModule,
    build_omd_plan,
    is_omd_apply_term,
    to_batched_omd_module,
)
from catopt.regime import default_rules
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.trace_lift import lift_scan_to_trace
from catopt.xcarrier import (
    gather_apply_stack,
    gather_applyd_stack,
    omd_tree_lift,
)

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _P(name: str, shape) -> Param:
    return Param(name=name, typ=TensorType(tuple(shape)))


def _rand(shape, seed_shift: int = 0):
    g = torch.Generator().manual_seed(4321 + seed_shift)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _chain_maps(a_t: Param, xs, steps: int):
    """The shared prefix-chain of diagonal maps: ``f_i =
    affd_compose(aff_diag(a[i], x[i]), f_{i-1})`` — the shape
    gather_applyd_stack + extraction emit for an unrolled scan."""
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a_t, arg1=0, arg2=t),
            Op.make("select", xs, arg1=0, arg2=t),
        )
        for t in range(steps)
    ]
    fs = [leaves[0]]
    for t in range(1, steps):
        fs.append(Op.make("affd_compose", leaves[t], fs[-1]))
    return fs


def _stacked_maps(fs):
    """``stack(affd_a f_i)`` / ``stack(affd_b f_i)`` coefficient terms."""
    a_map = Op.make("stack", *(Op.make("affd_a", f) for f in fs), dim=0)
    b_map = Op.make("stack", *(Op.make("affd_b", f) for f in fs), dim=0)
    return a_map, b_map


def _scan_attn_omd_term(T: int, D: int, seed: int = 0):
    """Export _ScanAttn, bounded-saturate, lift, extract the omd member.

    Same recipe as bench_omd.py: core CARRIER_LAWS saturation then the
    non-local lifts (bounded — the full build_egraph re-saturation does
    not terminate quickly at T>=64).
    """

    class _ScanAttn(torch.nn.Module):
        def __init__(self, T, D):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(T, D) * 0.1)
            self.h0 = torch.nn.Parameter(torch.randn(D) * 0.1)
            self.wq = torch.nn.Linear(D, D, bias=False)
            self.wk = torch.nn.Linear(D, D, bias=False)

        def forward(self, x):
            h = self.h0
            outs = []
            for t in range(x.shape[0]):
                h = self.a[t] * h + x[t]
                outs.append(h)
            v = torch.stack(outs)
            s = self.wq(x) @ self.wk(x).transpose(-1, -2)
            return torch.softmax(s, dim=-1) @ v

    torch.manual_seed(seed)
    m = _ScanAttn(T, D).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(default_rules(), root, max_iterations=4, max_nodes=200_000)
    lifts = (
        lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
    )
    if lifts:
        eg.rebuild()
    rc = eg.find(root)
    nodes = [n for n in eg.get_class(rc).nodes if n.op == "omd_apply"]
    assert nodes, "omd_tree_lift did not offer an omd_apply member"
    node = sorted(nodes, key=repr)[0]
    term = eg.extract_best(rc, flops_cost, overrides={rc: node})
    if term is None:
        args = [eg.any_term(eg.find(c)) for c in node.children]
        term = Op.make("omd_apply", *args, **dict(node.attrs))
    return m, x, term, ir, src


# ---------------------------------------------------------------------------
#  Detection / plan structure
# ---------------------------------------------------------------------------


def test_is_omd_apply_term():
    Tq, K, d = 4, 5, 3
    s, a, b, h = (
        _P("s", (Tq, K)),
        _P("a", (K, d)),
        _P("b", (K, d)),
        _P("h", (d,)),
    )
    good = Op.make("omd_apply", Op.make("omd_elem", s, a, b), h)
    assert is_omd_apply_term(good)
    comp = Op.make(
        "omd_compose",
        Op.make("omd_elem", s, a, b),
        Op.make("omd_elem", s, a, b),
    )
    assert is_omd_apply_term(Op.make("omd_apply", comp, h))
    # negatives
    assert not is_omd_apply_term(
        Op.make("om_apply", Op.make("om_elem", s, a))
    )
    assert not is_omd_apply_term(Op.make("matmul", s, a))
    assert not is_omd_apply_term(s)
    # foreign leaf in the tree
    assert not is_omd_apply_term(
        Op.make(
            "omd_apply",
            Op.make("omd_compose", Op.make("omd_elem", s, a, b), s),
            h,
        )
    )


def test_plan_chain_mode_on_prefix_stacks():
    """stack(affd_a f_i) over the shared prefix chain → chain plan."""
    T, Tq, d = 8, 3, 4
    a_p, x_v = _P("p_a", (T, d)), _P("x", (T, d))
    s, h = _P("s", (Tq, T)), _P("h", (d,))
    fs = _chain_maps(a_p, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    term = Op.make("omd_apply", Op.make("omd_elem", s, a_map, b_map), h)
    plan = build_omd_plan(term)
    assert plan is not None
    assert plan["map_mode"] == "chain"
    assert plan["chain_domain"] == "diag"
    assert len(plan["chain_leaves"]) == T
    # leaf parts are select slices of two base tensors → gather path
    assert plan["chain_a_gather"] is not None
    assert plan["chain_b_gather"] is not None
    assert len(plan["stack_seeds"]) == 2


def test_plan_forest_mode_on_disjoint_trees():
    """Projections of unrelated compose trees → forest (level-batched)."""
    d = 4
    a = [_P(f"a{i}", (d,)) for i in range(4)]
    b = [_P(f"b{i}", (d,)) for i in range(4)]
    # two independent compose trees over different leaf sets
    t1 = Op.make(
        "affd_compose",
        Op.make("aff_diag", a[1], b[1]),
        Op.make("aff_diag", a[0], b[0]),
    )
    t2 = Op.make(
        "affd_compose",
        Op.make("aff_diag", a[3], b[3]),
        Op.make("aff_diag", a[2], b[2]),
    )
    a_map = Op.make(
        "stack", Op.make("affd_a", t1), Op.make("affd_a", t2), dim=0
    )
    b_map = Op.make(
        "stack", Op.make("affd_b", t1), Op.make("affd_b", t2), dim=0
    )
    s, h = _P("s", (3, 2)), _P("h", (d,))
    term = Op.make("omd_apply", Op.make("omd_elem", s, a_map, b_map), h)
    plan = build_omd_plan(term)
    assert plan is not None
    assert plan["map_mode"] == "forest"
    assert len(plan["forest"]["diag"]["leaves"]) == 4
    assert (
        len(plan["forest"]["diag"]["gather"]) == 1
    )  # one compose level


# ---------------------------------------------------------------------------
#  Numerical equivalence — hand-built terms
# ---------------------------------------------------------------------------


def _params_dict(*ps):
    return {p.name: p for p in ps}


def test_batched_matches_generic_single_elem():
    """omd_apply(omd_elem(s, stack a_i, stack b_i), h) — the emitted
    scan-attention shape — agrees with serial eval and eager math."""
    T, Tq, d = 12, 5, 6
    p_a, p_h = _P("p_a", (T, d)), _P("p_h", (d,))
    p_s = _P("p_s", (Tq, T))
    x_v = Var("x", TensorType((T, d)))

    fs = _chain_maps(p_a, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    term = Op.make(
        "omd_apply", Op.make("omd_elem", p_s, a_map, b_map), p_h
    )
    ir = IR(root=term, inputs=[x_v], params=_params_dict(p_a, p_h, p_s))

    pv = {
        "p_a": _rand((T, d), 1),
        "p_h": _rand((d,), 2),
        "p_s": _rand((Tq, T), 3),
    }
    x = _rand((T, d), 4)

    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched and bat.map_mode == "chain"

    # eager reference: sequential scan then softmax @ values
    a_v, h0, s_v = pv["p_a"], pv["p_h"], pv["p_s"]
    h = h0.clone()
    outs = []
    for t in range(T):
        h = a_v[t] * h + x[t]
        outs.append(h)
    ref = torch.softmax(s_v, dim=-1) @ torch.stack(outs)

    with torch.no_grad():
        o_gen, o_bat = gen(x), bat(x)
    assert bat.fallbacks == 0
    assert (o_gen - ref).abs().max().item() < 1e-12
    assert (o_bat - ref).abs().max().item() < 1e-12
    assert (o_bat - o_gen).abs().max().item() < 1e-12


def test_batched_matches_generic_two_blocks():
    """omd_compose of two omd_elem leaves — exercises the level-batched
    FlashAttention combine (where/isfinite branches)."""
    T, Tq, d = 10, 4, 5
    p_a, p_h = _P("p_a", (T, d)), _P("p_h", (d,))
    p_s1, p_s2 = _P("s1", (Tq, T)), _P("s2", (Tq, T))
    x_v = Var("x", TensorType((T, d)))

    fs = _chain_maps(p_a, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    term = Op.make(
        "omd_apply",
        Op.make(
            "omd_compose",
            Op.make("omd_elem", p_s1, a_map, b_map),
            Op.make("omd_elem", p_s2, a_map, b_map),
        ),
        p_h,
    )
    ir = IR(
        root=term,
        inputs=[x_v],
        params=_params_dict(p_a, p_h, p_s1, p_s2),
    )
    pv = {
        "p_a": _rand((T, d), 10) * 0.3,
        "p_h": _rand((d,), 11),
        "s1": _rand((Tq, T), 12),
        "s2": _rand((Tq, T), 13),
    }
    x = _rand((T, d), 14)
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched
    with torch.no_grad():
        diff = (gen(x) - bat(x)).abs().max().item()
    assert bat.fallbacks == 0
    assert diff < 1e-12


def test_batched_masked_scores_match():
    """A fully-masked (-inf) score block keeps om's exact NaN semantics:
    the where/isfinite guards absorb the dead block, and a row that is
    ALL masked stays NaN — identical to the serial evaluator."""
    T, Tq, d = 6, 3, 4
    p_a, p_h = _P("p_a", (T, d)), _P("p_h", (d,))
    p_s1, p_s2 = _P("s1", (Tq, T)), _P("s2", (Tq, T))
    x_v = Var("x", TensorType((T, d)))
    fs = _chain_maps(p_a, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    term = Op.make(
        "omd_apply",
        Op.make(
            "omd_compose",
            Op.make("omd_elem", p_s1, a_map, b_map),
            Op.make("omd_elem", p_s2, a_map, b_map),
        ),
        p_h,
    )
    ir = IR(
        root=term,
        inputs=[x_v],
        params=_params_dict(p_a, p_h, p_s1, p_s2),
    )
    s1 = _rand((Tq, T), 20)
    s2 = _rand((Tq, T), 21)
    s2[0, :] = float("-inf")  # row 0 live only in block 1
    s1[1, :] = float("-inf")
    s2[1, :] = float("-inf")  # row 1 fully masked → NaN
    pv = {
        "p_a": _rand((T, d), 22) * 0.2,
        "p_h": _rand((d,), 23),
        "s1": s1,
        "s2": s2,
    }
    x = _rand((T, d), 24)
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    with torch.no_grad():
        o_gen, o_bat = gen(x), bat(x)
    assert bat.fallbacks == 0
    assert torch.isnan(o_gen[1]).all() and torch.isnan(o_bat[1]).all()
    assert torch.allclose(
        o_gen, o_bat, atol=1e-12, rtol=1e-12, equal_nan=True
    )


def test_batched_dense_fiber_applym():
    """omd_applym — dense per-key maps ``aff(A_t, c_t)`` chained and
    stacked via aff_A/aff_b projections; fa @ h + fb over l."""
    T, Tq, d = 7, 3, 4
    p_A, p_h = _P("p_A", (T, d, d)), _P("p_h", (d,))
    p_s = _P("p_s", (Tq, T))
    x_v = Var("x", TensorType((T, d)))

    leaves = [
        Op.make(
            "aff",
            Op.make("select", p_A, arg1=0, arg2=t),
            Op.make("select", x_v, arg1=0, arg2=t),
        )
        for t in range(T)
    ]
    fs = [leaves[0]]
    for t in range(1, T):
        fs.append(Op.make("aff_compose", leaves[t], fs[-1]))
    a_map = Op.make("stack", *(Op.make("aff_A", f) for f in fs), dim=0)
    b_map = Op.make("stack", *(Op.make("aff_b", f) for f in fs), dim=0)
    term = Op.make(
        "omd_applym", Op.make("omd_elem", p_s, a_map, b_map), p_h
    )
    ir = IR(root=term, inputs=[x_v], params=_params_dict(p_A, p_h, p_s))
    pv = {
        "p_A": _rand((T, d, d), 30) * 0.2,
        "p_h": _rand((d,), 31),
        "p_s": _rand((Tq, T), 32),
    }
    x = _rand((T, d), 33)
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched
    assert bat.map_mode == "chain"
    assert bat._plan["chain_domain"] == "dense"
    with torch.no_grad():
        diff = (gen(x) - bat(x)).abs().max().item()
    assert bat.fallbacks == 0
    assert diff < 1e-12


def test_batched_forest_mode_equivalence():
    """Non-prefix projections take the level-batched forest path and
    stay exact."""
    d = 4
    a = [_P(f"a{i}", (d,)) for i in range(4)]
    b = [_P(f"b{i}", (d,)) for i in range(4)]
    t1 = Op.make(
        "affd_compose",
        Op.make("aff_diag", a[1], b[1]),
        Op.make("aff_diag", a[0], b[0]),
    )
    t2 = Op.make(
        "affd_compose",
        Op.make("aff_diag", a[3], b[3]),
        Op.make("aff_diag", a[2], b[2]),
    )
    a_map = Op.make(
        "stack", Op.make("affd_a", t1), Op.make("affd_a", t2), dim=0
    )
    b_map = Op.make(
        "stack", Op.make("affd_b", t1), Op.make("affd_b", t2), dim=0
    )
    s, h = _P("s", (3, 2)), _P("h", (d,))
    xv = Var("x", TensorType((1,)))  # dummy input so forward(*xs) works
    term = Op.make("omd_apply", Op.make("omd_elem", s, a_map, b_map), h)
    ir = IR(root=term, inputs=[xv], params=_params_dict(s, h, *a, *b))
    pv = {
        p.name: _rand(p.typ.shape, 40 + i)
        for i, p in enumerate([s, h, *a, *b])
    }
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched and bat.map_mode == "forest"
    x = _rand((1,), 60)
    with torch.no_grad():
        diff = (gen(x) - bat(x)).abs().max().item()
    assert bat.fallbacks == 0
    assert diff < 1e-12


def test_projection_inside_tensor_context():
    """matmul(E, stack(affd_a f_i)) — a projection inside an ordinary
    tensor term is memo-seeded; the surrounding op still evaluates."""
    T, Tq, d = 6, 4, 3
    p_a, p_h = _P("p_a", (T, d)), _P("p_h", (d,))
    p_s, p_E = _P("p_s", (Tq, T)), _P("p_E", (T, T))
    x_v = Var("x", TensorType((T, d)))
    fs = _chain_maps(p_a, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    a_map2 = Op.make(
        "matmul", p_E, a_map
    )  # projection used inside matmul
    term = Op.make(
        "omd_apply", Op.make("omd_elem", p_s, a_map2, b_map), p_h
    )
    ir = IR(
        root=term, inputs=[x_v], params=_params_dict(p_a, p_h, p_s, p_E)
    )
    pv = {
        "p_a": _rand((T, d), 70) * 0.3,
        "p_h": _rand((d,), 71),
        "p_s": _rand((Tq, T), 72),
        "p_E": _rand((T, T), 73),
    }
    x = _rand((T, d), 74)
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched
    with torch.no_grad():
        diff = (gen(x) - bat(x)).abs().max().item()
    assert bat.fallbacks == 0
    assert diff < 1e-12


# ---------------------------------------------------------------------------
#  End-to-end: real extracted member, T = 16 / 64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [16, 64])
def test_scanattn_omd_member_end_to_end(T):
    """The extracted omd_apply member lowers batched and stays fp64-
    exact vs both the eager model and the serial IRModule."""
    m, x, term, ir, src = _scan_attn_omd_term(T, 64)
    assert is_omd_apply_term(term)
    opt_ir = IR(root=term, inputs=ir.inputs, params=ir.params)
    bat = to_batched_omd_module(opt_ir, src).eval()
    gen = ir_to_torch_module(opt_ir, src).eval()
    assert bat.is_batched
    assert bat.map_mode in ("chain", "forest")
    with torch.no_grad():
        ref = m(x)
        o_gen, o_bat = gen(x), bat(x)
    assert bat.fallbacks == 0
    assert (o_gen - ref).abs().max().item() < 1e-12
    assert (o_bat - ref).abs().max().item() < 1e-12
    assert (o_bat - o_gen).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  Fallback / determinism
# ---------------------------------------------------------------------------


def test_fallback_non_omd_root():
    """Non-omd IR: BatchedOmdModule delegates to serial evaluation."""
    from catopt.models import SwiGLU

    torch.manual_seed(0)
    m = SwiGLU(16, hidden_mult=2).eval()
    x = torch.randn(2, 3, 16)
    ir, src = export_to_ir(m, x)
    mod = to_batched_omd_module(ir, param_values=src)
    assert isinstance(mod, BatchedOmdModule)
    assert not mod.is_batched
    ref = ir_to_torch_module(ir, param_values=src)
    mod.eval()
    ref.eval()
    with torch.no_grad():
        assert torch.equal(mod(x.clone()), ref(x.clone()))
        assert (m(x.clone()) - mod(x.clone())).abs().max().item() < 1e-6


def test_fallback_bare_compose_tree():
    """An omd_compose tree NOT under omd_apply falls back cleanly —
    serial eval returns the (m, l, fa, fb) tuple."""
    Tq, K, d = 3, 4, 5
    s, a, b = _P("s", (Tq, K)), _P("a", (K, d)), _P("b", (K, d))
    t = Op.make(
        "omd_compose",
        Op.make("omd_elem", s, a, b),
        Op.make("omd_elem", s, a, b),
    )
    xv = Var("x", TensorType((1,)))
    ir = IR(root=t, inputs=[xv], params=_params_dict(s, a, b))
    pv = {
        "s": _rand((Tq, K), 80),
        "a": _rand((K, d), 81),
        "b": _rand((K, d), 82),
    }
    mod = to_batched_omd_module(ir, pv)
    assert not mod.is_batched
    with torch.no_grad():
        out = mod(torch.randn(1, dtype=torch.float64))
    assert isinstance(out, tuple) and len(out) == 4


def test_determinism():
    """Two forward calls and two module instances are bitwise-equal."""
    T, Tq, d = 8, 3, 4
    p_a, p_h = _P("p_a", (T, d)), _P("p_h", (d,))
    p_s = _P("p_s", (Tq, T))
    x_v = Var("x", TensorType((T, d)))
    fs = _chain_maps(p_a, x_v, T)
    a_map, b_map = _stacked_maps(fs)
    term = Op.make(
        "omd_apply", Op.make("omd_elem", p_s, a_map, b_map), p_h
    )
    ir = IR(root=term, inputs=[x_v], params=_params_dict(p_a, p_h, p_s))
    pv = {
        "p_a": _rand((T, d), 90),
        "p_h": _rand((d,), 91),
        "p_s": _rand((Tq, T), 92),
    }
    x = _rand((T, d), 93)
    bat1 = to_batched_omd_module(ir, pv).eval()
    bat2 = to_batched_omd_module(ir, pv).eval()
    with torch.no_grad():
        o1, o2 = bat1(x), bat1(x)
        o3 = bat2(x)
    assert torch.equal(o1, o2)
    assert torch.equal(o1, o3)


def test_omd_apply_term_accepting_bare_term():
    """to_batched_omd_module also accepts a bare term (wraps in IR)."""
    Tq, K, d = 2, 3, 4
    s, a, b, h = (
        _P("s", (Tq, K)),
        _P("a", (K, d)),
        _P("b", (K, d)),
        _P("h", (d,)),
    )
    term = Op.make("omd_apply", Op.make("omd_elem", s, a, b), h)
    xv = Var("x", TensorType((1,)))
    ir = IR(root=term, inputs=[xv], params=_params_dict(s, a, b, h))
    pv = {
        "s": _rand((Tq, K), 100),
        "a": _rand((K, d), 101),
        "b": _rand((K, d), 102),
        "h": _rand((d,), 103),
    }
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(term, pv).eval()
    assert bat.is_batched
    x = _rand((1,), 104)
    with torch.no_grad():
        assert (gen(x) - bat(x)).abs().max().item() < 1e-12


def test_batched_dense_compose_levels_no_fallback():
    """Dense-fiber ``omd_compose`` levels: fa carries (…,Tq,o,i) while
    the row weight is (…,Tq,1) — the batched rescale needs the
    trailing-1 broadcast the serial _omd_compose does, or every
    forward falls back silently (audit finding R3)."""
    T, Tq, d = 4, 3, 5
    p_A, p_h, p_s1, p_s2 = [
        _P(n, s)
        for n, s in (
            ("p_A", (T, d, d)),
            ("p_h", (d,)),
            ("p_s1", (Tq, T)),
            ("p_s2", (Tq, T)),
        )
    ]
    x_v = Var("x", TensorType((T, d)))

    def dense_map():
        leaves = [
            Op.make(
                "aff",
                Op.make("select", p_A, arg1=0, arg2=t),
                Op.make("select", x_v, arg1=0, arg2=t),
            )
            for t in range(T)
        ]
        fs = [leaves[0]]
        for t in range(1, T):
            fs.append(Op.make("aff_compose", leaves[t], fs[-1]))
        return (
            Op.make("stack", *(Op.make("aff_A", f) for f in fs), dim=0),
            Op.make("stack", *(Op.make("aff_b", f) for f in fs), dim=0),
        )

    a1, b1 = dense_map()
    a2, b2 = dense_map()
    term = Op.make(
        "omd_applym",
        Op.make(
            "omd_compose",
            Op.make("omd_elem", p_s1, a1, b1),
            Op.make("omd_elem", p_s2, a2, b2),
        ),
        p_h,
    )
    ir = IR(root=term, inputs=[x_v], params={})
    pv = {
        "p_A": _rand((T, d, d), 30) * 0.2,
        "p_h": _rand((d,), 31),
        "p_s1": _rand((Tq, T), 32),
        "p_s2": _rand((Tq, T), 33),
    }
    x = _rand((T, d), 34)
    gen = ir_to_torch_module(ir, pv).eval()
    bat = to_batched_omd_module(ir, pv).eval()
    assert bat.is_batched
    with torch.no_grad():
        diff = (gen(x) - bat(x)).abs().max().item()
    assert bat.fallbacks == 0  # the batched dense compose must fire
    assert diff < 1e-6
