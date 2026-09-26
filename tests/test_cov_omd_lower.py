# ruff: noqa: RUF002, RUF003
"""Coverage tests for catopt.omd_lower — projection/leaf recognizers,
plan rejection paths (inside_map_leaf projections, domain mismatches,
non-uniform signatures), the opaque-leaf path in _leaf_part, index
gathers in _eval_leaf_parts, the serial compose fallback, and the
runtime fallback counter.  CPU only."""

import pytest
import torch

from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.omd_lower import (
    BatchedOmdModule,
    _compose_pair,
    _concrete,
    _flatten_map,
    _is_omd_tree,
    _leaf_sig,
    _part_gather,
    _select_index,
    _stack_dim,
    build_omd_plan,
    is_omd_apply_term,
    to_batched_omd_module,
)
from catopt.torch_bridge import ir_to_torch_module


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


# ---------------------------------------------------------------------------
#  Small recognizers
# ---------------------------------------------------------------------------


def test_helpers_stack_dim_select_concrete():
    assert _stack_dim({"dim": 1}) == 1
    assert _stack_dim({"arg1": 2}) == 2
    assert _stack_dim({"dim": "x"}) == 0
    assert _stack_dim({}) == 0

    base = _v("b", 8, 4)
    assert _select_index(
        Op.make("select", base, arg1=0, arg2=3)
    ) == (base, 0, 3)
    assert (
        _select_index(
            Op.make(
                "select", base, arg1=0, arg2="i", validate=False
            )
        )
        is None
    )
    assert _select_index(Op.make("relu", base)) is None
    assert _select_index(base) is None

    assert _concrete((2, 3))
    assert not _concrete(())
    assert not _concrete("nope")
    assert not _concrete((2, "x"))


def test_is_omd_tree_and_apply_edges():
    s, a, b, h = _p("s", 3, 4), _p("a", 4, 2), _p("b", 4, 2), _p("h", 2)
    e = Op.make("omd_elem", s, a, b)
    assert _is_omd_tree(e)
    pkg = Op.make("omd", _p("m", 1), _p("l", 1), _p("fa", 1), _p("fb", 1))
    assert _is_omd_tree(pkg)
    comp = Op.make("omd_compose", e, pkg)
    assert _is_omd_tree(comp)
    assert not _is_omd_tree(Op.make("omd_compose", e, s))
    assert not _is_omd_tree(s)
    # memoised repeat
    memo = {}
    assert _is_omd_tree(comp, memo)
    assert _is_omd_tree(comp, memo)
    assert is_omd_apply_term(Op.make("omd_apply", comp, h))
    assert is_omd_apply_term(Op.make("omd_applym", comp, h))
    assert not is_omd_apply_term(Op.make("omd_apply", s, h))
    assert not is_omd_apply_term(comp)


def test_flatten_map_and_leaf_sig():
    a, b = _p("a", 3), _p("b", 3)
    l1 = Op.make("aff_diag", a, b)
    l2 = Op.make("aff_diag", _p("a2", 3), _p("b2", 3))
    t = Op.make("affd_compose", l1, l2)
    out = []
    _flatten_map(t, out)
    # right child applies first → l2 comes before l1
    assert out == [l2, l1]

    assert _leaf_sig(l1, "diag") == ("diag", (3,))
    # diag leaf with a != b shape → False
    bad = Op.make("aff_diag", _p("a", 3), _p("b2", 4))
    assert _leaf_sig(bad, "diag") is False
    A, bb = _p("A", 3, 3), _p("bb", 3)
    assert _leaf_sig(Op.make("aff", A, bb), "dense") == (
        "dense",
        (3, 3),
    )
    # dense leaf with non-square A → False
    assert (
        _leaf_sig(Op.make("aff", _p("A", 2, 3), _p("b", 2)), "dense")
        is False
    )
    # opaque term → None (no static guarantee)
    assert _leaf_sig(_v("m", 2, 3), "diag") is None
    assert _leaf_sig(Op.make("relu", a), "diag") is None


def test_part_gather():
    base = _v("x", 8, 4)
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", base, arg1=0, arg2=i),
            Op.make("select", base, arg1=0, arg2=i),
        )
        for i in range(3)
    ]
    g = _part_gather(leaves, 0)
    assert g is not None and g[0] is base and g[2] == [0, 1, 2]
    # non-leaf op in the leaf list → None
    leaves_bad = leaves[:1] + [_v("z", 4)]
    assert _part_gather(leaves_bad, 0) is None
    # arg not a select → None
    leaves2 = [
        Op.make("aff_diag", _p(f"a{i}", 4), _p(f"b{i}", 4))
        for i in range(3)
    ]
    assert _part_gather(leaves2, 0) is None
    # different bases → None
    leaves3 = [
        Op.make(
            "aff_diag",
            Op.make(
                "select",
                base if i else _v("y", 8, 4),
                arg1=0,
                arg2=1,
            ),
            Op.make("select", base, arg1=0, arg2=i),
        )
        for i in range(2)
    ]
    assert _part_gather(leaves3, 0) is None


def test_compose_pair_domains():
    torch.manual_seed(0)
    fa = torch.randn(3, 3, dtype=torch.float64)
    fb = torch.randn(3, dtype=torch.float64)
    ga = torch.randn(3, 3, dtype=torch.float64)
    gb = torch.randn(3, dtype=torch.float64)
    A, B = _compose_pair(fa, fb, ga, gb, "dense")
    assert torch.allclose(A, fa @ ga)
    assert torch.allclose(B, fa @ gb + fb)
    da, db = (
        torch.randn(4, dtype=torch.float64),
        torch.randn(4, dtype=torch.float64),
    )
    ga2, gb2 = (
        torch.randn(4, dtype=torch.float64),
        torch.randn(4, dtype=torch.float64),
    )
    A2, B2 = _compose_pair(da, db, ga2, gb2, "diag")
    assert torch.allclose(A2, da * ga2)
    assert torch.allclose(B2, da * gb2 + db)


# ---------------------------------------------------------------------------
#  build_omd_plan rejections
# ---------------------------------------------------------------------------


def test_plan_rejects_map_leaf_with_nested_projection():
    """A map leaf whose own args project another map — the batched
    leaf level would need map values that depend on leaf values.
    Declined honestly."""
    d = 4
    a, b, x = _p("a", d), _p("b", d), _v("x", d)
    f_inner = Op.make("affd_compose", Op.make("aff_diag", a, b),
                      Op.make("aff_diag", a, x))
    # leaf arg0 projects f_inner → inside_map_leaf scan hits the
    # projection and declines
    bad_leaf = Op.make("aff_diag", Op.make("affd_a", f_inner), x)
    mp = Op.make("affd_compose", Op.make("aff_diag", a, b), bad_leaf)
    s, h = _p("s", 3, 2), _p("h", d)
    amap = Op.make("affd_a", mp)
    bmap = Op.make("affd_b", mp)
    term = Op.make(
        "omd_apply", Op.make("omd_elem", s, amap, bmap), h
    )
    plan = build_omd_plan(term)
    assert plan is None


def test_plan_rejects_domain_mismatch():
    """A map registered for the wrong domain (dense aff under a diag
    projection) → the plan declines."""
    d = 4
    A, bb = _p("A", d, d), _p("bb", d)
    dense = Op.make("aff", A, bb)
    # affd_a expects a diag map; handing it a dense compose → mismatch
    amap = Op.make("affd_a", dense)
    s, h = _p("s", 3, 2), _p("h", d)
    term = Op.make(
        "omd_apply", Op.make("omd_elem", s, amap, _p("bm", 2)), h
    )
    plan = build_omd_plan(term)
    assert plan is None


def test_plan_rejects_nonuniform_leaf_sigs():
    """Compose trees whose leaves violate the shape contract (diag
    with a != b) → forest signature check fails → None."""
    d = 4
    t1 = Op.make(
        "affd_compose",
        Op.make("aff_diag", _p("a1", d), _p("b1", d)),
        Op.make("aff_diag", _p("a2", d), _p("b2", d + 1)),  # a2 != b2
    )
    t2 = Op.make(
        "affd_compose",
        Op.make("aff_diag", _p("a3", d), _p("b3", d)),
        Op.make("aff_diag", _p("a4", d), _p("b4", d)),
    )
    amap = Op.make("stack", Op.make("affd_a", t1), Op.make("affd_a", t2), dim=0)
    bmap = Op.make("stack", Op.make("affd_b", t1), Op.make("affd_b", t2), dim=0)
    s, h = _p("s", 3, 2), _p("h", d)
    term = Op.make("omd_apply", Op.make("omd_elem", s, amap, bmap), h)
    assert build_omd_plan(term) is None


def test_plan_no_projections_map_mode_none():
    """omd leaves with plain tensor args (no map projections) →
    map_mode None; the om tree still gets a batched schedule."""
    s1, s2 = _p("s1", 3, 4), _p("s2", 3, 4)
    a, b, h = _p("a", 4, 5), _p("b", 4, 5), _p("h", 5)
    comp = Op.make(
        "omd_compose",
        Op.make("omd_elem", s1, a, b),
        Op.make("omd_elem", s2, a, b),
    )
    term = Op.make("omd_apply", comp, h)
    plan = build_omd_plan(term)
    assert plan is not None and plan["map_mode"] is None
    assert len(plan["omd_leaves"]) == 2


# ---------------------------------------------------------------------------
#  BatchedOmdModule runtime
# ---------------------------------------------------------------------------


def _chain_term(T, Tq, d):
    """The standard emitted-scan chain term (same shape as
    test_omd_lower's): omd_apply(omd_elem(s, stack a_i, stack b_i), h)."""
    a_p, x_v = _p("p_a", T, d), _v("x", T, d)
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a_p, arg1=0, arg2=t),
            Op.make("select", x_v, arg1=0, arg2=t),
        )
        for t in range(T)
    ]
    fs = [leaves[0]]
    for t in range(1, T):
        fs.append(Op.make("affd_compose", leaves[t], fs[-1]))
    a_map = Op.make("stack", *(Op.make("affd_a", f) for f in fs), dim=0)
    b_map = Op.make("stack", *(Op.make("affd_b", f) for f in fs), dim=0)
    s = _p("s", Tq, T)
    h = _p("h", d)
    term = Op.make("omd_apply", Op.make("omd_elem", s, a_map, b_map), h)
    return term, x_v, a_p, s, h


def test_omd_module_chain_and_fallbacks():
    T, Tq, d = 8, 3, 4
    term, x_v, a_p, s, h = _chain_term(T, Tq, d)
    ir = IR(root=term, inputs=[x_v], params={})
    pv = {
        "p_a": _rand((T, d), 1) * 0.3,
        "s": _rand((Tq, T), 2),
        "h": _rand((d,), 3),
    }
    x = _rand((T, d), 4)
    mod = to_batched_omd_module(ir, param_values=pv)
    assert mod.is_batched and mod.map_mode == "chain"
    assert not mod.is_graph_captured
    mod.drop_cuda_graph()  # no-op on CPU
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(x)
        o_g = gen(x)
    assert mod.fallbacks == 0
    assert (o_b - o_g).abs().max().item() < 1e-12


def test_omd_module_runtime_fallback_counts():
    """A plan-valid term that throws at runtime → serial eval result
    and fallbacks incremented (the drop-in contract).

    Mixed leaf dtypes are legal IR but torch.stack can't batch them —
    the batched path raises, the wrapper falls back to serial eval and
    counts the fall."""
    Tq, Ts, d = 3, 4, 2
    a, b = _p("a", d), _p("b", d)
    h = _v("h", d)
    mp = Op.make(
        "affd_compose",
        Op.make("aff_diag", a, b),
        Op.make("aff_diag", a, b),
    )
    amap = Op.make("affd_a", mp)
    bmap = Op.make("affd_b", mp)
    # a MAP term sits in the s slot: legal IR (typing can't know it
    # evaluates to a pair).  The batched leaf pass calls omd_elem on
    # the tuple → raises inside the schedule → counted fallback; the
    # serial evaluator then surfaces the same genuine error.
    bad_elem = Op.make("omd_elem", mp, amap, bmap)
    term = Op.make("omd_apply", Op.make("omd_compose", bad_elem, bad_elem), h)
    ir = IR(root=term, inputs=[h], params={})
    pv = {
        "a": _rand((d,), 1) * 0.3,
        "b": _rand((d,), 2) * 0.3,
    }
    mod = to_batched_omd_module(ir, param_values=pv)
    assert mod.is_batched
    with pytest.raises(Exception):
        mod(_rand((d,), 4))
    assert mod.fallbacks == 1


def test_omd_nonuniform_leaves_serial_compose():
    """omd leaves with mismatched tuple shapes → the batched compose
    can't stack → serial compose loop, same result."""
    Tq, d = 3, 4
    s1, s2 = _p("s1", Tq, 4), _p("s2", Tq, 5)  # different key dims
    a1, b1 = _p("a1", 4, d), _p("b1", 4, d)
    a2, b2 = _p("a2", 5, d), _p("b2", 5, d)
    h = _v("h", d)
    comp = Op.make(
        "omd_compose",
        Op.make("omd_elem", s1, a1, b1),
        Op.make("omd_elem", s2, a2, b2),
    )
    term = Op.make("omd_apply", comp, h)
    ir = IR(root=term, inputs=[h], params={})
    pv = {
        "s1": _rand((Tq, 4), 1),
        "s2": _rand((Tq, 5), 2),
        "a1": _rand((4, d), 3),
        "b1": _rand((4, d), 4),
        "a2": _rand((5, d), 5),
        "b2": _rand((5, d), 6),
    }
    hv = _rand((d,), 7)
    mod = to_batched_omd_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(hv)
        o_g = gen(hv)
    assert mod.fallbacks == 0
    assert (o_b - o_g).abs().max().item() < 1e-12


def test_omd_prefix_scan_single_and_padding():
    """_prefix_scan: T==1 short-circuit; T not a multiple of the block
    size → identity padding keeps exact prefix maps."""
    mod = BatchedOmdModule.__new__(BatchedOmdModule)
    torch.manual_seed(0)
    A1 = torch.randn(1, 4, dtype=torch.float64)
    B1 = torch.randn(1, 4, dtype=torch.float64)
    A2, B2 = mod._prefix_scan(A1, B1, "diag")
    assert A2 is A1 and B2 is B1

    T, d = 7, 3  # pad to 9 (3 blocks of 3)
    av = torch.randn(T, d, dtype=torch.float64) * 0.5
    bv = torch.randn(T, d, dtype=torch.float64)
    Pa, Pb = mod._prefix_scan(av, bv, "diag")
    # verify every prefix against the serial fold
    for t in range(T):
        pa = torch.ones(d, dtype=torch.float64)
        pb = torch.zeros(d, dtype=torch.float64)
        for j in range(t + 1):
            pa, pb = av[j] * pa, av[j] * pb + bv[j]
        assert torch.allclose(Pa[t], pa, atol=1e-12), t
        assert torch.allclose(Pb[t], pb, atol=1e-12), t

    # dense domain too
    Ad = torch.randn(T, d, d, dtype=torch.float64) * 0.1
    Bd = torch.randn(T, d, dtype=torch.float64)
    Pa2, Pb2 = mod._prefix_scan(Ad, Bd, "dense")
    for t in range(T):
        pa = torch.eye(d, dtype=torch.float64)
        pb = torch.zeros(d, dtype=torch.float64)
        for j in range(t + 1):
            pa, pb = Ad[j] @ pa, Ad[j] @ pb + Bd[j]
        assert torch.allclose(Pa2[t], pa, atol=1e-10), t
        assert torch.allclose(Pb2[t], pb, atol=1e-10), t


def test_omd_opaque_map_leaf_part():
    """An opaque (non-aff/aff_diag) map leaf: _leaf_part indexes the
    evaluated pair/tensor exactly like the generic evaluator."""
    d = 4
    a, b = _p("a", d), _p("b", d)
    M = _v("M", 2, d)  # (2, d) tensor — opaque pair carrier
    t1 = Op.make("affd_compose", Op.make("aff_diag", a, b), M)
    # per-key coefficient terms are stack(affd_a f_i) — a single map
    # still gets the lifted stack wrapper
    amap = Op.make("stack", Op.make("affd_a", t1), dim=0)
    bmap = Op.make("stack", Op.make("affd_b", t1), dim=0)
    s, h = _p("s", 3, 1), _p("h", d)
    term = Op.make("omd_apply", Op.make("omd_elem", s, amap, bmap), h)
    plan = build_omd_plan(term)
    assert plan is not None
    # a single projected map is trivially a prefix chain → chain mode;
    # either way the opaque leaf M reaches _leaf_part's ev()-index path
    leaves = (
        plan["chain_leaves"]
        if plan["map_mode"] == "chain"
        else plan["forest"]["diag"]["leaves"]
    )
    assert M in leaves or any(lf is M for lf in leaves)
    ir = IR(root=term, inputs=[M], params={})
    pv = {
        "a": _rand((d,), 1) * 0.3,
        "b": _rand((d,), 2) * 0.3,
        "s": _rand((3, 1), 3),
        "h": _rand((d,), 4),
    }
    Mv = _rand((2, d), 5)
    mod = to_batched_omd_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(Mv)
        o_g = gen(Mv)
    assert mod.fallbacks == 0
    assert (o_b - o_g).abs().max().item() < 1e-12


def test_omd_non_omd_ir_fallback():
    torch.manual_seed(0)
    x = _v("x", 4)
    ir = IR(
        root=Op.make("tanh", x), inputs=[x],
        input_names={"x"}, params={},
    )
    mod = to_batched_omd_module(ir)
    assert not mod.is_batched and mod.map_mode is None
    with torch.no_grad():
        out = mod(torch.full((4,), 0.5))
    assert torch.allclose(out, torch.tanh(torch.full((4,), 0.5)))
