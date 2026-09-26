"""Coverage wave-2 tests — model blocks plus the executor/traced
lowering edge paths that the main suites deliberately leave open:

* ``catopt.models`` — ``ResidualMLP``/``ParallelConv`` forwards and the
  analytic flop helpers, exercised on real tensors.
* ``catopt.scan_lower`` — the composition-level memo hit, non-zero-axis
  leaf gathers, and the graph-capture CPU behaviour.
* ``catopt.om_lower`` — the domain-recogniser rejections, the
  ``bmm_qk`` rank-gap/``k_gather`` paths, and replay/compile edges.
* ``catopt.omd_lower`` — forest-mode plans, deferred-affine walker
  rejections, and the serial-compose fallback inside batched eval.
* ``catopt.trace`` / ``catopt.trace_lift`` — law check-fns in both
  directions and the lift pass's structural decline/cycle-cut paths.
"""
# ruff: noqa: RUF059 — test-idiom unpacking

import torch
import torch.nn.functional as F

import catopt.trace_lift as TL
from catopt.egraph import EClass, EGraph
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.om_lower import (
    _analyze_elem_group,
    _qk_parts,
    _slice_index,
    _sliced_gather,
    _stretch,
    build_om_plan,
    to_batched_om_module,
)
from catopt.omd_lower import (
    _leaf_sig,
    build_omd_plan,
    to_batched_omd_module,
)
from catopt.scan_lower import (
    _select_index,
    build_scan_plan,
    to_batched_scan_module,
)
from catopt.torch_bridge import ir_to_torch_module
from catopt.trace import (
    _check_collapse,
    _check_expand,
    _check_slide,
    _check_slide_rev,
    _check_superpose,
    _check_tighten_in,
    _check_tighten_out,
    _check_vanish_merge,
    _check_vanish_split,
    _derive_expand,
    _derive_slide,
    _derive_slide_rev,
    _derive_vanish_split,
    _usize_split2,
    _usize_total,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(99 + seed)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _cyclic_eclass(eg, prefix="cyc"):
    """An e-class whose every member is self-referential — the
    degenerate shape the recognisers' memo walks must cut."""
    leaf = eg.add_term(_p(f"{prefix}{len(eg._classes)}", 2))
    f = eg.add_enode("neg", (leaf,))
    eg.union(f, leaf)
    cid = eg.find(f)
    for n in list(eg._classes[cid].nodes):
        if n.op == "leaf":
            eg._classes[cid].nodes.discard(n)
    return cid


def _compose(opname, leaves):
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        opname,
        _compose(opname, leaves[:mid]),
        _compose(opname, leaves[mid:]),
    )


# ---------------------------------------------------------------------------
#  catopt.models — real forwards for the never-instantiated blocks
# ---------------------------------------------------------------------------


def test_residual_mlp_forward_matches_manual():
    from catopt.models import ResidualMLP

    torch.manual_seed(0)
    m = ResidualMLP(dim=8, hidden_mult=3).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    with torch.no_grad():
        out = m(x)
        ref = m.fc2(F.silu(m.fc1(m.norm(x)))) + x
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref)


def test_parallel_conv_forward_is_sum_of_branches():
    from catopt.models import ParallelConv

    torch.manual_seed(0)
    m = ParallelConv(4, 8, branches=3, kernel=3).eval().double()
    x = torch.randn(2, 4, 12, 12, dtype=torch.float64)
    with torch.no_grad():
        out = m(x)
        ref = sum(c(x) for c in m.convs)
    assert out.shape == (2, 8, 10, 10)
    torch.testing.assert_close(out, ref)


def test_flop_helpers_match_formulas():
    from catopt.models import MatrixChain, ParallelLinear

    assert ParallelLinear.flops((8, 4), 32) == 2 * 32 * 8 * 4
    dims = (16, 8, 4, 2)
    assert MatrixChain.flops(dims, 3) == 2 * 3 * sum(
        dims[k] * dims[k + 1] for k in range(3)
    )
    # fused = precompute (W2@W3 then W1@(W2W3)) + runtime
    assert (
        MatrixChain.fused_flops(dims, 3)
        == 2
        * (dims[1] * dims[2] * dims[3] + dims[0] * dims[1] * dims[3])
        + 2 * 3 * dims[0] * dims[3]
    )
    # and the module still forwards
    mc = MatrixChain(8, 4, 2, 6).eval()
    out = mc(torch.randn(5, 8))
    assert out.shape == (5, 6)


# ---------------------------------------------------------------------------
#  scan_lower — recogniser + runtime edges
# ---------------------------------------------------------------------------


def test_select_index_getitem_nonint_index():
    base = _v("b", 8, 4)
    assert _select_index(Op.make("getitem", base, index=2)) == (
        base,
        0,
        2,
    )
    # a non-integer index is not a leaf-select pattern
    assert (
        _select_index(
            Op.make("getitem", base, index="x", validate=False)
        )
        is None
    )


def test_scan_plan_visit_memo_hit_on_shared_subtree():
    """The same compose subtree appearing twice is visited once —
    the second visit resolves through the level memo."""
    A, b, h = _p("A", 3, 3), _p("b", 3), _p("h", 3)
    sub = Op.make(
        "aff_compose",
        Op.make("aff", A, b),
        Op.make("aff", _p("A2", 3, 3), _p("b2", 3)),
    )
    root = Op.make("apply", Op.make("aff_compose", sub, sub), h)
    plan = build_scan_plan(root)
    assert plan is not None
    # shared subtree → its leaves appear once in the leaf list
    assert len(plan["leaves"]) == 2
    assert len(plan["levels"]) == 2


def test_scan_batched_leaf_gather_on_nonzero_dim():
    """b-parts as ``select(base, dim=1, i)`` → the gather path
    movedims before index_select; batched still matches serial."""
    T, d = 4, 3
    x = _v("x", d, T)  # (d, T): leaves slice along dim 1
    leaves = [
        Op.make(
            "aff",
            _p(f"A{t}", d, d),
            Op.make("select", x, dim=1, index=t),
        )
        for t in range(T)
    ]
    root = Op.make("apply", _compose("aff_compose", leaves), _p("h", d))
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    mod = to_batched_scan_module(ir)
    plan = build_scan_plan(mod.eval_mod._root)
    assert plan["leaf_b_gather"][1:] == (1, [0, 1, 2, 3])
    pv = {f"A{t}": _rand((d, d), t) * 0.2 for t in range(T)} | {
        "h": _rand((d,), 9)
    }
    xv = _rand((d, T), 5)
    gen = ir_to_torch_module(ir, param_values=pv)
    mod = to_batched_scan_module(ir, param_values=pv)
    with torch.no_grad():
        torch.testing.assert_close(mod(xv), gen(xv))


def test_scan_capture_and_replay_edges(monkeypatch):
    """CPU tensors under a (fake) CUDA flag raise the documented
    ValueError; a captured-graph slot replays verbatim."""
    # a real scan plan so capture_cuda_graph proceeds past the
    # plan-is-None guard to the input check
    T, d = 3, 4
    x = _v("x", T, d)
    leaves = [
        Op.make(
            "aff_diag",
            _p("a", d),
            Op.make("select", x, dim=0, index=t),
        )
        for t in range(T)
    ]
    root = Op.make(
        "applyd", _compose("affd_compose", leaves), _p("h", d)
    )
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    pv = {"a": _rand((d,), 1).clamp(-0.9, 0.9), "h": _rand((d,), 2)}
    mod = to_batched_scan_module(ir, param_values=pv)
    assert mod.is_batched
    # CPU no-op capture leaves no graph
    assert mod.capture_cuda_graph(_rand((T, d), 7)) is mod
    assert mod._graph is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        mod.capture_cuda_graph(torch.randn(T, d))
    except ValueError as e:
        assert "CUDA" in str(e)
    else:  # pragma: no cover — defensive
        raise AssertionError("expected ValueError for CPU inputs")
    monkeypatch.undo()

    # a captured-graph slot replays verbatim on same-shaped inputs
    xv = _rand((T, d), 3)

    class _FakeGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    sentinel = torch.full((d,), 3.0)
    mod._graph = _FakeGraph()
    mod._graph_inputs = [xv]
    mod._graph_out = sentinel
    out = mod(xv.clone())
    assert out is sentinel and mod._graph.replays == 1
    # a dtype-mismatched input misses the replay guard and runs eager
    out2 = mod(xv.clone().float())
    assert out2.shape == (d,)
    mod.drop_cuda_graph()


# ---------------------------------------------------------------------------
#  om_lower — domain-recogniser edges
# ---------------------------------------------------------------------------


def test_slice_index_and_gather_rejections():
    bnc = _v("bnc", None, 8)
    b48 = _v("b48", 4, 8)
    b49 = _v("b49", 4, 9)
    # select/getitem/chunk with non-int position attrs are declined
    assert (
        _slice_index(
            Op.make("select", b48, dim="x", index=0, validate=False)
        )
        is None
    )
    assert (
        _slice_index(Op.make("getitem", b48, index="x", validate=False))
        is None
    )
    assert (
        _slice_index(
            Op.make(
                "chunk", b48, chunks=3, dim=1, index="x", validate=False
            )
        )
        is None
    )
    assert (
        _slice_index(
            Op.make(
                "chunk", b48, chunks="n", dim=1, index=0, validate=False
            )
        )
        is None
    )
    # getitem resolves to a dim-0 select
    assert _slice_index(Op.make("getitem", b48, index=2)) == (
        b48,
        "select",
        0,
        2,
        None,
        None,
    )
    # chunk on a non-concrete base → part size is unknown
    assert _slice_index(
        Op.make("chunk", bnc, chunks=4, dim=1, index=0)
    ) == (bnc, "chunk", 1, 0, 4, None)
    # ...as is a chunk that doesn't divide the axis evenly
    assert _slice_index(
        Op.make("chunk", b49, chunks=4, dim=1, index=0)
    ) == (b49, "chunk", 1, 0, 4, None)
    # anything else isn't a slice
    assert _slice_index(Op.make("relu", b48)) is None
    assert _slice_index(b48) is None
    # split parts: equal sizes → known part; ragged → None
    assert _slice_index(
        Op.make("split", b49, sizes=(3, 3, 3), dim=1, index=1)
    ) == (b49, "chunk", 1, 1, 3, 3)
    sp = _slice_index(
        Op.make("split", b49, sizes=(3, 2, 4), dim=1, index=2)
    )
    assert sp is not None and sp[-1] is None
    assert (
        _slice_index(
            Op.make("split", b49, dim=1, index=0, validate=False)
        )
        is None
    )
    # gathers over a symbolic base decline
    sym_parts = [
        _slice_index(Op.make("select", bnc, dim=0, index=i))
        for i in range(2)
    ]
    assert _sliced_gather(sym_parts) is None
    # parts must share base, kind, and dim
    b48b = _v("b48b", 4, 8)
    mixed_base = [
        _slice_index(Op.make("select", b48, dim=0, index=0)),
        _slice_index(Op.make("select", b48b, dim=0, index=1)),
    ]
    assert _sliced_gather(mixed_base) is None
    # ...and be the i-th slice in order
    out_of_order = [
        _slice_index(Op.make("select", b48, dim=0, index=0)),
        _slice_index(Op.make("select", b48, dim=0, index=2)),
    ]
    assert _sliced_gather(out_of_order) is None
    # select parts must cover the whole sliced axis
    partial = [
        _slice_index(Op.make("select", b48, dim=0, index=i))
        for i in range(2)
    ]
    assert _sliced_gather(partial) is None
    # chunk parts must share the divisor n
    mixed_n = [
        _slice_index(Op.make("chunk", b48, chunks=4, dim=1, index=0)),
        _slice_index(Op.make("chunk", b48, chunks=2, dim=1, index=1)),
    ]
    assert _sliced_gather(mixed_n) is None
    # ...and the number of parts must equal n (with real part sizes)
    short = [
        _slice_index(Op.make("chunk", b49, chunks=3, dim=1, index=i))
        for i in range(2)
    ]
    assert _sliced_gather(short) is None


def test_qk_parts_rejections():
    q, k = _v("q", 4, 8), _v("k", 2, 4, 8)
    # the second operand must be a single transpose
    assert _qk_parts(Op.make("matmul", q, k)) is None
    # non-integer transpose dims decline
    assert (
        _qk_parts(
            Op.make(
                "matmul",
                q,
                Op.make(
                    "transpose", k, dim0="x", dim1=-1, validate=False
                ),
            )
        )
        is None
    )
    # a concrete k transposed off its last two axes declines
    assert (
        _qk_parts(
            Op.make(
                "matmul",
                q,
                Op.make("transpose", k, dim0=0, dim1=1),
            )
        )
        is None
    )
    # a symbolic k transposed on non-last-two axes declines;
    # on the last two it resolves normally
    knc = _v("knc", None, 4, 8)
    assert (
        _qk_parts(
            Op.make(
                "matmul",
                q,
                Op.make("transpose", knc, dim0=0, dim1=1),
            )
        )
        is None
    )
    assert _qk_parts(
        Op.make(
            "matmul",
            q,
            Op.make("transpose", knc, dim0=-2, dim1=-1),
        )
    ) == (q, knc)


def test_analyze_elem_group_modes():
    q = _v("q", 4, 8)
    vs = [_v(f"v{i}", 4, 4) for i in range(2)]
    # k slices along a NON-key dim → not dense_qk → bmm_qk + k_gather
    Kb = _v("Kb", 4, 16)
    leaves = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul",
                q,
                Op.make(
                    "transpose",
                    Op.make("chunk", Kb, chunks=2, dim=-1, index=i),
                    dim0=-2,
                    dim1=-1,
                ),
            ),
            vs[i],
        )
        for i in range(2)
    ]
    grp = _analyze_elem_group(leaves)
    assert grp["s_mode"] == "bmm_qk"
    assert grp["k_gather"] == (Kb, "chunk", -1, 2, 8)
    # differing k shapes can't bmm → per-leaf stack
    leaves2 = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul",
                q,
                Op.make(
                    "transpose", _v(f"k{i}", 2 + i, 8), dim0=-2, dim1=-1
                ),
            ),
            vs[i],
        )
        for i in range(2)
    ]
    assert _analyze_elem_group(leaves2)["s_mode"] == "stack"


def test_stretch_same_rank_broadcast():
    """gap == 0 and trailing-dim broadcasting are the direct routes."""
    t = torch.randn(3, 1, 4)
    assert _stretch(t, torch.Size([2, 4])).shape == (3, 2, 4)
    # already-matching tail → returned unchanged
    t2 = torch.randn(3, 2, 4)
    assert _stretch(t2, torch.Size([2, 4])) is t2
    # a rank-deficient tail gets a leading reshape before expanding
    t3 = torch.randn(3, 4)
    assert _stretch(t3, torch.Size([2, 2, 4])).shape == (3, 2, 2, 4)


# ---------------------------------------------------------------------------
#  om_lower — bmm_qk runtime paths (k_gather, rank gaps, select gather)
# ---------------------------------------------------------------------------


def _om_apply_ir(elems, inputs):
    root = Op.make("om_apply", _compose("om_compose", elems))
    names = {v.name for v in inputs}
    return IR(
        root=root, inputs=list(inputs), input_names=names, params={}
    )


def _qk_leaf(q, k, v):
    return Op.make(
        "om_elem",
        Op.make("matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)),
        v,
    )


def test_om_batched_k_gather_on_nonkey_chunk_dim():
    """k_i = chunks of one base along the FEATURE axis → k_gather is
    built but dense_qk fails → eval_sliced emits the batched stack."""
    n, T, d, dv = 2, 6, 4, 4
    q = _v("q", T, d)
    Kb = _v("Kb", 4, n * d)
    vs = [_v(f"v{i}", 4, dv) for i in range(n)]
    elems = [
        _qk_leaf(
            q, Op.make("chunk", Kb, chunks=n, dim=-1, index=i), vs[i]
        )
        for i in range(n)
    ]
    ir = _om_apply_ir(elems, [q, Kb, *vs])
    mod = to_batched_om_module(ir)
    plan = build_om_plan(mod.eval_mod._root)
    grp = plan["leaf_groups"][0]
    assert grp["s_mode"] == "bmm_qk" and grp["k_gather"] is not None
    gen = ir_to_torch_module(ir)
    tq, tK = _rand((T, d), 1), _rand((4, n * d), 2)
    tv = [_rand((4, dv), 3 + i) for i in range(n)]
    with torch.no_grad():
        torch.testing.assert_close(mod(tq, tK, *tv), gen(tq, tK, *tv))


def test_om_bmm_qk_rank_gap_expands():
    """q rank 2 vs stacked k rank 3 → q is reshaped+expanded;
    q rank 4 vs k rank 2 → k is reshaped+expanded."""
    n, B, H, T, d, dv = 3, 2, 2, 6, 4, 4

    # gap > 0: q (T,d), k_i (B,K,d)
    q = _v("q", T, d)
    ks = [_v(f"k{i}", B, 4, d) for i in range(n)]
    vs = [_v(f"v{i}", B, 4, dv) for i in range(n)]
    ir = _om_apply_ir(
        [_qk_leaf(q, k, v) for k, v in zip(ks, vs, strict=True)],
        [q, *ks, *vs],
    )
    mod = to_batched_om_module(ir)
    assert (
        build_om_plan(mod.eval_mod._root)["leaf_groups"][0]["s_mode"]
        == "bmm_qk"
    )
    gen = ir_to_torch_module(ir)
    tq = _rand((T, d), 1)
    tk = [_rand((B, 4, d), 2 + i) for i in range(n)]
    tv = [_rand((B, 4, dv), 10 + i) for i in range(n)]
    with torch.no_grad():
        torch.testing.assert_close(mod(tq, *tk, *tv), gen(tq, *tk, *tv))

    # gap < 0: q (B,H,T,d), k_i (K,d); v must carry the batch dims
    q2 = _v("q2", B, H, T, d)
    ks2 = [_v(f"k{i}", 4, d) for i in range(n)]
    vs2 = [_v(f"v{i}", B, H, 4, dv) for i in range(n)]
    ir2 = _om_apply_ir(
        [_qk_leaf(q2, k, v) for k, v in zip(ks2, vs2, strict=True)],
        [q2, *ks2, *vs2],
    )
    mod2 = to_batched_om_module(ir2)
    assert (
        build_om_plan(mod2.eval_mod._root)["leaf_groups"][0]["s_mode"]
        == "bmm_qk"
    )
    gen2 = ir_to_torch_module(ir2)
    tq2 = _rand((B, H, T, d), 20)
    tk2 = [_rand((4, d), 30 + i) for i in range(n)]
    tv2 = [_rand((B, H, 4, dv), 40 + i) for i in range(n)]
    with torch.no_grad():
        torch.testing.assert_close(
            mod2(tq2, *tk2, *tv2), gen2(tq2, *tk2, *tv2)
        )


def test_om_batched_select_v_gather():
    """v_i = select(V, 0, i) → eval_sliced hits the movedim identity
    for the 'select' kind; batched still matches serial."""
    n, B, T, dv = 3, 2, 6, 4
    s, V = _v("s", B, T, 4), _v("V", n, B, 4, dv)
    elems = [
        Op.make("om_elem", s, Op.make("select", V, dim=0, index=i))
        for i in range(n)
    ]
    ir = _om_apply_ir(elems, [s, V])
    mod = to_batched_om_module(ir)
    grp = build_om_plan(mod.eval_mod._root)["leaf_groups"][0]
    assert grp["v_gather"][1] == "select" and grp["v_gather"][2] == 0
    gen = ir_to_torch_module(ir)
    ts, tV = _rand((B, T, 4), 1), _rand((n, B, 4, dv), 2)
    with torch.no_grad():
        torch.testing.assert_close(mod(ts, tV), gen(ts, tV))


def test_om_compile_replay_and_capture_edges(monkeypatch):
    B, T, d, dv = 2, 6, 4, 4
    q, k, v = _v("q", B, T, d), _v("k", B, 4, d), _v("v", B, 4, dv)
    leaf = _qk_leaf(q, k, v)
    ir = _om_apply_ir([leaf, _qk_leaf(q, k, v)], [q, k, v])
    mod = to_batched_om_module(ir)
    tq, tk, tv = (
        _rand((B, T, d), 1),
        _rand((B, 4, d), 2),
        _rand((B, 4, dv), 3),
    )
    with torch.no_grad():
        ref = mod(tq, tk, tv)
    mod.compile(backend="eager")
    with torch.no_grad():
        torch.testing.assert_close(mod(tq, tk, tv), ref)

    class _FakeGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    sentinel = torch.full((1,), 9.0)
    mod._graph = _FakeGraph()
    mod._graph_inputs = [tq, tk, tv]
    mod._graph_out = sentinel
    assert mod(tq.clone(), tk, tv) is sentinel
    assert mod._graph.replays == 1
    # mismatched shape falls through to eager
    assert mod(torch.zeros(1, 1, 1)).shape == (1, 1, 1)
    mod.drop_cuda_graph()

    # module properties + CPU no-op capture
    assert mod.is_batched and mod.n_levels >= 1 and mod.n_blocks >= 1
    assert not mod.is_graph_captured
    assert mod.capture_cuda_graph(tq, tk, tv) is mod  # CPU: no-op
    assert mod._graph is None

    # capture on CPU tensors under a fake CUDA flag → ValueError
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        mod.capture_cuda_graph(torch.randn(2, 3))
    except ValueError as e:
        assert "CUDA" in str(e)
    else:  # pragma: no cover — defensive
        raise AssertionError("expected ValueError for CPU inputs")

    # a non-batched module compiles/captures as a no-op too
    plain = to_batched_om_module(
        IR(
            root=Op.make("relu", _v("p", 4)),
            inputs=[_v("p", 4)],
            input_names={"p"},
            params={},
        )
    )
    assert (
        not plain.is_batched
        and plain.n_levels == 0
        and plain.n_blocks == 0
    )
    assert (
        plain.compile() is plain and plain.capture_cuda_graph() is plain
    )


# ---------------------------------------------------------------------------
#  omd_lower — walker rejections and forest/serial-eval paths
# ---------------------------------------------------------------------------


def test_omd_leaf_sig_opaque_two_arg_op():
    """A 2-arg op that is no carrier leaf returns no signature."""
    assert (
        _leaf_sig(Op.make("add", _p("a", 4), _p("b", 4)), "diag")
        is None
    )


def test_omd_plan_rejections():
    """The deferred-affine walk declines the invalid map shapes it
    documents — each maps to a specific guard."""
    d = 4
    A, bb, a, b = _p("A", d, d), _p("bb", d), _p("a", d), _p("b", d)
    s, h, bm = _p("s", 3, 2), _p("h", d), _p("bm", 2)

    # same map object registered under both domains → conflict
    mp = Op.make("aff", A, bb)
    conflict = Op.make(
        "omd_apply",
        Op.make(
            "omd_elem",
            s,
            Op.make("affd_a", mp),
            Op.make("aff_A", mp),
        ),
        h,
    )
    assert build_omd_plan(conflict) is None

    # a stack of projections INSIDE a map leaf's args is rejected
    inner = Op.make(
        "affd_compose",
        Op.make("aff_diag", a, b),
        Op.make("aff_diag", a, _v("x", d)),
    )
    leaf_with_stack = Op.make(
        "aff_diag",
        Op.make(
            "stack",
            Op.make("affd_a", inner),
            Op.make("affd_a", inner),
            dim=0,
        ),
        _v("x", d),
    )
    bad_leaf = Op.make(
        "omd_apply",
        Op.make(
            "omd_elem",
            s,
            Op.make(
                "affd_a",
                Op.make(
                    "affd_compose",
                    Op.make("aff_diag", a, b),
                    leaf_with_stack,
                ),
            ),
            bm,
        ),
        h,
    )
    assert build_omd_plan(bad_leaf) is None

    # dense aff_compose under a diag projection → wrong domain compose
    dense = Op.make(
        "aff_compose", Op.make("aff", A, bb), Op.make("aff", A, bb)
    )
    wrong_dom = Op.make(
        "omd_apply",
        Op.make("omd_elem", s, Op.make("affd_a", dense), bm),
        h,
    )
    assert build_omd_plan(wrong_dom) is None

    # prefix chains of one domain with non-uniform signatures →
    # neither chain nor forest can host them
    l1 = Op.make("aff_diag", _p("a1", 3), _p("b1", 3))
    l2 = Op.make("aff_diag", _p("a2", 4), _p("b2", 4))
    nonuni = Op.make(
        "omd_apply",
        Op.make(
            "omd_elem",
            s,
            Op.make("affd_a", Op.make("affd_compose", l2, l1)),
            bm,
        ),
        h,
    )
    assert build_omd_plan(nonuni) is None


def test_omd_plan_mixed_domains_go_forest():
    """A stack mixing diag and dense projections takes the forest
    path (the prefix-chain fast path requires one domain)."""
    d = 4
    a, b, A, bb = _p("a", d), _p("b", d), _p("A", d, d), _p("bb", d)
    mpd = Op.make(
        "affd_compose",
        Op.make("aff_diag", a, b),
        Op.make("aff_diag", _p("a2", d), _p("b2", d)),
    )
    mpA = Op.make(
        "aff_compose",
        Op.make("aff", A, bb),
        Op.make("aff", _p("A2", d, d), _p("bb2", d)),
    )
    term = Op.make(
        "omd_apply",
        Op.make(
            "omd_elem",
            _p("s", 3, 2),
            Op.make(
                "stack",
                Op.make("affd_a", mpd),
                Op.make("aff_A", mpA),
                dim=0,
            ),
            _p("bm", 2),
        ),
        _p("h", d),
    )
    plan = build_omd_plan(term)
    assert plan is not None and plan["map_mode"] == "forest"
    assert set(plan["forest"]) == {"diag", "dense"}


def _omd_chain(T, d, gather_dim=0, idxs=None, stack_dim=0):
    """A chain-mode omd_apply term over select-sliced aff_diag leaves.

    ``gather_dim`` selects the base-tensor axis the leaf parts read;
    ``idxs`` chooses non-contiguous indices; ``stack_dim`` rotates the
    projection stacks (wrapped in a transpose when non-zero)."""
    idxs = list(range(T)) if idxs is None else idxs
    N = max(idxs) + 1
    a_shape = (d, N) if gather_dim else (N, d)
    x_shape = (d, N) if gather_dim else (N, d)
    a_p, x_v = _p("p_a", *a_shape), _v("x", *x_shape)
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a_p, dim=gather_dim, index=i),
            Op.make("select", x_v, dim=gather_dim, index=i),
        )
        for i in idxs
    ]
    fs = [leaves[0]]
    for t in range(1, len(idxs)):
        fs.append(Op.make("affd_compose", leaves[t], fs[-1]))

    def _stk(proj):
        node = Op.make("stack", *(proj(f) for f in fs), dim=stack_dim)
        if stack_dim:
            node = Op.make("transpose", node, dim0=0, dim1=1)
        return node

    s_p = _p("s", 3, len(idxs))
    h_p = _p("h", d)
    term = Op.make(
        "omd_apply",
        Op.make(
            "omd_elem",
            s_p,
            _stk(lambda f: Op.make("affd_a", f)),
            _stk(lambda f: Op.make("affd_b", f)),
        ),
        h_p,
    )
    return term, x_v, a_p, s_p, h_p


def test_omd_chain_stack_on_nonzero_dim():
    """stack(affd_*, dim=1) seeds get the movedim spine — output
    still equals the serial evaluation."""
    T, d = 6, 4
    term, x_v, a_p, s_p, h_p = _omd_chain(T, d, stack_dim=1)
    plan = build_omd_plan(term)
    assert plan["map_mode"] == "chain" and len(plan["stack_seeds"]) == 2
    ir = IR(root=term, inputs=[x_v], params={})
    pv = {
        "p_a": _rand((T, d), 1).clamp(-0.9, 0.9),
        "s": _rand((3, T), 2),
        "h": _rand((d,), 3),
    }
    mod = to_batched_omd_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        torch.testing.assert_close(
            mod(_rand((T, d), 4)), gen(_rand((T, d), 4))
        )
    assert mod.fallbacks == 0


def test_omd_chain_gathers_nonzero_dim_and_sparse_idx():
    """Leaf parts along dim 1 → movedim gathers; non-contiguous index
    lists → index_select.  Both agree with serial eval."""
    T, d = 6, 4
    term, x_v, a_p, s_p, h_p = _omd_chain(T, d, gather_dim=1)
    plan = build_omd_plan(term)
    assert (
        plan["chain_a_gather"][1] == 1
        and plan["chain_b_gather"][1] == 1
    )
    ir = IR(root=term, inputs=[x_v], params={})
    pv = {
        "p_a": _rand((d, T), 1).clamp(-0.9, 0.9),
        "s": _rand((3, T), 2),
        "h": _rand((d,), 3),
    }
    xv = _rand((d, T), 4)
    mod = to_batched_omd_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        torch.testing.assert_close(mod(xv), gen(xv))
    assert mod.fallbacks == 0

    idxs = [0, 2, 4, 6, 8, 10]
    term2, x_v2, a_p2, s_p2, h_p2 = _omd_chain(len(idxs), d, idxs=idxs)
    plan2 = build_omd_plan(term2)
    assert plan2["chain_a_gather"][2] == idxs
    ir2 = IR(root=term2, inputs=[x_v2], params={})
    pv2 = {
        "p_a": _rand((12, d), 1).clamp(-0.9, 0.9),
        "s": _rand((3, len(idxs)), 2),
        "h": _rand((d,), 3),
    }
    mod2 = to_batched_omd_module(ir2, param_values=pv2)
    gen2 = ir_to_torch_module(ir2, param_values=pv2)
    with torch.no_grad():
        torch.testing.assert_close(
            mod2(_rand((12, d), 4)), gen2(_rand((12, d), 4))
        )
    assert mod2.fallbacks == 0


def test_omd_batched_with_packaging_leaf_and_nonuniform_vals():
    """An ``omd`` packaging node inside the compose tree falls to the
    generic evaluator; leaves whose triples disagree in shape force
    the serial per-leaf compose loop — both still correct."""
    Tq, T, d = 3, 6, 4
    s_p, h_p = _p("s", Tq, T), _p("h", d)
    pkg = Op.make(
        "omd",
        _p("m", Tq, 1),
        _p("l", Tq, 1),
        _p("fa", Tq, d),
        _p("fb", Tq, d),
    )
    elem = Op.make("omd_elem", s_p, _p("aa", T, d), _p("bb", T, d))
    term = Op.make("omd_apply", Op.make("omd_compose", elem, pkg), h_p)
    dummy = _v("dummy", 1)  # IRModule.forward takes at least one arg
    ir = IR(root=term, inputs=[dummy], input_names={"dummy"}, params={})
    pv = {
        "s": _rand((Tq, T), 1),
        "aa": _rand((T, d), 2),
        "bb": _rand((T, d), 3),
        "h": _rand((d,), 4),
        "m": _rand((Tq, 1), 5),
        "l": _rand((Tq, 1), 6).abs() + 0.5,
        "fa": _rand((Tq, d), 7),
        "fb": _rand((Tq, d), 8),
    }
    mod = to_batched_omd_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        torch.testing.assert_close(
            mod(torch.zeros(1)), gen(torch.zeros(1))
        )
    assert mod.fallbacks == 0

    # a leaf whose triple broadcast-shapes differ from the first
    # leaf's → non-uniform → the serial per-leaf compose loop runs
    # (the triples broadcast pairwise, so semantics is preserved)
    comp = Op.make(
        "omd_compose",
        Op.make(
            "omd_elem", _p("s1", 3, 4), _p("a1", 4, d), _p("b1", 4, d)
        ),
        Op.make(
            "omd_elem", _p("s2", 1, 4), _p("a2", 4, d), _p("b2", 4, d)
        ),
    )
    ir2 = IR(
        root=Op.make("omd_apply", comp, _v("h", d)),
        inputs=[_v("h", d)],
        input_names={"h"},
        params={},
    )
    pv2 = {
        "s1": _rand((3, 4), 1),
        "s2": _rand((1, 4), 2),
        "a1": _rand((4, d), 3),
        "b1": _rand((4, d), 4),
        "a2": _rand((4, d), 5),
        "b2": _rand((4, d), 6),
    }
    mod2 = to_batched_omd_module(ir2, param_values=pv2)
    gen2 = ir_to_torch_module(ir2, param_values=pv2)
    hv = _rand((d,), 7)
    with torch.no_grad():
        torch.testing.assert_close(mod2(hv), gen2(hv))
    assert mod2.fallbacks == 0


def test_omd_capture_and_replay_edges(monkeypatch):
    # a real chain plan so capture proceeds to the input check
    T, d = 4, 3
    term, x_v, a_p, s_p, h_p = _omd_chain(T, d)
    ir = IR(root=term, inputs=[x_v], input_names={"x"}, params={})
    pv = {
        "p_a": _rand((T, d), 1).clamp(-0.9, 0.9),
        "s": _rand((3, T), 2),
        "h": _rand((d,), 3),
    }
    xv = _rand((T, d), 4)
    mod = to_batched_omd_module(ir, param_values=pv)
    assert mod.is_batched
    # CPU no-op capture leaves no graph
    assert mod.capture_cuda_graph(xv) is mod and mod._graph is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        mod.capture_cuda_graph(xv)
    except ValueError as e:
        assert "CUDA" in str(e)
    else:  # pragma: no cover — defensive
        raise AssertionError("expected ValueError for CPU inputs")
    monkeypatch.undo()

    class _FakeGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    sentinel = torch.full((1,), 5.0)
    mod._graph = _FakeGraph()
    mod._graph_inputs = [xv]
    mod._graph_out = sentinel
    assert mod(xv.clone()) is sentinel
    mod.drop_cuda_graph()


# ---------------------------------------------------------------------------
#  trace — usize helpers and law check/derive guards
# ---------------------------------------------------------------------------


def _b2d(name, r, c):
    return {name: _p(name, r, c)}


def test_usize_helpers():
    assert _usize_total(4) == 4
    assert _usize_total((2, 3)) == 5
    assert _usize_total(2.0) == 2
    assert _usize_total("dim") == 0
    assert _usize_total(None) == 0
    assert _usize_split2({"$attr:UV": (2, 3)}) == (2, 3)
    assert _usize_split2({"$attr:UV": 5}) is None
    assert _usize_split2({"$attr:UV": (1, 2, 3)}) is None
    assert _usize_split2({"$attr:UV": (2, "x")}) is None
    assert _usize_split2({}) is None


def test_trace_vanish_checks_and_derives():
    # split side: missing usable-size attr → veto
    bound = {"f": _p("f", 8, 8)}
    assert not _check_vanish_split(bound)
    assert _derive_vanish_split(bound) is None
    bound = {**bound, "$attr:UV": (3, 2)}
    assert _check_vanish_split(bound)
    assert _derive_vanish_split(bound) == {
        "$attr:DU": 3,
        "$attr:DV": 2,
    }
    # f too small for the split → veto
    assert not _check_vanish_split(
        {"f": _p("f2", 2, 2), "$attr:UV": (3, 2)}
    )

    # merge side: needs explicit dims and a 2-D f at least du+dv wide
    bound = {"f": _p("f", 6, 6), "$attr:DU": 4, "$attr:DV": 2}
    assert _check_vanish_merge(bound)
    assert not _check_vanish_merge({**bound, "$attr:DV": "x"})
    assert not _check_vanish_merge({**bound, "f": _p("f2", None, 6)})
    assert not _check_vanish_merge({**bound, "f": _p("f3", 4, 6)})


def test_trace_superpose_and_tighten():
    bound = {
        "f": _p("f", 5, 7),
        "g": _p("g", 6, 8),
        "$attr:DU": 3,
        "$attr:DV": 2,
        "$attr:UV": (3, 2),
    }
    # every dim has room for its traced block
    assert _check_superpose(bound)
    # UV must account for both traced blocks
    assert not _check_superpose({**bound, "$attr:UV": (2, 2)})
    # f must hold its du-block in BOTH dims; g likewise for dv
    assert not _check_superpose({**bound, "f": _p("f2", 2, 7)})
    assert not _check_superpose({**bound, "g": _p("g2", 5, 1)})
    # unresolvable shapes → veto
    assert not _check_superpose(
        {**bound, "f": _p("f3", None, 7), "g": _p("g3", 1, 4)}
    )
    # missing dims → veto
    assert not _check_superpose(
        {"f": _p("f", 5, 7), "g": _p("g", 6, 6)}
    )

    # tighten_out: fs[0] must be du + k's out-dim; tighten_in: fs[1]
    # must be du + j's in-dim — the context map is folded through.
    b_out = {"f": _p("f", 6, 9), "k": _p("k", 7, 4), "$attr:DU": 2}
    assert _check_tighten_out(b_out)
    assert not _check_tighten_out({**b_out, "k": _p("k2", 7, 5)})
    assert not _check_tighten_out({**b_out, "k": _p("k3", None, 4)})
    b_in = {"f": _p("f", 6, 9), "j": _p("j", 7, 4), "$attr:DU": 2}
    assert _check_tighten_in(b_in)
    assert not _check_tighten_in({**b_in, "j": _p("j2", 8, 4)})


def test_trace_slide_checks_and_derives():
    # slide: g (dy, du+dx) widens on the output side
    good = {
        "h": _p("h", 3, 3),
        "g": _p("g", 5, 7),
        "$attr:DU": 3,
        "$attr:DX": 4,
    }
    assert _check_slide(good)
    assert _derive_slide(good) == {"$attr:DY": 2}
    assert not _check_slide({**good, "$attr:DU": "x"})
    assert not _check_slide({**good, "g": _p("gx", None, 7)})
    assert not _check_slide({**good, "h": _p("h2", 3, 5)})
    assert not _check_slide({**good, "g": _p("g2", 5, 6)})
    assert not _check_slide({**good, "g": _p("g3", 3, 7)})
    assert _derive_slide({**good, "g": _p("g4", None, 7)}) is None
    assert _derive_slide({**good, "$attr:DU": "x"}) is None
    assert _derive_slide({**good, "g": _p("g5", 3, 7)}) is None

    # slide_rev: g (du+dy, dy) widens on the input side
    good = {
        "h": _p("h", 3, 3),
        "g": _p("g", 7, 5),
        "$attr:DU": 3,
        "$attr:DY": 4,
    }
    assert _check_slide_rev(good)
    assert _derive_slide_rev(good) == {"$attr:DX": 2}
    assert not _check_slide_rev({**good, "$attr:DY": "x"})
    assert not _check_slide_rev({**good, "g": _p("gx", None, 5)})
    assert not _check_slide_rev({**good, "g": _p("g2", 6, 5)})
    assert not _check_slide_rev({**good, "g": _p("g3", 7, 3)})
    assert _derive_slide_rev({**good, "g": _p("g4", None, 5)}) is None


def test_trace_expand_and_collapse():
    # expand: f wider than the traced du block in both dims
    bound = {"f": _p("f", 6, 8), "$attr:DU": 4}
    assert _check_expand(bound)
    # f=(du+2, du+4) → the expand splits each axis at du
    assert _derive_expand(bound) == {
        "$attr:RS": (4, 2),
        "$attr:CS": (4, 4),
    }
    assert not _check_expand({**bound, "$attr:DU": "x"})
    assert not _check_expand({**bound, "$attr:DU": 0})
    assert not _check_expand({**bound, "f": _p("f2", None, 8)})
    assert not _check_expand({**bound, "f": _p("f3", 4, 8)})
    assert not _check_expand({**bound, "f": _p("f4", 6, 4)})
    assert _derive_expand({**bound, "f": _p("f5", None, 8)}) is None
    assert _derive_expand({**bound, "$attr:DU": "x"}) is None

    # collapse: the split RS/CS must each start with the traced block
    bound = {"$attr:DU": 4, "$attr:RS": (4, 2), "$attr:CS": (4, 3)}
    assert _check_collapse(bound)
    assert not _check_collapse({**bound, "$attr:DU": "x"})
    assert not _check_collapse({**bound, "$attr:DU": 0})
    assert not _check_collapse({"$attr:DU": 4})
    assert not _check_collapse({**bound, "$attr:RS": (3, 2)})
    assert not _check_collapse({**bound, "$attr:CS": (5, 3)})
    assert not _check_collapse({**bound, "$attr:RS": (4,)})


def _eval_root(term, pv, *xs):
    """Lower a closed term to a module and run it (a dummy input keeps
    IRModule's required forward arg satisfied)."""
    dummy = Var("z", TensorType((1,)))
    ir = IR(root=term, inputs=[dummy], input_names={"z"}, params={})
    mod = ir_to_torch_module(ir, param_values=pv)
    arg = xs[0] if xs else torch.zeros(1, dtype=torch.float64)
    with torch.no_grad():
        return mod(arg)


def test_trace_ops_numeric_laws():
    """The traced-category torch bindings satisfy the laws they
    implement — evaluated end-to-end through IRModule."""
    f = _rand((7, 6), 1)  # feedback block 2
    g = _rand((5, 5), 2)  # feedback block 1
    pv = {"F": f, "G": g}
    # Superposing: Tr^{U⊗V}(f ⊗ g) = Tr^U(f) ⊕ Tr^V(g)
    lhs = _eval_root(
        Op.make(
            "trace",
            Op.make("parl", _p("F", 7, 6), _p("G", 5, 5), u1=2, u2=1),
            usize=3,
        ),
        pv,
    )
    rhs = _eval_root(
        Op.make(
            "bdiag",
            Op.make("trace", _p("F", 7, 6), usize=2),
            Op.make("trace", _p("G", 5, 5), usize=1),
        ),
        pv,
    )
    assert lhs.shape == rhs.shape
    torch.testing.assert_close(lhs, rhs)

    # Yanking: Tr^U(swap) = id_U — exercises cswap+trace+eye.
    lhs = _eval_root(
        Op.make("trace", Op.make("cswap", d1=3, d2=3), usize=3), {}
    )
    # cswap/eye mint at the default dtype — compare in its dtype
    torch.testing.assert_close(lhs, torch.eye(3, dtype=lhs.dtype))

    # Vanishing: Tr^0(F) = F (usize must be ≥1 for the binding to
    # act — a usize below the block size leaves f untouched)
    out = _eval_root(Op.make("trace", _p("F", 7, 6), usize=0), pv)
    assert torch.equal(out, f)

    # inv / bdiag bindings on real tensors
    m = _rand((4, 4), 3)
    out = _eval_root(
        Op.make(
            "bdiag",
            Op.make("inv", _p("M", 4, 4)),
            Op.make("eye", dim=2),
        ),
        {"M": m},
    )
    torch.testing.assert_close(
        out,
        torch.block_diag(
            torch.linalg.inv(m), torch.eye(2, dtype=torch.float64)
        ),
    )


# ---------------------------------------------------------------------------
#  trace_lift — term resolution + structural decline paths
# ---------------------------------------------------------------------------


def test_lift_consistent_shapes_rejections():
    eg = EGraph()
    d = 3
    h0 = eg.add_term(_p("h0", d))
    m = eg.add_term(_p("m", d))
    i = eg.add_term(_p("i", d))
    assert TL._consistent_shapes(eg, "diag", [m], [i], h0) == d
    # h0 not 1-D → None
    bad_h0 = eg.add_term(_p("h0b", d, d))
    assert TL._consistent_shapes(eg, "diag", [m], [i], bad_h0) is None
    # dense kind needs (d,d) maps — a (d,) map declines
    assert TL._consistent_shapes(eg, "dense", [m], [i], h0) is None
    # a wrong-shaped in declines
    bad_i = eg.add_term(_p("ib", d, d))
    assert TL._consistent_shapes(eg, "diag", [m], [bad_i], h0) is None


def test_prefer_term_edges():
    eg = EGraph()
    # cid already in _seen → the metavar guard fires
    x = eg.add_term(_v("x", 2))
    assert (
        TL._prefer_term(
            eg, x, frozenset({"apply"}), _seen=frozenset({x})
        )
        is None
    )
    # a class absent from the graph → None
    del eg._classes[x]
    assert TL._prefer_term(eg, x, frozenset()) is None

    # preferred op member tried first but cyclic → falls to the leaf
    eg2 = EGraph()
    x2 = eg2.add_term(_v("x2", 2))
    a2 = eg2.add_enode("apply", (x2, x2))
    eg2.union(a2, x2)
    M = eg2.find(a2)
    term = TL._prefer_term(eg2, M, TL._CARRIER_OPS)
    assert isinstance(term, Var) and term.name == "x2"

    # a member whose child resolves to a cyclic-only class fails the
    # recursion; with no other member the class resolves to None.
    eg3 = EGraph()
    cyc = _cyclic_eclass(eg3)
    leaf = eg3.add_term(_v("lf", 2))
    P = eg3.add_enode("foo", (cyc, leaf))
    assert TL._prefer_term(eg3, P, frozenset()) is None


def test_carrier_plan_declines():
    # a class containing an apply member that cannot resolve to a
    # term (cyclic-only) → None before any plan work
    eg = EGraph()
    x = eg.add_term(_v("x", 2))
    a = eg.add_enode("apply", (x, x))
    eg.union(a, x)
    M = eg.find(a)
    for n in list(eg._classes[M].nodes):
        if n.op == "leaf":
            eg._classes[M].nodes.discard(n)
    assert TL._carrier_plan(eg, M) is None

    # apply/applyd member exists but the map is no affine carrier tree
    eg2 = EGraph()
    bad = eg2.add_term(
        Op.make("apply", Op.make("relu", _p("m", 4)), _p("h", 4))
    )
    assert TL._carrier_plan(eg2, eg2.find(bad)) is None


def _spine(eg, h0, A, X, T):
    """``h_t = A[t] ⊙ h_{t-1} + X[t]`` built over select slices."""
    term, eid = h0, eg.add_term(h0)
    for t in range(T):
        a_t = Op.make("select", A, dim=0, index=t)
        b_t = Op.make("select", X, dim=0, index=t)
        term = Op.make("add", Op.make("mul", a_t, term), b_t)
        eid = eg.add_term(term)
    return eid


def test_spine_walk_edges():
    d = 3
    # ambiguous decomposition: both mul factors walk to a base at the
    # same length with different signatures → vetoed, not guessed
    eg = EGraph()
    tie = eg.add_term(
        Op.make(
            "add", Op.make("mul", _p("ma", d), _p("h7", d)), _p("i7", d)
        )
    )
    assert TL._Spine(eg).plan(eg.find(tie)) is None

    # a self-referential step chain is cut by the active-set guard and
    # the walk still returns the OTHER, finite decomposition
    eg2 = EGraph()
    x4 = eg2.add_term(_v("x4", d))
    mu = eg2.add_enode("mul", (eg2.add_term(_p("pa", d)), x4))
    inn = eg2.add_term(_p("pi", d))
    ad = eg2.add_enode("add", (mu, inn))
    eg2.union(ad, x4)
    M = eg2.find(ad)
    res = TL._Spine(eg2)._walk(M)
    assert res is not None and len(res[0]) == 1

    # deleted product/state classes are skipped, not crashed on
    eg3 = EGraph()
    t3 = eg3.add_term(
        Op.make(
            "add", Op.make("mul", _p("m3", d), _p("h3", d)), _p("i3", d)
        )
    )
    rc = eg3.find(t3)
    mul_cid = next(
        eg3.find(c)
        for n in eg3._classes[rc].nodes
        for c in n.children
        if n.op == "add"
    )
    del eg3._classes[mul_cid]
    assert TL._Spine(eg3).plan(rc) is None

    eg4 = EGraph()
    h5 = _p("h5", d)
    t5 = eg4.add_term(
        Op.make("add", Op.make("mul", _p("m5", d), h5), _p("i5", d))
    )
    del eg4._classes[eg4.add_term(h5)]
    res = TL._Spine(eg4)._walk(eg4.find(t5))
    # the deleted state declines; the sibling factor decomposition
    # may still win but nothing may crash
    assert res is None or isinstance(res[0], list)

    # a product class carrying BOTH mul and matmul members yields
    # candidates of each kind from one child
    eg5 = EGraph()
    h5b = _p("h5b", d)
    a5 = eg5.add_term(_p("a5", d))
    mu = eg5.add_enode("mul", (a5, eg5.add_term(h5b)))
    mm = eg5.add_enode(
        "matmul", (eg5.add_term(_p("A5", d, d)), eg5.add_term(h5b))
    )
    eg5.union(mu, mm)
    pair = eg5.find(mu)
    root5 = eg5.add_enode("add", (pair, eg5.add_term(_p("i5", d))))
    res5 = TL._Spine(eg5)._walk(eg5.find(root5))
    assert res5 is None or isinstance(res5[0], list)

    # two add members yielding the SAME (map, state, in) decomposition
    # — the dedup arm keeps the first, not a tie
    eg6 = EGraph()
    h6 = _p("h6", d)
    sel = Op.make("select", _p("A6", 1, d), dim=0, index=0)
    i6 = _p("i6", d)
    n1 = eg6.add_enode(
        "add", (eg6.add_term(Op.make("mul", sel, h6)), eg6.add_term(i6))
    )
    n2 = eg6.add_enode(
        "add", (eg6.add_term(i6), eg6.add_term(Op.make("mul", sel, h6)))
    )
    eg6.union(n1, n2)
    res6 = TL._Spine(eg6)._walk(eg6.find(n1))
    assert res6 is not None and len(res6[0]) == 1


def test_spine_shorter_alternative_candidate():
    """A len-2 decomposition found first isn't displaced by the
    sibling's len-1 alternative — the elif's False arm runs."""
    d = 3
    eg = EGraph()
    h6 = _p("h6", d)
    A6, X6 = _p("A6", 2, d), _p("X6", 2, d)
    step_t = Op.make(
        "add",
        Op.make("mul", Op.make("select", A6, dim=0, index=0), h6),
        Op.make("select", X6, dim=0, index=0),
    )
    root_t = Op.make(
        "add",
        Op.make("mul", Op.make("select", A6, dim=0, index=1), step_t),
        Op.make("mul", Op.make("select", X6, dim=0, index=1), h6),
    )
    root = eg.add_term(root_t)
    plan = TL._Spine(eg).plan(eg.find(root))
    assert plan is not None and plan.T == 2


def test_spine_mixed_domains_and_lift_prefix_checks():
    """Mixed dense/diag steps decline; multiple spine plans trigger
    the interior-h0 and prefix-mismatch guard arms in maximal_only."""
    d = 3
    eg = EGraph()
    inner_t = Op.make(
        "add",
        Op.make("matmul", _p("MA", d, d), _p("hb", d)),
        _p("i1", d),
    )
    root_t = Op.make(
        "add", Op.make("mul", _p("ma", d), inner_t), _p("i2", d)
    )
    root = eg.add_term(root_t)
    assert TL._Spine(eg).plan(eg.find(root)) is None

    # three raw spines: two share h0 with different maps (prefix check
    # must inspect and reject), one has a different h0 (skipped early)
    eg2 = EGraph()
    h0 = _p("h0", d)
    h0b = _p("h0b", d)
    _spine(eg2, h0, _p("A1", 4, d), _p("X1", 4, d), 4)
    _spine(eg2, h0, _p("A2", 2, d), _p("X2", 2, d), 2)
    _spine(eg2, h0b, _p("A3", 3, d), _p("X3", 3, d), 3)
    lifts = TL.lift_scan_to_trace(eg2, min_steps=2, channel_splits=[])
    assert sorted(lift.T for lift in lifts) == [2, 3, 4]

    # a stale e-class key (its canonical target was merged away) is
    # skipped by the liveness guard rather than crashing the pass
    eg3 = EGraph()
    x = eg3.add_term(_v("zz", 2))
    f = eg3.add_enode("neg", (x,))
    eg3.union(f, x)
    merged = eg3.find(f)
    del eg3._classes[merged]
    eg3._classes[x] = EClass(id=x)  # stale key: find(x) → deleted canon
    assert TL.lift_scan_to_trace(eg3, min_steps=2) == []


def test_emit_and_offer_broken_refs():
    """A member ref that cannot resolve marks the emit broken; the
    joint offer is skipped and channel splits continue instead."""
    eg = EGraph()
    d = 4
    cyc = _cyclic_eclass(eg)
    good = eg.add_term(_p("good", d))
    cid = eg.add_term(_p("root", d))
    plan = TL._Plan(
        "diag", 2, d, maps=[cyc, cyc], ins=[good, good], h0=good
    )
    lifts = TL._offer(eg, cid, plan, [(2, 2)], "test", True)
    assert lifts == []

    em = TL._Emit(eg, "prov")
    eid_out, term = em.ref(cyc)
    assert eid_out == cyc and term is None and em.broken


def test_lift_witness_no_source():
    """When no representative exists the witness is None — the offer
    still returns a plan (the union just isn't replayable)."""
    eg = EGraph()
    cyc = _cyclic_eclass(eg)
    h = _p("h", 4)
    # normal path: a src exists → a named witness
    cid = eg.add_term(_p("r", 4))
    w = TL._lift_witness(eg, cid, Op.make("add", h, h), 0, "prov")
    assert w is not None and w.name.startswith("prov#")
    # cyclic src → no witness
    assert (
        TL._lift_witness(eg, cyc, Op.make("add", h, h), 0, "prov")
        is None
    )


# ---------------------------------------------------------------------------
#  trace_lift — real lifts: dense spines, carrier plans, channel splits
# ---------------------------------------------------------------------------


def _dense_spine(eg, h0, A, IX, T):
    """``h_t = A_t @ h_{t-1} + I_t`` over select slices of dense maps."""
    term, eid = h0, eg.add_term(h0)
    for t in range(T):
        a_t = Op.make("select", A, dim=0, index=t)
        i_t = Op.make("select", IX, dim=0, index=t)
        term = Op.make("add", Op.make("matmul", a_t, term), i_t)
        eid = eg.add_term(term)
    return eid


def test_lift_dense_spine_emits_joint_trace():
    """A dense matmul spine yields a joint trace offer; dense carriers
    get no channel partitions."""
    d = 3
    eg = EGraph()
    _dense_spine(eg, _p("h0", d), _p("A", 3, d, d), _p("I", 3, d), 3)
    lifts = TL.lift_scan_to_trace(eg, min_steps=2)
    assert (
        len(lifts) == 1 and lifts[0].kind == "dense" and lifts[0].T == 3
    )


def test_lift_diag_channel_split_emission():
    """channel_splits='auto' emits the joint trace AND the balanced
    parl channel-split member for a diagonal spine."""
    d = 4
    eg = EGraph()
    _spine(eg, _p("h0", d), _p("A", 4, d), _p("X", 4, d), 4)
    lifts = TL.lift_scan_to_trace(
        eg, min_steps=2, channel_splits="auto"
    )
    # joint offer + one (2,2) split offer
    assert len(lifts) == 2
    assert all(
        lf.kind == "diag" and lf.T == 4 and lf.d == d for lf in lifts
    )


def test_partitions_explicit_and_invalid():
    d = 4
    plan = TL._Plan("diag", 2, d, maps=[], ins=[], h0=0)
    # explicit valid split
    assert TL._partitions(plan, [(2, 2)]) == [(2, 2)]
    # invalid parts (wrong arity, non-ints, wrong sum) are filtered
    assert TL._partitions(plan, [(1, 1), "x", (2, "a"), (2, 2)]) == [
        (2, 2)
    ]
    # dense plans never split
    assert (
        TL._partitions(
            TL._Plan("dense", 2, d, maps=[], ins=[], h0=0), "auto"
        )
        == []
    )


def test_carrier_plan_success_via_lift():
    """An applyd/affd_compose member in a class yields a carrier-path
    plan — no spine walk needed."""
    d = 3
    eg = EGraph()
    a_p, x_v = _p("pa", 3, d), _v("x", 3, d)
    h0 = _p("h0", d)
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a_p, dim=0, index=t),
            Op.make("select", x_v, dim=0, index=t),
        )
        for t in range(3)
    ]
    tree = leaves[0]
    for lf in leaves[1:]:
        tree = Op.make("affd_compose", lf, tree)
    cid = eg.add_term(Op.make("applyd", tree, h0))
    plan = TL._carrier_plan(eg, eg.find(cid))
    assert plan is not None and plan.T == 3 and plan.kind == "diag"
    lifts = TL.lift_scan_to_trace(eg, min_steps=2, channel_splits=[])
    assert any(lf.T == 3 for lf in lifts)

    # a carrier term whose h0 is the wrong rank declines AFTER the
    # scan plan is built (consistent-shape check)
    bad = eg.add_term(Op.make("applyd", tree, _p("h0b", d, d)))
    assert TL._carrier_plan(eg, eg.find(bad)) is None


def test_lift_root_filter_and_nonmaximal():
    """root_eid restricts the class scan; maximal_only=False keeps
    prefix plans that would otherwise be dropped."""
    d = 3
    eg = EGraph()
    states = []
    term, eid = _p("h0", d), eg.add_term(_p("h0", d))
    A, X = _p("A", 4, d), _p("X", 4, d)
    for t in range(4):
        term = Op.make(
            "add",
            Op.make(
                "mul",
                Op.make("select", A, dim=0, index=t),
                term,
            ),
            Op.make("select", X, dim=0, index=t),
        )
        eid = eg.add_term(term)
        states.append(eid)
    other = eg.add_term(_p("other", d))
    # root_eid=other → the spine classes are skipped
    assert TL.lift_scan_to_trace(eg, root_eid=other) == []
    # maximal_only=False → interior prefixes offer too
    lifts = TL.lift_scan_to_trace(
        eg, min_steps=2, channel_splits=[], maximal_only=False
    )
    assert sorted(lf.T for lf in lifts) == [2, 3, 4]
    # root_eid on the tip gives exactly that plan
    lifts_tip = TL.lift_scan_to_trace(
        eg, root_eid=states[-1], min_steps=2, channel_splits=[]
    )
    assert [lf.T for lf in lifts_tip] == [4]


def test_lift_interior_detected_after_merge():
    """Unioning an interior state class into a new canonical id makes
    the step_states membership check miss it — the h0/map-prefix
    detector still marks it interior (no duplicate F emission)."""
    d = 3
    eg = EGraph()
    states = []
    term, eid = _p("h0", d), eg.add_term(_p("h0", d))
    A, X = _p("A", 4, d), _p("X", 4, d)
    for t in range(4):
        term = Op.make(
            "add",
            Op.make(
                "mul",
                Op.make("select", A, dim=0, index=t),
                term,
            ),
            Op.make("select", X, dim=0, index=t),
        )
        eid = eg.add_term(term)
        states.append(eid)
    # merge the T=2 state class UNDER an unrelated class — union-by-
    # rank makes the dummy's id canonical, so the state's raw eid no
    # longer matches the canonical key the step_states check uses
    dummy = eg.add_enode("foo", (eg.add_term(_p("dm", d)),))
    eg.union(dummy, states[1])
    lifts = TL.lift_scan_to_trace(eg, min_steps=2, channel_splits=[])
    # the interior T=2 plan is still suppressed — only T=4 emits
    assert [lf.T for lf in lifts] == [4]
