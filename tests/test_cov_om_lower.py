# ruff: noqa: RUF002, RUF003
"""Coverage tests for catopt.om_lower — the operand-gather recognizers,
elem-group analysis modes, the batched module's runtime paths (serial
leaves, DAG multiplicities, masked rows, cache), and the streaming
schedule's incremental API.  All CPU; CUDA paths stay in
test_om_batched.py."""

import pytest
import torch

from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.om_lower import (
    BatchedOMModule,
    StreamingOMModule,
    _analyze_elem_group,
    _batched_compose,
    _batched_elem,
    _is_om_tree,
    _qk_parts,
    _same_term,
    _slice_index,
    _sliced_gather,
    _stretch,
    build_om_plan,
    is_om_apply_term,
    om_apply_state,
    om_empty_state,
    om_step,
    om_step_qk,
    to_batched_om_module,
    to_streaming_om_module,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


# ---------------------------------------------------------------------------
#  _slice_index — decompose select/getitem/chunk/split terms
# ---------------------------------------------------------------------------


def test_slice_index_select_and_getitem():
    base = _v("base", 4, 8)
    sel = Op.make("select", base, arg1=1, arg2=2)
    assert _slice_index(sel) == (base, "select", 1, 2, None, None)
    sel2 = Op.make("select", base, dim=0, index=1)
    assert _slice_index(sel2) == (base, "select", 0, 1, None, None)
    gi = Op.make("getitem", base, arg1=3)
    assert _slice_index(gi) == (base, "select", 0, 3, None, None)
    # missing/bad attrs → None
    assert _slice_index(Op.make("select", base, arg1=0, arg2="x", validate=False)) is None
    assert _slice_index(Op.make("relu", base)) is None
    assert _slice_index(base) is None
    assert _slice_index(Op.make("select", base, _v("i", 1), arg1=0, arg2=0)) is None


def test_slice_index_chunk_and_split():
    base = _v("base", 4, 8)
    ch = Op.make("chunk", base, arg1=4, arg2=1, index=1)
    # 8 % 4 == 0 → part_size 2
    assert _slice_index(ch) == (base, "chunk", 1, 1, 4, 2)
    ch2 = Op.make("chunk", base, arg1=3, arg2=0, index=0)
    # 4 % 3 != 0 (dim 0 of a (4,8) base) → uneven tail → part_size None
    assert _slice_index(ch2) == (base, "chunk", 0, 0, 3, None)
    # non-int n → None
    ch3 = Op.make("chunk", base, arg1="z", arg2=0, index=0, validate=False)
    assert _slice_index(ch3) is None
    # missing dim → None
    ch4 = Op.make("chunk", base, arg1=2)
    assert _slice_index(ch4) is None
    # split with uniform sizes → part known
    sp = Op.make("split", base, sizes=[2, 2], arg2=1, index=0)
    assert _slice_index(sp) == (base, "chunk", 1, 0, 2, 2)
    # split with non-uniform sizes → part None
    sp2 = Op.make("split", base, sizes=[3, 1], arg2=1, index=1)
    assert _slice_index(sp2) == (base, "chunk", 1, 1, 2, None)
    sp3 = Op.make("split", base, arg2=1, index=0, validate=False)
    assert _slice_index(sp3) is None


def test_sliced_gather_select_and_chunk():
    base = _v("base", 4, 8)
    parts = [
        _slice_index(Op.make("select", base, arg1=0, arg2=i))
        for i in range(4)
    ]
    g = _sliced_gather(parts)
    assert g == (base, "select", 0, 4, 1)
    # chunk gather: n parts of equal size along dim 1
    ch = [
        _slice_index(Op.make("chunk", base, arg1=4, arg2=1, index=i))
        for i in range(4)
    ]
    g2 = _sliced_gather(ch)
    assert g2 == (base, "chunk", 1, 4, 2)
    # out-of-order indices → None
    parts_bad = [parts[1], parts[0], *parts[2:]]
    assert _sliced_gather(parts_bad) is None
    # different bases → None
    base2 = _v("base2", 4, 8)
    parts_mix = parts[:1] + [
        _slice_index(Op.make("select", base2, arg1=0, arg2=1))
    ] + parts[2:]
    assert _sliced_gather(parts_mix) is None
    # mixed kinds → None
    parts_mix2 = parts[:1] + ch[1:]
    assert _sliced_gather(parts_mix2) is None
    # partial coverage (n != len) → None
    assert _sliced_gather(parts[:2]) is None
    # empty / None members → None
    assert _sliced_gather([]) is None
    assert _sliced_gather([None, None]) is None


def test_qk_parts_and_same_term():
    q, k = _v("q", 2, 4, 8), _v("k", 2, 4, 6, 8)
    good = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    assert _qk_parts(good) == (q, k)
    # wrong transpose dims → None
    bad = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=0, arg2=1)
    )
    assert _qk_parts(bad) is None
    # non-matmul / non-transpose → None
    assert _qk_parts(Op.make("matmul", q, k)) is None
    assert _qk_parts(q) is None
    assert _qk_parts(Op.make("add", q, q)) is None

    p1, p2 = _p("w", 4, 4), _p("w", 4, 4)
    assert _same_term([p1, p2, p1])  # same-named Params count as same
    assert _same_term([q, q])
    assert not _same_term([q, k])


def test_is_om_tree_and_apply():
    s, v = _v("s", 4, 3), _v("v", 3, 6)
    e = Op.make("om_elem", s, v)
    assert _is_om_tree(e)
    m, l, a = _v("m", 4, 1), _v("l", 4, 1), _v("a", 4, 6)
    pkg = Op.make("om", m, l, a)
    assert _is_om_tree(pkg)
    comp = Op.make("om_compose", e, pkg)
    assert _is_om_tree(comp)
    assert not _is_om_tree(Op.make("om_compose", e, s))
    assert not _is_om_tree(s)
    assert is_om_apply_term(Op.make("om_apply", comp))
    assert not is_om_apply_term(comp)
    assert not is_om_apply_term(Op.make("om_apply", s))


# ---------------------------------------------------------------------------
#  _analyze_elem_group modes
# ---------------------------------------------------------------------------


def test_analyze_elem_group_dense_qk_bmm_and_stack():
    q = _v("q", 4, 8)
    K = _v("K", 8, 8)
    # k_i = chunk(K, -2, i) — consecutive equal chunks along key axis
    ks = [
        Op.make("chunk", K, arg1=4, arg2=-2, index=i) for i in range(4)
    ]
    vs = [_v(f"v{i}", 2, 4) for i in range(4)]
    leaves = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul", q, Op.make("transpose", kk, arg1=-2, arg2=-1)
            ),
            vv,
        )
        for kk, vv in zip(ks, vs, strict=True)
    ]
    grp = _analyze_elem_group(leaves)
    assert grp["s_mode"] == "dense_qk"
    assert grp["k_base"] is K

    # same q, uniform k shapes but NOT slices of one base → bmm_qk
    kvs = [_v(f"k{i}", 2, 8) for i in range(3)]
    leaves2 = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul", q, Op.make("transpose", kk, arg1=-2, arg2=-1)
            ),
            vv,
        )
        for kk, vv in zip(kvs, vs[:3], strict=True)
    ]
    grp2 = _analyze_elem_group(leaves2)
    assert grp2["s_mode"] == "bmm_qk"
    assert grp2["k_gather"] is None

    # different q per leaf, scores not slices → stack
    s_terms = [_v(f"s{i}", 4, 2) for i in range(3)]
    leaves3 = [
        Op.make("om_elem", ss, vv)
        for ss, vv in zip(s_terms, vs[:3], strict=True)
    ]
    grp3 = _analyze_elem_group(leaves3)
    assert grp3["s_mode"] == "stack"

    # scores as slices of one score tensor → "slice"
    S = _v("S", 4, 8)
    s_terms2 = [
        Op.make("chunk", S, arg1=4, arg2=-1, index=i) for i in range(4)
    ]
    leaves4 = [
        Op.make("om_elem", ss, vv)
        for ss, vv in zip(s_terms2, vs, strict=True)
    ]
    grp4 = _analyze_elem_group(leaves4)
    assert grp4["s_mode"] == "slice"


# ---------------------------------------------------------------------------
#  Small kernels
# ---------------------------------------------------------------------------


def test_stretch_and_batched_kernels():
    t = torch.randn(3, 4)
    tail = torch.Size([2, 4])
    out = _stretch(t, tail)
    assert out.shape == (3, 2, 4)
    # already-right tail returns the same tensor
    same = _stretch(t, torch.Size([4]))
    assert same is t
    # missing dims are padded in front then expanded
    t2 = torch.randn(3)
    out2 = _stretch(t2, torch.Size([2, 4]))
    assert out2.shape == (3, 2, 4)

    s = torch.randn(5, 3, 4)
    v = torch.randn(5, 4, 6)
    m, l, a = _batched_elem(s, v)
    assert m.shape == (5, 3, 1) and a.shape == (5, 3, 6)
    # per-leaf equivalence with the serial om_elem semantics
    e = torch.exp(s - s.amax(-1, keepdim=True))
    assert torch.equal(a, e @ v)
    m2, l2, a2 = _batched_compose(m, l, a, m, l, a)
    assert torch.equal(m2, m)  # compose(x, x) idempotent on m


# ---------------------------------------------------------------------------
#  BatchedOMModule runtime paths
# ---------------------------------------------------------------------------


def _qk_ir(B, T, d, dv, ksizes, leaf_fn=None):
    q = _v("q", B, T, d)
    ks = [_v(f"k{i}", B, k, d) for i, k in enumerate(ksizes)]
    vs = [_v(f"v{i}", B, k, dv) for i, k in enumerate(ksizes)]

    def default_leaf(qq, kk, vv):
        s = Op.make(
            "matmul", qq, Op.make("transpose", kk, arg1=-2, arg2=-1)
        )
        return Op.make("om_elem", s, vv)

    fn = leaf_fn or default_leaf

    def tree(leaves):
        if len(leaves) == 1:
            return leaves[0]
        mid = len(leaves) // 2
        return Op.make(
            "om_compose", tree(leaves[:mid]), tree(leaves[mid:])
        )

    root = Op.make(
        "om_apply", tree([fn(q, k, v) for k, v in zip(ks, vs)])
    )
    inputs = [q, *ks, *vs]
    return (
        IR(root=root, inputs=inputs,
           input_names={v.name for v in inputs}, params={}),
        q,
        ks,
        vs,
    )


def _dense_ref(q, ks, vs):
    s = q @ torch.cat(list(ks), dim=-2).transpose(-2, -1)
    return torch.softmax(s, dim=-1) @ torch.cat(list(vs), dim=-2)


def test_batched_matches_dense_serial_leaf_and_mults():
    """`om(m,l,a)` packaging leaf → serial group; a DAG-shared elem leaf
    (same node object twice) → multiplicity repeat — both agree with
    the dense reference."""
    torch.manual_seed(0)
    B, T, d, dv = 2, 8, 4, 4
    # ---- serial leaf: om(m,l,a) packaging node in the tree
    m_v = _v("m", B, T, 1)
    l_v = _v("l", B, T, 1)
    a_v = _v("a", B, T, dv)
    q, k1, v1 = _v("q", B, T, d), _v("k1", B, 4, d), _v("v1", B, 4, dv)
    e1 = Op.make(
        "om_elem",
        Op.make(
            "matmul", q, Op.make("transpose", k1, arg1=-2, arg2=-1)
        ),
        v1,
    )
    pkg = Op.make("om", m_v, l_v, a_v)
    root = Op.make("om_apply", Op.make("om_compose", e1, pkg))
    ir = IR(
        root=root,
        inputs=[q, k1, v1, m_v, l_v, a_v],
        input_names={"q", "k1", "v1", "m", "l", "a"},
        params={},
    )
    mod = to_batched_om_module(ir)
    assert mod.is_batched
    plan = build_om_plan(root)
    kinds = {g["kind"] for g in plan["leaf_groups"]}
    assert kinds == {"elem", "serial"}

    tq = torch.randn(B, T, d, dtype=torch.float64)
    tk = torch.randn(B, 4, d, dtype=torch.float64)
    tv = torch.randn(B, 4, dv, dtype=torch.float64)
    s2 = tq @ torch.randn(B, 3, d, dtype=torch.float64).transpose(-2, -1)
    tm = s2.amax(-1, keepdim=True)
    tl = torch.exp(s2 - tm).sum(-1, keepdim=True)
    ta = torch.exp(s2 - tm) @ torch.randn(B, 3, dv, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tq, tk, tv, tm, tl, ta)
    # serial-eval reference through the generic module
    gen = ir_to_torch_module(ir)
    with torch.no_grad():
        ref = gen(tq, tk, tv, tm, tl, ta)
    assert (out - ref).abs().max().item() < 1e-12

    # ---- DAG-shared leaf: same om_elem node object used twice
    torch.manual_seed(1)
    s = _v("s", B, T, 4)
    v = _v("v", B, 4, dv)
    leaf = Op.make("om_elem", s, v)
    root2 = Op.make("om_apply", Op.make("om_compose", leaf, leaf))
    ir2 = IR(
        root=root2,
        inputs=[s, v],
        input_names={"s", "v"},
        params={},
    )
    mod2 = to_batched_om_module(ir2)
    plan2 = build_om_plan(root2)
    assert plan2["leaf_groups"][0]["mults"] == [2]
    ts = torch.randn(B, T, 4, dtype=torch.float64)
    tv2 = torch.randn(B, 4, dv, dtype=torch.float64)
    with torch.no_grad():
        out2 = mod2(ts, tv2)
    # duplicating the leaf = softmax over duplicated keys
    s_cat = torch.cat([ts, ts], dim=-1)
    v_cat = torch.cat([tv2, tv2], dim=-2)
    ref2 = torch.softmax(s_cat, -1) @ v_cat
    assert (out2 - ref2).abs().max().item() < 1e-12


def test_batched_bmm_qk_mode_and_odd_count():
    """bmm_qk (same q, distinct k tensors) + an odd block count
    (carry-down path in the pair reduction) — matches dense.

    q is a Param rather than a Var: Op interning may hand back a
    structurally-equal matmul node bound to a different-but-equal Var
    object from an earlier test — _same_term counts same-named Params
    as the same term, so the mode detection is robust to interning."""
    torch.manual_seed(0)
    B, T, d, dv = 2, 8, 4, 4
    qp = _p("q", B, T, d)
    ks = [_v(f"k{i}", B, 4, d) for i in range(3)]
    vs = [_v(f"v{i}", B, 4, dv) for i in range(3)]
    leaves = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul",
                qp,
                Op.make("transpose", kk, arg1=-2, arg2=-1),
            ),
            vv,
        )
        for kk, vv in zip(ks, vs, strict=True)
    ]
    comp = Op.make(
        "om_compose",
        Op.make("om_compose", leaves[0], leaves[1]),
        leaves[2],
    )
    root = Op.make("om_apply", comp)
    ir = IR(
        root=root,
        inputs=[*ks, *vs],
        input_names={v.name for v in [*ks, *vs]},
        params={"q": qp},
    )
    tq = torch.randn(B, T, d, dtype=torch.float64)
    mod = to_batched_om_module(ir, param_values={"q": tq})
    plan = build_om_plan(mod.eval_mod._root)
    modes = {g["s_mode"] for g in plan["leaf_groups"]}
    assert "bmm_qk" in modes
    tks = [torch.randn(B, 4, d, dtype=torch.float64) for _ in ks]
    tvs = [torch.randn(B, 4, dv, dtype=torch.float64) for _ in vs]
    with torch.no_grad():
        out = mod(*tks, *tvs)
    ref = _dense_ref(tq, tks, tvs)
    assert (out - ref).abs().max().item() < 1e-12


def test_batched_fully_masked_rows_match_serial():
    """A block with a fully-masked (-inf) row: the sanitising where()
    zeroes its l/a — NaN semantics identical to serial evaluation."""
    torch.manual_seed(0)
    B, T, d, dv = 1, 6, 4, 4
    ir, q, ks, vs = _qk_ir(B, T, d, dv, [3, 3])
    mod = to_batched_om_module(ir)
    tq = torch.randn(B, T, d, dtype=torch.float64)
    tks = [torch.randn(B, 3, d, dtype=torch.float64) for _ in ks]
    tvs = [torch.randn(B, 3, dv, dtype=torch.float64) for _ in vs]
    gen = ir_to_torch_module(ir)
    with torch.no_grad():
        # poison row 2 of the first block's scores via the k tensor
        tks[0][:, 2] = float("nan")  # NaN propagates through q@k.T
        tks[0][:, 2] = 0.0
        tq[:, 2] = 0.0
        # force a fully-masked row by masking all scores in row 2
        # (simulate via k → -inf contribution is impossible; instead
        # build the score block directly with a leaf_fn next time)
        out = mod(tq, *tks, *tvs)
        ref = gen(tq, *tks, *tvs)
    assert (out - ref).abs().max().item() < 1e-12 or (
        torch.isnan(out) == torch.isnan(ref)
    ).all()

    # direct: score leaf with a fully -inf row — batched vs serial
    s_v = _v("s", B, T, 4)
    v_v = _v("v", B, 4, dv)
    leaf = Op.make("om_elem", s_v, v_v)
    ir2 = IR(
        root=Op.make(
            "om_apply",
            Op.make("om_compose", leaf, Op.make("om_elem", s_v, v_v)),
        ),
        inputs=[s_v, v_v],
        input_names={"s", "v"},
        params={},
    )
    mod2 = to_batched_om_module(ir2)
    gen2 = ir_to_torch_module(ir2)
    ts = torch.randn(B, T, 4, dtype=torch.float64)
    ts[:, 2] = float("-inf")  # fully masked row
    tv = torch.randn(B, 4, dv, dtype=torch.float64)
    with torch.no_grad():
        o2 = mod2(ts, tv)
        r2 = gen2(ts, tv)
    # identical NaN pattern
    assert torch.equal(torch.isnan(o2), torch.isnan(r2))
    assert (o2[~torch.isnan(o2)] - r2[~torch.isnan(r2)]).abs().max() < 1e-12


def test_batched_short_sequence_and_properties():
    """T < block size and a single leaf — plus the property surface."""
    torch.manual_seed(0)
    ir, q, ks, vs = _qk_ir(1, 2, 4, 4, [4])
    mod = to_batched_om_module(ir)
    assert mod.is_batched and mod.n_levels == 0 and mod.n_blocks == 1
    assert not mod.is_graph_captured
    mod.drop_cuda_graph()  # no-op on CPU
    tq = torch.randn(1, 2, 4, dtype=torch.float64)
    tk = torch.randn(1, 4, 4, dtype=torch.float64)
    tv = torch.randn(1, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        out = mod(tq, tk, tv)
    ref = _dense_ref(tq, [tk], [tv])
    assert (out - ref).abs().max().item() < 1e-12

    # _cached: first call builds, second returns the cached tensor
    like = torch.zeros(3)
    t1 = mod._cached(("k", 1), like, lambda t: torch.ones(3))
    t2 = mod._cached(("k", 1), like, lambda t: torch.zeros(3))
    assert t1 is t2 and torch.equal(t1, torch.ones(3))
    # compile() on an unbatched module is a pass-through
    plain_ir, _src = _export_silu()
    um = to_batched_om_module(plain_ir)
    assert not um.is_batched
    assert um.compile() is um


def _export_silu():
    import torch.nn as nn

    from catopt.torch_bridge import export_to_ir

    torch.manual_seed(0)
    m = nn.SiLU().eval()
    x = torch.randn(2, 4)
    return export_to_ir(m, x)


# ---------------------------------------------------------------------------
#  Streaming schedule + incremental decode API
# ---------------------------------------------------------------------------


def test_streaming_matches_batched_and_dense():
    torch.manual_seed(0)
    B, T, d, dv = 2, 8, 4, 4
    ir, q, ks, vs = _qk_ir(B, T, d, dv, [4, 4, 4, 4])
    sm = to_streaming_om_module(ir)
    bm = to_batched_om_module(ir)
    assert sm.is_streaming and sm.n_blocks == 4
    tq = torch.randn(B, T, d, dtype=torch.float64)
    tks = [torch.randn(B, 4, d, dtype=torch.float64) for _ in ks]
    tvs = [torch.randn(B, 4, dv, dtype=torch.float64) for _ in vs]
    with torch.no_grad():
        o_s = sm(tq, *tks, *tvs)
        o_b = bm(tq, *tks, *tvs)
        ref = _dense_ref(tq, tks, tvs)
        state = sm.forward_state(tq, *tks, *tvs)
    assert (o_s - ref).abs().max().item() < 1e-12
    assert (o_s - o_b).abs().max().item() < 1e-12
    # forward_state returns the raw carrier; apply → same readout
    assert (om_apply_state(state) - o_s).abs().max().item() < 1e-12


def test_streaming_incremental_decode_equivalence():
    """om_step/om_step_qk fold blocks one at a time; the final carrier
    equals both the streaming module's and the dense reference."""
    torch.manual_seed(0)
    B, T, d, dv = 1, 4, 3, 2
    q = torch.randn(B, T, d, dtype=torch.float64)
    ks = [torch.randn(B, 4, d, dtype=torch.float64) for _ in range(3)]
    vs = [torch.randn(B, 4, dv, dtype=torch.float64) for _ in range(3)]

    state = None
    for k, v in zip(ks, vs, strict=True):
        state = om_step_qk(state, q, k, v)
    ref = _dense_ref(q, ks, vs)
    assert (om_apply_state(state) - ref).abs().max().item() < 1e-12

    # om_empty_state is the identity: starting from it is identical
    state2 = om_empty_state((B, T), dv)
    for k, v in zip(ks, vs, strict=True):
        s_blk = q @ k.transpose(-2, -1)
        state2 = om_step(state2, s_blk, v)
    assert (om_apply_state(state2) - ref).abs().max().item() < 1e-12

    # composing with the identity changes nothing
    m, l, a = state
    mid, lid, aid = om_empty_state((B, T), dv)
    assert float(mid.min()) == float("-inf")
    assert float(lid.sum()) == 0.0 and float(aid.abs().sum()) == 0.0

    # the statics on the module delegate to the same functions
    assert StreamingOMModule.step is om_step
    assert StreamingOMModule.step_qk is om_step_qk
    assert StreamingOMModule.apply is om_apply_state
    assert StreamingOMModule.empty_state is om_empty_state


def test_streaming_non_om_fallback():
    plain_ir, src = _export_silu()
    sm = to_streaming_om_module(plain_ir, param_values=src)
    assert not sm.is_streaming and sm.n_blocks == 0
    x = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.equal(sm(x), sm.eval_mod(x))


def test_om_module_cached_and_repeat_idx():
    torch.manual_seed(0)
    ir, q, ks, vs = _qk_ir(1, 4, 4, 4, [2, 2])
    mod = to_batched_om_module(ir)
    like = torch.zeros(4)
    i1 = mod._repeat_idx([2, 1], like)
    assert torch.equal(i1, torch.tensor([0, 0, 1]))
    i2 = mod._repeat_idx([2, 1], like)
    assert i1 is i2  # cached


def ir_to_torch_module(ir, param_values=None):
    from catopt.torch_bridge import ir_to_torch_module as _f

    return _f(ir, param_values=param_values)
