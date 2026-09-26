"""Coverage tests for catopt.scan_lower — nested-apply folding, the
aff-tree/domain recognizers, leaf-shape and gather analyses, the
diagonal carrier, the leaf-b gather fast path, and the shared-A (LTI)
expand path.  CPU only; CUDA paths live in test_scan_batched.py."""
# ruff: noqa: RUF059 — test-idiom unpacking

import torch

from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.scan_lower import (
    _fold_nested_apply,
    _is_aff_tree,
    _leaf_b_gather,
    _leaf_shapes_consistent,
    _make_bottom_row,
    _select_index,
    build_scan_plan,
    is_scan_apply_term,
    to_batched_scan_module,
)
from catopt.torch_bridge import ir_to_torch_module


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


# ---------------------------------------------------------------------------
#  _fold_nested_apply
# ---------------------------------------------------------------------------


def test_fold_nested_apply_dense_and_diag():
    f, g, h = _v("f", 1), _v("g", 1), _v("h", 4)
    nested = Op.make("apply", f, Op.make("apply", g, h))
    out = _fold_nested_apply(nested)
    assert out.op == "apply"
    assert out.args[0].op == "aff_compose"
    assert out.args[1] is h
    # deeper chain folds bottom-up into one spine
    k = _v("k", 1)
    deep = Op.make("apply", f, Op.make("apply", g, Op.make("apply", k, h)))
    out2 = _fold_nested_apply(deep)
    assert out2.args[0].op == "aff_compose"
    assert out2.args[1] is h
    # diagonal variant uses affd_compose
    dn = Op.make("applyd", f, Op.make("applyd", g, h))
    outd = _fold_nested_apply(dn)
    assert outd.args[0].op == "affd_compose"
    # non-apply terms pass through unchanged
    leaf = _v("x", 3)
    assert _fold_nested_apply(leaf) is leaf
    plain = Op.make("relu", leaf)
    assert _fold_nested_apply(plain).op == "relu"


def test_is_aff_tree_domains():
    A, b = _p("A", 4, 4), _p("b", 4)
    leaf = Op.make("aff", A, b)
    assert _is_aff_tree(leaf) == "aff"
    tree = Op.make(
        "aff_compose", leaf, Op.make("aff", _p("A2", 4, 4), b)
    )
    assert _is_aff_tree(tree) == "aff"
    a, bb = _p("a", 4), _p("bb", 4)
    dleaf = Op.make("aff_diag", a, bb)
    assert _is_aff_tree(dleaf) == "aff_diag"
    dtree = Op.make(
        "affd_compose", dleaf, Op.make("aff_diag", _p("a2", 4), bb)
    )
    assert _is_aff_tree(dtree) == "aff_diag"
    # mixed domain: aff leaf under affd_compose → None
    assert (
        _is_aff_tree(Op.make("affd_compose", dleaf, leaf)) is None
    )
    # wrong compose op for the domain → None
    assert (
        _is_aff_tree(Op.make("aff_compose", dleaf, dleaf)) is None
    )
    # mixed leaves under one compose → None
    assert (
        _is_aff_tree(Op.make("aff_compose", leaf, dleaf)) is None
    )
    # foreign node / non-Op / wrong arity → None
    assert _is_aff_tree(Op.make("add", A, A)) is None
    assert _is_aff_tree(A) is None
    assert _is_aff_tree(Op.make("aff", A)) is None
    # memoised shared subtree: same tree object re-checked
    memo = {}
    assert _is_aff_tree(tree, memo) == "aff"
    assert _is_aff_tree(tree, memo) == "aff"


def test_is_scan_apply_term_variants():
    A, b, h = _p("A", 4, 4), _p("b", 4), _p("h", 4)
    aff = Op.make("aff", A, b)
    assert is_scan_apply_term(Op.make("apply", aff, h))
    a, bb = _p("a", 4), _p("bb", 4)
    affd = Op.make("aff_diag", a, bb)
    assert is_scan_apply_term(Op.make("applyd", affd, h))
    # carrier mismatch: apply over aff_diag → False
    assert not is_scan_apply_term(Op.make("apply", affd, h))
    assert not is_scan_apply_term(Op.make("applyd", aff, h))
    assert not is_scan_apply_term(aff)
    assert not is_scan_apply_term(Op.make("add", aff, h))
    # nested apply folds into a scan-apply shape
    nested = Op.make("apply", aff, Op.make("apply", aff, h))
    assert is_scan_apply_term(nested)


def test_leaf_shapes_consistent_and_select_index():
    A, b = _p("A", 4, 4), _p("b", 4)
    l1 = Op.make("aff", A, b)
    l2 = Op.make("aff", _p("A2", 4, 4), _p("b2", 4))
    assert _leaf_shapes_consistent([l1, l2])
    assert not _leaf_shapes_consistent([])
    # diag leaf needs a == b shapes
    good_d = Op.make("aff_diag", _p("a", 4), _p("bb", 4))
    bad_d = Op.make("aff_diag", _p("a", 4), _p("bb", 3))
    assert _leaf_shapes_consistent([good_d])
    assert not _leaf_shapes_consistent([bad_d])
    # dense leaf needs b == A[:-1]
    bad_dense = Op.make("aff", A, _p("b", 3))
    assert not _leaf_shapes_consistent([bad_dense])
    # one leaf with a different shape → False
    l3 = Op.make("aff", _p("A3", 5, 5), _p("b3", 5))
    assert not _leaf_shapes_consistent([l1, l3])

    base = _v("x", 8, 4)
    assert _select_index(
        Op.make("select", base, dim=0, index=2)
    ) == (base, 0, 2)
    assert _select_index(
        Op.make("getitem", base, index=1)
    ) == (base, 0, 1)
    assert (
        _select_index(
            Op.make("select", base, dim=0, index="i", validate=False)
        )
        is None
    )
    assert _select_index(base) is None
    assert _select_index(Op.make("relu", base)) is None


def test_leaf_b_gather():
    base = _v("x", 8, 4)
    leaves = [
        Op.make(
            "aff",
            _p(f"A{i}", 4, 4),
            Op.make("select", base, dim=0, index=i),
        )
        for i in range(4)
    ]
    g = _leaf_b_gather(leaves)
    assert g is not None
    assert g[0] is base and g[1] == 0 and g[2] == [0, 1, 2, 3]
    # different bases → None
    leaves[1] = Op.make(
        "aff",
        _p("A1", 4, 4),
        Op.make("select", _v("y", 8, 4), dim=0, index=1),
    )
    assert _leaf_b_gather(leaves) is None
    # non-select b → None
    leaves[0] = Op.make("aff", _p("A0", 4, 4), _p("b", 4))
    assert _leaf_b_gather(leaves) is None


# ---------------------------------------------------------------------------
#  build_scan_plan
# ---------------------------------------------------------------------------


def _compose(opname, leaves):
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        opname, _compose(opname, leaves[:mid]),
        _compose(opname, leaves[mid:]),
    )


def test_build_scan_plan_dense_diag_and_rejections():
    T, d = 4, 3
    A = _p("A", d, d)
    h = _p("h", d)
    x = _v("x", T, d)
    # LTI recurrence: every leaf shares the same A term
    leaves = [
        Op.make("aff", A, Op.make("select", x, dim=0, index=t))
        for t in range(T)
    ]
    root = Op.make("apply", _compose("aff_compose", leaves), h)
    plan = build_scan_plan(root)
    assert plan is not None
    assert plan["leaf_a_shared"] is True
    assert plan["leaf_b_gather"] is not None
    assert plan["diagonal"] is False
    assert plan["root_slot"] >= 0
    # diagonal carrier
    a = _p("a", d)
    dleaves = [
        Op.make("aff_diag", a, Op.make("select", x, dim=0, index=t))
        for t in range(T)
    ]
    droot = Op.make("applyd", _compose("affd_compose", dleaves), h)
    dplan = build_scan_plan(droot)
    assert dplan is not None and dplan["diagonal"] is True
    # non-scan root → None
    assert build_scan_plan(Op.make("relu", h)) is None
    # non-uniform leaf shapes → None
    bad = [
        Op.make("aff", A, _p("b0", d)),
        Op.make("aff", _p("A5", 5, 5), _p("b5", 5)),
    ]
    badroot = Op.make(
        "apply", Op.make("aff_compose", *bad), h
    )
    assert build_scan_plan(badroot) is None


def test_make_bottom_row():
    like = torch.zeros(4, 4)
    row = _make_bottom_row(like)
    assert row.shape == (1, 1, 5)
    assert row[..., -1] == 1.0 and row[..., :-1].abs().sum() == 0
    row2 = _make_bottom_row(torch.zeros(3))
    assert row2.shape == (1, 1, 3) or row2.shape[-1] == 3


# ---------------------------------------------------------------------------
#  BatchedScanModule — dense and diagonal forward
# ---------------------------------------------------------------------------


def _scan_ir_dense(T, d, shared_a=True, b_gather=True):
    A = _p("A", d, d)
    h = _p("h", d)
    x = _v("x", T, d)
    leaves = []
    for t in range(T):
        a_t = A if shared_a else _p(f"A{t}", d, d)
        b_t = (
            Op.make("select", x, dim=0, index=t)
            if b_gather
            else _p(f"b{t}", d)
        )
        leaves.append(Op.make("aff", a_t, b_t))
    root = Op.make("apply", _compose("aff_compose", leaves), h)
    inputs = [x] if b_gather else []
    ir = IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params={},
    )
    return ir, A, h, x, leaves


def _dense_ref_scan(As, bs, h):
    # aff_compose(f, g) = f∘g applies g FIRST — so the leaf list order
    # is reverse-chronological: leaf[0] applies last.  Iterate the
    # leaves right-to-left to reproduce the composed map eagerly.
    hh = h.clone()
    for A, b in zip(As[::-1], bs[::-1], strict=True):
        hh = A @ hh + b
    return hh


def test_batched_dense_scan_matches_serial():
    torch.manual_seed(0)
    T, d = 6, 4
    ir, A, h, x, leaves = _scan_ir_dense(T, d)
    mod = to_batched_scan_module(ir)
    assert mod.is_batched and mod.n_levels > 0
    Av = torch.randn(d, d, dtype=torch.float64) * 0.3
    xv = torch.randn(T, d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"A": Av, "h": hv}
    bat = to_batched_scan_module(ir, param_values=pv)
    gen = ir_to_torch_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = bat(xv)
        o_g = gen(xv)
    ref = _dense_ref_scan(
        [Av] * T, [xv[t] for t in range(T)], hv
    )
    assert (o_b - o_g).abs().max().item() < 1e-12
    assert (o_b - ref).abs().max().item() < 1e-12


def test_batched_diag_scan_matches_serial():
    torch.manual_seed(0)
    T, d = 6, 4
    a, h, x = _p("a", d), _p("h", d), _v("x", T, d)
    leaves = [
        Op.make("aff_diag", a, Op.make("select", x, dim=0, index=t))
        for t in range(T)
    ]
    root = Op.make("applyd", _compose("affd_compose", leaves), h)
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    mod = to_batched_scan_module(ir)
    assert mod.is_batched
    plan = build_scan_plan(mod.eval_mod._root)
    assert plan["diagonal"]
    av = torch.randn(d, dtype=torch.float64).clamp(-0.9, 0.9)
    xv = torch.randn(T, d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv)
    mod = to_batched_scan_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(xv)
        o_g = gen(xv)
    # eager reference: leaves apply right-to-left — the composed map
    # is the recurrence run over reversed inputs
    hh = hv.clone()
    for t in range(T - 1, -1, -1):
        hh = av * hh + xv[t]
    assert (o_b - hh).abs().max().item() < 1e-12
    assert (o_b - o_g).abs().max().item() < 1e-12


def test_batched_leaf_gather_and_nonshared_a():
    """Leaf b's as non-contiguous index_select + movedim, and distinct
    a terms — exercises the per-leaf stack path."""
    torch.manual_seed(0)
    T, d = 4, 3
    x = _v("x", 8, d)
    As = [_p(f"A{t}", d, d) for t in range(T)]
    leaves = [
        Op.make(
            "aff",
            As[t],
            Op.make("select", x, dim=0, index=2 * t),
        )
        for t in range(T)
    ]
    root = Op.make("apply", _compose("aff_compose", leaves), _p("h", d))
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    mod = to_batched_scan_module(ir)
    plan = build_scan_plan(mod.eval_mod._root)
    assert plan is not None
    assert not plan["leaf_a_shared"]
    g = plan["leaf_b_gather"]
    assert g is not None and g[2] == [0, 2, 4, 6]
    Avs = [torch.randn(d, d, dtype=torch.float64) * 0.3 for _ in As]
    xv = torch.randn(8, d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {f"A{t}": Avs[t] for t in range(T)} | {"h": hv}
    gen = ir_to_torch_module(ir, param_values=pv)
    mod = to_batched_scan_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(xv)
        o_g = gen(xv)
    ref = _dense_ref_scan(Avs, [xv[2 * t] for t in range(T)], hv)
    assert (o_b - ref).abs().max().item() < 1e-12
    assert (o_b - o_g).abs().max().item() < 1e-12


def test_batched_nonscan_fallback_and_properties():
    torch.manual_seed(0)
    x = _v("x", 4)
    ir = IR(
        root=Op.make("relu", x), inputs=[x],
        input_names={"x"}, params={},
    )
    mod = to_batched_scan_module(ir)
    assert not mod.is_batched and mod.n_levels == 0
    assert not mod.is_graph_captured
    mod.drop_cuda_graph()
    # CPU: capture_cuda_graph is a no-op
    assert mod.capture_cuda_graph(torch.randn(4)) is mod
    with torch.no_grad():
        out = mod(torch.full((4,), -1.0))
    assert torch.equal(out, torch.zeros(4))


def test_batched_same_a_eval_expand():
    """Distinct term objects that evaluate to the SAME tensor hit the
    runtime ``all(a is leaf_As[0])`` expand path: two Var leaves with
    the same name resolve to one env tensor."""
    torch.manual_seed(0)
    T, d = 3, 2
    h = _p("h", d)
    xs = [_v(f"x{t}", d) for t in range(T)]
    # two distinct Var objects, SAME name "a" → leaf_a_shared False
    # (Var identity differs) but ev() returns the same tensor
    a1, a2 = _v("a", d, d), _v("a", d, d)
    leaves = [
        Op.make("aff", a1, xs[0]),
        Op.make("aff", a2, xs[1]),
        Op.make("aff", a1, xs[2]),
    ]
    root = Op.make("apply", _compose("aff_compose", leaves), h)
    ir = IR(
        root=root,
        inputs=[a1, *xs],
        input_names={"a", *{v.name for v in xs}},
        params={},
    )
    mod = to_batched_scan_module(ir)
    plan = build_scan_plan(mod.eval_mod._root)
    assert plan is not None and not plan["leaf_a_shared"]
    av = torch.randn(d, d, dtype=torch.float64) * 0.2
    xvs = [torch.randn(d, dtype=torch.float64) for _ in xs]
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"h": hv}
    gen = ir_to_torch_module(ir, param_values=pv)
    mod = to_batched_scan_module(ir, param_values=pv)
    with torch.no_grad():
        o_b = mod(av, *xvs)
        o_g = gen(av, *xvs)
    ref = _dense_ref_scan([av] * T, xvs, hv)
    assert (o_b - o_g).abs().max().item() < 1e-12
    assert (o_b - ref).abs().max().item() < 1e-12
