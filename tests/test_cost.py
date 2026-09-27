"""Tests for cost models."""

import types

import pytest
from catopt.cost import (
    _LAUNCH_S,
    LOWERINGS,
    CostModel,
    _local_roofline,
    count_cost,
    depth_cost_for,
    executor_cost_for,
    executor_overhead,
    flops_cost,
    fused_cost_for,
    lowering_aware_cost_for,
    param_bytes_cost,
    param_bytes_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt.ir import Const, Op, Param, TensorType, Var


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


def test_count_cost_leaves():
    x = Var("x", TensorType((1, 4)))
    assert count_cost(x) == 0.0


def test_count_cost_simple():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("neg", x)
    assert count_cost(op) == 1.0  # one non-view op


def test_count_cost_nested():
    x = Var("x", TensorType((1, 4)))
    op = Op.make("add", Op.make("mul", x, Const(1)), Const(0))
    assert count_cost(op) == 2.0  # add + mul (Consts are leaves)


def test_count_cost_view_ops():
    x = Var("x", TensorType((1, 4)))
    # transpose is a view op, should be free
    op = Op.make("transpose", x)
    assert count_cost(op) == 0.0


def test_broadcast_shape_inference():
    """Elementwise ops must broadcast, not take shapes[0].

    Regression test: a missing broadcast here previously made the cost
    model credit mul((B,T,1),(B,T,C)) with only B*T elements, which
    fabricated a 1.98x 'optimization' out of two identical forms.
    """
    from catopt.cost import _infer_op_shape

    a = Var("a", TensorType((4, 8, 1)))
    b = Var("b", TensorType((4, 8, 32)))

    assert _infer_op_shape(Op.make("mul", a, b)) == (4, 8, 32)
    assert _infer_op_shape(Op.make("add", a, b)) == (4, 8, 32)
    assert _infer_op_shape(Op.make("mul", b, a)) == (4, 8, 32)

    # The two broadcast orderings must cost the same — they are the same work.
    c1 = flops_cost(Op.make("mul", a, b))
    c2 = flops_cost(Op.make("mul", b, a))
    assert c1 == pytest.approx(c2)


def test_flops_cost_scalar():
    c = Const(2.0)
    assert flops_cost(c) == 0.0


def test_flops_cost_matmul():
    x = Var("x", TensorType((128, 64)))
    W = Param("W", TensorType((64, 32)))
    op = Op.make("matmul", x, W)
    # matmul: 2 * 128 * 64 * 32 = 524288 FLOPs
    assert flops_cost(op) == pytest.approx(524288)


def test_flops_cost_elementwise():
    x = Var("x", TensorType((128, 64)))
    op = Op.make("neg", x)
    # neg: 1 * 128 * 64 = 8192 FLOPs
    assert flops_cost(op) == pytest.approx(8192)


def test_cost_model_class():
    x = Var("x", TensorType((4, 4)))
    cm = CostModel()
    op = Op.make("add", x, x)
    # add of two (4,4) tensors: 1 * 4 * 4 = 16
    assert cm(op) == pytest.approx(16)


def test_cost_model_matmul_heavier():
    x = Var("x", TensorType((128, 64)))
    W = Param("W", TensorType((64, 32)))
    cm = CostModel()
    op = Op.make("matmul", x, W)
    # matmul: 2 * 128 * 64 * 32 = 524288
    assert cm(op) == pytest.approx(524288)


def test_concat_chunk_shapes():
    """concat joins along dim; chunk splits it — shape inference."""
    from catopt.cost import _infer_op_shape

    a = Param("A", TensorType((8, 4)))
    b = Param("B", TensorType((8, 4)))
    cat = Op.make("concat", a, b, dim=0)
    assert _infer_op_shape(cat) == (16, 4)

    x = Var("x", TensorType((32, 16)))
    y = Op.make("chunk", x, chunks=2, dim=-1, index=0)
    assert _infer_op_shape(y) == (32, 8)

    # data-movement ops cost nothing
    assert flops_cost(cat) == 0.0
    assert flops_cost(y) == 0.0


def test_param_bytes_counts_values():
    """param_bytes_cost = number of stored values under Param leaves."""
    x = Var("x", TensorType((4, 64)))
    W = Param("W", TensorType((64, 64)))
    assert param_bytes_cost(Op.make("linear", x, W)) == 64 * 64
    # vars, consts, and bare leaves other than Param store nothing
    assert param_bytes_cost(x) == 0.0
    assert param_bytes_cost(Const(2.0)) == 0.0


def test_param_bytes_dedups_shared_names():
    """A weight read by two consumers is stored once — dedup by name."""
    x = Var("x", TensorType((4, 64)))
    y = Var("y", TensorType((4, 64)))
    W = Param("W", TensorType((64, 64)))
    t = Op.make("add", Op.make("linear", x, W), Op.make("linear", y, W))
    assert param_bytes_cost(t) == 64 * 64
    # Two distinct Param objects spelled the same are still one weight.
    W2 = Param("W", TensorType((64, 64)))
    t2 = Op.make(
        "add", Op.make("linear", x, W), Op.make("linear", y, W2)
    )
    assert param_bytes_cost(t2) == 64 * 64


def test_param_bytes_source_tensors():
    """source_tensors is authoritative for numel; TensorType is fallback."""
    import torch

    x = Var("x", TensorType((4, 8)))
    W = Param(
        "W", TensorType((None, None))
    )  # shape unknown at type level
    t = Op.make("linear", x, W)
    src = {"W": torch.zeros(8, 16)}
    assert param_bytes_cost(t, src) == 8 * 16
    # names absent from source_tensors fall back to the TensorType
    U = Param("U", TensorType((3, 5)))
    t2 = Op.make(
        "add", Op.make("linear", x, W), Op.make("linear", x, U)
    )
    assert param_bytes_cost(t2, src) == 8 * 16 + 3 * 5
    # the bound-closure form prices identically
    assert param_bytes_cost_for(src)(t2) == 8 * 16 + 3 * 5


def test_param_bytes_factorised_cheaper():
    """Chained narrow factors store fewer values than the dense weight."""
    x = Var("x", TensorType((4, 64)))
    W = Param("W", TensorType((64, 64)))
    V = Param("V", TensorType((8, 64)))
    U = Param("U", TensorType((64, 8)))
    dense = Op.make("linear", x, W)
    chained = Op.make("linear", Op.make("linear", x, V), U)
    assert param_bytes_cost(chained) == 8 * 64 + 64 * 8
    assert param_bytes_cost(chained) < param_bytes_cost(dense)


def test_rank1_matmul_shapes():
    """matvec/vec-mat/dot infer real shapes, not the matrix's shape.

    Regression test: matmul(B, b1) with B (o,h), b1 (h,) previously
    inferred (o,h) — the MATRIX's shape — so the fused bias in
    assoc_linear_bias's RHS broadcast _INVALID against the (…,o)
    linear output and priced at _INVALID_COST.
    """
    from catopt.cost import _infer_op_shape

    A = Param("A", TensorType((8, 4)))
    M = Param("M", TensorType((4, 8)))
    v = Param("v", TensorType((4,)))
    _w = Param("w", TensorType((8,)))
    assert _infer_op_shape(Op.make("matmul", A, v)) == (8,)  # matvec
    assert _infer_op_shape(Op.make("matmul", v, M)) == (8,)  # vec-mat
    assert _infer_op_shape(Op.make("matmul", v, v)) == ()  # dot
    # batched matrix-vector keeps the batch dims
    B = Param("B", TensorType((2, 8, 4)))
    assert _infer_op_shape(Op.make("matmul", B, v)) == (2, 8)


def test_linear_bias_broadcast_shapes():
    """linear(x, W, b) broadcasts the bias slot: (o,) and the
    column-vector disguise (o,1) are rank-1 biases; a provably
    wrong bias is ill-typed (_INVALID), not silently ignored."""
    from catopt.cost import _INVALID, _infer_op_shape

    x = Var("x", TensorType((4, 16)))
    W = Param("W", TensorType((8, 16)))
    b = Param("b", TensorType((8,)))
    bc = Param("bc", TensorType((8, 1)))
    assert _infer_op_shape(Op.make("linear", x, W, b)) == (4, 8)
    assert _infer_op_shape(Op.make("linear", x, W, bc)) == (4, 8)
    bad = Param("bad", TensorType((7,)))
    assert _infer_op_shape(Op.make("linear", x, W, bad)) is _INVALID


def test_assoc_linear_bias_rhs_finite_cost():
    """The composed biased-linear rewrite prices finite.

    linear(linear(x,A,b1),B,b2) == linear(x, B@A, B·b1) + b2 — the
    RHS must infer a VALID shape (…,o) and cost real FLOPs under
    every model.  Before the matvec case existed, matmul(B, b1)
    inferred (o,h) and the outer add went _INVALID, so this member
    could never win extraction on cost — the rule existed but its
    product was unselectable except via the bias-slot dodge.
    """
    from catopt.cost import (
        _INVALID_COST,
        _infer_op_shape,
        dag_cost,
        depth_cost,
        roofline_cost,
    )

    i, h, o = 32, 128, 32
    x = Var("x", TensorType((4, i)))
    A = Param("A", TensorType((h, i)))
    B = Param("B", TensorType((o, h)))
    b1 = Param("b1", TensorType((h,)))
    b2 = Param("b2", TensorType((o,)))
    rhs = Op.make(
        "add",
        Op.make(
            "linear",
            x,
            Op.make("matmul", B, A),
            Op.make("matmul", B, b1),
        ),
        b2,
    )
    assert _infer_op_shape(rhs) == (4, o)
    for cost in (flops_cost, depth_cost, roofline_cost):
        assert 0 < dag_cost(rhs, cost) < _INVALID_COST

    # … and the natural spelling add(linear(x, BA), B·b1) — which the
    # workaround bias-slot spelling existed to avoid — is valid too.
    alt = Op.make(
        "add",
        Op.make("linear", x, Op.make("matmul", B, A)),
        Op.make("matmul", B, b1),
    )
    assert _infer_op_shape(alt) == (4, o)
    assert 0 < flops_cost(alt) < _INVALID_COST


def test_assoc_linear_bias_rule_member_extracts_finite():
    """End-to-end through the e-graph: the assoc_linear_bias rewrite
    fires and its RHS member sits in the class with a finite cost —
    extraction must never see _INVALID_COST on the real shape."""
    from catopt.cost import _INVALID_COST, _shape_of
    from catopt.egraph import EGraph
    from catopt_core.laws import ASSOC_LINEAR_BIAS

    i, h, o = 8, 16, 8
    x = Var("x", TensorType((4, i)))
    A = Param("A", TensorType((h, i)))
    B = Param("B", TensorType((o, h)))
    b1 = Param("b1", TensorType((h,)))
    b2 = Param("b2", TensorType((o,)))
    src = Op.make("linear", Op.make("linear", x, A, b1), B, b2)

    eg = EGraph()
    eid = eg.add_term(src)
    eg.run([ASSOC_LINEAR_BIAS], eid, max_iterations=4, max_nodes=2000)
    assert eg.rule_fires.get("assoc_linear_bias", 0) >= 1
    best = eg.extract_best(eid, flops_cost)
    assert _shape_of(best) == (4, o)
    assert 0 < flops_cost(best) < _INVALID_COST


def test_cost_preference_for_fewer_ops():
    """Cost model should prefer matmul chains with fewer total FLOPs.

    Uses funnel dimensions (128->64->32->8) where right-assoc (fused weight)
    is dramatically cheaper than left-assoc.
    """
    x = Var("x", TensorType((256, 128)))  # batch=256, d0=128
    A = Param("A", TensorType((128, 64)))
    B = Param("B", TensorType((64, 32)))
    C = Param("C", TensorType((32, 8)))

    # Left-assoc: ((x @ A) @ B) @ C  — many intermediate matmuls
    left_assoc = Op.make(
        "matmul", Op.make("matmul", Op.make("matmul", x, A), B), C
    )
    # Right-assoc: x @ (A @ (B @ C))  — fused weight
    right_assoc = Op.make(
        "matmul", x, Op.make("matmul", A, Op.make("matmul", B, C))
    )

    cost_left = flops_cost(left_assoc)
    cost_right = flops_cost(right_assoc)
    # Right-assoc should be cheaper for funnel dims
    assert cost_right < cost_left
    print(f"Left FLOPs:  {cost_left:.0f}")
    print(f"Right FLOPs: {cost_right:.0f}")
    print(f"Speedup: {cost_left / cost_right:.1f}x")


# ---------------------------------------------------------------------------
#  Lowering-aware pricing — executor_overhead / executor_cost_for /
#  fused_cost_for / lowering_aware_cost_for
# ---------------------------------------------------------------------------


def _scan_leaf(i: int, d: int = 4) -> Op:
    return Op.make("aff_diag", _p(f"a{i}", d), _p(f"b{i}", d))


def _balanced_tree(leaves: list[Op]) -> Op:
    while len(leaves) > 1:
        nxt = [
            Op.make("affd_compose", a, b)
            for a, b in zip(leaves[::2], leaves[1::2], strict=False)
        ]
        if len(leaves) % 2:
            nxt.append(leaves[-1])
        leaves = nxt
    return leaves[0]


def test_executor_overhead_generic_counts_ops():
    x = _v("x", 4, 8)
    # hand-built chain: add(mul(x, 2), neg(x)) -> 3 dispatched ops
    t = Op.make("add", Op.make("mul", x, Const(2.0)), Op.make("neg", x))
    assert executor_overhead(t, "generic") == 3.0
    # view ops still dispatch under the generic evaluator
    tv = Op.make("transpose", t, dim0=0, dim1=1)
    assert executor_overhead(tv, "generic") == 4.0
    # per-occurrence, not DAG-dedup: extract_best recovers local cost
    # as f(t) − Σf(children), exact only for additive fns — shared
    # subtrees are already billed once at the e-class level
    m = Op.make("mul", x, _p("W", 8, 8))
    assert executor_overhead(Op.make("add", m, m), "generic") == 3.0
    # leaves and Consts dispatch nothing
    assert executor_overhead(x, "generic") == 0.0
    assert executor_overhead(Const(1.0), "generic") == 0.0
    # memo reuse returns the cached count
    memo: dict = {}
    first = executor_overhead(tv, "generic", memo)
    assert executor_overhead(tv, "generic", memo) == first


def test_executor_overhead_batched_scan_compose_tree():
    leaves = [_scan_leaf(i) for i in range(4)]
    tree = _balanced_tree(leaves)
    t = Op.make("applyd", tree, _v("h", 4))
    # 4 leaves -> ceil(log2 4) = 2 batched levels; uniform aff_diag
    # leaves batch into ONE kind slot (a_shared/b_gather stack the
    # leaves — not T sequential evals); outside spine: the applyd
    # root (the h Var is a leaf — no dispatch).
    assert executor_overhead(t, "batched_scan") == 2 + 1 + 1
    # A 3-leaf tree: ceil(log2 3) = 2 levels.
    t3 = Op.make(
        "applyd",
        Op.make(
            "affd_compose",
            leaves[0],
            Op.make("affd_compose", leaves[1], leaves[2]),
        ),
        _v("h3", 4),
    )
    assert executor_overhead(t3, "batched_scan") == 2 + 1 + 1
    # A lone map leaf: 0 levels, one leaf-kind slot, the apply root.
    t1 = Op.make("applyd", leaves[0], _v("h1", 4))
    assert executor_overhead(t1, "batched_scan") == 0 + 1 + 1
    # A bare-Param leaf still pays one slot evaluation.
    tp = Op.make(
        "applyd",
        Op.make("affd_compose", _p("M", 4, 4), leaves[0]),
        _v("hp", 4),
    )
    assert executor_overhead(tp, "batched_scan") == 1 + 2 + 1
    # (levels=1; two leaf kinds — bare Param + aff_diag — 1+1 slots;
    # +1 applyd root)
    # A shared spine subtree dedups: compose(sc, sc) has 2 leaves.
    sc = Op.make("affd_compose", leaves[0], leaves[1])
    dag = Op.make(
        "applyd", Op.make("affd_compose", sc, sc), _v("h2", 4)
    )
    assert executor_overhead(dag, "batched_scan") == 1 + 1 + 1
    # (2 dedup'd aff_diag leaves -> 1 level; 1 leaf kind; +1 root)
    # A shared h-subterm across two scan roots hits the memo on the
    # second walk (same spine, same outside nodes).
    memo: dict = {}
    hs = Op.make("neg", _v("hs", 4))
    ta = Op.make("applyd", tree, hs)
    tb = Op.make("applyd", tree, Op.make("add", hs, hs))
    assert executor_overhead(ta, "batched_scan", memo) == 2 + 1 + 2
    assert executor_overhead(tb, "batched_scan", memo) == 2 + 1 + 3


def test_executor_overhead_non_scan_and_compiled():
    non = Op.make("neg", _v("x", 4))
    # Non-scan roots get the serial-fallback (generic) count.
    assert executor_overhead(non, "batched_scan") == executor_overhead(
        non, "generic"
    )
    # An apply op with no spine argument is not a scan term either.
    assert executor_overhead(Op.make("apply"), "batched_scan") == 1.0
    # Compiled dispatch is counted via fusion regions, not nodes.
    assert executor_overhead(non, "compiled") == 0.0
    with pytest.raises(ValueError, match="unknown lowering"):
        executor_overhead(non, "nope")


def test_executor_cost_for_adds_dispatch_overhead():
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 10.0,
    }
    x = _v("x", 4, 8)
    t = Op.make("add", Op.make("mul", x, Const(2.0)), Op.make("neg", x))
    oh = executor_overhead(t, "generic")
    fn = executor_cost_for(prof, lowering="generic")
    assert fn.__name__ == "executor_cost_for"
    assert fn.profile is prof
    assert fn.lowering == "generic"
    # Nanoseconds like roofline_cost: +10 us per dispatched op.
    assert fn(t) == pytest.approx(
        roofline_cost_for(prof)(t) + oh * 10e-6 * 1e9
    )
    # Memo reuse returns the cached value.
    memo: dict = {}
    assert fn(t, memo) == fn(t, memo)
    # depth base: critical path + dispatch.
    fd = executor_cost_for(prof, lowering="generic", base="depth")
    assert fd(t) == pytest.approx(
        depth_cost_for(prof)(t) + oh * 10e-6 * 1e9
    )
    # flops base: dispatch billed as us-to-flop launch-equivalents.
    ff = executor_cost_for(prof, lowering="generic", base="flops")
    assert ff(t) == pytest.approx(flops_cost(t) + oh * 10.0)
    # An object profile reads dispatch_us off attributes.
    ns_prof = types.SimpleNamespace(
        tflops=2.5, gbps=89.0, launch_us=8.7, dispatch_us=5.0
    )
    fo = executor_cost_for(ns_prof)
    assert fo(t) == pytest.approx(
        roofline_cost_for(ns_prof)(t) + oh * 5e-6 * 1e9
    )
    # No dispatch_us: falls back to the built-in launch constant —
    # on dicts and on plain profiles alike.
    no_disp = {"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7}
    for p in (None, no_disp, ns_prof.__class__(**no_disp)):
        fp = executor_cost_for(p)
        assert fp(t) == pytest.approx(
            roofline_cost_for(p)(t) + oh * _LAUNCH_S * 1e9
        )
    # batched_scan lowering on a non-scan term = generic overhead.
    fb = executor_cost_for(prof, lowering="batched_scan")
    assert fb(t) == pytest.approx(fn(t))
    # compiled delegates to the fusion model.
    fc = executor_cost_for(prof, lowering="compiled")
    assert fc(t) == pytest.approx(fused_cost_for(prof)(t))
    with pytest.raises(ValueError, match="unknown lowering"):
        executor_cost_for(prof, lowering="bogus")
    with pytest.raises(ValueError, match="unknown base"):
        executor_cost_for(prof, base="bogus")


def test_fused_cost_for_pointwise_region():
    x = _v("x", 512, 512)
    mul = Op.make("mul", x, x)
    t = Op.make("add", mul, Op.make("neg", x))
    fused = fused_cost_for()(t)
    unfused = roofline_cost(t)
    # The 3-op pointwise chain fuses to one kernel: the dominant
    # member's roofline + one dispatch — well under the per-op sum.
    dom = max(
        _local_roofline(t),
        _local_roofline(mul),
        _local_roofline(t.args[1]),
    )
    assert fused == pytest.approx(dom + _LAUNCH_S * 1e9)
    assert fused < unfused / 2
    # A fusible diamond dedups its shared member in the region.
    dia = Op.make("add", mul, mul)
    fused_dia = fused_cost_for()(dia)
    assert fused_dia == pytest.approx(
        max(_local_roofline(dia), _local_roofline(mul))
        + _LAUNCH_S * 1e9
    )
    # A non-fusible op keeps its own kernel: local + one dispatch.
    W = _p("W", 512, 512)
    mm = Op.make("matmul", x, W)
    assert fused_cost_for()(mm) == pytest.approx(
        _local_roofline(mm) + _LAUNCH_S * 1e9
    )
    # Mixed term: the GEMM prices normally, the pointwise tail fuses.
    mix = Op.make("neg", Op.make("mul", mm, mm))
    fmix = fused_cost_for()(mix)
    assert fmix == pytest.approx(
        _local_roofline(mm)
        + max(_local_roofline(mix), _local_roofline(mix.args[0]))
        + 2 * _LAUNCH_S * 1e9
    )
    assert fmix < roofline_cost(mix)
    # Leaves emit no kernel; memo reuse hits the cache.
    assert fused_cost_for()(x) == 0.0
    memo: dict = {}
    f = fused_cost_for({"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7})
    assert f(mix, memo) == f(mix, memo)


def test_lowering_aware_cost_for_picks_per_term():
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 10.0,
    }
    cost = lowering_aware_cost_for(prof)
    assert cost.__name__ == "lowering_aware_cost_for"
    assert cost.profile is prof
    x = _v("x", 512, 512)

    # Pure pointwise: the compiled lowering wins on fusion.
    pw = Op.make("add", Op.make("mul", x, x), Op.make("neg", x))
    assert cost.best_lowering(pw) == "compiled"
    assert cost(pw) == pytest.approx(fused_cost_for(prof)(pw))

    # A lone GEMM: fusion cannot shrink one kernel and there is no
    # scan to batch — every lowering prices identically and the
    # LOWERINGS order breaks the tie toward "generic".
    mm = Op.make("matmul", x, _p("W", 512, 512))
    assert cost.best_lowering(mm) == "generic"

    # An applyd compose tree: the batched scan's log-level dispatch
    # count beats per-node dispatch (generic) and per-node kernels
    # (compiled — the carrier ops are not pointwise-fusible).
    leaves = [_scan_leaf(i) for i in range(8)]
    scan = Op.make("applyd", _balanced_tree(leaves), _v("h", 4))
    assert cost.best_lowering(scan) == "batched_scan"
    assert cost(scan) == pytest.approx(
        executor_cost_for(prof, lowering="batched_scan")(scan)
    )

    # Memo reuse + a restricted lowering set.
    memo: dict = {}
    assert cost(pw, memo) == cost(pw, memo)
    only_generic = lowering_aware_cost_for(prof, lowerings=("generic",))
    assert only_generic.best_lowering(pw) == "generic"
    with pytest.raises(ValueError, match="non-empty"):
        lowering_aware_cost_for(prof, lowerings=())
    with pytest.raises(ValueError, match="unknown lowering"):
        lowering_aware_cost_for(prof, lowerings=("bogus",))


def test_lowering_aware_all_lowerings_constant():
    assert LOWERINGS == ("generic", "batched_scan", "compiled")
