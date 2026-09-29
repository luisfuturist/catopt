"""Tests for cost models."""

import types

import pytest
from catopt_core.cost import (
    _LAUNCH_S,
    _PEAK_BW,
    _PEAK_FLOPS,
    LOWERINGS,
    CostModel,
    count_cost,
    depth_cost_for,
    executor_cost_for,
    executor_overhead,
    flops_cost,
    fused_cost_for,
    fusion_regions,
    lowering_aware_cost_for,
    param_bytes_cost,
    param_bytes_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt_core.ir import Const, Op, Param, TensorType, Var


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
    from catopt_core.cost import _infer_op_shape

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
    from catopt_core.cost import _infer_op_shape

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
    from catopt_core.cost import _infer_op_shape

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
    from catopt_core.cost import _INVALID, _infer_op_shape

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
    from catopt_core.cost import (
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
    from catopt_core.cost import _INVALID_COST, _shape_of
    from catopt_core.egraph import EGraph
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
    from catopt_core.cost import _SOLVER_FACTOR

    x = _v("x", 4, 8)
    # hand-built chain: add(mul(x, 2), neg(x)) -> 3 dispatched ops
    t = Op.make("add", Op.make("mul", x, Const(2.0)), Op.make("neg", x))
    assert executor_overhead(t, "generic") == 3.0
    # view ops still dispatch under the generic evaluator
    tv = Op.make("transpose", t, dim0=0, dim1=1)
    assert executor_overhead(tv, "generic") == 4.0
    # per-occurrence, not DAG-dedup: extract_best recovers local cost
    # as f(t) - Σf(children), exact only for additive fns — shared
    # subtrees are already billed once at the e-class level
    m = Op.make("mul", x, _p("W", 8, 8))
    assert executor_overhead(Op.make("add", m, m), "generic") == 3.0
    # leaves and Consts dispatch nothing
    assert executor_overhead(x, "generic") == 0.0
    assert executor_overhead(Const(1.0), "generic") == 0.0
    # a param-only foldable subtree dispatches nothing — the lowerer
    # materialises it to a bound parameter before the first forward
    assert (
        executor_overhead(
            Op.make(
                "add",
                Op.make("mul", _p("P1", 8, 8), _p("P2", 8, 8)),
                x,
            ),
            "generic",
        )
        == 1.0
    )
    # ...but a param-only op the fold can't touch still runs.
    assert (
        executor_overhead(
            Op.make("trace", _p("M", 4, 4), usize=[4]), "generic"
        )
        == _SOLVER_FACTOR
    )
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
    x = _v("x", 4)
    non = Op.make("neg", x)
    # Non-scan roots get the serial-fallback (generic) count.
    assert executor_overhead(non, "batched_scan") == executor_overhead(
        non, "generic"
    )
    # An apply op with no spine argument is not a scan term either.
    assert executor_overhead(Op.make("apply"), "batched_scan") == 1.0
    # Compiled dispatch counts fusion regions (kernels), not nodes:
    # neg is pointwise — ONE fused region, not one per node.
    assert executor_overhead(non, "compiled") == 1.0
    assert executor_overhead(x, "compiled") == 0.0  # a leaf
    assert (
        executor_overhead(
            Op.make("neg", Op.make("matmul", x, _p("W", 4, 4))),
            "compiled",
        )
        == 2.0
    )  # the GEMM boundary + the fused neg
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


def _kernel_ns(flops: float, in_b: float, out_b: float) -> float:
    """One region's kernel time, per fused_cost_for's documented
    composition: max(Σ member FLOPs / peak, region traffic / bw)."""
    return max(flops / _PEAK_FLOPS, (in_b + out_b) / _PEAK_BW) * 1e9


def test_fused_cost_for_pointwise_region():
    x = _v("x", 512, 512)
    el_b = 512 * 512 * 4  # one fp32 tensor's bytes
    mul = Op.make("mul", x, x)
    t = Op.make("add", mul, Op.make("neg", x))
    fused = fused_cost_for()(t)
    unfused = roofline_cost(t)
    # The 3-op pointwise chain fuses to one kernel: Σ member flops
    # vs deduplicated region traffic (x read ONCE + the root write),
    # + one launch + one graph-level dispatch — well under the
    # per-op sum.  (No dispatch_us in the default profile → the
    # dispatch falls back to the launch constant.)
    exp = _kernel_ns(3 * 512 * 512, el_b, el_b) + _LAUNCH_S * 1e9 + 80_000.0
    assert fused == pytest.approx(exp)
    # The per-graph overhead (~80us fallback) dominates at this size —
    # a small fused graph is barely under the unfused price, which is
    # the measured reality: Inductor's guard+call overhead means a
    # tiny graph doesn't pay.
    assert fused < unfused
    # A fusible diamond dedups its shared member in the region.
    dia = Op.make("add", mul, mul)
    fused_dia = fused_cost_for()(dia)
    assert fused_dia == pytest.approx(
        _kernel_ns(2 * 512 * 512, el_b, el_b) + _LAUNCH_S * 1e9 + 80_000.0
    )
    # A non-fusible op keeps its own kernel: its flops dominate
    # the region traffic (reads x + W, writes the product).
    W = _p("W", 512, 512)
    mm = Op.make("matmul", x, W)
    assert fused_cost_for()(mm) == pytest.approx(
        _kernel_ns(2 * 512**3, 2 * el_b, el_b) + _LAUNCH_S * 1e9 + 80_000.0
    )
    # Mixed term: the GEMM is its own kernel, the pointwise tail
    # fuses reading the GEMM's output once.
    mix = Op.make("neg", Op.make("mul", mm, mm))
    fmix = fused_cost_for()(mix)
    assert fmix == pytest.approx(
        _kernel_ns(2 * 512**3, 2 * el_b, el_b)
        + _kernel_ns(2 * 512 * 512, el_b, el_b)
        + 2 * _LAUNCH_S * 1e9
        + 80_000.0  # per-graph: max(dispatch, measured overhead)
    )
    assert fmix < roofline_cost(mix)
    # Leaves emit no kernel; memo reuse hits the cache.
    assert fused_cost_for()(x) == 0.0
    memo: dict = {}
    f = fused_cost_for({"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7})
    assert f(mix, memo) == f(mix, memo)


def test_fused_cost_region_traffic():
    """Region bytes, not dominant member: externals read + boundary
    writes decide the fused kernel's time — two regions reading the
    SAME big input and a small one differ from one reading two big
    inputs, which dominant-member pricing could not see."""
    x = _v("x", 512, 512)
    y = _v("y", 512, 512)
    el_b = 512 * 512 * 4
    # one region {add}: reads x AND y (two externals), writes out.
    add = Op.make("add", x, y)
    f_add = fused_cost_for()(add)
    assert f_add == pytest.approx(
        _kernel_ns(512 * 512, 2 * el_b, el_b) + _LAUNCH_S * 1e9 + 80_000.0
    )
    # the same op reading ONE external twice dedups the read —
    # fewer bytes than the two-input form.
    add2 = Op.make("add", x, x)
    assert fused_cost_for()(add2) < f_add
    # Interior values stay in registers: add(mul, sigmoid) fuses the
    # whole pointwise DAG — only x, y in and the root out cross
    # memory.  Feeding the same subtrees to a concat boundary splits
    # them into per-producer kernels that must WRITE their outputs —
    # more launches AND more bytes, so strictly pricier.
    pw = Op.make("add", Op.make("mul", x, y), Op.make("sigmoid", x))
    cat = Op.make(
        "concat", Op.make("mul", x, y), Op.make("sigmoid", x), dim=0
    )
    assert len(fusion_regions(pw)) == 1
    assert len(fusion_regions(cat)) == 3
    assert fused_cost_for()(cat) > fused_cost_for()(pw)
    # ...and a transparent view does not hide the boundary: neg's
    # value crosses the region through the transpose to reach `add`
    # in another region, so it is billed as a write.
    neg = Op.make("neg", x)
    t = Op.make(
        "add",
        Op.make("matmul", x, _p("W", 512, 512)),
        Op.make("transpose", neg, dim0=0, dim1=1),
    )
    member_regions = [
        sorted(tt.op for tt in r) for r in fusion_regions(t)
    ]
    assert sorted(member_regions) == [["add", "neg"], ["matmul"]]
    # Carrier packaging is transparent too: aff(neg, neg) forwards
    # BOTH args through to `add`'s region — neg stays interior (its
    # only opaque consumer is in-region), and the duplicated parent
    # edge is dedup'd in the boundary walk.
    neg2 = Op.make("neg", y)
    t2 = Op.make(
        "add",
        Op.make("matmul", x, _p("U", 512, 512)),
        Op.make("aff", neg2, neg2),
    )
    regions2 = [sorted(tt.op for tt in r) for r in fusion_regions(t2)]
    assert sorted(regions2) == [["add", "neg"], ["matmul"]]
    assert fused_cost_for()(t2) > 0


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

    # An applyd compose tree: the diagonal-carrier ops are pointwise
    # under the hood, so the compiled lowering fuses the whole spine
    # into one region — cheaper than the batched scan's log-level
    # dispatch count AND the generic per-node dispatch.  (This is the
    # measured-fidelity asymmetry the region model captures.)
    leaves = [_scan_leaf(i) for i in range(8)]
    scan = Op.make("applyd", _balanced_tree(leaves), _v("h", 4))
    assert cost.best_lowering(scan) == "compiled"
    assert cost(scan) == pytest.approx(
        executor_cost_for(prof, lowering="compiled")(scan)
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


# ---------------------------------------------------------------------------
#  Fusion regions — the compiled lowering's kernel partition
# ---------------------------------------------------------------------------


def _applyd_chain(n: int, d: int = 4) -> Op:
    """applyd over a balanced affd_compose tree of n aff_diag leaves."""
    return Op.make(
        "applyd",
        _balanced_tree([_scan_leaf(i, d) for i in range(n)]),
        _v("h", d),
    )


def _ops_of(regions) -> list[list[str]]:
    return [sorted(t.op for t in r) for r in regions]


def test_fusion_regions_pointwise_chain_scales_sublinear():
    """A T-step pointwise chain prices at ≤ O(log T) regions — one.

    The measured-fidelity fix: the compiled lowering fuses the whole
    chain into ~one kernel (0.05ms measured on the applyd spine at
    T=128), NOT the O(T) launches the generic evaluator pays (~0.9ms).
    Region counting is global — a whole-DAG partition — so the model
    is reporting/frontier-only (extract_best's subtractive local-cost
    decomposition can hide a sibling merge in a clamped local; the
    fusion_regions docstring carries the contract).
    """
    import math

    for t_steps in (4, 8, 128):
        # a left-leaning chain of unary pointwise ops
        t = _v("x", 4, 4)
        for _ in range(t_steps):
            t = Op.make("neg", Op.make("sigmoid", t))
        n_regions = len(fusion_regions(t))
        assert n_regions == 1
        assert n_regions <= math.ceil(math.log2(t_steps)) + 1
    # the cost is ~T-independent: one kernel regardless of chain
    # depth — member flops/reads grow with T but stay far below the
    # serial evaluator's O(T) dispatches.
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 10.0,
    }
    f = fused_cost_for(prof)
    assert f(_applyd_chain(128)) < f(_applyd_chain(4)) * 2


def test_fusion_regions_applyd_spine_is_one_region():
    """The diagonal-carrier scan spine fuses whole under compiled.

    affd_compose/applyd bindings are pure pointwise arithmetic on the
    carried pair (f0⊙g0, f0⊙g1+f1 / f0⊙h+f1) and the aff_diag leaves
    are tuple packaging — the whole spine is one pointwise region,
    matching Inductor's single fused kernel.  Generic counts one
    dispatch per NODE (2T); batched_scan counts O(log T) levels.
    """
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 10.0,
    }
    for t_steps in (8, 128):
        chain = _applyd_chain(t_steps)
        regions = fusion_regions(chain)
        assert len(regions) == 1
        assert {t.op for t in regions[0]} == {"applyd", "affd_compose"}
        # the overhead count IS the region count
        assert executor_overhead(chain, "compiled") == 1.0
        assert executor_overhead(chain, "generic") == 2 * t_steps
        compiled = executor_cost_for(prof, lowering="compiled")(chain)
        generic = executor_cost_for(prof, lowering="generic")(chain)
        batched = executor_cost_for(prof, lowering="batched_scan")(
            chain
        )
        assert compiled < generic
        assert compiled < batched
        # With DISTINCT param leaves, no leaf fast path engages —
        # the batched executor pays a per-leaf eval plus its level
        # machinery, so on this profile it cannot beat the serial
        # evaluator (the honest CPU negative of data-dependent
        # leaves; the fast-path case is pinned below).
        assert batched > generic


def test_batched_scan_latency_fast_paths():
    """Level-batched price beats generic only when leaf fast paths
    engage — shared transition term + select-of-one-base inputs."""
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        "leaf_eval_us": 5.0,
    }
    x = _v("x", 128, 8)
    u = Op.make("matmul", x, _p("W", 8, 8))
    gamma = _p("gamma", 8)
    # LTI scan: every leaf is aff_diag(gamma, select(u, 0, i)) —
    # leaf_a_shared + leaf_b_gather both engage.
    leaves = [
        Op.make(
            "aff_diag",
            gamma,
            Op.make("select", u, dim=0, index=i),
        )
        for i in range(128)
    ]
    scan = Op.make("applyd", _balanced_tree(leaves), _v("h", 8))
    generic = executor_cost_for(prof, lowering="generic")(scan)
    batched = executor_cost_for(prof, lowering="batched_scan")(scan)
    compiled = executor_cost_for(prof, lowering="compiled")(scan)
    # ~7 batched levels + one base eval + one gather ≪ ~256 serial
    # dispatches; the fused spine still beats everything.
    assert compiled < batched < generic
    # Same shape but data-dependent leaves (distinct transitions,
    # non-select inputs): no fast path → batched cannot win at this
    # size on this profile (the honest negative).
    leaves2 = [
        Op.make("aff_diag", _p(f"g{i}", 8), _p(f"b{i}", 8))
        for i in range(128)
    ]
    scan2 = Op.make("applyd", _balanced_tree(leaves2), _v("h", 8))
    assert executor_cost_for(prof, lowering="batched_scan")(
        scan2
    ) > executor_cost_for(prof, lowering="generic")(scan2)


def _aff_tree(leaves: list[Op]) -> Op:
    while len(leaves) > 1:
        nxt = [
            Op.make("aff_compose", a, b)
            for a, b in zip(leaves[::2], leaves[1::2], strict=False)
        ]
        if len(leaves) % 2:
            nxt.append(leaves[-1])
        leaves = nxt
    return leaves[0]


def test_batched_scan_latency_dense_and_leaf_edges():
    """The dense affine (``apply``) compose path and non pair-carrier
    leaf shapes take their own priced branches."""
    from catopt_core.cost import _leaf_gather_base, _leaf_shared_a

    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        "leaf_eval_us": 5.0,
    }
    f = executor_cost_for(prof, lowering="batched_scan")
    # dense affine spine: apply over aff_compose of aff(A,b) leaves —
    # the (d+1)x(d+1) bmm compose path (4 calls/level, s**1.5 flops).
    d = 4
    aff_leaves = [
        Op.make("aff", _p(f"A{i}", d, d), _p(f"b{i}", d))
        for i in range(8)
    ]
    dense_scan = Op.make("apply", _aff_tree(aff_leaves), _v("h", d))
    b = f(dense_scan)
    g = executor_cost_for(prof, lowering="generic")(dense_scan)
    assert b > 0 and g > 0
    # leaves with no pair structure (a bare Param leaf popped FIRST
    # in the spine — aff_diag last arg — falls to the per-leaf-eval
    # arm) — still a finite positive price.
    spine = Op.make(
        "affd_compose",
        Op.make("aff_diag", _p("a", d), _p("bb", d)),
        _p("M", d, d),
    )
    odd = Op.make("applyd", spine, _v("h2", d))
    assert f(odd) > 0
    # a spine with a shared compose subtree: the DAG walk dedups the
    # revisit — same price structure, finite positive cost.
    sc = Op.make(
        "affd_compose",
        Op.make("aff_diag", _p("g", d), _p("b", d)),
        Op.make("aff_diag", _p("g", d), _p("b2", d)),
    )
    shared_spine = Op.make("affd_compose", sc, sc)
    assert f(Op.make("applyd", shared_spine, _v("h3", d))) > 0
    # helper edge cases — shared-a needs uniform first args.
    diag = Op.make("aff_diag", _p("g", d), _p("b", d))
    eye = Op.make("eye", dim=d)  # zero-arg leaf
    assert not _leaf_shared_a([])
    assert _leaf_shared_a([diag, diag])
    assert not _leaf_shared_a(
        [diag, Op.make("aff_diag", _p("g2", d), _p("b", d))]
    )
    # identically-NAMED Params count as one shared transition even
    # when they are different objects (the LTI rule).
    assert _leaf_shared_a(
        [
            Op.make("aff_diag", _p("g", 4), _p("b", d)),
            Op.make("aff_diag", _p("g", 8), _p("b", d)),
        ]
    )
    assert not _leaf_shared_a([diag, eye])
    # gather needs every b = select(same base, same dim).
    u = Op.make("matmul", _v("x", 8, d), _p("W", d, d))
    g_leaves = [
        Op.make(
            "aff_diag", _p("g", d), Op.make("select", u, dim=0, index=i)
        )
        for i in range(4)
    ]
    assert _leaf_gather_base(g_leaves) is u
    # non-select b / missing attrs / mismatched base → None.
    assert _leaf_gather_base([diag, diag]) is None
    bad_dim = Op.make(
        "aff_diag",
        _p("g", d),
        Op.make("select", u, dim="x", index=0, validate=False),
    )
    assert _leaf_gather_base([g_leaves[0], bad_dim]) is None
    other_base = Op.make(
        "aff_diag",
        _p("g", d),
        Op.make("select", _v("z", 8, d), dim=0, index=0),
    )
    assert _leaf_gather_base([g_leaves[0], other_base]) is None
    # a zero-arg leaf also breaks the pair check.
    assert _leaf_gather_base([g_leaves[0], eye]) is None
    # a leaf with <2 args breaks it too.
    one_arg = Op.make("neg", _p("q", d))
    assert _leaf_gather_base([g_leaves[0], one_arg]) is None


def test_batched_scan_profile_fallbacks():
    """None / dict-without-leaf_eval profiles use the conservative
    4*dispatch leaf-eval fallback; a dict WITH leaf_eval_us uses it."""
    scan = Op.make(
        "applyd",
        Op.make("affd_compose", _scan_leaf(0), _scan_leaf(1)),
        _v("h", 4),
    )
    f_none = executor_cost_for(None, lowering="batched_scan")
    f_dict = executor_cost_for(
        {
            "tflops": 2.5,
            "gbps": 89.0,
            "launch_us": 8.7,
            "dispatch_us": 1.0,
            "leaf_eval_us": 1000.0,
        },
        lowering="batched_scan",
    )
    # a huge leaf_eval_us must dominate the leaf materialisation
    assert f_dict(scan) > f_none(scan)
    # object profile missing leaf_eval_us → same fallback as dicts.
    ns = types.SimpleNamespace(tflops=2.5, gbps=89.0, launch_us=8.7)
    f_obj = executor_cost_for(ns, lowering="batched_scan")
    assert f_obj(scan) > 0


def test_fusion_regions_hard_boundaries():
    """matmul / reductions / materialising layout ops split regions."""
    x = _v("x", 64, 64)
    # Two distinct GEMMs + a pointwise tail: 3 kernels — the fused
    # {add, neg, sigmoid} plus one per matmul.
    mm1 = Op.make("matmul", x, _p("W", 64, 64))
    mm2 = Op.make("matmul", x, _p("U", 64, 64))
    t = Op.make("add", Op.make("neg", mm1), Op.make("sigmoid", mm2))
    regions = fusion_regions(t)
    assert len(regions) == 3
    assert frozenset({mm1}) in regions
    assert frozenset({mm2}) in regions
    assert frozenset({t, t.args[0], t.args[1]}) in regions
    # A reduction is a boundary too: pointwise siblings still fuse.
    red = Op.make("add", Op.make("sum", x, dim=1), Op.make("mul", x, x))
    r_ops = _ops_of(fusion_regions(red))
    assert sorted(r_ops) == [["add", "mul"], ["sum"]]
    # Materialising layout ops bound regions; a gather does too.
    cat = Op.make("concat", x, x, dim=0)
    r_ops = _ops_of(fusion_regions(Op.make("neg", cat)))
    assert sorted(r_ops) == [["concat"], ["neg"]]
    gat = Op.make("index_select", x, dim=0, index=(0, 1))
    r_ops = _ops_of(fusion_regions(Op.make("neg", gat)))
    assert sorted(r_ops) == [["index_select"], ["neg"]]


def test_fusion_regions_transparent_plumbing_bridges():
    """Views + carrier packaging emit no kernel and don't split a
    region — a pointwise consumer unions with their descendants."""
    x = _v("x", 4, 4)
    y = _v("y", 4, 4)
    neg = Op.make("neg", x)
    t = Op.make(
        "add",
        Op.make("transpose", neg, dim0=0, dim1=1),
        y,
    )
    regions = fusion_regions(t)
    assert len(regions) == 1
    assert regions[0] == frozenset({t, neg})  # transpose is invisible
    # A lone packaging op is no kernel at all — a tuple assembly.
    assert (
        fusion_regions(Op.make("aff_diag", _p("a", 4), _p("b", 4)))
        == ()
    )
    assert fusion_regions(Op.make("transpose", x, dim0=0, dim1=1)) == ()
    assert (
        fused_cost_for()(Op.make("transpose", x, dim0=0, dim1=1)) == 0.0
    )


def test_fusion_regions_folded_params_are_free():
    """Param-only subtrees the lowerer folds contribute no kernel —
    the compiled graph reads them as materialised inputs."""
    x = _v("x", 64, 64)
    fold = Op.make("matmul", _p("W1", 64, 64), _p("W2", 64, 64))
    assert fusion_regions(fold) == ()
    assert fused_cost_for()(fold) == 0.0
    # ...including when the fold feeds a pointwise region: the
    # mul(P1,P2) subtree is an input, so only `add` is a region member
    # (and it does not union with the folded mul).
    t = Op.make(
        "add", x, Op.make("mul", _p("P1", 64, 64), _p("P2", 64, 64))
    )
    assert fusion_regions(t) == (frozenset({t}),)
    # the fold is billed as a kernel INPUT — read once by the add
    # kernel — not as a kernel of its own.
    assert fused_cost_for()(t) > 0


def test_fusion_regions_solver_and_leaves():
    """Solver ops are singleton regions billed _SOLVER_FACTOR
    dispatches; leaves emit nothing."""
    from catopt_core.cost import _SOLVER_FACTOR

    inv = Op.make("inv", _p("M", 4, 4))
    regions = fusion_regions(inv)
    assert regions == (frozenset({inv}),)
    # the extern solve dwarfs a dispatch — same floor the generic
    # count carries
    assert fused_cost_for()(inv) >= _SOLVER_FACTOR * _LAUNCH_S * 1e9
    # leaves and non-Op terms partition to nothing
    assert fusion_regions(_v("x", 4)) == ()
    assert fusion_regions(Const(1.0)) == ()
    assert fused_cost_for()(Const(1.0)) == 0.0
    # memo reuse returns the cached partition
    memo: dict = {}
    chain = _applyd_chain(4)
    assert fusion_regions(chain, memo) == fusion_regions(chain, memo)


# ---------------------------------------------------------------------------
#  Measured op_kernel_ns table — the shape-dependent kernel floor
# ---------------------------------------------------------------------------

#: A profile dict carrying measured kernel times far above the
#: roofline estimate, so the floor is visible in the price.
_PROF_KERNEL = {
    "tflops": 2.5,
    "gbps": 89.0,
    "launch_us": 8.7,
    "dispatch_us": 5.0,
    "op_kernel_ns": {
        "matmul": {"128x128x128": 200000.0, "512x512x512": 5000000.0},
        "pointwise": {"4096": 5000.0, "1048576": 400000.0},
        "reduce": {"1048576": 300000.0},
        "concat": {"1048576": 250000.0},
        "stack": {"1048576": 260000.0},
        "index_select": {"65536": 20000.0},
    },
}


def _bare(prof: dict) -> dict:
    """The same profile without the kernel table."""
    return {k: v for k, v in prof.items() if k != "op_kernel_ns"}


def test_kernel_signature_op_classes():
    from catopt_core.cost import _kernel_signature

    x = _v("x", 64, 64)
    assert _kernel_signature(
        Op.make("matmul", x, _p("W", 64, 64)), {}
    ) == (
        "matmul",
        (64.0, 64.0, 64.0),
    )
    # linear: (M,K,N) = (rows, in-features, out-features)
    lin = Op.make("linear", _v("l", 8, 16), _p("LW", 32, 16))
    assert _kernel_signature(lin, {}) == ("matmul", (8.0, 16.0, 32.0))
    # reduce buckets the streamed INPUT numel, the rest output numel
    assert _kernel_signature(Op.make("sum", x, dim=1), {}) == (
        "reduce",
        (4096.0,),
    )
    assert _kernel_signature(Op.make("concat", x, x, dim=0), {}) == (
        "concat",
        (8192.0,),
    )
    assert _kernel_signature(Op.make("stack", x, x, dim=0), {}) == (
        "stack",
        (8192.0,),
    )
    assert _kernel_signature(Op.make("add", x, x), {}) == (
        "pointwise",
        (4096.0,),
    )
    cls, sig = _kernel_signature(
        Op.make("index_select", x, dim=0, index=(0, 1)), {}
    )
    assert cls == "index_select" and sig == (128.0,)


def test_kernel_signature_fallbacks():
    """Unshapeable / un-bucketed ops return None → roofline path."""
    from catopt_core.cost import _kernel_signature

    # ill-typed op: _INVALID output shape
    bad = Op.make("add", _v("a", 2), _v("b", 3))
    assert _kernel_signature(bad, {}) is None
    # unknown (non-tuple) output shape
    xs = Var("xs", TensorType(None))
    assert _kernel_signature(Op.make("add", xs, xs), {}) is None
    # rank-1 matmul result (matvec): no (M,K,N) bucket
    mv = Op.make("matmul", _p("A", 4, 8), _v("v", 8))
    assert _kernel_signature(mv, {}) is None
    # batched matvec: matrix result but vector weight — K unreadable
    bmv = Op.make("matmul", _p("B", 2, 8, 4), _v("v", 4))
    assert _kernel_signature(bmv, {}) is None
    # non-int output dim / non-int reduction dim / unknown weight
    assert (
        _kernel_signature(
            Op.make("matmul", _v("x", 4, 8), _p("Wn", 8, None)), {}
        )
        is None
    )
    assert (
        _kernel_signature(
            Op.make("matmul", _v("x", 4, 8), _p("Wk", None, 8)), {}
        )
        is None
    )
    assert (
        _kernel_signature(
            Op.make(
                "matmul", _v("x", 4, 8), Var("w", TensorType(None))
            ),
            {},
        )
        is None
    )
    # ops without a measured class keep the roofline price
    assert (
        _kernel_signature(Op.make("contiguous", _v("z", 8, 8)), {})
        is None
    )


def test_kernel_lookup_parsing():
    """Table parsing: nearest-bucket in log space; malformed entries
    are skipped; empty/unusable tables disable the floor."""
    from catopt_core.cost import _kernel_lookup

    assert _kernel_lookup(None) is None
    assert _kernel_lookup({}) is None
    # non-dict class entries and unparseable keys are skipped — a
    # table with nothing usable disables the lookup entirely.
    assert _kernel_lookup({"matmul": 5}) is None
    assert _kernel_lookup({"pointwise": {"bogus": 1.0}}) is None
    kns = _kernel_lookup(
        {
            "matmul": {
                "bad": 1.0,
                "128x128x128": 100.0,
                "512x512x512": 500.0,
            },
            "pointwise": {"4096": 10.0},
            "junk": {"x": 1.0},
            "nd": 42,
        }
    )
    assert kns is not None
    mm = Op.make("matmul", _v("x", 128, 128), _p("W", 128, 128))
    assert kns(mm, {}) == 100.0
    # 384³ is nearer (log-space) to 512³ than to 128³
    mm2 = Op.make("matmul", _v("x", 384, 384), _p("W", 384, 384))
    assert kns(mm2, {}) == 500.0
    # a class absent from the table misses
    assert kns(Op.make("sum", _v("x", 4, 4), dim=0), {}) is None
    # as does an op with no signature at all
    assert kns(Op.make("contiguous", _v("a", 4)), {}) is None
    # bucket keys with a different signature arity never match
    kns2 = _kernel_lookup({"pointwise": {"4x4": 10.0}})
    assert kns2 is not None
    assert (
        kns2(Op.make("add", _v("a", 4, 4), _v("b", 4, 4)), {}) is None
    )


def test_roofline_measured_kernel_floor():
    """A measured bucket above the roofline estimate floors the price;
    one below it never undercuts."""
    mm = Op.make("matmul", _v("x", 2048, 128), _p("W", 128, 128))
    # sig (2048,128,128): nearer to 128³ (log-dist 4) than 512³ (6)
    assert roofline_cost_for(_PROF_KERNEL)(mm) == pytest.approx(
        200000.0
    )
    assert roofline_cost_for(_bare(_PROF_KERNEL))(mm) < 200000.0
    # a measured value BELOW the roofline does not lower the price
    small = dict(_PROF_KERNEL)
    small["op_kernel_ns"] = {"matmul": {"128x128x128": 1.0}}
    assert roofline_cost_for(small)(mm) == pytest.approx(
        roofline_cost_for(_bare(_PROF_KERNEL))(mm)
    )
    # measured classes: reduce / concat / stack / index_select /
    # pointwise all floor at their buckets when those exceed roofline
    prof = dict(_PROF_KERNEL)
    x1m = _v("big", 1024, 1024)
    assert roofline_cost_for(prof)(
        Op.make("sum", x1m, dim=1)
    ) == pytest.approx(300000.0)
    assert roofline_cost_for(prof)(
        Op.make("concat", _v("c", 512, 1024), _v("d", 512, 1024), dim=0)
    ) == pytest.approx(250000.0)
    assert roofline_cost_for(prof)(
        Op.make("stack", _v("c", 512, 1024), _v("d", 512, 1024), dim=0)
    ) == pytest.approx(260000.0)
    assert roofline_cost_for(prof)(
        Op.make("index_select", _v("g", 256, 256), dim=0, index=(0, 1))
    ) == pytest.approx(20000.0)
    # pointwise add at 1M elems floors at the 1M bucket
    assert roofline_cost_for(prof)(
        Op.make("add", x1m, _v("b2", 1024, 1024))
    ) == pytest.approx(400000.0)
    # …while a small pointwise op keeps the launch-floor roofline
    assert (
        roofline_cost_for(prof)(Op.make("neg", _v("s", 8, 8)))
        < 100000.0
    )
    # and an op with no measured class misses the table entirely
    assert roofline_cost_for(prof)(
        Op.make("contiguous", _v("z", 8, 8))
    ) == pytest.approx(
        roofline_cost_for(_bare(prof))(
            Op.make("contiguous", _v("z", 8, 8))
        )
    )
    # object profiles read the table off the attribute too
    ns_prof = types.SimpleNamespace(
        tflops=2.5,
        gbps=89.0,
        launch_us=8.7,
        op_kernel_ns={"matmul": {"128x128x128": 200000.0}},
    )
    assert roofline_cost_for(ns_prof)(mm) == pytest.approx(200000.0)
    # an empty table is no table
    ns_empty = types.SimpleNamespace(
        tflops=2.5, gbps=89.0, launch_us=8.7, op_kernel_ns={}
    )
    assert roofline_cost_for(ns_empty)(mm) == pytest.approx(
        roofline_cost_for(_bare(_PROF_KERNEL))(mm)
    )


def test_measured_floor_through_executor_and_depth():
    """The kernel table flows through executor_cost_for (generic,
    batched_scan, depth base), fused_cost_for, depth_cost_for and
    lowering_aware_cost_for."""
    prof = dict(_PROF_KERNEL)
    bare = _bare(prof)
    mm = Op.make("matmul", _v("x", 2048, 128), _p("W", 128, 128))
    # generic lowering: roofline arm + dispatch, both table-floored
    assert executor_cost_for(prof, lowering="generic")(
        mm
    ) > executor_cost_for(bare, lowering="generic")(mm)
    # depth base and fused/compiled lowering
    assert executor_cost_for(prof, lowering="generic", base="depth")(
        mm
    ) > executor_cost_for(bare, lowering="generic", base="depth")(mm)
    assert fused_cost_for(prof)(mm) > fused_cost_for(bare)(mm)
    assert depth_cost_for(prof)(mm) > depth_cost_for(bare)(mm)
    assert lowering_aware_cost_for(prof)(mm) == pytest.approx(
        min(
            executor_cost_for(prof, lowering=lw)(mm) for lw in LOWERINGS
        )
    )
    # batched_scan on a scan term: leaf-operand evals price through
    # the table (the gather base is a matmul here)
    u = Op.make("matmul", _v("u", 128, 8), _p("W", 8, 8))
    leaves = [
        Op.make(
            "aff_diag",
            _p("g", 8),
            Op.make("select", u, dim=0, index=i),
        )
        for i in range(8)
    ]
    scan = Op.make("applyd", _balanced_tree(leaves), _v("h", 8))
    assert executor_cost_for(prof, lowering="batched_scan")(
        scan
    ) >= executor_cost_for(bare, lowering="batched_scan")(scan)
    # flops base ignores the table (it is flop-denominated)
    assert executor_cost_for(prof, lowering="generic", base="flops")(
        scan
    ) == executor_cost_for(bare, lowering="generic", base="flops")(scan)


def test_fused_cost_region_measured_work_floor():
    """In the fusion model each region floors at the sum of its
    members' measured kernel work (solo launches stripped)."""
    prof = dict(_PROF_KERNEL)
    bare = _bare(prof)
    # multi-member pointwise region: three 1M-numel members each
    # floor at the 400µs bucket minus the launch — clearly above the
    # traffic roofline of the fused kernel.
    a, b = _v("a", 1024, 1024), _v("b", 1024, 1024)
    pw = Op.make("add", Op.make("mul", a, b), Op.make("neg", a))
    f_table = fused_cost_for(prof)(pw)
    f_roof = fused_cost_for(bare)(pw)
    # kernel = sum(m_i - launch); +1 fused launch +1 graph dispatch.
    exp = 3 * (400000.0 - 8700.0) + 8700.0 + 80_000.0
    assert f_table == pytest.approx(exp)
    assert f_table > f_roof
    # a singleton region of a measured class floors at its bucket
    mm = Op.make("matmul", _v("x", 512, 512), _p("W", 512, 512))
    assert fused_cost_for(prof)(mm) == pytest.approx(
        5000000.0 - 8700.0 + 8700.0 + 80_000.0
    )
    # a region whose members have no measured class is unchanged
    ct = Op.make("contiguous", _v("z", 8, 8))
    assert fused_cost_for(prof)(ct) == fused_cost_for(bare)(ct)


def test_graph_overhead_profile_paths():
    """_profile_graph_overhead_s: None / dict / object branches all
    feed the fused per-graph charge — the inductor guard overhead."""
    from catopt_core.cost import fused_cost_for

    x = _v("x", 8, 8)
    t = Op.make("neg", x)
    # default fallback (no profile at all)
    d = fused_cost_for()(t)
    # dict profile with an explicit overhead
    prof = dict(_PROF_KERNEL)
    prof["graph_overhead_us"] = 200.0
    hi = fused_cost_for(prof)(t)
    assert hi > d  # 200us > 80us fallback
    # object profile via a namespace
    import types

    obj = types.SimpleNamespace(
        tflops=2.5,
        gbps=89.0,
        launch_us=8.7,
        dispatch_us=5.0,
        leaf_eval_us=20.0,
        op_kernel_ns={},
        graph_overhead_us=0.0,
    )
    lo = fused_cost_for(obj)(t)
    assert lo < d  # 0us overhead < 80us fallback
