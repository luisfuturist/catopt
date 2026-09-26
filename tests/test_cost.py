"""Tests for cost models."""

import pytest

from catopt.cost import (
    CostModel,
    count_cost,
    flops_cost,
    param_bytes_cost,
    param_bytes_cost_for,
)
from catopt.ir import Const, Op, Param, TensorType, Var


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
    """The eps axis: chained low-rank factors beat the dense weight."""
    x = Var("x", TensorType((4, 64)))
    W = Param("W", TensorType((64, 64)))
    V = Param("eps_v_W_0", TensorType((8, 64)))
    U = Param("eps_u_W_0", TensorType((64, 8)))
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
