"""Tests for cost models."""

import pytest
from catopt.ir import Op, Var, Const, Param, TensorType
from catopt.cost import count_cost, flops_cost, CostModel


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
    left_assoc = Op.make("matmul", Op.make("matmul",
                                  Op.make("matmul", x, A), B), C)
    # Right-assoc: x @ (A @ (B @ C))  — fused weight
    right_assoc = Op.make("matmul", x, Op.make("matmul", A, Op.make("matmul", B, C)))

    cost_left = flops_cost(left_assoc)
    cost_right = flops_cost(right_assoc)
    # Right-assoc should be cheaper for funnel dims
    assert cost_right < cost_left
    print(f"Left FLOPs:  {cost_left:.0f}")
    print(f"Right FLOPs: {cost_right:.0f}")
    print(f"Speedup: {cost_left/cost_right:.1f}x")
