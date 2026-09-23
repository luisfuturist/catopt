"""Tests for rewrite rules."""

import pytest
from catopt.ir import Op, Var, Const, Param, TensorType, op_repr
from catopt.egraph import EGraph, Rewrite
from catopt.rules import (
    all_rules, SIMPLIFICATION_RULES, CATEGORICAL_RULES,
    COMM_ADD, COMM_MUL, ASSOC_ADD, ASSOC_MUL,
    ID_ADD, ID_MUL, DOUBLE_NEG, SUB_TO_ADD,
    SILU_EXPAND, SQUARE_EXPAND,
    DISTRIBUTE_MUL, FACTOR_MUL,
    NATURALITY_SCALAR, NATURALITY_SCALAR_REV,
    ASSOC_MATMUL, ASSOC_MATMUL_REV,
)


def test_rules_have_correct_names():
    assert COMM_ADD.name == "comm_add"
    assert ASSOC_MATMUL.name == "assoc_matmul"
    assert NATURALITY_SCALAR.name == "naturality_scalar"


def test_all_rules_contains_all():
    """ALL_RULES should be the union of simplification and categorical rules."""
    combined = SIMPLIFICATION_RULES + CATEGORICAL_RULES
    assert len(all_rules()) == len(combined)


def test_id_add_simplifies():
    """add(x, 0) should simplify to x."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("add", x, Const(0))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([ID_ADD], eid, max_iterations=5, max_nodes=100)

    from catopt.cost import count_cost
    best = eg.extract_best(eid, count_cost)
    # The best should be just "x" (leaf, 0 cost)
    # Note: after simplification, the e-class should contain both add(x,0) and x.
    # The lowest-cost term is x (cost 0) vs add(x,0) (cost 1).
    assert isinstance(best, Var) and best.name == "x"


def test_id_mul_simplifies():
    """mul(x, 1) should simplify to x."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("mul", x, Const(1))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([ID_MUL], eid, max_iterations=5, max_nodes=100)

    from catopt.cost import count_cost
    best = eg.extract_best(eid, count_cost)
    assert isinstance(best, Var) and best.name == "x"


def test_silu_expands():
    """silu(x) should expand to mul(x, sigmoid(x))."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("silu", x)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SILU_EXPAND], eid, max_iterations=5, max_nodes=100)

    # The e-class should now contain both silu(x) and mul(x, sigmoid(x))
    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "silu" in op_names  # original
    assert "mul" in op_names   # expanded


def test_sub_converts_to_add():
    """sub(a, b) should convert to add(a, neg(b))."""
    a = Var("a", TensorType((1, 4)))
    b = Var("b", TensorType((1, 4)))
    term = Op.make("sub", a, b)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SUB_TO_ADD], eid, max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "sub" in op_names  # original
    assert "add" in op_names   # converted


def test_square_expands():
    """square(x) should expand to mul(x, x)."""
    x = Var("x", TensorType((1, 4)))
    term = Op.make("square", x)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SQUARE_EXPAND], eid, max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    op_names = {n.op for n in root_class.nodes}
    assert "square" in op_names
    assert "mul" in op_names


def test_naturality_round_trip():
    """W @ (x * c) and (W @ x) * c should be equivalent."""
    x = Var("x", TensorType((1, 4)))
    W = Param("W", TensorType((4, 4)))
    c = Const(2.0)

    # Start with: W @ (x * c)  — matmul(W, mul(x, c))
    term = Op.make("matmul", W, Op.make("mul", x, c))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([NATURALITY_SCALAR, NATURALITY_SCALAR_REV], eid,
           max_iterations=5, max_nodes=100)

    root_class = eg.get_class(eid)
    # Should contain both forms
    assert len(root_class.nodes) >= 2


def test_matmul_associativity():
    """Both association orders of matmul should be reachable."""
    x = Var("x", TensorType((1, 4)))
    A = Param("A", TensorType((4, 4)))
    B = Param("B", TensorType((4, 4)))
    C = Param("C", TensorType((4, 4)))

    left = Op.make("matmul", Op.make("matmul", Op.make("matmul", x, A), B), C)
    eg = EGraph()
    eid = eg.add_term(left)
    eg.run([ASSOC_MATMUL, ASSOC_MATMUL_REV], eid,
           max_iterations=10, max_nodes=10000)

    root_class = eg.get_class(eid)
    # Both association orders should be present
    assert len(root_class.nodes) >= 2
