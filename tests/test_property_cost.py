"""Property tests for the cost models (``catopt_core.cost``).

Pins the algebraic invariants the extractor relies on: costs are
finite and non-negative for arbitrary terms, and the commutative
elementwise ops (``add``/``mul``) price identically under operand
swaps — the property that lets extraction treat commuted forms as
genuinely equal work.
"""

from __future__ import annotations

import math

import pytest
from catopt_core.cost import (
    _shape_of,
    count_cost,
    dag_cost,
    flops_cost,
    launch_aware_cost,
)
from catopt_core.ir import Const, Op, TensorType, Var
from hypothesis import given

from tests.test_property_strategies import terms, uniform_terms


@given(terms())
def test_costs_are_finite_and_non_negative(t):
    for fn in (flops_cost, count_cost, launch_aware_cost):
        c = fn(t)
        assert c >= 0.0
        assert math.isfinite(c)
    assert dag_cost(t, flops_cost) >= 0.0
    assert math.isfinite(dag_cost(t, launch_aware_cost))


@given(uniform_terms(max_leaves=6), uniform_terms(max_leaves=6))
def test_flops_invariant_under_commutative_swap(a, b):
    for op in ("add", "mul"):
        assert flops_cost(Op.make(op, a, b)) == flops_cost(
            Op.make(op, b, a)
        )
        assert count_cost(Op.make(op, a, b)) == count_cost(
            Op.make(op, b, a)
        )


@pytest.mark.xfail(
    reason=(
        "cost._infer_op_shape short-circuits when ANY operand shape is "
        "unknown, returning shapes[0] (or None) instead of the "
        "symmetric broadcast — so add(a, b) and add(b, a) price "
        "differently when the FIRST operand's shape is unknown"
    ),
    strict=False,
)
def test_flops_invariant_under_commutative_swap_unknown_shape():
    known = Var("x", TensorType((2,)))
    unknown = Op.make("max", Const(0.0), Const(0.0))
    assert _shape_of(unknown) is None
    assert flops_cost(Op.make("add", known, unknown)) == flops_cost(
        Op.make("add", unknown, known)
    )


@given(terms(max_leaves=6))
def test_unary_wrapping_never_reduces_cost(t):
    base = flops_cost(t)
    for op in ("neg", "square", "exp", "sigmoid", "silu"):
        assert flops_cost(Op.make(op, t)) >= base


@given(terms(max_leaves=6))
def test_leaf_contribution_is_zero(t):
    if isinstance(t, Op):
        return
    assert flops_cost(t) == 0.0
    assert count_cost(t) == 0.0
    assert launch_aware_cost(t) == 0.0
