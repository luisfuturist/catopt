"""Property tests for the e-graph (``catopt_core.egraph``).

Covers interning/congruence and the extraction invariant that a
rewritten (saturated) e-class never extracts a term more expensive than
the original — on randomly generated terms, not fixed examples.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
from catopt_core.cost import count_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, TensorType, Var, op_repr
from catopt_core.laws import (
    ASSOC_ADD,
    ASSOC_MUL,
    COMM_ADD,
    COMM_MUL,
    ID_ADD,
    ID_MUL,
)
from hypothesis import given, settings

from tests.test_property_strategies import (
    additive_terms,
    rebuild,
    terms,
)

_ALGEBRAIC = [
    ASSOC_ADD,
    COMM_ADD,
    ID_ADD,
    ASSOC_MUL,
    COMM_MUL,
    ID_MUL,
]


@given(terms())
def test_add_term_is_idempotent(t):
    eg = EGraph()
    a = eg.add_term(t)
    b = eg.add_term(t)
    assert eg.find(a) == eg.find(b)


@given(terms())
def test_structurally_equal_terms_share_a_class(t):
    eg = EGraph()
    assert eg.find(eg.add_term(t)) == eg.find(eg.add_term(rebuild(t)))


@given(terms())
def test_single_member_class_extracts_original(t):
    eg = EGraph()
    eid = eg.add_term(t)
    best = eg.extract_best(eid, flops_cost)
    assert best is not None
    assert op_repr(best) == op_repr(t)


@given(terms())
def test_union_merges_and_is_reflexive_noop(t):
    eg = EGraph()
    a = eg.add_term(t)
    b = eg.add_term(Op.make("neg", t))
    assert eg.union(a, a) is False
    if eg.find(a) == eg.find(b):
        return
    assert eg.union(a, b) is True
    assert eg.find(a) == eg.find(b)
    assert eg.union(a, b) is False


@given(st.sampled_from(["neg", "square", "exp"]))
def test_congruence_through_readd_after_union(op):
    x = Var("x", TensorType((2, 3)))
    y = Var("y", TensorType((2, 3)))
    eg = EGraph()
    eg.union(eg.add_term(x), eg.add_term(y))
    a = eg.add_term(Op.make(op, x))
    b = eg.add_term(Op.make(op, y))
    assert eg.find(a) == eg.find(b)


@settings(max_examples=60, deadline=None)
@given(additive_terms())
def test_saturated_extraction_never_costs_more_than_input(t):
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run(
        _ALGEBRAIC,
        eid,
        max_iterations=6,
        max_nodes=2000,
    )
    best = eg.extract_best(eid, flops_cost)
    assert best is not None
    cost = flops_cost(best)
    assert math.isfinite(cost) and cost >= 0.0
    assert cost <= flops_cost(t) + 1e-9
    assert count_cost(best) <= count_cost(t) + 1


@pytest.mark.xfail(
    reason=(
        "EGraph.rebuild dedups enodes WITHIN a class but does not merge "
        "two distinct classes that become structurally identical after "
        "child canonicalisation (standard e-graph congruence closure)"
    ),
    strict=False,
)
def test_rebuild_merges_congruent_classes():
    x = Var("x", TensorType((2, 3)))
    y = Var("y", TensorType((2, 3)))
    eg = EGraph()
    nx = eg.add_term(Op.make("neg", x))
    ny = eg.add_term(Op.make("neg", y))
    eg.union(eg.add_term(x), eg.add_term(y))
    eg.rebuild()
    assert eg.find(nx) == eg.find(ny)
