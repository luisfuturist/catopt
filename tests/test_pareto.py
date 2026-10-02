"""Tests for the Pareto-frontier cost landscape.

Exercises every line and branch of ``catopt_core.pareto``: vector
construction and validation, dominance ordering, the non-dominated
frontier, and the scalarized argmin.
"""

from __future__ import annotations

import pytest
from catopt_core.pareto import (
    CostVector,
    best,
    dominates,
    pareto_frontier,
)


def test_cost_vector_stores_parallel_fields():
    v = CostVector(("flops", "latency"), (3.0, 1.5))
    assert v.dims == ("flops", "latency")
    assert v.values == (3.0, 1.5)


def test_cost_vector_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        CostVector(("a", "b"), (1.0,))


def test_from_mapping_defaults_to_key_order():
    v = CostVector.from_mapping({"a": 1.0, "b": 2.0})
    assert v == CostVector(("a", "b"), (1.0, 2.0))


def test_from_mapping_honours_explicit_dims():
    v = CostVector.from_mapping({"a": 1.0, "b": 2.0}, dims=("b", "a"))
    assert v == CostVector(("b", "a"), (2.0, 1.0))


def test_from_mapping_missing_dim_raises_key_error():
    with pytest.raises(KeyError):
        CostVector.from_mapping({"a": 1.0}, dims=("a", "b"))


def test_getitem_and_as_dict():
    v = CostVector(("a", "b"), (1.0, 2.0))
    assert v["b"] == 2.0
    assert v.as_dict() == {"a": 1.0, "b": 2.0}


def test_getitem_missing_dim_raises_key_error():
    v = CostVector(("a",), (1.0,))
    with pytest.raises(KeyError):
        v["missing"]


def test_dominates_strictly_better():
    a = CostVector(("x", "y"), (1.0, 2.0))
    b = CostVector(("x", "y"), (2.0, 3.0))
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_equal_is_not_domination():
    a = CostVector(("x", "y"), (1.0, 1.0))
    b = CostVector(("x", "y"), (1.0, 1.0))
    assert not dominates(a, b)


def test_dominates_incomparable_is_false():
    a = CostVector(("x", "y"), (1.0, 2.0))
    b = CostVector(("x", "y"), (2.0, 1.0))
    assert not dominates(a, b)


def test_dominates_requires_matching_dims():
    a = CostVector(("x",), (1.0,))
    b = CostVector(("y",), (2.0,))
    with pytest.raises(ValueError, match="different dims"):
        dominates(a, b)


def _items():
    return [
        ("p0", CostVector(("x", "y"), (1.0, 3.0))),
        ("p1", CostVector(("x", "y"), (2.0, 2.0))),
        ("p2", CostVector(("x", "y"), (3.0, 1.0))),
        ("p3", CostVector(("x", "y"), (2.0, 2.0))),
        ("p4", CostVector(("x", "y"), (3.0, 3.0))),
    ]


def test_pareto_frontier_keeps_non_dominated_in_order():
    items = _items()
    front = pareto_frontier(items, key=lambda it: it[1])
    assert front == items[:4]


def test_pareto_frontier_empty():
    assert pareto_frontier([], key=lambda it: it[1]) == []


def test_best_unweighted_sums_every_dim():
    items = [
        ("a", CostVector(("x", "y"), (2.0, 2.0))),
        ("b", CostVector(("x", "y"), (1.0, 1.0))),
    ]
    assert best(items, key=lambda it: it[1])[0] == "b"


def test_best_weighted_picks_weighted_minimum():
    items = [
        ("cheap", CostVector(("flops", "latency"), (1.0, 10.0))),
        ("fast", CostVector(("flops", "latency"), (10.0, 1.0))),
    ]
    picked = best(
        items,
        key=lambda it: it[1],
        weights={"latency": 1.0},
    )
    assert picked[0] == "fast"


def test_best_ties_break_toward_input_order():
    items = [
        ("first", CostVector(("x",), (1.0,))),
        ("second", CostVector(("x",), (1.0,))),
    ]
    assert best(items, key=lambda it: it[1])[0] == "first"


def test_best_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        best([], key=lambda it: it[1])
