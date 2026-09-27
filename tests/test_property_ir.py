"""Property tests for term interning / hash-consing (``catopt_core.ir``).

Targets the invariants hand-written example tests only sample: for
*every* small term, ``Op.make`` is idempotent, structural equality
coincides with identity, the content hash agrees with ``__eq__``, attr
insertion order is irrelevant, and the ``attrs`` schema contract is
idempotent.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from catopt_core.attrs import validate_attrs
from catopt_core.ir import Op, TensorType, Var, op_repr
from hypothesis import given

from tests.test_property_strategies import leaves, rebuild, terms

_ATTRS = st.dictionaries(
    st.sampled_from(["a", "b", "c"]),
    st.one_of(st.integers(min_value=-3, max_value=3), st.booleans()),
    max_size=3,
)

_SCHEMA_CASES = st.sampled_from(
    [
        ("transpose", {"dim0": 1, "dim1": 2}),
        ("chunk", {"chunks": 2, "dim": 0, "index": 1}),
        ("concat", {"dim": -1}),
        ("split", {"sizes": (2, 2), "dim": 0, "index": 0}),
        ("getitem", {"index": (0, 1)}),
    ]
)


@given(terms())
def test_make_is_idempotent(t):
    if not isinstance(t, Op):
        return
    assert Op.make(t.op, *t.args, **dict(t.attrs)) is t


@given(terms())
def test_structurally_equal_terms_are_identical(t):
    clone = rebuild(t)
    assert clone is t
    assert clone == t
    assert hash(clone) == hash(t)


@given(terms())
def test_repr_is_stable_under_rebuild(t):
    assert op_repr(rebuild(t)) == op_repr(t)


@given(st.sampled_from(["add", "mul"]), _ATTRS)
def test_attr_insertion_order_is_irrelevant(op, attrs):
    x = Var("x", TensorType((2,)))
    rev = dict(reversed(list(attrs.items())))
    assert Op.make(op, x, x, **attrs) is Op.make(op, x, x, **rev)


@given(st.integers(0, 9), st.integers(10, 19))
def test_distinct_attr_values_are_distinct_terms(v1, v2):
    x = Var("x", TensorType((2,)))
    assert Op.make("mul", x, x, a=v1) != Op.make("mul", x, x, a=v2)


@given(leaves())
def test_leaf_terms_round_trip(t):
    assert rebuild(t) is t


@given(st.lists(st.one_of(st.none(), st.integers(0, 5)), max_size=4))
def test_tensor_type_size_is_product_or_none(dims):
    tt = TensorType(tuple(dims))
    if any(d is None for d in dims):
        assert tt.size is None
    else:
        expected = 1
        for d in dims:
            expected *= d
        assert tt.size == expected


@given(_SCHEMA_CASES)
def test_validate_attrs_is_idempotent(case):
    op, attrs = case
    once = validate_attrs(op, attrs)
    assert once == attrs
    assert validate_attrs(op, once) == once


@given(st.sampled_from(["transpose", "chunk", "concat", "getitem"]))
def test_validate_attrs_rejects_undeclared_positional(op):
    with pytest.raises(ValueError):
        validate_attrs(op, {"arg9": 0})


@given(st.sampled_from([("concat", 1), ("chunk", 1), ("getitem", 1)]))
def test_declared_positional_spelling_is_preserved(case):
    op, i = case
    assert validate_attrs(op, {f"arg{i}": 0}) == {f"arg{i}": 0}


def test_positional_write_wins_over_canonical_name():
    assert validate_attrs("concat", {"dim": 0, "arg1": 5}) == {"dim": 5}
