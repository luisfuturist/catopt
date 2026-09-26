"""Hash-consed terms: structural interning, content hashing, GC safety.

Phase 2a of plan 0001 — terms are interned value objects, which lets
every memo key on the term object directly instead of ``id(t)`` (the
GC id-reuse hazard that required keepalive lists).
"""

from __future__ import annotations

import gc

from catopt.ir import Const, Op, Param, TensorType, Var

x = Var("x", TensorType((4,)))
w = Param("w", TensorType((4, 4)))


def test_identical_terms_are_the_same_object():
    t1 = Op.make("mul", x, x, a=1)
    t2 = Op.make("mul", x, x, a=1)
    assert t1 is t2
    t3 = Op.make("mul", x, x, a=2)
    assert t3 is not t1 and t3 != t1


def test_nested_structure_interns_whole_dags():
    inner1 = Op.make("mul", x, w)
    t1 = Op.make("add", inner1, x)
    inner2 = Op.make("mul", x, w)
    t2 = Op.make("add", inner2, x)
    assert t1 is t2  # whole DAG shares, children included


def test_content_hash_matches_equality():
    a = Op.make("add", x, Const(1.0))
    b = Op.make("add", x, Const(1.0))
    c = Op.make("add", x, Const(2.0))
    assert hash(a) == hash(b)
    assert {a: "hit"}[b] == "hit"
    assert a != c


def test_attr_order_invariant():
    t1 = Op.make("sdpa", x, x, x, scale=0.5, is_causal=True)
    t2 = Op.make("sdpa", x, x, x, is_causal=True, scale=0.5)
    assert t1 is t2 and t1 == t2


def test_list_vs_tuple_attrs_are_distinct():
    lt = Op.make("split", x, sizes=[3, 5], dim=0, index=0, validate=False)
    tt = Op.make("split", x, sizes=(3, 5), dim=0, index=0, validate=False)
    assert lt != tt
    assert lt is not tt


def test_term_survives_as_memo_key_without_keepalive():
    """A term used as a dict key stays resolvable after GC churn —
    the property keepalive lists used to defend manually."""
    t = Op.make("mul", x, x)
    memo = {t: "shape"}
    for i in range(50):
        _ = Op.make("mul", x, Var(f"tmp{i}", TensorType((4,))))
    gc.collect()
    assert memo[t] == "shape"
    # And a structurally-equal term hits the same memo entry:
    assert memo.get(Op.make("mul", x, x)) == "shape"


def test_interning_is_gc_safe():
    t = Op.make("mul", x, x, flag="gone")
    del t
    gc.collect()
    rebuilt = Op.make("mul", x, x, flag="gone")
    assert rebuilt.attrs["flag"] == "gone"  # fresh or interned, equal
