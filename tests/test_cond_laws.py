"""Tests for the declarative condition DSL (``catopt_core.laws.cond``).

A law's side condition is the one part of a rewrite that used to be
opaque code (``check=lambda bound: ...``), blocking the lemma-store
seam: patterns serialize fine, lambdas do not.  The ``cond`` field on
``Rewrite`` is pure data — a tuple tree interpreted against the same
``bound`` environment ``check`` sees — and ``Rewrite.__post_init__``
folds it into ``check`` so every evaluation site keeps the single
``rule.check`` convention.

Covered surface:

* every interpreter op, branch by branch — the strictness contract
  (unprovable declines), the ``None``-dim wildcards where a pred says
  ``*-compat``, the shape-spec forms (metavar + ``mm-out``);
* the strictness/degradation edge cases: ``_INVALID`` shapes, missing
  metavars, out-of-range dims, malformed nodes (``ValueError``, not a
  decline);
* guard composition — ``cond`` + ``check`` conjoin, cond first;
* canonicalisation — a list tree (``json.loads`` output) is normalised
  to tuples at construction, so the rule stays hashable and equal;
* the serialization round-trip — ``cond_to_data`` → JSON →
  ``cond_from_data`` rebuilds a rule that fires identically (the
  lemma-store seam);
* regression on the migrated tensor laws — the ``_check_*`` aliases
  ARE the same data the rules carry, and the fingerprint covers the
  cond (a data edit must invalidate the synthesis cache).
"""

from __future__ import annotations

import functools
import json

import pytest

from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.laws.cond import (
    as_check,
    compile_guard,
    cond_from_data,
    cond_to_data,
    eval_cond,
)
from catopt_core.laws.tensor import (
    FACTOR_MUL,
    SOFTMAX_FOLD,
    WEIGHT_FACTOR,
    _check_sum_keepdim,
)
from catopt_core.rulecache import ruleset_fingerprint


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _b(**kv):
    return kv


# ---------------------------------------------------------------------------
#  eval_cond — combinators and the malformed-node contract
# ---------------------------------------------------------------------------


def test_eval_cond_bool_leaves_and_combinators():
    b = {"a": _v("a", 2, 3)}
    assert eval_cond(True, b) is True
    assert eval_cond(False, b) is False
    assert eval_cond(("and", True, ("rank", "a", "==", 2)), b)
    assert not eval_cond(("and", True, ("rank", "a", "==", 3)), b)
    assert eval_cond(("or", False, ("rank", "a", "==", 2)), b)
    assert not eval_cond(("or", False, ("rank", "a", "==", 3)), b)
    assert eval_cond(("not", ("rank", "a", "==", 3)), b)


def test_eval_cond_combinator_short_circuits():
    # ``and``/``or`` must not evaluate the rest once decided — the
    # second clause here would raise on a missing attr if reached.
    b: dict = {}
    assert not eval_cond(("and", False, ("attr-eq", "MISSING", 1)), b)
    assert eval_cond(("or", True, ("attr-eq", "MISSING", 1)), b)


def test_eval_cond_malformed_is_valueerror_not_decline():
    b: dict = {}
    with pytest.raises(ValueError):
        eval_cond(("bogus-op", "a"), b)  # unknown operator
    with pytest.raises(ValueError):
        eval_cond((), b)  # empty node
    with pytest.raises(ValueError):
        eval_cond(17, b)  # non-bool, non-node
    with pytest.raises(ValueError):
        eval_cond("shaped", b)  # bare string is not a node
    with pytest.raises(ValueError):
        eval_cond(("not",), b)  # unary op missing its operand


# ---------------------------------------------------------------------------
#  Shape predicates — shaped / concrete / scalar / uniform / ones-but-last
# ---------------------------------------------------------------------------


def test_shaped_and_concrete():
    shaped = _v("a", 2, 3)
    wild = _v("w", None, 3)
    illtyped = Op.make("add", _v("i", 2), _v("j", 3))  # _INVALID
    b = {"a": shaped, "w": wild, "bad": illtyped}
    assert eval_cond(("shaped", "a"), b)
    assert eval_cond(("shaped", "w"), b)  # tuple with None dims: shaped
    assert not eval_cond(("shaped", "bad"), b)  # provably ill-typed
    assert not eval_cond(("shaped", "missing"), b)  # unbound: decline
    assert eval_cond(("concrete", "a"), b)
    assert not eval_cond(("concrete", "w"), b)  # None dim: not concrete
    assert not eval_cond(("concrete", "bad"), b)


def test_scalar_uniform_ones_but_last():
    b = {
        "s": _v("s"),  # () scalar
        "u": _v("u", 1, 1),
        "o": _v("o", 1, 5),  # ones-but-last
        "t": _v("t", 2, 1),
        "c": Const(3.0),  # leaf const: shape ()
    }
    assert eval_cond(("scalar", "s"), b)
    assert eval_cond(("scalar", "c"), b)
    assert not eval_cond(("scalar", "u"), b)
    assert not eval_cond(("scalar", "missing"), b)
    assert eval_cond(("uniform", "u"), b)
    assert eval_cond(("uniform", "s"), b)  # scalar is all-of-nothing
    assert not eval_cond(("uniform", "t"), b)
    assert eval_cond(("ones-but-last", "o"), b)
    assert eval_cond(("ones-but-last", "u"), b)
    assert not eval_cond(("ones-but-last", "t"), b)


# ---------------------------------------------------------------------------
#  Rank / shape comparison predicates
# ---------------------------------------------------------------------------


def test_rank_predicates():
    b = {"a": _v("a", 2, 3), "b": _v("b", 5), "c": _v("c", 9, 9)}
    for cmp_, k, want in (
        ("==", 2, True),
        ("!=", 1, True),
        (">=", 2, True),
        (">", 2, False),
        ("<", 3, True),
        ("<=", 1, False),
    ):
        assert eval_cond(("rank", "a", cmp_, k), b) is want
    assert not eval_cond(("rank", "missing", "==", 2), b)
    assert eval_cond(("rank-eq", "a", "c"), b)  # both rank 2
    assert not eval_cond(("rank-eq", "a", "b"), b)
    assert not eval_cond(("rank-eq", "a", "missing"), b)
    assert not eval_cond(("rank-eq", "missing", "a"), b)


def test_shape_eq_and_compat():
    a = _v("a", 2, 3)
    same = _v("s", 2, 3)
    wild = _v("w", None, 3)  # None dims: wildcard under *-compat only
    wild2 = _v("w2", 2, None)
    diff = _v("d", 2, 4)
    b = {"a": a, "s": same, "w": wild, "w2": wild2, "d": diff}
    assert eval_cond(("shape-eq", "a", "s"), b)
    assert not eval_cond(("shape-eq", "a", "w"), b)  # exact: 2 != None
    assert not eval_cond(("shape-eq", "a", "missing"), b)
    assert not eval_cond(("shape-eq", "missing", "a"), b)
    # None dims are wildcards under shape-compat, either side
    assert eval_cond(("shape-compat", "a", "w"), b)
    assert eval_cond(("shape-compat", "w", "a"), b)
    assert eval_cond(("shape-compat", "w", "w2"), b)  # None vs None/2
    assert not eval_cond(("shape-compat", "a", "d"), b)  # provable


def test_shape_compat_rank_mismatch_declines():
    b = {"a": _v("a", 2, 3), "e": _v("e", 1, 2, 3), "w": _v("w", None)}
    assert not eval_cond(("shape-compat", "a", "e"), b)
    assert not eval_cond(("shape-compat", "a", "missing"), b)
    assert not eval_cond(("shape-compat", "missing", "a"), b)
    # rank-1 wildcard vs rank-2: rank mismatch declines even w/ Nones
    assert not eval_cond(("shape-compat", "w", "a"), b)


def test_dim_predicates():
    a = _v("a", 2, 3)
    wd = _v("w", None, 3)  # dim 0 unknown
    c = _v("c", 4, 3)
    b = {"a": a, "w": wd, "c": c}
    # dim-eq: strict equality, None == None counts as equal
    assert eval_cond(("dim-eq", "a", -1, "c", 1), b)  # 3 == 3
    assert not eval_cond(("dim-eq", "a", 0, "c", 0), b)  # 2 != 4
    assert not eval_cond(("dim-eq", "a", 5, "c", 0), b)  # out of range
    assert not eval_cond(("dim-eq", "a", 0, "c", 9), b)
    assert eval_cond(("dim-eq", "w", 0, "w", 0), b)  # None == None
    assert not eval_cond(("dim-eq", "a", 0, "w", 0), b)  # 2 vs None
    assert not eval_cond(("dim-eq", "missing", 0, "a", 0), b)
    # dim-compat: None on either side is a wildcard
    assert eval_cond(("dim-compat", "a", 0, "w", 0), b)
    assert eval_cond(("dim-compat", "w", 0, "a", 0), b)
    assert eval_cond(("dim-compat", "a", -1, "c", 1), b)
    assert not eval_cond(("dim-compat", "a", 0, "c", 0), b)
    assert not eval_cond(("dim-compat", "missing", 0, "a", 0), b)
    assert not eval_cond(("dim-compat", "a", 0, "missing", 0), b)
    # dim-eq-const
    assert eval_cond(("dim-eq-const", "c", -1, 3), b)
    assert not eval_cond(("dim-eq-const", "c", -1, 4), b)
    assert not eval_cond(("dim-eq-const", "c", 7, 3), b)
    assert not eval_cond(("dim-eq-const", "missing", 0, 3), b)


def test_bcast_into_and_mm_shape_ok():
    small = _v("s", 4)
    big = _v("b", 2, 4)
    bad = _v("z", 3)
    m1, m2 = _v("m", 2, 3), _v("n", 3, 4)
    nm = _v("q", 9)
    b = {"s": small, "b": big, "z": bad, "m": m1, "n": m2, "q": nm}
    assert eval_cond(("bcast-into", "s", "b"), b)  # (4,) → (2,4)
    assert not eval_cond(("bcast-into", "z", "b"), b)  # (3,) can't
    assert not eval_cond(("bcast-into", "missing", "b"), b)
    # "big" must itself be a tuple shape
    assert not eval_cond(("bcast-into", "s", "missing"), b)
    assert eval_cond(("mm-shape-ok", "m", "n"), b)  # (2,3)@(3,4)
    assert not eval_cond(("mm-shape-ok", "m", "q"), b)  # 3 != 9
    assert not eval_cond(("mm-shape-ok", "m", "missing"), b)
    # the mm-out spec composes as a shape spec under every predicate
    assert eval_cond(("rank", ("mm-out", "m", "n"), "==", 2), b)
    assert eval_cond(("dim-eq", ("mm-out", "m", "n"), 1, "n", 1), b)
    assert eval_cond(("shaped", ("mm-out", "m", "n")), b)
    # mm-out of an unshapeable pair resolves to a non-tuple → decline
    assert not eval_cond(("rank", ("mm-out", "m", "q"), "==", 2), b)


def test_shape_spec_nonmetavar_forms_decline():
    b = {"a": _v("a", 2, 3)}
    # a spec that is neither a metavar name nor ("mm-out", a, b) is
    # unknown → every shape predicate declines.
    assert not eval_cond(("shaped", ("not-a-spec", "a")), b)
    assert not eval_cond(("shaped", ("mm-out", "a")), b)  # arity 2
    assert not eval_cond(("shaped", 7), b)  # non-str, non-seq
    # mm-out over an ill-matched pair resolves to a non-tuple shape
    assert not eval_cond(("rank", ("mm-out", "a", "a"), "==", 2), b)


# ---------------------------------------------------------------------------
#  Axis predicates — transpose pairs and the single-axis check
# ---------------------------------------------------------------------------


def test_axes_last2_and_distinct():
    x = _v("x", 2, 3, 4)
    b = {"x": x, "$attr:D0": -2, "$attr:D1": -1}
    assert eval_cond(("axes-last2", "x", "D0", "D1"), b)
    b2 = {"x": x, "$attr:D0": 0, "$attr:D1": -1}
    assert not eval_cond(("axes-last2", "x", "D0", "D1"), b2)
    b3 = {"x": x, "$attr:D0": 2, "$attr:D1": 1}
    assert eval_cond(("axes-last2", "x", "D0", "D1"), b3)  # swapped ok
    # rank < 2 has no last-two pair
    b4 = {"x": _v("y", 5), "$attr:D0": 0, "$attr:D1": 0}
    assert not eval_cond(("axes-last2", "x", "D0", "D1"), b4)
    # unshaped / out-of-range dims decline
    assert not eval_cond(("axes-last2", "missing", "D0", "D1"), b)
    bad = {"x": x, "$attr:D0": 0, "$attr:D1": 99}
    assert not eval_cond(("axes-last2", "x", "D0", "D1"), bad)
    # axes-distinct: a normalized pair with d0 != d1
    assert eval_cond(("axes-distinct", "x", "D0", "D1"), b)
    same = {"x": x, "$attr:D0": 1, "$attr:D1": 1}
    assert not eval_cond(("axes-distinct", "x", "D0", "D1"), same)
    assert not eval_cond(("axes-distinct", "missing", "D0", "D1"), b)


def test_axes_pair_unbound_attrs_mean_bare_transpose():
    # Neither $attr bound = the bare ``t()`` spelling — the implicit
    # last-two swap (same contract as layout's ``_bound_axes``).
    x = _v("x", 2, 3, 4)
    b = {"x": x}
    assert eval_cond(("axes-last2", "x", "D0", "D1"), b)
    assert eval_cond(("axes-distinct", "x", "D0", "D1"), b)
    # one bound + one unbound normalizes through ``_axis_pair``
    half = {"x": x, "$attr:D0": 1}
    assert eval_cond(("axes-last2", "x", "D0", "D1"), half) is False
    assert eval_cond(
        ("axes-distinct", "x", "D0", "D1"),
        {"x": x, "$attr:D0": 0, "$attr:D1": 1},
    )


def test_axes_eq_two_pairs():
    x = _v("x", 2, 3, 4)
    # inner pair (1,2) vs outer pair (2,1): distinct each, same set
    b = {
        "x": x,
        "$attr:I0": 1,
        "$attr:I1": 2,
        "$attr:O0": 2,
        "$attr:O1": 1,
    }
    assert eval_cond(("axes-eq", "x", "I0", "I1", "O0", "O1"), b)
    diff = dict(b, **{"$attr:O1": 0})
    assert not eval_cond(("axes-eq", "x", "I0", "I1", "O0", "O1"), diff)
    # a degenerate pair (d0 == d1) declines
    degen = dict(b, **{"$attr:O1": 2})
    assert not eval_cond(("axes-eq", "x", "I0", "I1", "O0", "O1"), degen)
    degen1 = dict(b, **{"$attr:I1": 1})
    assert not eval_cond(
        ("axes-eq", "x", "I0", "I1", "O0", "O1"), degen1
    )
    # unshapeable term or unnormalizable pair declines
    assert not eval_cond(
        ("axes-eq", "missing", "I0", "I1", "O0", "O1"), b
    )
    badpair = dict(b, **{"$attr:O0": "x"})  # non-int attr
    assert not eval_cond(
        ("axes-eq", "x", "I0", "I1", "O0", "O1"), badpair
    )


def test_axis_attr_normalizes_against_rank():
    q = _v("q", 2, 4, 8)
    b = {"q": q, "$attr:SD": -1}
    assert eval_cond(("axis", "q", "SD", -1), b)
    b2 = {"q": q, "$attr:SD": 2}
    assert eval_cond(("axis", "q", "SD", -1), b2)  # 2 % 3 == -1 % 3
    b3 = {"q": q, "$attr:SD": 0}
    assert not eval_cond(("axis", "q", "SD", -1), b3)
    # non-int attr, rank-0 shape, unshaped term all decline
    assert not eval_cond(
        ("axis", "q", "SD", -1), {"q": q, "$attr:SD": "x"}
    )
    assert not eval_cond(
        ("axis", "q", "SD", -1), {"q": _v("z"), "$attr:SD": 0}
    )
    assert not eval_cond(("axis", "missing", "SD", -1), b)


# ---------------------------------------------------------------------------
#  Term / attr predicates
# ---------------------------------------------------------------------------


def test_op_leaf_const_term_eq():
    op = Op.make("relu", _v("x", 2))
    b = {"t": op, "v": _v("v", 2), "c": Const(1.0)}
    assert eval_cond(("op-in", "t", ("relu", "gelu")), b)
    assert not eval_cond(("op-in", "t", ("tanh",)), b)
    assert not eval_cond(("op-in", "v", ("relu",)), b)  # not an Op
    assert eval_cond(("leaf", "v"), b)
    assert eval_cond(("leaf", "c"), b)
    assert eval_cond(("leaf", "missing"), b)  # unbound is not an Op
    assert not eval_cond(("leaf", "t"), b)
    assert eval_cond(("const", "c"), b)
    assert not eval_cond(("const", "v"), b)
    assert eval_cond(("term-eq", "v", "v"), b)
    assert not eval_cond(("term-eq", "v", "c"), b)
    # two missing metavars resolve to the same absence
    assert eval_cond(("term-eq", "m1", "m2"), b)


def test_const_numeric_predicates():
    b = {
        "n": Const(3), "f": Const(0.5), "s": Const(True),
        "v": _v("v", 2),
    }
    assert eval_cond(("const-num", "n"), b)
    assert eval_cond(("const-num", "f"), b)
    assert eval_cond(("const-num", "s"), b)  # bool is int
    assert not eval_cond(("const-num", "v"), b)  # no .value
    assert not eval_cond(("const-num", "missing"), b)
    assert eval_cond(("const-cmp", "n", ">", 1), b)
    assert eval_cond(("const-cmp", "n", "==", 3), b)
    assert eval_cond(("const-cmp", "n", "!=", 4), b)
    assert eval_cond(("const-cmp", "n", "<=", 3), b)
    assert eval_cond(("const-cmp", "n", ">=", 3), b)
    assert not eval_cond(("const-cmp", "n", "<", 3), b)
    assert not eval_cond(("const-cmp", "v", "<", -1e30), b)


def test_attr_predicates():
    b = {
        "$attr:RK": True,
        "$attr:RD": (-1,),
        "$attr:SD": 2,
        "$attr:NM": "foo",
    }
    assert eval_cond(("attr-is", "RK", True), b)
    assert not eval_cond(("attr-is", "RK", 1), b)  # identity, not ==
    assert not eval_cond(("attr-is", "MISSING", True), b)
    assert eval_cond(("attr-is", "MISSING", None), b)  # absent is None
    assert eval_cond(("attr-eq", "SD", 2), b)
    assert not eval_cond(("attr-eq", "SD", 3), b)
    assert not eval_cond(("attr-eq", "MISSING", 3), b)
    assert eval_cond(("attr-in", "SD", (0, 2, -1)), b)
    assert not eval_cond(("attr-in", "SD", (0, 1)), b)
    assert not eval_cond(("attr-in", "MISSING", (0, 1)), b)
    # attr-type over every declared kind
    assert eval_cond(("attr-type", "SD", "int"), b)
    assert eval_cond(("attr-type", "SD", "number"), b)
    assert eval_cond(("attr-type", "RK", "bool"), b)
    assert eval_cond(("attr-type", "NM", "str"), b)
    assert eval_cond(("attr-type", "RD", "tuple"), b)
    assert not eval_cond(("attr-type", "SD", "float"), b)
    assert not eval_cond(("attr-type", "MISSING", "int"), b)
    fb = {"$attr:F": 0.5}
    assert eval_cond(("attr-type", "F", "float"), fb)
    assert eval_cond(("attr-type", "F", "number"), fb)
    # attr-len: tuple attrs only
    assert eval_cond(("attr-len", "RD", "==", 1), b)
    assert eval_cond(("attr-len", "RD", ">=", 1), b)
    assert not eval_cond(("attr-len", "RD", "==", 2), b)
    assert not eval_cond(("attr-len", "SD", "==", 1), b)  # not a tuple


# ---------------------------------------------------------------------------
#  Guard composition + the check-shaped view
# ---------------------------------------------------------------------------


def test_compile_guard_conjunction():
    cond = ("rank", "a", "==", 2)
    g = compile_guard(cond, None)
    assert g({"a": _v("a", 2, 3)})
    assert not g({"a": _v("a", 2)})
    # cond + check conjoin, cond evaluated first
    calls: list[str] = []

    def ck(bound):
        calls.append("check")
        return bound["a"].typ.shape[0] == 2

    g2 = compile_guard(cond, ck)
    assert g2({"a": _v("a", 2, 3)})
    assert calls == ["check"]
    calls.clear()
    assert not g2({"a": _v("a", 5, 5)})
    assert calls == ["check"]
    calls.clear()
    assert not g2({"a": _v("a", 2)})  # cond fails → check not reached
    assert calls == []


def test_as_check_is_a_check_shaped_partial():
    f = as_check(("scalar", "c"))
    assert isinstance(f, functools.partial)
    assert f({"c": Const(1)})
    assert not f({"c": _v("c", 2)})


def test_rewrite_post_init_folds_and_canonicalizes():
    lhs = Op.make("add", "a", "b")
    # a list tree (what json.loads hands back) canonicalizes to tuples
    r = Rewrite(
        name="t_list", lhs=lhs, rhs="a", cond=["scalar", "a"]
    )
    assert r.cond == ("scalar", "a")
    assert r.check is not None
    assert r.check({"a": Const(1), "b": _v("b", 2)})
    assert not r.check({"a": _v("a", 2), "b": _v("b", 2)})
    hash(r)  # canonical tuple tree keeps the frozen rule hashable
    # a tuple-tree cond is stored verbatim and conjoins with check
    r2 = Rewrite(
        name="t_both",
        lhs=lhs,
        rhs="a",
        cond=("scalar", "a"),
        check=lambda b: b["b"].typ.shape == (2,),
    )
    assert r2.cond == ("scalar", "a")
    assert r2.check({"a": Const(1), "b": _v("b", 2)})
    assert not r2.check({"a": Const(1), "b": _v("b", 3)})
    assert not r2.check({"a": _v("a", 1), "b": _v("b", 2)})
    # cond-free rules are untouched
    r3 = Rewrite(name="t_none", lhs=lhs, rhs="a")
    assert r3.cond is None and r3.check is None


# ---------------------------------------------------------------------------
#  Serialization — the lemma-store seam
# ---------------------------------------------------------------------------


def test_cond_data_roundtrip_is_canonical():
    c = ("and", ("scalar", "a"), ("or", ("rank", "b", ">=", 2), False))
    data = cond_to_data(c)
    assert data == ["and", ["scalar", "a"], ["or", ["rank", "b", ">=", 2], False]]
    assert all(isinstance(x, list) for x in data[1:])
    # JSON round-trip: parse → from_data → identical canonical form
    back = cond_from_data(json.loads(json.dumps(data)))
    assert back == c
    # non-seq leaves pass through; from_data also accepts tuples
    assert cond_to_data(True) is True
    assert cond_from_data(7) == 7
    assert cond_from_data(c) == c
    assert cond_from_data(None) is None


def test_every_shipped_cond_roundtrips_through_json():
    """All 32 migrated rules' conds survive the store wire format."""
    seen = 0
    for rule in all_rules():
        if rule.cond is None:
            continue
        seen += 1
        blob = json.dumps(cond_to_data(rule.cond))
        back = cond_from_data(json.loads(blob))
        assert back == rule.cond, rule.name
        # a rule rebuilt from the record carries the same condition
        rebuilt = Rewrite(
            name=rule.name,
            lhs=rule.lhs,
            rhs=rule.rhs,
            cond=json.loads(blob),  # list tree: canonicalized
        )
        assert rebuilt.cond == rule.cond
    assert seen == 32


def test_rebuilt_rule_fires_identically_in_egraph():
    """The lemma-store claim end-to-end: deserialize a cond, rebuild
    the rule, and the e-graph reaches the same members."""
    blob = json.dumps(cond_to_data(FACTOR_MUL.cond))
    rebuilt = Rewrite(
        name="factor_matmul_rt",
        lhs=FACTOR_MUL.lhs,
        rhs=FACTOR_MUL.rhs,
        cond=json.loads(blob),
    )
    x, w, a, b = _v("x", 5, 4), _p("W", 4, 3), _p("a", 4, 3), _p("b", 4, 3)
    src = Op.make(
        "add", Op.make("matmul", x, Op.make("matmul", w, a)),
        Op.make("matmul", x, Op.make("matmul", w, b)),
    )
    for rule in (FACTOR_MUL, rebuilt):
        eg = EGraph()
        root = eg.add_term(src)
        eg.run([rule], root, max_iterations=4, max_nodes=10_000)
        merged = Op.make(
            "matmul", x, Op.make("matmul", w, Op.make("add", a, b))
        )
        assert eg.find(root) == eg.find(eg.add_term(merged)), rule.name
    # and the decline path survives the same round-trip: the
    # documented rank-1 counterexample refuses under both the shipped
    # and rebuilt rule
    x2, a2, b2 = _v("x2", 16), _p("a2", 16), _p("b2", 16, 16)
    bad = Op.make(
        "add", Op.make("matmul", x2, a2), Op.make("matmul", x2, b2)
    )
    for rule in (FACTOR_MUL, rebuilt):
        eg = EGraph()
        root = eg.add_term(bad)
        eg.run([rule], root, max_iterations=4, max_nodes=10_000)
        merged = Op.make("matmul", x2, Op.make("add", a2, b2))
        assert eg.find(root) != eg.find(eg.add_term(merged)), rule.name


def test_fingerprint_covers_cond_data():
    """A cond edit must invalidate the synthesis cache — the folded
    guard's hook signature is shared across all cond-carrying rules."""
    lhs = Op.make("add", "a", "b")
    r1 = Rewrite(name="r", lhs=lhs, rhs="a", cond=("scalar", "a"))
    r2 = Rewrite(name="r", lhs=lhs, rhs="a", cond=("scalar", "b"))
    r3 = Rewrite(name="r", lhs=lhs, rhs="a")
    fp1 = ruleset_fingerprint([r1])
    fp2 = ruleset_fingerprint([r2])
    fp3 = ruleset_fingerprint([r3])
    assert fp1 != fp2  # different condition → different fingerprint
    assert fp1 != fp3  # guarded vs unguarded
    # same data re-spelled as lists fingerprints identically
    r1l = Rewrite(name="r", lhs=lhs, rhs="a", cond=["scalar", "a"])
    assert ruleset_fingerprint([r1l]) == fp1


# ---------------------------------------------------------------------------
#  Regression — the migrated tensor laws keep their verdicts
# ---------------------------------------------------------------------------


def test_migrated_rules_carry_cond_and_folded_check():
    migrated = [
        r for r in all_rules() if r.cond is not None
    ]
    assert len(migrated) == 32
    for r in migrated:
        assert callable(r.check), r.name
    # gqa_absorb stays check-only; glu_fold conjoins cond + a
    # procedural split-axis parity check
    by_name = {r.name: r for r in all_rules()}
    gqa = by_name["gqa_absorb_repeat"]
    assert gqa.cond is None and gqa.check is not None
    glu = by_name["glu_fold"]
    assert glu.cond is not None and glu.check is not None


def test_check_aliases_are_the_same_data_as_rule_cond():
    """``as_check`` aliases cannot drift from the rule's cond — they
    ARE the cond, viewed through the ``check`` calling convention."""
    assert _check_sum_keepdim.func is eval_cond
    assert _check_sum_keepdim.args == (SOFTMAX_FOLD.cond,)
    # verdict parity on an accept and a decline binding
    ok = {"$attr:RK": True, "$attr:RD": (-1,)}
    bad = {"$attr:RK": False, "$attr:RD": (-1,)}
    for bound in (ok, bad):
        assert _check_sum_keepdim(bound) == eval_cond(
            SOFTMAX_FOLD.cond, bound
        ) == SOFTMAX_FOLD.check(bound)


def test_softmax_fold_cond_verdicts():
    keepdim_int = {"$attr:RK": True, "$attr:RD": -1}
    keepdim_tuple = {"$attr:RK": True, "$attr:RD": (-1,)}
    no_keepdim = {"$attr:RK": False, "$attr:RD": (-1,)}
    missing = {"$attr:RD": (-1,)}
    multi_axis = {"$attr:RK": True, "$attr:RD": (0, 1)}
    non_seq = {"$attr:RK": True, "$attr:RD": "x"}
    for bound in (keepdim_int, keepdim_tuple):
        assert SOFTMAX_FOLD.check(bound)
    for bound in (no_keepdim, missing, multi_axis, non_seq):
        assert not SOFTMAX_FOLD.check(bound)


def test_weight_factor_cond_verdicts():
    w, w2 = _p("W", 4, 4), _p("W2", 4, 4)
    batched = (_p("W", 2, 4, 4), _p("W2", 4, 4))  # min rank >= 2 ok
    vec = (_p("W", 4), _p("W2", 4, 4))  # rank-1 vs rank-2: veto
    scalar = (_p("W"), _p("W2", 4, 4))
    assert WEIGHT_FACTOR.check({"W": w, "W2": w2})
    assert WEIGHT_FACTOR.check({"W": batched[0], "W2": batched[1]})
    assert not WEIGHT_FACTOR.check({"W": vec[0], "W2": vec[1]})
    assert not WEIGHT_FACTOR.check({"W": scalar[0], "W2": scalar[1]})
    assert not WEIGHT_FACTOR.check({"W": w})  # unshaped W2 declines
