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
from catopt_core.laws.serialize import (
    law_from_data,
    law_to_data,
    missing_hooks,
)
from catopt_core.laws.tensor import (
    CHUNK_SINGLE,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    GQA_ABSORB,
    LINEAR_CHANNEL_SCALE,
    LINEAR_CHANNEL_TO_ROW_SCALE,
    LINEAR_ROW_SCALE,
    LINEAR_ROW_SCALE_REV,
    MUL_RESHAPE_INERT_L,
    MUL_UNSQ_PAD_L,
    MUL_UNSQ_PAD_R,
    SDPA_FOLD_DIV_NOMASK,
    SDPA_FOLD_NOMASK,
    SOFTMAX_FOLD,
    SOFTSIGN_FOLD,
    SUB_UNSQ_PAD_L,
    TRANSPOSE_NOOP,
    WEIGHT_DISTRIBUTE,
    WEIGHT_DISTRIBUTE_LINEAR,
    WEIGHT_FACTOR,
    WEIGHT_FACTOR_LINEAR,
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
#  View-output shape specs — select / slice / chunk / transpose
# ---------------------------------------------------------------------------


def test_select_out_spec():
    """``select-out`` drops the bound axis (mirrors typing.select)."""
    b = {
        "u": _v("u", 2, 3, 4),
        "$attr:D": 1,
        "s": _v("s", 2, 4),
    }
    assert eval_cond(("shape-eq", ("select-out", "u", "D"), "s"), b)
    assert not eval_cond(
        ("shape-eq", ("select-out", "u", "D"), "s"),
        dict(b, s=_v("s", 2, 3)),
    )
    # a negative axis normalizes mod rank
    neg = {"u": _v("u", 2, 3), "$attr:D": -1, "s": _v("s", 2)}
    assert eval_cond(("shape-eq", ("select-out", "u", "D"), "s"), neg)
    # rank-0 / unbound axis / non-int axis / bad arity decline
    assert not eval_cond(
        ("shaped", ("select-out", "z", "D")),
        {"z": _v("z"), "$attr:D": 0},
    )
    assert not eval_cond(
        ("shaped", ("select-out", "u", "D")),
        {"u": _v("u", 2, 3)},
    )
    assert not eval_cond(
        ("shaped", ("select-out", "u", "D")),
        {"u": _v("u", 2, 3), "$attr:D": "x"},
    )
    # bad arity (one arg) declines
    assert not eval_cond(
        ("shaped", ("select-out", "u")),
        {"u": _v("u", 2, 3), "$attr:D": 0},
    )


def test_slice_out_spec():
    """``slice-out`` replaces the axis with the sliced extent."""
    b = {
        "u": _v("u", 4, 6),
        "$attr:D": -1,
        "$attr:S": 0,
        "$attr:E": 2,
        "$attr:ST": 1,
        "s": _v("s", 4, 2),
    }
    assert eval_cond(("shape-eq", ("slice-out", "u", "D", "S", "E", "ST"), "s"), b)
    # None start/end/step take the torch defaults 0 / dim / 1
    full = {"u": _v("u", 4, 6), "$attr:D": 0, "s": _v("s", 4, 6)}
    assert eval_cond(
        ("shape-eq", ("slice-out", "u", "D", None, None, None), "s"), full
    )
    # a stepped slice: ceil(extent / step)
    stepped = {
        "u": _v("u", 8),
        "$attr:D": 0,
        "$attr:S": 0,
        "$attr:E": 8,
        "$attr:ST": 2,
        "s": _v("s", 4),
    }
    assert eval_cond(
        ("shape-eq", ("slice-out", "u", "D", "S", "E", "ST"), "s"), stepped
    )
    # a non-positive step cannot rewrite the extent — the axis keeps
    # its bound dim (the mirror-decline arm, not an error)
    nostep = {
        "u": _v("u", 4, 6),
        "$attr:D": -1,
        "$attr:S": 0,
        "$attr:E": 2,
        "$attr:ST": 0,
        "s": _v("s", 4, 6),
    }
    assert eval_cond(
        ("shape-eq", ("slice-out", "u", "D", "S", "E", "ST"), "s"), nostep
    )
    # unbound axis / rank-0 / bad arity decline
    assert not eval_cond(
        ("shaped", ("slice-out", "u", "D", None, None, None)),
        {"u": _v("u", 4)},
    )
    assert not eval_cond(
        ("shaped", ("slice-out", "z", "D", None, None, None)),
        {"z": _v("z"), "$attr:D": 0},
    )
    assert not eval_cond(("shaped", ("slice-out", "u", "D")), b)


def test_chunk_out_spec():
    """``chunk-out`` divides the axis by the chunk count."""
    b = {"u": _v("u", 4, 6), "$attr:C": 2, "$attr:D": -1, "s": _v("s", 4, 3)}
    assert eval_cond(("shape-eq", ("chunk-out", "u", "C", "D"), "s"), b)
    assert not eval_cond(
        ("shape-eq", ("chunk-out", "u", "C", "D"), "s"),
        dict(b, **{"$attr:C": 3}),
    )
    # non-positive chunk count / rank-0 / unbound axis / bad arity decline
    assert not eval_cond(
        ("shaped", ("chunk-out", "u", "C", "D")),
        {"u": _v("u", 4, 6), "$attr:C": 0, "$attr:D": -1},
    )
    assert not eval_cond(
        ("shaped", ("chunk-out", "z", "C", "D")),
        {"z": _v("z"), "$attr:C": 2, "$attr:D": 0},
    )
    # bad arity (two args) declines
    assert not eval_cond(
        ("shaped", ("chunk-out", "u", "C")),
        {"u": _v("u", 4, 6), "$attr:C": 2, "$attr:D": -1},
    )
    # a symbolic (non-int) dim leaves it unchanged — the spec only
    # rewrites a concrete int extent
    sym = {
        "u": Var("u", TensorType((4, None))),
        "$attr:C": 2,
        "$attr:D": -1,
        "s": Var("s", TensorType((4, None))),
    }
    assert eval_cond(
        ("shape-eq", ("chunk-out", "u", "C", "D"), "s"), sym
    )


def test_transpose_out_spec():
    """``transpose-out`` swaps the bound axes (defaults -2 / -1)."""
    b = {
        "u": _v("u", 2, 3, 4),
        "$attr:D0": 0,
        "$attr:D1": -1,
        "s": _v("s", 4, 3, 2),
    }
    assert eval_cond(("shape-eq", ("transpose-out", "u", "D0", "D1"), "s"), b)
    # None args take the bare last-two swap (d0=-2, d1=-1)
    last2 = {"u": _v("u", 2, 3, 4), "s": _v("s", 2, 4, 3)}
    assert eval_cond(
        ("shape-eq", ("transpose-out", "u", None, None), "s"), last2
    )
    # rank-0 / unbound axis / bad arity decline
    assert not eval_cond(
        ("shaped", ("transpose-out", "z", "D0", "D1")),
        {"z": _v("z"), "$attr:D0": 0, "$attr:D1": 1},
    )
    # bad arity (two args) declines
    assert not eval_cond(
        ("shaped", ("transpose-out", "u", "D0")),
        {"u": _v("u", 2, 3), "$attr:D0": 0, "$attr:D1": 1},
    )


# ---------------------------------------------------------------------------
#  Broadcast-alignment predicates — the view-commute guard vocabulary
# ---------------------------------------------------------------------------


def test_axis_align_eq():
    """``axis-align-eq``: the bound axis is the same grid axis."""
    # rank-2 vs rank-1: axis -1 right-aligns identically
    assert eval_cond(
        ("axis-align-eq", "a", "b", "D"),
        {"a": _v("a", 2, 3), "b": _v("b", 3), "$attr:D": -1},
    )
    # axis 0 of a rank-1 operand maps past a rank-2 partner's axis 0
    assert not eval_cond(
        ("axis-align-eq", "a", "b", "D"),
        {"a": _v("a", 4), "b": _v("b", 3, 4), "$attr:D": 0},
    )
    # equal ranks always align
    assert eval_cond(
        ("axis-align-eq", "a", "b", "D"),
        {"a": _v("a", 2, 3), "b": _v("b", 5, 6), "$attr:D": 1},
    )
    # unknown side / rank-0 / unbound axis decline
    assert not eval_cond(
        ("axis-align-eq", "a", "z", "D"),
        {"a": _v("a", 2, 3), "$attr:D": 0},
    )
    assert not eval_cond(
        ("axis-align-eq", "a", "b", "D"),
        {"a": _v("a"), "b": _v("b", 3), "$attr:D": 0},
    )
    assert not eval_cond(
        ("axis-align-eq", "a", "b", "D"),
        {"a": _v("a", 2, 3), "b": _v("b", 4)},
    )


def test_bcast_dim_inv():
    """``bcast-dim-inv``: the partner is constant along the axis."""
    # V's rank is too low to reach U's axis 0 -> invariant
    assert eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", 8), "u": _v("u", 4, 8), "$attr:D": 0},
    )
    # V's aligned extent at U's axis -1 is 2 (>1) -> varies
    assert not eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", 2), "u": _v("u", 2, 2), "$attr:D": -1},
    )
    # extent exactly 1 -> invariant
    assert eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", 1), "u": _v("u", 2, 2), "$attr:D": -1},
    )
    # a scalar V is invariant everywhere
    assert eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v"), "u": _v("u", 2, 3), "$attr:D": -1},
    )
    # rank-0 U / unknown V / unbound axis decline
    assert not eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", 3), "u": _v("u"), "$attr:D": 0},
    )
    assert not eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"u": _v("u", 2, 3), "$attr:D": 0},
    )
    assert not eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", 3), "u": _v("u", 2, 3)},
    )
    # a None extent at the mapped position is not provably 1 -> decline
    assert not eval_cond(
        ("bcast-dim-inv", "v", "u", "D"),
        {"v": _v("v", None), "u": _v("u", 2, 3), "$attr:D": -1},
    )


def test_view_commute_guard_composes_and_roundtrips():
    """The wrap-form guard the auto-cond bank mints for an index view."""
    cond = (
        "and",
        ("bcast-dim-inv", "V", "U", "A_dim"),
        (
            "bcast-eq",
            ("select-out", "U", "A_dim"),
            "V",
            ("select-out", ("bcast", "U", "V"), "A_dim"),
            ("select-out", ("bcast", "U", "V"), "A_dim"),
        ),
    )
    # accepts: scalar V (invariant), U/V broadcastable
    ok = {
        "U": _v("U", 2, 3),
        "V": _v("V"),
        "$attr:A_dim": -1,
        "$attr:A_index": 0,
    }
    assert eval_cond(cond, ok)
    # declines: V varies along the selected axis
    bad = {
        "U": _v("U", 2, 2),
        "V": _v("V", 2),
        "$attr:A_dim": -1,
        "$attr:A_index": 0,
    }
    assert not eval_cond(cond, bad)
    # the guard is pure data — the lemma-store round-trip
    data = cond_to_data(cond)
    assert cond_from_data(json.loads(json.dumps(data))) == cond


def test_shaped_bcast_is_the_broadcastable_predicate():
    """``shaped(("bcast", A, B))`` composes into ``bcast-ok``."""
    ok = {"a": _v("a", 4), "b": _v("b", 3, 4)}
    fail = {"a": _v("a", 4), "b": _v("b", 3, 3)}
    assert eval_cond(("shaped", ("bcast", "a", "b")), ok)
    assert not eval_cond(("shaped", ("bcast", "a", "b")), fail)
    # an unbound side is a wildcard — ``_broadcast`` returns the known
    # shape, so the composite still resolves (and declines only on a
    # *provable* mismatch)
    assert eval_cond(("shaped", ("bcast", "a", "z")), ok)


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
    """All 45 cond-carrying rules' conds survive the store wire format."""
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
    assert seen == 45


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
    assert len(migrated) == 45
    for r in migrated:
        assert callable(r.check), r.name
    # every shipped guard is data now — the once-procedural laws all
    # carry cond (glu_fold's parity, the rms pair's trailing block,
    # gqa_absorb's repeat chains)
    by_name = {r.name: r for r in all_rules()}
    for name in (
        "glu_fold",
        "rms_norm_fold",
        "rms_norm_fold_nogain",
        "gqa_absorb_repeat",
    ):
        assert by_name[name].cond is not None, name


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


def test_mm_addends_rank_equal_requires_shape_eq():
    """Equal rank does not align the addends: two rank-1 addends that
    broadcast (``a=(1,)``, ``b=(4,)``) mint a RHS whose ``matmul(W, a)``
    / ``matmul(W, b)`` cannot both contract — the derivable-gate audit's
    rhs-err fix (``shape-eq`` on the rank-equal branch)."""
    good = {"W": _v("W", 4), "a": _v("a", 4), "b": _v("b", 4)}
    bad = {"W": _v("W", 4), "a": _v("a", 1), "b": _v("b", 4)}
    assert DISTRIBUTE_MUL.check(good)
    assert not DISTRIBUTE_MUL.check(bad)
    assert not FACTOR_MUL.check(bad)
    # rank>=2 with the SAME contraction axis (-2) stays covered by the
    # second branch (a rank-2 + rank-3 batch broadcast).
    batched = {
        "W": _p("W", 5, 4),
        "a": _v("a", 4, 3),
        "b": _v("b", 2, 4, 3),
    }
    assert DISTRIBUTE_MUL.check(batched)


def test_mm_addends_rank_ge2_requires_contraction_axis():
    """The rank>=2 branch must align the CONTRACTION axis (-2), not
    merely the rank: ``a=(3,5)``, ``b=(1,5)`` broadcasts the
    contraction axis (1 -> 3) yet ``matmul(W, a)``/``matmul(W, b)``
    cannot contract with one ``W`` — a measured rhs-err the old
    leading-axis-padding argument accepted.  The minimal closure
    ``dim-eq(a, -2, b, -2)`` declines it and keeps every sound
    rank>=2 mixed-rank binding."""
    from catopt_core.egraph.terms import _term_instantiate
    from catopt_discovery import oracle as vo

    bad = {"W": _p("W", 4, 3), "a": _v("a", 3, 5), "b": _v("b", 1, 5)}
    assert not DISTRIBUTE_MUL.check(bad)
    assert not FACTOR_MUL.check(bad)
    # the same binding is a measured rhs-err, not an equal site.
    lhs = _term_instantiate(DISTRIBUTE_MUL.lhs, bad)
    rhs = _term_instantiate(DISTRIBUTE_MUL.rhs, bad)
    assert vo.eval_instance(lhs, rhs)[0] == "rhs-err"
    # a sound rank>=2 mixed-rank pair (shared -2) still clears.
    good = {
        "W": _p("W", 5, 4),
        "a": _v("a", 4, 3),
        "b": _v("b", 2, 4, 3),
    }
    assert DISTRIBUTE_MUL.check(good)
    lhs = _term_instantiate(DISTRIBUTE_MUL.lhs, good)
    rhs = _term_instantiate(DISTRIBUTE_MUL.rhs, good)
    assert vo.eval_instance(lhs, rhs)[0] == "equal"


def test_mm_weights_rank_ge2_uses_the_spelling_axis():
    """The summed weights' contraction axis differs by spelling:
    ``matmul(x, W)`` contracts ``W[-2]``, ``linear(x, W)`` (= ``x @
    W.T``) contracts ``W[-1]``.  The matmul guard declines a
    ``W=(2,3)``/``W2=(1,3)`` pair; the linear guard declines
    ``W=(2,3)``/``W2=(2,1)`` — and each accepts the other's sound
    case, so a single shared clause could not be correct."""
    w_mm_bad = {"x": _v("x", 4, 2), "W": _v("W", 2, 3), "W2": _v("W2", 1, 3)}
    assert not WEIGHT_FACTOR.check(w_mm_bad)
    assert not WEIGHT_DISTRIBUTE.check(w_mm_bad)
    w_lin_bad = {
        "x": _v("x", 4, 3),
        "W": _v("W", 2, 3),
        "W2": _v("W2", 2, 1),
    }
    assert not WEIGHT_FACTOR_LINEAR.check(w_lin_bad)
    assert not WEIGHT_DISTRIBUTE_LINEAR.check(w_lin_bad)
    # each spelling keeps its own sound rank>=2 mixed-extent case.
    w_mm_ok = {"x": _v("x", 5, 4), "W": _v("W", 4, 3), "W2": _v("W2", 2, 4, 3)}
    assert WEIGHT_DISTRIBUTE.check(w_mm_ok)
    w_lin_ok = {"x": _v("x", 2, 5), "W": _v("W", 3, 5), "W2": _v("W2", 1, 5)}
    assert WEIGHT_DISTRIBUTE_LINEAR.check(w_lin_ok)


def test_mm_weights_rank_equal_requires_shape_eq():
    """The summed weights: two rank-1 weights of different extent
    broadcast but the minted ``linear``/``matmul`` RHS cannot denote —
    the audit's rhs-err fix, on the shared weight guard."""
    good = {"x": _v("x", 4), "W": _v("W", 4), "W2": _v("W2", 4)}
    bad = {"x": _v("x", 4), "W": _v("W", 1), "W2": _v("W2", 4)}
    assert WEIGHT_FACTOR.check(good)
    for rule in (
        WEIGHT_FACTOR,
        WEIGHT_DISTRIBUTE,
        WEIGHT_FACTOR_LINEAR,
        WEIGHT_DISTRIBUTE_LINEAR,
    ):
        assert not rule.check(bad), rule.name


def test_linear_channel_scale_requires_linear_well_typed():
    """The channel fold is a value identity only where ``F.linear(x,
    W)`` denotes: a scalar ``x`` or a rank-1 ``W`` disagreeing with
    ``x``'s in-feature axis mints an ill-typed RHS — the audit's
    rhs-err fix."""
    good = {"x": _v("x", 2, 3), "c": _v("c", 3), "W": _p("W", 4, 3)}
    assert LINEAR_CHANNEL_SCALE.check(good)
    scalar_x = {"x": _v("x"), "c": _v("c", 4), "W": _p("W", 4)}
    assert not LINEAR_CHANNEL_SCALE.check(scalar_x)
    mismatch = {"x": _v("x", 1), "c": _v("c", 4), "W": _p("W", 4)}
    assert not LINEAR_CHANNEL_SCALE.check(mismatch)


def test_linear_row_scale_requires_r_broadcasts_into_output():
    """The row scale must broadcast INTO ``linear(x, W)``'s output
    without adding axes: a rank-1 ``r=(1,)`` against a scalar output
    grows ``mul(linear(x,W), r)`` by an axis — the audit's unequal
    fix."""
    good = {
        "x": _v("x", 2, 2, 3),
        "r": _v("r", 2, 2, 1),
        "W": _p("W", 4, 3),
    }
    assert LINEAR_ROW_SCALE.check(good)
    assert LINEAR_ROW_SCALE_REV.check(good)
    # r=(1,) against a scalar output: mul(out, r) gains an axis.
    bad = {"x": _v("x", 1), "r": _v("r", 1), "W": _p("W", 1)}
    assert not LINEAR_ROW_SCALE.check(bad)
    assert not LINEAR_ROW_SCALE_REV.check(bad)


def test_gqa_absorb_repeat_requires_rank4_operands():
    """``sdpa`` + ``enable_gqa`` repeats along the head axis at ``-3``;
    the pattern's ``transpose(1, 2)`` only lands the heads there when
    the leaves are rank-4 ``(b, t, h, d)``.  A rank-3 spelling —
    ``q=(2,6,4)``, ``k=v=(2,3,4)`` with the consistent
    ``unsq(-2) → expand (2,3,2,4) → reshape (2,6,4)`` chain —
    satisfies every repeat clause but its minted ``enable_gqa`` RHS
    cannot contract: a measured ``rhs-err``
    (``Expected size for first two dimensions of batch2 tensor to be:
    [2, 6] but got: [2, 3]``).  The ``rank == 4`` clauses decline it.
    """
    from catopt_core.egraph.terms import _term_instantiate
    from catopt_discovery import oracle as vo

    # the measured counterexample: all repeat clauses pass, rank-3
    # leaves make the enable_gqa RHS ill-typed.
    bad = {
        "q": _v("q", 2, 6, 4),
        "k": _v("k", 2, 3, 4),
        "v": _v("v", 2, 3, 4),
        "$attr:UDk": -2,
        "$attr:ESk": (2, 3, 2, 4),
        "$attr:RSk": (2, 6, 4),
        "$attr:UDv": -2,
        "$attr:ESv": (2, 3, 2, 4),
        "$attr:RSv": (2, 6, 4),
        "$attr:D": 1e-5,
        "$attr:C": False,
    }
    assert not GQA_ABSORB.check(bad)
    # …and it was a real rhs-err, not a vacuous decline: the
    # instantiated RHS genuinely cannot denote.
    lhs = _term_instantiate(GQA_ABSORB.lhs, bad)
    rhs = _term_instantiate(GQA_ABSORB.rhs, bad)
    assert vo.eval_instance(lhs, rhs)[0] == "rhs-err"
    # the rank-4 (b, t, h, d) corner — the binding the chained
    # enumerator mints — still clears and evaluates equal.
    good = {
        "q": _v("q", 2, 3, 4, 4),
        "k": _v("k", 2, 3, 2, 4),
        "v": _v("v", 2, 3, 2, 4),
        "$attr:UDk": -2,
        "$attr:ESk": (2, 3, 2, 2, 4),
        "$attr:RSk": (2, 3, 4, 4),
        "$attr:UDv": -2,
        "$attr:ESv": (2, 3, 2, 2, 4),
        "$attr:RSv": (2, 3, 4, 4),
        "$attr:D": 1e-5,
        "$attr:C": False,
    }
    assert GQA_ABSORB.check(good)
    lhs = _term_instantiate(GQA_ABSORB.lhs, good)
    rhs = _term_instantiate(GQA_ABSORB.rhs, good)
    assert vo.eval_instance(lhs, rhs)[0] == "equal"


def test_linear_scale_guards_cap_the_weight_arity():
    """``F.linear`` refuses a rank>=3 weight (``t() expects a tensor
    with <= 2 dimensions``), but ``mm-shape-ok`` / ``mm-out`` over
    ``W.T`` resolve a rank>=3 ``W.T`` happily — so both linear-scale
    guards conjoin the ``rank(W) <= 2`` ceiling.  Measured: the only
    site it declines is a rank-3-``W`` ``both-err``; every equal site
    keeps."""
    ch3 = {"x": _v("x", 4, 3), "c": _v("c", 3), "W": _p("W", 2, 4, 3)}
    assert not LINEAR_CHANNEL_SCALE.check(ch3)
    row3 = {"x": _v("x", 2, 2, 3), "r": _v("r", 2, 2, 1), "W": _p("W", 2, 3, 4)}
    assert not LINEAR_ROW_SCALE.check(row3)
    assert not LINEAR_ROW_SCALE_REV.check(row3)
    # rank-2 weights still clear.
    assert LINEAR_CHANNEL_SCALE.check(
        {"x": _v("x", 4, 3), "c": _v("c", 3), "W": _p("W", 4, 3)}
    )
    assert LINEAR_ROW_SCALE.check(
        {"x": _v("x", 2, 2, 3), "r": _v("r", 2, 2, 1), "W": _p("W", 4, 3)}
    )


# ---------------------------------------------------------------------------
#  View-guard predicates — the cond forms for the view-oracle's
#  mechanical separators (see project/retros/cond-dsl-view-guards.md)
# ---------------------------------------------------------------------------


def test_unsq_out_and_getitem_out_specs():
    u = _v("u", 2, 3)
    b = {"u": u, "$attr:D": 0}
    # ("unsq-out", T, K) inserts a 1 at K mod (rank+1)
    b2 = {"u": u, "$attr:D": -3}
    assert _shape_helper(("unsq-out", "u", "D"), b) == (1, 2, 3)
    assert _shape_helper(("unsq-out", "u", "D"), b2) == (1, 2, 3)
    assert _shape_helper(("unsq-out", "u", "D"), {"u": u, "$attr:D": 1}) == (2, 1, 3)
    # unshaped operand / non-int attr → None → declines
    assert _shape_helper(("unsq-out", "u", "D"), {"u": u}) is None
    assert _shape_helper(("unsq-out", "missing", "D"), b) is None
    assert _shape_helper(("unsq-out", "u", "D", "extra"), b) is None
    # ("getitem-out", T) drops dim 0
    assert _shape_helper(("getitem-out", "u"), b) == (3,)
    assert _shape_helper(("getitem-out", "u"), {"u": _v("s")}) is None
    assert _shape_helper(("getitem-out", "u", "x"), b) is None
    assert _shape_helper(("getitem-out", "missing"), b) is None


def test_bcast_and_reshape_out_specs():
    u, v = _v("u", 2, 3), _v("v", 1, 3)
    b = {"u": u, "v": v, "$attr:S": (2, 3), "$attr:NEG": (-1, 3)}
    # ("bcast", T, T) broadcasts two specs; an unresolvable side is a
    # wildcard (None), matching _broadcast's convention.
    assert _shape_helper(("bcast", "u", "v"), b) == (2, 3)
    assert _shape_helper(("bcast", "missing", "v"), b) == (1, 3)
    assert _shape_helper(("bcast", "u"), b) is None  # arity
    # ("reshape-out", T, NAME) resolves -1 and validates numel.
    assert _shape_helper(("reshape-out", "u", "S"), b) == (2, 3)
    assert _shape_helper(("reshape-out", "u", "NEG"), b) == (2, 3)
    bad = {"u": u, "$attr:S": (2, 4)}  # numel mismatch → None
    assert _shape_helper(("reshape-out", "u", "S"), bad) is None
    assert _shape_helper(("reshape-out", "missing", "S"), b) is None
    bad2 = {"u": u, "$attr:S": 4}  # non-tuple target
    assert _shape_helper(("reshape-out", "u", "S"), bad2) is None
    assert _shape_helper(("reshape-out", "u", "S", "x"), b) is None
    # -1 that cannot be resolved leaves a None slot in the shape —
    # the same "unknown dim" posture the typing rule reports.
    odd = {"o": _v("o", 3), "$attr:S": (-1, 2)}
    assert _shape_helper(("reshape-out", "o", "S"), odd) == (None, 2)
    wild = {"w": _v("w", 3), "$attr:S": (-1, 4)}
    assert _shape_helper(("reshape-out", "w", "S"), wild) == (None, 4)


def _shape_helper(ref, bound):
    from catopt_core.laws.cond import _shape

    return _shape(bound, ref)


def test_dim_eq_attr_declines():
    # strictness: unshaped terms and non-int attr indices decline.
    b = {"u": _v("u", 2, 3), "$attr:D": 0}
    assert not eval_cond(("dim-eq-attr", "u", "D", "missing", "D"), b)
    assert not eval_cond(("dim-eq-attr", "u", "MISSING", "u", "D"), b)


def test_dim_mod():
    # ("dim-mod", T, K, m, r): the attr-named dim satisfies a modular
    # congruence — glu_fold's split-axis parity is ("dim-mod","u","D",2,0).
    u = _v("u", 4, 8)
    b = {"u": u, "$attr:D": -1}
    assert eval_cond(("dim-mod", "u", "D", 2, 0), b)
    assert eval_cond(("dim-mod", "u", "D", 4, 0), b)  # 8 % 4 == 0
    assert not eval_cond(("dim-mod", "u", "D", 3, 0), b)  # 8 % 3 != 0
    assert not eval_cond(("dim-mod", "u", "D", 2, 1), b)  # 8 % 2 != 1
    # an odd axis declines — the glu veto
    odd = {"u": _v("o", 4, 3), "$attr:D": -1}
    assert not eval_cond(("dim-mod", "u", "D", 2, 0), odd)
    # strictness: unshaped T, non-int axis, out-of-range axis,
    # non-int or zero modulus, and a None dim ON the axis all decline
    assert not eval_cond(("dim-mod", "missing", "D", 2, 0), b)
    assert not eval_cond(("dim-mod", "u", "MISSING", 2, 0), b)
    assert not eval_cond(
        ("dim-mod", "u", "D", 2, 0), {"u": u, "$attr:D": (-1,)}
    )
    assert not eval_cond(
        ("dim-mod", "u", "D", 2, 0), {"u": u, "$attr:D": 5}
    )
    assert not eval_cond(("dim-mod", "u", "D", 0, 0), b)  # mod 0
    assert not eval_cond(("dim-mod", "u", "D", "x", 0), b)
    wild = {"u": _v("w", 4, None), "$attr:D": -1}
    assert not eval_cond(("dim-mod", "u", "D", 2, 0), wild)
    # ...but a None dim OFF the axis is fine
    off = {"u": _v("w", None, 8), "$attr:D": -1}
    assert eval_cond(("dim-mod", "u", "D", 2, 0), off)
    # scalar u has no axis to name
    assert not eval_cond(
        ("dim-mod", "u", "D", 2, 0), {"u": _v("s"), "$attr:D": 0}
    )


def test_attr_eq_attr():
    b = {"$attr:A": (1, 2), "$attr:B": (1, 2), "$attr:C": (1, 3)}
    assert eval_cond(("attr-eq-attr", "A", "B"), b)
    assert not eval_cond(("attr-eq-attr", "A", "C"), b)
    # an unbound side declines, it does not compare None == None
    assert not eval_cond(("attr-eq-attr", "A", "MISSING"), b)
    assert not eval_cond(("attr-eq-attr", "NOPE", "MISSING"), b)
    # a legitimately None-bound attr still counts as bound
    assert eval_cond(
        ("attr-eq-attr", "A", "B"), {"$attr:A": None, "$attr:B": None}
    )


def test_attr_cmp_dim():
    u = _v("u", 2, 8)
    b = {"u": u, "$attr:E": 8, "$attr:D": -1}
    assert eval_cond(("attr-cmp-dim", "E", ">=", "u", "D"), b)
    assert not eval_cond(("attr-cmp-dim", "E", "==", "u", "D"), {"u": u, "$attr:E": 3, "$attr:D": -1})
    assert not eval_cond(("attr-cmp-dim", "E", ">=", "missing", "D"), b)
    assert not eval_cond(("attr-cmp-dim", "E", ">=", "u", "MISSING"), b)
    assert not eval_cond(("attr-cmp-dim", "MISSING", ">=", "u", "D"), b)
    # out-of-range axis and unknown dims decline
    far = {"u": u, "$attr:E": 8, "$attr:D": 9}
    assert not eval_cond(("attr-cmp-dim", "E", ">=", "u", "D"), far)
    wild = {"u": _v("w", 2, None), "$attr:E": 8, "$attr:D": -1}
    assert not eval_cond(("attr-cmp-dim", "E", ">=", "w", "D"), wild)


def test_bcast_eq_predicate():
    u, v, w = _v("u", 2, 3), _v("v", 1, 3), _v("w", 2, 1)
    b = {"u": u, "v": v, "w": w}
    assert eval_cond(("bcast-eq", "u", "v", "u", "w"), b)  # (2,3)==(2,3)
    assert not eval_cond(("bcast-eq", "u", "v", "v", "v"), b)  # (2,3)!=(1,3)
    assert not eval_cond(("bcast-eq", "u", "missing", "u", "v"), b)
    # an ill-typed broadcast side declines, it does not wildcard
    bad = {"u": u, "v": _v("z", 5), "w": w}
    assert not eval_cond(("bcast-eq", "u", "v", "u", "w"), bad)


def test_ones_before():
    u = _v("u", 1, 1, 4)
    b = {"u": u, "$attr:D": 2}
    assert eval_cond(("ones-before", "u", "D"), b)  # u[:2] all 1
    assert eval_cond(("ones-before", "u", "D"), {"u": _v("x", 2, 1, 4), "$attr:D": 0})
    assert eval_cond(("ones-before", "u", "D"), {"u": _v("x", 1, 2, 4), "$attr:D": 1})
    # a non-1 dim at or before the axis fails the pairing
    assert not eval_cond(("ones-before", "u", "D"), {"u": _v("x", 2, 1, 4), "$attr:D": 1})
    assert not eval_cond(("ones-before", "u", "D"), {"u": _v("x", 2, 1, 4), "$attr:D": 2})
    assert not eval_cond(("ones-before", "missing", "D"), b)
    assert not eval_cond(("ones-before", "u", "MISSING"), b)
    # negative axis normalizes mod (rank+1)
    assert eval_cond(("ones-before", "u", "D"), {"u": u, "$attr:D": -2})
    assert not eval_cond(("ones-before", "u", "D"), {"u": _v("x", 2, 1, 4), "$attr:D": -2})


def test_axes_noop():
    u = _v("u", 2, 3)
    same = {"u": u, "$attr:D0": 0, "$attr:D1": 0}
    neg_same = {"u": u, "$attr:D0": -2, "$attr:D1": 0}
    swap = {"u": u, "$attr:D0": 0, "$attr:D1": 1}
    assert eval_cond(("axes-noop", "u", "D0", "D1"), same)
    assert eval_cond(("axes-noop", "u", "D0", "D1"), neg_same)
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), swap)
    # swapping two size-1 axes is still a semantic no-op
    flat = _v("f", 1, 1, 4)
    both1 = {"u": flat, "$attr:D0": 0, "$attr:D1": 1}
    assert eval_cond(("axes-noop", "u", "D0", "D1"), both1)
    one1 = {"u": _v("g", 1, 3, 4), "$attr:D0": 0, "$attr:D1": 1}
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), one1)
    # rank-1: every valid pair is (0, 0)
    vec = {"u": _v("v", 4), "$attr:D0": 0, "$attr:D1": -1}
    assert eval_cond(("axes-noop", "u", "D0", "D1"), vec)
    far = {"u": _v("v", 4), "$attr:D0": 0, "$attr:D1": 5}
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), far)
    noattr = {"u": _v("v", 4)}
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), noattr)
    # scalar / unshaped / out-of-range pair decline
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), {"u": _v("s"), "$attr:D0": 0, "$attr:D1": 0})
    assert not eval_cond(("axes-noop", "missing", "D0", "D1"), swap)
    oor = {"u": u, "$attr:D0": 0, "$attr:D1": 7}
    assert not eval_cond(("axes-noop", "u", "D0", "D1"), oor)


def test_flat_pair_unsq():
    # The wr pairing: reshape(mul(u,v),S) vs mul(unsq(u,d),reshape(v,S))
    # reads u through the same flat index map.
    cond = ("flat-pair-unsq", "U", "A_dim", "V", ("reshape-out", "V", "B_shape"))
    # u=(4,), S=(4,1), nd=1 — trailing-1 insert keeps stride-1 reads.
    b = {"U": _v("u", 4), "V": _v("v", 4),
         "$attr:A_dim": -1, "$attr:B_shape": (4, 1)}
    assert eval_cond(cond, b)
    # u=(4,), S=(1,4), nd=0 — leading-1 insert keeps stride-1 reads.
    b2 = {"U": _v("u", 4), "V": _v("v", 4),
          "$attr:A_dim": 0, "$attr:B_shape": (1, 4)}
    assert eval_cond(cond, b2)
    # u=(4,), v=(2,3,4), S=(6,4), nd=0 — the observed equal instance.
    b3 = {"U": _v("u", 4), "V": _v("v", 2, 3, 4),
          "$attr:A_dim": 0, "$attr:B_shape": (6, 4)}
    assert eval_cond(cond, b3)
    # u with leading 1-dims: they are skipped by the pairing check.
    b4 = {"U": _v("u", 1, 4), "V": _v("v", 1, 4),
          "$attr:A_dim": 0, "$attr:B_shape": (1, 4)}
    assert eval_cond(cond, b4)
    # stride mismatch: S=(4,2) splits the multiplied axis.
    nb = {"U": _v("u", 4), "V": _v("v", 2, 4),
          "$attr:A_dim": -1, "$attr:B_shape": (4, 2)}
    assert not eval_cond(cond, nb)
    # extent mismatch: the grid dim does not carry u's extent.
    nb_x = {"U": _v("u", 4), "V": _v("v", 2, 2),
            "$attr:A_dim": -1, "$attr:B_shape": (4, 1)}
    assert not eval_cond(cond, nb_x)
    # S rank too small for the unsqueeze output → out of range.
    nb2 = {"U": _v("u", 2, 3), "V": _v("v", 2, 3),
           "$attr:A_dim": 0, "$attr:B_shape": (6,)}
    assert not eval_cond(cond, nb2)
    # ill-typed / unshaped / missing attr decline
    assert not eval_cond(cond, {"U": _v("u", 4), "$attr:A_dim": 0,
                                "$attr:B_shape": (4, 1)})
    assert not eval_cond(cond, {"U": _v("u", 4), "V": _v("v", 4),
                                "$attr:A_dim": 0, "$attr:B_shape": (4, 4)})


def test_flat_map_unsq():
    # The wl naturality: mul(unsq(u,d),reshape(v,S)) == unsq(mul(u,v),d)
    cond = ("flat-map-unsq", "U", "A_dim", "V", ("reshape-out", "V", "B_shape"))
    # u=(4,), v=(4,), S=(4,), d=0 — grids (1,4) agree, maps coincide.
    b = {"U": _v("u", 4), "V": _v("v", 4),
         "$attr:A_dim": 0, "$attr:B_shape": (4,)}
    assert eval_cond(cond, b)
    # u=(3,4), v=(4,), S=(4,1), d=2 — the observed ALiBi-like equal.
    b2 = {"U": _v("u", 3, 4), "V": _v("v", 4),
          "$attr:A_dim": 2, "$attr:B_shape": (4, 1)}
    assert eval_cond(cond, b2)
    # u=(4,), v=(2,3,4), S=(1,2,3,4), d=0 — observed equal instance.
    b3 = {"U": _v("u", 4), "V": _v("v", 2, 3, 4),
          "$attr:A_dim": 0, "$attr:B_shape": (1, 2, 3, 4)}
    assert eval_cond(cond, b3)
    # u=(2,3), v=(1,3), S=(1,3), d=0 — v's leading broadcast-1 dim.
    b4 = {"U": _v("u", 2, 3), "V": _v("v", 1, 3),
          "$attr:A_dim": 0, "$attr:B_shape": (1, 3)}
    assert eval_cond(cond, b4)
    # v-map mismatch at equal grids: v=(2,), S=(2,1) reads v[j]
    # where the mul-order reads v[k] — same grid, wrong pairing.
    nb = {"U": _v("u", 2, 2), "V": _v("v", 2),
          "$attr:A_dim": 0, "$attr:B_shape": (2, 1)}
    assert not eval_cond(cond, nb)
    # u-map mismatch: the insertion shifts a non-1 u dim out of the
    # grid position the mul-order reads it at.
    nb_u = {"U": _v("u", 3), "V": _v("v", 2, 3, 1),
            "$attr:A_dim": 1, "$attr:B_shape": (2, 1, 1, 3)}
    assert not eval_cond(cond, nb_u)
    # v misreads: v=(4,), S=(2,2) — numel ok but the flat map splits.
    nb = {"U": _v("u", 4), "V": _v("v", 4),
          "$attr:A_dim": -1, "$attr:B_shape": (2, 2)}
    assert not eval_cond(cond, nb)
    # grid mismatch: inserted axis does not reproduce the LHS grid.
    nb2 = {"U": _v("u", 2, 3), "V": _v("v", 2, 3),
           "$attr:A_dim": 1, "$attr:B_shape": (2, 3)}
    assert not eval_cond(cond, nb2)
    # ill-typed broadcast, unshaped spec, missing attr decline.
    nb3 = {"U": _v("u", 2), "V": _v("v", 3),
         "$attr:A_dim": 0, "$attr:B_shape": (6,)}
    assert not eval_cond(cond, nb3)
    assert not eval_cond(cond, {"U": _v("u", 4),
                              "$attr:A_dim": 0, "$attr:B_shape": (4,)})
    wild = {"U": _v("u", 4, None), "V": _v("v", 4),
            "$attr:A_dim": 0, "$attr:B_shape": (4,)}
    assert not eval_cond(cond, wild)


def test_tail_block_spec():
    # ("tail-block", T, NAME): resolves to shape(T)[-k:] iff the bound
    # reduce dims name exactly T's last k axes — the rms pair's
    # normalized_shape, one spec for check AND derive.
    u = _v("u", 2, 4, 8)
    b = {"u": u, "$attr:MD": (-2, -1)}
    assert _shape_helper(("tail-block", "u", "MD"), b) == (4, 8)
    assert _shape_helper(("tail-block", "u", "MD"), {"u": u, "$attr:MD": (0, 1, 2)}) == (2, 4, 8)
    # bare-int and list spellings normalize the same way
    assert _shape_helper(("tail-block", "u", "MD"), {"u": _v("x", 4, 8), "$attr:MD": -1}) == (8,)
    assert _shape_helper(("tail-block", "u", "MD"), {"u": _v("x", 4, 8), "$attr:MD": [1]}) == (8,)
    # declines: non-trailing, duplicate, out-of-range, over-rank dims
    for dims in ((0,), (1,), (-1, -1), (-4,), (), None, True, (True,), "x", (1.5,)):
        assert _shape_helper(
            ("tail-block", "u", "MD"), {"u": u, "$attr:MD": dims}
        ) is None, dims
    # more dims than rank is no block either
    assert _shape_helper(
        ("tail-block", "u", "MD"), {"u": _v("x", 4, 8), "$attr:MD": (0, 1, -1)}
    ) is None
    # u's shape must be concrete — the spec IS the minted attr
    assert _shape_helper(
        ("tail-block", "u", "MD"), {"u": _v("w", None, 4, 8), "$attr:MD": (-2, -1)}
    ) is None
    assert _shape_helper(("tail-block", "missing", "MD"), b) is None
    # and arity is validated like every spec
    assert _shape_helper(("tail-block", "u"), b) is None
    # as a guard: shaped proves the block, shape-eq gates on it
    assert eval_cond(("shaped", ("tail-block", "u", "MD")), b)
    assert not eval_cond(("shaped", ("tail-block", "u", "MD")), {"u": u, "$attr:MD": (0,)})
    w = _p("w", 4, 8)
    assert eval_cond(("shape-eq", "w", ("tail-block", "u", "MD")), {**b, "w": w})
    assert not eval_cond(("shape-eq", "w", ("tail-block", "u", "MD")), {**b, "w": _p("w2", 8)})
    # scalar u: no rank for a block to live in
    assert not eval_cond(
        ("shaped", ("tail-block", "s", "MD")), {"s": _v("s"), "$attr:MD": (-1,)}
    )


def _repeat_bound(r=4, *, ud=3, es=None, rs=None, k_shape=(1, 4, 2, 3)):
    """A well-formed unsqueeze→expand→reshape binding (repeat_chain)."""
    k = _p("k", *k_shape)
    us = (*k_shape[:ud], 1, *k_shape[ud:])
    if es is None:
        es = (*us[:ud], r, *us[ud + 1 :])
    if rs is None:
        rs = (*us[: ud - 1], us[ud - 1] * r, *us[ud + 1 :])
    return {"k": k, "$attr:UDk": ud, "$attr:ESk": es, "$attr:RSk": rs}


def test_repeat_chain_predicate():
    # ("repeat-chain", T, UD, ES, RS): unsq→expand→reshape IS
    # repeat_interleave on T's dim d-1.
    cond = ("repeat-chain", "k", "UDk", "ESk", "RSk")
    ok = _repeat_bound()
    assert eval_cond(cond, ok)
    # non-int dim attr / non-tuple shapes / unshaped base → decline
    assert not eval_cond(cond, {**ok, "$attr:UDk": "3"})
    assert not eval_cond(cond, {**ok, "$attr:ESk": 5})
    assert not eval_cond(cond, {**ok, "$attr:RSk": 5})
    assert not eval_cond(cond, {"k": _v("k"), "$attr:UDk": 1,
                                "$attr:ESk": (1, 1), "$attr:RSk": (1,)})
    # symbolic dims in the base decline
    assert not eval_cond(cond, _repeat_bound(k_shape=(None, 4, 2, 3)))
    # unsqueeze at position 0 is not a repeat_interleave pattern
    assert not eval_cond(cond, _repeat_bound(ud=0))
    # expand must match the unsqueezed rank
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1, 4, 2, 4)})
    # the repeat factor must be an int > 1
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1, 4, 2, 1, 3)})
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1, 4, 2, 1.5, 3)})
    # expand may only grow the inserted dim
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1, 8, 2, 4, 3)})
    # reshape must have the base rank and merge the repeated dims
    assert not eval_cond(cond, {**ok, "$attr:RSk": (1, 4, 8)})
    assert not eval_cond(cond, {**ok, "$attr:RSk": (1, 4, 2, 4, 3)})


def test_repeat_heads_predicate():
    # ("repeat-heads", A, B, UD, ES): sa[-2] == sb[-2] * es[d] —
    # query heads = kv heads times the repeat factor.
    cond = ("repeat-heads", "q", "k", "UDk", "ESk")
    ok = {"q": _p("q", 1, 4, 8, 3), "k": _p("k", 1, 4, 2, 3),
          "$attr:UDk": 3, "$attr:ESk": (1, 4, 2, 4, 3)}
    assert eval_cond(cond, ok)  # 8 == 2 * 4
    assert not eval_cond(cond, {**ok, "q": _p("q", 1, 4, 7, 3)})
    # unshaped / low-rank / symbolic operands decline
    assert not eval_cond(cond, {**ok, "q": _v("q")})
    assert not eval_cond(cond, {**ok, "q": _p("q", 8)})
    assert not eval_cond(cond, {**ok, "q": _p("q", 1, 4, None, 3)})
    assert not eval_cond(cond, {**ok, "k": _p("k", 1, None, 2, 3)})
    # non-int axis / non-tuple expand shape / factor slot misses
    assert not eval_cond(cond, {**ok, "$attr:UDk": "3"})
    assert not eval_cond(cond, {**ok, "$attr:ESk": 5})
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1,)})
    # a non-int factor declines
    assert not eval_cond(cond, {**ok, "$attr:ESk": (1, 4, 2, 1.5, 3)})


# ---------------------------------------------------------------------------
#  The decoded guards — one cond per conditional view-oracle candidate
#  (project/retros/cond-dsl-view-guards.md has the full table; these
#  pin the accept/decline verdicts on canonical true/false envs).
# ---------------------------------------------------------------------------

_UNSQ_STRIP_COND = (
    "and",
    ("ones-before", "U", "A_dim"),
    ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"),
)


def test_unsqueeze_strip_guard():
    # mul(unsq(u,d), v) == mul(u,v) iff u dims before the inserted axis
    # are all 1 AND the two broadcast grids coincide.
    yes = {"U": _v("u", 4), "V": _v("v", 3, 4), "$attr:A_dim": 0}
    yes2 = {"U": _v("u", 2, 3), "V": _v("v", 1, 2, 3), "$attr:A_dim": 0}
    no_axis = {"U": _v("u", 2, 3), "V": _v("v", 1, 2, 3), "$attr:A_dim": 1}
    no_grid = {"U": _v("u", 4), "V": _v("v", 3, 3), "$attr:A_dim": 0}
    assert eval_cond(_UNSQ_STRIP_COND, yes)
    assert eval_cond(_UNSQ_STRIP_COND, yes2)
    # ones-before covers the non-pad insertions with leading 1s too.
    pad = {"U": _v("u", 1, 3), "V": _v("v", 2, 1, 3), "$attr:A_dim": 1}
    assert eval_cond(_UNSQ_STRIP_COND, pad)
    assert not eval_cond(_UNSQ_STRIP_COND, no_axis)
    assert not eval_cond(_UNSQ_STRIP_COND, no_grid)


def test_slice_strip_guard():
    cond = (
        "and",
        ("or", ("attr-is", "A_start", None), ("attr-eq", "A_start", 0)),
        ("or", ("attr-is", "A_end", None),
         ("attr-cmp-dim", "A_end", ">=", "U", "A_dim")),
    )
    full = {"U": _v("u", 2, 3), "$attr:A_dim": -1, "$attr:A_start": 0,
            "$attr:A_end": 3}
    full_huge = {"U": _v("u", 2, 3), "$attr:A_dim": -1,
                 "$attr:A_start": 0, "$attr:A_end": 2**62}
    none_attrs = {"U": _v("u", 2, 3), "$attr:A_dim": -1,
                  "$attr:A_start": None, "$attr:A_end": None}
    partial = {"U": _v("u", 2, 3), "$attr:A_dim": -1,
               "$attr:A_start": 0, "$attr:A_end": 2}
    offset = {"U": _v("u", 2, 3), "$attr:A_dim": -1,
              "$attr:A_start": 1, "$attr:A_end": 3}
    for bound in (full, full_huge, none_attrs):
        assert eval_cond(cond, bound)
    for bound in (partial, offset):
        assert not eval_cond(cond, bound)


def test_getitem_strip_guard():
    cond = (
        "and",
        ("leaf", "U"),
        ("dim-eq-const", "U", 0, 1),
        ("attr-in", "A_index", (0, -1)),
        ("bcast-eq", ("getitem-out", "U"), "V", "U", "V"),
    )
    yes = {"U": _v("u", 1, 4), "V": _v("v", 1, 4), "$attr:A_index": 0}
    yes2 = {"U": _v("u", 1, 4), "V": _v("v", 2, 1, 4), "$attr:A_index": -1}
    # grid mismatch: v drops the retained leading axis.
    no_grid = {"U": _v("u", 1, 4), "V": _v("v", 4), "$attr:A_index": 0}
    # extent > 1: getitem picks a row, not a broadcast no-op.
    no_ext = {"U": _v("u", 2, 4), "V": _v("v", 2, 4), "$attr:A_index": 0}
    # tuple producer bound: the RHS eq(u,v) does not denote.
    tup = {"U": Op.make("topk", _v("w", 2, 4), k=2), "V": _v("v", 2, 4),
           "$attr:A_index": 0}
    for bound in (yes, yes2):
        assert eval_cond(cond, bound)
    for bound in (no_grid, no_ext, tup):
        assert not eval_cond(cond, bound)


def test_reshape_unsq_family_guards_roundtrip():
    """The three mixed-family guards are pure data — round-trip by name."""
    guards = {
        "id": (
            "and",
            ("shape-eq", "V", ("reshape-out", "V", "B_shape")),
            ("ones-before", "U", "A_dim"),
            ("bcast-eq", ("unsq-out", "U", "A_dim"),
             ("reshape-out", "V", "B_shape"), "U", "V"),
        ),
        "wr": (
            "and",
            ("bcast-into", "U", "V"),
            ("bcast-into", ("unsq-out", "U", "A_dim"),
             ("reshape-out", "V", "B_shape")),
            ("flat-pair-unsq", "U", "A_dim", "V",
             ("reshape-out", "V", "B_shape")),
        ),
        "wl": (
            "flat-map-unsq", "U", "A_dim", "V",
            ("reshape-out", "V", "B_shape"),
        ),
    }
    raw = json.loads(json.dumps(cond_to_data(guards["id"])))
    assert cond_from_data(raw) == guards["id"]
    raw = json.loads(json.dumps(cond_to_data(guards["wr"])))
    assert cond_from_data(raw) == guards["wr"]
    raw = json.loads(json.dumps(cond_to_data(guards["wl"])))
    assert cond_from_data(raw) == guards["wl"]
    # the id guard on the canonical equal env (identity reshape + pad)
    yes = {"U": _v("u", 4), "V": _v("v", 3, 4),
           "$attr:A_dim": 0, "$attr:B_shape": (3, 4)}
    assert eval_cond(guards["id"], yes)
    no = {"U": _v("u", 4), "V": _v("v", 3, 4),
          "$attr:A_dim": 1, "$attr:B_shape": (3, 4)}
    assert not eval_cond(guards["id"], no)
    # the wr guard on the canonical equal env
    yes = {"U": _v("u", 4), "V": _v("v", 4),
           "$attr:A_dim": -1, "$attr:B_shape": (4, 1)}
    assert eval_cond(guards["wr"], yes)
    # wl canonical equal env
    yes = {"U": _v("u", 4), "V": _v("v", 4),
           "$attr:A_dim": 0, "$attr:B_shape": (4,)}
    assert eval_cond(guards["wl"], yes)


# ---------------------------------------------------------------------------
#  The promoted discovery objects — auto-cond / fold_object / guarded
#  composition outputs that became shipped laws
#  (``project/retros/promoted-laws.md``).  Each pin: the guard accepts
#  its measured region and declines the off-region binding, the rule
#  fires through the real ``EGraph`` path, and the whole record is
#  serializable data.
# ---------------------------------------------------------------------------


def _merged(rule, src, dst):
    """Run *rule* alone; return whether src and dst ended up merged."""
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    return eg.find(root) == eg.find(eg.add_term(dst))


def _rt(rule):
    """JSON round-trip the law record; the rebuilt rule fires the same."""
    import json as _json

    return law_from_data(
        _json.loads(_json.dumps(law_to_data(rule)))
    )


PROMOTED_LAWS = (
    SOFTSIGN_FOLD,
    TRANSPOSE_NOOP,
    CHUNK_SINGLE,
    MUL_UNSQ_PAD_L,
    MUL_UNSQ_PAD_R,
    SUB_UNSQ_PAD_L,
    MUL_RESHAPE_INERT_L,
    SDPA_FOLD_NOMASK,
    SDPA_FOLD_DIV_NOMASK,
    LINEAR_CHANNEL_TO_ROW_SCALE,
)


def test_promoted_laws_are_full_data():
    """Every promoted law serializes whole — no procedural hooks."""
    for rule in PROMOTED_LAWS:
        assert missing_hooks(rule) == (), rule.name
        rebuilt = _rt(rule)
        assert law_to_data(rebuilt) == law_to_data(rule), rule.name


def test_promoted_softsign_fold():
    x = _v("x", 4, 8)
    src = Op.make(
        "div", x, Op.make("add", Op.make("abs", x), Const(1))
    )
    assert _merged(SOFTSIGN_FOLD, src, Op.make("softsign", x))
    assert _merged(_rt(SOFTSIGN_FOLD), src, Op.make("softsign", x))
    # the decomposed spelling without the +1 is not the fold's pattern
    bad = Op.make("div", x, Op.make("abs", x))
    assert not _merged(SOFTSIGN_FOLD, bad, Op.make("softsign", x))
    assert SOFTSIGN_FOLD in all_rules()


def test_promoted_transpose_noop():
    # equal axes: an explicit no-op swap strips to the operand
    u = _v("u", 2, 3)
    src = Op.make("transpose", u, dim0=0, dim1=0)
    assert _merged(TRANSPOSE_NOOP, src, u)
    # both swapped extents are 1: the "transpose" permutes nothing
    w = _v("w", 4, 1, 1)
    src = Op.make("transpose", w, dim0=-2, dim1=-1)
    assert _merged(TRANSPOSE_NOOP, src, w)
    # a real swap declines — (8,8) transposed on (-2,-1) is not u
    t = _v("t", 8, 8)
    assert not _merged(
        TRANSPOSE_NOOP, Op.make("transpose", t, dim0=-2, dim1=-1), t
    )
    assert _merged(_rt(TRANSPOSE_NOOP), src, w)


def test_promoted_chunk_single():
    u = _v("u", 4, 8)
    src = Op.make("chunk", u, chunks=1, dim=-1, index=0)
    assert _merged(CHUNK_SINGLE, src, u)
    assert _merged(_rt(CHUNK_SINGLE), src, u)
    # a two-chunk index read is not the whole tensor
    src2 = Op.make("chunk", u, chunks=2, dim=-1, index=0)
    assert not _merged(CHUNK_SINGLE, src2, u)


def test_promoted_unsq_pad_strips():
    # u broadcast-padded at dim 0: grids coincide, pad view is dead.
    u, v = _v("u", 8, 8), _v("v", 1, 1, 1)
    src = Op.make("mul", Op.make("unsqueeze", u, dim=0), v)
    assert _merged(MUL_UNSQ_PAD_L, src, Op.make("mul", u, v))
    assert _merged(
        _rt(MUL_UNSQ_PAD_L), src, Op.make("mul", u, v)
    )
    assert _merged(
        SUB_UNSQ_PAD_L,
        Op.make("sub", Op.make("unsqueeze", u, dim=0), v),
        Op.make("sub", u, v),
    )
    # the off-region binding: a non-pad insertion axis declines
    u2, v2 = _v("u2", 8, 8), _v("v2", 8, 8)
    bad = Op.make("mul", Op.make("unsqueeze", u2, dim=1), v2)
    assert not _merged(MUL_UNSQ_PAD_L, bad, Op.make("mul", u2, v2))
    assert not _merged(SUB_UNSQ_PAD_L, bad, Op.make("sub", u2, v2))


def test_promoted_unsq_pad_right():
    # the right-operand twin: v's leading pad strips inside mul
    u, v = _v("u", 1, 8, 8), _v("v", 8, 8)
    src = Op.make("mul", u, Op.make("unsqueeze", v, dim=0))
    assert _merged(MUL_UNSQ_PAD_R, src, Op.make("mul", u, v))
    assert _merged(_rt(MUL_UNSQ_PAD_R), src, Op.make("mul", u, v))
    # v=(1,1,1) unsqueezed at 0 adds a real leading axis — declines
    u2, v2 = _v("u2", 8, 8), _v("v2", 1, 1, 1)
    bad = Op.make("mul", u2, Op.make("unsqueeze", v2, dim=0))
    assert not _merged(MUL_UNSQ_PAD_R, bad, Op.make("mul", u2, v2))


def test_promoted_reshape_inert_strip():
    # (h,w) -> (1,h,w) is broadcast-inert inside the elementwise op
    u, v = _v("u", 2, 6), _v("v", 1, 2, 6)
    src = Op.make("mul", Op.make("reshape", u, shape=(1, 2, 6)), v)
    assert _merged(MUL_RESHAPE_INERT_L, src, Op.make("mul", u, v))
    assert _merged(_rt(MUL_RESHAPE_INERT_L), src, Op.make("mul", u, v))
    # a permuting reshape changes the pairing — declines
    u2, v2 = _v("u2", 2, 3), _v("v2", 3, 2)
    bad = Op.make("mul", Op.make("reshape", u2, shape=(3, 2)), v2)
    assert not _merged(MUL_RESHAPE_INERT_L, bad, Op.make("mul", u2, v2))


def test_promoted_sdpa_nomask_folds():
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-1, dim1=-2)
    )
    src = Op.make("matmul", Op.make("softmax", scores, dim=-1), v)
    want = Op.make("sdpa", q, k, v, scale=1.0)
    assert _merged(SDPA_FOLD_NOMASK, src, want)
    assert _merged(_rt(SDPA_FOLD_NOMASK), src, want)
    # scaled twin: a numeric S mints the reciprocal scale attr
    src2 = Op.make(
        "matmul",
        Op.make(
            "softmax", Op.make("div", scores, Const(4.0)), dim=-1
        ),
        v,
    )
    want2 = Op.make("sdpa", q, k, v, scale=0.25)
    assert _merged(SDPA_FOLD_DIV_NOMASK, src2, want2)
    assert _merged(_rt(SDPA_FOLD_DIV_NOMASK), src2, want2)
    # a rank-1 Q declines — the minted sdpa cannot denote
    q1 = _v("q1", 4)
    bad = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "matmul",
                q1,
                Op.make("transpose", k, dim0=-1, dim1=-2),
            ),
            dim=-1,
        ),
        v,
    )
    assert not _merged(SDPA_FOLD_NOMASK, bad, want)


def test_promoted_channel_to_row_scale():
    # linear(x, W∘c) -> c·linear(x,W) — the guarded composite fires on
    # the scalar-weight-scale site the corpus measured.
    x, w = _v("x", 2, 4), _p("W", 3, 4)
    src = Op.make("linear", x, Op.make("mul", w, Const(2.0)))
    want = Op.make("mul", Op.make("linear", x, w), Const(2.0))
    assert _merged(LINEAR_CHANNEL_TO_ROW_SCALE, src, want)
    assert _merged(_rt(LINEAR_CHANNEL_TO_ROW_SCALE), src, want)
    # recorded derivation: the lemma is a two-premise consequence
    assert LINEAR_CHANNEL_TO_ROW_SCALE.derivation == (
        "linear_channel_scale_rev",
        "linear_row_scale",
    )
    assert LINEAR_CHANNEL_TO_ROW_SCALE.kind == "lemma"
    # a channel scale that is NOT a row scale declines: c=(4,) on a
    # 3x4 weight — bcast-into(c, out) fails (out's last dim is 3)
    c2 = _v("c", 4)
    bad = Op.make("linear", x, Op.make("mul", w, c2))
    want2 = Op.make("mul", Op.make("linear", x, w), c2)
    assert not _merged(LINEAR_CHANNEL_TO_ROW_SCALE, bad, want2)
