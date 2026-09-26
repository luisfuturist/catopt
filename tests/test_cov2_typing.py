"""Coverage-gap tests for catopt.typing.

``tests/test_typing.py`` pins the layer's contract on the happy paths.
This file drives the arms and edges it never reaches:

* the mixed-shape early return (a ``None``-shaped operand under a
  shaped one propagates the first shape, not ``None``);
* ``linear``'s rank-0 weight fall-through and the column-bias
  squeeze-vs-raw broadcast fallback;
* ``sum``/``mean`` non-keepdim reduces, ``reshape`` without a ``shape``
  attr, ``squeeze`` on a real base, ``getitem`` passthrough;
* ``slice`` on an unknown-length axis, ``embedding`` with a non-rank-2
  weight, ``index_select`` on a ``()`` base, ``split`` with
  un-parseable ``sizes``;
* ``conv2d``'s fallback return (str padding / short ranks) and the
  partial-``None`` spatial-dim arcs;
* ``_broadcast``'s negative-dim coercion on the *right* operand;
* carrier handler edges: ``trace`` on a non-matrix member, ``bdiag``
  on non-matrix members, ``cswap`` with non-int attrs;
* the rank-0 carrier-member ``() -> None`` policy on the remaining
  axis ops.

Defensive branches believed unreachable by construction (listed for
``pragma: no cover`` in ``test_defensive_branch_inventory``): the
``base is None`` guards inside ``sum``/``unsqueeze``/``stack`` — the
``_infer_op_shape`` dispatch filters ``None`` operand shapes before
the match, so they can never observe one.
"""

import pytest

from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.typing import (
    _INVALID,
    _SHAPE_RULES,
    _broadcast,
    _infer_op_shape,
    _numel,
    _shape_of,
    register_shape_rule,
)


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


x234 = _v("x234", 2, 3, 4)
x48 = _v("x48", 4, 8)
scalar = Const(1.0)
#: An op whose shape is ``None`` — a carrier-internal ``()`` member
#: read as a tensor — used to exercise mixed-shape propagation.
unknown = Op.make("transpose", scalar)


SHAPE_CASES = [
    # --- mixed-shape early return -------------------------------------
    # A None operand under a shaped one: the first shape survives.
    pytest.param(
        Op.make("add", x234, unknown), (2, 3, 4), id="mixed/first-shaped"
    ),
    pytest.param(
        Op.make("add", unknown, unknown), None, id="mixed/all-unknown"
    ),
    # --- linear -------------------------------------------------------
    # Rank-0 weight: no in-features axis to consume — fall through to
    # the operand shape rather than fabricating one.
    pytest.param(
        Op.make("linear", x48, scalar), (4, 8), id="linear/scalar-weight"
    ),
    # Column bias on a (B,o) output: the squeezed (B,) broadcast is
    # ill-typed, so the raw (B,1) broadcast must win — (B,o), not (B,B).
    pytest.param(
        Op.make(
            "linear",
            _v("x", 4, 16),
            _p("W", 8, 16),
            _p("b", 4, 1),
        ),
        (4, 8),
        id="linear/column-bias-raw-broadcast",
    ),
    # Vector input x (in,) -> (out,).
    pytest.param(
        Op.make("linear", _v("v", 16), _p("W", 8, 16)),
        (8,),
        id="linear/vector-x",
    ),
    # --- reductions ---------------------------------------------------
    pytest.param(
        Op.make("sum", x234, dim=1), (2, 4), id="sum/dim-no-keepdim"
    ),
    pytest.param(
        Op.make("mean", x234, dim=(0, 2)),
        (3,),
        id="mean/multi-dim-no-keepdim",
    ),
    pytest.param(
        Op.make("sum", x234, dim=[0, 1], keepdim=True),
        (1, 1, 4),
        id="sum/list-dims-keepdim",
    ),
    pytest.param(
        Op.make("sum", x234, axis=2),
        (2, 3),
        id="sum/axis-attr-spelling",
    ),
    # --- reshape ------------------------------------------------------
    pytest.param(
        Op.make("reshape", x234), (2, 3, 4), id="reshape/no-shape-attr"
    ),
    pytest.param(
        Op.make("reshape", x234, shape=(-1, 6)),
        (4, 6),
        id="reshape/minus-one-leading",
    ),
    # -1 that cannot be resolved (numel not divisible) -> unknown dim,
    # never a negative dim poisoning downstream broadcasting.
    pytest.param(
        Op.make("reshape", x234, shape=(5, -1)),
        (5, None),
        id="reshape/minus-one-unresolvable",
    ),
    # Two -1 dims resolve identically — the product then fails the
    # numel guard and the member is flagged ill-typed.
    pytest.param(
        Op.make("reshape", x234, shape=(4, -1, -1)),
        _INVALID,
        id="reshape/minus-one-ambiguous-invalid",
    ),
    # --- squeeze / unsqueeze ------------------------------------------
    pytest.param(
        Op.make("squeeze", _v("v", 2, 1, 4), dim=1),
        (2, 4),
        id="squeeze/middle-dim",
    ),
    pytest.param(
        Op.make("squeeze", _v("v", 2, 1, 4), dim=-2),
        (2, 4),
        id="squeeze/negative-dim-attr",
    ),
    pytest.param(
        Op.make("unsqueeze", x234, dim=1),
        (2, 1, 3, 4),
        id="unsqueeze/insert-dim",
    ),
    pytest.param(
        Op.make("unsqueeze", scalar, dim=0),
        (1,),
        id="unsqueeze/scalar-to-vector",
    ),
    # --- getitem / select / unbind ------------------------------------
    pytest.param(
        Op.make(
            "getitem",
            Op.make("split", _v("v", 8), sizes=(3, 5), index=0),
            index=1,
        ),
        (3,),
        id="getitem/passthrough",
    ),
    pytest.param(
        Op.make("select", scalar, dim=0, index=0),
        None,
        id="policy/select-of-scalar",
    ),
    pytest.param(
        Op.make("unbind", scalar, dim=0),
        None,
        id="policy/unbind-of-scalar",
    ),
    pytest.param(
        Op.make("chunk", scalar, chunks=2),
        None,
        id="policy/chunk-of-scalar",
    ),
    pytest.param(
        Op.make("flatten", scalar, start_dim=0),
        None,
        id="policy/flatten-of-scalar",
    ),
    pytest.param(
        Op.make("index_select", scalar, dim=0, index=(0,)),
        None,
        id="policy/index-select-of-scalar",
    ),
    # --- slice --------------------------------------------------------
    # Slicing an unknown-length axis keeps it unknown — no fabricated
    # extent.
    pytest.param(
        Op.make(
            "slice",
            _v("v", 2, None, 4),
            dim=1,
            start=0,
            end=None,
            step=1,
        ),
        (2, None, 4),
        id="slice/unknown-extent",
    ),
    pytest.param(
        Op.make(
            "slice",
            x234,
            dim=2,
            start=1,
            end=None,
            step=1,
        ),
        (2, 3, 3),
        id="slice/open-end",
    ),
    pytest.param(
        Op.make(
            "slice",
            x234,
            dim=2,
            start=None,
            end=3,
            step=1,
        ),
        (2, 3, 3),
        id="slice/none-start",
    ),
    # --- embedding ----------------------------------------------------
    pytest.param(
        Op.make(
            "embedding", _p("W", 16, 8, 4), _v("idx", 2, 3)
        ),
        None,
        id="embedding/rank3-weight",
    ),
    pytest.param(
        Op.make("embedding", scalar, _v("idx", 2)),
        None,
        id="embedding/scalar-weight",
    ),
    # --- index_select ---------------------------------------------------
    pytest.param(
        Op.make("index_select", x234, dim=1, index=(0, 2)),
        (2, 2, 4),
        id="index_select/tuple-index",
    ),
    pytest.param(
        Op.make("index_select", x234, dim=1, index=2),
        (2, None, 4),
        id="index_select/non-seq-index-unknown",
    ),
    # --- split --------------------------------------------------------
    # A ``sizes`` attr that is neither list nor int -> extent unknown.
    pytest.param(
        Op.make("split", _v("v", 8), sizes=None, index=0, dim=0),
        (None,),
        id="split/none-sizes",
    ),
    # --- conv2d -------------------------------------------------------
    pytest.param(
        Op.make(
            "conv2d",
            _v("x", 2, 3, 8, 8),
            _p("w", 4, 3, 3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
        ),
        (2, 4, 4, 4),
        id="conv2d/tuple-attrs",
    ),
    # str padding ('same'/'valid') — can't compute H' -> partial tuple.
    pytest.param(
        Op.make(
            "conv2d",
            _v("x", 2, 3, 8, 8),
            _p("w", 4, 3, 3, 3),
            padding="same",
        ),
        (2, 4, None, None),
        id="conv2d/string-padding",
    ),
    # Short-rank weight -> the (N, O, ?, ?) guess.
    pytest.param(
        Op.make(
            "conv2d", _v("x", 2, 3, 8, 8), _p("w", 4, 3, 3)
        ),
        (2, 4, None, None),
        id="conv2d/rank3-weight",
    ),
    # Degenerate (scalar) weight: even the (N,O,?,?) guess is unfounded.
    pytest.param(
        Op.make("conv2d", _v("x", 2, 3, 8, 8), scalar),
        (2, 3, 8, 8),
        id="conv2d/scalar-weight",
    ),
    # Unknown input spatial dims -> that output dim stays None while
    # the known one is still computed.
    pytest.param(
        Op.make(
            "conv2d",
            _v("x", 2, 3, None, 8),
            _p("w", 4, 3, 3, 3),
        ),
        (2, 4, None, 6),
        id="conv2d/unknown-h",
    ),
    pytest.param(
        Op.make(
            "conv2d",
            _v("x", 2, 3, 8, None),
            _p("w", 4, 3, 3, 3),
        ),
        (2, 4, 6, None),
        id="conv2d/unknown-w",
    ),
    pytest.param(
        Op.make(
            "conv2d",
            _v("x", 2, 3, 8, 8),
            _p("w", 4, 3, None, 3),
        ),
        (2, 4, None, 6),
        id="conv2d/unknown-kernel-h",
    ),
    # --- carrier handlers ----------------------------------------------
    # A non-matrix trace member: not len-2 -> unknown, not a fabricated
    # scalar.
    pytest.param(
        Op.make("trace", scalar, usize=1), None, id="trace/scalar-member"
    ),
    pytest.param(
        Op.make("trace", _v("v", 4, 5, 6), usize=1),
        (4, 5, 6),
        id="trace/rank3-member",
    ),
    # bdiag/parl need rank-2 int-dimmed members; anything else reports
    # the first member's shape (or None).
    pytest.param(
        Op.make("bdiag", scalar, _v("m", 2, 2)),
        None,
        id="bdiag/scalar-member",
    ),
    pytest.param(
        Op.make("bdiag", _v("a", 2, None), _v("b", 3, 3)),
        (2, None),
        id="bdiag/unknown-dims",
    ),
    pytest.param(
        Op.make("parl", _v("a", 2), _v("b", 3, 3)),
        (2,),
        id="parl/rank1-member",
    ),
    pytest.param(
        Op.make("cswap", d1=1.5, d2=2), None, id="cswap/non-int-attrs"
    ),
    pytest.param(
        Op.make("eye", dim="d"), (None, None), id="eye/symbolic-dim"
    ),
    # --- broadcasting policy shapes -------------------------------------
    # eq/lt family broadcast like elementwise arithmetic.
    pytest.param(
        Op.make("eq", _v("a", 4, 1), _v("b", 1, 8)),
        (4, 8),
        id="cmp/broadcast",
    ),
    # where(c, x, y) broadcasts all three against each other.
    pytest.param(
        Op.make(
            "where", _v("c", 4, 1), _v("x", 1, 8), _v("y", 4, 8)
        ),
        (4, 8),
        id="where/three-way-broadcast",
    ),
    pytest.param(
        Op.make("where", scalar, scalar, scalar), (), id="where/scalars"
    ),
    # --- shape-preserving pass-throughs ---------------------------------
    pytest.param(
        Op.make("dropout", x48, p=0.0, train=False),
        (4, 8),
        id="dropout/passthrough",
    ),
    pytest.param(
        Op.make("to", x48), (4, 8), id="to/passthrough"
    ),
    pytest.param(
        Op.make("type_as", x48, x234), (4, 8), id="type_as/passthrough"
    ),
    pytest.param(
        Op.make("masked_fill", x48, _v("m", 4, 8), scalar),
        (4, 8),
        id="masked_fill/passthrough",
    ),
    # --- default arm ----------------------------------------------------
    pytest.param(
        Op.make("narrow", x234, dim=1, start=0, length=2),
        (2, 3, 4),
        id="narrow/default-arm-first-operand",
    ),
    pytest.param(
        Op.make("some_unknown_op", scalar),
        None,
        id="default/scalar-to-none",
    ),
    pytest.param(
        Op.make("some_unknown_op"),
        None,
        id="default/zero-arg-unknown",
    ),
]


@pytest.mark.parametrize("term,expected", SHAPE_CASES)
def test_infer_op_shape_gaps(term, expected):
    got = _shape_of(term)
    if expected is _INVALID:
        assert got is _INVALID
    else:
        assert got == expected


# ---------------------------------------------------------------------------
#  _broadcast edges
# ---------------------------------------------------------------------------


def test_broadcast_negative_dim_right_operand():
    """A -1 on the RIGHT operand degrades to unknown too — the old
    asymmetric code only coerced the left side."""
    assert _broadcast((4, 8), (-1, 8)) == (None, 8)
    assert _broadcast((-1, 8), (4, 8)) == (None, 8)
    # Both sides unresolved.
    assert _broadcast((-1, 4), (8, -1)) == (None, None)
    # Unknown dims are sticky — a None never resolves to the other
    # side's concrete extent (conservative, never a mismatch).
    assert _broadcast((None, 4), (3, 4)) == (None, 4)
    assert _broadcast((3, 4), (None, 4)) == (None, 4)
    # One-dim broadcasting from each side.
    assert _broadcast((1, 4), (2, 4)) == (2, 4)
    assert _broadcast((2, 4), (1, 4)) == (2, 4)
    # Equal non-1 dims take the value.
    assert _broadcast((3, 4), (3, 4)) == (3, 4)
    # Hard mismatch still poisons.
    assert _broadcast((3, 4), (5, 4)) is _INVALID


def test_broadcast_empty_shapes():
    """() against a rank-n shape broadcasts as a scalar operand."""
    assert _broadcast((), (2, 3)) == (2, 3)
    assert _broadcast((2, 3), ()) == (2, 3)
    assert _broadcast((), ()) == ()


# ---------------------------------------------------------------------------
#  dispatch/registry edges
# ---------------------------------------------------------------------------


def test_invalid_operand_poisons_before_rule():
    """An _INVALID child short-circuits the whole term — the registered
    rule is never consulted."""
    bad = Op.make("add", _v("a", 4, 8), _v("b", 4, 7))
    assert _shape_of(bad) is _INVALID
    # A registered-rule op with an invalid operand is still poisoned:
    # the rule dispatch sits AFTER the _INVALID check.
    traced = Op.make("trace", bad, usize=2)
    assert _shape_of(traced) is _INVALID


def test_zero_arg_rule_gets_empty_shapes():
    """A registered zero-arg op's rule is invoked with ``shapes = ()``
    before the empty-shapes early return."""
    seen = []
    try:
        register_shape_rule(
            "zz_const",
            lambda op, shapes: seen.append(shapes) or (9, 9),
        )
        assert _shape_of(Op.make("zz_const")) == (9, 9)
        assert seen == [()]
    finally:
        _SHAPE_RULES.pop("zz_const", None)


def test_rule_overrides_builtin_arm():
    """A registered rule wins over the match arm for the same op."""
    x = _v("zzx", 2, 3)
    assert _shape_of(Op.make("neg", x)) == (2, 3)
    try:
        register_shape_rule("neg", lambda op, shapes: (5,))
        assert _shape_of(Op.make("neg", x)) == (5,)
    finally:
        _SHAPE_RULES.pop("neg", None)
    assert _shape_of(Op.make("neg", x)) == (2, 3)


def test_shape_of_non_term_is_none():
    """A bare non-term leaf (a metavar string, an int) is unknown."""
    assert _shape_of("metavar") is None
    assert _shape_of(7) is None
    assert _shape_of(None) is None


def test_numel_and_helpers():
    assert _numel(()) == 1
    assert _numel(None) == 1
    assert _numel((3, None)) == 3


# ---------------------------------------------------------------------------
#  defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------


def test_defensive_branch_inventory():
    """Documents the arms that cannot fire through ``_shape_of`` —
    candidates for ``# pragma: no cover`` rather than contorted tests:

    * ``_infer_op_shape`` line ~203-204 — ``sum``/``mean`` ``base is
      None``: the dispatch early-returns when ANY operand shape is
      ``None`` (lines 107-110), so the match arm only ever sees
      non-``None`` bases.
    * ``unsqueeze`` ``base is None -> return None`` (~274-275) — same
      reason.
    * ``stack`` ``base is None -> return None`` (~293-294) — same
      reason.
    * ``_infer_op_shape`` line ~341-342 for ``embedding`` is reachable
      only via non-rank-2 weights (covered above); the ``wsh is None``
      conjunct itself is dead for the same dispatch reason.
    """
    # The dispatch guarantee these guards rely on, demonstrated:
    # any term with a None-shaped operand never reaches the match arm.
    none_arg = Op.make("add", unknown, unknown)
    assert _infer_op_shape(none_arg) is None
