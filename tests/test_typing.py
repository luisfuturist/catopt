"""Contract tests for catopt.typing — the shape/type-inference layer.

Phase 1a extracted ``_shape_of`` / ``_infer_op_shape`` / ``_INVALID`` /
``_broadcast`` / ``_numel`` out of catopt.cost and moved the carrier
op cases into ``register_shape_rule`` handlers.  These tests pin the
contract the layer publishes:

* shapes are best-effort tuples (dims may be ``None``);
* ``_INVALID`` = provably ill-typed (poisonous to extraction);
* ``None`` = unknown — including the carrier-internal ``()`` policy
  (a ``()`` operand under an axis op, or a carrier's ``()`` slot read
  as a tensor, reports unknown, never a fabricated scalar);
* ``()`` = genuine scalar only;
* carrier ops (aff/apply/om/omd/trace/bdiag/parl families) answer via
  their registered handlers in ``_SHAPE_RULES``.
"""

import pytest

from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.typing import (
    _INVALID,
    _SHAPE_RULES,
    _broadcast,
    _numel,
    _shape_of,
    register_shape_rule,
)

B, T, K, D = 4, 8, 3, 16

x234 = Var("x234", TensorType((2, 3, 4)))
x_BTk = Var("x_BTk", TensorType((B, T, K)))
x_BTd = Var("x_BTd", TensorType((B, T, D)))
vec8 = Var("vec8", TensorType((8,)))
mat = Var("mat", TensorType((10, 10)))
scalar = Const(1.0)


#: (op term, expected shape) — one row per contract point.
SHAPE_CASES = [
    # --- slice incl. strided step (arg4) ----------------------------------
    pytest.param(
        Op.make("slice", x234, arg1=2, arg2=0, arg3=None, arg4=1),
        (2, 3, 4),
        id="slice/full-range",
    ),
    pytest.param(
        Op.make("slice", x234, arg1=2, arg2=0, arg3=None, arg4=2),
        (2, 3, 2),
        id="slice/strided-step2",
    ),
    pytest.param(
        Op.make("slice", x234, arg1=2, arg2=1, arg3=4, arg4=2),
        (2, 3, 2),
        id="slice/bounded-strided",
    ),
    pytest.param(
        Op.make("slice", x234, dim=1, arg2=0, arg3=2, arg4=1),
        (2, 2, 4),
        id="slice/dim-attr-spelling",
    ),
    # --- split: both attr spellings + section index -----------------------
    pytest.param(
        Op.make("split", vec8, sizes=(3, 5), index=1, dim=0),
        (5,),
        id="split/sizes-index-spelling",
    ),
    pytest.param(
        Op.make("split", vec8, arg1=(3, 5), arg3=0, dim=0),
        (3,),
        id="split/arg1-arg3-spelling",
    ),
    pytest.param(
        Op.make("split", vec8, arg1=4, arg3=1, dim=0),
        (4,),
        id="split/equal-sections",
    ),
    pytest.param(
        Op.make("split", vec8, sizes=(3, 5), index=5, dim=0),
        (None,),
        id="split/index-past-end-unknown",
    ),
    # --- chunk ------------------------------------------------------------
    pytest.param(
        Op.make("chunk", Var("c", TensorType((2, 8))), chunks=4, dim=-1),
        (2, 2),
        id="chunk/last-dim",
    ),
    # --- concat arity: variadic cat sums every operand on the axis --------
    pytest.param(
        Op.make(
            "concat",
            Var("a", TensorType((2, 3))),
            Var("b", TensorType((2, 5))),
            Var("c", TensorType((2, 1))),
            dim=1,
        ),
        (2, 9),
        id="concat/three-operands",
    ),
    # Rank mismatch is best-effort, not poisonous: returns the base shape.
    pytest.param(
        Op.make(
            "concat",
            Var("a", TensorType((2, 3))),
            Var("b", TensorType((5,))),
            dim=1,
        ),
        (2, 3),
        id="concat/rank-mismatch-falls-back",
    ),
    # --- reshape numel guard ----------------------------------------------
    pytest.param(
        Op.make("reshape", x234, shape=(4, 6)),
        (4, 6),
        id="reshape/preserving",
    ),
    pytest.param(
        Op.make("reshape", x234, shape=(4, -1)),
        (4, 6),
        id="reshape/minus-one-resolved",
    ),
    pytest.param(
        Op.make("reshape", x234, shape=(4, 7)),
        _INVALID,
        id="reshape/numel-mismatch-invalid",
    ),
    # --- broadcast ----------------------------------------------------------
    pytest.param(
        Op.make(
            "mul",
            Var("a", TensorType((4, 8, 1))),
            Var("b", TensorType((4, 8, 32))),
        ),
        (4, 8, 32),
        id="broadcast/trailing-one",
    ),
    pytest.param(
        Op.make(
            "add",
            Var("a", TensorType((4, 8))),
            Var("b", TensorType((4, 7))),
        ),
        _INVALID,
        id="broadcast/hard-mismatch-invalid",
    ),
    # --- scalar semantics: () is genuine only at real scalar ops ----------
    pytest.param(Op.make("sum", x234), (), id="sum/full-reduce-scalar"),
    pytest.param(
        Op.make("sum", x234, dim=1, keepdim=True),
        (2, 1, 4),
        id="sum/keepdim",
    ),
    pytest.param(
        Op.make("matmul", vec8, vec8), (), id="matmul/dot-scalar"
    ),
    # --- the () -> None carrier policy --------------------------------------
    pytest.param(
        Op.make("transpose", scalar), None, id="policy/transpose-of-scalar"
    ),
    pytest.param(
        Op.make("slice", scalar, dim=0),
        None,
        id="policy/slice-of-scalar",
    ),
    pytest.param(
        Op.make("sum", scalar, dim=0),
        None,
        id="policy/explicit-dim-reduce-of-scalar",
    ),
    pytest.param(
        Op.make("concat", scalar, scalar, dim=0),
        None,
        id="policy/concat-of-scalars",
    ),
    # --- carrier ops report via registered handlers -----------------------
    pytest.param(
        Op.make("om", x_BTk, x_BTk, x_BTd),
        (B, T, D),
        id="carrier/om-accumulator-shape",
    ),
    pytest.param(
        Op.make("om", x_BTk, x_BTk, scalar),
        None,
        id="carrier/om-scalar-slot-unknown",
    ),
    pytest.param(
        Op.make("om_elem", x_BTk, x_BTd),
        (B, T, D),
        id="carrier/om-elem-applied-shape",
    ),
    pytest.param(
        Op.make("apply", Var("f", TensorType((D, D))), x_BTd),
        (B, T, D),
        id="carrier/apply-evaluates-h",
    ),
    pytest.param(
        Op.make("aff", Var("A", TensorType((D, D))), vec8),
        (D, D),
        id="carrier/aff-linear-part-shape",
    ),
    pytest.param(
        Op.make("aff", scalar, vec8),
        None,
        id="carrier/aff-scalar-slot-unknown",
    ),
    pytest.param(
        Op.make("applyd", Var("f", TensorType((D,))), vec8),
        (8,),
        id="carrier/applyd-evaluates-h",
    ),
    pytest.param(
        Op.make("trace", mat, usize=4),
        (6, 6),
        id="carrier/trace-drops-usize",
    ),
    pytest.param(
        Op.make("trace", mat, usize=(2, 2)),
        (6, 6),
        id="carrier/trace-usize-list",
    ),
    pytest.param(
        Op.make(
            "bdiag",
            Var("p", TensorType((2, 3))),
            Var("q", TensorType((4, 5))),
        ),
        (6, 8),
        id="carrier/bdiag-juxtaposes",
    ),
    pytest.param(
        Op.make(
            "parl",
            Var("p", TensorType((2, 3))),
            Var("q", TensorType((4, 5))),
        ),
        (6, 8),
        id="carrier/parl-juxtaposes",
    ),
    # --- zero-arg constant morphisms (registered, attrs-only) -------------
    pytest.param(Op.make("eye", dim=5), (5, 5), id="const-morphism/eye"),
    pytest.param(
        Op.make("cswap", d1=2, d2=3),
        (5, 5),
        id="const-morphism/cswap",
    ),
    # --- shape-preserving ops w/ attrs (default arm) ----------------------
    pytest.param(
        Op.make(
            "rms_norm",
            Var("h", TensorType((4, 64))),
            Param("g", TensorType((64,))),
            dim=(64,),
            arg3=1e-6,
        ),
        (4, 64),
        id="rms_norm/dim-attr-passthrough",
    ),
]


@pytest.mark.parametrize("term,expected", SHAPE_CASES)
def test_infer_op_shape(term, expected):
    got = _shape_of(term)
    if expected is _INVALID:
        assert got is _INVALID
    else:
        assert got == expected


def test_broadcast_helper():
    assert _broadcast((4, 8, 1), (8, 32)) == (4, 8, 32)
    assert _broadcast(None, (2, 3)) == (2, 3)
    assert _broadcast((2, 3), None) == (2, 3)
    assert _broadcast((4, 8), (4, 7)) is _INVALID
    # Unresolved -1 dims degrade to unknown, never a mismatch.
    assert _broadcast((-1, 8), (4, 8)) == (None, 8)


def test_numel_helper():
    assert _numel(None) == 1
    assert _numel(()) == 1
    assert _numel((2, 3, 4)) == 24
    assert _numel((2, None, 4)) == 8


def test_register_shape_rule_hook():
    """The registry is real dispatch: a custom rule installs and
    _infer_op_shape consults it ahead of the match's default arm."""
    try:
        register_shape_rule(
            "myop_test", lambda op, shapes: (7, *shapes[0][1:])
        )
        got = _shape_of(Op.make("myop_test", x234))
        assert got == (7, 3, 4)
    finally:
        _SHAPE_RULES.pop("myop_test", None)
    # After unregistering the same term falls back to shapes[0].
    assert _shape_of(Op.make("myop_test", x234)) == (2, 3, 4)


def test_leaf_typing():
    assert _shape_of(Const(3.0)) == ()
    v = Var("v", TensorType((None, 4)))
    assert _shape_of(v) == (None, 4)


def test_memo_is_id_keyed_dag():
    """Shared subterms resolve once through the memo (DAG, not tree)."""
    shared = Op.make("mul", x234, x234)
    term = Op.make("add", shared, shared)
    memo: dict = {}
    assert _shape_of(term, memo) == (2, 3, 4)
    assert id(shared) in memo
