"""Tests for ``catopt_core.opdata`` — declaring an op as data.

Plan 0019: a view / relayout-class op is a *declaration* — an
:class:`~catopt_core.opdata.OpDef` (name, arity, tags, attr schema, and
a shape rule written in the existing shape-spec DSL).  These tests
declare two real torch spellings absent from the registry
(``view_as``, ``swapaxes``) and show they are first-class: registered,
shaped, mintable, matchable and serializable — then that the seam is
additive (``reset_declarations`` restores the shipped state).
"""

import json

import pytest
from catopt_core import opdata
from catopt_core import opmeta as om
from catopt_core import typing as ty
from catopt_core.attrs import ATTR_REQUIRED, ATTR_SCHEMA
from catopt_core.ir import Op, TensorType, Var
from catopt_core.meta import match_pattern

#: A shape-preserving view: output is operand 0's shape.
_VIEW_AS = opdata.OpDef(
    name="view_as",
    arity=2,
    tags=frozenset({om.RELAYOUT, om.VIEWISH}),
    shape="arg1",  # output = operand 1's shape
)

#: An axis-swap view: the schema + a transpose shape spec.
_SWAPAXES = opdata.OpDef(
    name="swapaxes",
    arity=1,
    tags=frozenset({om.RELAYOUT, om.VIEWISH}),
    attrs={1: "dim0", 2: "dim1"},
    required=frozenset({"dim0", "dim1"}),
    shape=("transpose-out", "arg0", "dim0", "dim1"),
)


@pytest.fixture(autouse=True)
def _clean_declarations():
    """Isolate each test — the declaration seam is process-global."""
    opdata.reset_declarations()
    yield
    opdata.reset_declarations()


def _var(name, shape=(2, 3)):
    return Var(name, TensorType(shape))


# ---------------------------------------------------------------------------
#  Declaration registers into the live registry
# ---------------------------------------------------------------------------


def test_declared_op_joins_the_live_registry():
    assert om.meta("view_as") is None
    d = opdata.declare_op(_VIEW_AS)
    assert d is _VIEW_AS
    m = om.meta("view_as")
    assert m is not None and m.arity == 2
    assert om.tags_of("view_as") == frozenset({om.RELAYOUT, om.VIEWISH})
    assert om.arity("view_as") == 2
    # the live accessor sees it ...
    assert "view_as" in om.ops(om.RELAYOUT)
    assert "view_as" in om.REGISTRY
    # ... but the frozen shipped projection does not (the honest
    # boundary: the *_OPS constants are import-time snapshots).
    assert "view_as" not in om.RELAYOUT_OPS


def test_declared_op_shapes_through_typing():
    opdata.declare_op(_VIEW_AS)
    x, y = _var("x", (2, 3)), _var("y", (4, 5))
    term = Op.make("view_as", x, y)
    assert ty.shape_of(term) == (4, 5)


def test_declared_axis_op_shapes_and_mints():
    opdata.declare_op(_SWAPAXES)
    x = _var("x", (2, 3))
    term = Op.make("swapaxes", x, dim0=0, dim1=1)
    assert ty.shape_of(term) == (3, 2)
    # the declared schema + required attrs are enforced at mint.
    assert om.attrs("swapaxes") == {1: "dim0", 2: "dim1"}
    assert om.required("swapaxes") == frozenset({"dim0", "dim1"})
    assert ATTR_SCHEMA["swapaxes"] == {1: "dim0", 2: "dim1"}
    assert ATTR_REQUIRED["swapaxes"] == frozenset({"dim0", "dim1"})


def test_declared_axis_op_declines_on_an_unbound_axis():
    opdata.declare_op(_SWAPAXES)
    x = _var("x", (2, 3))
    # a non-int axis is not provable — the DSL declines (unknown).
    assert ty.shape_of(Op.make("swapaxes", x, dim0="D", dim1=1)) is None


def test_declared_op_without_a_shape_spec_uses_the_default():
    opdata.declare_op(
        opdata.OpDef(name="myview", arity=1, tags=frozenset({om.RELAYOUT}))
    )
    assert "myview" not in ty._SHAPE_RULES
    x = _var("x", (7, 8))
    assert ty.shape_of(Op.make("myview", x)) == (7, 8)


def test_declare_accepts_the_data_form():
    d = opdata.declare_op(opdata.opdef_to_data(_SWAPAXES))
    assert d == _SWAPAXES
    assert om.arity("swapaxes") == 1


def test_declaring_a_shipped_name_raises():
    with pytest.raises(ValueError, match="shipped op"):
        opdata.declare_op(opdata.OpDef(name="add"))
    assert om.meta("add").tags == frozenset(
        {om.COMMUTATIVE, om.POINTWISE_BINARY}
    )


def test_redeclaring_the_same_name_is_idempotent():
    opdata.declare_op(_VIEW_AS)
    opdata.declare_op(_VIEW_AS)
    assert opdata.reset_declarations() == ["view_as"]


# ---------------------------------------------------------------------------
#  The attr schema — mint validation, with and without required attrs
# ---------------------------------------------------------------------------


def test_mint_validation_enforces_the_declared_schema():
    opdata.declare_op(_SWAPAXES)
    x = _var("x", (2, 3))
    # a positional argN at an undeclared position is malformed.
    with pytest.raises(ValueError, match="no declared canonical name"):
        Op.make("swapaxes", x, arg5=0)
    # a fully-attributed term missing a required attr dies at mint.
    with pytest.raises(ValueError, match="missing required"):
        Op.make("swapaxes", x, dim0=0)
    # a bare term is a partial term and passes.
    assert Op.make("swapaxes", x).attrs == {}
    # the positional spelling at a declared position is preserved.
    assert Op.make("swapaxes", x, arg1=0, arg2=1).attrs == {
        "arg1": 0,
        "arg2": 1,
    }


def test_attrs_without_required_registers_a_schema_only():
    opdata.declare_op(
        opdata.OpDef(
            name="swapdims",
            arity=1,
            tags=frozenset({om.RELAYOUT}),
            attrs={1: "dim0", 2: "dim1"},
        )
    )
    assert ATTR_SCHEMA["swapdims"] == {1: "dim0", 2: "dim1"}
    assert "swapdims" not in ATTR_REQUIRED


# ---------------------------------------------------------------------------
#  Matching — a declared op is a first-class pattern/term op
# ---------------------------------------------------------------------------


def test_declared_op_matches_a_law_pattern():
    opdata.declare_op(_SWAPAXES)
    x = _var("x", (2, 3))
    pat = Op.make("swapaxes", "u", dim0=0, dim1=1)
    term = Op.make("swapaxes", x, dim0=0, dim1=1)
    assert match_pattern(pat, term) == {"u": x}


# ---------------------------------------------------------------------------
#  Serialization — a declaration *is* data
# ---------------------------------------------------------------------------


def test_opdef_json_roundtrip():
    data = opdata.opdef_to_data(_SWAPAXES)
    assert data["attrs"] == [[1, "dim0"], [2, "dim1"]]
    assert data["tags"] == [om.RELAYOUT, om.VIEWISH]  # sorted
    back = opdata.opdef_from_data(json.loads(json.dumps(data)))
    assert back == _SWAPAXES


def test_opdef_from_data_defaults_optional_fields():
    d = opdata.opdef_from_data({"name": "z"})
    assert d == opdata.OpDef(name="z")


def test_shape_codec_handles_nested_specs():
    spec = ("bcast", ("unsq-out", "arg0", "dim"), "arg1")
    encoded = opdata._shape_to_data(spec)
    assert encoded == ["bcast", ["unsq-out", "arg0", "dim"], "arg1"]
    assert opdata._shape_from_data(encoded) == spec
    assert opdata._shape_to_data(None) is None


# ---------------------------------------------------------------------------
#  Teardown — the seam is additive and reversible
# ---------------------------------------------------------------------------


def test_reset_restores_the_shipped_state():
    registry_before = set(om.REGISTRY)
    schema_before = set(ATTR_SCHEMA)
    rules_before = set(ty._SHAPE_RULES)
    opdata.declare_op(_SWAPAXES)
    assert "swapaxes" in om.REGISTRY
    assert "swapaxes" in ATTR_SCHEMA
    assert "swapaxes" in ty._SHAPE_RULES
    removed = opdata.reset_declarations()
    assert removed == ["swapaxes"]
    assert set(om.REGISTRY) == registry_before
    assert set(ATTR_SCHEMA) == schema_before
    assert set(ty._SHAPE_RULES) == rules_before
    assert om.meta("swapaxes") is None
    assert "swapaxes" not in ATTR_SCHEMA


def test_reset_with_nothing_declared_is_a_noop():
    assert opdata.reset_declarations() == []
