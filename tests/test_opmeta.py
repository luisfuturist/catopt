"""Tests for ``catopt_core.opmeta`` — the op-metadata registry.

The registry is the single source the op *vocabulary* sets derive
from.  These tests pin three things:

* the registry's own composition (schema ops registered, projections
  are exactly the tag filter);
* the **cross-module agreement** that used to be folklore — the two
  ``_VIEW_OPS`` share one source, the derived subsets nest, the
  defining-property arities hold, and the adapter's aten alias map
  introduces no unregistered IR op;
* the consumer modules actually *read* the registry (identity, not a
  copy), so a set can no longer drift from its twin.
"""

import pytest
from catopt_core import opmeta as om
from catopt_core.attrs import ATTR_REQUIRED, ATTR_SCHEMA

# ---------------------------------------------------------------------------
#  Registry composition
# ---------------------------------------------------------------------------


def test_registry_is_immutable():
    with pytest.raises(TypeError):
        om.REGISTRY["nope"] = None


def test_every_schema_op_is_registered():
    assert set(ATTR_SCHEMA) <= set(om.REGISTRY)


def test_attrs_and_required_index_the_schema():
    # Indexed, not copied: the schema keeps exactly one home.
    assert om.attrs("transpose") is ATTR_SCHEMA["transpose"]
    assert om.required("transpose") == ATTR_REQUIRED["transpose"]
    assert om.attrs("not_an_op") == {}
    assert om.required("not_an_op") == frozenset()


def test_projections_are_the_tag_filter():
    for tag in (
        om.RELAYOUT,
        om.COST_VIEW,
        om.GENERATOR_VIEW,
        om.VIEWISH,
        om.REDUCE_DIM,
        om.POINTWISE,
        om.POINTWISE_UNARY,
        om.POINTWISE_BINARY,
        om.ACTIVATION,
        om.WRITE,
        om.TABLE,
        om.COMMUTATIVE,
    ):
        assert om.ops(tag) == frozenset(
            n for n, m in om.REGISTRY.items() if tag in m.tags
        )


def test_named_projection_constants_match_ops():
    assert om.ops(om.RELAYOUT) == om.RELAYOUT_OPS
    assert om.ops(om.COST_VIEW) == om.COST_VIEW_OPS
    assert om.ops(om.GENERATOR_VIEW) == om.GENERATOR_VIEW_OPS
    assert om.ops(om.VIEWISH) == om.VIEWISH_OPS
    assert om.ops(om.REDUCE_DIM) == om.REDUCE_DIM_OPS
    assert om.ops(om.POINTWISE) == om.POINTWISE_OPS
    assert om.ops(om.POINTWISE_UNARY) == om.POINTWISE_UNARY_OPS
    assert om.ops(om.POINTWISE_BINARY) == om.POINTWISE_BINARY_OPS
    assert om.ops(om.ACTIVATION) == om.ACTIVATION_OPS
    assert om.ops(om.WRITE) == om.WRITE_OPS
    assert om.ops(om.TABLE) == om.TABLE_OPS
    assert om.ops(om.COMMUTATIVE) == om.COMMUTATIVE_OPS


def test_accessors_handle_unknown_ops():
    assert om.meta("not_an_op") is None
    assert om.tags_of("not_an_op") == frozenset()
    assert om.arity("not_an_op") is None
    assert om.ops("not_a_tag") == frozenset()


def test_meta_carries_arity_and_tags():
    m = om.meta("transpose")
    assert m is not None
    assert m.name == "transpose"
    assert m.arity == 1
    assert om.RELAYOUT in m.tags
    # An undeclared arity stays None rather than being guessed.
    assert om.arity("matmul") is None


# ---------------------------------------------------------------------------
#  Cross-module agreement — the point of the registry
# ---------------------------------------------------------------------------


def test_generator_view_is_a_relayout_subset():
    """The two ``_VIEW_OPS`` share one source; the generator uses a
    documented subset of the transparent-relayout set."""
    assert om.GENERATOR_VIEW_OPS <= om.RELAYOUT_OPS
    assert om.VIEWISH_OPS <= om.RELAYOUT_OPS


def test_view_sets_agree_across_modules():
    """``pipeline._VIEW_OPS`` and ``signature._VIEW_OPS`` are the same
    *source*, and the generator's alphabet is a subset of it — the
    measured inconsistency the registry removes."""
    from catopt_discovery import pipeline as pl
    from catopt_orchestrator.morphisms import signature as sig

    assert set(pl._VIEW_OPS) <= set(sig._VIEW_OPS)
    assert set(pl._VIEW_OPS) == set(om.GENERATOR_VIEW_OPS)
    assert set(sig._VIEW_OPS) == set(om.RELAYOUT_OPS)


def test_consumers_read_the_registry_object():
    """Each consumer set *is* the registry projection — identity, not
    a copy, so a twin cannot drift."""
    from catopt_core import typing as ty
    from catopt_core.cost import basic as cost_basic
    from catopt_core.laws import layout
    from catopt_discovery import oracle as orc
    from catopt_discovery import pipeline as pl
    from catopt_orchestrator.morphisms import signature as sig

    assert cost_basic._VIEW_OPS is om.COST_VIEW_OPS
    assert ty._COMMUTATIVE_BROADCAST is om.COMMUTATIVE_OPS
    assert sig._VIEW_OPS is om.RELAYOUT_OPS
    assert sig._POINTWISE_OPS is om.POINTWISE_OPS
    assert sig._ACT_OPS is om.ACTIVATION_OPS
    assert sig._MUT_OPS is om.WRITE_OPS
    assert sig._TABLE_OPS is om.TABLE_OPS
    assert orc._VIEWISH is om.VIEWISH_OPS
    assert orc._REDUCTION_DIM_OPS is om.REDUCE_DIM_OPS
    assert set(layout._POINTWISE_UNARY) == set(om.POINTWISE_UNARY_OPS)
    assert set(layout._POINTWISE_BINARY) == set(om.POINTWISE_BINARY_OPS)
    assert set(pl._POINTWISE) == set(om.POINTWISE_BINARY_OPS)


def test_defining_property_arities_hold():
    for op in om.GENERATOR_VIEW_OPS:
        assert om.arity(op) == 1, op
    for op in om.VIEWISH_OPS:
        assert om.arity(op) == 1, op
    for op in om.REDUCE_DIM_OPS:
        assert om.arity(op) == 1, op
    for op in om.POINTWISE_BINARY_OPS:
        assert om.arity(op) == 2, op
    for op in om.POINTWISE_UNARY_OPS:
        assert om.arity(op) == 1, op
    # A relayout is one operand, bar the two-operand re-layouts.
    for op in om.RELAYOUT_OPS:
        assert om.arity(op) in (1, 2), op
    assert om.arity("expand_as") == 2
    assert om.arity("type_as") == 2


def test_commutative_ops_are_broadcast_symmetric():
    # The registry agrees with the shape-symmetry it documents.
    assert frozenset({"add", "mul", "eq", "ne"}) == om.COMMUTATIVE_OPS


# ---------------------------------------------------------------------------
#  The adapter boundary — the aten alias map
# ---------------------------------------------------------------------------


def test_aten_alias_map_mints_only_registered_ops():
    """Every IR op the exporter can mint has registry metadata.

    The alias map legitimately lives in the adapter (core is
    torch-free); this is the check that keeps it honest.
    """
    from catopt_torch.torch_bridge import (
        _ATEN_TO_IR,
        _IR_TO_TORCH_EXTRA,
    )

    merged = {**_ATEN_TO_IR, **_IR_TO_TORCH_EXTRA}
    assert om.validate_aten_map(merged) == []


def test_validate_aten_map_reports_offenders():
    assert om.validate_aten_map({"x": "not_an_op"}) == ["not_an_op"]
    assert om.validate_aten_map({"x": "add"}) == []
