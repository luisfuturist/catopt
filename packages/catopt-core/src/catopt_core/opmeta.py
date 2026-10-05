"""Op metadata registry — one home for the op vocabulary.

The op *vocabulary* used to be hand-listed in a dozen modules: the
view-op sets (``cost._VIEW_OPS``, ``signature._VIEW_OPS``,
``pipeline._VIEW_OPS``, ``oracle._VIEWISH``), the pointwise sets
(``layout._POINTWISE_UNARY`` / ``_POINTWISE_BINARY``,
``signature._POINTWISE_OPS``, ``pipeline._POINTWISE``), the spine
inventories (``signature._ACT_OPS`` / ``_MUT_OPS`` / ``_TABLE_OPS``),
the reduction-axis set (``oracle._REDUCTION_DIM_OPS``) and the
operand-symmetric set (``typing._COMMUTATIVE_BROADCAST``).  The copies
drifted: ``pipeline._VIEW_OPS`` and ``signature._VIEW_OPS`` describe
the *same* concept and disagree, and adding an op meant editing N
files.

This module is the one home.  :data:`REGISTRY` is keyed by op name and
each entry carries the op's operand *arity* and its classification
*tags*; every consumer set is a **named projection** of the registry
(:func:`ops` / the ``*_OPS`` constants below), so a set can no longer
drift from its twin — they are the same data.

What the registry deliberately does *not* hold:

* the **attr schema** — ``catopt_core.attrs.ATTR_SCHEMA`` /
  ``ATTR_REQUIRED`` are already single-source; the registry *indexes*
  them (:func:`attrs` / :func:`required`) rather than copying;
* the **aten alias map** — ``catopt_torch.torch_bridge._ATEN_TO_IR``
  lives in the adapter (``catopt-core`` imports no torch) and is
  *validated against* the registry at the boundary
  (:func:`validate_aten_map`);
* the **numeric kernel bodies** — the FLOP formulas, the torch
  bindings and the shape rules stay code; only their metadata is data.

The tag vocabulary is closed: a new concept is a new tag here, never a
new hand list elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from catopt_core.attrs import ATTR_REQUIRED, ATTR_SCHEMA

__all__ = [
    "ACTIVATION",
    "ACTIVATION_OPS",
    "COMMUTATIVE",
    "COMMUTATIVE_OPS",
    "COST_VIEW",
    "COST_VIEW_OPS",
    "GENERATOR_VIEW",
    "GENERATOR_VIEW_OPS",
    "POINTWISE",
    "POINTWISE_BINARY",
    "POINTWISE_BINARY_OPS",
    "POINTWISE_OPS",
    "POINTWISE_UNARY",
    "POINTWISE_UNARY_OPS",
    "REDUCE_DIM",
    "REDUCE_DIM_OPS",
    "REGISTRY",
    "RELAYOUT",
    "RELAYOUT_OPS",
    "TABLE",
    "TABLE_OPS",
    "VIEWISH",
    "VIEWISH_OPS",
    "WRITE",
    "WRITE_OPS",
    "OpMeta",
    "arity",
    "attrs",
    "meta",
    "ops",
    "required",
    "tags_of",
    "validate_aten_map",
]

# ---------------------------------------------------------------------------
#  Tag vocabulary — one name per concept
# ---------------------------------------------------------------------------

#: Transparent relayout / read op — a single-child term that re-lays or
#: re-reads its input without computing on it (``signature._VIEW_OPS``).
RELAYOUT = "relayout"
#: Ops that emit no kernel — the cost model's zero-cost view set
#: (``cost._VIEW_OPS``).  A different axis from :data:`RELAYOUT`: it
#: adds carrier *packaging* (``aff``/``om``) and constant morphisms.
COST_VIEW = "cost-view"
#: The discovery generator's naturality alphabet — a documented
#: *subset* of :data:`RELAYOUT` (``pipeline._VIEW_OPS``).
GENERATOR_VIEW = "generator-view"
#: Views the numeric oracle can attribute-instantiate
#: (``oracle._VIEWISH``) — a subset of :data:`RELAYOUT`.
VIEWISH = "viewish"
#: Reductions whose ``dim`` attr accepts an axis or a tuple of axes
#: (``oracle._REDUCTION_DIM_OPS``).
REDUCE_DIM = "reduce-dim"
#: Multi-operand pointwise op — the "context position" test
#: (``signature._POINTWISE_OPS``).
POINTWISE = "pointwise"
#: Unary pointwise op that commutes with any operand permutation
#: (``layout._POINTWISE_UNARY``).
POINTWISE_UNARY = "pointwise-unary"
#: Binary pointwise op — the same-rank commutation family
#: (``layout._POINTWISE_BINARY`` / ``pipeline._POINTWISE``).
POINTWISE_BINARY = "pointwise-binary"
#: Activation on a block's spine (``signature._ACT_OPS``).
ACTIVATION = "activation"
#: arg0-mutating write, functionalised in the IR
#: (``signature._MUT_OPS``).
WRITE = "write"
#: Ops whose operand positions are all table/index roles
#: (``signature._TABLE_OPS``).
TABLE = "table"
#: Elementwise ops whose result is invariant under operand swap
#: (``typing._COMMUTATIVE_BROADCAST``).
COMMUTATIVE = "commutative"

# ---------------------------------------------------------------------------
#  The classification table — the one place the op sets are written down
# ---------------------------------------------------------------------------

#: ``op name -> classification tags``.  The single source every
#: ``*_OPS`` projection below is derived from.  An op absent from this
#: table is still registered if it carries an attr schema (seeded from
#: ``ATTR_SCHEMA``) or is a declared IR-only op (:data:`_IR_ONLY_OPS`).
_CLASSIFIED: dict[str, frozenset[str]] = {
    "abs": frozenset({POINTWISE_UNARY}),
    "add": frozenset({COMMUTATIVE, POINTWISE_BINARY}),
    "addcdiv": frozenset({POINTWISE}),
    "addcmul": frozenset({POINTWISE}),
    "aff": frozenset({COST_VIEW}),
    "aff_diag": frozenset({COST_VIEW}),
    "alias": frozenset({RELAYOUT}),
    "amax": frozenset({REDUCE_DIM}),
    "amin": frozenset({REDUCE_DIM}),
    "argmax": frozenset({REDUCE_DIM}),
    "argmin": frozenset({REDUCE_DIM}),
    "atan2": frozenset({POINTWISE}),
    "bfloat16": frozenset({RELAYOUT}),
    "bitwise_and": frozenset({POINTWISE}),
    "bitwise_or": frozenset({POINTWISE}),
    "broadcast": frozenset({COST_VIEW}),
    "broadcast_to": frozenset({RELAYOUT, VIEWISH}),
    "chunk": frozenset({COST_VIEW, RELAYOUT, VIEWISH}),
    "clamp": frozenset({POINTWISE}),
    "clamp_max": frozenset({POINTWISE}),
    "clamp_min": frozenset({POINTWISE}),
    "clone": frozenset({RELAYOUT}),
    "contiguous": frozenset({RELAYOUT}),
    "copy": frozenset({WRITE}),
    "count_nonzero": frozenset({REDUCE_DIM}),
    "cswap": frozenset({COST_VIEW}),
    "detach": frozenset({RELAYOUT}),
    "detach_": frozenset({RELAYOUT}),
    "div": frozenset({POINTWISE, POINTWISE_BINARY}),
    "double": frozenset({RELAYOUT}),
    "embedding": frozenset({TABLE}),
    "eq": frozenset({COMMUTATIVE, POINTWISE}),
    "exp": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "expand": frozenset({GENERATOR_VIEW, RELAYOUT, VIEWISH}),
    "expand_as": frozenset({RELAYOUT}),
    "eye": frozenset({COST_VIEW}),
    "flatten": frozenset({RELAYOUT, VIEWISH}),
    "flip": frozenset({RELAYOUT}),
    "float": frozenset({RELAYOUT}),
    "fmax": frozenset({POINTWISE}),
    "fmin": frozenset({POINTWISE}),
    "fmod": frozenset({POINTWISE}),
    "gather": frozenset({TABLE}),
    "ge": frozenset({POINTWISE}),
    "gelu": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "getitem": frozenset({RELAYOUT, VIEWISH}),
    "gt": frozenset({POINTWISE}),
    "half": frozenset({RELAYOUT}),
    "heaviside": frozenset({POINTWISE}),
    "index": frozenset({TABLE}),
    "index_add": frozenset({WRITE}),
    "index_put": frozenset({WRITE}),
    "index_select": frozenset({TABLE}),
    "isclose": frozenset({POINTWISE}),
    "item": frozenset({TABLE}),
    "le": frozenset({POINTWISE}),
    "leaf": frozenset({COST_VIEW}),
    "lerp": frozenset({POINTWISE}),
    "linalg_vector_norm": frozenset({REDUCE_DIM}),
    "log": frozenset({POINTWISE_UNARY}),
    "logical_and": frozenset({POINTWISE}),
    "logical_or": frozenset({POINTWISE}),
    "logical_xor": frozenset({POINTWISE}),
    "lt": frozenset({POINTWISE}),
    "masked_fill": frozenset({POINTWISE}),
    "max": frozenset({REDUCE_DIM}),
    "maximum": frozenset({POINTWISE}),
    "mean": frozenset({REDUCE_DIM}),
    "median": frozenset({REDUCE_DIM}),
    "min": frozenset({REDUCE_DIM}),
    "minimum": frozenset({POINTWISE}),
    "mode": frozenset({REDUCE_DIM}),
    "movedim": frozenset({RELAYOUT, VIEWISH}),
    "mul": frozenset({COMMUTATIVE, POINTWISE, POINTWISE_BINARY}),
    "nanmean": frozenset({REDUCE_DIM}),
    "nansum": frozenset({REDUCE_DIM}),
    "narrow": frozenset({RELAYOUT, VIEWISH}),
    "ne": frozenset({COMMUTATIVE, POINTWISE}),
    "neg": frozenset({POINTWISE_UNARY}),
    "nonzero": frozenset({TABLE}),
    "numel": frozenset({TABLE}),
    "om": frozenset({COST_VIEW}),
    "one_hot": frozenset({TABLE}),
    "pad": frozenset({RELAYOUT}),
    "permute": frozenset({RELAYOUT, VIEWISH}),
    "pow": frozenset({POINTWISE}),
    "prod": frozenset({REDUCE_DIM}),
    "relu": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "remainder": frozenset({POINTWISE}),
    "repeat": frozenset({RELAYOUT}),
    "reshape": frozenset(
        {COST_VIEW, GENERATOR_VIEW, RELAYOUT, VIEWISH}
    ),
    "roll": frozenset({RELAYOUT}),
    "rsqrt": frozenset({POINTWISE_UNARY}),
    "scatter": frozenset({WRITE}),
    "scatter_add": frozenset({WRITE}),
    "scatter_reduce": frozenset({WRITE}),
    "sdpa": frozenset({ACTIVATION}),
    "searchsorted": frozenset({TABLE}),
    "select": frozenset({GENERATOR_VIEW, RELAYOUT, VIEWISH}),
    "select_scatter": frozenset({WRITE}),
    "sigmoid": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "silu": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "slice": frozenset({GENERATOR_VIEW, RELAYOUT, VIEWISH}),
    "slice_scatter": frozenset({WRITE}),
    "softmax": frozenset({ACTIVATION}),
    "split": frozenset({COST_VIEW, RELAYOUT, VIEWISH}),
    "sqrt": frozenset({POINTWISE_UNARY}),
    "square": frozenset({POINTWISE_UNARY}),
    "squeeze": frozenset({GENERATOR_VIEW, RELAYOUT, VIEWISH}),
    "std": frozenset({REDUCE_DIM}),
    "std_mean": frozenset({REDUCE_DIM}),
    "sub": frozenset({POINTWISE_BINARY}),
    "sum": frozenset({REDUCE_DIM}),
    "take_along_dim": frozenset({TABLE}),
    "tanh": frozenset({ACTIVATION, POINTWISE_UNARY}),
    "tensor_split": frozenset({RELAYOUT}),
    "to": frozenset({RELAYOUT}),
    "transpose": frozenset(
        {COST_VIEW, GENERATOR_VIEW, RELAYOUT, VIEWISH}
    ),
    "tril": frozenset({RELAYOUT}),
    "triu": frozenset({RELAYOUT}),
    "type_as": frozenset({RELAYOUT}),
    "unbind": frozenset({RELAYOUT, VIEWISH}),
    "unflatten": frozenset({RELAYOUT}),
    "unsqueeze": frozenset({GENERATOR_VIEW, RELAYOUT, VIEWISH}),
    "var": frozenset({REDUCE_DIM}),
    "var_mean": frozenset({REDUCE_DIM}),
    "view": frozenset({RELAYOUT, VIEWISH}),
    "where": frozenset({POINTWISE}),
    "xlogy": frozenset({POINTWISE}),
}

#: Declared operand arity (operand count), where it is a defining
#: property of the classification.  An op absent here has an
#: undeclared arity (:func:`arity` returns ``None``) — most IR ops
#: take one tensor plus attrs, but the registry does not guess.
_ARITY: dict[str, int] = {
    # relayout — one operand, bar the two-operand re-layouts
    "slice": 1,
    "select": 1,
    "narrow": 1,
    "getitem": 1,
    "unsqueeze": 1,
    "squeeze": 1,
    "reshape": 1,
    "view": 1,
    "expand": 1,
    "expand_as": 2,
    "broadcast_to": 1,
    "permute": 1,
    "transpose": 1,
    "flatten": 1,
    "unflatten": 1,
    "movedim": 1,
    "contiguous": 1,
    "detach": 1,
    "detach_": 1,
    "clone": 1,
    "to": 1,
    "type_as": 2,
    "float": 1,
    "double": 1,
    "half": 1,
    "bfloat16": 1,
    "repeat": 1,
    "chunk": 1,
    "split": 1,
    "tensor_split": 1,
    "unbind": 1,
    "roll": 1,
    "flip": 1,
    "pad": 1,
    "triu": 1,
    "tril": 1,
    "alias": 1,
    # cost-view
    "broadcast": 1,
    "leaf": 0,
    "aff": 2,
    "om": 3,
    "aff_diag": 2,
    "eye": 0,
    "cswap": 0,
    # reduce-dim
    "sum": 1,
    "mean": 1,
    "prod": 1,
    "amax": 1,
    "amin": 1,
    "max": 1,
    "min": 1,
    "argmax": 1,
    "argmin": 1,
    "median": 1,
    "mode": 1,
    "var": 1,
    "std": 1,
    "var_mean": 1,
    "std_mean": 1,
    "nansum": 1,
    "nanmean": 1,
    "count_nonzero": 1,
    "linalg_vector_norm": 1,
    # pointwise families
    "add": 2,
    "mul": 2,
    "sub": 2,
    "div": 2,
    "neg": 1,
    "abs": 1,
    "silu": 1,
    "relu": 1,
    "sigmoid": 1,
    "tanh": 1,
    "gelu": 1,
    "exp": 1,
    "sqrt": 1,
    "rsqrt": 1,
    "square": 1,
    "log": 1,
}

#: IR ops the exporter can mint that carry neither an attr schema nor a
#: classification tag.  Declared so the adapter's alias map validates
#: against the registry (:func:`validate_aten_map`) rather than
#: silently minting an unregistered op.
_IR_ONLY_OPS: tuple[str, ...] = (
    "conv1d",
    "diag_sum",
    "gammaln",
    "inv",
    "linear",
    "matmul",
)


@dataclass(frozen=True)
class OpMeta:
    """The metadata the registry holds for one op.

    Attributes
    ----------
    name : str
        The canonical IR op spelling — the lowering token the sink
        dispatches on.
    arity : int | None
        The operand count where it is a defining property; ``None``
        when undeclared.
    tags : frozenset[str]
        The op's classification — a subset of the tag vocabulary.

    The attr schema is *not* a field: :func:`attrs` / :func:`required`
    index ``ATTR_SCHEMA`` / ``ATTR_REQUIRED`` (already single-source),
    so the schema has exactly one home.

    """

    name: str
    arity: int | None
    tags: frozenset[str]


def _build_registry() -> dict[str, OpMeta]:
    """Compose the registry from its three declared sources.

    The attr-schema ops (so every schema'd op is registered), the
    classification table, and the IR-only vocabulary.
    """
    names = set(ATTR_SCHEMA) | set(_CLASSIFIED) | set(_IR_ONLY_OPS)
    return {
        name: OpMeta(
            name=name,
            arity=_ARITY.get(name),
            tags=_CLASSIFIED.get(name, frozenset()),
        )
        for name in names
    }


#: ``op name -> OpMeta``.  Immutable — a mapping proxy over the built
#: dict, so a caller cannot mutate the single source.
REGISTRY: Mapping[str, OpMeta] = MappingProxyType(_build_registry())


# ---------------------------------------------------------------------------
#  Accessors
# ---------------------------------------------------------------------------


def meta(name: str) -> OpMeta | None:
    """Return the :class:`OpMeta` for *name*, or ``None`` if absent."""
    return REGISTRY.get(name)


def ops(tag: str) -> frozenset[str]:
    """Return every op carrying *tag* (the projection seam)."""
    return frozenset(
        name for name, m in REGISTRY.items() if tag in m.tags
    )


def tags_of(name: str) -> frozenset[str]:
    """Return *name*'s tags, or the empty set if it is not registered."""
    m = REGISTRY.get(name)
    return m.tags if m is not None else frozenset()


def arity(name: str) -> int | None:
    """Return the declared operand count of *name*, or ``None``."""
    m = REGISTRY.get(name)
    return m.arity if m is not None else None


def attrs(name: str) -> Mapping[int, str]:
    """Return *name*'s positional-attr schema (``ATTR_SCHEMA``).

    The registry indexes the schema; it never copies it, so there is
    one home for the canonical attr names.
    """
    return ATTR_SCHEMA.get(name, {})


def required(name: str) -> frozenset[str]:
    """Return *name*'s required canonical attrs (``ATTR_REQUIRED``)."""
    return ATTR_REQUIRED.get(name, frozenset())


# ---------------------------------------------------------------------------
#  Named projections — the consumer sets, derived not hand-listed
# ---------------------------------------------------------------------------

#: Transparent relayout ops (``signature._VIEW_OPS``).
RELAYOUT_OPS: frozenset[str] = ops(RELAYOUT)
#: Zero-kernel ops for the cost model (``cost._VIEW_OPS``).
COST_VIEW_OPS: frozenset[str] = ops(COST_VIEW)
#: The discovery generator's view alphabet (``pipeline._VIEW_OPS``).
GENERATOR_VIEW_OPS: frozenset[str] = ops(GENERATOR_VIEW)
#: Oracle-instantiable views (``oracle._VIEWISH``).
VIEWISH_OPS: frozenset[str] = ops(VIEWISH)
#: Axis-or-tuple reductions (``oracle._REDUCTION_DIM_OPS``).
REDUCE_DIM_OPS: frozenset[str] = ops(REDUCE_DIM)
#: Multi-operand pointwise ops (``signature._POINTWISE_OPS``).
POINTWISE_OPS: frozenset[str] = ops(POINTWISE)
#: Unary pointwise commutation family
#: (``layout._POINTWISE_UNARY``).
POINTWISE_UNARY_OPS: frozenset[str] = ops(POINTWISE_UNARY)
#: Binary pointwise commutation family
#: (``layout._POINTWISE_BINARY`` / ``pipeline._POINTWISE``).
POINTWISE_BINARY_OPS: frozenset[str] = ops(POINTWISE_BINARY)
#: Spine activations (``signature._ACT_OPS``).
ACTIVATION_OPS: frozenset[str] = ops(ACTIVATION)
#: Functionalised arg0-mutating writes (``signature._MUT_OPS``).
WRITE_OPS: frozenset[str] = ops(WRITE)
#: Lookup-table operand roles (``signature._TABLE_OPS``).
TABLE_OPS: frozenset[str] = ops(TABLE)
#: Operand-symmetric elementwise ops
#: (``typing._COMMUTATIVE_BROADCAST``).
COMMUTATIVE_OPS: frozenset[str] = ops(COMMUTATIVE)


# ---------------------------------------------------------------------------
#  Boundary validation
# ---------------------------------------------------------------------------


def validate_aten_map(mapping: Mapping[str, str]) -> list[str]:
    """Return the aten→IR targets of *mapping* absent from the registry.

    The adapter's alias map (``torch_bridge._ATEN_TO_IR``) is the one
    op-name table that legitimately lives outside core — it is the
    torch boundary, and ``catopt-core`` imports no torch.  This is the
    check that keeps it honest: every IR op the exporter can mint must
    be a registered op, so an alias cannot introduce a spelling the
    rest of the engine has no metadata for.  Returns the sorted
    offending names (empty when clean).
    """
    missing = {
        target for target in mapping.values() if target not in REGISTRY
    }
    return sorted(missing)
