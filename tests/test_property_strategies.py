"""Shared Hypothesis strategies for the ``test_property_*`` suite.

Not a test module: it only defines strategies/helpers, so pytest
collects nothing here.  Import with ``from
tests.test_property_strategies import ...``.
"""

from __future__ import annotations

import hypothesis.strategies as st
from catopt_core.ir import Const, Op, Param, TensorType, Var

#: ``op_repr`` is the leaf key and ``canonicalize``'s sort key, so a
#: leaf name must determine the leaf's type.  The name encodes the
#: shape (``v2_3`` = Var of shape ``(2, 3)``), giving distinct reprs
#: for distinct leaves and the same interned object for equal ones.
_SHAPES = st.lists(
    st.integers(min_value=1, max_value=4), min_size=0, max_size=3
).map(tuple)

#: Unary elementwise ops with no involution / identity interaction, so
#: term-level coherence rewrites act predictably on them.
UNARY = ["neg", "square", "exp", "sigmoid", "silu"]
BINARY = ["add", "mul", "sub", "div", "max"]

#: Unary ops that are NOT involutions (canonicalize never collapses a
#: pair of them), used by the uniform-shape cost-preservation property.
UNIFORM_UNARY = ["square", "exp", "sigmoid", "silu"]


@st.composite
def leaves(draw):
    if draw(st.booleans()):
        return Const(
            draw(
                st.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=-4.0,
                    max_value=4.0,
                )
            )
        )
    shape = draw(_SHAPES)
    tag = "_".join(str(d) for d in shape)
    typ = TensorType(shape)
    return (
        Var(f"v{tag}", typ)
        if draw(st.booleans())
        else Param(f"p{tag}", typ)
    )


def _extend(children):
    unary = st.builds(
        lambda op, a: Op.make(op, a),
        st.sampled_from(UNARY),
        children,
    )
    binary = st.builds(
        lambda op, a, b: Op.make(op, a, b),
        st.sampled_from(BINARY),
        children,
        children,
    )
    return st.one_of(unary, binary)


def terms(max_leaves: int = 8):
    """Random small IR terms (Var/Const/Param leaves + elementwise ops)."""
    return st.recursive(leaves(), _extend, max_leaves=max_leaves)


def uniform_terms(max_leaves: int = 8):
    """Terms whose every leaf has shape ``(2, 3)`` — elementwise ops keep
    that shape, so the term's FLOPs depend only on its op multiset."""

    def extend(children):
        unary = st.builds(
            lambda op, a: Op.make(op, a),
            st.sampled_from(UNIFORM_UNARY),
            children,
        )
        binary = st.builds(
            lambda op, a, b: Op.make(op, a, b),
            st.sampled_from(["add", "mul"]),
            children,
            children,
        )
        return st.one_of(unary, binary)

    base = st.builds(
        lambda n: Var(n, TensorType((2, 3))),
        st.sampled_from(["x", "y", "z", "w"]),
    )
    return st.recursive(base, extend, max_leaves=max_leaves)


@st.composite
def additive_terms(draw):
    """Left/right-nested ``add``/``mul`` chains over distinct ``(2, 3)``
    Vars — a bounded saturation domain for the e-graph properties."""
    n = draw(st.integers(min_value=1, max_value=5))
    vs = [Var(f"v{i}", TensorType((2, 3))) for i in range(n)]
    term: object = vs[0]
    for v in vs[1:]:
        op = draw(st.sampled_from(["add", "mul"]))
        term = (
            Op.make(op, term, v)
            if draw(st.booleans())
            else Op.make(op, v, term)
        )
    return term


def rebuild(t):
    """Structurally-equal reconstruction of *t* via ``Op.make``."""
    if isinstance(t, Op):
        return Op.make(
            t.op, *(rebuild(a) for a in t.args), **dict(t.attrs)
        )
    return t
