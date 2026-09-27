"""Property tests for rewrite soundness at the term level.

Three axes, all on randomly generated terms:

* the pattern machinery (``match_pattern`` / ``instantiate_pattern``)
  round-trips: instantiating a rule LHS with random concrete bindings
  and re-matching recovers the same term;
* ``canonicalize`` is idempotent, collapses commutativity/associativity
  and identity, and preserves FLOPs on uniform-shape terms;
* the coherent (comm/assoc/identity/involution) rewrites, applied
  anywhere via ``apply_rewrite_at``, preserve the canonical form.
"""

from __future__ import annotations

import hypothesis.strategies as st
from catopt_core.cost import flops_cost
from catopt_core.ir import Const, Op
from catopt_core.laws import (
    ALL_RULES,
    ASSOC_ADD,
    ASSOC_MUL,
    COMM_ADD,
    COMM_MUL,
    DOUBLE_NEG,
    ID_ADD,
    ID_MUL,
)
from catopt_core.meta import (
    _positions,
    apply_rewrite_at,
    canonicalize,
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from hypothesis import given, settings

from tests.test_property_strategies import terms, uniform_terms

_COHERENT = [
    ASSOC_ADD,
    COMM_ADD,
    ID_ADD,
    ASSOC_MUL,
    COMM_MUL,
    ID_MUL,
    DOUBLE_NEG,
]


@settings(max_examples=250, deadline=None)
@given(st.data())
def test_lhs_match_instantiate_roundtrip(data):
    rule = data.draw(st.sampled_from(ALL_RULES))
    subst = {}
    for mv in pattern_metavars(rule.lhs):
        subst[mv] = (
            data.draw(st.sampled_from([0, 1, 2, -1]))
            if mv.startswith("$attr:")
            else data.draw(terms(max_leaves=4))
        )
    term = instantiate_pattern(rule.lhs, subst)
    found = match_pattern(rule.lhs, term)
    assert found is not None
    assert instantiate_pattern(rule.lhs, found) == term


@settings(max_examples=250, deadline=None)
@given(terms())
def test_canonicalize_is_idempotent(t):
    once = canonicalize(t)
    assert canonicalize(once) == once


@given(terms(max_leaves=5), terms(max_leaves=5))
def test_canonicalize_collapses_commutativity(a, b):
    for op in ("add", "mul"):
        assert canonicalize(Op.make(op, a, b)) == canonicalize(
            Op.make(op, b, a)
        )


@given(terms(max_leaves=6))
def test_canonicalize_drops_identities(t):
    assert canonicalize(Op.make("add", t, Const(0))) == canonicalize(t)
    assert canonicalize(Op.make("mul", t, Const(1))) == canonicalize(t)


@settings(max_examples=150, deadline=None)
@given(uniform_terms())
def test_canonicalize_preserves_flops_on_uniform_shapes(t):
    assert flops_cost(canonicalize(t)) == flops_cost(t)


@settings(max_examples=250, deadline=None)
@given(terms(max_leaves=6), st.data())
def test_coherent_rewrites_preserve_canonical_form(t, data):
    rule = data.draw(st.sampled_from(_COHERENT))
    for path, _ in _positions(t):
        rewritten = apply_rewrite_at(rule, t, path)
        if rewritten is not None:
            assert canonicalize(rewritten) == canonicalize(t)
