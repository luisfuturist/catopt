"""Law-side content tables — the rule vocabulary as data.

``catopt_core.opmeta`` is the one home for the *op* vocabulary; this
module is the parallel home for the *law* vocabulary the search
machinery reads — pure data, no logic.  The direction (plan 0018 →
plan 0019 → the de-hardcoding retros) is that anything a learned
policy could plausibly choose is enumerable here rather than buried
as a Python constant inside an algorithm:

* :data:`COHERENT_RULE_NAMES` — the coherence classification
  ``catopt_core.meta.classify_rules`` splits on (which rules only
  generate equivalent bracketings).  A policy that re-learns "which
  laws are coherence" edits this table, never ``meta.py``.
* :data:`AC_IDENTITY` / :data:`ASSOC_ONLY` — the canonicalizer's
  monoid vocabulary: which ops carry a dropped identity, and which
  associative-but-not-commutative chains flatten and rebuild
  balanced.  Content the canonicalization policy reads; the
  flattening algorithm stays in ``meta.py``.
* :data:`INSTANTIATE_LEAF_SHAPES` / :data:`INSTANTIATE_ATTR_POOL` —
  the bounded instantiation bank the guarded-candidate validator
  enumerates (``meta._instantiation_stream``): which leaf-shape
  profiles and which attribute values a candidate's side conditions
  are tried against.  A learned enumerator's candidate set.

What is deliberately *not* here: budgets (``_MAX_INSTANTIATIONS``
and friends are resource knobs, not content), the flatten/rebuild
algorithms, and the law bodies themselves (``catopt_core.laws.*`` —
replayable ``Rewrite`` objects, a different record format).
"""

from __future__ import annotations

__all__ = [
    "AC_IDENTITY",
    "ASSOC_ONLY",
    "COHERENT_RULE_NAMES",
    "INSTANTIATE_ATTR_POOL",
    "INSTANTIATE_LEAF_SHAPES",
]

# ---------------------------------------------------------------------------
#  Coherence classification
# ---------------------------------------------------------------------------

#: Rules that express pure coherence: symmetry/associativity/identity/
#: involution.  They generate every *equivalent bracketing* of the same
#: computation — the search-space the e-graph should never store.
#: ``om_assoc``/``om_assoc_rev`` live in catopt_carriers.om (same law
#: family: associativity of a monoid's compose).
COHERENT_RULE_NAMES: frozenset[str] = frozenset(
    {
        "comm_add",
        "comm_mul",  # SMC symmetry
        "assoc_add",
        "assoc_mul",  # additive/multiplicative associativity
        "id_add",
        "id_mul",  # monoid units
        "double_neg",  # involution
        "assoc_matmul",
        "assoc_matmul_rev",  # associativity of composition
        "aff_assoc",
        "aff_assoc_rev",  # affine-map monoid associativity
        "affd_assoc",
        "affd_assoc_rev",  # diagonal-affine monoid assoc.
        "om_assoc",
        "om_assoc_rev",  # online-softmax monoid assoc.
    }
)

# ---------------------------------------------------------------------------
#  Canonicalizer vocabulary
# ---------------------------------------------------------------------------

#: Commutative + associative ops with a dropped identity element.
AC_IDENTITY: dict[str, float] = {"add": 0.0, "mul": 1.0}

#: Associative but NOT commutative ops: chains flatten in order and
#: rebuild balanced.  For ``aff_compose`` the balanced form is the
#: parallel-scan (Blelloch) bracketing — computed, not searched.
ASSOC_ONLY: frozenset[str] = frozenset(
    {"matmul", "aff_compose", "affd_compose"}
)

# ---------------------------------------------------------------------------
#  Candidate-instantiation bank
# ---------------------------------------------------------------------------

#: Leaf shapes tried when instantiating a candidate LHS for validation.
#: ``(4,4)`` satisfies shape-checked guards on rank-2 terms; ``()``
#: catches guards requiring scalar bindings.  A candidate whose guards
#: need a mixed/other profile is rejected — conservative, never
#: unsound.
INSTANTIATE_LEAF_SHAPES: tuple = ((4, 4), ())

#: Values enumerated for attribute metavariables in a candidate LHS
#: (dims first — they dominate; a few shapes for view-style attrs).
INSTANTIATE_ATTR_POOL: tuple = (
    -1,
    -2,
    1,
    2,
    0,
    -3,
    3,
    4,
    -4,
    (4, 4),
    (4,),
    (2, 4, 4),
)
