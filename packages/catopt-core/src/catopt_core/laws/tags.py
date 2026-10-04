"""Rule tags — the intrinsic classification of a rewrite rule.

Tags are **constants**, never bare strings: a typo cannot silently
miss a budget or a preset, and the vocabulary stays discoverable.
A rule carries them on :attr:`catopt_core.egraph.Rewrite.tags` —
they describe the rule's *nature* (intrinsic metadata); scheduling
priority is a :class:`~catopt_core.laws.RuleSet` concern instead.

* :data:`SIMPLIFICATION` — the basic algebraic simplifications
  (identities, inverses, small decompositions).
* :data:`CATEGORICAL` — the distributivity/naturality/product laws
  the search's insight lives in.
* :data:`FUSION` — structural folds that collapse a consumer pattern
  into one kernel (qkv / swiglu / sdpa / gqa).
* :data:`CARRIER` — the carrier-package law families (om / trace /
  the cross-carrier seams).
* :data:`SYMMETRY` — commutativity/associativity and the scale-hoist
  naturality rules: the Catalan-blowup closure generators.  Opt-in —
  never in :data:`~catopt_core.laws.DEFAULT`.
* :data:`EXPANSIVE` — anything whose saturation closure is
  combinatorially explosive (superset of ``SYMMETRY``); the pipeline
  applies the per-rule enode budget to tagged rules.
* :data:`SUBSUMED` — term-local fusion rules the non-local pairing
  pass replaces (``swiglu_fuse`` / ``parallel_mul_fuse`` /
  ``qkv_fuse`` / ``qkv_fuse_asym``).
* :data:`LAYOUT` — the transpose/layout migration laws.
* :data:`SCAN` — the core scan monoids (dense + diagonal affine).
* :data:`DECODE` — the decode/KV-cache carrier laws.
* :data:`ATTENTION` — the attention-path laws (rotary composition and
  scale commutation, the right-multiply absorb, the score-scale
  migration) — opt-in, never in
  :data:`~catopt_core.laws.DEFAULT`.
* :data:`REDUNDANT` — a derivable rule whose spelling is a literal
  alpha-duplicate of another shipped rule (the same 2-cell carried by
  two rule objects — e.g. ``weight_distribute_matmul`` is
  ``distribute_matmul_over_add`` under metavar renaming).  Combined
  with :attr:`~catopt_core.egraph.Rewrite.derivation` it completes the
  axiom/lemma taxonomy: ``rule.kind`` reports ``"axiom"`` /
  ``"lemma"`` / ``"redundant"``.  Annotation, not scheduling — the
  tag never joins a preset or a budget.
"""

SIMPLIFICATION = "simplification"
CATEGORICAL = "categorical"
FUSION = "fusion"
CARRIER = "carrier"
SYMMETRY = "symmetry"
EXPANSIVE = "expansive"
SUBSUMED = "subsumed"
LAYOUT = "layout"
SCAN = "scan"
DECODE = "decode"
ATTENTION = "attention"
REDUNDANT = "redundant"

__all__ = [
    "ATTENTION",
    "CARRIER",
    "CATEGORICAL",
    "DECODE",
    "EXPANSIVE",
    "FUSION",
    "LAYOUT",
    "REDUNDANT",
    "SCAN",
    "SIMPLIFICATION",
    "SUBSUMED",
    "SYMMETRY",
]
