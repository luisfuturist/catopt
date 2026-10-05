# ADR 0004 — Abstraction as a legal move

**Status**: proposed.
**Date**: 2026-10.
**Depends on**: ADR 0002 (the thesis), ADR 0003 (the four dimensions).
**Related**: plan 0017 (implementation).

## Context

The game today is **fixed objects × fixed rules**: tensor terms (the
objects) rewritten by a static law library (the moves).  Two facts
about it are now measured, not hypothetical:

1. **The richer move already exists, in embryo.**  Carriers lift a
   *runtime abstraction* into the e-graph (`carrierize` introduces the
   online-softmax monoids, `omd`, the affine scans as first-class
   objects the laws then operate on).  The fold laws close the other
   direction: a manually-spelled composition
   (`div(exp, sum(exp))`, `x·rsqrt(mean x²)`, `chunk+sigmoid`) is
   folded into one dispatched kernel — "expose a representation" as a
   rewrite.  Every shipped machine-found law is one of these two
   shapes: a **fold** or a **bridge**.  None is an arbitrary rewrite.
2. **The naive version of this ADR is falsified.**  Asking a learned
   policy to invent laws directly was measured ~10× worse than
   enumeration (retro `learned-proposal` results; `meta_game` is the
   probe).  Direct neural law invention does not scale — but
   *structured construction* demonstrably does: the discovery pipeline
   (census → propose → verify → measure → rank) has shipped six fold
   laws by composing a small vocabulary of schemas over the corpus.

Meanwhile the serialization layer caught up: a law is now **data** —
pattern + `cond`/`dspec` guards + tags + `derivation` — 57/61 rules
serialize completely, and `derivation=` replays into real
`Certificate`s (`law_lemma_cert`, `--admit`).  The one scar is
`derive=`: computing RHS attrs the LHS lacks stays procedural (a
Python hook that cannot serialize).

The pressure this ADR answers: **what is a move?**  If the answer
stays "a pre-approved `R(...)` in `laws/*.py`", then the search space
is closed under human authorship — the machine can only reorder what
it is given.  The measured alternative: let the *object language*
grow — declare an abstraction as data, let the referee admit it, let
search use it.

## Decision

**The unit of invention is a constructed object, not an arbitrary
rewrite — and an admissible abstraction is serializable data the
referee replays, not Python hooks the policy mints.**

Concretely:

1. **Objects-as-data.**  An "introduced abstraction" is a record:
   its pattern/structure, its `cond` precondition, its `derive`
   semantics, its tags — the same shape a serialized law already
   takes (`LAW_FORMAT` v2).  The `lemmas` store generalizes from
   "verdicts + law records" to **declared objects**: the evidence DB
   becomes the theorem store.  An object whose conditions cannot be
   written declaratively is *not admissible as data* — it falls back
   to the existing Python-hook path with the same review bar a
   hand-written `R(...)` faces today, never a privileged side-door.
2. **`derive=` moves toward data.**  Most shipped derives are
   "pull the dim out of the tuple" / "copy the attr" — a small DSL
   (`dspec` already exists for the shape side).  Extending it to
   cover the remaining procedural derives is the honest blocker on
   fully-serializable introduced objects; plan 0017 takes it first.
3. **The player's action space is *construction operations*, not
   rewrites.**  `introduce abstraction | fold representation |
   compose abstractions | specialize runtime | bridge spellings` —
   the same operations the discovery machinery already performs by
   hand.  The policy chooses *which generator invests compute where*
   (the `meta_game` probe's actual role); it never mints a semantics
   string the referee hasn't admitted.  **RL stays the guide, not
   the source of truth** — ADR 0002 Rule 3 survives unchanged.
4. **The verifier for a synthesized abstraction is the adversarial
   layer, not instance-checking.**  An admitted object must clear
   the same gauntlet a shipped law clears now: numeric oracle →
   view/index oracle → typed-pay gate → certificate replay.  The
   `select_mul` unsoundness is the standing lesson that "verified on
   all observed sites" is not "sound" — an introduced object carries
   its precondition as a `cond`, and the sweep layer tests the
   false region, not just the true one.
5. **The four dimensions hold.**  Semantics decides equivalence;
   search decides what to explore (now including which constructions
   to attempt); evaluation prices it; execution runs it.  A
   synthesized abstraction is a *search* artifact that must pass the
   *semantics* gate — exactly like a law today, not a new class of
   trusted input.

## Consequences

- **The object language grows at the rate the verifier admits.**
  Corpus and oracle coverage bound what can be honestly introduced —
  a synthesized abstraction without a satisfiable `cond` is rejected
  before it can mint an ill-typed program (the typed-pay gate is the
  standing implementation of this rule).
- **Serialization becomes the admission format.**  A stored object
  replays into a live `Rewrite`/`Op` the way `--admit` already
  reconstructs a stored lemma; the seam exists, the record type
  generalizes.
- **`meta_game` graduates from probe to seam.**  Its action space
  was already "which generator" rather than "which rewrite"; the
  guide wires in where the design already expected it.
- **Honest scars carried forward.**  `derive=`-as-data is bounded
  (attr arithmetic beyond the DSL stays procedural); the
  12 documented unreachable-defensive lines and the `cond` gaps
  (`w:v_commutes_view`/`id:same_pairing` need the broadcast-grid
  predicates the DSL just gained) are the standing limitations.

## Status of this ADR

Proposed — the design direction, not the implementation.  Plan 0017
sequences the work: derive-as-data first, the object record second,
the guide third, the adversarial admission path throughout.
