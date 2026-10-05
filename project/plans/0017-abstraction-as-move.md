# Plan 0017 — abstraction as a legal move

Status: starting.  Implements ADR 0004 — the unit of invention is a
constructed *object*, declared as serializable data and admitted by
the adversarial referee, never a minted rewrite the policy hands to
a trusted seam.

## Stages

| Stage | What | Depends on |
|---|---|---|
| 0 | `derive=` toward data — extend `dspec` to cover the remaining procedural derives ("pull dim from tuple", attr arithmetic); count the serializable total | `laws/cond.py`, `dspec` machinery |
| 1 | **The object record** — generalize `lemmas`/`law_to_data`/`law_from_data` into a declared-object record (pattern + cond + derive + tags + cert) so a stored *abstraction* reconstructs into a live, firing object the same way `--admit` reconstructs a stored lemma | `laws/serialize.py`, `evidence.py` |
| 2 | **The guide seam** — `meta_game`'s real role: the policy chooses *which generator invests compute where* (census/vocab/oracle/generator passes), not which rewrite to fire.  Wire the action space to the generator inventory; measure against the enumeration baseline | `meta_game.py`, `pipeline.py` |
| 3 | **Adversarial admission for synthesized objects** — a stored object passes numeric oracle → view oracle → typed-pay gate → cert replay before it's usable; precondition must exist as a `cond` (the `w:v_commutes_view`/`id:same_pairing` gaps are the standing `cond` work) | stage 1, the oracles |

## Honest boundaries carried in

- A synthesized abstraction whose `cond` can't be written
  declaratively is not admissible as data — it goes through the same
  Python-hook review path as a hand-written law.  No side-door.
- `derive=`-as-data is bounded — attribute arithmetic beyond the DSL
  stays procedural; the serializable total is reported honestly, not
  forced.
- The verifier stays the authority at every stage: RL guides
  generator investment, never mints semantics (ADR 0002 Rule 3).

## Exit criteria

A stored, serializable *introduced object* — declared as data — that
the pipeline admits through the adversarial gauntlet and that fires
identically to its declared semantics, with a replayable certificate.
The first candidate is an object the machinery already understands
(a carrier-style abstraction or a conditional view law from the
oracle's guard table), not a novel one — the seam is what's new, not
the object.
