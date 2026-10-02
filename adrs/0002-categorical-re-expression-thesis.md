# ADR 0002 — The categorical re-expression thesis

**Status**: accepted.
**Date**: 2026-09.
**Amended by**: ADR 0003 — Rule 3 only.  RL remains a non-goal as a
*mechanism of reach*; it is allowed as a replaceable *policy* over the
structure (the four-dimension split).

## Context

The project's capabilities grew independently — carrier monoids,
non-local lifts, the morphism engine, bounded-error rewrites, exact
weight sharing — each justified on its own and each documented in a
different place (README, REPORT.md, RESEARCH_WRITEUP.md, plans
0009–0013). Two things now force a single anchor:

1. ADR 0001 falsified weight-space structure, and the structure
   census showed exploitable structure is **architectural** (GQA
   replication, ties, MoE, adapters), not numerical.
2. Plan 0013 proposes a generalization (contraction search) that
   reads as a *third* competing generality mechanism alongside the
   term-level law library (0009) and the morphism engine (0011).

Without one stated thesis the docs drift: the pitch still leads
with "compression with a receipt" while ADR 0001 closed weight
compression, and exact/bounded modes read as two features rather
than one object.

## Decision

Adopt one thesis; make every capability answer to it.

**Thesis.** A tensor program is a morphism in a (hyper)graph
monoidal category. Optimization is search in the quotient of the
free category on the IR's generators, by the laws of the structures
the program instantiates — monoids, comonoids, traces, contraction.
Op-level compilers rewrite *generators*; this rewrites *morphisms up
to structure*, which is why it reaches programs they provably
cannot.

The interpretation is **not unique**, and choosing it is the creative
act: the same graph reads as plain composition (nothing to do), a
shared-input comonoid (pairing), a low-rank approximation (bounded),
or a contraction problem — each exposing different laws. So
**discovery is choosing the richest interpretation**, not applying
laws inside a fixed one. Reach is the **closure** of the generators:
the law library is the generating cells, and everything they
generate is the reachable set.

**Goal.** Given a model and its weights, automatically produce a
cheaper program that computes the same function — or one provably
within an `error_budget` — and return it with a replayable
certificate. Weights are source material: reused exactly (tying,
folding, GQA replication) or derived within the bound (KV sharing,
head merging, factoring). The **function** is fixed, not the
weights.

**Mechanism.** Lift the model to the categorical structure — choose
the richest structural interpretation — search structural moves on
that small structure, and verify each candidate by certificate
before delivery. Construction from recognized structure is the
primary path; saturation is the fallback.

**Objective.** Minimize cost (memory, compute, decode latency)
subject to the certificate constraint. The constraint is a hard
filter — accept/reject — not a soft reward.

Six rules follow and are binding:

1. **Name your structure.** Every capability must name the
   categorical structure it exploits. A feature that cannot name
   one is not in the thesis and does not ship under it.
2. **The certificate is the artifact.** The deliverable is a
   cheaper program *and* its proof — exact equivalence or a
   certified bound, never silent approximation.
3. **Directed search, not learning.** You never enumerate the
   space: evaluate states (verifier + cost) and prune (cost bound,
   decomposition). That handles exponential joint move spaces — the
   Catalan bracketing search already does. RL is a non-goal:
   structure is what makes search beat it.  *(Amended by ADR 0003:
   this holds for **reach**; a learned **policy** selecting among the
   structure's rewrites is now permitted.)*
4. **Construction primary, saturation fallback.** Where the
   structure names the target, build the member directly
   (`lift_scan_to_applyd`); saturation is the completeness
   fallback, and its truncation (budgets) can miss, never
   mis-prove.
5. **Exact and bounded are one object.** Exact rewrites are the
   zero-error elements of a category enriched over (cost, error);
   bounded rewrites are lax morphisms carrying a 2-cell. The two
   modes are not separate features.
6. **Scope claims to structure.** Parity where Inductor fuses
   well; wins where work is reduced or Inductor cannot codegen.
   Honest declines, never a wrong answer.

## Capability map

| Capability | Structure | Reaches |
|---|---|---|
| Carriers `aff`/`om`/`trace` | monoid + homomorphism + trace | Blelloch scan, streaming softmax, resolvents |
| Pairing / QKV merge | comonoid `Δ` + naturality | one GEMM + split views |
| Weight-chain fold | contraction order | k-chain → 1 GEMM |
| Morphism engine (0011) | block-level signature algebra | cross-layer windows, KV latent share |
| Contraction search (0013) | hypergraph contraction order | discovered sharing |
| Bounded / `error_budget` (0012) | enriched / lax morphism | certified approximation |
| Tying / CSE | congruence in the free category | parameter dedup |

## Higher structure (the engine's justification)

The e-graph plus certificate machinery is a computational higher
category: 1-cells are programs, 2-cells are rewrites (the merge log
/ witnesses / certificate steps), 3-cells are coherence —
`_close_congruence` unions enodes that became identical after child
canonicalisation, i.e. asserts two construction paths are the same
cell. Saturation is an ∞-tower that does not terminate, so plan
0010's budgeted expansive rules are a *truncation*, and a truncation
can never emit a false cell. This is the internal justification for
both certificate composition (2-cells compose vertically and
horizontally) and the budget mechanism — **not** a headline claim.

## Alternatives considered

- **Reachability only.** Keep the writeup's framing (structure
  generates + names) without a re-expression goal. Rejected: it
  states why the reach exists but not what the project is for,
  leaving the docs free to drift.
- **Weight-space compression** (the original bet). Falsified by
  ADR 0001 and Phase 5; retained only as the structural corner.
- **Higher category as the public headline.** Rejected: it must do
  work in the code (above) or it is decoration; kept as the
  engine's internal justification.
- **Three independent generality mechanisms** (0009/0011/0013).
  Rejected: they are a hierarchy — each level's coverage bottleneck
  is replaced by structure-driven search one level up.

## Consequences

- README, REPORT.md, RESEARCH_WRITEUP.md and plans 0009–0013 cite
  this ADR as the anchor; "compression with a receipt" is scoped to
  **structural** compression.
- 0009 → 0011 → 0013 is one arc (term laws → block laws → block
  search), not three bets.
- Bounded mode (0012) is restated as the enrichment's lax case,
  unifying it with exact mode.
- New features must name their structure; structure-less features
  are rejected under this thesis.
- Plan 0014 stages the conversion path this thesis implies.

## Follow-ups (mandatory)

This ADR is **not in force** until the following are done — the
consistency it claims does not exist until every document cites it.
Each item is a required post-condition of this decision, to be
completed *after* this ADR is accepted.

- [ ] `README.md` — cite ADR 0002 as the anchor; re-scope
      "compression with a receipt" to **structural** compression
      (not weight-space; see ADR 0001); link `docs/overview.md`
      and `docs/mechanism.md`.
- [ ] `docs/overview.md` (new) — the "project envisioned" summary:
      what it is / how / the edge / what it exploits / what it
      delivers / the unique thing (certified approximation) /
      scope / direction.
- [ ] `project/REPORT.md` — cite ADR 0002 as the thesis; frame
      Phase 5 / §10.2 as the structural corner that survives.
- [ ] `project/RESEARCH_WRITEUP.md` — cite ADR 0002; state the
      re-expression *goal*, not reachability alone.
- [ ] `project/plans/0009-composable-rule-sets.md` — note its
      hierarchy position (term-level laws).
- [ ] `project/plans/0011-morphism-engine.md` — note its position
      (block-level laws) and that 0013 replaces its hand-written
      law list with search.
- [ ] `project/plans/0012-bounded-error-rewrites.md` — restate as
      the enrichment's lax case; exact and bounded are one object.
- [ ] `project/plans/0013-contraction-search.md` — note it is the
      discovery feeder in the 0009→0011→0013 arc; sequence it after
      plan 0014's delivery stages.
- [ ] `AGENTS.md` — add the anchor thesis and the "name your
      structure" rule to Conventions.
- [ ] `project/TODO.md` — link these follow-ups so they are not
      lost.
- [ ] `pitch/` — align the deck's claim with the structural scope.

**Enforcement:** a new feature or plan that does not name its
categorical structure (Rule 1) is rejected under this thesis; a
document that contradicts this ADR is a bug.
