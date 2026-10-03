# ADR 0003 — Evaluation is an independent dimension

**Status**: accepted.
**Date**: 2026-10.
**Depends on**: ADR 0002 (the thesis).  **Amends**: ADR 0002 Rule 3.

## Context

CatOpt today proves *reach* and prices it with a scalar, static,
calibrated cost model.  The pipeline is lift → saturate → extract →
certify → lower.  Pricing is a `CostFn` (selected by a `Criterion`)
over a `TargetProfile`'s measured constants.  Measurement exists, but
only as an autotune *selector* inside one strategy (`Autotuned` via the
`Meter` port), and its numbers are folded back into the cost model as
multiplicative corrections.

Three pressures force a wider frame:

1. **Reach is certified; selection is one-dimensional.**
   `extract_best` minimizes a single scalar.  The equivalence frontier
   (`discover_alternatives` → `SearchResult.alternatives`) already
   enumerates many equivalent programs, then collapses the landscape
   to one winner — the information a hardware-aware chooser would need
   is discarded.
2. **"How does this run here?" is not "how many FLOPs/bytes?"**
   `catopt_core.profile.TargetProfile` holds *target constants*
   (tflops, gbps, launch µs, measured corrections) — not *program
   characteristics*.  The two are conflated, and the measurement →
   model loop is ad hoc.
3. **ADR 0002 Rule 3 declares RL a non-goal.**  That is right for
   *reach* — structure generates it, and search should not learn what
   algebra already knows.  It wrongly forecloses *policy* — choosing
   which rewrite to fire next — as an independently replaceable part.

The opportunity: make evaluation a first-class, pluggable dimension,
so the engine defines a **hardware-independent semantic search
space** and learns *later* how hardware maps abstract program
properties to runtime — without the core depending on a GPU, a
profiler, or a policy.

## Decision

Adopt **four independent dimensions** with explicit ports, and keep
them independent.

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
   │            │             │              │
 "equivalent?" "explore?"  "how does it   "what actually
                            look / run?"    ran?"
```

**Rule.** Semantics, search, evaluation and execution are independent
dimensions; no layer may become responsible for another's question.

Each dimension answers exactly one question:

| Dimension | Question | Owns | Must not |
|---|---|---|---|
| Semantics | Is it equivalent? | `EGraph`, `verify_certificate` | price or time anything |
| Search | What should we explore? | `Engine`, policy | decide equivalence |
| Evaluation | What does it look like / what might run well? | profiler, cost, performance model | change the function |
| Execution | What actually ran? | `Sink`, `Meter`, hardware adapter | define semantics |

### Decisions in detail

1. **Evaluation splits from execution.**  A `Profiler` observes a
   program *without running it on the target*; an `Executor` runs it;
   the existing `Meter` measures a run — there is **no separate
   `Benchmark` port**; `Meter` gains provenance and failure
   classification.  **Profiling does not imply a GPU** — the static
   half is torch-free, in core.
2. **Keep the frontier; do not scalarize early.**  The candidate set
   is what `discover_alternatives` already enumerates
   (`SearchResult.alternatives`); selection keeps the non-dominated
   subset over named dimensions.  A scalar `best()` is a convenience
   view, not the internal contract — today's `criteria={...}` weighted
   blend is that view.  Note: `extract_best` minimizes one scalar by
   construction, so true multi-objective *extraction* is new work, not
   a re-view of the existing one.
3. **One reference hardware backend, not one architecture.**  The
   first GPU profiler/benchmarker is the first *instance* behind the
   ports.  `cuda` / `rocm` / `tpu` names never enter core; the
   dependency arrow points core ◀ adapter, never core ▶ CUDA.
4. **Policy is a replaceable port, RL included.**  RL is a *policy*
   over an already-structured space — never the mechanism of reach.
   The first policies are random / greedy / heuristic / existing; RL
   is one more conforming value.  `Policy.choose(state, actions)`
   threads through the existing `Engine.run` schedule / `stop` hook —
   it is not a parallel search mechanism.  This **amends ADR 0002
   Rule 3**.
5. **The performance model is a port.**  Initially analytical, later
   learned; it *predicts*, it never decides correctness and never
   bounds the semantic space.
6. **Benchmarks are last.**  Rewrite benchmarks only after the
   boundaries settle, around research questions, not just speed
   numbers.

**Feasibility is hard; performance is soft.**  Two mechanisms already
bound the search and must stay distinct: `supported_ops` is a *hard*
feasibility bound (a form using an unlowerable op prices at `+inf` and
is never selected); a performance model is a *soft* ranking (it orders
feasible members and may never drop one).  Invariant 8 makes this
binding.

## The ports — mapped onto what already exists

The design reuses catopt's existing hexagonal seams; it does not add a
parallel universe of abstractions.

| Concept | Existing seam | Action |
|---|---|---|
| `Source` (`capture`) | `catopt_core.ports.Source.to_ir` | reuse; name the phase verb |
| semantic space | `EGraph`, `Engine` | reuse |
| whole-pipeline policy | `catopt_core.ports.Strategy` | reuse |
| in-search action policy | — | **new** `Policy` (`choose(state, actions)`) |
| cost / ranking | `CostFn`, `Criterion`, `catopt_core.cost` | reuse |
| static program features | — (see naming note) | **new** `Profiler` → `ProgramFeatures` |
| measurement | `catopt_core.ports.Meter` / `TimingResult` | reuse; add provenance |
| lower / verify / execute | `Sink` (`Capabilities` + `lower` + `verify`) | reuse |
| hardware description | `catopt_core.profile.TargetProfile` | reuse, do not duplicate |
| performance prediction | — | **new** `PerformanceModel` (`predict(features, hardware)`) |

**Naming note (binding).**  `profile` is already taken:
`catopt_core.profile.TargetProfile` names *target constants*.  The new
static characterization of a *program* must use a distinct name —
proposed `ProgramFeatures` in a new `catopt_core.features` module —
and never be called a "profile".  "Profile" keeps exactly one meaning:
a measured target.

**No new numeric type.**  Features are plain floats (`flops`, bytes,
`depth`), not a `Quantity` abstraction; `depth` already exists as
`DepthCriterion`.  `parallelism` / `reuse` do not exist yet — they are
new feature computations, not renames.

## Invariants (binding)

1. `catopt_core` imports no `torch` / `numpy` / GPU library
   (import-linter already enforces this).
2. Semantic equivalence is independent of performance.  A rewrite's
   legality is decided by laws and certificates alone.
3. Profiling is observation, never semantics.
4. Hardware-specific behavior lives behind ports.
5. Search policy cannot invalidate correctness: the certificate is a
   **hard filter** (accept/reject), never a soft reward.
6. Extraction stays deterministic and testable without learning.
7. Benchmarking is never part of semantic correctness.
8. The performance model may *rank* the semantic space; it may never
   *prune* it in a way that loses a certified alternative.
9. Measurement is provenance-carrying: commit, hardware, driver,
   framework/compiler version, shapes, dtype, warmup, iterations,
   statistics, seed.
10. No public API exposes implementation details unnecessarily.

## Relationship to ADR 0002

- **Amends Rule 3.**  "Directed search, not learning" stays true of
  *reach*: structure is what makes search beat enumeration.  It is
  amended to allow a *learned selector* over that structure.  The
  certificate constraint (Rule 2) and "construction primary,
  saturation fallback" (Rule 4) are unchanged.
- **Keeps Rule 1 binding for evaluation too.**  A profiler or
  performance model must name the property it observes or predicts
  ("occupancy", "memory traffic", "launch count") — an unnamed
  feature is decoration.
- **Keeps Rules 5–6.**  Exact/bounded remain one object; scope stays
  structure-dependent, and honest declines stay honest.

## Generality — the precise claim

CatOpt is general **broadly, not magically**.  The claim, stated so it
can be falsified:

- **Any model expressible in CatOpt's IR** can be optimized; the
  `Source` port decides expressibility (torch: `torch.export`).
- **Known operations** get algebraic transformations — the law library
  acts on the structures those ops instantiate.
- **Unknown / custom operations** are opaque and preserved safely.
  Mechanically: an op absent from the sink's table drops out of
  `supported_ops` (`TorchSink.supported_ops`), so `backend_cost`
  prices every form using it at `+inf` and extraction never selects a
  form that rewrites it; a missing binding fails loudly at eval
  (strict) or yields `None` (fold), never a silent mis-lowering.
- **More rules / carriers** enlarge the space — reach is the closure
  of the law library, so a richer library is a strictly larger
  reachable set (ADR 0002, Rule 1).
- **The same optimizer** runs across model architectures and — once
  evaluation is a port — across hardware.

Core idea:

> **Don't build an optimizer for every model.  Build an optimizer for
> the algebra that models are made from.**

This is the framework / law-library split the README states as "the
honest split": the *framework* is model- and backend-agnostic; the
*law library* is narrow, and it is the only thing that bounds reach.
An unmatched block is an opaque boundary, never a wrong answer.

## Scope / non-goals

- **No façade package.**  Plan 0008 retired `import catopt`; this ADR
  does not revive it.  New public surface stays at the domain-package
  paths (`catopt_orchestrator.optimize`, `catopt_core.egraph`, …).
- **`catopt-core` stays zero-dependency.**
- **RL does not own the optimizer.**  It is a player, not the board.
- **No change to the certificate / verify contract.**
- **Not a universal-speedup claim.**  Evaluation makes selection
  honest; it does not create structure where none exists.

## Deferred to the companion plan

Named here so they are not lost; their design belongs to the plan, not
this ADR:

- **Failure classification.**  OOM / timeout / kernel failure / NaN /
  device-unavailable must be *classified*, not crash — a `Meter`
  contract, with a backend-contract test.
- **Performance dataset storage.**  Where `(features, hardware,
  measurement)` tuples live — extend `TargetProfile`'s measured tables
  or a new artifact; must round-trip and carry provenance.
- **The `Policy` protocol's exact signature** (`state`, `actions`,
  reward) and its wiring into `Engine.run`.

## Staged rollout

Ordered so the architecture is correct before it is measured.
Benchmarks are deliberately last.

```text
0  Recon — map current boundaries; find violations
1  Boundaries — name the four dimensions; add the ports
2  Profiling abstraction — static `Profiler` → `ProgramFeatures`
3  One GPU backend — executor + benchmarker + device metadata
4  Evaluation — cost vectors + Pareto frontier (keep the landscape)
5  Game / search API — state, actions, rule book, evaluator
6  Policies — random / greedy / beam / existing (no ML)
7  Learned policy — RL as one more `Policy`
8  Performance model — analytical first, then learned
9  Public API — concepts, not internal classes
10 Docs — Diátaxis (tutorials / how-to / reference / explanation)
11 Tests — unit, property, metamorphic, backend-contract, failure
12 Benchmarks — rewritten around research questions
```

## Alternatives considered

- **Keep the scalar cost model + autotune (status quo).**  Rejected:
  it discards the frontier and keeps the four questions fused, so
  every new hardware feature edits the core.
- **Start with RL.**  Rejected: without a profiler, a frontier and a
  performance model there is no environment and no training signal —
  the agent would learn against a scalar the model already computes.
- **A new top-level `catopt` façade for the "clean" API.**  Rejected:
  directly contradicts plan 0008; the API is a *re-export surface over
  domain packages*, never a package again.
- **Put program features in `catopt_core.profile`.**  Rejected: name
  collision — that module is target constants.  Features get their
  own module.
- **Make the GPU implementation the abstraction.**  Rejected: CUDA is
  the first *instance*, not the interface.

## Consequences

- The engine gains an explicit `Policy` seam below `Strategy`, so
  RL/MCTS/beam are interchangeable without touching search.
- `discover_alternatives` becomes the entry to a Pareto frontier, not
  a top-k list; `criteria={...}` becomes the scalar view of it.
- `Meter` gains provenance and failure classification; `Autotuned`
  becomes one consumer of evaluation, not its owner.
- New protocols are new surface: they must be typed (`ty`), tested
  (100% coverage), and free of dead code (`vulture`).
- The road to a learned performance model is opened without changing
  the semantic contract — the central invariant the project protects.
- **A port is not done until something reaches it.**  Four of these
  ports shipped typed, tested and green while *no code in
  `packages/` consumed them* — and every gate passed, because no gate
  catches an unconsumed port.  "Wired" is a deliverable of its own;
  see plan 0016's Wiring table.
- **`Engine` / `Policy` needs a capability declaration.**  This ADR
  says an engine *may* accept a policy, so a caller cannot tell: a
  policy-less engine (the shape `NativeEngine` has) raises a bare
  `TypeError` rather than declining clearly.  Either the port gains an
  explicit capability flag or the call is guarded — left open here,
  tracked in plan 0016's Known gaps.

## Follow-ups (mandatory)

All complete — this ADR is **in force**.

- [x] `project/adrs/0002-...md` — add a pointer that Rule 3 is amended
      by 0003.
- [x] `AGENTS.md` — add the four-dimension rule and the "profile =
      target constants only" naming rule to Conventions.
- [x] `docs/mechanism.md` — add the evaluation dimension after
      "certify" (syntax → … → certify → **evaluate**).
- [x] `README.md` — scope the headline to include evaluation, and
      state the "not a façade" constraint.
- [x] A companion **plan** — `project/plans/0016-evaluation-dimension.md`
      carries the full staged rollout, per-stage gates and acceptance
      tests.  This ADR records the decision; the plan records the work.
- [x] `project/TODO.md` — link this ADR and its companion plan.

**Enforcement:** a change that lets a layer answer another dimension's
question — a cost model that changes semantics, a policy that decides
equivalence, a profiler that runs on the target — is a bug under this
ADR.
