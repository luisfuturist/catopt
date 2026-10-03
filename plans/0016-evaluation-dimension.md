# Plan 0016 — evaluation as an independent dimension

Status: landed — stages 0–12, with the gaps named below.  Depends on
ADR 0003 (the four-dimension decision) and the seams it reuses (plan
0006 `Strategy`, 0007 backend ports, 0010 `Engine`).  Amends nothing;
it stages what ADR 0003 decides.

## Landed so far

| Stage | State | Where |
|---|---|---|
| 0 recon | done | `project/retros/four-dimension-recon.md` |
| 1 boundaries | done | `catopt_core.ports`: `Profiler` / `Policy` / `PerformanceModel` |
| 2 profiling | done | `catopt_core.features` (`ProgramFeatures`, `StaticProfiler`) |
| 3 one GPU backend | done | `catopt_core.failures` taxonomy + `TimingResult` provenance/`timeout_s` + classified `TorchMeter`/autotune/CUDA-runner failures + `calibrate` provenance; **one timing contract** — `catopt_core.timing` (warmup/n_calls, median, IQR) is what `TorchMeter`, `calibrate` and `benchkit.Runner` all reduce through; one CUDA backend |
| 4 evaluation | done | `catopt_core.pareto` (`CostVector`, `frontier`, `best`) |
| 5 game API | done | `catopt_core.game` (`Action` / `GameState` / `RuleBook` / `transition` / `Evaluator`) |
| 6 policies | done | `catopt_core.policies` (random / existing / greedy / beam-score) |
| 7 learned policy | done | supervised: `catopt_core.trajectories` + `catopt_torch.learned_policy` + `tools/train_search_policy.py`; **RL**: `catopt_core.search_env` + `catopt_torch.rl` + `tools/train_rl_policy.py` |
| 8 performance model | done | `catopt_core.perf_model` (`AnalyticalPerformanceModel`) |
| 9–11 | done | `docs/evaluation.md`; property tests |
| 12 benchmarks | done | `bench/suites/evaluation/policy_value.py` |

Stage 7's measured results:
`project/retros/stage7-learned-policy-results.md` (supervised),
`project/retros/stage7-rl-results.md` (RL), and
`project/retros/stage7-multifamily-results.md` (the three-family
mixture, scored per family).

## Wiring — what actually consumes each port

The stages landed modules; this is what makes them *operational*
rather than merely tested:

| Port / module | Consumed by |
|---|---|
| `Policy` | `EGraph.run(..., policy=)` — consulted once per iteration; `search(...)` and `Optimizer.optimize(...)` thread it through |
| `pareto` | `SearchResult.frontier(cost_fns)` — the non-dominated set over named axes |
| `Profiler` + `PerformanceModel` | `PredictedCriterion` — pass it as `criteria=` |
| `failures` | `TorchMeter`, `autotune`, the CUDA runner |
| `timing` | `TorchMeter`, `calibrate`, `benchkit.Runner` |

**The lesson: tested is not wired.**  Until this pass four of those
had no consumer anywhere in `packages/`, with every gate green — no
existing gate catches an unconsumed port.

## Known gaps

1. **`Engine` / `Policy` is under-specified.**  The `Engine` port says
   an engine *"may"* accept `policy`, so a caller cannot tell.
   `search` passes `policy` whenever it is set, so a policy-less
   engine (the shape `NativeEngine` has) raises a bare `TypeError`
   instead of working or declining clearly.  Latent: `catopt-native`
   is not installed here, so no test covers it.  Needs a capability
   declaration or a guarded call.
2. **RL collapses on the multi-family mixture** — measured.  The
   baseline fix landed (a per-episode standardized advantage in
   place of a single running-mean baseline): it solves `dup` and
   keeps `linear`, but `chain` still collapses — a shared-net
   winner-take-all under a sparse reward, not a reward-scale bias
   (see `retros/stage7-multifamily-results.md`).  A denser or
   state-conditioned signal is still needed.

## Goal

Make evaluation a first-class, pluggable dimension so the engine
defines a hardware-independent semantic search space and learns —
later — how hardware maps program features to runtime, without the
core depending on a GPU, a profiler, or a policy.

The four dimensions (ADR 0003):

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
```

Benchmarks are deliberately **last**: make the architecture correct,
then measure it.

## Non-goals

- **No façade package.**  Plan 0008 retired `import catopt`; this plan
  does not revive it.
- **`catopt-core` stays zero-dependency.**  The static profiler is
  pure Python over the IR.
- **No universal-speedup claim.**  Evaluation makes selection honest;
  it creates no structure.
- **No change to the certificate / verify contract.**
- **RL is not the point.**  It is one `Policy` among several, and the
  last to land.

## Gates (every stage)

Per AGENTS.md, all of these pass before a stage is done:

```sh
uv run pytest
.venv/bin/ty check
.venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture
.venv/bin/lint-imports
.venv/bin/bandit -c .bandit.yaml -r packages
.venv/bin/semgrep --config .semgrep.yml packages
.venv/bin/python tools/radon_ratchet.py
```

Coverage stays `fail_under = 100`.  **Every new Protocol ships with a
conformance test** (the `tests/` pattern the existing ports use), and
no new suppression is added to pass a gate.

## Stages

### 0 — Recon

Map the current boundaries and find violations of the four dimensions.

- Deliverable: `project/retros/four-dimension-recon.md` — every place
  a layer answers another's question, each with a disposition
  (landed).
- Gate: the note exists and each finding has a disposition.  [done]
- Risk: none; this is measurement.

### 1 — Boundaries

Name the four dimensions; add the ports ADR 0003 lists, reusing the
existing seams.

- Deliverable: `catopt_core.ports` gains the dimension contracts that
  do not already exist — `Profiler`, `Policy`, `PerformanceModel` —
  and the mapping table in ADR 0003 becomes a docstring contract.
  `Strategy` stays the whole-pipeline policy; `Policy` is the
  in-search action selector, threaded through `Engine.run`'s
  schedule / `stop` hook.
- Gate: import-linter still forbids core → adapter; each new Protocol
  has a conformance test.
- Risk: protocol sprawl — keep the surface minimal; prefer extending
  `Meter` over adding `Benchmark`.

### 2 — Profiling abstraction

Static, torch-free program features.

- Deliverable: `catopt_core.features` with `ProgramFeatures`
  (`flops`, bytes read/written, temporary bytes, `depth`,
  `operations`; `parallelism` / `reuse` only when computed) and a
  `Profiler` that produces it from an IR with **no execution**.
  Plain floats — no `Quantity` type.
- Gate: features are deterministic and pure-Python; a property test
  pins composition stability (`features(A∘B)` is a function of
  `features(A)`, `features(B)` where the structure allows).
- Risk: `parallelism` / `reuse` are new computations, not renames —
  if they cannot be computed honestly, they are omitted, not faked.

### 3 — One GPU backend

The first hardware *instance*, behind ports.

- Deliverable: a hardware adapter providing compilation/execution,
  warmup, synchronization, repeated measurement, robust statistics,
  environment metadata, failure classification, and reproducibility
  controls.  `Meter` is reused and gains provenance +
  failure-classification; no `Benchmark` port.
- Gate: a **backend-contract test** every future backend must pass
  (the same abstract suite, run on CPU and CUDA); failures are
  *classified* (OOM / timeout / kernel failure / NaN /
  device-unavailable), not crashes.
- Risk: CUDA must not leak into core; the adapter imports core, never
  the reverse.

### 4 — Evaluation

Keep the landscape; stop scalarizing early.

- Deliverable: a cost-vector evaluation over the candidate set
  `discover_alternatives` already enumerates
  (`SearchResult.alternatives`), a Pareto/frontier API, and a scalar
  `best()` view.  `criteria={...}` becomes that scalar view, not the
  internal contract.
- Gate: property test that the frontier is non-dominated and
  deterministic; extraction stays single-objective (multi-objective
  *extraction* is new work, explicitly out of scope here).
- Risk: confusing ranking with feasibility — `supported_ops` is a
  hard bound, the frontier is a soft ranking (ADR 0003 invariant 8).

### 5 — Game / search API

Formalize state, action, rule book, transition, evaluator.

- Deliverable: the game value types mapped onto the e-graph, with the
  roles kept distinct:

  | Role | Component |
  |---|---|
  | Rulebook (legal moves) | `catopt_core.laws` |
  | Board (state) | e-graph |
  | Referee (soundness) | verifier + certificate — a move is legal only if the law licenses it and the certificate replays |
  | Score estimator (predicts) | cost model / `PerformanceModel` |
  | Scoreboard (measures) | profiler / `Meter` |
  | Player | `Policy` |

- Gate: the mapping is documented in `docs/mechanism.md`; the types
  are exercised without any ML.
- Risk: over-abstracting — the game API must reduce to the existing
  `Engine.run`, not fork it.  The referee is **not** the e-graph: the
  board does not enforce soundness, the certificate does.

### 6 — Policies (no ML)

- Deliverable: `RandomPolicy`, `GreedyPolicy`, `BeamPolicy`, and a
  policy wrapping the existing search; all satisfy `Policy`.
- Gate: each policy is deterministic under a seed; a policy cannot
  change *what* is certified, only *which* rewrites are tried.
- Risk: a policy that prunes certified alternatives — forbidden by
  ADR 0003 invariant 5/8.

### 7 — Learned policy

RL as one more `Policy`.

The shape:

```text
pretrained search policy → runtime rules / model / hardware
   → guided exploration → measured results → continuous learning
```

- **Pretraining is on trajectories**, not on the rules.  The laws are
  the *action space*; the policy learns which legal move to try,
  where, and in what order, from `(state, action, outcome)` sequences.
- **Runtime context** the policy sees: the current e-graph, the
  available rule set, and the hardware / feature context.
- **Runtime rules without retraining** holds only if the action
  representation is **structural** — a new rule is embedded by its
  shape / local e-graph signature, not a fixed vocabulary index — and
  unseen actions carry a nonzero prior (uniform or heuristic).
  Otherwise a new rule is an out-of-distribution action and needs at
  least fine-tuning.  This is a load-bearing design decision, not an
  implementation detail.
- **Continuous learning**: fine-tune / retrain later from new search
  and benchmark data.
- Deliverable: an `RLPolicy` over the state/action space, pretrained
  on trajectories, rewarded by the deterministic evaluator (verifier
  + cost).
- Gate: RLPolicy is interchangeable with the stage-6 policies; it
  never bypasses the certificate; a **new rule added at runtime is
  scored without retraining** (the out-of-distribution test).
- Risk: training against a scalar the model already computes — the
  reward must be the *evaluator*, not a hand-tuned scalar.

### 8 — Performance model

- Deliverable: `PerformanceModel.predict(features, hardware)` —
  analytical first, learned later — predicting latency from features.
- Gate: prediction error is reported (predicted vs measured); the
  model may rank, never prune.
- **Two learners, not one.**  The policy learns *where to search*
  (reward = evaluator); the performance model learns *how hardware
  responds to features* (reward = measurement error).  Hardware data
  feeds the model; the model's predictions feed the policy.  They must
  not be collapsed — that is the four-dimension mistake in miniature.
- Risk: a learned model that leaks the target into core — it stays
  behind the port.

### 9 — Public API

- Deliverable: concepts, not internal classes, at the **domain-package
  paths** (`catopt_orchestrator.*`, `catopt_core.*`).  The phase verbs
  reuse existing names (`Source.to_ir`; `search` / `lower`).
- Gate: no `catopt` package; `docs/api.md` regenerated.
- Risk: re-introducing the façade — rejected (ADR 0003).

### 10 — Documentation (Diátaxis)

- Deliverable: tutorials / how-to / reference / explanation, with the
  reference tied to the public API.
- Gate: every public name is documented; examples run.
- Risk: docs drift from the API — the `bench` drift guards are the
  precedent for enforcing this.

### 11 — Tests

- Deliverable: unit, property, metamorphic, backend-contract, and
  **failure** tests; the failure suite is the one that classifies
  OOM / timeout / NaN rather than crashing.
- Gate: 100% coverage; `typeguard` runtime contracts on the new
  modules (grow `tools/runtime_types.sh`).
- Risk: instrumentation time — keep heavy suites out of the
  instrumented list, as today.

### 12 — Benchmarks (last)

- Deliverable: benchmarks rewritten around **research questions**
  (search, profiling, hardware-model error, discovery, RL-vs-
  heuristic, end-to-end), each preserving provenance (commit,
  hardware, driver, framework/compiler, shapes, dtype, warmup,
  iterations, statistics, seed, configuration).
- Gate: the existing `bench` harness and its drift guards still pass;
  verdicts are typed (`win` / `parity` / `regression` / `negative` /
  `inconclusive`).
- Risk: benchmarking the architecture before it is stable — that is
  why this is stage 12.

## Honest risks (plan-level)

- **Scope.**  Twelve stages is a program, not a patch; each stage
  lands independently and green, and none is required for the others
  to be useful.
- **Reach is unchanged.**  Evaluation improves *selection*; the
  semantic reach is still the law library's closure (ADR 0002,
  Rule 1).  No stage here widens what the optimizer can find.
- **Structure-conditional wins.**  Better selection cannot create
  structure a model does not have; the honest declines stay honest.

## See also

- `project/adrs/0003-evaluation-is-an-independent-dimension.md` — the
  decision this plan stages.
- `project/adrs/0002-categorical-re-expression-thesis.md` — the
  thesis; Rule 3 amended by 0003.
- `project/plans/0006-public-api-search-lower.md` (`Strategy`),
  `0007-backend-abstraction.md` (`Meter` / ports),
  `0010-performance.md` (`Engine`).
