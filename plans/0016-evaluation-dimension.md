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

## Measured results

What measuring the landed stages showed.  Every number is
reproducible from the named retro; ratios are to the exact optimum
where one exists, to best-found otherwise (an upper bound on the
optimum, so the measured gaps are conservative).

### Contraction ordering — the precondition

`project/retros/contraction-precondition.md`.  Tensor-contraction
ordering is a space where the one-step greedy cost-model oracle is
*provably* suboptimal, so a learned `Policy` has something to win.

* classic matrix chains — greedy loses **66.7 %** of instances
  (mean 2.77×, worst 15.3×);
* general tensor networks — greedy loses **71.7 %** (mean 2.09×,
  worst 23.0×);
* catopt **e-graph** rule space (`assoc_matmul` over `Var` leaves,
  the priced spelling) — the one-step oracle loses **53.3 %**
  (worst 8.75×); saturation reaches exactly the DP optimum on
  **60/60** chains (`contraction-ladder.md`);
* catopt **diagram** move space — **no headroom**: `search_moves`
  ties greedy on **4/4** chains.  The window reify folds the chain
  into one morphism and the composed weight is a param-only subtree
  priced at 0, so ordering has no cost signal there.

### Contraction ordering — expressible but not priceable

`project/retros/contraction-cost-signal.md` asks the harder
prerequisite: can the *diagram* space price an order at all?

* **No.**  All 6 bracketings are reachable in the root e-class, but
  every one prices at **0** under `flops_cost` / `launch_aware_cost`
  / `count_cost`; extraction breaks the tie by member order and
  picks a suboptimal bracketing on **37/40** chains (worst 10.7×).
* **Billing the fold is unsound** — measured blast radius **31 test
  failures across 15 files** — and it would *misprices runtime*: the
  folded weight is materialised once at compile time, so steady-state
  runtime is one GEMM whatever the order.  The discount is correct.
* The order signal lives in the **`Var`-leaf e-graph rule space**,
  which prices it directly.  Verdict: contraction ordering is **out
  of scope for the diagram move space** (a fusion space).

### Contraction ordering at scale — the gate

`project/retros/contraction-scale.md` tests the other end, where the
exact DP is intractable.

* **Equality saturation dies at n ≈ 12** (assoc) / **n ≈ 8** (AC):
  the e-class member count is **Catalan**, so the fixed point is
  reached only for n ≤ 12 and n = 13 does not finish in 200 s.  The
  exact subset DP reaches **n ≈ 18** — saturation is *dominated* by
  the DP, so there is no scale where it is the right tool.
* **Greedy is materially worse at scale**: **13.3× mean / 28× worst**
  above best-found at n = 60 (2.3× at n = 20/40; the n = 30 draw is
  mild at 1.15×).  Best-found is an upper bound, so greedy is *at
  least* 13× off optimal at n = 60.
* **Controls validate the players** against the DP at n = 8–16:
  bounded best-first `search` reaches 1.01–1.09× (1.55× on one hard
  draw); `one-step` hits the optimum at n = 8.
* The headroom is above **greedy**, not above saturation.  The real
  baselines are `search` (n ≤ 30) and `restart` (n ≤ 60), already
  1.00–1.04× best-found.

### Coordination — the hand-written pairing heuristic

`project/retros/coordination-optimality.md` asks whether the shipped
`extract_paired` / `_select_best_term` policy is optimal.

* **Near-optimal.**  Under the shipped default model it is at the
  optimum on **92.2 %** of enumerable draws (naive greedy: 37.3 %);
  under `count_cost` it is optimal on **100 %**.
* The residual gap is **one kernel dispatch** on ~10 % of draws
  (max rel 4.76 % under the default model, 0.35 % under
  `launch_aware`; `flops_cost` carries no coordination signal).
* The gap is an **all-or-nothing artefact**: `_select_best_term`
  compares only *two* terms (greedy vs force-*every*-group), so it
  cannot fuse one group while declining another.  The fix is a
  **per-group decision** (a `{0,1}^G` vector, `G` ≤ 3 here) — not
  ML — and its ceiling is a few dispatches.

### Learned contraction ordering — the decisive test

`project/retros/contraction-policy.md` trains a REINFORCE policy on
n = 8–12 with a greedy-completion critic and generalises to
n = 20/30/40 with no large-n data.

* **beats `greedy` decisively** — **0.41–0.69×** its cost at every
  scale and seed;
* **ties bounded `search`** — **0.93–1.09×** (seed-dependent);
* **comparable to `restart`** as one rollout — 0.78–1.03×;
* **beats `restart` only with restarts on both sides** —
  **0.73–0.87×** (64 sampled rollouts vs 64 randomised-greedy);
* **imitation generalises worse** — trained on the exact DP optimum,
  it degrades monotonically (`imitation / restart`: 1.10 → 2.25 →
  **5.72** at seed 0, n = 20 → 30 → 40);
* controls: the policy is **1.12–1.18×** the true optimum at
  n = 8–12 — a genuine player, not a random ordering.

Two silently-corrupting RL bugs were found and fixed: the REINFORCE
loss multiplied `logps` (step-major) against `advantages`
(episode-major) — a *permuted* gradient — and un-standardised
advantages diverge (one catastrophic move dominates).  With both
fixed the policy improves monotonically (1.24 → 1.14 → 1.09).

The honest headline: **a learned contraction-ordering policy is a
viable player, not a new regime** — as a single rollout it does not
beat the best cheap player.

### The other measured negatives

* **RL collapses on a multi-family mixture**
  (`stage7-multifamily-results.md`): the supervised
  per-`(features, rule)` classifier does not dilute (rank 1.00 on
  all three families); RL still collapses `chain` to the floor after
  the diagnosed reward-scale bug was fixed — a shared-net
  **winner-take-all** under a sparse reward, not a scale bias.
* **A model-backed criterion cannot steer extraction**
  (`eval-axis-selection.md`): `PredictedCriterion` prices a form by
  the *whole* subtree, but `extract_best` recovers a local cost by
  the **additive** marginal `c(t) − Σc(children)`, so a
  non-additive criterion is mis-ranked or collapses to bit-identical
  values.  Measured: pluggable evaluation changes the extracted
  program in **0/7** families, and `SearchResult.frontier` yields
  **0 genuine** trade-offs over 7 families × 21 axis pairs.

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

**Settled by measurement — do not re-open.**

* **Contraction ordering in the diagram move space is out of scope.**
  The composed weight is param-only, so the shipped runtime models
  price every bracketing at 0; the fold discount is *correct* for
  runtime (billing it is unsound — 31 test failures) and the diagram
  space is a **fusion** space.  The order signal lives in the
  `Var`-leaf e-graph rule space, which prices it directly
  (`contraction-cost-signal.md`).
* **Equality saturation is the wrong tool for contraction ordering.**
  It is exponential in the e-class member count (Catalan) and dies at
  n ≈ 12 (assoc) / n ≈ 8 (AC), below the exact subset DP's n ≈ 18.
  The small-board "saturation = optimum" result was a tiny-space
  artefact (`contraction-scale.md`).
* **The coordination heuristic is near-optimal.**  The shipped
  `extract_paired` / `_select_best_term` policy is at the optimum on
  ~92 % of enumerable draws; the residual is exactly one kernel
  dispatch on ~10 %, an all-or-nothing group-policy artefact.  It is
  a per-group decision, not an ML problem
  (`coordination-optimality.md`).

**Open.**

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
3. **`PredictedCriterion` cannot steer extraction.**  `extract_best`
   recovers a node's local cost by subtracting its children's costs —
   exact only for an *additive* cost function.  A `PerformanceModel`
   prediction is a function of the whole subtree, so the marginal is
   either mis-ranked (`swiglu`) or collapses to bit-identical values
   for forms whose true predictions differ by 25 % (`linattn`).
   Measured by `python -m bench run eval_axis`: pluggable evaluation
   does **not** change the extracted program (0/7 families), and the
   true roofline argmin is target-invariant in 7/7.  Either the model
   criterion is confined to post-hoc ranking, or extraction needs a
   dense (non-marginal) pricing path for non-additive criteria.
4. **Beating the strong cheap heuristics at scale** — the decisive
   open thread.  The learned contraction-ordering policy crushes
   greedy (0.41–0.69×) but only *ties* bounded `search` (0.93–1.09×)
   and beats `restart` only when both sides get restarts (0.73–0.87×).
   The headroom `contraction-scale.md` measured is above *greedy*; the
   strong cheap players already capture most of it.  Sketched in the
   open thread below — not yet its own plan file, because the
   direction is still a choice.

## Contingent spike — growing the law library

If the headroom hunt comes back empty (every space measured is small
enough for saturation to close it, so a policy can only tie), the
remaining direction is to widen the **action space** rather than the
search:

**The condition was tested and not met.**  The headroom hunt
(`contraction-scale.md`) found a space where saturation *breaks*
(n ≥ 12) and greedy is materially worse — so the remaining direction
is beating the cheap players, not widening the action space.  The
spike below stays contingent on a *future* empty result.

* Today a `Policy` acts on **2-cells** — it chooses which *known* law
  to fire, so reach is the library's closure.  ADR 0002 frames the
  e-graph + certificate machinery as a higher category (1-cells =
  programs, 2-cells = rewrites, 3-cells = coherences); the two levels
  above the policy are unused.
* A policy could instead **propose an equality** (a lemma, a derived
  rule, a chosen coherence) and let the referee verify it.  The
  striking part: `verify_certificate` replays a derivation and has no
  notion of "library vs proposal", so propose-then-verify needs **no
  change to the safety story** — reach grows, soundness is preserved
  by construction.
* The hard part is verification cost.  Proving a proposed equality is
  as hard as the search itself (undecidable in general here); a
  numeric check is *evidence, not proof*.  So the design is a
  **spectrum**: propose → numeric check → admit as a conjecture under
  the existing enrichment (`error_budget`, the lax case) → upgrade to
  a proof if a derivation is found.
* This is a **design spike, not a stage** — it changes what an action
  *is*, and it is the step that would turn "search a fixed library"
  into "search that grows its own library".

## Open thread — beating the cheap heuristics at scale

The decisive test (`contraction-policy.md`) leaves one question open:
the learned policy *ties* the best cheap player, so is there a regime
where learning beats `search` / `restart` outright?  A plan is
warranted for this thread, but the direction is still a choice, so it
stays a **sketch** inside plan 0016 — promote it to its own plan file
only once a direction is picked.

Candidate directions, each falsifiable:

1. **Compute-equal comparison.**  The policy rollout and `restart`
   were matched in *rollout count*, not per-step cost; a learned
   rollout is the more expensive one.  Measure wall-clock at an equal
   budget before claiming a win.
2. **A learned critic / state-conditioned policy.**  The RL critic is
   the hand-built greedy completion; a learned value baseline — or
   the supervised per-`(features, rule)` classifier, which did *not*
   dilute — may close the `search` gap.  The mixture collapse is a
   shared-net winner-take-all, so a state-conditioned head is the
   natural next architecture.
3. **The per-group coordination decision.**  The ~10 % residual is a
   `{0,1}^G` fuse-or-not vector, not a search problem; it is a small,
   deterministic fix to `_select_best_term` and needs no ML.  This is
   the shovel-ready direction.
4. **A new regime, not a better player.**  The measured headroom is
   above greedy; the strong cheap players already capture most of it.
   If a learned policy cannot dominate them at equal cost, the honest
   conclusion is that contraction ordering is *solved by cheap
   heuristics*, and the evaluation dimension's value is elsewhere
   (the coordination decision, the additivity seam, the mixture).

Acceptance for any direction: beat the *best cheap player*
(`search` / `restart`) at an **equal wall-clock budget** on
n = 20/30/40 with a seed-robust margin — or record the negative.

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
