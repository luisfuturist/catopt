# Executor-aware pricing under real wall-clock — where the routing inversion actually lives

`law-wallclock-verification.md` found the pipeline's dispatch-count
cost model praising SSM extractions that measure *slower* than raw:
the picked term lowers to `BatchedScanModule`, which loses to the
unrolled path at these dims.  `tools/executor_cost_probe.py` now
localises the blind spot precisely, measures the executor families on
the real model shapes, and demonstrates that a *measured* executor
price — applied at both decision points that need it — restores the
correct routing on every case.

**The headline: the inversion is an *extraction* blind spot plus a
*calibration* gap, and it needs measured prices at two places, not
one.  Under corrected delivered pricing the greedy extraction picks
the measured-fastest member on all six SSM cases (the paired stage
too, on every case traced) — and the shipped pipeline still delivers
`BatchedScanModule`, because `_carrier_upgrade` re-inverts the
decision comparing `_delivered_cost` at hardcoded `profile=None`
constants.  Re-running
that same comparison under the measured factors keeps the right
member: all six SSM cases then ship `add`-rooted `IRModule`s at
0.89–1.23× of raw instead of `apply[d]`-rooted `BatchedScanModule`s
at 0.37–0.54×.  The public `search(cost_fn=…)` seam exists and works;
the carrier-upgrade comparison is the part that cannot see a measured
price today.**

## 1. Where the executor choice is priced today

Three places, mapped on the code:

* **Extraction** (`optimize.py:240-254` `_default_cost_fn` →
  `executor_cost_for(lowering="generic")`, wrapped by
  `backend_cost`): every root-eclass member is priced as if the
  per-node `IRModule` evaluator runs it — regardless of which
  executor the term routes to.  This is where the `apply[d]` member
  wins: its collapsed compose spine counts fewer dispatches than the
  `add`-rooted member (sel-cost 3.13e5 vs 3.48e5 on DiagonalSSM
  small) — but `add` is delivered by `IRModule` while `applyd` is
  delivered by `BatchedScanModule`, which the price never sees.
* **Delivery routing** (`optimize.py:334-369` `_lower_extracted` /
  `_route_spec`): term-level — the first carrier `ExecutorSpec`
  whose `accepts` probe holds claims the term
  (`adapters.py:153-267` `_executor_specs`: scan / om_batched /
  omd_batched / om_streaming / trace, then generic).  No price
  comparison at all — routing is a pure shape test on whatever
  extraction selected.
* **`_carrier_upgrade`** (`optimize.py:291-331`, called from
  `_select_best_term` at `:1678`): a post-extraction pass that can
  only swap a carrier member *in* — it force-extracts each
  carrier-apply enode of the root e-class and compares
  `_delivered_cost` (`optimize.py:270-288`), which bills the term
  under the executor it would route to.  Its `profile` parameter
  exists end-to-end (`_carrier_upgrade(profile=)` →
  `_delivered_cost(profile=)` → `executor_cost_for(profile, …)`) —
  and is *never passed*: `_select_best_term` calls it without a
  profile and `search()` exposes no `profile=` argument at all, so
  the comparison always runs on the built-in RTX-2050 fallbacks
  (`_profile_constants` / `dispatch_us≈launch` / `leaf_eval_us=15`).

So on the shipped path the carrier member won the *generic-priced*
extraction outright on all five SSM models (the probe's
`post_greedy_diff` flag is `False` on all six SSM cases — the
post-greedy stages changed nothing).  The batched-vs-generic
delivered comparison that
*could* have caught the regression exists only inside
`_carrier_upgrade`, and it runs uncalibrated.

## 2. The measured executor table

Per case, every root-eclass alternative
(`SearchResult.alternatives(top_k)`) was lowered through BOTH its
routed executor and the generic `IRModule`, verified against the raw
model, and timed on CUDA — warmup 50 / timed 200, per-iter sync,
median ms — eager and under manual CUDA-graph replay (the
`law_wallclock` methodology verbatim).  µs below.

Same extracted term, two deliveries — `BatchedScanModule` vs the
serial `IRModule` eval of the *same* `apply[d]` term:

| case | raw eager | batched E | batched G | generic E | generic G |
|---|---|---|---|---|---|
| SelectiveSSM small | 158 | 366 | 125 | 182 | 118 |
| SelectiveSSM large | 276 | 513 | 247 | 323 | 221 |
| DiagDenseSSM small | 143 | 332 | 88 | 161 | 85 |
| DiagDenseSSM large | 252 | 486 | 171 | 289 | 156 |
| DiagonalSSM small | 100 | 274 | 79 | 104 | 36 |
| DiagonalSSM large | 164 | 410 | 135 | 185 | 50 |

`BatchedScanModule` costs 1.9–2.7× the generic eval of the *same
term* at these dims — the level-schedule's gathers/copies dominate a
4–8-step serial loop.  And the member the model preferred over it:

| case | `apply[d]`→scan E | `add`→generic E | modeled delivered (applyd vs add) |
|---|---|---|---|
| SelectiveSSM small | 366 | 158 | 5.9e5 < 7.7e5 — **wrong** |
| SelectiveSSM large | 513 | 266 | 1.0e6 < 1.5e6 — **wrong** |
| DiagDenseSSM small | 332 | 137 | 5.6e5 < 7.0e5 — **wrong** |
| DiagDenseSSM large | 486 | 300 | 9.4e5 < 1.5e6 — **wrong** |
| DiagonalSSM small | 274 | 85 | 6.6e5 > 5.6e5 — right |
| DiagonalSSM large | 410 | 150 | 1.1e6 < 1.1e6 — **wrong** |

The *modeled* delivered comparison is wrong on 5/6 SSM cases —
measurably, not just in absolute error.  Measured/modeled ratios on
the SSM terms: batched 0.39–0.62, generic 0.14–0.20 — the model
over-prices BOTH families, but the generic delivery ~3× more, so the
batched route wins a modeled comparison it loses on the clock.
(Geomean factors over all measured alternatives, eager:
`batched ×0.50`, `generic ×0.063` — the generic figure is dragged
down by the hybrids' wildly over-modeled terms, up to 2.7e9 vs 616 µs
modeled; on the SSM terms alone generic runs ~×0.18.)

`reshape`-rooted alternatives failed lowering with a CUDA/CPU device
mismatch in a `cat` (constant folding leaves a CPU tensor the lowered
module consumes alongside CUDA params) — recorded honestly as `-`;
at ~300× the modeled cost they were never competitive anyway.

## 3. Feeding it back — the counterfactual

The probe re-runs `search` with a public-seam cost model —
`_delivered_cost_fn`: each term priced under the executor it routes
to (`"batched_scan"` for plannable apply-roots, `"generic"` else),
optionally × the measured per-family factor — then, on the same
e-graph, re-runs the `_carrier_upgrade` comparison under corrected
delivered prices (the paired-extraction stage is not reproducible
post-hoc — its group objects aren't recorded — but under this cost
model it agreed with greedy on every case checked).

What each lever does, measured eagerly in µs:

| case | raw | shipped | cf-modeled | cf-measured | corrected-sel |
|---|---|---|---|---|---|
| SelectiveSSM small | 158 | 366 apply/scan | 386 apply | 431 apply | **179 add/IRModule** |
| SelectiveSSM large | 276 | 513 apply/scan | 505 apply | 520 apply | **282 add/IRModule** |
| DiagDenseSSM small | 143 | 332 apply/scan | 312 apply | 315 apply | **126 add/IRModule** |
| DiagDenseSSM large | 252 | 486 apply/scan | 424 apply | 429 apply | **221 add/IRModule** |
| DiagonalSSM small | 100 | 274 applyd/scan | **82 add** | 246 applyd | **82 add/IRModule** |
| DiagonalSSM large | 164 | 410 applyd/scan | 381 applyd | 374 applyd | **133 add/IRModule** |
| HybridBlock small | 189 | 227 linear | 221 linear | 195 linear | 197 linear (same) |
| HybridBlock large | 224 | 313 linear | 374 linear | 256 linear | 249 linear (same) |
| TwoLayerHybrid small | 285 | 366 stack | 338 stack | 311 stack | 310 stack (same) |
| TwoLayerHybrid large | 416 | 616 stack | 616 stack | 426 stack | 429 stack (same) |
| ManualSoftmaxAttn small | 107 | 90 linear | 140 linear | 107 linear | 104 linear (same) |
| ManualSoftmaxAttn large | 261 | 260 linear | 250 linear | 208 linear | 258 linear (same) |

* `cf-modeled` = extraction under *uncalibrated* delivered pricing.
  Flips only DiagonalSSM small (the one case the modeled delivered
  comparison already gets right); ships the carrier member on the
  other five.  No case regresses.
* `cf-measured` = extraction under corrected delivered pricing —
  **greedy picks `add` on all six SSM cases** (`greedy_root=add` in
  every row) — and the shipped pipeline still delivers
  `apply[d]`→`BatchedScanModule` on all six: `_carrier_upgrade`
  re-inverts it.  Instrumented trace on DiagonalSSM small,
  SelectiveSSM large and DiagDenseSSM small: greedy→`add`,
  paired→`add`, shipped `_carrier_upgrade`→`apply[d]` (uncalibrated
  delivered prices 5.6e5 < 7.0e5 / 1.0e6 < 1.5e6), corrected
  upgrade→`add`.
* `corrected-sel` = the identical selection chain with the upgrade
  comparison priced under the measured factors: lands on the
  measured-fastest member on **all 12 cases** — `add`→`IRModule` on
  all six SSMs, same member as shipped elsewhere (correct — those
  were already the best members).  All verified `pass`.

The inversion flips on **all five SSM models at both sizes**.
ManualSoftmaxAttention needed no fix (its member was already
measured-best and stays so; the `softmax_fold` win is preserved).
Honest ordering vs raw, eager: corrected-sel beats raw on DiagDense
(126 vs 143 / 221 vs 252) and DiagonalSSM (82 vs 100 / 133 vs 164),
and lands within ~2–13 % *slower* on SelectiveSSM (179 vs 158 /
282 vs 276).  Under graph replay the corrected picks beat or match
raw everywhere except DiagDenseSSM small (77 vs 66 µs — the borderline
case; run-to-run noise on it is comparable to the gap).

## 4. What the shipped measured machinery already does — `Autotuned`

The `Autotuned` strategy (plan 0006/0007,
`orchestrator/autotune.py`) re-lowers the extracted term through
`eager` (raw), `generic`, `batched` candidates, verifies and times
each on-device, and ships the measured winner.  On these cases:

* Every SSM model and both hybrids: winner = **`eager`** — the raw
  model.  The batched candidate measured slower than eager on all
  ten (276–711 µs vs 104–497 µs), and the generic delivery of the
  same extracted term also lost — so the shipped mechanism honestly
  declines the slower executor.  But it can only re-*deliver* the
  term extraction picked: the `add`-rooted member is upstream of it
  and unreachable, so it ships raw and leaves the corrected member's
  win — where there is one (~14–22 % vs raw on DiagDense /
  DiagonalSSM; SelectiveSSM's best member still loses to raw) — on
  the table.
* ManualSoftmaxAttention: winner = `batched` (the generic IRModule
  delivery) — the pipeline's real win preserved.
* Its `profile=` channel (`record_measured` / `corrected_price_ns` /
  `shape_bucket` — `catopt_core/profile.py`) is exactly the
  measured-feedback table the extraction-side correction needs; the
  consumers just don't reach back into `_select_best_term`.

## 5. The seam, precisely

* `search(cost_fn=…)` is public and sufficient for the extraction
  half — `backend_cost` wraps it for feasibility (ADR 0003 clean:
  the model ranks, `supported_ops` gates).  No packages change was
  needed to demonstrate either corrected pricing.
* The blocker is `_carrier_upgrade`'s delivered-cost comparison at
  `optimize.py:323,328`.  The `profile` plumbing already exists —
  `_carrier_upgrade(profile=)`, `_delivered_cost(profile=)`,
  `executor_cost_for(profile, lowering=…)`, and `backend_cost`
  already forwards a `profile` marker attribute off the wrapped
  cost fn (`cost/backend.py:135-137`).  The minimal wiring is
  `profile=getattr(cost_fn, "profile", None)` at the
  `_select_best_term` call site (`optimize.py:1678`) — three lines.
  What is *missing* for the measured correction specifically:
  `_delivered_cost` composes `executor_cost_for(profile)` but never
  consults `measured_ns` / `corrections`, so a per-family measured
  factor (the only thing that flips the ordering here — the model's
  generic-vs-batched relative error is ~3×) cannot ride the profile
  yet.  Closing that needs a candidate-name + shape-bucket contract
  inside `_delivered_cost` — a small design decision, deliberately
  not taken in this tools-level probe.
* Non-additivity: `_batched_scan_latency` is a whole-spine price;
  as an `extract_best` model `_delivered_cost_fn` is approximate
  (documented in the tool).  Empirically the member total still
  equals its delivered price — local = f(applyd) − f(spine) −
  f(state) — so the argmin is the delivered-price comparison the
  routing decision actually makes.  On these cases it picks right;
  it is not a proven-general extractor.

## 6. What remains unfixed

* `_carrier_upgrade` still compares at `profile=None` — the shipped
  pipeline re-inverts a corrected extraction.  The three-line
  `profile=` thread plus a corrections-consumer in `_delivered_cost`
  is the packages-side fix this measurement motivates.
* The corrected member is only *near*-parity on SelectiveSSM
  (0.89–0.98× raw): even the best member carries `IRModule` dispatch
  overhead a hand-written loop doesn't pay.  The honest "raw <
  optimized" ordering is restored on 4/6 SSM cases eagerly, 5/6
  under graph — the rest needs a delivery-side answer (runner), not
  a cost-model one.
* Corrected factors are same-shape measurements — a demonstration of
  the mechanism, not a deployed calibration.  The honest granularity
  is the profile's per-(candidate, bucket) `corrections` table.
* Timing noise is real at these sizes: SelectiveSSM small's
  measured-best member flipped between two probe runs (add 487 µs
  vs 158 µs across runs — first-call folding effects).  The
  corrected-selection verdict holds either way (both members beat
  the shipped `apply`→scan), but single-run medians at ~0.9–1.1× of
  raw are within noise.

## 7. Gates

* `.venv/bin/ruff check tools/executor_cost_probe.py` — pass
* `.venv/bin/ruff format --check tools/executor_cost_probe.py` — pass
* `tools/executor_cost_probe.py` runs to completion (~9 min);
  `--json` writes the machine-readable measured cost table.
* No `packages/` changes — the demonstration rides the public
  `search(cost_fn=…)` seam and post-hoc re-runs of the shipped
  selection stages; the identified fix is documented above, not
  applied.
* Full pytest/coverage intentionally not run (measurement task, RAM).
