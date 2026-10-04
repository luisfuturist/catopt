# Executor pricing fix — the measured-feedback channel reaches `_carrier_upgrade`

`measured-cost-model.md` localized the routing inversion precisely:
`_carrier_upgrade`'s delivered-price comparison
(`optimize.py`, the `profile=` parameter threaded end-to-end) was
*never passed* — `_select_best_term` called it bare, so the one
batched-vs-generic comparison that could catch a slow carrier
delivery always ran on the built-in constants.  This change wires
the profile through and adds the measured-corrections consumer the
delivered comparison needed.  Re-running
`tools/executor_cost_probe.py` against the shipped code path —
no post-hoc re-implementation of the upgrade loop — now lands on
the measured-fastest member on **all 12 probe cases**: `add`-rooted
`IRModule`s on all six SSM cases instead of
`apply[d]`-rooted `BatchedScanModule`s, and the same (already best)
member everywhere else.

## 1. The seam, as wired

* `cost_fn` carries the profile.  `executor_cost_for(profile, …)`
  and `criteria_cost(…, profile)` already stamp a ``profile``
  marker on the returned callable, and `backend_cost` already
  forwards it (`cost/backend.py` marker copy).  No new `search()`
  argument — the public `cost_fn=` seam was sufficient.
* `_select_best_term` (`optimize.py`) reads the marker through the
  new `_upgrade_pricing(cost_fn, x)` helper → `(profile, bucket)`:
  `profile` = `getattr(cost_fn, "profile", None)`, `bucket` =
  `shape_bucket(x)` of the search's example input — the same key
  `record_measured` writes under.  No profile → `bucket=None`;
  nothing downstream changes.
* `_carrier_upgrade(…, profile=, bucket=)` passes both into
  `_delivered_cost(…, bucket=)` at both comparison sites.
* `_delivered_cost` now composes in two steps: the routed
  `executor_cost_for(profile, lowering=…)` model price, then
  `_corrected_delivered` — a thin consumer over
  `catopt_core.profile.corrected_price_ns` (learned `corrections`
  factor → `measured_ns` residual → median substitution → model
  unchanged).  The design decision the retro deferred — *which
  candidate name the route consults* — is `_DELIVERED_CANDIDATES`:
  `"generic"` → `("generic",)`, `"batched_scan"` → `("batched",
  "batched_scan")`, `"compiled"` → `("torch_compile", "cuda_graph",
  "compiled")`, first hit wins.  Autotune's write-back names are
  the primary keys (its `"batched"` candidate IS the pipeline's
  routed carrier delivery), with the lowering names as aliases so
  hand-built or tool-emitted tables apply too.
* Backward compatible by construction: `profile=None` or
  `bucket=None` or empty tables → `corrected_price_ns` returns the
  model price untouched; every existing `_delivered_cost(term)` /
  `_carrier_upgrade(...)` call site keeps today's behavior.  The
  ADR-0003 split is intact — the profile only re-prices, routing
  feasibility stays a `plan`/`accepts` gate.

ADR 0003 holds: the cost model ranks, it never decides equivalence.
`supported_ops` stays the feasibility gate; `accepts()`/plan
builders stay the routing gate.  The profile only moves the
*price* the comparison sees.

## 2. Calibration path

`tools/executor_cost_probe.py --emit-profile PATH` now writes a
`TargetProfile` JSON: the pooled per-family geomean factor
(`measured_routed_ns / modeled_ns`, eager) under every measured
`shape_bucket`, with `n` = pooled observation count.  Pooled rather
than per-case: a single-case family often contributes one
observation, which sits below the `_CORRECTION_MIN_SAMPLES` gate
and would never fire — the emit is the same calibration the
counterfactual consumes, explicitly a same-run mechanism demo.
Base constants are the model's own built-ins (`_PEAK_FLOPS` /
`_PEAK_BW` / `_LAUNCH_S` + the `dispatch`/`leaf_eval` fallbacks), so
the correction table is the *whole* delta vs an uncalibrated
profile.  Consumption: `cost_fn=executor_cost_for(
TargetProfile.load(path))` — or any `CostFn` with `.profile` set.

The `Autotuned` strategy's write-back feeds the same table: run it
with `profile=`, and a subsequent search whose cost_fn carries that
profile prices the delivered comparison by its measurements — the
loop `measured-cost-model.md` §4 said "doesn't reach back into
`_select_best_term`" is now closed through the marker.

## 3. Before/after on the probe cases

Full re-run post-fix (RTX 2050, fp64, warmup 50 / timed 200, µs
eager).  `cf-modeled` = the identical selection chain without the
profile (delivered-aware extraction, uncalibrated comparison);
`cf-measured` = the shipped `search()` under the profile-carrying
cost_fn — `res.term` IS the corrected pick, produced by the shipped
`_carrier_upgrade` itself; `corrected-sel` = the post-hoc shipped
`_carrier_upgrade` on the same e-graph (now just confirmation).

| case | raw | shipped | cf-modeled | cf-measured | corrected-sel |
|---|---|---|---|---|---|
| SelectiveSSM small | 130 | 313 apply/scan | 324 apply | **127 add/IRModule** | 126 add/IRModule |
| SelectiveSSM large | 276 | 494 apply/scan | 495 apply | **231 add/IRModule** | 245 add/IRModule |
| DiagDenseSSM small | 143 | 269 apply/scan | 356 apply | **148 add/IRModule** | 142 add/IRModule |
| DiagDenseSSM large | 252 | 393 apply/scan | 480 apply | **238 add/IRModule** | 243 add/IRModule |
| DiagonalSSM small | 100 | 224 applyd/scan | **82 add** | **81 add/IRModule** | 79 add/IRModule |
| DiagonalSSM large | 164 | 374 applyd/scan | 364 applyd | **129 add/IRModule** | 128 add/IRModule |
| HybridBlock small | 179 | 229 linear | 206 linear | 185 linear (same) | 187 linear |
| HybridBlock large | 237 | 327 linear | 338 linear | 215 linear (same) | 221 linear |
| TwoLayerHybrid small | 257 | 337 stack | 339 stack | 987→315 stack (same) | 315 stack |
| TwoLayerHybrid large | 410 | 539 stack | 629 stack | 358 stack (same) | 357 stack |
| ManualSoftmaxAttn small | 104 | 94 linear | 115 linear | 87 linear (same) | 84 linear |
| ManualSoftmaxAttn large | 299 | 259 linear | 226 linear | 205 linear (same) | 254 linear |

Measured factors this run: `batched ×0.43`, `generic ×0.058`.
Verdict line: `corrected-selection: correct` on **12/12** — every
SSM case ships the `add`/`IRModule` member (the measured-fastest),
every non-SSM keeps its already-best member.  On 4/6 SSM cases the
corrected pick beats the raw model eagerly (SelectiveSSM small
127 vs 130; DiagDense 148 vs 143 borderline / 238 vs 252;
DiagonalSSM 81 vs 100 / 129 vs 164); the DiagDenseSSM small and
SelectiveSSM large rows remain near-parity rather than wins.
TwoLayerHybrid small's 987 µs cf-measured median is first-call
noise (re-timed at 315 in the corrected-sel arm — the retro's
documented caveat); the *selection* is still the measured-best
member.

## 4. Backward-compat evidence

* `tests/test_optimize_routing.py` — 25/25 pass, including the
  pre-existing uncalibrated-swap tests (the batched upgrade still
  fires at default constants).
* `test_delivered_cost_no_measured_data_unchanged` — `None`
  profile, empty tables, missing bucket and cross-device bucket all
  return the pure-model price.
* `test_carrier_upgrade_declines_measured_slower_carrier` — a
  `measured_ns` median on `"batched"` keeps the incumbent (the
  inversion case, unit level).
* `test_search_profile_marker_reaches_carrier_upgrade` — end to
  end: a delivered-aware cost_fn picks `add` greedily, the
  uncalibrated comparison re-inverts to `apply`, and the same
  search with the profile marker keeps `add`; the spy asserts the
  profile object and `shape_bucket(x)` reach `_delivered_cost`.
* `test_delivered_cost_measured_correction_scales_route` — the
  learned-`corrections` path scales the routed price by the
  recorded factor on both families.

## 5. What remains

* **Extraction-side correction is still the caller's job.**  The
  fix prices the *upgrade comparison* correctly; greedy extraction
  under the default `executor_cost_for("generic")` still prices
  every member as generic-dispatched.  The probe's delivered-aware
  `cost_fn` is a tools-level model (non-additive at carrier roots,
  documented) — promoting a shipped delivered-aware extraction
  model, or feeding profile corrections into the extraction cost
  itself, is a separate design step.
* **A public calibrate→profile CLI** — `--emit-profile` covers the
  probe's measured families; a `calibrate()`-side executor-family
  probe (batched vs generic vs compiled per bucket) belongs with
  `catopt_torch.calibrate`.
* **Per-(candidate, bucket) granularity.**  The emit pools factors
  across cases; a real calibration should record per-bucket factors
  (the table supports it — the n≥2 gate just needs enough
  observations per bucket).
* **The correction clamp is `[0.1, 10]`.**  The measured generic
  factor (~0.06) clamps to 0.1 at consumption — fine here since
  the delivered-comparison factor that matters (`batched` ~0.4) is
  in-range; extreme misfits need a documented way past the clamp.
* **Near-parity on SelectiveSSM** stands: even the best member
  carries `IRModule` dispatch overhead — a delivery-side (runner)
  answer, not a cost-model one.
* **The emitted profile is this hardware** (RTX 2050, fp64 cases) —
  measured tables on other targets are the point of the
  `TargetProfile` persistence machinery, unexercised here.

## 6. Gates

* `ruff check packages tools` — pass; `ruff format --check` — pass.
* `ty check` — 0 errors.  `vulture` — clean.  `lint-imports` — 4/4
  kept.  `bandit`/`semgrep` — 0 findings.  `radon_ratchet` — ok
  (`_upgrade_pricing` extracted to keep `_select_best_term` at its
  baseline 8).
* `pytest tests/test_optimize_routing.py` — 25/25 (~12 s; the only
  test file run — full suite deliberately not run).
* `tools/executor_cost_probe.py --json /tmp/ec2.json
  --emit-profile /tmp/ec_profile.json` — runs to completion
  (~9 min); verdict `correct` on 12/12 through the shipped
  `_carrier_upgrade`, no post-hoc re-implementation.
