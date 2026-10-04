# Calibrated profiles as a public path — `delivered_cost_for` + `tools/calibrate_profile.py`

`executor-pricing-fix.md` wired the measured-feedback channel end to
end but left the claim "the cost model is measured, never assumed"
true only on this dev box: `--emit-profile` writes a same-run
artifact, and the corrected extraction ran on the probe's
tools-level `_delivered_cost_fn`.  Two gaps remained, both named in
that retro's §5: a *public calibrate→profile CLI*, and a *shipped
delivered-aware extraction model*.  This change ships both — the
public path now runs the probe's 12/12 corrected routing through
production code only.

## 1. The seam, as wired

No new pipeline parameter.  The `cost_fn=` port was sufficient —
the addition is one public factory in
`catopt_orchestrator.optimize`:

* `delivered_cost_for(profile=None, *, x=None, bucket=None,
  compiled=False) -> CostFn` — `_delivered_cost` lifted into an
  extraction model: carrier-apply-rooted plannable terms bill under
  `"batched_scan"`, everything else under `"generic"`, and the
  routed price is corrected by `corrected_price_ns` under
  `(candidate, bucket)` — the same consumption contract the
  upgrade comparison already ran.  `x=` derives the
  `shape_bucket` the corrections were recorded under (an explicit
  `bucket=` wins); `bucket=None` is the delivered-aware but
  uncalibrated model — the probe's `cf-modeled` arm, now a shipped
  value.  The closure carries the `profile` marker
  `backend_cost` forwards, so `_select_best_term` prices
  `_carrier_upgrade` off the same table under the search input's
  OWN bucket — selection and upgrade consume one calibration.
* `_delivered_cost` gained a `memo` parameter (threaded into the
  per-lowering `executor_cost_for` call) so `delivered_cost_for`
  shares the extraction memo — `_memo_dispatch` sees `memo` in the
  signature and binds the shared dict.  Every existing call site
  keeps today's behavior.
* Exported as `catopt_orchestrator.delivered_cost_for`.

ADR 0003 holds: the corrections only re-price — feasibility stays
`supported_ops`/`accepts`, and the non-additive-at-carrier-roots
caveat is the documented one `fused_cost_for` and
`lowering_aware_cost_for` already carry.  The alternative — a
`profile=` argument on `search()` — was rejected: it could feed
`_resolve_cost_fn`'s defaults and the upgrade comparison, but it
cannot make *extraction* delivered-aware without changing the
default selection model for every caller.  The inversion needs the
delivered-aware price AT extraction, which is a choice of cost
model, i.e. exactly what `cost_fn=` is for.

## 2. The calibration entry point

`tools/calibrate_profile.py` — the documented "run this on your
machine" script:

* Phase 1: `catopt_torch.calibrate.calibrate()` — the measured
  constants sweep (peaks, launch, dispatch, leaf-eval, graph
  overhead, `op_kernel_ns`).
* Phase 2 (CUDA only): imports — not duplicates — the probe's
  measured-table machinery: `_probe_case` searches each of
  `law_wallclock._cases()`, lowers every root alternative through
  its routed executor AND the generic one, times on CUDA;
  `_correction_factors`/`_correction_ratios` pool the
  measured/modeled ratios per family; a new `_corrections_table`
  helper (extracted from `_emit_profile`, which now calls it)
  assembles `{candidate: {bucket: {"factor", "n"}}}` under every
  measured `shape_bucket`.  `--input-device` keys the buckets for
  searches whose example input isn't CPU-side; `--cases`,
  `--warmup`, `--iters`, `--quick` bound the sweep;
  `--skip-executor-corrections` writes constants-only on any
  machine.
* The corrections merge into the calibrated base via
  `dataclasses.replace` — unlike `--emit-profile` (built-in
  constants + corrections), the emitted artifact is fully measured.
  `TargetProfile.load(path)` reads it back; `--save` persists under
  `profiles_dir()`.

## 3. End-to-end proof on the probe cases

Full run on the dev box (RTX 2050): `tools/calibrate_profile.py`
→ `calibrated_profile.json` (measured factors this run:
`batched ×0.518`, `generic ×0.067` — the generic factor clamps to
0.1 at consumption, cf. the retro's `[0.1, 10]` note).  Then a
plain `Optimizer(backend=TorchBackend())` — `opt.search(model, x,
cost_fn=delivered_cost_for(profile, x=x))` + `opt.lower(res, x,
verify=True)`, no monkeypatch, no hand-rolled cost fn — on all 12
probe cases:

| case | picked root | route | module | verified |
|---|---|---|---|---|
| SelectiveSSM small / large | add | generic | IRModule | yes |
| DiagDenseSSM small / large | add | generic | IRModule | yes |
| DiagonalSSM small / large | add | generic | IRModule | yes |
| HybridBlock small / large | linear | generic | IRModule | yes |
| TwoLayerHybrid small / large | stack | generic | IRModule | yes |
| ManualSoftmaxAttn small / large | linear | generic | IRModule | yes |

**6/6 SSM cases ship the generic `add` member**, and every non-SSM
keeps its measured-best member — the identical member-level result
the probe's `cf-measured`/`corrected-sel` arms produced through the
tools-level cost fn (12/12, `executor-pricing-fix.md` §3).  The
difference is that the selection now comes from shipped API:
`res.term` IS the corrected pick produced by `extract_best` +
`_carrier_upgrade` under `delivered_cost_for`.

## 4. Honest boundaries

* **Extraction-side correction needs the delivered-aware model.**
  `cost_fn=executor_cost_for(profile)` calibrates constants and the
  upgrade comparison, but greedy extraction still prices every
  member generic — `_carrier_upgrade` can only swap a carrier
  member IN, never demote a carrier pick.  The public path to the
  SSM routing fix is `delivered_cost_for(profile, x=x)`; forgetting
  the `x=` leaves extraction uncalibrated (delivered-aware,
  model-priced — `cf-modeled` behavior).
* **Corrections are pooled per-family, per-bucket.**  The emitted
  table puts one geomean factor under every measured bucket —
  same-run granularity, transferred across buckets by
  `corrected_price_ns`'s dampened nearest-same-device lookup.
  Per-case factors need n≥2 observations each; the machinery
  supports it, the sweep doesn't produce it.
* **`n` is the pooled count, not a confidence.**  `n ≥ 2` is the
  `corrected_price_ns` gate, not a statistical claim.
* **The correction clamp `[0.1, 10]` bounds transfer.**  Measured
  generic ~0.067 clamps to 0.1 — still correct-signed here, but a
  target where generic is even cheaper hits the floor.
* **The probe arm is CPU-input buckets by default** (the probe's
  convention: search inputs are CPU tensors, timing happens on
  CUDA).  A search whose example input lives on CUDA needs
  `--input-device cuda` — cross-device buckets never interpolate.
* **Profile is this hardware + dtype** (fp64 probes; constants at
  `--dtype`).  Other targets/dtypes recalibrate — that is the point
  of the persistence machinery.

## 5. Gates

* `ruff check packages tools` — pass; `ruff format --check` — pass.
* `ty check` — 0 errors.  `vulture` — clean.  `lint-imports` — 4/4
  kept.  `bandit`/`semgrep` — 0 findings.  `radon_ratchet` — ok
  (`delivered_cost_for` and `_delivered_cost`'s new `memo` arm are
  below threshold).
* `pytest tests/test_optimize_routing.py` — 29/29 (4 new:
  routed-lowering billing + profile marker, compiled arm, bucket
  keying/x-derivation, and the shipped search keeping the `add`
  pick where the unbucketed search re-inverts — no spies, no
  hand-rolled cost fn).
* `tools/calibrate_profile.py --out /tmp/calibrated_profile.json` —
  full 12-case run (~5 min) completed; `TargetProfile.load` round
  trip verified; `--quick --skip-executor-corrections` smoke on the
  constants-only path.
