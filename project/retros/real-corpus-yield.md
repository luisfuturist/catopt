# Retro — real-corpus yield: `shippable = 0` on honest modules

Date: 2026-10-05
Context: `project/retros/corpus-round-4.md` (`shippable: 0 -> 11`),
`project/retros/shippable-audit.md` ("demonstrated usefulness is
corpus-circular"), commit `e2a4051` (the `purpose_built` ledger,
`real_workloads()`, the `--real-only` flag).  The audit *argued*
circularity from `fire_cases` provenance; this retro *measures* the
counterfactual: drop the 46 purpose-built spellings and re-run the
whole intake -> pipeline harness on what is left.

**Headline: `shippable: 0 -> 0`.**  On real modules alone — the
torch-native `nn.*` workloads and the realistic compound assemblies —
the pipeline ships nothing.  All 11 shipped rules, and every one of
the 30 newly-firing guards' *equal* sites, live exclusively on
purpose-built workloads.

## The measurement

Two intake runs from the committed tree, same seed (0), same vocab
(`derived`), no holdout:

```sh
.venv/bin/python -m catopt_discovery.intake \
    --out /tmp/full_intake_corpus.json \
    --tensors /tmp/full_intake_tensors.pt \
    --json /tmp/full_results.json
.venv/bin/python -m catopt_discovery.intake --real-only \
    --out /tmp/real_intake_corpus.json \
    --tensors /tmp/real_intake_tensors.pt \
    --json /tmp/real_results.json
```

(`--out`/`--tensors` are redirected to `/tmp` so the gitignored
`tools/intake_corpus.json` stays as last generated — see *Caveats*.)
Both runs ingested, measured the census delta, and ran the full
baseline-vs-enlarged pipeline delta.  A second probe loaded the two
side-files, re-ran `workload_gen._run_pipeline` on each enlarged
corpus for per-proposal `fire_cases`, and swept each of round-4's 30
newly-firing guards — as the *bare* `Rewrite` from
`generator_pools` — over every corpus case via `real_matches` +
`evidence._guarded_evals` (`torch.manual_seed(0)` per case), bucketing
sites by provenance (`bench` / `model` / real-intake / purpose-built).

No code changed.  `--real-only` is plumbed correctly end to end:
`main` filters `candidates()` through `real_workloads()` before
`ingest`, and ingestion, the census delta, the status table, the
side-file write and the pipeline delta all consume the filtered list.
No plumbing fix was needed.

## Full vs real — the honest table

The registry holds **256** candidates (the round-4 table's "254" was
the *exported* count; 2 torch-native modules are export-rejected in
both runs).  `--real-only` keeps 210 of them; `_PURPOSE_BUILT` drops
46 (9 round-3 + 37 round-4 spellings).

| metric | baseline (bench+models) | corpus + full intake | corpus + real-only intake |
|---|---|---|---|
| intake candidates | — | 256 | 210 |
| exported | — | 254 | 208 |
| ingested (fp64-verified) | — | 201 | 155 |
| census-only (unbound ops) | — | 53 | 53 |
| verify-failed | — | 0 | 0 |
| rejected | — | 2 | 2 |
| probe-eligible (≤120 nodes) | — | 201 | 155 |
| corpus terms | 110 | 364 | 318 |
| op-tuples | 211 | 725 | 702 |
| new op-tuples vs baseline | — | 514 | 491 |
| new shapes | — | 1112 | 1066 |
| new ops | — | 193 | 193 |
| proposals | 54 | 79 | 79 |
| new proposals | — | 27 | 25 |
| proposals with `fires > 0` | — | 69 | 41 |
| proposals with `paid > 0` | — | 33 | 9 |
| **shippable** | **0** | **11** | **0** |

Three readings:

1. **The spellings add no ops.**  All 193 previously-absent ops, and
   the entire census-only backlog, come from real modules — the
   purpose-built subset contributes only 23 op-tuples and 46 shapes of
   pure *re-spelling*.  That is what it was for.
2. **The real corpus is not dead — it is just unshippable.**  41
   proposals still fire on the real corpus (the e-graph applies the
   bare pattern somewhere) and 9 still pay.  Every paying proposal is
   refused at `truth` (conditional equality — the view oracle's
   "no single mechanical feature separates" / `w:v_commutes_view`
   verdicts) except `grammar:sub_to_add_dup` (`not new`) and
   `grammar:mul_distribute` (never lowers cost).  The unguarded
   unconditional equalities — the class that can clear the truth gate
   — simply do not occur on real modules.
3. **`shippable` is entirely the spellings.**  Every one of the 11
   shipped rules' `fire_cases` is a `_PURPOSE_BUILT` workload:

   | rule | firing cases (all purpose-built) |
   |---|---|
   | `sub_self` | `ScalarSelfCancel` |
   | `grammar:pow_one` | `PowOneHead` |
   | `reshape_reshape` | `DoubleReshapeHead` |
   | `square_neg` | `SquareNegHead` |
   | `factor_left` | `SharedFactorMixture`, `QuadraticFeature` |
   | `factor_right` | `SharedFactorMixtureRight` |
   | `factor_sub_left` | `SharedFactorContrast` |
   | `factor_sub_right` | `SharedFactorContrastRight` |
   | `neg_add` | `NegatedSum` |
   | `exp_add` | `ExpProductHead` |
   | `sub_neg` | `SubNegBias` |

   (`factor_left`'s second site `QuadraticFeature` is the
   `d*d + d*e` self-product — also the `FALSE_mul_factor` site.)

## The 30 newly-firing guards — real vs spelling-gated

The sweep answers the question at two levels: does the *pattern*
occur on real modules (sites / bare-rule fires), and does the law
*hold* there (`equal` sites — the region the minted `cond` must
cover; a minted guard's accepted region is a subset of the equal
sites on the measured domain).  Result: **all 30 are
spelling-gated** — not one has an equal site outside `_PURPOSE_BUILT`.
They split into two honest classes:

**23 guards: the pattern does not occur on real modules at all.**
`census:sub_unsqueeze` (→ `PairwiseSubLift`), `exp_add`, `neg_add`,
`sub_neg`, `sub_self`, `square_neg`, `mul_neg_left`, `mul_zero`,
`reshape_reshape`, `factor_left`/`factor_right`/`factor_sub_left`/
`factor_sub_right`, `grammar:FALSE_mul_factor`, `grammar:add_inv`,
`grammar:exp_distribute`, `grammar:neg_distribute`, `grammar:pow_one`,
`grammar:sigmoid_neg`, `grammar:square_mul`, `grammar:sub_add_factor`,
`select_add`, `select_sub` — every equal site is on a purpose-built
workload (one case each; `factor_left` two).  (`select_add`/`select_sub` dedup to
`census:add_select`/`census:sub_select` under `generator_pools`'s
merged-pool key order; on the real corpus they are not even
*proposed* — the select-under-elementwise tuple only exists via
`SelectGateSum`/`SelectGateDiff`.)

**7 wrap/mirror guards: the pattern occurs on real modules, but only
in the corner where the law is false.**  This is the sharper finding:

| guard | real-corpus sites (cases) | equal on real | equal on purpose-built |
|---|---|---|---|
| `mixed:mul_unsqueeze_l_w` | 9 sites / 6 cases — `SelectiveSSM`, `HardDispatch`, `MoEMLP`, `TopKRouter`, `ALiBiAttention`, `LinearAttention` | **0** | 2 (`ChannelGateBroadcast`, `LiftedScalarScale`) |
| `mixed:mul_unsqueeze_r_w` | 5 / 2 — `DiagDenseSSM`, `LinearAttention` | **0** | 2 (`HeadGateBroadcast`, `LiftedScalarScaleRight`) |
| `mixed:sub_unsqueeze_l_w` | 4 / 4 — `CodebookQuantizer`, `ALiBiAttention`, `RelativePositionBias`, `SlidingWindowMask` | **0** | 2 (`ContrastiveCenter`, `LiftedScalarCenter`) |
| `mixed:mul_chunk_r_id` | 3 / 3 — `RegluMLP`, `SwiGLU`, `SwiGLUBlock` | **0** | 1 (`SingleChunkGate`) |
| `mixed:mul_chunk_r_w` | 3 / 3 — same | **0** | 3 |
| `mixed:add_transpose_r_id` | 4 / 3 — `ConformerBlock`, `PostNormBlock`, `PreNormBlock` | **0** | 3 |
| `mixed:add_transpose_r_w` | 4 / 3 — same | **0** | 3 |

Real architectures genuinely spell these shapes — SwiGLU/ReGLU chunk
their gate, ALiBi/relative-position code lifts a bias through
`unsqueeze`, Conformer/pre/post-norm blocks add transposed residuals —
but always with a *non-trivial* second operand (a differently-shaped
gate, `chunks ≥ 2`, a real transpose).  Those are exactly the sites
the minted `bcast-eq`/`axes-noop`/`chunks==1` guard must decline, and
the bare-rule sweep confirms it: `equal == 0` on every real case
(some evaluate `unequal`, some `rhs-err`/`other-err` — none equal).
The purpose-built spellings are the only places the broadcast-trivial
/ no-op corner is inhabited.  (The pipeline's `fires` column still
counts these real sites — the e-graph applies the unchecked pattern
and the truth gate refuses it downstream — which is why the
`*_w` proposals show `intake=1..3` fires even real-only.)

## fp64 verify — real vs purpose-built

| subset | exported | ingested | census-only | verify-failed | pass rate |
|---|---|---|---|---|---|
| full corpus | 254 | 201 | 53 | 0 | **79.1 %** |
| real-only | 208 | 155 | 53 | 0 | **74.5 %** |
| purpose-built alone | 46 | 46 | 0 | 0 | **100 %** |

The gap is a provenance tell: the census-only class is *entirely*
real — torch-native modules whose exported graphs carry ops the
bridge has no binding for (`broadcast_tensors`, `affine_grid`, …) —
while every purpose-built spelling was written inside the bound
vocabulary (it had to verify to fill a guard region).  Both export
rejections are real too (`nn.CTCLoss` — `DynamicOutputShapeException`;
`nn.GaussianNLLLoss` — data-dependent guard), unchanged across runs.
Purpose-built modules do not "verify differently" — they verify
perfectly, *because* verification was part of their spec.

## What the real corpus does yield

Not zero — just nothing *shippable*.  The honest residual:

- **Paying sites exist on real modules.**  Nine proposals have
  `paid > 0` real-only.  Four were already paying on the baseline
  (`mixed:mul_select_l_id`, `census:mul_select`, `linear_factor`,
  `recognize:softmax` — all refused at earlier gates).  Four gain
  paid sites *from real intake workloads*:
  `mixed:mul_transpose_l_id` paid 0 → 4 (`AxialAttention`,
  `ConformerBlock`, `PostNormBlock`, `PreNormBlock`,
  `nn.MultiheadAttention` — real no-op-transpose sites),
  `mixed:sub_unsqueeze_l_id` 0 → 3, `mixed:mul_unsqueeze_l_id` 1 → 2,
  `mixed:add_unsqueeze_r_id` 0 → 1 (`RelativePositionBias` alone).
  (`mixed:mul_unsqueeze_r_id` also fires on `intake:LinearAttention`,
  but paid stays at its baseline 1 — the added fire does not pay.)
- **All of them are refused at `truth`** — they are the conditional
  view-identity family (`id:same_pairing` / `unsq:d_in_pad` /
  "no single mechanical feature separates").  The real corpus's
  yield bottleneck is not firings or pay; it is that every paying
  equality on real modules is *conditional* and the minted guards are
  never admitted (the auto-cond → gauntlet path's known limit,
  unchanged).
- The two proposals minted only on the full corpus,
  `census:add_select` and `census:sub_select`, are the
  purpose-built's own census shadows — they vanish with the
  spellings.

## Bottom line

On real modules alone: **0 rules ship, and 0 of the 30 newly-firing
guards have a non-empty truth region.**  `shippable = 11` is now
*measured* corpus-circular, not merely argued — the same pipeline,
same seed, same gates, minus the 46 spellings, yields nothing
shippable.  The purpose-built corpus did honest work elsewhere: it
proved the guards' regions are inhabitable *in principle* and it
grew the gate stack's coverage, but it cannot count as evidence that
the discovered laws pay on programs nobody wrote to spell them.  The
real corpus's genuine signal is the conditional view-identity sites
that fire and pay — the next yield lever is admitting (or
refuting) those guards, not minting more spellings.

## Caveats / limits

- **Bare-rule sweep, not re-minted `cond`s.**  The guard table uses
  the proposals' own `check`/`derive` plus fp64 `equal`/`unequal`
  per site — for the unguarded laws that is the firing criterion
  exactly; for the view guards the minted `cond`'s accepted region
  is a subset of the equal sites on the measured domain, so
  `equal == 0` on real cases bounds the guarded region from above.
  (`other-err`/`rhs-err` sites were counted but are not firings.)
- **`--real-only` writes the filtered census.**  Without `--out`, a
  real-only run replaces `tools/intake_corpus.json` with the
  real-only subset — consistent with "the file IS what was ingested",
  but the full side-file is then gone until re-generated.  These
  runs redirected `--out`/`--tensors` to `/tmp`; the repo state is
  untouched.  Worth a flag or a docstring line if the semantics ever
  surprise — recorded, not fixed (measurement only).
- **Measured, not proven.**  Same seeded draws and the same
  `synth_limit`-bounded windows as the retro whose numbers this
  checks; a region outside the corpus is unmeasured by construction.
- **The ledger is a judgement call at the edges.**  `_PURPOSE_BUILT`
  marks the 46 round-3/4 pattern spellings; a couple of earlier
  compound candidates (e.g. `_ScalarRsub`) are borderline but stayed
  "real" — the honest reading does not change if they move.
- **Measured at HEAD `e2a4051`.**  Both intake runs and the guard
  sweep completed before a concurrent in-flight sibling edit
  (`oracle.py`, `laws/tensor.py`, `coherence*.py`, tests) landed in
  the working tree ~21:49; the probe's pipeline re-runs reused the
  already-imported modules, so every number here is the committed
  tree's.  Re-run the same two commands to re-measure after that
  edit settles.
