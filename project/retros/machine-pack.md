# Retro — the machine law pack: the store's admitted objects ship

Date: 2026-10-06
Context: `project/retros/human-bar.md` measured the yield sitting
admitted-but-unshipped — `affd_step_lift` wins −4.07 % on AdaLNBlock
where the whole human library scores 0, and a `full + store` arm won
6/22 against either side alone.  The store held sixteen
gauntlet-cleared objects no ruleset contained; this task made them
deployable as a first-class ruleset value.

## What shipped

`catopt_discovery.machine_pack` — the loader and the wiring seam:

* `load_pack(store, *, names=MACHINE_STORED, gauntlet_corpus=None)`
  reads a store's `lemmas` rows, keeps the admitted names, rebuilds
  each through `evidence.admit_object`, and requires **full data** —
  `missing_hooks == ()` on the record *and* the rebuilt `Rewrite`.
  Returns a `MachinePack`: the `rules` tuple plus the honest load
  report (`loaded` / `skipped` / `excluded` — every row accounted
  for, `rows == len(loaded) + len(skipped) + len(excluded)`).  Skips
  carry reasons (`reconstruct: …`, `missing hooks: …`,
  `duplicate rule name`, `gauntlet: …`), never crash the load.
* `store_laws(store)` → `tuple[Rewrite, ...]`;
  `store_ruleset(store)` → `RuleSet("machine_store", …)`;
  `machine_default(store)` → `default_rules() + pack` — the one-call
  opt-in spelling for `opt.search(model, x, rules=…)`.  The seam is
  the existing `rules=` parameter; the orchestrator cannot import the
  discovery package (the layers contract pins `catopt_discovery`
  above everything), so the composition helper lives on the discovery
  side and the orchestrator needed **no edits**.  Nothing lands in
  `DEFAULT` — the pack stays a distinct set and the provenance audit
  reads the boundary.

**The "usable" question.**  The store never persists the gauntlet
verdict — `lemmas` rows are object records, `verdicts` rows are the
pipeline's corpus-scoped measurements, and `Gauntlet.usable` is a
runtime report.  The pack's admission filter is therefore the
**provenance ledger**: `MACHINE_STORED`
(`catopt_core.laws.provenance`) names exactly the objects the
committed tests pin `usable` plus the promotion audit's admitted set
— sixteen unshipped names.  `names=None` trusts a store wholesale;
`gauntlet_corpus=` re-runs `run_gauntlet` per row for a store the
ledger does not cover (with the honest caveat that the novelty gate
then refuses objects whose pattern since shipped — the promoted
`mul_unsqueeze_l_id`/`sub_unsqueeze_l_id`/`mul_reshape_l_id` would
read `duplicate` today, so a re-admission under the current library
yields a strict subset of the ledger's snapshot).

**Collision policy.**  The ledger omits the four name-colliding
objects — `om_lift` (the human carrier law ships under that name),
`softsign_fold` / `sdpa_fold_{,div_}nomask` (promoted under their own
names).  Without that, `default + pack` would raise on a
name/different-rule pair (`RuleSet.union` is deliberately strict).

## The load table

Store seeded with the sixteen `MACHINE_STORED` objects — the eight
guarded view strips (`{mul,sub,add}_unsqueeze_l_id`,
`mul_unsqueeze_r_id`, `{mul,add}_transpose_l_id`, `mul_chunk_l_id`,
`mul_reshape_l_id`), the metavar-attr fold `sdpa_fold_div_nomask_g`,
and the seven constructed lifts/composites (`aff_step_lift`,
`aff_scan2_lift`, `affd_step_lift`, `affd_scan2_lift`,
`affd_scan4_lift`, `silu_fold_commuted`, `channel_then_row_scale`) —
spelled verbatim from the committed test constructors:

| metric | count |
|---|---|
| `lemmas` rows | 16 |
| loaded | **16** |
| skipped | 0 |
| excluded | 0 |
| `missing_hooks == ()` members | 16/16 |

The skip paths are exercised in `tests/test_machine_pack.py`: a
procedural-check record reports `missing hooks: check`, an
unknown-kind record reports `reconstruct: …`, a vanished row reports
`reconstruct: row vanished`, a same-name body reports `duplicate rule
name`, a ledger-absent name (`om_lift`) is excluded, and
`gauntlet_corpus=` re-admission refuses a known-false row at
`gauntlet: truth:`.

## The zoo delta

Same harness as the human-bar retro: `Optimizer(backend=TorchBackend())`,
`search` → `lower`, fp64 verify at rtol 1e-4, `dag_cost` on the
result's own cost fn.  Arms: `default` (229 rules), `default+store`
(245), `store` (16).  `Δ%` is the verified extraction's dag-cost
delta; `v` verified, `!` term changed, `—` no fire, `x` unlowerable.

| zoo model | default | default+store | store |
|---|---|---|---|
| GatedAttentionUnit | −0.003 v! | −0.003 v! | +0.000 v |
| AdditiveAttention | — | — | — |
| TalkingHeadsAttention | — | — | — |
| CosineAttention | −0.006 v! | −0.006 v! | +0.000 v |
| FiLMHead | — | — | — |
| AdaLNBlock | +0.000 v | **−4.070 v!** (`affd_step_lift`) | **−4.070 v!** |
| LoRAAdapter | −19.985 v! | −19.985 v! | +0.000 v |
| MLPMixerBlock | 0.000 v!* | 0.000 v!* | 0.000 v!* |
| FNOBlock | x (`fft_rfft2`) | x | x |
| GCNLayer | — | — | — |
| AffineCoupling | — | — | — |
| SoftSlotMoE | — | — | — |
| ExpertChoiceMoE | — | — | — |
| CapsuleRouting | — | — | — |
| ChunkedRetention | — | — | — |
| DeepEquilibrium | — | — | — |
| WaveNetGate | — | — | — |
| TimeCondConv | −4.996 v! | −4.996 v! (+`mul_unsqueeze_r_id`) | −4.996 v! (both strips) |
| MoSHead | 0.000 v | 0.000 v | — |
| CovarianceHead | 0.000 v | 0.000 v | — |
| SplineKAN | −0.003 v! | −0.003 v! | +0.000 v |
| HighwayGate | — | — | — |

`*` the nonlocal param-dedup lift — a pipeline pass, identical
across arms.

**Score: default 5/22 · default+store 6/22 · store-alone 2/22.**
The union is strictly better than the human set — AdaLNBlock joins
exactly as the human-bar retro predicted — and the pack alone
reproduces the two wins it owns (AdaLNBlock −4.07 %, TimeCondConv
−5.0 %, both verified exact, max_rel 0.0).

## Regressions and honest deltas

* **No fire-but-verify-fail anywhere**: every changed extraction
  verified fp64 in every arm (21/21 lowerable; FNOBlock stays
  unlowerable — the binding gap, unchanged).
* **No regression inside `default+store`**: all five human wins hold
  verbatim (`assoc_linear` −20 %, the pad strips −5 %, the micro
  drops); the pack adds one win and one extra fire
  (`mul_unsqueeze_r_id` on TimeCondConv, redundant with the shipped
  `mul_unsq_pad_*` twins' coverage).
* **Versus the retro's raw 20-object store arm** (3/22): the pack
  carries the 16 ledger names, so the store-alone arm loses
  CosineAttention's −0.006 % and SoftSlotMoE's neutral fire — both
  rode the *store* `om_lift`, deliberately excluded for the name
  collision.  The union arm is unaffected (the human `om_lift` is
  already in `default`).  Cost of the honest collision policy: the
  pack cannot co-carry a same-named machine object.
* **Cost-model caveat stands** (same as the zoo/human-bar retros):
  `Δ%` is executor-model dag-cost, not wall-clock; the AdaLN term
  routes generic.

## The store-yield number

**Store yield shipped: 16/16 admitted objects deployable, +1 verified
zoo win (6/22 vs 5/22).**  The admitted-but-unused residue the
human-bar retro flagged is now one `rules=machine_default(db)` call.

## Files

- `packages/catopt-discovery/src/catopt_discovery/machine_pack.py` —
  the loader + `machine_default` wiring seam.
- `tests/test_machine_pack.py` — 13 tests: load/exclude/skip
  honesty, full-data pin, gauntlet re-admission, the RuleSet
  composition, and the AdaLNBlock behavioral pin.
- Probe: `/tmp/machine_pack_probe.py` (ephemeral, per convention) —
  seeds the 16-object store and measures the three-arm zoo table
  above.
- Gates: `pytest tests/test_machine_pack.py` 13 passed (~4 s);
  ruff check / format, ty, vulture, lint-imports clean.
  `radon_ratchet` reports two over-threshold *new* functions in
  `object_synthesis.py` (`_fold_comb`, `specialize`) — the sibling
  task's uncommitted diff, untouched here; `machine_pack` itself is
  under the threshold throughout.  Not committed.
