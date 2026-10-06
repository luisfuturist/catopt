# Retro: the shipped-guard tightening — 7 guarded laws, one class

**The derivable-gate audit's follow-up** — the 7 shipped guarded laws
the audit measured unclean, plus the `gqa_absorb_repeat` sweep crash.
Date: 2026-10.

## What the audit left on the table

`project/retros/derivable-gate.md` swept every shipped guarded law's
guard-accepted region and found **7** with measured counterexamples —
human-admitted, sound-*intent* laws whose guards accept sites where the
equality is false (`unequal`) or the minted RHS cannot denote
(`rhs_err`).  The gate now *refuses* such objects, but the shipped laws
were never gauntleted (human-admitted), so the audit was the first time
their regions were measured at all.  Same class as the `swiglu_fuse`
rank-1 fix (`61d0d99`): a guard that is tight enough for the corpus and
loose enough to admit a degenerate binding.

This change fixes all 7, and the `gqa_absorb_repeat` crash the audit
recorded (`typing._infer_op_shape`'s `d % (…)` on a str attr metavar).

## The diagnosis — each rule, its bad sites, the exact miss

Every bad site is a **shape the guard did not check**, not a wrong
predicate.  The audit's `_site_outcome` distinguishes `rhs_err` (torch
raises evaluating the minted RHS) from `unequal` (the two sides differ);
the fix is a declarative clause that declines exactly the region where
the identity's *definedness* (or truth) fails.

| rule | bad sites (sample) | the exact miss |
|---|---|---|
| `select_mul` | `u=(2,3)`, `v=(2,3,1)`, `D=0` — 8 rhs-err | `dim-eq-attr` pins `u[D]==v[D]`, but `mul(u,v)` must *broadcast*; `u⊙v` is ill-typed off the selected axis |
| `distribute_matmul_over_add` | `a=(1,)`, `b=(4,)` — 4 rhs-err | `rank-eq` is not extent-equality: rank-1 addends broadcast yet `matmul(W,a)`/`matmul(W,b)` cannot both contract |
| `weight_distribute_matmul` | `W=(1,)`, `W2=(4,)` — 4 rhs-err | same, on the summed weights |
| `weight_distribute_linear` | `W=(1,)`, `W2=(4,)` — 4 rhs-err | same; the minted `linear(x,W)+linear(x,W2)` cannot denote |
| `linear_channel_scale` | `x=()`, or `x=(1,)` vs `W=(4,)` — 16 rhs-err | the channel fold is a value identity only where `F.linear(x,W)` denotes; the guard never checked `x` at all |
| `linear_row_scale` | `W=(1,)`, `r=(1,)`, `x=(1,)` — 3 unequal, 9 rhs-err | the row scale must broadcast *into* `linear(x,W)`'s output; a rank-1 `r=(1,)` against a scalar output grows the RHS by an axis |
| `linear_row_scale_rev` | same 3 unequal | the reverse spelling of the same law |

The `linear_*` misses are the interesting ones: `linear` is `x @ W.T`,
and neither the channel nor the row guard ever constrained `x` or the
*output*.  The `linear_row_scale` `unequal` is a genuine value
counterexample — `linear((1,)∘(1,), (1,)) = ()` but
`mul(linear((1,),(1,)), (1,)) = (1,)`.

## The fix — declarative clauses, no new atoms

Every addition composes *existing* cond predicates and shape specs; the
law stays pure data (JSON-safe, no callables).

- **`select_mul`** — `∧ ("shaped", ("bcast", "u", "v"))`.  The `bcast`
  spec resolves `_broadcast(u, v)`; `shaped` is true iff it is a real
  tuple (not `_INVALID`).  That *is* "`mul(u,v)` denotes".
- **`_COND_MM_ADDENDS` / `_COND_MM_WEIGHTS`** — the rank-equal branch
  gains `("shape-eq", a, b)`.  Equal rank does not mean equal extents;
  rank ≥ 2 differing extents stay covered by the second branch (so this
  tightens *only* the rank-1 corner the audit measured).
- **`_COND_CHANNEL_SCALE`** — `∧ _COND_LINEAR_XW`, where
  `_COND_LINEAR_XW = ("mm-shape-ok", "x", _SPEC_WT)` and
  `_SPEC_WT = ("transpose-out", "W", None, None)` is `W.T`.  `F.linear`
  is `x @ W.T`, so `mm-shape-ok(x, W.T)` *is* "`F.linear(x,W)`
  denotes" — the contraction and the batch broadcast, exactly torch's
  `x.matmul(W.t())` lowering.
- **`_COND_ROW_SCALE`** — `∧ ("bcast-into", "r", _SPEC_LINEAR_OUT)`,
  where `_SPEC_LINEAR_OUT = ("mm-out", "x", _SPEC_WT)` is
  `linear(x,W)`'s output shape.  `bcast-into(r, out)` is true iff
  `broadcast(r, out) == out` — *r* adds no axes, which is exactly the
  naturality condition.

No atom was added: `shaped` + `bcast`, `shape-eq`, `mm-shape-ok` +
`transpose-out`, and `bcast-into` + `mm-out` are all existing
vocabulary.  `transpose-out` with `None` axis args defaults to the
last-two swap (`-2`/`-1`) and is the identity on a rank-1 weight —
matching torch's 1-D-weight `F.linear`.

## The crash — a str attr metavar in shape inference

`gqa_absorb_repeat`'s pattern carries `unsqueeze(k, dim="UDk")` with
`UDk` an *unbound* attr metavar.  The sweep's `_operand_shape` runs
shape inference on the uninstantiated pattern, so `attr_of` returns the
**string** `"UDk"`, and `_infer_op_shape`'s `unsqueeze` branch did
`d % (len(base) + 1)` → `TypeError: not all arguments converted during
string formatting`, aborting the whole sweep.

Fix: the branch declines (`return None`, the contract's
unprovable→unknown) when `d` is not an `int`.  `acc=0` on the synth
region is the honest consequence — the uninstantiated bindings are
unknowable — and the **real** firing is untouched (real graphs bind
`UDk` to an int before shape inference ever sees it;
`test_gqa_absorb_repeat_kv` still fires and verifies).

## Measured — per-rule before / after

Guard-accepted synth region at the default 360-site window
(`evidence._guarded_evals ∘ evidence._synth_sites`):

| rule | before eq/ne/rerr/acc | after eq/ne/rerr/acc |
|---|---|---|
| `select_mul` | 57 / 0 / **8** / 81 | 57 / 0 / **0** / 57 |
| `distribute_matmul_over_add` | 5 / 0 / **4** / 48 | 5 / 0 / **0** / 36 |
| `weight_distribute_matmul` | 5 / 0 / **4** / 41 | 5 / 0 / **0** / 29 |
| `weight_distribute_linear` | 4 / 0 / **4** / 41 | 4 / 0 / **0** / 29 |
| `linear_channel_scale` | 24 / 0 / **16** / 128 | 24 / 0 / **0** / 25 |
| `linear_row_scale` | 22 / **3** / **9** / 232 | 22 / **0** / **0** / 23 |
| `linear_row_scale_rev` | 22 / **3** / 0 / 232 | 22 / **0** / **0** / 23 |
| `gqa_absorb_repeat` | **CRASH** | 0 / 0 / 0 / 0 (no crash) |

Every `equal` count is preserved exactly (the fix declines only
non-equal sites); every `unequal`/`rhs_err` is gone.  A full re-sweep
of **all 36 shipped guarded rules** now reports `unclean: none` (the
other 29 were already clean).

The shared-cond siblings move with the tightened guards and stay clean:

| rule | before eq/ne/rerr/acc | after |
|---|---|---|
| `factor_matmul` | 5 / 0 / 0 / 48 | 5 / 0 / 0 / 36 |
| `weight_factor_matmul` | 5 / 0 / 0 / 41 | 5 / 0 / 0 / 29 |
| `weight_factor_linear` | 4 / 0 / 0 / 41 | 4 / 0 / 0 / 29 |
| `linear_channel_scale_rev` | 24 / 0 / 0 / … | 24 / 0 / 0 / 25 |

## The composite the audit's hole admitted is cured

`channel_then_row_scale = compose(linear_channel_scale_rev,
linear_row_scale)` inherited `linear_row_scale`'s rank-1 blind spot (the
audit measured 19 equal / **1 unequal**), which is why the derivable
gate refused it.  Guard transport is the premises' conjunction, so
tightening `linear_row_scale` tightened the composite: its region is now
**19 / 0 / 0** — clean, so the derivation-backed composite clears the
truth gate.  The premise fix *cured* the inherited counterexample; the
gate no longer has to refuse it.

## Tests

- `tests/test_select_laws.py` — `test_select_mul_declines_when_operands_do_not_broadcast`
  (`u=(2,3)`, `v=(2,3,1)`: `u[D]==v[D]` yet `u⊙v` is ill-typed →
  declined; same-rank broadcasting operands still accepted).
- `tests/test_cond_laws.py` — `test_mm_addends_rank_equal_requires_shape_eq`,
  `test_mm_weights_rank_equal_requires_shape_eq`,
  `test_linear_channel_scale_requires_linear_well_typed`,
  `test_linear_row_scale_requires_r_broadcasts_into_output` — the
  per-rule accept/decline verdicts on the bad and good bindings.
- `tests/test_intake_defects.py` — `test_infer_op_shape_declines_on_str_attr_metavar`
  (the exact crash) and `test_gqa_absorb_repeat_sweep_no_longer_crashes`
  (the sweep completes; the guard is total).
- `tests/test_typing_edges.py` — a `SHAPE_CASES` param
  `unsqueeze/str-attr-metavar-declines` (the `_infer_op_shape` arm).
- **The derivable-gate retro's gate pins** were re-pointed at
  *purpose-built* unclean guards (the pre-tightening guards), so the
  gate's logic (`derivation does not override a measured
  counterexample`) stays pinned independently of the shipped library's
  now-clean regions:
  - `tests/test_discovery_gauntlet.py` —
    `test_derivation_does_not_override_a_measured_counterexample`,
    `test_derivation_cannot_override_a_measured_rhs_err`,
    `test_derivation_cannot_override_an_unequal_site` now use
    `_LOOSE_ROW_SCALE` / `_LOOSE_SELECT_MUL`; a new
    `test_tightened_premise_cures_the_composite` pins the composite's
    clean region.
  - `tests/test_discovery_synthesis.py` —
    `test_declarative_transport_refused_by_an_inherited_counterexample`
    → `test_declarative_transport_cleared_by_the_tightened_premise`
    (the composite now clears), and
    `test_transport_inherits_the_premise_blind_spot` →
    `test_transport_carries_the_tightened_premise_guard` (the
    transported clause declines the old blind-spot binding).

## Honest limits

- **The fix is measured-region-tight, not a proof.**  `shape-eq` on the
  rank-equal branch declines the rank-1 mismatch the bank found; the
  rank ≥ 2 branch still trusts the audit's argument (leading-axis
  padding) and would accept a rank-2 pair whose contraction axes
  disagree (`a=(2,3)`, `b=(1,3)`) — a latent site outside the bank.
  `mm-shape-ok` composition would close it but changes the accepted
  region more than the "smallest addition" bar wants; recorded, not
  taken.
- **`bcast-into` needs the output shape to resolve.**  A row-scale site
  with a fully symbolic `x`/`W` (no inferable output shape) now
  declines.  Conservative, never wrong — the same posture the other
  shape-reading predicates take.
- **The `linear` "well-typed" read rides `transpose-out`.**  It is exact
  for the 1-D and 2-D weight spellings `F.linear` supports; a rank ≥ 3
  weight (which torch refuses) is not distinguished.
- **`gqa_absorb_repeat`'s synth region is empty** after the crash fix
  (the guard declines every uninstantiated binding).  That is the
  crash's honest residue, not a new defect: the rule is human-admitted
  and its real firing is unaffected.

## Follow-up — `gqa_absorb_repeat` rank-4 (the chained-bindings hole)

Same class, one more law — the eighth.  `chained-bindings.md`'s
operand-chained attr domains made the rank-3 binding *mintable*,
surfacing the latent hole its §6 recorded: `q=(2,6,4)`,
`k=v=(2,3,4)` with the consistent `unsq(-2) → expand (2,3,2,4) →
reshape (2,6,4)` chains satisfies `repeat-chain`/`attr-eq-attr`/
`repeat-heads` (`q[-2]=6 == k[-2]·r`), but the minted `sdpa(...,
enable_gqa=True)` RHS cannot contract — `enable_gqa` repeats along
the head axis at `-3`, and the pattern's `transpose(1, 2)` only
lands the heads there when the leaves are rank-4 `(b, t, h, d)`.
Measured: `rhs-err` (`RuntimeError: Expected size for first two
dimensions of batch2 tensor to be: [2, 6] but got: [2, 3]`); the
enumerator mints 6 such bases per rank-3 leaf binding (the `{D, C}`
product).

Fix: `("rank", "q", "==", 4)`, `("rank", "k", "==", 4)`,
`("rank", "v", "==", 4)` prepended to `_COND_GQA_ABSORB` — the
existing `rank` atom, no new vocabulary.  Every `equal` site is
preserved: all real fires are rank-4 (`test_gqa_absorb_repeat_kv`,
`test_gqa_absorb_real_firing_and_veto`), and every non-rank-4
accepted site was a `rhs-err`/`unequal` anyway (the repeated axis
lands at `-2`, not `-3`, off rank 4).  The synth window stays
`0/3000` all-declined — clean — and the accepted region is
non-empty: the rank-4 corner binding clears and evaluates `equal`
(the equal corner's ~5M-env depth remains the cap-depth limitation
`chained-bindings.md` §6 names, not a guard failure).

Tests: `test_cond_laws.test_gqa_absorb_repeat_requires_rank4_operands`
(the decline, the measured `rhs-err`, the corner `equal`),
`test_laws_rewrite_edges.test_gqa_absorb_declines_a_rank3_term`
(`apply_rule` mints no enodes) + a rank-3 decline case in
`test_check_gqa_absorb_both_directions`; the `test_discovery_oracle`
pin's hand-built binding moved to the rank-4 corner spelling.

## Gates

`pytest tests/test_cond_laws.py tests/test_law_serialize.py
tests/test_select_laws.py tests/test_rulesets.py
tests/test_discovery_gauntlet.py` plus the touched files
(`test_discovery_synthesis.py`, `test_intake_defects.py`,
`test_typing_edges.py`, `test_typing.py`, `test_contracts.py`,
`test_synthesis_guards.py`) — **658 passed, 19 skipped**.  The
`linear`/`row`-scale firing tests (`test_normlinear_folds_channel_scale`,
`test_row_scale_rejects_data_scale`,
`test_channel_scale_rejects_row_scale`,
`test_rmsnorm_roundtrip_and_optimize`, `test_gqa_absorb_repeat_kv`, …)
pass unchanged.  `ruff check packages tools`, `ruff format --check
packages`, `ty check`, `vulture`, `lint-imports`, `radon_ratchet` — all
clean; the new `_infer_op_shape` branch is covered by the new tests (no
`# pragma: no cover`).  No new suppressions; no new cond atom.
`evidence.py`, `oracle.py`, `intake.py`, `object_synthesis.py`,
`meta_game.py` untouched.
