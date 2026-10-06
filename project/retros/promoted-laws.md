# Retro: honest promotion — the admitted objects, audited for the library

Date: 2026-10-05
Context: follow-up to plan 0017 / the admission-gauntlet and auto-cond
rounds.  The store now holds a dozen objects that cleared the
gauntlet — reconstruct, full-data, measure, truth, novelty,
typed-pay, closure, cert all green.  That is *admission*, not
promotion: the gauntlet proves an object is sound, paying and
serializable on the measured domain, but it does not decide whether
the object is a law the library lacks, a respelling of one it has,
or corpus-specific material that should not ship.  This retro records
that call.

Promoted code lives in `catopt_core.laws.tensor`; nothing under
`catopt_discovery` changed (`oracle.py`/`evidence.py`/`intake.py`/
`opmeta`/`opdata` untouched).

## The rule

For each admitted object: **(a)** a genuine library addition — a law
the library lacks that fires on real modules, pays, and is clean on
its declared region; **(b)** duplicate/subsumed — an existing law
already covers it (or a strictly more general law does); **(c)** not
worth shipping — corpus-specific, no pay, or no clean region.

## Promotion table

| object | verdict | shipped as | evidence |
|---|---|---|---|
| `softsign_fold` | **promoted** | `softsign_fold` | `div(x, abs(x)+1) → softsign(x)`; unguarded, 3 ops → 1 kernel; sweep 17/17 equal; fires the `nn.Softsign` intake spelling |
| `sdpa_fold_nomask` | **promoted** | `sdpa_fold_nomask` | mask-free `matmul(softmax(q·kᵀ), v)` — the `nn.MultiheadAttention` default path the `add`/`masked_fill` family never matched; generalized to the family guard (`_COND_SDPA_BASE`) plus the measured `rank(Q,V) >= 2` floor (rank-1 `Q` mints a non-denoting `sdpa` — `rhs-err`) |
| `sdpa_fold_div_nomask` | **promoted** | `sdpa_fold_div_nomask` | same, with `div`-by-`S` scale; `dspec` mints `scale = 1/S`; guard is `_COND_SDPA_SCALED ∧ rank(Q,V) ≥ 2` |
| `sdpa_fold_div_nomask_g` | **subsumed** | (folded into `sdpa_fold_div_nomask`) | the `_g` twin's metavar-axes pattern IS the promoted rule; its minted guard is the same last-two-axes fact the family carries — no separate spelling needed |
| `mul_unsqueeze_l_id` | **promoted** | `mul_unsq_pad_l` | first inhabitant; `bcast-eq ∧ bcast-into` region (≡ `ones-before ∧ bcast-eq`); sweep 23/23 equal, corpus pays on the ALiBi pad spelling |
| `mul_unsqueeze_r_id` | **promoted** | `mul_unsq_pad_r` | right-operand twin; sweep 23/23 equal |
| `sub_unsqueeze_l_id` | **promoted** | `sub_unsq_pad_l` | same pad region under `sub` — the asymmetric op proves the guard, not commutativity, carries it |
| `mul_reshape_l_id` | **promoted** | `mul_reshape_inert_l` | single-clause `bcast-eq(reshape-out(u,S), v, u, v)`; equal grids force a 1-axis insert/remove — a permuting reshape changes an extent and the grids disagree; sweep 60/60 equal |
| `mul_transpose_l_id` | **subsumed** | `transpose_noop` | the honest law is the *view-identity* itself: `transpose(u, d0, d1) → u` under `axes-noop`; once the transpose merges into `u`'s e-class, congruence does the mul/add strip in every context |
| `add_transpose_l_id` | **subsumed** | `transpose_noop` | same — the wrapped spellings stay store objects |
| `mul_chunk_l_id` | **subsumed** | `chunk_single` | `chunk(u, chunks=1, d, i) → u` under `attr-eq(C, 1)` — the single-chunk partition IS the tensor; the mul-wrapped strip follows by congruence |
| `linear_channel_scale_rev ∘ linear_row_scale` | **promoted** | `linear_channel_to_row_scale` | derivable composite, `derivation=("linear_channel_scale_rev", "linear_row_scale")`, kind `lemma`; both premises are SYMMETRY-gated (opt-in), so under `DEFAULT` this term-local edge is genuinely new reach; transported guard is the premises' conjunction over the shared `c`; paid 1 on corpus |
| `om_lift` | **duplicate — rejected** | — | `catopt_carriers.om.OM_LIFT`/`om_lift_plain` already ship `matmul(softmax(s,d), v) → om_apply(om_elem(s,v))` with the metavar-dim guard; the constructed literal `dim=-1` pattern is a strict subset |
| `affd_step_lift` | **duplicate — rejected** | — | alpha-identical to the shipped `affd_lift` (`SCAN_DIAG_LAWS`): same `add(mul(a,h),x) → applyd(aff_diag(a,x),h)`, same `_COND_AFFD_STATE` |
| `affd_scan2_lift` | **rejected — derivable unroll** | — | two `affd_lift` firings already produce exactly this member under `SCAN_DIAG_LAWS` saturation; a fixed-arity-2 shortcut is corpus-depth-shaped, not a law |
| `affd_scan4_lift` | **rejected — derivable unroll** | — | same at arity 4 — the corpus's 4-deep `DiagonalSSM` spines are why it paid; arity-specific composites are not library axioms |

## What the promotion changed

`ALL_RULES` 61 → **71** (`SIMPLIFICATION_RULES` 23 → 30,
`CATEGORICAL_RULES` 38 → 41, `SDPA_FOLD_RULES` 12 → 14).

- Declared split: 55 axioms / 14 lemmas / 2 redundant (the composite
  is the new lemma; the other nine promotions are axioms).
- Serialization census: **71/71 full-data** — every promoted law is
  pure pattern + `cond`/`dspec`/`derivation` data;
  `missing_hooks == ()` throughout.  The hook-free (no `check`, no
  `derive`) count rose 24 → 25 (`softsign_fold` is unguarded).
- `cond` users: 36 → **45**; `dspec` users: 16 → **18**.
- `FUSION`-tagged: 17 → **19** (the two `sdpa` nomask folds).
- Measured coherence (`--emit-basis`): 47 axioms / 13 lemmas /
  2 redundant / **9 no-instance** — the promoted conditional laws the
  catalogue's instancer cannot build a concrete site for (they are
  shape-guarded, not corpus-spelled); `linear_channel_to_row_scale`
  among them — its derivation is declared and store-certified, the
  catalogue just found no instance to replay it on.

## Guards — what the promoted laws carry

- `transpose_noop`: `("axes-noop", "u", "D0", "D1")` — swap axes
  coincide mod rank, or both swapped extents are 1.
- `chunk_single`: `("attr-eq", "CK", 1)` — one section is the tensor.
- `mul_unsq_pad_{l,r}` / `sub_unsq_pad_l`: the auto-derived pair
  `bcast-eq(unsq-out(t,UD), other, t, other) ∧ bcast-into(t,
  unsq-out(t,UD))` — grid coincidence plus "the operand fits its own
  padded shape" (which collapses to `ones-before` on shapes).
- `mul_reshape_inert_l`: `bcast-eq(reshape-out(u,RS), v, u, v)` —
  broadcast-grid equality alone, exactly as minted.
- `sdpa_fold_nomask` / `sdpa_fold_div_nomask`: the family guard
  (`_COND_SDPA_BASE` / `_COND_SDPA_SCALED`) plus `rank(Q) ≥ 2 ∧
  rank(V) ≥ 2` — `sdpa` needs rank ≥ 2 operands and the mask-free LHS
  still denotes on a rank-1 `Q`/`V`; `K`'s floor is implied by
  `axes-last2`.  (The masked family gets away without the clause —
  the mask operand's read makes a rank-1 `Q` lhs fail anyway; a
  latent guard-residual worth noting.)
- `linear_channel_to_row_scale`: the transported conjunction
  `("and", _COND_CHANNEL_SCALE, _COND_ROW_SCALE_c)` — the scalar `c`
  corner satisfies both sides.

## Sweeps re-measured at promotion

Each promoted rule was re-run through `_guarded_evals` over
`_synth_sites` (360-site window) plus a real-site firing check
(`EGraph.run` → merged e-class on the corpus spelling, decline on an
off-region binding):

| law | accepted | equal | unequal | rhs_err | fires |
|---|---|---|---|---|---|
| `softsign_fold` | 17 | 17 | 0 | 0 | `div(x,abs(x)+1)` → `softsign` |
| `transpose_noop` | 7 | 7 | 0 | 0 | `(2,1,1).transpose(-2,-1)` |
| `chunk_single` | 12 | 12 | 0 | 0 | `chunk(u,1,-1,0)` |
| `mul_unsq_pad_l` | 23 | 23 | 0 | 0 | `mul(unsq(u,0),(1,1,1))` |
| `mul_unsq_pad_r` | 23 | 23 | 0 | 0 | `mul((1,8,8),unsq(v,0))` |
| `sub_unsq_pad_l` | 23 | 23 | 0 | 0 | `sub(unsq(u,0),(1,1,1))` |
| `mul_reshape_inert_l` | 60 | 60 | 0 | 0 | `(2,6)→(1,2,6)` pad |
| `sdpa_fold_nomask` | 7 | 6 | 0 | 0 | spelled attention → `sdpa` |
| `sdpa_fold_div_nomask` | 3 | 3 | 0 | 0 | scaled spelling → `sdpa(…,scale=1/s)` |
| `linear_channel_to_row_scale` | 19 | 19 | 0 | 0 | `linear(x, W·c)` → `c·linear(x,W)` |

## Test fallout — the honest post-promotion state

Promoting a pattern makes the *stored object* a library duplicate
(`relation` keys on the `(lhs, rhs)` alpha pair — `cond` is not part
of the key).  The store's novelty gate now reports:

- `mul_unsqueeze_l_id`, `sub_unsqueeze_l_id`, `mul_reshape_l_id` →
  `duplicate` (alpha-equal to `mul_unsq_pad_l`, `sub_unsq_pad_l`,
  `mul_reshape_inert_l`);
- `softsign_fold` → `duplicate`;
- `channel_then_row_scale` → `duplicate` (same pattern as
  `linear_channel_to_row_scale`);
- `mul_unsqueeze_r_id` stays `new` — the record's rhs is `mul(v, u)`,
  the shipped law's is `mul(u, v)`;
- `mul_transpose_l_id`, `add_transpose_l_id`, `mul_chunk_l_id` stay
  `new` — only the *generic* view identities shipped, not the wrapped
  strips;
- the `sdpa` objects stay `new` — the promoted rules carry metavar
  attrs (`dim="SD"`, `scale="SC"`) where the records pin literals;
- `affd_*` / `om_lift` stay `new` — `SCAN_DIAG_LAWS` and `OM_LAWS` are
  not in `ALL_RULES` (the novelty library), so the duplicate verdict
  rests on this audit, not the alpha-key check.

Gauntlet tests that pinned a now-shipped object's `usable` verdict
were repointed at still-unshipped admitted siblings
(`mul_transpose_l_id`, `add_unsqueeze_l_id`) and one new test pins the
duplicate verdict itself
(`test_promoted_object_now_reports_library_duplicate`).

## Limitations / follow-ups

- The `add`/`div`/`getitem`-wrapped unsqueeze strips and the
  dropout-wrapped `sdpa_fold_nomask` form are natural siblings that
  were **not** measured — deliberately left unshipped rather than
  extrapolated.
- `mul_unsq_pad_l`'s shipped guard is the auto-cond pair
  (`bcast-eq ∧ bcast-into`), semantically the `ones-before ∧
  bcast-eq` region the first inhabitant carried.  The minted clause
  is the measured one; `ones-before` would read clearer but the
  promotion rule is "the auto-derived guard, verbatim".
- The shipped `sdpa_fold_add*`/`masked_fill*` family carries a latent
  `rank(Q) ≥ 2` blind spot the nomask promotions needed explicitly —
  masked sites mask it because the lhs fails to denote there.  A
  guard-residuals tightening candidate, not promoted-rule scope.
- Pre-existing (uncommitted, pre-dating this work) `oracle.py`
  enumeration changes shift three discovery-test pins
  (`test_guarded_truth_sweep_counts` 23→26 equal,
  `test_sdpa_fold_addmul_region_appears_at_the_guarded_cap` one
  rhs-err at the grown window, `test_auto_cond_g_twin_the_attr_case`
  mints `axis(V,TD2,-1)` for the `_g` twin) and two
  `test_discovery_oracle` index pins (`_first_equal_index` 327→337,
  `_reshape_targets` dedup) — unrelated to the promoted laws; the
  enumeration WIP owns those.
