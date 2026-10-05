# Retro: auto-cond — the conditional → guarded → admitted path

Date: 2026-10-05
Context: `guide-real-run.md` §7 — the arena shipped dozens of
candidates, `usable: yes` stayed 0 for every guide, and the refusal
census named the missing move: `truth` ×612 refusals are conditional
equalities whose guard exists in no declarative form.  This change
builds that move: a constructor that mints the smallest declarative
`cond` covering the measured domain, and an additive admission hook
that retries a truth-refused object with it.  `oracle.py`,
`meta_game.py`, `pipeline.py`, `laws/` untouched — the mechanism sits
in `object_synthesis.py` (`auto_cond_object` + helpers) and
`evidence.py` (`run_gauntlet(auto_cond=...)`, `_auto_cond_retry`,
`--auto-cond`).  Tests in `tests/test_discovery_autocond.py`.

## The mechanism

1. **Measure the bare domain** (`_measure_domain`).  The probe is the
   candidate's pattern with `cond`/`check` dropped — the found guard
   must be a complete claim, not a strengthening of a half-written
   one.  `derive`/`dspec` stays: it is part of how the rule
   instantiates, and a veto is a firing abort (`declined`), not a
   site.  Each env is scored by `_site_outcome` — the same
   check→derive→instantiate→fp64-eval order a firing consults.
   Sites partition into `equal` (must accept), `bad` =
   `unequal`+`rhs-err` (must decline), `other` (`lhs-err`/`both-err`/
   `env-err` — don't-care, counted), `declined` (outside the
   universe).

   The synth side unions **two windows** of the same enumeration.
   `evidence._synth_sites` dedups the instantiated `(lhs, rhs)` pair
   *before* `derive` runs — a vetoed env still burns a window slot;
   `oracle.synthesize` dedups on the post-derive instance signature.
   The windows expose different corners of the same domain, and the
   difference is not theoretical: the `_g` twin's counterexample
   corner — `transpose(K, dim0=0, dim1=0)` on a square `K`, spelling
   `Q@K` where `sdpa` computes `Q@Kᵀ` — lives inside the second
   window and outside the first.  `_oracle_sites` mirrors the
   `synthesize` dedup exactly; the union is the measured domain.
   Real-corpus matches (`shape_proposal.real_matches` + `_term_match`)
   measure identically — a real `unequal` is a site the guard must
   decline.

2. **Stability.**  `_env_for` fills leaves with ambient
   `torch.randn` draws, so a binding whose verdict is
   value-sensitive — measured: a real `sub(unsqueeze…) ·` term whose
   `pow(param, -0.25)` NaNs on a negative draw — can flip outcome
   between the search's measurement and the re-sweep.
   `_stable_outcome` scores every evaluative site twice and classes
   a disagreement as `unstable` — must-decline: the minted guard
   never admits a firing region whose verdict is a coin flip.
   (Detection is per-draw; a site that agrees twice counts as
   stable — the same standard the guarded sweep itself applies.)

3. **The predicate bank** (`_pred_bank`) is the cond DSL's own
   vocabulary instantiated over the pattern's metavariables and attr
   metavariables — leaf/shape/rank predicates, `rank-eq`/
   `shape-eq`/`bcast-into`/`mm-shape-ok`/`dim-eq`/`term-eq` pairs,
   `attr-type`/`attr-is`/`attr-len`/`axis`/`dim-mod`/`dim-eq-attr`/
   `attr-cmp-dim`, `op-in` over ops actually observed bound,
   `attr-eq` over values actually observed in the domain, and the
   per-view-op strip vocabulary (`_VIEW_NODE_PREDS`:
   `ones-before`/`flat-*`/`axes-*`/`bcast-eq` over computed shape
   specs — `unsq-out`/`reshape-out`/`getitem-out`).  Every predicate
   is evaluated over every site once (`_pred_masks`); a predicate
   that raises anywhere is dropped — a non-total guard is not
   admissible data.  Verdict-identical spellings dedupe by repr.

4. **Minimal conjunction** (`_min_cover`).  Iterative deepening over
   clause count k = 1, 2, 3 (`max_clauses`); DFS branches on the
   uncovered bad site with the fewest covering killers; tie-breaks on
   fewest accepted non-equal sites, then lexicographic repr —
   deterministic.  A refusal is data: `no declarable conjunction of
   <= k covers E equal / B bad sites (C covering predicates)`.  The
   "not tuned to one counterexample" property is structural — a
   clause only counts if it accepts *every* equal site, and the
   search is smallest-first.

5. **Minting** (`_mint_guarded`).  `ConstructedObject(rule=...,
   kind="abstraction", construction=("auto-cond", name))`: found
   `cond` on the original lhs/rhs, `dspec` kept, procedural
   `check`/`derive` kept as callables (the serializer flags them as
   `missing_hooks` — the record stays honest about what is not data).
   The minted object faces `run_gauntlet` unchanged — construction
   is a claim, the gauntlet is the referee.

6. **The admission hook** (`run_gauntlet(auto_cond=True)`).
   `_auto_cond_armed` fires the retry at exactly two refusals:
   `truth` (the candidate measured conditional) and `full-data` with
   `missing_hooks == ["check"]` (the reconstructed rule is already
   the bare pattern — the missing `check` is the slot a found `cond`
   fills).  On a found cover the object record is rewritten *in
   place* under the same alpha key — `store_object`'s upsert — now
   carrying `cond` and `kind="abstraction"` — and the gauntlet
   re-runs once.  The report carries `rep.auto_cond` (found, clauses,
   measured-domain counts, refusal detail, `first_reason`) and a
   prepended `auto-cond` stage; a refused search keeps the original
   verdict untouched.

## Measured — the real corpus

`default_gauntlet_corpus()` (276 terms / 158 probe cases), the static
proposal pool (`generator_pools` — 66 deduplicated candidates, the
enumeration arms' inventory), `synth_limit=360` per window.  37 of
the 66 candidates measured conditional (equal>0, bad>0) and were
stored, gauntleted bare, then re-gauntleted armed.  Result:

| outcome | count |
|---|---|
| **usable: yes** — admitted | **1** |
| guard minted, truth clean, refused at `typed-pay` (fires=0 — no real accepted site) | 19 |
| no declarable conjunction ≤3 | 15 |
| refused before truth (`novelty`/`full-data`) | 2 |

…plus the pool's 29 non-conditional candidates (all-equal or
no-domain), which the hook correctly never touches.

### The admitted set

`mixed:sub_unsqueeze_l_id` — `sub(unsqueeze(u, A_dim), v) →
sub(u, v)` with `bcast-eq(unsq-out(U,A_dim), V, U, V) ∧
bcast-into(U, unsq-out(U,A_dim))` — synth region 23 accepted / 23
equal / 0 unequal, real region 1 accepted / 1 equal.  The found
guard is semantically the hand-written `ones-before ∧ bcast-eq`
region (`bcast-into(U, unsq-out)` is the broadcastable-pad clause).

Its `mul` twin is the instructive near-miss: it minted the same
strip pair **plus** `leaf(U)` — the search detected the corpus's
one accepted site as `unstable` (the `pow(param, -0.25)` term
evaluates NaN on a negative draw) and declined it — so the guarded
object is inert on this corpus and `typed-pay` refuses, honestly.
The verdict that flips per draw is the corpus's, not the
constructor's.

### Guards minted but refused later (fires=0)

Nineteen candidates got a verified guarded region — every accepted
synth site equal — then failed `typed-pay` because the guard's region
fires nowhere on the real corpus.  Honest outcome worth listing: the
guard is real, the *region* is empty in practice.

| candidate | minted cond (clauses) |
|---|---|
| `mul_unsqueeze_l_id` | the `sub` twin's pair **∧ `leaf(U)`** — the extra clause is the unstable-site decline (above) |
| `mul_slice_l_id` | `attr-cmp-dim(A_end,==,U,A_dim) ∧ attr-eq(A_start,0)` — the full-extent slice |
| `mul_transpose_l_id` / `add_transpose_l_id` | `axes-noop(U,A_dim0,A_dim1)` — the swap is an identity |
| `mul_unsqueeze_l_w`/`_r_w`/`sub_unsqueeze_l_w` | `bcast-into(V,U) ∧ bcast-into(V,unsq-out)` — pad target |
| `mul_unsqueeze_r_id` | same strip pair as the l twin |
| `eq_getitem_l_id` | `bcast-eq(getitem-out,V,U,V) ∧ leaf(U)` |
| `mul_reshape_l_id` | `bcast-eq(reshape-out(U,S),V,U,V)` — reshape broadcast-inert |
| `mul_chunk_l_id` | `attr-eq(A_chunks,1)` — the 1-chunk no-op |
| `mul_unsqueeze_reshape_wl` | `flat-map-unsq(U,A_dim,V,reshape-out(V,B_shape))` — the cond-dsl retro's exact guard |
| `mul_unsqueeze_reshape_id` | `bcast-eq(reshape-out,V,…)` ×2 |
| scalar-corner grammar finds (`mul_zero`, `sub_self`, `add_inv`, `div_self`, `mul_zero_left`) | `rank(A)==0` — the only binding where `a*0=0`-style rewrites are *shape*-equal (rhs is a scalar literal; the sweep's shape-strict compare refuses tensor lhs) |
| `FALSE_mul_factor` | `term-eq(M0,M1)` — the guarded candidate is the true `x*x` law |

### Refused — nothing declarable ≤3 clauses

Fifteen candidates: `census:sub_unsqueeze`, `mixed:mul_slice_l_w`,
`mixed:mul_select_l_w`, `mixed:add_select_r_w`, `mixed:eq_getitem_l_w`,
`mixed:mul_reshape_l_w`, `mixed:mul_transpose_l_w`,
`mixed:add_transpose_l_w`, `mixed:mul_chunk_l_w`,
`mixed:add_getitem_l_w`, `mixed:mul_unsqueeze_reshape_wr`,
`select_add`, `select_sub`, `slice_mul`, `reshape_transpose`.  Two honest
sub-cases: regions needing a vocabulary the bank does not have
(the `*_w` "wrap" forms need predicates over `w`'s broadcast into the
stripped shape — several report 0 covering predicates, meaning no
bank clause even accepts the equal class), and `recognize:softmax`
stood down correctly — `full-data` with `missing_hooks = ["check",
"derive"]` is not an auto-cond question (a missing `derive` is a
derivation gap, not a guard gap).

## Limits — what this does not claim

- **Measured, not proven.**  The minted `cond` is exact on the
  measured domain — the two capped enumeration windows plus the real
  matches.  A counterexample outside both windows stands until a
  wider sweep reaches it; that is the same blind spot every guarded
  law carries, and the guarded-region sweep remains the truth gate's
  own lens.  `cond=True` (vacuous cover) is minted when the domain
  shows no bad site — the flag is in `detail`, not hidden; it did
  not occur in this pool (every minted object carried real clauses).
- **The bank is finite and hand-scoped.**  Regions separated by a
  predicate that exists in `cond` but not in `_pred_bank` (e.g. the
  `*_w` wrap forms) refuse as "no declarable conjunction".  Growing
  the vocabulary is additive; each atom is already interpreter-
  checked.
- **`other`-bucket sites are don't-care.**  The tie-break prefers
  covers accepting fewest of them, but an `lhs-err` site accepted by
  the guard is not a defect — the guarded sweep counts it
  separately and a firing aborts the same way.
- **Cap ≤ 3 clauses.**  `max_clauses` bounds the search; "no cover"
  means "no cover of ≤ k", not "no cond exists".
- **The pool is the static inventory.**  The measured set is
  `generator_pools`'s 66 candidates; `gap_gen`/`workload_gen`
  minted objects from the guided runs are not in it — their
  conditional finds (the 69/8 and 80/38 regions) are the same
  families, and the constructor does not care where a candidate was
  minted.
- **Procedural remainders stay flagged.**  A candidate carrying a
  non-declarable `check`/`derive` keeps it on the minted rule (a
  veto is a firing abort) and the record's `missing_hooks` reports
  it — a stored object whose guard is part-procedural does not
  claim full serializability.

## The load-bearing tests

`tests/test_discovery_autocond.py`: the known-conditional strip
mints a separating guard whose every clause is load-bearing (drop
one and a bad site re-enters) and whose `k-1` cap refuses; the `_g`
attr case mints the `axes-last2`-region guard; three refusal classes
(no equal site / no declarable cover / no evaluable domain);
`_min_cover` smallest-first unit tests; and the gauntlet wiring —
bare conditional stored, armed re-run rewrites the record under the
same alpha key and clears all eight gates, the refused search leaves
the record untouched, and the `missing_hooks == ["check"]` trigger
runs the same path from `full-data`.

One object admitted; thirty-six honest refusals with reasons.
