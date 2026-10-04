# View/index oracle — every "unproven" resolved, zero ships

`intake-round-2.md` left 66 candidates, 33 firing, **0 shippable**,
and named the next gate: *view/index identities whose verification
needs a different oracle* — gather/index semantics and honest shape
instantiation, not the elementwise numeric probe that reads a single
real match.

This retro records the oracle (`tools/law_view_oracle.py`), its
wiring into `tools/law_pipeline.py`, and the run over the current
166-workload corpus.  Headline:

* **Every view-family candidate now resolves** — the 35 candidates
  whose patterns carry a view/index op split **1 true / 29
  conditional / 5 false / 0 unproven**.  The "unproven" bucket is
  empty.
* **The one strictly-true candidate does not fire**
  (`reshape_reshape`, a layout schema: equal on all 193 synthesized
  evaluable instances, 0 real matches).  **No true+paying candidate
  exists**, so nothing is reported for admission.
* **The oracle found a latent unsoundness in a shipped law**:
  `census:mul_select` — the proposal mirroring the shipped
  `select_mul` rule — is **conditional**, not unconditionally true:
  20 of its synthesized evaluable instances are counterexamples, and
  a direct check confirms `mul(sel(u,0,0), sel(v,0,0)) ≠
  sel(mul(u,v),0,0)` for `u=(4,)`, `v=(2,4)` (broadcast expands
  `u` along the *selected* axis; the shipped rule's structural
  same-dim/same-index guard does not prevent it).  Every real
  corpus match satisfies the missing condition — which is why it
  has never bitten — but the rewrite is wrong in general.
  The missing guard is declarable: `("rank-eq","u","v")` ∧
  `("dim-compat","u","D","v","D")` (or the stronger `shape-eq`,
  which covers all observed usage).  **Reported for review; not
  patched.**  The same verdict lands on the `select_add` /
  `select_sub` / `slice_mul` / `census:sub_unsqueeze` proposals of
  the same family.
* **The interesting output is the conditional+paying set**: nine
  candidates that are true *only* under a mechanically named shape
  guard, yet produce cost drops at real firing sites.  They are
  listed for review in §3 — none was admitted; they are evidence for
  `check`-guarded laws, not laws yet.
* **The false-but-paying phenomenon is now evidence-backed**:
  `mixed:add_getitem_l_id` — the pipeline's #1 ranked candidate — has
  4 fires, 4 paid, 40 % extracted-cost drop, and **zero agreeing
  instances in 82 evaluable**.  The old oracle left it "unproven";
  it is plain false.

Reproduce:

    .venv/bin/python tools/law_view_oracle.py --json /tmp/vo.json
    .venv/bin/python tools/law_pipeline.py --vocab derived \
        --json /tmp/pipeline-view-oracle.json
    .venv/bin/python tools/law_pipeline.py --vocab derived \
        --no-view-oracle        # pre-oracle behaviour

CPU-only; the pipeline run is ~9 min (the oracle adds ~3 min of
synthesis over the reach saturations).

## 1. Why the generic oracle could not answer

`law_proposal._numeric_true` instantiates **one** concrete match —
the first whose `check`/`derive` hooks pass — and evaluates both
sides on random fp64 tensors.  Three failure modes, all visible in
round-2's "unproven" rows:

1. **Wrong instance.**  For `mul(select(u),v) → mul(u,v)` the first
   match binds `v` to a select-of-something whose shape makes the
   instantiated `mul(u,v)` ill-typed.  The eval raises, the oracle
   abstains → `num_true=None`.  A later match that *would* prove the
   candidate is never tried.
2. **Wrong leaf semantics.**  `getitem` in this IR picks either a
   tuple element (every real match in the corpus: `topk`,
   `var_mean`, `cummax`, …) or a dim-0 tensor index.  A metavar under
   `getitem` is not honestly a fresh `Var`; instantiating it as one
   tests a different law and misses that `add(getitem(topk),v) →
   add(topk,v)` is *ill-typed*, not merely false.
3. **No satisfiability answer.**  Whether a both-sides-well-typed
   instance exists at all — and whether the equality holds on the
   *whole* satisfiable region — is the question a conditional-law
   classifier must answer, and a single-sample probe cannot.

## 2. The oracle

`tools/law_view_oracle.py` verifies a pattern pair in two passes and
returns a five-way verdict.

### Real-match sweep

`sweep_real` applies the proposal's `check`/`derive` hooks to
**every** real match (not the first viable), instantiates the RHS
exactly as a firing would, and evaluates each side tri-state:
`equal` / `unequal` / `lhs-err` / `rhs-err` / `env-err`.  An
ill-typed RHS is recorded as its own outcome — it is the signal that
distinguishes "the rewrite mints garbage here" from "the equality
fails here".  The bound operand's *kind* (tensor vs tuple value) is
recorded per match.

### Satisfiable instantiation

`synthesize` enumerates a bounded domain:

* **viewed metavars** (under a view/index op): a shape bank
  `{(4,), (2,3), (3,4), (2,2), (2,3,4), (2,3,1)}` plus, under
  `getitem`, **tuple-producing terms** (`topk`, `var_mean`, `cummax`
  over a shared `(2,4)` leaf) — both `getitem` meanings get tested;
* **attr metavars**: per view node, options valid for the bound
  operand's shape — `select`/`slice`/`chunk`/`narrow`/`unbind` dims
  and indices within extent, `unsqueeze` dims in `[-r-1, r]`,
  `transpose`/`movedim`/`permute` axis pairs, `reshape`/`expand`
  same-numel/broadcastable targets, `split` size tuples.  An op the
  oracle cannot instantiate honestly returns no options and the
  combination is skipped, never guessed;
* **free metavars**: a `Const(0.5)` scalar, a scalar `Var`, and a
  *derived* shape bank recomputed per attr choice — the viewed
  operand's shape, the view node's output shape, every single-axis-1
  insertion of each (the broadcast-pad region the naturality
  candidates hinge on), and a mismatched `(7,7)` control.

Each instantiation is deduped on `(repr, bindings)` — `op_repr`
renders bound `Var`s by name only, so the sig must carry the
bindings or every shape assignment collapses to one instance — and
both sides are evaluated fp64 (`_MAX_INSTANCES = 360`).

### Mechanical guard features

Per instance the oracle computes features that *name* the region
where the equality can hold:

* `w:v_commutes_view` — the naturality precondition: push the free
  operand `v`, broadcast to the *unviewed* grid `bcast(u,v)`, through
  the view, and check it equals `v` broadcast to the *viewed* grid
  `bcast(g(u),v)`.  Computed on tensors with the op's real binding.
* `id:same_pairing` / `id:out_shape_eq` — the strip (`_id`)
  preconditions: `g(u)` and `u` broadcast to the same elements on the
  common grid, and `bcast(g(u),v)` has `bcast(u,v)`'s shape.
* `tr:noop`, `rs:noop`, `sl:full`, `ck:single` — degenerate-view
  markers (a transpose that swaps equal axes, an identity reshape, a
  full-range slice, `chunks=1`).
* `unsq:d_in_pad`, `u_tuple`, `v_scalar`, `v_uniform`,
  `u_kind`/`u_shape`/`v_shape` — bookkeeping.

`_separating_feature` then finds a feature (or a pair, as `a ∧ b`)
whose true-set on the both-sides-typed instances is exactly the
equal-set — an absent feature counts as false, since a missing
pairing feature means there was no common broadcast grid.

### Verdict

| verdict        | meaning |
|----------------|---------|
| `true`         | every both-typed instance equal *and* no ill-typed-RHS instance |
| `conditional`  | equal and unequal instances both exist, or all evaluable agree but some LHS-ok instance mints an ill-typed RHS — `view_guard` names the separator |
| `false`        | both-typed instances exist, none agree |
| `ill-formed`   | the LHS evaluates somewhere the RHS never can (rewrite target does not denote) |
| `unproven`     | nothing evaluated at all |

## 3. Pipeline wiring and the run

`law_pipeline.measure` runs the oracle on every non-derivable
candidate whose patterns contain a view/index op (`_has_view_op`),
default-on, disabled by `--no-view-oracle`.  `Evidence` gains
`view_verdict` / `view_guard` / `view_note`; the verdict maps
`true→num_true=True`, `false|ill-formed→False`,
`conditional→None` — so a conditional candidate stays out of the
ship set until its guard exists as a `check`.  `no_ship_reason`
reports `conditional truth (view-oracle): <guard>`.

Run (66 proposals, 35 view candidates):

| verdict | candidates |
|---------|------------|
| **true** | `reshape_reshape` — 193/193 equal synthesized instances; 0 real matches |
| **conditional** | all 29 remaining view candidates |
| **false** | `add_getitem_l_id`, `sub_getitem_r_id`, `mul_select_l_id`, `add_select_r_id`, `sub_getitem_r_w` |
| **unproven** | — none — |

### Strictly-true candidates that fire: none

`reshape_reshape` never fired.  No unconditional true+paying
candidate exists → nothing to admit.

### The conditional+paying set (for review — not admitted)

Nine candidates pay at real sites but are true only under their
guard; the old pipeline called them "unproven" or "false".  The
middle column is the real-match sweep — how the **actual** firing
sites evaluated:

| candidate | guard (mechanical) | real-match sweep | paid | drop |
|-----------|--------------------|------------------|------|------|
| `mixed:mul_slice_l_id` | `id:same_pairing` (full-range slice) | 0 eq / 5 ne / 12 rerr | 1 | 25.0 % |
| `mixed:mul_unsqueeze_l_id` | `id:out_shape_eq ∧ id:same_pairing` | 1 eq / 4 ne / 3 rerr | 2 | 14.3 % |
| `mixed:mul_unsqueeze_r_id` | `id:out_shape_eq ∧ id:same_pairing` | 0 eq / 4 ne / 0 rerr | 1 | 14.3 % |
| `mixed:mul_unsqueeze_reshape_id` | not isolated | 0 eq / 0 ne / 1 rerr | 1 | 13.0 % |
| `mixed:mul_unsqueeze_reshape_wr` | not isolated | 0 eq / 0 ne / 1 rerr | 1 | 8.7 % |
| `mixed:sub_unsqueeze_l_id` | not isolated | 1 eq / 0 ne / 1 rerr | 1 | 8.7 % |
| `mixed:mul_unsqueeze_reshape_wl` | not isolated | 0 eq / 0 ne / 1 rerr | 1 | 4.3 % |
| `mixed:mul_transpose_l_id` | `id:same_pairing` (≡ `d0==d1`) | 0 eq / 2 ne / 0 rerr | 1 | 2.8 % |
| `mixed:eq_getitem_l_id` | `id:out_shape_eq` | 0 eq / 4 ne / 0 rerr | 1 | 2.1 % |

For comparison, the conditional **naturality** candidates (paid = 0)
are where the real sites actually live inside the guard:

| candidate | real-match sweep | verdict |
|-----------|------------------|---------|
| `mixed:add_select_r_w` | 4 eq / 0 ne / 0 rerr | conditional — true at all real sites; the old single-instance oracle's `true` |
| `mixed:mul_select_l_w` | 24 eq / 0 ne / 1 rerr | conditional — guard `w:v_commutes_view` |
| `mixed:mul_slice_l_w` | 5 eq / 0 ne / 12 rerr | conditional — naturality holds at 5 sites, RHS ill-typed at 12 |
| `mixed:mul_transpose_l_w` | 2 eq / 0 ne / 0 rerr | conditional — `w:v_commutes_view` |
| `mixed:mul_unsqueeze_l_w` | 0 eq / 5 ne / 3 rerr | conditional — real sites pair `unsqueeze` with broadcast-incompatible operands |
| `mixed:add_getitem_l_w` | 0 eq / 0 ne / 4 rerr | conditional — all real sites are tuple-`U`, RHS never denotes |
| `mixed:eq_getitem_l_w` | 0 eq / 0 ne / 4 rerr | conditional — same tuple-`U` signature |

The three `mul_unsqueeze_reshape_*` candidates pay at their single
(ALiBiAttention) site **while minting an ill-typed RHS there** —
the e-graph admits the member, the cost model prices it, and only
the downstream gates (extraction verify, certificate) stop it.
Reach is not an oracle.

And `reshape_transpose` (the layout schema, 65 matches, 58 fires):
**all 60 distinct real instances evaluate unequal** — the naive
reshape/transpose interchange is wrong at every real site (the split
must respect the transposed axes); 39 equal instances exist only in
the degenerate synthesized region.  It paid twice at ~0 % anyway.

Reading the guards as law conditions: `id:*` strip candidates are
true exactly when the view is a semantic no-op on the broadcast
grid — a `check` of the form "the view output equals its operand
under broadcast, and the two multiply-output shapes coincide" (the
`shape-eq`/`dim-compat`/`axis` predicates of the condition DSL cover
the `out_shape_eq` half; `same_pairing` needs a broadcast-equality
predicate the DSL does not yet have).  `w:v_commutes_view` is the
naturality guard — `v` must broadcast to the viewed grid the way it
broadcasts through the view — expressible as `dim`-wise
`bcast`-compatibility once `cond` can talk about broadcast grids
along a chosen axis.

Honest counterpoint: a `conditional` verdict with a paying site does
**not** mean the site satisfies the guard — `eq_getitem_l_id` fires
on `SharedExpertMoE` and pays there while the guard fails in
general.  The reach measurement prices the *rule*, not the guard;
admission still needs `check` to exclude the false region.  None of
the nine is admitted.

### The false-but-paying evidence ledger

| candidate | fires / paid | drop | oracle evidence |
|-----------|--------------|------|-----------------|
| `mixed:add_getitem_l_id` | 4 / 4 | 40.0 % | 0 agreeing of 82 evaluable; all real matches are tuple-`U` (`topk`/`var_mean`) where the RHS is ill-typed |
| `mixed:sub_getitem_r_id` | 4 / 3 | 40.0 % | 0 of 82 |
| `mixed:mul_select_l_id` | 40 / 3 | 40.0 % | 0 of 205 — was "unproven"; genuinely false (a select always removes the multiplied axis) |
| `mixed:add_select_r_id` | 4 / 1 | 20.0 % | 0 of 185 |
| `mixed:sub_getitem_r_w` | 4 / 0 | — | 0 of 77 |

The 40 %-drop rows are the cost model telling the truth about cost
and the oracle telling the truth about semantics — the two signals
disagree, and the oracle wins.

## 4. Limits (honest)

* **The instance domain is finite.**  Shape bank, tuple sources and
  attr options are enumerated, not solved — `true` means "equal on
  every evaluable instance tested", and `conditional`'s equal-set
  may be a union the separator under-covers (`no single mechanical
  feature separates` on 15 of 29).  A *false* verdict is strong (it
  exhibits a real, well-typed counterexample); a *true* verdict is
  inductive evidence, not a proof — the same epistemic level the
  numeric oracle always claimed.
* **Metavars bind to terms, the oracle binds them to leaves.**
  Real `U`/`V` bindings are arbitrary subterms (`var_mean`,
  `select(a)`, `exp(x)`); the synthesis covers leaves plus three
  tuple producers.  Guards stated over `U`'s *value* properties
  (rank, dims) transfer to compound bindings; guards keyed on `U`
  being a tensor leaf do not — the getitem naturality rows show this
  (`w:v_commutes_view` for tensor-`U`, ill-typed RHS for tuple-`U`,
  a `("leaf","U")`-style condition in DSL terms).
* **`_attr_options` covers 18 ops.**  A view op outside the table
  skips the combination — honest abstention, never a guess.
* **Real `u_kind` classification assumes the bound term evaluates.**
  A match whose bound `U` itself fails to evaluate reports
  `u_kind=""` — counted, not guessed.
* **Coverage vs cost.**  The enumeration caps at 360 instances per
  candidate and dedups; the 14–20 view candidates take ~3 min.
* The oracle *describes* guards; it cannot admit them.  `cond.py`
  has no broadcast-grid predicate yet, so even the cleanly-named
  guards (`w:v_commutes_view`, `id:same_pairing`) need a DSL
  extension before a `check` can express them.

## 5. What changed

* `tools/law_view_oracle.py` — new (oracle, CLI table/JSON driver).
* `tools/law_pipeline.py` — `Evidence.{view_verdict,view_guard,
  view_note}`; oracle call in `measure`; `no_ship_reason` for
  `conditional`/`ill-formed`; report tally + guard list;
  `--no-view-oracle`; JSON dump fields; `view_oracle` param on
  `run_pipeline`.
* `tests/test_law_view_oracle.py` — 13 tests: tri-state eval,
  verdicts on known candidates, mechanical-feature agreement with
  the equal set, real-match sweep, abstention.
* `project/retros/view-index-oracle.md` — this file.

No law files touched; nothing admitted.
