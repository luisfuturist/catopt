# Retro: cond-DSL guards for the conditional view/index candidates

Date: 2026-02-10
Context: follow-up to `view-index-oracle.md`.  The view/oracle pass
found 35 candidates — 1 unconditionally true (`reshape_reshape`, no
real firing site), 5 genuinely false, and 29 conditional.  This task
decodes the conditional guards the oracle named into the declarative
condition DSL (`catopt_core.laws.cond`), verifies them against the
synthesized and real corpus bindings, and reports — it does **not**
admit laws.  `tensor.py`, `scan.py`, `torch_bridge.py`,
`law_pipeline.py` and `serialize.py` were not touched.

## What the oracle meant by its guard names

The oracle's mechanical guards were shape-level checks it evaluated
per binding; this retro's job is to restate each one as pure-data
`cond` tuples over the same `bound` environment, with the same
strictness contract (if the guard cannot prove its claim from the
binding, it declines).

The oracle's features decompose as:

- `id:out_shape_eq` — the post-view broadcast grid equals the
  stripped broadcast grid: `bcast(view_out, v) == bcast(u, v)`.
- `id:same_pairing` — for `unsqueeze`, the inserted axis is a
  broadcast-pad position: every operand dim before it (after `mod
  rank+1` normalization) is `1`, so the broadcast reads the operand
  through the same index map as the uninserted form.
- For `slice` and `transpose` the strip is literal: a full-extent
  slice is a no-op term and a no-op transpose is a no-op term, so the
  guard is about the *view* alone — output shape cannot change.
- For the `reshape`-crossing naturals (`id`/`wr`/`wl`) the oracle
  reported "no single mechanical feature separates" — the honest
  guards are conjunctions the oracle did not name.

## DSL additions

New shape specs resolvable anywhere a spec is accepted
(`packages/catopt-core/src/catopt_core/laws/cond.py`):

- `("bcast", T1, T2)` — `_broadcast` over two nested specs; an
  unresolvable side is a wildcard (`None`), matching `_broadcast`.
- `("unsq-out", T, K)` — the `unsqueeze` output shape: a `1` at
  `$attr:K mod (rank+1)`.
- `("reshape-out", T, NAME)` — the resolved reshape target: `-1`
  folds through numel, numel mismatch resolves to `None` (decline —
  the strictness contract does not need `_INVALID`'s poison marker).
- `("getitem-out", T)` — drop dim 0: the tensor-index output shape.
  `()` for a rank-0/rank-1 source or unresolvable input.

Dispatch is by `ref[0]` through `_SPEC_OPS`; unknown tags resolve to
`None` (decline), preserving the old `mm-out` behavior.

New predicates (all `_OPS`-registered, all serializing by name):

- `("attr-cmp-dim", NAME, CMP, T, K)` — `$attr:NAME CMP
  shape(T)[$attr:K]`; the slice-covering check `end >= shape(u)[dim]`.
- `("bcast-eq", A, B, C, D)` — `bcast(A,B) == bcast(C,D)`; every spec
  must resolve to a tuple and both broadcasts must succeed —
  `id:out_shape_eq` verbatim.
- `("ones-before", T, K)` — `shape(T)[:K mod (rank+1)]` all `1`;
  `id:same_pairing` for unsqueeze verbatim.
- `("axes-noop", T, D0, D1)` — transpose is a semantic no-op on `u`:
  normalized `d0 == d1`, *or* both swapped extents are `1` (swapping
  two size-1 axes changes neither the shape tuple nor the broadcast
  pairing — several oracle-equal instances were exactly this, and the
  rank-1 operand degenerates to the same axis entirely).
- `("flat-pair-unsq", T, K, G, S)` — the *wr* naturality's `u`-side:
  for every non-`1` `u`-dim, the flat-read digit it contributes sits
  at the same stride in `G = bcast(u,v)` and in `S` (the reshape
  target).  All specs concrete-int; any mismatch declines.
- `("flat-map-unsq", T, K, V, S)` — the *wl* naturality: the grids
  coincide (`bcast(unsq(u), S) == insert1(bcast(u,v), K mod rank+1)`)
  AND each operand's flat-read coefficient map agrees across the two
  orders (u via `unsq`-broadcast vs mul-broadcast-shifted; v via
  `reshape`-broadcast vs mul-broadcast-shifted).

Helpers `_viewed_map`/`_shifted_map`/`_flat_map_ok`/`_stride`/
`_concrete_specs`/`_unsq_shape`/`_reshape_shape`/`_fold_minus1`
implement the flat-read maps and shape arithmetic; all stay inside the
complexity ratchet.

## Guard table

`U`, `V` are the term metavars as in the oracle's patterns
(`mul_unsqueeze_r_id` binds the free operand as `V` too — the pattern
spells `mul(V, unsq(U,d))`, so the same strip guard applies verbatim).

| candidate | cond | oracle separation | real corpus |
|---|---|---|---|
| `mul_slice_l_id` | `("and", ("or", ("attr-is","A_start",None), ("attr-eq","A_start",0)), ("or", ("attr-is","A_end",None), ("attr-cmp-dim","A_end",">=","U","A_dim")))` | exact: accepts all 57 equal, declines 127 unequal + errs (42 `both-err` accepted — the strip is literal so ill-typed mul bindings are admitted; see limits) | 0/21 accepted — all real slices are partial |
| `mul_unsqueeze_l_id` | `("and", ("ones-before","U","A_dim"), ("bcast-eq",("unsq-out","U","A_dim"),"V","U","V"))` | exact: 25/25 equal accepted, 0 others | 1/8 accepted (`U=(8,8)`, `V=(1,1,1)`, `dim=0`) — verified equal |
| `mul_unsqueeze_r_id` | same strip guard (pattern binds the unsq operand as `U`) | exact: 25/25 | 0/4 |
| `sub_unsqueeze_l_id` | same strip guard | exact: 26/26 | 1/2 accepted (`U=(8,)`, `V=(8,1)`, `dim=0`) — verified equal |
| `mul_transpose_l_id` | `("axes-noop","U","A_dim0","A_dim1")` | exact: 47/47 equal (d0==d1 plus both-1 swaps and rank-1) | 0/3 — real transposes are real swaps |
| `eq_getitem_l_id` | `("and", ("leaf","U"), ("dim-eq-const","U",0,1), ("attr-in","A_index",(0,-1)), ("bcast-eq",("getitem-out","U"),"V","U","V"))` | honest region accepted (all 4 hand-built true envs evaluate `equal`); the oracle's 36 `equal` are all declined — see the eq caveat | 0/4 — all real `U` bindings are tuple producers |
| `mul_unsqueeze_reshape_id` | `("and", ("shape-eq","V",("reshape-out","V","B_shape")), ("ones-before","U","A_dim"), ("bcast-eq",("unsq-out","U","A_dim"),("reshape-out","V","B_shape"),"U","V"))` | exact: 6/6 | 0/1 (the ALiBi site has `S=(-1,1,1)` ≠ `V` shape) |
| `mul_unsqueeze_reshape_wr` | `("and", ("bcast-into","U","V"), ("bcast-into",("unsq-out","U","A_dim"),("reshape-out","V","B_shape")), ("flat-pair-unsq","U","A_dim","V",("reshape-out","V","B_shape")))` | exact: 13/13 | 0/1 |
| `mul_unsqueeze_reshape_wl` | `("flat-map-unsq","U","A_dim","V",("reshape-out","V","B_shape"))` | exact: 19/19 | 1/1 accepted — see the pow caveat |

`mul_reshape_l_id` was not in scope of the requested set, but note it
carries the same `id:out_shape_eq` flavor (the `eq_getitem` row's
bcast-eq clause is the same mechanism).

## The `eq_getitem_l_id` caveat

The oracle recorded 36 equal / 84 unequal / 16 rhs-err.  Every
`equal` instance is a coincidence of the metric, not the semantic
region: `eq` is a Boolean pointwise op and the generated data is
`torch.randn` — with probability ≈1 **both** sides are the all-False
tensor, so "equal" means "both happened to be all-False".  A crafted
binding (`u=(2,3)`, `i=1`, `v=(2,3)` with `v[0]=u[0]`, `v[1]≠u[1]`)
produces `[[T,T,T],[F,F,F]]` vs `[[F,F,F],[F,F,F]]` — genuinely
unequal.

The semantic precondition for the strip is: `getitem` is a broadcast
no-op on `u` — `u` a tensor (leaf, not a tuple producer), `u[0]==1`,
and the index selects the single leading element (`0` or `-1` for
leading-index forms).  The guard accepts exactly that region and
declines all 36 oracle-"equal" bindings, which is the correct
behavior under "equal for all values of the bound vars" rather than
"equal at one random draw".  The finite oracle cannot see this
distinction; the retro's earlier claim that "id:out_shape_eq
separates" is true *of its metric* — the equal-set there is the
coincidence region.

Undercoverage warning: the shape bank contains no leading-extent-1
tensor bound to `U` in a `getitem` site, so the true region had to be
verified by hand-built envs (4/4 accepted and `equal`).  The
`index ∈ {0,-1}` clause covers only leading-element indices; other
honest extensions (e.g. trailing singleton slices) are out of the
pattern's evidence.

## The `pow` caveat on the real `wl` site

The one real `mul_unsqueeze_reshape_wl` match (ALiBiAttention) binds
`V` to `pow(detach_(contiguous(c_lifted)), div(neg(arange4),4))`.
`_shape_of` reports `V`'s shape as `()` — `pow`'s shape rule returns
the base shape and ignores broadcast against the exponent — but the
evaluated value is `(4,)`, so the oracle correctly recorded
`rhs-err` (the instantiated RHS is ill-typed).  The guard accepts
the binding *on the shapes the IR believes*.  This is a pre-existing
`pow` shape-rule inaccuracy in `typing.py`, not a guard error — but
it is a real limitation: cond verdicts inherit shape-inference
inaccuracy.  Flagged for a follow-up fix; out of scope here.

## Verification

- `tests/test_cond_laws.py` — +10 test functions covering every new
  branch (spec arities, `-1` folding, numel mismatch, rank-1/no-op
  transpose, map/grid mismatches, missing-attr declines) plus the
  guard conds as data with canonical accept/decline envs and a
  `cond_to_data`/`cond_from_data` JSON round-trip.
- Synth sweep: every proposed guard evaluated over all synthesized
  bound envs (`vo.synthesize`, `limit=360`) — the table's "exact"
  rows mean `accepted == equal` and `declined == {unequal, *-err}`
  on the finite domain, with the slice/transpose rows accepting
  `both-err` bindings whose strip is literal (documented above).
- Real corpus: `real_matches` over benchmark+model+intake terms,
  each matched binding evaluated — acceptances listed in the table;
  every acceptance was verified to evaluate `equal` (except the
  pow caveat row).
- `pytest tests/test_cond_laws.py tests/test_derive_laws.py
  tests/test_law_serialize.py tests/test_law_view_oracle.py -q`:
  115 passed.  `ruff check`, `ruff format --check`, `ty check` clean
  on touched files.  `radon_ratchet` clean.  `cond.py` at 100%
  statement+branch coverage under the focused set plus
  `test_select_laws.py` (its `dim-eq-attr` consumers).

## Honest limitations

1. The guards are exact on the *synthesized finite domain* and
   consistent with derivation, but they are not mechanically proven.
   `flat-pair-unsq`/`flat-map-unsq` encode the flat-index-map
   equality — the derivation in the retro text is the argument; no
   symbolic engine checked it.
2. The slice/transpose guards accept `both-err` bindings (the view
   is a no-op but `mul(u,v)` is ill-typed).  Harmless — the matcher
   only fires on evaluable LHS instances — but the guard itself does
   not assert well-typedness; by design it states the *view's*
   precondition.
3. `getitem`'s guard presumes the tuple-index interpretation
   (`getitem-out` drops dim 0) while `U` must be a `leaf` for the
   `eq` RHS to denote — `leaf` is a coarse proxy for "tensor, not a
   tuple-valued expr".  A tuple-typed non-leaf `U` still declines;
   there is no "is-a-tensor" pred in the DSL, and the retro flags
   this as the coarsest point of the guard set.
4. The `wr`/`wl` guards assume `u`/`v` are the *operand* shapes; the
   flat-index reasoning is over row-major flat order — the same
   convention `_broadcast` and `reshape` share, stated explicitly in
   the predicate docstrings.
5. `sub_unsqueeze_l_id` carries the identical strip guard to `mul`'s
   — `sub` shares broadcasting semantics, but the retro notes the
   law family differs (RHS is `sub(u,v)`); the cond is identical
   because the *precondition* is about the view strip only.
6. No candidate is admitted.  The conds exist as verified data;
   `law_pipeline`'s admission machinery owns the next step, and
   `eq_getitem_l_id` in particular needs a tuple-aware type pred
   before it can be trusted in the pipeline — its real firing site
   (`cross_entropy` block-diagonal case) pays cost-wise but the
   semantic guard there is exactly the honest `u[0]==1` clause,
   which that site does not satisfy.
