# Retro: chained view bindings — closing the `gqa_absorb_repeat`
# enumeration gap

Date: 2026-10.  Follow-up to `guard-residuals.md`'s documented
residual:

> `gqa_absorb_repeat`'s guard accepts a hand-built chained binding
> (`q=(2,6,4)`, `k=v=(2,3,4)`, `UD=2`, `ES=(2,3,2,4)`, `RS=(2,6,4)`)
> but the enumerator never mints it — `_attr_domains` reads each
> view node's operand shape from the *pre-view* term.

Owned seams: `catopt_discovery/oracle.py`,
`tests/test_discovery_oracle.py`, this file.  No law, `cond.py`,
`evidence.py`, `intake.py`, or metadata file touched — the fix is
purely in how the enumerator *enumerates*, not in what the laws
mean.

## 1. The break, precisely

The GQA LHS carries K and V through
`transpose(reshape(expand(unsqueeze(k, dim=UDk), shape=ESk),
shape=RSk), dim0=1, dim1=2)`.  `_attr_domains` computed each view
node's attr domain over `_operand_shape(instantiate(args[0]))` —
which follows a metavariable-attred operand down to the **leaf
fallback**, so:

- `unsqueeze` saw the leaf shape (`(2,3,4)`-class) — correct;
- `expand` saw the *same* leaf shape, not `unsqueeze`'s output —
  it could never mint `(2,3,2,4)`/`(2,3,2,2,4)`-style targets that
  grow the *inserted* axis;
- `reshape` likewise saw the leaf, so the `head*r` merge target
  `(2,6,4)`/`(2,3,4,4)` never appeared.

Two smaller gaps compounded it: the `expand` table only grew the
*first* 1-extent (to 3), and `_reshape_targets` merged only the two
edge dim pairs — a rank-5 operand's interior merge was never
produced.  And the pinned *nested-view* boundary test
(`transpose(transpose(u, A, B), A, B)` → `TypeError`) documented the
same root cause.

## 2. The fix

`oracle.py`, three mechanics, order-preserving:

1. **Chained attr groups** — `_attr_groups` now union-finds over
   (a) identical metavariable-name sets (existing rule: shared names
   resolve once) and (b) *operand links*: a view node whose first
   operand is another metavariable-attred node joins its group.
   `_chain_domain` enumerates a linked group **innermost-first**:
   under each partial assignment the operand is instantiated with
   the drawn `$attr:` bindings, so `_operand_shape`/`_shape_of`
   returns the real intermediate shape — the cond DSL's
   `unsq-out`/`reshape-out`/`transpose-out` vocabulary made concrete
   through the same typing engine the eval side uses.  Consistency
   is constructive: a metavariable shared inside the chain
   constrains later draws (same resolve-once semantics as
   `_attr_merge` across groups).  Non-view consumers
   (`dropout(softmax(...))`) are deliberately *not* linked — their
   domains are shape-independent and linking only reshuffles the
   diagonal (measured: it pushed `sdpa_fold_add_drop`'s first
   accepted site from inside the 360 window out to ~4748).

2. **Table coverage for the chain's shapes** — `expand`/
   `broadcast_to` now grow *every* 1-extent by {2,3} (the repeat
   factor lands on the unsqueezed axis), appended after the
   original seats; `_reshape_targets` adds interior adjacent merges
   for rank>=4 operands and the cap moved 7→12 — appended, so
   existing domain indices keep their seats.

3. **Rank-4 leaf coverage, scoped** — `_CHAIN_VIEWED_SHAPES` adds
   the `(2,3,2,4)`/`(2,3,4,4)` pair to viewed banks **only for
   patterns containing an operand-chained view**
   (`_has_chained_view`).  An unconditional append also perturbed
   chain-free enumerations (measured: `sdpa_fold_addmul` gained an
   `rhs_err` site inside its 2000-cap window — a clean region
   acquiring a counterexample); scoping keeps every chain-free rule
   byte-identical.

4. **Lazy pull** — `_synth_bases`/`_binding_envs` materialized the
   *entire* (viewed x attr) product up front (~1.6M+ bases for GQA —
   the sweep literally could not start).  `_LazySeq` gives indexed
   pull access; `_diag_groups` probes groups lazily;
   `_binding_envs` pulls one base per round.  The yield order is
   unchanged (pinned by an eager-reconstruction equivalence test
   plus the existing `first_equal`/corner indices).  `_lhs_out_shapes`
   is skipped when no free metavariable can consume it.

## 3. Measured

### The corner exists now

Leaf binding `k=v=(2,3,2,4)`, `q=(2,3,4,4)` (the `(b,t,h_kv,d)` /
`(b,t,h,d)` pair — `repeat-heads` needs `q[-2]==k[-2]*r`, and
`sdpa`/`enable_gqa` needs rank 4; rank-3 spellings *were*
guard-accepted but `rhs-err` — the guard did not check rank, a
latent hole the enumeration could now see.  **Fixed** in a
follow-up: the cond gained `("rank", q|k|v, "==", 4)` — see
`shipped-guard-tightening.md`):

- the k/v chain groups mint `{UD=-2, ES=(2,3,2,2,4), RS=(2,3,4,4)}`
  at option index 29 of 151 consistent triples;
- the merged base passes `rule.check` and evaluates **equal**
  (pinned in `test_gqa_absorb_repeat_guard_accepts_a_chained_binding`).

### Region windows

| rule | before @360 | after @360 | @2000 |
|---|---|---|---|
| `gqa_absorb_repeat` | (unmeasurable — eager materialization) | 0/360 all declined | 0/2000 all declined |
| `qkv_fuse_asym` | 0/360, 0/2000 | 0/360, 0/2000 | — |
| `mul_reshape_inert_l` | 60 acc / 60 eq | 60 acc / 60 eq (identical) | 68/68 |
| `sdpa_fold_add`(+drop) | 15 acc / 5 eq, first eq 139 | identical | — |
| `sdpa_fold_addmul`(+drop) | 6 acc / 2 eq, first eq 327 | identical | — |
| `sdpa_fold_adddiv`(+drop) | 6 acc / 2 eq, first eq 327 | identical | — |
| all other guarded rules | — | identical (enumerations byte-equal) | — |

No guarded region shrank: every chain-free enumeration produces the
same env stream (grouping unchanged, banks unchanged, diagonal
order unchanged); the two chained patterns were all-declined before
and are all-declined within the windows after — but now *reachable
in principle*, and demonstrably non-empty at depth.

### The corner's true depth

The corner lives at viewed-binding index 504 of 512 (leaf index-sum
19 — the appended rank-4 pair necessarily sits late) and base index
~9445 inside it (first accepted) / ~11083 (first `equal`,
`D=1e-5`, `C=False`).  In `_binding_envs` order that is env index
≈ 5.0M / 5.7M — **outside every cap** (360 default, 2000 escalate).
The guarded region is now honest about what it cannot reach: the
gap moved from "the domain cannot express the binding" to "the
binding sits deep in a fair diagonal".  That residual is the
cap-depth problem `enumeration-fairness.md` already names, not a
domain failure.

### Other rules rescued

`qkv_fuse_asym` gained a consistent `(SZ, S)` chain domain (reshape
over `split` output instead of leaf fallback) but stays all-declined
in-window — its starvation is elsewhere (leaf structures), not the
chain.  No other guarded rule changed.

### Cleanliness of the GQA region when reached

At the corner binding the first accepted sites are `D=0.5`/`D=1.0`
dropout draws — `sdpa` applies dropout stochastically per eval, so
those read `unequal` (an RNG artifact of materializing `dropout_p`,
not a law defect).  `D=1e-5` sites evaluate `equal`.  So even when
reached, the synthesized region reports accepted+unequal — the
honest verdict on a law carrying a stochastic attr the enumeration
correctly proposes.

## 4. Cost delta

- `_synth_bases`/`_binding_envs` went from eager full-materialization
  (≈0.46 s per binding × bindings — GQA never finished) to
  pull-only-what's-probed; GQA's first 2000 sites enumerate in
  seconds.
- For chain-free patterns the enumeration is identical — no cost
  delta beyond an `O(1)` `_has_chained_view` scan.
- For chained patterns the per-binding attr product shrinks: the
  consistent joint domain (151 triples per GQA chain) replaces what
  independent per-node domains would have needed (~5×13×50 each)
  plus merge pruning.

## 5. Tests

- `test_chain_domain_resolves_through_view_outputs` — the
  `unsqueeze→expand→reshape` spec unit test: joint assignments are
  minted, consistent (`ES`/`RS` equal the instantiated inner
  outputs), and contain the repeat-merge triple.
- `test_chain_domain_edge_paths` — unbound-operand fallback,
  empty-option veto, intra-chain metavar disagreement.
- `test_synth_bases_skips_fully_conflicted_binding` — a binding
  whose whole product merge-conflicts contributes no group (the
  eager `if bases:` indexing preserved).
- `test_nested_view_candidate_chains_consistently` — the former
  `TypeError` boundary now enumerates a consistent chain group and
  `synthesize` yields all-equal instances.
- `test_binding_envs_lazy_pull_matches_eager_order` — the lazy pull
  stream equals an eager reconstruction elementwise.
- `test_gqa_absorb_repeat_guard_accepts_a_chained_binding` — the
  hand-built pin stands, plus the chained domain contains the
  corner triples and the corner binding evaluates `equal`.

## 6. Remaining limitations

- The GQA corner is *deep* (~5M envs) — dominated by the
  leaf-binding index (504/512) and the two 151-option chain domains.
  Reaching it within a sweep needs either a deeper cap policy or a
  further ordering improvement; both are out of scope.
- The chain enumeration covers only direct `args[0]` links; a view
  buried inside a wider compound operand still resolves through
  `_operand_shape`'s leaf fallback (unchanged, honest posture).
- ~~`gqa_absorb_repeat`'s guard accepts rank-3 bindings whose RHS
  sdpa cannot contract~~ — **RESOLVED** (follow-up, same change
  class as `shipped-guard-tightening.md`): `_COND_GQA_ABSORB` gained
  `("rank", "q", "==", 4)`/`("rank", "k", "==", 4)`/`("rank", "v",
  "==", 4)` — the pattern's `transpose(1, 2)` lands the head axis at
  sdpa's `-3` slot only at rank 4.  Measured on the minted rank-3
  chain (`UD=-2, ES=(2,3,2,4), RS=(2,6,4)` over `q=(2,6,4)`,
  `k=v=(2,3,4)` — 6 bases): old guard accepts all → `rhs-err`;
  tightened guard declines all.  The rank-4 corner still accepts and
  evaluates `equal`.
