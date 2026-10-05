# Retro: enumeration fairness — the (viewed × attr) diagonal

Date: 2026-10-08
Context: `cap-policy.md`'s deep finding — the starved guards' first
equal site sat at a **fixed enumeration index** (793 for
`sdpa_fold_addmul` / `_adddiv`, 469 for `sdpa_fold_add`),
independent of the cap.  The region was never rare; the "fair"
interleave deterministically placed it past the window.  The cap
policy escalated as a workaround; this is the ordering fix it was
standing in for.  `evidence.py`, `laws/`, `intake.py`,
`object_synthesis.py` were not touched.

## The starvation, characterized

The enumeration has three nesting levels, and only two of them were
diagonalized:

1. the **viewed-binding × attr-combination** product — the *base*
   dimension (`_synth_bases`);
2. the **free-operand** product inside each base (`_diag_product`);
3. the **base × free-combo** interleave (`_binding_envs`, index-sum).

`attr-sweep.md` fixed (2) and (3).  Level (1) stayed **shape-major**:
`for viewed in _viewed_bindings(...): for combo in _diag_product(...)`.
Every attr combination of one viewed shape was served before the next
shape opened.

### `sdpa_fold_add` — the base dimension

Guard (`_COND_SDPA_BASE`): `axes-last2(K, TD1, TD2) ∧ axis(Q, SD,
-1)` — the transpose must be exactly the last two axes of a rank ≥ 2
`K`, and the softmax over the last axis.  `K` is the only viewed
metavar (`transpose`); `M`, `Q`, `V` are free.

The base list is grouped by `K`'s shape, in `_VIEWED_SHAPES` order:
`(4,)` first.  For rank-1 `K` the transpose table admits two axis
pairs (`(0,0)`, `(0,-1)`) × four softmax axes × three scales = **24
bases, all guard-declined** (the guard needs rank ≥ 2).  Then the
first rank-2 shape `(2,3)` opens: its diagonal reaches
`TD = (-2,-1) ∧ SD = -1` at diagonal index 3 — the transpose clause
is satisfied at the *first* axis pair (`(-2,-1)` is last-two for rank
2), so only the softmax clause costs a diagonal step.

| | base index | free index | first equal site |
|---|---|---|---|
| shape-major (old) | **27** | 3 | **468** |
| viewed×attr diagonal (new) | **13** | 3 | **139** |

The equal site's index is `triangular(base + free)` — the base index
dominates.  The free index is 3 either way: `M` at bank index 1
(`Const(0.5)`), `Q` and `V` at 0 (the operand-derived shape).  So the
corner is buried by the **attr-base dimension**, not the leaf bank and
not the free diagonal.

For `sdpa_fold_addmul` / `_adddiv` the extra `S` metavar lengthens the
free bank; the base structure is identical (same 324 bases, corner at
base 27 → 13), and the first equal site moves **792 → 327**.

### `om_lift ∘ om_split` — the leaf-bank dimension

The composed object `om_chunk2` (`matmul(softmax(concat(s1,s2),-1),
concat(v1,v2)) → om_apply(om_compose(om_elem(s1,v1), om_elem(s2,v2)))`)
has **no viewed metavar** — `concat` is not in `VIEWISH`.  Its guard
(`om_split`'s transported `_check_om_concat_dims`) needs all four
operands rank ≥ 2 with `s1[-1] == v1[-2]`, `s2[-1] == v2[-2]`.

All four operands are free, so the base dimension is a single group
(16 attr bases) and the burying dimension is the **leaf bank**: the
derived bank leads with the degenerate sentinels `()`, `(1,)`, `(4,)`,
so the first rank-2 entry is at bank index 5.  With four independent
banks and a rank conjunction, the index-sum is ≥ 20:

| | index |
|---|---|
| first all-rank-2 corner | **156135** |
| first equal site | **114750** |

This is the *same class* — a conjunction of clauses over independent
dimensions, each dimension leading with a degenerate corner — but the
buried dimension is the leaf bank, not the attr base.  (See
"Remaining blind spots".)

## What changed

One helper and a rewrite of `_synth_bases`'s nesting, all in
`catopt_discovery.oracle`:

1. **`oracle._diag_groups`** — the ragged-product companion of
   `_diag_product`.  `groups[v]` is one viewed binding's list of
   attr-combination bases (lengths differ — a rank-1 operand's view
   table admits fewer axis pairs); the helper yields `groups[v][a]` in
   increasing `v + a` (Cantor order).  A cap now truncates a *corner*
   of the `(viewed × attr)` space instead of a whole viewed binding.

2. **`_synth_bases`** materializes each viewed binding's attr bases
   into a group and yields through `_diag_groups`.  The `_attr_domains`
   veto (`None`) and the `_attr_merge` conflict-skip are unchanged;
   `_binding_envs` still interleaves `(base, free-combo)` by index-sum
   one level up.  The whole enumeration is now a nested Cantor diagonal
   at every level.

The reorder is a **pure permutation** — the same sites, the same
banks, the same values.  Only the order inside the cap changes.

## Measured

Across all 35 measurable shipped guarded rules (`gqa_absorb_repeat`
excluded — its enumerator boundary is a separate, slow case):

| metric | shape-major | diagonal |
|--------|-------------|----------|
| non-empty accepted region | 19 | **25** |
| total envs @360 | 11949 | 11949 |
| rules whose region shrank | — | **0** |

Six rules surface (region non-empty, at least one equal site, no
counterexample):

| rule | first equal (old → new) | acc@360 | eq@360 |
|------|-------------------------|---------|--------|
| `sdpa_fold_add` | 468 → **139** | 0 → 15 | 0 → 5 |
| `sdpa_fold_addmul` | 792 → **327** | 0 → 6 | 0 → 2 |
| `sdpa_fold_adddiv` | 792 → **327** | 0 → 6 | 0 → 2 |
| `sdpa_fold_add_drop` | >2500 → **139** | 0 → 15 | 0 → 5 |
| `sdpa_fold_addmul_drop` | >2500 → **327** | 0 → 6 | 0 → 2 |
| `sdpa_fold_adddiv_drop` | >2500 → **327** | 0 → 6 | 0 → 2 |

Every other guarded rule's accepted count, equal count and first-equal
index is byte-for-byte unchanged (`select_mul`, `softmax_fold`,
`glu_fold`, `naturality_scalar`, the `weight_*` / `linear_*` scales,
`swiglu_fuse`, `parallel_mul_fuse`, …).  **No regression.**

**The escalation is now unnecessary for the headline folds.**  Under
`_guarded_truth` (which applies the selective cap) the six
`sdpa_fold_add*` rules no longer escalate — they accept at the first
window:

| | guarded-truth envs (6 folds) | total (9-rule sample) |
|---|---|---|
| shape-major | 6 × 2360 | 18892 |
| diagonal | 6 × 360 | **6892** |

A 12000-env reduction: the ordering fix *pays* for itself by removing
the second window the policy used to buy.  `rms_norm_fold` /
`assoc_linear_bias` still escalate (their gaps are shape/kind, not
ordering) — the policy remains a latent safety net.

## The declined alternative — a rank-fair leaf bank

The natural twin for the `om_chunk2` / `assoc_linear_bias` class is to
reorder the **free leaf bank** the way `_diag_groups` reorders the
base dimension — a round-robin over the bank's *rank classes*, so the
lowest-index shape of each rank precedes any rank's deep tail (the
rank-2 corner would move from bank index 5 to ~2).

Measured (keep the documented index-0/1/2 seats, round-robin the rest):

| rule | accepted | note |
|------|----------|------|
| `linear_channel_scale` / `_rev` | 24 → 20 | **shrink** |
| `linear_row_scale` / `_rev` | 22 → 20 | **shrink** |
| `naturality_scalar` / `_rev` | 148 → 145 | **shrink** |
| `assoc_linear_bias` / `_rev` | 0 → 0 | no rescue |

It regresses six rules and rescues **none** of the target class: the
`assoc_linear_bias` guard needs *five* simultaneous rank clauses
(`A`,`B` rank 2, `b1` rank 1, `b2` scalar/rank 1, `x` rank ≥ 1), so a
bank reorder that moves a rank-2 shape forward also moves the rank-1
shapes the other rules already measure on.  The value-bank retro
pinned index-0/1/2 for exactly this reason; moving them is not paid
for.  **Declined** — recorded, not hidden.

## Remaining blind spots

1. **The free-bank class is a different dimension.**  A guard that is
   a conjunction of *rank* clauses over several independent **free**
   operands (`assoc_linear_bias`, `om_chunk2`) is buried by the leaf
   bank, which the viewed×attr diagonal does not touch.  The
   rank-fair reorder above is the obvious lever and is measured
   counterproductive; a guard-aware bank (order each operand's shapes
   by the clauses the guard names) would need the rule's `cond` inside
   `_binding_envs`, which is a signature change this change did not
   take.
2. **`gqa_absorb_repeat` is still unmeasurable.**  Its enumerator
   raises (the oracle's pinned single-level-view boundary); it is
   outside both the cap policy and this reorder.
3. **The shape/kind gaps are untouched.**  `rms_norm_fold*` (a
   `tail-block` shape fact), `sdpa_fold_masked_fill*` (a bool mask
   leaf) still measure 0 — reordering cannot construct a value or a
   kind the bank never mints.
4. **Eager materialization grew slightly.**  `_synth_bases` now
   builds each viewed binding's attr bases into a list before
   yielding; `_binding_envs` already materialized all bases, so the
   upfront cost is the same order (~O(bases)) — bounded, but noted.
