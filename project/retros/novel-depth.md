# Retro: novel depth — constructed objects past the catalog

**Plan 0017 / ADR 0004, stage 5** — the frontier after
`novel-abstraction.md`: not another pipeline candidate re-minted, but
objects the machinery composes that *no ruleset contains*.  Date:
2026-10.

## What the corpus made constructible

The survey (`impact`/`intake` terms, 276 real terms): the largest
composable units that *fire on real models* are (a) the spelled
attention-score block — `matmul(softmax(matmul(q,kᵀ)·s, -1), v)` — and
(b) the nested selective-scan spine —
`add(mul(a_t, add(mul(a_{t-1}, …), x_{t-1})), x_t)` — spelled 4-deep in
`DiagonalSSM`, `HybridBlock`, `TwoLayerHybrid` (and densely in
`LinearRecurrence`).  ALiBiAttention arrives already `sdpa`-folded;
the RNN cells arrive as fused kernels — neither is a construction
site.

Two facts about the shipped surface decided the targets:

- Every `sdpa_fold_*` law requires an `add`/`masked_fill` **mask**
  operand.  The mask-free block — `softmax(q@kᵀ)@v` and its
  `div`-scaled twin — is spelled by `nn.MultiheadAttention`,
  `ManualAttention`, `HybridBlock`, `TwoLayerHybrid` and folds to
  *nothing*.
- The diagonal-affine carrier (`aff_diag`/`applyd`,
  `SCAN_DIAG_LAWS`) ships the *one-step* lift and a step-composer —
  never the fused n-block object.  And it is *cheap* in the e-graph
  where the dense `aff` carrier was not.

## The machinery delta

`object_synthesis.py` grew the declarative surface, not new
combinators:

- `fold_object` accepts a **full term spec** as the RHS (was: unary
  kernel name only) — the fold generalises to "spelled composition →
  dispatched form" for abstractions like `sdpa(q,k,v,scale=…)`.
- All three ops take `cond=`/`dspec=` — the object's own declarative
  guard/derive, the same serializable data shipped laws carry.  For
  `compose_objects` this is the constructor *declaring* the
  composite's guard; full cond-transport is still unwired.

Nothing procedural was admitted: every object is `serializable` with
`missing_hooks == []`.

## Constructed objects and verdicts (real corpus, `--admit-object --gauntlet`)

| object | op | statement | verdict |
|---|---|---|---|
| `sdpa_fold_nomask` | fold | `matmul(softmax(q@kᵀ,−1),v) → sdpa(q,k,v,scale=1)` | **usable** — 3 real matches, guarded sweep 11eq synth + 3eq real, paid 3/3, closure 1.03× |
| `sdpa_fold_div_nomask` | fold | `matmul(softmax(q@kᵀ/S,−1),v) → sdpa(q,k,v,scale=1/S)` | **usable** — ManualAttention, num_true, paid, 1.11× |
| `affd_step_lift` | lift | `a⊙h+x → applyd(aff_diag(a,x),h)` | **usable** — 24 matches, 16 fires, paid 3, 1.15× |
| `affd_scan2_lift` | compose | two-block diagonal scan → nested `applyd` | **usable** — 12 matches, 12 fires, paid, 1.19× |
| `affd_scan4_lift` | compose∘compose | four-block scan, self-composed | **usable** — 4 matches, 4 fires, 1.43× |
| `sdpa_fold_div_nomask_g` | fold (guarded twin) | same as div, but `scale="SC"` metavar + cond | **refused at truth** — synth region *empty* |

`sdpa_fold_nomask` is the task's "named attention-score-block
object": one dispatched abstraction for the spelled block, guarded on
the transpose axes (`axes-last2`), carrying its `cond` as data.

`affd_scan4_lift` is the deep test: `compose(affd_scan2_lift,
affd_scan2_lift)` — **both edges are constructed objects**; the
derivation names `affd_scan2_lift` alone, a premise in no shipped
ruleset.  The machinery composed its own prior construction and the
gauntlet admitted the result.

`affd_scan2_lift` likewise composes a *constructed* premise with
itself (`affd_step_lift` — the honest caveat: its body mirrors the
shipped `affd_lift`, which lives in `SCAN_DIAG_LAWS`, outside the
`ALL_RULES` base the novelty gate diffs against).

## The honest blockers — measured

1. **The guarded-region sweep cannot enumerate attr metavars on
   non-view ops.**  `oracle._attr_options` covers view ops only;
   `softmax(dim="SD")` or `sdpa(scale="SC")` in a pattern empties the
   synth domain — *even for the shipped* `sdpa_fold_*` rules
   (verified: 0 synth sites for `sdpa_fold_addmul`).  A guarded object
   whose RHS mints an attr metavar is unadmittable: the gate requires
   ≥1 synthesized equal.  Worked around, honestly: `sdpa_fold_nomask`
   keeps `dim=-1`/`scale=1.0` literal (the *pattern* is the guard on
   those axes; metavar transpose dims stay enumerable), and
   `sdpa_fold_div_nomask` moves the guard into the **derive** — the
   `dspec` float-veto declines non-`Const` scales at fire time, so no
   unsound member is minted, while truth rides the numeric oracle.
   The `_g` twin is the documented counterfactual: identical math,
   guarded the "right" way, refused on `synth: 0 accepted`.
2. **Guarded premises still can't compose when the guard needs real
   shapes.**  `compose(om_lift, om_split)` — the two-block chunked-
   attention object — returns `None`: `om_split`'s dim-deriving check
   declines on metavar terms.  (Guarded composition *does* work when
   the predicate accepts symbolic bindings — `leaf h` treats a metavar
   as a leaf, which is exactly how `affd_scan2_lift`/`scan4` built.)
   Cond-transport remains the seam.
3. **Carrier price is a carrier property, not a lift property.**  The
   dense `aff` lifts died at closure last stage (2.72× / 6.89×
   enodes); the diagonal `affd` family clears at 1.03–1.43× on the
   same gate.  The verdict moved with the monoid, not the
   construction — an object the verifier *needs* is one whose
   abstraction is cheap to keep around.

## Where this leaves plan 0017

The object language now contains: two fused-attention objects covering
the mask-free spelling the shipped fold family missed, a guarded
carrier lift declared as data, and a self-composed n-block carrier
object — all `usable` through the full adversarial path on the real
corpus.  The standing frontier: extending the oracle's attr domain so
guarded objects with minted RHS attrs (`scale="SC"`) can be swept, and
cond-transport for guarded composition.  `laws/` untouched; promotion
still manual.
