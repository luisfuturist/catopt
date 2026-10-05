# Retro: guarded composition — transporting premise guards

**Plan 0017 / ADR 0004** — `novel-depth.md` blocker #2 ("Guarded premises
still can't compose when the guard needs real shapes").  Date: 2026-10.

## The failure

`compose_objects` fires each premise with
`catopt_core.meta.apply_rewrite_at`, which evaluates the premise's guard
on the *symbolic* binding.  A guard that reads real tensor extents can't
decide there.  Reproduced on `compose(om_lift, om_split)`:

```
intermediate: (om_apply (om_elem (concat 's1','s2',dim=SD),
                                   (concat 'v1','v2',dim=VD)))
match at (0,): {'s1':'s1','s2':'s2','v1':'v1','v2':'v2',
                '$attr:SD':'SD','$attr:VD':'VD'}
_check_om_concat_dims -> False
```

`om_split`'s `_check_om_concat_dims` short-circuits at
`if not (isinstance(sd, int) and isinstance(vd, int)): return False` —
the concat dims are *metavariable strings*, not the ints the check needs
— and, even given ints, it then reads `_vshape(s1)`…`_vshape(v2)`, which
are leaves with no shape.  The predicate is **undecidable** on the
symbolic binding, not false; the composition aborted as `None`.

## What changed

`object_synthesis.compose_objects` gained **guard transport**.  A
premise is fired in two steps:

1. **Decidable** (`apply_rewrite_at`, unchanged) — the cond DSL and its
   shape specs decide leaf/rank/spec clauses without concrete values
   (how `affd_scan2/4` always built: `("leaf", "h")` reads a metavar as
   a leaf).  Nothing is transported.
2. **Structural** (`_structural_fire`) — the LHS alone matches; the
   premise is fired *without* its guard and the guard is **transported**
   into the composite, so the composite is admissible exactly where the
   premise would have fired.  A premise whose LHS matches nowhere still
   fails the composition as `None`, honestly.

Transport has two flavours, chosen by what the premise's guard *is*:

* **declarative** (`_declarative_clause`) — a purely declarative
  `cond` (no procedural `check` remainder, every metavariable bound to a
  plain composite metavariable name) is re-expressed by renaming its
  metavariable references and ANDed into the composite's own `cond`.
  The composite stays **pure data** (serializable), so the gauntlet's
  guarded-region sweep can rule on it.
* **procedural** (`_guard_transport`) — a `check`/`derive` with a code
  remainder is re-run at fire time on the premise's *own* binding (the
  intermediate re-instantiated from the composite's binding).  The
  composite is sound but **non-serializable**: the store flags
  `missing_hooks == ["check"]` and the gauntlet's full-data gate refuses
  it.  Construction is a claim; a claim data cannot carry is refused as
  such.

The **first premise's guard** is transported too (`_head_transport`) —
it guards the composite's LHS, so dropping it would leave the composite
under-guarded (measured: without it
`linear_channel_scale_rev ∘ linear_row_scale`'s guarded region carried 3
`unequal` sites; with it, 1 — the premise's own, below).  It is folded in
declaratively when re-expressible; a guard bound to a *compound* subterm
(the specialization replaced a metavar with a term, e.g. `affd`'s
`h ↦ add(mul(a1,h),x1)`) is left to the caller's `cond` declaration (the
pre-existing contract) unless none was given, in which case it rides a
procedural check.

## Measured

Battery: every ordered pair of the 108 shipped rules (11664 pairs; the
`ALL_RULES` base plus `OM_LAWS`), `compose_objects(r1, r2)`:

| | before | after |
|---|---|---|
| pairs that build | 106 | **136** |
| — decidable (unchanged) | 106 | 106 |
| — guard-transported (**new**) | 0 | **30** |

The 30 new builds:

| class | count | gauntlet verdict |
|---|---|---|
| declarative, tautology (inverse pairs) | 18 | refused — `novelty: relation=tautology` |
| declarative, non-tautological | 2 | 1 **usable**, 1 refused — `typed-pay: paid=0` |
| procedural, tautology | 2 | refused — `full-data: dropped hooks: check` |
| procedural, non-tautological | 8 | refused — `full-data: dropped hooks: check` |

Per-composite, the non-tautological new builds:

| composite | transport | verdict |
|---|---|---|
| `linear_channel_scale_rev ∘ linear_row_scale` | declarative | **usable** — 19eq/1ne synth, 1eq real, paid 1 |
| `linear_row_scale_rev ∘ linear_channel_scale` | declarative | refused — `typed-pay: paid=0` |
| `om_lift ∘ om_split` (constructed premise) | procedural | refused — `full-data: dropped hooks: check` |
| `sdpa_cat_{0..6}_dim ∘ om_split` (7) | procedural | refused — `full-data: dropped hooks: check` |
| `square_to_pow ∘ pow_to_rsqrt` | procedural | refused — `full-data: dropped hooks: check` |

`om_lift ∘ om_split` — the headline — now **builds**:

```
(matmul (softmax (concat 's1','s2',dim=SD), dim=-1), (concat 'v1','v2',dim=VD))
  -> (om_apply (om_compose (om_elem 's1','v1'), (om_elem 's2','v2')))
```

and its transported check *is* `om_split`'s guard re-evaluated on the
premise's binding: it accepts the aligned chunk pair
(`s1(5,3) s2(5,4) v1(3,6) v2(4,6)`, SD=-1, VD=-2) and declines a
row-axis score concat (SD=0) and a mis-contracted key dim (v2=(9,6)).

## Honest findings

1. **Transport is faithful — it inherits, never adds, a blind spot.**
   `linear_channel_scale_rev ∘ linear_row_scale`'s one `unequal` synth
   site is *exactly* `linear_row_scale`'s own: the premise's
   `_COND_ROW_SCALE` accepts a rank-1 degenerate weight (`W=c=(1,)`)
   where `linear` mis-evaluates.  The composite's guard is the premises'
   conjunction, so it can only be as tight as its premises
   (`tests/test_discovery_synthesis.py::
   test_transport_inherits_the_premise_blind_spot`).  The composite
   clears the gauntlet via *derivability* (the premises are shipped
   rules), which short-circuits the region sweep — a gauntlet posture,
   not a transport claim.

2. **A procedural premise guard is unadmittable, by construction.**
   `om_split`'s check is code, not a declarative `cond`, so the
   composite carries a code hook and the record cannot replay it
   (`missing_hooks == ["check"]`, refused at `full-data`).  The honest
   route to admitting the chunked-attention object is to give `om_split`
   a declarative `cond` — which the existing atoms cannot do exactly
   (`_chunks_compatible` needs "all dims except the concat axis equal" +
   broadcast, and `shape-compat` is too strong off-axis).

3. **`om_lift ∘ om_split`'s synthesized region is starved anyway.**
   Even were the guard data, the guarded sweep finds 0 accepted sites in
   the first 200k bindings: the guard needs rank ≥ 2 chunks with aligned
   dims, and `_binding_envs` interleaves the free-operand diagonal with
   the attr bases, so the first all-rank-2 corner lands at ~156k sites
   (the attr-sweep retro's "cap starves deep regions", worse here
   because every score/value metavar needs a rank-2 shape).  The *real*
   region accepts 1 and is equal.  The truth gate needs ≥ 1 equal synth
   site, so the object would be refused at `truth` even with a
   declarative guard — `synth_limit` is the knob, not this change.

4. **`first`'s guard is the caller's declaration when it is not
   re-expressible.**  For a compound binding (`affd`'s `h`) the guard
   cannot be renamed onto a composite metavariable, so the caller's
   `cond=` stands as the composite's declared guard — the pre-existing
   contract, unchanged (`affd_scan2/4` still clear the gauntlet).

## Where this leaves plan 0017

Guarded composition no longer declines silently: `compose_objects`
builds the composite and carries the premise guards, as data where it
can and as code where it must — and the gauntlet rules on the result.
The frontier moves from "can it compose?" to "can the guard be
*expressed*?": a declarative `om_split` cond (needs an off-axis
compat atom), and a synthesized region that reaches the aligned-chunk
corner.  `laws/` untouched; promotion still manual.
