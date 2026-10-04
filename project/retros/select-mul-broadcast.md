# Retro — `select_mul` was latently unsound (broadcast along the selected axis)

## What happened

The view/index oracle (`tools/law_view_oracle.py`, built to resolve the
intake-r2 "unproven" bucket) ran a real-match sweep + satisfiable-
instance synthesis on every view-family candidate. Its proposal mirror
of `select_mul` — `census:mul_select` — came back **conditional**, and
the counterexample is real:

    u = (4,), v = (2,4), D = 0
    mul(sel(u,0,0), sel(v,0,0)) = u[0] · v[0]   # scalar × vector → [0,0,0,0]
    sel(mul(u,v), 0,0)          = (u · v)[0]   # u broadcasts → u · v[0]
                                               #          = [0,1,4,9]

Broadcasting `mul(u, v)` along the *selected* axis changes the result.
The structural precondition the law's comment described ("shared
dim/index metavars enforce it, no check needed") was true as far as it
went — it missed that `mul`'s broadcast is the other degree of freedom.

## Why it never bit

All 13 real corpus sites had `shape(u)[D] == shape(v)[D]` — the sites
that motivated the law (same-shape projections) satisfy the missing
condition by construction. The shipped `sink.verify` oracles only ever
saw the true region. Exactly the matmul-unsoundness pattern: a law
true on every real site, false on adversarial bindings, invisible
until an adversarial oracle runs.

## Fix

New cond predicate `("dim-eq-attr", A, K, B, K2)` — dims named by
attr metavars (`shape(u)[D] == shape(v)[D]`); `select_mul` now carries
`cond=("dim-eq-attr", "u", "D", "v", "D")`. Broadcast along *other*
axes stays allowed (u=(2,1), v=(2,4), D=0 still fires — both sides
broadcast identically post-selection). The law stays fully
serializable — the guard is data. Test:
`tests/test_select_laws.py::test_select_mul_broadcast_axis_guard`.

## Second catch in the same sweep

`pow`'s shape rule returned the base shape unconditionally —
`pow((), (4,))` reported `()` while aten evaluates `(4,)`. Fixed in
`80b133f` (broadcast like the other elementwise binaries). The
conditional-candidate sweep surfaced it because the ALiBiAttention
`mul_unsqueeze_reshape_wl` site was accepted on an IR-believed shape
execution contradicts.

## The meta-lesson (twice now)

"Shipped + verified on all real sites" is not "sound". Both the
matmul rank bug and this broadcast bug were caught the same way — an
oracle that *synthesizes bindings outside the observed corpus* (the
adversarial witness generator, the view oracle's shape bank). The
adversarial layer is the part of the referee that keeps earning its
keep; every shipped law that lacks a `cond` should eventually get a
synthesis pass over its metavar bindings.

## Honest limits

- `dim-eq-attr` declines when either side's shape is unknown — the law
  now won't fire on shape-less binds (conservative, never wrong).
- `select_add`/`select_sub`/`slice_mul`/`sub_unsqueeze` proposals have
  the same latent issue as *candidates* — none are shipped laws yet,
  and the conditionals their proposals resolved to are the guard
  table in `project/retros/cond-dsl-view-guards.md`.
