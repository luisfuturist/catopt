# Retro: the attr sweep — enumerating metavars on non-view ops

Date: 2026-10-05
Context: plan 0017 / ADR 0004 — `novel-depth.md` blocker #1.  The
guarded-region sweep (`evidence._synth_sites`, mirroring
`oracle.synthesize`) enumerated leaf shapes and view-op attrs only:
`_attr_options` returned `None` for any op outside the view table, so
one str-valued attr on `softmax`/`sdpa`/`sum` emptied the *entire*
synthesized domain — the guarded truth gate then refused every such
object on `synth: 0 accepted`.  `meta_game.py`, `pipeline.py`,
`laws/` were not touched.

## What changed

1. **Generic attr domains** (`oracle._attr_kind` / `_kind_domain` /
   `_generic_attr_options`).  `_attr_options`'s wildcard now types
   each metavar'd attr key: positional spellings canonicalize
   through `catopt_core.attrs.ATTR_SCHEMA` (`sdpa.arg6` -> `scale`),
   `(op, name)` overrides beat the name table (`rms_norm.dim` is a
   normalized-shape tuple, not an axis; `eye.dim` is an int), and
   reduction ops' `dim` enumerates both the scalar and tuple
   spellings (`_REDUCTION_DIM_OPS`).  Kinds: `axis` (the operand's
   shape-valid axes, `_dims`), `red-dims`, `int` {0,1,2},
   `float` {0.5,1.0,1e-5}, `bool` {False,True}, `shape` (trailing
   blocks).  A key the table cannot type — tensor-valued
   (`attn_mask`), string payloads (`equation`), unknowns (`frob.x`)
   — stays honestly unenumerable: `None`, the node is skipped.
   Only metavar'd keys enumerate; literal attrs never veto.

2. **The enumeration machinery moved into `oracle`**
   (`_metavar_parents`, `_synth_bases`, `_binding_envs`,
   `_attr_merge`, `_lhs_out_shapes`); `evidence._synth_sites` is
   `oracle._binding_envs` + instantiation + pair-dedup, and its
   helpers delegate verbatim.  One enumeration, two consumers.

3. **Fair ordering.**  Three changes, all measured necessary:
   `_diag_product` (Cantor order inside every product),
   `_binding_envs` interleaves `(base_index, free_index)` by
   index-sum — the old nesting served one base's ~10⁴ free combos
   before the next base opened — and `_leaf_bindings`' free bank
   leads with the *bound operand's own shape* (derived shapes
   before the static bank; `Const` at index 1, scalar `Var` at 2).
   The evaluable corner is not a low-index corner of the
   scalar-led product: for the `_g` twin the first equal site sat
   ~5,900 sites in under scalar-led order, ~39 under operand-led.

4. **`synthesize` takes `derive=`.**  `verify_view_candidate` rides
   the proposal's `derive` on synth bindings exactly as
   `sweep_real` does — the computed `$attr:` value overrides the
   enumerated placeholder (a metavariable the rule derives is not
   a free choice), and a vetoed binding is skipped, not counted.
   Without it, `sdpa(scale="SC")` minted wrong scales and produced
   false `unequal` counterexamples on derived-attr objects.

## Measured

The `_g` twin (`sdpa_fold_div_nomask_g`, `scale="SC"` metavar +
`axes-last2`/`const-num` guard + `dspec`), limit=360:

|                          | before | after                          |
|--------------------------|--------|--------------------------------|
| synth sites              | 0      | 360                            |
| accepted                 | 0      | 53                             |
| equal / unequal / rhs-err| 0/0/0  | **16 / 0 / 0**                 |
| other-err / declined     | 0/0    | 37 / 307                       |
| gauntlet verdict         | truth: FAIL (vacuous) | **usable: True** (8/8 gates) |

Real region: 1 accepted / 1 equal — the ManualAttention firing.

Shipped guarded laws, same sweep (every one was 0 sites before):

| law | accepted | equal | other |
|-----|----------|-------|-------|
| `softmax_fold`  | 164 | 139 | 25 other-err, 196 declined |
| `glu_fold`      | 9   | 9   | 3 declined |
| `swiglu_fuse`   | 65  | 0   | **4 rhs-err**, 61 other-err |
| `sdpa_fold_*` (12) | 0 | 0 | 360 declined |
| `rms_norm_fold*`   | 0 | 0 | 360 declined |

Two honest new findings the widened domain surfaced:

- **`swiglu_fuse` mints ill-typed members its guard accepts** —
  `shape-compat A B` passes on rank-1 bindings (`A=(4,), B=(4,)`)
  where the fused `linear(x, concat(A,B,0))` + `chunk` RHS cannot
  denote.  The shipped law has a latent mint defect the old
  envelope could not see; a rank guard on the operands is the fix
  (laws/ untouched — recorded here).
- **`sdpa_fold_div_nomask` needed a rank cond.**  The raw oracle
  went `conditional (44 equal / 0 unequal; 4 ill-typed-RHS)` —
  rank-1 `Q` mints `sdpa` below its 2-D floor while the `matmul`
  spelling still evaluates.  The object now declares
  `rank >= 2` on Q/K/V and clears the gauntlet on its guarded
  region (19 equal / 19 accepted).

`compose(om_lift, om_split)` still returns `None` — **unchanged**.
The blocker is `om_split`'s shape-derived `check` declining on
symbolic bindings at compose time (blocker #2 of `novel-depth.md`),
not attr enumeration.  Cond-transport remains the seam.

## What it cost

- Capped sweep counts shifted honestly: `mul_unsqueeze_l_id`
  synth region now lands 23 equal / 321+ declined at 360 (was 25).
- The separator `verify_view_candidate` names for that strip
  changed: `unsq:d_in_pad` — the same broadcast-pad guard, found
  directly rather than inside an `id:`-prefixed conjunction.
- `softmax(Const(0.5), d) == 2·0.5` — a degenerate equality the
  early-`Const` bank now reaches; the `softmax(U,D) -> mul(U,2)`
  strip honestly reads `conditional`, not `false`, on that
  binding.
- Eagerly materialized bases: `_binding_envs` builds the whole
  (viewed x attr) base list before interleaving — bounded, but
  ~O(bases) upfront where the old loop was lazy.

## Remaining blind spots

1. **The cap still starves deep regions.**  `sdpa_fold_addmul` has
   9 equal sites by limit 2000 — 0 inside 360: more metavars mean
   ~10⁵ envs and a multi-clause guard keeps the accepted corner
   deep.  The ordering truncates a corner honestly; for heavy
   patterns `synth_limit` is the knob, not a bigger default.
2. **Finite kind domains.**  `float` probes {0.5, 1.0, 1e-5} —
   `sdpa_fold_div_nomask` verifies only because `derive` rewrites
   SC anyway; a rule whose truth needs `scale=1/√d` *enumerated*
   would not see it.  Enumerated values are proposals, the eval
   disposes — but unsampled values stay unprobed.
3. **Unenumerable attrs stay unenumerable** — tensor-valued
   (`attn_mask`), list-shaped (`sizes`), string payloads
   (`equation`): the honest `None` skip means such patterns still
   get no synth region.
4. **`other_err` is information-poor.**  37 of 53 accepted `_g`
   sites eval-err — scalar/const operands minting ill-typed
   `matmul`s.  The bank could learn "plausible for this op"
   ordering per parent op the way `getitem` gets tuple-sources.
5. **The sweep and the fire path can still disagree** on
   attr-domain edge cases (e.g. an `arg6` spelling instantiating
   differently than the canonical name) — same risk class as the
   old view tables, now wider.
