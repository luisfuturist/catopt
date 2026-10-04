# Derive DSL — `dspec`, closing the last non-data hook

`condition-dsl.md` made the *side condition* serializable; a law
still carried one opaque callable whenever its RHS needed an
attribute the LHS could not bind — `derive=lambda bound:
{"$attr:SZ": ...}`.  This change adds the twin DSL: a **`dspec`** is
a `{NAME: expr}` map (or tuple of pairs) of declarative value
expressions evaluated against the same `bound` environment, folded
into `rule.derive` at `Rewrite.__post_init__` exactly the way `cond`
folds into `rule.check`.  With it, **57 of the 61 shipped tensor
laws are now full-data** — up from 43 — and the 16 `SCAN_DIAG_LAWS`
scan rules joined them too (their two procedural *checks* —
`_affd_state_like` / `_affd_unit_state_like` — were one-line `cond`
trees each and rode along).

Reproduce:

    .venv/bin/python -m pytest tests/test_derive_laws.py \
        tests/test_law_serialize.py tests/test_cond_laws.py -q
    .venv/bin/python -c "
    from catopt_core.laws import ALL_RULES
    from catopt_core.laws.serialize import missing_hooks
    print(sum(not missing_hooks(r) for r in ALL_RULES), '/', len(ALL_RULES))"

## 1. The mechanism

* `Rewrite.dspec: Any = None` (`egraph/types.py`) — the declarative
  derive spec, stored as data.  `__post_init__` *canonicalises*
  (`derive_from_data` — dicts and pair lists become sorted tuples of
  `(NAME, expr)` pairs, hashable and `==` to their store-loaded
  twins) and *folds* — `compile_derive(spec, derive)` becomes the
  rule's `derive`, evaluating the spec first, then any procedural
  remainder (a `None` from either vetoes; the maps merge, code wins
  collisions).  Every evaluation site — `apply_rule`, certificate
  replay, meta's composite guards — stays on the single
  `rule.derive` convention.
* `derive=` *accepts* a spec too: a non-callable value is recast to
  `dspec`, and an `as_derive(spec)` partial is recognised by shape
  (`eval_derive` target, one positional arg) and re-folded — so a
  spec is full data however it arrived.  A spec given twice
  (`dspec=` plus a spec-shaped `derive=`) raises `ValueError`.
* `laws.cond` — same module, second half: `eval_derive` (the
  interpreter — returns the `{"$attr:NAME": v}` map or `None` on
  veto), `compile_derive` (the fold), `as_derive` (the
  derive-shaped `functools.partial` view for the kept `_derive_*`
  test aliases), `derive_to_data` / `derive_from_data` (JSON
  canonical forms).  Declines raise an internal `_Decline`;
  malformed nodes raise `ValueError` — a bug, not a veto, same
  contract as `eval_cond`.
* `laws.serialize` — `_proc_derive` probes the fold the way
  `_proc_check` does (a spec-only `compile_derive` closure shares
  one code object), `missing_hooks` flags only a genuine code
  remainder, the record carries `"dspec"`, and `LAW_FORMAT` bumped
  to **2** (new field + new reconstruction semantics; old v1
  records miss cleanly).
* `rulecache.ruleset_fingerprint` hashes `derive_to_data(r.dspec)`
  — the folded closure's signature is shared across every
  spec-carrying rule, so without this a spec edit could not
  invalidate the synthesis cache (the same bug cond had, fixed the
  same way).
* `R(..., dspec=...)` forwards it; law sources read
  `dspec=_DSPEC_*` next to the `_COND_*` constants they already
  carry.

## 2. The grammar

    dspec := {NAME: expr, ...} | ((NAME, expr), ...)
    expr  := int | float | bool | None            # literal
           | ("lit", v)                           # explicit literal
           | ("attr", NAME) | ("attr0", NAME)     # $attr read / unwrap
           | ("const", T)                         # bound leaf's .value
           | ("shape", T) | ("dim", T, i)         # inferred shape / dim
           | ("leaf-dim", T, i)                   # declared .typ dim
           | ("len", e)
           | ("tuple", e...) | ("concat", e...)
           | ("add"|"sub"|"mul"|"fdiv"|"floordiv", e, e)
           | ("neg"|"recip"|"float"|"int", e)
           | ("bcast", T, T)                      # broadcast, concrete

`T` is the cond DSL's *shape spec* (metavar name or `("mm-out", …)`);
`NAME` keys mint under `"$attr:NAME"`.  Strictness is the cond
posture: anything uncomputable — missing attr, unknown or
non-concrete shape, non-numeric const, zero division, out-of-range
index — vetoes the whole spec.

## 3. The survey — every shipped `derive=`

| Laws | What the hook computed | Result |
| --- | --- | --- |
| `softmax_fold` | `SD` = `dim` tuple unwrapped (`(-1,) → -1`, scalar passthrough) | `{"SD": ("attr0","RD")}` |
| `qkv_fuse_asym` | `SZ` = `(|Q|,|K|,|V|)` out-dims of three bound weights | `{"SZ": ("tuple", leaf-dim×3)}` |
| 12 `sdpa_fold_{add,masked_fill}{mul,div,}{,_drop}` | `SC` = `float(S.value)` / `1/that` / `1.0` | `float∘const`, `recip∘float∘const`, literal `1.0` |
| 4 `affd_lift_unit*` | `US` = `broadcast(shape h, shape x)` | `{"US": ("bcast","h","x")}` |
| `rms_norm_fold`, `rms_norm_fold_nogain` | `ND` = trailing-block normalized shape `u.shape[-k:]`, `EP` = `float(eps.value)` | **stays Python** — needs reduce-dims set normalization (`{d%r}` = `{r-k..r-1}`), not shape arithmetic |
| `glu_fold`, `gqa_absorb_repeat` | none (`check`-only blockers) | unchanged |

Census after migration (`missing_hooks` over `ALL_RULES`):

* **57 full-data** (was 43): the 14 derive-only laws joined.
* **4 still flagged**: `glu_fold` (`check`), `gqa_absorb_repeat`
  (`check`), the rms pair (`check` + `derive`).  All four are
  check-blocked — making their derives declarative alone would not
  have made them serializable anyway.
* The scan set outside `ALL_RULES` went to **16/16 full-data**: the
  two state guards were expressible `cond`s verbatim —
  `Op → op in {add,sub,apply,applyd}` else leaf — and the four unit
  lifts carry `dspec` too.

## 4. Semantics kept verbatim (and one deliberate upgrade)

* `leaf-dim` reads *declared* `.typ.shape` only and requires every
  dim non-`None` — an `Op` binding declines, exactly the old
  strictness (`_shape_of` inference would have *widened* what the
  hook accepted).
* `attr0` indexes only `tuple` (a list attr passes through
  verbatim) — the old `isinstance(dims, tuple)` check verbatim.
* `bcast` resolves both operands through `_shape` and feeds
  non-tuple results to `_broadcast` as `None` — the old hook's
  "unshaped side is a wildcard" behaviour kept, *including* the
  case where `h` is unshaped and `x` wins.
* Upgrade, documented: a zero divisor or unindexable empty tuple
  now *declines* instead of raising — the old `1.0/s` /
  `dims[0]`-on-`()` would have crashed `apply_rule`.  Declining is
  the DSL's total-predicate posture and strictly safer.

## 5. Limits — honest ones

* The grammar covers exactly what the 18 sites needed plus the
  obvious arithmetic/table siblings; there is no reduce-fold
  (`sum`/`prod` over dims), no set normalization, no
  metavar-*term* outputs (the `derive` contract can mint non-`$attr`
  bindings in principle — no shipped site needed it).
* `rms_norm_fold`'s derive is irreducible here: `u.shape[-k:]` with
  `k = |dims|` *and* the trailing-block side condition are really
  one computation — putting half in a spec gains nothing while the
  check stays procedural.
* The `affd` unit lift's one side effect —
  `_LeafRegistry.register(Const(1.0))`, needed because
  `_instantiate` keys leaf enodes by repr without registering —
  moved to module import.  The registry is a single global table,
  so import-time registration is identical to per-firing.
* `LAW_FORMAT` is 2 now: lemma stores holding v1 records miss the
  version check and must be re-stored (the intended semantics of
  the bump — a v2 record read by v1 code would have silently
  minted `"serializable": true` while dropping the derive).
