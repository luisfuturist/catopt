# Derive → data, finished — stage 0 of plan 0017

`derive-declarative.md` closed the *derive* gap for 14 laws; ADR 0004
asks for the residual to shrink honestly, not by force.  This stage
enumerated the four laws still carrying a procedural hook — all four
were *check*-blocked, not derive-blocked — and every one migrated to
`cond`/`dspec`.  **The shipped library is now 61/61 full-data**, and
`SCAN_DIAG_LAWS` remains 16/16.

Reproduce:

    .venv/bin/python -m pytest tests/test_derive_laws.py \
        tests/test_cond_laws.py tests/test_law_serialize.py \
        tests/test_lemma_certificates.py -q
    .venv/bin/python -c "
    from catopt_core.laws import ALL_RULES
    from catopt_core.laws.serialize import missing_hooks
    print(sum(not missing_hooks(r) for r in ALL_RULES), '/', len(ALL_RULES))"

## 1. What the four hooks actually computed

| Law | Hook(s) | What it computed | Declarative form |
| --- | --- | --- | --- |
| `glu_fold` | `check` | parity of the attr-named split axis — `u.shape[D % rank]` a known even int (odd axes split 2+1 under `chunk` while `glu` halves exactly) | `("dim-mod", "u", "D", 2, 0)` — new predicate: `sa[K mod rank] % m == r` |
| `rms_norm_fold` | `check` + `derive` | reduce dims name exactly u's last `k` axes **and** `w.shape == u.shape[-k:]`; mints `ND = u.shape[-k:]`, `EP = float(eps)` | `("shape-eq", "w", ("tail-block", "u", "MD"))` + `{"ND": ("shape", ("tail-block","u","MD")), "EP": ("float", ("const","EPS"))}` |
| `rms_norm_fold_nogain` | `check` + `derive` | same trailing-block proof, no `w` gate | `("shaped", ("tail-block", "u", "MD"))` + the same dspec |
| `gqa_absorb_repeat` | `check` | unsq→expand→reshape is `repeat_interleave` on both kv operands, equal expand shapes, `q.heads == k.heads × r` | `("repeat-chain", k/v, UD·, ES·, RS·)` + `("attr-eq-attr", "ESk", "ESv")` + `("repeat-heads", "q", "k", "UDk", "ESk")` |

Census after migration (`missing_hooks` over `ALL_RULES`):

- **61 full-data** (was 57): the derive DSL's last procedural derive
  and the three remaining procedural checks are all gone.
- The one genuinely new *shape spec* is `tail-block`; the new
  *predicates* are `dim-mod`, `attr-eq-attr`, `repeat-chain`,
  `repeat-heads` (plus internal helpers `_attr_dims`,
  `_trailing_block`, `_repeat_merge_ok`, `_concrete_rank2`,
  `_rep_factor`, kept under the radon threshold).

## 2. The new DSL atoms

- **`("dim-mod", T, K, m, r)`** — the dim the bound attr `K` names
  satisfies a modular congruence.  Covers the parity/halving class:
  any "the split axis must be even / divisible / aligned" guard.
- **`("tail-block", T, NAME)`** (shape spec) — resolves to
  `shape(T)[-k:]` iff attr `NAME`'s reduce dims name exactly T's last
  `k` axes — the `F.*_norm` `normalized_shape` contract.  One spec
  serves *three* sites: `shaped` (the block exists), `shape-eq` (the
  gain IS the block), and `("shape", spec)` in the dspec (mint the
  block).  This is the composition the last retro called irreducible:
  "`u.shape[-k:]` and the trailing-block side condition are really one
  computation" — so they became one spec.
- **`("repeat-chain", T, UD, ES, RS)`** — the copy-map chain:
  `reshape(expand(unsqueeze(T,UD),ES),RS)` is `repeat_interleave` on
  dim `d-1`.  A real class, not one-off: `decode_laws` consumes the
  same verification through `_check_repeat_chain` (which now delegates
  to the predicate — the helper and the rule's cond cannot drift).
- **`("attr-eq-attr", A, B)`** — two bound attrs compare equal.
  The obvious `attr-eq` sibling; an unbound side declines.
- **`("repeat-heads", A, B, UD, ES)`** — `sa[-2] == sb[-2] × es[d]`,
  `d = UD mod (rank sb + 1)`.  The one law-specific atom — same
  posture `flat-map-unsq` set: a coherent named property whose args
  are data, housed in the fixed interpreter rather than a per-law
  lambda.

## 3. Semantics kept verbatim (and two documented widenings)

- Every strictness posture is preserved: the strict reading of reduce
  dims (bare int / tuple / list of non-`bool` ints), concrete-shape
  requirements, `d != 0` after normalization, `r > 1`, expand-may-
  only-grow-the-inserted-dim, `rs == merged` — each is transcribed
  1:1 from the Python hooks it replaced, and the branch-by-branch
  verdicts are pinned by the pre-existing hook tests (untouched:
  `test_rms_norm_laws`, `test_glu_fold_law`, `test_laws_rewrite_edges`
  still call the same `_check_*`/`_derive_*` names — now `as_check` /
  `as_derive` aliases or thin `eval_cond` delegates, so they *are* the
  same data the rules carry).
- Widening 1 (documented): `("float", ("const", "EPS"))` accepts a
  numeric-looking *string* Const (`float("1e-5")` parses) where the
  hand hook's `isinstance(eps, (int, float))` vetoed.  The rule's
  `("const-num", "EPS")` cond runs first in the folded guard, so no
  firing site can reach it — identical to the accepted
  `("float", ("const","S"))` widening on the sdpa specs.
- Widening 2 (documented): a zero modulus / non-int modulus in
  `dim-mod` declines instead of raising — the DSL's total-predicate
  posture, strictly safer.

## 4. The honest residual

Nothing shipped is procedural anymore — `missing_hooks` is empty
over the whole library.  What remains is a *vocabulary* boundary,
not a scar:

- The DSL still has no term-level iteration or lookup beyond named
  atoms — a guard that needs open-ended computation (data-dependent
  loops the interpreter has no predicate for, external state) still
  takes the `check=`/`derive=` hook path, flagged honestly by
  `missing_hooks`.  That is the ADR's documented escape hatch, not a
  cheat.
- The honesty-path tests no longer have a shipped flagged law to
  stand on — they now exercise the `serializable: false` record with
  a *synthetic* cond+check+derive rule (`_flagged_rule` in
  `test_law_serialize`), which is a cleaner unit of the contract
  anyway: the codec's flag/drop/rebuild behaviour no longer depends
  on which laws happen to be procedural.
- `repeat-heads` is the one bespoke predicate added for a single
  law; it follows the `flat-map-unsq` precedent (a named property is
  data even when only one law uses it).  `tail-block` and
  `repeat-chain` are already multi-site (`check`+`derive`+`_shape`)
  or multi-package (decode_laws) atoms.
- `_rms_normalized_shape` and `_check_repeat_chain` stay importable
  from `laws.tensor` — both now delegate to the DSL, so the test and
  carrier call sites can never drift from the shipped conds.
- `_check_glu_fold` additionally stays a `def` (delegating to
  `eval_cond`): the discovery emitter's hook-name collision probe
  reads tensor.py *source* (`def\s+name\s*\(`), and an `as_check`
  alias would have silently dropped the emitted-dedup note.
