# Condition DSL — side conditions as data, the lemma-store seam

A rewrite law is a 2-cell: a term pair plus a side condition.  Every
part of that was already data — lhs/rhs patterns are `Op` trees that
`rulecache` round-trips through JSON — except the condition, which
was an opaque Python callable (`check=lambda bound: ...`).  That one
field kept a law from being a serializable record: you cannot ship a
`lambda` through a store, cannot diff two conditions, cannot ask
"which laws mention rank" without running them.

This change adds `catopt_core.laws.cond`, a small declarative DSL for
the side condition, and migrates **29 of the 54 shipped tensor laws**
onto it.  A `cond` is a nested tuple tree (lists after `json.loads`)
interpreted against the same `bound` environment `check` sees —
`{metavar: resolved_term, "$attr:NAME": value}` — by `eval_cond`, a
total predicate that never calls back into user code.

Reproduce:

    .venv/bin/python -m pytest tests/test_cond_laws.py -q
    .venv/bin/python -c "
    import json
    from catopt_core.laws.tensor import FACTOR_MUL
    from catopt_core.laws.cond import cond_to_data
    print(json.dumps(cond_to_data(FACTOR_MUL.cond)))"

## 1. The mechanism

* `Rewrite.cond: Any = None` (`egraph/types.py`) — the declarative
  condition, stored as data.  `__post_init__` does two things:
  *canonicalises* (`cond_from_data` — list trees become tuple trees,
  so a store-loaded cond hashes and compares equal to its
  source-spelled twin) and *folds* — `compile_guard(cond, check)`
  becomes the rule's `check`, evaluating `cond` first then any
  procedural remainder.  Every evaluation site — `apply_rule`,
  certificate replay, term-level matching, meta's composite guards —
  stays on the single `rule.check` convention.  Nothing downstream
  changed.
* `laws.base.R(..., cond=...)` forwards it.  `check` and `cond` may
  coexist and conjoin; `derive` deliberately stays Python — it
  *computes* RHS attributes, a different role from the condition's
  verdict.
* `laws.cond` — the interpreter (`eval_cond`), the fold
  (`compile_guard`), the `check`-shaped view (`as_check` — a
  `functools.partial(eval_cond, cond)`, which
  `rulecache._hook_sig` fingerprints explicitly), and the JSON
  canonical forms (`cond_to_data` / `cond_from_data`).
* `rulecache.ruleset_fingerprint` now hashes `cond_to_data(r.cond)`
  alongside the hook signatures.  That one matters: every
  cond-carrying rule shares *one* `compile_guard.<locals>.guard`
  closure — identical module, qualname and source — so the hook
  signature alone cannot see a `cond` edit.  A data change must
  invalidate the synthesis cache; now it does.

## 2. The grammar

    cond := bool
          | ("and", c1, ...) | ("or", c1, ...) | ("not", c) | PRED

    PRED := ("shaped", T) | ("concrete", T) | ("scalar", T)
          | ("uniform", T) | ("ones-but-last", T)
          | ("rank", T, CMP, k) | ("rank-eq", A, B)
          | ("shape-eq", A, B) | ("shape-compat", A, B)
          | ("dim-eq", A, i, B, j) | ("dim-compat", A, i, B, j)
          | ("dim-eq-const", T, i, k) | ("bcast-into", T, U)
          | ("mm-shape-ok", A, B)
          | ("axes-last2", T, D0, D1) | ("axes-distinct", T, D0, D1)
          | ("axes-eq", T, A0, A1, B0, B1) | ("axis", T, NAME, k)
          | ("op-in", T, ops) | ("leaf", T) | ("const", T)
          | ("term-eq", A, B)
          | ("const-num", T) | ("const-cmp", T, CMP, v)
          | ("attr-is", NAME, v) | ("attr-eq", NAME, v)
          | ("attr-in", NAME, vs) | ("attr-type", NAME, K)
          | ("attr-len", NAME, CMP, k)

`T`/`A`/`B` are *shape specs*: a metavar name (resolved through
`_shape_of` on the bound term) or `("mm-out", T, T)` — the matmul
output shape of two nested specs.  `CMP` is `== != < <= > >=`.

The strictness contract is the same posture the hand-written checks
took: **anything a predicate cannot prove declines** — unknown or
provably-ill-typed (`_INVALID`) shapes, missing attrs, non-int dims —
with `None`-dims as wildcards only where an op says `*-compat` /
`dim-eq`.  A migrated law keeps its decline behaviour verbatim.
Malformed nodes raise `ValueError`: a malformed condition is a bug,
not a decline.

## 3. The census — what migrated, what stayed Python

**29 / 54** `ALL_RULES` carry `cond=` — 17 static sites plus the 12
rules `_make_sdpa_fold_rules` generates (3 scale spellings × 2
dropout-wrap variants × 2 mask kinds), whose six check functions
became six cond constants composed off `_COND_SDPA_BASE`:

* **rank/shape guards** — `distribute/factor_matmul`,
  `weight_{factor,distribute}{,_linear}` (the mixed-rank addend
  unsoundness guard), `swiglu_fuse` / `parallel_mul_fuse`
  (`shape-compat` — the fuse-pair shape equality),
  `assoc_linear_bias{,_rev}` (the `B·b1 + b2` compose chain: seven
  clauses of rank + dim-eq),
  `linear_{channel,row}_scale{,_rev}` and `naturality_scalar{,_rev}`
  (the diagonal/scale broadcast predicates);
* **attr guards** — `softmax_fold` (`keepdim is True` ∧ single
  reduce axis), the whole `sdpa_fold_*` family (score-transpose
  axes-last-two, softmax last-axis, `Const`-numeric scale,
  masked-fill `F < -1e30`).

**What stays `check=` — one rule.**  `gqa_absorb_repeat` walks a
nested `transpose(reshape(expand(unsqueeze(k))))` repeat-chain
*structurally*, then indexes bound attr tuples and does head-count
arithmetic — genuinely procedural.  `derive=` hooks stay Python
everywhere for the same reason: they compute, they do not judge.
The compat aliases keep test code honest: `_check_sum_keepdim`,
`_check_mm_rhs_*`, `_check_fuse_pair`, `_check_linear_bias_compose`,
`_check_sdpa_*`, `_check_score_transpose`, `_check_softmax_dim` are
now `as_check(COND)` — the *same data* the rule carries, viewed
through the `check` calling convention, so the helper and the rule
can never drift.

**Out of scope:** the `check=` callables in `laws/attention.py`
(17), `laws/layout.py` (11) and `laws/scan.py` (6).  Most of
layout's are already expressible (`axes-*` was built for them);
attention's uniform/bounds predicates are the next audit.

## 4. Migration audit — verdicts preserved

Every migrated `check` was re-derived as data and compared clause by
clause; the decline tests (`test_matmul_factor_laws`,
`test_softmax_laws`, `test_laws_rewrite_edges`, …) pass unchanged.
Two corners tightened *deliberately*, both consistent with the
codebase's existing posture:

* `axes-last2` normalises through `typing._axis_pair`, which rejects
  `|dim| >= rank` — where the old `%`-arithmetic check could fire on
  a malformed transpose (`dim=6` on a rank-4 `K` folded onto the
  last-two pair).  Such a term is ill-typed anyway — the stricter
  decline matches what `laws.layout` already does.
* the rank-0 score-transpose case that `ZeroDivisionError`ed inside
  the old check (a raise *is* a reject through the matcher) is now a
  plain decline — same verdict, no exception.

## 5. The round-trip, demonstrated

```python
blob = json.dumps(cond_to_data(FACTOR_MUL.cond))
rule2 = Rewrite(name="factor_matmul_rt",
                lhs=FACTOR_MUL.lhs, rhs=FACTOR_MUL.rhs,
                cond=json.loads(blob))   # list tree: canonicalised
# rule2.cond == FACTOR_MUL.cond ; saturating with either reaches
# the same merged member — and both decline the rank-1
# counterexample the check was written for.
```

`tests/test_cond_laws.py` runs exactly this: all 29 shipped conds
survive `cond_to_data` → JSON → `cond_from_data`, a rebuilt rule
fires identically in a real `EGraph`, and two rules differing only
in `cond` data fingerprint differently.

## 6. The lemma-store path

A law record is now fully serialisable: `{name, lhs, rhs, cond,
derivation}` — `rulecache._enc_term` covers the patterns,
`cond_to_data` the side condition.  What a store still cannot carry
is the *procedural* remainder (`gqa_absorb_repeat`'s `check`, every
`derive`) and the runtime-minted rules' composed guards (already
handled by the `guard_pats` re-expression spec).  The shipped-library
seam is closed: conditions are inspectable, diffable data — a law
tool can enumerate guard clauses, a store can ship them, and the
fingerprint proves a data edit reaches the cache key.
