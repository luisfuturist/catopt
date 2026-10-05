# Retro: the cond bank — widening the view-commute vocabulary

Date: 2026-10-06
Context: follow-up to `auto-cond.md`.  The auto-cond constructor
(`object_synthesis.auto_cond_object`) measures a conditional
candidate's bare domain, enumerates the cond DSL's vocabulary over the
pattern's metavariables, and mints the smallest conjunction covering
every measured `equal` site and declining every `bad` one.  The first
measurement left **15 of 37** conditional candidates with *no
declarable conjunction ≤ 3 clauses* — the constructor's honest refusal
when the bank has no predicate separating the measured classes.  This
change widens the bank where the measured gaps are: four new
view-output shape specs and two broadcast-alignment atoms in
`catopt_core.laws.cond`, plus the *commutation* predicate form the
`f(view(u), v) -> view(f(u, v))` "wrap" candidates need.  Nothing else
moved: `laws/tensor.py`, `evidence.py`, `meta_game.py` untouched.

## 1. Reproducing the refusals

The measurement is the retro's own: `default_gauntlet_corpus()` (276
terms / 158 probe cases), the static proposal pool
(`meta_game.generator_pools(census_op, real_terms, "derived")`), then
`auto_cond_object(rule, corpus_terms=corpus.real_terms,
synth_limit=360)` per candidate.  A candidate is *conditional* when
the bare pattern measures `equal > 0 and bad > 0`; a *refusal* is
`object is None` with `"no declarable conjunction"` in `detail`.

Two honest measurement caveats, both pre-existing:

- **The pool tracks the corpus.**  The auto-cond retro's census was 66
  candidates / 37 conditional / 15 no-cover refusals over the
  276-term corpus.  A sibling commit grew the corpus to 327 terms
  (`corpus: +51 workloads`), so a fresh build measures 70 / 41 / 19 —
  the extra `_r_w` mixed-view twins appear with the added workloads.
  The pool is deterministic for a fixed corpus; the *counts* move with
  it, the *classes* do not.  This retro reports both the retro's named
  15 and a whole-pool delta.
- **`equal`/`bad` splits are value-sensitive.**  `_stable_outcome`
  draws leaves from ambient `torch.randn`, so a borderline site can
  flip; the admit/refuse *verdicts* below were reproduced across runs.

## 2. The refusal classes

The 15 named refusals decompose into three structural classes, all
variants of "a view op must commute with a broadcast":

| class | shape of the candidate | count |
|---|---|---|
| **wrap** `f(g(u), v) -> g(f(u, v))` | one operand viewed, pushed through | 8 (`mixed:*_w`) |
| **pair** `f(g(a), g(b)) -> g(f(a, b))` | both operands viewed | 4 (`census:mul_select`, `select_add`, `select_sub`, `slice_mul`) |
| **view-view** `g1(g2(a)) -> g2(g1(a))` | nested views | 1 (`reshape_transpose`) |

The measured domain for every one of them is separated by the same
fact — **does the view commute with the broadcast?** — which the bank
could not name, because:

1. **No output-shape spec for four view ops.**  `_view_specs` builds
   specs only for `unsqueeze` / `reshape` / `getitem`; `select`,
   `slice`, `chunk`, `transpose` had no `*-out` spec, so no predicate
   could read `g(u)`'s shape at all.  Several wrap candidates reported
   **0 covering predicates** — not even one bank clause accepted the
   equal class.
2. **No commutation predicate.**  The bank emits the *strip* form
   `bcast(g(u), v) == bcast(u, v)` (`bcast-eq(s, m, u, m)`), never the
   *wrap* form `bcast(g(u), v) == g(bcast(u, v))` — the RHS view of the
   broadcast needs a spec over a `bcast` operand, which the missing
   specs could not express.
3. **No broadcast-alignment atoms.**  Even with the shapes named, the
   shape equation alone is insufficient: `select` at axis `d` commutes
   with a broadcast only when `d` lands on the *same grid axis* in both
   operands, and the pushed-through partner must be **constant along
   that axis** — the value-level conditions behind the surviving
   counterexamples (e.g. `select(A=(4,), 0)` vs `select(B=(3,4), 0)`
   match in shape but not in values).

## 3. The DSL additions (`packages/catopt-core/src/catopt_core/laws/cond.py`)

All pure data — JSON-serializable tuple trees, no lambdas.  Every new
atom mirrors the corresponding `_infer_op_shape` branch and declines
(`None` / `False`) whenever the fact is unprovable.

**Shape specs** (accepted anywhere a spec is read, and composable —
`("select-out", ("bcast", "U", "V"), "D")` resolves):

| spec | meaning |
|---|---|
| `("select-out", T, D)` | `select` output: drop the axis `$attr:D` names |
| `("slice-out", T, D, S, E, STEP)` | `slice` output: axis `$attr:D` replaced by `ceil((min(end, dim) - start)/step)`; `None` args take the torch defaults `0`/dim/`1` |
| `("chunk-out", T, C, D)` | `chunk` element: `s[$attr:D] // $attr:C` |
| `("transpose-out", T, D0, D1)` | `transpose` output: swap axes `$attr:D0`/`$attr:D1` (defaults `-2`/`-1`) |

**Predicates** (registered in `_OPS`):

| predicate | meaning |
|---|---|
| `("axis-align-eq", A, B, K)` | the `$attr:K` axis sits at the same right-aligned position in both shapes (`d % rank - rank` agrees) — the index view commutes with the broadcast |
| `("bcast-dim-inv", V, U, K)` | `V` is broadcast-invariant along `U`'s `$attr:K` axis (no such axis, or its aligned extent is exactly `1`) |

**No other atom was needed** — the commutation guard composes the
existing `bcast-eq` / `shape-eq` with the new specs, and
"broadcastable" composes as `("shaped", ("bcast", A, B))` (the missing
`bcast-ok`; `bcast-into` is one-directional and does not cover it).
Two bank-level compositions the generator emits:
`("or", ("axes-noop", V, D0, D1), ("rank", V, "<=", 1))` for the
transpose partner, and `("and", ("axis-align-eq", A, B, K), ...)` for
the pair form.

## 4. The admit delta

Measured over the retro's named 15 (of which 14 remain in the grown
pool — `mixed:mul_unsqueeze_reshape_wr` is no longer enumerated),
**11 now mint a guard**; the 3 that still refuse are the slice family
(`mixed:mul_slice_l_w`, `slice_mul`) and `reshape_transpose`.  Over the
current whole pool (70 candidates / 41 conditional / 19 stock
no-cover refusals), **14 of the 19 now admit**:

| candidate | minted guard |
|---|---|
| `census:sub_unsqueeze` | `bcast-eq(unsq-out(U), unsq-out(V), unsq-out(bcast(U,V)), ·)` |
| `census:mul_select`, `select_add`, `select_sub` | `axis-align-eq(U,V,D) ∧ bcast-eq(select-out(U,D), select-out(V,D), select-out(bcast(U,V),D), ·)` |
| `mixed:mul_select_l_w`, `mixed:add_select_r_w` | `bcast-dim-inv(V,U,A_dim) ∧ bcast-eq(select-out(U,A_dim), V, select-out(bcast(U,V),A_dim), ·)` |
| `mixed:eq_getitem_l_w`, `mixed:add_getitem_l_w` | `bcast-eq(getitem-out(U), V, getitem-out(bcast(U,V)), ·) ∧ leaf(U)` |
| `mixed:mul_reshape_l_w` | `bcast-eq(reshape-out(U,A_shape), V, reshape-out(bcast(U,V),A_shape), ·)` |
| `mixed:mul_chunk_l_w` | `bcast-eq(chunk-out(U,A_chunks,A_dim), V, chunk-out(bcast(U,V),…), ·)` |
| `mixed:mul_transpose_l_w`, `mixed:add_transpose_l_w` | `bcast-eq(transpose-out(U,…), V, transpose-out(bcast(U,V),…), ·) ∧ or(axes-noop(V,…), rank(V)<=1)` |

Every minted guard was verified on its own sweep: the guarded region
has `equal > 0`, `unequal == 0`, `rhs_err == 0`.  Each is pure data
(`missing_hooks == ()`); the `bcast-eq` clause is the shape half and
the `axis-align-eq` / `bcast-dim-inv` / `axes-noop` clause is the
value-level half.

### What still refuses, and why

- **The slice family** (`mixed:mul_slice_l_w`, `slice_mul`).  A
  *partial* slice pushes through a broadcast only when the slice
  commutes at the flat-read level: the sliced extent must align, and
  the partner must be constant along the sliced axis *or* the slice
  must be full-extent.  The shape specs and `bcast-dim-inv` get the
  full-extent and invariant cases but not the partial-extent alignment
  — that is the `flat-pair-unsq`-style flat-index map the existing
  machinery implements only for `unsqueeze`/`reshape`.  Declaring a
  slice wrap honestly needs a flat-read predicate per index view.
- **`reshape_transpose`.**  The nested view-view shape equation
  (`shape-eq(transpose-out(reshape-out(A,S),…), reshape-out(
  transpose-out(A,…),S))`) covers all equal sites and kills 108/119 bad
  ones, but the residue — `S` flattening `A` so the reshape reorders
  relative to the transpose — needs a flat-layout condition, the same
  gap as the slice family.  No ≤ 3-clause cover exists in the current
  bank.
- **`recognize:softmax`** refuses `no declarable conjunction` at the
  `auto_cond_object` level, but it is *not* a guard gap: it stands
  down at `full-data` with `missing_hooks == ["check", "derive"]` — a
  derivation gap, not a condition gap (unchanged from the auto-cond
  retro).

## 5. Production wiring — an honest gap

`object_synthesis._pred_bank` is the only place that enumerates the
bank, and it is out of this change's write scope.  The new specs and
atoms live in `cond.py`; the *generator* that emits the commutation
predicates for a pattern lives in `tests/test_discovery_autocond.py`
(`_view_commute_preds` + `_install_view_commute_bank`), which
monkeypatches `obs._pred_bank` so the delta is measurable.  **The
production bank needs the same additive emission** — extend
`object_synthesis._view_specs` with the four view ops and add the
commutation emissions to `_spec_preds` / a new `_view_commute_preds`.
Until then the minted guards exist as verified data but
`auto_cond_object` in production still refuses these candidates.  This
is the one follow-up the change does not close.

## 6. The load-bearing tests

- `tests/test_cond_laws.py` — per-predicate: `select-out` / `slice-out`
  (including the `ceil`/`step` arithmetic and the `None` defaults) /
  `chunk-out` / `transpose-out` (including the `-2`/`-1` defaults),
  `axis-align-eq`, `bcast-dim-inv`; the composed wrap guard with
  canonical accept/decline envs; the `shaped(bcast)` composite; and
  the JSON round-trip of the composed guard.
- `tests/test_discovery_autocond.py` — the generator names the
  select-wrap commutation predicate, and the stock bank refuses the
  select wrap while the widened bank mints a guard whose every
  accepted site is `equal`.
- `tests/test_law_serialize.py` / `tests/test_cond_laws.py` counts: the
  cond-carrying-rule pins (`seen == 36`, `len(migrated) == 36`, the 61
  full-data census) **did not move** — this change adds vocabulary, not
  laws.  (The task anticipated they would move; they would only if new
  shipped laws used the new atoms, which this change does not add.)

Verification: `pytest tests/test_cond_laws.py tests/test_derive_laws.py
tests/test_law_serialize.py tests/test_discovery_autocond.py -q` → 120
passed.  `ruff check` / `ruff format --check` clean on `cond.py`;
`ty check` 0 errors; `radon_ratchet` ok (3175 functions); `vulture`
and `lint-imports` clean.

## 7. Honest limits

- **Measured, not proven.**  Each minted guard is exact on the measured
  domain (the two capped enumeration windows plus the real matches); a
  counterexample outside both windows stands until a wider sweep
  reaches it — the same blind spot every guarded law carries.
- **Two classes remain unnameable** without a flat-read predicate
  (slice, reshape-transpose).  The atoms added here are the *shape*
  vocabulary; the flat-index vocabulary is the next increment.
- **`bcast-dim-inv` presumes right-aligned broadcast**, the same
  convention `_broadcast` and `reshape` share (row-major flat order).
- **The pool tracks the corpus**, so the counts (15 → 11 on the retro's
  list; 19 → 14 on the grown pool) are measured against a fixed corpus
  and reported honestly; the *classes* are structural.
