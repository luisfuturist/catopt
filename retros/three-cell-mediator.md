# `silu_fold` shipped — the mediating 3-cell of the silu × swiglu critical pair

`law-coherence-catalogue.md` §4 measured the shipped library's only
coherence gap: two genuinely divergent critical pairs —

* `silu_expand × swiglu_fuse`
* `silu_mul_form × swiglu_fuse`

— both faces of the same overlap.  On the shared redex

    (mul (silu (linear x, A)), (linear x, B))

`silu_expand` at path `(0,)` (or `silu_mul_form` at the root) produces
the *expand-side reduct*

    (mul (mul (linear x, A), (sigmoid (linear x, A))), (linear x, B))

while `swiglu_fuse` at the root produces the *fuse-side reduct*

    (mul (silu (chunk (linear x, (concat A, B, dim=0)), c=2, d=-1, i=0)),
         (chunk (linear x, (concat A, B, dim=0)), c=2, d=-1, i=1))

and the two had **no common reduct** under the whole library: the
expansion destroys the `mul(silu ·, ·)` skeleton the fuse pattern
needs, and no rule rebuilt a `silu`.  Eqsat held both members anyway —
the gap mattered for *ordered* rewriting, where expanding first kills
the fuse path permanently.

This retro records **admission of the mediator**: `silu_fold`,
`mul(x, sigmoid(x)) → silu(x)` — the definitional inverse of
`silu_expand`, tagged `SIMPLIFICATION`, in `SIMPLIFICATION_RULES` and
therefore in `DEFAULT`.  With it the expand-side reduct folds its gate
back — `mul((g·σg), u) → mul(silu(g), u)` — restoring the fuse redex,
so both one-step reducts rejoin under the library.

## 1. Choosing the mediator

Three candidates were costed:

* **`silu_fold`** (shipped) — one three-node pattern, the exact
  structural inverse of an existing rule, mediates **both** pairs
  (and any future expand-kills-redex pair, not just swiglu).  Inverse
  pairs are kept deliberately in this library — eqsat needs both
  directions reachable (`pow_to_square`/`square_to_pow`,
  `naturality_scalar`/`_rev`, …), and this is the same 2-cell carried
  the other way.
* A **conditional fuse** matching the expanded spelling —
  `mul(mul(linear(x,A), σ(linear(x,A))), linear(x,B)) →` the
  chunk/concat RHS expanded.  Also correct (both reducts meet at the
  expanded-fused term), but it duplicates `swiglu_fuse`'s whole RHS
  machinery in a second rule, fixes only the swiglu pair, and adds a
  composite where a primitive exists.
* A genuine **un-fuse** — `mul(silu(chunk(z,2,d,0)), chunk(z,2,d,1))`
  back to the two-linear form.  This is the shape the catalogue's
  prose gestured at, but it cannot reach the expand-side reduct alone:
  un-fusing `linear(x, concat(A,B))` yields
  `linear(x, chunk(cat,2,0,i))`, and proving that equal to
  `linear(x, A)` needs a *second* mediating law
  (`chunk(concat(a,b,0),2,0,i) → arg_i`).  Two rules where one
  suffices, and the weight-splitting direction is the one the cost
  model never wants anyway.

Numeric truth was checked before building the `R(...)`
(`law_proposal._numeric_true`-style, fp64): `mul(x, σ(x)) == silu(x)`
is `True`; the false control `mul(x, σ(y))` is `False` — confirming
the shared `x` metavariable is the *load-bearing* precondition, not a
formality.  Like `select_mul`'s shared `dim`/`index`, the side
condition lives in the pattern; no `check`/`derive` hooks needed.

```python
SILU_FOLD = R(
    "silu_fold",
    Op.make("mul", "x", Op.make("sigmoid", "x")),
    Op.make("silu", "x"),
    law="The manual-silu fold: x · σ(x) = silu(x) — the definitional "
    "inverse of silu_expand, and the mediating 3-cell of the "
    "silu_expand/silu_mul_form × swiglu_fuse critical pairs.",
    tags=_SIM,
)
```

## 2. Confluence, measured

`.venv/bin/python tools/law_coherence.py` before → after:

| measure | before | after |
|---|---|---|
| shipped rules | 53 | 54 |
| co-firing pairs probed | 44 | 45 |
| pair-confluent | 37 | 38 |
| library-mediated | 5 | 7 |
| **divergent** | **2** | **0** |
| equivalence classes > 1 | 10 | 11 |
| inverse pairs | 13 | 14 |
| composite direct edges | 1 | 2 |
| primitive (basis) | 30 | 29 |

* Both critical pairs now report **lib-mediated** — the witness is
  `silu_fold` itself: the expand-side reduct folds its gate and the
  fuse fires.  (Under `{A, B}` alone they still cannot rejoin — the
  mediator is a third rule; `lib-mediated` is exactly the verdict the
  `pow_to_square`-mediated `square_expand × square_to_pow` pair
  already carries.  Instance-level local confluence under the whole
  library is the restored property.)
* A new inverse class `{silu_expand, silu_fold}`: each is now
  `derivable` with the other as its only essential premise — the
  catalogue's basis count correctly drops 30 → 29 (the pair is one
  basis element).
* A second **composite** direct edge appeared:
  `silu_fold ⇒ silu_mul_form` — the fold alone merges the mul-form
  instance's sides (the expanded member folds back).  Consequence:
  `silu_mul_form` lost its *essential* premise — it had exactly one
  proof (via `silu_expand`); it now has two independent ones.
* The only new co-firing pair the law adds is `silu_fold × comm_mul`,
  confluent: `mul(σx, x)` commutes to `mul(x, σx)` and folds.
* Under `--with-layout` (131 rules): **divergent: 0** as well.  The
  `silu_expand × transpose_push/pull_silu(_bare)` pairs report
  lib-mediated — checked against the fold-less universe: they were
  *already* lib-mediated, so this law fixed exactly the two catalogued
  divergences and no hidden ones.

## 3. Closure safety — measured, not argued

`ALL_RULES` vs `ALL_RULES \ {silu_fold}` on real exports (8 iters,
100 k enode cap):

    SwiGLU                  79 / 79 enodes, 31 / 31 classes
    ResidualMLP             16 / 16
    TransformerBlock      2211 / 2211  (both stop=max_iterations)
    GatedResidualBlock      16 / 16
    ManualSoftmaxAttention  18 / 18

**Zero enode delta everywhere, zero counted fires.**  Reason, not
luck: wherever a `mul(x, σ(x))` member exists it was *produced by*
`silu_expand` inside `silu(x)`'s e-class, so the fold's RHS is already
there — the union is a no-op (`merged=False`), and `rule_fires`
honestly does not count it.  The law only ever *adds* a member where
the manual spelling occurs without the kernel — at most one enode per
e-class.  That is also why it is a fixed-point-safe inverse pair:
both directions saturate to the same two-member class and stop.

`law_bench` (`--laws silu_fold --sizes 128,256`, CUDA): fired, RHS
member found, picked, verify pass — and it *pays* where it applies:

    silu_fold  128   cost 4.953e4 -> 2.329e4   0.0146ms -> 0.0102ms  1.431x
    silu_fold  256   cost 6.425e4 -> 2.918e4   0.0292ms -> 0.0112ms  2.617x

On a manual-`x·σ(x)` term the fold replaces two dispatches with one —
the same value class as `softmax_fold`: a corpus that spells the
kernel by hand gets the kernel back.

## 4. Admission surface

* `packages/.../laws/tensor.py` — `SILU_FOLD` + the 3-cell comment;
  added to `SIMPLIFICATION_RULES` (so `DEFAULT`, `ALL_RULES`,
  `ALL_RULES_WITH_LAYOUT`, `SIMPLIFICATION` preset pick it up
  automatically).
* `laws/__init__.py` — exported (`SILU_FOLD`, `__all__`).
* `bench/suites/correctness/law_bench.py` — `LAW_CASES["silu_fold"]`
  (`_one_var(mul(x, sigmoid(x)))`).
* `tests/test_contracts.py` — `_LAW_SPECS["silu_fold"] = {"x": (2,3)}`
  (fuzzer now covers it; `check`/`derive` are absent — nothing to
  satisfy).
* `tests/test_laws_structure.py` — counts 53→54, 15→16, 130→131,
  189→190; `SILU_FOLD` in `PUBLIC_NAMES`.
* `tools/law_coherence.py` — docstring/CLI universe counts.
* `tests/test_silu_fold_laws.py` (10 tests) — the `select_mul` /
  `softmax_fold` shape: registration/tags, match-instantiate
  round-trip, fp64 soundness (`allclose 1e-15` — the fused kernel
  rounds differently, not bitwise), fires + RHS-member, two declines
  (`mul(u, σ(v))` u≠v — matcher veto; sigmoid-free mul), **the
  critical pair itself** (fuse alone fires 0 on the expanded reduct;
  fold+fuse fires both, reducts share an e-class, certificate replays
  with `rules_used ⊆ {silu_fold, swiglu_fuse}`), end-to-end on a real
  `x·σ(x)` export (fires, cost drops, cert replays, `sink.verify`),
  default-pipeline fire + verified lower, and the inverse-pair
  fixed-point pin.

## 5. Honest limits

* **Mediation is library-level, not pairwise.**  Under `{A, B}` alone
  the two reducts still diverge — the probe's `lib-mediated` verdict
  is the correct reading: the 3-cell is a third 2-cell, which is the
  whole point.  Ordered rewriting over a *subset* that omits
  `silu_fold` still kills the fuse path; the law must ship (it does —
  `DEFAULT`).
* **One spelling order.**  The fold binds `mul(x, σ(x))`, the order
  `silu_expand` emits; `mul(σ(x), x)` folds only where `comm_mul`
  runs (it is in `ALL_RULES` but tagged `SYMMETRY`, hence out of
  `DEFAULT` — an exported graph spelling `σ(x)*x` inside the default
  pipeline will not fold).  The coherence pairs are unaffected — the
  expanded form is always the canonical order.
* **Instance-level probe.**  As §6 of the catalogue retro states:
  confluence is measured on one co-firing instance per pair, not a
  full critical-pair analysis.  Other overlap modes of these two laws
  were not enumerated — though the LHS skeletons leave little room:
  `swiglu_fuse` overlaps `silu_expand`/`silu_mul_form` only at the
  `silu`/`mul` spine this law restores.
* **No second divergence surfaced.**  Both universes re-probe clean;
  the retro does not claim global confluence, only that the
  catalogue's measured divergences are now mediated.

## 6. Gates

* `.venv/bin/ruff check packages tools` — pass; new test file clean
  under `ruff check`/`format --check` (file-level RUF002/3 noqa for
  math notation, per convention; the pre-existing `I001` drift in
  `test_contracts.py` is on HEAD, untouched).
* `.venv/bin/ruff format --check packages tools` — pass (136 files).
* `.venv/bin/ty check` — pass (0 errors).
* `.venv/bin/vulture` — exit 0.
* `.venv/bin/lint-imports` — 4 contracts kept.
* `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 findings.
* `.venv/bin/semgrep --config .semgrep.yml packages` — 0 findings.
* `.venv/bin/python tools/radon_ratchet.py` — ok (1948 functions).
* **Focused tests** (no full-suite run — 25 min serial on this
  machine): `tests/test_silu_fold_laws.py` standalone (10 passed),
  `test_laws_structure` + `test_rules` + `test_rulesets` +
  `test_contracts` (269 passed, 14 skipped), `test_property_laws` +
  `test_property_egraph`, and the four SwiGLU/silu tests of
  `test_torch_integration` — all green.  Coverage was not re-measured;
  the new rule's branches are exercised by the standalone file.

## Verdict

The catalogue named exactly one missing coherence; the library now
carries it.  `silu_fold` is the cheapest possible mediator — the
inverse of the rule whose expansion caused the split — and it earns
its place twice over: it *is* the 3-cell (both pairs rejoin, zero
divergent in both universes) and it is a paying law in its own right
(manual silu folds to one dispatch, ~1.4–2.6× on the bench case, zero
closure cost where it does not apply).
