# The numeric oracle's broadcast hole, closed

`law-meta-game.md` §3.2/§4 flagged a soundness hole in the law
pipeline's numeric-truth oracle and deferred the fix: "the shipped
`_allclose` accepts broadcastable rank mismatches — it certified
`select(X) = unsqueeze(select(X))`, a rank-changing 'equality'.  Worth
a line in the oracle's file eventually".  This retro records the fix
and the regression audit it required.

## 1. The hole

`tools/law_proposal.py::_allclose` compared two evaluated sides with
`torch.allclose(a.to(float64), b.to(float64))` and nothing else.
`torch.allclose` broadcasts its operands, so `(n,)` vs `(1, n)` — and
`(d, d)` vs `()` — compare equal whenever the values broadcast-agree.
The oracle is the referee for every proposed law that `verify_law`
cannot derive: a broadcast acceptance means a candidate can score
"true" while the two sides have **different ranks** — different types.
The meta-game found it in the wild: `select(V,d,i) ->
unsqueeze(select(V,d,i))` scored `true` and fired 8 times before the
`paid` gate kept it off the ship list.

The helper is shared, so the hole was wider than one call site:

* `law_proposal._numeric_true` — the pipeline's truth oracle
  (`law_pipeline.measure`, and `law_meta_game.Referee.evaluate`).
* `law_vocab._commutes` — the pointwise naturality probe that derives
  the `--vocab derived` op alphabet (`f(v(x)) = v(f(x))`).

`law_verifier.verify_law` is unaffected — it is pure e-graph
derivability, no tensor comparison.  `law_wallclock`'s
`torch.allclose` compares a module against itself (same shape by
construction); unrelated.

## 2. The fix

One check in `tools/law_proposal.py::_allclose`: exact `torch.Size`
equality **before** the elementwise comparison.  Two outputs are equal
only when their shapes are identical *and* the values agree within
tolerance.  The tuple (multi-output) case recurses through the same
predicate, so a carrier tuple is strict per element.

What was deliberately *not* changed:

* **Dtype promotion stays.**  Both sides are still cast to fp64 before
  comparing — a `Const` leaf lowers to an integer tensor, and a dtype
  equality check would re-break `x*0 = 0`-style comparisons (the
  promotion was itself a fix; see the docstring).  Structural dtype
  discipline is already enforced: a tensor is never equal to a tuple,
  and tuples must match length elementwise.
* **Tolerance stays `1e-6` fp64** for same-shape comparisons — the
  kernel-vs-manual softmax case needs the slack.  The fix tightens the
  *shape* precondition, never the numeric one.

## 3. Regression audit — did any past verdict change?

Re-ran `tools/law_pipeline.py --vocab derived` before and after the
fix (same tree, `--json`, all 50 candidates) and diffed the `true`
column per candidate name.  **Three flips, all `True → False`, all the
same hole class:**

| candidate | equality | old | new |
|---|---|---|---|
| `grammar:div_self` | `div(x, x) -> Const(1)` | true | **false** |
| `grammar:mul_zero_left` | `mul(Const(0), x) -> Const(0)` | true | **false** |
| `grammar:add_inv` | `add(x, neg(x)) -> Const(0)` | true | **false** |

Each is a rank-collapse to a scalar `Const`: the LHS evaluates to a
`(4, 4)` tensor, the RHS to a `()` scalar.  They broadcast-agree, so
the old oracle said true — but as rewrites they are type-incorrect (a
`()`-typed term cannot replace a `(d, d)`-typed hole), exactly the
class the meta-game flagged.  The flips are the fix working, not
regressions.

**No verdict changed.**  No candidate flipped `derivable` or
`shippable`; the ship list was empty before and after, and the flipped
three only gained an honest `false (numeric oracle rejects)` no-ship
reason.  The two known-good shapes are unaffected:

* `census:mul_select` (the shipped `select_mul` shape) — still
  `num_true=yes`; a real same-shape naturality.
* `recognize:softmax` (the softmax fold) — still `num_true=yes`; both
  sides are the same shape and the fp64 slack was untouched.

The proposal set itself is unchanged (same 50 candidates) — the
stricter `_allclose` did not alter the derived vocabulary's
pointwise/view classification on this corpus.

## 4. Reproducer

The meta-game's exact case through `_numeric_true`:

    select(u, dim=0, index=0) = unsqueeze(select(u, dim=0, index=0), dim=0)
       (4,)                         (1, 4)
        -> was True; now False

Verified directly as well: `(n,)` vs `(1, n)` → `False`, `(4, 4)`
zeros vs scalar `0` → `False`, same-shape int64-vs-fp64 → `True`
(dtype promotion intact), tuple-vs-tensor and tuple-rank-mismatch →
`False`, and a real same-shape identity (`x*y + x*z = x*(y+z)`) →
`True`.

## 5. Honesty / limits

* **The audit is same-tree, not committed-run.**  The corpus has grown
  since `law-pipeline.md` / `law-vocab-derived.md` were committed
  (30 models vs 22, 50 proposals vs 35), so the committed table cannot
  be diffed literally; the before/after `--json` diff on this tree is
  the honest comparison.  The committed headline results — the
  `select_mul` and `recognize:softmax` truth verdicts — re-verify.
* **The same hole remains in `packages/`**, deliberately untouched
  here: `catopt_torch.meta_eval._eval_allclose` also calls bare
  `torch.allclose`.  It feeds `catopt_core.meta.synthesize_rules`,
  not the law pipeline; worth the same one-line fix as a follow-up.
* **Single-instance caveat stands.**  `_numeric_true` still checks one
  matched instance (per `law-meta-game.md` §4); the fix makes each
  check type-faithful, not exhaustive.
* **Broadcast-true laws are now unreachable** through the oracle —
  e.g. `mul(x, Const(0)) -> Const(0)` is correctly rejected.  A
  spelling *with* the matching view on the RHS
  (`-> expand(Const(0), …)`) still compares same-shape.  That is the
  intended semantics: `TensorType` carries rank, so rank-changing
  "equalities" are not equalities.

## Gates

`tools/`-only change (`tools/law_proposal.py::_allclose` + docstring);
no `packages/` edit:

* `.venv/bin/ruff check tools/law_proposal.py` — pass
* `.venv/bin/ruff format --check tools/law_proposal.py` — pass
* `tools/law_pipeline.py --vocab derived` before/after — three
  `true→false` flips, all rank-collapse candidates (§3); no
  `derivable`/`shippable` verdict changed
* reproducer `select(u) = unsqueeze(select(u))` — `True → False`
* per the task bound, full pytest/coverage not run
