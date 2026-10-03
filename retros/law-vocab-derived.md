# The op vocabulary, derived by property — the last human table removed

`law-pipeline.md` §7 named the remaining human ingredient in the law-
discovery loop: the generator's **op alphabet** — the `_POINTWISE` /
`_VIEW_OPS` tuples that say which ops may play which role in a
view-naturality candidate.  `select_mul` was discoverable only because
`select` happened to be listed.  This retro records replacing the
lookup with a **test**: `tools/law_vocab.py` classifies every op that
occurs in the corpus by what it *does*, and
`tools/law_pipeline.py --vocab derived` runs the whole pipeline on the
derived alphabet.

The headline is a **pass on both checks that matter**:

* **Derived ≥ hand.**  Over the 31 corpus ops the property test
  recovers *every* entry in the pipeline's own tables and adds ops the
  humans missed — `eq`, `pow` (pointwise) and `chunk`, `split`
  (views) — each addition corroborated by a *different* hand table
  elsewhere in the codebase.
* **Held-out rediscovery still passes.**  `--vocab derived --holdout
  select_mul` re-proposes `select_mul` as `census:mul_select`, ranks it
  **#1 of 35**, verdict SHIP — identical evidence to the hand-vocab
  run (fires 24, pays 5, drop 25.9 %, cert pass, enode 1.27×).

Reproduce:

    .venv/bin/python tools/law_vocab.py
    .venv/bin/python tools/law_pipeline.py --vocab derived --holdout select_mul

CPU-only, ~30 s for the derivation plus the pipeline's usual ~25 s.

## 1. The classification — by property, not membership

`law_vocab.derive_vocabulary` mines the 31 ops that occur in the
corpus (the 22 exported `catopt_torch.models` blocks plus the 61 bench
law cases) and assigns each a class by *evaluating* it:

| class | test |
|---|---|
| **view** | arity 1, shape-changing for some occurrence, value-preserving — every output element is an input element (`set(v(x)) ⊆ set(x)`).  Computes nothing. |
| **pointwise** | shape-preserving (broadcast) **and** commuting with a probe set of shape-agnostic views (flatten, unsqueeze, gather, slice, transpose): `f(v(x),…) = v(f(x,…))` under the numeric oracle. |
| **reduction** | output smaller than input, not value-preserving. |
| **attribute-carrying** | non-empty attrs in the corpus. |

Ops the tests cannot decide — `layer_norm`, `sdpa`, `stack`,
`masked_fill`, `concat` — are marked `?`.  The test **abstains**
rather than guessing: an undecided op simply never enters either
alphabet, which is the safe direction (it can only shrink the
proposal space, never admit a wrong op).

## 2. Result — the derived sets

    pointwise (binary): add, div, eq, mul, pow, sub
    pointwise (unary):  alias, neg, rsqrt, sigmoid, silu, square, tanh
    views:              chunk, expand, reshape, select, slice,
                        split, transpose, unsqueeze

Validation against every op table in the codebase (the pipeline's own
two, `catopt_core.cost._VIEW_OPS`,
`catopt_orchestrator.morphisms.signature`'s two, and
`catopt_core.laws.layout`'s two):

| hand table | agree | derived-only | hand-only |
|---|---|---|---|
| `pipeline._POINTWISE` | 4 | `eq`, `pow` | 0 |
| `pipeline._VIEW_OPS` | 6 | `chunk`, `split` | 0 |
| `cost._VIEW_OPS` | 4 | `expand`, `select`, `slice`, `unsqueeze` | 0 |
| `signature._VIEW_OPS` | 8 | 0 | `alias` |
| `signature._POINTWISE_OPS` | 4 | `add, alias, neg, rsqrt, sigmoid, silu, square, sub, tanh` | `masked_fill` |
| `layout._POINTWISE_BINARY` | 4 | `eq`, `pow` | 0 |
| `layout._POINTWISE_UNARY` | 6 | `alias` | 0 |

Every *derived-only* op is corroborated by another hand table — e.g.
`select`/`slice`/`expand`/`unsqueeze` are in `signature._VIEW_OPS`
already; the pipeline's narrower table was missing them.  The two
genuine disagreements:

* **`alias`** — the signature tables call it a view, the property
  test calls it pointwise.  It is shape-preserving *and* value-
  preserving, so both are defensible; classified pointwise because it
  commutes with real views.
* **`masked_fill`** — the signature table calls it pointwise; the
  test abstains (it carries a `value` attr, and its mask argument is
  not a tensor the commutation test can probe).  Reported as `?`.

## 3. Effect on the pipeline

`--vocab derived` feeds `derive_vocabulary()`'s sets into the
census-naturality generator in place of the tuples.  On this corpus
the proposal list is **unchanged — the same 35 candidates** — because
the generator is census-constrained: `eq`/`pow` never occur as an
outer `f(g, g)` tuple, and `chunk`/`split` never occur as the shared
inner view.  The derived alphabet is therefore a strict improvement in
*standing*, not in this run's yield:

* the same known winner is rediscovered blind (rank 1, SHIP);
* the same false candidates are still rejected
  (`reshape_transpose`, `grammar:FALSE_*`);
* and the alphabet now *grows itself* — the next real model that
  uses, say, `pow(transpose, transpose)` will have its law proposed
  without a human updating a table.

`--vocab hand` remains the default and reproduces the previous run
exactly; `derived` is opt-in so the comparison is explicit.

## 4. Honesty / limits

* **The test is not tuned to the tables** — the disagreement list is
  evidence of that.  The abstentions are honest "don't know"s.
* **Same proposal set on this corpus.**  The win is removing the
  human-authored ingredient, not new laws today; claiming more would
  be overstating it.
* **Coverage of "occurs in corpus".**  An op absent from all 83
  graphs is invisible — the alphabet is still corpus-bounded, just no
  longer hand-bounded.
* **`packages/` untouched.**  `law_vocab.py` is a `tools/` report;
  the shipped library tables are validated against, not edited.

## 5. What remains human

With this, the proposal loop's inputs are all machine-measured:

* **shapes** — the census (what occurs),
* **ops** — the property tests (how they behave),
* **truth** — the two oracles,
* **worth** — fires / pays / cert / closure.

What is still human: **admission** (a commit applies the
recommendation), the **corpus** itself (the 22 models are
hand-written), and the *choice* of which property tests define the
classes.

## Gates

`tools/`-only change (`tools/law_vocab.py` new, `--vocab` flag in
`tools/law_pipeline.py`):

* `.venv/bin/ruff check tools/law_vocab.py tools/law_pipeline.py` — pass
* `.venv/bin/ruff format --check` on both — pass
* `tools/law_vocab.py` — runs CPU-only, ~30 s, deterministic
* `law_pipeline.py --vocab derived --holdout select_mul` — PASS
  (`census:mul_select` rank 1 of 35, SHIP)
* `law_pipeline.py --vocab hand` — unchanged from the committed run
