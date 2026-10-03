# The law-discovery pipeline — census → propose → verify → measure → rank

The loop that produced `select_mul` (`law-shape-aware.md`) was a
sequence of one-off tools with a human deciding which experiment to
run next.  This retro records the loop made **runnable end to end** —
one entry point, `tools/law_pipeline.py` — and the **held-out
rediscovery** that validates it.

The headline is a **pass**:

* **Held-out rediscovery.**  With `select_mul` — and only it —
  removed from the search rule set (51 rules), the pipeline
  re-proposes it, verifies it with both oracles, measures it, and
  ranks it **#1 of 35**, shippable.  The winning candidate is
  **census-generated**: the pipeline's mechanical view-naturality
  generator emits it from the census's frequent `mul(select, select)`
  op-tuple, independently of the hand-written schema that first named
  it.
* **Run for real** (current library, no holdout): **no further
  candidates clear the bar.**  `select_mul` is now a library duplicate;
  every other firing candidate is false, a duplicate/inverse, or
  closure-unsafe.

Reproduce:

    .venv/bin/python tools/law_pipeline.py
    .venv/bin/python tools/law_pipeline.py --holdout select_mul

CPU-only, no network, ~25 s per run.  Output is stable (see
*Honesty*, §6).

## 1. The pipeline

`tools/law_pipeline.py` composes the existing tools — it adds the
composition, a closure-safety ratio, and the ranking; it does not
re-implement any stage.

| stage | what it does | reused from |
|---|---|---|
| **census** | op-tuple / shape frequencies of the real corpus | `law_shape_census.run_census` |
| **propose** | unify candidate equalities, de-dup by alpha-normal key | `law_shape_proposal.schemas`, `law_proposal.schema_candidates` (+ a census-driven view-naturality generator) |
| **verify** | *both* oracles: numeric truth **and** derivability | `law_proposal._numeric_true`, `law_verifier.verify_law` |
| **measure** | fires, cost delta, lowered-module `sink.verify` | `law_impact._probe` |
| | end-to-end reach, certificate replay, **closure ratio** | `law_impact._saturate` / `_cert_ok` (base set parameterized) |
| **rank** | one deterministic order + ship/no-ship verdict | new (the only new decision logic) |

Three proposal sources feed one pool:

1. **census-naturality** — for every frequent op-tuple `f(g, g)` (a
   pointwise *f* whose two operands are the *same* view op *g*), emit
   `f(g(u,A), g(v,A)) -> g(f(u,v), A)`, with *g*'s own attr keys as
   shared metavariables.  This is census → propose done by machine.
2. **shape-aware** — `law_shape_proposal.schemas` (the hand-written
   grammar from the shape-aware retro).
3. **algebraic-grammar** — `law_proposal.schema_candidates`,
   abstracted to patterns.

De-dup is by `law_proposal._key` (the alpha-normal equality), and a
duplicate's provenance is **merged** into `sources`, so a candidate
reachable from the census generator is recorded as such.

### The ship gate and the ranking

A candidate **ships** iff *all* of:

    truth      : derivable OR numerically true on a real instance
    new        : not a tautology / duplicate / inverse of the library
    fires > 0  : the rule fires on at least one real model
    paid > 0   : adding it strictly lowers a model's extracted cost
    verify_fail == 0 : the lowered before/after modules agree
    cert_fail == 0   : the extraction certificate still replays
    closure_ratio <= 2.0 : enodes(out)/enodes(in) stays bounded

The ranking key is, in order: *shippable*, then end-to-end cost drop,
paid-model count, firing count, truth, applicability, and (as a
tie-break) the smaller closure ratio.  The key was fixed **before**
the runs — nothing was tuned to make `select_mul` win.

## 2. Held-out rediscovery — the validation

`select_mul` is shipped (`catopt_core.laws.tensor.SELECT_MUL`, in
`SIMPLIFICATION_RULES`).  Remove it from the search rule set and
re-run:

    .venv/bin/python tools/law_pipeline.py --holdout select_mul

```
search rule set: 51 rules (held out: select_mul)
proposals: 35; firing on a real model: 4; shippable: 1

rank candidate             family             census true rel  match fires paid drop% cert  enode ship
   1 census:mul_select     census-naturality      25  yes new     25    24    5  25.9 pass 1.27x SHIP
   2 linear_factor         linear-control          4  yes dup      3     2    2   0.0 pass 1.00x   no
   3 reshape_transpose     layout                 31   no new     31    23    1   0.0 pass 1.39x   no
   4 grammar:mul_distribute algebraic-grammar      0  yes new     12    24    0   0.0 pass 43.6x   no

held out: select_mul — rediscovered as census:mul_select:
          rank 1 of 35, shippable=True
proposed by: census-naturality, shape-aware — census-generated=True
verdict: PASS — the pipeline ranks the known winner top
```

**Where `select_mul` lands: #1 of 35, shippable.**  Every gate is
green and the evidence is end-to-end:

* **census** — its LHS op-tuple `mul(select, select)` is the census's
  #5 op-tuple (25 sites, 6 terms); the *generator* is driven by that
  count, not by the name `select_mul`.
* **truth** — the numeric oracle confirms `mul(sel(u),sel(v)) =
  sel(mul(u,v))` on the real leaf shapes; it is **not** derivable from
  the 51-rule set (a genuine new primitive).
* **applicability** — 25 equality-enforced matches across the corpus.
* **fires / pays** — 24 firings on 5 models (SelectiveSSM,
  DiagDenseSSM, DiagonalSSM, HybridBlock, TwoLayerHybrid); the
  extracted cost drops **25.9 %**.
* **certificate** — replays (`cert=pass`) on every reach row.
* **closure** — `enode(out)/enode(in) = 1.27×`, well under the 2.0
  limit.

The three runners-up are rejected for three *different* reasons, which
is the point of the gate set:

* `linear_factor` fires and pays on 2 models but is a **duplicate** of
  the shipped `weight_factor_linear` — the framework rediscovers an
  existing law (the sanity check).
* `reshape_transpose` fires 23 times and lowers cost on 1 model, but
  the **numeric oracle says false** — the cost-only trap
  `law-shape-aware.md` §6 warned about.
* `grammar:mul_distribute` fires 24 times but **never pays**, and its
  **closure ratio is ~44×** — a search hazard the closure check flags.

## 3. Run for real — no further candidates

Without a holdout (`ALL_RULES`, 52 rules):

```
proposals: 35; firing on a real model: 4; shippable: 0

-- ship recommendations --
  none — no candidate clears the ship bar
```

The four firing candidates, and why each fails:

| candidate | census | true | rel | fires | paid | closure | reason |
|---|---|---|---|---|---|---|---|
| `census:mul_select` | 25 | yes | duplicate | 24 | 5 | 1.00× | not new (already shipped) |
| `linear_factor` | 4 | yes | duplicate | 2 | 2 | 1.00× | not new (shipped) |
| `reshape_transpose` | 31 | **no** | new | 23 | 1 | 1.39× | false (numeric oracle) |
| `grammar:mul_distribute` | 0 | yes | new | 24 | 0 | ~36× | fires but never pays (and closure-unsafe) |

The remaining 31 proposals are `inapplicable` (no equality-enforced
real match — the factorization and elementwise-algebra families),
`no firing on a real model`, or `duplicate` / `inverse`.  **Honest
answer: the pipeline finds nothing new worth shipping on the current
library and corpus.**

## 4. What the pipeline buys

* **One command, one verdict.**  The four earlier tools each answered
  a piece; the pipeline runs them in order and returns a ranked
  ship/no-ship decision with the evidence attached.
* **The closure-safety check is load-bearing.**  It is the signal the
  earlier tools lacked: `grammar:mul_distribute` fires *more* than
  `select_mul` (24 vs 24) yet is a 36–44× closure blow-up and pays
  nothing — a rule a fires-only or cost-only proposer might have
  ranked highly.  The ratio (enodes out / enodes in, from the impact
  tool's reach saturation) is now a first-class gate.
* **De-dup by equality, not by name.**  The census generator, the
  hand-written schema, and the algebraic grammar converge on the same
  alpha-normal key; the merged provenance records *every* source, so
  "was this rediscovered from the census?" is answered structurally.

## 5. The ranking, verbatim

    key = (shippable, cost_drop, paid, fires, truth, matches,
           relaxed, -closure_ratio)          # descending

`select_mul` is the unique candidate with `shippable=1`, so it is
rank 1 by the first component alone; the tie-breaks order the rest
(cost drop first, then paid models, then fires).  No component was
adjusted after seeing the result.

## 6. Honesty about what this does and does not show

* **The proposal grammar is not fully machine-invented.**  The
  pipeline's winning candidate is *census-generated* (the
  view-naturality generator reads the census's frequent `f(g,g)`
  op-tuples), which is stronger than re-measuring the hand-written
  schema — but the generator's op tables (`_POINTWISE`, `_VIEW_OPS`)
  are still human-authored.  The held-out test therefore validates
  the **judgment chain** (both oracles + measurement + ranking) and
  the **census → propose** step; it does not show that the *op
  vocabulary* is machine-invented.  The generator is also deliberately
  unsound-by-construction (e.g. it proposes a reshape naturality that
  is false in general) — the numeric oracle, not the generator, is
  what makes the pool trustworthy.
* **The exact closure ratio varies run to run.**  The e-graph's node
  sets iterate in hash order, which the per-run string-hash seed
  perturbs, so the *enode counts* of an expansive candidate
  (`grammar:mul_distribute`) wobble (36×–44×).  The **verdict** is
  stable: three independent held-out runs each ship exactly
  `census:mul_select` at rank 1.  The ratio is used as a bounded gate
  (≤ 2.0), not a precise score.
* **The gain is dispatch-count, under the pipeline's own model.**
  The 25.9 % drop is `executor_cost_for("generic")`'s per-dispatch
  term (the same caveat `law-shape-aware.md` §6 records), measured at
  the models' own dims with the pipeline's bounded saturation.
* **The corpus is 22 models + 61 bench cases.**  `select` appears
  because the SSM exporter unrolls the recurrence into per-step
  indexing; a `scan`-based exporter would not contain the shape.
* **`packages/` is untouched.**  The pipeline is a `tools/` report;
  the ship recommendation is reported, not made.

## 7. Recommendation (reported, not made)

* **No new law to ship.**  On the current library and corpus the only
  shippable candidate is `select_mul`, already in `SIMPLIFICATION_RULES`.
  The honest output is "no further candidates clear the bar."
* **Keep the pipeline as the standing gate.**  Any future proposed
  law should clear the same seven gates before adoption; the
  `--holdout <rule>` mode is the regression test that the gate still
  ranks a known winner top.
* **Documented next step.**  To close the last human gap, replace the
  `_POINTWISE` / `_VIEW_OPS` tables with a view/pointwise
  *classification* derived from the ops' own algebra (or from the
  census's shape statistics), so the generator's vocabulary is
  machine-invented too.

## Gates

Run from the main worktree, HEAD plus `tools/law_pipeline.py` (no
`packages/` change):

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass (126 files)
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/python tools/radon_ratchet.py` — pass (1879 functions)
* `uv run pytest -q` — pass (3364 passed, 31 skipped)
