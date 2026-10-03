# Candidate law proposal — the open half of "the AI invents laws"

The rediscovery retro (`law-rediscovery.md`) closed the *verify*
half: a proposed equality is decided in milliseconds by putting both
sides in a fresh e-graph and saturating under the known laws.  It left
the *propose* half open — 28 of 51 library laws are primitive axioms,
the 23 "derivable" ones are 22 construction artifacts plus one genuine
composite, and nothing generated good candidates.

This retro closes — or at least *bounds* — the propose half.  It
builds four proposal strategies in `tools/law_proposal.py`, verifies
every candidate with `tools/law_verifier.py::verify_law`, and reports
**yield**: proposed / derivable / numerically-true / genuinely-new /
useful.

The headline: **candidate proposal is solvable, and one strategy
(schema enumeration) produces genuinely-new, *useful* laws the library
does not contain.**  But `verify_law` cannot confirm them — it is a
*derivability* oracle, not a truth oracle — so a second, numeric
oracle is required.  The retro's own "compose the laws" strategy is
confirmed to be a dead end for *value*: composites verify but never
pay.

Reproduce: `.venv/bin/python tools/law_proposal.py`
(~28 s, CPU-only, no network; `--json PATH` for machine-readable
output).  Output is deterministic across runs and `PYTHONHASHSEED`
values (the near-miss representative is chosen by `op_repr`, not by
the e-graph's hash-ordered member set).

## 1. What the tool measures

For every candidate ``lhs = rhs``:

* **derivable** — `verify_law(lhs, rhs, ALL_RULES)`: the library
  already proves it (a composite; the retro's `derivable` column);
* **num-true** — both sides agree on random fp64 tensors
  (`torch.allclose`, tol `1e-6`).  This is the truth proxy for a
  candidate `verify_law` *cannot* prove, because it is not a
  consequence of the library.  Shape mismatch counts as *false* (equal
  terms share a shape); an unbound op is *undecidable* (`None`);
* **genuinely new** — structurally not a tautology, duplicate, or
  inverse of any library rule (the retro's own line: leaf-abstracted
  `op_repr` keys, attrs canonicalised);
* **useful** — the candidate is *true* (derivable or num-true) **and**
  adding it to `ALL_RULES` and re-saturating a real term drops the
  extracted cost under the shipped `flops_cost` model.  A merely-true
  law that never changes a cost is *not* useful.

Usefulness is measured by saturating the candidate's own LHS instance
(a real, well-typed term) under `ALL_RULES`, then under
`ALL_RULES + {candidate}`, both bounded by the pipeline's
`EXPANSIVE`-rule budget (the Catalan closure generators are the reason
a small term can still reach thousands of e-nodes).  A *derivable*
candidate provably cannot lower the cost — its members are already
reachable — so its delta is always zero; that is exactly the
composite dead end below.

## 2. Strategy 1 — near-miss mining

Saturate each of 16 curated real terms (matmul chains, swiGLU, an
attention path, elementwise algebra, …) under `ALL_RULES`, take one
deterministic representative per e-class, and propose ``a = b``
whenever two representatives from *different* classes sit within a
tree distance of 1 — "almost equal", but no library rule closes the
gap (else they would share a class).

**Result: 16 proposed, 0 derivable, 0 numerically true, 16 new,
0 useful.**  Every candidate is a *false* equality.  The
representatives are structurally close because they share a shape
(`matmul(matmul(a,b),c)` vs `matmul(b,c)`), not because they are
almost-equal programs.  The numeric oracle rejects all 16; the
verifier (correctly) refuses all 16.  **Near-miss mining over a
saturated graph proposes false laws.**  This is the honest negative
for the strategy.

## 3. Strategy 2 — composition of existing laws

For each ordered pair of library laws ``(r1, r2)``, instantiate `r1`
on a real term and fire `r2` at every position of the rewritten form;
the candidate is ``(r1.lhs instance, r1.rhs with r2 applied)``.  Every
candidate is derivable by construction — the rediscovery retro's
"composites verify".

**Result: 26 proposed, 26 derivable, 26 numerically true, 26 new,
0 useful.**  The composites are reorderings and re-spellings
(`assoc_add+comm_add`, `silu_expand+comm_mul`, `sub_to_add+comm_add`,
…) that *verify* but leave the extracted cost untouched.  The retro's
suspicion is now measured, not assumed: **a composite of library laws
is never useful** — adding a derivable rule cannot shrink the set of
reachable members, so extraction over the e-class is unchanged.  The
retro's `silu_mul_form` (the one genuine emergent law) is exactly this
kind of object: true, new, and worth zero on the cost axis.

## 4. Strategy 3 — algebraic schema enumeration

The strategy the retro did not try.  Enumerate a small grammar of
algebraic identities over the op vocabulary — distributivity,
factoring, absorption, annihilators, inverse elements, involutions —
instantiated on concrete shapes.  The set deliberately mixes true
identities with false ones (so the numeric oracle is exercised) and
includes a library duplicate (so the structural classifier is
exercised).

**Result: 23 proposed, 2 derivable, 20 numerically true, 22 new,
11 useful & new.**

This is the decisive positive.  Eleven candidates are true,
structurally new, and *lower a program's cost* — laws the library
simply does not contain:

| law | lhs → rhs | cost |
|---|---|---|
| `mul_factor` | `x*y + x*z → x*(y+z)` | 48 → 32 |
| `mul_factor_right` | `y*x + z*x → (y+z)*x` | 48 → 32 |
| `neg_factor` | `-x + -y → -(x+y)` | 48 → 32 |
| `square_neg` | `(-x)² → x²` | 32 → 16 |
| `exp_factor` | `eˣ·eʸ → e^(x+y)` | 48 → 32 |
| `mul_zero` | `x*0 → 0` | 16 → 0 |
| `mul_zero_left` | `0*x → 0` | 16 → 0 |
| `pow_one` | `x¹ → x` | 32 → 0 |
| `sub_self` | `x - x → 0` | 16 → 0 |
| `add_inv` | `x + (-x) → 0` | 32 → 0 |
| `div_self` | `x / x → 1` | 64 → 0 |

`mul_factor`/`mul_factor_right` and `mul_zero`/`mul_zero_left` are
the same law up to `comm_mul`, so this is **9 distinct laws**, not 11
— an honest discount the raw table does not apply.

None is a duplicate or inverse of a library rule (the structural
classifier agrees: 22/23 new; the one non-new is the deliberate
`sub_to_add` duplicate), and `verify_law` proves none of them (they
are *new axioms*, not consequences).  The two that `verify_law` *does*
prove — `square_mul` and the duplicate — are the ones that change no
cost.

Note the split the retro asked for, *true* vs *useful*: 20 schemas are
numerically true, only 11 are useful.  `mul_distribute`, `neg_distribute`,
`exp_distribute`, `sub_add_factor`, `div_add`, `mul_neg`, `sigmoid_neg`
are all true and all worthless — they expand or hold the node count.
The three `FALSE_*` schemas are rejected by the numeric oracle (the
oracle is falsifiable, not a rubber stamp).

## 5. Strategy 4 — ranking a candidate pool (learned vs heuristic vs random)

Does a scorer rank *useful* candidates above random?  Three scorers
order the pool; the metric is **average precision** (robust to pool
composition).  The random baseline is `useful / pool`.

| pool | learned | heuristic (cost Δ) | random |
|---|---|---|---|
| full (65, 11 useful) | 0.31 | 0.28 | 0.17 |
| non-derivable (37, 11 useful) | 0.34 | 0.52 | 0.30 |

* The **cost-delta heuristic** (`flops(lhs) − flops(rhs)`) is the
  strongest signal on the non-derivable sub-pool (0.52 vs 0.30) but
  still far from perfect: a value scorer cannot tell a true
  cost-reducing law from a *false* cost-reducing one, which is exactly
  the near-miss population.
* The **learned rule-value net** (trained on single-rule deltas over
  the seeds, the stage-7 machinery) beats random only marginally
  (0.34 vs 0.30) — it does not transfer from library rules to novel
  candidates.
* Neither scorer reaches high AP: **usefulness requires a *soundness*
  judgment the score does not encode.**  Ranking helps, but the
  derivability/truth oracles remain load-bearing.

The full-pool vs non-derivable split matters: on the full pool the
scorers are partly rewarded for ranking derivable composites *low*,
which is a derivability question, not a value question.

## 6. The yield table

```
strategy      prop  deriv  ntrue   new  useful  u_new
-----------------------------------------------------
composite       26     26     26    26       0      0
near-miss       16      0      0    16       0      0
schema          23      2     20    22      11     11
```

`deriv` = `verify_law` proves it; `ntrue` = numeric oracle; `new` =
not a duplicate/inverse; `useful` = true **and** lowers a cost;
`u_new` = useful **and** new.

## Verdict

* **Proposal is not unsolved — it is *grammar-bound*.**  Schema
  enumeration found 9 distinct genuinely-new, useful laws (elementwise
  distributivity/factoring, the neg/exp homomorphisms, `square(-x)`,
  the annihilators, `x-x`, `x+(-x)`, `x/x`).  The library is missing
  ordinary algebra, and the machine can propose, verify, and *price*
  it.
* **But `verify_law` alone cannot adopt them.**  It proves
  *derivability*, and a genuinely-new law is by definition not
  derivable — so it reports "not derivable" for every useful
  candidate.  A second oracle (numeric truth on random tensors) is
  what confirms a new axiom.  Propose-then-verify is now closed on
  *both* halves, but the verify half is two oracles, not one.
* **The retro's own proposal idea — composition — is a dead end for
  value.**  All 26 composites verify, all 26 are new, none is useful:
  a derivable rule cannot shrink the reachable set.  "Composites
  verify" was true; "composites pay" is false.
* **Near-miss mining proposes false laws.**  Structurally-close
  e-class representatives are close because they share a *shape*, not
  because they are almost-equal programs; 0/16 are numerically true.
* **Ranking helps but does not solve proposal.**  A cost-delta
  heuristic beats random on the non-derivable pool (0.52 vs 0.30); a
  learned scorer does not transfer (0.34 vs 0.30).  Neither can judge
  soundness, so the truth oracle stays load-bearing.

## Honesty about what "the machine invented"

The found laws are **textbook algebraic identities**, not emergent
surprises, and the schema grammar is **human-authored**.  The
machine's contribution is the loop — enumerate from the grammar,
reject with `verify_law` where derivable, confirm with the numeric
oracle where new, and *price* with the cost model to separate the 20
true from the 11 useful.  That loop is real and it works; the
*creativity* is still in the grammar.  A machine that can only verify
laws it already has is the honest baseline — this tool is one step
past it (it can also *adopt* a law it can numerically confirm and
price), but the grammar is the ceiling.

## Caveats (stated, not tuned)

* **Conditional soundness.**  `div_self` (`x/x = 1`) needs `x ≠ 0`
  (`0/0 = nan`); `mul_zero`, `sub_self`, `add_inv` carry the usual
  `inf`/`nan` caveats.  These are *conditional* laws and would need a
  `check` hook before entering the library.  The unconditional,
  exactly-sound ones are `square_neg`, `pow_one`, and the
  real-arithmetic distributivity/homomorphism laws (fp-approximate,
  like the library's own `distribute_matmul_over_add`).
* **Cost model.**  Usefulness is measured with `flops_cost`; a
  different model (param bytes, launch-aware) could re-rank the
  "merely true" schemas.
* **Program.**  Usefulness is measured on the candidate's own LHS
  instance; a whole-model probe is the natural next step.
* **Schema grammar.**  Hand-authored, ~23 schemas; it is not
  exhaustive and it is the binding constraint on proposal quality.
* **`verify_law` scope.**  It decides derivability on tiny terms with
  a bounded saturation; the `EXPANSIVE` budget bounds the cost check,
  and a `stop == max_nodes` on a large term is reported, never read as
  "not useful".

## Gates

Run from the main worktree, HEAD plus `tools/law_proposal.py` only
(no `packages/` change):

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors; `tools/` is out of scope)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/python tools/radon_ratchet.py` — pass (1873 functions)
* `uv run pytest -q` — 3339 passed, 31 skipped
* coverage unchanged (no `packages/` file touched)
