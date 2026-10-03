# Shape-aware law proposal — propose against the shapes real models contain

The proposal retro (`law-proposal.md`) enumerated an *algebraic*
grammar and found 11 true, cost-reducing laws; the impact retro
(`law-impact.md`) then measured them on real graphs and got **zero
firings** — the ops were present, the *shapes* were not.  Proposal was
decoupled from reality: it optimised algebraic novelty, not
applicability.

This retro closes that loop.  It mines the shape distribution of the
real corpus (`tools/law_shape_census.py`), then drives a proposer
(`tools/law_shape_proposal.py`) from the *frequent shapes* and asks the
only question that matters: **is there a true, cost-reducing equality
whose LHS is (or contains) that shape?**

The headline is a decisive **positive** where the earlier work was a
decisive negative.  The census shows the corpus's most common algebraic
shape is `mul(select, select)` (24 sites, the SSM recurrence
`B_t ⊙ x_t`), and the naturality law

    mul(select_k(u), select_k(v)) -> select_k(u * v)      ("select_mul")

is **true** (every one of the 24 real sites), **new** (no library
rule), **fires** on 5 real models (24 firings) and **pays** end-to-end:
the pipeline's own cost model drops the extracted cost 16–26 %, the
lowered before/after modules agree under `sink.verify`, and the
certificate still replays.  The 11 earlier laws fired **0** times on
the same corpus.

The complementary negative also holds, now per-schema: the shape the
earlier laws were aimed at — `add(mul, mul)` — *is* frequent (21
sites), but **never carries a shared factor** (0 of 21), so factoring
is inapplicable.  Shape-awareness finds a law exactly where the shape's
*precondition* is satisfiable, and finds nothing where it is not.

Reproduce:

    .venv/bin/python tools/law_shape_census.py --json /tmp/census.json
    .venv/bin/python tools/law_shape_proposal.py --json /tmp/shape.json

CPU-only, no network, a few minutes.  Output is deterministic.

## 1. The shape census (the deliverable that outlives the experiment)

`tools/law_shape_census.py` reads the same corpus the impact tool
probes — the 60 bench law cases plus the 22 exported
`catopt_torch.models` blocks — and reports three tables.  Corpus: **82
terms, 610 op nodes, 124 distinct op-tuples, 322 distinct shapes.**

**Op-tuple census** (parent op + child-op tuple; count / distinct
terms):

```
 count  terms  op-tuple
    47     24  linear(·, ·)
    31     11  transpose(reshape)
    28      5  select(linear)
    24      6  select(·)
    24      5  mul(select, select)
    23      8  reshape(linear)
    21      7  add(mul, mul)
    20      4  select(sigmoid)
    16     12  matmul(·, ·)
    16      6  linear(mul, ·)
    16      3  mul(slice, ·)
    12      3  mul(select, add)
     9      9  mul(·, ·)
     8      8  sdpa(transpose, transpose, transpose)
     8      2  add(matmul, mul)
```

**Shape census** — each subterm abstracted to a leaf-numbered shape
(repeated leaves stay repeated; `share = yes` means a placeholder
repeats).  The frequent shapes are *flat* views (`linear`, `matmul`,
`mul`, `add`) and the `select`-indexed SSM step:

```
 count  terms share  shape
    47     24     -   linear(a, b)
    16     12     -   matmul(a, b)
     9      9     -   mul(a, b)
     6      6     -   transpose[dim0=-2,dim1=-1](a)
     5      5     -   mul(select[dim=0,index=3](linear(a, b)), select[dim=0,index=3](c))
     4      4   yes   mul(a, rsqrt(add(mean[dim=(-1,),keepdim=True](pow(a, 2)), 1e-06)))
```

**Sharing census** — for every binary op node, whether its operands
are structurally identical (`same-kids`) or merely share a leaf
(`shared-leaf`), aggregated by op-tuple:

```
 op-tuple                  sites same-kids shared-leaf
 mul(select, select)          24         0           4
 add(mul, mul)                21         0          19
 matmul(·, ·)                 16        16           0
 mul(slice, ·)                16         0           4
 mul(select, add)             12         0          12
 mul(·, ·)                     9         9           0
 add(matmul, mul)              8         0           7
 add(linear, linear)           4         4           3
```

The sharing column is the census's sharpest finding.  `add(mul, mul)`
has 19 sites that share *some* leaf — but that leaf is buried under a
`slice`/`select` (the RoPE rotary `x₁·c + x₂·s`, the SSM step
`a_t·h + b_t·x`), not a factor of the two products.  The shape that
looks factorable is the shape where factoring is invalid; the
per-schema match count below makes that exact.

## 2. The shape-aware proposer

`tools/law_shape_proposal.py` defines a small, honest library of
**schemas** — generic metavariable equalities over the real op
vocabulary — and measures each against the corpus:

* **relax** — real subterms matching the LHS with repeated *argument*
  metavariables relaxed to wildcards (the pure shape); attribute
  metavariables are left shared, so a precondition carried on attrs
  (the `select` dim/index) is still enforced.
* **match** — real subterms matching the LHS with **all** equalities
  enforced (`_term_match` requires a repeated metavariable to bind a
  structurally equal subterm).  Subterms are deduped by identity, so a
  shared node counts once.  This is the *applicable* count.
* **true** — the instantiated LHS and RHS agree numerically on random
  fp64 tensors of the *real* leaf shapes (the numeric oracle the
  proposal retro uses for new axioms — reused, not reinvented).
* **deriv** — `law_verifier.verify_law` proves it from `ALL_RULES`
  (then it is a composite and cannot lower a cost).
* **rel** — structurally a `duplicate` / `inverse` of a library rule,
  or `new`.
* **cost / use** — saturating a matched real term under `ALL_RULES`
  vs `ALL_RULES + {schema}` and re-extracting (the pipeline's own
  `backend_cost(executor_cost_for("generic"), sink.supported_ops)`);
  `use = yes` iff true **and** strictly cheaper.

The schemas target the census's frequent shapes: the elementwise
factorizations (`add(mul,mul)`, `sub(mul,mul)`), the `select`/`slice`
naturality, the layout fusions, the elementwise algebra, and two
library-duplicate controls (`linear_factor`, `matmul_factor`) so the
duplicate detector is exercised.

## 3. Results

### 3.1 match / truth / usefulness (21 schemas)

```
schema            relax  match  true deriv        rel           cost   use
factor_left          21      0     - False        new              -    no
factor_right         21      0     - False        new              -    no
factor_mid           21      0     - False        new              -    no
factor_sub_left       4      0     - False        new              -    no
factor_sub_right      4      0     - False        new              -    no
select_mul           24     24   yes False        new 6.96e4->5.22e4  yes
select_add            0      0     - False        new              -    no
select_sub            0      0     - False        new              -    no
slice_mul             0      0     - False        new              -    no
reshape_reshape       0      0     - False        new              -    no
reshape_transpose    31     31    no False        new 3.51e4->3.51e4   no
neg_add               0      0     - False        new              -    no
sub_neg               0      0     - False        new              -    no
mul_neg_left          0      0     - False        new              -    no
exp_add               0      0     - False        new              -    no
square_neg            0      0     - False        new              -    no
linear_factor         4      3   yes  True  duplicate 1.74e4->1.74e4  no
matmul_factor         3      2   yes  True    inverse 1.89e4->1.89e4  no
mul_zero              0      0     - False        new              -    no
id_mul_lit            1      1   yes  True  duplicate        0->0     no
sub_self              5      0     - False        new              -    no
```

Read the `relax`/`match` gap.  It is the impact retro's finding, made
per-schema:

* `factor_left/right/mid`: shape present **21** times, equality holds
  **0** times.  The `add(mul, mul)` sites never share a factor.
* `sub_self`: the `sub(_, _)` shape appears 5 times; `x - x` never.
* `select_mul`: shape present 24, equality holds 24 — the SSM step's
  two selects always carry the same `dim` **and** `index`, so the
  naturality precondition is *satisfied at every site*.
* `reshape_transpose`: shape present 31, but the numeric oracle says
  **false** — `reshape` then `transpose` is not `transpose` then
  `reshape`.  A cost-only proposer would have shipped it (it lowers the
  cost on 1 model, see below); the truth oracle rejects it.

### 3.2 firing on real models (each schema run alone)

```
schema                fires  changed  paid  cases
select_mul               24        5     5  SelectiveSSM,DiagDenseSSM,DiagonalSSM,HybridBlock,TwoLayerHybrid
reshape_transpose        23        7     1  AttentionBlock,GQAAttention,TransformerBlock,…
linear_factor             2        2     2  ParallelLinear,DeepParallel
```

`select_mul` is the only *true & new* schema that fires.  `linear_factor`
fires and pays but is a **duplicate** of the shipped `weight_factor_linear`
(the framework rediscovers an existing law, which is the sanity check);
`reshape_transpose` fires and cost-lowers on 1 model but is **false**.

The 8 elementwise-algebra schemas (`neg_add`, `sub_neg`, `exp_add`,
`square_neg`, `mul_neg_left`, `mul_zero`, `sub_self`) and the remaining
factorization schemas fire **0** times — the proposal retro's 11 laws,
re-derived shape-aware, are still dead on this corpus.

### 3.3 reach — end-to-end (ALL_RULES vs ALL_RULES + schema)

```
schema             model                                cost  fires  cert
select_mul         SelectiveSSM           6.79e+05->5.57e+05      8  pass
select_mul         DiagDenseSSM           6.09e+05->4.87e+05      8  pass
select_mul         DiagonalSSM             4.7e+05->3.48e+05      8  pass
select_mul         HybridBlock            6.87e+05->5.66e+05      8  pass
select_mul         TwoLayerHybrid         1.17e+06->9.31e+05     16  pass
reshape_transpose  AttentionBlock         1.48e+05->1.48e+05      6  pass
reshape_transpose  TransformerBlock       4.44e+05->4.44e+05      9  pass
```

Adding `select_mul` to the law universe drops the extracted cost on
**5 of the 22 models** — the SSM/hybrid family — by 16–26 %:

| model | cost | drop |
|---|---|---|
| SelectiveSSM | 6.787e5 → 5.569e5 | 17.9 % |
| DiagDenseSSM | 6.091e5 → 4.873e5 | 20.0 % |
| DiagonalSSM | 4.698e5 → 3.480e5 | 25.9 % |
| HybridBlock | 6.874e5 → 5.656e5 | 17.7 % |
| TwoLayerHybrid | 1.175e6 → 9.310e5 | 20.8 % |

The certificate replays (`cert=pass`) on every one, and lowering both
terms through the pipeline's `_lower_extracted` + `sink.verify`
(rtol 1e-4) **passes** — the two forms are numerically equal:

```
SelectiveSSM     fires=4 changed=True 6.787e+05->5.569e+05 verify=pass
DiagDenseSSM     fires=4 changed=True 6.091e+05->4.873e+05 verify=pass
DiagonalSSM      fires=4 changed=True 4.698e+05->3.480e+05 verify=pass
HybridBlock      fires=4 changed=True 7.048e+05->5.830e+05 verify=pass
TwoLayerHybrid   fires=8 changed=True 1.192e+06->9.484e+05 verify=pass
```

## 4. The verdict, plainly

**Yes — shape-awareness produces a law that FIRES on a real model.**
`select_mul` is true, genuinely new, fires 24 times on 5 models and
pays 16–26 % end-to-end.  The earlier 11 laws fired **0** times.  The
difference is *not* the algebraic content (the earlier grammar
contained distributivity and factoring too) — it is that the proposer
was pointed at a shape whose precondition real graphs satisfy.

The decisive negative from the impact retro also survives, sharpened:
the frequent `add(mul, mul)` shape (21 sites) admits **no** shared
factor, so the one shape that *looks* factorable is the one where
factoring is inapplicable.  Shape-awareness cannot manufacture an
equality that the shape does not carry.

## 5. The one law that fires and pays: `select_mul`

    mul(select(u, dim=D, index=I), select(v, dim=D, index=I))
        -> select(mul(u, v), dim=D, index=I)

* **True.**  `select` picks the `I`-th slice along `D`; elementwise
  `mul` commutes with slicing.  All 24 real sites are numerically
  equal, and the lowered before/after modules verify.
* **New.**  Not a duplicate or inverse of any of the 51 `ALL_RULES`,
  nor of the `LAYOUT_RULES`/`SCAN_LAWS` groups.
* **Where it comes from.**  It is the `select` analogue of the shipped
  `transpose_push_mul` / `transpose_pull_mul` (elementwise `mul`
  through a view op).  The library already pushes `mul` through
  `transpose`; it never learned to push it through `select`, and the
  census shows `select` is the frequent view op in the SSM family.
* **Why it pays.**  The pipeline's `executor_cost_for("generic")` is
  roofline **plus a per-dispatch overhead**.  `mul(select, select)` is
  *four* dispatched ops (`linear`, `select`, `select`, `mul`); the RHS
  is *three* (`linear`, `mul`, `select`).  The law removes exactly one
  dispatched op per site, and that dispatch term is what the 16–26 %
  drop measures.  The gain holds across `d_inner ∈ {8…256}`
  (17.9 % → 16.3 %) — it is an op-count reduction, not a size artefact.

## 6. Honesty about what this does and does not show

* **The gain is dispatch-count, under the pipeline's own model.**
  `select_mul` removes one dispatched op; the measured drop is the
  cost model's dispatch term (`_profile_dispatch_s`), not a roofline
  saving.  On a backend whose `select` is free (a pure view) the law
  would still be a *fusion* (fewer kernels) but the number would
  shrink; the honest claim is "the pipeline's selection model prefers
  the fused form", which is what `optimize_model` selects with.
* **`reshape_transpose` is a cost-only trap.**  It fires on 23 sites
  and lowers the cost on 1 model, yet the numeric oracle says **false**.
  This is the impact retro's lesson in reverse: a proposer scored on
  cost alone would ship a false law; the truth oracle is load-bearing.
* **The corpus is 22 models, not "all models."**  It spans
  MLP/attention/SSM/hybrid/conv blocks.  `select` appears because the
  SSM exporter unrolls the recurrence into per-step indexing; a model
  that used a `scan` op instead would not contain the shape.  The law
  is real for the graphs *as exported*.
* **The schema grammar is still human-authored** — 21 schemas, as in
  the proposal retro.  Shape-awareness changed *which* shape the
  grammar is pointed at, not the grammar's reach.  It is the binding
  constraint, and it is why 8 elementwise schemas still fire 0: no
  schema targets the `mul(select, add)` gating shape (12 sites), and
  none should — the only true law there (`distribute`) expands.
* **Firing is structural and size-independent.**  It is measured at
  the models' own dims; the reach comparison uses the pipeline's
  bounded saturation (EXPANSIVE budget 600, `max_iterations=6`).
* **No law was tuned to fire.**  The winning schema is a textbook
  naturality identity; the corpus is read by the census, not
  special-cased.

## 7. Recommendation (reported, not made)

`packages/` is untouched.  On this evidence, `select_mul` is the one
candidate worth a library entry:

* **`select_mul`** — `mul(select(u,D,I), select(v,D,I)) ->
  select(mul(u,v), D,I)`, tagged `SIMPLIFICATION` (or alongside the
  `transpose_push_*`/`transpose_pull_*` layout family).  It is true,
  new, fires and pays on 5 models with the certificate and lowered
  modules verifying.  A `check` hook is unnecessary: the shared
  `dim`/`index` attribute metavariables make the matcher enforce the
  precondition.  It should be *measured on the target corpus* (this
  tool) before adoption, exactly as the impact retro argues.

Everything else is dead weight on this corpus and should **not** be
added: the factorization and elementwise-algebra schemas fire 0 times
(the shape's precondition never holds), `reshape_transpose` is false,
and `linear_factor`/`matmul_factor` are already shipped.

The larger lesson: a law proposal pipeline should be **scored on the
corpus it will run on**, and driven by the *frequent* shapes of that
corpus — not by algebraic novelty over synthetic witnesses.  The
census is the input the proposer was missing; `select_mul` is the
payoff of supplying it.

## Gates

Run from the main worktree, HEAD plus `tools/law_shape_census.py` and
`tools/law_shape_proposal.py` (no `packages/` change):

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors; `tools/` is out of scope)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/python tools/radon_ratchet.py` — pass
* `uv run pytest -q` — pass
