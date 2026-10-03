# Law impact — do the 11 proposed laws actually pay on real models?

The proposal retro (`law-proposal.md`) found 11 genuinely-new,
cost-reducing laws absent from `ALL_RULES` — mul-over-add factoring,
neg-distribution, `square(neg x) = square x`, the `exp` homomorphism,
the annihilators, `pow x 1`, `x - x`, `x + (-x)` and `x / x`.  Every
one is numerically true and lowers the extracted cost on the
*synthetic* term it was designed for.  It left the applicability
question open: **do these laws fire on real graphs, and when they do,
do they pay?**

This retro answers it with `tools/law_impact.py` — a single tool that
encodes the 11 laws *locally* (never in `packages/`, never in the
shipped `ALL_RULES`), then measures firing, cost delta and reach on
real graphs.

The headline is a decisive negative: **not one of the 11 laws fires on
any of the 60 bench law cases or the 22 real model graphs.**  The
synthetic control fires all 11 (proving the harness detects firings),
the relaxed-pattern census shows *why* the real corpus never fires
(the law shapes are mostly absent; the two present shapes —
`add(mul, mul)` and `sub(_, _)` — never carry the required equality),
and adding the laws changes nothing about saturation: e-node counts,
extracted cost and certificate replay are all bit-identical.

Reproduce: `.venv/bin/python tools/law_impact.py`
(~2 min, CPU-only, no network; `--json PATH` for machine-readable
output).  Output is deterministic.

## 1. What the tool measures

Three steps, all on the pipeline's own machinery:

1. **Firing** — each law is run *alone* over every case (bench law
   cases at size 32, the 22 real models exported to IR, and a
   synthetic control = each law's own LHS instance).  Firing counts
   are read from `EGraph.rule_fires`, zeros included.
2. **Cost delta** — for every firing, the pipeline cost model
   (`backend_cost(executor_cost_for(lowering="generic"),
   sink.supported_ops)` — the model `optimize_model` selects with)
   gives the before/after cost; when extraction changed the program,
   both are lowered through `_lower_extracted` and compared with
   `sink.verify` (rtol 1e-4).
3. **Reach** — each real graph is saturated twice, under `ALL_RULES`
   and under `ALL_RULES + {the 11 laws}`, both with the pipeline's
   bounded saturation (EXPANSIVE rules budgeted at 600 enodes,
   `max_iterations=6`, `max_nodes=60_000`).  Reported: e-node count,
   e-class count, extracted cost, whether the extracted term changed,
   and whether the certificate still replays (`eg.certificate` →
   `verify_certificate`).

The laws are unconditional (the pattern already restricts the shape);
the conditional caveats the proposal retro records (`x/x` needs
`x != 0`; the annihilators carry the usual `inf`/`nan` fp caveat) are
noted below, not encoded as `check` hooks — the question is whether
the *pattern* matches at all.

## 2. The 11 laws, encoded in the tool

| name | lhs → rhs |
|---|---|
| `cand_mul_factor` | `x*y + x*z → x*(y+z)` |
| `cand_mul_factor_right` | `y*x + z*x → (y+z)*x` |
| `cand_neg_factor` | `-x + -y → -(x+y)` |
| `cand_square_neg` | `(-x)² → x²` |
| `cand_exp_factor` | `eˣ·eʸ → e^(x+y)` |
| `cand_mul_zero` | `x*0 → 0` |
| `cand_mul_zero_left` | `0*x → 0` |
| `cand_pow_one` | `x¹ → x` |
| `cand_sub_self` | `x - x → 0` |
| `cand_add_inv` | `x + (-x) → 0` |
| `cand_div_self` | `x / x → 1` |

## 3. Step (a) — do they fire? (the firing table)

Each law run alone over every case; `synth` is the control.

```
law                       bench  model  synth  total  paid
----------------------------------------------------------
cand_mul_factor               0      0      1      1     1
cand_mul_factor_right         0      0      1      1     1
cand_neg_factor               0      0      1      1     1
cand_square_neg               0      0      1      1     1
cand_exp_factor               0      0      1      1     1
cand_mul_zero                 0      0      1      1     1
cand_mul_zero_left            0      0      1      1     1
cand_pow_one                  0      0      1      1     1
cand_sub_self                 0      0      1      1     1
cand_add_inv                  0      0      1      1     1
cand_div_self                 0      0      1      1     1
```

**Zero real firings across 60 bench cases and 22 models.**  Every law
fires exactly once on its own designed witness — the control that the
harness is sound.

The real corpus, in full: `SwiGLU`, `RMSNorm`, `ResidualMLP`,
`ParallelLinear`, `DeepParallel`, `NormLinear`, `MatrixChain`,
`AttentionBlock`, `GQAAttention`, `TransformerBlock`, `ParallelBlock`,
`RepeatKVAttention`, `EagerAttention`, `AdditiveMaskAttention`,
`ParallelConv`, `LinearRecurrence`, `LinearAttention`,
`SelectiveSSM`, `DiagDenseSSM`, `DiagonalSSM`, `HybridBlock`,
`TwoLayerHybrid` — all exported cleanly; none fires any law.  The
whole-set run (all 11 at once) also fires only on the synthetic
controls.

## 4. Step (a2) — *why* not? (the relaxed-pattern census)

A firing table of zeros is only useful if it is not a harness bug.  To
separate "the op is absent" from "the shape is present but the
equality precondition fails", the tool re-matches each law's LHS with
its repeated metavariables *relaxed* to fresh wildcards — a pure
structural pattern.  Over the 931 distinct subterms of the real corpus
(bench + models):

```
law                       relaxed  fired  interpretation
--------------------------------------------------------
cand_add_inv                    0      0  shape absent in the corpus
cand_div_self                   1      0  shape present, precondition fails
cand_exp_factor                 0      0  shape absent in the corpus
cand_mul_factor                21      0  shape present, precondition fails
cand_mul_factor_right          21      0  shape present, precondition fails
cand_mul_zero                   0      0  shape absent in the corpus
cand_mul_zero_left              0      0  shape absent in the corpus
cand_neg_factor                 0      0  shape absent in the corpus
cand_pow_one                    0      0  shape absent in the corpus
cand_square_neg                 0      0  shape absent in the corpus
cand_sub_self                   5      0  shape present, precondition fails
```

Two shapes *are* common in real graphs, and the census shows exactly
why the laws miss them:

* **`add(mul, mul)` — 21 sites, never a shared factor.**  The RoPE
  rotary is `add(mul(x₁, cos), mul(x₂, sin))`; the Mamba/SSM step is
  `add(mul(a_t, h), mul(b_t, x_t))`.  Both are `x₁·c + x₂·s`-shaped —
  the two products have *different* factors, so `mul_factor`
  (`x*y + x*z → x*(y+z)`) has nothing to factor.  The 21 sites are
  precisely the ones where factoring would be wrong.
* **`sub(_, _)` — 5 sites, operands differ.**  The RoPE cases are
  `sub(mul(x₁,cos), mul(x₂,sin))`; the bench `sub_to_add` case is a
  bare `sub(x, y)`.  None is `x - x`.
* **`div(_, _)` — 1 site, operands differ.**  `AdditiveMaskAttention`
  has `div(matmul(...), sqrt(...))`; not `x / x`.

The other eight laws' shapes (`mul`-with-literal-0, `pow`-with-1,
`add(neg, neg)`, `square(neg)`, `mul(exp, exp)`, `add(x, neg x)`)
simply never appear.  A real model does not multiply by a literal
zero, raise to the literal first power, or negate two operands and add
them — the ops exist, the *patterns* do not.

This is the honest answer to "why zero": it is not that the harness
failed to see the graphs.  It is that textbook identities fire on the
expressions a human writes *when constructing a synthetic witness*,
not on the expressions `torch.export` emits from a neural network.

## 5. Step (b) — when they fire, do they pay?

Every firing is on a synthetic control, and every one pays and
verifies:

```
source    case                 law                    fires      cost in     cost out  chg  verify
--------------------------------------------------------------------------------------------------
synthetic cand_add_inv         cand_add_inv               1     3.48e+04            0  yes    pass
synthetic cand_div_self        cand_div_self              1     1.74e+04            0  yes    pass
synthetic cand_exp_factor      cand_exp_factor            1    5.221e+04     3.48e+04  yes    pass
synthetic cand_mul_factor      cand_mul_factor            1    5.221e+04     3.48e+04  yes    pass
synthetic cand_mul_factor_right cand_mul_factor_right      1    5.221e+04     3.48e+04  yes    pass
synthetic cand_mul_zero        cand_mul_zero              1     1.74e+04            0  yes    pass
synthetic cand_mul_zero_left   cand_mul_zero_left         1     1.74e+04            0  yes    pass
synthetic cand_neg_factor      cand_neg_factor            1    5.221e+04     3.48e+04  yes    pass
synthetic cand_pow_one         cand_pow_one               1     1.74e+04            0  yes    pass
synthetic cand_square_neg      cand_square_neg            1     3.48e+04     1.74e+04  yes    pass
synthetic cand_sub_self        cand_sub_self              1     1.74e+04            0  yes    pass
```

The laws are real: on their designed term, extraction picks the
cheaper member, the lowered before/after modules agree (verify
`pass`), and the annihilators collapse to a constant (`cost 0`).  The
tool confirms the proposal retro's cost table — and confirms that none
of it transfers to a real model.

## 6. Step (c) — do they add reach?

Saturating each real graph with and without the 11 laws leaves every
quantity identical (a representative slice; all 22 models are `no`):

```
model                     enodes       classes                   cost  chg   cert
---------------------------------------------------------------------------------
SwiGLU                    79->79        31->31     8.71e+04->8.71e+04   no   pass
ResidualMLP               16->16        13->13     8.71e+04->8.71e+04   no   pass
AttentionBlock            31->31        30->30     1.48e+05->1.48e+05   no   pass
TransformerBlock      1246->1246      311->311     4.44e+05->4.44e+05   no   pass
ParallelBlock         1194->1194      324->324     3.57e+05->3.57e+05   no   pass
SelectiveSSM          4729->4729      931->931     6.79e+05->6.79e+05   no   pass
TwoLayerHybrid          118->118       87->87     1.17e+06->1.17e+06   no   pass
```

* e-node and e-class counts are unchanged — the laws introduce no new
  enode because they never fire;
* the extracted cost is unchanged — no cheaper member was offered;
* **the certificate still replays** (`pass` on every model, with and
  without the laws) — adding rules that do not fire cannot break the
  proof, and the tool checks it rather than assuming it.

## Verdict

**We found 11 true, useful-on-paper laws, and none of them fires on
any real model or bench term.**  On a real graph the reach is
*bit-identical* with and without them: same enodes, same cost, same
replayed certificate.  The measured improvement on a real model is
**exactly zero** — for all 11, on all 22 models, on all 60 bench
cases.

This is the decisive test the proposal retro asked for, and it is the
honest test of whether law-finding is worth anything *by itself*:

* The proposal loop is sound — enumerate, verify, price — and it does
  produce true, cost-reducing laws.  But "cost-reducing on the term you
  designed the law for" is nearly vacuous: a law whose LHS you wrote by
  hand always fires on that hand-written term.
* Applicability is the binding constraint, and it is *empirical*.  A
  neural network's exported graph is built from a small vocabulary of
  shaped primitives — linear/reshape/transpose/sdpa/softmax/mul/add —
  and the elementwise-algebra identities a symbolic search finds live
  in a region of term-space the graph never visits.
* The one shape that *is* common — `add(mul, mul)`, the RoPE rotary
  and the SSM recurrence — is exactly the shape where factoring is
  *invalid*, because the two products never share a factor.  The
  census makes that concrete rather than hand-wavy: 21 sites, 0 shared
  factors.

The corollary for the search's value: a law proposal pipeline should be
scored on the *corpus* it will run on, not on synthetic witnesses.  The
proposal retro's "useful" column (a synthetic-term cost drop) is
necessary but nowhere near sufficient; `tools/law_impact.py` is the
missing applicability gate.

## Honesty about what this does and does not show

* **The laws are true and the harness is sound.**  The synthetic
  control fires every law and verifies the lowered result; the census
  re-matches the shapes independently of the e-graph matcher.  The
  zeros are real zeros.
* **The corpus is 22 models, not "all models."**  It spans MLP/attention/
  SSM/hybrid/conv blocks, including the export forms (`linear`,
  `layer_norm`, `rsqrt`, `mean`, `pow(x,2)`, `sdpa`, `chunk`, `cat`)
  that the shipped pipeline targets.  A model written in a different
  style could contain an `add(mul, mul)` with a shared factor — but no
  standard neural-network block does, and the 21 real `add(mul,mul)`
  sites here are all RoPE/SSM, none factorable.
* **`mul_factor`'s 21 relaxed sites include bench cases**, not only
  models: the RoPE bench cases are the rotary pattern.  The census
  mixes the two corpora on purpose (both are "real terms"); the model
  slice alone contributes the SSM recurrences, all non-factorable too.
* **Firing is measured at size 32 / small model dims.**  Firing is a
  *structural* property — the shape either matches or it does not —
  and is independent of the size knob; the reach comparison uses the
  same small graphs, where saturation already reaches the fixed point
  on the bench cases (the deep models stop at `max_iterations`, as the
  pipeline's own bounded saturation does).
* **No law was tuned to fire.**  Nothing in the tool special-cases a
  corpus; the laws are the proposal retro's exact 11, encoded verbatim.

## Caveats (stated, not tuned)

* **Conditional soundness.**  `cand_div_self` (`x/x = 1`) needs
  `x ≠ 0` (`0/0 = nan`); `cand_mul_zero`, `cand_sub_self`,
  `cand_add_inv` carry the usual `inf`/`nan` caveats.  These would
  need a `check` hook before entering the library — moot here, since
  none fires.
* **Cost model.**  Usefulness is priced with the pipeline's
  `backend_cost(executor_cost_for("generic"), sink.supported_ops)`.
  A different model (param bytes, launch-aware) could re-rank the
  synthetic deltas, but cannot make a non-firing law fire.
* **Reach budget.**  The reach comparison uses the pipeline's bounded
  saturation (EXPANSIVE budget 600, `max_iterations=6`).  The deep
  models stop at `max_iterations`; the counts are identical between
  the two runs either way, which is the point.

## Recommendation (reported, not made)

None of the 11 laws belongs in `packages/` or `ALL_RULES`: on the
evidence here they are dead weight — zero firings, zero reach, zero
cost.  If any were ever to be adopted, it should be gated on an
applicability measurement over the target corpus (this tool), not on
the synthetic-term cost drop the proposal retro reports.  The tool is
the recommendation: run it before adding a law.

## Gates

Run from the main worktree, HEAD plus `tools/law_impact.py` only
(no `packages/` change):

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check tools/law_impact.py` — pass
  (`tools/grammar_proposal.py` is unformatted at HEAD; a shared-worktree
  file owned by another agent — not touched here)
* `.venv/bin/ty check` — pass (0 errors; `tools/` is out of scope)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/python tools/radon_ratchet.py` — pass (1873 functions)
* `uv run pytest -q` — pass (3340 passed, 31 skipped).  One earlier
  full-suite run flaked 2 `tests/test_benchkit.py` cases with
  `ModuleNotFoundError: No module named 'plotly'` (plotly is installed
  and both pass in isolation — a pre-existing test-isolation flake,
  unrelated to this tool, which pytest does not collect); the rerun
  was clean.
