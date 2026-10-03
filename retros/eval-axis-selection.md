# Evaluation-axis selection — does pluggable evaluation change the answer?

Plan 0016 (ADR 0003, the EVALUATION dimension) wired four ports into
the pipeline: `Policy` steers `EGraph.run`, `pareto` backs
`SearchResult.frontier`, and `Profiler` + `PerformanceModel` back
`PredictedCriterion`.  Every case probed until now found that
`criteria=PredictedCriterion(...)` selected the **same** program as
the default cost model, and that `SearchResult.frontier({...})`
returned a **single** member.

This retro records the search for the case that turns that wiring
from an *enabler* into a *result* — and the honest verdict where it
does not exist.

Measured by the new suite
`bench/suites/evaluation/eval_axis.py` (registered as `eval_axis`,
`python -m bench run eval_axis --device cpu --quick`).  CPU-only,
deterministic, ~3 s.

> **Revision note.**  An earlier run of this suite reported a WIN on
> the per-target axis.  That run predated commit `ecfbf50`
> (`features: bill runtime traffic, not phantom bytes`), which made
> `catopt_core.features.compute_features` bill **runtime** traffic
> only — view-op outputs (`_VIEW_OPS`) and subtrees with no data
> input (`has_var_leaf` false) are no longer charged.  The old WIN
> was a profiler artefact, and the suite's accounting double-
> subtracted the (now already-absent) phantom slice, printing a
> nonsensical 149 % phantom share.  This revision re-derives every
> number against the fixed profiler; **all three findings are now
> NEGATIVE.**

## Verdict

| Claim | Verdict | Evidence |
|---|---|---|
| A bandwidth-bound target extracts a different certified program than the compute / launch targets | **NEGATIVE** | 0/7 families: the bandwidth pick is never distinct from *both* the compute and launch picks; it equals the launch pick everywhere |
| The root-class frontier exposes a genuine multi-axis trade-off | **NEGATIVE** | 0 genuine frontiers over 7 families × 21 axis pairs; 6 >1 results are 1 exact tie + 5 float-noise pairs |
| The residual target-sensitivity traces to a hardware trade-off | **NEGATIVE** | the true roofline value ranks the same form first under all three targets in 7/7 families; the move is the additive marginal decomposition of the non-additive `PredictedCriterion` |

## Method

Seven families, each exported and saturated once: `chain3`
(three-deep linear chain), `rect` (rectangular two-chain, k > d),
`swiglu` (SwiGLU MLP), `qkv_sum` (three same-input projections
summed), `linattn` (unnormalised attention `(Q Kᵀ) V`),
`deep_parallel` (`(x@W1 + x@W2) @ W3`), `parallel8` (eight-expert
projection sum).

Per family the suite records

1. the **root-class frontier** — `SearchResult.frontier` over every
   pair of the built-in axes (`flops`, `launch`, `count`, `depth`,
   `peak`, `params`, `roofline`); a >1 frontier is classified
   *genuine* (every axis spreads ≥ 1 %), *tie* (some axis exactly
   flat) or *noise* (float-only spread);
2. the **per-target extraction** — the search re-run under
   `criteria=PredictedCriterion(AnalyticalPerformanceModel(),
   hardware=<target>)` for a compute-bound (`tflops=1e-3`), a
   bandwidth-bound (`gbps=0.05`) and a launch-bound
   (`launch_us=5000`) synthetic `TargetProfile`.  Every pick is
   certified (`lower` → `sink.verify`) and priced two ways:
   * `marginal` — the DAG-sum the search's extraction actually
     minimizes (`dag_cost` over the criterion blend);
   * `true` — the whole-program model value
     (`model.predict(features, target)`).

   The `marginal`/`true` split is the mechanism probe: for a
   non-additive criterion they diverge exactly when the additive
   marginal decomposition mis-ranks a form.

Candidates priced: 21 root-class members total (5 + 3 + 1 + 3 + 2 +
4 + 3), each on 7 axes = 147 axis pairs; 28 searches
(default + 3 targets × 7 families); 21 picks lowered and verified.

## Finding 1 — per-target extraction: **NEGATIVE**

Under the fixed profiler the bandwidth story is gone.  The
bandwidth-bound pick is **never** distinct from both the compute and
the launch picks; it equals the launch pick in every family.  The
three targets disagree among themselves in exactly **one** family
(`swiglu`, compute vs bandwidth/launch), and only two families move
at all against the default:

| family | root alts | distinct certified | bandwidth vs default | targets agree |
|---|---|---|---|---|
| `chain3` | 5 | 1 | same | yes |
| `rect` | 3 | 1 | same | yes |
| `swiglu` | 1 | **2** | same | **no** (compute ≠ bw/launch) |
| `qkv_sum` | 3 | 1 | same | yes |
| `linattn` | 2 | **2** | **differs** | yes |
| `deep_parallel` | 4 | 1 | same | yes |
| `parallel8` | 3 | 1 | same | yes |

Every pick passed `sink.verify` (`max_rel` ≤ 5.7e-7).

**`swiglu`** — the divergence is on the *compute* axis, not the
bandwidth axis.  Default / bandwidth / launch pick the **fused**
gate/up form (`split(linear(x, concat(W1, W3)))` with `silu`), 7 ops,
34 816 write bytes.  The **compute** target alone picks the
`silu`-*expanded* form `mul(a, sigmoid(a))` — 8 ops, 43 008 write
bytes, **same** flops (794 624).

**`linattn`** — the divergence is on the *model* axis, not the
target axis.  The **default** (built-in latency model) picks the
left-associated `(Q Kᵀ) V` (8 388 608 flops, 294 912 write bytes);
**all three** predicted targets agree on the right-associated
`Q (Kᵀ V)` (10 485 760 flops, 393 216 write bytes).  The three
targets never disagree with each other here — the *built-in* model
moved, not the target.

So plugging a different **target** in does not extract a different
program for a bandwidth reason anywhere.  The deliverable the plan
hoped for — "two different targets, two different certified
programs" — does not survive an accurate profiler.

## Finding 2 — the frontier: **NEGATIVE**

`SearchResult.frontier` over all 147 axis pairs returned **0 genuine**
multi-axis trade-offs.  The 6 results with >1 member:

* **1 exact tie** — `linattn` `count × params`: both frontier members
  price *identically* (`{count: 5, params: 12288}`); `pareto_frontier`
  keeps both because neither strictly dominates.
* **5 float-noise pairs** — `chain3` `{flops,launch,count} × depth`,
  `depth × {params,roofline}`: the calibrated `depth_cost_for`
  leaves a ~2e-12 relative spread between associativity variants
  (`8930.112359550563` vs `8930.112359550561`), so neither dominates.

No pair ever produced two members that trade off on *both* axes.

**Why.**  Two structural reasons, both measured:

1. `SearchResult.frontier` enumerates only **root-class enodes**
   (`extract_alternatives` forces one root enode at a time).  A
   distinction that lives in a *sub-class* — an associativity or
   fusion choice under the root — is invisible to it.  `linattn`'s
   root class has 2 enodes; the fused-vs-unfused projection
   distinction is a child-e-class choice, never priced by `frontier`.
   (The scope is now documented explicitly on
   `SearchResult.frontier`; widen it deliberately when a sub-class
   trade-off matters.)
2. Within the root class, the param-only fold discount makes one form
   dominate on **every** built-in axis.  E.g. `chain3`'s
   weights-first root (`linear(x, W3@(W2@W1))`) has the lowest
   `flops`, `launch`, `count` *and* `params`; it is the Pareto
   optimum, not a trade-off point.

So the *enabler* is wired, but the *result* — a >1 frontier — does
not exist in the structures reachable here.

## Finding 3 — the mechanism: **NEGATIVE**

Even the residual target-sensitivity (Finding 1's 2/7 families) is
not a hardware trade-off.  The **true** roofline value
(`model.predict(compute_features(t), target)`) ranks the **same**
form first under all three targets in **7/7** families.  The move is
an artefact of pricing the **non-additive** `PredictedCriterion`
through the **additive** marginal decomposition `extract_best` uses
(`local = c(t) − Σc(children)`).

| family | target | search picks | true argmin | `marginal` of the two forms | `true` of the two forms |
|---|---|---|---|---|---|
| `swiglu` | compute | mul-expanded | **silu-fused** | 0.000793028 vs 0.000795076 | 0.000798624 vs **0.000798124** |
| `linattn` | compute | right-assoc | **left-assoc** | 0.006293956 vs 0.006293956 | 0.01049026 vs **0.008393108** |
| `linattn` | bandwidth | right-assoc | **left-assoc** | 0.00622842 vs 0.00622842 | 0.01671618 vs **0.01278402** |
| `linattn` | launch | right-assoc | **left-assoc** | 0.025000062 vs 0.025000062 | 0.045000167 vs **0.045000128** |

Two failure modes, both visible:

* **`swiglu`** — the marginal *mis-ranks*: the additive
  decomposition prices the expanded form 0.26 % lower
  (0.000793028 < 0.000795076) while the true model prices it
  0.06 % *higher* (0.000798624 > 0.000798124).  The search minimizes
  the wrong number and picks the strictly worse form.
* **`linattn`** — the marginal *collapses*: the two forms are priced
  **bit-identically** on every axis (0.006293956 / 0.00622842 /
  0.025000062) while the true values differ by up to **25 %**
  (0.008393108 vs 0.01049026).  The additive decomposition erases the
  distinction entirely; the tie is broken by search order, landing on
  the right-associated form the true model ranks *worse*.

`PredictedCriterion`'s own docstring warns about this: a prediction is
a function of the whole subtree, so it is non-additive inside
extraction.  The finding quantifies the consequence — the search
optimizes the approximation, not the prediction.

Note the axis that *does* differ for the right reason is `count` /
launch: the fused form replaces three GEMMs with one (a real kernel
reduction).  That is why `launch` prefers the fused form — but the
byte axis, and the whole bandwidth story, was the (now-fixed) profiler
artefact.

## What this means

1. **With an accurate profiler there is no per-target result.**  The
   pluggable evaluator (`PredictedCriterion` +
   `AnalyticalPerformanceModel`) is still wired and still steers
   extraction — but no target extracts a program the others would
   not, and the true model ranks identically across regimes.  This is
   the honest negative the plan's "evaluation dimension" needs to
   record: the *enabler* exists, the *result* does not.
2. **The profiler fix (commit `ecfbf50`) was necessary and
   sufficient to kill the artefact.**  `compute_features` now bills
   runtime traffic only, agreeing with the built-in view- and
   fold-aware cost models.  The suite no longer splits `bytes_written`
   (there is no phantom slice to subtract); it compares the *priced*
   traffic and the marginal-vs-true prices directly.
3. **The residual search-level sensitivity is a non-additivity
   artefact.**  `PredictedCriterion` is a whole-program function
   priced through `extract_best`'s additive marginal recovery.  A
   genuinely non-additive criterion (whole-DAG `dag_exact` pricing, or
   a criterion whose marginal is exact) would remove it.  *Actionable:*
   price `PredictedCriterion` with `dag_exact`-style whole-term
   evaluation, or document that it is an approximate in-search axis.
4. **`SearchResult.frontier` is too narrow to be the discovery
   surface — and now says so.**  It prices root-class enodes only, so
   a sub-class trade-off is invisible.  The scope is documented on the
   method; widening it (a `diverse_classes` union, or a frontier over
   candidates gathered under several cost fns) is a deliberate,
   separate change.
5. **No genuine hardware trade-off exists in the reachable rewrite
   set.**  Catopt's laws are Pareto-improving identities (fold,
   factor, fuse, reassociate) — every one reduces work on at least
   one axis without increasing another, in the real executor.  A
   genuine trade-off would need a compute-vs-memory rewrite
   (recomputation, spilling) that the law set does not contain.

## Reproduce

```sh
.venv/bin/python -m bench run eval_axis --device cpu --quick
# canonical (non-quick), with artifacts under bench/results/:
.venv/bin/python -m bench run eval_axis --device cpu
```

Artifacts: `bench/results/eval_axis.{json,md,html}` (JSON carries the
per-family `aux`: every pick's signature, runtime features
(`read` / `write` / `traffic`), the search `marginal` and the true
model value under all three targets, plus the frontier rows).
