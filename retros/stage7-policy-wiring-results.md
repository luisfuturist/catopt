# Stage 7 — wiring the learned policy into the pipeline: results

Plan 0016 stage 7's last mile.  The `Policy` seam was already consumed
— `EGraph.run(..., policy=...)` and `Optimizer.optimize(...,
policy=...)` both take one — but nothing learned could actually run
through it: the players lived in `tools/` games and the optimizer never
consulted them.  This note wires a learned policy into the real
pipeline and measures whether it helps.

**Short verdict: it does not.**  The learned policy reaches the *exact*
same extracted cost as the shipped ordering on every held-out real
model — never better — and is consistently slower; on the large
e-graphs it also inflates the enode count.  It is safe (the certificate
replays and the equivalence class is unchanged), and that is the whole
of what it delivers.

## What was built

All changes are additive and live in `catopt-torch` plus tests:

| file | what |
|---|---|
| `packages/catopt-torch/src/catopt_torch/graph_policy.py` | `GraphFeaturePolicy` — a `Policy` usable straight from `EGraph.run` / `Optimizer.optimize` |
| `packages/catopt-torch/src/catopt_torch/model_trajectories.py` | `model_rule_samples` — real-model trajectories, reusing `catopt_core.trajectories`'s encoding verbatim |
| `packages/catopt-torch/src/catopt_torch/learned_policy.py` | `LearnedPolicy.features_of` (the derivation seam), batched `scores`, cached structural rule vectors |
| `tests/test_graph_policy.py`, `tests/test_learned_policy.py` | tests + 100% coverage of the new lines |
| `tools/train_graph_policy.py` | train on real models, then race default / learned / random on held-out ones |

No `catopt-core` change was needed.

## The wiring gap it closes

`EGraph.run`'s schedule hook `_scheduled` (`egraph/core.py:1468`)
constructs the state it offers the policy as

```python
state = GameState(self, root_eid, iteration)
```

— i.e. **no `features`**.  The engine never needs them, so it does not
carry them.  But `LearnedPolicy.choose` reads `state.features`, so a
`LearnedPolicy` raises as soon as it is handed to the real engine.  That
is the gap: the learned players in `tools/` were never drivable by
`optimize`/`run`.  `GraphFeaturePolicy` derives the features itself.

## The pinned state contract

A learned policy is coupled to the semantics of its features — the
multifamily retro records a correctness fix in `catopt_core.features`
silently regressing a trained policy.  So the derivation is **pinned,
not recomputed per iteration**:

* features are computed **once per e-graph**, on the *first* consult —
  iteration 0, before any rule has fired — and cached for the rest of
  the run;
* at that instant the e-graph holds exactly the exported program, so the
  features are `compute_features(ir.root)`, **bit-identical** to the
  training contract of `trajectories.rule_samples` /
  `learned_policy.train_rule_value`.

Recomputing from a mid-search rewrite instead would move the input off
the trained distribution.  `tests/test_graph_policy.py` pins this: the
derived features `== compute_features(<exported root>)`, and a second
consult on the same e-graph returns the same cached object.

## Training

Supervised, not RL.  The multifamily retro found the supervised
per-`(features, rule)` target is dense and multi-family-stable while
REINFORCE collapses to a winner-take-all under the sparse reward; the
supervised signal is the right one to test *wiring*, which is the
question here.  Data: 10 real models from `catopt_torch.models` exported
through `TorchSource`, `rule_samples` on each root → 510 samples,
`RuleValueNet` (hidden 64, 2000 epochs, GPU).

**The signal is nearly empty.**  Of 510 `(program, rule)` samples only
**8** are strictly improving (`delta_cost > 0`) — 1.6%:

```
train swiglu           rules= 51 improving=  0
train rmsnorm          rules= 51 improving=  1
train attention        rules= 51 improving=  0
train transformer      rules= 51 improving=  2
train residual_mlp     rules= 51 improving=  0
train parallel_linear  rules= 51 improving=  1
train deep_parallel    rules= 51 improving=  1
train norm_linear      rules= 51 improving=  2
train matrix_chain     rules= 51 improving=  1
train gqa              rules= 51 improving=  0
```

The synthetic families (`chain`/`dup`/`linear`) were *constructed* to
have a paying rule; real models mostly do not — a rule applied alone to
a real model's graph almost never lowers its `flops_cost`.  That is the
first reason the policy cannot help: there is almost nothing to learn.

## Measurement — enodes / time / cost, learned vs default

Held-out models (unseen widths / heads / depth), real pipeline
(`catopt_orchestrator.optimize.search`, `DEFAULT_RULES`, GPU, RTX 2050,
`torch 2.14.0+cu130`).  `enodes` = `stats["n_enodes"]`, `cost` =
`flops_cost(result.term)`.

### Fixed point (the shipped default)

```
swiglu*            default  enodes=    11 iters=2 cost=     557568.0 ms=  37.6
                   learned  enodes=    11 iters=2 cost=     557568.0 ms=  65.7
attention*         default  enodes=    18 iters=1 cost=    9289728.0 ms=  46.7
                   learned  enodes=    18 iters=1 cost=    9289728.0 ms=  55.5
transformer*       default  enodes=    50 iters=3 cost=   73368048.0 ms= 137.5
                   learned  enodes=    50 iters=2 cost=   73368048.0 ms= 162.5
deep_transformer*  default  enodes=    97 iters=3 cost=  689675904.0 ms= 154.2
                   learned  enodes=    97 iters=2 cost=  689675904.0 ms= 194.8
deep_parallel*     default  enodes=    20 iters=4 cost=     244224.0 ms=  25.3
                   learned  enodes=    20 iters=4 cost=     244224.0 ms=  77.2
matrix_chain*      default  enodes=    15 iters=3 cost=      73728.0 ms=  20.4
                   learned  enodes=    15 iters=3 cost=      73728.0 ms=  51.2
```

* **cost: identical, every model.**  No quality gain, none.
* **iterations: the learned order reaches the fixed point in 1 fewer
  iteration** on the two transformers (3→2) — the one thing it does
  *better*.
* **time: 1.4–3.1× slower.**  The per-iteration schedule overhead
  (below) swamps the saved iteration.
* **enodes: equal or +2** — within the redundant-duplicate noise.

### Saturation only (`EGraph.run`, no non-local passes) — the policy's real effect

```
transformer*       default  enodes=  2209 iters= 9 cost=  36379632.0 ms=  250.1
                   learned  enodes=  3493 iters=10 cost=  36379632.0 ms=  675.6
                   random   enodes=  2736 iters=11 cost=  36379632.0 ms=  410.2
deep_transformer*  default  enodes=  4835 iters= 9 cost= 333201024.0 ms=  749.3
                   learned  enodes=  6587 iters=10 cost= 333201024.0 ms= 1124.8
                   random   enodes=  5855 iters=11 cost= 333201024.0 ms= 1252.1
```

On the large e-graphs the learned order makes saturation **worse**:
+58% enodes / 2.7× time on `transformer*`, +36% enodes / 1.5× time on
`deep_transformer*` — *worse than a random policy* in both cases.  The
engine's declaration order is deliberately cheap-filtering-rules-first;
the learned order front-loads the cost-improving (fusion) rules the
classifier favours, which grow the e-graph before the filtering rules
can prune it.  Cost is again identical.

### Bounded and lazy (`max_iterations=1`, `stop="improving"`)

Same story: cost identical on every model, learned slower everywhere.
The one place a reordering *could* pay — a truncated run — does not,
because the non-local pairing/lift passes (which run after saturation,
outside the policy's reach) supply the cost-improving members anyway.

## Why cost cannot move (the structural reason)

The policy may only *reorder* the saturation rules (`EGraph.run`
docstring, ADR 0003 invariant 5).  The e-graph fixed point is
order-invariant, so the partition of terms — the equivalence class — is
order-invariant too, and so is the extracted cost.  A learned policy
*and a random policy* therefore both reach exactly the shipped
ordering's cost; the only degree of freedom is how much redundant
enode/dedup churn the order leaves behind, and the learned order is
worse at that than the shipped one.

## Safety — re-asserted for the wiring

* **The certificate still replays.**  For all 6 held-out models:
  `eg.certificate(ir.root, best, cost_fn=flops_cost)` then
  `verify_certificate(ir.root, cert)` returns a term with the same cost
  as the extracted best.  OK (6/6).
* **A random policy yields the identical equivalence class.**  Checked
  through the full pipeline and at the saturation level, comparing a
  Weisfeiler-Lehman canonical form of the e-graph (eid-independent, so
  eid renumbering and duplicate enodes do not count as a difference):
  the partition is identical for default / learned / random on all 6
  models.  OK (6/6).
* **Note on `n_enodes`.**  `n_enodes` is *not* a semantic invariant:
  reordering renumbers eids and can leave a redundant duplicate
  spelling in a class, so `n_enodes` differs by a few between orderings
  at the same fixed point.  The partition — the actual equivalence
  class — does not.  Any future safety test must compare the partition,
  not the enode count.

## Verdict

**Wiring the learned policy in does not help.**  Concretely:

* it reaches the **same extracted cost** as the shipped ordering on
  every held-out model, in every regime (fixed point, bounded, lazy,
  saturation-only) — never lower;
* it is **slower** (1.4–3.1× on the full pipeline; up to 2.7× on
  saturation-only), because the engine consults `choose` once per rule
  per iteration and each decision pays a net forward;
* on the large e-graphs it **inflates the enode count** (up to +58%),
  worse than a random policy;
* its only measured advantage is reaching the fixed point in **one
  fewer iteration** on the two transformers — which the decision
  overhead more than cancels.

This is a *structural* result, not a tuning failure.  A policy that
only reorders a confluent rewrite system cannot change the fixed point,
so it cannot change the extracted cost; the training signal on real
models is also nearly empty (1.6% improving rule-applications).  The
honest conclusion for the plan is that **the `Policy` seam is not a
lever for extraction quality** — that lives in the rules, the cost
model and the non-local passes, none of which the policy touches.

Two things would have to change for a search policy to matter, and
neither is a better-trained net:

1. **a regime where order changes the answer** — e.g. a hard
   node/iteration budget the *non-local passes* also respect, so firing
   high-yield rules first reaches a better cost before the cut;
2. **a denser signal on real models** — a multi-step / value target
   (the retro's next step) rather than the one-step `delta > 0` that is
   empty here.

## Reproduce

```sh
.venv/bin/python tools/train_graph_policy.py --device cuda
```

It prints the training label counts, the three held-out races (fixed
point, bounded, saturation-only, lazy) and the safety assertions, and
exits non-zero if any invariant fails.  Full log:
`project/retros/stage7-policy-wiring-run.txt`.

## Gates

`ruff check` + `format --check` (packages, tools), `ty check`,
`vulture` (0), `lint-imports` (4 kept), `radon_ratchet` (ok), `bandit`
(0), `semgrep` (0 findings), `coverage` on the touched files (100%).
