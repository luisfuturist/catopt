# Evaluation — the fourth dimension

ADR 0003 splits catopt into four independent dimensions.  This page
is the **evaluation** one: how catopt describes, ranks, and — when a
target is available — measures a program, without letting any of that
change what is *equivalent*.

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
```

## Static features — `catopt_core.features`

`ProgramFeatures` describes a program without running it: `flops`,
`bytes_read` / `bytes_written`, `temporary_bytes`, `depth`,
`operations`, `param_leaves`, `reuse` (arithmetic intensity), and
`parallelism` (average width).  Pure Python, torch-free.

```python
from catopt_core.features import StaticProfiler

features = StaticProfiler().profile(term)   # a ProgramFeatures
features.to_vector()                        # pin a model's input layout
```

Features are **not** a "profile": a profile
(`catopt_core.profile.TargetProfile`) is a *measured target*; features
describe a *program*.

## The frontier — `catopt_core.pareto`

Keep the landscape; do not scalarize early.  `CostVector` names the
dimensions, `pareto_frontier` keeps the non-dominated candidates, and
`best` is the scalar *view* over them.

```python
from catopt_core.pareto import CostVector, pareto_frontier, best

frontier = pareto_frontier(candidates, key=vector_of)
winner = best(candidates, key=vector_of, weights={"latency": 0.7, "memory": 0.3})
```

## The game — `catopt_core.game`

A thin formalization over the e-graph: state (`GameState`), action
(`Action`), rule book (`RuleBook`), transition (`transition`),
evaluator (`Evaluator`).  The **referee is not here** — legality is
the laws and the certificate.

## Policies — `catopt_core.policies`, `catopt_torch.learned_policy`

A `Policy` only *orders* the legal moves, so it can never change what
is certified.  `RandomPolicy` / `ExistingPolicy` / `GreedyPolicy` /
`BeamPolicy` ship first; `LearnedPolicy` is one more conforming value.

The learned policy scores `(program features ⊕ structural rule
vector)` with a small MLP (`RuleValueNet`) and picks the best action.
The rule vector is **structural** — root-op buckets, a full
pattern-tree hash, arities, check/derive flags — so a rule the model
has never seen is still scored by its shape, not a vocabulary index
(no retraining to add runtime rules).

```python
from catopt_torch.learned_policy import LearnedPolicy, train_rule_value

model = train_rule_value(samples, epochs=3000)     # GPU when available
policy = LearnedPolicy(model, {r.name: r for r in rules})
pick = policy.choose(GameState(None, 0, features=f), actions)
```

Pretrain, post-train, and predict in one command — see
`tools/train_search_policy.py` and the measured result in
[`project/retros/stage7-learned-policy-results.md`](../project/retros/stage7-learned-policy-results.md).

## Performance prediction — `catopt_core.perf_model`

`AnalyticalPerformanceModel` is the first `PerformanceModel`: a
roofline prediction `max(compute, memory) + launch overhead` from
features and a target.  It **ranks**; it never prunes the semantic
space.  A learned model is one more conforming value.

## Failures — `catopt_core.failures`

`FailureClass` (`OK` / `OOM` / `TIMEOUT` / `KERNEL` / `NONFINITE` /
`UNAVAILABLE` / `UNKNOWN`) and `classify(exc)` give measurement
failures a taxonomy instead of an ambiguous crash.

## The rule

Evaluation is an independent dimension: a profiler **observes**, a
policy **orders**, a model **ranks** — none of them decides
equivalence.  Feasibility (`supported_ops`, hard) and performance
(soft) stay distinct.
