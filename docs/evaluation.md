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

## Calibration — `catopt_torch.calibrate`, `tools/calibrate_profile.py`

The roofline/executor cost models price programs with *measured*
constants: `calibrate()` micro-benchmarks the machine it runs on
(peak FLOPS, bandwidth, launch/dispatch/leaf-eval/compiled-graph
overheads, and a per-op-class kernel table) and returns a
`TargetProfile`.  `tools/calibrate_profile.py` is the "run this on
your hardware" entry point — it writes a loadable profile JSON that
additionally carries the *executor-family corrections*
`tools/executor_cost_probe.py` measures (batched vs generic routed
deliveries, per shape bucket), so the delivered-price comparison in
`_carrier_upgrade` — and extraction under `delivered_cost_for` —
rank by this machine's numbers, not the built-in constants:

```sh
.venv/bin/python tools/calibrate_profile.py --out calibrated_profile.json
```

```python
from catopt_core.profile import TargetProfile
from catopt_orchestrator import Optimizer, delivered_cost_for
from catopt_torch.backend import TorchBackend

profile = TargetProfile.load("calibrated_profile.json")
opt = Optimizer(backend=TorchBackend())
res = opt.search(
    model, x, cost_fn=delivered_cost_for(profile, x=x)
)
mod = opt.lower(res, x, verify=True).module
```

Two consumption levels, both through the existing `cost_fn=` seam
(no new pipeline parameter):

* `executor_cost_for(profile)` — measured constants only: per-op
  prices calibrated, the `_carrier_upgrade` comparison corrected by
  the profile's `corrections`/`measured_ns` tables (the `profile`
  marker `backend_cost` forwards into `_select_best_term`).
* `delivered_cost_for(profile, x=x)` — additionally bills each term
  under the lowering it would actually be delivered by (batched
  carrier vs generic), corrected by the measured factors under
  `shape_bucket(x)`.  The delivered-aware extraction model the SSM
  routing inversion needs — measured-slower batched deliveries stop
  winning extraction.  Non-additive at carrier roots (documented
  caveat, same as `fused_cost_for`).

Corrections are per `(route, shape_bucket)` and clamped to
`[0.1, 10]`; the executor-correction phase needs CUDA
(`--skip-executor-corrections` writes a constants-only profile).
`profile.save()` / `TargetProfile.load(name)` persist under
`~/.cache/catopt/profiles`.

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

The model criterion is an **approximate in-search axis**, not a
faithful target chooser: extraction recovers a node's cost by an
*additive* marginal, exact only for an additive cost function, so a
whole-subtree prediction is either mis-ranked or collapses to
bit-identical values.  Measured in
[`eval-axis-selection.md`](../project/retros/eval-axis-selection.md):
a different target changes the extracted program in **0/7** families.
Use a `PerformanceModel` for post-hoc ranking; a dense (non-marginal)
pricing path is the open fix.

## Failures — `catopt_core.failures`

`FailureClass` (`OK` / `OOM` / `TIMEOUT` / `KERNEL` / `NONFINITE` /
`UNAVAILABLE` / `UNKNOWN`) and `classify(exc)` give measurement
failures a taxonomy instead of an ambiguous crash.

## Wiring — what actually consumes these

The ports are not decoration; each is reached from a real call:

| Port / module | Consumed by |
|---|---|
| `Policy` | `EGraph.run(..., policy=)` — the schedule consults it once per iteration; `search(..., policy=...)` threads it through |
| `Profiler` + `PerformanceModel` | `PredictedCriterion` — pass it as `criteria=` to price extraction by predicted runtime |
| `pareto` | `SearchResult.frontier(cost_fns)` — the non-dominated set over named axes |

```python
from catopt_core.cost import flops_cost, param_bytes_cost
from catopt_core.perf_model import AnalyticalPerformanceModel
from catopt_core.policies import GreedyPolicy
from catopt_orchestrator import PredictedCriterion
from catopt_orchestrator.optimize import search

res = search(
    model, x,
    source=TorchSource(),
    criteria=PredictedCriterion(AnalyticalPerformanceModel()),
    policy=GreedyPolicy(),
)
res.stats["criteria"]   # {'predicted': 1.0}
res.stats["policy"]     # 'greedy'
res.frontier({"flops": flops_cost, "memory": param_bytes_cost})
```

A policy may only *reorder* — every rule still runs, so the fixed point
and the certificate are unchanged.  A model may only *rank* — the
feasibility bound (`supported_ops`) is still what decides reach.

## The rule

Evaluation is an independent dimension: a profiler **observes**, a
policy **orders**, a model **ranks** — none of them decides
equivalence.  Feasibility (`supported_ops`, hard) and performance
(soft) stay distinct.
