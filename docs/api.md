# API surface

The orchestration surface lives in `catopt_orchestrator`
(backend-neutral — it imports no torch); the torch ports, runners and
autotune candidate builders live in `catopt_torch` / `catopt_cuda`;
the engine itself is `catopt_core`.  There is no `catopt` façade — import
the domain packages directly.

## The entry point

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
```

`Optimizer.optimize(model, x, **kw) -> (module, stats)`.  `stats`
carries `rule_fires`, `lowering`, `runner`, and the strategy's own keys.

## Choose how the result is delivered

```python
from catopt_cuda import CudaGraphRunner
from catopt_orchestrator import (
    Autotuned, ChainedRunner, Compositional, Optimizer,
)
from catopt_torch import TorchBackend, TorchCompileRunner
from catopt_torch.autotune import TORCH_BUILDERS

opt = Optimizer(backend=TorchBackend())

opt_mod, stats = opt.optimize(model, x, runner=TorchCompileRunner())
# torch.compile wraps the delivered module (stats["compiled"])

opt_mod, stats = opt.optimize(model, x, runner=CudaGraphRunner())
# captures the executor into a CUDA graph (stats["cuda_graph"]) —
# compile-free; works without Inductor

opt_mod, stats = opt.optimize(
    model, x,
    runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()]),
)
# runners compose left-to-right; the Runner protocol is duck-typed,
# so your own runner drops in

opt_mod, stats = opt.optimize(
    model, x, strategy=Autotuned(builders=TORCH_BUILDERS),
)
# re-lowers the same extracted term through each candidate executor
# ("generic", "batched", "torch_compile", "cuda_graph"), verifies
# each, times them on the real input, returns the measured winner

opt_mod, stats = opt.optimize(model, x, strategy=Compositional())
# multi-block models: optimizes each block against its captured real
# input, recomposes with per-block + end-to-end verification and
# automatic fallback (stats["blocks"], stats["end_to_end"])
```

## Steer what "cheapest" means

```python
from catopt_orchestrator import LatencyCriterion, MemoryCriterion

opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
    model, x,
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3,
)
# or the shorthand: criteria={"latency": 1.0, "memory": 0.5}
```

## Reference

| Name | Signature / role |
|---|---|
| `Optimizer` | `(backend=..., source=..., sink=..., composer=..., meter=..., criteria=..., runner=...)` — the configured entry point; `.optimize(model, x, **kw)` → `(module, stats)`, plus `.search` / `.lower` / `.discover` phase verbs |
| `Monolithic` / `Compositional` / `Autotuned` / `MorphismSearch` | the `strategy=` argument of `Optimizer.optimize`: whole-model search (default), per-block + recompose (`block_pred=`, `verify_tol=`, `cache=` replays structurally-identical blocks — N identical blocks cost ~1 search + N−1 verified replays, `max_cross_pairs=` re-judges adjacent pairs jointly), measured autotune (`candidates=`, `budget_s=`, `profile=` persists measured corrections, `builders=TORCH_BUILDERS`), or block-signature algebra (`optimize_morphisms` — lifts each block to a `BlockSig`, rewrites the tiny morphism graph with `WindowCompose`/`ResidualReassoc`/`NormCascade`/`WeightTie`/`KVLatentShare`/opt-in `CrossBlockCSE`, reifies into certified term rewrites) |
| `search` / `lower` | the phase verbs: `model -> SearchResult`, `SearchResult -> LowerResult` — re-lower one search under different runners |
| `discover_alternatives` | `(model, x, *, source, ...)` → `SearchResult` — enumerate the equivalence frontier (`.alternatives(top_k)`, `.certificate()`) |
| `SearchResult.frontier` | `(cost_fns: {axis: CostFn}, candidates=32, top_k=8)` → `[(CostVector, term)]` — the **non-dominated** members over named axes (`catopt_core.pareto`); feed them to `pareto.best` for the scalar view |
| `export_optimized` / `load_optimized` | `catopt_torch.export` — `(model, opt, path, fmt="module"|"safetensors"|"state_dict"|"torchscript", …)`; `.pt2` roundtrips run standalone, no catopt at inference |
| Rule sets | `search(..., rules=DEFAULT)` — composable `RuleSet` algebra (`FULL - SYMMETRY`, `WITH_LAYOUT`, presets in `catopt_core.laws.ruleset`) |
| Engines | `search(..., engine=NativeEngine())` — the pure-Python engine is the default/reference; `catopt-native` (PyO3/Rust) is an explicit opt-in (~17× on match-bound closures) |
| Detection passes | `search(..., detect_factors=True)` — certified low-rank weight factoring; `detect_specials=True` — exact dead/diag/dup/block-diag weight elision; `error_budget=` — certified bounded approximations (bound ledger + output-propagated verify) |
| Criteria | `LatencyCriterion`, `FlopsCriterion`, `DepthCriterion`, `MemoryCriterion("weights"|"peak"|"combined")`, `CompiledCriterion` — compose with `*` / `+`, or pass `{"axis": weight}` dicts |
| `PredictedCriterion` | `(model: PerformanceModel, profiler=None, hardware=None)` — a `Criterion` that prices a term by the model's **prediction** (`catopt_core.perf_model.AnalyticalPerformanceModel` is the shipped model). Non-additive inside extraction; opt in when you want a model to steer selection |
| `policy=` | on `search` / `Optimizer.optimize`: a `Policy` (`catopt_core.policies`: `RandomPolicy` / `ExistingPolicy` / `GreedyPolicy` / `BeamPolicy`; `catopt_torch.learned_policy.LearnedPolicy`; `catopt_torch.rl.RLPolicy`). Consulted once per saturation iteration by `EGraph.run(..., policy=)`; it may only **reorder** — every rule still runs, so the fixed point and the certificate are unchanged. Recorded in `stats["policy"]`; a policy-less engine declines with a clear error |
| Profiling | `catopt_core.features.StaticProfiler().profile(term)` → `ProgramFeatures` (flops, bytes, depth, reuse, …) — torch-free, computed without running the program |
| Measurement | `catopt_core.timing` — the one contract (`TorchMeter`, `calibrate`, `benchkit.Runner` all reduce through it); `catopt_core.failures.FailureClass` / `classify` — OOM / timeout / kernel / NaN / device-unavailable, and `TimingResult` carries `device` / `warmup` / `failure` provenance |
| Runners | `IdentityRunner` (default), `TorchCompileRunner()`, `CudaGraphRunner()`, `ChainedRunner([...])` — duck-typed `Runner` protocol |
| Ports | `Source` / `Sink` (`catopt_core.ports`; torch impls `catopt_torch.adapters.TorchSource`/`TorchSink`, bundled as `TorchBackend`) — a new backend implements `Sink`; the engine never imports it. The evaluation dimension adds `Profiler` / `Policy` / `PerformanceModel` (`catopt_core.ports`), each with a shipped implementation and a consumer in the pipeline (ADR 0003) |
| Verification | `catopt_core.egraph.verify_certificate` — replays the derivation shipped with every extracted program |
