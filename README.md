# catopt

Certified-equivalent program search over neural computation graphs —
it finds faster programs the compiler can't express, proves they're
equivalent (or, under `error_budget=`, certifies a proven
approximation bound), and hands them to a backend to run. The engine
and orchestrator are backend-agnostic (no torch imports); PyTorch is
the shipped reference backend — `catopt_torch` implements the ports.

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, example_input)
out = opt(x)        # same function as model(x), verified rtol=1e-4
```

One call runs the whole pipeline: the backend's `Source` lifts the
model to a typed IR (torch: `torch.export`), equality saturation
enumerates equivalent programs, a cost model extracts the cheapest
one the backend can execute, the lowered module is checked against
the original, and you get back a `torch.nn.Module` plus a stats dict (`stats["rule_fires"]`,
`stats["lowering"]`, `stats["runner"]`, …).

## Quickstart

```bash
uv sync                 # dev env — all workspace members editable
python demo.py          # 60-second end-to-end run on CPU
```

### Choose how the result is delivered

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

### Steer what "cheapest" means

```python
from catopt_orchestrator import LatencyCriterion, MemoryCriterion

opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
    model, x,
    criteria=LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3,
)
# or the shorthand: criteria={"latency": 1.0, "memory": 0.5}
```

## What it does

- **e-graph semantic search** — model *and* weights export into one
  term; saturation enumerates the equivalence class. Shared-input
  projections pair into one GEMM + split views (`pair` product law),
  weight chains fold weights-first, recurrences reassociate.
- **certified transforms** — every extracted program carries a
  replayable derivation `original → optimized`;
  `verify_certificate` rechecks each step standalone. Non-local
  passes lift recurrences into carrier forms the tensor IR can't
  reach: parallel scans, streaming softmax, closed-form resolvents.
- **measured or priced selection** — extraction is priced per-backend
  (`sink.supported_ops` bounds the search to executable forms), or
  measured end-to-end by the `Autotuned` strategy, which reports
  the winner honestly — including when it loses.

## When it helps — honest numbers

Measured on the dev box (RTX 2050 / CPU), all rows verified
equivalent:

| Regime | Result |
|---|---|
| Blocks with exploitable structure (`bench/suites/algebra/real_win_hunt.py`) | **~1.1–10× vs Inductor** — unnormalized-attention reassoc ~2×, PaLM parallel blocks ~1.2×, expert-sum weight fold ~7× |
| Carrier + CUDA-graph decode (`bench/suites/algebra/decode_scan_bench.py`, `decode_retnet`) | **1.65–3.4× vs Inductor / best non-carrier** — chunked scan carriers amortize to zero launches |
| Deep weight chains (`bench/suites/algebra/reassoc_scale.py`) | 8.9–16.1× vs Inductor — a form Inductor's post-grad graph provably can't reach |
| Morphism windows (`bench/suites/algebra/morphism_e2e.py`) | **12.4–12.5× measured wall** on 4-block chains at GEMM-bound sizes — term flops −92% fully translates; 10.7–11.4× vs plain Inductor (the compile-time weight fold is out of its reach) |
| Residual reassoc (`ResidualReassoc`) | term flops −66% → measured 1.3–2.2× (partial conversion — distributed adds/fillers eat headroom; +inductor recovers more) |
| KV latent sharing (`KVLatentShare`, opt-in) | kv flops/bytes −62.5%, module params −31% — a memory/params win, NOT wall-time (compute parity, −14% at tiny sizes — reported honestly) |
| Bounded rewrites (`error_budget=`) | certified-approximation mode: `search(..., error_budget=1e-3)` accepts rewrites whose propagated output bound fits the budget. **stories15M: 1.15–1.32× vs Inductor** (bound 1e-4→1e-2) — near-dup tied-head rows elide, KL≈0 at the tight end, bounds always recorded + verified-with-tolerance. `None` = exact only |
| Whole model E2E (`bench/suites/models/e2e_model.py`) | **~1.05× over plain Inductor** — pairing fires per block, verified fp64-exact |
| Real trained checkpoints (stories15M/110M) | Exact mode: parity (dense weights carry ~zero bitwise structure — measured by `bench/suites/core/structure_census.py`). Bounded mode: **1.15–1.32× vs Inductor** on stories15M via the collapsed tied head |

This is **not** a universal speedup. Attention and GEMM-bound code is
already optimal — expect a parity floor there — and the losses are
measured too: launch-bound decode cells (B=1, T≤64) lose 4–15% and a
batch-8 flat decode cell lands at ~0.56× vs Inductor. The wins live
where structure exists: shared-input projections, foldable weight
chains, unnormalized attention, recurrences.

## Demo

```bash
python demo.py                 # CPU
python demo.py --device cuda   # GPU (needs CUDA torch)
python demo.py --quick         # shorter timing loop
```

One command on a gated projection block (PaLM-style gates + a 10-deep
value chain): the e-graph search, the certificate replayed through
the standalone verifier, the extracted op tree, and a synced median
race vs eager and `torch.compile`. **~2.1× on CPU, ~2.6× on an RTX
2050** — from two transforms Inductor structurally cannot do
(projection concat + weight-first fold). `PYTHONHASHSEED=0` re-exec
makes it bit-for-bit reproducible.

## Benchmarks

Every claim above is a runnable suite under `bench/`.  The harness is
a real system, not a pile of scripts:

```bash
python -m bench list                       # the catalog
python -m bench run reassoc_scale --quick   # one suite → JSON+MD+HTML+plots
python -m bench run-all --quick             # every harnessed suite
python -m bench dashboard                   # cross-suite HTML index
```

Each suite states its conclusion as a typed **finding** (win / parity /
regression / negative) with the supporting metric, so every surface —
JSON, Markdown, the HTML dashboard, a Quarto document, Slidev assets —
is rendered from one canonical report.  See `bench/README.md`.

| Suite | Category | Measures |
|---|---|---|
| `reassoc_scale` | algebra | k-deep weight chain → 1 GEMM; dumps Inductor's post-grad graph to prove the form unreachable |
| `search_efficiency` | core | saturation cost vs the Catalan-sized program space |
| `real_win_hunt` | algebra | autotuned wins on realistic block topologies |
| `real_linear_attn` | algebra | scan lift on RetNet/GLA/delta-rule blocks, CPU+CUDA |
| `decode_scan_bench` | algebra | chunked decode on carriers, eager vs CUDA-graphed |
| `decode_bench` | models | launch-bound (B,T) sweep — the falsified hypothesis, losses included |
| `stories15m_bench` | models | real llama2.c checkpoints through `strategy=Compositional()` |
| `e2e_model`, `e2e_llm`, `e2e_models2` | models | whole-model E2E: llama-toy, ~0.4B prefill+decode, non-decoder shapes |
| `model_bench` | models | complete multi-block models: latency, peak memory, compile time |
| `cost_fidelity` | core | predicted-cost vs measured-latency rank correlation |
| `killer_demo` | algebra | `Autotuned` per-model lowering selection |
| `law_bench` | core | per-rewrite-law value harness |
| `morphism_e2e` | algebra | term-flops → wall-time conversion for the morphism laws |
| `bounded_e2e` | models | `error_budget` sweep on a real checkpoint (speedup vs bound vs KL/top-k drift) |
| `structure_census` | core | how much catopt-exploitable structure real trained weights carry |
| `bound_amplification` | core | weight-bound → output-error propagation on real activations |

## API surface

The orchestration surface lives in `catopt_orchestrator`
(backend-neutral — it imports no torch); the torch ports, runners
and autotune candidate builders live in `catopt_torch` /
`catopt_cuda`; the engine itself is `catopt_core`.

| Name | Signature / role |
|---|---|
| `Optimizer` | `(backend=..., source=..., sink=..., composer=..., meter=..., criteria=..., runner=...)` — the configured entry point; `.optimize(model, x, **kw)` → `(module, stats)`, plus `.search` / `.lower` / `.discover` phase verbs |
| `Monolithic` / `Compositional` / `Autotuned` / `MorphismSearch` | the `strategy=` argument of `Optimizer.optimize`: whole-model search (default), per-block + recompose (`block_pred=`, `verify_tol=`, `cache=` replays structurally-identical blocks — N identical blocks cost ~1 search + N−1 verified replays, `max_cross_pairs=` re-judges adjacent pairs jointly), measured autotune (`candidates=`, `budget_s=`, `profile=` persists measured corrections, `builders=TORCH_BUILDERS`), or block-signature algebra (`optimize_morphisms` — lifts each block to a `BlockSig`, rewrites the tiny morphism graph with `WindowCompose`/`ResidualReassoc`/`NormCascade`/`WeightTie`/`KVLatentShare`/opt-in `CrossBlockCSE`, reifies into certified term rewrites) |
| `search` / `lower` | the phase verbs: `model -> SearchResult`, `SearchResult -> LowerResult` — re-lower one search under different runners |
| `discover_alternatives` | `(model, x, *, source, ...)` → `SearchResult` — enumerate the equivalence frontier (`.alternatives(top_k)`, `.certificate()`) |
| `export_optimized` / `load_optimized` | `catopt_torch.export` — `(model, opt, path, fmt="module"|"safetensors"|"state_dict"|"torchscript", …)`; `.pt2` roundtrips run standalone, no catopt at inference |
| Rule sets | `search(..., rules=DEFAULT)` — composable `RuleSet` algebra (`FULL - SYMMETRY`, `WITH_LAYOUT`, presets in `catopt_core.laws.ruleset`) |
| Engines | `search(..., engine=NativeEngine())` — the pure-Python engine is the default/reference; `catopt-native` (PyO3/Rust) is an explicit opt-in (~17× on match-bound closures) |
| Detection passes | `search(..., detect_factors=True)` — certified low-rank weight factoring; `detect_specials=True` — exact dead/diag/dup/block-diag weight elision; `error_budget=` — certified bounded approximations (bound ledger + output-propagated verify) |
| Criteria | `LatencyCriterion`, `FlopsCriterion`, `DepthCriterion`, `MemoryCriterion("weights"|"peak"|"combined")`, `CompiledCriterion` — compose with `*` / `+`, or pass `{"axis": weight}` dicts |
| Runners | `IdentityRunner` (default), `TorchCompileRunner()`, `CudaGraphRunner()`, `ChainedRunner([...])` — duck-typed `Runner` protocol |
| Ports | `Source` / `Sink` (`catopt_core.ports`; torch impls `catopt_torch.adapters.TorchSource`/`TorchSink`, bundled as `TorchBackend`) — a new backend implements `Sink`; the engine never imports it |
| Verification | `catopt_core.egraph.verify_certificate` — replays the derivation shipped with every extracted program |

## Generality — honest split

The **framework** is general: e-graph saturation, verification,
cost extraction, backends, strategies, runners and rule sets are
all pluggable and model-agnostic. What is **narrow** is the *law
library*: like every rule-based optimizer (Halide, TASO, verified
compilers), catopt finds the structures its laws describe — an
unmatched block is an opaque boundary, never a wrong answer. The
morphism engine is the generality mechanism: laws target signature
*classes* (any residual chain, any shared-projection family)
rather than specific op trees, so coverage grows at the right
level of abstraction.

## Limits

- **Wins are regime-dependent** — the transform set is structural:
  pairing, folds, reassociation, carrier lifts. If the model is
  already dense-GEMM-bound with no shared structure, expect parity.
- **Search is compile-time work** — seconds per block; monolithic
  eqsat slows past ~8 blocks, which is why the `Compositional`
  strategy exists.
- **Inference only** — weight folding destroys per-layer gradients;
  no backward-graph rewriting.
- **Coverage gaps** — `matmul`+bias and grouped convs aren't
  pairable; reassociation needs unnormalized attention; masks must
  arrive materialized.
- **Dev-box numbers** — RTX 2050 (4 GB) / CPU; `calibrate()`
  re-targets the cost model, but magnitudes don't extrapolate to
  datacenter hardware.

## Install

Python ≥3.11 (developed on 3.13), `torch>=2.0`, `numpy>=1.24`.
uv-workspace monorepo: `packages/catopt-core` (zero-dependency
engine), `catopt-torch` (PyTorch adapters), `catopt-carriers`
(scan/attention carriers), `catopt-cuda` (the CUDA-graph runner),
`catopt-orchestrator` (the backend-neutral pipelines). The `catopt`
façade is gone — import the domain packages directly. Optional:
`packages/catopt-native` is the PyO3/Rust search engine (build with
maturin; opt-in via `engine=` — never auto-detected).

```bash
uv sync                                  # everything, editable

# or with pip:
pip install -e packages/catopt-core -e packages/catopt-torch \
    -e packages/catopt-carriers -e packages/catopt-cuda \
    -e packages/catopt-orchestrator

pip install -e packages/catopt-core      # engine only, zero deps
```

## Depth

- `bench/README.md` — per-suite protocols and expected verdicts.
- `AGENTS.md` — repo layout, verification commands, the port
  contracts.
