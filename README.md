# catopt

**A verified search engine over faster, provably equivalent versions of
your model.**

catopt takes a PyTorch model, searches the space of semantics-preserving
graph transformations, and returns a faster module with a replayable
equivalence certificate — then hands it to `torch.compile` so Inductor
does the codegen either way.

```text
PyTorch model
  → torch.export              (the graph, typed)
  → e-graph saturation        (equivalent programs, enumerated)
  → cost-based extraction     (the cheapest one, for your hardware)
  → verified module           (certificate replayed, fp64-checked)
  → torch.compile / Inductor  (same backend, fair fight)
```

```python
from catopt.optimize import optimize_model

opt, report = optimize_model(model, example_input)
out = opt(x)          # equivalent to model(x), certificate-backed
opt = torch.compile(opt)
```

Deep stacks use `optimize_compositional`, which optimizes each block
against its captured real input and recomposes with per-block
verification and automatic fallback.

## What it finds

The transforms are not handwritten recipes — they fall out of the
equivalence space, then a cost model decides per shape whether to take
them. All rows verified semantically equivalent (fp64 where stated);
RTX 2050, synced timing.

| Transform | Example | Result |
|---|---|---|
| **Projection pairing** (QKV, gate·up, parallel branches → 1 GEMM) | PaLM-style block, 5 projections fused | **1.24× vs Inductor** end-to-end |
| **FLOP reduction** (reassociation, weight merging, factorization) | DeepParallel b=4096 | **2.51×** GPU / **3.02×** CPU |
| **Asymptotic reassociation** | LinearAttention `(QKᵀ)V → Q(KᵀV)`: O(T²d)→O(Td²) | **8.0×** at T=2048 |
| **Attention fold** | `softmax(masked_fill(qkᵀ·s)) @ v` → `sdpa(is_causal)` | **4.6×** vs eager, **2.5×** under Inductor (nanoGPT, T=2048) |
| **Parallel-scan discovery** | LTI recurrence → balanced Blelloch tree | **6.3×** CUDA-graph, T=64 |
| **Diagonal-affine scan** | Mamba-faithful `a⊙h + b⊙x` | **4.4×** CUDA-graph, T=64 |
| **Streaming attention** | om monoid: O(1) state per KV block | **260×** vs sdpa-recompute at 65k cache; 89 MiB flat at 2M keys |
| **Conv pairing** | 4 parallel conv1×1 → 1 conv | **1.24–1.33×** all batch sizes |
| **Diagonal absorption** | `repeat_kv` → SDPA `enable_gqa` (llama2.c) | 1.12× at T=512 |

On unmodified community code — Karpathy's `llama2.c` — it automatically
rediscovers `MergedColumnParallelLinear` and `QKVParallelLinear`, the
transforms vLLM and TensorRT-LLM implement by hand.

## Why it finds what Inductor can't

> A program optimizer's reachable set is bounded by its semantic
> language, not its search strategy.

Tensor IRs rewrite *ops*. catopt **lifts programs into carriers** —
monoid objects where the same associativity law reaches structures no
op-level pattern can express:

- **Affine monoid `aff(A,b)`**: an unrolled recurrence
  `h_t = Ah_{t-1} + x_t` under matmul/add laws plateaus at depth ~1.5T —
  the balanced scan is unreachable because the partial-product pair is a
  cross-class object no term law synthesizes. In the carrier, the *same*
  associativity law produces the Blelloch tree: **depth 2T → ~2·log₂T**.
- **Online-softmax monoid `om(m,l,a)`**: FlashAttention's combine falls
  out of homomorphism + associativity — no `flash_attention` rule is
  ever written.
- **Product law `⟨f₁,…,f_k⟩ = (×fᵢ)∘Δ`**: `pair_shared_input_linears`
  groups shared-input projections at diagram level — one GEMM feeding k
  uneven split views, arbitrary arity, consumer-agnostic. Pattern
  matchers need a rule per consumer shape; this needs none.
- **`trace^U`** (traced monoidal structure): recurrences are fixpoints;
  the Joyal–Street–Verity axioms are rewrites, so loops become
  closed-form resolvents `P + Q(I−S)⁻¹R`.
- **Cross-carrier laws** (`xcarrier.py`): readouts push through scan
  evaluation, scans fold inside softmax elements, and attention over
  scanned values stays one recurrence — measured where the wall actually
  is (scores are quadratic in the state; no affine carrier reaches
  them).

Executors lower discovered carrier trees to level-batched GPU kernels
(`scan_lower.py`, `om_lower.py`, `omd_lower.py`); the IR carries
`Param` leaves as first-class terms, so a weight that no extracted
member references drops out of the state dict on its own — folding,
tying, and dedup are the same event.

## Verified, not hoped

Every extracted program carries a **certificate**: an ordered,
replayable derivation of `original → optimized` that
`verify_certificate` re-checks on real terms — derivational equivalence,
not numerical spot-checks. Non-local passes attach pointwise witnesses
so even constructed members replay standalone.

The verifier is load-bearing, not ceremonial. It has caught: a matcher
that skipped repeated-metavariable equality (a would-be false proof), a
broadcast-shape misinference that fabricated a 1.98× "win", a
well-typed but semantically wrong program (diff 9.83), and a benchmark
measuring CUDA submission time instead of execution. All
regression-tested.

Saturation scales: coherence stratification canonicalizes the
contractible law-spaces instead of searching them (2.0M → 351 enodes on
the T=8 recurrence; 4-layer models 768s → 4.9s; ≥8 blocks monolithic),
with `optimize_compositional` as the deeper path.

## Where it wins — and where it doesn't

The honest regime map, all measured:

| Regime | Verdict |
|---|---|
| FLOP-reducing rewrites (assoc/fold/factorize) | **Wins everywhere** — pure work reduction |
| Nonlinear-boundary folds (attention fold, reassociation) | **Wins** — up to 8×, asymptotic |
| Carrier lifts (scans, streaming attention) | **Wins in their regime** — depth & memory, not raw latency |
| Conv pairing | **Wins** — Inductor never fuses cuDNN calls |
| GEMM pairing on transformer blocks | **Parity** — ~40 non-GEMM kernels/layer dilute it |
| Real trained checkpoints (stories15M/110M) | **Parity** — all blocks transform and verify, no win at these sizes |
| Launch-bound decode cells (B=1, T≤64) | **Loses 4–15%** — split-view copies cost more than saved launches |
| Large cells (B≥8, T≥128, stories110M) | **Parity** — GEMM-shape efficiency washes out at ~1% |

Real checkpoints (`bench/stories15m_bench.py`): stories15M and
stories110M pass the full pipeline — 8/8 and 14/14 blocks optimize,
QKV + gate·up fuse, outputs verify to ~2e-5 — at parity with Inductor
(3.0 ms / 17.0 ms both ways). The mechanism is real (−37% GEMM launches,
profiler-verified); at these dimensions it just doesn't pay. The
launch-bound hypothesis was falsified — on blocks, on whole models,
and on the large-cell crossover sweep (`bench/decode_bench.py`).

**Controlled negative**: NormLinear loses slightly (0.98×) — Inductor
already fuses `x·rms·wn` into the GEMM's input read, so restructuring
can't promise a bandwidth win intra-kernel fusion already delivers.
That's the boundary: catopt wins on transforms that *change the graph*;
it can't beat a kernel-level fusion on the same graph.

`calibrate()` measures your device's peak FLOPS / bandwidth / launch
overhead and `roofline_cost_for(profile)` re-prices the search per
target — the same equivalence space, selected per backend.

## Limitations

- **Nothing discovered is novel to practitioners** — fused QKV, flash
  attention, the linear-attention identity are all known. The
  contribution is automatic discovery + verification + per-shape choice,
  not new math. (A transform nobody encoded hasn't appeared yet; the
  machinery for one exists.)
- **Inference only** — weight folding destroys per-layer gradients;
  backward-graph rewriting (AOTAutograd) is unimplemented.
- **Chunked attention loses to fused SDPA head-to-head** when K,V fit on
  device — the om win is feasibility (bounded memory, incremental
  state), not throughput.
- **Coverage gaps** — `matmul`+bias and grouped convs aren't pairable;
  reassociation needs unnormalized attention; masks must arrive
  materialized.
- **All numbers are an RTX 2050 (4 GB)** — `calibrate()` makes
  re-targeting mechanical, but don't extrapolate magnitudes to
  datacenter hardware.

## Layout

The repo is a **uv workspace monorepo** — the engine is split into
domain distributions so each carries only its own dependencies:

```
packages/catopt-core/       the torch-free semantic engine (zero deps)
  ir, attrs, typing         term IR, attr schemas, shape inference
  egraph/                   union-find, matching, saturation, proof
  laws/                     equational laws + non-local pairing passes
  cost, meta, rulecache     cost algebra, rule synthesis, law cache
  ports, ops                Source/Sink + protocol boundaries, OpTable
packages/catopt-torch/      the PyTorch adapters (deps: core + torch)
  adapters                  TorchSource / TorchSink (the ports)
  torch_bridge              torch.export → IR → executable module
  executors/                BatchedExecutorBase + level_schedule
  models/, report           small model zoo; OptReport + verify gate
packages/catopt-carriers/   semantic carriers (deps: core + torch)
  om, xcarrier, trace       online-softmax / deferred / traced monoids
  *_lower, trace_lift       lowerers + the non-local lift passes
packages/catopt-eps/        opt-in certified-approximation toolkit
  eps, act_eps, ibp         weight offers, site wraps, interval bounds
packages/catopt-optimize/   pipeline orchestrators (deps: all above)
  optimize, regime, calibrate
catopt/                     façade — public API + compat aliases;
                            every historical `catopt.X` path resolves
                            to its new home via sys.modules
bench/                      real-checkpoint benchmarks: stories15M/110M,
                            decode sweep, e2e smoke, omd attention stack
tests/                      1609 tests
project/                    orphan branch: plans, ADRs, retrospectives
```

`catopt-core` installs standalone — the IR, e-graph, laws, and cost
algebra run with zero dependencies (no torch). Integrations plug in
per-domain: `pip install -e packages/catopt-core` for just the engine.

**Ports.**  The pipeline's two ends are named ports, so the frontend and
backend are swappable without touching the engine.  A `Source` lifts a
model to IR (`model → (IR, leaf values)`); a `Sink` lowers IR back to a
runnable and owns its op set and equivalence gate.  PyTorch is the
shipped pair — `TorchSource` / `TorchSink` are the defaults for
`optimize_model` and `discover_alternatives` (`source=` / `sink=`
override them).  A sink's `supported_ops` bounds extraction: any member
using an op the backend can't lower prices at `+inf` (`backend_cost`),
so the optimizer only commits to forms the sink can execute.  A new
backend implements `Sink`; `catopt-core` never imports it.

```bash
uv sync                       # installs all workspace members editable
python main.py                # demo: all transform families
python bench/fetch.py         # checkpoints → ~/.cache/catopt
python bench/stories15m_bench.py --device cuda
python bench/decode_bench.py --device cuda --quick   # launch-bound sweep
python -m pytest tests/ -q
```

## References

E-graphs: Willsey et al., *egg*. Scans: Blelloch, *Prefix Sums and Their
Applications*. FlashAttention: Dao et al. Traced categories:
Joyal–Street–Verity. llama2.c: Karpathy.
