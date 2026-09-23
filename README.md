# catopt

**Categorical semantics + equality saturation as a neural-network graph optimizer.**

`catopt` translates PyTorch models into a typed symmetric-monoidal IR, explores
semantics-preserving rewrites with an e-graph, extracts a lower-cost program,
and lowers it back through `torch.compile`/TorchInductor. Every comparison in
this README uses the same model, weights, backend, and inputs — only the graph
representation differs.

```text
PyTorch model
    → torch.export
    → typed CatOpt IR
    → e-graph / equality saturation   (categorical + algebraic laws)
    → cost-based extraction           (DAG-aware, coordinated)
    → executable PyTorch module
    → torch.compile / TorchInductor
    → benchmark + equivalence check
```

## Results at a glance

Measured on an RTX 2050 (per-iteration `cuda.synchronize`, interleaved
baseline/optimized, lower quartile of 30 reps) and CPU. All rows verified
semantically equivalent.

| Transform family | Examples | GPU | CPU |
|---|---|---|---|
| **FLOP-reducing** (reassociation, weight merging, factorization) | MatrixChain, ParallelLinear, DeepParallel | **1.49–2.51×** | **1.60–6.22×** |
| **Asymptotic reassociation** | LinearAttention `(QK^T)V → Q(K^TV)`: O(T²d) → O(Td²) | **8.0×** at T=2048, d=64 | — |
| **Diagonal absorption** | `repeat_kv` → SDPA `enable_gqa` (llama2.c) | **1.12×** at T=512 | — |
| **Same-FLOP pairing** (fused projections) | SwiGLU gate/up, QKV, GQA, 5-way ParallelBlock | parity at compute-bound; **1.19× launch-bound toy**; 0.82–0.99× real llama2.c blocks | ~1.0× |
| **Same-FLOP conv pairing** | 4× parallel conv1×1 branches | **1.24–1.33× at all batch sizes** | — |
| **Norm folding** | NormLinear | 0.98× (controlled negative) | 0.90× |

**Headline capability:** pointed at unmodified community code —
Karpathy's `llama2.c` Llama 2 implementation — the pipeline
automatically rediscovers `MergedColumnParallelLinear` (w1/w3 fusion)
and `QKVParallelLinear` (asymmetric wq/wk/wv fusion), the transforms
vLLM and TensorRT-LLM implement by hand. Verified to float noise.

## The mechanism: the product law is non-local

`⟨f₁,…,f_k⟩ = (f₁ × … × f_k) ∘ Δ` — pair morphisms by shared domain.
A term-local `lhs → rhs` rewrite can only express this through a
consumer pattern (`mul(l₁,l₂)`, `sdpa(h₁,h₂,h₃)`), which is why
pattern-matching compilers need a handwritten rule per consumer shape
and still cannot generalize.

`pair_shared_input_linears` is a **diagram-level pass**: it groups
`linear` e-nodes by input e-class and offers each member
`splitᵢ(linear(x, cat(W₁,…,W_k)))` — arbitrary arity, asymmetric
output dims, consumer-agnostic. It subsumes the specialized
`swiglu_fuse`, `qkv_fuse`, `qkv_fuse_asym`, `parallel_mul_fuse` rules;
on a PaLM-style parallel block it produces **one GEMM feeding five
uneven split views**, a shape no term-local rule combination reaches.

The cost: extraction stops being locally decomposable — the shared
GEMM amortizes only if *all* members coordinate. `extract_paired`
performs override-based coordinated extraction (members forced to
splits, consumers steered onto member-reaching enodes) and keeps the
result only if true DAG cost beats the greedy term.

## Full measurements

### GPU (RTX 2050, synced timing)

| Program | Equiv | Inductor (ms) | CatOpt (ms) | Speedup |
|---|---:|---:|---:|---:|
| MatrixChain b=4096 | 8e-09 | 0.188 | 0.126 | **1.49×** |
| ParallelLinear b=4096 | 3e-06 | 1.763 | 0.720 | **2.45×** |
| DeepParallel b=4096 | 2e-06 | 5.420 | 2.159 | **2.51×** |
| SwiGLU b=4096 / b=128 | 0.0 / 3e-07 | 12.51 / 0.780 | 12.49 / 0.751 | 1.00× / 1.04× |
| Attention fused QKV b=64 T=256 | 0.0 | 23.71 | 24.47 | 0.97× |
| GQA fused QKV b=64 T=256 | 0.0 | 17.42 | 17.99 | 0.97× |
| TransformerBlock b=64 T=512 | 0.0 | 169.8 | 174.8 | 0.97× |
| ParallelBlock b=64 T=256 (5 proj → 1 GEMM) | 5e-07 | 73.95 | 75.12 | 0.98× |
| **ParallelBlock b=4 T=64 (launch-bound)** | 5e-07 | 1.193 | 1.004 | **1.19×** |
| NormLinear b=256 | 8e-06 | 4.29 | 4.38 | 0.98× |

Kernel-count evidence (`torch.profiler`, 10 forwards of an attention
block): original issues 40 GEMM + 10 SDPA calls; optimized issues
**20 GEMM + 10 SDPA** — fused QKV halves the GEMM count mechanically.

### CPU

| Program | Equiv | Inductor (ms) | CatOpt (ms) | Speedup |
|---|---:|---:|---:|---:|
| MatrixChain b=128 / b=4096 | 7e-09 / 5e-09 | 0.048 / 0.240 | 0.030 / 0.039 | **1.60× / 6.22×** |
| DeepParallel b=4096 | 2e-06 | 1.362 | 0.451 | **3.02×** |
| ParallelLinear b=4096 | 3e-06 | 0.878 | 0.393 | **2.24×** |
| SwiGLU b=128 / b=4096 | 0.0 / 6e-08 | 2.800 / 94.93 | 2.642 / 97.18 | 1.06× / 0.98× |
| Attention QKV b=64 T=256 | 3e-08 | 157.1 | 166.7 | 0.94× |
| NormLinear b=256 | 3e-06 | 39.91 | 44.52 | 0.90× |

### Community code, unmodified (llama2.c)

| Module | Found | Verified | GPU b=4 | GPU b=64 |
|---|---|---|---|---|
| `FeedForward` | w1/w3 → 1 GEMM + 2 splits | 1.3e-07 | 1.04× | 1.06× |
| `Attention` | wq/wk/wv → 1 GEMM + uneven splits | 3.3e-07 | 1.00× | 0.97× |
| `TransformerBlock` | 4 pairing groups in one pass | 4.8e-07 | 0.97× | 0.98× |

Stacked N-layer llama2.c blocks (pairing pass only, no saturation):
**linear scaling** — n=1…16 layers, 2 pairing groups per layer
(QKV + gate/up), all verified ≤2.4e-06, pipeline time dominated by
`torch.export`, not the optimizer.

| llama2.c stacked, GPU | b=1 T=1 | b=1 T=8 | b=4 T=64 | b=64 T=256 |
|---|---|---|---|---|
| 4 layers | 0.89× | 0.82× | 0.95× | 0.99× |
| 2 layers b=1 T=1 / 1 layer b=1 T=1 | 0.90× | | 0.88× | |

**The decode-regime hypothesis was falsified on real blocks.** Pairing
pays where *projections dominate* the kernel count (toy ParallelBlock:
1.19× at b=4). In a real transformer block, RoPE + SDPA + norms +
residuals contribute ~40 kernels per layer — fusing 3 GEMMs saves ~2
launches of ~40 while handing strided views to downstream reshapes.
The honest regime boundary is sharper than "launch-bound wins":
pairing needs launch-bound **and** GEMM-dominated to pay.

**Conv pairing is the exception — it pays at every size measured**
(1.24–1.33× on 4 parallel conv1×1 branches, b=1…64). Inductor does not
fuse cuDNN conv calls, so 4→1 is genuine kernel reduction, and one
wide conv reuses the input better than four narrow ones. The same
diagram-level pass produces it — `pair_shared_input` is now generic
over projection signatures (linear, conv2d), with a compat cluster key
per op (conv members must share stride/padding/dilation/groups and
kernel dims; asymmetric out-channels supported).

**Two nonlinear-boundary transforms landed.** (a) *Diagonal
absorption*: `unsqueeze→expand→reshape` before SDPA is the copy map
Δ (llama2.c's `repeat_kv`); the `gqa_absorb_repeat` rule pushes the
duplication inside the kernel via `enable_gqa`, deleting the
materialisation — discovered automatically on unmodified community
code, 1.12× at T=512 (the saving scales with T). (b) *Linear-attention
reassociation*: `(QK^T)V` → `Q(K^TV)` is the O(T²d) → O(Td²) identity
the linear-transformer literature is built on — the e-graph finds it
from associativity alone and the cost model picks it by shape:
**8.0× at T=2048, d=64**, verified exact in fp64 (rel err 8e-16).
Neither transform is expressible by Inductor: one requires recognising
that a view chain equals a kernel flag, the other requires reordering
matmul composition — both outside local pattern fusion.

**The calibrated cost model predicts the crossover.** `roofline_cost`
constants are measured on the target GPU (2.5 TFLOPS, 89 GB/s,
8.7 µs launch). Predicted vs measured direction agrees on all tested
cases; at the boundary the magnitude is right (ParallelBlock b=4:
predicted 1.08×, measured 1.08×). With `cost_fn=roofline_cost` the
pipeline *accepts* pairing where it wins and *declines* it on real
llama2.c blocks where it loses — per-shape, cost-driven selection.
Residual gap: the model prices the transform, not the ~5% `IRModule`
lowering overhead visible on real blocks.

## What the experiments establish

- **Inductor genuinely misses these transforms** — measured, not
  assumed (up to 3.35× headroom on DeepParallel, within 11% of a
  hand-derived reference). They require creating new parameters, which
  is outside kernel fusion's capability class.
- **Value splits cleanly by regime.** FLOP-reducing algebraic laws pay
  on every backend. Same-FLOP pairing pays where launches dominate
  (small-batch serving — the regime where fused QKV is standard
  practice) and is free otherwise. The cost model, not a hard rule,
  decides per shape.
- **NormLinear is the controlled negative**: Inductor already fuses
  `x·rms·wn` into the GEMM's input read, so graph-level norm folding
  loses on both backends — graph restructuring cannot promise a
  bandwidth win that intra-kernel fusion already delivers.
- **The verifier is load-bearing.** It caught four real bugs: a
  matcher that didn't enforce repeated-metavariable equality (would
  have emitted a false proof), broadcast-shape misinference that
  fabricated a 1.98× "win", unchecked scale-metavariable binding that
  produced a well-typed but semantically wrong program (diff 9.83), and
  a benchmark that measured CUDA submission time instead of execution
  (fabricating 1.10–1.20× GPU "wins"). All regression-tested.
- **Scaling required three DAG-aware fixes** — `add_term`, cost fns, and
  lowering's `_uses_input`/`collect` all used unmemoised tree walks that
  are exponential on shared-subterm DAGs (llama2.c n≥4 hung for
  minutes; now linear). Saturation itself remains the scaling wall:
  AC/distributive rules explode combinatorially on real graphs, so the
  profitable path at scale is the O(n) pairing pass + bounded
  saturation, not full eqsat.

## Honest limitations

- **Nothing found yet is novel to practitioners.** Fused QKV, merged
  gate/up, `enable_gqa` absorption, and the linear-attention identity
  are all known — the contribution is automatic discovery + formal
  verification + cost-driven choice, including one transform with
  *asymptotic* (not constant-factor) impact.
- **The FLOP-reducing wins are degenerate cases** — linear-only DAGs
  collapse to one linear, which a domain expert writes in one line.
  What is demonstrated is that Inductor misses them and the derivation
  is error-prone by hand (the verifier caught transpose-order mistakes
  twice).
- **Weight folding is inference-only** — folding destroys per-layer
  gradients; sound only on frozen graphs.
- **Pairing covers `linear` and `conv2d`** — `matmul`+bias, grouped
  convs, and learned-scale norms are not yet pairable; the mechanism is
  generic (per-op compat cluster key).
- **Reassociation applies only to unnormalised attention** — softmax in
  the middle blocks the `(QK^T)V → Q(K^TV)` law, so it benefits
  linear-attention-style models and generic 3-matmul chains, not
  standard softmax attention.
- **`enable_gqa` absorption requires the exact repeat pattern** —
  `unsqueeze→expand→reshape` merging the head dim; other duplication
  shapes (repeat vs repeat_interleave orderings) are correctly rejected
  but not optimised.
- **`roofline_cost` is calibrated to this GPU** (RTX 2050) — other
  hardware needs the constants re-measured; and it prices the
  transform, not the ~5% `IRModule` lowering overhead on real blocks.
- **Small absolute timings** on a 4 GB mobile GPU; magnitudes should
  not be extrapolated to datacenter hardware.

## Repository layout

| Path | Role |
|---|---|
| `catopt/ir.py` | Typed term algebra, symmetric-monoidal generator registry |
| `catopt/egraph.py` | Union-find, e-matching, saturation, attr metavariables, `check`/`derive` hooks, DAG-aware + coordinated extraction |
| `catopt/rules.py` | 35 laws + `pair_shared_input_linears` non-local pass |
| `catopt/cost.py` | `count_cost`, `flops_cost`, `launch_aware_cost`, `roofline_cost`, `dag_cost` |
| `catopt/torch_bridge.py` | `torch.export` → IR, IR → `IRModule`, compile-time weight folding |
| `catopt/optimize.py` | `optimize_model` pipeline with equivalence verification |
| `catopt/models/` | Benchmark modules (incl. llama2.c-compatible blocks) |
| `main.py`, `bench_gpu.py` | Demos and benchmark drivers |
| `tests/` | 80 tests: equivalence, soundness regressions, pairing |

## Reproduce

```bash
python main.py                     # full demo: all transform families
python main.py --large-batch 4096  # large-batch timing
python -m pytest tests/ -q         # 87 tests
python bench_gpu.py                # GPU table (requires CUDA)
```

## References

- [Inductor passes — PyTorch dev discuss](https://dev-discuss.pytorch.org/t/inductor-passes/2742)
- [RFC: Polyhedral optimization pass for Inductor](https://dev-discuss.pytorch.org/t/rfc-polyhedral-optimization-pass-for-pytorch-inductor/3341)
- [Equivalence Hypergraphs: DPO Rewriting for Monoidal E-Graphs (LICS 2025)](https://doi.org/10.1109/LICS65433.2025.00023)
- [Rewriting for Traced Monoidal Closed Categories (UCL)](https://discovery.ucl.ac.uk/id/eprint/10211429)
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — community model used verbatim
