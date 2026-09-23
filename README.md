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
| **Attention fold** (flash-attention transform) | `softmax(masked qk^T·s) @ v` → `sdpa(is_causal)` — nanoGPT eager path | **1.8–4.6×** eager; **1.1–2.5×** under Inductor | — |
| **Asymptotic reassociation** | LinearAttention `(QK^T)V → Q(K^TV)`: O(T²d) → O(Td²) | **8.0×** at T=2048, d=64 | — |
| **Parallel-scan discovery** (affine monoid) | `h_t = A h_{t-1} + x_t` → balanced Blelloch tree; fires on input-dependent selective SSMs (`A_t = I+Δ_t·A`) | **2.8×** level-batched; **6.3×** CUDA-graph at T=64 | — |
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

**Three nonlinear-boundary transforms landed.**
(a) *Attention fold* — the flash-attention transform:
`softmax(masked_fill(qk^T·s, mask, −inf)) @ v` is literally SDPA's
semantics, so the fold is sound for *any* mask: boolean fill-masks
invert to keep-masks, additive masks pass straight through. A
post-extraction pass then evaluates the (parameter-only) mask and, if
it is exactly lower-triangular, replaces it with `is_causal=True` —
no mask op at all. On **unmodified nanoGPT** eager attention:
**4.6× vs eager, 2.5× under Inductor at T=2048** — Inductor's 17
SDPA patterns miss the `masked_fill` form (verified in its generated
code: `bmm` + fused softmax + `bmm`). (b) *Diagonal absorption*:
`unsqueeze→expand→reshape` before SDPA is the copy map Δ (llama2.c's
`repeat_kv`); `gqa_absorb_repeat` pushes it inside the kernel via
`enable_gqa`, 1.12× at T=512. (c) *Linear-attention reassociation*:
`(QK^T)V` → `Q(K^TV)`, O(T²d) → O(Td²) — **8.0× at T=2048**,
verified exact in fp64 (rel err 8e-16).

**The equivalence class is enumerable — the discovery-engine view.**
`discover_alternatives(model, x)` runs the same pipeline but returns
the top-k cheapest *distinct* members of [G] instead of one winner,
plus `rule_fires` provenance and `diverse_classes` (e-classes holding
structurally different but provably-equal programs — e.g.
`sdpa(transpose³,logical_not)` ≡ `matmul(softmax,transpose)`).
Novelty has levels, and the frontier report makes them inspectable:

- **Level 1–2** (known transform / generalization): fused QKV,
  merged gate/up, conv pairing, `enable_gqa`, flash fold —
  all demonstrated on unmodified community code.
- **Level 3** (emergent composition of laws): llama2.c Attention
  extracts a term combining `split(linear)` pairing enodes **and**
  `enable_gqa` sdpa — "fused QKV with internal head-broadcast".
  No single rule encodes it; it arises from product law ∘ diagonal
  absorption via coordinated extraction. Likewise
  `naturality_scalar` ∘ `sdpa_fold` in nanoGPT.
- **Level 4** (transform nobody encoded): not yet — every
  *result* remains practitioner-known even when the *derivation*
  is emergent.  But the language-boundary claim is now *measured*:
  an unrolled LTI recurrence `h_t = A h_{t-1} + x_t` (SSM-style
  fold) under pure matmul/add laws (distribute + assoc, no comm —
  commutativity is the explosive law: T=16 saturates at 883 enodes
  without it vs 112k+ with) plateaus at **1.5·T critical-path
  depth** — the balanced scan is unreachable because the pair
  (partial-product, partial-sum) is a cross-class object no term
  law synthesises.  Lift the steps into the **affine-map monoid**
  — `aff(A,b)`, `aff_compose` (the (A,b)∘(C,d)=(A·C, A·d+b) law),
  `apply` — and *the same associativity law alone* reaches the
  balanced Blelloch tree: **depth 2T → ~2·log₂T** (T=64: 128→12),
  fp64-exact.  Same laws, richer domain, asymptotically different
  reachable set — that is the "search cannot exceed its language"
  thesis demonstrated, not asserted.  And the payoff is real now:
  `scan_lower.BatchedScanModule` packs each affine map into a
  homogeneous matrix `[[A,b],[0,1]]` — aff_compose becomes a
  **batched matmul per tree level** — so the discovered tree
  executes in ~log T launches: **2.8× faster than the sequential
  recurrence** (6.3× with CUDA-graph capture) at T=64, d=32.
  It fires unprompted on **input-dependent selective dynamics**
  (`SelectiveSSM`: `A_t = I + Δ_t·A`, `B_t = B_θ(x_t)` — the
  Mamba-style core): depth 70→17 at T=32, fp64-exact.  And the
  Mamba-faithful elementwise form `a_t⊙h + b_t⊙x_t` is covered by
  the **diagonal-affine carrier** `aff_diag`/`affd_compose`/
  `applyd` (`SCAN_DIAG_LAWS`) — the same monoid restricted to
  diagonal linear parts, O(d) per compose instead of O(d³).

- **The same mechanism discovers chunked attention** (nonlinear
  recurrence): the online-softmax monoid `om(m,l,a)` — running
  max, exp-sum, weighted numerator — composes by the
  FlashAttention combine law, and `om_elem` is a *monoid
  homomorphism* over concat'd key blocks (`catopt/om.py`).  Dense
  `softmax(q·cat(kᵢ)ᵀ)@cat(vᵢ)` splits into
  `om_apply(⊕ᵢ om_elem(...))` — the streaming/chunked
  decomposition falls out of homomorphism + associativity, with
  no `flash_attention` rule written.  fp64-verified 8.9e-15.
  `om_lower.BatchedOMModule` level-batches the om tree (batched
  `om_elem` scores collapse to one dense `q@Kᵀ` GEMM): up to
  **4.4× over the serial carrier eval** in launch-bound regimes,
  and within **1.13–1.6× of dense** with the optional Inductor
  compile path.  Honest caveat: dense `matmul+softmax+matmul`
  (and certainly fused `sdpa`) still wins outright — chunked
  attention is *reachable and near-parity*, not yet a win; the
  payoff regime is bounded-memory/streaming execution.

**2-morphisms are first-class data; 3-morphisms are computed, not
stored.** Every `union` records a `ProofEdge` witness and every
instantiated enode carries its creating rule (`track_proofs`,
~4% overhead). `certificate(src, dst)` reconstructs an ordered
positional derivation and `verify_certificate` replays each step
on real terms independent of the e-graph — *derivational*
equivalence, not just numerical agreement. 100% of merges are
replayable on rule-driven cases; non-local passes (the pairing
pass) are honestly flagged `egraph_dependent`, and `strict=True`
surfaces exactly those gaps.

**Coherence stratification eliminates the saturation wall.**
`meta.py` classifies laws as *coherent* (assoc/comm/id — the
spaces they generate are contractible, so a canonical form
suffices) vs *contentful* (distribute/lift/fold — saturate
these). `canonicalize` computes normal forms eagerly (balanced
bracketings — the Blelloch shape *is* the canonical form), then
only contentful laws run. On the T=8 recurrence: **2,011,701 →
351 enodes (~5,700×), 182s → 0.05s**, same fp64-exact result.
This is Mac Lane coherence operationalized as a scheduler —
empirically it is the difference between unusable and instant.

**Rules synthesize themselves.** `meta.synthesize_rules` does
critical-pair completion: compose ordered rule pairs symbolically
or on seed terms, filter tautologies/duplicates, validate each
candidate by replay + fp64 evaluation. Fed
`SCAN_LAWS \ {aff_lift_step}`, it emits the unfolded equivalent
of `AFF_LIFT_STEP` — a rule previously hand-written is now
derived. This is the meta-optimization loop: certified composite
paths distill back into the law set.

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
  minutes; now linear). Saturation scaling has a name now —
  **commutativity is the explosive law** (permutation space): the
  order-preserving fragment saturates in O(T²) enodes where the
  full AC set blew past 100k.  The profitable paths at scale are
  the O(n) pairing pass, order-preserving law sets, and monoid
  domains (`aff`, `om`) that move structure into carriers.

## Honest limitations

- **Nothing found yet is novel to practitioners.** Fused QKV, merged
  gate/up, `enable_gqa` absorption, flash attention, and the
  linear-attention identity are all known — the contribution is
  automatic discovery + formal verification + cost-driven choice,
  including transforms with *asymptotic* (not constant-factor) impact.
- **SDPA-fold coverage is bounded** — mul/div score scaling,
  masked_fill and additive masks, optional eval-mode dropout;
  `is_causal` specialization additionally requires the mask to be
  parameter-only and exactly lower-triangular (verified by evaluating
  it). Other mask constructions fold to `attn_mask` but not
  `is_causal`.
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
| `catopt/egraph.py` | Union-find, e-matching, saturation, attr metavariables, `check`/`derive` hooks, DAG-aware + coordinated extraction, proof-carrying merges (`certificate`/`verify_certificate`) |
| `catopt/meta.py` | Coherence stratification (`canonicalize`, `stratified_run`) + critical-pair rule synthesis (`synthesize_rules`) |
| `catopt/rules.py` | 35 laws + `pair_shared_input_linears` non-local pass |
| `catopt/cost.py` | `count_cost`, `flops_cost`, `launch_aware_cost`, calibrated `roofline_cost`, `depth_cost` (critical path), `dag_cost` |
| `catopt/torch_bridge.py` | `torch.export` → IR, IR → `IRModule`, compile-time weight folding |
| `catopt/optimize.py` | `optimize_model` pipeline with equivalence verification |
| `catopt/om.py` | Online-softmax monoid laws (chunked attention) |
| `catopt/om_lower.py` | Level-batched chunked-attention executor + CUDA graphs/compile |
| `catopt/scan_lower.py` | Level-batched parallel-scan executor + CUDA graphs |
| `catopt/models/` | Benchmark modules (llama2.c blocks, `ssm.py` selective SSMs) |
| `main.py`, `bench_gpu.py` | Demos and benchmark drivers |
| `tests/` | 185 tests: equivalence, soundness, pairing, monoid domains |

## Reproduce

```bash
python main.py                     # full demo: all transform families
python main.py --large-batch 4096  # large-batch timing
python -m pytest tests/ -q         # 185 tests
python bench_gpu.py                # GPU table (requires CUDA)
```

## References

- [Inductor passes — PyTorch dev discuss](https://dev-discuss.pytorch.org/t/inductor-passes/2742)
- [RFC: Polyhedral optimization pass for Inductor](https://dev-discuss.pytorch.org/t/rfc-polyhedral-optimization-pass-for-pytorch-inductor/3341)
- [Equivalence Hypergraphs: DPO Rewriting for Monoidal E-Graphs (LICS 2025)](https://doi.org/10.1109/LICS65433.2025.00023)
- [Rewriting for Traced Monoidal Closed Categories (UCL)](https://discovery.ucl.ac.uk/id/eprint/10211429)
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — community model used verbatim
