# catopt

**A categorical-semantics compiler for neural-network graphs: equality
saturation over semantic carriers, with proofs.**

`catopt` translates PyTorch models into a typed symmetric-monoidal IR,
explores semantics-preserving rewrites with an e-graph, extracts a
lower-cost program, and lowers it back through
`torch.compile`/TorchInductor. Every comparison below uses the same
model, weights, backend, and inputs — only the graph representation
differs.

```text
PyTorch model
    → torch.export
    → typed CatOpt IR                    (1-morphisms)
    → e-graph / equality saturation      (2-morphisms as data)
    → coherence stratification           (3-morphisms computed, not stored)
    → cost-based extraction + certificate
    → executable PyTorch module
    → torch.compile / TorchInductor
    → benchmark + equivalence check
```

## The claim

> A program optimizer's reachable set is bounded by its semantic
> language, not by its search strategy.

Tensor-level IRs (and pattern-matching compilers like Inductor) can only
express rewrites over ops. `catopt` **lifts programs into semantic
carriers** — monoids, diagonal maps, online-softmax states — where the
*same associativity law* reaches structures the op-level language
provably cannot. This is measured, not asserted:

- An unrolled LTI recurrence `h_t = A h_{t-1} + x_t` under pure
  matmul/add laws plateaus at **1.5·T critical-path depth** — the
  balanced scan is unreachable because the pair (partial-product,
  partial-sum) is a cross-class object no term law synthesises.
- Lift the steps into the **affine-map monoid** `aff(A,b)` and *the same
  associativity law alone* reaches the balanced Blelloch tree:
  **depth 2T → ~2·log₂T**, fp64-exact.
- The online-softmax monoid `om(m,l,a)` derives **FlashAttention's
  combine as a law**: `softmax(q·cat(kᵢ)ᵀ)@cat(vᵢ)` →
  `om_apply(⊕ᵢ om_elem(...))` falls out of homomorphism + associativity,
  with no `flash_attention` rule written.

The contribution is a **discovery engine**: laws in, verified +
certified + measured transforms out — including structures whose
*derivation* is emergent even when the destination is practitioner-known.

## Semantic carriers

Each carrier is a monoid object in the IR; lifting and lowering rules
connect it to the tensor domain. Associativity in the carrier is what
produces the parallel/decomposed forms.

| Carrier | Maps | Discovers | Wall-clock |
|---|---|---|---|
| `aff(A,b)` dense affine | `h ↦ Ah+b` | Blelloch parallel scan | **6.3×** (CUDA-graph, T=64) |
| `aff_diag(a,b)` diagonal affine | `h ↦ a⊙h+b` | elementwise scan (Mamba-faithful SSMs) | **4.4×** (CUDA-graph, T=64) |
| `om(m,l,a)` online softmax | running max/exp-sum/numerator | chunked/flash attention, streaming KV | memory-feasibility win (below) |
| tensor domain | — | folds, pairing, reassociations | up to **8×** |

Executors (`scan_lower.py`, `om_lower.py`) lower discovered trees into
level-batched GPU kernels: `aff_compose` becomes a batched matmul per
tree level via homogeneous-matrix packing `[[A,b],[0,1]]`; the diagonal
tree batches as elementwise ops; om trees batch `om_elem` scores into a
single `q@Kᵀ` GEMM.

## Higher morphisms

**2-morphisms are first-class data; 3-morphisms are computed, not
stored.** The e-graph is a truncation of the program ∞-groupoid, with a
configurable `truncation_level`:

- **Level 1 — pure quotient.** E-classes only; minimal memory.
- **Level 2 — witnesses.** Every `union` records a `ProofEdge`
  (rule + substitution); `certificate(src,dst)` reconstructs an ordered
  positional derivation and `verify_certificate` replays it on real
  terms — *derivational* equivalence, not numerical spot-checks.
  ~4–10% overhead.
- **Level 3 — lazy coherences.** `all_proofs`/`coherent_paths` enumerate
  *alternate derivations* between two terms on demand (bounded BFS over
  the term-rewriting space). Nothing is stored: coherence is a property
  of the rewriting space, computed when asked, then thrown away.

**Coherence stratification eliminates the saturation wall.** Laws are
classified *coherent* (assoc/comm/id — the spaces they generate are
contractible, so a canonical form suffices) vs *contentful*
(distribute/lift/fold — saturate these). `canonicalize` computes normal
forms eagerly — the Blelloch shape *is* the canonical form — then only
contentful laws run. On the T=8 recurrence: **2,011,701 → 351 enodes
(~5,700×), 182s → 0.05s**, same fp64-exact result. This is Mac Lane
coherence operationalized as a scheduler.

**Rules synthesize themselves.** `meta.synthesize_rules` performs
critical-pair completion: compose rule pairs on seed terms, validate
each candidate by replay + fp64 evaluation. Fed
`SCAN_LAWS \ {aff_lift_step}`, it emits the unfolded equivalent of a
previously hand-written derived rule. Certified composite paths distill
back into the law set — the meta-optimization loop is closed.

## The mechanism: the product law is non-local

`⟨f₁,…,f_k⟩ = (f₁ × … × f_k) ∘ Δ` — pair morphisms by shared domain. A
term-local `lhs → rhs` rewrite can only express this through a consumer
pattern (`mul(l₁,l₂)`, `sdpa(h₁,h₂,h₃)`), which is why pattern-matching
compilers need a handwritten rule per consumer shape and still cannot
generalize.

`pair_shared_input_linears` is a **diagram-level pass**: it groups
`linear` e-nodes by input e-class and offers each member
`splitᵢ(linear(x, cat(W₁,…,W_k)))` — arbitrary arity, asymmetric output
dims, consumer-agnostic. It subsumes the specialized `swiglu_fuse`,
`qkv_fuse`, `qkv_fuse_asym`, `parallel_mul_fuse` rules; on a PaLM-style
parallel block it produces **one GEMM feeding five uneven split views**,
a shape no term-local rule combination reaches. Extraction stops being
locally decomposable — `extract_paired` performs coordinated extraction
and keeps the result only if true DAG cost beats the greedy term.

## Results

Measured on an RTX 2050 (per-iteration `cuda.synchronize`, interleaved
baseline/optimized, lower quartile of 30 reps) and CPU. All rows
verified semantically equivalent.

| Transform family | Examples | GPU | CPU |
|---|---|---|---|
| **FLOP-reducing** (reassociation, weight merging, factorization) | MatrixChain, ParallelLinear, DeepParallel | **1.49–2.51×** | **1.60–6.22×** |
| **Attention fold** | `softmax(masked qk^T·s) @ v` → `sdpa(is_causal)` — nanoGPT eager path | **1.8–4.6×** eager; **1.1–2.5×** under Inductor | — |
| **Asymptotic reassociation** | LinearAttention `(QK^T)V → Q(K^TV)`: O(T²d) → O(Td²) | **8.0×** at T=2048, d=64 | — |
| **Parallel-scan discovery** (affine monoid) | `h_t = A h_{t-1} + x_t` → balanced Blelloch tree; fires on input-dependent selective SSMs | **2.8×** batched; **6.3×** CUDA-graph | — |
| **Diagonal-affine scan** | Mamba-faithful `a_t⊙h + b_t⊙x_t`, O(d)/compose | **4.4×** CUDA-graph at T=64 | — |
| **Streaming / bounded-memory attention** (om monoid) | KV streams beyond VRAM: 89 MiB flat at 2M keys; O(1)/step incremental state | **260×** vs sdpa-recompute at 65k cache | — |
| **Diagonal absorption** | `repeat_kv` → SDPA `enable_gqa` (llama2.c) | **1.12×** at T=512 | — |
| **Same-FLOP pairing** | SwiGLU gate/up, QKV, GQA, 5-way ParallelBlock | parity compute-bound; **1.19×** launch-bound | ~1.0× |
| **Same-FLOP conv pairing** | 4× parallel conv1×1 branches | **1.24–1.33×** at all batch sizes | — |
| **Norm folding** | NormLinear | 0.98× (controlled negative) | 0.90× |

**Headline capability:** pointed at unmodified community code —
Karpathy's `llama2.c` — the pipeline automatically rediscovers
`MergedColumnParallelLinear` (w1/w3 fusion) and `QKVParallelLinear`
(asymmetric wq/wk/wv fusion), the transforms vLLM and TensorRT-LLM
implement by hand. Verified to float noise.

**Hybrid models compose carriers.** On a Jamba-style SSM→attention
block, both carriers coexist in one e-graph (444 enodes, saturates in
0.7s) and extraction produces terms mixing `applyd` and `om_apply`
(fp64-exact). Cross-domain transforms fire unaided: `assoc_linear`
fused attention `out_proj` into the *next* SSM's input projections —
an inter-layer weight merge across the carrier seam.

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

**The decode-regime hypothesis was falsified on real blocks.** Pairing
pays where *projections dominate* the kernel count (toy ParallelBlock:
1.19× at b=4). In a real transformer block, RoPE + SDPA + norms +
residuals contribute ~40 kernels per layer — fusing 3 GEMMs saves ~2
launches of ~40. The honest regime boundary: pairing needs launch-bound
**and** GEMM-dominated to pay. **Conv pairing is the exception** —
1.24–1.33× at every size measured, because Inductor does not fuse cuDNN
conv calls.

**Three nonlinear-boundary transforms.** (a) *Attention fold* —
`softmax(masked_fill(qk^T·s, mask, −inf)) @ v` is literally SDPA's
semantics, so the fold is sound for *any* mask; a post-extraction pass
evaluates the (parameter-only) mask and replaces it with
`is_causal=True` when exactly lower-triangular. On unmodified nanoGPT:
**4.6× vs eager, 2.5× under Inductor at T=2048** — Inductor's 17 SDPA
patterns miss the `masked_fill` form. (b) *Diagonal absorption* —
`unsqueeze→expand→reshape` before SDPA is the copy map Δ;
`gqa_absorb_repeat` pushes it inside via `enable_gqa`. (c)
*Linear-attention reassociation* — `(QK^T)V → Q(K^TV)`, **8.0× at
T=2048**, fp64 rel err 8e-16.

**The equivalence class is enumerable.** `discover_alternatives(model,
x)` returns the top-k cheapest *distinct* members of [G] with
`rule_fires` provenance and `diverse_classes` — e-classes holding
structurally different but provably-equal programs. Novelty levels:

- **Level 1–2** (known transform / generalization): fused QKV, merged
  gate/up, conv pairing, `enable_gqa`, flash fold — all on unmodified
  community code.
- **Level 3** (emergent composition): "fused QKV with internal
  head-broadcast" (product law ∘ diagonal absorption); `out_proj`
  fused into the next SSM's projections in the hybrid model.
- **Level 4** (transform nobody encoded): not yet — every *result*
  remains practitioner-known even when the *derivation* is emergent.
  The machinery where one could appear — monoid domains, completion,
  certificates, frontier enumeration, cross-carrier models — is built.

**The calibrated cost model predicts the crossover.** `roofline_cost`
constants are measured on the target GPU (2.5 TFLOPS, 89 GB/s, 8.7 µs
launch). Predicted vs measured direction agrees on all tested cases; at
the boundary the magnitude is right (ParallelBlock b=4: predicted 1.08×,
measured 1.08×). The pipeline *accepts* pairing where it wins and
*declines* it on real blocks where it loses — per-shape, cost-driven.

## What the experiments establish

- **Inductor genuinely misses these transforms** — measured, not
  assumed (up to 3.35× headroom on DeepParallel, within 11% of a
  hand-derived reference). They require creating new parameters, which
  is outside kernel fusion's capability class.
- **Value splits cleanly by regime.** FLOP-reducing laws pay on every
  backend. Same-FLOP pairing pays where launches dominate and GEMMs
  dominate the kernel count. The cost model, not a hard rule, decides
  per shape.
- **NormLinear is the controlled negative**: Inductor already fuses
  `x·rms·wn` into the GEMM's input read, so graph-level folding loses —
  restructuring cannot promise a bandwidth win that intra-kernel fusion
  already delivers.
- **The verifier is load-bearing.** It caught four real bugs: a matcher
  that didn't enforce repeated-metavariable equality (would have
  emitted a false proof), broadcast-shape misinference that fabricated
  a 1.98× "win", unchecked scale-metavariable binding that produced a
  well-typed but semantically wrong program (diff 9.83), and a
  benchmark that measured CUDA submission time instead of execution
  (fabricating 1.10–1.20× "wins"). All regression-tested.
- **Saturation scaling is now understood and managed** — commutativity
  is the explosive law (permutation space); coherence stratification
  canonicalizes it rather than searching it. The profitable paths at
  scale are the O(n) pairing pass, stratified law sets, and monoid
  carriers that move structure out of the search entirely.

## Honest limitations

- **Nothing found yet is novel to practitioners.** Fused QKV, merged
  gate/up, `enable_gqa`, flash attention, and the linear-attention
  identity are all known — the contribution is automatic discovery +
  formal verification + cost-driven choice, including transforms with
  asymptotic impact.
- **Inference-only.** Weight folding destroys per-layer gradients; all
  wins are forward-pass. Backward-graph rewriting via joint
  (AOTAutograd) graphs is unimplemented future work.
- **Chunked attention loses to fused sdpa head-to-head** whenever K,V
  fit on device (sdpa is already score-bounded; ~1.8–2× latency win for
  sdpa). The om win is *feasibility* — `StreamingOMModule` evaluates the
  om tree as a bounded-memory fold (~57 MiB transient flat in T_kv,
  verified identical to the hand-written benchmark fold) and
  `om_step_qk` gives O(block) incremental state updates (~40× vs
  sdpa-recompute at 65k cache, CUDA-graph capturable). The fixed-query
  incremental mode does not model per-token decode.
- **Masked chunked attention needs a "mask distributes over concat"
  law** — causal masks carry positional offsets that the current IR
  cannot yet express per-block; additive masks compose freely.
- **SDPA-fold coverage is bounded** — mul/div score scaling,
  masked_fill and additive masks, optional eval-mode dropout;
  `is_causal` requires the mask to be parameter-only and exactly
  lower-triangular.
- **The FLOP-reducing wins are degenerate cases** — linear-only DAGs
  collapse to one linear, which a domain expert writes in one line.
  Demonstrated: Inductor misses them and hand derivation is error-prone
  (the verifier caught transpose-order mistakes twice).
- **Pairing covers `linear` and `conv2d`** — `matmul`+bias, grouped
  convs, and learned-scale norms are not yet pairable.
- **Reassociation applies only to unnormalised attention** — softmax
  blocks the `(QK^T)V → Q(K^TV)` law.
- **`roofline_cost` is calibrated to this GPU** (RTX 2050, 4 GB);
  magnitudes should not be extrapolated to datacenter hardware.

## Roadmap

- **Regime-adaptive architecture**: one weight set, multiple certified
  forms — recurrent (decode), parallel-scan (train/prefill), chunked
  (bounded memory). Extract the Pareto frontier across cost models and
  dispatch per deployment regime.
- **Backward-graph rewriting**: joint fwd+bwd (AOTAutograd) graphs —
  the only path to training-side wins; weight-merging laws currently
  destroy gradients.
- **More weight-preserving dualities**: RepVGG-style branch merging,
  conv↔GEMM, head reshaping, MHA↔GQA directions — each a new
  architecture over the same parameters.
- **Masked chunked attention**: "mask distributes over concat" with
  positional offsets — the remaining gap to causal om.
- **Guarded-rule synthesis**: extend completion to `check`/`derive`
  rules so the om/attention lemma library derives itself.

## Repository layout

| Path | Role |
|---|---|
| `catopt/ir.py` | Typed term algebra, symmetric-monoidal generator registry |
| `catopt/egraph.py` | Union-find, e-matching, saturation, `truncation_level` (1–3), proof-carrying merges (`certificate`/`verify_certificate`/`coherent_paths`), DAG-aware + coordinated extraction |
| `catopt/meta.py` | Coherence stratification (`canonicalize`, `stratified_run`) + critical-pair rule synthesis (`synthesize_rules`) |
| `catopt/rules.py` | Laws + `pair_shared_input_linears` non-local pass |
| `catopt/cost.py` | `count_cost`, `flops_cost`, `launch_aware_cost`, calibrated `roofline_cost`, `depth_cost`, `dag_cost` |
| `catopt/torch_bridge.py` | `torch.export` → IR, IR → `IRModule`, compile-time weight folding |
| `catopt/optimize.py` | `optimize_model` pipeline with equivalence verification |
| `catopt/om.py` | Online-softmax monoid laws (chunked/streaming attention) |
| `catopt/om_lower.py` | Level-batched + streaming chunked-attention executors, incremental om state, CUDA graphs/compile |
| `catopt/scan_lower.py` | Level-batched parallel-scan executor (dense + diagonal carriers) + CUDA graphs |
| `catopt/models/` | Benchmark modules (llama2.c blocks, `ssm.py` selective/diagonal SSMs, `hybrid.py` SSM+attention) |
| `main.py`, `bench_gpu.py` | Demos and benchmark drivers |
| `tests/` | 231 tests: equivalence, soundness, pairing, carriers, certificates, truncation, hybrid, streaming |

## Reproduce

```bash
python main.py                     # full demo: all transform families
python main.py --large-batch 4096  # large-batch timing
python -m pytest tests/ -q         # test suite
python bench_gpu.py                # GPU table (requires CUDA)
```

## References

- [Inductor passes — PyTorch dev discuss](https://dev-discuss.pytorch.org/t/inductor-passes/2742)
