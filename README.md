# catopt

Verified search over semantics-preserving rewrites of neural-network
computation graphs. catopt lifts a model to a typed IR, saturates an
e-graph with equational laws, extracts a cheaper member by cost model,
and returns a program with a replayable equivalence certificate.

The engine (`catopt-core`) is backend-agnostic — IR, e-graph, laws,
and cost algebra run with zero dependencies. PyTorch is the shipped
integration through `Source`/`Sink` ports; `torch.compile`/Inductor is
used as the benchmark baseline and the torch sink's codegen. Target
pricing is a pluggable cost model (`calibrate()` +
`roofline_cost_for`); nothing in the engine names CUDA, Triton, or
torch.

```text
model (any Source)
  → typed IR                  (frontend port: TorchSource shipped)
  → e-graph saturation        (equivalent programs, enumerated)
  → cost-based extraction     (the cheapest member, for your backend)
  → verified program          (certificate replayed, fp64-checked)
  → backend lowering          (Sink port: TorchSink → torch.compile)
```

```python
from catopt.optimize import optimize_model

opt, report = optimize_model(model, example_input)
out = opt(x)          # equivalent to model(x), certificate-backed

opt, report = optimize_model(model, example_input, compile=True)
# ^ torch.compile is applied to the delivered module; report["compiled"]
```

Deep stacks use `optimize_compositional`, which optimizes each block
against its captured real input and recomposes with per-block
verification and automatic fallback.

## Mechanism

A program optimizer's reachable set is bounded by its semantic
language, not its search strategy. Tensor IRs rewrite *ops*; catopt
lifts programs into *carriers* — monoid objects where the same
associativity law reaches structures no op-level pattern expresses:

- **Affine monoid `aff(A,b)`** — an unrolled recurrence
  `h_t = Ah_{t-1} + x_t` under matmul/add laws plateaus at depth ~1.5T;
  the balanced scan is unreachable because the partial-product pair is
  a cross-class object no term law synthesizes. In the carrier, the
  same associativity law produces the Blelloch tree: depth 2T →
  ~2·log₂T. `lift_scan_to_applyd` constructs the carrier member
  directly from the recognized recurrence spine — a nonlocal pass,
  since carrier-law saturation is combinatorially explosive at long
  horizons.
- **Online-softmax monoid `om(m,l,a)`** — FlashAttention's combine
  follows from homomorphism + associativity; no `flash_attention` rule
  is written.
- **Product law `⟨f₁,…,f_k⟩ = (×fᵢ)∘Δ`** — `pair_shared_input_linears`
  groups shared-input projections at diagram level: one GEMM feeding k
  uneven split views, arbitrary arity, consumer-agnostic.
- **`trace^U`** (traced monoidal structure) — recurrences are
  fixpoints; the Joyal–Street–Verity axioms are rewrites, so loops
  become closed-form resolvents `P + Q(I−S)⁻¹R`.
- **Cross-carrier laws** (`xcarrier.py`) — readouts push through scan
  evaluation; scans fold inside softmax elements; attention over
  scanned values stays one recurrence.

Executors lower carrier trees to level-batched kernels
(`scan_lower.py`, `om_lower.py`, `omd_lower.py`). The IR carries
`Param` leaves as first-class terms; a weight no extracted member
references drops out of the state dict — folding, tying, and dedup
are the same event.

## Results

All rows verified semantically equivalent (fp64 where stated); RTX
2050, synced timing.

| Transform | Example | Result |
|---|---|---|
| Projection pairing (QKV, gate·up, parallel branches → 1 GEMM) | PaLM-style block, 5 projections fused | 1.24× vs Inductor end-to-end |
| FLOP reduction (reassociation, weight merging, factorization) | DeepParallel b=4096 | 2.51× GPU / 3.02× CPU |
| Asymptotic reassociation | `(QKᵀ)V → Q(KᵀV)`: O(T²d)→O(Td²) | 8.0× at T=2048 |
| Weights-first fold (k-deep chain → 1 GEMM) | `x @ W₁@…@W₁₆` — Inductor's post-grad graph keeps all k `mm`s | 15.9× vs Inductor GPU, k=16 |
| Scan lift on real blocks | RetNet/GLA/delta-rule — `optimize_model` selects + delivers the carrier | 3× vs eager CPU+GPU at T=128; 3.9× composed (`compile=True`); the only schedule produced at GLA T=2048 |
| Attention fold | `softmax(masked_fill(qkᵀ·s)) @ v` → `sdpa(is_causal)` | 4.6× vs eager, 2.5× under Inductor (nanoGPT, T=2048) |
| Parallel-scan discovery | LTI recurrence → balanced Blelloch tree | 6.3× CUDA-graph, T=64 |
| Diagonal-affine scan | `a⊙h + b⊙x` (Mamba-faithful) | 4.4× CUDA-graph, T=64 |
| Streaming attention | om monoid: O(1) state per KV block | 260× vs sdpa-recompute at 65k cache; 89 MiB flat at 2M keys |
| Conv pairing | 4 parallel conv1×1 → 1 conv | 1.24–1.33×, all batch sizes |
| Diagonal absorption | `repeat_kv` → SDPA `enable_gqa` (llama2.c) | 1.12× at T=512 |

On unmodified community code (Karpathy's `llama2.c`) it rediscovers
`MergedColumnParallelLinear` and `QKVParallelLinear` — the transforms
vLLM and TensorRT-LLM implement by hand.

## Regime map

Measured, including the losses:

| Regime | Outcome |
|---|---|
| FLOP-reducing rewrites (assoc/fold/factorize) | Wins — pure work reduction |
| Nonlinear-boundary folds (attention fold, reassociation) | Wins — up to 8×, asymptotic |
| Carrier lifts (scans, streaming attention) | Wins in their regime — depth and memory, not raw latency |
| Conv pairing | Wins — Inductor never fuses cuDNN calls |
| GEMM pairing on transformer blocks | Parity — ~40 non-GEMM kernels/layer dilute it |
| Real trained checkpoints (stories15M/110M) | Parity — all blocks transform and verify, no win at these sizes |
| Scan lift on linear-attention blocks | 3× vs eager delivered end-to-end; 3.9× composed with `compile=True`; loses to a compiled Inductor where it can compile |
| Inductor compile wall on unrolled recurrences | Inductor's compile grows superlinearly in T (54–85s at T=2048; GLA T=2048 exceeds 60s timeout). CatOpt produces a certified O(log T) schedule there — but its own pipeline is slower than Inductor's compile where Inductor succeeds (262s at T=2048) |
| Launch-bound decode cells (B=1, T≤64) | Loses 4–15% — split-view copies cost more than saved launches |
| Large cells (B≥8, T≥128, stories110M) | Parity — GEMM-shape efficiency washes out at ~1% |

Real checkpoints (`bench/stories15m_bench.py`): stories15M and
stories110M pass the full pipeline — 8/8 and 14/14 blocks optimize,
QKV + gate·up fuse, outputs verify to ~2e-5 — at parity with Inductor
(3.0 ms / 17.0 ms both ways). The mechanism is real (−37% GEMM
launches, profiler-verified); at these dimensions it does not pay.
The launch-bound hypothesis was falsified on blocks, on whole models,
and on the large-cell crossover sweep (`bench/decode_bench.py`).

## Cost-model fidelity

`bench/cost_fidelity.py` measures predicted-cost vs measured-latency
rank correlation. Findings and the fixes they drove:

- ρ ≈ 0.95 in the win regime (k-chain) — the model orders the
  frontier correctly and picks the winner.
- ρ ≈ −0.3 on scan blocks — inverted. Same term, same predicted cost,
  6–30× latency spread across lowerings: Inductor's fused kernel is
  rank-1 measured but priced like the generic per-leaf evaluator.
- Fixed: **executor-aware pricing** (`executor_overhead` /
  `executor_cost_for`, calibrated `dispatch_us`/`leaf_eval_us`);
  **lowering routing** (carrier-apply roots lower through their
  level-batched executor, `stats["lowering"]`, generic fallback);
  **honest param-only discount** (a `trace` resolvent member hid a
  14.5s `linalg.solve` inside a nominally compile-time-free subtree;
  only genuinely foldable subtrees now bill zero, and solver ops carry
  a surcharge).
- Carrier selection is nonlocal: additive extraction can't price a
  batched spine, so `_carrier_upgrade` force-extracts root-class
  carrier enodes and compares delivered prices — each term billed
  under the executor it would route to.
- Residual gap: ρ ≈ 0.3 on scan blocks — identical terms run 0.043ms
  Inductor vs ~0.9ms generic eval; structural cost can't predict
  realized kernel fusion. Pricing compiled lowerings by fusion
  regions is open work.
- Constraint: extraction requires additive cost functions —
  `min`-over-lowerings is non-additive and corrupts `extract_best`'s
  local-cost decomposition, so it serves reporting, not selection.

## Verification

Every extracted program carries an ordered, replayable derivation
`original → optimized`; `verify_certificate` re-checks it on real
terms — derivational equivalence, not numerical spot-checks.
Non-local passes attach pointwise witnesses so constructed members
replay standalone.

The verifier has caught: a matcher that skipped repeated-metavariable
equality (a would-be false proof), a broadcast-shape misinference that
fabricated a 1.98× "win", a well-typed but semantically wrong program
(diff 9.83), and a benchmark measuring CUDA submission time instead of
execution. All regression-tested.

Saturation scales by coherence stratification (2.0M → 351 enodes on
the T=8 recurrence; 4-layer models 768s → 4.9s; ≥8 blocks monolithic);
`optimize_compositional` is the deeper path.

## Limitations

- **Compute, not weights** — weights are compile-time constants folded
  into the graph (constant folding, not reparameterization). The
  weight-space axis (INR/Kronecker/monarch) was falsified and archived
  in `project/retros/`; the ε approximation toolkit (`catopt-eps`)
  moved off main to the `weight-eps` branch.
- **Nothing discovered is novel to practitioners** — fused QKV, flash
  attention, the linear-attention identity are known. The contribution
  is automatic discovery + verification + per-shape choice, not new
  math.
- **Inference only** — weight folding destroys per-layer gradients;
  backward-graph rewriting (AOTAutograd) is unimplemented.
- **Chunked attention loses to fused SDPA head-to-head** when K,V fit
  on device — the om win is bounded memory and incremental state, not
  throughput.
- **Coverage gaps** — `matmul`+bias and grouped convs aren't pairable;
  reassociation needs unnormalized attention; masks must arrive
  materialized.
- **RTX 2050 (4 GB) numbers** — `calibrate()` makes re-targeting
  mechanical; magnitudes don't extrapolate to datacenter hardware.

## Layout

uv workspace monorepo — the engine is split into domain distributions
so each carries only its own dependencies:

```
packages/catopt-core/       the torch-free semantic engine (zero deps)
  ir, attrs, typing         term IR, attr schemas, shape inference
  egraph/                   union-find, matching, saturation, proof
  laws/                     equational laws + non-local pairing passes
  cost, meta, rulecache     cost algebra, rule synthesis, law cache
  ports, ops                Source/Sink + protocol boundaries, OpTable
packages/catopt-torch/      PyTorch adapters (deps: core + torch)
  adapters                  TorchSource / TorchSink (the ports)
  torch_bridge              torch.export → IR → executable module
  executors/                BatchedExecutorBase + level_schedule
  models/, report           model zoo; OptReport + verify gate
packages/catopt-carriers/   semantic carriers (deps: core + torch)
  om, xcarrier, trace       online-softmax / deferred / traced monoids
  *_lower, trace_lift       lowerers + the non-local lift passes
packages/catopt-optimize/   pipeline orchestrators (deps: all above)
  optimize, regime, calibrate
catopt/                     façade — public API + compat aliases
bench/                      benchmarks: stories15M/110M checkpoints,
                            decode sweep, reassoc_scale,
                            real_linear_attn (scan lift, CPU+CUDA),
                            cost_fidelity (predicted vs measured)
tests/                      test suite
project/                    orphan branch: plans, ADRs, retrospectives
```

`catopt-core` installs standalone (`pip install -e
packages/catopt-core`) — zero dependencies.

**Ports.** A `Source` lifts a model to IR (`model → (IR, leaf
values)`); a `Sink` lowers IR to a runnable, owns its op set, and owns
the equivalence gate. `TorchSource`/`TorchSink` are the defaults for
`optimize_model` and `discover_alternatives` (`source=`/`sink=`
override). A sink's `supported_ops` bounds extraction: members using
ops the backend can't lower price at `+inf` (`backend_cost`), so the
optimizer only commits to executable forms. A new backend implements
`Sink`; `catopt-core` never imports it.

```bash
uv sync                       # installs all workspace members editable
python bench/fetch.py         # checkpoints → ~/.cache/catopt
python bench/stories15m_bench.py --device cuda
python bench/decode_bench.py --device cuda --quick
python bench/real_linear_attn.py --device cuda --quick
python -m pytest tests/ -q
```

## References

- Willsey et al., *egg: Fast and Extensible Equality Saturation*
  (POPL 2021) — the e-graph / equality-saturation machinery.
- Blelloch, *Prefix Sums and Their Applications* (1990) — the scan
  structure the affine carrier reaches.
- Dao et al., *FlashAttention* (NeurIPS 2022) — the om-monoid combine
  is this recurrence, derived rather than encoded.
- Joyal, Street & Verity, *Traced Monoidal Categories* (1996) — the
  `trace` structure used for the closed-form resolvent.
- Sun et al., *Retentive Network* (2023); Yang et al., *Gated Linear
  Attention* (ICML 2024); Yang et al., *Parallelizing Linear
  Transformers with the Delta Rule* (NeurIPS 2024); Gu & Dao, *Mamba*
  (2023) — the linear-attention/SSM families the scan benchmarks
  instantiate.
- Karpathy, *llama2.c* — the unmodified codebase the pairing rules
  rediscover vLLM/TensorRT-LLM-style merged projections on.
- Paszke et al., *PyTorch*; `torch.compile`/Inductor — the shipped
  `Source`/`Sink` pair and the benchmark baseline.
