# CatOpt — research writeup

Verified search over semantics-preserving rewrites of neural-network
computation graphs. Engine: `catopt-core` (torch-free e-graph, laws,
cost algebra); shipped backend: PyTorch via `Source`/`Sink` ports.
Baseline throughout: `torch.compile`/Inductor, RTX 2050 (4 GB) +
i5-12500H, synced timing, fp64 verification where stated.

## 1. Claim

A program optimizer's reachable set is bounded by its semantic
language, not its search strategy. Tensor-level IRs rewrite *ops*;
rewrites expressed over *categorical structure* — monoid carriers,
traced monoidal fixpoints, products as `⟨f₁,…,f_k⟩ = (×fᵢ)∘Δ` —
reach programs no op-level pattern composes to. Precisely: the
balanced scan of an unrolled affine recurrence `h_t = Ah_{t-1}+x_t`
saturates at depth ~1.5T under matmul/add term laws because the
partial-product pair `(ΠA, ΣAⁱxᵢ)` is a cross-class object; in the
affine carrier the *same* associativity law produces the Blelloch
tree (depth ~2·log₂T). The claim is about reachability under a
fixed law set, not about a smarter heuristic.

## 2. Mechanism

`model → typed IR → e-graph saturation (equational laws) →
cost-based extraction → verified program → backend lowering`.

- **Carrier monoids**: `aff(A,b)` (linear recurrences → scans),
  `om(m,l,a)` (online softmax — FlashAttention's combine *follows*
  from homomorphism + associativity), `trace^U` (JSV axioms as
  rewrites; loops → resolvents `P+Q(I−S)⁻¹R`), cross-carrier laws
  (`xcarrier.py`) pushing readouts through scans and scans inside
  softmax elements.
- **Nonlocal lifts**: `lift_scan_to_applyd` constructs the carrier
  member directly from the recognized recurrence spine — carrier-law
  saturation is combinatorially explosive at long T (358k enodes on
  retnet T=128 vs the 100k cap), so the fold is built, not
  saturated. Pointwise witnesses make constructed members replay
  standalone.
- **Extraction**: additive cost over the e-graph; `backend_cost`
  prices members using ops the sink can't lower at +inf. Carrier
  selection is nonlocal — additive extraction can't price a batched
  spine — so `_carrier_upgrade` force-extracts root-class carrier
  enodes and compares *delivered* prices per lowering.
- **Certificates**: every extracted program carries an ordered
  replayable derivation `original → optimized`; `verify_certificate`
  re-checks derivational equivalence on real terms (fp64), not
  numerical spot-checks. It has caught a false-proof matcher bug, a
  shape misinference that fabricated a 1.98× "win", a well-typed
  wrong program (diff 9.83), and a launch-time-vs-execution timing
  bug. All regression-tested.

## 3. Verified results

All rows fp64-verified where stated; bench artifacts named for
reproduction.

| Transform | Result | Artifact |
|---|---|---|
| Weights-first fold, k-chain `x@W₁@…@W₁₆` → 1 GEMM (Inductor keeps all k `mm`s) | **15.9× vs Inductor GPU** (k=16, d=512, R=65536); ≈k× scaling across 36-cell sweep | `reassoc_scale_20260926-142951.md`, `GPU_RUN.md` |
| Asymptotic reassoc `(QKᵀ)V → Q(KᵀV)`, O(T²d)→O(Td²) | 8.0× at T=2048 | README results table |
| Scan lift, retnet/GLA/delta blocks — `optimize_model` selects + delivers carrier end-to-end | root=`applyd`, fp64-exact (rel 8e-16), **3.36× vs eager** T=128 CUDA; 3.9× with `compile=True` | `real_linear_attn_20260926-235234.md` |
| Chunked decode carriers, CUDA-graph captured | **1.65–2.8× vs best non-carrier** (GPU); delta C=8: 29.7µs vs 49.1µs; un-captured carrier loses (dispatch) | `decode_scan_20260927-015244.md` |
| GLA T=2048 — Inductor compile exceeds 60s timeout | CatOpt delivers certified batched schedule (262s pipeline), runtime parity w/ eager | plan `0005-cost-model-fidelity.md` §scale |
| Streaming attention (om monoid) | 260× vs sdpa-recompute at 65k cache; 89 MiB flat at 2M keys | README results table |
| Projection pairing (QKV, gate·up) on llama2.c | rediscovers `MergedColumnParallelLinear`/`QKVParallelLinear`; 1.24× vs Inductor | `stories15m_bench` |
| FLOP reduction (reassoc/merge/factorize) | 2.51× GPU / 3.02× CPU, DeepParallel b=4096 | README results table |
| Autotuned selection (`optimize_model_autotuned`) | matrix_chain 2.24× CPU; picks honest about losses (linear_recurrence → eager) | `killer_demo_20260927-021522.md` |

**Cost-model fidelity** (`cost_fidelity_*`): ρ ≈ 0.95 in the win
regime (k-chain — correct ordering *and* pick). Scan regime was
inverted (ρ −0.24…−0.38): same term, same predicted cost, 6–30×
latency spread across lowerings. Fixes: executor-aware pricing
(`executor_cost_for`, calibrated `dispatch_us`/`leaf_eval_us`),
lowering routing (`stats["lowering"]`), honest param-only discount
(a `trace` resolvent hid a 14.5s `linalg.solve`), and
`fusion_regions` (predicted Inductor kernel count — the 128-leaf
`applyd` spine bills ~1 kernel, predicted 18.7µs ≈ measured, not
~256 phantom dispatches). Post-fix ρ: `lowering_min` 0.167→0.261,
`exec_generic` 0.070→0.316 — positive but not yet predictive; the
frontier on scan blocks is thin (4 near-identical alts).

## 4. Honest losses and boundaries

- **Inductor wins wherever it compiles the raw graph.** Executor
  per-leaf dispatch makes the uncompiled carrier lose (retnet T=128:
  0.184×; CPU chunked decode: 0.5–0.8×, all six cells LOSS in
  `decode_scan_20260927-010340.md`). The CUDA-graph capture is what
  amortizes the carrier's launch structure to zero.
- **CPU decode is a loss**; launch-bound cells (B=1, T≤64) lose
  4–15% — split-view copies cost more than saved launches
  (`decode_bench`, `GPU_RUN.md` §decode).
- **Pipeline cost**: 262s at T=2048 vs Inductor's 45s compile where
  Inductor succeeds (retnet T=2048: CatOpt 215.8s→1.04ms vs
  Inductor 45.4s→0.256ms — Inductor still faster at runtime).
- **Toy dims**: d≤64, T≤2048, B=1, RTX 2050 (4 GB, no
  max_autotune_gemm). Real checkpoints (stories15M/110M): all blocks
  transform and verify (~2e-5), −37% GEMM launches profiler-verified,
  **parity** on wall time.
- Chunked om-attention loses to fused SDPA head-to-head when K,V fit
  on device; the win is memory/incremental state, not throughput.

## 5. Open work

- **Executor leanness**: per-leaf dispatch is the loss driver;
  fusion/tiling inside `scan_lower`/`om_lower` or codegen for the
  batched levels.
- **Fusion-aware extraction**: `fusion_regions` is landed for
  *reporting*; completing it for *selection* needs either additive
  decomposition that respects kernel boundaries or more
  `_carrier_upgrade`-style coordinated passes.
- **Pipeline scalability**: spine walk + saturation at large T is
  the compile-time wall (215–262s); incremental matching already
  took 4-layer models 768s→4.9s, same direction needed for T.
- **Scale validation**: production dims (d≥1024, B≥8), datacenter
  GPU, trained linear-attention checkpoints. `calibrate()` makes
  re-targeting mechanical; magnitudes don't extrapolate.
- Thicker frontiers: scan-block alternates are near-identical;
  discrimination currently rides on executor variants, not form
  diversity.

## 6. Verdict

Publishable as a **reachability + verification** contribution, not
as a speedup paper. What is proven and verified: categorical
semantics expose transforms outside the tensor-IR reachable set
(Blelloch scan from associativity alone; om-combine = FlashAttention
derived, not encoded; closed-form resolvents via `trace`), the
pipeline *delivers* them end-to-end under certificates (not just
finds them in an e-graph), and the verifier has repeatedly caught
would-be false claims — the methodology is the result as much as
the numbers. The performance story is real but narrow: wins where
work is asymptotically reduced (fold, reassoc, log-depth scan) or
where Inductor cannot produce code at all (GLA T=2048), and losses
wherever Inductor's fused kernels apply — executor overhead, not
search, is the gap. Nothing discovered is novel to practitioners;
the contribution is automatic discovery + certified equivalence +
per-shape selection. Framed as "verified reachability with a real,
regime-bounded performance win," it is a solid systems/PL workshop
or short-paper result; a full venue paper needs the scale
validation and executor work above.
