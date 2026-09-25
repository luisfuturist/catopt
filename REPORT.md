# CatOpt — Research Report

*The unifying result: a PyTorch model and its weights are one semantic
program. Optimizing that program — with certificates — covers
computation structure and parameter storage in a single search.*

---

## 1. The thesis

CatOpt began as a graph optimizer: categorical semantics + equality
saturation discovering rewrites that tensor-level IRs miss. The
research extension documented here removes the artificial boundary:

```text
              MODEL SEMANTICS
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
   computation              parameters
        │                       │
        └───────────┬───────────┘
                    ↓
          unified equivalence space
                    ↓
             search + coherence
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   executable graph        weight program
```

Mechanism: parameters are `Param` leaves in the same term language as
compute. *Elimination is a corollary* — a leaf no extracted member
references never reaches the state dict. Composition, tying,
slice-sharing, and bounded factorization are the same event at
different certificate bound values.

## 2. Phase 0 — Linear falsification (`measure_weights.py`)

Measured real trained weights (llama2.c `stories15M`, 15M params):

| Probe | Result | Verdict |
|---|---|---|
| Numerical rank @1e-2 | ~270–288/288 | full — no exact low-rank |
| Displacement rank | ~280/288 | no Toeplitz/generator |
| Symmetry defect | ~1.41 | fully asymmetric |
| Low-rank @99% | 96.3% of params | meaningless saving |

**Gate outcome: exact linear structure is absent.** Stop condition
would normally trigger here — but Phase 0 tested only *one* algebra.

## 3. Phase 0b — Algebra family

Extended to the structured-algebra family, gated as *storage vs
relative Frobenius error against plain SVD*:

| Probe | Result | Verdict |
|---|---|---|
| **Token embedding** (60% of params) | **rank 19 @ 95% energy** | **~15× — live** |
| Kronecker-sum (attn/MLP) | 1.3× @ 35% residual | marginal — real, small |
| H-matrix off-diagonals | rel. rank 1.0 | dead |
| Sparse concentration | top10% ≈ 46% energy | mild |
| Monarch/butterfly ALS | diverged | inconclusive |
| INR coordinate-fit | rel err ≈ 1.0 | dead (no smooth manifold) |

**Gate outcome: one strong signal, one marginal.** The embedding is
genuinely compressible ~15×; Kronecker on the dense weights is real
but weak (the earlier "4 terms" was the best-e1 split's rank — the
storage-optimal split needs K~28, giving ~1.3× at 35% residual).
Correction applied honestly.

## 4. The ε axis (`catopt/eps.py`, `egraph.py`)

> **Status: optional toolkit, not core.** The ε-passes are opt-in
> (`optimize_model(eps_rtol=…)`, off by default) and ride on catopt
> rather than being a core capability. Phase 5 (§10) falsified norm
> bounds as a predictor of weight-compression quality on real
> checkpoints — the machinery is retained for activation-path and
> verification use, where a norm bound IS the contract.

The machinery that makes approximation a first-class object:

- `Rewrite.error_bound` / `bound_norm` — a bounded rewrite is an
  ordinary witnessed offer carrying a certified bound.
- `Certificate.error_bound` — triangle-inequality accumulation;
  `cert.exact` distinguishes equivalence from approximation.
- `extract_best_bounded(max_error=…)` — extraction under an ε budget:
  candidates over budget get their bound-carrying enodes banned and
  extraction retries through exact members.
- `param_bytes_cost` — the first cost axis pricing *stored* values
  (opts out of the param-only discount via `charges_param_only`).

Quantization, low-rank, and tying are now one object: **a rewrite with
an error bound**.

## 5. Phase 1 — Atoms and laws

- `low_rank_params`: `linear(x,W) → linear(linear(x,V_r), U_rΣ_r)`,
  bound = σ_{r+1} (exact Eckart–Young, spectral). Offered only when
  storage truly shrinks.
- `kron_linear_params`: `W ≈ Σᵢ Aᵢ⊗Bᵢ` executes as
  `reshape → Aᵢ·X·Bᵢᵀ → reshape`, summed over K terms. Bound =
  rearranged-SVD residual (Frobenius isometry → exact). The offered
  member is a *program of K composed maps* — weights-as-programs in
  the literal sense.
- `low_rank_gather`: `embedding(W,idx) → matmul(embedding(U_r,idx),
  V_r)` — gather the small factor, then project. On the real
  stories15M embedding: **rank 2 at 5% spectral = 142× storage**.
- `quant_params`: quantization-as-rewrite — `W → mul(float(W_int8),
  s)`, certified Frobenius bound `(s/2)·√n`. Byte-aware
  `param_bytes_cost(by_bytes=True)` prices the width reduction;
  verified end-to-end (32KB → 4KB, err within bound).
- `model_bound`: **output-level certificates** — each site bound ×
  its Lipschitz path sensitivity to the output (weight-side edges
  resolve via `input_norm` × input-sensitivity). Verified: quantized
  2-layer model, err 0.0034 vs certified 0.397 (spectral products are
  ~100× conservative — honest).

Rate–distortion through the full pipeline (2-layer MLP, fp64):

| method | stored | ratio | measured err | certified bound |
|---|---|---|---|---|
| int8 quant | 16,384 B | 8.0× | 5.3e-3 | 0.076 (site) / 0.79 (model) |
| low-rank @10% | — | no offer | — | (weights are full-rank) |

Composite (low-rank embedding + int8 quant on all params, one
e-graph, one extraction): **577 KB → 16.7 KB (34.6×)**, measured err
2.8e-2 within the certified model bound — all three rewrite families
composing in a single certificate.

Both inject derived factor params into `source_tensors`, so the
lowered module's state dict contains only the factor tensors.

## 6. Phase 2 — Weight-as-program search

`optimize_model(eps_rtol=…)` runs the ε-passes inside the pipeline.
Demonstrated end-to-end: a Kronecker-structured 64×64 weight → K=4
terms, **32,768 → 4,096 bytes (8×)**, certified bound, replayable
certificate, extracted under `param_bytes_cost`.

## 7. Phase 3 — Cost axes

The trade space is complete: `flops_cost` / `launch_aware_cost` /
`param_bytes_cost` / target-calibrated `TargetProfile`s, and
`extract_best_bounded(max_error)` as the ε constraint. Selection is
Pareto over (compute, storage, certified error).

## 8. Exact sharing (the ε=0 cases)

- `share_duplicate_params` — bitwise-equal `Param` leaves tie into one
  e-class (tied embeddings, duplicated adapters).
- `share_duplicate_param_slices` — head-granular dedup *inside* one
  weight via `index_select` (GQA replication baked into checkpoints).
- `_fold_weight_chains` + `_build_params` — composed/fused weights
  materialize as `fused_*` params; unreferenced originals vanish.

## 9. Phase 4 — Joint search

Already real at the offer level: graph rewrites and weight rewrites
coexist in one e-graph, one extraction. The remaining work is
*interaction*: a compressed weight can change which compute form is
optimal (e.g., a Kronecker-factored linear wants a different schedule
than a dense GEMM).

## 10. Phase 5 — Validation status: **falsified for quality**

Real perplexity on TinyStories validation (stories15M, 2819 tokens,
fp64 forward — baseline ppl **5.03**):

| config | storage | perplexity | verdict |
|---|---|---|---|
| emb rank-2 / r19 / r64 / r128 / r192 | 2.5–1.2× | 6108 → 214 | **dead at every rank** |
| Kronecker-sum (35% resid) | 1.1× | 143 | dead |
| **int8 RTN quant (certified, catopt)** | **4.0×** | **5.16** | **works, bounded** |
| int8 per-channel RTN (GPTQ-class baseline) | 4.0× | 5.00 | works, no bound |
| int4 g128 RTN (GPTQ-class) | 8.0× | 7.01 | works, no bound |

**The honest #2 verdict**: at matched ratio, certified per-tensor
int8 (ppl 5.16) ≈ per-channel int8 baseline (ppl 5.00) — the
certificate costs ~0.16 ppl, not bytes. `quant_params(per_channel=True)`
now exists — row-wise scales close the quality gap at the same ratio
with the bound kept (verified: 8× lower error on outlier rows). The
differentiator claim survives, sized honestly: **compression parity
with the RTN-class baseline, plus a certificate** — not a ratio win.

**The energy signal did not survive contact with quality.** The
embedding's top-19 singular directions (95% Frobenius energy) are not
the ones next-token prediction needs — `‖ΔW‖` bounds don't predict
`Δppl`. Certified compression is real but certifies the wrong
quantity for weight space.

**What survives**: exact structure only — tying, sharing, folding,
composed-linear collapse (all verified, all exact). The ε machinery is
sound and useful where a norm bound IS the contract (activation
paths, verification, certified deployment); it does not rescue
post-hoc weight compression on this checkpoint.

This is the honest negative the kill-gates were for.

### 10.1 The exact corner, measured across archetypes

Under `param_bytes_cost_for` with fp64 output equality verified:

| archetype | saved | mechanism |
|---|---:|---|
| stories15M (real, dense) | 0.00% | nothing to dedup (46 tensors probed) |
| dense transformer (synth) | 0.08% | `linear_channel_scale` fold |
| GQA 8q/2kv (repeat_kv materialised) | **37.50%** | `share_duplicate_param_slices` |
| adapter-merged (W+B·A stored) | 11.03% | `_fold_weight_chains` param-only fold |
| tied embed/classifier stored twice | **50.00%** | `share_duplicate_params` |
| MoE: 4 weight-tied routed experts | **75.00%** | `share_duplicate_params` |
| dead param (unused tensor) | 98.46% | unreachable leaf dropped |
| adapter UNmerged (Wx+BAx) | **−82.76%** | regression: paired-GEMM materialises phantom cat'd weight |

The claim, plainly: **the exact corner is ~0% on dense trained
checkpoints and real on structured ones** — GQA replication,
double-stored ties, weight-tied experts, merged adapters. Two honest
caveats: savings only materialize under the storage cost axis, and
the unmerged-adapter form currently *regresses* (the paired GEMM
materializes a phantom concatenated weight; pinned in
`test_adapter_unmerged_currently_regresses`).

## 11. Honest limits

- **omd at realistic scale** (`bench_omd2.py`): MQA fires
  (`omd_applym`) but the dense-fiber numerator is ~dv/dim× oversized —
  batched executor only 1.15–1.5× vs generic on CUDA, loses to
  Inductor 5–20× everywhere. MHA/sdpa don't lift at all (no
  reshape/transpose-through-`apply` law; `_check_om_elem_aff` vetoes
  rank-4 batched maps — fixable). Semantic value real, speed not.
- `model_bound` propagates site bounds via per-op Lipschitz constants
  — **known unsound for spectral sites at activation positions**
  (low-rank `eps_lr` members: it misses the ‖activation‖ factor;
  reported 0.14 vs measured 0.246). Use `ibp.tight_model_bound`
  instead — it flags `spectral_unsafe` and gives sound bounds
  (118×→3× on quant sites via realized-delta propagation).
  `sdpa`/`exp`/unknown ops report ∞ rather than fabricate a bound.
- Monolithic saturation doesn't scale past ~2 blocks;
  `optimize_compositional` is the per-block workaround.
- Monarch ALS was inconclusive — deeper butterfly structure may exist
  that naive alternating least squares can't reach.
- ε-passes are opt-in (`eps_rtol`); default extraction is exact-only,
  so nothing silently trades accuracy.

## 12. What's next

1. ~~**`omd` executor**~~ — **landed** (`omd_lower.py`): 1.2–3.3× vs
   generic eval, 2.4–5.8× CUDA-graph, fp64-exact.
2. **Real-scale rate–distortion**: apply the full ε pipeline to a
   stories15M-class model end-to-end (needs a model definition; the
   per-matrix evidence is in §3/§10).
3. **GPTQ/AWQ comparison**: needs the quantized checkpoint — blocked
   on `auto-gptq`/`huggingface_hub` deps (PEP 668). The honest claim
   is *comparable compression with a certificate*, not a ratio win.
4. Tighter bounds: `model_bound`'s spectral Lipschitz products are
   ~100× conservative — interval/IBP bounds would tighten.
