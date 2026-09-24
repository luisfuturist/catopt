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
| **Kronecker-sum** (attn/MLP) | ~4 terms @95% | **beats SVD 15–31% — live** |
| H-matrix off-diagonals | rel. rank 1.0 | dead |
| Sparse concentration | top10% ≈ 46% energy | mild |
| Monarch/butterfly ALS | diverged | inconclusive |
| INR coordinate-fit | rel err ≈ 1.0 | dead (no smooth manifold) |

**Gate outcome: two live signals, both ε-approximate.** The weights
aren't structured — they're *compressible*.

## 4. The ε axis (`catopt/eps.py`, `egraph.py`)

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

## 10. Phase 5 — Validation status

- Real-weight measurement: done (Phases 0/0b above).
- End-to-end rate–distortion vs GPTQ/AWQ: **not yet** — the
  differentiator is the certificate, so the bar is "comparable
  compression *with* a bound," not beating GPTQ's raw ratio.
- omd executor: the cross-carrier `omd` member wins flops/launch
  extraction but runs 0.54–0.71× slower than eager through the generic
  evaluator — needs a dedicated blocked-scan executor (est. 2–9×).

## 11. Honest limits

- Bounds are site-local (spectral / Frobenius); whole-model
  propagation needs per-op Lipschitz constants — not yet computed.
- Monolithic saturation doesn't scale past ~2 blocks;
  `optimize_compositional` is the per-block workaround.
- Monarch ALS was inconclusive — deeper butterfly structure may exist
  that naive alternating least squares can't reach.
- ε-passes are opt-in (`eps_rtol`); default extraction is exact-only,
  so nothing silently trades accuracy.

## 12. What's next

1. `kron_linear_params` at real scale — apply to stories15M attention
   weights and report the model-level rate–distortion curve.
2. An `omd` executor (blocked associative scan + hoisted coefficient
   maps).
3. Lipschitz propagation for output-level ε bounds.
4. GPTQ/AWQ comparison on the same checkpoint.
