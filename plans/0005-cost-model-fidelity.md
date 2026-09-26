# Plan 0005 — Cost-model fidelity vs measured latency

Status: measured (bench landed) — fix identified, not yet implemented

## Measured (RTX 2050 + CPU)

- chain k=4 (win regime): rho 0.91-0.97 all cost fns, correct pick.
- retnet T=128 (loss regime): rho -0.24 to -0.38 — INVERTED.
  inductor pred 8/measured 1; canon_irmod pred 2-6/measured 9.
- Root cause: term-level cost is blind to the lowering — same term
  6-30x latency across executors (fused kernel vs per-leaf eval).
  trace-carrier members hide linalg.solve (14.5s) under normal prices.
- attn/swiglu frontier: 1 distinct member at T=128 (thin).

## Named fix (next plan)

Price (term, lowering) pairs — executor-aware cost terms:
per-level launch count for batched carriers, per-leaf generic-eval
dispatch for IRModule, fused-kernel discount for compiled lowering;
hidden solver ops (linalg.solve) priced honestly.

## Hypothesis

Parity/loss vs Inductor may be a *cost-model fidelity* gap, not a
search-space gap. The study: does the predicted-cost ordering of
equivalence-class members correlate with measured end-to-end latency?

## Method

`bench/cost_fidelity.py` (benchkit Report):

Per cell (model, shape, device):
1. `discover_alternatives(m, x, top_k)` → the frontier of distinct
   members + rule-fire provenance.
2. For each alternative × each lowering (generic IRModule, batched
   carrier executor where applicable, torch.compile'd):
   - verify fp64 vs eager (skip-and-record on failure),
   - measure latency via benchkit Runner.
3. Re-price every candidate under each cost fn: `flops_cost`,
   `launch_aware_cost`, `depth_cost`, `roofline_cost_for(calibrate())`,
   `depth_cost_for(profile)`, `param_bytes_cost_for`.
4. Report per cost fn per cell: Spearman ρ, Kendall τ, pick-accuracy
   (argmin predicted == argmin measured?), residual breakdown
   (which form/executor is systematically mispriced).

Cells: matrix chain k∈{4,8,16} (win regime), retnet/gla T∈{128,512}
(loss regime), AttentionBlock/SwiGLU (parity regime).

## Expected findings → fixes

- If ρ is high per cost fn but the selection still loses →
  selection/extraction issue, not cost.
- If a form/executor is systematically *under*-predicted-cost
  (batched scan measured slower than priced) → the missing term is
  executor dispatch (per-level launches + per-leaf eval), not graph
  FLOPs. Fix: price the lowering's launch structure, not just the
  term's.
- If ρ is weak everywhere → the cost model needs per-backend
  calibration; `calibrate()` exists — measure the delta before/after.

## Deliverables

- `bench/cost_fidelity.py` → JSON + MD + predicted-vs-measured
  scatter plots.
- A cost-model fix for any identified unmodeled term, then a re-run
  showing ρ improvement and selection changes.
- README update with the fidelity numbers (honest either way).
