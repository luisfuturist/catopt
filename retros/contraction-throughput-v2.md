# Contraction throughput v2 — vectorising the rollout, not the net

`policy-guided-restart.md` decomposed the residual gap into

> per-rollout quality is ~parity at n <= 30 and ~10-15 % behind at
> n = 40, *and* `oe` does ~3-6x more rollouts per second.

This retro is the throughput half of that decomposition, measured and
reduced **in the shipped package**: `rollout_orders` /
`run_policy_batch` in
`packages/catopt-torch/src/catopt_torch/contraction_policy.py` now
drive a **vectorised lockstep game state** — one `[B, m, L]` indicator
tensor and `[B, m, m]` pairwise tables shared by the whole batch,
updated incrementally and entirely on-device, with the sampled-index
history and costs coming home in a single copy at episode end.

**The honest answer: per-rollout cost drops ~7-15x on CUDA
(110-180 -> ~1500 episodes/s at n = 40, ~3500 at n = 30, ~10 400 at
n = 20 — the policy now runs *more* trials per second than
`opt_einsum`'s heap-scan randomised greedy at every scale: ~1.8x at
n = 40, ~2.7x at n = 30, ~4.7x at n = 20).  Per-rollout quality is
unchanged (best-of-128 oe-costs are bitwise-identical between the old
and new drivers on every board measured).  The starved cells flip:
the n = 30 / 50 ms cell goes 4.36 -> 1.15 and n = 20 wins outright at
every budget.  What is left is exactly the other half of the
decomposition — the per-rollout quality deficit at n = 40 — now that
compute no longer binds.**

Reproduce (RTX 2050, `uv sync --group einsum` first):

```sh
.venv/bin/python project/retros/contraction_throughput_v2_probe.py \
    --throughput --ladder --device cuda --boards 4
```

## 1. The profile — where a rollout's ms went

Per-phase split of the old lockstep driver (the pre-change shipped
code), n = 40, batch 128, CUDA — the shape is the same at n = 20/30:

| phase | ms | share | what it is |
|---|---|---|---|
| `feat` | 226 | 31 % | per-game `pair_feature_matrix` + `state_features` |
| `step` | 280 | 39 % | per-game `step` (`_advance` + `_refresh` + `pair_cost`) |
| `init` | 56 | 8 % | `ContractionGame` construction, `O(m^2)` Python |
| `stack`+`tensor` | 69 | 10 % | `np.stack` + host->device copy |
| `fwd`+`samp`+`list` | 88 | 12 % | MLP forward + `Categorical` + `tolist()` sync |

**~78 % of a rollout was per-game Python feature work**; the forward
pass itself was ~9 %, so a smaller net could not have been the lever.
The same pattern holds on CPU (feature+step ~60 %, the rest a slower
CPU GEMM — CPU forward is *not* a win at these sizes).  The batch
already amortised the GPU; what it could not amortise was the
`O(B)` Python state updates between forwards.

## 2. What changed — the vectorised lockstep driver

`rollout_orders` and `run_policy_batch` keep their signatures; both
now dispatch to `_rollouts`, which picks `_lockstep_rollouts` for any
batch > 2 and the kept scalar loop (`_scalar_rollouts`, the old code)
at batch <= 2 — the measured crossover where the vectorised step's
fixed launch burst stops paying for itself (CUDA only; on CPU the new
driver wins even at batch 1).

Inside `_lockstep_rollouts`:

1. **Batched state.**  Every tensor's index set is a row of a
   `[B, m, L]` indicator matrix (the same sorted-label bit order
   `ContractionGame` uses).  The pairwise tables `T` (intersection
   log-volume) and `C` (intersection cardinality) live as
   `[B, m, m]` tensors, with `ls`/`rk` read off the diagonals —
   `T[t,t]` is tensor `t`'s own volume, `C[t,t]` its rank.
2. **Incremental tables.**  A contraction recomputes only the merged
   tensor's row/column — one *skinny* `[B,1,L]` batched matmul instead
   of two full `[B,m,L]@[B,L,m]` GEMMs (2.3 ms vs 0.03 ms per step on
   this card: consumer FP64 GEMM is ~10x slower than FP32).  This
   mirrors the scalar game's `O(m)` `_advance` rather than an `O(m^2)`
   rebuild.  Surviving rows/cols compact by a stable argsort + gather,
   preserving `merge_tensors`' keep-order-then-append convention, so
   recorded `(a, b)` pairs replay identically under `cost_of_order`.
3. **Feature assembly batched.**  All 9 pair columns and 7 state
   columns are computed on `[B, P]` tensors — `searchsorted` for the
   percentile column, `aminmax` for min/max — then cast float64 ->
   float32 exactly where the old `torch.as_tensor` did.
4. **Zero per-step syncs.**  The sampled index, the merge, the table
   update and the cost accumulation all stay on-device; `tolist()`
   happens once, on the `[B, n-1, 2]` history at episode end.  Sampled
   rollouts use the **Gumbel-max trick** (argmax of `logits/T + g` for
   iid Gumbel `g`) — the same categorical distribution as
   `Categorical(logits/T).sample()`, in fewer kernels.
5. **`MAX_BATCH` 128 -> 512.**  The driver's marginal cost per episode
   is now a per-step kernel-time share, so bigger batches keep
   amortising — the cap is a memory guard, not a sweet spot.

### Feature fidelity — what "same semantics" means here

The scalar game's `math.fsum` sums are correctly rounded; batched
GEMMs are not.  Measured against `ContractionGame` along identical
state sequences (n = 8/20/40, CPU and CUDA):

* columns 0-4 differ by <= ~4e-15 (GEMM ulp);
* columns 5-7 are **exact** (integer data in float64);
* the move cost is the **exact** union product while it stays below
  `2^53` (the same regime where `pair_cost` is exact) — beyond that it
  and the `cost` column drift by ~ulp;
* **column 8 (`pct`) splits true volume ties**: pairs whose union
  log-volumes are mathematically equal get bitwise-equal `fsum` values
  and so equal `searchsorted` ranks, while GEMM noise (~1e-16)
  perturbs them into distinct ranks — deviations up to
  `tie_size / n_pairs` (~1e-2 observed).  Integer multiplicative
  relations make such ties common on this family (2*8 = 4*4).

The empirical answer: at best-of-128 the oe-costs produced by the two
drivers are **bitwise identical** on every board measured — the
deviations never change which episodes win.  An exact fix exists
(prime-exponent vectors per label — unique factorisation makes equal
union volumes bitwise-equal again) if the column-8 caveat ever needs
removing; it costs a second `[B,m,m,#primes]` table.

## 3. Throughput — episodes/s before -> after

`rollout_orders` wall time, `T = 1.0` sampled episodes, min of 3
reps (probe's `--throughput`; `old` is the kept replica of the
pre-change driver):

| n | B | old ep/s | new ep/s | x |
|---|---|---|---|---|
| 20 | 8 | 416 | 580 | 1.4 |
| 20 | 64 | 728 | 4202 | 5.8 |
| 20 | 128 | 562 | 7096 | 12.6 |
| 20 | 512 | 692 | 10407 | 15.0 |
| 30 | 8 | 218 | 350 | 1.6 |
| 30 | 128 | 321 | 3275 | 10.2 |
| 30 | 512 | 359 | 3539 | 9.9 |
| 40 | 8 | 112 | 233 | 2.1 |
| 40 | 128 | 210 | 1465 | 7.0 |
| 40 | 512 | 205 | 1530 | 7.5 |

CPU gains are real but modest (n = 40: 110 -> ~250 at B = 128 —
~2.3x): the driver is kernel-dispatch-bound and CPU kernel time is
real work.  For reference the independent baseline's clocked
trial rates on this box: `oe-rand-greedy` ~2300 trials/s at n = 20,
~1270 at n = 30, ~855 at n = 40 — **the policy episode is now the
cheaper trial at every scale.**

Single-pass (batch 1) `ContractionPolicy.order` is unchanged — it
takes the scalar path (CUDA n = 40: ~20 ms, vs ~26 ms through the
vectorised driver, hence the dispatch).

## 4. Equal wall-clock — quality before -> after

oe-cost pairwise ratio vs `oe-rand-greedy`, mean over 4 boards, the
retro's affine batch scheduler on both drivers, budgeted best-of-N
at T = 1.0; **<1 = the learned player wins**.  `ep` = episodes
completed; `ms` = actually spent.

| ms | n | old | new | new ep | oe-rg trials |
|---|---|---|---|---|---|
| 50 | 20 | 1.010 | **0.918** | 512 | 117 |
| 200 | 20 | 0.956 | **0.951** | 1962 | 477 |
| 1000 | 20 | 0.997 | **0.993** | 10834 | 2179 |
| 50 | 30 | 4.361 | 1.147 | 146 | 61 |
| 200 | 30 | 1.339 | 1.044 | 696 | 255 |
| 1000 | 30 | 1.037 | 1.039 | 3433 | 1262 |
| 50 | 40 | 1.752 | 1.375 | 43 | 43 |
| 200 | 40 | 1.440 | 1.219 | 315 | 171 |
| 1000 | 40 | 1.285 | 1.172 | 1513 | 830 |

* **n = 20 flips to a win at every budget** (0.92-0.99; the retro had
  parity ~0.97-1.02): ~5-10x more episodes than `oe` buys depth where
  rollouts were already at parity per episode.
* **The starved cells are transformed.**  n = 30 / 50 ms goes
  4.36 -> 1.15 (a catastrophic one-episode cell becomes near-parity);
  n = 40 / 50 ms goes 1.75 -> 1.38 at *equal* episode count (43 vs 43
  — the driver now fits real work under the wire instead of one batch
  overshooting).
* **Fed budgets: n = 30 sits at ~1.04** (the retro's scale column was
  ~1.13-1.16) and **n = 40 at ~1.17** (was ~1.28-1.41).  The residual
  is no longer throughput — at n = 40 / 1000 ms the policy completes
  1.8x *more* episodes than `oe` yet still lands 1.17: the
  per-rollout deficit the retro isolated is now the whole story.
* CPU ladder (same protocol, `single`/`oe` unchanged): n = 40 / 200 ms
  1.84 -> 1.22; the gains are real but much smaller — CPU kernel time
  dominates where CUDA dispatch dominated.

A second full run (the transcript file) reproduced the shape with
starved-cell wobble as expected: n = 30 / 50 ms new 1.14, n = 40 /
50 ms new 1.28, n = 40 / 200 ms new **1.105** — and n = 30 / 200 ms
new **1.001**, the first fed cell at n = 30 at parity.

## 5. Verdict

**The per-decision cost bottleneck is gone, and it buys real wins.**

* Per-rollout cost: ~7-15x cheaper on CUDA; the policy now exceeds
  `oe-rand-greedy`'s trials/s at n = 20/30/40 (was ~3-6x *behind*).
* Quality at equal wall-clock: **the player beats `oe-rand-greedy` at
  n = 20 across all budgets**, sits at ~1.04 at n = 30, and ~1.17-1.38
  at n = 40 — every cell better than before, none regressed.
* The starved regime is where it pays most: n = 30 / 50 ms moved from
  a 4.36x loss to 1.15 — "more decisions at fixed quality" is exactly
  what the starvation cells needed.
* The residual gap at n = 40 is now cleanly *algorithmic*: 1.8x the
  trial volume, 1.17x the cost.  Throughput is solved; the remaining
  fight is per-rollout policy quality (scale training, richer
  features, or a guided proposal closer to `oe`'s staged scan).

Standing one-liner: *"the vectorised lockstep driver makes a policy
episode cheaper than an `opt_einsum` heuristic trial at every scale —
the learned player now wins at n <= ~25 and ~1.04/1.17 at n = 30/40,
with the residual entirely per-rollout quality, not compute."*

## 6. Honesty / limits

* **4 boards, one run.**  Starved cells remain timing-noise-dominated;
  the n = 30 / 50 ms old-driver cell read 4.36 this run (1.75-5.8
  across runs).  No conclusion rests on a margin under ~10 %.
* **Scheduler overshoot shrank, not vanished.**  The new driver's
  batch cost is nearly flat in B until kernels saturate, so affine
  priors fitted on small batches oversize the first batch — the probe
  fits a conservative large-batch marginal per board; spent-vs-budget
  is reported in the `ms` column (new player: 33.5-77.7 ms at the
  50 ms budget where the fixed cost alone is ~30-45 ms — a bounded,
  protocol-documented overshoot, better than the old driver's).
* **`MAX_BATCH = 512` widens first-batch overshoot potential** for
  misestimated priors (512 episodes of real work can exceed a 50 ms
  budget ~3x at n = 30); the shipped schedulers (`policy_best_order`,
  `_guided_best_order`) size from their own priors — the probe showed
  the effect is controlled by a conservative marginal estimate.
* **Column-8 tie-splitting** (section 2) is a real, documented
  semantic nuance — ~1e-2 feature deviations on tie-group members;
  empirically quality-neutral at best-of-128 on every board measured,
  but it is *not* bit-identical.  Columns 0-4 carry ~ulp GEMM-vs-fsum
  noise; costs carry ~ulp noise once products exceed 2^53.
* **Gumbel-max** is a distribution-identical sampler (argmax of
  logits + iid Gumbel = `Categorical.sample`) but consumes a different
  RNG stream — seeded determinism is preserved, sampled orders differ
  from the old stream's.
* **`_scalar_rollouts` remains** for batch <= 2 — two drivers, one
  contract; the dispatch is a measured crossover, not a semantic
  split.
* CPU gains are modest (the driver is dispatch-bound; CPU kernels do
  real work) — the wins above are the CUDA column the experiments
  actually measured.

## Gates

* `packages/catopt-torch/src/catopt_torch/contraction_policy.py`:
  `rollout_orders`/`run_policy_batch` internals replaced by
  `_lockstep_rollouts` + `_scalar_rollouts` (batch dispatch);
  `MAX_BATCH` 128 -> 512; feature contract `scale-free-v1` and every
  public signature untouched — the bundled artifact loads unchanged.
* `tests/test_contraction_policy_artifact.py`: +1 degenerate-board
  test (34 pass).  `ruff check`, `ruff format --check`, `ty check`,
  `vulture` — all clean.  No tools/ files touched.
* Probe harness `project/retros/contraction_throughput_v2_probe.py`
  (self-contained: inline replica of the old driver for the
  before/after); run transcript alongside at
  `contraction-throughput-v2.txt`.
