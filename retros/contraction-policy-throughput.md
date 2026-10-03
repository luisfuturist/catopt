# Learned contraction ordering — removing the feature-construction bottleneck

Plan 0016 follow-up, fourth half.  `contraction-policy-compute.md`
measured the per-decision cost of the learned policy and named the
open problem in its own reading of the numbers:

> A policy forward pass is cheap per decision — 84-93 us — *below* the
> heuristic scan at n = 40.  … A full policy rollout is ~4x a
> randomised-greedy rollout.  The rollout cost is dominated not by the
> MLP but by the Python-side **feature construction**
> (`all_pair_features` for every state).

`contraction-policy-einsum.md` then showed the consequence: the policy
gets ~4x fewer rollouts than the randomised-greedy baselines, and loses
to `opt_einsum`'s `RandomGreedy` at n = 40.  This retro does the
engineering half: profile the rollout, remove the dominant cost, and
re-measure — **honestly, without touching the feature semantics or the
budgets.**

**The honest answer: the bottleneck is real, large, and removed — pure
feature construction is 4.3x cheaper at n = 40 (18.3 ms -> 4.3 ms; 28.9
ms -> 7.1 ms including the GPU forward) and the policy gets ~3x more
rollouts at equal wall-clock (35-40 -> 100-153 at 1 s).  That flips the
comparison against `opt_einsum`'s staged `greedy` at n = 40 (from a
consistent loss to a win).  It does NOT flip the comparison against
`opt_einsum`'s `RandomGreedy`: the policy still loses at n = 40, by
~1.7-2.2x at 1 s — essentially the same margin as before the fix,
because quality saturates with rollout count.**

Reproduce (RTX 2050, `uv sync --group einsum` first):

```sh
.venv/bin/python project/retros/contraction_policy_feature_check.py
.venv/bin/python tools/contraction_policy.py --mode time \
    --instances 6 --seed 0 --device cuda
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 0
```

## What changed

**`tools/contraction_policy.py` only — nothing under `packages/`
changed.**  `ContractionGame` keeps its exact interface (`ts`, `pairs`,
`step`, `state_features`, `all_pair_features`, `cost`, `greedy_ref`)
and its exact **feature values**; only its internals and cost changed.

1. **Bitmask tensors.**  Each tensor's index set is mirrored as an
   integer bitmask over the instance's labels, so a pair's shared
   indices are one `int &` (and `int.bit_count()` for the
   cardinality) instead of a `frozenset` intersection.
2. **Incremental pairwise statistics.**  The two quantities the
   features need per pair — the `log2` intersection volume and the
   intersection cardinality — are maintained across steps: a
   contraction recomputes only the row/column of the freshly merged
   tensor (`O(m)` per step), not the whole `O(m^2)` table.  The old
   `_refresh` rebuilt all of it every step, and `all_pair_features`
   then rebuilt *the same intersections a second time* in a Python
   loop.
3. **NumPy-vectorised assembly.**  The `[n_pairs, 9]` feature matrix
   is built in one shot per step (`np.searchsorted` for the percentile
   column, broadcasting for the normalised columns) instead of a
   Python loop over pairs; `_batch_inputs` stacks the matrices with
   `np.stack` and hands them to torch in one conversion.
4. **Exact arithmetic preserved.**  Every feature value is the same
   IEEE operation on the same exactly-rounded `math.fsum` volume
   (`fsum` is order-independent, so a bitmask recomputation of a
   tensor volume is bit-identical to the `frozenset` one).  The
   `pct` column is `np.searchsorted` on the sorted union array, which
   is exactly `bisect_left` on the sorted list; the ranks and
   intersection cardinalities are small integers, exact in float64.

Nothing was added to `packages/`; no suppression, no ratchet change.

### The numerical-identity guarantee

A policy is coupled to the *semantics* of its features
(`stage7-multifamily-results.md`: a `features.py` change silently
regressed a learned policy), so a speedup that moved a feature value
would be worthless.  `project/retros/contraction_policy_feature_check.py`
defines the **pre-optimisation scalar construction inline** and drives
both implementations along the *same* action sequence, asserting the
state and pair feature vectors are equal **bit for bit** on every step:

```text
OK — 333 instances, features bit-identical
```

333 instances: 240 hypergraph boards (n = 8-40, 40 seeds), 3 chains,
6 non-zero-start-cost / explicit-`n0` games, 80 bond-network boards
(n = 8-40, 20 seeds) and the four real attention/MLP einsums.  Step
costs agree too.  The trained policy is therefore valid on the new
features.

## 1. The profile — where a rollout's time actually goes

Two measurements of the **original** rollout.  Left: pure Python, no
torch, min over trials (the honest feature cost).  Right: the same
rollout with the GPU forward, batch 8 (phases overlap sync, so the
percentages are approximate).

| n | rollout (features) | step / `_refresh` | pair features | state feats | MLP fwd+sample |
|---|---|---|---|---|---|
| 20 | 4.07 ms | 1.28 ms (31%) | 2.57 ms (63%) | 0.03 ms | 0.09 ms |
| 30 | 13.51 ms | 4.08 ms (30%) | 8.99 ms (67%) | 0.06 ms | 0.02 ms |
| 40 | 30.64 ms | 9.02 ms (29%) | 20.87 ms (68%) | 0.08 ms | 0.02 ms |

Phase split including torch (batch 8, n = 20/30/40): **pair features
30/47/49%**, `_refresh` 16/19/19%, Python->tensor conversion 13/14/15%,
MLP forward 12/6/5%, categorical sampling 18/9/7%.

The conclusion the retro's reading predicted, now measured directly:
**~95% of a rollout is Python-side feature construction** — pair
features (~65%) plus the `_refresh` bookkeeping that recomputes the
same intersections (~30%).  The MLP forward is **under 1%**.  That is
the bottleneck; it is not the network, and it is not the forward pass.

## 2. The new rollout cost

Pure-Python feature cost, min over trials (identical instance, seed 0):

| n | before | after | speedup | after (rollouts/s) |
|---|---|---|---|---|
| 20 | 2.79 ms | 1.26 ms | **2.0x** | 791 |
| 30 | 7.90 ms | 2.32 ms | **3.4x** | 431 |
| 40 | 18.28 ms | 4.29 ms | **4.3x** | 233 |
| 60 | 61.9 ms | 10.5 ms | **5.9x** | 95 |

The tool's own per-decision table (mid-game state, batch 64, warm;
`--mode time`), before -> after.  The forward pass is untouched
(84-93 us at batch 1) — only the rollout moved:

| n | policy rollout | randomised-greedy rollout | pol / restart |
|---|---|---|---|
| 20 | 3.39 -> **1.70 ms** | 0.82 -> 0.90 ms | 4.1 -> **1.9** |
| 30 | 13.25 -> **5.18 ms** | 3.22 -> 3.31 ms | 4.1 -> **1.6** |
| 40 | 30.27 -> **7.85 ms** | 7.49 -> 7.90 ms | 4.0 -> **0.99** |

At n = 40 a policy rollout is now **cheaper** than a randomised-greedy
rollout (0.99x); the retro's "~4x more expensive per rollout" penalty is
gone at the scale that matters.  End-to-end (including the GPU forward,
batch 64, min over trials):

| n | greedy rollout | policy before | policy after | speedup | after / restart |
|---|---|---|---|---|---|
| 20 | 0.75 ms | 3.52 ms | 1.58 ms | 2.2x | 2.10 |
| 30 | 3.13 ms | 12.55 ms | 4.36 ms | 2.9x | 1.39 |
| 40 | 6.84 ms | 28.86 ms | 7.05 ms | **4.1x** | 1.03 |
| 60 | 28.91 ms | 98.64 ms | 13.90 ms | 7.1x | 0.48 |

## 3. Equal-wall-clock vs `opt_einsum` at n = 40

Same tool, same budgets, same instance family
(`random_bond_network`), 3 instances/scale.  `oe-rand-greedy` is the
best-found on every n = 40 board, so `learned / oe-rand-greedy` equals
the learned ratio-to-best column.  **< 1 = the learned policy wins.**
The retro's two seeds (old code) are shown for comparison; the new
numbers are four seeds.

| budget | metric | retro s0 / s1 | new s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| 50 ms | learned/oe-rand-greedy | 5.54 / 4.20 | 5.61 | 4.31 | 175.1 | 333.0 |
| 50 ms | learned/oe-greedy | 3.64 / 3.44 | 3.64 | 3.44 | 79.7 | 52.3 |
| 200 ms | learned/oe-rand-greedy | 2.82 / 1.82 | 1.94 | 1.83 | 1.87 | 1.92 |
| 200 ms | learned/oe-greedy | 1.68 / 1.27 | **1.03** | 1.13 | **0.66** | **1.04** |
| 1000 ms | learned/oe-rand-greedy | 2.06 / 1.96 | 1.89 | 1.74 | 1.66 | 2.17 |
| 1000 ms | learned/oe-greedy | 1.105 / 1.136 | **0.95** | **0.91** | **0.59** | 1.28 |

Rollouts the policy actually completed at n = 40 (the throughput win):

| budget | retro s0 | new s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| 50 ms | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 200 ms | 5.5 | 20.7 | 25.3 | 17.3 | 19.3 |
| 1000 ms | ~35-40 | 124.3 | 153.3 | 100.0 | 120.3 |

(The retro's 1 s figure is quoted from its own text — "~40 rollouts" on
the bond family, 35 on the retro's hypergraph family; both are pre-fix.)

## 4. The verdict

**Success criterion: does the policy beat `opt_einsum`'s randomised
greedy at n = 40 at equal wall-clock?  No — it still loses.**  At 1 s
the ratio is **1.66-2.17x** across four seeds (mean ~1.87); at 200 ms,
1.83-1.94x.  This is essentially the retro's pre-fix margin (~2.0x): the
~3.4x more rollouts bought only **~10%** of quality.

The reason is visible in the rollouts table: the policy's quality
*saturates*.  Going from 20 rollouts (200 ms) to ~120 rollouts (1 s) —
6x more work — moves the n = 40 ratio from ~1.9 to ~1.8.  A compute
speedup therefore cannot close a 1.8x *quality* gap; the gap is the
policy's ordering quality per rollout against a staged randomised
greedy that never forms an outer product, not the throughput.

Two things the fix **did** change:

* **It beats `opt_einsum`'s staged `greedy` at n = 40 at 1 s.**  The
  retro measured 1.105 / 1.136 (a loss, both seeds); now 0.59-1.28,
  i.e. a win on 3 of 4 seeds (mean ~0.93).  The policy is a better
  single-rollout chooser than `oe-greedy`; the extra rollouts let it
  show that at n = 40.
* **The per-rollout cost penalty is gone at n = 40** (pol/restart 4.0 ->
  0.99).  The original limitation — "the learned player *is* the more
  expensive one per rollout" — no longer holds at the scale where it
  mattered.

So the engineering goal is met (the measured bottleneck is removed and
the policy is no longer compute-starved), but the **research verdict is
unchanged: `RandomGreedy` still wins at n = 40.**  More compute alone
will not flip it.

## Honest limitations

* **The quality ratios are noisy.**  The anytime players are
  wall-clock-bound and the host was loaded (load average ~6 on 16
  cores); the policy's n = 40 ratio moves ~10-20% run to run.  The
  *deterministic* facts — the feature cost, the forward cost, the
  mechanism counts, `oe-greedy`'s result — reproduce; the verdict here
  rests only on gaps (>= 1.6x) far larger than that spread.  No
  conclusion rests on a margin below it.
* **Three instances per scale.**  Enough to establish the ~1.8x
  `RandomGreedy` gap and its sign; not to resolve 5-10%.
* **The n = 40 / 50 ms cell is a one-rollout regime** (the policy fits a
  single rollout; `learned/oe-rand-greedy` 4.3-333).  It is reported as
  measured, not averaged away.
* **`oe-rand-greedy` is deadline-bounded** (`max_time`), not
  iteration-bounded, so its rollout count is not directly comparable to
  the policy's; it is the fair equal-time opponent, and it wins.
* **One policy, one architecture, one cost model** — the same caveats
  as the three prior retros (a 64-wide MLP trained by REINFORCE on
  n = 8-12 with a greedy critic).
* **`optimal` is still a non-comparison**: `opt_einsum`'s exact DP is
  DNF at n >= 10 within 1 s.

## Reproduce

```sh
uv sync --group einsum
# 1. the feature-identity assertion (no GPU):
.venv/bin/python project/retros/contraction_policy_feature_check.py
# 2. the per-decision / equal-wall-clock ladder on the retro's family:
.venv/bin/python tools/contraction_policy.py --mode time \
    --instances 6 --seed 0 --device cuda
# 3. the head-to-head vs opt_einsum (~6 min/seed):
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 0
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 1
```

Gates at this HEAD: `ruff check packages tools` clean, `ruff format
--check` clean, `ty check` clean, `vulture` exit 0, radon ratchet ok,
`pytest` green (3340 passed, 31 skipped).  Nothing under `packages/`
changed; the default `dev` group is unchanged.
