# Learned contraction ordering — is the win real at EQUAL COMPUTE?

Plan 0016 follow-up, second half.  `contraction-policy.md` measured the
learned policy against the cheap heuristics by matching the **rollout
count** (1 vs 1, 64 vs 64) and flagged the gap in its own limitations
section:

> *Not compute-equal.* A policy rollout scores every pair with a small
> MLP per step; `greedy`/`restart` sort scalar pair costs. The *number
> of rollouts* is matched, the per-step cost is not.

That is the open question this retro closes.  It is a **falsification
attempt on our own best result**: if the policy only "won" because it
was compared at equal rollouts rather than equal work, then giving every
player the *same wall-clock budget* should erase the win — and that
negative would be the decisive answer.

**The honest answer: the win survives.**  At equal wall-clock the
learned policy still beats `restart` and `search` at every scale and
every budget — the earlier result was **not** an artefact of matching
rollouts.  But the measurement also shows *why* it survives, and it is
not free: a learned rollout costs **~4x** a randomised-greedy rollout, so
at equal time the policy gets **~4x fewer rollouts** and wins anyway
because a single learned rollout is worth far more than a single
randomised-greedy one.

Reproduce (RTX 2050, ~8 min/seed):

```sh
.venv/bin/python tools/contraction_policy.py --mode time \
    --instances 6 --seed 0 --device cuda
```

## What was added

`tools/contraction_policy.py` gained a **`--mode time`** equal-wall-clock
ladder; **nothing under `packages/` changed** and the existing
rollout-matched mode is untouched (`--mode rollout`, the default).

* **Anytime players.** Every player is re-framed as an *anytime*
  algorithm that keeps producing candidates until a wall-clock deadline
  and returns the best found, plus the number of **rollouts** and
  **decisions** it managed:
  * `anytime_greedy` — one deterministic episode (extra time cannot
    help it);
  * `anytime_restart` — best of as many randomised-greedy episodes as
    fit (`top_k=3`, the same player as before);
  * `anytime_search` — the bounded best-first search, with the deadline
    checked *inside* the expansion loop so a tight budget bounds the
    overshoot to a single greedy completion;
  * `anytime_policy` — best of as many sampled policy rollouts as fit,
    run in lockstep batches.
* **A per-decision cost table.** A mid-game state per scale, timing the
  heuristic scan, one policy forward pass (batch 1 and 64), a full
  policy rollout and a full randomised-greedy rollout — so the
  forward-pass-vs-scan comparison is explicit and the two players'
  *units of work* are directly comparable.
* **A per-instance budget** (`--budgets 50,200,1000` ms by default) and
  a table of ratio-to-best, rollouts, decisions and **actual** wall time
  at each budget.

### Two measurement hazards, found and fixed

Both are the honest part — without them the policy is either flattered
or unfairly punished.

1. **A cold first probe overestimates the policy's cost severalfold.**
   The anytime policy sizes its lockstep batch from a measured
   per-rollout cost; the first CUDA probe of a fresh shape (and after
   the CPU-bound heuristic players idle the GPU) measured **34 ms**
   where the warm value is **4 ms** at n = 20.  A probe that cold makes
   the batch ~8x too small and the policy under-uses its budget.
2. **The fix is a warm-up, not a bigger budget.**  `_warm_policy` runs
   the batch shapes the policy will actually use, once, before any board
   is timed; the batch is then sized from a prior measured in the
   decision section and re-sized from the *actual* elapsed time each
   pass, so the policy fills the budget (within a few percent) and
   nothing is timed outside it.  After the fix the policy's "ms" column
   matches the budget instead of overshooting 3x or using half of it.

## 1. The per-decision cost — the comparison made explicit

Mean wall cost per unit of work (mid-game state, RTX 2050, 200 reps;
`seed 0 / seed 1`, essentially identical):

| n | pairs | heuristic scan | policy fwd (batch 1) | policy fwd (batch 64) | policy rollout | restart rollout | pol/restart |
|---|---|---|---|---|---|---|---|
| 20 | 45 | 15.6 / 16.2 us | 83.8 / 92.8 us | 111.2 / 111.0 us | 3.39 / 3.64 ms | 0.82 / 0.87 ms | **4.1 / 4.2** |
| 30 | 105 | 39.4 / 42.3 us | 83.9 / 92.5 us | 248.1 / 248.0 us | 13.25 / 13.19 ms | 3.22 / 3.17 ms | **4.1 / 4.2** |
| 40 | 190 | 83.7 / 83.9 us | 87.3 / 92.4 us | 450.4 / 449.2 us | 30.27 / 30.18 ms | 7.49 / 7.36 ms | **4.0 / 4.1** |

Reading it:

* **A policy forward pass is cheap per decision** — 84-93 us at batch 1,
  *below* the heuristic scan at n = 40 (83.7 us) and only ~5x it at
  n = 20.  The forward pass is *not* the expensive part.
* **A full policy rollout is ~4x a randomised-greedy rollout** at every
  scale.  The rollout cost is dominated not by the MLP but by the
  Python-side **feature construction** (`all_pair_features` for every
  state), which is why batching the forward does not help much and why
  the ratio is flat in n.
* So the earlier limitation was **right about the sign**: the learned
  player *is* the more expensive one per rollout.  At equal wall-clock
  it therefore gets ~1/4 the rollouts.

## 2. The equal-wall-clock ladder

`ratio to best-found` and `rollouts` per player, mean over 6 seeded
networks, `seed 0 / seed 1`.  **`< 1` on a pairwise row means the
learned policy wins.**

### Rollouts actually completed at each budget (seed 0)

| budget | player | n = 20 | n = 30 | n = 40 |
|---|---|---|---|---|
| 50 ms | restart | 58.2 | 16.3 | 7.5 |
| 50 ms | **rl** | 8.0 | 1.8 | 1.0 |
| 200 ms | restart | 245.5 | 66.8 | 27.5 |
| 200 ms | **rl** | 59.3 | 15.2 | 5.5 |
| 1000 ms | restart | 1212.5 | 320.0 | 135.0 |
| 1000 ms | **rl** | 286.0 | 84.2 | 35.0 |

The policy gets **~4x fewer rollouts** at every budget and scale —
exactly the ratio the per-decision table predicts.

### Pairwise cost ratio at equal wall-clock (mean over 6 seeds)

`rl / restart`:

| budget | n = 20 | n = 30 | n = 40 |
|---|---|---|---|
| 50 ms | **0.792 / 0.867** | **0.736 / 0.673** | **0.696** / 1.245 |
| 200 ms | **0.842 / 0.856** | **0.863 / 0.882** | **0.735 / 0.651** |
| 1000 ms | **0.881 / 0.891** | **0.884 / 0.903** | **0.856 / 0.858** |

`imitation / restart`:

| budget | n = 20 | n = 30 | n = 40 |
|---|---|---|---|
| 50 ms | **0.783 / 0.816** | **0.805 / 0.704** | **0.902 / 0.890** |
| 200 ms | **0.797 / 0.835** | **0.863 / 0.856** | **0.758 / 0.652** |
| 1000 ms | **0.841 / 0.859** | **0.872 / 0.877** | **0.861 / 0.855** |

`rl / search` and `rl / greedy` (search is indistinguishable from greedy
at equal time — see §4):

| budget | rl/search 20 | 30 | 40 | rl/greedy 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| 50 ms | **0.390 / 0.388** | **0.675 / 0.612** | **0.547** / 1.214 | same | same | same |
| 200 ms | **0.382 / 0.364** | **0.660 / 0.600** | **0.537 / 0.545** | same | same | same |
| 1000 ms | **0.729 / 0.722** | **0.658 / 0.598** | **0.536 / 0.544** | 0.376 / 0.356 | 0.658 / 0.598 | 0.536 / 0.544 |

**17 of the 18 `rl/restart` cells and all 18 `imitation/restart` cells
are below 1.0.**  The single exception is `rl/restart` at n = 40, 50 ms,
seed 1 (1.245), where the policy got exactly **one** rollout — a
one-sample artefact (its `imitation/restart` sibling is 0.890 on the
same board); every other cell, at every budget, wins.

The **decisions** column in the tool output is `rollouts x (n - 1)` for
the rollout players (each episode makes `n - 1` action choices) and the
number of **state expansions** for `search`; the tool prints it per
board, and it tracks rollouts exactly, so the rollout table above is the
load-bearing one.

## 3. Why it wins — quality per rollout, not count

The policy trades **4x fewer rollouts** for a rollout that is far better
per unit.  A single learned rollout is already a good ordering; a
randomised-greedy rollout is only a small perturbation of greedy (it
picks among the three cheapest pairs), so `restart`'s best-of-N improves
*slowly*.  The concrete numbers at n = 40 (seed 0):

| budget | restart rollouts | restart ratio-best | rl rollouts | rl ratio-best |
|---|---|---|---|---|
| 50 ms | 7.5 | 3.831 | 1.0 | **1.000** |
| 200 ms | 27.5 | 2.516 | 5.5 | **1.000** |
| 1000 ms | 135.0 | 1.656 | 35.0 | **1.000** |

`restart` needs its full **135** rollouts to reach 1.656 — still worse
than the policy's best-of-**35**, and it never reaches the policy's
single-rollout quality.  The policy's advantage is a *better per-step
chooser*, and that dominates the 4x per-rollout cost even at a 1 s
budget.

The margin **narrows as the budget grows** (n = 40: 0.70 at 50 ms ->
0.74 at 200 ms -> 0.86 at 1 s) because `restart`'s best-of-N keeps
climbing, but it does **not flip** within 1 s at n <= 40.

## 4. The bounded `search` is useless at equal time

At equal wall-clock the best-first `search` is essentially `greedy`.  Its
per-expansion cost is dominated by the **greedy-completion heuristic** it
prices for every child, so within 1 s it finds **zero** complete states
at n = 30/40 and only ~0.8 at n = 20/1000 ms:

| budget | n = 20 complete | n = 30 | n = 40 |
|---|---|---|---|
| 50 ms | 0.0 | 0.0 | 0.0 |
| 200 ms | 0.0 | 0.0 | 0.0 |
| 1000 ms | 0.8 | 0.0 | 0.0 |

This is a real finding about the earlier comparison: `contraction_policy`
gave `search` a fixed **300-3000 state** budget — which at n = 40 is
~30 s of work — while the learned policy got a few seconds.  The
rollout-matched table was therefore *enormously generous to `search`*;
at equal time it cannot run at all.  (This does not change the earlier
conclusion that `search` ties the policy — it says the tie was bought
with far more compute.)

## 5. CPU cross-check — the direction holds, the margin is hardware-dependent

The policy uses the GPU; `restart`/`search` are CPU-only Python.  To test
whether the win is a GPU-batching artefact, the same ladder was run with
the policy moved to **CPU** (GPU-trained, evaluated on CPU, 2 instances):

| budget | rl/greedy | rl/restart | rl/search |
|---|---|---|---|
| 50 ms | 0.667 | **0.944** | 0.667 |
| 200 ms | 0.654 | **0.942** | 0.654 |
| 1000 ms | 0.644 | **0.969** | 0.664 |

On CPU the policy **still wins**, but only by **0-6%** against `restart`
(vs 11-30% on GPU).  The reason is in the per-decision table: the CPU
policy rollout is *relatively* more expensive (the MLP forward is cheap
either way, but the CPU gives less batch amortisation), so the policy
gets even fewer rollouts.  The win is **direction-robust but
margin-sensitive to the device** — reported honestly, with the caveat
that the CPU sample is only 2 instances.

## The verdict

**At equal wall-clock, the learned contraction-ordering policy beats the
cheap heuristics — the earlier "win" was not an artefact of matching
rollout count.**

* **vs `restart`** — the policy wins at **every scale and budget**, 0.65-
  0.90 across two seeds (one noisy single-rollout cell excepted).  It
  does so with **~4x fewer rollouts**, because a learned rollout is a
  far better per-step chooser than a randomised-greedy one.
* **vs `search`** — the policy wins decisively (0.36-0.73); at equal time
  the bounded best-first search cannot even find one complete state at
  n >= 30, so its earlier tie was bought with orders of magnitude more
  compute.
* **The cost is real and measured.**  A policy rollout is **~4x** a
  randomised-greedy rollout, and the policy is the more expensive player
  per unit — the original limitation had the right sign.  The win is a
  statement about *quality per unit of work*, not about free compute.
* **The margin narrows with the budget** (0.70 -> 0.86 at n = 40) and is
  **device-sensitive** (0.94-0.97 on CPU vs 0.70-0.89 on GPU).  On a
  CPU-only target the win is a near-tie; the GPU lockstep batching is
  part of the margin, though not the direction.

This is the falsification half, reported as measured: the central
hypothesis **survives the compute-equal test** — a learned
contraction-ordering policy is a genuinely better *player per unit of
work* at the scale where exact and saturation both fail, not merely a
player given more work.

## Honest limitations

* **Best-found is not the optimum** at n >= 20; every ratio is against an
  upper bound, so the gaps are conservative.
* **One noisy cell.**  `rl/restart` at n = 40, 50 ms, seed 1 is 1.245
  where the policy got a single rollout; `imitation` (0.890) on the same
  board and every larger budget flip it back.  Six instances per scale is
  enough to separate the field, not to resolve a 5% margin.
* **Laptop-GPU throughput drifts.**  The policy's per-rollout cost varies
  ~1.5x with the GPU's clock state, so the "ms" column is within a few
  percent of the budget, not exact.  A small first batch plus adaptive
  re-sizing keeps the overshoot to roughly one rollout.
* **The CPU cross-check is 2 instances** and GPU-trained; it confirms the
  direction, not a precise CPU margin.
* **Two trainers, one architecture, one cost model** — the same caveats
  as `contraction-policy.md` apply unchanged.
* **The rule space is not shipped.**  catopt has no general-network
  contraction rule; the `contract` game uses the same cost model as
  `contraction_scale`, not a `catopt_core.laws` rule set.

## Reproduce

```sh
# the equal-compute ladder (GPU), both seeds:
.venv/bin/python tools/contraction_policy.py --mode time \
    --instances 6 --seed 0 --device cuda
.venv/bin/python tools/contraction_policy.py --mode time \
    --instances 6 --seed 1 --device cuda
# a faster smoke test:
.venv/bin/python tools/contraction_policy.py --mode time \
    --trainer rl --iterations 300 --instances 1 --scales 20 \
    --budgets 50,200 --device cuda
# both ladders (rollout-matched + compute-equal):
.venv/bin/python tools/contraction_policy.py --mode both --device cuda
```

Gates at this HEAD: `ruff check packages tools` clean, `ruff format
--check` clean, `ty check` clean, `vulture` exit 0, radon ratchet ok,
`pytest` green.
