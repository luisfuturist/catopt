# Distilling the strong teacher — the edge partially transfers, the wall-clock gap doesn't

`contraction-train-scale.md` closed the training-distribution lever
(scale training took the learned policy to ~1.04× of `oe-rand-greedy`
at n = 40) and left one untested lever: **distill the strong teacher
itself**.  Its §2 had the tantalising hint — a supervised net fit to
the *deterministic* `oe-greedy` rolled out *better than its teacher*
(roll/teacher 0.82).  If a student can beat the order it was trained
to copy, maybe supervising on `oe-rand-greedy`'s trajectories imports
its advantage too.

The caveat, flagged then and measured now: `oe-rand-greedy`'s edge is
best-of-N *restarts* — a search procedure.  Its returned path is the
argmin of ~10²-10³ stochastic trials, and a single order's per-step
labels cannot carry the restart structure.  `tools/contraction_distill.py`
runs the experiment with two distillation arms that bracket the
question:

* `oe-best` — supervise on the *selected* order (what
  `ce.oe_random_greedy_order` returns): the retro's proposal.
* `oe-all` — supervise on **every trial's** trajectory (collected via
  `RandomGreedy.setup`'s trial function + `ssa_to_linear`, capped at 8
  trials/board): the policy learns the teacher's per-step
  *distribution* — the closest a one-pass policy can get to sampling
  like the teacher.

against `rl` (the curriculum REINFORCE regime, retrained fresh) and
`dp` (the existing imitation trainer: exact DP ≤ 14, our best-of-32
restart fallback above), plus a `dp-small` control arm (pure-DP labels
only, the shipped imitation recipe).

Reproduce:

    uv sync --group einsum
    .venv/bin/python tools/contraction_distill.py --device cuda \
        --iterations 1800 --per-n 24 --epochs 200 \
        --teacher-repeats 128 --teacher-budget 0.75 --trial-cap 8
    # control arm separately:
    .venv/bin/python tools/contraction_distill.py --device cuda \
        --arms dp-small --iterations 1

**Feature contract note.**  `catopt_torch.contraction_policy` was
mid-edit (feature-v2 work in flight) during this experiment.  Both runs
bound the working tree at import time, which reported
`scale-free-v1` (`STATE_DIM=7`, `PAIR_DIM=9`; file md5
`2333e552f953570853d7c9a2fabe4990` at launch, unchanged afterwards);
all arms in a run share one contract.  The mechanism is
contract-agnostic — nothing here depends on the extra v2 columns.

Train/test split: 24 boards/scale × scales (8,12,16,20,24) for
training, held-out seeds for eval — same curriculum, same boards for
every supervised arm.  Training cost (CUDA): rl 289 s; teacher trials
6 s (128/board); oe-best fit 54 s (1800 rows); oe-all fit 514 s
(14 400 rows); dp fit 96 s; dp-small 18 s.

## 1. Held-out agreement — the distilled policies do not decide like the teacher

Walk `oe-rand-greedy`'s selected order on unseen n = 40 boards (8
boards, 0.75 s teacher budget); per state: top-1 agreement and the
Spearman ρ between policy scores and the negated pair cost; plus
roll/teacher (policy's own argmax rollout over the teacher order's
cost):

| player | top-1 | ρ vs −cost | roll/teacher |
|---|---|---|---|
| our-greedy | 0.103 | — | — |
| rl | 0.298 | 0.881 | 5.43 |
| oe-best | 0.269 | 0.840 | 5.15 |
| oe-all | 0.298 | 0.542 | **3.12** |
| dp | 0.131 | 0.920 | 518 123 (!) |
| dp-small | 0.202 | 0.793 | 29.1 |

No distilled arm reproduces more than ~30 % of the teacher's decisions
on held-out boards — the same ceiling the RL policy hits.  Two details
matter: `oe-all`'s ρ drops to 0.54 (it *unlearned* pure cost-greedy
ranking — it samples sub-cheapest pairs, like the stochastic teacher),
and it rolls out closest to its teacher (3.12× — still >1, but half of
RL's 5.4×).

## 2. The ladder — equal single-pass, equal rollouts, equal wall-clock

Pairwise vs `opt_einsum` players, mean over 3 boards, both cost models
(they agree to 3 decimals everywhere; oe model shown, ours identical).
`<1` = learned wins.

**Single-pass** (argmax rollout vs `oe-greedy` / 1-episode
`RandomGreedy`):

| n | rl | oe-best | oe-all | dp | dp-small |
|---|---|---|---|---|---|
| 20 | 1.05 | **1.01** | 1.09 | 108 | 3.62 |
| 30 | **0.83** | 1.46 | 0.98 | 9 251 | 4.08 |
| 40 | 4.15 | 3.60 | 2.89 | 2 670 | 13.8 |

Single argmax passes are brittle at n = 40 for *every* learned player
(even `oe-rand-1` is 1.12× `oe-greedy`) — the policy is a sampler, not
a deterministic player.  No news there.

**Equal rollouts** (best-of-64 sampled vs `RandomGreedy(max_repeats=64)`):

| n | rl | oe-best | oe-all | dp | dp-small |
|---|---|---|---|---|---|
| 20 | 0.96 | 1.00 | 1.07 | 2.17 | 1.17 |
| 30 | 0.94 | 0.95 | 1.10 | 55.2 | 1.90 |
| 40 | 1.26 | 1.17 | **0.91** | 127 | 1.38 |

**The headline row**: `oe-all` at n = 40 beats `oe-rand-greedy` at
equal sample count — 0.914 — the first time *any* policy arm beats the
strong teacher in any equal comparison at the top scale.  Distilling
the trial *distribution* (not just the argmin order) transferred enough
of the restart edge that the student's smoothed sampling distribution
out-searches 64 raw teacher trials.  `oe-best` ≈ `rl` ≈ parity — a
single selected order indeed cannot teach restarts.

**Equal wall-clock** (`policy / oe-rand-greedy`, ms per instance):

| budget | n | rl | oe-best | oe-all | dp | dp-small |
|---|---|---|---|---|---|---|
| 200 | 20 | 1.00 | 1.01 | 1.18 | 1.81 | 1.18 |
| 200 | 30 | 1.02 | 1.01 | 1.33 | 44.9 | 1.94 |
| 200 | 40 | 1.21 | 1.35 | 1.45 | 203 | 2.12 |
| 1000 | 20 | 1.00 | 0.99 | 1.07 | 1.79 | 1.15 |
| 1000 | 30 | 1.01 | 0.96 | 1.06 | 16.6 | 1.84 |
| 1000 | 40 | 1.56 | 1.20 | 1.30 | 104 | 1.61 |

At equal wall-clock nobody beats the teacher: `oe-rand-greedy` runs
~1 300 heuristic trials/s while a policy rollout costs ~2-7 ms at
n = 40 (~150-500 rollouts/s).  `oe-best` reaches ~0.96-1.35×;
`oe-all`'s equal-rollouts edge evaporates because it cannot afford
enough samples.  Against `oe-greedy`, every distilled/RL arm wins at
adequate budget (0.62-0.99 at n = 40).

## 3. Side finding — the `dp` arm's collapse is the *fallback* teacher's fault

The `dp` arm (existing imitation on the curriculum) collapses
catastrophically at n = 40 (roll/teacher 518 123; ladder ratios
100-2 700×).  Mechanism, measured on the training boards themselves:

| n | teacher labels | outer-product step fraction | order cost vs oe-rand-greedy |
|---|---|---|---|
| 12 | exact DP | 0.015 | 0.96 (as good as oe) |
| 20 | restart fallback | 0.246 | 47.7× |
| 24 | restart fallback | 0.225 | 32.6× |

`oe-rand-greedy`'s orders are outer-free (0.000).  The fallback teacher
— our best-of-32 top-3 restart — produces outer-eager orders 30-48×
worse than the oe teacher, and 60 % of the curriculum boards (n =
16,20,24) carry its labels.  The arm faithfully learned a bad teacher.
The `dp-small` control (exact DP only, n ≤ 14) is sane — top-1 0.202,
1.4-2.1× off the strong teacher at n = 40 — so the collapse is the
fallback labels plus lost scale coverage, not supervision per se.

## 4. Verdict

* **Partially transfers.**  Distilling `oe-rand-greedy`'s *trial
  distribution* (`oe-all`) is the first arm to beat the teacher at
  equal rollouts at n = 40 (0.914) — so part of the advantage is
  learnable per-step knowledge, not purely restart scaffolding.
  Distilling the *selected order* (`oe-best`) only reaches RL parity,
  confirming the caveat: one order can't teach the restart structure.
* **The residual is now more precisely structural**: not "best-of-N
  can't be learned" but "best-of-N at ~0.7 ms/trial out-samples a
  ~5 ms/rollout policy".  The distilled distribution is competitive
  *per sample*; the throughput gap (≈4-10×) is what keeps
  `oe-rand-greedy` ahead at equal wall-clock.
* **Held-out agreement stays low (~0.27-0.30) yet rollouts keep
  improving** — consistent with the sufficiency probe's picture: the
  features express only a slice of the teacher's choices, and the
  learned smooths rather than copies.
* **The existing imitation teacher is unsafe at curriculum scales on
  the bond family**: its `dp_max` fallback silently labels large boards
  with orders ~40× worse than the target.  If imitation is ever re-run
  there, the fallback must be `oe`-strength or the arm should stop at
  `dp_max`.

## 5. Honesty / limits

* **3 boards per eval cell, 1 seed** — the same caveat as the prior
  retro; the `oe-all` 0.914 could be partly wobble, though it is
  consistent with its lowest roll/teacher and lowest ρ signature.
* **Training-time teacher ≠ eval-time teacher budget**: labels came
  from ≤128 trials (≈0.75 s cap); the eval teacher fills its full
  budget (~1 300 trials at 1 s).  `oe-all`'s equal-rollouts win is
  thus over a *stronger-per-sample* teacher distribution than its own
  labels.
* **`oe-all` saw ≤8 trials/board** (trial-cap), not the full 128 —
  its dataset is biased toward early trials (including the
  deterministic trial 0).
* **Contract drift**: trained against `scale-free-v1` while the
  package was mid-edit; rerun under v2 before citing numbers elsewhere.
* **`dp-small` was a separate run** (`--arms dp-small`), same boards
  and eval seeds.

## Gates

`tools/`-only change (`contraction_distill.py` new; nothing else
touched — the package was left to the in-flight v2 edit):

* `.venv/bin/ruff check` / `ruff format --check` on the file — pass
* Runs to completion: `contraction-distillation-run.txt` (961 s
  training + eval) and `contraction-distillation-dp-small-run.txt`
* No pytest run — tools-only experiment, no shipped code changed
