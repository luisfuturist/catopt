# Contraction synthesis — distilled oe-all + lockstep driver + guided restart beats the teacher at equal wall-clock

Three retros each carried half of a result:

* `contraction-distillation.md` — a policy distilled on **every**
  `oe-rand-greedy` trial trajectory (`oe-all`) beats the teacher at
  **equal rollouts**, n = 40: 0.914.  Its per-sample quality is at or
  above the teacher's; what it lacked was trial throughput.
* `contraction-throughput-v2.md` — the vectorised `_lockstep_rollouts`
  driver in `catopt_torch.contraction_policy` runs **more episodes/s
  than the baseline's trials/s** (~1.8x at n = 40).  What it lacked
  was per-rollout quality (RL-v1 sits ~1.17 at n = 40 fed).
* `policy-guided-restart.md` — the affine-scheduled guided-restart
  protocol (`tools/contraction_guided_restart.py`) is the honest way
  to spend that throughput: best-of-N sampled episodes under a
  wall-clock budget, with an equal-rollout arm that separates quality
  from rate.

This retro runs the synthesis the three imply: **train the oe-all
distilled policy and drive it with the lockstep-guided ladder.**  The
prediction was per-rollout ~0.9 x throughput ~1.8x -> ~0.9 territory
at n = 40.

**That is what happens, and it is the first outright win.**  The
distilled guided player beats `oe-rand-greedy` at equal wall-clock at
n = 40 — **8 of 9 T = 1.0 cells below 1.0 across three seeds
(0.91-1.01), every cell below 1.0 at its best temperature**, best cell
**0.888** (seed 8, 1000 ms, T = 2.0) — while the bundled RL-v1 policy
under the *identical* driver, boards and protocol stays >= 1.0
(1.01-1.33).  The attribution is clean: the same driver, the same
boards; only the weights differ.

Reproduce (RTX 2050, `uv sync --group einsum` first):

```sh
# 1. retrain the oe-all arm and serialise it (new --save flag writes
#    <stem>-<arm>.pt in the shipped artifact format):
.venv/bin/python tools/contraction_distill.py --device cuda \
    --arms oe-all --save /tmp/distill-synthesis
# -> /tmp/distill-synthesis-oe-all.pt  (scratch artifact; the bundled
#    .pt is untouched)

# 2. the ladder against every baseline, distilled policy:
.venv/bin/python tools/contraction_guided_restart.py --device cuda \
    --boards 4 --policy /tmp/distill-synthesis-oe-all.pt

# 3. the attribution arm — bundled RL-v1 artifact, same seed/boards:
.venv/bin/python tools/contraction_guided_restart.py --device cuda \
    --boards 4

# 4. seed stability (seed offsets also redraw the board sets):
.venv/bin/python tools/contraction_guided_restart.py --device cuda \
    --boards 4 --seed 4 --policy /tmp/distill-synthesis-oe-all.pt
.venv/bin/python tools/contraction_guided_restart.py --device cuda \
    --boards 4 --seed 8 --scales 30,40 \
    --policy /tmp/distill-synthesis-oe-all.pt
```

Transcripts: `contraction-synthesis-distill-run.txt`,
`contraction-synthesis-guided-{seed0,seed4,seed8}.txt`,
`contraction-synthesis-guided-rlv1-seed0.txt` alongside this file.

## 1. Equal wall-clock — distilled guided / oe-rand-greedy

oe-cost pairwise ratio, mean over 4 boards, T = softmax temperature
of the sampled episodes; **<1 = the learned player wins**.  Seed 0.

| ms | n | T0.5 | T1.0 | T2.0 |
|---|---|------|------|------|
| 50 | 20 | 1.220 | 1.601 | 2.858 |  (starved: 1 episode, see §5)
| 50 | 30 | **0.911** | **0.961** | 1.112 |
| 50 | 40 | **0.921** | **0.981** | 1.043 |
| 200 | 20 | **0.971** | **0.967** | 1.025 |
| 200 | 30 | **0.947** | **0.943** | 1.134 |
| 200 | 40 | **0.908** | **0.928** | 1.009 |
| 1000 | 20 | **0.957** | **0.961** | **0.998** |
| 1000 | 30 | **0.951** | **0.934** | 1.000 |
| 1000 | 40 | **0.970** | **0.971** | 1.003 |

**Every fed cell at n >= 30 is below 1.0 at T <= 1**, and n = 20 is
below 1.0 at both fed budgets.  The same ladder's other anchors at fed
cells: guided beats `oe-greedy` everywhere (0.51-0.90), beats the
distilled single pass everywhere (0.36-0.92), and sits at 0.005-0.26
of the uniform-restart control's cost (4-200x better) — the policy,
not restarts per se, does the work.  The starved n = 20 / 50 ms cell
loses to all three (1 episode — §5).

## 2. Attribution — the bundled RL-v1 policy on the same driver/boards

Identical run, bundled `contraction_policy_curriculum.pt` (RL-v1),
seed 0, same boards.  /oe-rand-greedy:

| ms | n | T0.5 | T1.0 | T2.0 |
|---|---|------|------|------|
| 50 | 30 | 1.076 | 1.076 | 1.072 |
| 50 | 40 | 1.109 | 1.006 | 1.103 |
| 200 | 20 | 1.006 | 0.993 | 0.996 |
| 200 | 30 | 1.118 | 1.114 | 1.006 |
| 200 | 40 | 1.114 | 1.035 | 1.007 |
| 1000 | 20 | 1.010 | 1.002 | 0.996 |
| 1000 | 30 | 1.145 | 1.038 | 0.967 |
| 1000 | 40 | 1.334 | 1.197 | 1.137 |

(n = 20 / 50 ms row omitted — starved to 1 episode, 1.29-2.09.)

RL-v1 is itself much improved under the lockstep driver vs the
guided-restart retro's numbers (n = 40: 1.0-1.3 where that retro had
1.28-1.41 — the throughput half alone bought ~15-20 points) — but it
**never crosses below 1.0 at n = 40**.  The distill-vs-RL gap at
n = 40 / 1000 ms is 0.97 vs 1.20: **the missing piece was per-rollout
policy quality, exactly as the throughput retro predicted** ("the
residual is no longer throughput — it is per-rollout quality").

## 3. The mechanism, decomposed — quality x rate

Equal-rollout arm (all players re-run at the guided-T1 clocked count
N; gfix/oe-fixed = per-rollout quality):

| seed | n | N=~90 | N=512 | N=1536 | clocked episodes vs oe trials (1000 ms) |
|---|---|-------|-------|--------|------------------------------------------|
| 0 | 40 | 1.039 | 0.987 | 1.046 | 1536 vs 708 (2.2x) |
| 4 | 40 | 1.055 | 1.008 | 1.010 | 1536 vs 730 (2.1x) |
| 8 | 40 | 1.061 | 0.952 | **0.913** | 1536 vs 737 (2.1x) |

Distilled per-rollout quality at n = 40 sits at **parity-to-slightly-
better than oe's proposal** (0.91-1.06 across seeds and N; RL-v1 runs
1.2-1.38 at N >= 512 on the same boards — its old ~15% deficit).  Supervising on the
teacher's *trial distribution* really did transfer the proposal, not
just the argmin.  Then the driver supplies ~2.1x the trial volume, and
best-of-1536 beats best-of-~720 by 3-9% — that is the whole win: the
two halves are roughly independent multipliers and both now sit >= 1.

Temperature: the optimum wanders per board set (T0.5 best on seed 0,
T2 best on seed 8) — the distilled net already *is* a calibrated
sampling distribution, so the sweep only nudges it.  **T = 1.0 is the
robust default**: <= ~1.01 in every fed cell at every seed.

Corroboration from the training run's own eval
(`contraction-synthesis-distill-run.txt`, 3 boards, shipped linear
`policy_best_order` scheduler over the same lockstep driver): oe-all
vs oe-rand-greedy at n = 40 reads **0.886 @ 200 ms** and 1.019 @
1000 ms — the win is not an artefact of the affine scheduler; the
affine protocol simply spends the budget more evenly at 1000 ms.

## 4. Seed stability at the winning cells

n = 40, guided / oe-rand-greedy (T1 / best temp).  Each seed redraws
the board set too, so this is stability over *new instances*, not just
new sampling streams:

| budget | seed 0 | seed 4 | seed 8 |
|---|---|---|---|
| 50 ms | 0.981 / 0.921 | 0.972 / 0.972 | 0.983 / 0.965 |
| 200 ms | 0.928 / 0.908 | 0.983 / 0.983 | 0.949 / 0.944 |
| 1000 ms | 0.971 / 0.970 | 1.009 / 0.951 | 0.912 / **0.888** |

Nine T1 cells, one marginal over-run (1.009) — everything else below
1.0.  n = 30 / 1000 ms T1: 0.934 / 0.987 / 0.991; n = 20 / 1000 ms T1:
0.961 / 0.953.  The win is stable, not a seed artefact.

## 5. Single-pass argmax under the new driver — better, still not a player

The distill retro warned that a single argmax pass is brittle at
n = 40 (oe-all read 2.89x `oe-greedy` there).  On these boards the
distilled argmax pass lands /oe-greedy = **1.045 (seed 0), 1.275
(seed 4), 0.921 (seed 8)** — enormously better than RL-v1's 3.30 on
the same seed-0 boards, and one board set even beats oe-greedy
outright.  But it does not beat the restart players: vs oe-rand-greedy
the argmax reads 1.4-3.7 at n = 40.
Honest summary: argmax is now a plausible *fallback*, not a player —
sampling + best-of-N remains the whole point of the player.

Starved cells stay broken as before: at n = 20 / 50 ms the affine
prior's fixed-cost estimate exceeds the budget, so guided runs **one**
episode and loses (1.2-3.6 across seeds, meaningless); the same hit
n = 30 / 50 ms on seed 8 (1 episode, 3.1-7.0).  The n >= 30 / 50 ms
"win" cells conversely overspent (see §6) — treat all 50 ms numbers at
n >= 20 as noise-adjacent.

## 6. Honesty / limits

* **Batch overshoot inflates the 50/200 ms wins at n >= 30.**  The
  affine scheduler commits to one maximal batch; at n = 30/40 it
  spent ~20-90 % over budget (50 ms cells: ~62-94 ms; 200 ms cells:
  ~240-356 ms).  The **clean cells** — spent within ~5 % of budget —
  are n = 20 @ 200/1000 ms, n = 30 @ 1000 ms, n = 40 @ 1000 ms, and
  the distilled player wins *all* of them; the short-budget wins are
  real but partially bought with extra milliseconds.  At n = 20 /
  200 ms it *underspent* (135-160 ms) and still won.
* **4 boards/cell.**  Same caveat as every retro in this series; the
  seed reruns partially compensate (12 distinct n = 40 boards total,
  all won at T <= 2 at fed budgets).
* **Training nondeterminism.**  This oe-all run shows the retro's
  signature with stronger flavour: roll/teacher 3.08 (was 3.12),
  top-1 0.324 (was 0.298), rho-vs-cost **0.093** (was 0.542) — this
  copy unlearned cost-ranking even more thoroughly, i.e. it samples
  the teacher's distribution rather than ranking by pair cost.  Its
  best-of-64 at n = 40 also improved on the retro: **0.872** vs 0.914.
  Treat the exact rho as run-to-run wobble; the direction is the same.
* **`oe-rand-greedy` is per-board deterministic** given the trial
  count (trial r seeded by r); the policy's sampling stream is not —
  the pairwise ratios at equal clock are exact per board for oe,
  sampled for the policy.
* **The equal-rollout arm's N is the clocked count**, so "per-rollout
  parity" is measured at the policy's own volumes; at very small N
  (~90) oe's free deterministic trial 0 shows — gfix/oe reads
  ~1.04-1.06 there while the clocked arm still wins on volume.
* **Artifact is scratch** (`/tmp/distill-synthesis-oe-all.pt`,
  trainer `distill-oe-all`, `scale-free-v1` contract, git_sha
  62b84b8): deliberately not shipped — the bundled artifact stays the
  RL curriculum weights.  Reproduce with the §0 commands.

## 7. Verdict

**Outperform.**  The combination — oe-all distillation for
per-rollout quality, the lockstep driver for trial throughput, the
affine guided scheduler for spending it — is the first learned
contraction player to beat `opt_einsum`'s randomised greedy at equal
wall-clock at the top scale: **0.89-1.01 at n = 40 across three seeds
at fed budgets** (0.89-0.98 at the best temperature), 0.93-0.99 at
n = 30, 0.95-0.97 at n = 20.  Under the
same driver the RL-v1 policy stays >= 1.0 at n = 40, so the margin is
attributable to the distilled proposal quality, not the machinery.

Standing one-liner: *"distilling the teacher's trial distribution gave
the policy ~parity per rollout, and the lockstep driver gives it ~2x
the teacher's trial rate — best-of-more at parity quality beats
best-of-fewer, and the learned player finally wins at n = 40
(0.91-0.98, three seeds)."*

What is left on the table: the scheduler's batch overshoot (a
split-the-batch refinement would clean up the short-budget cells), the
starved-prior failure at 50 ms, and the fact that per-rollout quality
is *parity*, not better — the remaining upside is a proposal that is
strictly better than oe's (feature-v2's heuristic-encoded columns are
the natural next lever).

## Gates

* `tools/contraction_distill.py`: +73 lines — `--save` (serialise
  each trained arm as `<stem>-<arm>.pt` via the shipped
  `save_contraction_policy`, with provenance meta) and its `_save_arms`
  / `_git_sha` helpers.  No `packages/` edits; the bundled artifact
  untouched.  `contraction_guided_restart.py` unchanged — its existing
  `--policy` flag is the weights path.
* `.venv/bin/ruff check` / `ruff format --check` on the touched
  file — pass.
* No pytest run — tools-only experiment, no shipped code changed.
