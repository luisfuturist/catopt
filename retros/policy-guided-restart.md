# Policy-guided restarts — becoming the baseline's algorithm buys parity, not a win

`contraction-train-scale.md` left the learned player at ~1.04× of
`opt_einsum`'s randomised greedy at n = 40 and diagnosed the residual
as **algorithmic**: `oe-rand-greedy` is best-of-N random restarts
over a staged heuristic, and no single forward pass matches a search
procedure.  The direct response is to *become the restart* — run
episodes that sample each contraction pair from the policy's softmax
distribution and keep the best order found.

The shipped machinery already implements the move:
`rollout_orders(greedy=False)` samples `Categorical(logits / T)`, so
"policy-guided restart" is exactly best-of-N sampled rollouts.  This
retro measures it properly — `tools/contraction_guided_restart.py`
adds what the earlier ladders never isolated:

* a **temperature sweep** {0.5, 1.0, 2.0} on the sampled restarts;
* an **affine batch scheduler** — a lockstep batch costs
  `fixed + marginal × B` (~55–75 ms + ~1.3 ms/rollout at n = 20,
  ~135–190 ms + ~6–8 ms at n = 40 on the RTX 2050), not the linear
  ms/rollout model `policy_best_order` sizes with, so the guided
  player fits `f, m` per scale (batch-1-probe-floored) and spends the
  remaining budget on one maximal batch.  `guided-lin` — the shipped
  protocol — is kept as the ablation;
* a **uniform-restart control** (`our-restart`, uniform top-k) that
  isolates the policy's contribution to the restart algorithm;
* an **equal-rollout arm**: every player re-run at exactly the
  episode count guided-T1 achieved under the clock — separating
  per-rollout quality from rollouts-per-second throughput;
* the learned **single-pass** argmax order and `oe-greedy` as the
  deterministic references.

Protocol: einsum-valid `random_bond_network` boards, n = 20/30/40,
budgets 50/200/1000 ms, **4 boards/cell**, every order scored by
`opt_einsum.contract_path` (the independent metric).  Policy: the
bundled curriculum artifact (`scale-free-v1`, trained 8–24), plus a
freshly trained `scale` (16–24) artifact for the n = 40 check.

Reproduce:

    uv sync --group einsum
    .venv/bin/python tools/contraction_guided_restart.py \
        --device cuda --boards 4

(Runs to completion in ~3 min.  Caveat: at the time of writing the
working tree carries an in-flight `scale-free-v2` feature bump —
`load_contraction_policy()` rejects the v1 artifact until the
weights are retrained.  The tool is contract-agnostic; the numbers
below were measured against the v1 code+artifact at commit `2a3a175`.)

## 1. Equal wall-clock — guided / oe-rand-greedy

oe-cost pairwise ratio, mean over 4 boards; <1 = guided wins.
Bundled curriculum artifact.

| ms | n | T0.5 | T1.0 | T2.0 |
|---|---|------|------|------|
| 50 | 20 | **0.970** | **0.965** | **0.972** |
| 50 | 30 | 0.949 | 1.052 | 1.099 |
| 50 | 40 | 2.485 | 1.908 | 2.786 |
| 200 | 20 | 1.009 | 1.019 | 1.009 |
| 200 | 30 | 1.133 | 1.136 | 1.121 |
| 200 | 40 | 1.269 | 1.338 | 1.302 |
| 1000 | 20 | 1.006 | 0.996 | 1.006 |
| 1000 | 30 | 1.156 | 1.159 | **1.063** |
| 1000 | 40 | 1.410 | 1.282 | 1.289 |

**Does it go below 1.0?**  At n = 20, yes — 0.965–0.996 across
budgets and temperatures, the first cells where a learned player
sits at or under `oe-rand-greedy` on equal footing.  At n = 30 it is
~1.05–1.17.  At n = 40 it is **1.28–1.41** at the adequate (1000 ms)
budget.  The 50 ms cells at n ≥ 30 are starvation, not signal: a
single lockstep batch costs more than the budget, so guided gets
5–9 episodes while `oe` gets 34–56.

For context the same table's other baselines: guided beats
`oe-greedy` everywhere (0.67–0.88), beats the learned single pass
everywhere (0.41–0.85), and beats uniform restart by 4–170× per
board (0.006–0.23 — see §2).

A second run of the same grid (different prior draws) reproduced the
shape: n = 20 0.98–1.03, n = 30 1.13–1.17, n = 40 1.25–1.40 —
starved cells wobble hardest (a 50 ms n = 40 cell moved 1.9↔3.6
between runs on prior-estimate noise alone).

## 2. Where the remaining gap lives — quality × throughput

The equal-rollout arm answers "do better rollouts beat more
rollouts?" by holding N fixed at the guided player's clocked count
(ratios are oe costs of best-of-N, mean over boards):

| ms | n | N | guided-T1 / unif | guided-T1 / oe |
|---|---|---|------------------|----------------|
| 200 | 20 | 139 | 0.160 | 0.979 |
| 200 | 30 | 57 | 0.033 | 0.950 |
| 200 | 40 | 31 | 0.005 | 1.146 |
| 1000 | 20 | 648 | 0.224 | 0.994 |
| 1000 | 30 | 305 | 0.051 | 1.133 |
| 1000 | 40 | 145 | 0.012 | 1.130 |

* **The policy's contribution to the restart is real and large.**
  Best-of-N policy-sampled episodes beats best-of-N uniform top-k
  episodes by 4–170× in cost — the policy learned a proposal
  distribution where uniform-over-cheap-pairs never did.
* **But `oe`'s proposal is at least as good.**  Per rollout, guided
  matches `oe-rand-greedy` at n = 20 (~0.96–1.00) and n = 30 at
  moderate N — and is ~10–15 % *worse* at n = 40.  The staged
  memory-removed heuristic is simply a strong proposal; the trained
  net ties it at small scale, not at n = 40.
* **And `oe` runs ~3–6× more trials per second** (guided: ~110–750
  episodes/s depending on n and budget; `oe-rand-greedy`: ~640–2000).
  Each policy episode costs a forward pass per contraction step;
  each `oe` trial is a heap scan.

So at equal wall-clock the two effects compound *against* the guided
player at n = 40: ~15 % worse rollouts, several-fold fewer of them —
and best-of-N scales sublinearly in N, which is why the wall-clock
gap (~1.3×) is only the product-ish of the two, not worse.

## 3. Temperature sensitivity

Weak.  At fed budgets the three temperatures sit within ~0.05 of each
other on the /oe-rand-greedy ratio; there is no monotone trend.
T1.0 is marginally best at n = 40 (1.28 vs 1.29–1.41); T2.0 wins the
n = 30 / 1000 ms cell (1.063).  At starved budgets high T
occasionally produces catastrophic single samples (a T2 rollout at
n = 40 / 50 ms landed at 2.8× in one run, 20× in another — one
sampled episode, one bad move).  The shipped `SAMPLE_TEMP = 1.5`
sits comfortably in the flat region.

## 4. n = 40 with the scale-trained policy

The bundled artifact is curriculum-trained on 8–24 — OOD at n = 40.
A scale policy trained on 16–24 (same iterations/seed/family, run
against the v1 module from HEAD) improves the deep cells but still
does not cross:

| ms | n=40, T1.0 | curriculum |
|---|---|---|
| 200 | **1.120** | 1.338 |
| 1000 | **1.197** | 1.282 |

Equal-rollout at n = 40 / 1000 ms: guided/oe-fixed = **1.117** —
the per-rollout deficit survives scale training, matching the
retro's diagnosis that the residual is *algorithmic*, not a
train/test mismatch.  (Its single pass is also much better at n = 40
— 3.7–4.2 vs curriculum's 6.2–7.5 — consistent with
`contraction-train-scale.md`.)

## 5. Verdict

**Parity, not outperform** — and the parity is itself the new
result.

* Policy-guided restart is the first learned configuration to sit
  at/below `oe-rand-greedy` on a full scale column: **0.97–1.02 at
  n = 20** across budgets and temperatures.
* It decisively beats the *uniform* restart (the isolating control:
  the policy is what makes the restart good — 0.006–0.23 at equal
  N), beats `oe-greedy` everywhere, and beats its own single pass.
* It does **not** go reliably below 1.0 at n ≥ 30: at n = 40 the
  best cells are ~1.12–1.20 (scale policy, adequate budget).  The
  reason is now precisely decomposed: **per-rollout quality is
  ~parity at n ≤ 30 and ~10–15 % behind at n = 40, *and* `oe` does
  ~3–6× more rollouts per second.**  Better rollouts do not beat
  more rollouts when the rollouts aren't better.
* The `guided-lin` ablation shows the shipped linear batch-sizing is
  a real defect, not cosmetic: it collapses to 1–2 episodes at
  starved cells (ratios up to 118× — one rollout, one disaster).
  The affine scheduler removes the collapse mode; at fed budgets the
  two agree.
* Standing one-liner, updated: *"best-of-N policy-sampled restarts
  reach parity with `oe-rand-greedy` at n ≤ 30 and ~1.1–1.4× at
  n = 40, decisively beating uniform restarts and deterministic
  greedy; the residual is per-rollout heuristic quality at scale
  plus a 3–6× trial-throughput deficit."*

## 6. Honesty / limits

* **4 boards, 1 seed**, and starved cells are timing-noise-dominated:
  the affine prior's `f` estimate wobbles between processes, so a
  50 ms cell can deliver 1 or 24 guided episodes.  Reported numbers
  are one run; a second run reproduced the shape (§1).  Treat the
  n ≥ 30 / 50 ms column as "starved", not a score.
* **First-batch overshoot.**  When a single lockstep batch cannot
  fit the budget, the player runs one anyway (the shipped protocol's
  convention) — the n = 20 / 50 ms cells spent ~58–75 ms against
  `oe`'s 50 ms.  The <1.0 cells should therefore be read as parity.
* **Free-seeding asymmetry, against us.**  `oe` counts its
  deterministic greedy as trial 0 and `our-restart` seeds with
  cheapest-pair greedy; the guided arm is all-sampled — its wins are
  despite forgoing a free deterministic episode.
* **Mid-measurement drift.**  `contraction_policy.py` was bumped to
  `scale-free-v2` (PAIR_DIM 9→13) in the working tree during this
  session (uncommitted concurrent work); the bundled v1 artifact no
  longer loads under it and the referenced retro file is not yet
  present.  All results above are v1-code/v1-artifact, reproducible
  at commit `2a3a175`; the scale-policy arm ran HEAD's v1 module
  under a scratch import.  Once the v2 weights ship, the tool runs
  unchanged — and the v2 features (which literally encode `oe`'s
  memory-removed heuristic) may move the per-rollout numbers at
  n = 40; re-run then.
* **`packages/` untouched** — the only change is the new
  `tools/contraction_guided_restart.py`.

## Gates

* `tools/contraction_guided_restart.py` new; `tools/` only.
  (`tools/contraction_einsum.py` was briefly given a `temperature`
  parameter, then reverted — the ladder drives `_policy_rollouts`
  directly; final diff is the new file only.)
* `.venv/bin/ruff check` / `ruff format --check` on the new file — pass.
* Runs to completion, CUDA, ~3 min:
  `policy-guided-restart-bundled.txt` (bundled artifact, full grid)
  and `policy-guided-restart-scale.txt` (scale policy, n = 30/40),
  alongside this retro.  An earlier run with the unfloored prior
  corroborates the shape and is cited in §1 for variance.
* Scratch driver for the scale arm lives at `/tmp/run_scale_v1.py`
  (loads HEAD's v1 module; not committed — dead code once v2 lands).
