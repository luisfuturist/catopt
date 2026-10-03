# Learned contraction ordering — can a policy beat the cheap heuristics?

Plan 0016 follow-up.  `contraction-scale.md` measured the player ladder
(greedy, one-step, bounded best-first `search`, randomised `restart`)
on seeded random tensor networks and found **headroom above greedy** —
up to 13x mean at n = 60 — but that the real baselines to beat are
`search` / `restart`, already 1.00-1.04 of best-found.  This retro is
the decisive test of the project's central hypothesis:

> Can a **learned** contraction-ordering policy beat the cheap
> heuristics at the scale where exact and saturation both fail?

**The honest answer: yes, but modestly — and only against the right
baseline.**  A small policy trained by REINFORCE on n = 8-12, with a
greedy-completion critic, **generalises to n = 20/30/40** and:

* **beats `greedy` decisively** — 0.41-0.69x its cost at every scale;
* **matches the bounded `search`** — 0.93-1.09x (a tie, seed-dependent);
* **matches or beats `restart`** — 0.78-1.03x as a single rollout, and
  **0.73-0.87x when it is allowed 64 restarts** like `restart` is.

So the large headroom is still **above `greedy`**, not above the strong
cheap players: the learned policy *ties* the best cheap player as a
single rollout and *beats* it once both sides get restarts.  It is a
real but not a dramatic win, and it is not uniform across instances.

Reproduce: `.venv/bin/python tools/contraction_policy.py --device cuda`
(~4-5 min on an RTX 2050; seeded, deterministic on the GPU).

## What was built

`tools/contraction_policy.py` — self-contained; **nothing under
`packages/` changed**.  It imports the players and cost model straight
from `contraction_scale` (so the comparison is on the *same* cost), and
re-implements only the game and the policy.

The **game** is the contraction rule space, framed on states directly
(not on an e-graph):

* **state** — the multiset of remaining tensors (a partial contraction),
  described by 7 **scale-free** features (progress, cost-so-far over the
  greedy reference, rank statistics, the pair-cost spread and minimum,
  all normalised by the state's own scale);
* **action** — one legal `contract` of a tensor pair, described by 9
  scale-free features (the pair's log-union cost relative to the state
  minimum, operand and intersection sizes, the merged size, ranks, and
  the pair's cost percentile);
* **reward** — the (negative) pairwise cost the move incurs, exactly the
  structural cost `contraction_scale` prices.

The policy is a small MLP scoring `(state (+) pair) -> logit` — the
shape `catopt_torch.rl.PolicyNet` uses — so the action space is open
(the number of pairs changes every step).  Two trainers are provided.

### Trainer 1 — REINFORCE with a greedy-completion critic

Each step's advantage is `V(s) - inc - V(s')`, with `V` the
**greedy-completion cost** from the state — a cheap, scale-free
heuristic critic.  Summed over an episode it telescopes to
`greedy_ref - episode_cost`, so maximising it *is* beating greedy.  This
is the "reward = the resulting cost" framing, with a hand-built value
baseline instead of a learned one.

### Trainer 2 — imitation of the exact DP-optimal order

At n = 8-12 the subset DP is affordable, so the exact optimal
contraction tree is a dense label.  The second policy is trained by
masked cross-entropy on the optimal pair at each state along a
DP-optimal order — the *strongest* available signal, included as an
ablation.

## Two bugs that cost real work (and are the honest part)

Both were found while chasing the policy's instability; both are
methodological, not tuning:

1. **Log-probs and advantages were built in different orders.**
   `logps` was collected step-major, `advantages` episode-major, so the
   elementwise product in the REINFORCE loss was a *permutation* — a
   silently corrupted gradient.  Fixed by flattening both episode-major.
2. **Un-standardised advantages diverge.**  A single catastrophic move
   makes `V(s')` astronomically large, so `A` has a heavy negative tail
   that dominates the gradient.  With the order bug *and* the
   un-standardised tail, the greedy policy blew up to 14x / 5.5x the
   n = 8-12 optimum at 900 / 1800 iterations.  With both fixed (order
   aligned, advantage standardised over the batch) it improved
   monotonically, 1.24 -> 1.14 -> 1.09 over the same 300 / 900 / 1800
   iterations.

Training is **deterministic on the GPU** (verified: identical parameter
hashes across separate processes), so the numbers below reproduce.

## 1. Controls — the policy reaches near-optimal where it can be checked

Players / the **exact** subset DP optimum (mean over 6 seeds), after
1800 RL iterations and 300 imitation epochs:

| n | greedy | restart | rl | rl-restart | imitation | imitation-restart |
|---|---|---|---|---|---|---|
| 8 | 1.242 | 1.023 | 1.149 | 1.139 | 1.154 | 1.000 |
| 10 | 1.328 | 1.089 | 1.120 | 1.083 | 1.214 | 1.032 |
| 12 | 2.277 | 1.184 | 1.176 | 1.034 | 1.214 | 1.005 |

The learned policy is **1.12-1.18x** the true optimum at n = 8-12 —
clearly better than greedy (1.24-2.28x), a little behind `restart`
(1.02-1.18x).  It is a genuine player, not a random ordering.

## 2. At scale — ratio to best-found

`ratio to best-found`, mean over 6 seeded networks, `seed 0 / seed 1`
(the exact optimum is unknown at these sizes, so "best-found" is an
upper bound on the true optimum):

| n | greedy | one-step | search | restart | rl | rl-restart | imitation | imitation-restart |
|---|---|---|---|---|---|---|---|---|
| 20 | 5.76 / 6.08 | 1.59 / 1.54 | 1.28 / 1.25 | 1.59 / 1.56 | **1.13 / 1.37** | **1.06 / 1.04** | 1.74 / 1.47 | 1.01 / 1.00 |
| 30 | 2.76 / 2.90 | — | 1.12 / 1.16 | 1.36 / 1.35 | **1.06 / 1.24** | **1.02 / 1.04** | 2.93 / 2.19 | 1.00 / 1.00 |
| 40 | 12.29 / 12.13 | — | — | 1.49 / 1.49 | **1.17 / 1.01** | **1.00 / 1.00** | 10.91 / 2.20 | 1.00 / 1.00 |

(`one-step` runs only at n <= 20, `search` only at n <= 30, as in
`contraction_scale`.)

## 3. At scale — the pairwise ratios that actually decide it

Ratio-to-best is *not* the same as head-to-head, because one
catastrophic instance dominates a player's mean.  The direct
learned-vs-cheap ratios (mean over 6 seeds, `seed 0 / seed 1`; **< 1
means the learned policy wins**):

| comparison | n = 20 | n = 30 | n = 40 |
|---|---|---|---|
| rl / greedy | **0.41 / 0.45** | **0.67 / 0.69** | **0.54 / 0.55** |
| rl / restart | **0.78** / 0.98 | **0.84** / 1.02 | 1.03 / **0.87** |
| rl / search | 0.93 / 1.09 | **0.97 / 0.99** | — |
| rl-restart / restart | **0.73 / 0.74** | **0.82 / 0.84** | **0.87 / 0.87** |
| imitation / restart | 1.10 / 1.07 | 2.25 / 1.92 | 5.72 / 2.03 |
| imitation-restart / restart | **0.69 / 0.72** | **0.81 / 0.81** | **0.87 / 0.87** |

Reading it honestly:

* **vs `greedy` — a decisive win, both seeds, every scale.**  0.41-0.69x.
  This is the easy half; it only confirms the policy learned something.
* **vs `search` (bounded best-first) — a tie.**  0.93-1.09x across the
  two seeds; seed 0 wins by ~7 % at n = 20, seed 1 loses by ~9 %.  No
  reliable win.
* **vs `restart` (best of 64 randomised-greedy episodes) — comparable.**
  0.78x (seed 0) and 0.98x (seed 1) at n = 20; ~0.84-1.02x at n = 30;
  1.03x / 0.87x at n = 40.  A single deterministic rollout holds its own
  against 64 randomised-greedy restarts.
* **With restarts (best of 64 sampled policy rollouts) vs `restart` — a
  clean win at every scale and seed**: 0.73-0.87x.  The learned scorer
  is a strictly better per-step chooser than "cheapest pair" inside the
  same restart budget.

## 4. The imitation ablation — strong supervision overfits

The imitation policy is trained on the **exact optimum**, yet it is the
*worse* learner at scale.  It degrades monotonically with n
(`imitation / restart`: 1.10 -> 2.25 -> 5.72 at seed 0), reaching
greedy-level or worse by n = 40, while the RL policy improves.

The reading: the DP-optimal *local* choice depends on global structure
that the local features cannot see, so supervised imitation learns
spurious small-n patterns that do not transfer.  REINFORCE, optimising
the *actual* objective through a scale-free critic, learns a robust
local rule instead.  (With restarts the imitation policy recovers —
0.69-0.87x restart — because the bad greedy rollout is discarded.)

## The verdict

**A learned policy can beat the cheap heuristics at the scale where
exact and saturation both fail — but by a modest, baseline-dependent
margin, not a landslide.**

* It **generalises from n = 8-12 to n = 20/30/40** with no
  retraining and no large-n data — the scale-free features are doing
  their job.
* It **beats `greedy` decisively** (0.41-0.69x) and **beats `restart`
  once both are given restarts** (0.73-0.87x).
* As a **single rollout it ties the bounded `search`** (0.93-1.09x) and
  is **comparable to `restart`** (0.78-1.03x) — it does not dominate
  them.
* So the honest headline is: **the headroom `contraction-scale.md`
  measured is above `greedy`; the strong cheap players already capture
  most of it, and the learned policy only edges past them when given the
  same restart budget.**  A learned contraction-ordering policy is a
  viable *player*, not a new regime.

This is the falsification half of the result, reported as measured: the
central hypothesis is **partly supported** — learning generalises and
wins against greedy and against `restart`-with-restarts — and **partly
not** — a single learned rollout does not beat the best cheap player.

## Honest limitations

* **Best-found is not the optimum.**  At n >= 20 there is no exact
  reference; every scale ratio is against an upper bound, so the
  measured gaps are conservative.
* **Seed sensitivity.**  The rl/search and rl/restart comparisons flip
  around 1.0 between the two training seeds (the n = 20 rl/search is
  0.93 at seed 0, 1.09 at seed 1).  Six instances per scale is enough to
  separate greedy from the field, not to resolve a 5 % margin.
* **Not compute-equal.**  A policy rollout scores every pair with a
  small MLP per step; `greedy`/`restart` sort scalar pair costs.  The
  *number of rollouts* is matched (1 vs 1, 64 vs 64), the per-step cost
  is not — the learned player is the more expensive one.
* **Two trainers, one architecture, one cost model.**  All numbers use
  the classic scalar-mult cost and a 64-wide MLP.  A bigger net, a
  learned critic, or a launch-aware cost would shift the exact ratios.
* **The rule space is not shipped.**  catopt has no general-network
  contraction rule; the `contract` game here is built on the same cost
  model as `contraction_scale`, not a `catopt_core.laws` rule set.
* **The policy only orders.**  It chooses among legal contractions; it
  cannot change the answer (ADR 0003 invariant 5).

## Reproduce

```sh
.venv/bin/python tools/contraction_policy.py --device cuda --seed 0 --instances 6
# seed 1 (robustness):
.venv/bin/python tools/contraction_policy.py --device cuda --seed 1 --instances 6
# faster / CPU smoke test:
.venv/bin/python tools/contraction_policy.py --trainer rl --iterations 300 \
    --instances 1 --scales 20 --device cpu
```

Gates at this HEAD: `ruff check packages tools` clean, `ruff format
--check` clean, `ty check` clean, `vulture` exit 0, radon ratchet ok,
import-linter 4/4 kept, `pytest` 3325 passed / 31 skipped.
