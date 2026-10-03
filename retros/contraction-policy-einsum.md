# Learned contraction ordering vs opt_einsum — the field baseline

Plan 0016 follow-up, third half.  `contraction-policy.md` and
`contraction-policy-compute.md` measured a learned contraction-ordering
policy beating **our own** players (`greedy`, `restart`, bounded
`search`) at equal wall-clock, generalising from n = 8-12 to
n = 20/30/40.  Both retros flagged the same hole in their own
limitations sections: *every baseline was ours*.  `opt_einsum` ships
the players that matter — its staged `greedy` and the exact `optimal`
DP — so it is the honest yardstick.  This retro is the falsification
attempt against it.

**The honest answer: the learned policy is a genuine player — it is the
only one of ours that survives the swap — but the retros' headline
"we beat greedy" was measured against a baseline `opt_einsum` beats by
two orders of magnitude.**

* **Our own `greedy` collapses on valid einsums** — 71-940x worse than
  best-found, and `restart` (best of ~1000 randomised rollouts at 1 s)
  is still 3.7-600x worse.  `opt_einsum`'s greedy is 1.2-2.1x.
* **The learned policy beats `opt_einsum`'s greedy at n = 20**
  (0.85-0.88x, both seeds, every budget) and at n = 30 for budgets
  >= 200 ms (0.83-0.90x) — but **loses at n = 40** (1.10-3.64x, only
  approaching parity at 1 s).
* **It does not beat `opt_einsum`'s randomised greedy at equal time** —
  n = 20 is a near-tie (0.93-1.06x), n = 30 and n = 40 lose
  (1.05-5.6x).  `optimal` cannot be reached at all: opt_einsum's exact
  DP already times out at n = 10 with a 1 s budget.
* **A methodological finding of its own:** the retros' instance family
  (`contraction_scale.random_network`) is **not a valid einsum** — its
  indices repeat up to 11x — so it cannot be handed to opt_einsum at
  all.  On a valid-einsum family the ordering problem is *different*
  (sparse, outer-product-heavy) and our players collapse.

Reproduce (RTX 2050, ~2.5 min/seed including 60 s of training):

```sh
uv sync --group einsum
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 0
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 1
```

## What was added

**`opt_einsum` is a new opt-in dependency group**, mirroring the
existing `oracle` group for `egglog`: it is *not* in `dev`, nothing
under `packages/` imports it, and `tools/contraction_einsum.py` exits
with a `uv sync --group einsum` hint when the group is absent.  The pin
is `opt_einsum>=3.4.0,<4` — 3.4.0 was published 2024-09-26, well over
the 7-day floor, and the upper bound keeps the lockfile off an untested
major.

**`tools/contraction_einsum.py`** — self-contained; **nothing under
`packages/` changed**.  It reuses the existing machinery
(`contraction_scale`'s players and cost model, `contraction_policy`'s
policy net and training loop) and adds only the comparison:

* a **valid-einsum** instance family (`random_bond_network`) plus four
  genuine attention/MLP einsums;
* order-producing wrappers for every player (our players and
  opt_einsum's), so **one order can be scored with either cost model**;
* an **equal-wall-clock** ladder: our `restart` and the learned policy
  fill the budget, opt_einsum's `RandomGreedy` fills it via `max_time`,
  and `optimal` runs under a deadline (DNF if it cannot finish);
* two ratio tables — the ratio to best-found under **our** cost model
  and under **opt_einsum's** own `contract_path` cost — so the
  comparison cannot be tilted by our cost model.

## 0. The retro's instance family is not an einsum

`contraction_scale.random_network` builds a *hypergraph*: each index is
sampled into every tensor independently, so a single index can appear in
many tensors.  Max multiplicity by scale:

| n | retro family | bond network | indices (retro / bond) |
|---|---|---|---|
| 8 | 5 | 2 | 4 / 11 |
| 20 | 9 | 2 | 10 / 26 |
| 30 | 11 | 2 | 15 / 39 |
| 40 | 7 | 2 | 20 / 55 |

An einsum sums an index that appears twice and keeps one that appears
once; an index appearing 3+ times has no pairwise-contraction semantics
at all (opt_einsum still *parses* it, but contracts to a different
result).  So the retro's boards **cannot be given to opt_einsum**, and
the comparison must move to a family where every index appears in at
most two tensors (a *bond*) or once (an *open* leg).  The tool's
`random_bond_network` uses the same n and the same seeds; it is
necessarily sparser (indices >= 1.4 n), because einsum-validity forces
it.

## 1. Control — where the exact optimum is known

At n = 8 our exact subset DP and opt_einsum's exact DP agree to the
FLOP factor (`oe-opt / 2dp = 1.000 / 0.995`), and the learned policy
reaches the optimum (`learned/dp = 1.000-1.007`).  Our `greedy` is
1.9-3.0x the optimum; opt_einsum's greedy is 1.29-1.59x.  At n = 10
opt_einsum's `optimal` already **times out** (DNF) on 2 of 3 instances
at seed 0 and all 3 at seed 1, at a 1 s budget — so there is no exact
reference beyond n = 10.

## 2. Mechanism — why our greedy loses

`opt_einsum`'s greedy is *staged*: Hadamard products first, then
maximise removed size, and **outer products only when forced**.  Our
cheapest-pair greedy happily contracts a cheap disjoint pair early and
then pays for the oversized intermediate.  Mean outer-product steps per
order:

| n | our-greedy | oe-greedy | steps |
|---|---|---|---|
| 20 | 4.0 / 4.3 | 0.0 | 19 |
| 30 | 6.3 / 5.7 | 0.0 | 29 |
| 40 | 7.7 / 8.3 | 0.0 | 39 |

(seed 0 / seed 1.)  That single difference explains the two-orders-of-
magnitude gap: opt_einsum never forms an outer product, we do so
several times per episode.

## 3. The equal-wall-clock ladder

`ratio to best-found`, mean over 3 seeded networks, **seed 0 / seed 1**.
The our-cost and oe-cost ratios coincide to ~3 significant figures for
every player (they differ only in the 2nd-3rd decimal for the orders
that contain outer products, because opt_einsum weights those at factor
1 instead of 2), so a single table is shown — the two independent
metrics rank the orders identically.

**50 ms per instance**

| n | our-greedy | our-restart | learned | oe-greedy | oe-rand-greedy | oe-optimal |
|---|---|---|---|---|---|---|
| 20 | 98.9 / 71.5 | 6.5 / 10.6 | **1.06 / 1.01** | 1.22 / 1.16 | 1.08 / 1.09 | DNF |
| 30 | 282.7 / 100.5 | 277.8 / 95.6 | **1.35 / 1.45** | 1.33 / 1.23 | 1.00 / 1.00 | DNF |
| 40 | 597.5 / 598.2 | 597.5 / 598.2 | 5.54 / 4.20 | 1.66 / 1.37 | 1.00 / 1.00 | DNF |

**200 ms per instance**

| n | our-greedy | our-restart | learned | oe-greedy | oe-rand-greedy | oe-optimal |
|---|---|---|---|---|---|---|
| 20 | 98.9 / 71.6 | 5.2 / 5.9 | **1.04 / 1.01** | 1.22 / 1.16 | 1.01 / 1.06 | DNF |
| 30 | 288.7 / 147.1 | 80.2 / 29.8 | **1.12 / 1.23** | 1.34 / 1.52 | 1.00 / 1.03 | DNF |
| 40 | 602.7 / 743.0 | 399.9 / 510.7 | 2.82 / 1.82 | 1.72 / 1.57 | 1.00 / 1.00 | DNF |

**1000 ms per instance**

| n | our-greedy | our-restart | learned | oe-greedy | oe-rand-greedy | oe-optimal |
|---|---|---|---|---|---|---|
| 20 | 101.0 / 71.7 | 3.8 / 5.0 | **1.06 / 1.01** | 1.25 / 1.17 | 1.00 / 1.01 | DNF |
| 30 | 302.5 / 147.7 | 30.0 / 18.0 | **1.08 / 1.22** | 1.38 / 1.55 | 1.00 / 1.05 | DNF |
| 40 | 886.4 / 939.9 | 558.9 / 533.0 | 2.06 / 1.96 | 2.10 / 1.85 | 1.00 / 1.00 | DNF |

Reading it honestly:

* **Our own baselines are the story.**  Cheapest-pair `greedy` is 71-940x
  worse than best-found; `restart` — best of 1100-1200 randomised-greedy
  episodes at 1 s — is still 3.8x (n = 20) to 559x (n = 40) worse.
  `opt_einsum`'s greedy, which is *free* (one pass, < 1.5 ms), is
  1.2-2.1x.
* **The learned policy is the only competitive player of ours.**  It is
  within 1.01-1.06x at n = 20 and 1.08-1.45x at n = 30, i.e. it is the
  player that closes almost all of the gap our heuristics leave.
* **At n = 40 it is starved.**  At 50 ms the policy gets exactly **one**
  rollout (its per-rollout cost is ~36 ms at n = 40) and lands 4-5.5x;
  by 1 s it has ~40 rollouts and reaches 2.06 / 1.96x.
* **`optimal` never runs** at n >= 20: opt_einsum's exact DP is DNF at
  every budget.  (It is also DNF at n = 10 / 1 s on 2 of 3 instances.)

## 4. Learned vs opt_einsum, head to head

`learned / opponent` (mean over 3 seeds; **< 1 = the learned policy
wins**).  The two cost models give the same column to ~3 significant
figures.

| budget | n | / oe-greedy (s0 / s1) | / oe-rand-greedy (s0 / s1) |
|---|---|---|---|
| 50 ms | 20 | **0.867 / 0.878** | **0.994 / 0.933** |
| 50 ms | 30 | 1.051 / 1.279 | 1.353 / 1.453 |
| 50 ms | 40 | 3.639 / 3.441 | 5.537 / 4.201 |
| 200 ms | 20 | **0.851 / 0.876** | 1.026 / **0.955** |
| 200 ms | 30 | **0.900 / 0.847** | 1.124 / 1.207 |
| 200 ms | 40 | 1.680 / 1.272 | 2.824 / 1.821 |
| 1000 ms | 20 | **0.851 / 0.868** | 1.062 / **0.994** |
| 1000 ms | 30 | **0.846 / 0.827** | 1.079 / 1.182 |
| 1000 ms | 40 | 1.105 / 1.136 | 2.061 / 1.962 |

**Answers to the three questions.**

1. *Does the learned policy beat opt_einsum's `greedy` at equal time?*
   **Yes at n = 20 and n = 30 (>= 200 ms), no at n = 40.**  The n = 20
   win is clean and seed-stable (0.85-0.88x across both seeds and all
   three budgets); n = 30 wins at 200 ms / 1 s (0.83-0.90x) but loses at
   50 ms; n = 40 loses everywhere (1.10-3.64x), though it closes to
   1.10-1.14x by 1 s.
2. *Does it approach `optimal`?*  **Untestable at scale** —
   opt_einsum's exact DP cannot finish past n = 10 within 1 s.  Where it
   can be checked (n = 8) the learned policy *equals* the optimum.
3. *Does it beat opt_einsum's randomised greedy (its strongest anytime
   player)?*  **No, not at scale.**  n = 20 is a tie (0.93-1.06x); n = 30
   and n = 40 lose (1.05-5.6x).  `RandomGreedy` is the fair equal-time
   opponent — it is opt_einsum's own restart — and it wins.

## 5. Real einsums (attention / MLP block)

Four genuine, small einsums, ordered by every player (oe-cost of each
order shown; all orders verified numerically identical):

| case | ops | our-greedy | our-restart | learned | oe-greedy | oe-optimal |
|---|---|---|---|---|---|---|
| attention `QK^T` (`bqe,bke->bqk`) | 2 | 4096 | 4096 | 4096 | 4096 | 4096 |
| attention `@V` (`bqk,bkh->bqh`) | 2 | 4096 | 4096 | 4096 | 4096 | 4096 |
| bilinear `XWY^T` (`btd,de,btf->bte`) | 3 | 3.28e4 | 3.28e4 | 3.28e4 | 3.28e4 | 3.28e4 |
| MLP stack `XW1..W5` (`btd,de,ef,fg,gh,hi->bti`) | 6 | 5.24e5 | 5.24e5 | 5.24e5 | 1.21e6 | 5.24e5 |

On the small block einsums **everything ties** — 2-3 operands leave no
ordering freedom worth having.  On the 6-operand MLP stack our greedy
is **optimal** (5.24e5, equal to `oe-optimal`) and opt_einsum's greedy
is 2.3x worse.  So the collapse in §3 is *not* universal: it is specific
to the sparse, outer-product-heavy multi-operand regime, which is
exactly the regime the large-n ladder lives in.

(A full transformer block is **not** a single einsum — its Q/K/V share
the batch axis, so an index would appear 3+ times — which is why the
real cases are the block's individual matmuls and their chains.)

## The verdict

**A decisive, partly-negative result.**  Falsified against the standard
tool:

* **Our self-built baselines are weak, and that is the load-bearing
  finding.**  On valid einsums our `greedy` is 71-940x worse than
  best-found and `restart` 3.7-600x, while opt_einsum's greedy is
  1.2-2.1x and its randomised greedy is 1.00-1.05x.  The retros' wins
  ("0.41-0.69x greedy", "0.65-0.90x restart") were measured against a
  baseline opt_einsum beats by two orders of magnitude.  **The earlier
  wins were against weak self-built baselines.**
* **The learned policy itself is not falsified.**  It is the only
  player of ours that competes: it beats opt_einsum's greedy at n = 20
  (0.85-0.88x, seed-stable) and n = 30 (0.83-0.90x at >= 200 ms).  The
  policy learned something our heuristics did not — it largely stops
  forming outer products.
* **But it does not beat the standard tool overall.**  It loses to
  opt_einsum's greedy at n = 40, loses to opt_einsum's randomised
  greedy everywhere at scale, and cannot be compared to `optimal` at
  all.  On the honest yardstick the learned policy is a *competitive*
  player, not a better one.
* **The instance family mattered more than the policy.**  The retro's
  hypergraph family is not a valid einsum; on a valid-einsum family the
  problem is sparse and outer-product-heavy, which is precisely where
  our players fail.

This is the falsification half, reported as measured: the central
hypothesis is **not supported** as stated ("a learned policy beats the
cheap heuristics at scale") — on opt_einsum's players it does not — but
the narrower claim survives: the learned policy is a *real* player that
beats the standard greedy in the mid-scale regime, and the weak part of
the earlier result was the baselines, not the learner.

## Honest limitations

* **The family had to change.**  opt_einsum cannot represent the retro's
  hypergraph boards, so the comparison is on a valid-einsum family with
  the same n and seeds.  It is sparser (indices >= 1.4 n) and therefore
  outer-product-heavy; the absolute ratios are family-specific, though
  the *direction* (our greedy collapses, the policy does not) is not.
* **Three instances per scale.**  Enough to separate the field
  (100x gaps), not to resolve a 5-10% margin; the n = 20 learned-vs-
  oe-greedy win is seed-stable, the n = 30 win flips at 50 ms.
* **`optimal` is a non-comparison.**  opt_einsum's exact DP times out at
  n >= 10 within 1 s, so "approaches optimal" is untestable at the
  scales that matter.  Our own subset DP reaches n = 12 but is also
  intractable at n >= 20.
* **One policy, one architecture, one cost model.**  Same caveats as the
  two prior retros: a 64-wide MLP trained by REINFORCE on n = 8-12 with
  a greedy critic; a different net or critic would shift the ratios.
* **`oe-rand-greedy` is deadline-bounded, not iteration-bounded.**  It
  fills the budget with `max_time`, so it gets as many randomised-greedy
  trials as fit — the same anytime contract our `restart` and the policy
  get.  It is the fair equal-time opponent, and it wins.
* **Device-sensitive margin.**  The learned policy uses the GPU; the
  opt_einsum players are CPU-only Python.  As in
  `contraction-policy-compute.md`, GPU batching is part of the policy's
  margin — but here it does not overcome `RandomGreedy` at scale.
* **The anytime players vary run to run.**  `restart`, the policy and
  `oe-rand-greedy` are all wall-clock-bound, so their exact ratios move
  by ~10% between runs (a re-run of seed 0 gave `our-restart` 7.08 vs
  6.54 at n = 20 / 50 ms).  The deterministic sections — the control,
  the mechanism counts and opt_einsum's own greedy — reproduce exactly.
  No conclusion here rests on a margin below that spread.

## Reproduce

```sh
uv sync --group einsum
# the full comparison, both seeds (~2.5 min each on an RTX 2050):
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 0
.venv/bin/python tools/contraction_einsum.py --device cuda --seed 1
# a fast smoke test (no training):
.venv/bin/python tools/contraction_einsum.py --trainer none \
    --scales 20 --control 8 --instances 1 --budgets 50 --device cpu
```

Gates at this HEAD: `ruff check packages tools` clean, `ruff format
--check` clean, `ty check` clean, `vulture` exit 0, radon ratchet ok,
`pytest` green.  The default `dev` group is unchanged; `opt_einsum`
lives only in the new opt-in `einsum` group.
