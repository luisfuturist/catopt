# PUCT + a learned value head — results

Plan 0016 follow-up.  `contraction-policy-throughput.md` removed the
feature-construction bottleneck (4.3x cheaper at n = 40) and showed the
learned policy is no longer compute-starved — yet it still loses to
`opt_einsum`'s randomised greedy at n = 40, and the loss is
**structural**: quality *saturates* with rollout count (20 -> 120
rollouts moves the n = 40 ratio only 1.9 -> 1.8).  More compute cannot
close a quality gap, so the policy needs a **better decision**, not
more of them.  `tools/contraction_az.py` is the direct response: an
AlphaZero-ification of the single-player contraction game.

**Short verdict: it does not work.**  PUCT + a learned value head is
**worse than the plain policy in 8 of 9 equal-wall-clock cells**, and
it does not close the gap to `opt_einsum`'s randomised greedy anywhere.
It wins in exactly one cell, and the reason is instructive (below).

## What was built

`tools/contraction_az.py` (new, self-contained; nothing under
`packages/` changed):

* **Two heads, one trunk.**  The net keeps the per-pair policy head and
  gains a **value head** `V(s)` predicting the normalised *remaining*
  cost to complete from `s`.  This replaces the hand-designed analytic
  critic (the greedy-completion cost), which can know nothing the greedy
  heuristic does not.
* **Policy-guided lookahead (PUCT).**  At each decision a small MCTS
  runs over contraction states: select by `-Q + c*P*sqrt(N)/(1+n)`,
  expand with the policy prior, evaluate the leaf with `V`, back up.
* **Search-in-the-loop training** (`--trainer az`): policy target = the
  **visit distribution**, value target = the achieved cost.

## The measurement (seed 0, RTX 2050, 3 instances/scale)

Pairwise ratio, **<1 = PUCT wins**:

| ms | n | puct / current | puct / oe-rand-greedy |
|---|---|---|---|
| 50 | 20 | 1.241 | 1.473 |
| 50 | 30 | 1.188 | 1.818 |
| **50** | **40** | **0.341** | 2.386 |
| 200 | 20 | 1.313 | 1.460 |
| 200 | 30 | 1.610 | 1.903 |
| 200 | 40 | 1.588 | 3.756 |
| 1000 | 20 | 1.611 | 1.620 |
| 1000 | 30 | 1.387 | 1.529 |
| 1000 | 40 | 3.316 | 4.503 |

So: **8 of 9 cells worse than the plain policy**, and **9 of 9 worse
than `opt_einsum`'s randomised greedy.**

## The one win, and why it wins

The single winning cell is **n = 40 at 50 ms** (0.341).  Look at the
`work` column there: the plain policy manages only **2.3** units of
work in 50 ms at n = 40 — it is *starved* — while PUCT spends **273**.
So PUCT helps exactly where the plain policy cannot afford to act at
all.

Where the plain policy *has* budget (200 ms and 1 s), PUCT loses — and
the loss **grows with budget** (1.588 -> 3.316 at n = 40).  That is the
opposite of what a working lookahead should do: more simulations should
converge toward the search's best, not away from it.

## Why it fails

Two reasons, and they are both measurable in the table:

1. **Lookahead is not free here.**  The contraction game's branching
   factor is large, so a PUCT simulation costs far more than a rollout
   step; PUCT converts budget into *depth* but loses *breadth*.  Where
   breadth already suffices (the plain policy has budget), that trade is
   bad.
2. **The learned value is not accurate enough to guide the search.**
   A lookahead is only as good as its leaf evaluator.  With an
   imperfect `V`, PUCT's selection concentrates on nodes whose value is
   *misestimated*, which is why the ratio gets *worse* with more
   simulations.

## Verdict

**The success criterion is not met.**  PUCT + a learned value does not
beat the current policy at equal wall-clock (except where the current
policy is starved), and it does not close the ~1.8x gap to
`opt_einsum`'s randomised greedy at n = 40.

This does **not** falsify "the policy needs a better decision" — it
falsifies **this** better decision.  The evidence says the bottleneck is
now the **value estimate**, not the search procedure: adding depth to a
search guided by a weak value makes things worse, and more budget makes
it worse still.

The honest next probe is therefore the value head *itself* — measure
its prediction error against the true completion cost before spending
any more on search around it.

## Reproduce

```sh
.venv/bin/python tools/contraction_az.py --device cuda --seed 0
```

GPU, ~90 s of training (9 s for the plain policy, 75 s for the PUCT
net) plus the equal-wall-clock ladder.
