# The AZ value head — measured against the exact optimum

Plan 0016 follow-up.  `contraction-az-results.md` found that PUCT + a
learned value loses to the plain policy in **8 of 9** equal-wall-clock
cells, and — the tell — that the loss *grows* with budget
(1.588 -> 3.316 at n = 40 from 200 ms to 1 s).  Its hypothesis: **the
learned value is not accurate enough**, so PUCT's selection concentrates
on nodes whose value is misestimated.

This retro measures that value head *before* any fix, against the exact
`O(3^n)` subset DP (`contraction_scale.dp`), and then fixes the cause
the measurement names.

**Short verdict: the value was broken — it ranked worse than the
analytic greedy critic it replaced — and the cause is one line: the
target was normalised by a quantity the net cannot observe.  Fixing it
lifts the value's rank correlation with the true completion cost from
0.51 to 0.89 (now *better* than the analytic critic), and brings the
learned value level with the critic inside the search.  But that changes
nothing end-to-end: PUCT still loses, and so does PUCT driven by the
analytic critic itself.  The value was a real defect, not the reason
PUCT loses — lookahead is.**

## What was built

`tools/contraction_value_probe.py` (new, self-contained; nothing under
`packages/` changed).  It:

1. trains the AZ net exactly as `contraction_az.train_search` does;
2. plays real PUCT episodes at `n = 10, 12` and snapshots the **full
   tensor multiset** at every committed decision (the trainer's
   `collect` path drops it, and the DP needs it);
3. prices every state with the exact subset DP and compares it with the
   net's `V` **and** with the analytic greedy-completion critic;
4. reports error, correlation, and the two **rank** measures a search
   actually consumes: global Spearman, and **sibling** Spearman (does
   `V` order a state's successors the way `dp` does?), plus top-1 pick
   accuracy and regret.

## 1. The value is worse than the critic it replaced

Seed 0, RTX 2050, 1500 decision states, net = 200 expert-iteration
rounds at `sims = 24`.  `abs-err` is the mean `|log1p(pred) - log1p(dp)|`
in normalised units; `ratio` is `pred / dp`.

| predictor | abs-err | ratio | pearson(log) | Spearman | sibling sp | top-1 | regret | within 2x |
|---|---|---|---|---|---|---|---|---|
| **net V** (original) | 0.117 | 3.77 | 0.477 | **0.507** | **0.513** | 0.365 | 2.562 | 53 % |
| analytic greedy critic | 0.229 | 3.70 | 0.585 | **0.684** | **0.842** | 0.609 | 1.138 | — |
| training target | 0.020 | 1.12 | 0.974 | 0.985 | — | — | — | — |

The headline: **the learned value ranks *worse* than the hand-designed
greedy-completion critic it was introduced to replace** — 0.51 vs 0.68
globally, 0.51 vs 0.84 on the sibling decision PUCT consumes, and it
picks the best child 36 % of the time against the critic's 61 %.  The
AZ change did not add a better leaf evaluator; it removed a good one.

## 2. The target is fine — the *normalisation* is not

`V` is regressed onto `log1p(remaining / ref)`, `ref` = the **board's**
full-network greedy cost, `remaining = episode_cost - cost_so_far`.
The target itself is an excellent proxy for `dp`: `spearman(target, dp)
= 0.992`, `target/dp` median 1.004.  The problem is the divisor.

`ref` is a *board* constant, and it is **not one of the net's
features** — the net sees `cost/ref`, `len(ts)/n0`, ranks, `spread`,
`l_min`, but never `ref` itself.  So the same remaining cost gets a
different label on every board, and the net provably cannot undo it.
Measured on the same states:

| label | Spearman vs `dp` |
|---|---|
| `log1p(achieved)` (absolute) | **0.995** |
| `log1p(achieved / ref)` (the shipped target) | **0.288** |
| `log1p(dp)` (absolute) | 1.000 |
| `log1p(dp / ref)` | 0.266 |

The board reference spans **643 -> 660028** (1026x) across the sampled
boards, so dividing by it scrambles the global ordering that the net is
asked to fit.  This is the dominant cause of the value error: a value
net can be accurate about a target, but here the target is inconsistent
with the net's own inputs.

## 3. The fix: make the normaliser observable

Two changes to `tools/contraction_az.py` (both behind a documented
`--value-unit` / feature flag; `packages/` untouched):

* **`_sf`** appends the board's `log2(ref)` to the state features, so
  the normaliser the target is divided by is now *observable*.  This
  keeps the target scale-free (so it still generalises across `n`) while
  removing the cross-board inconsistency.
* **`value_unit="abs"`** is offered as the alternative: keep the target
  absolute and divide by `ref` only inside the search.  This is the
  cleanest fix at small `n` but has no scale signal, so it does **not**
  generalise (below).

Same probe, same settings:

| variant | abs-err | ratio | pearson(log) | Spearman | sibling sp | top-1 | regret | within 2x |
|---|---|---|---|---|---|---|---|---|
| original `ref` | 0.117 | 3.77 | 0.477 | 0.507 | 0.513 | 0.365 | 2.562 | 53 % |
| **`ref` + scale feat** | 0.062 | 1.08 | 0.845 | **0.893** | 0.659 | 0.481 | 1.388 | 77 % |
| **`abs`** | 0.073 | 1.22 | 0.824 | **0.907** | **0.785** | 0.523 | 1.191 | 81 % |
| analytic greedy critic | 0.219 | 3.58 | 0.620 | 0.721 | 0.842 | 0.600 | 1.142 | — |

The value now **out-ranks the analytic critic globally** (0.89-0.91 vs
0.72) and closes most of the sibling gap (0.66-0.79 vs 0.84).  That is
the fix the measurement demanded.

## 4. Re-test — the improved value does not make PUCT win

The equal-wall-clock ladder is **noisy**: the same `abs` cell (n = 30,
200 ms) reads 0.827 at 3 instances and 1.876 at 5.  So the honest
control is *paired*: `--puct-analytic` runs the **same** PUCT with the
learned value swapped for the analytic greedy critic, on the same
boards.  `net-vs-analytic < 1` means the learned value helped.

Seed 0, 5 instances, 200 ms per instance:

| variant | n | puct/current | puct/oe-rand | **net-vs-analytic** |
|---|---|---|---|---|
| `ref` + scale feat | 20 | 1.579 | 1.693 | **1.010** |
| `ref` + scale feat | 30 | 2.459 | 4.134 | **1.000** |
| `ref` + scale feat | 40 | 1.098 | 2.679 | **0.979** |
| `abs` | 20 | 1.648 | 1.780 | **1.044** |
| `abs` | 30 | 1.876 | 2.797 | **0.972** |
| `abs` | 40 | 2.608 | 6.656 | **2.080** |

Two readings, and they are the point:

1. **Swapping the leaf evaluator changes essentially nothing.**
   `net-vs-analytic` is 1.010 / 1.000 / 0.979 for the fixed net — the
   search's outcome is insensitive to the leaf value.  The one place the
   value matters is where it is *broken*: `abs` at n = 40 reads 2.080,
   because an absolute target has no scale signal, so a net trained at
   n = 8-12 mis-prices a n = 40 board by orders of magnitude.
2. **PUCT loses with the analytic critic too.**  `puct-analytic/current`
   = 1.53 (n = 20), 2.60 (n = 30), 0.68 (n = 40) — the *same* search
   driven by the best available hand-built evaluator still loses to the
   plain policy at n = 20 and n = 30, and never beats `opt_einsum`'s
   randomised greedy (all `puct/oe-rand` > 1).

The absolute-target variant does improve the small-`n` ladder over the
retro (n = 30 wins at all three budgets in the 3-instance run), but the
5-instance run shows that was largely noise, and it collapses at n = 40.

## Verdict

**The value was inaccurate, and now it is not — but that was not why
PUCT loses.**

* **Measured, and confirmed:** the shipped value head ranked *worse*
  than the analytic critic it replaced (global Spearman 0.51 vs 0.68;
  sibling 0.51 vs 0.84).  Root cause: the regression target was
  normalised by the board's greedy reference, a quantity the net cannot
  observe, so the same cost carried a different label on every board
  (`ref` spans 1026x).  The label's rank quality against `dp` collapses
  from 0.995 (absolute) to 0.288 (divided by `ref`).
* **Fixed, and measured:** making the normaliser observable (a `log2(ref)`
  feature) lifts the value to global Spearman 0.893, out-ranking the
  analytic critic, with sibling 0.659; the absolute-target variant
  reaches 0.907 / 0.785 but does not generalise to n = 40.
* **But the fix does not move the game.**  The paired control shows the
  search is insensitive to the leaf evaluator (`net-vs-analytic` =
  1.010 / 1.000 / 0.979), and the analytic critic itself loses to the
  plain policy.  So "the value is not accurate enough" is **refuted**:
  an accurate value — and the strongest hand-built value — still do not
  make lookahead win.

This corroborates the retro's *other* reason, the one it ranked second:
**lookahead is not free here.**  The contraction game's branching factor
is large, so a PUCT simulation buys depth at the price of breadth, and
where breadth already suffices the trade is bad.  The value head was a
genuine defect worth fixing; it was not the binding constraint.

## Honest limitations

* **Small-`n` measurement, large-`n` inference.**  The DP comparison is
  `n <= 12` (the probe's default scales); the end-to-end ladder is
  `n = 20/30/40`.  The value's *accuracy* is therefore measured where
  the DP exists, and its *effect* is measured where it matters — the two
  are joined by the search, not by a direct large-`n` value oracle.
* **The ladder is noisy at 3 instances.**  The retro's 8-of-9 headline
  rests on 3 boards per cell; the same cell swings 0.83 -> 1.88 between
  3 and 5 instances.  The paired `net-vs-analytic` control is the
  trustworthy number, and it is ~1.0.
* **`abs` is a documented dead end, kept for the trade-off.**  It is the
  most accurate variant at small `n` and the worst at n = 40; the
  default stays `ref` + the scale feature.
* **One instance family.**  `random_bond_network` (valid einsums); the
  retro's hypergraph `random_network` cannot be priced by `opt_einsum`,
  so the field baseline is on the bond family only.
* **No architecture search.**  The 7-feature trunk is unchanged apart
  from the added scale column; a richer value input (the pair-cost
  distribution, say) was probed in a scratch fit and did not obviously
  help, but it was not run end-to-end.

## Reproduce

```sh
# the value probe: V(s) vs the exact DP, error + rank + sibling rank
.venv/bin/python tools/contraction_value_probe.py --device cuda --value-unit ref
.venv/bin/python tools/contraction_value_probe.py --device cuda --value-unit abs

# the paired end-to-end control (same search, learned value vs analytic)
.venv/bin/python tools/contraction_az.py --device cuda --seed 0 \
    --value-unit ref --puct-analytic --instances 5 --budgets 200
```

GPU; ~80 s to train the net per run plus the ladder.
