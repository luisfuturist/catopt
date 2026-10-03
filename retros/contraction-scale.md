# Contraction ordering at scale — does the search break?

Plan 0016 stage 7 (ADR 0003).  The contraction-ordering precondition
(`contraction-precondition.md`) and ladder (`contraction-ladder.md`)
measured greedy vs the exact optimum on **small** boards (3-8 tensors)
and found that equality saturation *ties* the optimum — `saturate ==
optimal` on 60/60 chains.  But on a board small enough to enumerate,
saturation's **completeness** trivially equals optimality
(Catalan(4) = 14): the result was an artefact of the space being tiny,
not evidence that the search scales.

This retro tests the other end, as a falsification attempt:

> At a scale where the exact optimum is intractable, does the search
> actually break — and if it does, is some cheaper player materially
> worse?

**The search breaks, hard — and there is headroom above greedy.**
Equality saturation in the contraction-ordering rule space is
*exponential* (Catalan growth in e-class members); it stops finishing
at **n ≈ 12**, far below where the subset DP becomes intractable
(n ≈ 18) and far below the scale the players are meant to work at.
At n = 20…60 greedy is materially worse than the best found by cheaper
players — up to **13× mean, 28× worst** — so the domain is **open**,
with the honest caveat below about *which* search the headroom is
above.

Reproduce: `.venv/bin/python tools/contraction_scale.py`
(~3 min, CPU-only, seeded; `--instances K`, `--sat-deadline S`).

## What was measured

Seeded random **general tensor networks** (hypergraphs): `n` tensors,
mixed ranks/degrees, shared indices, sizes 2-8.  Four scales
(n = 20, 30, 40, 60) where the `O(3^n)` subset DP is hopeless, and five
**controls** (n = 8…16) small enough to enumerate exactly, to validate
the players.

Players, all on the same per-instance cost (the classic pairwise
`∏ sizes over the union of the two carried index sets`):

* **greedy** — contract the cheapest pair (classic, `O(n³)`);
* **one-step** — greedy with one step of lookahead (greedy completion);
* **search** — a bounded best-first search over contraction states,
  the abstract analogue of
  `catopt_orchestrator.diagram_search.search_moves`;
* **restart** — best of 64 randomised-greedy episodes;
* **saturate** — equality saturation in a *contraction-ordering* rule
  space, run under explicit `max_nodes` / `max_iterations` budgets and
  a wall-clock deadline;
* **dp** — the exact `O(3^n)` subset DP (controls only).

The rule space is catopt's own e-graph: a binary `contract` op whose
associativity (×2) + commutativity generate *every* binary contraction
tree over a leaf multiset — the whole contraction-order space.  A
structural cost function prices each `contract` node as the pairwise
cost above; it is additive, so the extractor's DAG-aware billing
recovers the exact pairwise cost.  catopt ships no general-network
contraction rule (only `assoc_matmul` for chains), so this is a
faithful extension of its e-graph, not a shipped rule set.

## 1. The search breaks — saturation is exponential

Per-scale, under a 6 s wall-clock deadline per run (`max_nodes =
200 000`, `max_iterations = 64`):

| board | rules | stop | enodes | classes | secs | best |
|---|---|---|---|---|---|---|
| net n=8 | assoc | fixed_point | 353 | 36 | 0.03 | 3673 |
| net n=8 | AC | deadline | 107 980 | 30 856 | 6.0 | 2081 |
| net n=10 | assoc | fixed_point | 2604 | 55 | 1.4 | 2781 |
| net n=10 | AC | deadline | 275 092 | 105 840 | 6.0 | 1680 |
| net n=12 | assoc | deadline | 11 703 | 3882 | 6.0 | 4290 |
| net n=12 | AC | deadline | 308 097 | 102 280 | 6.0 | 2866 |
| net n=20 | assoc | deadline | 379 909 | 189 974 | 6.0 | n/a |
| net n=20 | AC | deadline | 367 476 | 183 577 | 6.0 | n/a |
| net n=40 | assoc | deadline | 366 637 | 183 359 | 6.0 | n/a |
| net n=60 | assoc | deadline | 326 167 | 163 143 | 6.3 | n/a |
| net n=60 | AC | deadline | 263 784 | 130 211 | 6.3 | 8.57e+13 |
| chain n=10 | assoc | fixed_point | 2604 | 55 | 1.0 | 576 |

With an **unbounded** deadline (supplementary run, seed 0):

| board | rules | stop | enodes | secs |
|---|---|---|---|---|
| net n=8 | AC | > 600 s | > 107 980 | > 600 |
| net n=10 | assoc | fixed_point | 2604 | 0.8 |
| net n=12 | assoc | fixed_point | 22 122 | **139.8** |
| net n=13 | assoc | deadline | 49 995 | 200 |
| chain n=12 | assoc | fixed_point | 22 126 | 90.7 |

So the **fixed point** is reached only for n ≤ 12 (associativity alone);
n = 13 does not finish in 200 s, and n ≥ 20 cannot complete a single
iteration.  The e-class member count is **Catalan**, not polynomial:
an e-class for a length-`k` leaf range holds every binary tree over it,
`Catalan(k-1)`.  Commutativity multiplies that further — the **AC** rule
set already breaks at **n = 8** (> 600 s).

**Consequence:** equality saturation is *dominated* by the exact subset
DP.  Both are exponential, but the DP is exact, cheaper, and reaches
n ≈ 18, while saturation dies at n ≈ 12.  There is **no scale at which
saturation is the right tool** for contraction ordering.  The
`contraction-ladder` claim "saturation closes the gap exactly" is true
only in the tiny regime it measured (n = 4…8) — exactly the artefact
this retro was asked to test.

## 2. Greedy is materially worse than best-found at scale

Per-scale ratios to the best cost any player found, mean over 3 seeds
(greedy also shows the worst):

| n | best (median) | greedy mean (max) | one-step | search | restart |
|---|---|---|---|---|---|
| 20 | 11 146 | **2.33** (3.31) | 1.25 | 1.12 | 1.11 |
| 30 | 590 380 | 1.15 (1.32) | — | 1.00 | 1.03 |
| 40 | 1.72e+08 | **2.33** (5.00) | — | — | 1.00 |
| 60 | 1.24e+12 | **13.32** (27.95) | — | — | 1.00 |

* **greedy loses** at every scale, growing with `n`: 2.3× (n=20),
  2.3× (n=40), **13.3× mean / 28× worst** (n=60).  The n=30 draw is the
  mild case (1.15×) — reported honestly; greedy's variance is the point.
* **search** (bounded best-first) beats greedy where it can run
  (n ≤ 30), and at n = 30 it *is* the best found (1.00).
* **restart** (64 randomised-greedy episodes) is the best cheap player
  at n = 40 and 60, where the best-first search is infeasible.

Because "best-found" is an **upper** bound on the true optimum,
greedy/best-found is a **lower** bound on greedy/optimum: greedy is at
least 13× off optimal at n = 60.

## 3. Controls validate the players against the exact DP

| n | dp | greedy/dp | one-step/dp | search/dp | restart/dp |
|---|---|---|---|---|---|
| 8 | 2081 | 1.616 | 1.000 | 1.012 | 1.012 |
| 10 | 1629 | 1.314 | 1.071 | 1.019 | 1.050 |
| 12 | 2675 | 2.618 | 1.116 | 1.075 | 1.099 |
| 14 | 3575 | 1.908 | 1.580 | 1.550 | 1.442 |
| 16 | 3251 | 1.395 | 1.329 | 1.091 | 1.335 |

The players are sound: `search` reaches 1.01-1.09× the DP optimum at
n = 8…16 (1.55× at n = 14, a genuinely hard draw — it is a *bounded
heuristic* search, not exact), and `one-step` hits the optimum at
n = 8.  Greedy is 1.3-2.6× off at these sizes, matching the
precondition's "greedy loses" finding.

## The gate

**Headroom exists — the domain is open.**  Both halves of the gate hold:

1. **Saturation fails to finish** at n ≥ 12 (assoc) / n ≥ 8 (AC) — see
   §1.  The search breaks well before the exact optimum is intractable.
2. **A cheaper player is materially worse**: greedy is up to 13× (mean)
   / 28× (worst) above best-found at n = 60 — see §2.

So a learned contraction-ordering policy has real quality headroom at
scale.  **The honest caveat:** the headroom is above *greedy*, not
above *saturation* — saturation is not a viable player there at all.
The cheap players that recover most of the gap (bounded best-first
`search`, randomised `restart`) are the real baselines a learned policy
must beat.  At n = 30 the bounded search already closes the gap to
1.00×, so the win there is small; the headroom is large at n = 40/60,
where search is infeasible and only greedy/restart run.

This is a **falsification that succeeded**: the small-board
"completeness = optimality" result did **not** survive scaling, and the
one-step oracle is not the binding baseline at scale.

## Honest limitations

* **Best-found is not the optimum.**  At n ≥ 20 there is no exact
  reference; every ratio in §2 is against an upper bound on the true
  optimum, so the measured greedy gap is conservative (a lower bound).
* **Saturation is a *specific* search.**  "The search breaks" means
  equality saturation in this rule space breaks.  A different exact
  search (the DP) reaches n ≈ 18; a different heuristic (best-first /
  restarts) runs at n = 60.  The finding is about *saturation*, not
  about all search.
* **The rule space is not shipped.**  catopt has no general-network
  contraction rule; the `contract` AC space here is built on its
  e-graph and cost extractor, not a rule set in `catopt_core.laws`.
* **One cost model family.**  All numbers use the classic scalar-mult
  cost.  A launch-aware or roofline model would shift the exact ratios,
  not the exponential-vs-polynomial conclusion.
* **Three seeds per scale.**  The scale table averages 3 seeded
  networks; greedy's variance across draws is large (the n=30 draw is
  mild), so read the mean *and* the max, not either alone.

## Reproduce

```sh
.venv/bin/python tools/contraction_scale.py            # ~3 min
.venv/bin/python tools/contraction_scale.py --instances 1 --seed 1
# the unbounded saturation curve (§1, supplementary):
.venv/bin/python -c "
import importlib.util as u
s = u.spec_from_file_location('cs', 'tools/contraction_scale.py')
cs = u.module_from_spec(s); s.loader.exec_module(cs)
a, ar, c = cs.contract_rules()
for n in (8, 10, 12, 13):
    t, sz = cs.random_network(n, 0)
    r = cs.saturate(t, sz, (a, ar), max_nodes=2_000_000,
                    max_iterations=200, deadline=200)
    print(n, r['stop'], r['enodes'], f\"{r['secs']:.1f}s\")
"
```
