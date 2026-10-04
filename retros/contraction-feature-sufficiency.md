# Feature sufficiency — the gap closes, and it doesn't matter

`contraction-train-scale.md` §2 measured the last untested lever on the
contraction thread: a supervised net over the shipped `scale-free-v1`
features fits `opt_einsum` oe-greedy's decisions at n = 40 with only
~0.60 held-out top-1 — the features *partially* express the strong
player's choice.  This retro asks whether better features close that
sufficiency gap, and whether closing it helps the player.

The answer is the cleanest result on this thread yet — and it is a
**ship-or-not negative for the artifact**:

* **The sufficiency gap is real and closable**: four cheap pair
  features lift held-out top-1 from 0.551 to 0.714 (+16 pts, over a
  +5 pt ship bar).
* **But end-to-end the new features *hurt* the sampled player** at
  n = 40 (1.69–1.70 vs oe-rand-greedy, up from 1.21 — confirmed on
  two independent RL seeds), while making the *argmax* single pass
  3x better.  Feature sufficiency was **not the binding bottleneck**;
  rollout diversity is.
* **Nothing ships.**  The v2 bump was implemented, trained, measured
  and reverted; the bundled artifact stays `scale-free-v1`.  The only
  durable changes are `tools/contraction_feature_probe.py` and this
  note.

Reproduce:

    uv sync --group einsum
    .venv/bin/python tools/contraction_feature_probe.py --device cuda \
        --scales 40 --boards 12 --holdout 12 --epochs 400
    .venv/bin/python tools/contraction_feature_probe.py --device cuda \
        --e2e bundled --e2e-budgets 200 --e2e-scales 20,30,40

## 1. What the teacher actually scores — a code read, not a guess

Before inventing features, we read `opt_einsum/paths.py`
(`ssa_greedy_optimize`).  Three mechanism facts the v1 features do not
encode:

* **The score is not our cost.**  oe-greedy minimises
  `cost_memory_removed = size(merged) - size(a) - size(b)` — the
  *memory reduction* of the intermediate — not the union-product FLOP
  count our `pair_cost` (and every v1 feature) uses.
* **Outer products never enter the queue.**  Candidates are pushed
  per shared dimension; a disjoint pair is only assessed when nothing
  shares an index.  Hence the retro's 0.0 vs 7.7 outer-product-step
  asymmetry — it is structural, not scored.
* **Ties are real and positional.**  The heap key is
  `(score, id2, id1)` on exact *integer* sizes; small int extents tie
  often (~0.19 extra ties per state), and ssa ids grow in creation
  order — which our list positions track, since merges append.

Direct measurement on the replayed states: **84 % of oe-greedy's
actions are exactly `argmin` of the raw memory-removed score** over
index-sharing pairs, rising to **~90 %** once ties break by position.
The residual ~10 % is the heap's incremental pruning (only the best
partner per (tensor, dim) is queued) — a genuinely stateful,
algorithmic detail no per-pair feature can see.

## 2. Sufficiency — v1 features + the teacher's own metric

Probe: `tools/contraction_feature_probe.py`.  Same methodology as the
train-scale §2 — replay oe-greedy's order into `(state, action)`
samples at n = 40 (12 train / 12 held-out bond boards, 468 states
each), fit the same MLP at masked cross-entropy, 400 epochs, hidden 64.
Feature sets are column slices over `v1 + candidates`; candidates
tested in two rounds:

* round 1 (mechanism-motivated, blind): outer-product indicator,
  merged rank, one-step lookahead, merged connectivity, and two state
  features (outer-product fraction, non-outer cost gap);
* round 2 (after §1's code read): `mr-score` (memory-removed,
  normalised by the union size, computed in log space — all exponents
  ≤ 0), `mr-rank` (exact-integer mr rank under the `(score, pos_b,
  pos_a)` tie-break), `pos-a`/`pos-b` (normalised positions).

| spec | width | train top-1 | held-out top-1 |
|---|---|---|---|
| v1 (shipped) | 16 | 0.998 | 0.551 |
| v2 (v1 + round-1 six) | 22 | 0.955 | 0.511 |
| **v3 (v1 + mr-score, mr-rank, pos-a, pos-b)** | 20 | 0.998 | **0.714** |
| v4 (everything) | 26 | 1.000 | 0.667 |

Per-feature ablations off v4 (held-out): `no-mr-rank` **0.547**
(removing it loses the whole gain — the exact-integer rank under the
teacher's tie-break is the load-bearing feature), `no-pos-a` 0.686,
`no-mr-score` 0.705; every round-1 feature ablates to 0.54–0.72,
i.e. noise.  The round-1 six — the candidates the retro's mechanism
section suggested — are flat-to-negative alone (v2, −4 pts).

A telling detail: **train top-1 was already ~1.0 under v1**.  The old
features can *fit* the teacher on-train; they just fit the wrong
invariants (union-FLOP order), so the fit doesn't transfer.  The mr
features give the net the teacher's *actual* ranking, and transfer
follows.

Supervised argmax rollouts / teacher order on held-out boards: v1
0.680, v3 0.708, v4 0.874 (below 1 = the student beats its teacher —
consistent with the earlier finding that a smoothed learned order can
price better than the order it was trained to copy).

## 3. End-to-end — the interesting part

If sufficiency were the bottleneck, a policy trained on the v3 column
set should beat the v1 artifact.  The bump was implemented behind a
new contract (`scale-free-v2`, `PAIR_DIM` 9 → 13, append-only
columns), the curriculum regime (8,12,16,20,24; 1800 iterations;
~430 s) retrained on it, and the ladder run on the same boards,
same protocol, same seeds as a v1 control run — 200 ms equal
wall-clock, best-of-N sampled rollouts:

| n | v1/oe-greedy | v2/oe-greedy | v1/oe-rand-greedy | v2/oe-rand-greedy |
|---|---|---|---|---|
| 20 | 0.818 | 0.818 / 0.817 | 0.954 | 0.977 / 0.966 |
| 30 | 0.850 | 0.855 / 0.873 | 1.033 | 1.026 / 1.076 |
| 40 | **0.763** | 1.122 / 1.095 | **1.209** | 1.704 / 1.690 |

(v2 columns show both RL seeds — the n = 40 regression is not wobble.)

And the deterministic **argmax** single pass, same boards:

| n | v1/oe-greedy | v2/oe-greedy | v1/oe-rand-greedy | v2/oe-rand-greedy |
|---|---|---|---|---|
| 20 | 1.053 | 0.888 | 1.315 | 1.110 |
| 30 | 0.833 | 0.926 | 1.016 | 1.154 |
| 40 | 4.153 | **2.150** | 9.477 | **3.230** |

Two opposite motions, one mechanism: the mr features pull the policy's
probability mass onto the teacher's argmin.  The argmax pass inherits
the teacher's per-step quality — a 3x improvement at n = 40 — but the
sampled player's edge was never per-decision quality: it is best-of-N
*exploration*, and a sharper policy restarts into the same basin.
Sharper policies, less diverse rollouts, worse anytime player.

## 4. Verdict

* **Sufficiency was real but not binding.**  The features can now
  express the strong player's choices (+16 pts held-out, ~80 % of the
  ~90 % structural ceiling) and the single-pass player improved ~2-3x
  at n = 40 — but the shipped player is the sampled anytime one, and it
  regressed.  Ship-or-not: **do not ship**; the v2 package bump and
  the retrained artifact were reverted, `scale-free-v1` stays bundled.
* **The bottleneck is diversity, not expressiveness.**  An anytime
  learned player that beats best-of-N randomised greedy needs rollout
  *variance with good coverage*, which sits on the opposite side of
  the sharpness the teacher's metric rewards.  This refines — not
  contradicts — the earlier "the residual is algorithmic" verdict: the
  algorithmic advantage isn't only restarts-as-search, it's that the
  restart ensemble's value is *orthogonal* to single-decision
  imitation fidelity.
* **If a single-pass player ever becomes the target** (e.g., a
  latency-bound embedding where 39 forward passes must do all the
  work), the v2 features are a measured ~2-3x win at n = 40 and the
  diff to re-derive them lives in this note and the probe.
* **What this does not change:** oe-rand-greedy remains unbeaten at
  adequate budgets; v1's standing (~1.2x at n = 40 sampled, ~0.76x vs
  oe-greedy) is still the honest one-liner.

## 5. Honesty / limits

* **3 boards per cell, 2 seeds for v2.**  The n = 40 sampled
  regression replicates across independent trainings (1.704 / 1.690)
  and is far outside the run-to-run wobble the earlier retro recorded;
  n = 20/30 cells are noise-level either way.
* **200 ms only.**  At 50 ms the learned player starves regardless;
  at 1000 ms the ordering could differ — the direction (argmax up,
  sampled down) is mechanism-driven, the magnitudes are not.
* **The ~10 % ceiling residual is real.**  The teacher's heap prunes
  to the best partner per (tensor, dim); modelling queue membership
  faithfully needs state the per-pair feature contract cannot carry.
  A `queued?` approximation was tried and admits everything (100 %
  queue fraction) — it does not discriminate.
* **`mr-score` is exact-integer ranked but log-space valued.**  The
  rank column uses Python ints (real ties preserved); the value
  column is the scale-free `[-2, 1]` normalisation — verified equal
  to the shipped derivation to ~1e-15 before the revert.

## Gates

* `.venv/bin/ruff check` / `ruff format --check` on
  `tools/contraction_feature_probe.py` — pass.
* `pytest tests/test_contraction_policy_artifact.py -q` — 33 green
  (run both under the v2 bump, with the retrained artifact, and after
  the revert).
* Final tree state: package module, tests and bundled artifact all
  `git checkout`-restored; the only additions are this probe and this
  retro.
