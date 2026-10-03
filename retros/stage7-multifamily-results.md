# Stage 7 — multi-family search policy: results

Plan 0016 stage 7, the multi-family gap (ADR 0003).  The supervised
policy (`stage7-learned-policy-results.md`) and the RL policy
(`stage7-rl-results.md`) were both trained and measured on ONE family
(matmul chains).  This note trains each on a **three-family mixture**
and measures it **per family**, so the question "does multi-family
training help, or dilute?" gets numbers instead of a guess.

## What changed

* `tools/families.py` (new) — the three training families and the
  per-family scoring.  Torch-free except the linear builder, which
  exports a small torch model through `TorchSource`.
* `tools/train_search_policy.py` — the mixture is now the default
  (`--train-families`), and it prints a **per-family** rank / hit-rate
  table.  `--train-families chain` reproduces the one-family baseline.
* `tools/train_rl_policy.py` — same mixture default, plus a per-family
  race table (random / declaration / greedy / RL / no-op).
* `catopt_torch/rl.py` — **the REINFORCE baseline fix** (this
  revision): the single running-mean baseline over returns is
  replaced by a **per-episode standardized advantage**.  The
  advantage is now shift- and scale-invariant, so one family's
  reward scale cannot bias another's.  `train_reinforce` /
  `RLPolicy` are unchanged for callers.
* `catopt_core.trajectories.rule_vector` is untouched (its structural
  encoding is load-bearing).

The three families, each with a guaranteed strictly-improving rule
(so a "hit" is a real find, not a coin flip):

| family | shape | paying rule |
|---|---|---|
| `chain`  | `a·(b·c)`, `k > m` | `assoc_matmul` |
| `dup`    | `square(t) + t·t`   | `square_expand` (CSE: the two spellings unify) |
| `linear` | `relu(linear(linear(x)))`, via `TorchSource` | `assoc_linear_bias` |

Metrics per held-out program, matching
`bench/suites/evaluation/policy_value.py`: **rank** of the picked rule
by true delta (1 = best; 1-based *average* ranks, ties share) and
**hit** (the pick's delta is strictly positive).  Rank 26.5 is the
floor — a pick tied with every non-improving rule (51 rules).

## Supervised policy (cuda, RTX 2050, torch 2.14.0+cu130)

`RuleValueNet`, 3000 epochs, hidden 96, 8 held-out programs per family,
seed 0.  Training programs: 60 (one family) vs 60 / 180 (mixture).

One-family baseline (`--train-families chain`, 60 chains):

```text
family     n mean_rank hit_rate  mean_delta
chain      8      1.00     1.00        95.0
dup        8     26.50     0.00         0.0
linear     8     26.50     0.00         0.0
```

Mixture, default (`chain,dup,linear`, 60/family = 180 programs):

```text
family     n mean_rank hit_rate  mean_delta
chain      8      1.00     1.00        95.0
dup        8      1.00     1.00        35.2
linear     8      1.00     1.00      2647.0
```

The matched-count mixture (20/family = 60 programs, the same training
size as the baseline) gives **identical** numbers.  Stable across
seeds 0–2.

**Verdict: no dilution.**  The mixture keeps the chain pick at rank
1.00 while also solving `dup` and `linear` — which the one-family
baseline never learns (rank 26.5, hit 0).  The supervised policy is a
per-`(features, rule)` classifier, so it can hold all three families
at once.

### Where dilution does appear (low-data stress)

At 500 epochs / hidden 32, the mixture only dilutes when the per-family
signal is scarce.  Chain-only with 3 chains stays at rank 1.00; the
mixture degrades the *chain* family while `dup` and `linear` stay
perfect (seeds 0/1/2):

| mixture per family | chain mean_rank (seed 0/1/2) | chain hit_rate |
|---|---|---|
| 3 (9 programs)  | 1.00 / 10.56 / 7.38 | 1.00 / 0.62 / 0.75 |
| 4 (12 programs) | 1.00 / 7.38 / 1.00  | 1.00 / 0.75 / 1.00 |
| 5 (15 programs) | 1.00 / 4.19 / 7.38  | 1.00 / 0.88 / 0.75 |
| 8 (24 programs) | 1.00 / 1.00 / 1.00  | 1.00 / 1.00 / 1.00 |

So the mixture is safe at the default (60/family) and down to
~8 programs/family; below that the family with the *smallest* signal
(`chain`, whose normalized payoff is the smallest) is the first to
dilute — the other two survive.

## RL policy (`train_reinforce`, REINFORCE)

2000 episodes, horizon 6, hidden 64, 8 held-out per family, seed 0.
Training programs: 60 (one family) vs 60 (mixture, 20/family).

One-family baseline (`--train-families chain`, 60 chains):

```text
family     n mean_rank hit_rate  mean_delta
chain      8      1.00     1.00        95.0
dup        8     26.50     0.00         0.0
linear     8     26.50     0.00         0.0

family        random declaration      greedy          rl       no-op
chain          866.8       875.8       780.8       780.8       875.8
dup            116.4       116.4        81.1       116.4       116.4
linear        3515.0      3515.0       868.0      3515.0      3515.0
```

The RL player matches the one-step greedy oracle on chains (780.8 =
780.8) and does nothing on the other two.  Stable across seeds.

Mixture, default (`chain,dup,linear`, 20/family = 60 programs).

### The baseline bug (before the fix)

A single **running-mean baseline over returns**
(`baseline = 0.98 * baseline + 0.02 * returns.mean()`), so one scalar
serves every family.  The env's reward is the *normalized* improvement
`(before − after) / before` (`catopt_core.search_env`), whose best
achievable value differs sharply by family:

| family | mean best normalized reward | typical before |
|---|---|---|
| chain  | 0.106 | ~792 |
| dup    | 0.312 | ~75 |
| linear | 0.757 | ~3916 |

The scalar baseline therefore sits *between* the families: the loud
`linear` steps get a large positive advantage and the quiet
`chain`/`dup` steps a **negative** one — so their *correct* rule is
pushed *away*.  The mixture collapses onto `linear`:

```text
family     n mean_rank hit_rate  mean_delta
chain      8     26.50     0.00         0.0
dup        8     26.50     0.00         0.0
linear     8      1.00     1.00      2647.0

family        random declaration      greedy          rl       no-op
chain          851.2       875.8       780.8       875.8       875.8
dup            116.4       116.4        81.1       116.4       116.4
linear        3515.0      3515.0       868.0       868.0      3515.0
```

It matches greedy on `linear` (868.0 = 868.0) and does *nothing* on
`chain`/`dup` (its pick ties the no-op floor).  Not a data-scarcity
effect: 60/family (180 programs) collapses the same way, seeds 0–2.

### The fix (after) — a per-episode standardized advantage

`catopt_torch.rl._advantage` now **standardizes each episode's
returns** (zero mean, unit variance) before the policy-gradient
update, replacing the single running mean.  Because standardization
is shift- *and* scale-invariant, the family reward scale drops out
entirely: within an episode the best action always scores positive
and the worst negative, whatever the family.  It needs **no family
label** — the env stays family-blind, and the public API
(`train_reinforce`, `RLPolicy`) is unchanged.

```text
family     n mean_rank hit_rate  mean_delta
chain      8     26.50     0.00         0.0
dup        8      1.00     1.00        35.2
linear     8      1.00     1.00      2647.0

family        random declaration      greedy          rl       no-op
chain          875.8       875.8       780.8       875.8       875.8
dup            116.4       116.4        81.1        81.1       116.4
linear        3515.0      3515.0       868.0       868.0      3515.0
```

The fix lands `dup` (rank 26.5 → 1.00, hit 0 → 1.00) and keeps
`linear`, now matching greedy on both — and the winner is **no longer
the loudest family**.  Across seeds 0–2 `dup` is solved every seed and
`linear` two of three.

> **Correction (later audit) — this table is stale at HEAD.**  An
> audit for a *different* RL bug class found no bug in the shipped
> code (the step/return alignment is correct, and `_advantage` always
> standardizes — now pinned by six mutation-verified tests), but
> re-running this measurement deterministically at HEAD reproduces the
> **"before"** table above (`chain 26.50`, `dup 26.50`,
> `linear 1.00`), not this one.
>
> The cause is **not** the reward baseline: it is
> `catopt_core.features` changing after `b16f358` — the view/fold-aware
> traffic accounting (`:178-196`, commit `ecfbf50`) altered the RL
> policy's **state input**.  Reverting `features.py` to `b16f358`
> reproduces this "after" table **exactly** (`dup 1.00/1.00`).  So the
> per-episode standardization fix is intact and correct; its
> documented *benefit* simply does not survive the feature change.
>
> The lesson is architectural, not numerical: **a policy is coupled to
> the semantics of its features**, so a correctness fix elsewhere can
> silently invalidate a learned policy.  Any change to
> `ProgramFeatures` must be treated as a change to every learned
> policy trained on it.

### The residual — a winner-take-all, not a scale bias

`chain` still collapses (rank 26.5).  The *scale* bias is gone, but
the collapse survives as a **winner-take-all race** of the shared
policy net:

* The trained net's rule ranking is nearly *state-independent*: for a
  `chain` state it orders `assoc_linear_bias > square_expand >
  linear_channel_scale > assoc_matmul` — the **same order** it gives a
  `linear` state.  It separates `dup` (reuse ~0.09) but conflates
  `chain` with `linear` (reuse ~0.95 both), so `chain` inherits
  `linear`'s pick.
* Controls that change the baseline or the conditioning do **not**
  break the tie (2000 episodes, seeds 0–2): a per-state running-mean
  baseline keyed by root op; a per-state mean+scale baseline; input
  standardization; entropy bonuses 0.02–3.0; step-reward instead of
  discounted returns-to-go; `patience` 6; shuffled program order;
  `hidden` 256; and 12k episodes.  Each still solves exactly one or
  two families.  `chain`, `dup` and `linear` are each learnable
  **alone** (rank 1.00), so it is mixture interference, not capacity.
* The reward is sparse: a `chain` episode samples ~2 of 51 rules
  before `patience` ends it, so the single paying rule is hit in ~2%
  of episodes.  Whichever family's signal first pushes the net to a
  near-deterministic preference wins the race, and the others stop
  being explored.

So the diagnosed *scale* bug is fixed; the mixture still collapses
for a second, orthogonal reason (shared-net winner-take-all under a
sparse reward).  The supervised policy does not suffer this because
its per-`(features, rule)` target is dense.

## Honest verdict

* **Supervised: multi-family training helps, and does not dilute** at
  any practical data size.  The mixture holds rank 1.00 on all three
  families where the one-family policy holds it on one — a strict
  improvement, not a trade.  Dilution is real but confined to a
  very-low-data regime (≤5 programs/family, 500 epochs), where the
  smallest-signal family (`chain`) loses first.
* **RL: the diagnosed baseline bug is fixed, but the mixture still
  collapses.**  A per-episode standardized advantage removes the
  reward-scale domination — `dup` is now solved (26.5 → 1.00) and
  `linear` stays solved, and the winner is no longer the loudest
  family.  But `chain` still collapses to the floor: the shared net
  converges to a near-global rule preference before the sparsest
  family's signal accumulates (a winner-take-all, not a scale bias).
  Fixing *that* needs a denser or state-conditioned signal — the
  supervised policy, or a value baseline trained on the same mixture —
  not a better scalar baseline.
* The mixture remains the default so the failure is visible and
  measurable, not hidden behind a one-family demo.

## Reproduce

```sh
# supervised — one-family baseline vs the mixture (default)
.venv/bin/python tools/train_search_policy.py --device cuda \
    --train-families chain --per-family 60 --epochs 3000 --hidden 96
.venv/bin/python tools/train_search_policy.py --device cuda \
    --train-families chain,dup,linear --per-family 60 --epochs 3000 --hidden 96

# RL — one-family baseline vs the mixture (default)
.venv/bin/python tools/train_rl_policy.py --device cuda \
    --train-families chain --train 60 --episodes 2000 --horizon 6 --hidden 64
.venv/bin/python tools/train_rl_policy.py --device cuda \
    --train-families chain,dup,linear --train 20 --episodes 2000 --horizon 6 --hidden 64
```

Both tools score **every** family on every run, so a one-family
baseline honestly reports rank 26.5 / hit 0 on the families it never
saw.
