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
* **No package code changed.**  `catopt_core.trajectories.rule_vector`
  is untouched (its structural encoding is load-bearing).

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

Mixture, default (`chain,dup,linear`, 20/family = 60 programs):

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

**Verdict: the RL policy dilutes catastrophically.**  It collapses
onto the `linear` family — matching greedy there (868.0 = 868.0) but
doing *nothing* on `chain` and `dup` (its pick ties the no-op floor).
The collapse is not a data-scarcity effect: 60/family (180 programs)
collapses the same way, seeds 0–2.

### Why — the reward scale is not comparable across families

The env's reward is the *normalized* improvement
`(before − after) / before` (`catopt_core.search_env`).  The best
achievable normalized reward differs sharply by family:

| family | mean best normalized reward | typical before |
|---|---|---|
| chain  | 0.106 | ~792 |
| dup    | 0.312 | ~75 |
| linear | 0.757 | ~3916 |

REINFORCE here uses a single **running-mean** baseline over all
families, so the high-reward `linear` steps get a large positive
advantage and the low-reward `chain`/`dup` steps a negative one — the
policy is pushed to always play the `linear` rule.  Two-family probes
confirm the mechanism (20/family, seed 0/1/2):

| training families | result |
|---|---|
| `chain,dup`    | both learned (seed 0, 2); collapses onto `dup` at seed 1 |
| `chain,linear` | collapses onto `linear` (all seeds) |
| `dup,linear`   | both learned at seed 0; collapses onto `linear` (seeds 1, 2) |

The collapse tracks the reward ordering, not the family *identity*:
whenever the highest-reward family is present, the policy tends to
collapse onto it.

## Honest verdict

* **Supervised: multi-family training helps, and does not dilute** at
  any practical data size.  The mixture holds rank 1.00 on all three
  families where the one-family policy holds it on one — a strict
  improvement, not a trade.  Dilution is real but confined to a
  very-low-data regime (≤5 programs/family, 500 epochs), where the
  smallest-signal family (`chain`) loses first.
* **RL: multi-family training dilutes badly** with the current
  objective.  A single running-mean baseline over families whose
  normalized rewards differ ~7× makes the policy collapse onto the
  highest-reward family.  The mixture default is therefore *honest
  but bad* for RL today.
* The fix is a **per-family (or per-state) baseline / reward
  normalization** — REINFORCE's baseline must be conditioned on
  something that makes the families comparable.  That is future work;
  the mixture remains the default so the failure is visible and
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
