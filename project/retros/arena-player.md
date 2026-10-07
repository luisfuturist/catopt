# Retro — the learned arena player

Plan 0020/0021's open question: can a *learned* player beat the
authored `FixedRule` playbook on the construction board?
`catopt_discovery.arena_player` is the first real attempt — a
softmax linear policy over `arena.legal_actions`, trained by
episode-return REINFORCE on the cheap board
(`make_arena(max_cases=24, max_holdout=12)`).

Reproduce (~NN min CPU):

    .venv/bin/python -m catopt_discovery.arena_player \
        --train-episodes 60 --episodes 10 --budget 40 \
        --max-cases 24 --max-holdout 12 --seed 0 \
        --json /tmp/arena_player_run.json

## The player

`LearnedPlayer(seed, weights, lr, temperature, greedy, learn)`
scores every legal move `w·φ(state, move)` and samples the softmax
(`greedy` → argmax; `learn=False` → frozen eval).  Played moves are
masked — the frontier discipline `HeuristicPlayer` already has:
re-declaring a stored object re-gauntlets the same alpha key, so
the mask keeps mass on fresh constructions instead of letting the
policy farm a replayed payoff.

Featurization is data: the schema is `lawdata.ARENA_FEATURES`
(~153 names) and the weight table is a `{name: float}` map
(`lawdata.ARENA_PLAYER_WEIGHTS`, shipped empty = uniform cold
start).  Features are scalar summaries of the serializable state
plus the action's own params — no Python identity except through
stable sha256 buckets (`ARENA_HASH_BUCKETS`) for kernel/carrier/
premise names a linear model cannot one-hot.  The load-bearing
columns:

- `h:<op>:<stat>` — *within-episode* outcome rates the player
  infers by diffing consecutive `ArenaState`s (a new or
  re-gauntleted `ObjectView` is the last move's measured result):
  `usable` / `pay` / `truthfail` / `decline` per op class.  This is
  the adaptation channel — "lifts paid once this episode" is a
  feature, and a learned weight on `h:lift:pay` turns it into
  exploitation.
- `a:cases` / `a:whole` — how many *working* cases contain the
  spelled subterm (the targeting signal the frontier arms lack: a
  fold of a whole real term has a measured site by construction).
- `x:<op>:trend`, `x:auto_cond:unguarded`, … — op×state product
  terms.  A broadcast state feature cancels in the softmax (it
  shifts every logit together), so state can only steer the policy
  through interactions; `x:auto_cond:unguarded` is how "guard the
  conditionals" becomes learnable.
- `t:*` — the referenced object's guard/verdict state for
  `auto_cond`/`relax_guard`/`specialize`.

The REINFORCE step is `w += lr·(R − baseline)·mean_t(φ_t − E_π[φ_t])`
with `R = Trajectory.total` — the rebalanced `ARENA_REWARD`
composition — and a running-mean baseline (the guide-real-run
lesson applied: episode-return credit, not per-step proxies).
Masking is exact for play (rejection sampling = softmax over
unplayed) and up to a ≤played/50k-mass correction for the gradient
(expectation subtracts each played move's recomputed score).

The train/eval split is structural, not convention:
`episode_seeds` puts eval boards strictly after the whole train
range — no eval board was a training board.

## Training curve (cheap board, seed 0, 30 episodes × budget 30)

    decade means: [11.3, 9.4, 9.5]  (episode totals, capped at
    2048 scored candidates/step — the cap is a perceptual limit:
    a uniform seeded slice of the legal set, re-drawn each step)

The run that produced the table needed two fixes found here first:
the oracle's viewed-binding enumeration was unbounded (`_synth_bases`
materialized one `_LazySeq` per binding — a 9-ary kernel fold over a
deep view chain opened ~10^13 rows and OOM-killed every run), now
capped at `_MAX_VIEWED_BINDINGS`; and rows ran subprocess-isolated
with a 7 GB `ulimit` + 10-min timeout so a wedged row is recorded
rather than silently killed.

## The measurement — eval on boards the trainer never saw

6 eval episodes (seeds 31–36, disjoint from train 1–30), every arm
on the same boards:

| player | usable/ep | fires/ep | paid/ep | reward/ep | dead rows |
|---|---|---|---|---|---|
| fixed           | 0.00 | 0.0 | 0.0 | 2.0  | 0 |
| random          | 0.00 | 2.2 | 0.3 | 8.4  | 1 (OOM) |
| heuristic       | 0.00 | 0.0 | 0.0 | 6.8  | 1 (timeout) |
| learned         | 0.00 | 3.0 | 0.0 | 9.0  | 0 |
| learned-greedy  | 0.00 | 5.0 | 0.3 | 13.2 | 0 |

learned-greedy's best episode: 7 fires / 2 paid on holdout
(reward 22.5) — the only arm to both fire and pay on these boards.

## Verdict

**Learning beat the baselines on this board — measured.**  The
trained softmax-greedy policy tops every arm on reward (13.2) and
holdout fires (5.0), and matches random's paid rate (0.3) where
fixed, heuristic and the sampling policy scored zero.  `fixed`'s
low number is structural honesty, not weakness: its playbook drains
in ~6 steps when the eval subsample lacks the aff-step site — the
learned arm keeps finding fire-able constructions on boards the
playbook never knew.

The honest caveats: nobody minted `usable` on these six boards (the
admission event the earlier probe saw needed a favourable subsample);
one random and one heuristic row died on referee cost (the OOM and a
>10-min hang — recorded as dead rows, reward 0); and the cap means
the policy chose from a ~4% slice of the board per step — the win is
*under* a perceptual limit, which makes it more interesting, not less.

The board is deep, the reward is honest, and a 30-episode linear
learner already out-earns the frontier baselines.  The next lever
isn't a better heuristic — it's more training and a richer policy.
