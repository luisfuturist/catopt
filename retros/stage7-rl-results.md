# Stage 7 — RL search policy: results

Plan 0016 stage 7, the RL half (ADR 0003).  The supervised policy is
in `stage7-learned-policy-results.md`; this is the *full RL* player.

## What was built

* `catopt_core.search_env` — `SearchEnv` + `StepResult`: one program
  per episode; each step applies one rule to the e-graph; the reward
  is the **normalized cost improvement** that step produced.  Bounded
  by `horizon`, ended early by `patience`.  Torch-free, 100% covered.
* `catopt_torch.rl` — `PolicyNet` (scores each rule from the state
  plus its **structural** vector), `RLPolicy` (the `Policy` port),
  `train_reinforce` (**REINFORCE** with a running-mean baseline).
* `tools/train_rl_policy.py` — trains on the GPU, then races the
  players on held-out programs.

## Measured result

RTX 2050, `torch 2.14.0+cu130`, 2000 episodes, horizon 6; held-out
chains with shapes 2–7 (unseen in training):

| player | mean final cost |
|---|---|
| random | 377.0 |
| declaration | 377.0 |
| no-op (no moves) | 377.0 |
| greedy (one-step oracle) | **361.0** |
| **rl** | **361.0** |

The RL policy **matches the one-step cost-model oracle** and beats
random, declaration-order and doing nothing.  It learned the
associativity move from reward alone — no labels.

## Honest limits

* **The oracle is reachable here.**  `greedy` is the one-step
  cost-model oracle from the current state (`rule_samples`), so no
  learned policy can beat it on this board; matching it is the
  available win.  The interesting regime — where 1-step lookahead is
  myopic — needs deeper chains, and is not measured yet.
* **Shallow credit.**  The reward is the immediate normalized
  improvement; discounting carries it across a 6-step episode but
  there is no learned value baseline beyond the running mean.
* **No action masking.**  The action space is every rule; rules that
  cannot fire simply yield zero reward.  Masking would sharpen the
  signal.
* **One family.**  Chains only.  Mixed families diluted the
  *supervised* policy (see the stage-7 retro); the RL policy has not
  been tested on them.
* **Still only a player.**  The policy orders legal moves; the
  certificate still decides correctness.  Nothing here can make a
  program wrong.

## Reproduce

```sh
.venv/bin/python tools/train_rl_policy.py --device cuda \
    --episodes 2000 --train 60 --horizon 6 --hidden 64
```
