# Stage 7 — learned search policy: results and findings

Plan 0016 stage 7 (ADR 0003).  What was built, what was measured, and
what the numbers honestly say.

## What was built

* `catopt_core.trajectories` — `RuleSample` + `rule_samples` + a
  **structural** `rule_vector` (root-op buckets, full pattern-tree
  hash buckets, arities, check/derive flags).  Torch-free.
* `catopt_torch.learned_policy` — `RuleValueNet` (an MLP) +
  `LearnedPolicy` (a `Policy`) + `train_rule_value`.
* `tools/train_search_policy.py` — pretrain on synthetic matmul
  chains, post-train on small real torch models (exported to IR), then
  predict a rule for a held-out program and score it against the best
  and the mean.

The policy only *orders* legal moves; the certificate still decides
correctness (ADR 0003 invariant 5).  Nothing here can make a program
wrong — only choose a better one.

## Measured result

Trained on the GPU (RTX 2050, `torch 2.14.1+cu130`), 3000 epochs:

```text
device: cuda
pretrain samples: 3060
posttrain samples: 408 (+3060 replay)
held-out cost before: 150
policy picks: assoc_matmul  (delta +54)
best delta: +54; mean +1
picked rank: 1/51
```

The held-out program is `a(3,5)·(b(5,2)·c(2,3))` — the expensive
bracketing, with **k=5 unseen in training** (training used 2–4).  The
policy picks `assoc_matmul`, which is the best rule (rank 1 of 51),
improving the extracted cost by 54 FLOPs — the associativity move the
engine's cost model already rewards, chosen by a learned model.

## Findings (each cost real work)

1. **The structural encoding collided.**  Root-op buckets alone gave
   51 rules → 23 distinct vectors.  Adding a full pattern-tree hash
   took it to 49/51.  Without this the policy cannot tell two `matmul`
   laws apart.
2. **Raw-delta regression collapses to the mean.**  Most rules leave
   the cost unchanged, so MSE predicts ~0 everywhere.  Training on
   `delta > 0` with BCE fixed the ranking.
3. **Input standardisation is load-bearing.**  Raw features (FLOPs in
   the thousands) next to 0/1 one-hots starve the MLP; fitting
   mean/std buffers moved the held-out pick from rank 13 to rank 2.
4. **Post-training forgets.**  Fine-tuning on the small-model family
   alone drops the chain pick from rank 1 to rank 2 — catastrophic
   forgetting.  Replaying the pretraining family during post-training
   restores rank 1.  This is why the script trains on `pre + post`.
5. **It generalises across shapes.**  The held-out shape (k=5) was
   never seen; the policy still ranks the right rule first.

## Honest limitations

* **One family.**  The demo trains on matmul chains.  A mixed training
  set (chains + elementwise) dilutes the signal and the pick drops to
  rank 2 — the policy is not yet a general-purpose player.
* **Deeper chains.**  On a 3-matmul chain the policy picks the right
  *family* (`assoc_matmul`) but the wrong direction (best was
  `assoc_matmul_rev`, +8) — rank 30/51.  More structure in the rule
  encoding, or a value model over partial extractions, would help.
* **Not full RL.**  This is supervised pretraining on one-step
  trajectories, not policy-gradient RL.  The `Policy` port is the
  seam: a true RL policy is one more conforming value (the plan's
  stage 7 target), trained against the deterministic evaluator.
* **Single-step credit.**  `rule_samples` scores each rule applied
  alone; it does not credit sequences of rules.  Multi-step
  trajectories are the natural next dataset.

## Reproduce

```sh
.venv/bin/python tools/train_search_policy.py --device cuda \
    --pretrain 60 --posttrain 8 --epochs 3000 --hidden 96
```
