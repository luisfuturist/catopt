# Contraction ordering — is there headroom above the one-step greedy oracle?

Plan 0016 stage 7 (ADR 0003).  A learned / RL search policy can only
*add* value where the one-step greedy cost-model oracle is not already
optimal.  Every board measured so far — small matmul chains in the
e-graph rule space (`stage7-rl-results.md`) — had greedy optimal, so
the policy could at best *match* it.  This retro answers one question
with measurements:

> Is there a tensor-program search space where the one-step greedy
> cost-model oracle is **provably suboptimal**?

**Yes.**  Tensor-contraction ordering is such a space — in the classic
literature (NP-hard, greedy known to lose), in a generic measurement,
and inside catopt's *own* machinery (the e-graph rule space).

Reproduce: `.venv/bin/python tools/contraction_precondition.py`
(~24 s, CPU-only, seeded; `--instances N --seed S`).

## What was measured

Three layers, none tuned to make greedy lose; the random sweeps are
seeded and ties are reported as ties.

### 1. Matrix chains — greedy vs the O(n³) DP

Greedy = "contract the cheapest adjacent pair" (lowest index on ties).
Optimal = the classic interval DP.

| case | dims | greedy | optimal | ratio |
|---|---|---|---|---|
| clrs | 30,35,15,5,10,20,25 | 50625 | 15125 | 3.35 |
| small-a | 20,7,5,22,31 | 16540 | 7210 | 2.29 |
| small-b | 26,1,40,32,22 | 52080 | 2556 | **20.4** |
| small-c | 11,25,31,11,32,35,40,39 | 61028 | 60302 | 1.01 |
| flat | 10,20,30,40,50 | 38000 | 38000 | 1.00 (tie) |

Random sweep (n=60, 4–8 matrices, dims 1–40): **greedy loses 66.7 %**
of instances, mean ratio 2.77, median 1.43, p90 6.69, **worst 15.3×**.

### 2. General tensor networks — greedy vs an exact subset DP

3–6 tensors, mixed ranks, shared indices (hyperedges).  Greedy =
"contract the cheapest pair" (min union size).  Optimal = the subset
DP over the *parity* characterisation (a subset's intermediate carries
exactly the indices appearing an odd number of times), **validated
against exhaustive enumeration on 200 random cases** (0 mismatches).

| case | greedy | optimal | ratio |
|---|---|---|---|
| spiky | 41620 | 2404 | **17.3** |
| chain-hyper | 13728 | 2592 | 5.30 |
| mixed-rank | 6488 | 6488 | 1.00 (tie) |

Random sweep (n=60): **greedy loses 71.7 %**, mean ratio 2.09, median
1.06, p90 3.87, **worst 23.0×**.

So the domain is exactly what the literature says: greedy is far from
optimal, frequently and by large factors.

## 3. catopt's own machinery

### 3a. The diagram contraction-move space — greedy ties search

`catopt_orchestrator.diagram_search.search_moves` (bounded best-first
lookahead over diagram states) vs the greedy `_apply_moves` driver, run
on `nn.Linear` chain models (the diagram-liftable spelling of a matrix
chain), over the same dims as §1:

| dims | initial | greedy | search | search < greedy |
|---|---|---|---|---|
| 2,3,14,14,3,8 | 2485 | 129 | 129 | no |
| 30,35,15,5,10,20,25 | 19206 | 2202 | 2202 | no |
| 26,1,40,32,22 | 16404 | 386 | 386 | no |
| 11,25,31,11,32,35,40,39 | 46591 | 3433 | 3433 | no |

**Search never beats greedy on a chain (0/4).**  Two structural reasons,
both honest and both confirmed in the code:

1. **The move set collapses the whole chain.**  `ReorderCompose` folds
   every contiguous window of a chain into one morphism, and greedy
   claims *widest-first* — so greedy already folds the entire chain into
   a single block.  Search explores the orderings but lands on the same
   single-block state; the widest window subsumes every narrower one.
2. **The composed weight product is priced at zero.**  The reified term
   is `linear(x, matmul(Wₙ, … matmul(W₁, W₀)))` — the composed weight
   is a **param-only** subtree, and `catopt_core.cost.dag_cost`
   discounts param-only subtrees that fold to a parameter
   ("compile-time work — charged 0").  So the matrix-chain product has
   *no cost signal at all* under `flops_cost` / `launch_aware_cost` /
   `count_cost`: the final cost is just the one activation GEMM
   (e.g. 129 = 2·4·2·8 + 1 launch for dims 2,3,14,14,3,8).

In other words: the diagram move space is a **fusion** space, not a
**contraction-ordering** space.  On contraction-ordering instances there
is nothing for search — or a policy — to find.

### 3b. The e-graph rule space — the priced ordering, and real headroom

The matrix-chain ordering *does* live in catopt — in the e-graph rule
space, when the operands are `Var` (activation) leaves, so the products
are priced.  `assoc_matmul` reassociates `a·(b·c) ↔ (a·b)·c`; equality
saturation closes the full set of bracketings.

Validation first: on 30 random chains, **saturation reaches exactly
`2 × DP-optimal`** every time (the 2× is the FLOP constant; `flops_cost`
prices a matmul at `2·M·N·K`).  So the saturated optimum *is* the true
matrix-chain optimum — the e-graph space is the right place to score
greedy.

The one-step greedy oracle is exactly `tools/train_rl_policy.py`'s
`_greedy_rule`: pick the rule with the largest `rule_samples` delta,
apply it, repeat for `horizon` steps until `patience` stalls.

| dims | initial | sat-opt | 2·DP | 1-step | greedy (h6) |
|---|---|---|---|---|---|
| 2,3,14,14,3,8 | 4576 | 1216 | 1216 | **1560** | **1560** |
| 30,35,15,5,10,20,25 | 95000 | 30250 | 30250 | 30250 | 30250 |
| 26,1,40,32,22 | 59224 | 5112 | 5112 | 5112 | 5112 |
| 11,25,31,11,32,35,40,39 | 332514 | 120604 | 120604 | **184618** | 120604 |

Random sweep (n=30, 4–8 matrices), ratio vs the saturated optimum:

* **one-step oracle: loses 53.3 %** of instances, mean 1.72, p90 3.57,
  **worst 8.75×**;
* greedy episode (h6, p2): loses **6.7 %**, mean 1.003, worst 1.09.

The `2,3,14,14,3,8` row is the clean proving instance.  Greedy takes one
`assoc_matmul` step (4576 → 1560) and then **every rule's delta is 0** —
a strict local optimum.  The optimum 1216 is one more `assoc_matmul`
away, but that second step is *non-improving in isolation* (the single
application to the current best term raises the cost), so the one-step
oracle never takes it.  Applying `assoc_matmul` twice on the
*accumulated* e-graph reaches 1216 exactly.  That is a provable gap in
catopt's own priced space, reachable by a policy that tolerates a
non-improving step — the reward shaping question, made concrete.

## Verdict on the precondition

**Satisfied — with a precise location.**

* Generic matrix chains: greedy loses 67 % (worst 15×).
* Generic tensor networks: greedy loses 72 % (worst 23×).
* catopt **diagram** move space: **no headroom** — greedy ties search;
  the chain-ordering problem is param-discounted away.
* catopt **e-graph** rule space: one-step oracle loses 53 % (worst
  8.75×); multi-step greedy episode loses 6.7 %.

So the answer to "can a learned policy beat one-step greedy?" is **yes
in the e-graph rule space over activation-tensor chains**, and **no in
the diagram contraction-move space** as it stands.  The stage-7 RL retro
("greedy is the one-step oracle here; matching it is the available
win") was correct *for the 3-tensor chains it measured*; it is **not**
true for 4+ tensor chains, where the one-step oracle is myopic.

### What the state / action / reward would be

Concrete, over catopt's existing seams (`GameState`, `Action`,
`SearchEnv`, `RuleBook`):

* **state** — the e-graph plus the extracted best term
  (`GameState.eg`, `GameState.root_eid`), described by
  `ProgramFeatures` (root op, arity, per-leaf shape ranks/sizes, degree
  of each shared index) — the static features the stage-7 policies
  already read.
* **action** — one legal rule application at the root e-class
  (`Action(rule_name)`), i.e. `assoc_matmul` / `assoc_matmul_rev` /
  the categorical and scan laws; the natural contraction-move action set.
* **reward** — the **normalized cost improvement** `(before − after) /
  before` (`SearchEnv.step` already returns exactly this), plus the
  *lookahead* the one-step oracle lacks: a shaped term that credits a
  non-improving step when it *enables* a later improvement (e.g. a
  potential-based shaping `γ·V(s') − V(s)` with a learned `V`, or a
  small positive reward for any rule that grows the closure toward a
  cheaper extraction).  The `2,3,14,14,3,8` instance is the test case:
  the policy must be willing to take the second `assoc_matmul` even
  though its immediate reward is 0.

The certificate still decides correctness (ADR 0003 invariant 5): the
policy only *orders* legal rewrites, so it can only make the program
cheaper, never wrong.

## Honest limitations

* **Chain-dominated instance set.**  §1/§3 use matrix chains; §2 adds
  small mixed-rank networks but there is no catopt *diagram* spelling of
  a general tensor network, so §3 measures chains only.  A
  diagram-level contraction network (tensors as activation hyperedges)
  does not exist in the move set today.
* **Saturation is the optimum *here*.**  The e-graph optimum equals
  `2 × DP` on every chain tested because `assoc_matmul` closes the full
  bracketing space; on a board where the closure is budget-truncated
  (`rule_budgets`, `stop="improving"`) the "optimum" is only the
  reachable one.
* **One cost model family.**  All §3 numbers use `flops_cost`
  (the `SearchEnv` default).  `launch_aware_cost` would add a per-op
  launch term and could shift the exact ratios, though not the
  suboptimality (the bracketings differ in FLOPs, not kernel count).
* **Greedy is *a* baseline, not *the* baseline.**  The one-step oracle
  is the plan's baseline; a stronger classical baseline (e.g. a
  beam over contraction moves) would shrink the measured headroom.

## Reproduce

```sh
.venv/bin/python tools/contraction_precondition.py --instances 60
# chain-only, to see the proving instance in isolation:
.venv/bin/python -c "
from catopt_core.laws import all_rules
from tools.contraction_precondition import (
    _var_chain, egraph_sat_opt, egraph_greedy)
r = all_rules(); t = _var_chain((2,3,14,14,3,8))
print('sat', egraph_sat_opt(t, r),
      'one-step', egraph_greedy(t, r, horizon=1, patience=1))
"
```
