# Coordination optimality — is the hand-written pairing heuristic optimal?

The extraction *coordination* question, measured (ADR 0003, the
SEARCH/EVALUATION boundary).  Per-class greedy extraction cannot see
that `k` members each choosing `split_i(fused)` share ONE fused GEMM,
so `EGraph.extract_paired` (`catopt_core.egraph.extract`) forces every
pairing member to its split, steers member-reaching consumers through
it, and `optimize._select_best_term` keeps whichever of the forced term
and the greedy term is cheaper.

The question this retro answers:

> Is that hand-written coordination policy optimal, or is there a
> *quality* gap a learned policy could close?

**Verdict: the heuristic is near-optimal — a big win over naive greedy,
with a small, real, model-dependent residual gap.**  Under the shipped
default cost model it is at the optimum on ~92 % of enumerable
instances and loses by exactly **one kernel dispatch** on the rest
(~10 %); under `count_cost` and `flops_cost` it is optimal on
everything enumerated.  The gap is an *all-or-nothing* artefact, not a
search-coverage one.

> **Fixed (later).**  The all-or-nothing artefact is gone: the group
> decision is now **per-group** (commit `ab14439`), so the extractor
> can fuse a profitable group while declining an unprofitable one.
> Measured on this family: **92.2 % → 100 % at the optimum**,
> 4 instances improved, **0 regressions**; widening
> (`--instances 120 --space-cap 16384`): 79 → 85 at-opt, 6 improved,
> 0 worse.  The numbers below are the *pre-fix* baseline and are kept
> as the record of what the gap was.
>
> One correction the fix surfaced: this retro's "1-3 groups" holds for
> *this probe family*, but a real model feeds `extract_paired` **~90
> groups** (a 4-block MiniGPT through `_reify_merge`).  The exhaustive
> branch is therefore capped at G ≤ 3, with a bounded fallback above
> it that can never be worse than the old all-forced term.

Reproduce: `.venv/bin/python tools/coordination_probe.py`
(~80 s, CPU-only, seeded, no torch).  `--instances N --space-cap C`
widen the sweep.

## The action space (bounded, stated explicitly)

Coordination is the choice of which enode each *coordinated* class
takes.  The probe enumerates exactly the classes the shipped heuristic
touches:

* every **pairing-member class** -> any of its enodes (its own
  projection, its `split_i(fused)`, or a rule-introduced member);
* every **steering-candidate class** — a reachable class holding both a
  member-reaching and a bypassing enode, i.e. exactly the classes
  `extract_paired` steers -> any of its enodes.

Every other class stays greedy, as the shipped heuristic leaves them.
The space is `|A| = prod_c len(nodes(c))` over those classes.  A draw
is skipped when `|A|` exceeds `--space-cap` (default 4096); the skip
count is reported, not hidden.

**The bound is validated.**  On every instance small enough to
enumerate the *full* per-class space (every reachable multi-enode
class), the coordination optimum equals the full optimum — 24 such
instances in the default run, 0 mismatches.  So on those the
coordination space is the true optimum of any extracted term, not a
restriction of it.

## The ladder

Seeded random family: 1-3 shared-input projection groups under a random
consumer (`sub`/`add`/`mul`/`silu`-gated) and a random root op, all
shapes well-typed.  The shipped extraction default is
`executor_cost_for(lowering="generic")` (`optimize._default_cost_fn`);
`launch_aware_cost`, `count_cost`, `flops_cost` are scored alongside.
Ratios are to the coordination optimum; `%at-opt` counts ties.

### Executor (the shipped default) — n=80 draws, 51 analysed

| player  | mean ratio | %at-opt | worst  | n suboptimal |
|---------|-----------:|--------:|-------:|-------------:|
| naive   |     1.0192 |  37.3 % | 1.1111 |           32 |
| shipped |     1.0019 |  92.2 % | 1.0476 |        **4** |
| optimal |     1.0000 | 100.0 % | 1.0000 |            0 |

Shipped gaps: 4 instances (seeds 12, 17, 25, 35), **max abs 8701 ns =
exactly one dispatch**, max rel 4.76 %, median rel 4.76 %.

### launch_aware — same draws

| player  | mean ratio | %at-opt | worst  | n suboptimal |
|---------|-----------:|--------:|-------:|-------------:|
| naive   |     1.0021 |  37.3 % | 1.0048 |           32 |
| shipped |     1.0002 |  92.2 % | 1.0035 |        **4** |
| optimal |     1.0000 | 100.0 % | 1.0000 |            0 |

Shipped gaps: same 4 seeds, max abs **3** (≤ #groups), max rel 0.35 %.

### count_cost — same draws

| player  | mean ratio | %at-opt | worst  | n suboptimal |
|---------|-----------:|--------:|-------:|-------------:|
| naive   |     1.2597 |  37.3 % | 1.6667 |           32 |
| shipped |     1.0000 | 100.0 % | 1.0000 |            0 |
| optimal |     1.0000 | 100.0 % | 1.0000 |            0 |

### flops_cost — same draws

Every player ties the optimum (0 gaps, 0 suboptimal): fusing `k`
shared-input projections is FLOP-neutral, so FLOPs carry no
coordination signal at all.  This is why the gap only appears under
kernel-count / dispatch-priced models.

### Robustness

Widening the space cap (n=60, `--space-cap 16384`, 43 analysed) finds
the same story with one more gap instance (seed 40): shipped 88.4 %
at-opt, **5** suboptimal — i.e. ~12 % of enumerable draws, never more
than one dispatch each.  The gap *rate* is stable (~8-12 %); the gap
*magnitude* is always one dispatch per declined group.

## What the gap is (all four instances are the same shape)

The shipped policy is **all-or-nothing**: `_select_best_term` compares
only two terms — plain greedy, and "force *every* member of *every*
group to its split, then steer".  When one group's fusion is
unprofitable and another's is profitable, it cannot fuse the good one
without also fusing the bad one.

The unprofitable case is concrete: a group consumed by `add` gains a
**weight-merge bypass** from `WEIGHT_FACTOR_LINEAR`
(`add(linear(x,W1),linear(x,W2)) -> linear(x, W1+W2)`) — one GEMM on a
summed weight, *half* the FLOPs.  Forcing that group to its split pays
the fused GEMM's two outputs instead.  So when the `add` group sits
beside a `sub` group (no fuse rule, no weight-merge — the only fusion
is pairing, which genuinely wins), the forced term is net worse than
greedy and the shipped policy declines *both*, missing the `sub`
group's launch saving.

Seed 35 is the minimal case:

```
src     = mul(linear(x1, W1_0+W1_1),  sub(linear(x0,W0_0), linear(x0,W0_1)))
naive   = mul(linear(x1, W1_0+W1_1),  sub(linear(x0,W0_0), linear(x0,W0_1)))   87011.9
shipped = (same)                                                               87011.9
optimal = mul(sub(split0(FA), split1(FA)), linear(x1, W1_0+W1_1))              87011.5
          FA = linear(x0, concat(W0_0, W0_1))
```

The optimum fuses the `sub` group's two projections into one shared
GEMM and keeps the `add` group's weight-merge bypass; the shipped
heuristic can do one or the other, not both.  On the hand-built mix
cases the same effect shows as 0.24-0.28 % under `launch_aware_cost`
and 2 kernel launches under `count_cost` (which the shipped *does*
recover there, because forcing the `add` group is free under a pure
kernel count).

## Headline findings

1. **The heuristic is a large win over naive greedy.**  Under the
   default model it lifts `%at-opt` from 37 % to 92 %; under
   `count_cost` from 37 % to 100 %.
2. **The residual quality gap is small and dispatch-bounded.**  ~10 %
   of enumerable draws, by **exactly one kernel dispatch** each (max
   rel 4.76 % under the default model, 0.35 % under `launch_aware`).
   Never more than one dispatch per declined group.
3. **The gap is a policy-shape artefact, not a coverage one.**  The
   coordination space is reachable; the heuristic simply cannot take a
   *mix* of per-group fuse decisions.
4. **`flops_cost` has no coordination signal** (0 gaps everywhere) —
   pairing is FLOP-neutral by construction.  The headroom only exists
   where kernel count / dispatch time is priced.

## The action space a learned policy would choose

The shipped heuristic collapses the space `|A|` to **two** candidate
terms (greedy vs force-all).  The measured `|A|` (default run):
min 4, max 3840; 1-3 groups; 2-9 coordinated classes.  A learned
policy would choose:

* **per pairing group: fuse or not** — a binary vector in `{0,1}^G`,
  `G` = number of groups (1-3 here; `2^G` ≤ 8); *this is the whole
  gap*;
* **per steering candidate: which member-reaching route** — which of
  its enodes (product of class sizes), the routing choice the shipped
  steering fixes to a greedy minimum.

That is a tiny, discrete, structured decision — the natural home for a
learned policy.  But the *payoff* is bounded by the number of groups
(≤ one dispatch per group fused correctly), so the ceiling on the
quality headroom is small: at most a few dispatches, only in the
minority of graphs that mix a profitable and an unprofitable group.

## Honest limitations

* **Bounded enumeration.**  Draws with `|A|` above the cap (29/80 at
  the default 4096) are excluded; the larger-cap check shows the rate
  is stable, but a fully general optimum is not computed.
* **Coordination space, not all terms.**  The optimum is over the
  member + steering classes; the full per-class space was checked to
  agree only where enumerable (24 instances).
* **Kernel-launch unit.**  The gap is measured in each model's unit;
  under the default executor model one dispatch is 8700 ns, so a 4.76 %
  relative gap is still ~0.02 % of a large roofline-dominated term.
* **Linear groups only.**  `pair_shared_input_convs` is the conv
  analogue and is not exercised here (no torch); the mechanism is
  shared.
* **Family shape.**  The random family is small graphs (≤ 3000 enodes,
  ≤ 3 groups).  Deeper stacks could hold more groups, but the gap is
  bounded by group count either way.

## Reproduce

```sh
.venv/bin/python tools/coordination_probe.py                    # n=80, cap=4096
.venv/bin/python tools/coordination_probe.py --instances 60 --space-cap 16384
```
