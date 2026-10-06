# The law-order board — measured shallow for quality, real for schedule

Plan 0021 depth question 1.  The contraction player wins on the
contraction-ordering board; this probe asks whether the **order laws
are applied during e-graph saturation + extraction** changes the
extracted program — i.e. whether a learned law-order player has a
real board.

Instrument: `catopt_discovery.law_order` (new).  Each `(term,
regime)` cell runs `EGraph.run` under six orderings — `declared`
(`policy=None`), `reversed`, `random:{0,7}` (`RandomPolicy`, seeded),
`expansive_last` / `expansive_first` (`GreedyPolicy` on the
`EXPANSIVE` tag) — then records enode/class/proof-edge counts,
per-rule fire counts, the extracted cost + spelling, the priced
top-8 root frontier, wall-clock, and whether the extracted term's
certificate replays under `verify_certificate`.

Reproduce:

    .venv/bin/python -m catopt_discovery.law_order --corpus models --regime all
    .venv/bin/python -m catopt_discovery.law_order --corpus zoo --regime prod

Boards: `zoo` — the 22-model held-out set (`catopt_discovery.zoo`);
`models` — the 44-case real corpus (`impact.model_cases`, the same
`catopt_torch` blocks the discovery pipeline measures against).
Regimes: `prod` (the `search()` defaults — `EXPANSIVE` rules capped
at 2048 new enodes each, `max_nodes=100k`), `tight` (128/20k),
`improving` (`stop="improving"`, patience 3, 2048/60k).

## Numbers

**Zoo (22 cells):** `answer-diff=0 contents-diff=0 cert-fail=0` —
and the board is mostly *empty*: 15/22 workloads record **zero**
proof edges (no law fires), the rest ≤ 4.  Consistent with
`model-zoo-yield.md`: the shipped laws' LHSs don't spell on held-out
architectures, so there is no ordering to matter.

**Models (132 cells = 44 cases x 3 regimes):**
`answer-diff=0 cert-fail=0`, `contents-diff=24`.
Across a second sweep at expansive budgets 8/32/128 on the 26 boards
with any rule fires (~350 arm comparisons total): **zero extracted-cost
or extracted-term differences, including the full top-16 priced
frontier.**

Where order does move things (models corpus, 26 firing boards):

| case | regime | ordering | enodes | classes | proof edges | wall |
|---|---|---|---|---|---|---|
| SelectiveSSM | prod (2048) | declared | 4912 | 429 | 4483 | 1000 ms |
|  |  | reversed | 7495 | 429 | 7065 | **40724 ms** |
|  |  | random:0 | 5743 | 429 | 5309 | 4527 ms |
|  |  | random:7 | 5070 | 429 | 4626 | 4777 ms |
|  |  | expansive_last | 4956 | 429 | 4489 | 1071 ms |
|  |  | expansive_first | 4912 | 429 | 4483 | 1051 ms |
| SelectiveSSM | tight (128) | declared | 675 | 203 | 461 | 77 ms |
|  |  | reversed | 701 | 197 | 492 | 84 ms |
|  |  | random:0 | 700 | 217 | 476 | 77 ms |
|  |  | random:7 | 679 | 214 | 458 | 86 ms |
|  |  | expansive_last | 723 | 209 | 507 | 68 ms |
|  |  | expansive_first | 675 | 203 | 461 | 76 ms |

Every row extracts cost **2248.0**, identical term spelling,
identical top-8 frontier, `certificate_ok=True`.

Smaller boards show the same shape at smaller scale: SwiGLU ne
11 vs 12, TransformerBlock 54 vs 55, ParallelBlock 43 vs 44,
LinearRecurrence 59 vs 61–69, KernelizedAttention 25 vs 26,
DiagDenseSSM 93 vs 94–98 — contents move, class count usually does
not, answer never does.

## Why contents diverge

Two mechanisms, both measured:

1. **Quotient-collapsed instantiation.**  `apply_rule` instantiates a
   rule's RHS resolving children through the *current* union-find.
   If an earlier rule already unioned `mul(g, sigmoid(g))` into the
   `silu` class, a later rule whose RHS contains `mul(g, sigmoid(g))`
   sees its instantiation collapse onto an existing enode — hash-cons
   dedup — and contributes nothing.  Reverse the order and the same
   rule materializes a *distinct* enode that a third rule later
   merges.  Minimal reproducer, `mul(silu(a@b), a@c)` under DEFAULT:
   `declared` fires `{silu_expand}` (ne=9, nc=8); `reversed` fires
   `{silu_mul_form, silu_fold}` (ne=10, nc=8) — the extra enode is the
   mul-form RHS instantiated before the expand could collapse it.
   The **partition converges** (nc=8 either way); the *stored enode
   set* does not.
2. **Budget-relative truncation.**  `rule_budgets` suspends an
   EXPANSIVE rule once its lifetime enode spend is exhausted; the
   resulting "fixed point" is only fixed relative to that truncation,
   and the schedule decides *which* subset of the closure the budget
   bought.  On SelectiveSSM/tight all six arms suspend 4–5 expansive
   rules with different spend distributions → class counts diverge
   (197–217).  Under `max_nodes` truncation order decides whether
   the run converges at all: at cap 5000, `declared` reached a fixed
   point at 4970 enodes while `reversed`/`random:0` blew past the cap
   (prod-scale probing, stop=`max_nodes`).

## Why the answer doesn't move

- A `Policy` may only **reorder** the rules `EGraph.run` fires —
  the schedule cannot change which equalities are assertable, only
  which enodes get materialized within a budget (ADR 0003 inv. 5;
  `EGraph._scheduled` still runs every offered rule).
- `extract_best` prices *terms*, not enodes.  The extracted winner
  on every corpus board is produced by direct LHS hits on the input
  spelling — the fusion/simplification folds (`silu_fold`,
  `assoc_matmul`, sdpa/rms/qkv family) — none of which lives deep in
  the budgeted expansive closure that order perturbs.  Even with the
  expansive rules capped at **8 enodes**, the same argmin survives.
- The priced **frontier** (top-16 root alternatives) is likewise
  identical across arms, including on SelectiveSSM where the raw
  enode set differs by 1.5x — the order noise lives in the closure's
  tail, not in the extractable surface.
- The only channel by which order *could* change the extracted
  program — an ordering that suspends a rule before a
  payoff-critical match site exists, so the optimal member is never
  materialized — never occurred in ~350 arm comparisons.

## Verdict

**The law-order board is shallow as a quality game.**  A learned
player ranking legal law applications has nothing to win on
extracted cost: under fixed-point *or* budgeted-truncated *or*
`stop="improving"` saturation, every ordering measured reaches the
identical extracted term.  The payoff surface is flat — there is no
gap to learn.

What ordering does buy is **schedule efficiency and convergence
safety**: up to **40.7x** wall-clock spread on the largest board
(`reversed` spent 25x the proof edges to reach the same answer),
with order deciding convergence-vs-cap under node budgets and
iteration count under lazy saturation.  That is a *defensive*
board — "don't pick the order that explodes" — not a progressive
one.  If a learned player has a role here it is as a saturation
*scheduler* minimizing time-to-fixed-point, a different objective
than the contraction player's cost board and a poor fit for plan
0021's "player vs enumeration on extracted cost" bar.

Depth in plan 0021 therefore lives (if anywhere) in the *other*
action dimensions: mid-search abstraction introduction and the
mixed action space — not in law order.

## Honesty / limits

- Corpus-bounded negative: 44 real blocks + 22 zoo models, one
  machine, CPU timings.  A program whose optimum sits inside the
  truncated expansive closure would break the invariance — the probe
  didn't find one; `law_order.py` stays in-tree to re-run the
  question whenever the law library or corpus grows.
- Per-process determinism is exact (seeded arms replay bit-for-bit);
  across processes `PYTHONHASHSEED` shifts set-iteration order and
  can move *contents* counts slightly — never the extracted answer,
  which `extract_best`'s canonical member sort immunizes.
- `SearchEnv`-style episodes (one rule per step, horizon-bounded)
  conflate ordering with *subset choice* under the horizon — a
  different board, not measured here.
- BoardRun `frontier` is top-k of the root class only; sub-class
  extraction divergence is out of scope (`SearchResult.frontier`'s
  documented limit applies).

## Gates

- `packages/catopt-discovery/src/catopt_discovery/law_order.py` new;
  `tests/test_law_order.py` new (17 tests, ~3 s).  No engine/laws/
  sibling-file edits.
- `pytest tests/test_law_order.py` — pass.  `ty check`, `ruff check`,
  `ruff format --check`, `vulture`, `lint-imports`, `radon_ratchet` —
  all clean.
- Full suite / coverage gates not run (per probe instructions);
  nothing committed.
