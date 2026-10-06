# Retro — arena legal-move enumeration + the first construction depth probe

Plan 0020 phase 2.  `legal_actions(state)` enumerates the live
construction moves; `depth_probe` plays the three baseline players
on the real board (`python -m catopt_discovery.arena`).

## The board is large

Legal move count at episode start: **~143k–152k**, growing to ~187k
after an episode (each stored object adds relax/specialize moves).
The action space is combinatorial — composite spellings × kernels ×
carrier bases × premise pairs × guard clauses × metavariable pins.
This is not a scheduler with four arms; it is a deep board.

## The probe (2 episodes × budget 8, seed 0)

| player | steps | usable | holdout fires | holdout paid | reward |
|---|---|---|---|---|---|
| fixed  | 5/5 | 0 | 8 / 0 | **2 / 0** | 23.6 / 17.0 |
| random | 8/8 | 0 | 2 / 0 | 0 / 0 | 24.4 / 24.0 |
| greedy | 8/8 | 0 | 0 / 0 | 0 / 0 | 24.0 / 24.0 |

## Honest read

- **Move choice matters** — the authored playbook's `compose` step
  (`comm_mul ∘ silu_fold` → `silu_swapped`) pays 2 on holdout;
  greedy pays nothing.  The payoff surface is not flat.
- **Nobody mints `usable` in 8 moves** — the easy constructions are
  already in the store; admitting a new object takes more horizon
  than the probe's budget.
- **Reward is stage-clear-dominated** — random/greedy score ~24 by
  clearing 3 stages per applied move; holdout pays are the signal
  that separates them.
- Episode variance is real (fixed ep1: 0 fires) — different split
  seeds move the holdout.

## What's wired

`relax_guard` / `specialize` are registered arena actions with
`Action.relax_guard`/`Action.specialize` classmethods; `ObjectView`
exposes `cond_clauses`/`leaf_metavars`/`attr_metavars` (the object's
own stored data — safe for the player); `legal_actions` enumerates
per-clause relax moves and per-metavar pin moves over the
`lawdata.SPECIALIZE_{SCALARS,AXES}` banks.  The pin banks are data —
a richer pin is a data change, not code.

## The 24-step probe (3 episodes × budget 24, seed 0)

| player | usable | holdout fires | holdout paid | reward |
|---|---|---|---|---|
| fixed  | 0 | 8/0/0 | **2/0/0** | 23.6/17/17 |
| random | 0 | 2/2/1 | 0/1/0 | 66/72/66 |
| greedy | 0 | 0/0/0 | 0/0/0 | 72/72/72 |

Two findings:

- **Nobody reaches `usable` in 24 moves.**  The admitted-object bar
  needs longer horizons or guided play — random and greedy churn
  through applied-but-unproductive constructions.
- **The reward is stage-clear-dominated.**  Greedy maxes reward
  (72.0) with *zero* holdout yield — `stages_cleared + 0.2·fires +
  2·paid` lets three cleared stages outscore a paying object until
  it admits.  **Fixed as data**: `lawdata.ARENA_REWARD` is now
  `stage=0.1, usable=10, fire=0.5, paid=5` — the objective
  (admitted, paying objects) outweighs a full budget of churn
  (24 × ~0.3 ≈ 7 < one usable's 10).  Under the old weights the
  same fixed episode scores 23.6; under the new, ~15–16 — still the
  only paying line, now correctly top of the board.

## Verdict for plan 0021

The construction board is **deep enough to measure**: large action
space, payoff differences between players, honest holdout.  The
board's depth for a *learned* player is the open question — the
reward no longer rewards churn, so a longer-horizon probe (or a
learned policy over the ~150k-move set) is the next measurement.

## The cheap board / long horizon probe

The real board's ~5-8 s/step (a full gauntlet over 261 working terms
plus a 50-case probe) made the depth question unmeasurable: 24 steps
cost ~1 h.  `make_arena` now takes `max_cases` / `max_holdout` —
seeded sha256-ranked subsamples drawn *after* the honest split, so
the holdout stays real-only and probe-eligible and can never leak
back into working.  At `max_cases=24, max_holdout=12` the board has
~35k-51k legal moves, a deep step costs ~1 s, and the probe below
(5 episodes x budget 60, four arms) ran in **~15 min**.

The probe (`max_cases=24`, `max_holdout=12`, seed 0, budget 60):

| player | steps | usable | holdout fires | holdout paid | reward |
|---|---|---|---|---|---|
| fixed     | 6-7/episode (drains) | **1/0/0/0/1** | 4/0/0/0/4 | 1/0/0/0/1 | 19.3/2.3/2.0/2.3/19.3 |
| random    | 60 | 0 | 0/1/1/5/0 | 0/0/0/2/0 | 14-24 |
| greedy    | 60 | 0 | 0 | 0 | 18.0 all |
| heuristic | 60 | 0 | 0 | 0 | ~17.8 all |

Three findings:

- **`usable` is reachable but only through authored knowledge.**  The
  fixed playbook admits the aff-step lift in 2 of 5 episodes (the
  subsample decides whether the working set keeps a measuring site
  and the 12-case probe keeps a paying one).  No enumerated-frontier
  arm — random, greedy, or the new heuristic — mints a usable object
  in 300 moves.
- **Move ordering separates on *depth*, not yet on yield.**  The
  heuristic (`lawdata.ARENA_MOVE_ORDER`: auto_cond -> compose ->
  relax -> fold -> ingest -> specialize -> lift, each move spent
  once) gets ~29% of its plays past truth+novelty vs ~2% random and
  0% greedy — but converts none to holdout fires.  Its episode is
  34 composes + 26 auto_conds: the preference walk *never drains the
  top classes*, so at this budget it never reaches the fold/lift
  mass where firing objects live.  A ranked order is a frontier
  policy; the tail ranks are unreachable when the head class has
  ~14k members.
- **The bottleneck is `truth`, then `typed-pay`.**  Stage-failure
  aggregates over the run: random `truth` 199 / `full-data` 72 /
  `declined` 24 / `typed-pay` 5; greedy `truth` 300 (every largest-
  spec fold is a false equality); heuristic `truth` 154 /
  `typed-pay` 86 / `declined` 60.  `truth` kills most minted
  constructions — the board's constructions are mostly *false*
  equalities, not unmeasured ones.  For deeper arms the wall moves
  to `typed-pay`: true, novel objects that never fire or pay on the
  probe (12 holdout cases is a small target — a `usable` needs
  fires>0 *and* paid>0 *there*).  `full-data` (72, all random) is
  the compose-products-drop-check-hooks class.

Two infrastructure findings the run surfaced:

- **Cert materialization gap.**  An enumerated `compose` whose
  specialize map leaves an attr metavariable free (the `sdpa_fold_*`
  family's `SC` — `INSTANCE_ATTR_DEFAULTS` covers TD/SD/DP/DT but
  not `SC`) produced a rule `lemma_cert.materialize` raised
  `KeyError` on — an episode-killing crash.  `Arena.step` now
  converts referee-side failures into an honest `applied=False`
  refusal (`referee declined: ...` in the note); ~1 in 60 random
  moves hits it.
- **Cheap-board auto-cond starvation.**  All 12 heuristic declines
  per episode are `auto_cond` refusals — "no equal site in the
  measured domain": a 24-case working corpus measures too few equal
  sites for the guard synth to cover.  The rescue class is weaker on
  the small board by construction.

Verdict, sharpened: the board is deep-but-winnable — `usable` is
*reachable*, not far-away-invisible; the blocker is that enumerated
constructions are mostly false (`truth`) and the true ones mostly
don't pay on the probe (`typed-pay`).  A learned player's edge, if
any, is in *targeting* — picking constructions that fire on real
workloads — which is exactly what the frontier arms cannot do.

## Post-rebalance probe (cheap board, 3 eps × budget 40)

Under `ARENA_REWARD` = `stage=0.1, usable=10, fire=0.5, paid=5`:

| player | usable | fires | paid | reward |
|---|---|---|---|---|
| fixed  | **1/0/0** | 4/0/0 | 1/0/0 | **19.3**/2.3/2.0 |
| random | 0 | 0/1/1 | 0 | 9.7–10.7 |
| greedy | 0 | 0 | 0 | 12.0 |
| heuristic | 0 | 0 | 0 | 9.7 |

The inversion is fixed: the episode that mints a usable, paying
object tops the board (19.3), churn policies score below it.
Stage-failure aggregate: `truth` kills most constructions
(72–120/player), then `typed-pay`/`novelty`/`full-data`.
