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

## Verdict for plan 0021

The construction board is **deep enough to measure**: large action
space, payoff differences between players, honest holdout.  The next
probe needs a longer horizon (or a smaller/cheaper corpus) before
"does a learned player beat the rule" is answerable — 8 moves does
not reach `usable`.
