# Plan 0021 — the meta-arena (the fuller game)

Status: drafted.  Builds on plan 0020 (the construction arena).

## Why

Plan 0020's arena measures whether a learned player can navigate
*object construction* (corpus-level).  The discussion's fuller
frame is **program-level meta-search**: one player whose action
space mixes rewrite moves, abstraction moves and execution moves
over a single candidate program —

```text
Action =
    apply law              (search dimension)
  | introduce abstraction  (HANDL/opdata-declared object mid-search)
  | compose abstractions   (plan 0020 ops)
  | specialize runtime     (evaluation dimension)
  | change execution       (executor/carrier/cost-model choice)
  | expose/fold representation
```

The e-graph + certificate stays the referee (ADR 0003): the player
can explore aggressively because no move can make a program wrong.

## Pieces that already exist

- `catopt_core.search_env` / `policies` — the move-ordering harness
  (the apply-law board).
- `object_synthesis` + `arena` (0020) — the abstraction moves.
- `opdata` — the declaration seam: "introduce abstraction" mints a
  declared op + its unfold/fold pair, insertable into the mid-search
  vocabulary.
- `cost.executor` + carriers — the execution-strategy move set.

## Depth questions (measured, in order)

1. Does a **guided law-order player** beat beam/greedy on real
   graphs?  (The contraction player won on its board; the law-order
   board is unmeasured.)
2. Does **introducing an abstraction mid-search** reach optima no
   law sequence reaches?  (`silu_fold` mediators hint yes.)
3. Does the **mixed** action space reward context-dependent play —
   i.e. is the meta-game non-shallow where the scheduling game was
   not?

## Bars

- Player vs enumeration/greedy on the same board, verified
  certificates, equal budget.
- Human bar unchanged: machine-admitted paying objects on the zoo.
