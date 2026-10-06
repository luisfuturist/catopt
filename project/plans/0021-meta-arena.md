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

## Update — measured (night session)

Q1 is settled by `law_order`: **law-application order is a schedule
dimension, not a quality one** — extracted term + priced frontier
identical under ~350 ordering arms on the real corpus; wall-clock
swings up to 40× and under node caps order decides
convergence-vs-truncation.  So `apply law` ordering belongs to the
meta-arena as a *cost* lever (how much board you can afford), not a
quality lever.

Q2/Q3 are open; the construction arena (0020) is live and shows
real depth (150k moves, move choice matters on holdout pay).

## HANDL formalism notes (for the mixed arena's vocabulary)

From `handl-lang/platform/research/PAPER.md`:

- `>>>[k]` axis composition: axis 0 = horizontal (cells across a
  shared object — program composition), axis 1 = vertical (cells
  along a morphism — rewrite stacking).  CatOpt's `compose` is a
  horizontal 2-cell composite; `id₂ ∘₀ α` ("whiskering") — applying
  a 2-cell *inside* a context — is the apply-law-in-position move
  an e-graph move already performs implicitly; exposing it as a
  first-class move is the "apply law" action's real content.
- HANDL resolves 2-cells as **weak** (proof-relevant) — matches
  catopt's certificates: a derivation is data, replayed, not an
  equality assertion.
- `law_inference` is the strict fast-path "where a canonical form
  suffices" — exactly catopt's oracle/verifier fast path vs the
  e-graph's proof-relevant saturation.
- Coherence questions ("do two derivations of the same 2-cell
  commute") are 3-cell content — catopt's coherence catalogue
  (`divergent: 1` on the RMSNorm pair) is already that measurement.

So the meta-arena's action table, one level more precise:

```text
Action =
    whisker 2-cell            (apply law in position — axis-0 paste)
  | stack 2-cells             (vertical composition — derivation step)
  | introduce declared object (opdata declaration mid-search)
  | construct object          (fold/lift/compose/relax/specialize)
  | specialize runtime        (evaluation move — cost model swap)
  | change execution          (executor/carrier choice)
  | ingest corpus             (evidence-scope rotation)
```
