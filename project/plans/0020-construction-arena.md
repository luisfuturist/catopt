# Plan 0020 — the construction arena (a better game for a better player)

Status: drafted.

## Why

The meta-game experiment measured something precise: **the pipeline
*scheduling* game is shallow** — the optimal arm-ordering is a fixed
rule a trained guide converges to but cannot beat (`law-meta-game.md`,
`guide-real-run.md`).  That is a finding about *that* layer, not
about learned play in general.  The space that is actually deep —
which objects to construct, compose, relax, specialize — was never
given to a player.  This plan defines the game where a learned
player could have an edge, plus the honest bars for "outperforms
humans".

## The game

- **Board.**  The evidence store + the real corpus + the declared
  object set + the shipped law library.  State is inspectable as
  data (proposals, regions, verdicts, pays).
- **Moves — structured construction operations** (each already
  proven mechanically in `object_synthesis`/`evidence`):
  `fold`, `lift`, `compose` (with guard transport), `auto_cond`,
  `relax_guard` *(new)*, `specialize` *(new)*, `ingest` (corpus
  direction).
- **Referee.**  The existing eight-stage gauntlet.  The player never
  decides equivalence (ADR 0003).
- **Score.**  Admitted objects that **pay on held-out real
  workloads** — zoo-discipline: pay on seeded spellings does not
  count (the corpus-circular lesson).

## Reward design (fixes the measured failures)

- **Dense partial credit** — each gauntlet stage cleared is progress
  (reconstruct→measure→truth→novelty→typed-pay→closure→cert), so
  credit assignment reaches the construction that caused it.
- **Hindsight attribution** — an ingest/compose action is scored on
  what it later enables, not its immediate delta.
- **Holdout gating** — reward reads `fires/pays` on a split the
  player cannot write into.

## Bars (honest, measurable)

1. Beat the fixed-rule baseline on *admitted-paying* objects per
   oracle call — if the optimum is still a rule, the game is
   honestly shallow at this level too.
2. Machine-admitted paying objects on the held-out zoo vs the
   human-authored library's on the same code — the "outperforms
   humans" bar.

## Phases

1. **Arena** — `arena.py`: actions as data, state view, referee
   wiring, episode driver.
2. **Baselines + depth probe** — fixed-rule, random, enumeration
   players; does context change the best move?  (Depth gate.)
3. **The player** — learned policy (RL or lookahead search) on the
   arena, vs baselines.
4. **Human bar** — machine vs human authored laws, scored on the zoo.
