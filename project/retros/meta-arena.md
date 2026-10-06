# Retro — the meta-arena (plan 0021 scaffold + Q2 probe)

Landed: `catopt_discovery.meta_arena` — the mixed board from
`project/plans/0021-meta-arena.md`.  One program under a live
`EGraph` at an enode budget; the action space mixes the two
measured dimensions:

* `fire(rule)` — one law application (the schedule/cost lever);
* `saturate(rules, budget)` — a bounded closure step (the old
  game's whole move, now one option);
* `declare(construction)` — an `object_synthesis` construction
  (fold/lift/compose/specialize) inserted into the live ruleset
  mid-search, with the definitional *unfold* pair minted when the
  constructed RHS is a faithful abbreviation;
* `extract()` — terminal: extract, build + replay the certificate
  (`verify_certificate`), score the delta vs baseline.

Reward is `lawdata.META_ARENA_REWARD` as data: applied non-terminal
moves pay `step + enode·<enodes added>`; extract pays `step` and
collects `delta·(baseline − cost)/baseline` only when the
certificate replays; declines score 0.  `META_SATURATE_BUDGETS`
enumerates the saturate arms.  Players: `ScriptedPlayer`
(saturate-then-extract — the baseline to beat), `GreedyPlayer`
(cheapest immediate — takes the payout the moment the extraction
beats baseline), `RandomPlayer`.  `python -m
catopt_discovery.meta_arena` reproduces the table.

## The honest columns

Two report fields exist because a naive read of `cost` lies:

* `cost_unfolded` — the extracted term with every declared
  abbreviation expanded back to its spelled form, repriced.  A
  fresh declared op prices at the `_OP_FLOPS` default
  (1 flop/elem), so a fresh-name fold "wins" *by cost-model
  construction*; unfolding collapses it to baseline.  A real gain
  survives unfolding.
* `used_declared` — the declared kernel ops the extraction picked.

A caveat kept explicit (module docstring): a fold to an
*already-named* op is a *claim* — the certificate certifies the
derivation, not the claim (the gauntlet lives in
`arena`/`evidence`, deliberately not wired into this board).  The
`legal_actions` enumerator therefore only offers fresh-name folds
(sound by definition); kernel claims stay playbook/probe-reachable.

## The Q2 probe (5 corpus terms)

Arms per case: `scripted` (search-only), `greedy`, `random`
(budget 160), `declare` (declare → saturate → extract per
candidate; best certified win).  `cost` = extracted flops_cost;
`unfold` = `cost_unfolded`; every extraction's certificate
replayed.

| case | base | scripted | greedy | random | declare | unfold | winner |
|---|---|---|---|---|---|---|---|
| silu_site (`mul(x,σx)`, DEFAULT−silu_fold) | 48.0 | 48.0 | 16.0¹ | 16.0¹ | 16.0 | 48.0 | foldabs_0 (+unfold) |
| softsign_site (`div(x,|x|+1)`, DEFAULT−softsign_fold) | 96.0 | 96.0 | 80.0¹ | 96.0 | 16.0 | 96.0 | softsign_decl (+unfold) |
| sub_gap (`add(x,−y)`, full DEFAULT) | 32.0 | 32.0 | 16.0¹ | 32.0 | 16.0 | 32.0 | sub_decl (+unfold) |
| nested_site (`tanh(div(x,|x|+1))`, DEFAULT−softsign_fold) | 128.0 | 128.0 | 112.0¹ | 112.0¹ | 16.0 | 128.0 | foldabs_2 (+unfold) |
| control (`add(mul,matmul)`, full DEFAULT) | 160.0 | 160.0 | 16.0¹ | 160.0 | 16.0 | 160.0 | foldabs_0 (+unfold) |

¹ every non-scripted "win" carries `used_declared = foldabs_*` —
the same artifact the `unfold` column exposes.  All 30 terminal
extractions verified.

## Verdict

**Q2: yes — mid-search abstraction reaches optima no law sequence
reaches, on the cases where the declared object is real.**

* `sub_gap` is the clean witness: under the *full* `DEFAULT` set
  `sub(x, y)` is unreachable (the library only spells it
  `add(x, −y)` — `sub_to_add` fires the expand direction);
  scripted extraction stays at baseline.  One `declare` inserts
  the fold+unfold pair, and the certified extraction is
  `sub(x, y)` at half the FLOPs.  `softsign_site` recovers the
  withheld shipped fold the same way (96 → 16).
* **But the dominant strategy on this board is minting unpriced
  names.**  Wherever a fresh `foldabs_*` fold is legal, *every*
  non-scripted player finds it — greedy and random "beat"
  scripted on four of five cases, always via the 1 flop/elem
  default price of a name nothing can lower.  `cost_unfolded`
  collapses every one of those to baseline.  The schedule game
  taught that order is a cost lever; this board teaches that
  *price* is the quality lever: `backend_cost(cost_fn,
  sink.supported_ops)` is the documented fix — a declared op
  outside `supported_ops` prices infinite and the artifact
  disappears from the reward entirely.
* Q3 (is the mixed game non-shallow?) is *suggestive, not
  settled*: the winning moves on every case were construction
  moves, never scheduling moves — matching the law-order verdict
  (order is a cost lever).  Whether a learned player beats the
  scripted baseline *once artifacts are priced out* is the next
  measurement; the board, referee and honest columns for it now
  exist.

## Housekeeping notes

* `MetaArena.step` / `meta_probe` were refactored to fit the
  radon ratchet (`_accept`/`_reward`/`_declare_best` split).
* One uncovered arc remains in the module (99% file coverage):
  `extract`'s `best is None` early return — the root class
  always has members, so it is unreachable-defensive.
* Pre-existing ratchet drift (not this change): `EGraph.apply_rule`
  reads 25 vs baseline 24 — commit 7ed07f0 added a branch without
  regenerating `tools/complexity_baseline.json`.
