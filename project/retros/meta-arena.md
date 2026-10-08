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

## Feasibility pricing (the artifact fix)

The first probe's hole: a `declare` minting a *fresh op name* was
priced by the `_OP_FLOPS` default (1 flop/elem), so
`sub(x,y) → foldabs_0(x,y)` "halved" measured cost with no kernel
behind it.  Fixed at the extraction boundary, not in a report
column — the board's `feasible_cost` is
`backend_cost(cost_fn, supported)` composed with a declared-op
expansion (`meta_arena._feasible_cost`):

* op in `supported` → its `cost_fn` price (a sink lowers it);
* declared-but-unlowered op (a minted name with a recorded
  definitional unfold) → bills its *spelled* form: an abbreviation
  never extracts cheaper than what it abbreviates;
* neither supported nor defined → the never-win sentinel
  (`_INVALID_COST` — finite, so the `local = c(t) − Σc(children)`
  decomposition inside `extract_best`/`dag_cost` never degenerates
  to `inf − inf = nan`; reported as `+inf`).

`supported` is a board parameter (`MetaArena(supported=)`): bind a
sink's `supported_ops` for the production price — `main()` does
exactly that (`_torch_supported()` — lazy, the module stays
torch-free).  The default is the ambient vocabulary — every op the
program or the *base* ruleset spells — which is honest for this
board because the shipped library's names are ops real kernels
exist for; a `declare`-minted name is never among them, so the
bound freezes fresh names out by construction.

Extraction stays sound: a declared-but-unlowered member extracts
at *parity* with its spelled form (same e-class under the minted
unfold law — `unfold()` materialises the runnable program), never
below it; a name with no definition at all is infeasible and the
extraction simply cannot pick it.

The columns post-fix: `cost` is the feasible price; `cost_unfolded`
still expands *every* declaration (supported ones too), so the gap
`unfolded − cost` is exactly the premium a supported kernel earns;
`used_declared` names the declared ops the extraction picked.

A caveat kept explicit (module docstring): a fold to an
*already-named* op is a *claim* — the certificate certifies the
derivation, not the claim (the gauntlet lives in
`arena`/`evidence`, deliberately not wired into this board).  The
`legal_actions` enumerator therefore only offers fresh-name folds
(sound by definition); kernel claims stay playbook/probe-reachable.

## The Q2 probe, re-run under the bound (5 corpus terms)

Same arms as before (`scripted` / `greedy` / `random` budget 160 /
`declare`), now feasibility-priced.  Bound:
`TorchSink().supported_ops` — `sub`, `silu`, `softsign` are all
torch-lowerable, so a kernel claim on them earns its `cost_fn`
price; `foldabs_*` names are unlowered, so they bill their spelled
form.  Every terminal extraction's certificate replayed.

| case | base | scripted | greedy | random | declare | unfold | winner |
|---|---|---|---|---|---|---|---|
| silu_site (`mul(x,σx)`, DEFAULT−silu_fold) | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | silu_decl (+unfold) |
| softsign_site (`div(x,|x|+1)`, DEFAULT−softsign_fold) | 96.0 | 96.0 | 96.0 | 96.0 | 16.0 | 96.0 | softsign_decl (+unfold) |
| sub_gap (`add(x,-y)`, full DEFAULT) | 32.0 | 32.0 | 32.0 | 32.0 | 16.0 | 32.0 | sub_decl (+unfold) |
| nested_site (`tanh(div(x,|x|+1))`, DEFAULT−softsign_fold) | 128.0 | 128.0 | 128.0 | 128.0 | 48.0 | 128.0 | softsign_decl (+unfold) |
| control (`add(mul,matmul)`, full DEFAULT) | 160.0 | 160.0 | 160.0 | 160.0 | 160.0 | 160.0 | foldabs_0 (+unfold) |

What changed, reading down the columns:

* **The artifact is gone from the reward, not just the report.**
  Greedy and random now land on baseline *everywhere* — a
  fresh-name fold spends a move and buys nothing (it extracts at
  spelled parity, `cost == baseline`, reward `−step`), so
  cheapest-immediate play correctly finds no payout in it.
* **The real declares still win — now provably sink-backed.**
  `sub_gap` 32→16 (`sub` is torch-lowerable and priced 1 flop/elem
  — vs the spelled `add(x,-y)`'s 32, so `unfolded − cost = 16` is
  the claim's earned premium); `softsign_site` 96→16 and
  `nested_site` 128→48 (`tanh(softsign(x))`) the same way — claims
  whose kernel names exist in the bound.
* **`silu_site` is the honest flat case.**  `silu` is supported,
  and `flops_cost` prices it 3 flops/elem — identical to the
  spelled `mul(x,σx)` (16+32).  The shipped fold is a *launch-count*
  win, invisible to a FLOPs model: `cost == unfolded == 48`.
  Under the bound the board now reports that plainly instead of
  crediting a fresh name for it.
* **Control stays flat** — `foldabs_0` "wins" the declare column
  only by being the first certified candidate at baseline cost.

## Verdict

**Q2: yes — mid-search abstraction reaches optima no law sequence
reaches, and after feasibility pricing every surviving win names a
kernel a sink can lower.**

* `sub_gap` is the clean witness: under the *full* `DEFAULT` set
  `sub(x, y)` is unreachable (the library only spells it
  `add(x, -y)` — `sub_to_add` fires the expand direction);
  scripted extraction stays at baseline.  One `declare` inserts
  the fold+unfold pair, and the certified extraction is
  `sub(x, y)` at half the FLOPs — `sub` is in the bound, so the
  supported price counts.  `softsign_site` recovers the withheld
  shipped fold the same way (96 → 16), and `nested_site` reuses
  the same claim inside a `tanh` wrapper (128 → 48).
* **The unpriced-name strategy is dead by construction.**  In the
  first run greedy and random "beat" scripted on four of five
  cases, always via the 1 flop/elem default price of a name
  nothing could lower.  Under the bound those folds extract at
  spelled parity — `cost == baseline`, reward `−step` — so no arm
  finds payout in them, and the reward needs no separate honesty
  column to stay truthful (`cost_unfolded` now *confirms* rather
  than *corrects*: `unfolded − cost` is the claimed kernel's
  premium, zero for a fresh name).  The one caveat that survives:
  a *supported* name still prices at the model's table weight —
  `silu`'s 3 flops/elem happens to equal its spelled form, so the
  board honestly reports "no FLOPs win"; whether the claim's
  *kernel* is actually faster is the meter's question, not the
  search's.
* Q3 (is the mixed game non-shallow?) is *suggestive, not
  settled*: every win on the board is a construction move — and
  specifically a kernel *claim* against the bound, never a
  scheduling move or a fresh-name definition.  Whether a learned
  player beats the scripted baseline is the next measurement; the
  board, referee and honest pricing for it now exist.

## Housekeeping notes

* `MetaArena.step` / `meta_probe` were refactored to fit the
  radon ratchet (`_accept`/`_reward`/`_declare_best` split); the
  feasibility change added `_supported_or` / `_feasible_cost` /
  `_reported` / `_rel_delta` / `_arg_at` helpers — all under the
  rank-C ceiling, existing baselines unchanged.
* One uncovered arc remains in the module (99% file coverage):
  `extract`'s `best is None` early return — the root class
  always has members, so it is unreachable-defensive.
* Pre-existing ratchet drift (not this change): `EGraph.apply_rule`
  reads 25 vs baseline 24 — commit 7ed07f0 added a branch without
  regenerating `tools/complexity_baseline.json`.

## The ``handle`` move — scoped interpretation, landed

Sanada's arrow handler as an arena move: ``declare`` mints the
request, ``Action.handle(obj, handler)`` assigns its scoped
interpretation — a row of :data:`lawdata.HANDLERS`
(tag → ``{pattern, kernel, args}``, pure data).

* **Legality is checked, not trusted** — the handler's pattern
  must alpha-cover the object's spelled body (canonical spec
  comparison, metavar repeats enforced); a non-covering handler
  declines honestly.
* **Pricing** — the live ``interpretations`` table sits inside
  ``feasible_cost``'s expand: a handled name rewrites to its
  kernel *before* the spelled-parity expansion, so a supported
  kernel earns its real cost and an unsupported one prices
  infeasible (the move applies; the bound refuses it at pricing).
* **Measured delta** — under ``count_cost`` a ``mysilu`` fold
  priced 3 spelled / **2 handled**; ``cost_unfolded`` still
  reports spelled (3) — the fused premium is visible per column.
  Under ``flops_cost`` the same pair is flat: fused and spelled
  move the same data — the launch-count evaluation gap made
  *playable*, not just documented.
* **Memo caveat found the honest way** — the e-graph's shared
  cost memo assumes ``(cost_fn, term)`` purity; a handler makes
  pricing stateful, so ``_handle`` clears the memo on assignment.
  Pinned by ``TestHandleMove``.

Enumeration: ``legal_actions`` offers a ``handle`` per declared
body x covering handler (once — already-handled names are not
re-offered; re-interpretation is still spellable explicitly).
8 tests pin legality, pricing, memo invalidation and the
supported-bound honesty.
