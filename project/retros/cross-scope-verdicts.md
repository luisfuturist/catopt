# Cross-scope verdicts — the attributable history view

Plan 0017, the cross-scope stage.  The corpus-arms retro
(`guide-corpus-arms.md` §1) deferred a helper it named
`verdicts_across_scopes`: `latest_verdicts` only answers the
*current* scope, so a guide whose corpus grew lost sight of every
pre-rotation verdict.  The loss was deliberate then — a verdict's
`fires`/`paid` columns are corpus-dependent, and serving them raw
across scopes would conflate measurements.  This stage ships the
honest version of the cross-scope view: rows grouped by the
`corpus_hash` they were measured under, with the corpus-dependence
split written into the API rather than left to convention.

Reproduce:

    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --budget 40 --guide-step 6 --guide-corpus 6 \
        --plays-cap 12 --seed 0

Code in `catopt_discovery/evidence.py` (the new function sits next
to `latest_verdicts`; nothing else in the file was restructured)
and `catopt_discovery/meta_game.py`; tests in
`tests/test_discovery_evidence.py` /
`tests/test_discovery_meta_game.py`.

## 1. The API shape — grouped, never merged

`evidence.verdicts_across_scopes(conn, meta_base)` returns

    {corpus_hash: {alpha_key: verdict_row}}

— one `latest_verdicts`-shaped table per corpus scope, over every
`corpus_hash` that shares `meta_base`'s `rules_hash` and
`code_rev`.  Three decisions and their reasons:

- **Per-scope dicts, not a merged bag.**  The earlier stage's
  warning was that a flat "all verdicts for key K" view lets a
  reader compare `fires` measured on corpus A with `paid` measured
  on corpus B.  Grouping by scope makes the boundary structural:
  you cannot flatten the result without seeing the `corpus_hash`
  each row carries (rows are `SELECT *` dicts — `corpus_hash`,
  `run_id`, `ts` ride along, so attribution survives).
- **Only `corpus_hash` varies.**  `rules_hash`/`code_rev` still
  filter — a verdict under a holdout ruleset or a different verifier
  revision is a different context entirely, not "history".  The
  `"unknown"`-rev exclusion matches `latest_verdicts`.
- **The invariant split is named.**  `CORPUS_DEPENDENT_COLS`
  enumerates the columns whose value is a measurement *of* the
  corpus: `census_sites`, `matches`, `fires`, `fire_cases_json`,
  `changed`, `paid`, `verify_fail`, `drop_pct`, `cert`,
  `enode_ratio` — and `verdict` itself, deliberately: `SHIP`
  conflates truth with pay, so the label is a corpus-A answer to a
  corpus-A question.  The readable-across-scope columns are
  `numeric_true` (an equality measured on a real instance is a fact
  about the candidate — the corpus supplied the instance, not the
  truth), `derivable` and `relation` (both `rules_hash`-scoped,
  which the query holds fixed).

`GuideObs` gains `history`: `verdicts_across_scopes` minus the
current scope (which `verdicts` already serves — a row appears
under exactly one group, so the two fields cannot double-count).
`obs.verdicts` keeps its current-scope contract unchanged.

## 2. The honest limit: what history can and cannot say in-run

Worth stating precisely, because it shapes what a guide can do with
the seam.  Within one run the arena corpus only *grows*, and the
referee's `dedup`/`by_key` survive rotation — so a candidate is
adjudicated at most once (except `gap_gen`'s explicit
re-adjudication).  Two consequences:

- A `no-instance` target's prior-scope rows are always themselves
  `no-instance`.  A prior-TRUE target — "the equality held on
  corpus A but no instance exists on corpus B" — **cannot arise
  in-run**: a match found under A ⊆ B stays a match.
- Prior-TRUE rows appear only when the store outlives a corpus:
  a shared `conn` across runs on different slices (the persistent
  store's raison d'être), or a future corpus arm that *removes*
  terms.

So `PriorAwareGuide` (the one informed move this stage adds)
subclasses `EnumerationGuide` and reads only `numeric_true` from
`obs.history`: when `gap_gen` has a live target — approximated as
the keys whose newest verdict across scopes is no-instance
(`numeric_true` NULL, `matches` 0; the `_gap_done` bookkeeping is
arena-side, so the set can overcount a spent miss) — whose prior
row was TRUE, the witness `gap_gen` would synthesize converts a
*proven* equality rather than a guess, and `gap_gen` takes the
allocation ahead of the fixed order.  A FALSE prior is refutation
and does not trigger; corpus-dependent columns are never read.

## 3. Measurement — the trigger needs a store that outlives a corpus

**On the standard board (fresh store per run), prior-aware is
exactly enumeration — and that's the predicted result, not a
missed opportunity.**  The `--guide --guide-corpus 6 --budget 40
--seed 0` run (the stage-3 recipe, one extra guide):

| guide | yield_tf | vs enum |
|---|---|---|
| enumeration | .462 | 1.00× |
| prior-aware | .462 | 1.00× — same trace, same arms, same best |

The trigger is structurally quiet on a fresh store (§2), so the
row measures the seam's presence cost: one duplicate enumeration
run when `--guide-corpus` arms the board.  Honest negative —
recorded as such rather than dressed up.

**On a shared store the trigger fires and changes the schedule.**
The demo (a script, mirroring the new tests): corpus A = the tiny
six-term slice plus one `mul(div(x,y), y)` workload — the candidate
`mul(div(U,V),V) -> U` adjudicates SHIP there (fires=1, paid=1);
corpus B = the tiny slice alone — the same candidate adjudicates
`no-instance` and becomes a `gap_gen` target.  One `:memory:`
store shared across both arenas, `budget=8`, `step=2`:

| guide | trace (arm, draws) | first_true_at |
|---|---|---|
| enumeration | shape-aware×1 → workload×2 → workload×2 → gap×1 | draw 6 |
| prior-aware | shape-aware×1 → **gap×1** → workload×2 → workload×2 | draw 2 |

The prior-scope TRUE verdict (`numeric_true`, corpus-invariant)
let the guide spend the witness draw on a *known* equality at its
first live-target opportunity — three rounds earlier conversion,
same total yield on this board (the budget reached everything
either way).  On a board where the gap queue is contested — many
targets, budget tight — earliness is the yield: the draws go to
proven equalities first instead of a FIFO order blind to the
store's memory.

## 4. What stays deliberately un-served

- **No aggregate "lifetime SHIP" count on `GuideObs`.**  Summing
  `paid`/`fires` across scopes is exactly the conflation the shape
  exists to prevent; a consumer wanting a run-level harvest number
  can count `numeric_true`/`relation` keys across groups (the arena
  tallies already carry lifetime counts anyway).
- **No cross-run dedup of history.**  On a shared store, `history`
  honestly includes *other* runs' scopes — `run_id` is on every row
  for a reader that cares; the obs doesn't decide which runs are
  "this" run's.
- **The negative prior is read but not acted on beyond the
  trigger.**  A target FALSE under every prior scope is not
  suppressed from `gap_gen` — worth measuring before shipping the
  suppression (a refuted equality could still be declared with a
  guard, so "FALSE under A" is weaker evidence against witnessing
  than "TRUE under A" is for it).
- **`obs.history` is a read seam, not a target list.**  The guide
  cannot address `gap_gen` *at* a specific candidate — the arm
  keeps its oldest-first target order; history informs whether the
  queue is worth paying, not which entry is.

## 5. Interfaces shipped

- `evidence.CORPUS_DEPENDENT_COLS` — the documented dependent/
  invariant column split.
- `evidence.verdicts_across_scopes(conn, meta_base)` —
  `{corpus_hash: {alpha_key: row}}`, scope keys `rules_hash`/
  `code_rev` still binding.
- `GuideObs.history` — `verdicts_across_scopes` minus the current
  scope; `GuideObs.verdicts` contract unchanged.
- `meta_game.PriorAwareGuide` — enumeration + the prior-TRUE gap
  override; `meta_game._no_instance_keys` — the newest-verdict
  reconstruction of the live target set.
- `run_guide_experiment` registers `"prior-aware"` whenever
  `--guide-corpus` arms the board.
- 10 new tests: the column split, grouping/stamping, scope-key
  filtering, newest-per-scope collapsing; history emptiness,
  scope tagging across two rotations; the prior-aware trigger,
  its absence on no-instance/FALSE priors, the release after
  conversion, and fresh-store equivalence to enumeration.

Verification: `pytest tests/test_discovery_evidence.py
tests/test_discovery_meta_game.py -q` → 93 passed; `ruff check` /
`ruff format --check` clean on `evidence.py` and `meta_game.py`
(the tests carry the repo's documented pre-existing `tests/`
format drift; the new hunks are formatted); `ty` and the radon
ratchet report only the sibling agent's uncommitted `oracle.py`
sweep work — nothing in either file this stage touched.
