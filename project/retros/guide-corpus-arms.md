# Corpus arms on the board — scope rotation and the re-measured game

Plan 0017 stage 3.  The stage-2 retro (`guide-seam.md`) measured an
honest negative — no guide beat enumeration — and named the reason:
the regime where guiding should pay (pool ≫ budget) needs the
corpus-generating arms, and those were excluded because a mid-game
corpus mutation would silently mix evidence scopes.  This stage
fixes the scope problem, puts `workload_gen` / `gap_gen` on the
board as legal `Allocation` arms, and re-measures.

Reproduce (each run ~1–3 min CPU):

    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --budget 40 --guide-step 6 --guide-corpus 6 \
        --plays-cap 12 --seed 0
    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --budget 70 --guide-step 8 --guide-corpus 8 \
        --plays-cap 12 --seed 0

Everything lives in `catopt_discovery/meta_game.py`; tests in
`tests/test_discovery_meta_game.py`.

## 1. The scope-key fix — the store was already honest

Investigation result first: `evidence.record_run` needed **no
change**.  It writes `meta["corpus_hash"]` per call, so each run row
is already scoped by the corpus it was measured on.  The blocker
was entirely in `meta_game`: `GuideArena.meta` was built once and
never rotated, so a mutated corpus would have kept writing verdicts
under the *stale* hash — silently attributing corpus_B measurements
to corpus_A.  `latest_verdicts` likewise needed nothing: it filters
by the scope keys it is given, so once `meta["corpus_hash"]` rotates
the lookup is correct by construction.

The seam, implemented in `GuideArena._ingest`:

- each ingested workload appends to `ref.terms` (the instance
  search), `ref.cases` (the firing probe) and the census corpus;
- `meta["corpus_hash"]` is recomputed over the grown
  `ref.terms` (same seam the constructor uses) and `meta["ts"]`
  refreshes — `run_id` stays, so the store sees one run measured
  across scopes;
- `GuideObs.verdicts` reads `latest_verdicts` at the **current**
  hash — post-mutation the guide sees only evidence attributable to
  the corpus it just grew.  Old rows remain in the store under
  their own scope, retrievable by hash — attribution is never
  destroyed, only re-scoped.

No `evidence.py` helper was needed.  One worth noting for the
object-record owner (not patched): a `verdicts_across_scopes` /
history-by-corpus view — rows for an alpha key grouped by
`corpus_hash` — would let a guide (or an audit) see *stale*
evidence tagged with the scope it belongs to.  Currently
`latest_verdicts` only answers the current scope, so a guide that
grows the corpus loses sight of corpus_A verdicts entirely; the
arena's own `ArmStat` tallies carry the lifetime view instead.
That lossy observation is deliberate this stage — a verdict's
`fires`/`paid` columns are corpus-dependent, so serving them
cross-scope would conflate measurements.

Mid-game regrowth does not resurrect stale candidates: the
referee's `dedup`/`by_key` caches are corpus-scoped in effect — a
previously `no-instance` candidate is only re-adjudicated through
`gap_gen` (below), never silently.

## 2. The corpus arms

`CORPUS_ARMS = ("workload_gen", "gap_gen")` — armed by
`GuideArena(gen_cap=N)`, `N` draws each; a draw costs a draw (the
same currency as refereeing a proposal) and is tracked per-arm.

- **`workload_gen`** — one draw mints one verified workload:
  census-resample or real-model mutate, then the five-gate
  `workload_gen.valid_term` (op term, node budget, sink-lowerable
  ops, well-typed, torch-evaluable, corpus-novel) — the same gates
  the intake path enforces — then `term_to_case` and `_ingest`.  It
  scores 0 immediately; its payoff is indirect: the census pools
  **regrow** after every ingestion (`_regrow_pools` re-derives the
  three corpus-derived generators and appends only keys never
  queued or adjudicated, so the merged-pool dedup semantics hold).
  A failed mint is a `gen-miss` draw — the draw is spent, nothing
  is invented.
- **`gap_gen`** — live only while an untargeted `no-instance`
  candidate exists.  A draw synthesizes a witness workload for the
  oldest such candidate via `gap_gen.gen_cases_for` (the instance
  must re-match its own pattern, satisfy check/derive, pass the
  same validity gates), ingests the min/ctx/graft embeddings, and
  then **re-adjudicates the candidate under the grown corpus** —
  the stale dedup entry is dropped, the referee re-searches and
  re-measures, and the new verdict is recorded under the rotated
  scope (oracle call honestly attributed to the arm).  A failed
  synthesis is a `gap-miss`; an arm with no targets is not live.

`EnumerationGuide` schedules corpus arms last
(`ENUMERATION_ORDER + CORPUS_ARMS`) — the pipeline's fixed
inventory first, then growth; `enum-corpus-first` is the flipped
control, reported alongside when the arms are armed.
`LearnedGuide`'s arm featurization gains an is-corpus flag.

## 3. Re-measurement — does any guide beat enumeration now?

Pool on the real slice: 52 proposals + `plays_cap` build draws +
`2 × guide-corpus` corpus draws — with `--guide-corpus 6`,
`--plays-cap 12`, budget 40 the drawable space (76) finally exceeds
the budget.  Yield = true+firing candidates per oracle call.

### 3.1 Budget 40, step 6, gen_cap 6, seeds 0/3/5

| guide | seed 0 | seed 3 | seed 5 |
|---|---|---|---|
| enumeration (corpus last) | .462 (1.00×) | .462 | .462 |
| uniform | .714 (**1.55×**) | .200 (0.43×) | .571 (**1.24×**) |
| learned | .444 (0.96×) | .286 (0.62×) | .393 (0.85×) |
| enum-corpus-first | .538 (**1.17×**) | .478 (**1.04×**) | .480 (**1.04×**) |

### 3.2 Budget 70, step 8, gen_cap 8, seed 0

| guide | yield_tf | vs enum |
|---|---|---|
| enumeration (corpus last) | .259 | 1.00× |
| uniform | .516 | 1.99× |
| learned | .385 | 1.48× |
| enum-corpus-first | .531 | **2.05×** |

### 3.3 Reading

**The negative turned into a qualified positive, then back into a
schedule finding.**  With the corpus on the board:

- Allocation finally matters.  `gap_gen` is the strongest arm ever
  measured here: it converted `no-instance` candidates into
  true+firing (often paying) verdicts at roughly 4–6 finds per 6
  draws — seed 0 uniform went 6/6, corpus-first 8/8 at budget 70.
  These are candidates the corpus could never even evaluate; a
  synthesized witness makes the oracle's answer *reachable*.  The
  `SHIP` verdicts are scope-stamped and honestly mean "fires and
  pays on a corpus that now contains the witness workload" — the
  `gap_gen` retro's own acceptance test, automated.
- Any policy that *reaches* the corpus arms beats the
  proposals-first fixed order (uniform 1.55×/1.24×, learned 1.48×
  at loose budget).  Enumeration's fixed schedule is what loses —
  it structurally delays corpus growth behind a 52-draw queue it
  can't finish in-budget.
- **But no adaptive policy beats the static corpus-first
  schedule.**  `enum-corpus-first` ≥ every other guide at budget 70
  (2.05× vs uniform 1.99×, learned 1.48×) and is positive at every
  seed at budget 40 (1.04×–1.17×).  The winning move — grow the
  corpus early, referee over the grown inventory — is a constant
  schedule; the learned guide's ~7–10 reward observations per run
  still cannot beat knowing it a priori.  `workload_gen` showed 0
  immediate yield everywhere (its payoff is the pool regrowth —
  e.g. `census-mixed-view` emitted 16→18 on seed 0 — and new firing
  sites for later draws).
- Seed 3's uniform collapse (0.43×) is the old story in new arms:
  uniform burned 12 draws on `build` (0 true) and 6 on
  `workload_gen`, never reaching `gap_gen`.  When the high-yield
  arm needs *recognizing*, a uniform policy is coin-flip fragile;
  the fixed corpus-first order is not.

**The publishable verdict**: the stage-2 negative was about the
*inventory*, not the policy — pool ≫ budget does make allocation
matter (a 2× yield spread between schedules), and the corpus arms
are where the yield lives.  But on this board the optimum is a
static schedule, not a learned guide: corpus-first enumeration is
the champion, and "learning to allocate" still earns nothing over
"knowing to grow the corpus first".  The guide seam now honestly
reaches the corpus — what it hasn't yet produced is a regime where
adaptivity beats the right fixed order.

## 4. Interfaces shipped

- `CORPUS_ARMS` — `("workload_gen", "gap_gen")`.
- `GuideArena(gen_cap=, gen_rng=, corpus=, vocab=, supported=)` —
  corpus arms; `_ingest` (scope rotation) / `_regrow_pools` /
  `_draw_workload` / `_draw_gap`; per-arm `gen_left` budgets;
  `gap_gen` target tracking over `no-instance` adjudications.
- `GuideObs.corpus_size` / `scope_epoch`; verdict rows scoped by
  the current `corpus_hash`.
- `summary` gains `generated`, `corpus_size`, `scope_epoch`;
  `--guide-corpus` CLI; `enum-corpus-first` control when armed.
- `tests/test_discovery_meta_game.py` — 15 new tests: scope
  rotation isolation (old-scope rows retrievable, new rows stamped
  post-mutation), corpus-arm legality (intake gates), gap
  re-adjudication, all miss paths (`gen-miss`, `gap-miss`,
  `no-target`), pool-regrowth novelty/dedup, enumeration order, the
  `--guide-corpus` CLI path.

Verification: `pytest tests/test_discovery_meta_game.py
tests/test_discovery_experiments.py -q` → 124 passed; `ruff check`,
`ruff format --check`, `ty check`, `radon_ratchet` clean on
`meta_game.py` (the radon run reports one pre-existing over-threshold
new function in `object_synthesis.py` — another agent's untracked
file, untouched here).
