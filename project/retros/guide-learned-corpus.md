# Trained learned guide under the corpus regime — a second honest negative

Plan 0017 stage 4.  The stage-3 retro (`guide-corpus-arms.md`) put
the corpus arms on the board and found `enum-corpus-first` the
champion — but the learned arm policy had still only ever trained
*online*, ~`budget/step` reward observations per run (~7 at b=40),
nowhere near enough to learn an arm-quality table.  This stage gives
the learned guide a fair shot: real whole-game training episodes,
the observation fixes the corpus regime needs, and a re-measurement
— which turned up two measurement bugs whose correction *revises
the stage-3 table itself*.

Reproduce (each run ~3.5–4 min CPU; ~2.5 min is the training pass):

    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --guide-episodes 40 --budget 40 --guide-step 6 \
        --guide-corpus 6 --plays-cap 12 --seed 0
    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --guide-episodes 80 --budget 40 --guide-step 6 \
        --guide-corpus 6 --plays-cap 12 --seed 0   # diagnostic

Everything lives in `catopt_discovery/meta_game.py`; tests in
`tests/test_discovery_meta_game.py`.

## 1. The training path — reuse, no new trainer

`--guide-train` pre-trains the *build arm's* construction policy;
the arm policy had no trainer.  `train_guide(make_arena, guide,
episodes, budget)` is the whole-game analog of `train`: each
episode is one `run_guide` over a fresh arena, and the guide's own
`update` — REINFORCE on the batch-mean verdict score, running-mean
baseline — is the optimizer.  40 episodes × ~7 allocations ≈ ~280
gradient steps, ~2.5 min.  The trained net joins the comparison as
**`learned-trained`**, evaluated *frozen* (`learn=False`: samples
from the policy, never updates) so measured yield is attributable
to the weights, not in-run adaptation.  `--guide-episodes N` arms
it; `guide_train_hist` (per-episode `yield_tf`) is in the result.

Two observation fixes were needed for a fair shot — the old
features could not express the decision:

- **Arm identity.**  `_arm_vec` gave `workload_gen` and `gap_gen`
  *identical* features whenever their tallies matched (same
  is-corpus flag, same stats) — the high-yield arm and the
  indirect-payoff arm were indistinguishable by construction.
  `_arm_vec` now appends a one-hot over `_ARM_NAMES`
  (ADIM 8 → 14).
- **Corpus growth.**  `GuideObs` carried `corpus_size` /
  `scope_epoch` but `_state_vec` never featurized them — the net
  could not condition on "the corpus just grew".  Added
  (SDIM 6 → 8).

## 2. The measurement fixes — the stage-3 table was contaminated

Two latent bugs surfaced mid-run; both changed the numbers.

**Arena isolation.**  `Referee` stored `terms`/`cases` *by alias*,
and `run_guide_experiment` passed one pair of slice lists to every
arena — so each guide's corpus ingestions leaked into the *next*
guide's board (`corpus_size` was cumulative across runs: 46 → 79 →
129 in one comparison).  With corpus arms armed, "dedup state is
per-run" was true but "corpus is per-run" was false: the
later-running guides (learned, corpus-first) played on corpora
grown by the earlier ones — pre-witnessed `no-instance` targets
included.  `Referee` now copies its slice lists.  Consequence: the
stage-3 `enum-corpus-first` numbers were inflated; under clean
isolation its b=40 margin over enumeration is ~1.01×, not 1.04–1.17×.
A cautionary corollary: a first 60-episode training run *looked*
like learning (curve 0.357 → 0.720) purely because episodes
inherited an accumulating shared corpus — the curve measured corpus
growth, not policy improvement.

**Generator crash containment.**  Two latent crashes, both in files
outside this stage's ownership, are now honest misses instead of
run aborts (a failed mint is a `gen-miss`, a failed synthesis a
`gap-miss` — the arms' own documented contract):

- `workload_gen.corpus_stats` keys attr tables on
  `tuple(sorted(attrs.items()))` — a generated term carrying an
  unhashable (list) attr value passes `valid_term` but poisons
  every later stats rebuild (`TypeError: unhashable type: 'list'`,
  hit at seed 3).  Ingestion now gates on `GuideArena._stats_safe`
  (all subterm attr values hashable); the draw misses rather than
  the arm dying mid-game.
- `catopt_core.typing._infer_op_shape` divides by conv `stride`
  unguarded (`// st` — `ZeroDivisionError`) when
  `gap_gen.synthesize` instantiates a zero-stride RHS.
  Owners should fix at the source; containment is in meta_game.

## 3. Re-measurement — the trained guide vs the fixed orders

Budget 40, step 6, gen_cap 6, plays_cap 12 — the pool≫budget
regime where allocation matters.  Yield = true+firing per oracle
call, ratio vs enumeration.  40 training episodes before each
seed's comparison.

| guide | seed 0 | seed 3 | seed 5 | mean |
|---|---|---|---|---|
| enumeration (corpus last) | .462 (1.00×) | .462 | .462 | .462 |
| uniform | .769 (**1.67×**) | .200 (0.43×) | .571 (**1.24×**) | .513 |
| learned (cold online) | .476 (1.03×) | .500 (1.08×) | .222 (0.48×) | .399 |
| **learned-trained (40 ep)** | .524 (**1.13×**) | .412 (0.89×) | .235 (0.51×) | .390 |
| enum-corpus-first | .467 (1.01×) | .467 (1.01×) | .467 (1.01×) | .467 |

`learned-trained` beats `enum-corpus-first` at one of three seeds;
its mean is *below* the cold-online learner's and far below
uniform's.  **Verdict: the trained arm policy does not beat the
fixed corpus-first order** — an honest negative, same convention
as stage 2.  Two caveats that cut both ways: under the isolation
fix the champion it had to beat is much weaker at b=40 than the
stage-3 table claimed, and single-run yields are noisy (uniform
spans 0.43–1.67× on the same arms).

## 4. The diagnostic — it is the reward structure, not the budget

Training curves (per-episode `yield_tf`, decade means):

- seed 0 ×40: .351 .399 .435 .344
- seed 3 ×40: .408 .388 .396 .411
- seed 5 ×40: .404 .483 .377 .316
- seed 0 ×80: .351 .429 .371 .293 .235 .228 .241 .203 → eval .238 (0.52×)

Flat at 40 episodes and *declining* at 80 — doubling the training
budget makes the policy measurably worse (it drifts onto
`algebraic-grammar`+`build`, abandoning the high-yield arm).
Episode count is not the bottleneck.

Neither, apparently, is the observation: with identity one-hots
the net *can* express "prefer `gap_gen`", and it does reach the
corpus arms (≥6 gap draws at every seed).  What the signal cannot
convey is the schedule's *structure*:

- **`workload_gen` is unlearnable under this reward.**  Every draw
  scores exactly 0 — its payoff is only indirect (pool regrowth,
  e.g. `census-mixed-view` re-emitting on a grown census).  Batch
  REINFORCE can only learn to *avoid* it; a policy that skips it
  loses the regrowth.  The corpus-first schedule contains a
  provably zero-reward arm — invisible to an immediate-reward
  learner, whatever the obs.
- **No cross-allocation credit.**  `gap_gen` pays only because
  earlier proposal draws minted `no-instance` targets; the batch
  reward credits the gap allocation, never the draws that made it
  live.  "Proposals → no-instance → gap converts" is a temporal
  chain this per-allocation signal cannot assemble.
- **Variance.**  ≤6 draws/batch, verdict scores heavy-tailed
  (0–25), ~7 allocations/episode — advantage estimates are noise-
  dominated at ~280 updates, which is exactly what flat-to-
  declining curves look like.

The stage-3 reading survives correction in a sharper form: the
yield on this board is *schedule-shaped* — reach the corpus arms
(uniform's upside when it does: 1.24–1.67×; its coin-flip downside
when it doesn't: 0.43×, seed 3 again burning 12 draws on `build`) —
and on the current reward definition a learned guide is not given
enough signal to find that shape.

## 5. Interfaces shipped

- `train_guide(make_arena, guide, episodes, budget)` → per-episode
  `yield_tf` hist; `LearnedGuide.frozen()` / `learn=`; `--guide-episodes`;
  `learned-trained` arm; `guide_train_hist` in the result; report
  training line.
- `LearnedGuide` obs fixes: `_ARM_NAMES` identity one-hot (ADIM 14),
  `corpus_size`/`scope_epoch` in `_state_vec` (SDIM 8).
- `Referee` copies `terms`/`cases` — per-run corpus isolation.
- `GuideArena._stats_safe` ingestion gate; corpus-draw crashes →
  `gen-miss`/`gap-miss`.
- Tests +4 (arm identity, frozen no-update, `train_guide`,
  `--guide-episodes` CLI) → 128 passing.

Verification: `pytest tests/test_discovery_meta_game.py
tests/test_discovery_experiments.py -q` → 128 passed; `ruff check`,
`ruff format --check`, `ty check`, `radon_ratchet` clean on
`meta_game.py` (radon also reports two over-baseline regressions in
`catopt_core` `ir.py`/`egraph/types.py` — another agent's
uncommitted work, untouched here).
