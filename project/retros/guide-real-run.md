# The guide on the real corpus — rule vs policy, decided

Plan 0017 stage 5.  Stages 3–4 put the corpus arms on the board and
measured on the legacy ~16-case slice: `enum-corpus-first` won and the
trained arm policy's reward could not credit the corpus arms.  The open
question this stage answers on the *real* board (276 terms — bench +
models + the intake side-file; 158 probe cases — the pipeline's own
probe set): does the optimal schedule reduce to a rule —
`gap_gen`-when-targets, `workload_gen`-when-budget-loose — or is there
a non-obvious allocation a learned policy finds?  A second question
rides along: with the admission gauntlet finally wired to the arena,
how many *usable* objects does any guide actually produce?

Reproduce (~40–55 min CPU per seed; ~35 min is the 60-episode training
pass):

    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --guide-slice full --guide-episodes 60 --budget 90 \
        --guide-step 8 --guide-corpus 10 --plays-cap 8 \
        --guide-gauntlet --seed 0 --json /tmp/full_s0.json
    # seeds 0, 3, 5 at budget 90; seed 0 again at budget 80

Everything lives in `catopt_discovery/meta_game.py`; tests in
`tests/test_discovery_meta_game.py`.

## 1. What this run needed — the honest extensions

Four minimal additions in `meta_game.py`, no rewiring:

- **`--guide-slice full`** — the real arena.  `small` is the legacy
  slice unchanged (bench+models census, `_slice_cases` referee).
  `full` loads `intake.load_cases()` into the census/instance-search
  corpus (276 terms) and sets the firing probe to the pipeline's own
  probe set (`models + intake.probe_cases()` = 158 cases).
- **`--guide-gauntlet`** — after the comparison, every candidate the
  arena adjudicated `truth` or `unknown` (the only verdicts that can
  conceivably pass admission) is written to the arena's own store via
  `evidence.store_object` and faces `evidence.run_gauntlet`.  In
  `full` mode the gauntlet corpus *is* the arena's real corpus; in
  `small` mode it is `default_gauntlet_corpus`.  Reports
  `usable: yes/no` per guide plus the failing-stage census.  A shared
  memo dedups objects several guides adjudicated (a stored key's
  gauntlet outcome is corpus-deterministic).
- **`"trace"` in every summary** — the allocation sequence: arm,
  draws, batch score, and the `gap_gen` targets live at choose time.
  A run's *schedule* is now reportable, not only its tallies.
- **Per-episode arena seeds in `_pretrain_arm_guide`** —
  `make_arena(seed)`; training episodes roll `args.seed + 1 + ep`
  arenas while eval uses `args.seed`.  Previously every training
  episode played the *identically-seeded* board it was later scored
  on — the eval arena was literally inside the training set.
- `_cand_arm` — minting-arm attribution for the admission tally.

## 2. The board on the real corpus

Static pool: 58 proposals + 8 `build` plays + 20 corpus draws (10 per
arm) ≈ 86–90 draws, growing a few candidates per workload ingestion.
At `budget=90` the drawable space ≈ the budget; at `budget=80` there
is real allocation pressure (~10 draws of slack).  `gap_gen` is again
the monster arm — but with a twist measured below.

Per-episode cost on the real board ≈ 30–45 s (corpus draws dominate:
witness synthesis, the five intake gates, scope rotation, pool
regrowth over ~300 terms, re-adjudication).  60 training episodes ≈
35 min per seed — the run is feasible but no longer cheap.

## 3. Results — yield per oracle call (tf/call)

Budget 90, step 8, gen_cap 10, plays_cap 8, 60 training episodes:

| guide | seed 0 | seed 3 | seed 5 | mean |
|---|---|---|---|---|
| enumeration (corpus last) | .370 | .385 | .370 | .375 |
| uniform | .382 | .283 | .370 | .345 |
| learned (cold) | .283 | .356 | .268 | .302 |
| **learned-trained (60 ep)** | .277 | **.393** | .345 | .338 |
| enum-corpus-first | .357 | .364 | .377 | .366 |

Budget 80, seed 0 (40 episodes):

| guide | yield_tf | vs enum |
|---|---|---|
| enumeration | .304 | 1.00× |
| uniform | .447 | **1.47×** |
| learned | .400 | 1.31× |
| learned-trained | .244 | 0.80× |
| enum-corpus-first | .392 | **1.29×** |

## 4. The arm-visit pattern — the trained guide *does* learn the rule

`learned-trained`'s trace, seeds 0/3/5 at b=90 and s0 at b=80:

- **`gap_gen` whenever targets are live: 12/12 rounds.**  Every round
  the observation showed a live gap target, the frozen policy spent
  its allocation on `gap_gen` — the same behaviour `enum-corpus-first`
  exhibits (3/3 live rounds ×4 runs).  Enumeration-the-control had
  13–15 live-target rounds per run and fired gap only 1–2 of them —
  the fixed order cannot act on targets until the statics drain;
  the trained guide's edge over it is exactly those rounds.
- **`workload_gen` late.**  At seeds 3/5 it drew workload at rounds
  15–17 — after the statics drained (the "budget-loose" position).
  At seed 0 and b=80 it drew it mid-game instead.
- **Best statics first.**  `census-naturality` — the pool's known
  winner — opens the trace at all three b=90 seeds (score ~9).

So the trained schedule IS the hypothesized rule: fire `gap_gen` the
instant a target exists, keep `workload_gen` for when the board is
drained or the budget is loose, take the high-yield static arms
first.  Where it *diverges* is `build` — and the divergence is
instructive.

## 5. The wrinkle the rule needed — build-minted targets don't witness

`gap_gen`'s conversion depends on *which arm minted the target*.
`_draw_gap` witnesses the oldest untargeted `no-instance` candidate;
replaying the arena shows the build arm's constructions (random
5-op chains over `conv1d`/`pixel_shuffle`/`nan_to_num` attr metvars)
are un-synthesizable — `gen_cases_for` cannot mint a valid term
matching them, so every build-minted target ends in `gap-miss`.
Generator-minted `no-instance` candidates (naturality/grammar
schemas) synthesize fine: ~90–100 % conversion.

The correlation is exact across the four measured traces:

| run | build position | next gap batch |
|---|---|---|
| s0 b90 | r5 (before gap spent) | r6: drew 7, score 0.00 — all misses |
| s5 b90 | r8 | r9: drew 2, score 0.00 |
| s0 b80 | r6 | r7: drew 6, score 0.00 |
| s3 b90 | r11 (**after** gap spent) | no misses — gap went 10/10 |

The trained guide loses to the fixed order exactly on the runs where
it drew `build` early: its own plays minted unwitnessable targets,
which then ate the gap draws.  `enum-corpus-first` never trips this —
`build` is last in `ENUMERATION_ORDER`, so by the time it could play,
the synthesizable generator targets are already converted.  The
refined rule: **proposal arms mint targets → `gap_gen` converts them
→ `workload_gen` only when loose → `build` is a pure tax** (8 plays;
0 true in 18 of the 19 arm tallies that drew it, 1 true in the
b=80 enumeration run — and its targets are unwitnessable besides).

## 6. Rule vs policy — the verdict

**The optimum is a rule, and the trained policy converged to it.**
The learned schedule is not a different allocation: it is the fixed
schedule, learned — gap-when-live (12/12), workload-when-loose,
best-first.  Its one systematic deviation (not suppressing `build`)
costs it the margin: the runs where `build` came early are exactly
the runs it loses (s0 .75×, b80 .80×, s5 .93×); the one run where
`build` was safely last it *beat* the champion (s3 .393 vs .364 —
its gap-per-target interleave also landed `sub a (neg b) = add a b`
as the best find).  There is no non-obvious schedule hiding in this
board: the reward-visible degrees of freedom are "when does gap
fire" (answer: immediately) and "what mints targets" (answer:
generators, not builds), and the fixed corpus-first order already
embodies both.

Why the policy still can't *reliably* match the rule: the reward
structure is unchanged from the stage-4 diagnosis — `workload_gen`
draws still score exactly 0 (payoff is indirect), the credit for a
`gap_gen` hit lands on the gap allocation rather than the proposal
draws that minted the target, and `build` draws score 0 *immediately*
while poisoning the target pool — a delayed cost per-allocation
REINFORCE cannot see.  Training curves stay flat-to-mildly-rising at
60 episodes on the full board too (decade means s0 .323→.383,
s3 .352→.378, s5 .283→.380): the gradient exists, the signal doesn't.
`first shippable at draw 3` for the trained guides at all three seeds
(vs 13–77 for the enums) is the one place the learned schedule's
earliness shows up as real gain — it reaches the first ship earlier
because it fires gap at first opportunity.

## 7. The admission tally — `usable: yes` is 0 for every guide

The harder measurement this run adds: `--guide-gauntlet` stored every
reachable candidate (`truth`/`unknown` verdicts — the rest cannot
clear the truth gate on the same corpus) and ran the full admission
gauntlet on the real corpus.

| guide | tested (objects) | usable: yes |
|---|---|---|
| enumeration | 32–41 | **0** |
| uniform | 33–41 | **0** |
| learned | 29–46 | **0** |
| learned-trained | 31–44 | **0** |
| enum-corpus-first | 40–42 | **0** |

Refusal census across all 20 guide-runs: `truth` ×612,
`novelty` ×96, `full-data` ×34, `typed-pay` ×20.  Per run that's
roughly `truth` ×23–37, `novelty` ×2–5, `full-data` ×1–3,
`typed-pay` ×0–2 — three distinct honest negatives inside it:

- **`gap_gen` finds are conditional, not usable.**  Of the witnessed
  objects per run, ~9 fail the truth gate — `numeric=None` (the real
  corpus has no instance: the witness only exists on the synthesized
  workload) or `view=conditional` (the guarded sweep finds
  counterexamples: 69 equal / 8 unequal, 80 equal / 38 unequal…).
  About one per run passes truth+novelty then fails `typed-pay`
  (`fires=0` — a true law that fires nowhere real).  One at s3 also
  stood down at `full-data`.  The stage-3 caveat is now the
  measured fact: arena-`SHIP` means "true+firing+paying *on a corpus
  that contains the witness*", and admission requires truth on the
  real corpus or a written guard.  The candidates gap_gen mints are
  *conditional equalities witnessed into evidence* — exactly the
  objects that would need a `cond` to be declared.
- **`novelty` ×~5** is a fixed set: the pool's own best candidates
  (`census:mul_select` et al.) are shipped-law spellings —
  unclaimable by construction.
- **`full-data` ×1–3**: candidates carrying procedural hooks
  (the softmax fold's `check`/`derive`, build-minted
  `_bool_attr_check`/`_attr_bridge`) store flagged and stand down at
  stage 2 — the declared-object codec cannot carry them.

So the "did it find something never found" tally on the real corpus
is honestly **zero for every policy** — not because nothing was found
(gap_gen's witnesses are real measured evidence for real conditional
equalities) but because *nothing found this run survives declaration*.
The board's reward optimizes "pays on a grown corpus"; admission asks
"is a law".  That gap is now measured, per arm, per guide.

## 8. What this changes and what it doesn't

- The stage-3/4 verdict survives the real corpus, in sharper form:
  the optimum on this board is **fixed-decidable** — a static rule,
  not a learned policy.  The trained guide is a noisy approximation
  of it, occasionally lucky (s3 1.02×), losing on the mean (.338 vs
  .366 for the champion at b=90; .244 vs .392 at b=80).
- The new precision: the rule isn't just "corpus arms first" — it's
  **generator targets → gap converts → workload last → build never**.
  `build`'s role in the inventory is now measured dead weight on this
  board (0 true everywhere; its targets are unwitnessable).
- The admission layer separates two questions the yield table
  conflated: *yield* (arena-ship per call) is a schedule question —
  the rule wins; *usable objects* is an object-language question —
  0 for everyone, gated by conditional truth, not by allocation.
  A guide that maximized yield still mints zero usable objects; the
  missing move is guarding the conditional candidates (`cond`/`dspec`
  into the declared object), not scheduling differently.

## 9. Interfaces shipped

- `--guide-slice {small,full}` — the real-corpus arena
  (`_guide_corpus`); `--guide-gauntlet` + `_run_admissions` +
  `_admit_arena_candidates` + `_gauntlet_corpus_for` — the
  `usable: yes` pass; `_print_admissions` in the report.
- `run_guide` summaries gain `"trace"` (arm, draws, score,
  live gap targets per allocation).
- `_pretrain_arm_guide` rolls per-episode arena seeds
  (`make_arena(seed)`); eval arenas are boards the trainer never
  played.
- `GuideArena._cand_arm` — minting-arm attribution.
- No signature changes to `compare_guides`, `train_guide`,
  `Guide`, or `GuideArena`'s constructor; `small` mode is
  byte-compatible with stage 4.

Verification: `pytest tests/test_discovery_meta_game.py -q` →
48 passed; `ruff check`, `ruff format --check`, `ty check` clean on
`meta_game.py`.  Radon: the two functions the pass touches are back
at their pre-change complexity (the ratchet still reports the
pre-existing `catopt_core.typing._infer_op_shape` regression — a
sibling agent's uncommitted work, untouched here).
