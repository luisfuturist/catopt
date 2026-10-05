# The guide seam — a policy over the generator inventory

Plan 0017 stage 2.  ADR 0004 §3 assigns `catopt_discovery.meta_game`
its real role: the player does not pick rewrites, it picks **which
generator invests compute where**.  This stage builds that seam —
the action space, the observation source, three baselines — and
measures it against the fixed enumeration.  The verdict, plainly:
**the seam works and no guide beats enumeration in-budget** — the
measured parity is the finding (ADR 0004 predicted the game, not
the winner).

Reproduce (each run ~5 s CPU):

    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --budget 60 --guide-step 8 --seed 0
    .venv/bin/python -m catopt_discovery.meta_game --guide \
        --budget 30 --guide-step 6 --seed 5

Everything lives in `catopt_discovery/meta_game.py`; tests in
`tests/test_discovery_meta_game.py`.

## 1. The honest action space

The investable inventory is the pipeline's own proposal machinery —
`generator_pools` splits `pipeline.propose`'s five sources into
per-generator queues, deduplicated under `propose`'s merged-pool
semantics (a candidate reachable from two generators queues on the
first), so `EnumerationGuide` replays the baseline schedule exactly:

| arm | source | pool on the real slice |
|---|---|---|
| `census-naturality` | `pipeline._census_naturality` | 1 |
| `census-mixed-view` | `pipeline._census_mixed_naturality` | 16 |
| `pattern-recognition` | `pipeline._pattern_recognition` | 1 |
| `shape-aware` | `shape_proposal.schemas` | 19 |
| `algebraic-grammar` | `proposal.schema_candidates` | 15 |
| `build` | the `BuildGame` construction player — one draw is one `_play_once` | `plays_cap` |

"What does 'invest compute in X' mean" resolved concretely: a draw
is **one proposal refereed** — the unit a generator consumes — and
the cost is measured in **oracle calls**, the same currency the
meta-game's yield tables already count (a `no-instance` /
`tautology` / `repeat` draw costs a draw but no call; the arm stats
attribute calls honestly).  Wall-clock is *not* the unit: it is
unmeasurable per-draw and unplayable by a policy; corpus slice is
not a per-generator knob either — the slice is the board, shared by
every arm.

**Deliberate non-actions** (the boundary, stated once): the workload
and gap generators mutate the *corpus*, and a mutated corpus
invalidates the evidence store's `(corpus_hash, rules_hash,
code_rev)` scope — every cached verdict would silently refer to a
different context.  A corpus-mutating action needs a re-scoped
observation mid-game, not another queue, so `workload_gen` /
`gap_gen` stay outside the seam.  The emit/admission path is a
reporting sink, not a generator — a guide allocates *search*
compute only.  The derivability and view oracles are referee
machinery the pipeline already runs per-candidate, not investable
arms (they have no proposal queue to draw from).

## 2. The seam

```python
class Guide:                       # the policy interface
    def choose(self, obs: GuideObs) -> Allocation | None
    def update(self, reward: float, verdicts: list[Verdict]) -> None
```

- `Allocation(generator, n)` — draw up to `n` proposals from an arm.
- `GuideObs` — budget/spent, per-arm `ArmStat` tallies (drawn,
  oracle calls, true/firing/new-t+f/shippable counts, best score),
  remaining queue lengths, and `verdicts`: **`evidence.latest_verdicts`
  rows for the arena's scope**.  The observation source is the
  store itself — `GuideArena` writes each adjudicated verdict with
  `evidence.record_run` (dedup artifacts — tautology, repeat,
  mint-error — are never recorded, exactly the same honesty the
  store demands of the pipeline) and the guide reads back what a
  later audit would.  A `Verdict` is adapted to the pipeline's
  `Evidence` schema by `_verdict_evidence`; fields the referee does
  not measure stay at defaults, so a stored `SHIP` means "cleared
  the referee's bar" and no more.
- `GuideArena.invest` draws, referees every proposal through the
  *shared* `Referee` (dedup is cross-arm — a candidate two
  generators emit is still one candidate), tallies, records.
- `run_guide` plays until the draw budget is spent; `compare_guides`
  runs guide *factories* on fresh arenas over the same pools.

Three baselines: `EnumerationGuide` (fixed `ENUMERATION_ORDER`,
drains each arm — the control arm, bit-for-bit the pipeline's
schedule), `RandomGuide` (uniform over live arms), `LearnedGuide`
(the existing `_PolicyNet` + REINFORCE machinery re-pointed at
arms — a `(state ⊕ arm-features) -> logit` net, `Categorical`
sampling, running-mean baseline, reward = mean verdict score of
the batch).  **No new trainer**: the build arm's player policy is
optionally pre-trained by the existing `train` (`--guide-train`);
the guide itself learns online across its own allocations.

## 3. Guide vs enumeration — measured

Bounded task: within `N` total proposals drawn from the inventory
(pool = 52 candidates + the build arm), what does each policy find?
Yield = true+firing candidates per oracle call, the meta-game's
standing currency.

### 3.1 Loose budget (60 draws ≥ 52-pool), seeds 0 and 3

| guide | draws | calls | tf | new tf | ship~ | yield_tf | vs enum |
|---|---|---|---|---|---|---|---|
| enumeration | 60 | 26 / 25 | 7 / 7 | 2 / 2 | 0 | .269 / .280 | 1.00× |
| uniform | 60 | 22 / 26 | 5 / 5 | 1 / 3 | 0 | .227 / .192 | 0.84× / 0.69× |
| learned | 60 | 27 / 17 | 7 / 6 | 2 / 1 | 0 | .259 / .353 | 0.96× / 1.26× |

### 3.2 Scarce budget (30 draws < 52-pool), seeds 0 and 5

| guide | calls | tf | new tf | yield_tf | vs enum |
|---|---|---|---|---|---|
| enumeration | 9 / 9 | 3 / 3 | 1 / 1 | .333 / .333 | 1.00× |
| uniform | 9 / 11 | 4 / 3 | 0 / 1 | .444 / .273 | 1.33× / 0.82× |
| learned | 8 / 13 | 3 / 3 | 1 / 1 | .375 / .231 | 1.12× / 0.69× |

### 3.3 Reading

**No guide beats enumeration consistently — the differences are
hit-rate noise, in both directions.**  The measured spread across
seeds is 0.69×–1.33× (uniform wins one seed, loses the other;
learned likewise): a single `new`+firing candidate in a 10-draw
allocation moves the yield by a third, and the policy sees ~8–10
reward observations per run — far too few to learn the arm-quality
table the runs actually contain (measured: `algebraic-grammar`
runs ~6/15 true, `shape-aware` ~3/19, `census-mixed-view` ~1/16,
`build` 0 true in ≤25 plays).

The honest structural finding underneath the noise:

- **At loose budget the game is nearly degenerate** — every
  enumeration arm drains regardless of order, so guides differ only
  in how many draws they burn on `build` (uniform: 25 of 60).
  Allocation matters only when compute is scarce.
- **At scarce budget the opportunity is real but small**: the arm
  table above says a guide that deprioritizes `census-mixed-view`
  would roughly double yield-per-draw early — but the information
  to learn that costs several allocations of the very budget being
  optimized.  On a 52-proposal pool, enumeration's fixed order is
  already near the achievable optimum; a guide earns its keep only
  on a *larger* inventory (an enlarged corpus, more generator
  families) where the pool-to-budget ratio is high.
- **The `build` arm is the weak generator** measured in stage-1
  terms: 0 true candidates in ≤25 plays per run, consistent with
  the meta-game retro's ~0.4% truth density.  A guide that could
  learn "stop feeding build" would save draws — the learned arm
  sampled it ~10 times anyway (the sparse reward punishes slowly).

## 4. What the seam is for, honestly

The negative is not a defect — it replicates the meta-game retro's
conclusion one level up: **enumeration is a strong baseline when
the pool is small and the referee is cheap**.  The seam's value is
that it is now *a real interface, measured*: `Guide`/`GuideObs`/
`GuideArena`/`Allocation` is where a future policy (a bandit over
arm yields, a trained meta-policy over a bigger inventory) plugs in
without touching the referee, and the evidence store is already the
observation — accumulation is free.

What stage 2 did *not* show, and the record should say plainly:
nothing yet rewards guiding.  The regime where it should pay —
pool ≫ budget — is exactly what corpus-generating arms
(`workload_gen`, `gap_gen`) would produce, and those are out of
the seam for the scope-key reason in §1.  Re-scoping observations
mid-game is the honest next blocker for a *useful* guide, not a
better learner.

## 5. Interfaces shipped

- `Allocation`, `ArmStat`, `GuideObs` — the action and observation
  records.
- `GuideArena` — pools + shared `Referee` + in-memory (or
  file-backed, `conn=`) evidence store; `invest`/`observation`/
  `summary`.
- `Guide`, `EnumerationGuide`, `RandomGuide`, `LearnedGuide` —
  the policy interface and the three baselines.
- `generator_pools`, `run_guide`, `compare_guides`,
  `run_guide_experiment`, `_print_guide_report`; CLI `--guide`,
  `--guide-step`, `--guide-build`/`--no-guide-build`,
  `--guide-train`.
- `tests/test_discovery_meta_game.py` — 28 tests over the seam on
  the tiny corpus (arena accounting, store round-trip, dedup
  honesty, build arm, all three guides, the comparison harness, the
  `--guide` CLI path).
