# Retro — corpus round 4: the wrap spellings ship

Date: 2026-10-05
Context: `project/retros/corpus-round-3.md`.  Round 3 fired six of
the auto-cond set's nine inert *left* view-identity guards and left
`mul_slice_l_id` plus the `_w` "wrap" family standing: "the `_w`
guards need a third operand the round-3 models do not spell."  Round 4
supplies those spellings — 37 new `nn.*`-style compound workloads
targeting the *wrap* region `f(view(u), v) -> view(f(u, v))`, its
right-operand mirrors, the shared-factor algebra laws, the
neg/exp/square grammar finds and the scalar-corner annihilators.

The headline is not the guard count this time: the corpus growth
**shipped 11 rules**.  `shippable: 0 -> 11` — the first proposals the
intake corpus has ever pushed through the *whole* gate stack
(conditional truth → view oracle → typed-pay → closure).

Owned: `catopt_discovery/intake.py` (the workload registry + new
compound classes + the `_r()` scalar fix), `tests/test_intake_defects.py`
(the round-4 regressions), this retro.  No `torch_bridge.py` change
was needed — every new workload exported, bound and fp64-verified
against the boundary as shipped (0 verify-failed).

## The still-inert guards — re-measured

Re-running the auto-cond mint over the current pool
(`generator_pools` — 70 candidates) and sweeping each minted guard's
LHS over the real corpus (`real_matches` + `_guarded_evals`,
`torch.manual_seed(0)`) gives 56 minted guards.  Before round 4, 37
were inert (accepted == 0); after, 7.

The seven that stay inert are **structurally** unreachable, not
corpus gaps — each is a boundary/canonicalisation fact, not a missing
model:

| guard | why the region is empty |
|---|---|
| `mul_slice_l_id` / `_r_id` | needs `start == 0 ∧ end == size(dim)`; `torch.export` folds every full-extent slice to `alias`, so no `slice` node survives |
| `add_getitem_l_w` / `eq_getitem_l_w` / `eq_getitem_l_id` / `census:add_getitem` | need `getitem(U)` with `leaf(U)`; `x[0]`/`x[:, 0]` export as `select`, and `getitem` only appears when a *tuple-returning* op (`var_mean`, `topk`, `sort`) is indexed — its operand is never a leaf |
| `grammar:mul_zero_left` | LHS is `mul(Const(0), M0)`; torch canonicalises `0 * x` to `mul(x, 0)`, so the left-zero form never reaches the boundary |

The spelling each *reachable* inert guard needs (the region the
round-4 models fill), read off the guard's accepted synth bindings:

| guard | required spelling |
|---|---|
| `mul_unsqueeze_l_w` | `u.unsqueeze(d) * v` with `v` **scalar or the same shape as `u`** (broadcast-trivial through the lift) |
| `mul_unsqueeze_r_w` | `v * u.unsqueeze(d)`, same `v` condition |
| `sub_unsqueeze_l_w` | `u.unsqueeze(d) - v`, same `v` condition |
| `census:sub_unsqueeze` | `u.unsqueeze(d) - v.unsqueeze(d)` (both lifted) |
| `mul_chunk_r_w` | `v * chunk(u, c, d)[i]` with `v` scalar (c ≥ 2) or `v` broadcastable with the *whole* `u` |
| `mul_chunk_r_id` | `v * chunk(u, 1, d)[0]` — the one-chunk no-op, on the **right** |
| `add_transpose_r_w` | `v + u.transpose(d0, d1)` with `v` scalar / rank ≤ 1 / same shape |
| `add_transpose_r_id` | `v + u.transpose(d0, d1)` with the swap a no-op on `u` |
| `select_add` / `select_sub` | `a[:, i] + b[:, i]` (shared select, aligned axis) |
| `factor_left` / `factor_right` / `factor_sub_left` / `factor_sub_right` | `a*b ± a*c` (shared left/right factor) |
| `FALSE_mul_factor` | `d*d + d*e` — the `term-eq(M0, M1)` self-product region |
| `reshape_reshape` | `x.reshape(a, b, c).reshape(a, b*c)` |
| `neg_add`, `sub_neg`, `mul_neg_left`, `exp_add`, `square_neg`, `grammar:neg_distribute`, `grammar:sigmoid_neg`, `grammar:pow_one`, `grammar:div_add`, `grammar:exp_distribute`, `grammar:square_mul`, `grammar:sub_add_factor` | the literal grammar spellings (`(-x)+(-y)`, `x-(-y)`, `(-x)*y`, `exp(x)*exp(y)`, `square(-x)`, `-(x+y)`, `sigmoid(-x)`, `x**1`, `(x+y)/z`, `exp(x+y)`, `square(x*y)`, `(x-y)-z`) |
| `mul_zero`, `sub_self`, `add_inv`, `div_self` | `s*0`, `s-s`, `s+(-s)`, `s/s` on a **rank-0** operand |

The shape relationship is the whole story: the wrap guards accept
exactly when the second operand broadcasts *trivially* through the
view.  Round 3's `mul(unsqueeze…)` sites all carried a
differently-shaped `v` (a gate/stack) and correctly declined; the
round-4 sites carry a scalar or a same-shape `v`.

## Corpus growth

| metric | before | after |
|---|---|---|
| workloads | 217 | **254** |
| ingested (fp64-verified) | 164 | **201** |
| census-only | 53 | 53 |
| verify-failed | 0 | **0** |
| rejected | 2 | 2 |
| census terms | 327 | **364** |
| op-tuples | 708 | **725** |
| shapes | 1504 | **1533** |
| op nodes | 2135 | **2218** |

**No new ops.**  Round 4 introduced *no* previously-absent op — every
spelling is built from the census's existing vocabulary
(`unsqueeze`/`chunk`/`transpose`/`select`/`square`/`exp`/`neg`/`pow`/
`div`/`sigmoid`/`reshape`/`mul`/`add`/`sub`).  This round is pure
*spelling*, the round-3 lever taken to the mirror and the algebra.

### New workload families (37)

- **unsqueeze-wrap (gated broadcast)** — `ChannelGateBroadcast`,
  `LiftedScalarScale`, `HeadGateBroadcast`, `LiftedScalarScaleRight`,
  `LiftedScalarCenter`, `ContrastiveCenter`, `PairwiseSubLift`.
- **chunked-projection mirrors** — `ScaledChunkProjection`,
  `SingleChunkGate`, `ChunkHalfScale`.
- **transposed-add mirrors** — `NoopTransposeResidual`,
  `ScalarTransposeBias`, `Rank3TransposeAdd`.
- **select naturality** — `SelectGateSum`, `SelectGateDiff`.
- **shared-factor algebra** — `SharedFactorMixture`,
  `SharedFactorContrast`, `SharedFactorMixtureRight`,
  `SharedFactorContrastRight`, `QuadraticFeature`.
- **grammar spellings** — `DoubleReshapeHead`, `NegatedSum`,
  `NegDistributeHead`, `SubNegBias`, `NegatedScale`, `ExpProductHead`,
  `ExpSumHead`, `SquareNegHead`, `SquareMulHead`, `SigmoidNegGate`,
  `PowOneHead`, `SubAddFactorHead`, `DivAddHead`.
- **scalar-corner annihilators** — `ScalarAnnihilator`,
  `ScalarSelfCancel`, `ScalarSelfRatio`, `ScalarInverseSum`.

## Guard firing — the delta

Same pool (70 candidates), same 56 minted guards, seeded
(`torch.manual_seed(0)`), swept on the round-3 corpus vs the round-4
one:

| metric | round-3 corpus | round-4 corpus |
|---|---|---|
| minted guards | 56 | 56 |
| **firing** (accepted ≥ 1) | 19 | **49** |
| inert (accepted == 0) | 37 | **7** |
| total accepted sites | 138 | **189** |
| total equal sites | 138 | **189** |

**30 guards newly fire** (accepted 0 → ≥1), every one with
`unequal == 0 ∧ rhs_err == 0` on its accepted real sites:

`census:sub_unsqueeze`, `exp_add`, `factor_left`, `factor_right`,
`factor_sub_left`, `factor_sub_right`, `grammar:FALSE_mul_factor`,
`grammar:add_inv`, `grammar:exp_distribute`, `grammar:neg_distribute`,
`grammar:pow_one`, `grammar:sigmoid_neg`, `grammar:square_mul`,
`grammar:sub_add_factor`, `mixed:add_transpose_r_id`,
`mixed:add_transpose_r_w`, `mixed:mul_chunk_r_id`,
`mixed:mul_chunk_r_w`, `mixed:mul_unsqueeze_l_w`,
`mixed:mul_unsqueeze_r_w`, `mixed:sub_unsqueeze_l_w`, `mul_neg_left`,
`mul_zero`, `neg_add`, `reshape_reshape`, `select_add`, `select_sub`,
`square_neg`, `sub_neg`, `sub_self`.

The `_w` wrap family is the biggest move: `mul_unsqueeze_l_w`
(0 → 2 accepted), `mul_unsqueeze_r_w` (0 → 1), `sub_unsqueeze_l_w`
(0 → 2), `mul_chunk_r_w` (0 → 3), `add_transpose_r_w` (0 → 3),
`census:sub_unsqueeze` (0 → 1).  Round 3's six already-firing guards
kept firing and grew their match counts (`mul_unsqueeze_l_id` 10 → 12,
`sub_unsqueeze_l_id` 5 → 8, `mul_unsqueeze_r_id` 6 → 8);
`mul_transpose_l_id`'s match count held at 9.

## Pipeline delta — 11 rules ship

`python -m catopt_discovery.intake`:

```
corpus: 110 -> 364 terms, 211 -> 725 op-tuples
proposals: 54 -> 79
shippable: 0 -> 11
```

**New shippable rules (11):** `sub_self`, `grammar:pow_one`,
`reshape_reshape`, `square_neg`, `factor_left`, `factor_right`,
`factor_sub_left`, `factor_sub_right`, `neg_add`, `exp_add`,
`sub_neg`.

This is the round's real yield.  Round 3 proved yield was
*spelling-gated* but still moved only the conditional→paying half
(`shippable` stayed 0 — every moved proposal was refused by the view
oracle).  Round 4's spellings are *unconditional* equalities with a
mechanical shape story, so they clear the view oracle too and reach
`ship: Y`.

Notable shared-proposal moves (match → fires, paid):

| proposal | match | fires | paid |
|---|---|---|---|
| `mixed:sub_unsqueeze_l_id` | 1→8 | 1→8 | **0→7** |
| `mixed:mul_transpose_l_id` | 2→9 | 0→7 | **0→5** |
| `mixed:mul_unsqueeze_l_id` | 7→12 | 7→12 | **1→5** |
| `mixed:mul_unsqueeze_r_id` | 4→8 | 4→9 | **1→4** |
| `mixed:add_transpose_r_id` | (new) | 7 | **3** |
| `mixed:mul_chunk_r_id` | (new) | 6 | **3** |
| `factor_left` / `factor_sub_left` | 0→2 / 0→1 | 0→2 / 0→1 | **0→2 / 0→1** |
| `reshape_transpose` | 31→95 | 23→93 | 0→0 |

`factor_left/right`, `factor_sub_left/right`, `exp_add`, `neg_add`,
`sub_neg`, `square_neg`, `reshape_reshape`, `grammar:pow_one`,
`sub_self` all go `match 0 → ≥1, fires 0 → ≥1, paid 0 → ≥1, ship: Y`
in one step — the shared-factor and grammar spellings were simply
absent from the corpus, and their first real site both fires and pays.

The `*_w` proposals still do not ship (`w:v_commutes_view`,
`id:out_shape_eq`) — the wrap equality needs a *value-level* oracle,
which is `intake-round-2`'s finding unchanged.  What shipped is the
unconditional half.

## Defects found

**No lowering defect.**  0 verify-failed across 37 new workloads; no
`torch_bridge.py` change.  The only code change outside the workload
registry is `_r()`: the intake's example-input helper was
`torch.randn(*shape)`, which raised on the empty shape the
scalar-corner guards need.  It is now `torch.randn(tuple(shape))`, so
`_r()` yields a rank-0 tensor — the `rank(A) == 0` binding the
`mul_zero`/`sub_self`/`add_inv`/`div_self` guards require.

Two **minting** observations, recorded not "fixed" (they are the
auto-cond constructor's, outside this round's ownership):

- `grammar:div_add` and `grammar:div_self` are not numerically-true
  laws: `(x+y)/z != x/z + y/z` in fp64 and `0/0 != 1`, so the mint
  correctly *refuses* them once the measured domain contains the bad
  sites (detail: `no declarable conjunction of <= 3 covers 330 equal /
  30 bad sites`).  Whether they mint at all is RNG-state sensitive —
  the unseeded round-3 sweep minted them vacuously (`cond=True`, no
  bad site drawn), the seeded one does not.  The pipeline still fires
  them (the grammar pool's own unguarded proposal) and both are
  refused later (`fires but never lowers cost` / `numeric oracle
  rejects`).
- The mint set is RNG-state sensitive at the margin: `mul_unsqueeze_l_id`'s
  region contains a value-sensitive real site (the ALiBi
  `pow(param, -0.25)` term NaNs on a negative draw), so its
  equal/unequal split flips per draw.  The seeded comparison above
  pins the numbers.

## Load-bearing tests

`tests/test_intake_defects.py` round 6:

- `test_round4_workloads_verify` — parametrized over all 37
  workloads; each exports, lowers, fp64-verifies *and* carries its
  expected ops.
- Eleven `test_round4_*_now_fires` tests pin the yield lever: the
  wrap / mirror / shared-factor / scalar-corner guards accept a real
  round-4 site (`accepted ≥ 1`, `equal ≥ 1`, `unequal == 0`,
  `rhs_err == 0`).  These fail if a spelling stops being
  broadcast-trivial or a workload is dropped.
- Three `test_round4_*` tests pin the *structural* reasons the last
  seven guards stay inert: a full-extent slice folds to `alias`
  (no `slice` node); `x[0]`/`x[:, 0]` export as `select` while
  `getitem` needs a tuple-returning op; `0 * x` canonicalises to
  `mul(x, 0)`.

`test_discovery_intake.py`'s eager-build gate now covers 254
workloads and stays green.

## Limits

- **Measured, not proven.**  The grown corpus fires these guards on
  the sites it contains; a guard region outside both enumeration
  windows still stands until a wider sweep reaches it.
- **The side-file is generated.**  `tools/intake_corpus.json` +
  `intake_tensors.pt` are gitignored artifacts; the growth persists by
  re-running `python -m catopt_discovery.intake`, not by committing a
  blob.
- **Seven guards are boundary facts, not corpus gaps.**
  `mul_slice_l_id`/`_r_id`, the four `getitem` guards and
  `mul_zero_left` cannot fire on any `torch.export` graph as the
  boundary spells it.  They are corpus questions only if the boundary
  ever emits a surviving full-extent `slice` or a `getitem` on a leaf.
- **`div_add`/`div_self` are false in fp64.**  Their minting is
  RNG-state sensitive and their firing never pays; a future
  grammar-pool pass should drop them.
- **The `_w` wrap proposals still refuse shipping.**  The value-level
  view oracle is the next gate, unchanged from round 2.
