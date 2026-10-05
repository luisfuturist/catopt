# Retro — corpus round 3: the inert guards start firing

Date: 2026-10-05
Context: `project/retros/auto-cond.md` — the auto-cond constructor
minted **19** verified guards that fired **0 times** on the 276-term
corpus.  The guards are correct; the corpus was too small to contain
their sites.  The retro's own words: "the guard is real, the *region*
is empty in practice."  This round grows the corpus to fill those
regions — 51 new `nn.*`/compound workloads targeting the transformer,
normalisation, conv/attention, recurrent/scan and view-identity
spellings the census lacked.

Owned: `catopt_discovery/intake.py` (the workload registry + new
compound classes), `tests/test_intake_defects.py` (the round-3
regressions), this retro.  No lowering defect needed a
`torch_bridge.py` change this round — every new workload exported,
bound and fp64-verified against the boundary as shipped.

## Corpus growth

| metric | before | after |
|---|---|---|
| workloads | 166 | **217** |
| ingested (fp64-verified) | 114 | **164** |
| census-only | 52 | 53 |
| verify-failed | 0 | **0** |
| rejected | 2 | 2 |
| census terms | 276 | **327** |
| op nodes | 1586 | **2135** |
| op-tuples | 588 | **708** |
| shapes | 1084 | **1504** |

The two `census-only` additions are honest binding gaps, not
defects: `BroadcastTensors` (`broadcast_tensors`, already in the
backlog via `nn.MSELoss`/`nn.L1Loss`) and — before the fix — a
`SlidingWindowMask` that spelled the band complement as `~band`
(`bitwise_not`, unbound).  Rewriting it as `band.logical_not()`
(`logical_not`, bound) made it ingest; that is the one workload
shaped *by* the boundary rather than by the model.

### New op spellings (15)

`erf`, `erfc`, `erfinv`, `sqrt`, `relu6`, `fmin`, `minimum`,
`clamp_max`, `isclose`, `le`, `ge`, `ne`, `all`, `inv`,
`scatter_reduce` — each bound and fp64-verified through a round-3
workload.

Two probes *decomposed* rather than emitting their op: `F.softsign`
(`abs`/`add`/`div`) and `torch.trace(x @ x)` (`diag_sum`).  Both stay
in the registry as real models; the census simply spells them with
ops it already had.

## The yield lever — view-identity elementwise spellings

The auto-cond retro's inert guards separate `elementwise(view(u))`
from `elementwise(u)`.  The pre-round-3 corpus held the *patterns*
(8 `mul(unsqueeze…)`, 21 `mul(slice…)`, 3 `mul(transpose…)` sites)
but none where the view was a **semantic no-op**, so the guard's
region stayed empty.  Round 3 supplies the no-op spellings directly:

| workload | spelling | guard it feeds |
|---|---|---|
| `BroadcastPadLeft` | `u.unsqueeze(0) * v`, `u` a leaf | `mul_unsqueeze_l_id` |
| `BroadcastPadRight` | `u * v.unsqueeze(0)` | `mul_unsqueeze_r_id` |
| `BroadcastPadSub` | `u.unsqueeze(0) - v` | `sub_unsqueeze_l_id` |
| `NoopTransposeScale` | `u.transpose(-1, -2) * v` on `(C,1,1)` | `mul_transpose_l_id` |
| `NoopTransposeAdd` | `u.transpose(-1, -2) + v` on `(C,1,1)` | `add_transpose_l_id` |
| `InertReshapeScale` | `u.reshape(1, h, w) * v` | `mul_reshape_l_id` |
| `SingleChunkScale` | `u.chunk(1, -1)[0] * v` | `mul_chunk_l_id` |
| `SquaredDistance` | `d * d` (`mul(t, t)`) | `FALSE_mul_factor` |

Shapes matter: `mul_unsqueeze_l_id`'s guard is
`bcast-eq(unsq-out(U,d), V, U, V) ∧ bcast-into(U, unsq-out(U,d))`,
which holds for `u:(H,W)`, `v:(1,H,W)`, `d=0` — a front-pad
broadcast — and *not* for the `(B,T)`-into-`(B,1,T)` pad the corpus
already had.  `InertReshapeScale` was first written reshaping *down*
(`(H,1,W)→(H,W)`), which fails `bcast-eq`; reshaping *up* (inserting
a leading 1) is the broadcast-inert direction.

### Guard firing — the retro's exact guards

Reconstructed from the `auto-cond.md` table and evaluated on the
pre-round-3 corpus vs the grown one (`real_matches` + the guarded
sweep, seeded):

| guard | base | grown |
|---|---|---|
| `mul_unsqueeze_l_id` | 0 | **1** ← now fires |
| `mul_unsqueeze_r_id` | 0 | **1** ← now fires |
| `sub_unsqueeze_l_id` | 1 | 5 |
| `mul_transpose_l_id` | 0 | **1** ← now fires |
| `add_transpose_l_id` | 0 | **1** ← now fires |
| `mul_reshape_l_id` | 0 | **1** ← now fires |
| `mul_chunk_l_id` | 0 | **1** ← now fires |
| `FALSE_mul_factor` | 2 | 7 |
| `mul_slice_l_id` | 0 | 0 (see below) |
| guards firing | 2/9 | **8/9** |

Six previously-inert guards now fire; two already-firing regions
grew.  `mul_slice_l_id` stays empty — its guard needs
`start == 0 ∧ end == size(dim)`, i.e. a *full-extent* slice, and
`torch.export` folds every full-extent slice to `alias`, leaving no
`slice` node to match.  Its 21 corpus matches are all partial slices
the guard correctly declines.  Hitting it needs a symbolic-extent
slice; that is a future corpus question, not a missing predicate.

## Pipeline delta

`python -m catopt_discovery.intake` (writes the side-file census,
then the baseline-vs-enlarged mirror):

```
corpus: 110 -> 327 terms, 211 -> 708 op-tuples
proposals: 54 -> 79
shippable: 0 -> 0
```

**New proposals: 25** (the `mixed:`-family view identities the larger
corpus surfaces).  The shared proposals whose firing/paid moved:

| proposal | match | fires | paid |
|---|---|---|---|
| `mixed:mul_transpose_l_id` | 2→9 | 0→7 | **0→5** |
| `mixed:sub_unsqueeze_l_id` | 1→5 | 1→5 | **0→4** |
| `mixed:mul_unsqueeze_l_id` | 7→10 | 7→10 | **1→3** |
| `mixed:mul_unsqueeze_r_id` | 4→6 | 4→7 | **1→2** |
| `mixed:mul_reshape_l_id` | 2→4 | 0→1 | **0→1** |
| `mixed:mul_chunk_l_id` | 2→3 | 1→2 | **0→1** |
| `reshape_transpose` | 31→95 | 23→93 | 0→0 |

Plus two new proposals that already pay on landing:
`mixed:add_transpose_l_id` (match=2, fires=2, paid=1) and
`mixed:add_unsqueeze_r_id` (match=1, fires=1, paid=1).

**Read:** the corpus growth converted six inert guards into *paying*
ones (`paid` is the typed-pay gate — a merged fire that extracts
cheaper somewhere).  The `mul_transpose_l_id` jump (0→5) is the
single biggest move, and it is exactly the guard the retro listed as
"fires=0".  `shippable` stays **0**: every moved proposal is still
refused by the *view oracle* ("no single mechanical feature
separates" / `w:v_commutes_view`), which is `intake-round-2`'s
finding unchanged — the next yield gate is a view/index oracle, not
another intake round.  What moved here is the *conditional→paying*
half, not the shippable half.

## Defects found

**None this round.**  0 verify-failed across 51 new workloads.  The
round-1/2 lowering-defect classes (copy_ threading, dropped dtype
casts, the `chunk`/`split` implicit `dim`, `take_along_dim`'s
arg-2 `dim`, the int/float leaf and attr spellings) all stayed fixed.
The only boundary interaction was `~band` → `logical_not`, a
*missing binding* surfaced by a new spelling, not a lowering
defect — and it is one the workload was rewritten to avoid rather
than a bridge change.

## Load-bearing tests

`tests/test_intake_defects.py` round 5:

- `test_round3_view_identity_workloads_verify` — parametrized over
  the eight view-identity spellings; each exports, lowers and
  fp64-verifies *and* carries its expected view op under the
  elementwise op.
- `test_round3_absent_ops_are_bound` — parametrized over the ten
  new-op workloads; each new op is present in the exported term,
  bound, and fp64-verified.
- `test_round3_transpose_noop_guard_now_fires` /
  `test_round3_chunk_noop_guard_now_fires` — pin the yield lever:
  the retro's `axes-noop` and `attr-eq(A_chunks,1)` guards accept a
  real round-3 site (`region.accepted ≥ 1`, `unequal == 0`,
  `rhs_err == 0`).  These fail if the spelling stops being a no-op
  or the workload is dropped.

`test_discovery_intake.py`'s eager-build gate (every registry thunk
runs) now covers 217 workloads and stays green.

## Limits

- **Measured, not proven.**  The grown corpus fires these guards on
  the sites it contains; a guard region outside both enumeration
  windows still stands until a wider sweep reaches it — the same
  blind spot every guarded law carries.
- **The side-file is generated.**  `tools/intake_corpus.json` +
  `intake_tensors.pt` are gitignored artifacts; the corpus growth
  persists by *re-running* `python -m catopt_discovery.intake`, not
  by committing a blob.
- **Two spellings declined to appear.**  `mul_slice_l_id` needs a
  surviving full-extent slice (export folds it to `alias`); the
  `_w` "wrap" guards need a third operand the round-3 models do not
  spell.  Both are corpus questions, listed for round 4.
- **Concurrent edit.**  While this round ran, a parallel change was
  editing `catopt_core.laws.cond` (new predicates: `axis-align-eq`,
  `bcast-dim-inv`, `select-out`, `slice-out`, `chunk-out`,
  `transpose-out`) — additions only, no existing predicate changed,
  so the guard measurement above (which uses only pre-existing
  predicates) is unaffected.  The pipeline-delta numbers are a
  snapshot from before that edit landed.
