# Retro: the selective cap — rescuing a starved guarded region

Date: 2026-10-06
Context: the attr-sweep retro's "remaining blind spot #1" —
"the cap still starves deep regions".  The guarded-region sweep
(`evidence._synth_sites` over `oracle._binding_envs`) enumerated a
fixed fair-order domain and truncated at the default window
(`oracle._MAX_INSTANCES = 360`).  The shipped `sdpa_fold_*` folds and
`rms_norm_fold` measured **0 accepted sites** at that cap, so the
gauntlet's truth gate refused them as *vacuous* ("a guard that accepts
nothing provable is not verified") — even though they are shipped,
sound laws whose guard regions do exist.  `object_synthesis.py`,
`cond.py`, `meta_game.py`, `laws/` were not touched.

## The starvation, characterized

For `sdpa_fold_addmul` (guard `axes-last2(K) ∧ softmax dim=-1 ∧
const-num(S)`) the accepted corner is a *multi-clause* region.  The
enumeration is the diagonal over `(viewed × attr)` bases: at round
`total` it yields one free-combo from each of bases `0..total`.  The
base whose guard admits a site is **base #27 of 324**, and the base
before it (`K=(4,)`, rank-1 — the guard needs rank ≥ 2) is pure
decline, so the corner is reached only after the bases ahead of it
have each spent a round.

| limit | sites | accepted | equal | first equal at |
|-------|-------|----------|-------|----------------|
| 360   | 360   | **0**    | 0     | —              |
| 1000  | 1000  | 14       | 2     | 793            |
| 2000  | 2000  | 41       | 9     | 793            |
| 4000  | 4000  | 129      | 29    | 793            |

The first equal site is at a **fixed** enumeration index — 793 for
`sdpa_fold_addmul` / `sdpa_fold_adddiv`, 469 for `sdpa_fold_add` —
independent of the cap.  So the region is not absent; the window
simply stops before it.  A bigger cap reaches it (that is option (a)),
but a flat larger default would pay for *every* rule.

The full shipped guarded set (36 rules) splits into three failure
modes, not one (`gqa_absorb_repeat` raises in the enumerator's
nested-view boundary — the oracle's pinned single-level-view edge —
and is outside this policy entirely):

- **Cap-starved** — the region exists past the cap: the three
  `sdpa_fold_add{,mul,div}` folds (index 469/793), and the `_drop`
  twins + `assoc_linear_bias*` (which need ~16000 — the extra
  `dropout` metavars push the corner much deeper).
- **Domain-gapped** — the guard needs a leaf value the bank never
  mints: `sdpa_fold_masked_fill*` (`const-cmp F < -1e30`, but `F` is
  a free leaf filled only with `Const(0.5)` / scalar `Var`s),
  `div_sqrt_to_rsqrt` (`const-cmp ONE == 1`), `pow_to_rsqrt`
  (`const-cmp P == -0.5`), `rms_norm_fold*` (`const-cmp P == 2`).
  No cap reaches these — a bigger window widens the *space*, not the
  *value bank*.
- **Shallow** — `softmax_fold`, `glu_fold`, `select_mul`, the
  `*_scale`/`*_matmul` folds: the window already accepts (and
  usually proves) sites.

## What changed

A **selective cap policy**, split across the two owned seams.

1. **`oracle.escalate_limit` + `oracle._GUARDED_CAP`** (the
   enumeration/limit surface).  `_GUARDED_CAP = 2000` — the measured
   depth at which the `sdpa_fold_*` corners appear (9 equal for
   `_addmul`).  `escalate_limit(limit, *, guarded, accepted)` returns
   the second-phase cap: it raises to `_GUARDED_CAP` **only** when the
   rule is guarded *and* the window accepted nothing — the starvation
   signature.  An unguarded rule has no guard region to starve; a
   window that already accepted a site has proved the region is
   non-empty (the cap is not what is biting).  The returned cap is
   never below `limit`.

2. **`evidence._guarded_truth`** (the sweep's limit wiring).  The
   synthesized sweep runs at the first-phase window, then consults
   `escalate_limit` and re-runs once at the ceiling when it escalates.
   `GuardedRegion` gained an `envs` field — the enumeration cost — so
   the report names the price per domain; `_region_detail` prints it.

The common case is untouched: only a guarded rule whose window
accepted *nothing* pays the second sweep.

## Measured

Per shipped guarded rule — accepted at 360 vs under the policy, and
the env cost (sites evaluated; the escalating rules pay both windows):

| rule | acc@360 | acc@policy | equal@policy | envs |
|------|---------|------------|--------------|------|
| `sdpa_fold_addmul`      | 0 | **41**  | 9  | 2360 |
| `sdpa_fold_adddiv`      | 0 | **41**  | 9  | 2360 |
| `sdpa_fold_add`         | 0 | **140** | 30 | 2360 |
| `glu_fold`              | 9 | 9       | 9  | 12   |
| `softmax_fold`          | 164 | 164   | 139| 360  |
| `select_mul`            | 81 | 81     | 57 | 198  |
| `naturality_scalar`     | 148 | 148   | 19 | 360  |
| `swiglu_fuse`           | 5 | 5       | 0  | 360  |
| `rms_norm_fold`         | 0 | 0       | 0  | 2360 |
| `sdpa_fold_masked_fill` | 0 | 0       | 0  | 2360 |
| `sdpa_fold_addmul_drop` | 0 | 0       | 0  | 2360 |
| `assoc_linear_bias`     | 0 | 0       | 0  | 2360 |

Across all 35 measurable guarded rules (`gqa_absorb_repeat` raises in
the enumerator's documented nested-view boundary and is skipped):

| metric | default cap | selective cap |
|--------|-------------|---------------|
| rules with a non-empty accepted region | 17 | **20** |
| total envs evaluated | 11820 | 44270 |
| wall time (guarded sweeps) | 2.7 s | 6.1 s |

**Delta: 3 shipped guarded rules** (`sdpa_fold_addmul`,
`sdpa_fold_adddiv`, `sdpa_fold_add`) go from `synth: 0 accepted` /
truth-refused-as-vacuous to a non-empty, contradiction-free region
with an equal site — they now clear stage 4 instead of being refused
for a cap artefact.  18 rules escalate; 15 of those are domain-gapped
or need a deeper cap, so their escalation is spent without a rescue
(bounded: ~32450 extra envs, ~4 s).

## Cost

The escalation is *not* free but it is *bounded and selective*: an
unguarded rule and a shallow guarded rule keep the 360 default
(`glu_fold` pays 12, `select_mul` 198 — the enumeration exhausts
before the cap).  The extra work is confined to the guarded rules
whose window is empty, and the cost is reported in the gauntlet's
truth line (`synth: 9eq/0ne/0rerr (41 accepted, 1959 declined, 2360
envs)`), so a reader sees the window that produced a verdict.

## Remaining blind spots

1. **The value bank is still finite.**  The policy widens the
   *window*; it does not widen the *values*.  `sdpa_fold_masked_fill*`
   (`F < -1e30`), `div_sqrt_to_rsqrt` (`ONE == 1`), `pow_to_rsqrt`
   (`P == -0.5`), `rms_norm_fold*` (`P == 2`) need leaf constants the
   free bank never mints — a separate fix (a `Const` bank keyed by the
   guard's `const-cmp` demands, or the guard's own `dspec`-style
   proposal).  Recorded here, not fixed.
2. **The `_drop` twins need a deeper cap** (~16000) than the ceiling.
   The extra `dropout` metavars multiply the free product and push the
   corner out; the ceiling is set for the *headline* folds, not them.
   A *targeted* (base-major) phase would reach them at ~7776 envs
   (measured: a depth-4 base-major probe finds 468 accepted for
   `sdpa_fold_addmul_drop`) — the natural next step, deliberately not
   taken here to keep the change one policy, not an enumeration
   rewrite.
3. **The trigger is `accepted == 0`, not `equal == 0`.**  A window
   that accepted sites but found no *equal* one (`swiglu_fuse`,
   `parallel_mul_fuse`: 5 accepted / 0 equal) is not escalated — its
   `equal == 0` is a mint defect the attr-sweep retro already
   recorded, not starvation, and widening the window would not fix it.
4. **`auto_cond` keeps the 360 window.**  `object_synthesis`
   (`_measure_domain` / `_oracle_sites`) is out of this change's
   scope, so a bare `sdpa_fold` pattern that reaches the auto-cond
   retry still measures its 360 domain.  The retry is a different
   mechanism (minting a guard, not verifying one) and a sibling.
5. **Two sweeps, not one.**  An escalating rule pays the first
   window's envs *and* the second's; the generator is not resumed.
   The waste is the first window (≤360) — small beside the ceiling —
   but a resumed enumeration would be strictly cheaper.
