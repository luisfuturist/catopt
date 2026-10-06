# Retro: the guard residuals — the mm rank≥2 hole and the enumeration gaps

Date: 2026-10.  The follow-up sweep the earlier retros left open:
`shipped-guard-tightening.md`'s "Honest limits", `cond-bank.md`'s
"two classes remain unnameable", `cap-policy.md`'s five blind spots
and `value-bank.md`'s four.  Each residual is either **closed**
(declarative, minimal, equal-preserving) or **documented** with the
precise reason it is not honestly closable inside the owned seams
(`laws/tensor.py`, `laws/cond.py`, `catopt_core/typing.py`).

Owned seams touched: `laws/tensor.py` (the mm guards + the two
`linear`-scale guards).  `laws/cond.py` needed **no** new atom — every
closure composes existing vocabulary.  `oracle.py`, `evidence.py`,
`intake.py`, `object_synthesis.py` untouched (the residuals that live
there are documented, not edited).

## 1. The mm-guard rank≥2 branch — **CLOSED**

### The latent case is real (and a shipped test blessed it)

`shipped-guard-tightening.md` recorded the rank≥2 branch as "latent,
outside the bank": it trusts a leading-axis-padding argument and would
accept `a=(2,3)`, `b=(1,3)`.  Constructed:

```
distribute_matmul_over_add   W=(4,2), a=(2,3), b=(1,3)
  guard        -> True
  lhs == rhs   -> rhs-err: mat1 and mat2 shapes cannot be multiplied
                  (4x2 and 1x3)
```

Broadcast pads the *contraction* axis too (`b[-2]=1 -> 3`), so
`matmul(W, b)` cannot contract with the same `W` that `matmul(W, a)`
needs.  The residual was **not** outside the bank: the shipped test
`test_cond_laws.test_mm_addends_rank_equal_requires_shape_eq` asserted
the identical defect —

```
batched = {"W": _p("W", 4, 3), "a": _v("a", 3, 5), "b": _v("b", 1, 5)}
assert DISTRIBUTE_MUL.check(batched)   # measured rhs-err, not equal
```

So the old guard's region contained a *shipped* counterexample, not a
merely-latent one.  The fix updates that test to a sound rank≥2
mixed-rank binding and adds the decline + numeric pins.

### The fix

The rank≥2 branch gains the **contraction-axis** equality.  For
`matmul(W, ·)` the addends contract on axis `-2`:

```python
_COND_MM_ADDENDS = (
    "or",
    ("and", ("rank-eq", "a", "b"), ("rank", "a", ">=", 1),
            ("shape-eq", "a", "b")),
    ("and", ("rank", "a", ">=", 2), ("rank", "b", ">=", 2),
            ("dim-eq", "a", -2, "b", -2)),
)
```

`dim-eq` is the *smallest* addition that closes it: the rank-equal
branch's `shape-eq` already implies it, so no equal site is lost.  The
retro's `mm-shape-ok` composition (`mm-shape-ok(W,a) ∧
mm-shape-ok(W,b)`) also closes it but over-shrinks — it additionally
requires `W` shaped (measured **18** accepted vs `dim-eq`'s **20**),
so it was not taken.

### The summed weights need the *spelling's* axis

`_COND_MM_WEIGHTS` is shared by the `matmul` and `linear` spellings,
but the contraction axis differs:

* `matmul(x, W)` contracts `x[-1]` with `W[-2]`;
* `linear(x, W)` is `x @ W.T`, contracting `x[-1]` with `W.T[-2] = W[-1]`.

A single shared clause cannot be correct.  `_COND_MM_WEIGHTS` (matmul)
gains `dim-eq(W, -2, W2, -2)`; the two `linear` rules
(`weight_factor_linear`, `weight_distribute_linear`) move to a new
`_COND_MM_WEIGHTS_LINEAR` with `dim-eq(W, -1, W2, -1)`.  Measured that
the split is necessary, not cosmetic:

```
weight_distribute_matmul  W=(2,3) W2=(1,3) x=(4,2)  -> rhs-err
  dim-eq(-2) declines, dim-eq(-1) wrongly accepts
weight_distribute_linear  W=(2,3) W2=(2,1) x=(4,3)  -> rhs-err
  dim-eq(-1) declines, dim-eq(-2) wrongly accepts
```

### Measured region delta (default 360-site window)

| rule | before acc/eq/ne/rerr/oerr | after |
|---|---|---|
| `distribute_matmul_over_add` | 36 / 5 / 0 / 0 / 31 | **20 / 5 / 0 / 0 / 15** |
| `factor_matmul` | 36 / 5 / 0 / 0 / 31 | **20 / 5 / 0 / 0 / 15** |
| `weight_distribute_matmul` | 29 / 5 / 0 / 0 / 24 | **17 / 5 / 0 / 0 / 12** |
| `weight_factor_matmul` | 29 / 5 / 0 / 0 / 24 | **17 / 5 / 0 / 0 / 12** |
| `weight_factor_linear` | 29 / 4 / 0 / 0 / 25 | **17 / 4 / 0 / 0 / 13** |
| `weight_distribute_linear` | 29 / 4 / 0 / 0 / 25 | **17 / 4 / 0 / 0 / 13** |

Every `equal` count is **preserved exactly**; the declines are all
rank≥2 `both-err` sites (never an equal one) — the mixed-extent
contraction-axis-disagreement corner.  A full re-sweep of all **36
shipped guarded rules** reports `unclean: none`.

## 2. `gqa_absorb_repeat`'s empty synth region — **DOCUMENTED (enumeration gap)**

The region is **not** genuinely empty — the guard accepts a hand-built
binding (pinned in `test_discovery_oracle`):

```
q=(2,3,4,4), k=v=(2,3,2,4)
UDk=UDv=-2, ESk=ESv=(2,3,2,2,4), RSk=RSv=(2,3,4,4)  ->  guard True
```

`unsqueeze(-2)` gives `(2,3,2,1,4)`, `expand` grows dim -2 to
`(2,3,2,2,4)`, `reshape` merges dims 2–3 to `(2,3,4,4)` — a
`repeat_interleave` with `r=2`, and `q[-2] = 4 = k[-2]·r`.

(The earlier draft of this pin used the rank-3 spelling `q=(2,6,4)`,
`k=v=(2,3,4)` — every repeat clause held, but its minted
`enable_gqa` RHS could not contract: a measured `rhs-err`, the
latent hole `chained-bindings.md` §6 named.  The guard now carries
`("rank", q|k|v, "==", 4)` — `sdpa`'s head axis must sit at `-3`
after the pattern's `transpose(1, 2)` — see
`shipped-guard-tightening.md`'s follow-up section.)

The enumeration never mints it.  `oracle._attr_domains` resolves each
view node's operand shape through `_operand_shape`, which for
`expand(unsqueeze(k, dim="UDk"), shape="ESk")` falls back to the first
*leaf* (`k`) — the `unsqueeze`/`expand` attrs are still string
metavars, so `_shape_of` reports unknown and the walk finds `k`.  The
`expand` domain is therefore `[(2, *k.shape)]` (or a grown shape), and
the `reshape` domain `_reshape_targets(k.shape)` — neither chains onto
the `unsqueeze` output.  So `repeat-chain` can never hold on an
enumerated binding; the honest `acc=0` is the *enumeration's* residue,
not the guard's.

Closing it needs the attr domains to *chain* the view outputs (an
`oracle` enumeration change) — out of this change's scope.  The value
bank (`oracle._CONST_DOMAIN`) is not the blocker: no `const-cmp`
demands a value here.

## 3. `rms_norm_fold`/`_nogain`'s `tail-block` — **DOCUMENTED (spec correct; ordering artifact)**

The `tail-block` spec is **correct** — the guard accepts the binding it
should (pinned):

```
u=(2,3,4), w=(4,), MD=-1  ->  guard True   (tail = u[-1:] = (4,))
u=(2,3,4), w=(2,3,4), MD=(0,1,2)  ->  guard True  (tail = u[-3:])
```

So it is not a mis-specified guard.  The `0/6000` measurement is an
**enumeration-ordering artifact**.  Measured at the 360/6000/40000
windows:

* the front `_COND_RMS_FOLD` (keepdim ∧ numeric eps ∧ `P==2`) passes
  **94 / 6000** — and **every one** of those has `u=()` (rank 0), where
  `tail-block` is `None`;
* the full guard passes **0 / 40000**.

The four free operands (`EPS`, `P`, `u`, `w`) are enumerated by
`_diag_product` in index-sum order; the corner that satisfies both the
front and a non-scalar `u` (e.g. `u=(2,3,4)` at bank index 18,
`w=(4,)` at 4, `P=Const(2)` at 3, `EPS=Const(0.5)` at 1) has a free
index of ~2.4×10⁴, which the interleave with the (many) attr bases
pushes past ~10⁶ sites.  The bank holds the values; the *ordering*
buries the corner.  Closing it needs an enumeration change (a targeted
base-major phase, or an operand-derived `w` bank) in `oracle` — out of
scope.

## 4. The `_w` wrap family's flat-read blind spots — **DOCUMENTED (not cleanly declarable here)**

`cond-bank.md` left the slice family (`mixed:mul_slice_l_w`,
`slice_mul`) and `reshape_transpose` refusing, "needing a flat-read
predicate per index view".  Assessment:

* The **shape** half is already expressible and emitted: `slice-out`
  + `bcast-eq` give the grid equation, and `bcast-dim-inv` covers the
  "partner constant along the sliced axis" case; a **full-extent**
  slice (`start=0`, `end ≥ dim`) is expressible with `attr-eq` +
  `attr-cmp-dim`.  These already decline the value-level
  counterexamples (e.g. `slice(u, 0, 1..3)` against a length-3 `v`).
* The residue is the **partial** slice's flat index map: pushing a
  partial slice through a broadcast commutes only when the sliced
  extent *aligns* with the partner's — a strided/offset map the
  existing `_viewed_map` / `_shifted_map` machinery does not model (it
  is contiguous-only, built for `unsqueeze`/`reshape`).  A new atom
  would need that map, and its correctness could only be validated
  against the auto-cond measured domain — whose generator
  (`object_synthesis._view_commute_preds`) is out of this change's
  scope, and `oracle`'s enumeration is a sibling.  Adding an
  unvalidated atom would violate the "measured, not proven" discipline,
  so it is **not** cleanly declarable here: recorded, not taken.

## 5. The other named residuals

### Closed

**The `linear` rank≥3 weight** (`shipped-guard-tightening.md` honest
limit).  `F.linear` refuses a rank≥3 weight
(`t() expects a tensor with <= 2 dimensions`), but the guards read
`W.T` through `mm-shape-ok` / `mm-out`, which resolve a rank≥3 `W.T`
happily — so `linear_channel_scale` accepted a rank-3-`W` `both-err`.
Closed with one existing atom, `_COND_LINEAR_WT = ("rank", "W", "<=",
2)`, conjoined to both `_COND_CHANNEL_SCALE` and `_COND_ROW_SCALE`.
Measured: the only site it declines is the rank-3-`W` `both-err`;
every equal site keeps.

| rule | before acc/eq/oerr | after |
|---|---|---|
| `linear_channel_scale` | 25 / 24 / 1 | **24 / 24 / 0** |
| `linear_channel_scale_rev` | 25 / 24 / 1 | **24 / 24 / 0** |
| `linear_row_scale` | 23 / 22 / 1 | **22 / 22 / 0** |
| `linear_row_scale_rev` | 23 / 22 / 1 | **22 / 22 / 0** |

### Documented — not closable inside the owned seams

Every remaining named residual bottoms out in a module this change does
not own; each is a real gap, recorded with its owning seam.

| source | residual | owning seam |
|---|---|---|
| `cap-policy` #2 | `_drop` twins need a ~16000 cap (base-major phase) | `oracle` enumeration policy |
| `cap-policy` #3 | escalation triggers on `accepted==0`, not `equal==0` | `oracle`/`evidence` |
| `cap-policy` #4 | `auto_cond` keeps the 360 window | `object_synthesis` |
| `cap-policy` #5 | escalating rules pay both windows (no resume) | `evidence` |
| `value-bank` #1 | `nan == nan` reads `unequal` (tolerance) | `proposal._allclose` |
| `value-bank` #2 | `sdpa_fold_masked_fill*` needs a **bool** mask leaf | `oracle` dtype bank |
| `value-bank` #4 | the per-op const table is hand-keyed | `oracle` |
| `attr-sweep` #2 | finite kind domains (`float` probes 3 values) | `oracle` |
| `attr-sweep` #3 | unenumerable attrs (`attn_mask`, `equation`) | `oracle` |
| `attr-sweep` #4 | `other_err` is information-poor | `oracle` |
| `attr-sweep` #5 | sweep/fire can disagree on attr spelling | `oracle` |
| `cond-bank` §5 | production bank needs the view-commute emission | `object_synthesis` |
| `shipped-guard-tightening` | `bcast-into` declines a symbolic output shape | by design (conservative) |

None of these is a guard/spec defect in `laws/` — they are enumeration,
policy, or bank gaps in the sibling seams.

## Tests

* `tests/test_cond_laws.py` —
  `test_mm_addends_rank_ge2_requires_contraction_axis` (the latent
  `a=(3,5)`/`b=(1,5)` rhs-err is declined, a sound mixed-rank pair
  clears and evaluates `equal`); `test_mm_weights_rank_ge2_uses_the_spelling_axis`
  (matmul declines `W[-2]`-disagreement, linear declines `W[-1]`-
  disagreement, each keeps its own sound case);
  `test_linear_scale_guards_cap_the_weight_arity` (rank-3 `W`
  declined, rank-2 clears); the pre-existing
  `test_mm_addends_rank_equal_requires_shape_eq` was **corrected** — its
  `batched` binding was a measured rhs-err.
* `tests/test_discovery_oracle.py` —
  `test_gqa_absorb_repeat_guard_accepts_a_chained_binding` and
  `test_rms_norm_fold_guard_accepts_a_tail_block_binding` pin the
  "region non-empty, enumeration the gap" diagnosis for items 2 and 3.
* `tests/test_matmul_factor_laws.py` — the shipped firing/guard tests
  still pass unchanged (the `batched` rank-mismatch binding there was
  already sound).

## Gates

`pytest tests/test_cond_laws.py tests/test_rulesets.py
tests/test_law_serialize.py tests/test_discovery_oracle.py
tests/test_discovery_gauntlet.py -q` — **267 passed**; the touched
law files (`test_matmul_factor_laws`, `test_rms_norm_laws`,
`test_morphism_fusednorm_tie`, `test_synthesis_guards`,
`test_laws_structure`, `test_derive_laws`, `test_contracts`,
`test_select_laws`) — **327 passed, 19 skipped**; the synthesis
siblings (`test_discovery_synthesis`, `test_synthesis_seeds`,
`test_hybrid`, `test_square_rsqrt_laws`) — **62 passed**.  A full
36-rule guarded re-sweep reports `unclean: none`.  `ruff check
packages` / `ruff format --check packages` / `ty check` /
`radon_ratchet` (3281 functions) / `vulture` clean.  The one ruff
finding on `tests/test_cond_laws.py` (`I001`) is pre-existing drift
(the same file is `I001` at `HEAD`); `tests/**` is not format-checked.
No new cond atom, no new suppression.

## Honest limits

* **Measured, not proven.**  Each closure is exact on the 360-site
  window; a counterexample outside it stands until a wider sweep
  reaches it — the standing caveat every guarded law carries.
* **`dim-eq` on a `None` dim** passes only when *both* sides are
  `None` (the strict posture: a known-vs-unknown mismatch declines).
* **The gqa / rms regions stay empty** in the shipped enumeration.  The
  guards are correct and the real firings are untouched
  (`test_gqa_absorb_repeat_kv`, `test_rmsnorm_roundtrip_and_optimize`
  pass); only the *synth* sweep reads `0 accepted`, which the
  derivable-gate truth gate waives for a human-admitted law.
