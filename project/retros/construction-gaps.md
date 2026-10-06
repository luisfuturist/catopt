# Retro — closing the human-bar gaps by construction

Date: 2026-10-07
Context: `project/retros/human-bar.md` ablated the zoo arms and named
three machine-coverage gaps: no associativity/chain-composition object
(the ~−20% `LoRAAdapter` win), no self-product fold (the `mul_square`
spelling bridge on GAU/SplineKAN), no gather/dedup family.  The
mission was to close them **by construction** — the construction
operators in `catopt_discovery.object_synthesis` (`fold`, `lift`,
`compose`, `auto_cond`, `relax_guard`, `specialize`) used the way the
arena player would — with each candidate stored through the evidence
store and re-refereed by `evidence.run_gauntlet`.  No law bodies were
written or modified; `object_synthesis.py` needed no API addition.

## The constructions, and what the gauntlet said

Corpus: a tiny injected `GauntletCorpus` per object (the pattern's own
`TermCase`, `base_rules=ALL_RULES`), same shape the synthesis tests
use.  `usable` means every recorded stage passed.

### Gap 1 — chain composition (`assoc_linear`-equivalent)

`compose` rewrites the head premise's RHS through later premises whose
LHS structurally matches; it cannot mint `matmul(B,A)` into a RHS that
doesn't already mention `matmul`.  The arena spelling therefore goes
through **bridge folds** — `lin_as_mm` (`linear(x,W) → matmul(x,Wᵀ)`),
`mm_transpose_flip` (`tA·tB → (B·A)ᵀ`), `mm_t_as_linear` (the inverse
step) — used *only as compose premises*; nothing about them is stored.
The composite carries a declarative `cond` mirroring `assoc_linear`'s
contract (rank-2 weights, `dim-eq` chain alignment).

| construction | object | verdict | stage |
|---|---|---|---|
| verbatim fold `lin(lin(x,A),B)→lin(x,B·A)` | `assoc_linear_fold` | **declined** | novelty: alpha-duplicate of shipped `assoc_linear` |
| full compose → `lin(x, matmul(B,A))` | `assoc_linear_built` | **declined** | novelty: duplicate — the composed object *is* `assoc_linear` up to alpha |
| compose → `matmul(x, tA·tB)` | `linear_chain_mm` | **declined** | typed-pay: two non-foldable `transpose` ops bill like real work |
| compose → `matmul(x, transpose(matmul(B,A)))` | `linear_chain_t` | **usable** | cleared; extract pays 17404 vs 34805 dag-cost on the injected chain |
| cross-boundary `matmul(X, linear(Y,W))→(X·Y)·Wᵀ` | `mm_lin_r_assoc` | **declined** | typed-pay: reassociation mints a real `transpose`; cost-neutral |
| `matmul(linear(X,A),Y)→(X·Aᵀ)·Y` | `lin_mm_l_assoc` | **declined** | typed-pay: same transpose bill |
| 3-chain `lin(lin(lin(x,A),B),C)→matmul(x,(C·B·A)ᵀ)` | `linear_chain3_t` | **declined** | truth: guarded-sweep starvation — 0 accepted synth envs at the 2360-env cap; the one real match verifies equal |

**`linear_chain_t` is the object that matters**: the composed form
`lin(lin(x,A),B) → matmul(x, (B·A)ᵀ)` keeps the fused weight
`matmul(B,A)` inside a *foldable* subtree (matmul folds on param-only
chains; `transpose` does not), which is why this spelling pays and the
`matmul(x, tA·tB)` spelling doesn't.  It is alpha-*distinct* from
`assoc_linear` (rhs op `matmul` vs `linear`), so novelty admits it —
two spellings of the same fold now coexist, and the store holds the
machine-built one.  No missing capability surfaced: bridges + shipped
`assoc_matmul_rev` + a caller `cond` were sufficient.  (One compose
limitation did surface in the gather section: premise `derive`s do not
compose — see below.)

### Gap 2 — self-product folds

| construction | object | verdict | stage |
|---|---|---|---|
| `compose(mul_square, square_to_pow)` → `mul(x,x)→pow(x,2)` | `mul_to_pow2` | **usable** | cleared; derivation replays a strict 2-step cert against `ALL_RULES` |
| `e^a·e^b → e^{a+b}` | `exp_mul_fold` | **usable** | cleared |
| `a²·b² → (a·b)²` | `square_mul_fold` | **usable** | cleared; also `derivable` under shipped rules |
| `(-a)·(-b) → a·b` | `neg_neg_mul` | **usable** | cleared |
| `x^a·x^b → x^{a+b}` | `pow_add_exp` | **usable** | cleared on the measured domain |
| `\|a\|·\|b\| → \|a·b\|` | `abs_mul_fold` | **usable** | cleared |
| `rsqrt(a)·rsqrt(b) → rsqrt(a·b)` | `rsqrt_mul_fold` | **declined** | truth: measured counterexample — `a<0,b<0` gives `NaN·NaN` lhs vs finite rhs; no bank predicate can guard it |
| `add(x,x) → mul(x,2)` | `self_add_fold` | **declined** | closure: the minted member overflows the enode budget (2.50×) iterating comm/assoc |

`mul_to_pow2` is the gap-closer: it is the `x·x → pow` bridge the
store lacked, built by composing two shipped square bridges — its
`derivation` is real and the cert stage replays it.  The honest
declines are real findings: `rsqrt` product-separation is *false* on
the measured domain (NaN corner), not merely unproven.

### Gap 3 — gather / dedup

| construction | object | verdict | stage |
|---|---|---|---|
| `index_select(index_select(t,D,(0,1,2,3)),D,(0,2)) → index_select(t,D,(0,2))` | `isel_compose_lit` | **usable** | cleared — literal-index gather-compose |
| `concat(t[I],t[I],D) → t[I++I]` (dspec `concat`) | `cat_isel_dup` | **usable** | cleared — metavar index, derive is pure data |
| `concat(t[I],t[J],D) → t[I++J]` | `cat_isel_fold` | **usable** | cleared |
| `concat(t[(0,2)],t[(0,2)],D) → t[(0,2,0,2)]` | `cat_isel_dup_lit` | **usable** | cleared |
| `t[I][J] → t[K], K=I[J]` metavar | `isel_compose` | **declined** | full-data: needs a procedural `derive` — the dspec tuple vocabulary has no index-of-index form (`attr`/`attr0`/`concat`/`tuple`/`len`/shape-ops only) |
| `compose(index_select_dedup, index_select_id)` | `dedup_then_id` | **declined** | full-data: transported procedural `check` (and the composite RHS keeps the premise's unbound `U` metavar — premise `derive` outputs don't compose into a composite `dspec`) |
| `compose(dedup, dedup)` | `dedup_dedup` | **declined** | full-data: same transported check; the second premise's derive minted garbage attrs (`('V','I')`) from iterating a metavar string |

Two genuine machinery limits recorded honestly:

1. **No index-of-index in `dspec`.**  The declarative derive language
   can concatenate whole attr tuples and take `attr0`, but cannot
   express `K = (I[j] for j in J)` — the meta-composed gather stays
   procedural and full-data refuses it.
2. **`compose` doesn't compose `derive`s.**  A premise whose RHS mints
   attr metavars via a Python `derive` (e.g. `index_select_dedup`'s
   `U`, `VI`) leaves them unbound in the composite's RHS — the
   composite is unusable regardless of the transported guard.  The
   literal-index objects are the honest ceiling of what *data* can
   express today; general gather-compose would need a declarative
   element-index derive form, not a law body.

## Zoo arms — did it move the needle?

Same measurement as the human-bar retro: `Optimizer(backend=TorchBackend())`,
`search` → `lower` (fp64 verify, rtol 1e-4), `dag_cost` delta; arms
through the `rules=` seam.

| arm | contents | wins |
|---|---|---|
| `default` | `default_rules()` | 5/22 — GAU −0.003, CosineAttention −0.006, LoRAAdapter −19.985, TimeCondConv −4.996, SplineKAN −0.003 |
| `default + store` | + 20 store objects | 6/22 (+ AdaLNBlock −4.070) — retro reproduced |
| `default + constructed` | + 11 new objects | 5/22 — `linear_chain_t` fires on LoRAAdapter, `mul_to_pow2` on GAU + SplineKAN; wins already covered |
| `default + store + constructed` | | 6/22 — same six, three new firing sites |
| `machine` | promoted + store | 3/22 — CosineAttention, TimeCondConv, AdaLNBlock |
| `machine + constructed` | | **6/22** — adds GAU −0.003, LoRAAdapter **−19.985**, SplineKAN −0.003, all via constructed objects |

**Fires grew; combined-arm wins did not** — the new objects' paying
sites were already merged by the shipped equivalents.  The closure
shows in the **machine repertoire**: 3/22 → **6/22**, matching the
best combined arm the retro measured, and — the point of the mission —
now owning the zoo's largest win (−19.985 `LoRAAdapter`) and both
self-product zeros **by construction**, where the human bar previously
held them exclusively.

The gather/dedup objects are usable and deployable
(`machine_pack.load_pack` round-trips them — pinned in the test) but
fire nowhere in the 22-model zoo: no held-out model spells a
`concat`-of-gathers or a nested `index_select`.  Gather coverage is
capability-admitted, site-absent on this holdout.

## Per-gap verdict

| gap | closed? | by | honest residue |
|---|---|---|---|
| chain composition | **yes** | `linear_chain_t` (compose over bridge folds + `assoc_matmul_rev`) — usable, deploys, owns LoRA −19.985 in the machine arm | the `linear(x,·)` spelling itself is novelty-locked; deep (3+) chains starve the guarded sweep — a measurement window issue, not a wrong object |
| self-product | **yes** | `mul_to_pow2` (compose w/ replaying cert) + five elementwise folds — all usable | `rsqrt` product is measurably *false* (declined); `add(x,x)→mul(x,2)` overflows closure |
| gather/dedup | **partially** | `cat_isel_dup`/`cat_isel_fold`/`cat_isel_dup_lit`/`isel_compose_lit` — usable | metavar `t[I][J]→t[I[J]]` needs a dspec index-of-index form the language lacks; premise `derive`s don't compose; no zoo site fires |

## Caveats / limits

- **Cost-model fidelity** — same as the zoo retro: `Δ%` is the
  executor-aware dag-cost model under generic delivery, not wall
  clock.  The `linear_chain_t` pay rides the param-only fold discount
  (a `matmul` over `Param` leaves is compile-time work; `transpose`
  is deliberately not foldable — that's what made the paying spelling
  the paying spelling).
- **Guarded-sweep starvation** — `linear_chain3_t`'s decline is a
  sweep-coverage datum (0 accepted envs at the escalation cap on a
  6-conjunct guard), not evidence against the object; a wider synth
  window or a lazier cond could admit it.
- **Bridge objects are scaffolding** — `lin_as_mm`/`mm_transpose_flip`
  were never stored or measured as standalone claims; used unguarded
  they are unsound outside the composite's lhs domain.  Only the
  composite carried a gauntlet verdict.
- **Probe**: `/tmp/gap_zoo.py` (ephemeral, per the probe convention);
  store constructors and arms recomputable from
  `tests/test_construction_gaps.py`.

## Files

- `tests/test_construction_gaps.py` — 14 tests pinning each
  construction's store+gauntlet path: the usable objects, every
  honest decline with its exact stage, and the `machine_pack` deploy
  round-trip.
- `project/retros/construction-gaps.md` — this note.
- No source changes: `object_synthesis.py`'s constructors were
  sufficient for every admitted object; the two recorded limits are
  vocabulary gaps in `dspec`/compose, not missing operators.
- Gates: `pytest tests/test_construction_gaps.py` 14 passed (~0.6 s);
  `ruff check`/`ruff format --check` clean on the new files; radon
  ratchet unaffected (test file, no source edits).  Not committed.

## Coverage notes (unreachable-defensive arcs, verified)

- `object_synthesis.py:1481` / `:1507` — the `suv/suw is None`
  skips in `_wrap_preds`/`_pair_preds`: `views` entries are
  pre-filtered to spec-non-None nodes, so the recomputed spec can
  never be None there.
- `object_synthesis.py:2080->2079` — `_min_cover`'s
  `i not in chosen` skip: kill masks of already-chosen predicates
  are removed from `uncovered`, so they can never kill a
  still-uncovered site.

## Machinery limits — closed (derive transport + index-of-index)

Both measured limits are closed as *data*, not procedural escapes:

- **Compose transports derives.**  `compose_objects` now threads
  each premise's binding through the chain: a head premise's pure
  `dspec` entries become composite spec entries (substituted
  through the match and the accumulated *derived* map); procedural
  `check`/`derive` hooks ride a fire-time `_chain_hooks` pair that
  replays the premises' fire order — each premise's binding
  re-expressed under the accumulated env, its derive outputs feeding
  later premises.  A composite over procedural premises honestly
  records both hooks in `missing_hooks` (`["check","derive"]` — the
  pin changed *upward* in honesty); a composite over dspec premises
  serializes fully.
- **Index-of-index is a derive expr.**  The dspec language gained
  `("gather", e_table, e_sel)` (`t[I][J] → t[I[J]]` = `K ↦
  ("gather", ("attr","I"), ("attr","J"))`), `("posmap", …)`, the
  `("isel-out", T, D, I)` shape spec, and the `attr-range`
  predicate.  Attr positions in conds/specs/exprs now accept inline
  derive exprs — a transported guard can read a *derived* attr.
- **`attr-is`/`attr-eq`/`attr-in` keep the absent-is-None contract**:
  the new `_aval` resolver distinguishes unbound (`_MISSING`) from
  bound-None, but the value predicates map `_MISSING → None` — the
  synthesized optional-attr guards (`slice.start is None` = torch
  default) depend on it; the pin held.

**Measured:** `isel_compose` (`t[I][J] → t[I[J]]`) as a pure-data
object clears **all eight gauntlet stages including cert — `usable`**
— a construction class that was refused at `full-data` on a
technicality is now an admitted, serializable, deployable object.
The procedural spelling still declines honestly (and the pin proves
the data spelling replaces it, not hides it).
