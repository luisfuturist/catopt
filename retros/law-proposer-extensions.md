# Proposer extensions — mixed views, pattern recognition, one new SHIP

`law-corpus-expansion.md` §6 named the two best next-law leads the
enlarged corpus offered, and both were *proposer* gaps, not corpus or
oracle gaps:

1. the census-naturality generator only proposed `f(g(u,A), g(v,A)) →
   g(f(u,v), A)` — a pointwise `f` over two copies of the **same**
   view — while the corpus's frequent shapes are asymmetric
   (`mul(unsqueeze, stack)`, `mul(select, add)`, `mul(slice, ·)`);
2. `ManualSoftmaxAttention` spells softmax as `exp / sum`, a shape no
   naturality schema can reach — a recognition fold, not a
   commutation.

This retro records implementing both extensions in
`tools/law_pipeline.py` and re-running the validated pipeline.

The headline is the pipeline's **second machine-discovered SHIP**:

* `recognize:softmax` — `div(exp(u), sum(exp(u), dim, keepdim)) →
  softmax(u, dim)` — clears every gate: numerically true, genuinely
  new, fires on `ManualSoftmaxAttention`, lowers end-to-end extracted
  cost **19.0 %**, certificate replays, enode ratio **1.06×**.
* Held-out rediscovery still passes: `--holdout select_mul` ranks
  `census:mul_select` **#1 of 50** (was 35), SHIP, identical evidence —
  and the softmax candidate holds its SHIP at #2 under the reduced
  search set.
* The 14 new mixed-view candidates ship **zero** — but they produce
  the pipeline's most informative no-ship verdicts yet, including
  false-but-*paying* candidates caught by the truth and verify gates
  (§3).

Reproduce:

    .venv/bin/python tools/law_pipeline.py --vocab derived
    .venv/bin/python tools/law_pipeline.py --vocab derived --holdout select_mul

CPU-only, ~6 min (the new candidates that fire add reach saturations).

## 1. What was added (all in `tools/law_pipeline.py`)

* **`Proposal.check` / `Proposal.derive`** — the `Rewrite`
  side-condition hooks, threaded through `Proposal.as_rule()`.  A
  candidate whose RHS needs an attribute the LHS does not carry
  verbatim (the sum's `dim=(-1,)` tuple to softmax's `dim=-1` int) can
  now express it.  `_instance_from_match` applies the same hooks per
  real match — check veto / `None` derive skips the match — so the
  numeric and derivability oracles see exactly the instance a firing
  would produce.
* **`_census_mixed_naturality`** — for every census tuple `f(g, h)`
  with `f` binary-pointwise and distinct children where ≥1 is a view,
  emits a small family of `f(g(u,A), h(v,B)) → w(f(u,v))` candidates:
  `w` over each view present, plus the identity (strip).  Distinct
  views get distinct attr metavariables (`A_*` / `B_*`); a non-view
  operand stays a bare metavar, so `mul(select,·)` and
  `mul(select,add)` collapse onto one pattern and de-dup merges
  provenance.  No algebraic theory of view inversion — the oracles
  are the referee.
* **`_pattern_recognition`** — scans the census for
  *composed-then-reduced* chains `f(u(·), r(u(·)))` where `r(u(·))`
  also occurs, and emits the candidate a small `_RECOGNIZERS` table
  maps to.  One recognizer today: `(div, exp, sum) → softmax`, with
  `check` = keepdim ∧ single-axis and `derive` = unwrap the dim
  tuple.

Both feed `Proposal`s into the unchanged verify → measure → rank
path; no `packages/` law was touched.

## 2. Result — 50 proposals (was 35), 1 SHIP

| run | proposals | firing on a real model | shippable |
|---|---|---|---|
| `--vocab derived` | 50 | 14 | **1** (`recognize:softmax`) |
| `--vocab hand` | 50 | 14 | **1** (`recognize:softmax`) |
| `--vocab derived --holdout select_mul` | 50 | 14 | 2 (winner + softmax) |

Pattern recognition is not vocab-gated, so the SHIP is identical
under both alphabets.

The ship candidate's full evidence row:

    #1 recognize:softmax [pattern-recognition]
       true=True new=new census=1 match=1 fires=1 paid=1
       drop=19.0% cert=pass enode=1.06x   case: ManualSoftmaxAttention
       rule: (div (exp 'U'), (sum (exp 'U'), dim=RD, keepdim=RK))
             -> (softmax 'U', dim=SD)

The `R(...)` the pipeline's evidence supports admitting (a
recommendation — admission is the reviewed step, not done here):

```python
R(
    "softmax_fold",
    Op.make(
        "div",
        Op.make("exp", "U"),
        Op.make("sum", Op.make("exp", "U"), dim="RD", keepdim="RK"),
    ),
    Op.make("softmax", "U", dim="SD"),
    check=_check_sum_keepdim,    # keepdim ∧ single reduce axis
    derive=_derive_softmax_dim,  # {"$attr:SD": D[0]}
    tags=(SIMPLIFICATION,),
)
```

It is mathematically unremarkable — it *is* the definition of
softmax — which is exactly the point: a corpus that spells the kernel
by hand gets the kernel back, with a certificate.

## 3. What the mixed-view family found (and why none shipped)

14 candidates, four honest verdict classes:

* **True, fires, never pays** — `mixed:add_select_r_w`
  (`add(V, select(U,A)) → select(add(V,U),A)`): numerically true on
  its `LinearRecurrence` instance, fires 4, and the RHS does strictly
  more work (add on the un-selected tensor) so extraction never picks
  it.  A true law that is cost-useless in this orientation.
* **True but bench-only** — `mixed:mul_reshape_l_w` /
  `mixed:mul_transpose_l_w` (`mul(v(U,A), c) → v(mul(U,c),A)`): true
  on the `mul(view, const)` sites — but those live in bench cases, so
  the model-only firing probe sees 0.  Applicable ≠ reached.
* **False / unproven — including false-but-paying** —
  `mixed:add_select_r_id` is numerically *false*, yet "paid" one
  model with a 63.6 % drop: a wrong rewrite is cheaper — and the
  layered gates caught it twice (numeric oracle + `verify_fail=1`
  on the lowered module).  `mixed:mul_select_l_id` is the starker
  case: 41 matches, 40 fires, 3 pays, up to 55 % drop under the
  holdout — and still unproven (`mul(U,V)` is ill-shaped on the real
  instance; the oracle abstains rather than guessing).  The no-ship
  verdicts are the machinery working: a false law that lowers cost is
  a *hazard*, and the gates reject on truth before reach.
* **Unproven, no model firing** — the `mul_slice` pair: 16 LHS
  matches (the family's most-applicable LHS) but the RHS instance is
  ill-typed on the real match, and no model fires.

The MoE-dispatch lead specifically — `mul(unsqueeze, stack)` — yields
`mixed:mul_unsqueeze_l_w`/`_id`: unproven/strip-false.  The grouped
dispatch fold is not a view naturality; §6 stands amended on that
point, by measurement not guesswork.

## 4. Recognizer telemetry

The composed-then-reduced scan found 9 chains; 8 have no recognizer —
an honest map of missing folds for future work:

    (mul, linear, silu)        SwiGLU:  mul(silu(·), ·) pairs
    (mul, linear, gelu)        GEGLU:   mul(gelu(·), ·) pairs
    (mul, add, rsqrt)          RMSNorm: x * rsqrt(mean(x²)+eps)
    (matmul, transpose, T)     attention score shapes
    (matmul, linear, T)        linear-as-matmul forms
    (concat, slice, neg)       SSM carrier slicing

None was guessed — a chain without a recognizer stays a census fact.

## 5. Honesty / limits

* **`num_true` is single-instance.**  The softmax fold is true in
  general (modulo fp numerics); the `w`-variant trues are true *on
  their corpus instance* — a pattern-level law with no shape guard is
  only as general as the check it carries.  Admission review should
  weigh that, not just the verdict row.
* **No new two-view law tested.**  The `f(g,h)` both-views branch
  emitted nothing — no distinct-view pair occurs in the census; the
  code path exists for the first corpus that produces one.
* **Runtime grew** (~25 s → ~6 min): candidates that fire pay for a
  full with/without reach saturation per model.  Still bounded;
  worth watching as families grow.
* **`packages/` untouched** — tools-only change, as tasked.

## 6. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — pass
* `tools/radon_ratchet.py` — pass (scans `packages`; tool code is
  outside its target list)
* `uv run pytest -q` — **3367 passed, 31 skipped** (~8 min)
* `law_pipeline.py --vocab derived --holdout select_mul` — **PASS**
  (`census:mul_select` rank 1 of 50, SHIP)

## 7. Where the next law might hide (updated)

1. **A recognizer for `(mul, add, rsqrt)` or the SwiGLU/GEGLU
   chains** — the un-recognized list above is now the queue; each is
   one `_RECOGNIZERS` entry plus its check/derive.
2. **Inverse-orientation candidates** — several true `w`-variants
   failed only because the census form is already the cheap one; the
   reverse `v(f(u,v),A) → f(v(u,A),v)` direction may pay where the
   corpus carries the un-pushed shape (none today).
3. **`group_norm` attr schema** — unchanged; still the ConvNeXt
   blocker.
