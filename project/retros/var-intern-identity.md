# Retro — the `is`/`==` carrier-helper bug (weak intern table)

**Status:** fixed (commit `83fbb80`); regression test added.

## Symptom

A full serial `pytest tests/` run reported two failures that **passed
standalone** but failed in file order:

- `test_search_profile_marker_reaches_carrier_upgrade`
- `test_delivered_cost_for_search_keeps_generic_pick`

Both assert that the *uncalibrated* search on `LinearRecurrence`
re-inverts to the carrier member (`apply`/`applyd`). After
`test_cuda_graph_skipped_when_compiled` ran a `torch.compile`
`optimize` earlier in the file, the uncalibrated arm returned `add`
instead — the carrier upgrade no longer fired.

## Root cause

Two independent facts compose:

1. **Only `Op` is hash-consed.**  `Var`/`Param` are plain frozen
   dataclasses — `Var("x", T) == Var("x", T)` but is not the same
   object.  And the `Op._INTERN` table is a *weak* value dictionary:
   entries (and the leaf objects they hold) can be dropped by GC
   between mints.  An extracted term can therefore contain
   **equal-but-distinct `Var` leaves** — here, eight
   `select(x, dim=0, index=i)` enodes whose `x` args were `==`
   but not `is`.

2. **The cost model tested identity.**  `catopt_core.cost.executor`'s
   `_leaf_gather_base` (mirroring `scan_lower._leaf_b_gather`) used
   `b.args[0] is not base`; `_leaf_shared_a` used `a is a0`.  With
   distinct-equal `x` objects the gather recognition returned
   `None`, the batched-scan latency fell back to per-leaf pricing —
   **exactly 2×** on this term (313,258 → 626,470 ns) — and the
   carrier lost the delivered-price comparison.

   The executor's real `_leaf_b_gather` already uses `==` and carries
   the comment explaining why: *"`Op.make` returns the first interned
   node, whose args hold the original value-equal leaves, so identity
   is not a stable test for 'same base term' (weak intern table)."*
   The cost-model copy missed that correction.

## Why it was nondeterministic

Whether the extracted term's selects share one `x` object depends on
intern-table residency — a GC/lifetime property, not a semantic one.
`torch.compile` in the earlier test shifted allocation pressure and
table lifetimes just enough to materialize distinct `Var`s.  The bug
was latent from the day the helpers were written; nothing in the
compile run *caused* it.

## Fix

`is`/`is not` → `==`/`!=` in both `_leaf_gather_base` and
`_leaf_shared_a`, with the same comment the executor carries.
`tests/test_cost.py::test_batched_scan_helpers_value_equality`
constructs the distinct-equal-`Var` state directly (raw `Op(...)`
bypasses interning) and asserts gather/shared-a recognition still
fires — and still declines on a genuinely different base.

## Audit rule

**Any "same term" question on this IR is a `==` question, never `is`**
— identity holds only *within* one intern-table lifetime and only for
`Op`.  The executor knew this; the model copy did not.  The remaining
`is`-comparisons in both files were audited — these two were the only
term comparisons left.

## Second fix in the same commit (unrelated)

`test_results_doc_in_sync` failed because `docs/results.md` had a
hand-appended tooling section while the test requires exact
`render_results_doc` output.  Moved the section to
`docs/results_tooling.md`; the renderer now appends it verbatim, so
the generated file stays reproducible *and* the hand notes survive
regeneration.
