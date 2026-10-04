# Typed-pay gate — "pays" now requires the minted member to denote

`view-index-oracle.md` resolved every candidate's *truth* but left a
measurement bug standing: `law_pipeline.measure` counted a firing and
its cost drop **without checking that the instantiated RHS is
well-typed**.  `eg.rule_fires` counts merges, not valid mints; a
rewrite can mint a member that does not denote — `mul(u, v)` where
the bound operands do not broadcast, `add(u, v)` where the bound `u`
evaluates to a *tuple* (`getitem` over `topk`/`var_mean`/`cummax`), or
a view op whose attr only `%`-normalises into range inside the shape
rules while torch rejects it at eval.  The minted member enters the
e-class, the cost model prices it (an *unknown* shape falls back to
~free — `cost/basic.py`), extraction picks it, and the probe records
"paid".  Cost-dropping an invalid program is not evidence.

This retro records the audit now wired into `measure`, and the
before/after over the 166-workload corpus.

Headline: **33 firing candidates, 19 mint at least one ill-typed
member — 49 ill-typed fires, 11 probe pays and 12 reach drops
suppressed.**  Shippable stays 0 — the gate only makes the honest
count visible.

Reproduce:

    .venv/bin/python tools/law_pipeline.py --vocab derived \
        --evidence-db tools/evidence.db --json /tmp/pipeline-typed.json

## 1. What was measured before

`_probe` (`tools/law_impact.py`) runs the lone candidate rule on each
workload, reads `eg.rule_fires`, extracts the cheapest member, and
reports `paid` whenever `dag_cost` strictly dropped.  `f.paid` is a
*cost* verdict — nothing in it asks whether the cheaper program runs.
`_reach` likewise accrued every `base_cost > add_cost` row.

## 2. The audit

`measure`'s firing path is unchanged (`_probe` still owns fires,
cost delta, lowered-module verify); a second pass `_typed_probe`
re-runs the identical lone-rule saturation on a proof-tracking
e-graph — the default `truncation_level=2` already records every
application (`eg.applications`: matched e-class, fired `subst`,
RHS root enode) and every merge (`eg.merge_log`).

* **Per-fire split.**  Each recorded application that *merged*
  (aligned to a `merge_log` edge by frozen binding — merged apps are
  a prefix of each binding class, since a union never splits) has
  its RHS reconstructed at term level — bound metavars resolved via
  `eg._any_term_cached`, the same minimum-size representative the
  `check`/`derive` hooks saw, `$attr:` bindings taken from the record
  verbatim — then classified: *ill-typed* iff any subterm's
  `_shape_of` raises / returns `_INVALID` or `None` / carries a
  non-int or negative dim, or the term fails fp64 evaluation (the
  eval leg catches what `%`-normalised shape rules cannot see —
  `unsqueeze(u, dim=9)` on rank-2, and tuple-valued operands that
  shape-passthrough as tensors).  `Evidence.fires` keeps counting
  every merged fire; `fires_typed` + `fires_ill_typed` make it
  honest.
* **The pick check.**  `paid`/`changed`/`cost_drop` only accrue when
  the *extracted winner* denotes: `_pick_ill_typed(best, input…)`
  flags any bad subterm absent from the reference terms, or a `best`
  that fails evaluation where a reference evaluates.  A suppressed
  probe pay counts in `paid_ill_typed` (and names the case in
  `ill_typed_cases`); a suppressed reach drop counts in `reach_ill`.
  A doomed lowering on an ill-typed pick no longer double-counts as
  `verify_fail`.
* **The ship gate.**  `shippable` additionally requires
  `fires_ill_typed == 0`: a law that mints members that cannot
  denote is a search hazard even where it also pays — cheap-fallback
  ill-typed members are extractable.
* **The verdict surface.**  `no_ship_reason` reports, in order:
  `pays only on ill-typed sites (N suppressed, k/n fires ill-typed)`,
  `every fire mints an ill-typed member`, `fires but never lowers
  cost (k/n fires ill-typed)`, or — for a candidate that pays on
  typed sites yet still mints garbage — `mints ill-typed members on
  real sites`.

## 3. Before / after

Old columns from the evidence store (previous run's verdict rows);
new columns from this run.  `paid*` is the gated count; `paid_ill`
are the probe pays suppressed because the extracted winner was
ill-typed; `reach_ill` are suppressed saturation drops.

| candidate | old fires/paid/drop | fires (typed/ill) | paid | paid_ill | reach_ill | drop |
|---|---|---|---|---|---|---|
| `mixed:add_getitem_l_id` | 4/4/40 % | 4 (0/4) | **0** | 4 | 4 | **0 %** |
| `mixed:sub_getitem_r_id` | 4/3/40 % | 4 (0/4) | **0** | 3 | 3 | **0 %** |
| `mixed:eq_getitem_l_id` | 4/1/2 % | 4 (4/0) | **0** | 1 | 1 | **0 %** |
| `mixed:mul_unsqueeze_reshape_id` | 1/1/13 % | 1 (0/1) | **0** | 0 | 0 | **0 %** |
| `mixed:mul_unsqueeze_reshape_wl` | 1/1/4 % | 1 (0/1) | **0** | 0 | 0 | **0 %** |
| `mixed:mul_unsqueeze_reshape_wr` | 1/1/9 % | 1 (0/1) | **0** | 0 | 0 | **0 %** |
| `mixed:add_select_r_id` | 4/1/20 % | 4 (1/3) | **0** | 1 | 3 | 20 % † |
| `reshape_transpose` | 58/2/0 % | 58 (51/7) | **0** | 2 | 1 | 0 % |
| `mixed:mul_unsqueeze_l_id` | 8/2/14 % | 8 (5/3) | **2** | 0 | 0 | 14 % |
| `mixed:mul_unsqueeze_r_id` | 4/1/14 % | 4 (4/0) | 1 | 0 | 0 | 14 % |
| `mixed:mul_slice_l_id` | 5/1/25 % | 5 (5/0) | 1 | 0 | 0 | 25 % |
| `mixed:sub_unsqueeze_l_id` | 2/1/9 % | 2 (1/1) | 1 | 0 | 0 | 9 % |
| `mixed:mul_transpose_l_id` | 1/1/3 % | 1 (1/0) | 1 | 0 | 0 | 3 % |
| `mixed:mul_select_l_id` | 40/3/40 % | 40 (40/0) | 3 | 0 | 0 | 40 % |
| `mixed:mul_chunk_l_id` | 1/0/10 % | 1 (0/1) | 0 | 0 | 0 | 10 % † |
| `census:mul_select` | 24/5/0 % | 24 (24/0) | 5 | 0 | 0 | 0 % |
| `linear_factor` / `recognize:softmax` | 2/2, 1/1 | unchanged | — | — | — | — |

† the surviving `cost_drop` rows are *well-typed* extractions — the
gate suppresses only drops whose winner does not denote.

Zero-pay candidates with ill-typed fires the table omits:
`mul_slice_l_w` 5/4, `mul_unsqueeze_l_w` 5/3, `eq_getitem_l_w`,
`add_getitem_l_w`, `sub_getitem_r_w` all 0/4 (every fire mints an
ill-typed member — the tuple-valued-`U` signatures the oracle named),
`sub_unsqueeze_l_w` 1/1, `mul_chunk_l_w`, `add_transpose_l_id`,
`add_transpose_l_w` 0/1.

### What the split says

* **The `getitem` strips were pure inflation.**  All four real fires
  of `add_getitem_l_id` (VarNorm / StdNorm / KthMode / MinFmax
  intake sites) bind `U` to a tuple producer — `add(u, v)` can never
  denote.  paid 4→0, drop 40 %→0.  `sub_getitem_r_id` same.
* **The three `mul_unsqueeze_reshape_*` pays were the ALiBiAttention
  mint** the oracle flagged: 1 fire each, all ill-typed, all
  suppressed.  paid 1→0, drops 13/8.7/4.3 %→0.
* **`eq_getitem_l_id` is the subtle one**: all four merged fires
  reconstruct well-typed — yet its single pay (SharedExpertMoE)
  extracted a member that does not denote, so the pick check
  suppressed it.  The per-fire classification reads the binding's
  min-size representative; the extractor reads the cheapest — where
  the two disagree, the pick check is the second net.
* **Typed-but-false stays paid** — correctly: `mul_select_l_id`'s 40
  mints all denote (the oracle says the *equality* fails, not the
  program).  The pay gate answers "does it denote"; truth answers
  "does it preserve".  Neither asks the other's question.
* **Conditional candidates still paying on well-typed sites**:
  `mul_unsqueeze_l_id` (2 paid, plus 3 ill fires elsewhere),
  `mul_unsqueeze_r_id`, `mul_slice_l_id`, `sub_unsqueeze_l_id`,
  `mul_transpose_l_id` — all kept their honest pays and remain
  unshipped on the truth gate, not the pay gate.  The nine-candidate
  "conditional+paying" set from `view-index-oracle.md` §3 is now
  five true-but-guarded pays and four proven inflations
  (`eq_getitem_l_id` + the three `mul_unsqueeze_reshape_*`).

## 4. Honest limits

* **Representative-member reconstruction.**  Per-fire classification
  resolves each bound e-class to its min-size member — the same
  resolution `check`/`derive` hooks use, but not bit-identical to
  the enode the extractor may pick.  The pick check (`best` itself)
  closes the residual gap: a fire that reads typed can still land an
  ill-typed member in the winner, and `paid_ill_typed` catches it
  (`eq_getitem_l_id`).  Conversely a mint could in principle be
  flagged ill under one representative while denoting under another
  — conservative by construction: *unresolvable ⇒ ill-typed*.
* **Eval is a sample, not a proof.**  The eval leg runs one random
  fp64 env; it exists to catch attr-out-of-range and tuple-operand
  mints the `%`-normalising shape rules miss.  An env that cannot be
  built (non-int leaf dims) abstains — the shape verdict stands.
* **Recorded-but-unmerged mints are not fires.**  An application
  that created enodes without merging is invisible to `rule_fires`
  and to the split — its minted sub-enodes persist in the graph but
  are not counted.  The pick check still flags them if they reach
  the winner.
* **Ambient ill-typedness is not attributed.**  `pick_ill` discounts
  bad subterms the input (or the base-ruleset extraction) already
  carried, and eval-invalid winners are only suppressed where a
  reference evaluated cleanly — a corpus term that does not evaluate
  cannot accuse a mint.
* **`paid` counts typed extractions; a typed extraction can still be
  *false*.**  Suppression is a denotation gate, not an equivalence
  gate — `verify_certificate` / `sink.verify` remain the equivalence
  gates.
* **The verdict schema is frozen.**  `fires_typed` /
  `fires_ill_typed` / `paid_ill_typed` / `ill_typed_cases` /
  `reach_ill` ride the report and the JSON dump only; a cached
  verdict row restores them at 0 (the `code_rev` key already makes
  pre-gate rows unreachable).  The stored `paid` column now carries
  the gated count.
* **Cost.**  The audit re-saturates once per *firing* (case,
  proposal) pair — the run went ~9 min → ~11 min.

## 5. What changed

* `tools/law_pipeline.py` — the typedness-audit block
  (`_bad_subterms`, `_evals`, `_term_typed`, `_pick_ill_typed`,
  `_subst_key`, `_app_typed`, `_TypedAudit`, `_typed_probe`);
  `_fire` splits fires and gates `changed`/`paid`/`verify_fail` on
  the pick's denotation; `_reach_row` adds `add_typed` and `_reach`
  suppresses ill-typed drops into `reach_ill`; `Evidence` gains
  `fires_typed` / `fires_ill_typed` / `paid_ill_typed` /
  `ill_typed_cases` / `reach_ill`; `shippable` requires
  `fires_ill_typed == 0`; `no_ship_reason` surfaces the ill-typed
  reasons; the ranked table gains an `ill` column; the JSON dump
  carries the new fields.
* `tests/test_law_pipeline_typed.py` — 12 tests: shape/eval
  predicates, the per-fire split (typed, ill-typed-by-broadcast,
  ill-typed-by-tuple-operand), paid suppression at the `_fire` and
  `_reach_row` levels, and the verdict strings.
* `tools/evidence.db` — the new run's verdicts recorded under the
  new `code_rev` (untracked local artifact; rows are keyed by
  content hash, so the pre-gate rows remain for comparison).

No `laws/*`, `attrs.py`, `torch_bridge.py`, `serialize.py`, or
`law_evidence.py` schema changes.  Nothing admitted — the corpus has
no true+typed+paying candidate.
