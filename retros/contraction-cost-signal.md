# Contraction ordering in the diagram space — can it even be priced?

Plan 0016 follow-up.  The contraction-ordering precondition
(`contraction-precondition.md` §3a) claimed the diagram
contraction-move space carries **no cost signal** for a matrix-chain
ordering: `search_moves` ties greedy on every chain, because the
reified composed weight is a *param-only* subtree and the shipped
cost models discount it to zero.

This retro asks the harder prerequisite question for any learned
contraction-ordering policy:

> Can catopt *express* a contraction-order choice with a cost signal
> at all — and if not, what is the right substrate?

**Answer: it can express the choice, but the diagram space cannot
price it — and it should not.**  The order signal genuinely lives in
a *different* substrate (activation-tensor contraction), which catopt
already has in the e-graph rule space.

Reproduce: `.venv/bin/python tools/contraction_cost_probe.py`
(<~5 s, CPU-only, torch-free, seeded).

## 1. The claim reproduces — with the exact code path

The diagram window reify
(`catopt_orchestrator.morphisms.reify._reify` over a `linear` chain)
folds `linear(linear(x, W0), W1, …)` into
`linear(x, matmul(W4, matmul(W3, … matmul(W1, W0))))`.  The composed
weight is a pure `Param` subtree; the only data input (`x`) sits
*above* it.  Verified directly on the shipped diagram driver:

```
reorder_compose:blocks.0+…+blocks.4  -> chain+chain+chain+chain
  reified: (linear x, (matmul W4, (matmul W3, (matmul W2,
           (matmul W1, W0)))))
  diagram_search: initial_cost=2485.0  final_cost=129.0
```

`129 = 2·4·2·8 + 1 launch` — one activation GEMM, no order term.

### The zeroing is at two sites, both keyed on `_folds_to_param`

**Site A — `catopt_core.cost.dag_cost`** (`cost/params.py:448-454`).
`rec` skips any op subtree that has no `Var` leaf and folds:

```python
if (not has_var(t) and not bill_params
        and _folds_to_param(t, None, fold_memo)):
    return  # folds at compile time — free at runtime
```

**Site B — `EGraph.extract_best`** (`egraph/extract.py:515-523`).
The class-level `local` is zeroed for a param-only class:

```python
param_only = param_only and node.op in _FOLDABLE_OPS
local = max(cfn(term) - child_tc, 0.0)
if param_only and not bill_params:
    local = 0.0  # whole subtree folds at compile time
```

Both read `bill_params` from the model's `charges_param_only` marker
(`cost/params.py:409`, `egraph/extract.py:305`).

### Measured: the order is invisible

The probe prices the same product three ways under every shipped
model (dims `2,3,14,14,3,8`):

| term | flops_cost | launch_aware | count_cost | folds? |
|---|---|---|---|---|
| product, left-nested | 0.0 | 0.0 | 0.0 | yes |
| product, right-nested | 0.0 | 0.0 | 0.0 | yes |
| product, balanced | 0.0 | 0.0 | 0.0 | yes |
| `linear(x, left)` | 32.0 | 33.0 | 1.0 | — |
| `linear(x, right)` | 32.0 | 33.0 | 1.0 | — |
| `linear(x, balanced)` | 32.0 | 33.0 | 1.0 | — |

Raw (no fold) the signal is obvious — `left` 4576, `right` **1216**
(the DP optimum ×2), `balanced` 4424 — but `dag_cost` erases it.  The
product's FLOPs are removed *twice over*: the subtractive per-node
decomposition subtracts the child, and the fold skips it entirely.

### Worse: the space is *expressible* but not *priceable*

Saturating the product with `assoc_matmul` / `assoc_matmul_rev` puts
**all 6 bracketings in the root e-class** — so the diagram space
*can represent* the order.  But every bracketing prices at 0, so
`extract_best` resolves the tie by member order, which is arbitrary.
Over 40 random chains (seeded with the diagram's natural right-nested
fold):

* shipped models pick a **suboptimal bracketing on 37/40** chains,
  worst **10.7×** the DP optimum;
* the diagram's window reify delivers whatever the natural fold
  produced, and saturation + extraction **never improves it** (the
  cost model cannot see the difference).

So the diagram does not merely *fail to reward* a good order — it
delivers a bad one ~92 % of the time and cannot tell.

## 2. Is it fixable, and is the fix sound?

Three options, each evaluated.

### (a) Price the composed weight product (bill param-only folds)

Change `bill_params` so the fold's FLOPs are charged.  **Measured
blast radius: 31 test failures across 15 files.**  The fold discount
is load-bearing — it is *what lets extraction prefer rewrites that
move computation onto the weights* (`egraph/extract.py:254-256`:
"`x@(W1@W2@W3)` even though the weight product itself is not free in
FLOP terms").  Disabling it breaks, among others:

* the fusion/pairing wins — `test_pairing_pass_five_way_parallel_block`,
  `test_swiglu_fuse_*`, `test_qkv_fuse_produces_single_gemm`,
  `test_gqa_asym_fuse`, `test_transformer_block_stacks_fusions`;
* the extraction-choice tests — `test_lora_flop_arithmetic`,
  `test_matmul_assoc_picks_materialized_high_rank`,
  `test_score_scale_folds_into_wq`, `test_ir_module_fuses_weight_chain`;
* the diagram/morphism windows — `test_diagram.py` (4),
  `test_morphism_multiinput.py` (3), `test_morphisms.py` (2),
  `test_compositional_pairs.py` (5);
* the cost-model contract tests — `test_dag_cost_param_only_discount`,
  `test_fused_param_fold_is_not_pointwise`,
  `test_param_only_subtrees_bill_no_traffic`.

**Not sound as a silent change.**  And note it would be *wrong* for
runtime: the folded weight is materialised **once** at compile time,
so the chain's steady-state runtime really is one GEMM regardless of
the order.  Charging the product per-forward misprices runtime.  The
discount is not a bug; it is correct for a runtime model.

### (b) A dedicated contraction cost model

The lever exists: a `CostFn` with `charges_param_only = True` makes
both sites bill the fold.  Measured on the *product-only* extraction,
such an opt-in model recovers the **DP optimum on 40/40** chains
(vs 37/40 suboptimal under the shipped models).

But it is **not sound as a drop-in diagram cost model**:

* Applied to the *whole* program extraction it re-ranks
  folded-vs-unfolded forms, not bracketings — it picked the
  *unfolded* 5-GEMM chain (620) over any folded form, because it now
  prices compile-time weight work as if it were runtime.
* It only cleanly prices the order when the model is pointed at the
  **weight product in isolation** (or a tensor-network IR) — not at
  the diagram's runtime joint, which is what `ContractionSearch`
  extracts.
* It prices the **compile-time weight-materialisation** axis, which is
  a *different question* from runtime contraction ordering.  Pricing
  it would make the diagram optimise one-time weight folding, not
  inference cost.

So a dedicated model *can* price the order, but only by answering a
different question on a different axis, and only when isolated from
the joint.  It is not a clean fix.

### (c) Leave it alone — declare contraction ordering out of scope

**Sound and correct.**  In the diagram's spelling a matrix chain is
`Param` weights composed against a single activation `Var`.  The
weight product folds once at compile time; runtime is order-independent
by construction.  This is a **fusion** space, and the fold discount is
the *right* price for it.  A runtime contraction-ordering signal cannot
live here.

## 3. Verdict + recommendation

**Can a contraction-order choice be priced today?  No — not in the
diagram space, and not by reusing the fold-aware models.**  The choice
is *expressible* (all bracketings are in the e-graph) but not
*priceable*: the shipped models zero it, extraction is arbitrary and
suboptimal (37/40), and the fold discount is load-bearing (31 tests)
and correct for runtime.  Neither (a) nor (b) is sound as a runtime
diagram cost model.

**The right substrate is a tensor-network-level IR where each node is
a tensor and the cost is the contraction product** — i.e. products
over *activation* (`Var`) leaves, which are **not** folded and so are
priced by `flops_cost` directly.  catopt **already has this**: the
e-graph rule space over `Var`-leaf chains (`assoc_matmul`), where
`contraction-precondition.md` §3b and `contraction-ladder.md` show
saturation reaches the exact DP optimum (60/60) and the one-step
oracle loses 37 % (worst 4.1×).  That is where a learned
contraction-ordering policy has a priced state/action/reward — not
the diagram move space.

**Recommendation.**  Do not touch the fold discount.  Record
contraction ordering as **out of scope for the diagram move space**
(it is a fusion space; its runtime cost is order-independent by
construction), and route any contraction-ordering policy to the
`Var`-leaf e-graph rule space, which is already priced and already
optimal under saturation.  If a *compile-time weight-folding* cost
model is ever wanted (a legitimate but different axis), build it as a
separate opt-in `CostFn` over the weight product — never by widening
the runtime discount.

## Honest limitations

* **One instance family.**  §1's sweep is matrix chains (4–8
  matrices).  The diagram has no general tensor-network spelling, so
  the conclusion is measured on chains only — but the structural
  argument (param-only product → fold → 0) is spelling-independent.
* **"Suboptimal" is the compile-time axis.**  The 37/40 and 10.7×
  figures are the weight-product *materialisation* FLOPs.  Runtime is
  unaffected; that is the whole point.
* **The opt-in model was probed, not shipped.**  (b)'s 40/40 is a
  scratch measurement of an opt-in `charges_param_only` FLOP model;
  no production code was changed.
* **Greedy vs search is not re-litigated.**  The precondition retro's
  0/12 tie stands; this retro explains *why* at the cost-model level.

## Reproduce

```sh
.venv/bin/python tools/contraction_cost_probe.py
# the diagram driver's reified term (needs torch, ~seconds):
.venv/bin/python -c "
from tools.contraction_precondition import _chain_model, diagram_chain_costs
print(diagram_chain_costs((2,3,14,14,3,8)))
# -> {'initial': 2485.0, 'greedy': 129.0, 'search': 129.0}"
```
