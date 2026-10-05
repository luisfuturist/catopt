# Retro — de-hardcoding the op tables (plan 0018)

## What the pass did

The op *vocabulary* was hand-listed in a dozen modules.  Thirteen
duplicated sets became **named projections of one registry**
(`catopt_core.opmeta`), and a consistency test now pins the
relationships that used to be folklore.  No engine behaviour changed:
every projection is *byte-identical* to the literal it replaced
(verified before the literals were removed).

## What became data

| Concept (tag) | Consumers now reading the registry | Ops |
|---|---|---|
| `relayout` | `signature._VIEW_OPS` | 37 |
| `cost-view` | `cost.basic._VIEW_OPS` | 11 |
| `generator-view` | `pipeline._VIEW_OPS` | 7 |
| `viewish` | `oracle._VIEWISH` | 17 |
| `reduce-dim` | `oracle._REDUCTION_DIM_OPS` | 19 |
| `pointwise` | `signature._POINTWISE_OPS` | 32 |
| `pointwise-unary` | `layout._POINTWISE_UNARY` | 12 |
| `pointwise-binary` | `layout._POINTWISE_BINARY`, `pipeline._POINTWISE` | 4 |
| `activation` | `signature._ACT_OPS` | 8 |
| `write` | `signature._MUT_OPS` | 8 |
| `table` | `signature._TABLE_OPS` | 10 |
| `commutative` | `typing._COMMUTATIVE_BROADCAST` | 4 |

The registry itself is **166 ops**, composed from three declared
sources: the attr schema (`ATTR_SCHEMA`, indexed not copied), the
classification table (`_CLASSIFIED`, keyed by op), and a short
IR-only vocabulary (`_IR_ONLY_OPS`).

The `pipeline._VIEW_OPS` vs `signature._VIEW_OPS` inconsistency is
resolved by making both a projection of the one `relayout`/`generator-view`
source, with the subset relation machine-checked
(`GENERATOR_VIEW_OPS ⊆ RELAYOUT_OPS`).  The two are deliberately *not*
one identical object — see the honest boundary below.

## What stayed code, and why

* **Single-source tables.**  `ATTR_SCHEMA` / `ATTR_REQUIRED`,
  `cost.basic._OP_FLOPS`, `typing._SHAPE_RULES`, `_CARRIER_MODULES`
  each already live in exactly one place.  Moving them would add
  indirection, not remove duplication — so they stayed, and the
  registry *indexes* the schema rather than copying it
  (`om.attrs("transpose") is ATTR_SCHEMA["transpose"]`).
* **Numeric kernel bodies.**  The `_flops_of` arms (`matmul`,
  `conv2d`, `trace`), the torch bindings and the shape-rule bodies are
  implementations, not data.  Only their *metadata* moved.
* **Evaluation-dimension tables.**  `_FUSION_TRANSPARENT_OPS`,
  `_FUSION_POINTWISE_OPS`, `_SOLVER_OPS`, `_MEASURED_GATHER_OPS` are
  the cost model's own axes (ADR 0003 keeps evaluation independent);
  they stay with the cost model.
* **The aten alias map.**  `torch_bridge._ATEN_TO_IR` is the torch
  boundary — core imports no torch — so it stays in the adapter and is
  **validated against** the registry (`validate_aten_map`), never
  moved.
* **Derived carrier subsets.**  `trace_lift._BASE_VIEW_OPS` and
  `scan_lower._VIEW_OPS` are relayout *minus* the indexing ops, with a
  documented rationale; expressing the subtraction as a tag is a
  follow-up, not this pass.
* **The generator inventory.**  `meta_game._ARM_NAMES` /
  `ENUMERATION_ORDER` / `CORPUS_ARMS` enumerate *generators*, not ops —
  a different kind of inventory.

## Consistency tests added (`tests/test_opmeta.py`, 14 tests)

* every projection is exactly the registry's tag filter;
* `ATTR_SCHEMA ⊆ REGISTRY`, and `attrs` / `required` are the *same
  object* as the schema entries (one home for the attr half);
* `GENERATOR_VIEW_OPS ⊆ RELAYOUT_OPS` and `VIEWISH_OPS ⊆ RELAYOUT_OPS`;
* `set(pipeline._VIEW_OPS) ⊆ set(signature._VIEW_OPS)` — the measured
  inconsistency, now a checked relation;
* each consumer set **is** the registry projection (identity, not a
  copy) — the anti-drift guarantee;
* defining-property arities hold (generator/viewish/reduce/unary = 1,
  binary = 2, relayout ∈ {1, 2});
* `validate_aten_map` is clean over the adapter's alias map, so the
  exporter cannot mint an unregistered IR op.

## Gates

`ruff check`, `ruff format --check`, `ty check`, `vulture`,
`lint-imports`, `bandit`, `semgrep` (committed config), the radon
ratchet, and the touched test files (opmeta, layout, vocab, pipeline,
structure, cost, typing, regime, oracle, morphisms, property, carrier
lift — 1100+ tests) all pass.  `opmeta.py` is at **100 %** statement
and branch coverage.

## Honest boundary

The two `_VIEW_OPS` are **one source, not one object**.  The generator
uses a *narrower* view alphabet than the morphisms signature
deliberately — forcing identity would make the discovery generator
propose naturality over `getitem`/`permute`/dtype-casts, a real
behaviour change to the engine, not a de-hardcoding.  The registry
makes the subset relation explicit and machine-checked; the plan
records why equality is the wrong target.  This is the same
"de-duplicate, do not silently redefine" discipline the pass applies
everywhere.
