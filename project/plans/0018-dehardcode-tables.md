# Plan 0018 — de-hardcoding the op tables

Status: stages 0–4 landed — `catopt_core.opmeta` is the single source
and the thirteen duplicated sets are projections of it; the carrier
subsets and the evaluation-dimension tables stay code by design (see
the boundary).  Retro: `project/retros/dehardcode.md`.  The
architecture's claim is "objects/laws/guards are data"; the op
*vocabulary* is still hand-maintained in a dozen places.  This plan
gives the op metadata one home and makes the duplicated lists derive
from it.

## Thesis — de-hardcode means de-*duplicate*

Not every table is a leak.  Three kinds of table exist:

1. **Single-source, already data.**  `catopt_core.attrs.ATTR_SCHEMA`
   (the canonical positional-attr names), `cost.basic._OP_FLOPS` (the
   fallback FLOP weights), `typing._SHAPE_RULES` (the carrier shape
   registry), `_CARRIER_MODULES` (the carrier registration tuple).
   These live in exactly one place and are already the
   single-source-of-truth.  **They stay where they are** — moving them
   would add indirection, not remove duplication.
2. **Duplicated op sets.**  The same *concept* (view op, pointwise
   op) hand-listed in several modules, drifting apart.  These are the
   leak: a new op cannot be added without editing N files, and the
   copies silently disagree (`pipeline._VIEW_OPS` vs
   `signature._VIEW_OPS` is the measured example).  **These get one
   home.**
3. **Kernel bodies.**  The per-op numeric *implementations* —
   `_flops_of`'s `matmul`/`conv2d`/`trace` arms, the torch bindings,
   the shape-rule bodies.  Code, not data, and legitimately so.  The
   *metadata* around them (which op, what arity, what class) becomes
   data; the body stays code.

## Inventory (measured)

Duplicated / hardcoded op sets found in `catopt-core`,
`catopt-orchestrator`, `catopt-discovery`:

| # | Table | File:line | Kind | Disposition |
|---|---|---|---|---|
| 1 | `_VIEW_OPS` | `catopt_core/cost/basic.py:97` | zero-kernel cost views | → registry `cost-view` |
| 2 | `_VIEW_OPS` | `catopt_orchestrator/morphisms/signature.py:134` | transparent relayout | → registry `relayout` |
| 3 | `_VIEW_OPS` | `catopt_discovery/pipeline.py:213` | generator naturality alphabet | → registry `generator-view` |
| 4 | `_VIEWISH` | `catopt_discovery/oracle.py:96` | oracle-instantiable views | → registry `viewish` |
| 5 | `_REDUCTION_DIM_OPS` | `catopt_discovery/oracle.py:562` | axis-or-tuple reductions | → registry `reduce-dim` |
| 6 | `_POINTWISE_OPS` | `catopt_orchestrator/morphisms/signature.py:179` | multi-operand pointwise | → registry `pointwise` |
| 7 | `_ACT_OPS` | `catopt_orchestrator/morphisms/signature.py:62` | spine activations | → registry `activation` |
| 8 | `_MUT_OPS` | `catopt_orchestrator/morphisms/signature.py:117` | arg0-mutating writes | → registry `write` |
| 9 | `_TABLE_OPS` | `catopt_orchestrator/morphisms/signature.py:218` | lookup-table roles | → registry `table` |
| 10 | `_POINTWISE_UNARY` | `catopt_core/laws/layout.py:249` | unary commutation family | → registry `pointwise-unary` |
| 11 | `_POINTWISE_BINARY` | `catopt_core/laws/layout.py:265` | binary commutation family | → registry `pointwise-binary` |
| 12 | `_POINTWISE` | `catopt_discovery/pipeline.py:210` | generator pointwise alphabet | → registry `pointwise-binary` |
| 13 | `_COMMUTATIVE_BROADCAST` | `catopt_core/typing.py:117` | operand-symmetric ops | → registry `commutative` |

Adjacent sets **deliberately left as code** (documented boundary —
they are narrower *derivations* of a concept, or a different axis):

| Table | File:line | Why it stays |
|---|---|---|
| `_FUSION_TRANSPARENT_OPS` | `cost/fusion.py:45` | compiled-lowering plumbing; adds carrier *packaging* ops (`aff`/`om`/`omd`/`affd_*`) that are not IR views — a different concept |
| `_FUSION_POINTWISE_OPS` | `cost/params.py:355` | the fusion model's own union; a cost-model axis, not the op classification |
| `_SOLVER_OPS`, `_MEASURED_GATHER_OPS` | `cost/fusion.py:89`, `cost/roofline.py:233` | cost-model tie-breaks, evaluation-dimension data |
| `_SPLIT_VIEW_OPS` | `orchestrator/diagram.py:967` | the diagram substrate's narrow split-view subset |
| `_ELEMWISE`, `_DIM_OPS` | `laws/headshare.py:137` | a law module's local precondition set |
| `_BASE_VIEW_OPS` | `carriers/trace_lift.py:149` | a *derived subset* of relayout (indexing ops removed) with documented rationale |
| `scan_lower._VIEW_OPS` | `carriers/scan_lower.py:83` | relayout + the `t` spelling; carrier-local |
| `_ATEN_TO_IR` / `_IR_TO_TORCH_EXTRA` | `catopt_torch/torch_bridge.py:28` | the aten→IR alias map; **validated against** the registry at the boundary, never moved into core (core is torch-free) |
| `ATTR_SCHEMA` / `ATTR_REQUIRED` | `catopt_core/attrs.py:73` | already single-source; the registry *indexes* it |
| `_OP_FLOPS` | `cost/basic.py:48` | already single-source; evaluation-dimension data |
| `_ARM_NAMES` / `ENUMERATION_ORDER` / `CORPUS_ARMS` | `discovery/meta_game.py:1407` | a *generator* inventory, not an op vocabulary |

## Target design

`catopt_core.opmeta` — one declarative table keyed by op name:

```python
@dataclass(frozen=True)
class OpMeta:
    name: str
    arity: int | None          # operand count; None = undeclared
    tags: frozenset[str]       # the classification
    attrs: Mapping[int, str]   # ATTR_SCHEMA[op] (indexed, not copied)
    required: frozenset[str]   # ATTR_REQUIRED[op] (indexed)
```

* The registry is **seeded from `ATTR_SCHEMA`** (so every schema'd op
  is registered, and the attr half has one source) and **overlaid with
  the classification table** (`_CLASSIFIED: dict[str, frozenset[str]]`)
  and an explicit IR vocabulary.
* A small, closed **tag vocabulary** names each concept once
  (`relayout`, `cost-view`, `generator-view`, `viewish`, `reduce-dim`,
  `pointwise`, `pointwise-unary`, `pointwise-binary`, `activation`,
  `write`, `table`, `commutative`).
* Every consumer set becomes a **named projection**:
  `opmeta.ops(TAG)` / the module constants `RELAYOUT_OPS`,
  `COST_VIEW_OPS`, …  The module keeps its public name (`_VIEW_OPS`)
  as a thin binding, so call sites do not move.
* `opmeta.validate_aten_map(mapping)` is the boundary check: every
  aten target the exporter can mint must be a registered IR op.  It is
  *called by a test*, not by the bridge (core must not import torch).

## Migration order (each step green)

| Step | What | Gate |
|---|---|---|
| 0 | `catopt_core/opmeta.py` — registry + tags + projections + `validate_aten_map` | unit tests |
| 1 | `cost/basic._VIEW_OPS` ← `COST_VIEW_OPS`; `typing._COMMUTATIVE_BROADCAST` ← `COMMUTATIVE_OPS`; `laws/layout._POINTWISE_*` ← projections | cost/typing/layout tests |
| 2 | `orchestrator/morphisms/signature._VIEW_OPS` / `_POINTWISE_OPS` / `_ACT_OPS` / `_MUT_OPS` / `_TABLE_OPS` ← projections | morphisms tests |
| 3 | `discovery/pipeline._VIEW_OPS` / `_POINTWISE`, `discovery/oracle._VIEWISH` / `_REDUCTION_DIM_OPS` ← projections | discovery tests |
| 4 | consistency tests (`tests/test_opmeta.py`): projections agree, the two `_VIEW_OPS` share one source, the bridge's IR targets are registered | full opmeta + discovery + morphisms |

## Consistency tests added

* every projection is exactly the registry's tag filter (no drift);
* `GENERATOR_VIEW_OPS ⊆ RELAYOUT_OPS` — the two `_VIEW_OPS` are one
  *source* (the generator uses a documented subset, and the subset
  relation is now machine-checked rather than folklore);
* `VIEWISH_OPS ⊆ RELAYOUT_OPS`;
* defining-property arities hold (generator/relayout/viewish/reduce =
  1, binary = 2, unary = 1);
* every `ATTR_SCHEMA` op is registered, and `opmeta.attrs(op)` is the
  same object as `ATTR_SCHEMA[op]`;
* `validate_aten_map(torch_bridge._ATEN_TO_IR)` is clean.

## Honest boundary

* The two `_VIEW_OPS` are **not** identical objects, and forcing them
  to be would change the generator's behaviour (it would start
  proposing naturality over `getitem`/`permute`/`dtype`-casts).  They
  are one *source* with a checked subset relation — the honest
  resolution, recorded here rather than papered over.
* The registry is **torch-free**.  The aten alias map stays in the
  adapter and is validated, not moved.
* Numeric kernel bodies (`_flops_of` arms, bindings, shape-rule
  bodies) stay code; only their *metadata* is data.
* Carriers' derived subsets (`_BASE_VIEW_OPS`, `scan_lower._VIEW_OPS`)
  stay local for now — deriving them from the registry is a follow-up
  once their subtraction rationale is expressible as a tag.
