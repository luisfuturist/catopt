# Plan 0019 — ops as data (the structural slice)

Status: prototype landed.  `catopt_core.opdata` declares a *view /
relayout-class* op as data — name, arity, classification tags, attr
schema and a shape rule written in the **existing** shape-spec DSL —
and `declare_op` registers it into `catopt_core.opmeta`, the attr
schema and `catopt_core.typing`'s shape-rule registry, so a declared
op shapes, mints and matches through the same machinery a shipped op
does.  Kernel bodies, torch/aten spellings and cost entries stay
code, by design — this is the declaration seam for the structural
slice, not a general "define any op in JSON".  Retro:
`project/retros/ops-as-data.md`.

## Why

The architecture's claim is "the object language grows without
growing Python".  Plan 0018 made an op's *metadata* data
(`catopt_core.opmeta` — one classification table, 166 ops, named
projections).  But the op **set** and each op's **semantics** are
still code: a new primitive needs a Python shape rule, a torch
binding, a lowering.  This is the last hardcoded layer — the
HANDL-style frontier ("declare a new primitive", see
`project/retros/handl-analysis.md` §1.6: *does a tool reason about it
without knowing the runtime?*).

ADR 0004 (abstraction as a legal move) already established the
discipline for *laws*: an admissible abstraction is serializable data
the referee replays, never a minted Python hook.  An op is the other
half of the 2-cell — the object the law acts on.  This plan asks the
same question one level down: **what part of a primitive is
declarable data, and what is irreducibly code?**

## Inventory — what an op needs to be usable

Measured against the shipped vocabulary (166 registered ops).  For
each requirement: where it lives today, and whether it is data.

| # | Requirement | Where it lives | Kind |
|---|---|---|---|
| 1 | **name** (the lowering token) | `opmeta.REGISTRY` key | **data** (opmeta) |
| 2 | **arity** (operand count) | `opmeta._ARITY` | **data** (opmeta) |
| 3 | **classification tags** | `opmeta._CLASSIFIED` | **data** (opmeta) |
| 4 | **attr schema** (`{argN: canonical}`) | `attrs.ATTR_SCHEMA` | **data** (indexed by opmeta) |
| 5 | **required attrs** | `attrs.ATTR_REQUIRED` | **data** |
| 6 | **shape rule** | `typing._infer_op_shape` match arms + `typing._SHAPE_RULES` | **code** — *except the structural slice* |
| 7 | **torch/aten spelling** | `catopt_torch.torch_bridge._ATEN_TO_IR` | **code** (adapter boundary) |
| 8 | **lowering** (kernel body) | `torch_bridge._CORE_TORCH_BINDINGS` / carrier `TORCH_BINDINGS` | **code** |
| 9 | **cost entry** | `cost.basic._OP_FLOPS` (+ the `cost-view` tag) | **code** (FLOP formula) / **data** (the tag) |
| 10 | **carrier hook** | `ops.register_carrier`, `_CARRIER_MODULES` | **code** (packaging ops) |

Three buckets:

**(a) already data in `opmeta`** — 1–5.  Name, arity and tags are the
registry; the attr schema is already a positional
`{index: canonical-name}` table.  A declaration that supplies these
adds nothing new to the *model*, only to the *contents*.

**(b) declarable data — the structural slice** — 6, for view /
relayout-class ops.  The shape rules for
`unsqueeze`/`squeeze`/`select`/`slice`/`chunk`/`transpose`/`reshape`/
`getitem`/`bcast`/`mm-out`/`tail-block` **already exist as pure data**
in `catopt_core.laws.cond` — the shape-spec DSL its `cond`
predicates resolve (`("unsq-out", T, K)`, `("select-out", T, D)`,
`("slice-out", T, D, S, E, STEP)`, `("chunk-out", T, C, D)`,
`("transpose-out", T, D0, D1)`, `("reshape-out", T, NAME)`,
`("getitem-out", T)`, `("bcast", T, T)`, `("mm-out", T, T)`,
`("tail-block", T, NAME)`; a bare metavariable name is an operand
reference).  That DSL is *the* shape-spec DSL — this plan re-binds
its metavariables from law-pattern names to **operand positions**
(`"arg0"`/`"arg1"`) and its attr metavariables to the op's **attr
names** (`"dim"`), then compiles the spec into a `typing` shape rule.
A shape-preserving view is the spec `"arg0"`; `view_as` (output =
operand 1's shape) is `"arg1"`; `swapaxes` is
`("transpose-out", "arg0", "dim0", "dim1")`.

**(c) irreducibly code** — 7–10, plus the shape rules that are *not*
in the DSL.  A kernel body is a numeric implementation; a torch
binding is the adapter boundary (core imports no torch); a cost FLOP
formula is the evaluation dimension's own table (ADR 0003 keeps
evaluation independent); a carrier hook packages ops the core does
not own.  And several shipped shape rules are genuinely procedural —
`matmul` (rank promotion/demotion + batch broadcast), `conv1d`/
`conv2d` (window arithmetic), `index` (numpy advanced-indexing
layout), `linear` (column-bias squeeze), `einsum` (opaque equation),
the factory ops (`shape` attr verbatim).  Those stay match arms.

## Target design

`catopt_core.opdata` — an op *definition* as a value:

```python
@dataclass(frozen=True)
class OpDef:
    name: str
    arity: int | None = None
    tags: frozenset[str] = frozenset()
    attrs: Mapping[int, str] = {}      # {argN: canonical name}
    required: frozenset[str] = frozenset()
    shape: Any = None                  # a cond shape-spec, or None
```

`declare_op(defn)` is additive and reversible:

* **registry** — `opmeta.register_op_meta(OpMeta(name, arity, tags),
  attrs=…, required=…)` inserts into the live registry dict
  (`REGISTRY` is a `MappingProxyType` over it, so it reflects the
  addition) and into `ATTR_SCHEMA`/`ATTR_REQUIRED`.  Every accessor
  (`meta`/`ops`/`tags_of`/`arity`/`attrs`/`required`) sees the op with
  no change — they already read the live dicts.
* **mint contract** — because the declared schema joins `ATTR_SCHEMA`,
  `Op.make` enforces it: a positional `argN` at an undeclared position
  or a missing required attr raises at mint, exactly as for a shipped
  op.
* **shape rule** — the spec compiles into `typing.register_shape_rule`
  (a closure over the spec that binds `op.args`/`op.attrs` into the
  DSL's `bound` env and calls `cond._shape`).  `typing.shape_of` then
  answers the declared op with no change to `typing.py`.

`reset_declarations()` removes every declared op from the registry,
the attr schema and the shape-rule table, and returns the names — the
seam is process-global state, so a test (or a re-composition) undoes
it rather than leaking.

`opdef_to_data` / `opdef_from_data` are the JSON codec (tags sorted,
attr schema as `[[argN, name], …]`, the shape spec as nested lists) —
so a declaration *is* data the way a stored law is (`LAW_FORMAT`).

## Demo

`tests/test_opmeta_data.py` declares two real torch spellings that
are absent from the registry (verified by grep — no `swapaxes`,
`swapdims`, `view_as`, `ravel`, `atleast_*` anywhere in
`packages/`), then shows they are first-class:

* `view_as` — 2 operands, `{relayout, viewish}`, shape `"arg1"`;
* `swapaxes` — 1 operand, `{relayout, viewish}`, schema
  `{1: "dim0", 2: "dim1"}`, required `{dim0, dim1}`, shape
  `("transpose-out", "arg0", "dim0", "dim1")`.

It asserts: registry membership (`meta`/`ops`/`tags_of`/`arity`), shape
inference through `typing.shape_of`, mint validation (`Op.make` raises
on `arg5=…`, on a missing required attr), law matching
(`meta.match_pattern` binds a declared-op pattern against a declared-op
term), JSON round-trip, and clean teardown.

## Honest boundary

* **The frozen projections are the shipped vocabulary.**  `opmeta`'s
  `RELAYOUT_OPS` / `COST_VIEW_OPS` / … are `frozenset` snapshots
  computed at import.  A declared op is visible through the *live*
  accessors (`REGISTRY`, `meta`, `ops(tag)`, `tags_of`) but **not**
  through the frozen constants or the consumer sets bound to them
  (`signature._VIEW_OPS`, `cost.basic._VIEW_OPS`, …) — those are
  composed once at import.  Wiring a declaration into a consumer is a
  *recomposition* step (re-read `ops(tag)`, or re-import), not a live
  mutation.  This is the honest limit of the seam: it makes an op
  first-class in the *core vocabulary and machinery*, not in every
  already-bound consumer constant.
* **The kernel stays code.**  A declared op has no torch binding and
  no lowering — it shapes, mints and matches, but cannot be lowered or
  evaluated.  The seam declares the *object*, not its *execution*.
* **The torch/aten spelling stays in the adapter.**  `_ATEN_TO_IR` is
  the torch boundary (core imports no torch); `opmeta.validate_aten_map`
  is the check, not a move.
* **The shape rule is the DSL's, and inherits its strictness.**  The
  `cond` shape specs *decline* (resolve to `None`/unknown) where the
  hand-written `typing` arms sometimes return a partial shape or the
  `_INVALID` poison.  A declared reshape that mismatches numel
  therefore reads as *unknown*, not *ill-typed* — an honest difference,
  not a bug, and the reason the DSL covers the structural slice, not
  the whole dispatch.
* **Not thread-safe, not retroactive.**  A declaration mutates
  process-global state; declare before composing the engine.

## Follow-ups (out of scope here)

* Make the tag projections live views (or refresh-on-declare) so a
  declaration reaches consumer sets without a recomposition.
* A public shape-spec resolver in `catopt_core.laws.cond` — this
  prototype reuses its private `_shape`, which is the one coupling
  worth promoting to a seam.
* Extend the DSL (or add a second spec family) to cover `matmul`/
  `conv`/`index`/`linear` so a broader slice is declarable.
* Let a `TORCH_BINDINGS` entry ride alongside an `OpDef` so a
  declared op can also *lower* (the adapter-side half).
