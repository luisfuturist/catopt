# Retro — ops as data (plan 0019)

**What landed.** `catopt_core.opdata` — the declaration seam for a
view/relayout-class op.  An `OpDef` (name, arity, tags, attr schema,
shape spec in the existing `cond` DSL) serializes to JSON-safe data;
`declare_op` registers it into the live `opmeta.REGISTRY`, the attr
schema (`ATTR_SCHEMA`/`ATTR_REQUIRED`, so `Op.make` enforces it at
mint) and `typing._SHAPE_RULES` (via `register_shape_rule`), so a
declared op shapes, mints and matches through shipped machinery.
`reset_declarations` undoes it — tests do not leak process-global
state.  `opmeta` gained the registry half (`register_op_meta`,
`reset_declared`, the live `_REGISTRY` dict behind the mapping
proxy); declaring over a shipped name is a `ValueError`.

**What is declarable, measured.** Of the ten things an op needs to be
usable (plan 0019's inventory), five were already data (name, arity,
tags, attr schema, required attrs); the shape rule is now data *for
the structural slice* — any op whose output shape the `cond` DSL can
express (`"arg0"`, `("transpose-out", "arg0", "dim0", "dim1")`, …).
Kernel bodies, aten spellings, cost formulas and carrier hooks stay
code — the honest boundary the plan states.

**Known limits (all in the plan, kept honest).** The `*_OPS`
projections are import-time snapshots — a declared op is in the live
registry but not in the shipped sets; a DSL shape spec *declines* to
`None` where a hand-written arm would return the `_INVALID` poison;
declaration is not thread-safe or retroactive.

**Verification.** `tests/test_opmeta_data.py` (16 tests): declare a
`view_as`-style op end to end (mint → shape → match), the codec
round-trip, shadowing a shipped name refused, reset restores the
shipped state.  `opdata.py` and the touched `opmeta.py` lines are at
100% under the pinned coverage run.
