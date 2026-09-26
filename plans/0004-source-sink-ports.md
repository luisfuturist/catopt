# Plan 0004 — Source/Sink ports: pluggable graph frontends and backends

Status: implemented (this branch)
Date: 2026-09-26
Branch: `project` (orphan — plans live here, not on main)
Follows: `0003-monorepo-domain-packages.md`

## Goal

`catopt-core` is torch-free, but the *whole-graph* boundary was not
named: `optimize_model` hard-wired `export_to_ir` (torch.export →
ATen → IR) on the way in and `ir_to_torch_module` / `verify_module` on
the way out.  The ports layer named the per-op (`Binding`), executor,
verifier and cost contracts but never the two ends of the pipeline.

This plan names them — `Source` (`model -> (IR, leaves)`) and `Sink`
(`IR -> runnable`, plus its supported-op set and equivalence gate) —
and makes extraction **backend-relative**: a sink declares the ops it
can lower, and members outside that set price at `+inf`, so the search
commits only to forms the backend can execute.

Inductor is untouched: it was never a dependency, only the benchmark
baseline.

## Design

1. **`Source` / `Sink` protocols** in `catopt_core.ports` (stdlib only,
   `TYPE_CHECKING` torch refs).  `Sink` carries three members:
   - `supported_ops: frozenset[str]` — the backend's lowerable op set;
   - `ops: OpRegistry` — the lowering table const folds dispatch
     through;
   - `lower(ir, params) -> Executor` and
     `verify(ref, opt, inputs, *, rtol, atol) -> VerifyReport` — the
     module-level runtime and gate.
2. **`Binding`** — `TorchBinding` renamed to the backend-neutral
   spelling, with `TorchBinding` kept as an alias (same object) so
   existing annotations and `isinstance` checks are untouched.
3. **`backend_cost(cost_fn, supported_ops)`** in `catopt_core.cost` —
   wraps any `CostFn`, returning `+inf` for a term using an op outside
   the set.  Preserves `charges_param_only` / `dag_exact` / `profile`
   markers (so `dag_cost` and param-bytes pricing are unchanged) and
   forwards `memo` adaptively, matching `_memo_dispatch`.
4. **`catopt_torch.adapters`** — `TorchSource` (delegates to
   `export_to_ir`) and `TorchSink(ops=None)` (delegates to
   `ir_to_torch_module` / `report.verify_module`; `supported_ops` is
   the op table's key set).
5. **`optimize_model(..., source=None, sink=None)`** and
   `discover_alternatives(..., source=None, sink=None)` — default to
   the torch pair; price extraction through
   `backend_cost(cost_fn, sink.supported_ops)`; lower via
   `sink.lower`; verify via `sink.verify`.  The `ops` parameter still
   selects the default sink's table.

## Acceptance

- [x] `Source` / `Sink` protocols; `TorchSource` / `TorchSink` conform.
- [x] `backend_cost` bounds the reachable class; markers preserved.
- [x] `optimize_model` / `discover_alternatives` take `source` / `sink`.
- [x] A non-torch `NumpySink` runs the full extract → lower → verify
      path (`tests/test_pluggable_sink.py`) — the seam is real.
- [x] 1,609 tests green; coverage 100% line+branch.
- [x] pyright 0 errors / 2 warnings (baseline).
- [x] `ruff check packages catopt` clean.

## Notes

- **Verification is sink-owned.**  `Sink.verify` is module-level
  (`ref` / `opt` are runnables in the sink's runtime); the torch sink
  delegates to `report.verify_module`.  `optimize_model`'s built-in
  verbose check now routes through the sink, so the historical
  monkeypatch seam moved from `catopt.optimize.verify_module` to
  `catopt.adapters.verify_module`.
- **Two sink tiers are explicit.**  Core-IR ops are portable; the
  carrier lowerers (`scan_lower` / `om_lower` / `omd_lower`) are
  torch-specific, and a generic backend simply omits those ops from
  `supported_ops` — the carrier lifts then never win extraction for it.
- **The NumPy sink is a conformance double, not a shipped package** —
  it lives in `tests/` and covers the core op set only.
- **Pre-existing tooling drift (not introduced here).**  The installed
  ruff (0.16.9) disagrees with the committed style across `tests/` and
  `bench/` (isort grouping of first-party imports, plus format
  rewraps) — `ruff check`/`format --check` were already failing
  repo-wide at `HEAD`.  `packages/` + `catopt/` stay clean; a
  follow-up should pin ruff or add
  `[tool.ruff.lint.isort] known-first-party` and re-format once.
