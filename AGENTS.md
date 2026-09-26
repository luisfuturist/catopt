# Agent guide — catopt

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Monorepo layout: the domain packages
live under `packages/` (`catopt-core` — the torch-free engine;
`catopt-torch` — PyTorch adapters; `catopt-carriers` — carrier
laws/executors; `catopt-eps` — opt-in approximation toolkit;
`catopt-optimize` — pipeline orchestrators); `catopt/` is the façade
+ compat aliases (every historical `catopt.X` import path resolves to
its new home via `sys.modules` aliases). Tests in
`tests/`. The dev virtualenv is `.venv/` (uv-managed).

## Verification commands

Run these before finishing any change; all must pass.

```sh
uv run pytest                 # full test suite (pytest-xdist enabled)
.venv/bin/pyright             # typecheck — 0 errors required (warnings OK)
.venv/bin/ruff check          # lint
.venv/bin/ruff format --check # formatting
coverage run --source=catopt_core,catopt_torch,catopt_carriers,catopt_eps,catopt_optimize,catopt -m pytest tests/ -q
coverage report -m                                              # coverage (fail_under=100)
```

Pre-commit (`pre-commit install`) runs ruff check/format and pyright on
`packages/**/*.py` + `catopt/`.

Known drift: the installed ruff (0.16.9) flags pre-existing isort /
format differences across `tests/` and `bench/` that predate any
current change. `ruff check packages catopt` is the meaningful gate
(0 errors); repo-wide `ruff check` and `ruff format --check` are not
clean at `HEAD` under 0.16.9.

## Typecheck ratchet

- Checker: **pyright** (`[tool.pyright]` in `pyproject.toml`,
  `typeCheckingMode = "standard"`, `include = ["packages","catopt"]`, `extraPaths` per-package `src/`).
- Installed in `.venv` via the `dev` dependency group
  (`pyright>=1.1.380`; currently 1.1.414).
- `exclude` in `pyproject.toml` lists files that predate the ratchet —
  each entry documents its baseline error count. **Remove entries as
  files get annotated**; never add new ones. Excluded files are still
  analyzed when imported by checked files.
- New modules under `packages/*/src/` are checked automatically — keep them
  clean at `standard` strictness.
- Baseline was green at 0 errors / 2 warnings (`calibrate.py`
  `reportUnusedExpression`); warnings must not regress to errors.

## Ports (hexagonal boundary)

`catopt_core.ports` names every adapter contract as a
`@runtime_checkable` Protocol.  The whole-graph ends are `Source`
(`model -> (IR, leaves)`) and `Sink` (`IR -> runnable`, plus its
`supported_ops` set and the module-level equivalence gate); the per-op
port is `Binding` (`TorchBinding` is an alias of the same object).
`optimize_model` / `discover_alternatives` take `source=` / `sink=`
(defaults: `catopt_torch.adapters.TorchSource` / `TorchSink`).
Extraction is priced through
`catopt_core.cost.backend_cost(cost_fn, sink.supported_ops)`, so the
search only selects forms the backend can lower.  A new backend
implements `Sink`; nothing in `catopt-core` changes.

## Conventions

- Line length 72 (ruff). E501/B008/SIM108 intentionally ignored — see
  `[tool.ruff.lint]` for rationale.
- Coverage floor: `fail_under = 100` in `[tool.coverage.report]` —
  the suite is pinned at 100%; new branches need tests (or a
  justified `pragma: no cover`).
- Tests allocating CUDA tensors use the `requires_cuda` marker
  (auto-skipped when CUDA is absent).
