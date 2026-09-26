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
.venv/bin/vulture             # dead code (uses [tool.vulture] paths)
.venv/bin/lint-imports        # hexagonal boundary contracts
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

## Dead-code + architecture linting

Two dev-only ratchets (both in the `dev` dependency group, both
configured in `pyproject.toml`):

- **vulture** (`[tool.vulture]`, `min_confidence = 80`) flags
  unreachable / unused code across `packages` / `catopt` / `tests`.
  The suite is pinned at 100% coverage, so a genuine dead branch is a
  real finding, not noise.  Run `.venv/bin/vulture` (uses the
  configured `paths`); a non-zero exit (3) means dead code.
- **import-linter** (`[tool.importlinter]`) pins the hexagonal boundary:
  a `forbidden` contract makes `catopt_core` importing `catopt_torch` /
  `catopt_carriers` / `catopt_eps` / `catopt_optimize` — or the `torch`
  / `numpy` external packages — a hard error.  Run
  `.venv/bin/lint-imports`.  Core is a *sink* for adapter-pushed state
  (see `catopt_core.ops` *Backend wiring*), never a puller: the adapter
  registers its tables into core, core never imports the adapter.

## Differential oracle (opt-in, test-only)

`tests/test_egglog_oracle.py` cross-checks catopt's hand-rolled
equality-saturation search against the Rust **egglog** library on a
small op/law subset.  It is an **opt-in, test-only oracle** — not a
production dependency and not an engine swap: `catopt-core` stays
pure-Python / zero-dependency, and nothing under `packages/` or
`catopt/` imports `egglog`.  The ported prototype and its documented
limitations (untyped-by-shape terms, unconditional `check` rewrites, no
proof replay) live in `tests/egglog_oracle.py`.

`egglog` is in a dedicated `oracle` dependency group — deliberately
**not** the default `dev` group — so the repo stays light (egglog is a
compiled Rust extension).  The test begins with
`pytest.importorskip("egglog")`, so the whole file skips cleanly when
the group is absent and the rest of the suite stays green.

Run it explicitly:

```sh
uv sync --group oracle
.venv/bin/python -m pytest tests/test_egglog_oracle.py
```

It runs catopt's `EGraph` and egglog on the *same* law subset and
asserts they agree: the extracted lowest-FLOPs terms match up to
commutative-operand *representative* choice, the folded (param-
discounted) costs coincide, and the original plus both extracted terms
are numerically equal (torch ground truth + NumPy evaluation).  It is
fast (<~1 s) and needs no CUDA.  `egglog` ships wheels for CPython
>=3.12 only, so the group entry carries a `python_version` marker while
the workspace keeps `requires-python = ">=3.11"`.

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
