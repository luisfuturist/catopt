# Agent guide — catopt

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Package source lives in `catopt/`; tests in
`tests/`. The dev virtualenv is `.venv/` (uv-managed).

## Verification commands

Run these before finishing any change; all must pass.

```sh
uv run pytest                 # full test suite (pytest-xdist enabled)
.venv/bin/pyright             # typecheck — 0 errors required (warnings OK)
.venv/bin/ruff check          # lint
.venv/bin/ruff format --check # formatting
uv run pytest --cov=catopt --cov-report=term-missing   # coverage (fail_under=87)
```

Pre-commit (`pre-commit install`) runs ruff check/format and pyright on
`catopt/**/*.py`.

## Typecheck ratchet

- Checker: **pyright** (`[tool.pyright]` in `pyproject.toml`,
  `typeCheckingMode = "standard"`, `include = ["catopt"]`).
- Installed in `.venv` via the `dev` dependency group
  (`pyright>=1.1.380`; currently 1.1.414).
- `exclude` in `pyproject.toml` lists files that predate the ratchet —
  each entry documents its baseline error count. **Remove entries as
  files get annotated**; never add new ones. Excluded files are still
  analyzed when imported by checked files.
- New modules under `catopt/` are checked automatically — keep them
  clean at `standard` strictness.
- Baseline was green at 0 errors / 2 warnings (`calibrate.py`
  `reportUnusedExpression`); warnings must not regress to errors.

## Conventions

- Line length 72 (ruff). E501/B008/SIM108 intentionally ignored — see
  `[tool.ruff.lint]` for rationale.
- Coverage ratchet: `fail_under = 87` in `[tool.coverage.report]`,
  raised per refactor phase toward 100.
- Tests allocating CUDA tensors use the `requires_cuda` marker
  (auto-skipped when CUDA is absent).
