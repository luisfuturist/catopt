# Agent guide — catopt

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Monorepo layout: the domain packages
live under `packages/` (`catopt-core` — the torch-free engine;
`catopt-torch` — PyTorch adapters; `catopt-carriers` — carrier
laws/executors; `catopt-optimize` — pipeline orchestrators);
`catopt/` is the façade
+ compat aliases (every historical `catopt.X` import path resolves to
its new home via `sys.modules` aliases). Tests in
`tests/`. The dev virtualenv is `.venv/` (uv-managed).

## Verification commands

Run these before finishing any change; all must pass.

```sh
uv run pytest                 # full test suite (pytest-xdist enabled)
.venv/bin/ty check            # typecheck — 0 errors required (warnings OK)
.venv/bin/ruff check          # lint
.venv/bin/ruff format --check # formatting
.venv/bin/vulture             # dead code (uses [tool.vulture] paths)
.venv/bin/lint-imports        # hexagonal boundary contracts
.venv/bin/bandit -c .bandit.yaml -r packages catopt   # security SAST
.venv/bin/semgrep --config .semgrep.yml packages catopt   # dataflow (offline)
.venv/bin/python tools/radon_ratchet.py   # complexity ratchet
coverage run --source=catopt_core,catopt_torch,catopt_carriers,catopt_optimize,catopt -m pytest tests/ -q
coverage report -m                                              # coverage (fail_under=100)
```

Manual-stage gates (not run on every commit — network/slower):

```sh
.venv/bin/pip-audit                                          # dependency CVEs (network)
.venv/bin/semgrep --config p/python --config p/security-audit packages catopt  # registry rules (network)
.venv/bin/python -m pytest -q --typeguard-packages=catopt_core \
    tests/test_ir.py tests/test_egraph.py tests/test_laws_structure.py \
    tests/test_cost.py tests/test_interning.py tests/test_attrs.py  # runtime contracts
```

Pre-commit (`pre-commit install`) runs ruff check/format, **ty**,
import-linter, vulture, bandit, semgrep and the radon ratchet.  The
`manual`-stage hooks (typeguard, pip-audit, semgrep-registry) run with
`pre-commit run --hook-stage manual --all-files`.

Known drift: the installed ruff (0.16.9) flags pre-existing isort /
format differences across `tests/` and `bench/` that predate any
current change. `ruff check packages catopt` is the meaningful gate
(0 errors); repo-wide `ruff check` and `ruff format --check` are not
clean at `HEAD` under 0.16.9.

## Typecheck ratchet

- Checker: **ty** (`[tool.ty]` in `pyproject.toml`) — Astral's type
  checker, replacing the earlier pyright ratchet.
  `[tool.ty.src] include = ["packages","catopt"]` scopes checking to the
  shipped packages (tests/ and bench/ are not type-checked);
  `[tool.ty.environment]` sets the 3.13 target and the per-package
  `extra-paths`; `[tool.ty.terminal] error-on-warning = false` keeps the
  gate "0 errors, warnings OK".
- Installed in `.venv` via the `dev` dependency group (`ty>=0.0.84`;
  currently 0.0.84).
- `[tool.ty.src] exclude` lists files that predate the ratchet — each
  entry documents its ty-baseline error count + dominant rule.
  **Remove entries as files get annotated**; never add new ones.
  Excluded files are not reported, but are still analyzed when imported
  by a checked file.
- New modules under `packages/*/src/` are checked automatically — keep
  them clean.
- Baseline at migration: 198 errors / 19 files (down from pyright's 442
  / 22). `ty check` at `HEAD` is green (0 diagnostics).

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
  `catopt_carriers` / `catopt_optimize` — or the `torch`
  / `numpy` external packages — a hard error.  Run
  `.venv/bin/lint-imports`.  Core is a *sink* for adapter-pushed state
  (see `catopt_core.ops` *Backend wiring*), never a puller: the adapter
  registers its tables into core, core never imports the adapter.

## Property tests (Hypothesis)

`tests/test_property_*.py` are property-based tests over the torch-free
core: IR/`Op` interning, e-graph congruence + extraction cost bounds,
cost-model monotonicity, and law match/instantiate round-trips across
all `ALL_RULES` patterns.  The shared strategies live in
`tests/test_property_strategies.py` (not collected).  They are fast
(<~10 s, no CUDA/network) and run as part of the normal suite:

```sh
.venv/bin/python -m pytest tests/test_property_ir.py tests/test_property_egraph.py \
    tests/test_property_cost.py tests/test_property_laws.py
```

Two properties are `xfail(strict=False)` because they document genuine
open questions (asymmetric cost under operand swap when a shape is
unknown; `rebuild` does not close congruence across classes) — see the
test bodies.  Do not "fix" these by weakening the test; either resolve
the behaviour or keep the xfail.

## Static analysis & security

- **Bandit** (`.bandit.yaml`, dev dep): SAST over `packages` + `catopt`
  (tests/bench excluded).  The only finding is two `B112`
  (try/except/continue) in `catopt_core.meta`'s rule matcher, where a
  guard/derive raising is the *signal to reject a candidate* — a
  justified config-level skip.  Run `.venv/bin/bandit -c .bandit.yaml
  -r packages catopt`.
- **Semgrep** (`.semgrep.yml`, dev dep): the committed config is
  deterministic/offline (no-`eval`/`exec`, no-`shell=True`,
  no-unsafe-`yaml.load`).  The deeper registry scan
  (`--config p/python --config p/security-audit`) is a manual-stage
  gate; at migration it reported 0 findings.
- **pip-audit** (dev dep): audits the installed environment; manual
  stage (needs network).  At migration: 0 known vulnerabilities (the
  five first-party workspace packages are skipped as local editable
  installs).

## Complexity ratchet (radon)

The engine is math-heavy (47 rank-C, 11 D, 7 E, 5 F functions at
migration), so a hard ceiling is impractical.  Instead
`tools/radon_ratchet.py` pins every function's cyclomatic complexity in
`tools/complexity_baseline.json`: a function may not exceed its recorded
value, and a function **not** in the baseline must stay at or below
`THRESHOLD = 11` (rank C).  After an intentional complexity change,
regenerate with `--update` and review the diff.  Run
`.venv/bin/python tools/radon_ratchet.py`.

## Docstring quality (ruff `D`)

The pydocstyle ruleset is enabled through ruff (`select = [..., "D"]`),
which is the modern replacement for the standalone `pydocstyle`
package.  The currently-violated rules (missing-docstring `D1xx`, and
the `D202/D205/D209/D301/D400/D401/D403/D413` style backlog) are
ratcheted off in `[tool.ruff.lint] ignore` with a comment; remove
entries as docstrings are brought up to standard.  The gate applies to
shipped code only — `tests/**` and `bench/**` ignore `D`.

## Runtime contracts (typeguard)

`typeguard` (dev dep) enforces annotations at runtime.  Whole-suite
instrumentation is blocked by the ty annotation backlog (it surfaces
`str`-sentinel-vs-`tuple | None` returns in `typing.py`, etc.), so the
gate is a curated green subset run with `--typeguard-packages=catopt_core`
over `test_ir/test_egraph/test_laws_structure/test_cost/test_interning/
test_attrs` (manual stage).  It already caught two real bugs, now fixed:
`laws/tensor._head` was annotated `str` but takes an `Op`, and
`typing._infer_op_shape` passed a `tuple` to zero-arg shape rules typed
`list`.  Grow the file list as annotations are cleaned up.

## Mutation testing (mutmut)

Configured in `[tool.mutmut]` but **not yet operational** for this
monorepo.  mutmut 3.8 derives a mutant key from its path relative to
cwd (`packages.catopt-core.src.catopt_core.attrs.x_foo`) and expects the
module importable under that dotted name, but a per-package `src/`
layout imports as `catopt_core.attrs`.  Two shims are in place for the
fixable halves — the `pythonpath` roots in `[tool.pytest.ini_options]`
and the editable-finder drop in `tests/conftest.py` (both no-ops for
normal runs) — but the key derivation needs upstream support or a
single-package flat checkout.  Until then, prefer coverage + the
property tests for test-strength signal.

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
- Ratchets, not rewrites: the ty exclude list, the ruff `D` ignore
  list, the radon baseline and the typeguard subset are all "pin the
  current state, never regress" gates.  Tighten them as code improves;
  never widen them to make a change pass.
- The lockfile resolves a CUDA-enabled `torch` wheel on Linux.  Some
  CUDA-graph tests (`test_cov2_models`, `test_executor_base`) assume a
  CPU build (`torch.cuda.is_available() is False`); if your venv has
  the CUDA wheel and a GPU, install the CPU build for the suite:
  `uv pip install --torch-backend cpu --reinstall torch`.
