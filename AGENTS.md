# Agent guide — catopt

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Monorepo layout: the domain packages
live under `packages/` (`catopt-core` — the torch-free engine;
`catopt-torch` — PyTorch adapters; `catopt-carriers` — carrier
laws/executors; `catopt-cuda` — the CUDA-graph runner;
`catopt-orchestrator` — the backend-neutral pipeline). The `catopt`
façade is gone (plan 0008): `import catopt` fails and every name
lives at its real package path — `catopt_core.egraph`,
`catopt_torch.adapters`, `catopt_orchestrator.optimize`,
`catopt_cuda.CudaGraphRunner`, … Tests in
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
.venv/bin/bandit -c .bandit.yaml -r packages   # security SAST
.venv/bin/semgrep --config .semgrep.yml packages   # dataflow (offline)
.venv/bin/python tools/radon_ratchet.py   # complexity ratchet
coverage run --source=catopt_core,catopt_torch,catopt_carriers,catopt_orchestrator,catopt_cuda -m pytest tests/ -q
coverage report -m                                              # coverage (fail_under=100)
```

Manual-stage gates (not run on every commit — network/slower):

```sh
.venv/bin/pip-audit                                          # dependency CVEs (network)
.venv/bin/semgrep --config p/python --config p/security-audit packages  # registry rules (network)
sh tools/runtime_types.sh                                    # typeguard runtime contracts
MUTMUT_ONLY=catopt_core/ir.py MUTMUT_TESTS='tests/test_ir.py' tools/mutmut.sh run  # mutation testing
```

Pre-commit (`pre-commit install`) runs ruff check/format, **ty**,
import-linter, vulture, bandit, semgrep and the radon ratchet.  The
`manual`-stage hooks (typeguard, pip-audit, semgrep-registry) run with
`pre-commit run --hook-stage manual --all-files`.

Known drift: the installed ruff (0.16.9) still flags pre-existing
isort / format differences across `tests/` and `bench/`, which are not
format-checked.  The shipped source — `packages` and `tools` —
is clean under both `ruff check` and `ruff format --check`.

## Typecheck (ty)

- Checker: **ty** (`[tool.ty]` in `pyproject.toml`) — Astral's type
  checker, replacing the earlier pyright ratchet.
  `[tool.ty.src] include = ["packages"]` scopes checking to the
  shipped packages (tests/ and bench/ are not type-checked);
  `[tool.ty.environment]` sets the 3.13 target and the per-package
  `extra-paths`; `[tool.ty.terminal] error-on-warning = false`.
- Installed in `.venv` via the `dev` dependency group (`ty>=0.0.84`;
  currently 0.0.84).
- **The ratchet `exclude` list is empty.** The migration baseline was
  198 errors across 19 files (down from pyright's 442/22); every one of
  those files has since been annotated, so `ty check` is clean with no
  opt-outs.  Keep it that way — annotate; never add an exclude entry or
  a `# ty: ignore`.
- New modules under `packages/*/src/` are checked automatically.

## Dead-code + architecture linting

Two dev-only ratchets (both in the `dev` dependency group, both
configured in `pyproject.toml`):

- **vulture** (`[tool.vulture]`, `min_confidence = 80`) flags
  unreachable / unused code across `packages` / `tests`.
  The suite is pinned at 100% coverage, so a genuine dead branch is a
  real finding, not noise.  Run `.venv/bin/vulture` (uses the
  configured `paths`); a non-zero exit (3) means dead code.
- **import-linter** (`[tool.importlinter]`) pins the hexagonal boundary:
  a `forbidden` contract makes `catopt_core` importing `catopt_torch` /
  `catopt_carriers` / `catopt_orchestrator` — or the `torch`
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

No properties are xfailed: both former xfails are fixed.  The
commutative unknown-shape cost fallback in `_infer_op_shape` is
symmetric, and `EGraph.rebuild` now closes congruence across classes —
enodes that become identical after child canonicalisation are unioned
(see `_close_congruence`), to a fixed point.

## Static analysis & security

- **Bandit** (`.bandit.yaml`, dev dep): SAST over `packages`
  (tests/bench excluded).  The only finding is two `B112`
  (try/except/continue) in `catopt_core.meta`'s rule matcher, where a
  guard/derive raising is the *signal to reject a candidate* — a
  justified config-level skip.  Run `.venv/bin/bandit -c .bandit.yaml
  -r packages`.
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

The pydocstyle ruleset is enabled through ruff (`select = [..., "D"]`).
The `D` ignore list is now **empty**: the whole backlog (550 violations
— missing docstrings, summary/blank-line style, imperative mood, …) has
been cleared, so `ruff check packages` enforces `D` in full.  The
codebase adopts **D211** (blank line before a class docstring) and
**D212** (multi-line summary on the first line) over the mutually
exclusive D203/D213 — ruff prints its usual "incompatible" warning for
those pairs; that is the expected signature of the choice, not an
error.  The gate applies to shipped code only — `tests/**` and
`bench/**` ignore `D`.

## Runtime contracts (typeguard)

`typeguard` (dev dep) enforces annotations at runtime.
`tools/runtime_types.sh` instruments all four packages
(`--typeguard-packages=catopt_core,catopt_torch,catopt_carriers,catopt_orchestrator`)
and runs the curated (1372 tests) — a manual-stage gate that CI also
runs (~11 min).  It caught several real annotation bugs, all fixed:
`laws/tensor._head` was annotated `str` but takes an `Op`;
`typing._infer_op_shape` passed a `tuple` to zero-arg shape rules typed
`list`; `typing._shape_of` returned the `_INVALID` string sentinel under
a `tuple | None` return type; `EGraph.matches`/`_match` substituted
attribute metavariables (floats/strs) into `dict[str, int]`; and the
adapter `eval_term` / `ev_factory` / carrier-module `forward`s declared
`-> torch.Tensor` returns that are really carrier tuples or `None`
(widened to `Any`).  One parameter was renamed `memo` → `memo_env` to
dodge a typeguard 4.6 shadowing bug.  Files still left out (and why):
heavy saturation tests that exceed ~20 min under instrumentation
(`test_torch_integration`, `test_omd_lower`, `test_om_mha`), and a few
error-path tests that deliberately pass wrong-typed values.  Grow the
list in the script.

## Mutation testing (mutmut)

`tools/mutmut.sh` runs mutmut against a flat-layout sandbox
(`.mutmut-sandbox/`, gitignored): mutmut 3 derives mutant keys from
paths, but a per-package `src/` layout imports under a different dotted
name, so the wrapper symlinks each package at the sandbox top level
(path == import name) plus `tests/` and `uv.lock`, and writes
the matching `[tool.mutmut]`.  Scope a fast loop with `MUTMUT_ONLY` /
`MUTMUT_TESTS`; a surviving mutant is a weak-assertion finding (the
suite is pinned at 100% coverage, so it is not a coverage gap).  The
editable-finder drop in `tests/conftest.py` exists solely so the
sandbox's mutated copies win over uv's editable install.

## Differential oracle (opt-in, test-only)

`tests/test_egglog_oracle.py` cross-checks catopt's hand-rolled
equality-saturation search against the Rust **egglog** library on a
small op/law subset.  It is an **opt-in, test-only oracle** — not a
production dependency and not an engine swap: `catopt-core` stays
pure-Python / zero-dependency, and nothing under `packages/` or
`packages/` imports `egglog`.  The ported prototype and its documented
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
`catopt_orchestrator.Optimizer` takes the ports as a
`catopt_core.pipeline.Backend` value (torch:
`Optimizer(backend=TorchBackend())`, the
`catopt_torch.backend.TorchBackend` bundle of
`TorchSource`/`TorchSink`/`TorchComposer`/`TorchMeter`) or explicitly
(`source=`/`sink=`/…); `discover_alternatives` takes `source=`.
There is no default backend.
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
  (auto-skipped when CUDA is absent).  Tests that exercise the CPU
  no-op path of `capture_cuda_graph` force `torch.cuda.is_available()`
  off via `monkeypatch`, so they pass on a CUDA host too.
- Ratchets, not rewrites: the ruff `D` ignore list, the radon baseline
  and the typeguard file list are "pin the current state, never
  regress" gates.  Tighten them as code improves; never widen them to
  make a change pass.  The ty `exclude` list is empty — keep it so.
- No new suppressions.  Do not add `# type: ignore` / `# ty: ignore` /
  `# noqa` / `# nosec` / `# pragma: no cover` to make a gate pass; fix
  the code or add a justified, documented config entry (as `.bandit.yaml`
  does for the two `B112` `try/except/continue` sites).
- The lockfile resolves a CUDA-enabled `torch` wheel on Linux.  CI
  installs the CPU wheel (`uv pip install --torch-backend cpu
  --reinstall torch`) for determinism; the CUDA-graph tests no longer
  require it locally.

## Adding a rewrite law

Laws live in `packages/catopt-core/src/catopt_core/laws/` — `tensor.py`
(tensor algebra), `scan.py` (scan monoids).  Shape (via `laws.base.R`):
`R(name, lhs, rhs, law=..., check=..., derive=...)` — lhs/rhs are
`Op.make` pattern trees; bare `str` leaves are metavariables, `Const`
leaves are literals, and `str` attr values are attr metavariables bound
under `"$attr:"` keys.  `check(bound) -> bool` is the side-condition
hook: `bound` maps each metavar to a resolved member term (use
`laws.base._shape_of` for shape guards — the matcher cannot see types).
`derive(bound) -> dict | None` computes RHS attrs absent from the LHS
(`{"$attr:SZ": ...}`); `None` vetoes.  Every firing records a witness
(proof edge + rule provenance) — no extra bookkeeping needed.  Register
the rule in a group list at the bottom of its module
(`SIMPLIFICATION_RULES`, `CATEGORICAL_RULES`, `SDPA_FOLD_RULES`,
`SCAN_LAWS`, `SCAN_DIAG_LAWS`; `ALL_RULES`/`all_rules()` is the union).
Rules also carry tags from `catopt_core.laws.tags` (pass
`R(..., tags=...)`); `EXPANSIVE` marks the closure-generating rules the
pipeline budgets (`rules.tagged(EXPANSIVE)`).  The composable `RuleSet`
presets live in `catopt_core.laws.ruleset`.

Measure it: `python bench/law_bench.py --laws <name[,name|group]>`
`--sizes <d[,d]>` — per law: registered synthetic term → `eg.run` on
that rule alone → extraction (pipeline cost model) → `_lower_extracted`
→ `sink.verify` → timed before/after.  Table shows fired / rhs-member /
picked / verified / cost & ms before→after; non-firing laws report
honestly.  Add a builder in `LAW_CASES` keyed by rule name for new
laws.  Gates: `uv run pytest`, `.venv/bin/ty`,
`.venv/bin/ruff check` (packages), coverage stays 100.
