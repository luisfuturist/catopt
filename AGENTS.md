# Agent guide — catopt

Categorical optimization of neural-network computation graphs
(Python 3.13, torch + numpy). Monorepo layout: the domain packages
live under `packages/` (`catopt-core` — the torch-free engine;
`catopt-torch` — PyTorch adapters; `catopt-carriers` — carrier
laws/executors; `catopt-cuda` — the CUDA-graph runner;
`catopt-orchestrator` — the backend-neutral pipeline;
`catopt-discovery` — the law-discovery engine, formerly
`tools/law_*.py`; invoke as `python -m catopt_discovery.<mod>`;
the game layer is `engine` (the `Board` protocol + generic
episode/train/eval drivers), `players` (`LinearPolicy`, the
featurizer-injected learned arm), `play` (the domain registry +
`python -m catopt_discovery.play --domain meta|joint|search|torch|gen`
CLI; `--deliver` lowers+verifies the winning extraction on domains
that ship a deliver hook — torch does)).
The optional
`catopt-native` package — the PyO3/Rust search engine, excluded from
the uv workspace and built with maturin — is opt-in via `engine=`.
The `catopt` façade is gone (plan 0008): `import catopt` fails and
every name lives at its real package path — `catopt_core.egraph`,
`catopt_torch.adapters`, `catopt_orchestrator.optimize`,
`catopt_cuda.CudaGraphRunner`, … Tests in
`tests/`. The dev virtualenv is `.venv/` (uv-managed).

## Verification commands

Run these before finishing any change; all must pass.

```sh
uv run pytest                 # full test suite (~7.5 min, single-process —
                              # pytest-xdist is installed but the suite runs
                              # serially; see the note below before enabling
                              # `-n` on this suite)
.venv/bin/ty check            # typecheck — 0 errors required (warnings OK)
.venv/bin/ruff check          # lint
.venv/bin/ruff format --check # formatting
.venv/bin/vulture             # dead code (uses [tool.vulture] paths)
.venv/bin/lint-imports        # hexagonal boundary contracts
.venv/bin/bandit -c .bandit.yaml -r packages   # security SAST
.venv/bin/semgrep --config .semgrep.yml packages   # dataflow (offline)
.venv/bin/python tools/radon_ratchet.py   # complexity ratchet
coverage run -m pytest tests/ -q   # source list includes catopt_discovery
coverage report -m --omit="*/catopt_discovery/*"   # the five: fail_under=100
coverage report -m --include="*/catopt_discovery/*" --fail-under=99   # ratchet floor — measured 99%; tighten, never lower
```

> **Parallel-warning (measured, not theoretical).**  Do *not* run this
> suite — with or without coverage — under `pytest -n auto` on
> low-RAM machines.  The suite is single-process by design: the
> torch.compile-heavy tests (`test_torch_integration` &c.) spawn an
> inductor pool of **16 compile workers per pytest process**, so
> `-n N` multiplies the fan-out to ~`16·N` subprocesses.  On an 11 GB /
> 16-core host, `-n auto` OOM-froze the machine; even `-n 3` drove
> memory to 10.2 GB near the suite's tail.  The coverage gate is a
> commit-time check (~20-25 min serial under tracing); for iteration
> use a scoped `coverage run -m pytest tests/test_<file>.py`.
> If a bigger machine needs it anyway, cap both fan-outs:
> `TORCHINDUCTOR_COMPILE_THREADS=2 pytest -n 2 --cov=...` — untested.

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

Carrier machinery is the second half of the boundary.  The
orchestrator's carrier-aware passes (batched plan builders, non-local
lifts, carrier rule sets, carrier-root probes) reach the torch-coupled
`catopt_carriers` package through the `register_carriers` seam
(`catopt_orchestrator.carriers`), never by importing it: the carrier
package registers a `CarrierMachinery` value — all thunks, so
registration loads no tensor library — when it is imported.  Without a
registration the passes degrade to their carrier-free defaults.  The
resulting package graph is pinned by the import-linter `dependency
layers` contract: `catopt_cuda` above `catopt_torch`/`catopt_carriers`
(one mutually-dependent backend layer — the carriers' law modules
carry `TORCH_BINDINGS`, and the torch sink builds its executor table
from the carrier lowerings), above `catopt_orchestrator`, above
`catopt_native`, above `catopt_core`.  `catopt_torch` does not depend
on `catopt_cuda`; import `CudaGraphRunner` from `catopt_cuda`.

## Conventions

- Line length 72 (ruff). E501/B008/SIM108 intentionally ignored — see
  `[tool.ruff.lint]` for rationale.
- Coverage floor: `fail_under = 100` in `[tool.coverage.report]` —
  the suite is pinned at 100% for the five original packages; new
  branches need tests (or a justified `pragma: no cover`).
  `catopt_discovery` joined the source list after its staged climb
  and sits under a **pinned ratchet floor** (`--fail-under=99` at
  landing — tighten the number as the residual gaps close, never
  lower it; the unreachable-defensive arcs are documented in
  `project/retros/discovery-package.md`).
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
- Four independent dimensions (ADR 0003): **semantics / search /
  evaluation / execution**.  No layer answers another's question — a
  cost model must not change semantics, a policy must not decide
  equivalence, a profiler must not run on the target.  Feasibility
  (`supported_ops`, hard) and performance ranking (soft) stay
  distinct.
- "Profile" means one thing: a measured target
  (`catopt_core.profile.TargetProfile`).  Static characterization of
  a *program* is `ProgramFeatures` (a future `catopt_core.features`),
  never called a "profile".

## Adding a rewrite law

Laws live in `packages/catopt-core/src/catopt_core/laws/` — `tensor.py`
(tensor algebra), `scan.py` (scan monoids).  Shape (via `laws.base.R`):
`R(name, lhs, rhs, law=..., check=..., derive=...)` — lhs/rhs are
`Op.make` pattern trees; bare `str` leaves are metavariables, `Const`
leaves are literals, and `str` attr values are attr metavariables bound
under `"$attr:"` keys.  Prefer `cond=...` — a declarative side
condition in `catopt_core.laws.cond` (pure data: `("and", ("rank-eq",
"a", "b"), ("rank-ge", "a", 2))`, serializable via `cond_to_data`/
`cond_from_data`; 45 of 71 laws use it) — over `check(bound) -> bool`,
the escape hatch for conditions the DSL can't express: `bound` maps
each metavar to a resolved member term (use `laws.base._shape_of` for
shape guards — the matcher cannot see types).
`derive(bound) -> dict | None` computes RHS attrs absent from the LHS
(`{"$attr:SZ": ...}`); `None` vetoes.  Prefer `dspec=...` — the twin
declarative DSL in `catopt_core.laws.cond` (a `{NAME: expr}` map of
shape/attr expressions, serializable via `derive_to_data`/
`derive_from_data` and folded into `rule.derive` at construction);
18 of 71 laws use it, leaving **71 of 71 laws fully data-serializable**
(see `project/retros/derive-declarative.md`).  Every firing records a
witness
(proof edge + rule provenance) — no extra bookkeeping needed.  Register
the rule in a group list at the bottom of its module
(`SIMPLIFICATION_RULES`, `CATEGORICAL_RULES`, `SDPA_FOLD_RULES`,
`SCAN_LAWS`, `SCAN_DIAG_LAWS`; `ALL_RULES`/`all_rules()` is the union).
Rules also carry tags from `catopt_core.laws.tags` (pass
`R(..., tags=...)`); `EXPANSIVE` marks the closure-generating rules the
pipeline budgets (`rules.tagged(EXPANSIVE)`).  Each rule's kernel kind
lives on `Rewrite.derivation`/`rule.kind` — `R(..., derivation=(...))`
names the axioms that prove a lemma (`catopt_discovery.coherence
--emit-basis` emits the measured table; see
`project/retros/axiom-lemma-split.md`).  `catopt_discovery.lemma_cert`
materializes each recorded derivation as a replayable `Certificate`
(codec: `cert_to_data`/`cert_from_data` in `catopt_core.egraph.certs`;
see `project/retros/lemma-certificates.md`).  The composable `RuleSet`
presets live in `catopt_core.laws.ruleset`.

Measure it: `python -m bench.suites.correctness.law_bench --laws <name[,name|group]>`
(`--sizes <d[,d]>` selects sizes; the suite's own CLI is the entry point) —
per law: registered synthetic term → `eg.run` on that rule alone →
extraction (pipeline cost model) → `_lower_extracted` → `sink.verify` →
timed before/after.  Table shows fired / rhs-member / picked / verified /
cost & ms before→after; non-firing laws report honestly.  Add a builder
in `LAW_CASES` keyed by rule name for new laws.  Gates: `uv run pytest`,
`.venv/bin/ty`, `.venv/bin/ruff check` (packages), coverage stays 100.

Benchmarks are a package, not scripts: `python -m bench list` shows the
catalog; `python -m bench run <suite>` emits JSON + Markdown + HTML +
plotly plots (+ optional Quarto/Slidev) from one canonical `Report`;
`python -m bench dashboard` builds the cross-suite HTML index;
`python -m bench catalog` emits the docs.  Suites live in
`bench/suites/<intent>/` and expose `run_bench(args) -> Report`; shared
data (checkpoint loaders) is in `bench/common/`.  `bench/registry.py`
is the source of truth for a suite's **intent** (the question it
answers), its `question`/`expects`, `tier`, `status`, controlled
`mechanisms` and `needs_cuda` — the README tables and the dashboard
grouping are generated from it, so they cannot drift.  Details in
`bench/README.md`.
