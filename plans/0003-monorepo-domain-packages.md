# Plan 0003 — Monorepo: domain packages with isolated dependencies

Status: proposed
Date: 2026-09-26
Branch: `project` (orphan — plans live here, not on main)
Follows: `0002-genericity-and-shared-machinery.md` (phases A–D complete)

## Goal

Split `catopt/` from one flat package into a uv-workspace monorepo of
domain distributions. Verified dependency truth (audited 2026-09-26):

- **~half the codebase is genuinely torch-free** — the IR, attrs,
  typing, ports, the whole egraph package, the whole laws package,
  cost, meta, rulecache, trace_lift. That is the abstract engine:
  terms + rewriting + saturation + extraction + certificates + cost
  algebra. It can install and run with **zero deps** (not even torch —
  the carriers' own semantics, including the term-level rewrites and
  certificates, live there too).
- The torch-dependent half is integration: the bridge adapter,
  executors, models, carriers' torch bindings, eps/ibp evaluators,
  the optimizer orchestrators.

Isolating them means: `catopt-core` installs standalone (the semantic
engine, the research artifact); integrations plug in only when their
dependency (torch) is present.

## Design

Namespace rule: **distinct top-level namespaces** (`catopt_core`,
`catopt_torch`, `catopt_carriers`, `catopt_eps`, `catopt_optimize`) —
no PEP-420 path merging, no `pkgutil` extend_path tricks. The old
`catopt/` package at the repo root becomes the **façade + compat
layer**: `sys.modules` aliases map every historical `catopt.X` path
onto the new homes, so `from catopt.cost import flops_cost`,
`from catopt.egraph import EGraph`, `import catopt.rules` all keep
working unchanged (the 1,579-test suite and any external callers
don't move).

### Layout

```
pyproject.toml                  # workspace root + "catopt" meta dist
uv.lock                         # one lockfile for the workspace
packages/
  catopt-core/                  # dist "catopt-core" — ZERO deps
    pyproject.toml
    src/catopt_core/
      __init__.py               # domain re-exports
      ir.py  attrs.py  typing.py  ports.py  ops.py
      cost.py  meta.py  rulecache.py  rules.py      (rules = laws shim)
      egraph/  laws/
  catopt-torch/                 # "catopt-torch" — deps: catopt-core, torch
    pyproject.toml
    src/catopt_torch/
      __init__.py
      torch_bridge.py  report.py
      models/  executors/
  catopt-carriers/              # "catopt-carriers" — deps: core+torch
    pyproject.toml
    src/catopt_carriers/
      om.py  xcarrier.py  trace.py  trace_lift.py
      om_lower.py  omd_lower.py  scan_lower.py
  catopt-eps/                   # "catopt-eps" — deps: core+torch
    pyproject.toml
    src/catopt_eps/
      eps.py  act_eps.py  ibp.py
  catopt-optimize/              # "catopt-optimize" — deps: all above
    pyproject.toml
    src/catopt_optimize/
      optimize.py  regime.py  calibrate.py
catopt/                         # the meta-façade package (dist "catopt")
  __init__.py                   # public re-exports + sys.modules aliases
```

### Dependency DAG (enforced by each package's declared deps)

```
catopt-core (no deps)
    ↑            ↑
catopt-torch     catopt-eps?   (eps needs core + torch — NOT torch dist)
    ↑  ↑         │
    │  └── catopt-carriers (core + torch — torch the lib, not the dist;
    │                       lowerers import catopt_torch.executors →
    │                       carriers DOES depend on catopt-torch)
    │            ↑
    └──── catopt-optimize ──── eps optional (lazy import already)
```

Notes on the edges:

- `catopt-eps` needs `catopt-core` + `torch` + `catopt_torch.bridge`
  (ibp/eps eval via bindings) → depends on catopt-torch too.
- `catopt-carriers` lowerers use `catopt_torch.bridge` +
  `executors.base` → depends on catopt-torch.
- `optimize` uses `catopt.eps` lazily (optimize.py:580 inside the
  eps_rtol path) → eps stays an *optional* extra, not a required dep.
- `ops.py` resolves carrier bindings lazily (`OpTable.full()` →
  `importlib`) → lives in core; no hard carrier edge.
- `trace_lift` is torch-free; it lives with the carriers' domain
  (its laws transform carrier structure) — no isolation loss.

### Compat façade mechanics (`catopt/__init__.py`)

```python
# For each new submodule: alias the old path in sys.modules.
import sys as _sys
from catopt_core import ir as _ir
_sys.modules.setdefault(__name__ + ".ir", _ir)
# ... per-module, packages alias wholesale:
#   catopt.egraph -> catopt_core.egraph (its submodules follow naturally)
```

Package aliases (`catopt.egraph` → `catopt_core.egraph`) resolve
nested submodules automatically because the alias IS the real
package object. Only torch-domain aliases run lazily-in-`import catopt`
— `import catopt` implies the full install anyway.

## Phases

| Phase | Work | Verify |
|---|---|---|
| 1 scaffold | `packages/*` dirs, per-package pyprojects (name, version, `dependencies` per DAG), `[tool.uv.workspace] members`, move files, domain `__init__.py`s | `uv sync` resolves |
| 2 rewrite | sed-map `from catopt.X`/`import catopt.X` → new namespaces (internal + tests untouched — shims cover tests); write `catopt/__init__.py` façade + aliases | `pytest -n4` green |
| 3 isolation | clean venv `pip install packages/catopt-core` → `import catopt_core`, run a small `EGraph` saturation — prove zero-dep | script |
| 4 gates | coverage `source` updated (new namespaces + façade); ruff; pyright excludes re-mapped; AGENTS.md commands | 100% / clean |
| 5 docs | README layout block + quickstart (paths unchanged for users), plan log | — |

Mechanics: the move is ~45 files + ~600 `catopt.X` references — a
scripted sed over an explicit rename map, then the suite as the
oracle. Agents do phases in order; phases 1–2 in one agent (they're
entangled), 3–5 verify.

## Acceptance

- [ ] `pip install -e packages/catopt-core` in a torch-free venv:
      `import catopt_core`, `EGraph().run(rules)` works.
- [ ] `catopt-carriers` import fails *cleanly* without torch
      (ImportError naming torch, not a crash inside core).
- [ ] Full env: 1,579 tests green, coverage 100% (source = new pkgs).
- [ ] `from catopt.optimize import optimize_model` and every other
      historical path works (aliases) — zero test edits.
- [ ] Each package's `pyproject.toml` declares exactly its DAG deps.
- [ ] No cycles in the package DAG (core knows nothing about torch).

## Risks

- **Import-map misses** — sed catches `from catopt.X` but dynamic
  `importlib`/string names survive (`ops.py` carrier resolution by
  string name — those must be renamed in `_CARRIER_MODULES`).
- **`sys.modules` alias order** — aliases must be registered before
  any user `import catopt.X`; do them at the top of `catopt/__init__.py`,
  before the public re-exports.
- **Setuptools packaging per subpackage** — each pyproject needs
  `package-dir` + `packages.find` scoped to `src`.
