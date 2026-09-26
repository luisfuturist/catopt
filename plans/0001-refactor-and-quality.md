# Plan 0001 — Refactor architecture, enforce quality gates

Status: phases 0–3 complete (as of bc1ea0e)
Date: 2026-09-25
Branch: `project` (orphan — plans live here, not on main)

## Objective

Refactor catopt (~39k LOC, single-package) for extensibility,
composability, testability, decoupling, robustness, and performance —
and stand up the tooling that keeps it that way: **uv, pytest,
coverage (100% target), ruff (lint + format)**.

## Why now — the structural wounds, with evidence

| Wound | Cost it already extracted |
|---|---|
| `_shape_of` (cost.py) serves two masters — cost pricing **and** rewrite soundness — one giant `match` | slice-step bug, rank-0 policy patch, om_lift veto — wrong silently twice in one week |
| Attr spelling chaos — `arg1`/`dim`/`sizes` dual spellings, renames scattered across `_ATTR_RENAMES`, `_SCALAR_OPERAND_OPS`, `_POSITIONAL_ATTRS` | `rms_norm` missing binding (all transformer blocks failed); slice `arg4` silently dropped |
| `egraph.py` = 2,295-line god object | union-find + matching + saturation + extraction + certificates + caches, one file |
| Import-time global mutation — `_IR_TO_TORCH` mutated by `trace.py`, `xcarrier.py`, `act_eps.py` at import | ordering-dependent coupling; `import x  # noqa — registers bindings` |
| `id()`-keyed memos + `keepalive` lists | GC id-reuse hazard; code keeps lists alive to dodge it |
| `rules.py` 1,717 lines mixing laws, guards, and non-local passes | pairing passes aren't laws but live in the laws file |
| Loose stats dicts | `rep["blocks"]` ad-hoc keys; consumers guess the schema |
| Integration-heavy tests, thin per-op contracts | shape/binding bugs shipped past the suite |

Current scale (for phase ordering): egraph 2295, xcarrier 1880,
rules 1717, ibp 1388, cost 1301, meta 1297, regime 1151, om 1093,
torch_bridge 767, optimize 722.

## Tooling setup (do first — gates everything after)

Migrate to **uv** as the project/env manager; keep `pyproject.toml`
as the single source of truth.

- `uv` manages the venv and lockfile (`uv.lock` committed).
- Dev dependency group: `pytest`, `pytest-cov`, `ruff`, `coverage`
  (and `pytest-xdist` for speed — 572 tests already take ~70s).
- `pyproject.toml` gains:
  - `[tool.ruff]` + `[tool.ruff.lint]` — rule set agreed: `E, F, W,
    I, UP, B, SIM, RUF` (pyupgrade + bugbear + simplify; NOT
    docstring/coverage rules — noise).
  - `[tool.coverage.run]` — `source = ["catopt"]`,
    `omit = ["*/tests/*"]`; `branch = true`.
  - `[tool.coverage.report]` — `fail_under = 100`,
    `show_missing = true`, `exclude_also` for
    `if __name__ == "__main__":`, `def __repr__`,
    `if TYPE_CHECKING:`, `@overload`, `raise NotImplementedError`,
    `...` ellipsis bodies, `# pragma: no cover` markers for
    genuinely-unreachable defensive branches (each must carry a
    comment justifying the exclusion).
- **100% line+branch coverage on `catopt/` is the gate.** Interim
  ratchet: measure current coverage day 0, set `fail_under` to that
  value, then raise it as phases land so it never decreases. New
  code must carry tests in the same change.
- Pre-commit hooks (`.pre-commit-config.yaml`): `ruff check --fix`
  + `ruff format` + `pytest -q -x --ff` on staged paths.
- Optional CI: GitHub Actions — lint, format-check, pytest with
  coverage upload (`pytest --cov=catopt --cov-report=term-missing
  --cov-report=xml`), fail under threshold.
- `make` or `just`/`uv run` entrypoints documented in README:
  `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`,
  `uv run pytest --cov`.

## Phase 0 — Baseline + harness

1. uv setup, lockfile, dev deps, ruff config, coverage config.
2. `ruff format` once (whole-repo, one diff commit), `ruff check`
   fixes — mechanical only.
3. Coverage baseline report → checked-in `coverage-baseline.txt`
   on the `project` branch (not main).
4. Ratchet `fail_under` to the baseline.

Exit: CI-style command suite runs green; coverage number known.

## Phase 1 — Correctness architecture (the bug-shaped work)

The two classes that produced this session's real bugs.

### 1a. `catopt/typing.py` — shape/type inference owns itself

- Move `_shape_of`, `_INVALID`, `_broadcast`, shape semantics out of
  `cost.py`.
- Carrier ops register shape semantics via protocol —
  `register_shape(op, fn)` — from their own modules (om/xcarrier/
  trace declare their convention-vs-value semantics next to their
  laws), instead of one shared match everyone edits.
- cost.py keeps only pricing; callers in `om.py`/`rules.py`/
  `xcarrier.py`/`meta.py`/`ibp.py` repoint.
- The convention-vs-value split becomes *per-op data*, not a global
  docstring: a carrier member's value-shape resolver (`_xshape`)
  vs tensor-shape resolver is explicit in the op's registration.

### 1b. Attr canonicalization at the boundary

- `torch_bridge` maps `argN` → canonical semantic names at export —
  the ONLY place positional spellings exist.
- Per-op `AttrSchema` declared once (e.g.
  `slice: dim,start,end,step`; `split: sizes,dim,index`;
  `rms_norm: normalized_shape,eps`). Ops validate attrs at `Op.make`
  — a rewrite minting a malformed term fails at `union`, loudly,
  not at eval.
- Delete the `kw.get("arg1", kw.get("dim", ...))` fallback pattern
  everywhere downstream — dead code once the boundary canonicalizes.

### 1c. Per-op contract tests

- Table-driven tests: every op × every attr spelling × shape rule —
  the slice/`rms_norm` class of bug becomes a 1-line test failure.
- Binding coverage test: every op in `_ATEN_TO_IR` has a
  `_IR_TO_TORCH` binding (catches the missing-`rms_norm` class).
- Law soundness fuzzer: random well-typed terms, apply each rule's
  lhs/rhs on random tensors, assert `lhs(args) == rhs(args)` where
  both defined (bounded loop, seeded).

Exit: shape and attr bugs are mechanically impossible to ship
silently. Tests: contract suite green.

## Phase 2 — Core architecture

### 2a. Hash-consed terms

`Op`/`Var`/`Param`/`Const` become interned value objects (weakref
intern table keyed by structural identity). Benefits:

- `id()`-keyed memos and `keepalive` hacks die — memoization keys on
  content hash, GC-safe.
- Structural dedup of extracted terms is free (DAG sharing by
  construction).
- `==`/`hash` become semantic — simplifies extraction caching,
  `_class_of_term`, `_locate`.

This is a load-bearing choice: it touches `ir.py` and every place
that compares terms by `id()`. Audit `id(`/`is ` usages in
egraph/cost/meta before starting.

### 2b. `egraph.py` → `catopt/egraph/` package

- `unionfind.py` — find/union, proof-edge record.
- `nodes.py` — ENode, leaf registry (module-private, typed).
- `matching.py` — e-matching, instantiated substitutions.
- `saturate.py` — `run`, dirty-frontier, rule budgets.
- `extract.py` — `extract_best`, `extract_paired`, `dag_cost` usage.
- `proof.py` — `certificate`, `verify_certificate`, `coherent_paths`,
  `_class_of_term`, `_locate`.
- `__init__.py` re-exports the current public surface — no caller
  changes.

### 2c. Explicit registry composition

Replace import-side-effect registration with explicit tables:

- `OpTable` (torch bindings + shape rules + attr schemas) and
  `LawSet` — constructed by composition: `Ops.core() + Ops.carrier(
  "om", "aff", "trace")`, not by importing modules for side effects.
- `optimize_model(..., ops=..., laws=...)` parameters default to the
  full set — same behavior, visible dependency.
- `act_eps`/`ibp`/carriers stop mutating `_IR_TO_TORCH` at import;
  they export `bindings()`/`shape_rules()` dicts the table folds in.

### 2d. `rules.py` → `catopt/laws/` by domain

Each module owns one domain's full surface: law definitions,
side-condition `check`s, shape-rule registration, torch bindings,
witness constructors. `tensor.py` (fold/assoc/distribute),
`scan.py`, `om.py`, `trace.py`, `pairing.py` (non-local passes —
admit they're passes, not laws). Public `all_rules()` unchanged.

Exit: API-compatible internals; `id(`-keyed memos gone; no import
side effects; packages navigable.

## Phase 3 — Ergonomics + robustness

### 3a. Typed reports

`OptResult`, `BlockReport`, `ParamReport` dataclasses replace loose
stats dicts — fields typed, schema is code, not convention.
`optimize_model`/`optimize_compositional` return them; benches read
attributes not string keys.

### 3b. Uniform verify gate

`verify_equiv(ref, opt, inputs, tol) -> VerifyReport` — one
implementation used by optimize_model's internal check,
compositional per-block + e2e checks, and the bench files.
Currently near-duplicated three times with different tolerances and
fallback semantics.

### 3c. Observability

`logging` module (named logger `catopt`, levels DEBUG/INFO) replaces
`print(..., verbose=...)`; per-rule fire counts, saturation stats,
extraction decisions emitted as structured events — queryable in
tests via `caplog` / a `run_stats` object.

### 3d. Resilience

- OOM/failure policy: `optimize_model` failures inside
  compositional already fall back per-block — keep, but make the
  e-graph memory path bounded (`max_enodes` enforced with a cheap
  watermark check before rule application, not after).
- A `requires_gpu`/`requires_ckpt` pytest marker for bench-level
  tests so `pytest` stays green on CPU/CI without a GPU.

## Coverage plan — the 100% gate

- **Target: 100% line + branch coverage of `catopt/`**, verified by
  `pytest --cov=catopt --cov-branch --cov-report=term-missing`.
- Interim ratchet (day 0 → gate):
  baseline → `fail_under` set to baseline → every phase's exit
  requires the new floor. Public-API modules (ir, egraph,
  optimize, torch_bridge, cost, rules/laws) reach 100% first;
  executors and regime/eps tail last but still reach it.
- Honest exclusions only via `exclude_also`/`# pragma: no cover` —
  every exclusion carries a comment stating why the line is
  unreachable (defensive re-raise, platform branch, debug assert).
- Gap-closing order: contract tests (1c) cover the boundary ops
  wholesale; per-phase "cover the lines you touched" rule; a
  coverage audit pass at each phase exit listing the remaining
  uncovered lines (checked into `project/`).
- Benches, `main.py`, `measure_weights.py`, `exact_probe.py` stay
  outside the coverage gate (measurement harnesses, not product),
  and move under `bench/` if they're kept at all.

## Execution order

| Order | Item | Depends on | Risk |
|---|---|---|---|
| 0 | Tooling (uv, ruff, coverage, ratchet, hooks) | — | low |
| 1 | Per-op contract tests + binding coverage | 0 | low — pure tests |
| 2 | typing.py split | 1 | med — 5 call sites |
| 3 | Attr canonicalization + schemas | 2 | med — boundary touches everything |
| 4 | Hash-consed terms | 3 | high — id() audit first |
| 5 | egraph → package | 4 | low after interning |
| 6 | Registry composition | 5 | med — callers rewire |
| 7 | laws/ split | 6 | low |
| 8 | Typed reports + verify gate | any | low |
| 9 | Observability, resilience markers | any | low |

Every step lands green: suite + lint + coverage floor ratcheted.
No step changes public API without a deprecation shim.

## Acceptance criteria

- `uv run pytest --cov=catopt --cov-branch` → **100%** on `catopt/`,
  suite green.
- `uv run ruff check` clean; `ruff format --check` clean.
- No `id(`-keyed memoization; no `keepalive` workarounds; no
  import-side-effect registration.
- Public surface unchanged (`optimize_model`,
  `optimize_compositional`, `IRModule`, `all_rules`, cost fns,
  carriers) or shimmed.
- Benchmarks (`bench_gpu.py`, `bench/stories15m_bench.py`) still run
  and report the same regime verdicts — refactor must not silently
  move the numbers.

## Non-goals

- No new optimization laws/carriers in this plan — structure only.
- No API redesign of the user-facing entry points.
- eps/ibp/act_eps modules get the same structural treatment but no
  feature work.


---

## Progress log (main branch)

| Phase | Commit | Result |
|---|---|---|
| 0 tooling | `97b79d8` | uv venv (system-site torch), ruff config, coverage config, pre-commit; baseline 80% |
| 1a typing split | `0cdab72` | `catopt/typing.py`; `register_shape_rule` protocol; cost.py 1509→990 |
| 1b attr schemas | `709f2c0` | `catopt/attrs.py` ATTR_SCHEMA + mint-time validation; layer_norm eps bug found |
| 1c contract tests | `0cdab72` | `test_contracts.py` found `max` unbound + rms_norm eps positional on first run |
| 2a hash-consing | `2f2b809` | `Op.make` interns; ~20 id()-memos → content keys; keepalives deleted |
| 2b egraph→pkg | `99143f1` | `catopt/egraph/` = types/certs/terms/extract/proof/core via mixins |
| 2c registries | `3dce571` | `OpTable.core()/full()`; import side-effects removed |
| 2d laws/ split | `5c4919a` | `rules.py` 2020→60-line shim; laws/{base,tensor,scan,pairing} |
| lint cleanup | `adf030f` + sweep | ruff 876→0 repo-wide |
| coverage | `5c4919a` | 80%→87.6%; ibp 45→94, optimize 75→92, executors ≥89; 6 real bugs found+fixed |
| 3a/3b reports+verify | `90dca02`,`bc1ea0e` | `catopt/report.py` typed dataclasses + `verify_equiv`/`verify_module` gate |
| 3c logging | `90dca02` | `logging.getLogger("catopt.*")`; no compat prints needed |
| 3d resilience | `90dca02` | requires_cuda markers + OOM guard (`OptimizationResourceError`) — done pre-refactor |

Real bugs the refactor surfaced (beyond the two it was designed around): `max` op unbound, rms_norm eps positional, layer_norm eps misread, ibp `_inf_box` sentinel crash, `xs[0]` IndexError on param-only plans ×3 modules, `_matmul` dtype promotion, `_select_index` getitem spelling, unbind default dim, omd batched-seed memo silently dead (id-keyed vs content-keyed), `eps_rtol` docstring overclaim.

Current state (bc1ea0e): **1009 tests**, ruff 0 findings, coverage 87.77% (fail_under ratchet 87 → 100 target open).
