# Plan 0002 — Genericity: shared machinery, plug points, deduplication

Status: proposed
Date: 2026-09-26
Branch: `project` (orphan — plans live here, not on main)
Follows: `0001-refactor-and-quality.md` (phases 0–3 + hardening complete)

## Goal

After the structural refactor the codebase is modular and 100%-covered —
but it still re-implements the same small machines in several places.
This plan extracts the genuinely-shared machinery into homes keyed by
the ports layer (`catopt/ports.py` already names the boundaries), keeping
each executor/carrier free to override where its semantics differ.

Design rules:

1. **Merge only what's semantically identical.** Two functions that look
   alike but differ in semantics stay separate (e.g. `_eval_const`'s
   "param-only or None" vs `_eval`'s "full eval"). Duplication of *shape*
   without duplication of *meaning* is fine — we deduplicate meaning.
2. **Every merge must keep all 1,564 tests green** — the suite is the
   contract that two implementations were in fact identical.
3. **No new abstractions without ≥3 real call sites.** Ports stay
   Protocols — composition over inheritance; mixins only where a class
   today already contains the same method 3×.
4. **Backward compatibility** — public names keep working; moves land
   behind the existing shims.

## Phase A — Executor shared base (trivial-merge, ~60 lines)

Three batched executors (`om_lower`, `omd_lower`, `scan_lower`) already
conform to `catopt.ports.BatchedExecutor` and carry byte-identical:

- `capture_cuda_graph` (×3, ~18 lines each — identical, CUDA-only)
- `is_graph_captured` / `drop_cuda_graph` (×3)
- `_forward_impl` **prologue**: `x = xs[0] if xs else None` + env build
  from `self._inputs` (×3 — identical lines)
- `ev()` memo-eval closure shape (×5 sites total incl. streaming)

Work:

- New `catopt/exe/` package (executor implementations grouped):
  - `exe/base.py` — `_BatchedBase` mixin: `is_graph_captured`,
    `drop_cuda_graph`, `capture_cuda_graph` (one copy), `_input_env(xs)`
    returning `(x, env)`, `is_batched`/`fallbacks` counters.
  - `exe/scan.py`, `exe/om.py`, `exe/omd.py` — classes mix in the base;
    only their plan-execution sections stay bespoke.
  - Back-compat: `catopt/{scan,om,omd}_lower.py` re-export; OR keep
    modules in place and only extract the mixin (`exe/base.py` +
    `from catopt.exe.base import _BatchedBase`) — **decide from the
    audit map**: minimal-diff version = mixin only, no moves.
- Acceptance: 3 copies → 1; identical `_forward_impl` prologue gone;
  executor tests unchanged.

## Phase B — Dual-spelling attr reads (mechanical, ~42+ sites)

`attrs.get("dim", attrs.get("arg1", default))` patterns repeat across
torch_bridge, typing, ibp, laws/*, xcarrier, om*, eps*, meta, regime.
They're the post-canonicalization compatibility reads (declared dual
spellings are legal — the attrs contract is unchanged).

- New helper in `catopt/attrs.py`:
  `attr_of(op_or_term, *names, default=None)` — returns first present;
  reads `term.attrs` if given a term, dict otherwise.
- Convert every multi-name `.get(...)` chain to `attr_of`. Single-name
  reads stay `.get`.
- Mechanical sed-style sweep with a test run after each module.
- Acceptance: ~40 sites → helper calls; zero behavior change.

## Phase C — Memo prologue + eval-loop normalization (small)

- `memo = {} if memo is None else memo` + `k = (tag, term)` appears ~15+
  times with identical semantics (content-keyed DAG memo). Add
  `catopt/ir.py`: `memo_or(memo)` (or keep — micro). **Decision: only
  merge the prologue line, not the whole memo dict shape** — call sites
  need the dict locally anyway; a one-line helper `memo = m or {}` is
  all that's earned. Low value — may skip after audit shows exact count.
- `_eval_const` (optimize) vs `IRModule._eval` (torch_bridge) vs each
  executor's `ev()` — verify semantics differ where claimed; unify ONLY
  the identical parts (probably just the env/xs convention from Phase A).
- `share_duplicate_params` / `share_duplicate_param_slices` /
  `_folds_to_param` / `_fold_weight_chains` — audit the param-constness
  predicate: if they implement "term contains only param/const leaves"
  in 4 ways, extract `ir.py: is_param_only(term, source_tensors, memo)`
  once.
- Acceptance: predicate extracted if truly identical; tests pin it.

## Phase D — Plan-vocabulary typing (extensibility, opt-in)

Executor plans are `dict`-of-any (`plan["map_mode"]`, `plan["leaves"]`,
`plan["gather"]`). As a ports-layer polish:

- Typed plan dataclasses per executor (`ScanPlan`, `OmPlan`, `OmdPlan`)
  OR a shared `Plan` protocol + dataclass fields — keeps `.get(key)`
  consumers working via `asdict` at the boundary, OR keep dicts and
  document the vocabulary in `ports.py`.
- **Decision deferred to implementation**: prefer typed dataclasses
  with a `.get()` shim if any external consumer reads plans; otherwise
  plain dataclasses.
- Acceptance: plan construction sites build typed objects; executor
  internals keep working.

## Phase E — Recognizer consolidation (needs-care)

`_select_index`/`_sliced_gather`/`_qk_parts`/`_part_gather` recognizers
exist in law-side (xcarrier) and lowerer-side (om_lower/omd_lower)
forms — *similar shapes, different outputs* (laws emit offers; lowerers
emit plan rows). Audit decides whether a shared "gather recognizer"
kernel exists under both; merge only if the recognizer core is
bit-identical modulo output mapping. Otherwise document the intentional
parallel (law detects structure, lowerer consumes it).

## Out of scope

- Unifying laws and lowerers semantically (they answer different
  questions).
- Any public API rename.
- Meta/eps/regime internals beyond the shared predicates above.

## Execution

| Phase | Agent count | Size |
|---|---|---|
| A executor base | 1 | ~60 lines extracted, mixin into 3 classes |
| B attr_of sweep | 1 | ~42 sites, mechanical |
| C predicates | 1 | param-only predicate + eval/env dedup |
| D plan typing | 1 | 3 plan vocabularies |
| E recognizers | audit then decide | — |

A–C can run in parallel (disjoint files). D after A lands. E gated on
the audit map.

## Acceptance

- [ ] `capture_cuda_graph` exists once.
- [ ] Executor `_forward_impl` prologues share `_input_env`.
- [ ] `attr_of` used everywhere dual-spelling reads existed.
- [ ] Param-only predicate unified.
- [ ] All 1,564 tests green; 100% coverage maintained (extracted code
      arrives covered via its existing call sites).
- [ ] pyright clean on touched files; new files auto-checked.
- [ ] ruff clean.

## Risks

- **Behavior drift in "identical" code** — mitigated by the suite; the
  100%-coverage suite means a wrong merge fails a test.
- **`dict`→dataclass plan consumers** — grep for plan `.get(`/`["` uses
  before touching plan types.
- **Executor subclassing vs mixin** — mixin only; the classes keep
  their names/bases.
