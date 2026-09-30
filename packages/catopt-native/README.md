# catopt-native — the optional native search engine

A Rust/PyO3 port of the catopt equality-saturation **search core**,
behind the `catopt_core.ports.Engine` protocol.  Optional and
explicit: the pure-Python `catopt_core.egraph.EGraph` remains the
reference implementation, the correctness oracle, and the default
engine — installing this package changes **nothing** until
`engine=` is passed.

## What it is

* `src/` — the Rust core: union-find, hash-consed enode storage, the
  compiled-pattern matcher (`_m_bounded` semantics, attr metavars,
  shared-metavar consistency), RHS instantiation, rebuild/congruence
  closure, the dirty-frontier incremental saturation loop, per-rule
  enode budgets, and `min_term` member resolution for the
  `check`/`derive` bridge.
* `python/catopt_native/engine.py` — the `NativeEngine` adapter:
  term/pattern serialization, `bound`-dict materialisation through the
  injected `build_term` hook, and extraction via the **reference**
  `_ExtractMixin` over a materialised `_classes` view — so extracted
  terms and costs are identical *by construction*.
* `python/catopt_native/_native.pyi` — type stub for the extension.

## Scope boundary — search only

* **No proof machinery**: no merge log, no applications, no
  certificates (`truncation_level = 1`).  `union(..., witness=...)`
  accepts and drops proof metadata.  A run that needs a certificate
  uses the Python engine.
* **Non-local passes stay Python**: the pairing/carrier lifts that
  offer members through `union(witness=...)` run only on the Python
  engine — `optimize.search` skips them under `engine=` and records
  `stats["nonlocal_passes"]`.
* `check`/`derive` remain **Python callables**: the Rust matcher calls
  them through PyO3 with the same resolved `bound` dict the Python
  engine builds (real `Op`/leaf terms plus `$attr:` values).

## Building

Requires a Rust toolchain (cargo ≥ 1.75) and Python ≥ 3.11.  The
package is deliberately **not** a uv workspace member, so `uv sync`
never needs cargo — and will *uninstall* the wheel; rebuild it
afterwards.

```sh
# one-time: the PEP-517 build frontend
uv tool install maturin        # or: uv run --with maturin

# editable install into .venv
cd packages/catopt-native
VIRTUAL_ENV=../../.venv maturin develop --release

# or a plain wheel install
maturin build --release
pip install target/wheels/catopt_native-*.whl
```

`import catopt_native` works only after the extension is built; every
native test self-skips via `pytest.importorskip`/`skipif` when it is
not.

## Using it

```python
from catopt_native import NativeEngine
from catopt_orchestrator.optimize import search, Optimizer

res = search(model, x, source=source, engine=NativeEngine)
# or persistently:
opt = Optimizer(backend=backend, engine=NativeEngine)
res.stats["engine"]  # "native"
```

`engine=` accepts an instance, a class, or a zero-arg factory.  It is
**never auto-detected**.

## Differential oracle

`tests/test_native_engine.py` validates native vs Python on a corpus:

* unbudgeted runs — identical e-class counts, identical live enodes
  (stale hash-cons keys aside), identical extracted terms and costs;
* randomized Hypothesis corpus over the symmetry rules;
* check/derive + `$attr:` metavar rules (SDPA fold, naturality);
* a real torch-exported model through `search` end-to-end.

Measured (this machine, release build): ~**17×** on the match-bound
`assoc_matmul` Catalan closure (k=11: 18.4 s → 1.1 s), ~3× on
mixed-rule saturation, ~2.5× on budgeted expansive runs.  Per-rule
`rule_fires` counts and budget-truncated fragments are
enumeration-order-dependent (Python's own set-hash order also varies
run to run); the fixed point and extracted results are not.

## Not yet ported (deliberate)

* Extraction itself stays the Python `_ExtractMixin` over the
  `_classes` view (identical semantics by construction; the FFI
  boundary means a Rust extractor would still price terms in Python —
  the dominant win is the search).
* `stop="improving"` drives `run_iteration` from Python (re-extraction
  between iterations needs the reference extractor anyway).
* Certificate/`extract_best_bounded`, `any_term`/`_locate`,
  `matches`/`apply_rule` as public calls — Python-engine surface.
