# `softmax_fold` shipped — the second machine-discovered law, first with a side condition

`law-proposer-extensions.md` reported the pipeline's second SHIP
verdict: `recognize:softmax` —
`div(exp(u), sum(exp(u), dim, keepdim)) → softmax(u, dim)` — true on
the numeric oracle, new, firing once on `ManualSoftmaxAttention` for a
19.0 % end-to-end cost drop, certificate replaying, closure 1.06×.
This retro records **admission**: the law is now in the library as
`softmax_fold`, tagged `SIMPLIFICATION`, in `SIMPLIFICATION_RULES` —
so it is in `DEFAULT` and fires in the public pipeline.

It is also the first admitted machine-discovered law whose
precondition is **not structural**.  `select_mul`'s side condition
lived in the pattern (shared `dim`/`index` attr metavariables — the
matcher enforces it); the softmax fold's does not, and this admission
is the first shipped use of the `check`/`derive` hooks on a
simplification rule:

```python
def _check_sum_keepdim(bound) -> bool:
    """Guard the softmax fold: keepdim and a single reduce axis."""
    dims = bound.get("$attr:RD")
    if bound.get("$attr:RK") is not True:
        return False
    return isinstance(dims, int) or (
        isinstance(dims, tuple) and len(dims) == 1
    )


def _derive_softmax_dim(bound) -> dict:
    """Unwrap ``sum``'s ``dim`` tuple into ``softmax``'s scalar dim."""
    dims = bound.get("$attr:RD")
    return {"$attr:SD": dims[0] if isinstance(dims, tuple) else dims}


SOFTMAX_FOLD = R(
    "softmax_fold",
    Op.make(
        "div",
        Op.make("exp", "u"),
        Op.make("sum", Op.make("exp", "u"), dim="RD", keepdim="RK"),
    ),
    Op.make("softmax", "u", dim="SD"),
    law="exp(u) / Σ exp(u) = softmax(u): the manual normalization fold "
    "— a composed-then-reduced chain IS the kernel's definition.  "
    "Folds div+exp+sum to one dispatched op.",
    check=_check_sum_keepdim,
    derive=_derive_softmax_dim,
    tags=_SIM,
)
```

The two hooks answer two different questions:

* `check` carries the *equality* side condition — `x / Σ x` only
  broadcasts to a softmax when the sum keeps its dim (a dropped dim
  broadcasts wrongly, or not at all, against the numerator) and covers
  exactly one axis (softmax has no multi-axis image).  Both spellings
  the IR can carry are accepted: the exporter's `dim=(-1,)` tuple and
  a hand-minted scalar `dim=-1`.
* `derive` carries the *attr translation* — `softmax`'s `dim` is a
  scalar int while `sum`'s is a tuple, so the RHS attr cannot be bound
  verbatim from the match; it is computed per firing.

No matcher hacking was needed: `$attr:` bindings arrive in `bound` as
raw values (`True`, `(-1,)`), exactly as the pipeline's validated
proposal assumed.

## 1. IR verification before admission

Printed `export_to_ir(ManualSoftmaxAttention(8))` before writing the
rule — the actual attr types matched the proposal's assumptions:

    div (exp s), (sum (exp s), dim=(-1,), keepdim=True)

* `sum` carries `dim` as a **tuple** `(-1,)` and `keepdim` as a real
  `bool` — `sum` has no `ATTR_SCHEMA` entry, so its attrs pass through
  the exporter unvalidated.
* `softmax` is in `ATTR_SCHEMA` (`{1: "dim"}`) and `ATTR_REQUIRED`, so
  any fully-attributed softmax term must name `dim`; the RHS pattern's
  `dim="SD"` metavar satisfies the mint.
* The torch binding lowers `softmax` as
  `F.softmax(x, dim=int(attr_of(kw, "dim", default=-1)))` — the
  derived scalar is the exact form it reads.

## 2. Corpus measurement — law in vs out, shipped set

Thirty real model graphs (`tools/law_impact.model_cases`), saturated
under `ALL_RULES` vs `ALL_RULES \ {softmax_fold}` (6 iters, 60 k enode
cap, pipeline cost model):

    ManualSoftmaxAttention  enodes 17->18  classes 17->17
        cost 1.829e5 -> 1.481e5   −19.03%   fires 1   cert pass
    all other 29 models: 0 fires, 0% change, identical enode counts

* **1 firing total** — exactly the pipeline's count.
* **19.03 %** extracted-cost drop on the one model that spells
  softmax by hand; the extracted term literally reads
  `softmax(div(matmul, 2.828), dim=-1)`.
* **0 regressions** — every other model's extracted term, cost, enode
  and class counts are identical.
* **Certificates replay** on every model row (with and without the
  law).
* **Closure-safe:** enode ratios min/mean/max = 1.000 / 1.002 / 1.059
  — the 1.059 is the fold's own single extra enode, matching the
  pipeline's 1.06× and far under the 2.0 ship gate.  Single-direction
  and term-local, like `select_mul`: DEFAULT inclusion is what
  delivers the effect.

`law_bench` on the registered synthetic case (`sizes 64`, CUDA):

    softmax_fold  fired 1  rhs picked  verify pass
        cost 5.226e4 -> 1.742e4   ms 0.0203 -> 0.0103  (~2.0x)

## 3. Tests — the ship pattern, extended for check/derive

`tests/test_softmax_laws.py` (12 tests) mirrors
`tests/test_select_laws.py` and adds unit coverage the hooks need:

* registration/tagging (`SIMPLIFICATION_RULES`, `DEFAULT`,
  `{SIMPLIFICATION}`);
* match/instantiate round-trip — LHS metavars are exactly
  `{u, $attr:RD, $attr:RK}`; the RHS needs `derive` for `$attr:SD`;
* `check`/`derive` branch-by-branch (tuple/int dim, keepdim off,
  multi-axis, dimless);
* fp64 soundness — `allclose(atol=1e-15)`, not bitwise: the kernel's
  max-subtraction rounds differently (~5.6e-17);
* fires + RHS-is-member, on both the tuple- and int-`dim` spellings;
* three declines: `keepdim=False` (check), multi-axis `dim=(-2,-1)`
  (check), `div(exp u, sum(exp v))` with u ≠ v (the repeated-`u`
  metavar — matcher, no check needed);
* end-to-end `ManualSoftmaxAttention`: fires, cost drops, certificate
  replays, lowered before/after modules pass `sink.verify` fp64;
* fires inside the public `Optimizer(backend=TorchBackend())` default
  pipeline, `lower(verify=True)` passes.

Plus: a fuzzer spec in `test_contracts._LAW_SPECS`
(`{"u": (2,3)}, $attr:RD=(-1,), $attr:RK=True` — check accepts,
derive produces `dim=-1`), structure counts in `test_laws_structure`
(15 simplification, 53 `all_rules()`, 130 with-layout, 189 module
rewrites), and the `_c_softmax_fold` bench case.  The generic
Hypothesis round-trip over `ALL_RULES` covers the new rule with no
changes.

## 4. Surprises / notes

* **`sum.dim` is a tuple; `softmax.dim` is a scalar.**  The attr-type
  mismatch the pipeline's `derive` was built for is real in the
  export, not hypothetical.  `check` accepts the int spelling too
  because `Op.make` places no schema on `sum` — a hand-minted
  `sum(x, dim=-1)` is legal IR and folds correctly (tested).
* **`torch.softmax ≠ exp/sum` bitwise.**  The stabilized kernel
  subtracts the row max first; fp64 agreement is ~5.6e-17, so the
  soundness test uses a tight `allclose`, and `sink.verify`'s
  `rtol=1e-4` is what the end-to-end gate always meant by "equal".
* **The pipeline now reports the law as `duplicate`.**  `_relation`
  keys candidates alpha-normally, so a re-run finds
  `recognize:softmax` already in `ALL_RULES` and reports
  `not new (duplicate)` — the expected post-admission signature, same
  as `census:mul_select` after `select_mul` shipped.
* **Provenance.**  Proposed by `_pattern_recognition` (the
  composed-then-reduced census scan), not by naturality — the first
  SHIP from that proposer family, and the first that needed `check` /
  `derive` at all.

## 5. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass (128 files)
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/lint-imports` — 4 contracts kept
* `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 findings
* `.venv/bin/semgrep --config .semgrep.yml packages` — 0 findings
* `.venv/bin/python tools/radon_ratchet.py` — ok (1905 functions; the
  two new hooks are rank A)
* `uv run pytest -q` — **3382 passed, 31 skipped**
* `coverage run … && coverage report` — **100 %** (18175 stmts, 6856
  branches, 0 missed)
