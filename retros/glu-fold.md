# `glu_fold` shipped — the manual-GLU fold, second `chunk`-spelled kernel fold

`corpus-expansion-r2.md` §6 lead #2: the corpus had the kernel image
(`GluMLP`'s `glu` node — the only corpus `glu`) but no manual
spelling, so a `mul-split-sigmoid → glu` fusion law would see only one
side of the equation.  `softmax_fold`/`silu_fold` shipped from
exactly this recipe: put the hand-spelled math in the corpus, fold it
to one dispatched kernel op, certificate on.  This retro records the
closed pair: **`ManualGluMLP`** joined the corpus and **`glu_fold`**
is in the library — `SIMPLIFICATION`-tagged, in
`SIMPLIFICATION_RULES`, so it is in `DEFAULT` and fires in the public
pipeline.

## 1. The model — the manual spelling the lead asked for

`catopt_torch.models.ManualGluMLP` —
`x → chunk(up(x), 2, -1) → a·σ(b) → down`, the same up/down
projections as `GluMLP`.  Registered in
`tools/law_impact._model_cases()` (41 cases, 0 export errors) with a
`tests/test_corpus_models.py` case.

The export's getitem fold (`torch_bridge` rewrites
`getitem(split/chunk, i)` into the splitter carrying `index`) lands
the slice index on the `chunk` node itself, so the term is *already*
the law's shape — no `getitem` member intervenes:

    linear
      mul
        chunk {chunks 2, dim -1, index 0}   linear(x, p_up_weight)
        sigmoid
          chunk {chunks 2, dim -1, index 1} linear(x, p_up_weight)
      p_down_weight

Verified by printing `export_to_ir` before writing the rule — the
attr set `{chunks, dim, index}` is exactly what the pattern's keyset
must equal.

## 2. The law

```python
GLU_FOLD = R(
    "glu_fold",
    Op.make(
        "mul",
        Op.make("chunk", "u", chunks=2, dim="D", index=0),
        Op.make(
            "sigmoid",
            Op.make("chunk", "u", chunks=2, dim="D", index=1),
        ),
    ),
    Op.make("glu", "u", dim="D"),
    cond=_COND_GLU_FOLD,   # ("rank", "u", ">=", 1)
    check=_check_glu_fold,  # u.shape[D] a known even int
    tags=_SIM,
)
```

Most of the precondition is **structural** — no matcher hacking:

* the shared `u` metavar binds both chunk operands to the same
  e-class (chunks of different sources never match);
* the shared `D` attr metavar pins both to the same split axis *and*
  carries it to the RHS — no `derive` needed, unlike `softmax_fold`;
* literal `chunks=2` / `index=0|1` attrs pin the halves split and the
  gate order (`F.glu` computes `a ⊗ σ(b)`, `a` the first half) — the
  swapped-gate and `chunks=4` spellings cannot match.

The residual guard is the one thing the DSL cannot say: **parity of
the split axis**.  `glu` halves its `dim` exactly, but
`chunk(·, 2, d)` splits an odd axis *first-big* — `n=3` gives `2+1`,
and `(…,2)·(…,1)` still broadcasts, so an odd-axis redex evaluates
fine while its `glu` image raises at eval.  The `cond` DSL has no
predicate that indexes a shape by an *attr-named* axis (`axis`/
`dim-eq-const` take literal positions), so parity stays a procedural
`check` over `cond`'s expressible front (`u` shaped, rank ≥ 1 — a
scalar has no axis to halve).  Unknown/`None` dims ON the split axis
decline (the library's strict posture); `None` dims elsewhere do not
matter.  `missing_hooks` honestly reports `("check",)` — the serial
census is now 39 full-data / 14 derive / 2 check / 2 check+derive of
57 laws.

Derivability: **axiom** — nothing else in the library mints a `glu`
term, so `glu_fold` is a new kernel member (43 axioms / 57 rules; the
32 primitives + 11 cycle representatives).

## 3. Corpus measurement — law in vs out, shipped set

41 real model graphs (`tools/law_impact.model_cases`), saturated
under `DEFAULT` vs `DEFAULT \ {glu_fold}` (6 iters, 60 k enode cap,
pipeline cost model):

    ManualGluMLP  enodes 9->10  classes 9->9
        cost 8.706e4 -> 5.226e4   −39.98%   fires 1   cert pass
        extracted: linear(glu(linear(x, p_up), dim=-1), p_down)
    all other 40 models: 0 fires, identical enode counts and costs

* **1 firing total** — exactly the shape the corpus was missing.
* **−39.98 %** extracted-cost drop; the `glu` kernel literally lands
  in the extracted term.
* **0 regressions** — SwiGLU's `mul(silu(…), …)` and GegluMLP's
  `mul(gelu(…), …)` do not match (the gate op is not `sigmoid` over a
  `chunk`); `swiglu_fuse`'s own chunk RHS has `silu`, not `sigmoid`.
* **Closure-safe:** enode ratios min/mean/max = 1.000 / 1.003 / 1.111
  — the 1.111 is the fold's own single extra enode, far under the
  2.0 ship gate.  Term-local and single-direction.

`law_bench` on the registered synthetic case (`sizes 128`, CUDA):

    glu_fold  fired 1  rhs picked  verify pass
        cost 5.226e4 -> 1.743e4   ms 0.0213 -> 0.0124  (~1.72x)

## 4. Tests — the ship pattern, extended for cond+check

`tests/test_glu_fold_law.py` (16 tests) mirrors
`test_silu_fold_laws.py` and adds the guard's branch coverage:

* registration/tagging (`SIMPLIFICATION_RULES`, `DEFAULT`,
  `{SIMPLIFICATION}`, `kind == "axiom"`);
* match/instantiate round-trip — LHS metavars are exactly
  `{u, $attr:D}`; `D` carries to the RHS with no `derive`;
* fp64 soundness — `allclose(atol=1e-15)` on both `dim=-1` and
  `dim=0` (the axis is data, not a literal);
* fires + RHS-is-member, including an inner-axis (`dim=0`) instance;
* the matcher declines — different sources, different dims,
  `chunks=4`, swapped gate (`σ` on `index=0`), sigmoid-free
  `mul(chunk0, chunk1)`;
* the check declines — odd split axis (`(4,3)`), `None` ON the split
  axis, out-of-range/non-int dim attrs, scalar `u`; and the positive
  edge — `None` dims *off* the split axis still fire;
* end-to-end `ManualGluMLP`: fires, cost drops, certificate replays,
  lowered before/after modules pass `sink.verify`;
* fires inside the public `Optimizer(backend=TorchBackend())` default
  pipeline, `lower(verify=True)` passes;
* `glu(u)` itself is not a redex — the fold never loops.

Plus: a fuzzer spec in `test_contracts._LAW_SPECS`
(`{"u": (2,4)}, $attr:D=-1` — check accepts), updated structure counts
(57 `all_rules()`, 19 simplification, 134 with-layout, 193 module
rewrites, 43 axioms, 32 cond-carrying), the serializability census,
and the `_c_glu_fold` bench case.

## 5. Surprises / notes

* **The `n=3` corner is the real reason the check exists.**  A
  two-chunk split on an odd axis is not halves — and `(2+1)` still
  broadcasts against a `(…,1)` sigmoid operand, so the redex is
  *evaluable* yet its `glu` image raises.  This is the
  matcher-cannot-see-shapes class the strict shape-guard posture was
  invented for; without it the law would mint a member `sink.verify`
  can only reject, never prove.
* **One firing direction only.**  The LHS pins the exported operand
  order `mul(chunk0, σ(chunk1))`; the commuted spelling
  `σ(b)·a` exports as a different arg order and does not match —
  `COMM_MUL` is `SYMMETRY`-tagged and out of `DEFAULT`, so no
  canonicalization rescues it.  A second pattern instance would be a
  cheap follow-up if a real model needs it.
* **Concurrent landing.**  `rms_norm_fold` /
  `rms_norm_fold_nogain` (the §6 lead #1) shipped in the same working
  tree during this change — different regions of `tensor.py`, shared
  count pins updated to the joint state (57 rules, 32 conds, 43
  axioms).
* **`AGENTS.md` drift.**  Its "29 of 54 laws use it [cond]" line is
  now 32 of 57 — whoever commits last should refresh it.

## 6. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass (145 files)
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/lint-imports` — 4 contracts kept, 0 broken
* `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 findings
* `.venv/bin/semgrep --config .semgrep.yml packages` — 0 findings
* `.venv/bin/python tools/radon_ratchet.py` — ok (2038 functions;
  `_check_glu_fold` is rank A at complexity 5)
* scoped pytest (all touched areas — corpus, laws-structure,
  serialize, cond, contracts, rulesets, silu/matmul regression,
  egraph/meta/certificates, falsified pins, select-laws):
  **~1100 passed**, 0 failed
* scoped coverage of the new code: `ManualGluMLP` and
  `_check_glu_fold` fully covered by `test_glu_fold_law.py` +
  `test_corpus_models.py` (every check branch exercised)
* **full suite not run** — resource constraint on this box per the
  AGENTS.md parallel-warning; the coverage gate stays a commit-time
  check.
