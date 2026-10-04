# `rms_norm_fold` shipped — the fourth machine-discovered law family, first with a *shape-derived* attr

`corpus-expansion-r2.md` §6 named the clearest recognizer lead: the
corpus now carries BOTH spellings of RMSNorm — the manual
`x·rsqrt(mean(x²)+eps)·w` (`RMSNorm`, `NormLinear`, `TransformerBlock`,
`ParallelBlock`) and `NativeRmsNorm`'s fused `rms_norm` kernel call —
exactly the `softmax_fold` situation one norm family over.  This retro
records the admission: **two** laws now live in the library as
`rms_norm_fold` and `rms_norm_fold_nogain`, tagged `SIMPLIFICATION`,
in `SIMPLIFICATION_RULES` — so both are in `DEFAULT` and fire in the
public pipeline.

Why two: the gain-free `mul(u, rsqrt(mean(u²)+eps))` is *itself* a
real spelling (llama-style weight-free RMSNorm) and is also the inner
`mul` of the gained fold's LHS — so the pair covers `x·rms⁻¹` and
`x·rms⁻¹·w` with one shared side condition.  Inside a gained site the
nogain fold mints `mul(rms_norm(u), w)`, which lands in the same
e-class as the gained fold's `rms_norm(u, w)`; the cost model picks
the fused member.

It is the first admitted law whose `derive` mints a **shape**, not an
axis/flag translation:

```python
_COND_RMS_FOLD = (
    "and",
    ("attr-is", "MK", True),
    ("const-num", "EPS"),
    ("const-cmp", "P", "==", 2),
)

RMS_NORM_FOLD = R(
    "rms_norm_fold",
    Op.make("mul", Op.make("mul", "u", _rms_reduce()), "w"),
    Op.make("rms_norm", "u", "w", dim="ND", eps="EP"),
    law="x·rsqrt(mean(x²)+eps)·w IS rms_norm(x, w): ...",
    cond=_COND_RMS_FOLD,
    check=_check_rms_fold,
    derive=_derive_rms_norm,
    tags=_SIM,
)
# _rms_reduce() = rsqrt(add(mean(pow(u,P),dim=MD,keepdim=MK), EPS))
```

The three hooks split the precondition three ways:

* **structural** — the op-tree plus the shared `u` metavariable (the
  numerator and the `pow` operand must be the same e-class; the
  matcher enforces it, no check needed);
* **`cond` (the DSL front)** — `keepdim` is `True` (a dropped axis
  cannot broadcast the rms back), `eps` is a numeric `Const` leaf
  (the kernel's `eps` is a float attr — a tensor `eps` has no image),
  and the `pow` exponent is numerically 2 (`const-cmp`, so `2` and
  `2.0` both fold);
* **`check` (the procedural residual)** — what the DSL cannot say:
  the `mean`'s reduce dims must name *exactly u's last k axes*
  (`F.rms_norm` only normalizes a trailing block — the
  `dim-eq`/`axis` predicates can't express set-equality over an
  attr tuple), and, for the gained fold, `w`'s shape must BE that
  trailing block (`aten.rms_norm` rejects any other weight shape at
  eval — a mismatch would mint an unlowerable member).

`derive` then computes the two RHS attrs the LHS cannot bind
verbatim: `dim` is the *normalized shape* `u.shape[-k:]` — a shape
tuple, not the reduce dims — and `eps` unwraps the bound `Const`
leaf into the float attr.  Unknown/`None` dims and unshaped `u`/`w`
decline (the strict posture: a law never mints an attr it cannot
verify).

## 1. IR verification before admission

Printed `export_to_ir` on all four manual-spelling models plus the
kernel reference:

    RMSNorm:     mul(mul(x, rsqrt(add(mean(pow(x, 2), dim=(-1,), keepdim=True), 1e-06))), p_weight)
    NormLinear:  linear(<same>, p_proj_weight)
    TransformerBlock / ParallelBlock: <same> at every pre-norm site
    NativeRmsNorm: linear(rms_norm(x, p_norm_weight, dim=(16,), eps=1e-06), p_proj_weight)

* `mean` carries `dim` as a **tuple** `(-1,)` and `keepdim` as a real
  `bool`; `pow`'s exponent is a `Const(2)` operand (matching
  `pow_to_square`'s convention); `eps` is a `Const` leaf operand of
  `add`, not an attr.
* `rms_norm` is in `ATTR_SCHEMA` (`{1: "dim", 3: "eps"}`) and
  `ATTR_REQUIRED` (`dim`) — the RHS must name `dim`, and its value is
  the **normalized shape** `(16,)`, not the reduce axis `-1`.  The
  `sum`→`softmax` tuple→scalar mismatch `softmax_fold` hit is echoed
  one level up: axes→shape.
* The torch binding lowers `rms_norm` as
  `F.rms_norm(x, tuple(dim), weight=w, eps=float(eps))` — the derived
  attrs are the exact forms it reads.

## 2. Corpus measurement — laws in vs out

Forty real model graphs (`tools/law_impact._model_cases`), saturated
under `ALL_RULES` vs `ALL_RULES + {rms pair}` (6 iters, 60 k enode cap,
pipeline cost model):

    RMSNorm           enodes 23->25    cost 1.044e5 -> 1.741e4   −83.33%  fires {fold:1, nogain:1}
    NormLinear        enodes 34->36    cost 1.044e5 -> 3.482e4   −66.66%  fires {fold:1, nogain:1}
    TransformerBlock  enodes 1610->1664 cost 4.443e5 -> 3.051e5  −31.35%  fires {fold:2, nogain:2}
    ParallelBlock     enodes 1564->1675 cost 3.573e5 -> 2.876e5  −19.49%  fires {fold:1, nogain:1}
    all other 36 models: 0 fires, 0.00% change, identical enode counts

* **Exactly the four models the retro predicted** — every
  manual-RMSNorm corpus member folds; `ParallelBlock`'s shared `n`
  interns to ONE e-class, so one firing feeds all five projections.
* **0 regressions** — every other model's extracted term, cost, and
  enode count identical.
* **Certificates replay** on every model row (with and without the
  laws); `sink.verify` passes on lowered before/after modules for all
  four firing models (fp64, `rtol=1e-4` — see below: bitwise, in fact).
* **Closure-safe:** enode ratios min/mean/max = 1.000 / ~1.01 / 1.071
  — the 1.071 is ParallelBlock's new `rms_norm` members feeding
  downstream congruence, far under the 2.0 ship gate.  Single-
  direction and term-local, like `softmax_fold`.
* **Not derivable:** under `ALL_RULES` alone the manual term's
  extracted best stays `mul(x, mul(rsqrt(mean(square x)+eps), w))` —
  no `rms_norm` member is ever minted.

`law_bench` on the two registered synthetic cases (`sizes 64`, CUDA):

    rms_norm_fold         fired 1  rhs picked  verify pass
        cost 1.045e5 -> 1.743e4   ms 0.0413 -> 0.0153  (2.70x)
    rms_norm_fold_nogain  fired 1  rhs picked  verify pass
        cost 8.706e4 -> 1.742e4   ms 0.0367 -> 0.0150  (2.46x)

## 3. Decline battery — all eleven cases correct

| spelling | gained | nogain | why |
|---|---|---|---|
| `dim=(-1,)` tuple / `dim=-1` int / `dim=(1,)` | fires | fires | trailing-1 |
| `dim=(-2,-1)` on rank-3, `w` (d2,d3) | fires → `dim=(4,8)` | fires | multi-dim trailing block |
| `keepdim=False` | declines | declines | cond `attr-is` |
| `u` ≠ `v` (numerator vs `pow`) | declines | declines | shared metavar, no check |
| `dim=(0,)` non-trailing | declines | declines | check |
| `eps` a `Var`/`Param` | declines | declines | cond `const-num` |
| `pow` exponent 3 / 2.0 | declines / **fires** | declines / **fires** | `const-cmp` is numeric, not type-strict |
| `dim=(-1,-1)` dup, `(-3,)`/`(2,)` out-of-range | declines | declines | check |
| `w` wrong shape `(4,)` vs ns `(8,)` | declines | **fires** | gained veto; inner mul still folds |
| `u` unshaped `(None, 8)` | declines | declines | strict posture |

## 4. Tests

`tests/test_rms_norm_laws.py` (25 tests) mirrors
`tests/test_softmax_laws.py`/`test_glu_fold_law.py`:

* registration/tagging (`SIMPLIFICATION_RULES`, `DEFAULT`,
  `{SIMPLIFICATION}`);
* match/instantiate round-trips for both rules — LHS metavars exactly
  `{u, w, P, EPS, $attr:MD, $attr:MK}`; the RHS needs `derive` for
  `$attr:ND`/`$attr:EP`;
* `cond`/`check`/`derive` branch-by-branch (`_rms_normalized_shape`
  accepts tuple/int/multi-dim/full-rank trailing blocks; declines
  non-trailing, dup, out-of-range, empty, `None`, bool, non-int,
  unshaped);
* fp64 soundness — `torch.equal`, **bitwise**, not allclose: the
  kernel computes the same op sequence (unlike `torch.softmax`'s
  stabilizing max-subtraction);
* fires + RHS-is-member on tuple- and int-`dim` spellings, a
  multi-dim site, and the nogain-inside-gained double fire;
* all eleven decline cases above;
* end-to-end on all four real exports (fires, cost drops, cert
  replays, `sink.verify` fp64);
* the public `Optimizer(backend=TorchBackend())` default pipeline.

Plus: structure counts in `test_laws_structure` (19 simplification,
57 `all_rules()`, 134 with-layout, 193 module rewrites, 43 axioms),
the serialize census in `test_law_serialize` (the pair joins
`need_check` as `("check","derive")` — the cond is data, the
procedural remainder is named), `test_cond_laws`' migrated-cond count
(32), and the two `_c_rms_norm_fold*` bench cases.  The
`_LAW_SPECS` fuzzer deliberately cannot spec the pair — `EPS`/`P`
bind `Const` leaves, which the Var-binding spec machinery cannot
express (noted in the comment; they join the counted-not-fuzzed set).

## 5. Surprises / notes

* **Bitwise-equal fp64**, unlike `softmax_fold`'s 5.6e-17 — `rsqrt`,
  `mean`, `pow`, `mul`, `add` and the kernel compute the same op
  sequence, so the soundness test uses `torch.equal`.
* **`dim` means *shape* on `rms_norm`, axis on `mean`** — the attr is
  `normalized_shape` and must be *derived* (`u.shape[-k:]`), which is
  why `u` must be concretely shaped for the law to fire at all.  An
  unshaped graph simply declines (documented, not unsound).
* **The `w`-shape gate is a lowerability guard, not just soundness**:
  `aten.rms_norm` raises `RuntimeError` on a mismatched weight, so
  folding `x·rms⁻¹·w(wrong-shape)` would mint a member that can't
  lower.  The gained check demands `shape(w) == ns` exactly; the
  nogain twin still folds the inner `mul` there (verified).
* **`pow` only** — a graph spelling `x²` as `mul(x,x)` or `square(x)`
  does not match (no `mul→square` reverse bridge exists); under
  saturation the pow-class gains square/mul members but the reverse
  seeding doesn't happen.  Same class of documented miss as
  `softmax_fold`'s mul-orderings; a `mul_square` bridge is the
  follow-up if a corpus model ever spells it that way.
* **The mul nesting is the exporter's** — `mul(mul(x, rms), w)`.
  Alternate associations/orderings (`(x·w)·rms`, `w·(x·rms)`) reach
  the fold only via the opt-in SYMMETRY set, consistent with every
  other binary-op pattern in the library.
* **`eps` mismatches can't happen** — the bound `Const` leaf IS the
  minted attr; the decline-worthy cases are tensor `eps` bindings,
  which the cond rejects.
* **Provenance.**  The lead was recorded by the corpus census
  (`corpus-expansion-r2.md` §6.1) rather than emitted by the
  `_pattern_recognition` proposer — the admission followed the
  `softmax_fold` review path (oracle → derivability → corpus → cert →
  closure → declines → ship).

## 6. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass (145 files)
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/lint-imports` — 4 contracts kept
* `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 findings
* `.venv/bin/semgrep --config .semgrep.yml packages` — 0 findings
* `.venv/bin/python tools/radon_ratchet.py` — ok (2038 functions; the
  dims/ns helpers were split to stay rank-C)
* Scoped pytest — `test_rms_norm_laws` (25), `test_laws_structure`,
  `test_cond_laws`, `test_softmax_laws`, `test_glu_fold_law`,
  `test_silu_fold_laws`, `test_property_laws`, `test_law_serialize`,
  `test_contracts` (224+16skip — the pair skips the fuzzer spec by
  design), `test_rulesets`, `test_corpus_models`, `test_morphisms`,
  `test_morphism_fusednorm_tie`, `test_typing`, `test_cost_edge_cases`,
  `test_rules`, `test_meta`, `test_rulecache_codec`, `test_ports`,
  `test_pluggable_sink`, `test_select_laws`, `test_layout_laws`,
  `test_egraph_internals`, `test_e2e_pipeline`, `test_egraph_policy`,
  `test_perf_oracles`, `test_search_env`, `test_property_evaluation`,
  `test_matmul_factor_laws`, `test_trajectories`, plus the three
  rms-adjacent `test_torch_integration` tests — **all pass**
  (~900 tests).  Full suite not run on this box (the AGENTS.md
  parallel-warning); coverage on the new block is 100% under the
  scoped files.
