# The matmul factor/distribute laws were unsound on mixed-rank bindings — fixed

`law-gap-targeted-gen.md` §3 reported the finding and deferred the fix
(a `tools/`-scoped task): the shipped weight-merge law rewrites a real,
evaluable term into a numerically different one.  This retro records
the reproduction in the IR, the exact false-region, the guards shipped
in `packages/catopt-core/src/catopt_core/laws/tensor.py`, and the
rest-of-library audit the fix required.

## 1. The counterexample, reproduced

`tools/law_gap_targeted_gen.py` drew `x=(16,)`, `a=(16,)`, `b=(16,16)`
for `x@a + x@b → x@(a+b)` on attempt 67 of its adversarial search.
Reproduced as an IR term evaluated through the torch bindings
(`catopt_torch.meta_eval._eval_term`, fp64 randn):

    lhs = add(matmul(x, a), matmul(x, b))   # (16,)  — () broadcast over (16,)
    rhs = matmul(x, add(a, b))              # (16,)  — x @ ((16,)+(16,16) → (16,16))
    → allclose=False, max diff 12.83

Both sides are well-typed and evaluate — same shape `(16,)`, wrong
values.  The mechanism: `torch.matmul` contracts a rank-≥2 right
operand's axis `-2` but a rank-1 operand's *only* axis, while `add`
broadcast-aligns *trailing* axes.  `a+b` right-aligns the vector onto
`b`'s **last** axis, so `x@(a+b)` contracts `a_j·Σ_i x_i` per output
column `j` — the dot product `x·a` is never formed.  Meanwhile the LHS
computes `x·a` (a scalar) and broadcasts it across the matvec `x·b`.
The two "distributions" sum different pairings entirely.

All four unconditional members of the coherence catalogue's
{`distribute_matmul_over_add`, `factor_matmul`,
`weight_factor_matmul`, `weight_distribute_matmul`} equivalence class
shared the hole: they differ only in which side of the `matmul`/`add`
nesting the pattern spells, and the false region lives in the two
summed right-operand addends, not in `x`/`W`.

## 2. The exact false-region — enumerated, not guessed

~120 binding shapes were evaluated through the torch oracle, varying
the shared operand's rank (matrix `(m,k)`, vector `(k,)`) and the two
addends over `()`, `(k,)`, `(k,1)`, `(k,n)`, `(1,k)`, `(p,k,n)`,
`(1,k,n)`, `(n,n)`:

| addend ranks | verdict |
|---|---|
| equal, ≥1 | **sound** — always `True` |
| both ≥ 2, different (e.g. `(k,n)` + `(p,k,n)`) | **sound** — broadcast only replicates leading/batch axes; axis `-2` stays paired |
| exactly one rank-1, other ≥ 2 (e.g. `(k,)` + `(k,n)`, `(k,)` + `(k,1)`, `(k,)` + `(p,k,1)`) | **FALSE on every evaluable binding** — the vector's contraction axis lands on the partner's output axis |
| any rank-0 (scalar) addend | moot — `matmul` rejects a scalar operand; the LHS cannot arise in a real program |
| non-broadcastable addends | moot — the `add` itself doesn't eval |

Two boundary notes.  The false cases are exactly the ones where *both*
sides evaluate: `a=(k,) b=(k,n)` is broadcastable only when `k==n` (or
a degenerate `1`), so the hole is narrower than "any rank mix" — but
the retro's witness (`16 = 16`) is a perfectly ordinary square shape,
and `(k,)` + `(k,1)` needs no coincidence at all.  And the check is
therefore *rank*-equality-aware, not shape-equality: vetoing
`a=(k,n) b=(p,k,n)` would wrongly decline a sound batch-broadcast
merge.

The **left-operand** pair (`right_distribute_matmul`,
`right_factor_matmul` — `matmul(add(a,b), W)`) was enumerated the same
way across 24 cases including vector×matrix mixes: **all `True`**.
There the contraction axis is the *last* axis for every rank ≥ 1 —
which is exactly the axis broadcast aligns — so distributivity holds on
every evaluable binding.  Left unguarded deliberately.

## 3. The guards shipped

In `catopt_core/laws/tensor.py`:

* `_mm_rhs_addends_aligned(sa, sb)` — the core predicate: `len` equal,
  or `min(ra, rb) >= 2`; rank-0 vetoes.
* `_check_mm_rhs_addends` (keys `a`/`b`) on
  `distribute_matmul_over_add` and `factor_matmul`.
* `_check_mm_rhs_weights` (keys `W`/`W2`) on `weight_factor_matmul`,
  `weight_distribute_matmul`, **and** `weight_factor_linear` /
  `weight_distribute_linear`.

Both hooks are *strict*: a non-`tuple` `_shape_of` result (`None` —
unshaped or carrier-internal member; `_INVALID` — already poisoned)
declines.  That is the same posture as layout's
`_check_commute_binary`, and it is the sound one for a soundness
guard: a member whose shape the check cannot read cannot *prove* the
contraction axes align.  `None` *dims inside* a known-rank tuple still
pass — the veto is on rank structure, which `len()` reads fine.

The `linear` pair is mathematically safe even rank-mixed — a weight's
contraction axis is its *last* (`(o, in)` contracts `in`), the
broadcast axis — but a rank-1 weight mints a member `F.linear` cannot
lower, so the same guard is applied: it vetoes only bindings that
couldn't run anyway.

## 4. The audit found a second instance of the bug class

`swiglu_fuse` / `parallel_mul_fuse` mint
`chunk(linear(x, concat(A,B,0)), chunks=2, dim=-1)` — halving the fused
output assumes the paired weights have **equal output dims**.  With
`A=(4,i), B=(1,i)` the LHS `mul(silu(linear(x,A)), linear(x,B))`
evaluates fine (broadcastable `(m,4)*(m,1)`), but the RHS's `mul` of
the two uneven chunks fails to evaluate at all — an unevaluable member
asserted equal to a real term.  `_check_fuse_pair` now requires
provably-equal weight shapes (same rank; per-dim equal or `None`
wildcard — veto only provable mismatches).  `qkv_fuse` is *not*
affected: its three head views share the single `S` reshape attr
metavariable, which structurally forces equal output dims.

Rest of the audit, per family:

| law / family | verdict |
|---|---|
| `distribute_matmul_over_add`, `factor_matmul`, `weight_factor_matmul`, `weight_distribute_matmul` | **was unsound → now guarded** |
| `weight_factor_linear`, `weight_distribute_linear` | sound mathematically; **guarded** to bar unlowering members |
| `right_distribute_matmul`, `right_factor_matmul`, `right_factor_linear` | sound on every evaluable binding (contraction axis = broadcast axis); unguarded |
| `swiglu_fuse`, `parallel_mul_fuse` | **was unsound (unevaluable RHS) → now guarded** by `_check_fuse_pair` |
| `qkv_fuse`, `qkv_fuse_asym` | sound — structurally (shared `S`) / derived split sizes |
| `assoc_matmul`(+_rev) | sound — enumerated mixed-rank chains all equal; associativity is pure contraction re-ordering |
| `assoc_linear`(+_rev, +_bias family) | sound — same reassociation; `assoc_linear_bias*` already carries `_check_linear_bias_compose` |
| `naturality_scalar`(+_rev), `naturality_scalar_left`(+_rev), `linear_{channel,row}_scale*`, `linear_out_scale*`, rope scale laws | already guarded (scalar/row/channel/uniform side conditions) |
| `softmax_fold`, `sdpa_fold_*`, `gqa_absorb`, `select_mul` | already guarded |
| `comm_add`/`assoc_add`/`sub_to_add`/elementwise family | sound — broadcast is preserved pointwise by reassociation |
| scan `aff_*`/`affd_*` lifts | sound — `apply(aff(A,x),h)` *evaluates* `matmul(A,h)+x`; the lift changes the carrier, not the arithmetic |
| `factored`/`specials`/`headshare`/`pairing` passes | verify-gated machinery, not equational laws — members are offered under certified bounds |
| `orchestrator.morphisms.reify._distribute_over` | the same lemma, but a *constructed* equality offered under the pair's fp64 `verify` — guarded by verify at use; left as-is |
| `div`/`sub` factor proposals (`factor_left`, `div_add`, …) | tools/-level candidates only, never shipped |

## 5. What the cert machinery does and does not protect

Honest boundary, per the earlier retro's mitigation note:

* `sink.verify` at lowering is the **last** line: an extracted wrong
  member fails the lowered-module comparison and the optimization
  declines — so the bug could not silently ship a wrong model.  But it
  protects the *output*, not the *search*: the unsound member sat in
  the e-class as an equal citizen — it could win extraction (one
  matmul is cheaper than matvec+matmul+add), contaminate certificates
  that replay its edge, and poison any `verify_law`-style derivability
  argument made over the class.
* `verify_certificate` replays the *derivation*, not the *truth*: a
  cert over an unsound rule replays faithfully and is meaningless.
  The check-hook veto is the only layer that keeps the false member
  out of the e-graph in the first place — which is why the fix belongs
  in the law, not in verify.
* The corpus could never surface this: no real exported model carries
  a mixed-rank `x@a + x@B` — which is exactly the blind spot
  gap-targeted generation exists to probe.  The shape is
  real-world-*plausible* (a LoRA-flavoured `x@v` low-rank correction
  summed with `x@W`), just corpus-absent.

## 6. Regression evidence

`tests/test_matmul_factor_laws.py` (new, 15 tests): the counterexample
evaluates `allclose=False` under the torch oracle; each guarded rule
**declines** the mixed-rank binding (0 fires); equal-rank and
both-rank-≥2 mixed bindings still fire with the RHS member in the root
e-class; the 4-member equivalence class still holds under the guard
(each rule fires alone on its instance — note `factor_matmul` and
`weight_factor_matmul` mint the *identical* merged member, so a
joint-run second match is an honestly-uncounted no-op merge); the
certificate for the legal `weight_factor_matmul` merge still replays;
the fuse-pair guard declines `A=(4,4) B=(1,4)` and fires on equal
weights; the unguarded left pair is pinned sound (mixed-rank numeric
equality + firing).

`tools/law_coherence.py` re-run: the
{`distribute_matmul_over_add`, `factor_matmul`,
`weight_distribute_matmul`, `weight_factor_matmul`} class is intact
(duplicate+inverse edges preserved), 0 divergent critical pairs, all
54 rules still instanced — the bench `LAW_CASES` shapes are equal-rank
and unchanged.

## Gates

* `.venv/bin/ruff check packages tools` + `format --check` — pass
* `.venv/bin/ty check` — pass
* `.venv/bin/vulture` — pass · `.venv/bin/lint-imports` — 4/4 kept
* `.venv/bin/bandit -c .bandit.yaml -r packages` — 0 issues ·
  `semgrep --config .semgrep.yml packages` — 0 findings ·
  `radon_ratchet.py` — ok
* `.venv/bin/python tools/law_coherence.py` — class intact (§6)
* pytest (scoped, serial): `test_matmul_factor_laws` (15),
  `test_laws_structure`, `test_certificates`, `test_contracts`,
  `test_meta`, `test_silu_fold_laws`, `test_softmax_laws`,
  `test_laws_rewrite_edges`, `test_property_laws`, `test_moe_pairing`,
  `test_rulesets`, `test_select_laws`, `test_layout_laws`,
  `test_rules`, `test_backend_neutral`, `test_weight_specials`,
  `test_egraph_policy` — 528 passed; plus the
  `test_torch_integration` `-k swiglu/weight_factor/matmul` selection
  — 7 passed.  Per the task bound the full suite + 100% coverage gate
  was not run (25 min serial on this host).

Not committed.
