# Law rediscovery — can the library re-derive a law it dropped?

An operational, falsifiable test of the claim "the AI invents laws".
The test: take catopt's real law library
(`catopt_core.laws.ALL_RULES`, 51 rules), and for each law ask
**is it derivable from the OTHERS?**  If the machinery can decide
that — propose `lhs = rhs`, saturate, same e-class? — then law
verification is free and the only open question is *proposal*.

Reproduce: `.venv/bin/python tools/law_verifier.py`
(~1.3 s, CPU-only, no network; `--json PATH` for machine-readable
output).

## 1. The argument — verified

> A rewrite is a 2-cell; a law *about* rewrites is a 3-cell (ADR
> 0002).  Verifying `lhs = rhs` needs no new machinery: put both
> sides in a small e-graph, saturate under the KNOWN laws, and check
> whether they land in the same e-class.

**This held exactly.**  `tools/law_verifier.py::verify_law` is a
composition of shipped parts only:

* a fresh `EGraph()` (truncation level 2, proof tracking on);
* `add_term` both sides, `run(rules, root)`, then `find` on both;
* on a merge, `certificate` + `verify_certificate` — the existing
  proof machinery replays the derivation on *real terms*,
  independent of the e-graph.

No new semantics, no new data structures, nothing under `packages/`.
The decisive point in the argument — *law checks run on tiny terms,
so the saturation wall does not apply* — is confirmed empirically:
**every one of the 51 checks reached a fixed point in well under
10 ms** (the whole tool, 161 saturations, runs in ~1.3 s), the
largest graph being 77 e-nodes (`swiglu_fuse`).  The ~n=12 Catalan
wall is a whole-program phenomenon; a law's two sides are tiny.

### Witnesses are genuine derivations, not stubs

Equality is symmetric, so a merge can be witnessed from either
direction.  A rule present only as its *reverse* makes the forward
direction an `egraph_dependent` stub (the certificate search is
directional); `verify_law` therefore builds the certificate **both
ways and returns the one whose steps replay standalone**.  Every
derivable row below reports a `replayable=True` witness — a real
rule instance, not a trusted assertion.

## 2. Experiment 1 — the per-law rediscovery table

Verdict is "derivable" iff the other 50 rules already prove the law
(both sides land in one e-class); "primitive" iff they do not.  A
law's instance is taken from the `law_bench` `LAW_CASES` registry
(42 rules) or, for the 9 SDPA-fold spelling variants the registry
does not cover, built directly from the rule's own LHS pattern
(`generic_instance`).  **All 51 rules have a well-typed instance.**

### Derivable (23 / 51) — redundant given the others

| law | relation | witness rule | steps |
|---|---|---|---|
| `silu_mul_form` | **composite** | `silu_expand` | 1 |
| `distribute_matmul_over_add` | duplicate | `weight_distribute_matmul` | 1 |
| `factor_matmul` | duplicate | `weight_factor_matmul` | 1 |
| `weight_factor_matmul` | duplicate | `factor_matmul` | 1 |
| `weight_distribute_matmul` | duplicate | `distribute_matmul_over_add` | 1 |
| `pow_to_square` | inverse | `square_to_pow` | 1 |
| `square_to_pow` | inverse | `pow_to_square` | 1 |
| `right_distribute_matmul` | inverse | `right_factor_matmul` | 1 |
| `right_factor_matmul` | inverse | `right_distribute_matmul` | 1 |
| `weight_factor_linear` | inverse | `weight_distribute_linear` | 1 |
| `weight_distribute_linear` | inverse | `weight_factor_linear` | 1 |
| `assoc_linear` | inverse | `assoc_linear_rev` | 1 |
| `assoc_linear_rev` | inverse | `assoc_linear` | 1 |
| `assoc_linear_bias` | inverse | `assoc_linear_bias_rev` | 1 |
| `assoc_linear_bias_rev` | inverse | `assoc_linear_bias` | 1 |
| `naturality_scalar` | inverse | `naturality_scalar_rev` | 1 |
| `naturality_scalar_rev` | inverse | `naturality_scalar` | 1 |
| `assoc_matmul` | inverse | `assoc_matmul_rev` | 1 |
| `assoc_matmul_rev` | inverse | `assoc_matmul` | 1 |
| `linear_channel_scale` | inverse | `linear_channel_scale_rev` | 1 |
| `linear_channel_scale_rev` | inverse | `linear_channel_scale` | 1 |
| `linear_row_scale` | inverse | `linear_row_scale_rev` | 1 |
| `linear_row_scale_rev` | inverse | `linear_row_scale` | 1 |

### Primitive (28 / 51) — not derivable from the rest

`comm_add`, `comm_mul`, `assoc_add`, `assoc_mul`, `id_add`, `id_mul`,
`double_neg`, `sub_to_add`, `silu_expand`, `square_expand`,
`right_factor_linear`, `swiglu_fuse`, `parallel_mul_fuse`,
`qkv_fuse`, `qkv_fuse_asym`, `gqa_absorb_repeat`, and all 12
`sdpa_fold_*` rules.

### Reading the result honestly

23/51 sounds like "45 % of the library is emergent".  It is not.
The redundancy is almost entirely **by construction**:

* **18 laws derivable via an inverse rule** (nine inverse pairs:
  `pow_to_square`/`square_to_pow`, `right_distribute_matmul`/
  `right_factor_matmul`, `assoc_linear`/`assoc_linear_rev`,
  `assoc_matmul`/`assoc_matmul_rev`, `assoc_linear_bias`/`_rev`,
  `naturality_scalar`/`_rev`, `linear_channel_scale`/`_rev`,
  `linear_row_scale`/`_rev`, `weight_factor_linear`/
  `weight_distribute_linear`).  A rewrite and its reverse prove the
  same equality, so each derives the other under the "both sides in
  the graph" definition.  This is a *logical* redundancy, not an
  *operational* one: the library keeps both directions deliberately
  so a term appearing in *either* spelling can be rewritten — eqsat
  needs both to explore both bracketings from any starting form.
  These are not deletable.
* **4 laws that are exact duplicates** of another rule (see §4) —
  the only accidental redundancy found.  (These four are also
  mutually inverse, so the structural scan reports 13 inverse
  pair-lines; the 18 inverse-derived laws above are the nine clean
  pairs.)
* **Exactly one genuinely emergent law**: `silu_mul_form` is derived
  from `silu_expand` (a *different* law, applied inside a `mul`) —
  a 1-step composite.  It is the sole law that the library proves
  without containing a rule that is structurally its own inverse or
  copy.

So the library's core — commutativity, associativity, identity,
involution, subtraction, the SiLU/square decompositions, the product
fusions (`swiglu`/`qkv`), the SDPA fold, GQA absorption — is
**primitive axiom material**.  No law is derivable from the others
by genuine composition except `silu_mul_form`.  The "laws are
emergent" claim holds only in the weak sense that a *proposed*
equality can be mechanically *verified*; it does not hold in the
sense that the shipped axioms collapse into a smaller generating
set.

## 3. Experiment 2 — proposal (confirm new, reject false)

Both directions matter.  The battery (`run_proposals`) has three
parts, all 8/8 correct:

**Genuinely new equalities — CONFIRMED** (none is a single library
law; each needs a multi-step certificate):

| claim | certificate |
|---|---|
| `silu(neg(neg(x))) = mul(x, sigmoid(x))` | `double_neg`, `silu_expand` |
| `sub(x, neg(y)) = add(x, y)` | `double_neg`, `sub_to_add` |
| `add(matmul(W,a), matmul(W,b)) = matmul(W, add(b,a))` | `comm_add`, `factor_matmul` |

Each replays as a standalone derivation (`replayable=True`), so the
verifier does not merely *find* a merge — it *produces a proof*.

**False equalities — REJECTED** (the verifier is not a rubber stamp):

| claim | verdict |
|---|---|
| `add(x,y) = mul(x,y)` | not derivable |
| `matmul(A,B) = matmul(B,A)` | not derivable |
| `neg(x) = x` | not derivable |
| `sub(x,y) = sub(y,x)` | not derivable |

**A true-but-not-derivable conjecture — REJECTED**: `add(x,0) = x`
under `ALL_RULES \ {id_add}` is *semantically* true but the library
no longer proves it, so the verifier reports "not derivable" — the
correct answer for a *derivability* oracle.  Such an equality is
admissible only under the error budget, exactly as the framing says.

**No false equality was accepted** in any run.  A verifier that
accepts everything is worthless; this one did not.

## 4. Experiment 3 — soundness controls over all 51 laws

For every law with an instance (`run_controls`), two controls:

* **empty rule set** must *not* merge the two sides — guards against
  a free merge (interning bug, spurious union);
* **the law's own rule alone** must merge them — guards against
  rejecting a true equality.

Result: **51 laws checked, 0 failures.**  In particular, no law's
two sides ever merge with zero rules, so the merges in §2 are all
rule-driven.

### Structural side-finding: exact duplicate rules

Independent of the verifier (`structural_report`, metavariable
renaming normalised away):

```
DUPLICATE: ['distribute_matmul_over_add', 'weight_distribute_matmul']
DUPLICATE: ['factor_matmul',             'weight_factor_matmul']
```

`weight_distribute_matmul` is the *same rewrite* as
`distribute_matmul_over_add` — both are
`matmul(X, add(Y, Z)) -> add(matmul(X,Y), matmul(X,Z))` — and
`weight_factor_matmul` is the same as `factor_matmul`.  Same
`_CAT` tags, no `check`, no `derive`, both in
`CATEGORICAL_RULES`.  This is genuine (if harmless) dead weight: the
second copy is a no-op on every term the first already rewrites.
Actionable cleanup, not a correctness bug.

## Verdict

* **The verifier works and is falsifiable.**  It is a ~pure
  composition of `EGraph` / `find` / `certificate` /
  `verify_certificate`; it decides e-class membership on tiny terms
  and emits replayable proofs.  It confirms novel composites and
  rejects false equalities and non-derivable conjectures.
* **"Laws are emergent" is only weakly true.**  Of 51 laws, 28 are
  primitive (not derivable from the rest) and the 23 "derivable" ones
  are 22 construction artifacts (18 inverse-pair + 4 duplicate) plus
  exactly one genuine composite (`silu_mul_form`).  The library's
  axioms do not collapse.
* **What exists for "the AI invents laws" is the *verification*
  half, and it is complete.**  Proposal — the search for candidate
  equalities — is *not* provided here; it is the remaining open
  problem.  Propose-then-verify is now mechanically closed on the
  verify side.
* **No critical bug found**: no false equality accepted, no free
  merge, no rejected true law.
* Minor finding: two exact duplicate rules (`weight_*` copy
  `distribute_matmul_over_add` / `factor_matmul`).

## Caveats / limitations (stated, not tuned)

* **Definition of "derivable".**  The framing prescribes putting
  *both* sides in the graph.  Under that definition a law with its
  inverse in the library is always "derivable" (see §2).  Read the
  18 inverse rows as *logical* redundancy; they are *operationally*
  load-bearing.
* **Leaf identity.**  The e-graph keys leaves by `repr` — a `Var`'s
  repr is its name, so two Vars with the same name but different
  shapes are one leaf.  Instances must use distinct names per
  distinct value; the tool does.  A caller reusing a name would
  create a spurious merge.
* **Instance coverage.**  42/51 instances come from `law_bench`'s
  `LAW_CASES`; the 9 SDPA variants come from the pattern-driven
  `generic_instance`.  If a law's `check` vetoed every buildable
  instance it would be reported `no-instance` — none were.
* **Budget.**  `max_nodes = 200_000`, `max_iterations = 30`; no case
  approached either (all `stop == "fixed_point"`).  A `stop` of
  `max_nodes` would be reported, not silently read as "primitive".

## Gates

Run from the main worktree, HEAD plus this tool only (no `packages/`
change):

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors; `tools/` is out of scope)
* `.venv/bin/vulture` — exit 0
* `.venv/bin/python tools/radon_ratchet.py` — pass (1873 functions)
* `uv run pytest -q` — 3339 passed, 31 skipped
* coverage unchanged (no `packages/` file touched)
