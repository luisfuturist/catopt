# `handl` → `catopt`: what is reusable, what is analogous, what is a warning

Read-only survey of `/home/luis/GitHub/handl-lang/platform` (Rust, ~98 crates,
~133 ADRs under `project/adrs/`, a ~24-file Lean4 library under
`research/arrow-calculus-kernel-poc/lean/ArrowMinimalityLib/`).  Scope was the
seven questions in the brief.  Nothing in the handl repo was modified; this is
the only file written.

Method: read the ADRs named in the brief plus their dependencies, the Rust
kernel and transform engine, the platform/plugin crates, the research writeups
and Lean headers, and grepped the whole tree for e-graph / equality-saturation /
higher-category machinery.  Where I could not find something I say so.

---

## 0. The short version

`handl` is the closest thing in the wild to catopt's thesis.  Its own framing —
"a `Language` is a *presented 2-category*: sorts = 0-cells, ops = 1-cells,
laws/rewrites = 2-cells, handlers = equations" (`ADR-0100` §2) — is
catopt's ADR-0002 / plan 0014 thesis in different words ("the law library is the
generating 2-cells; the free higher category they generate *is* the reachable
set").  `ADR-0119`/`ADR-0120` even adopt catopt's `a ~[T]~> b` cell notation.

But the overlap is **design vocabulary, not implementation**.  `handl` is a
*term-rewriting* platform with a mechanized proof theory for a toy kernel.  It
has **no e-graph, no equality saturation, no cost-model-driven extraction, and
no derivational verification of real rewrites.**  Its certificates are runtime
value-equality checks; its proofs are about `arr/comp/first/left/loop` over an
abstract signature, not about a law library.  The reusable parts are the small,
sharp mechanisms below — not the grand unification.

---

## 1. Directly reusable

### 1.1 A CutElim normalizer as a strict fast-path for 2-cell equality

**Where:** `crates/handl-language/src/law_inference.rs` — `rewrite_root_td`
(lines 99–135), `rewrite_pass_td` (193–418), `normalize` (425–435),
`law_equal` (444–446).  Mechanized in
`lean/ArrowMinimalityLib/ProofAdvantage.lean` (`CutElim` / `CutElimStar` /
`cut_elim_preserves_proof`) and `CutElimCongruence.lean` (`CutElimCong` +
congruence + strong normalization).

**Mechanism.**  A tiny, fixed set of size-non-increasing local graph rewrites
(`Seq(id,G)⇒G`, `Seq(G,id)⇒G`, `Seq(First G, First H)⇒First(Seq G H)`,
`First(id)⇒id`, plus coproduct duals) is applied bottom-up to a fixpoint; two
terms are *law-equal* iff they share a normal form.  This decides a fragment of
2-cell equality **without replaying a derivation** and **without an e-graph**.

**Why catopt can use it.**  catopt already has the right substrate — `IR`/`Op`
is binder-free and structural, so normal-form comparison is decidable and cheap.
Add a normalizer next to `verify_certificate` (see Recommendations 1): use it to
(a) short-circuit certificate verification when
`normalize(lhs) == normalize(rhs)`, and (b) fast-reject rules whose two sides are
already normal-form-equal.  `CutElimCongruence.lean` supplies the missing piece
— congruence (rewriting inside subterms) plus a *complete* strong-normalization
proof for the six-rewrite system — which is exactly the analogue of catopt's
congruence closure.

**The honest boundary (reusable, and load-bearing):** `law_inference.rs`
lines 26–37 and `EquationalLawInference.lean` (the `SemanticLaw` structure,
lines 110–122) state that only **CutElim-proven structural laws** (functor
identity/fusion) are inferable from wiring.  Roundtrip, associativity,
sortedness are *semantic facts about base arrows* and need an explicit axiom
(`SemanticLaw.sound` is a hypothesis, not proven).  This boundary is the same
one catopt's frontier runs into: a normalizer solves the *structural* fragment;
the rest is the word problem.

### 1.2 Witness-carrying rewrites with a named-law registry

**Where:** `crates/plugin-engine/src/term.rs` — `Term::Rewrite { source,
target, witness }` (149–157), `RewriteWitness::{Named, Compute, HandlerHint,
Cast}` (241–252).  Lookup in
`crates/plugin-engine-exec-tree-walk/src/lib.rs::rewrite` (3854–3907):
`Named(name)` resolves through `ctx.lang.law(name)` / `ctx.lang.rewrite_rule(name)`.

**Mechanism.**  Every 2-cell carries *why* it is sound as a small enum; the
`Named` case references a law **by name** in the `Language` registry, so the
certificate does not have to embed the rule object.

**Why catopt can use it.**  catopt's `CertStep` already records
`rule`/`lhs`/`rhs`/`bindings`/`egraph_dependent` and `Certificate.rules` carries
the `Rewrite` objects so the cert is self-contained
(`packages/catopt-core/src/catopt_core/egraph/certs.py`).  handl's split is the
complementary design: cite by name, resolve against a registry.  Adopting a
named-law registry lets an *external* verifier re-check a certificate without
the producer's rule objects — useful once certificates cross a process/run
boundary (plan 0014 stage 2/3).

### 1.3 Separate the rule *source* from the rule *engine*

**Where:** `ADR-0094` §5 — `trait RuleSource { fn rules_for(&self, lang, pass)
-> Vec<TransformDef> }`, default `LanguageRuleSource` reads `lang.transforms`,
an alternate impl loads ad-hoc codemod rules **without merging them into the
`Language`**.

**Why catopt can use it.**  This is precisely catopt's "the language ships its
own laws" vs "a user runs an ad-hoc rule set" distinction.  catopt's
`ports.py` already names `RuleSetLike` / `RuleSetProvider`; add a `RuleSource`
port and let `EGraph.run` / `search` take one.  It cleanly decouples
`catopt_core.laws.ruleset` (the shipped presets) from a caller-supplied set.

### 1.4 Rule identity, priority, and `replaces`

**Where:** `ADR-0094` §2 — `TransformDef { id, pass, node_kind, priority, when,
rule, replaces, requires }`; identity key `pass + ":" + node_kind`; a first-class
rule may `replaces:` a lower-priority one; collisions without `replaces` are
errors.

**Why catopt can use it.**  catopt's `RuleSet` composition (plan 0009) has no
explicit collision/override discipline.  handl's `replaces` + `priority` +
error-on-unresolved-collision is a small, tested policy that would let rule sets
compose (e.g. a user set overriding a shipped law) without silent duplication.

### 1.5 `TransformMode::{Pass, Pipeline, ApplyAt, Plan}` — a dry-run mode

**Where:** `ADR-0094` §1 (`TransformRequest`/`TransformMode`), plus the
`TransformEngine::find_matches` match-only method.  `Plan` returns matched sites
+ proposed rewrites with no new arena; `ApplyAt` is a surgical rewrite.

**Why catopt can use it.**  `find_matches` is the analogue of e-matching used by
a lint/plan view, and `Plan` is a *dry-run* search that reports candidate
rewrites without committing — a useful shape for catopt's search/policy
instrumentation (`catopt_core.policies`, `catopt_core.trajectories`) and for a
"why did you not rewrite this?" diagnostic.  `ApplyAt` mirrors a targeted
single-site rewrite, which catopt's `_edge_path`/`_connect` already do
internally.

### 1.6 The "Term vs Op" decision criterion

**Where:** `ADR-0098` §14: *"can a tool transform, analyze, or reason about it
without knowing the target runtime?  If yes, it belongs in `Term`; if its
meaning is determined by a runtime-specific handler, it belongs in `Op`."*

**Why catopt can use it.**  This is a crisp, citable rule for what belongs in
`catopt_core.ir` (backend-neutral) vs behind a backend port.  It directly
supports catopt's import-linter hexagonal contract and the ADR-0003 independence
rule.  Recommend recording it verbatim in `ports.py` / ADR-0003.

### 1.7 Globular boundary law for composing 2-cells

**Where:** `ADR-0120` Phase 2 / `ADR-0119` §3, §5 — `Term::Seq { axis, fst, snd }`
(`crates/plugin-engine/src/term.rs` 72–90), typing rule
`tgt_k(a) = src_k(b)` for `a >>>[k] b`, enforced statically (`synth_axis_flow`
via `cell_boundary_`/`cell_paste_`) and dynamically (`cell2_comp_axis`).

**Why catopt can use it.**  catopt's frontier is "laws are 2-cells, coherence is
3-cells".  handl's contribution here is a concrete, cheap *typing rule* for
composing rewrites along a chosen axis, plus the observation that horizontal and
vertical 2-cell composition are the same `Seq` with different `axis` — no new
vocabulary.  This is the analogue of what catopt's `_close_congruence`
(`egraph/core.py`) does for coherence, stated as a boundary law.  Recommend
adopting `tgt_k = src_k` as the documented composition precondition in
`egraph/proof.py` / plan 0014 §Mechanism.

### 1.8 Deferred-contribution registration

**Where:** `crates/platform/src/plugin_model.rs` — `ContributionContext::contribute`
(92–116) and `PendingRule`/`PendingStatus` in `plugin.rs` (166–181, 293–328):
a contribution whose `RuleAcceptor` is not yet registered is queued and retried
as loading progresses.

**Why catopt can use it.**  catopt's `register_carriers` seam
(`catopt_orchestrator.carriers`) is the same problem — an extension registers
machinery that a later-loaded pass consumes.  handl's retry-on-missing-acceptor
generalizes catopt's import-time registration and removes load-order fragility.

---

## 2. Conceptually matching (same idea, different vocabulary)

| handl | catopt | note |
|---|---|---|
| Arrow calculus kernel: `arr`/`>>>`/`first`/`choice`/`loop`/`app` (`ADR-0052`; `Term` enum in `plugin-engine/src/term.rs`) | free monoidal category of ops/wires (`IR` + laws) | `Term` *is* a term language over the arrow calculus; catopt's `IR` is the same object with tensor ops as generators |
| `Language` = presented 2-category: sorts/ops/laws/rewrites/handlers = 0/1/2-cells/equations (`ADR-0100` §2) | ADR-0002 thesis; plan 0014 §Mechanism ("the free higher category they generate *is* the reachable set") | near-exact match; handl states it explicitly |
| `Model` trait with `carrier`/`run`/`rewrite` → `RewriteResult::{Equal,Replace,Proved,Compiled,Unsupported}` (`ADR-0100` §3) | four independent dimensions (ADR-0003) + `Ports`/`Backend` | see §3.1 — same "one description, many interpretations" story, but handl *unifies* where catopt *separates* |
| "Descriptions, not execution": `Term` describes, `Model` interprets, exec engines are handlers (`ADR-0052`, `ADR-0098` §3/§14) | semantics vs execution dimensions; `Source`/`Sink`/`Binding` ports | mechanism: `ModelCtx{ast,lang,current_node}` + `Op` dispatch order (handler stack → value grades → primitives), `ADR-0098` §16.1 |
| `Capability`/`Plugin`/`PluginRegistry`/`Kit`/`DefaultHost` (`crates/platform`) | `Ports` (runtime-checkable `Protocol`s) + import-linter contracts | handl = *runtime* capability registry with dep resolution; catopt = *static* structural contracts.  handl's `CapabilityRequest`/`Response` (JSON) is a uniform ABI; catopt's Protocols are more type-safe |
| "Language as data": dispatch on `lang.ast[kind]`/`eval`/`types`/`transforms`, never `kind ==` (`ADR-0044`, `ADR-0081`) | "laws target signature classes, not op trees" | handl's `AstShapeDef` roles ↔ catopt's `BlockSig`/signature (plan 0011 morphism engine); the engine-genericity rule is catopt's "a cost model must not know op trees" |
| `RewriteTrace` = free provenance because `AstRef` arena indices are stable (`ADR-0094` §1) | `ProofEdge`/`CertStep` path+rule provenance; `_birth`/`_oldest_term` ordering (`egraph/proof.py`) | same insight: provenance is an append, not a second structure |
| `RuleSource`, `find_matches`, `NativeRewrite` escape hatch (`ADR-0094`) | `RuleSetProvider`, e-matching, `catopt_native` engine seam | handl's `Native(name)` is a *data-dispatched* escape hatch (dispatch stays in `TransformDef`), not a `kind ==` branch — same discipline as catopt's registry seam |

---

## 3. Cautionary tales

### 3.1 Do NOT adopt `Model::rewrite` — it violates dimension independence

`ADR-0100` §3 makes the *model* decide a rewrite's meaning: `ValueModel.rewrite`
returns `Equal` (both sides produce the same value), `OptimizerModel` returns
`Replace`, `TypeModel` returns `Equal` on types.  The equality/semantics
question is answered by whichever model is running.  This is exactly the
anti-pattern catopt's **ADR-0003** forbids ("a cost model must not change
semantics, a policy must not decide equivalence").  handl's unification is
elegant for the "one description, many interpretations" slogan, but it lets the
evaluation/execution layer answer the semantics question.  catopt's four
*independent* dimensions are the better factoring; keep them.

### 3.2 The "one `Obj` world" is a research bet, not a pattern

`ADR-0106` collapses types, values, functions, proofs, grades and effects into a
single `Obj` universe (`Type : Obj := sym()`; "no special type universe, no
special proof language").  `ADR-0043`'s own Negative section concedes "Nobody
has shipped this in a production language."  This is maximal unification;
catopt's independence and its small fixed verifier are the opposite trade and
should stay.

### 3.3 handl's "proof objects" are largely definitional, and its runtime witness is a test, not a proof

- `PROOF_ADVANTAGE.md` §1 is explicit: the Curry–Howard `TyOf ↔ PrOf` bijection
  is "notation-for-notation identical … a restatement of the definitions."  The
  *non-definitional* result is cut-elimination-as-local-rewrites
  (`ProofAdvantage.lean`), and it is real — but it is a theorem about a toy
  calculus (`arr/comp/first/left/loop` over an abstract `Sig`), **not** about any
  concrete law library or optimizer.
- At runtime, `Term::Rewrite` is "verified" by *running both sides on the
  incoming value and comparing* (`plugin-engine-exec-tree-walk/src/lib.rs`
  3898–3906: `if a == b { Equal } else if is_rewrite { Rewritten }`).  That is a
  single-input testing oracle, **not** derivational verification.  `RewriteTrace`
  records `from/to/rule_id` and nothing re-checks it.

Catopt's differentiator is `verify_certificate`, which **replays the derivation
on real terms independent of the e-graph**.  handl has no analogue — do not read
"handl has proofs" as "handl certifies rewrites."

### 3.4 "Law inference" is much weaker than it sounds

`law_inference.rs` infers **only** the CutElim structural fragment (functor
identity/fusion).  Roundtrip/associativity/sortedness are explicitly out of
scope.  If catopt is tempted by "infer laws from structure," handl is the
cautionary data point: the inferable set is exactly the set of laws that *are*
the normalizer's rewrites.

### 3.5 The higher-category machinery is Proposed, partial, and open

`ADR-0119` is **Proposed**; `ADR-0100` is **Proposed** and truncates at 2-cells
("Higher cells are not represented directly"); `ADR-0120` records `k ≥ 2`
rejected and "`Cell2` is the only cell shape the value model carries."  `PAPER.md`
§13 lists higher associativity/unitor structure and interchange/coherence laws
as **open**.  handl is ahead of catopt in *vocabulary* for n-cells, not in
*implementation*.  Do not treat it as a reference solution for the 3-cell /
word-problem frontier.

### 3.6 Overloading a slot: `dim` as both dimension and effect-grade

`ADR-0120` §Context names the smell itself: "The `dim` field doubles as an
effect-grade string for `handle` routing and a morphism-dimension index — two
concepts in one slot."  (Resolved by merging into one *named-dimension*
concept.)  The lesson for catopt: keep `Op.attrs`, cost, and shape concerns
separate; the same overload is easy to introduce in `ir.py`.

### 3.7 Totality by opacity

`ADR-0119` §7 folds every derived/proof/path form into `Term::Opaque { data:
Value }` so `Term`↔`TermData` conversion is total.  Convenient, but the kernel
stops interpreting those records and their semantics live in tag-directed
consumers.  catopt's `Op` is opaque by *name* but its laws match on *signature*;
keep it that way rather than letting the IR become an untyped JSON blob.

### 3.8 Big-bang breaking migrations

`ADR-0052` ("big-bang cutover complete") and `ADR-0119` (breaking serial-format
changes to `Term::Arr`/`Op`) show handl repeatedly paying for breaking rewrites.
catopt's AGENTS.md discipline — "Ratchets, not rewrites" — is the better default.

---

## 4. What handl does NOT have that catopt needs

I searched the whole tree (`egraph`, `egglog`, `equality saturation`, `e-node`,
`enode`, case-insensitive).  The only hits are false positives (`single-node`,
`ProbeNode`, `referenceNode`, "compute-graph") plus a *related-work* citation of
CCLEMMA in `research/arrow-calculus-kernel-poc/PAPER_INVARIANT_INFERENCE.md`
(line 1258).  So:

1. **No e-graph / equality saturation.**  handl rewrites terms to a fixpoint
   (`RewriteEngine` bottom-up walk + `Traversal::Fixpoint{max}`, `ADR-0094` §4;
   `law_inference::normalize`).  It applies rules in a traversal order and
   cannot search the equivalence class the way `EGraph.run`/`extract_best` does.
   This is the single biggest gap: handl's reach is the *closure under one
   traversal*, not the congruence-closed equivalence class.
2. **No cost-model-driven extraction.**  No analogue of
   `extract_best`/`extract_alternatives`/`backend_cost`/`PerformanceModel`/
   Pareto/beam.  `ADR-0100` §3 says the `OptimizerModel` "rewrite[s] `Term` in
   place" — there is no ranking of alternatives.
3. **No derivational verification of real rewrites.**  No `verify_certificate`.
   Runtime witness = value-equality on one input (§3.3).  The Lean proofs are
   about the calculus, not a law library.
4. **No pluggable/learned search Policy.**  Traversal strategy is a fixed
   `Traversal::{BottomUp,TopDown,Fixpoint}` choice (`ADR-0094` §4), not a policy
   seam.  No `Policy` / learned policy analogue.
5. **No 3-cell coherence implementation.**  `CellComp`/`Whisker`/`CellId` are
   folded away (`ADR-0119` §4, `ADR-0120` Phase 5) and `k ≥ 2` is rejected.
   catopt's `_close_congruence` has no counterpart.
6. **No bounded-error rewrites.**  No `error_bound`/approximate-rewrite notion;
   catopt's plan 0012 / `Certificate.error_bound` is absent.
7. **No separate evaluation dimension.**  handl's "cost" is not a dimension; its
   profiler analogue (if any) is not separated from execution.

---

## 5. Recommendations (each tied to a concrete catopt artifact)

1. **Add a CutElim-style normalizer as the strict 2-cell fast-path** alongside
   `verify_certificate`.  New module (e.g. `catopt_core/laws/normalize.py`, or a
   `normalize`/`law_equal` pair in `egraph/proof.py`): canonicalize both sides of
   a law over `catopt_core.ir.Op` and compare.  Use it to short-circuit
   `verify_certificate` and to fast-reject normal-form-equal rules.  Scope it to
   the structural fragment; semantic laws (roundtrip, associativity) stay
   axiom-gated.  Cite handl `law_inference.rs` + `ProofAdvantage.lean`.
2. **Add a named-law registry so `CertStep` can cite a law by name**
   (`RewriteWitness::Named` + `lang.law(name)`).  Touches
   `catopt_core/egraph/certs.py` and `catopt_core/laws/ruleset.py`; makes
   certificates externally re-checkable (plan 0014 stage 2/3).
3. **Add a `RuleSource` port** next to `RuleSetLike`/`RuleSetProvider` in
   `ports.py`, and let `search`/`EGraph.run` take one.  Decouples shipped
   `RuleSet` presets from caller-supplied ad-hoc rule sets (`ADR-0094` §5).
4. **Adopt `replaces` + `priority` + collision-error** as the `RuleSet`
   composition discipline (plan 0009).  Prevents silent duplication when rule
   sets are merged.
5. **Adopt the Term-vs-Op decision criterion verbatim** ("can a tool
   transform/analyze/reason about it without knowing the target runtime?") in
   `ports.py` and ADR-0003.
6. **Adopt the globular boundary law `tgt_k(a) = src_k(b)`** as the documented
   precondition for composing 2-cells along an axis; wire it into
   `egraph/proof.py` (`_close_congruence`) and plan 0014 §Mechanism.  Gives
   catopt a cheap, precise statement of horizontal/vertical rewrite composition
   with no new vocabulary.
7. **Generalize the `register_carriers` seam with handl's deferred-contribution
   retry** (`ContributionContext::contribute` + `PendingRule`), so extension
   machinery can register before or after its consumer loads
   (`catopt_orchestrator/carriers.py`).
8. **Record a decision note reinforcing ADR-0003** against handl's
   `Model::rewrite` unification (§3.1) — a `PerformanceModel` must never decide
   equivalence — and against the "one `Obj` world" (§3.2).
9. **Use handl's Lean library as a *template for kernel metatheory*, not for
   law-library certification.**  The reusable pattern is: small binder-free
   calculus → local rewrite relation → preservation + strong-normalization
   proofs (`ProofAdvantage.lean`, `CutElimCongruence.lean`), with the honest
   axiom boundary (`EquationalLawInference.lean`).  The "equality is structural,
   no α-tax" argument (`PROOF_ADVANTAGE.md` §4) is a rigorous justification for
   *why catopt's 2-cell replay is cheap* and could be cited in ADR-0002 / plan
   0014 to contrast 2-cell replay (decidable) with the 3-cell word problem.

---

## Appendix — key handl artifacts consulted

- ADRs: `0052` (arrow calculus kernel), `0106` (object/arrow kernel), `0094`
  (transform engine), `0098` (one language, multiple interpreters), `0100`
  (directed HoTT limit), `0101` (frontier calculus), `0104b` (minimal Term
  core), `0119` (n-dimensional arrow calculus), `0120` (paper–impl alignment),
  `0043` (proofs as first-class values), `0067` (chunk certificates — deferred),
  `0044`/`0081` (language as data).
- Rust: `crates/plugin-engine/src/term.rs`, `term_data.rs`,
  `term_interpreter.rs`; `crates/handl-language/src/law_inference.rs`;
  `crates/host-cli/src/transform_categorical.rs`;
  `crates/plugin-engine-transform-rewrite/`; `crates/platform/src/{capability,
  plugin,plugin_model}.rs`; `crates/plugin-engine-exec-tree-walk/src/lib.rs`
  (`rewrite`).
- Research/Lean: `research/PAPER.md`;
  `research/arrow-calculus-kernel-poc/{FORMALISM_PROOF_GRAPHS,PROOF_ADVANTAGE,
  RESULTS_PROOF_GRAPHS}.md`;
  `lean/ArrowMinimalityLib/{ProofGraphs,ProofAdvantage,CutElimCongruence,
  EquationalLawInference,TraitInference,Termination,DependentProofGraphs,
  IteratedLoop}.lean`.
- Searched and found **absent**: e-graph / equality saturation / egglog
  implementation; cost-driven extraction; `verify_certificate`; pluggable
  search policy; 3-cell coherence; bounded-error rewrites.
