# How catopt works — syntax, structure, and the rewrite loop

The model graph is **syntax**; a category-theoretic **structure**
interprets it; the structure's **axioms** generate rewrites; **search**
picks the cheapest; the **certificate** proves it.

This is the conceptual pipeline behind ADR 0002.  It is deliberately
separate from the code walkthrough — it answers *why* the machinery
is shaped the way it is.

## The loop, step by step

Running example throughout: the recurrence `h_t = A·h_{t-1} + x_t`.

**1. Syntax.**  The graph as written — ops and wires.  Just a term,
no meaning.  *Example: a chain of `matmul(A, h)` + `add(x)`.*

**2. Structure.**  A mathematical object with two halves:

- **Signature** — its operations, i.e. the *shape*: "a fold over a
  sequence of `(A, b)` pairs."
- **Axioms** — equations those operations obey: associativity + unit.

*Example: the monoid `aff(A, b)`.*

**3. Match → the lift.**  Recognize that a piece of the graph has
the **signature's shape**.  That makes it a morphism *into* the
structure.  *Example: recognize the recurrence spine as an `aff`
fold.*  Matching is done on the **signature**.

**4. Rewrite.**  Now the structure's **axioms** become rewrite
rules.  *Example: associativity lets you regroup the fold.*
Rewriting is done with the **axioms**.  Side conditions
(`check` / `derive`) are the soundness boundary — the only layer
that keeps a false member out of the e-graph: adversarial witness
generation found three shipped matmul laws false on mixed-rank
bindings (and a fuse law minting an unevaluable member); all
guarded now.

**5. Saturate.**  Apply the axioms to a fixed point — this
*generates* every equivalent program.  The e-graph **is** this
closure (the consequences, not a search over them).

**6. Extract.**  Search the closure for the cheapest program the
backend can run.  *Example: the Blelloch tree.*

**7. Certify.**  The path original → extracted is a replayable
certificate (a 2-cell).

**8. Evaluate.**  Describe the extracted program's *characteristics*
(torch-free `ProgramFeatures`) and, when a target is available, its
runtime.  Evaluation is a separate dimension (ADR 0003): it **ranks**
candidates, it never changes what is equivalent.  A profiler observes;
it does not decide.

*Shortcut:* if the structure names the target directly, **construct**
it and skip steps 5–6.

```text
syntax → structure(signature + axioms) → match(signature)
       → rewrite(axioms) → saturate → extract → certify → evaluate
```

## The split that trips people up

| Half | What it is | What you do with it |
|---|---|---|
| **Signature** | sorts + operations — the *shape* of a morphism into the structure | **match** against it |
| **Axioms** | equations the operations satisfy | **rewrite** with them |

Matching is **not** based on the axioms.  You match on the
*shape* (signature); the axioms license the rewrites afterwards.
The precise statement: matching a structure = **finding a functor**
from the subgraph's free category into that structure — i.e.
finding an interpretation.

Two more consequences worth stating plainly:

- **Laws are entailed by the structure**, in two tiers: *axioms*
  (defining equations) and *consequences* (theorems derived from
  them).  Few axioms generate many consequences; the reach is their
  closure.  The split is *measured on the shipped library*, not
  asserted: `rule.kind` marks each member `axiom` / `lemma` /
  `redundant` (40 / 12 / 2 over `ALL_RULES`), and `rule.derivation`
  records the premise(s) proving each non-axiom.
- **Choosing the interpretation is the discovery act.**  The same
  graph reads as plain composition, a comonoid (pairing), a
  low-rank approximation (bounded), or a contraction problem —
  each exposing different laws.  Discovery = picking the richest
  reading, not applying laws inside a fixed one.

## The meta-layer — laws about laws, and where laws come from

The loop above runs *inside* a fixed law library.  Two mechanisms
operate on the library itself.

**Coherence — the 3-cell layer.**  A rewrite is a 2-cell; a law
*about* rewrites is a 3-cell.  `catopt_discovery.coherence` enumerates
the pair layer over the shipped library (co-firing critical pairs,
direct derivability edges); `catopt_discovery.coherence2` measures
depth 2 — stratification, reach under removal, and the mediator
table ("the coherence of `A × B` requires `C`").  The measured
structure: the derived layer of `ALL_RULES` is *flat* — zero laws
derivable from the primitives alone, every derivation a
self-supporting cycle (ten inverse pairs plus the 4-member matmul
class) plus one emergent law (`silu_mul_form`); the effective
basis is 40 = 29 primitives + 11 cycle seeds, marked on the rules
by `--emit-basis`.  Two honest findings: **derivable ≠
disposable** — removing a derivable law costs real bounded reach
(the twin proves the *equality*; only this direction's LHS fires
on those spellings — `linear_to_matmul_t` is provable yet the most
load-bearing derivable law under removal); and **divergence is
found and mediated** — `silu_fold` shipped to restore confluence
where `silu_expand` destroyed the `swiglu_fuse` redex, and it pays.

**Discovery — the closed loop.**  `catopt_discovery.pipeline` runs the
whole pipeline on a corpus of real exported models (94 terms / 163
op-tuples today): **census** the op-shape tuples that occur →
**propose** candidates (census-naturality, mixed-view,
pattern-recognition and grammar schemas — and the proposers' op
alphabet is itself property-derived by `catopt_discovery.vocab`, not
hand-listed) → **verify** each candidate (numeric oracle,
`sink.verify` on lowered modules, derivability oracle) →
**measure** (fires / cost deltas / cert replay / closure ratio) →
**rank** → **emit** (`--emit-admission` writes the `R(...)` block
plus generated tests as a reviewable patch — the emitted law is
bytecode-identical to the hand-written one in the held-out check).
Two shipped laws came out of this loop (`select_mul`,
`softmax_fold` — `silu_fold` came out of the coherence layer
above), and the loop audits itself:
adversarial witness generation (`catopt_discovery.gap_gen`)
produces bindings the corpus cannot — it falsified paying-but-false
candidates and the shipped matmul-family hole; the sqlite evidence
store (`--evidence-db` / `catopt_discovery.evidence`) caches verdicts
keyed by corpus × rules × code revision (~19× on a warm run) so
"what did we measure, and did it change?" is a query, not a
re-run.

## Where this lives in the code

| Step | Code |
|---|---|
| Signature | `BlockSig` / `InputSig` / `WeightRef` (`catopt_orchestrator.morphisms`) |
| Axioms | `catopt_core.laws` (`tensor.py`, `scan.py`); morphism laws' `check` / `derive`; the axiom/lemma marks `Rewrite.derivation` / `kind` + `tags.REDUNDANT` |
| Match (lift) | `lift_graph`, carrier recognition (`lift_scan_to_applyd`) |
| Saturate | `EGraph` |
| Extract | `extract_best`, `backend_cost` (measured pricing: `delivered_cost_for(profile, x=…)` / `executor_cost_for`) |
| Certify | `verify_certificate` |
| Evaluate | `catopt_core.features` (`ProgramFeatures`, `StaticProfiler`) · `SearchResult.frontier` · `PredictedCriterion`; `Meter` / `TargetProfile` for measurements (`tools/calibrate_profile.py`) |
| Discover (meta) | `catopt_discovery.pipeline` (census → propose → verify → measure → rank → emit) · `catopt_discovery.vocab` (property-derived op alphabet) · `catopt_discovery.gap_gen` (adversarial witnesses) · `catopt_discovery.emit` (admission patch) |
| Coherence (meta) | `catopt_discovery.coherence` (`--emit-basis`) · `catopt_discovery.coherence2` (stratification / reach / mediators) · `catopt_discovery.evidence` (sqlite verdict store) |

## See also

- `project/adrs/0002-categorical-re-expression-thesis.md` — the
  thesis this pipeline implements.
- `project/adrs/0003-evaluation-is-an-independent-dimension.md` — the
  evaluation dimension this pipeline appends.
- `project/plans/0014-certified-re-expression.md` — the staged
  path from proofs to delivery.
- `project/retros/axiom-lemma-split.md` and
  `project/retros/law-coherence-depth2.md` — the measured kernel
  and the depth-2 coherence structure.
- `project/retros/automated-admission.md` and
  `project/retros/law-gap-targeted-gen.md` — the emit step and
  the adversarial oracle half of the discovery loop.
