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
Rewriting is done with the **axioms**.

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
  closure.
- **Choosing the interpretation is the discovery act.**  The same
  graph reads as plain composition, a comonoid (pairing), a
  low-rank approximation (bounded), or a contraction problem —
  each exposing different laws.  Discovery = picking the richest
  reading, not applying laws inside a fixed one.

## Where this lives in the code

| Step | Code |
|---|---|
| Signature | `BlockSig` / `InputSig` / `WeightRef` (`catopt_orchestrator.morphisms`) |
| Axioms | `catopt_core.laws` (`tensor.py`, `scan.py`); morphism laws' `check` / `derive` |
| Match (lift) | `lift_graph`, carrier recognition (`lift_scan_to_applyd`) |
| Saturate | `EGraph` |
| Extract | `extract_best`, `backend_cost` |
| Certify | `verify_certificate` |
| Evaluate | `catopt_core.features` (`ProgramFeatures`, `StaticProfiler`) · `SearchResult.frontier` · `PredictedCriterion`; `Meter` / `TargetProfile` for measurements |

## See also

- `project/adrs/0002-categorical-re-expression-thesis.md` — the
  thesis this pipeline implements.
- `project/adrs/0003-evaluation-is-an-independent-dimension.md` — the
  evaluation dimension this pipeline appends.
- `project/plans/0014-certified-re-expression.md` — the staged
  path from proofs to delivery.
