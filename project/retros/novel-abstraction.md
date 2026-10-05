# Retro: the first machine-constructed object

**Plan 0017 / ADR 0004, stage 4** — "the unit of invention is a
constructed object."  Date: 2026-10.

Stages 0–3 made objects *legal*: any pipeline candidate can be declared
as data and run the gauntlet.  But both inhabitants
(`mul_unsqueeze_l_id`, `sub_unsqueeze_l_id`) were candidates the
census/proposal machinery had *found*.  The ADR's actual claim is
stronger — the player's action space is **construction operations**,
not rewrites the corpus coughs up.  This stage ships the seed of that
space: `catopt_discovery.object_synthesis`, the constructor for
declared objects, plus the first objects the machinery *builds* —
none of them a pipeline candidate.

## The module

`object_synthesis.py` is deliberately small.  Objects are built from
**term specs** — pure data (`str` metavar, numeric `Const`,
`(op, *specs[, attrs])` tuples) — through three construction
operations:

- **`fold_object(name, spelled, kernel)`** — *fold representation*:
  the spelled-out composition is the LHS, the dispatched kernel the
  RHS.  `fold_object("softsign_fold", ("div","X",("add",("abs","X"),1)),
  "softsign")`.
- **`lift_object(name, step, carrier, apply_op, state=None)`** —
  *introduce abstraction*: a spelled program step lifted into a
  runtime carrier — `step → apply_op(carrier, state)`.  The affine
  scan (`apply`/`aff`) and the online-softmax monoid
  (`om_apply`/`om_elem`) are both expressible as data.
- **`compose_objects(name, first, *rest, specialize=...)`** —
  *compose abstractions*: instantiate `first`'s pattern under
  `specialize`, then rewrite its RHS by each premise at the first
  firing position (`meta.apply_rewrite_at`, so premise guards evaluate
  on the symbolic binding and a non-firing step fails the whole
  construction honestly with `None`).  The composite's `derivation`
  names the premises — so a composite over shipped rules can carry a
  replayable certificate.

A construction returns a `ConstructedObject` — `rule` + `kind` +
`construction` trace — and `store_constructed` persists it through
`evidence.store_object` (when `env=` gives an instance and the
composite has a `derivation`, `lemma_cert.materialize` runs first and
the replayable cert rides inside the record).  Nothing in the module
admits: the record still faces `evidence.run_gauntlet` unchanged.
`evidence.py`, `laws/`, `pipeline.py` untouched — construction is a
claim, the gauntlet is the referee.

## The constructed inhabitants

Five objects were built, stored as `kind="abstraction"` records, and
admitted (`usable: yes`, injected corpus):

| object | op | statement | gates of note |
|---|---|---|---|
| `softsign_fold` | fold | `div(x, abs(x)+1) → softsign(x)` | num_true, paid (3×) |
| `aff_step_lift` | lift | `add(matmul(A,h),x) → apply(aff(A,x),h)` | pays (carrier < spelled) |
| `om_lift` | lift | `matmul(softmax(S,-1),V) → om_apply(om_elem(S,V))` | pays; verifies |
| `aff_scan2_lift` | compose | `add(matmul(A2, add(matmul(A1,h),x1)),x2) → apply(aff(A2,x2), apply(aff(A1,x1),h))` | exists in **no** ruleset |
| `silu_fold_commuted` | compose | `mul(sigmoid(X),X) → silu(X)` | derivable + **2-step cert replays strict** |

`aff_scan2_lift` is the cleanest claim: `compose(aff_lift, aff_lift)`
with the state metavar specialized to a second step — a carrier object
that appears in *no* shipped ruleset (`SCAN_LAWS` has the one-step
lift and the metavar-composed step separately; nobody wrote the
fused two-block form).  The machinery built it, the store kept it, the
gauntlet admitted it.

`silu_fold_commuted` is the first object whose **cert gate does real
work**: `derivation=("comm_mul","silu_fold")` materializes a linear
2-step certificate over the named premises, and stage 8 replays it
strict — not the "no derivation recorded" both stage-3 inhabitants
honestly reported.

## On the real corpus

`default_gauntlet_corpus` (276 terms, the pipeline's real probe set):

- **`softsign_fold`: usable — yes.**  One real firing site
  (`intake:nn.Softsign` spells the div/add/abs form verbatim), paid.
- **`om_lift`: usable — yes.**  11 real matches (the attention
  blocks), 8 fires, paid on 3, closure 1.22×.
- **`aff_step_lift`: usable — no, refused at closure.**  True
  (17 real matches, num_true), novel, pays (12 fires, paid=3) — but
  2.72× enodes on the saturating probe terms > the 2.0× guard.
  **`aff_scan2_lift` worse: 6.89×.**  The carrier lifts mint an
  `aff`/`apply` enode per `add(matmul(·,·),·)` site; on the real
  corpus that is closure inflation, and the gate sees it.  This is
  the gauntlet discriminating at scale — the object is sound and
  useful per-fire but too productive in the saturating reach.
- **`silu_fold_commuted`: usable — no, refused at truth.**  An honest
  edge worth naming: `derivable` is only measured when a real match
  supplies an instance, and no corpus term spells `mul(sigmoid(x),x)`
  (the corpus writes `mul(x,sigmoid(x))`).  matches=0 → the truth
  question is *unproven*, not failed — a constructed object with no
  real instance cannot even be asked "is it true".  On a corpus
  containing the swapped spelling it clears every gate, cert included.

Two honest negatives stand on the synthetic corpus:
`pow2_expand` (`pow(x,2) → mul(x,x)`, composed from shipped premises
— *true and provable*, but the `mul` spelling prices *above* `pow`
under the generic lowering → typed-pay refuses); and a deliberately
false fold (`abs(x) → square(x)`) dies at truth on `num_true=False`.

## What "constructed" bought — and honest limits

The measured claim is narrow but real: three construction operations
produce records that serialize completely, store, reconstruct, and
clear — or are honestly refused by — the same gauntlet a shipped law
faces.  The objects that pass are *usable by the pipeline*: stored
data, not Python hooks.

- **The lifts' `derivation` cannot certify under the current gate.**
  `stored_certificate` resolves premise names against `ALL_RULES`;
  `aff_lift`/`om`-family premises live in `SCAN_LAWS`/`OM_LAWS`, so a
  carrier composite's derivation is recorded metadata, never a cert.
  Certifying carrier-premised objects needs the gate to take the
  premise universe from the record — a deliberate next step.
- **`compose` is unconditional-only in practice.**  A premise guard
  sees symbolic metavar bindings (shape inference on metavar terms
  declines), so guarded premises mostly veto the composition — the
  honest failure, but it means `compose` today covers the
  unguarded fragment.  Composing *guarded* objects needs
  `cond`-transport (re-express premise conditions over the composite's
  metavars) — the `meta._compose_guards` machinery is the seam, not
  yet wired in.
- **`truth` needs a real instance.**  The unguarded truth gate reads
  `num_true`, which is measured on a corpus match only — a
  constructed object with no firing site in the corpus is unproven
  even when derivable.  A guarded object dodges this (the sweep
  synthesizes its region), which is an argument for declaring objects
  with their `cond` even when the guard is nearly tautological.
- **Closure is the real adversary for carriers.**  The carrier lifts
  pass truth/novelty/typed-pay cleanly and fail only at enode ratio —
  measured, not hypothetical: a runtime abstraction that fires
  everywhere costs the e-graph more than it is worth at full-corpus
  saturation.  A still-more-novel object (an `om` *compose* tree
  admitted as one object, a carrier with a `cond` bounding where the
  lift is legal) is where this bites next.
- **No promotion.**  `usable` remains "admitted through the
  gauntlet"; `laws/` is untouched, the promotion decision manual —
  unchanged from stage 3.

## Where this leaves plan 0017

The ADR's action space now has inhabitants built by each operation:
folds (`softsign_fold`), lifts (`aff_step_lift`, `om_lift`), and
composites (`aff_scan2_lift`, `silu_fold_commuted`) — the last
carrying the first constructed proof certificate.  The next rung is
the one the ADR actually cares about: a construction the verifier
*needs* — an object whose admitted form unlocks a program no amount
of reordering could reach — plus `cond`-transport so guarded objects
compose.
