# Retro: the admission gauntlet — a stored object earns "usable"

Date: 2026-10-04
Context: plan 0017 / ADR 0004, stage 3 — "abstraction as a legal
move".  Stages 0–2 made synthesized abstractions *legal objects*:
`laws.serialize` round-trips `law`/`abstraction`/`bridge` records,
`derivation` materializes as a replayable certificate, and
`evidence.py` learned `--add-object`/`--admit-object`.  But
`admit_object` only *reconstructs* a stored record into a live
`Rewrite` — reconstruction is not admission.  This stage closes the
loop: `--admit-object KEY --gauntlet` runs the same adversarial
gauntlet a shipped law faced, and reports `usable` only when every
gate passed.  `pipeline.py`, `oracle.py`, `cond.py`, `serialize.py`
and `laws/` were not touched.

## The seam stack

Three verbs, kept distinct:

- **store** (`--add-object` / `store_object`) — declare a record:
  pattern + `cond` + `dspec` + `kind` + `derivation` + optional
  `cert`, plus honest bookkeeping (`serializable`, `missing_hooks`).
- **reconstruct** (`--admit-object` / `admit_object`) — decode the
  record back into a live `Rewrite`; strict certificate replay is
  *shown*, not gated.
- **admit** (`--admit-object KEY --gauntlet` / `run_gauntlet`) — the
  record earns "usable" by clearing the gauntlet.  Exit code follows
  the verdict.

## The gates, in order

`run_gauntlet` runs eight stages; a failing gate stops the run —
later stages are not reached, not skipped:

1. **reconstruct** — the record decodes; unknown `kind` / missing
   row refuses.
2. **full-data** — `missing_hooks` empty.  A record that dropped a
   procedural `check`/`derive` admits a weaker rule than it
   declares.
3. **measure** — `pipeline.measure` on the rebuilt rule, hooks
   riding along on the `Proposal`: corpus matches, the numeric
   oracle, derivability, the raw view-oracle verdict, the firing
   probe, the typedness audit.  A measurement *error* is a failure
   to surface, not a pass — a malformed `cond` that raises on a real
   match is refused here.
4. **truth** — the declared equality must hold where the rule can
   fire.  Unguarded objects take the measured verdict (`num_true` or
   `derivable`).  A *guarded* object is measured differently — see
   below.
5. **novelty** — `relation == "new"`; a stored object that
   respells a library law is not a new inhabitant.
6. **typed-pay** — `fires > 0`, `fires_ill_typed == 0`, `paid > 0`,
   `verify_fail == 0`: every merged fire mints a well-typed member,
   extraction pays on at least one typed program, and the lowered
   modules agree (`_typed_probe` + `impact._verify`).
7. **closure** — `closure_safe` (no enode blow-up past the 4x
   guard), `reach_ill == 0`, and no *attributable* certificate
   replay failure — see below.
8. **cert** — a record carrying a `derivation` replays its
   materialized certificate strictly; corruption refuses the whole
   object.  A record without a derivation honestly reports "no
   derivation recorded".

## The guarded-region sweep — the real addition

The interesting gate is stage 4 for a `cond`-carrying object.  The
pipeline's own oracles answer "is this equality true" — for a
conditional law the honest answer is *conditional*, not true, and
the pipeline's ship bar correctly says no.  But a stored object
carries its own `cond`: the truth question for *it* is "does the
equality hold **on the bindings its guard accepts**".  The oracle
cannot ask that — it verifies patterns, not objects.

So `evidence.py` sweeps the guarded region itself
(`_guarded_truth`/`_guarded_evals`), reusing the oracle's *domain*
machinery — `_synth_sites` mirrors `oracle.synthesize`'s
enumeration (same leaf-binding banks, same shape-valid attribute
domains via `_viewed_bindings`/`_attr_domains`/
`_derived_free_shapes`) but yields the *binding environment* instead
of evaluating — and `_site_outcome` applies the rule's own `check`
and `derive` with firing-time semantics (a `check` raise is
`guard-err`, a `derive` `None`/raise is `declined`, an
uninstantiable RHS is `rhs-err` — the same abort reasons the engine
records).  The region passes only if every accepted evaluable
binding is `equal`, none errs, and the synthesized side exhibits at
least one `equal` — a guard that accepts nothing provable is
vacuous, not verified.  The same sweep runs over real corpus matches
(the `Schema`/`real_matches` machinery), counting `declined` on the
false region.

## One finding the sweep surfaced

`intake:SincKernel`'s reach row fails certificate replay under the
**base** ruleset too (`base_cert: FAIL (CertificateVerificationError)`,
`add_cert` the same, identical costs, no new fires) — an ambient
corpus instability, not attributable to any object.  The pipeline's
aggregate `cert_fail` counts *every* `add_cert` failure; charging it
to the object would condemn every firing candidate for a
pre-existing failure.  The gate therefore counts only
*attributable* failures — `add_cert != "pass" and
base_cert == "pass"` — and reports the ambient count separately.
`SincKernel` is worth its own bug (a baseline saturation whose
extraction certificate does not replay); this retro only records
that the gauntlet saw it.

## The first inhabitant

`mul_unsqueeze_l_id` — the oracle's strongest conditional —
stored as a serialized `abstraction` record:

```
mul(unsqueeze(U, dim=A_dim), V) -> mul(U, V)
cond := ("and", ("ones-before", "U", "A_dim"),
                ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"))
```

On the real corpus (`default_gauntlet_corpus`: 276 real terms,
158 probe cases, `ALL_RULES` base set):

```
[pass] reconstruct: abstraction rebuilds
[pass] full-data: the record carries every hook
[pass] measure: matches=8 census=4
[pass] truth: numeric=None derivable=False
       view=conditional (26 equal / 142 unequal evaluable
       instances; 40 ill-typed-RHS) | guarded:
       synth: 25eq/0ne/0rerr (25 accepted, 323 declined) |
       real: 1eq/0ne/0rerr (1 accepted, 7 declined)
[pass] novelty: relation=new
[pass] typed-pay: fires=1 typed=1 ill=0 paid=1 verify_fail=0
[pass] closure: enode=1.09x cert_fail=0 (ambient=1) reach_ill=0
[pass] cert: no derivation recorded
usable: yes — cleared
```

The firing site is `intake:ALiBiAttention` —
`mul(unsqueeze(softmax(linear …), -1), stack …)` — where the
inserted axis is exactly a broadcast pad; the fire pays an 8.7%
cost drop on that program.  On the synthesized domain the guard
accepts exactly the 25 bindings where the equality holds and
declines 323 (including the 142 where it is *false* — the
conditional the raw oracle named).  On the real corpus it accepts 1
and declines 7 — the guard bites on real programs, it is not a
tautology.  `usable: yes` — the first synthesized object admitted
through the gauntlet.

The sibling `sub_unsqueeze_l_id` (same pattern over `sub`, same
guard) was attempted second and also cleared: `usable: yes`, same
25/0/0 guarded synth region, real 1 accepted / 1 declined, 1 typed
paying fire on `intake:ALiBiAttention`, closure 1.13x.  Both
inhabitants pass the whole ladder — the fallback was not needed.

The honest negatives, all refused at the right gate: the *same
pattern with no `cond`* fails truth (`view=conditional`, `usable:
no`); `mul(u,v) -> add(u,v)` fails truth on the numeric oracle
(`num_true=False`); a `cond=True` guard that accepts the false
region fails truth with counterexamples; a true-but-vacuous object
(fires nowhere) fails typed-pay (`fires=0`); a library respelling
fails novelty (`relation=duplicate`); a non-serializable record
fails full-data; an unknown kind and a missing row fail
reconstruct; a corrupted derivation cert fails cert.

## Honest limitations

- **Finite domains.**  "Truth" means equal on every evaluable
  binding in the oracle's finite bank *that the guard accepts* —
  25/25 synthesized + 1/1 real here.  A counterexample outside the
  banks is not caught.  This is the same epistemic bound the shipped
  laws were admitted under; the gauntlet does not raise it.
- **Guard exactness is bounded by the DSL and shape inference.**  If
  the guard accepts a binding where the sides differ, the sweep
  sees it (`unequal`); but a guard written wrong in a way the DSL
  cannot express, or a shape the sink cannot infer, is outside the
  sweep's reach.  `guard_err` / `rhs-err` / `declined` counts keep
  those failures visible rather than silent.
- **The raw oracle verdict stays `conditional`.**  The guard is part
  of the object's semantics, not the oracle's — `verify_view_candidate`
  on the bare pattern still says conditional.  The gauntlet reports
  both readings; "usable" rests on the guarded one.
- **Corpus and backend scope.**  `measure` runs over the same
  real/bench/model/intake corpus and `TermCaseSink` the pipeline
  uses; "pays" means pays on *those* programs under *that* cost
  model.  A different backend's costing is not adjudicated here.
- **No certificate where no derivation exists.**  The synthesized
  objects carry `derivation=()` — there is nothing to replay, and
  the stage reports that absence honestly rather than minting a
  vacuous cert.  The cert gate's teeth were exercised on
  `silu_fold`'s real derivation (replays strict; a corrupted step
  refuses).
- **No automatic promotion.**  `usable` marks a stored object
  *admitted through the gauntlet* — usable by the pipeline — not a
  shipped law.  `laws/` remains human-admitted; nothing here writes
  to it.

## Where this leaves plan 0017

The chain the ADR demanded now exists end to end: a machine-named
conditional candidate can be declared as data (`abstraction` record
with `cond`), stored, reconstructed, and *admitted* — through the
same adversarial gates the shipped laws face — with a per-stage
report and a certificate slot that replays strictly when a
derivation exists.  Two inhabitants stand: `mul_unsqueeze_l_id` and
`sub_unsqueeze_l_id`, both `usable` on the real corpus.  The
remaining gap between "admitted object" and "shipped law" is the
promotion decision itself, which is deliberately still manual.
