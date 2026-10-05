# Retro: the shippable audit — 11 rules, both bars, no disagreement

Date: 2026-10.
Context: `project/retros/corpus-round-4.md` (the `shippable: 0 -> 11`
claim), `project/retros/derivable-gate.md` (the truth-gate
tightening) and `project/retros/admission-gauntlet.md` (the gauntlet
itself).

Corpus round 4 reported **`shippable: 0 -> 11`** from the pipeline's
own `measure`.  That is the *pipeline's* verdict — the candidate ship
bar — not the gauntlet's admission bar.  This audit runs each of the
11 through the full gauntlet (`evidence.run_gauntlet`) and reports
the result honestly.

**Headline: all 11 are `usable: yes`.**  The two bars do not
disagree anywhere on this set.  The interesting finding is *why*, and
the two places where they would disagree on other inputs.

## The per-rule verdict

Each of the 11 was stored as a `kind="law"` object and run through
`run_gauntlet` on the real corpus (`default_gauntlet_corpus`: bench +
models + intake, `ALL_RULES`, `TermCaseSink`):

| rule | pipeline `shippable` | gauntlet `usable` | first refusal |
|---|---|---|---|
| `sub_self` | yes | **yes** | — cleared |
| `grammar:pow_one` | yes | **yes** | — cleared |
| `reshape_reshape` | yes | **yes** | — cleared |
| `square_neg` | yes | **yes** | — cleared |
| `factor_left` | yes | **yes** | — cleared |
| `factor_right` | yes | **yes** | — cleared |
| `factor_sub_left` | yes | **yes** | — cleared |
| `factor_sub_right` | yes | **yes** | — cleared |
| `neg_add` | yes | **yes** | — cleared |
| `exp_add` | yes | **yes** | — cleared |
| `sub_neg` | yes | **yes** | — cleared |

**Zero refusals** — no rule stops at truth, novelty, typed-pay,
closure or cert.  Every one fires exactly once on its site
(`fires == fires_typed == paid`, `fires_ill_typed == 0`,
`verify_fail == 0`, `cert_fail == 0`, `closure_ratio <= 1.67x`).

The stage-4 truth detail for every rule is the *unguarded* form —
`numeric=True derivable=False` (or `derivable=True` for `sub_neg`) —
with **no `guarded:` sweep**.  That is the whole story.

## Why they agree — the gauntlet's teeth are dormant here

The gauntlet's one genuinely stricter stage is the **guarded-region
sweep** (`_guarded_truth`, stage 4 for a `cond`-carrying object).  All
11 rules are **unguarded** — `rule.cond is None and rule.check is
None` — so `_truth_gate` takes its unguarded branch:

```python
ok = evd.num_true is not False and (
    evd.derivable or evd.num_true is True
)
```

which is the pipeline's own `Evidence.truth`
(`derivable or num_true is True`) plus the derivable-gate tightening
(`num_true is not False`).  Every one of the 11 measures
`num_true is True`, so the tightening's extra clause never fires and
the two bars coincide *by construction*.  The gauntlet does not
independently re-verify these — it inherits the pipeline's
single-instance numeric oracle, exactly as an unguarded law always
has.

## The two systematic differences (real, but dormant)

The bars are not the same predicate; they diverge on inputs the 11 do
not present.  Both divergences are pinned as unit facts in the new
test file.

| dimension | pipeline `shippable` | gauntlet `usable` | who is stricter |
|---|---|---|---|
| truth, unguarded | `derivable or num_true is True` | `num_true is not False and (…)` | **gauntlet** — a measured counterexample outranks a derivation |
| truth, guarded | the same `truth` (no sweep) | full guarded-region sweep | **gauntlet** — by far |
| closure | `cert_fail == 0` (every `add_cert`) | *attributable* cert failures only | **pipeline** — the gauntlet excuses ambient failures |
| instance | `proposal.instance` (hand-written) | rebuilt proposal carries none → `_instance_from_match` | latent (grammar rules) |

- **Truth (derivable vs counterexample).**  The derivable-gate retro
  explicitly flagged `pipeline.Evidence.truth` as "the sibling
  instance … out of this change's scope".  This audit confirms it is
  a real divergence: `Evidence(num_true=False, derivable=True).truth`
  is `True` (pipeline), while the gauntlet's unguarded gate returns
  `False`.  None of the 11 is in that state.
- **Closure (ambient cert).**  The admission-gauntlet retro decided
  the gauntlet charges only failures *attributable* to the object,
  because the pipeline's aggregate `cert_fail` counts a pre-existing
  baseline break (`intake:SincKernel`).  So a rule with an ambient
  failure is pipeline-`shippable=False` but gauntlet-`usable=True` —
  the gauntlet is deliberately the *looser* one here.  All 11 have
  `cert_fail == 0`, so it does not bite.
- **Instance loss.**  `_measure_gate` rebuilds the proposal from the
  stored record, which carries no `instance`; the grammar proposals'
  hand-written instance is dropped and `_instance_from_match` is used
  instead.  For `grammar:pow_one` a real match exists, so the verdict
  is unchanged — but a grammar rule with a concrete instance and no
  real match would measure differently under the two bars (it cannot
  ship anyway, since `fires > 0` needs a site).

## The design question — align or document?

**Recommendation: document, do not change pipeline semantics.**  The
two properties answer different questions and their divergences are
each independently justified:

- `pipeline.Evidence.shippable` is the **candidate** bar — a proposal
  measured over the corpus, ranked, reported.  It has no object, no
  `cond`, no stored record.
- `evidence.run_gauntlet(...).usable` is the **stored-object** bar —
  a declared record admitted through the same ladder a shipped law
  faced, with a guarded sweep and a cert replay the pipeline does not
  run.

They are aligned in *intent* (truth ∧ new ∧ typed ∧ pays ∧
bounded-closure ∧ cert-clean) and identical for the unconditional,
hook-free laws — which is the whole of round 4's yield.  The two
genuine asymmetries are deliberate:

- the gauntlet's truth tightening is the *point* of the derivable-gate
  change (a proof must yield to a measured site);
- the gauntlet's closure leniency is the *point* of the ambient-cert
  decision (do not condemn an object for a corpus instability).

The one item worth a follow-up is the **pipeline's `cert_fail == 0`**:
it is over-strict relative to the gauntlet, because it charges ambient
failures the gauntlet deliberately excuses.  Aligning it would be a
pipeline-semantics change and is **out of this audit's scope** —
reported here, not made.  (The derivable-gate retro's flagged
`pipeline.Evidence.truth` follow-up is the same class: a real latent
divergence, dormant on the 11.)

## Are the 11 *useful*? — honest assessment

**They are true, well-typed, and they pay — but only on the corpus
grown to spell them.**  Measured against the *baseline* corpus
(bench + models, no intake) all 11 are `match=0 fires=0 paid=0
shippable=False`.  Every firing site is a round-4 intake workload:

| rule | site(s) | drop | derivable |
|---|---|---|---|
| `factor_left` | `intake:SharedFactorMixture`, `intake:QuadraticFeature` | 33% | no |
| `factor_right` | `intake:SharedFactorMixtureRight` | 33% | no |
| `factor_sub_left` | `intake:SharedFactorContrast` | 33% | no |
| `factor_sub_right` | `intake:SharedFactorContrastRight` | 33% | no |
| `exp_add` | `intake:ExpProductHead` | 33% | no |
| `neg_add` | `intake:NegatedSum` | 33% | no |
| `reshape_reshape` | `intake:DoubleReshapeHead` | 50% | no |
| `square_neg` | `intake:SquareNegHead` | 50% | no |
| `grammar:pow_one` | `intake:PowOneHead` | 100% | no |
| `sub_self` | `intake:ScalarSelfCancel` | 100% | no |
| `sub_neg` | `intake:SubNegBias` | 0% (pays) | **yes** |

So "`shippable: 0 -> 11`" is a statement about the *corpus's* new
spellings, not about real networks: none of the 11 is reached by a
pre-existing bench or model.  The 11 are, honestly:

- **real laws** (numerically true within the oracle's fp64 tolerance,
  not library duplicates — `relation=new`), and they pay;
- **elementary algebra** — `x-x=0`, `x**1=x`, `(-x)^2=x^2`,
  `x-(-y)=x+y`, `(-x)+(-y)=-(x+y)`, `e^x e^y=e^{x+y}`,
  `x*y ± x*z=x*(y±z)` — the textbook identities the library lacked
  (it already had the *matmul/linear* factoring controls, but not the
  elementwise ones);
- **spelled by contrived micro-modules** — several round-4 workloads
  are two-line `nn.Module`s written to produce the exact pattern
  (`(-x)+(-y)`, `x**1`, `s-s`), not natural model code.  The
  shared-factor and double-reshape sites are the most plausible;
  the scalar-corner and pure-grammar ones are spellings.

Verdict: the 11 are *usable* in the gauntlet's sense and *correct*,
but their demonstrated usefulness is **corpus-circular** — they are
recognized where round 4 deliberately taught the corpus to spell
them.  That is not a defect in the rules or the gates; it is the
honest reading of what "shipped" measures here.

## Tests

`tests/test_discovery_shippable_audit.py` (new) pins the audit's
verdict per rule on a small synthetic corpus carrying one concrete
spelling site each (fast — the whole file runs in ~4 s, no heavy
intake corpus):

- `test_shippable_rule_clears_the_gauntlet` (parametrized over all
  11) — `usable`, every stage passed, `fires == 1`, `paid == 1`,
  `cert_fail == 0`.
- `test_pipeline_shippable_and_gauntlet_usable_agree` (parametrized) —
  `measure().shippable is True`, `run_gauntlet().usable is True`,
  and the two are equal.
- `test_shippable_rules_are_unguarded` (parametrized) — the
  structural reason: `cond is None and check is None`.
- `test_gauntlet_truth_takes_the_unguarded_branch` — the truth gate
  leaves `synth_region is None` and the detail has no `guarded:`.
- `test_gauntlet_truth_is_stricter_on_a_derivable_counterexample` —
  the pipeline/gauntlet truth divergence (unit).
- `test_gauntlet_closure_is_looser_on_an_ambient_cert_failure` — the
  closure divergence (unit).

## Gates

`pytest tests/test_discovery_gauntlet.py tests/test_discovery_evidence.py
tests/test_discovery_intake.py -q` and `ruff check` / `ruff format
--check` / `ty check` / `radon_ratchet` / `vulture` are clean **against
the committed tree**.

> *Concurrent working-tree state.*  The audit was run while the tree
> carried an in-flight, uncommitted sibling edit (`oracle.py`,
> `laws/tensor.py`, `tests/test_cond_laws.py` — the value-bank
> widening).  That edit moves the enumeration, and three
> `test_discovery_gauntlet.py` tests that read region counts directly
> (`test_sdpa_fold_addmul_starved_at_the_default_cap`,
> `test_guarded_truth_escalates_a_starved_region`,
> `test_truth_gate_uses_the_selective_cap`) fail against it — the
> starved `sdpa_fold_addmul` window now accepts 6 sites instead of 0.
> Verified by re-running the gate in a `git worktree` at HEAD with the
> committed `oracle.py`: **115 passed**.  None of those tests touch
> this audit; all six new tests here pass in both trees.

No new suppressions.  `pipeline.py`, `evidence.py`, `oracle.py`,
`intake.py` untouched — the audit is read-only outside its own test
file and this retro.

## Limits

- **Measured, not proven.**  The gauntlet's truth for an unguarded law
  is the *single-instance* numeric oracle; a counterexample outside
  the sampled instance is not caught.  This is the same epistemic
  bound the shipped laws were admitted under.
- **Corpus-relative yield.**  "Pays" means pays on the corpus the
  pipeline measures; the 11 pay only on round-4 intake sites.
- **The two asymmetries are dormant, not absent.**  A future rule in
  a `derivable ∧ num_true=False` state, or one with an ambient cert
  failure, will make the pipeline and the gauntlet disagree — by
  design.  That is the case to watch, not this set.
