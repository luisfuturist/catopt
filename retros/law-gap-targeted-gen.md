# Gap-targeted synthesis — the gaps are real, and they were hiding more than absence

The workload-generation retro (`project/retros/law-workload-gen.md`)
measured that *undirected* generation cannot feed self-play:
resampling is confined to the census's tuple support (0 new op-tuples
by construction) and mutation's 48 new tuples contained zero
law-bearing shapes.  It left exactly one door open: **generation as
search** — synthesize a workload that *contains* a census-identified
unsatisfied shape, rather than hoping sampling hits one.  This retro
closes that loop with `tools/law_gap_targeted_gen.py`.

The pipeline (`tools/law_pipeline.py`, `--vocab derived`, 50
proposals) reports 28 candidates that never fired on a real model and
were not proven false — the "no firing on a real model" /
"inapplicable (no real match)" / "unproven (no oracle)" verdicts.
For each, the tool instantiates the proposal's LHS *pattern* to a
concrete term (metavariables → fresh `Var` leaves, repeated metavars
shared so `sub(x,x)` really self-subtracts; attribute metavariables
minted against the instantiated child shapes — `select.index` inside
the chosen `dim`, `reshape.shape` numel-preserving, `split.sizes`
summing right), gates it through the same machinery the census uses
(`_shape_of` concrete, `check`/`derive` honored under `"$attr:"`
bindings, sink-lowered ops, torch-evaluable, corpus-novel when the
corpus lacks the shape), embeds it three ways — bare (`min`), inside
`add(·, w)` (`ctx`), and grafted into a real model skeleton's
shape-equal leaf slot (`graft`) — then runs the pipeline's own
referees: `_probe` (lone-rule fire / cost / lowered-module verify)
and `_reach_row` (`ALL_RULES` vs `+rule`: end-to-end drop, cert
replay, enode ratio).  The numeric-truth and derivability oracles run
on the generated instance too — which for the 17 `num_true=None`
candidates is the **first instance they have ever had**.

Reproduce (~8 min CPU: the baseline pipeline ~4 min, synthesis
seconds, measurement a few minutes):

    .venv/bin/python tools/law_gap_targeted_gen.py --json /tmp/gap.json

## 1. The headline numbers

| measure | count |
|---|---|
| gap targets (fires=0, not proven-false) | 28 of 50 proposals |
| valid instances synthesized | **28/28** |
| now fire on a generated workload | **28/28** |
| pay on ≥1 generated case | **18/28** |
| would clear the ship bar *if the shape were real* | **13/28** |
| falsified by the generated instance (num_true=False) | 3 |
| over the 2.0× closure limit | 5 (two catastrophically: 83×, 687×) |
| verify FAIL (lowered modules differ) | 2 candidates (+2 with lowering errors) |

Generation-as-search works mechanically: every LHS pattern admits a
valid witness, every witness fires, and the verdicts move off
"inapplicable" for all 28.  The interesting question is what they
moved *to*.

## 2. Where the 28 landed — four honest buckets

**(a) Would ship on generated evidence (13).**  All contractive
simplifications: `grammar:pow_one`, `factor_left`, `factor_right`,
`factor_sub_left`, `factor_sub_right`, `select_add`, `select_sub`,
`slice_mul`, `reshape_reshape`, `neg_add`, `sub_neg`, `exp_add`,
`square_neg`.  On minimal, context, and grafted cases alike the lone
rule fires, extraction picks the rewrite (33–100 % cost drops), the
lowered module verifies, the cert replays, the closure stays ≤ 1.67×.
Caveat: `sub_neg` is *derivable* (verify_law proves `x−(−y)=x+y` from
the library) — it pays at the probe but the end-to-end reach drop is
0, so it is "SHIP" only under the pipeline's literal gate, adding
nothing the library can't already reach.  That leaves **12 genuinely
new paying laws** — if their shapes were real.

**(b) Fires but never pays (10).**  Every *expansive* direction:
`grammar:neg_distribute`, `sub_add_factor`, `div_add`,
`exp_distribute`, `square_mul`, `sigmoid_neg`, `mul_neg_left`, and the
mixed-view wrap variants `mixed:mul_reshape_l_w`,
`mixed:mul_transpose_l_w`, `mixed:mul_slice_l_w` (verified-true where
evaluable — and worthless: the rewrapped member costs what the input
costs).  The corpus wasn't missing a law; it was missing a *useless*
shape — the verdict the task brief predicted.

**(c) Falsified by generated evidence (3).**  `mixed:mul_slice_l_id`
(strip-the-view `mul(slice u, v) → mul(u,v)`) is shape-incoherent —
the generated instance made the oracle's first-ever evaluation and it
said NO, and the lowered modules *differ* (max_rel 0.82–1.20) on all
five probe cases.  `sub_self` and `mul_zero` are rejected on shape
grounds: `x−x → Const(0)` maps a shaped term to a scalar.  Note the
asymmetry the run exposed — `sink.verify` broadcasts (`sub(v,v)`
*verified* when the rewrite was taken) while the numeric oracle's
`_allclose` is shape-strict; the oracle's verdict is the stricter and
the right one at the root.

**(d) Hazards and counterexamples (the rest).**  `mul_zero` and
`sub_self` don't merely fail truth — adding them to the rule set
**explodes the closure**: up to 83× and 687× enode growth
(Const-minting rules give every const-consuming law a fresh target).
Three expansive-direction grammar laws also breach the 2.0× limit
modestly (`sigmoid_neg` 3.3×, `neg_distribute` 2.5×, `div_add` 2.3×)
— closure hazard, not just worthlessness.  The corpus could never
have surfaced any of this — it contains no `x*0` to fire on.
`id_mul_lit` (duplicate) and `matmul_factor` (inverse) fire and pay
but are relation-gated.

## 3. The finding that matters more than the table

`matmul_factor` (`add(matmul(x,a), matmul(x,b)) → matmul(x, a+b)` —
the *inverse* of shipped `weight_factor_matmul`) drew `x=(16,)`,
`a=(16,)`, `b=(16,16)` on attempt 67:  **`num_true=False`**, and the
lowered modules differ (max_rel 0.29–0.72 on `min1`).  The equality
is genuinely false on mixed-rank bindings: `matmul` contracts the
right operand's *first* axis while `add` broadcast-aligns the *last*
axis — `x·a` is a scalar and `x·B` a vector, so the LHS broadcasts
the dot product over the vector while the RHS contracts `a_j·Σx`,
not `Σx·a`.

**The same counterexample binding falsifies the shipped
`weight_factor_matmul`, `factor_matmul` and
`distribute_matmul_over_add`** — all unconditional.  `x@w + x@W`
with `w` a 1-D weight (a matvec plus a matmul, summed — a low-rank
correction, a perfectly plausible architecture shape) matches the
shipped LHS and rewrites to a numerically different term.  The corpus
never produced a mixed-rank `matmul` operand pair, so the
unsoundness was invisible to every prior measurement — the shape was
real-world-plausible but *corpus*-absent, which is precisely the
blind spot gap-targeted synthesis exists to probe.  Mitigation: the
orchestrator's `lower`-time `sink.verify` gate is the last line of
defense and would catch the wrong member at optimization time; the
rule itself remains unsound on that binding (a rank/shape `check` —
e.g. both weight metavars same rank — is the fix, and belongs in
`packages/`, out of this task's `tools/` scope).

`mixed:mul_slice_l_w` is the same phenomenon weaker: truth is
*binding-dependent* — one generated instance (broadcastable `v`)
says `num_true=yes`, another (full-shape `v`) says NO.  The law needs
a shape side-condition; the corpus's 16 real matches were
numerically undecidable (symbolic leaf dims) and the generated
witnesses show both outcomes exist.

## 4. Would these shapes occur in a real architecture?

Judgment, argued per bucket — the honest criterion from the task:
*a law that only pays on shapes no real architecture produces is
library weight without value.*

| candidate | pays | plausible in a real model? |
|---|---|---|
| `reshape_reshape` | yes | **yes** — consecutive `view`/`flatten` calls are idiomatic; corpus absence is likely a coverage accident (or the frontend fuses adjacent views before IR). The most plausible of the paying set. |
| `select_add` / `select_sub` | yes | **moderate** — same-index gathers feeding pointwise exist (positional arithmetic, attention biases), but the corpus says `mul(select,select)` (25 sites) is what real code writes; add/sub pairs are rarer cousins. |
| `slice_mul` | yes | **moderate** — gated same-range slices (`x[..:d]*y[..:d]`) occur in GLU-style blocks, though real GLUs slice *different* ranges. |
| `factor_*` (`xy+xz`) | yes | **moderate** — the corpus's 21 relaxed `add(mul,mul)` sites all lack the shared factor; real gating shares the *input* through `linear`/`matmul` (covered by `weight_factor_*`), not raw `mul`. |
| `neg_add`, `square_neg`, `exp_add`, `pow_one` | yes | **low–moderate** — unary-minus pairs, squared negations, exp products, `x**1` are artifacts more than idioms (`pow_one` mostly arises from computed exponents). |
| `sub_self`, `mul_zero` | (yes, but hazard/false) | **low** as source-level shapes — and moot: closure blow-up + shape-incoherence disqualify them on safety, not applicability. |
| `matmul_factor` mixed-rank | — | **yes, and that's the problem** — a rank-1 correction `x@v` summed with `x@W` is a LoRA-shaped term; the *shipped* rules already fire there. |

## 5. The verdict on generation-as-search

**It feeds the loop — and the loop's answer is mostly "the gaps are
real-world-absent for a reason."**  Three of the four outcomes were
predictable only in hindsight: the paying 12 are contractive
simplifications any compiler would ship on generality grounds, but
their generated-only pay is evidence of *capability*, not *value* —
the corpus gives no evidence a real graph ever contains `sub(x,x)`'s
neighbors.  The never-paying 9 confirm the corpus wasn't missing
laws, only useless shapes.  But the two outcomes nobody priced in are
the reason to keep the tool: **generation-as-search produced
evidence the corpus could not** — the first truth verdicts for 17
unproven candidates (two of them rejections), the first
closure-hazard measurement (Const-minting annihilators, up to ~600×),
and one shipped-rule counterexample.

So: generation-as-search is not a workload source for the census —
the generated terms are degenerate programs no architecture writes.
It is an **adversarial witness generator for candidate (and shipped)
laws** — the missing half of the oracle loop, not of the corpus loop.

## 6. Caveats

* One seed (1), one vocab (derived), ≤2 instances per candidate;
  per-instance truth (`mul_slice_l_w`) varies with the draw — the
  table reports the first witness per candidate, the JSON all rows.
* The probe feeds are `randn`, not trained weights; verify compares
  before/after on the same feed, so equivalence evidence is
  unaffected.
* `paid` is the lone-rule probe verdict; end-to-end reach drop is
  reported separately (`sub_neg`, `id_mul_lit`, `matmul_factor` pay
  at probe but move nothing the library can't already reach).
* The `matmul_factor` unsoundness finding was verified by direct
  torch evaluation (`allclose=False`, max diff ~12.8) on the
  counterexample binding — it is a tools/-level observation about
  shipped rules; fixing `packages/` is out of scope here.
* `--skip-baseline --only a,b` is the dev path (skips re-measuring
  the known-zero firing baseline); the default run re-derives the
  target set from a full `run_pipeline`.

## Gates

Run from the main worktree; `tools/law_gap_targeted_gen.py` added,
no `packages/` change:

* `.venv/bin/ruff check tools/law_gap_targeted_gen.py` — pass
* `.venv/bin/ruff format --check tools/law_gap_targeted_gen.py` —
  pass
* `.venv/bin/python tools/law_gap_targeted_gen.py` — runs to
  completion (~8 min incl. the baseline pipeline); tables above
* `.venv/bin/python tools/radon_ratchet.py` — pass (tools/ is
  outside the ratchet's `packages` target)

Per the task bound the full pytest/coverage gate was **not** run
(11 GB host); the tool's only shipped surface is `tools/`.
