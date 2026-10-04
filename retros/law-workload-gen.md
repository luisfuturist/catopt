# Workload generation for the census — real novelty, zero new laws

The corpus-expansion retro ended with the corpus as the bottleneck
and named the missing half of self-play: **new workloads feeding the
census**.  This retro builds that half — a term-level generator,
`tools/law_workload_gen.py`, that mints *valid, novel* programs two
ways and runs the validated pipeline on the enlarged corpus.  No
`packages/` change; generation happens at the IR-term level the
census reads.

The verdict, plainly: **generation produces real census novelty —
+25 op-tuples and +126 shapes on seed 1 (+23/+170 on seed 7) — and
the pipeline converts some of it into genuinely new proposals (4 per
seed, all census-mixed-view).  Nothing ships.**  Worse for the
self-play hypothesis, the *kind* of novelty explains why: resampling
provably cannot leave the census's tuple support (0 new tuples, by
construction), and mutation's new tuples are arbitrary
juxtapositions — `neg(neg(slice))`, `pow(linear,linear)`,
`eq(matmul,·)` — the equalities over which are mostly false.  The
corpus's law-bearing shapes (`f(g,g)` same-view pairs, `f(u,r(u))`
reduce-chains) are consequences of architecture semantics; no
sampling or point-mutation process mints the *justification*.  Two
seeds, same answer: **0 → 0 shippable.**

Reproduce (~7 min CPU: generation seconds, two pipeline runs the
rest):

    .venv/bin/python tools/law_workload_gen.py --n 60 --seed 1 \
        --json /tmp/lawgen.json
    .venv/bin/python tools/law_workload_gen.py --n 60 --seed 7 \
        --json /tmp/lawgen_s7.json

## 1. The two strategies and the validity gate

Every accepted term must (i) mint under `Op.make`'s attr contract,
(ii) type-check — `catopt_core.typing._shape_of` not `INVALID`, all
leaf shapes concrete, no `Var`/`Param` name collisions — (iii) use
only `TorchSink().supported_ops`, (iv) *evaluate*:
`TorchConcreteEval.eval_term` on random fp64 leaves must return a
tensor (the same oracle the pipeline's numeric-truth stage uses, one
level up), and (v) differ from every corpus root AND every corpus
subterm under the leaf-numbered `shape_key`.

* **Resample** — the census's own conditional distribution as a
  generative model: sample a corpus root op-tuple ∝ count, expand
  each child by sampling `P(children | op)`, mint attrs by
  resampling observed per-op attr dicts verbatim, fill leaf slots
  with `(parent, position)`-conditioned corpus leaf shapes
  (`var`/`param`/`const` distinguished).
* **Mutate** — a real `TermCase` under 1–3 point mutations: **swap**
  an op node for a same-arity census sibling (attrs resampled from
  the new op's observed dicts), **graft** a shape-equal corpus
  subterm into the site (the donor's leaves come along —
  cross-model composition for free), or **lift** a leaf to a
  shape-equal subterm.

On the lowering boundary, plainly: generated `TermCase`s carry fresh
`randn` feeds/params built from leaf shapes, so `law_impact._probe`
lowers and verifies them through the same `_lower_extracted` +
`sink.verify` path as real models — no `nn.Module`, no export.  The
verify-fail counts in the ranked evidence (false rewrites caught
through lowered-module comparison on generated terms) confirm the
machinery works end-to-end on them.

## 2. Generation results (two seeds, n=60 each)

| seed | strategy | attempts | accepted | rate | nodes~ | depth~ | new tuples | new shapes |
|---|---|---|---|---|---|---|---|---|
| 1 | resample | 586 | 30 | 5.1 % | 5 | 2 | **0** | 40 |
| 1 | mutate | 64 | 30 | 46.9 % | 14 | 5 | **25** | 86 |
| 7 | resample | 778 | 30 | 3.9 % | 5 | 2 | **0** | 47 |
| 7 | mutate | 85 | 30 | 35.3 % | 18 | 7 | **23** | 123 |

* **Resampling cannot mint a new op-tuple — and doesn't.**  Every
  node is sampled from a census-attested tuple, so the generated
  term's op-tuple census is a subset of the corpus's by
  construction.  Confirmed empirically: 0 new tuples both seeds.
  Its novelty is shape-level recombination only — and *shallow*
  recombination (mean depth 2): the (parent → child-position)
  conditional encodes local legality, not program depth.  A
  first-order op Markov chain regenerates fragments, not workloads.
* **Mutation is the only novelty source**: ~0.8 new tuples per
  accepted mutant.  Sample new tuples (seed 1): `mul(split,rsqrt)`,
  `add(transpose,·)`, `square(matmul)`, `pow(linear,linear)`,
  `gelu(·)`, `div(·,·)`, `layer_norm(linear,·,·)`; (seed 7):
  `add(reshape,·)`, `add(split,const)`, `batch_norm(relu,…,…,…,…)`,
  `eq(matmul,·)`, `sub(linear,linear)`.
* **Validity is honest, not tuned.**  Resample's ~4–5 % acceptance
  is what independent leaf-shape draws cost under the typing gate;
  mutation's ~35–47 % is what random rewiring costs.  Neither is
  the bottleneck — attempts are milliseconds.

## 3. The pipeline delta — the acceptance test

Enlarged corpus: 92 → 152 terms, 156 → 181 op-tuples (seed 1) /
→ 179 (seed 7).  All 60 generated terms feed the census and the
schema matchers; the 16-term firing/reach probe is interleaved
8+8 across strategies.

| measure | seed 1 | seed 7 |
|---|---|---|
| proposals | 50 → 54 | 50 → 54 |
| new proposals | 4 (mixed-view) | 4 (mixed-view) |
| new proposals firing | 0 | 3 (each once, on the minting mutant) |
| shippable | **0 → 0** | **0 → 0** |

The new proposals, in detail:

| seed | candidate | verdict |
|---|---|---|
| 1 | `mixed:add_transpose_l_{id,w}` | false (numeric oracle rejects) |
| 1 | `mixed:mul_split_l_id` | false |
| 1 | `mixed:mul_split_l_w` | **true**, new — match=1, fires 0 |
| 7 | `mixed:add_reshape_r_{id,w}` | unproven, fires 1 each on `gen:mutate:7` |
| 7 | `mixed:add_split_l_id` | fires 1 + **paid 1** — and *false* (the truth gate catching a paid lie) |
| 7 | `mixed:add_split_l_w` | true, fires 1, never lowers cost |

Shared candidates gained real evidence from generated workloads —
the interesting rows:

* `id_mul_lit` (duplicate of the shipped mul-identity): first firing
  AND first payment, on `gen:resample:4` — a resampled term minted
  `mul(a, 1)`, a degenerate law-shape the corpus never wrote.
* `matmul_factor` (inverse of shipped): first firing + payment on a
  mutant.
* `reshape_transpose`: fires 23 → 27 (s1) / 23 → 34 (s7, four gen
  cases) — still numerically false.
* `grammar:sub_to_add_dup`, `mixed:mul_unsqueeze_*`,
  `mixed:mul_select_*`: +1 gen firing each.

## 4. The honest verdict on self-play's missing half

**It is not generation-by-resampling.**  Three compounding reasons,
each now measured:

1. **Resampling is confined to the census's support.**  The
   conditional distribution can re-combine attested tuples but never
   create one: 0 new op-tuples is a theorem, not a sampling
   outcome.  Whatever a resampler adds at shape level is
   second-order — co-occurrence, not structure.
2. **Mutated novelty is semantically arbitrary.**  Mutants mint
   tuples real programs never produce — `eq(matmul,·)`,
   `batch_norm(relu)`, `pow(linear,linear)` — because point
   mutations preserve *local* legality only.  The laws the pipeline
   actually ships need *satisfied* structure: `mul(select_k u,
   select_k v)` exists because one tensor is read under two views;
   `div(exp u, sum exp u)` exists because someone spelled a
   normalization.  In 48 new tuples across two seeds: **zero
   `f(g,g)` same-view tuples, zero new reduce-chains** — the two
   shapes the law-bearing families consume.  Only ~2/25 tuples per
   seed even qualified as mixed-view fodder, and the equalities over
   them were false or worthless.
3. **The pipeline already said where the bottleneck is.**  Corpus
   expansion with *real* architectures moved `mul_distribute` from
   "inapplicable" to "fires but unsafe+worthless" — real programs
   carry the *justification* for their shapes.  Generated programs
   moved candidates from "absent" to "present" the same way
   (`matmul_factor`, `id_mul_lit`, `add_split_*`) — but the
   justifications are absent, so the equalities are false or the
   sites are degenerate.

**What this does and does not rule out.**  It rules out workload
generation as *sampling*: resampling the census cannot feed the
census, and undirected mutation feeds it noise-shaped novelty.  It
does not rule out generation as *search*: a mutator that hill-climbs
toward a census-identified **unsatisfied shape** — e.g. patterns
whose relaxed match is present but whose strict (equality-
constrained) match is absent, or the `f(u, r(u))` chains
`_reduce_chains` finds without a recognizer — is a different
algorithm pointed at a specific gap, not a resampler.  Equally open:
real deployment traces and new architecture families — the
corpus-expansion route that demonstrably DID change pipeline
verdicts.

The reusable pieces, whatever direction is next: the five-gate
validity checker (type-check + `eval_term` + dedup) makes any future
generator honest for free; `term_to_case` makes generated terms
first-class `TermCase`s (they lower, verify, and pay reach cost
exactly like exported models); and the census-delta + pipeline-delta
measurement is the acceptance test any workload source must pass.

## 5. Caveats

* Two seeds, n=60 terms each; per-seed new-tuple counts are
  Poisson-ish (25 vs 23) but the qualitative verdicts are identical.
* The firing/reach probe saw 16 of 60 generated terms (8 per
  strategy) — the census and matchers saw all 60.  A true candidate
  matching only an unprobed term reports `fires=0`
  (`mixed:mul_split_l_w`, seed 1 — match on a non-probed mutant).
* "Evaluable" = evaluates once on `randn` leaves through
  `eval_term`; the pipeline's own `sink.verify` is the stronger
  check and ran where candidates fired.
* Mutated feeds are random, not the original models' trained
  tensors — verify compares before/after on the *same* feed, so
  equivalence evidence is unaffected; cost magnitudes are not the
  point here.

## Gates

Run from the main worktree; `tools/law_workload_gen.py` added, no
`packages/` change:

* `.venv/bin/ruff check tools/law_workload_gen.py` — pass
* `.venv/bin/ruff format --check tools/law_workload_gen.py` — pass
* `.venv/bin/python tools/law_workload_gen.py --n 60 --seed 1` —
  runs to completion (~7 min incl. both pipeline runs); tables above
* `.venv/bin/python tools/law_workload_gen.py --n 60 --seed 7` —
  second-seed replication; same verdict

Per the task bound the full pytest/coverage gate was **not** run
(11 GB host); the tool's only shipped surface is `tools/`.
