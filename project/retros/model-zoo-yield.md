# Retro — the held-out model zoo: what fires, what pays, what ships

Date: 2026-10-06
Context: `project/retros/real-corpus-yield.md` measured
`shippable = 0` on the real half of the intake corpus and
`project/retros/shippable-audit.md` showed every ship was
corpus-circular.  Both measured *known* programs — the corpus the
laws were grown on, minus the spellings.  The decisive counterfactual
remained: **does the pipeline produce yield on real architectures
nobody targeted?**  This retro builds that holdout — a 22-module
zoo of architecture *families* absent from every corpus arm — and
measures both halves of the product on it: the shipped library
(`Optimizer`/`TorchBackend` end to end) and the discovery pipeline
(`workload_gen._run_pipeline` baseline-vs-enlarged, the intake
harness verbatim).

**Headline.**  On 22 held-out architectures: the *shipped* library
fires a law on 9 (plus a nonlocal param-dedup lift on a 10th),
changes the extraction on 6, and delivers a verified fp64 cost
reduction on 5 (best: `LoRAAdapter` **-20.0 %**,
`TimeCondConv` **-5.0 %**).  The *discovery* pipeline mints 16 new
proposals from the zoo's novel op-tuples, 21 proposals fire on zoo
cases, 6 pay somewhere — and **`shippable` stays `0 → 0`**.  The
optimiser yields on unseen architectures; the discovery engine still
does not ship off-corpus.

## The zoo

`catopt_discovery.zoo` — 22 `nn.Module` workloads, chosen by family
coverage, none registered in `intake.candidates()` and none written
to spell a law's region (`tests/test_discovery_zoo.py` pins the
disjointness against the registry, the `_PURPOSE_BUILT` ledger, the
model corpus, the bench `LAW_CASES` and the persisted
`intake_corpus.json` records).  Families the corpus never contained:

| family | zoo members |
|---|---|
| attention variants | `GatedAttentionUnit` (GAU, softmax-free), `AdditiveAttention` (Bahdanau), `TalkingHeadsAttention`, `CosineAttention` (SwinV2-style) |
| conditioning | `FiLMHead`, `AdaLNBlock` (DiT), `LoRAAdapter` |
| mixers / spectral / graphs | `MLPMixerBlock`, `FNOBlock`, `GCNLayer` |
| routing | `SoftSlotMoE`, `ExpertChoiceMoE`, `CapsuleRouting` |
| recurrence / conv | `ChunkedRetention` (RetNet), `DeepEquilibrium` (unrolled fixed point), `WaveNetGate`, `TimeCondConv` (DDPM residual block) |
| heads / objectives | `MoSHead`, `CovarianceHead` (VICReg), `SplineKAN`, `HighwayGate`, `AffineCoupling` (RealNVP) |

Ingestion (the intake's own `ingest()`, seed 0): **21 ingested
fp64-verified, 1 census-only, 0 rejected, 0 verify-failed.**  The one
census-only is honest and expected — `FNOBlock` exports but the
bridge has no `fft_rfft2` / `fft_irfft2` / `real` bindings.  The zoo
adds **+133 op-tuples, +244 shapes, +17 previously-absent ops**
(`einsum`, `permute`, `scatter_add`, `squeeze`, `linalg_vector_norm`,
`sqrt`, `log`, `clamp_min`, `rsub`, `zeros_like`, `detach_`,
`contiguous`, `expand_as`, `clamp`, the three fft pieces) — the
holdout is genuinely novel to the census, not a re-spelling of known
shapes.

## (a) The shipped library on the holdout — `Optimizer(backend=TorchBackend())`

`opt.search(model, x)` (default `DEFAULT_RULES`, fixed point,
executor-aware backend cost) then `opt.lower(res, x)` (generic
lowering, fp64 verify at rtol 1e-4).  All runs ~0.02–0.8 s.

| zoo model | enodes | rules fired | cost in → out | Δ% | changed | verify (max_rel) |
|---|---|---|---|---|---|---|
| GatedAttentionUnit | 22 | `mul_square`, `square_to_pow` | 182972.6 → 182966.8 | −0.003 | yes | pass (0.0) |
| AdditiveAttention | 28 | — | 261509.9 → same | 0 | no | pass (0.0) |
| TalkingHeadsAttention | 32 | — | 313779.6 → same | 0 | no | pass (0.0) |
| CosineAttention | 35 | `om_lift` | 365863.2 → 365840.2 | −0.006 | yes | pass (2.0e-16) |
| FiLMHead | 13 | — | 104495.6 → same | 0 | no | pass (0.0) |
| AdaLNBlock | 23 | — | 217830.1 → same | 0 | no | pass (0.0) |
| **LoRAAdapter** | 13 | `assoc_linear` (+1 pairing group) | 87045.3 → 69649.7 | **−20.0** | yes | pass (1.2e-16) |
| MLPMixerBlock | 21 | — (2 nonlocal lifts) | 156907.4 → same | 0 | yes* | pass (0.0) |
| FNOBlock | 15 | — | (unpriceable — unbound fft ops) | — | no | **lower fails: `fft_rfft2`** |
| GCNLayer | 19 | — | 226356.6 → same | 0 | no | pass (0.0) |
| AffineCoupling | 17 | — | 174054.7 → same | 0 | no | pass (0.0) |
| SoftSlotMoE | 14 | — | 148144.9 → same | 0 | no | pass (0.0) |
| ExpertChoiceMoE | 23 | — | 4.0e15 → same† | 0 | no | pass (0.0) |
| CapsuleRouting | 38 | `square_expand`, `square_to_pow` | 522100.6 → same | 0 | no | pass (0.0) |
| ChunkedRetention | 28 | `sub_to_add` | 252511.9 → same | 0 | no | pass (0.0) |
| DeepEquilibrium | 19 | — | 208938.8 → same | 0 | no | pass (0.0) |
| WaveNetGate | 16 | — | 156736.3 → same | 0 | no | pass (0.0) |
| **TimeCondConv** | 36 | `silu_expand`, `mul_unsq_pad_l`, `mul_unsq_pad_r` | 348289.4 → 330889.0 | **−5.0** | yes | pass (0.0) |
| MoSHead | 13 | `index_select_dedup`, `index_select_id` (+1 lift) | 139252.3 → same | 0 | no | pass (0.0) |
| CovarianceHead | 9 | `sub_to_add` | 78364.8 → same | 0 | no | pass (0.0) |
| SplineKAN | 11 | `mul_square`, `square_to_pow` | 87058.3 → 87055.5 | −0.003 | yes | pass (0.0) |
| HighwayGate | 14 | — | 139279.1 → same | 0 | no | pass (0.0) |

`*` `MLPMixerBlock`'s changed term reuses `p_norm1` at the second
norm site — legitimate, not a bug: both `nn.LayerNorm`s initialise
to weight=1/bias=0, so the bitwise-equal param dedup is exact
(max_rel 0.0).  On trained weights the lift would not apply; the
extraction kept it cost-neutral anyway.
`†` `ExpertChoiceMoE`'s 4e15 is four `_INVALID_COST` marks —
the executor cost model cannot price the gather/scatter-back
region's `scatter_add`/`reshape`/`expand` nodes there.  A pricing
artifact, not a correctness one: the module verifies.

**What paid, and how.**  Two substantial wins, three fidelity-level
drops:

- `LoRAAdapter`: `assoc_linear` rewrote `linear(linear(x,Wd),Wu)` →
  `linear(x, matmul(Wu,Wd))` — the low-rank chain fused into one
  weight product, a real LoRA-merge under the cost model.  −20 %.
- `TimeCondConv`: `mul_unsq_pad_{l,r}` — the *machine-admitted*
  guarded laws (`cond = bcast-eq ∧ bcast-into`, promoted in
  `promoted-laws.md`) — dropped the dead broadcast pad on the
  sinusoidal frequency table (`unsqueeze(v,0)` where `v` is
  `(4,)`-shaped).  −5.0 %, verified exactly.  **This is the first
  measured evidence that a minted `cond` law generalises to an
  architecture that was never in its admission corpus.**
- `CosineAttention`/`GatedAttentionUnit`/`SplineKAN`: `om_lift`
  (carrier map lift), `mul_square`+`square_to_pow` (the `x·x → square`
  spelling bridge) — sub-0.01 % each, all verified.

So the shipped side: **9/22 fire a law (10/22 counting the
nonlocal-lift pass), 6/22 change the extracted term, 5/22 pay,
21/21 lowerable verify fp64, 1 unlowerable (binding gap).**  The library is
not useless off-corpus — it is *modest*: two structurally real wins
out of 22, plus micro-drops.

## (b) The discovery pipeline on the holdout

Same harness as `real-corpus-yield.md`: `intake._run_delta` with
`bench+models` as the base corpus, `probe` = models + the 21
probe-eligible zoo cases (all ≤120 nodes), `vocab=derived`, no
holdout.

| metric | baseline (bench+models) | corpus + zoo |
|---|---|---|
| corpus terms | 110 | 132 |
| op-tuples | 211 | 344 |
| proposals | 54 | 70 (16 new: 3 `census:*_unsqueeze`, 13 `mixed:*`) |
| proposals firing | — | 28 |
| proposals paying | — | 10 |
| proposals firing on a zoo case | — | 21 |
| paying proposals that fire on zoo | — | 6 |
| **shippable** | **0** | **0** |

Every proposal that fires on the zoo is refused — and the refusal
reason is the same one the real corpus gave: the paying candidates
are all *conditional* view identities:

| proposal | zoo fire cases | fires | paid | first refusal |
|---|---|---|---|---|
| `mixed:mul_unsqueeze_l_id` | AdditiveAttention, GCNLayer, CapsuleRouting, TimeCondConv, MoSHead | 13 | 2 | conditional truth + `duplicate` (`mul_unsq_pad_l`) |
| `mixed:mul_unsqueeze_r_id` | AdditiveAttention, GCNLayer, ExpertChoiceMoE, CapsuleRouting, TimeCondConv | 11 | 1 | conditional truth (view oracle) |
| `mixed:add_unsqueeze_r_id` | AdditiveAttention, TimeCondConv | 3 | 1 | conditional truth (`unsq:d_in_pad`) |
| `mixed:sub_unsqueeze_l_id` | ChunkedRetention | 2 | 1 | conditional truth + `duplicate` (`sub_unsq_pad_l`) |
| `mixed:sub_unsqueeze_l_w` | ChunkedRetention | 2 | 1 | conditional truth (`w:v_commutes_view`) |
| `reshape_transpose` | TalkingHeads, CosineAttention | 30 | 1 | conditional truth |

Plus `grammar:mul_distribute` (fires on `AdaLNBlock`, numerically
true — never lowers cost) and `grammar:sub_to_add_dup` (fires on
`ChunkedRetention`/`CovarianceHead`, true but a library duplicate).
Eight more `mixed:*` proposals fire on zoo cases with `paid=0`.

**The paid sites are mostly fake drops.**  Re-probing each paying
candidate against its zoo cases (lone-rule `_probe` + lowered-module
verify) shows: of the **seven** paying zoo sites, six produce
extractions that `verify=FAIL` or `error` — the "id" strip deletes a
broadcast that was load-bearing, so the "pay" is the cost model
being charged for a program that computes something else
(`mul_unsqueeze_r_id` on `ExpertChoiceMoE`/`TimeCondConv`,
`add_unsqueeze_r_id`/`sub_unsqueeze_l_{id,w}` on
`TimeCondConv`/`ChunkedRetention`, `reshape_transpose` on
`CosineAttention`).  Exactly one pays *and* verifies:
`mixed:mul_unsqueeze_l_id` on `TimeCondConv` — the same site the
*shipped* `mul_unsq_pad_l` already covers (hence
`relation=duplicate`).  The pipeline counts these in `paid` *and*
`verify_fail` simultaneously, so even a loosened truth gate could
not ship them.  The guards and the verify gate are doing their
jobs: the loose sites are refused because they are wrong.

## (c) What would have fired if the guards were looser

Bare-rule sweep (`real_matches` over the 22 zoo terms →
`evidence._guarded_evals` per site, the round-4 probe verbatim) over
every proposal the enlarged corpus mints:

| proposal | zoo sites | accepted | equal | unequal | rhs-err | where |
|---|---|---|---|---|---|---|
| `mixed:mul_unsqueeze_l_id` | 6 | 6 | **1** | 0 | 5 | TimeCondConv (the pad strip is *true* there) |
| `mixed:mul_unsqueeze_r_id` | 6 | 6 | **1** | 0 | 4+1err | TimeCondConv |
| `mixed:add_unsqueeze_r_id` | 2 | 2 | **1** | 1 | 0 | TimeCondConv |
| `mixed:add_chunk_l_w` | 1 | 1 | **1** | 0 | 0 | AdaLNBlock (`chunk(u)+1` wrap commutes) |
| `grammar:mul_distribute` | 1 | 1 | **1** | 0 | 0 | AdaLNBlock (true, never pays) |
| `grammar:sub_to_add_dup` | 2 | 2 | 2 | 0 | 0 | ChunkedRetention, CovarianceHead |
| `reshape_transpose` | 6 | 6 | 0 | **6** | 0 | TalkingHeads, CosineAttention — all false |
| `mixed:{mul,sub,add}_unsqueeze_*_w` (4 rules) | 15 | all | 0 | 4 | 10+1err | real non-trivial broadcasts — the `_w` wraps mint ill-typed RHS or are outright unequal |
| `mixed:{mul,add}_chunk_{l,r}_{w,id}` (8 rules) | 14 | 14 | 1* | 1 | 12 | AdaLN/AffineCoupling/FiLM — chunk sites (*the one equal is `add_chunk_l_w` on AdaLN) |
| `mixed:{mul_slice_*, mul_slice_unsqueeze_*}` (5 rules) | 5 | 5 | 0 | 0 | 5 err | FNOBlock — complex-typed sites can't eval |

Reading: the held-out architectures DO carry sites where the loose
equality is true (7 equal sites across 6 proposals) — unlike the
real intake corpus, where every equal site was purpose-built.  But every
true site is either (i) already covered by a shipped guarded law
(the `unsq_pad`/`sub_unsq_pad` family — `relation=duplicate`), or
(ii) non-paying (`mul_distribute`, `add_chunk_l_w` pay 0).  The
*new* thing the zoo proves: the loose corners are inhabitable by
real code, and the shipped guards already sit exactly on the ones
that are both true and useful.

## The honest verdict

Three-layer answer to "does the pipeline produce yield on held-out
real architectures?"

1. **The shipped library yields — modestly.**  5/22 held-out
   modules get a verified cost reduction; two are structurally real
   (LoRA fold −20 %, dead broadcast-pad drop −5 %).  Nothing fails
   verification.  The product that pays on unseen programs exists —
   it is the optimiser + the (machine-admitted, guarded) law library.
2. **The discovery engine ships nothing off-corpus: `0 → 0`.**  The
   16 proposals the zoo's novel op-tuples mint either never fire,
   fire but don't pay, pay only through unsound extractions
   (verify=FAIL), or are already in the library.  The yield bottleneck
   is unchanged: real paying sites are *conditional* view identities,
   and the truth gate refuses them correctly — on the zoo the loose
   versions verify-fail at 6 of the 7 sites where they "pay".
3. **The verifier + guard machinery is where the value is.**  The
   one true-and-paying discovery site on the zoo is covered by a law
   the *guarded admission* path shipped — and that law generalises
   (`mul_unsq_pad_*` on `TimeCondConv`).  The realistic read: the
   discovery loop's demonstrated product is the verified, guarded law
   library it admits, not open-ended new-law discovery.  Discovery
   supply-side (census → proposal) works off-corpus — 16 novel
   proposals from 22 models — but the truth/verify gates correctly
   keep the ship set at zero, because nothing new is both true and
   worth it on this holdout.

If the project's goal is "an optimiser that helps on models nobody
targeted": **there is measured yield** — small, real, verified, and
concentrated in composition/fusion laws (matmul associativity,
broadcast pads, carrier lifts).  If the goal is "a discovery engine
that mints useful new laws on unseen architectures": **the honest
answer is still no** — not because proposals or firings are absent
(they are plentiful), but because every new paying equality on
held-out code is either conditional-wrong at its loose edge or
already shipped.

## Caveats / limits

- **Untrained weights.**  The zoo builds fresh-init modules; several
  signals (param dedup on `MLPMixerBlock`, the `index_select_*`
  dedup fires on `MoSHead`) live only on initialisation values.
  The wins that matter (`assoc_linear`, `mul_unsq_pad_*`) are
  structural, not value-dependent.
- **Cost-model fidelity.**  `paid`/`Δ%` are the executor-aware model's
  numbers, not wall-clock; `ExpertChoiceMoE`'s 4e15 shows the model
  still has unpriceable corners.  No timing claim is made.
- **The zoo is 22 hand-picked modules.**  Family coverage was the
  criterion; a different zoo could spell more or fewer of the
  conditional families.  The disjointness test pins only that this
  zoo was never ingested — not that it samples "all real programs".
- **One paid site is shared with the shipped law.**  The `id`
  variant of `mul_unsqueeze` paying on `TimeCondConv` is the same
  broadcast-pad region `mul_unsq_pad_l` already owns — net-new yield
  from *discovery* on the holdout is zero even at that site.
- **Probe measured while `gqa_absorb` gained its rank-4 cond.**
  The runs happened under the working tree carrying what is now
  `90573e5` ("requires rank-4 operands"); no zoo model spells that
  pattern, so every number above is unaffected either way.
- **Method.**  Probe scripts were ephemeral (`/tmp/zoo_probe.py`,
  `/tmp/zoo_pay_attrib.py`); every number above recomputes from the
  committed tree with `catopt_discovery.zoo`, `intake.ingest`,
  `intake._run_delta`, `evidence._guarded_evals` and
  `impact._probe` — no engine internals were modified, and nothing
  was written to `tools/intake_corpus.json` (the holdout stays
  out of the census).

## Files

- `packages/catopt-discovery/src/catopt_discovery/zoo.py` — the
  22-workload holdout registry (`zoo() -> list[Workload]`).
- `tests/test_discovery_zoo.py` — the holdout property (disjointness
  vs registry, `_PURPOSE_BUILT`, model/bench corpora, persisted
  side-file), the fp64 smoke on every workload, and the ingest-
  boundary pin.
- Gates: `ruff check`/`format`, `ty check`, `radon_ratchet`,
  `vulture` clean; `pytest tests/test_discovery_zoo.py` 29 passed.
  Not committed.
