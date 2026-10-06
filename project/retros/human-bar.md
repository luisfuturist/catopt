# Retro — the human bar: how much of the zoo yield is machine-admitted?

Date: 2026-10-06
Context: `project/retros/model-zoo-yield.md` measured the shipped
library at **5/22** verified wins on the held-out model zoo, and
`project/retros/promoted-laws.md` shipped 10 machine-admitted objects
into `ALL_RULES` (61 → 71).  The open question this retro answers:
**how much of that yield is machine-admitted vs human-authored — and
could a machine-built ruleset match the human one?**  The measurement
is an ablation, not an annotation read: four ruleset arms over the
same 22 workloads, same `Optimizer(backend=TorchBackend())` pipeline
(`search` → `lower`, fp64 verify at rtol 1e-4), dag-costed exactly as
the zoo retro priced it (the in/out numbers reproduce its table
cell-for-cell).

## The provenance mechanism

`catopt_core.laws.provenance` (new, registry-level — no law bodies
touched): `MACHINE_SHIPPED` names the ten promoted laws as they sit in
`ALL_RULES`; `MACHINE_STORED` names the gauntlet-cleared store objects
that stayed unshipped; `law_provenance(name_or_rule) -> "human" |
"machine"` is total.  Two collisions the registry states honestly:
`om_lift` names *both* a human carrier law
(`catopt_carriers.om.OM_LIFT`, metavar-dim guard) *and* a constructed
store object (literal `dim=-1`); and `softsign_fold` /
`sdpa_fold_{,div_}nomask` exist as shipped laws *and* as same-named
store spellings.  The registry resolves names; arm-level attribution
below compares **rule objects** (`id()` against the actual arm
membership), so the collision cannot launder a human law into machine
credit.

The committed `tools/evidence.db` holds no `lemmas` rows — admission
runs used ephemeral stores — so the store arm rebuilds the recorded
objects verbatim: the eight guarded view strips
(`{mul,sub,add}_unsqueeze_l_id`, `mul_unsqueeze_r_id`,
`{mul,add}_transpose_l_id`, `mul_chunk_l_id`, `mul_reshape_l_id`), the
four `fold_object` folds (`softsign_fold`, `sdpa_fold_nomask`,
`sdpa_fold_div_nomask`, `sdpa_fold_div_nomask_g`), and the seven
`lift_object`/`compose_objects` products (`om_lift`, `aff_step_lift`,
`aff_scan2_lift`, `affd_step_lift`, `affd_scan2_lift`,
`affd_scan4_lift`, `silu_fold_commuted`, `channel_then_row_scale`) —
20 objects, every one pinned `usable` by a committed gauntlet test or
the promotion audit.

## The four arms

| arm | contents | size |
|---|---|---|
| `full` | `DEFAULT_RULES` (core DEFAULT + carriers) | 59 rules |
| `human` | `full − MACHINE_SHIPPED` | 49 rules |
| `machine` | promoted laws + store objects (store spelling wins collisions) | 10 + 20 |
| `store` | the 20 store objects alone | 20 |

Per-model result — `Δ%` is the verified extraction's dag-cost delta
(`v` = verified, `!` = term changed, `—` = no fire, `x` = unlowerable):

| zoo model | full | human | machine | store |
|---|---|---|---|---|
| GatedAttentionUnit | **−0.003 v!** | **−0.003 v!** | +0.000 v | +0.000 v |
| AdditiveAttention | — | — | — | — |
| TalkingHeadsAttention | — | — | — | — |
| CosineAttention | **−0.006 v!** | **−0.006 v!** | **−0.006 v!** | **−0.006 v!** |
| FiLMHead | — | — | — | — |
| AdaLNBlock | +0.000 v | +0.000 v | **−4.070 v!** | **−4.070 v!** |
| LoRAAdapter | **−19.985 v!** | **−19.985 v!** | +0.000 v | +0.000 v |
| MLPMixerBlock | 0.000 v!* | 0.000 v!* | 0.000 v!* | 0.000 v!* |
| FNOBlock | x (`fft_rfft2` unbound) | x | x | x |
| GCNLayer | — | — | — | — |
| AffineCoupling | — | — | — | — |
| SoftSlotMoE | — | — | 0.000 v! | 0.000 v! |
| ExpertChoiceMoE | — | — | — | — |
| CapsuleRouting | 0.000 v | 0.000 v | — | — |
| ChunkedRetention | 0.000 v | 0.000 v | — | — |
| DeepEquilibrium | — | — | — | — |
| WaveNetGate | — | — | — | — |
| TimeCondConv | **−4.996 v!** | +0.000 v | **−4.996 v!** | **−4.996 v!** |
| MoSHead | 0.000 v | 0.000 v | — | — |
| CovarianceHead | 0.000 v | 0.000 v | — | — |
| SplineKAN | **−0.003 v!** | **−0.003 v!** | +0.000 v | +0.000 v |
| HighwayGate | — | — | — | — |

`*` `MLPMixerBlock`'s changed term is the nonlocal param-dedup lift —
a pipeline pass, not a law; identical across arms.

**Score: full 5/22 · human 4/22 · machine 3/22 · store 3/22** — and a
fifth arm (`full + store`, one run) wins **6/22**: the union is
strictly better than either side because `AdaLNBlock` only enters
through the store.

## Per-law attribution on the full library

| law | provenance | fired on | delivered? |
|---|---|---|---|
| `assoc_linear` | human | LoRAAdapter | **−19.985 %** (LoRA fold) |
| `mul_unsq_pad_l` + `mul_unsq_pad_r` | machine | TimeCondConv | **−4.996 %** (dead broadcast pads) |
| `om_lift` (carrier `OM_LIFT`) | human | CosineAttention | −0.006 % |
| `mul_square` + `square_to_pow` | human | GAU, SplineKAN (+CapsuleRouting via `square_expand`) | −0.003 % ×2 |
| `sub_to_add` | human | ChunkedRetention, CovarianceHead | cost-neutral |
| `index_select_dedup` + `index_select_id` | human | MoSHead | cost-neutral |
| `silu_expand` | human | TimeCondConv | fired, not load-bearing (machine arm wins without it) |

So of the library's five wins: **one is machine-admitted yield**
(`TimeCondConv`, the promoted pad strips) and **four are
human-authored** — including the only structurally large one
(`assoc_linear`, −20 %).  The machine half of `DEFAULT_RULES` fires on
exactly one held-out model and it pays.

## The machine ruleset alone — and where it beats the library

The machine arm (promoted + store) fires on 4 models and wins 3:

- `CosineAttention` −0.006 % — the *store* `om_lift` reproduces the
  carrier law's lift (both directions verified exact);
- `TimeCondConv` −4.996 % — `mul_unsq_pad_{l,r}` plus the store's
  `mul_unsqueeze_r_id`; the pure-store arm matches it with just
  `mul_unsqueeze_{l,r}_id`;
- **`AdaLNBlock` −4.070 % — a win `DEFAULT_RULES` misses entirely.**
  `affd_step_lift` folds `ln(x)·(1+scale) + shift` — DiT's AdaLN
  modulation — into `applyd(aff_diag(ln(x), shift), 1+scale)`; the
  generic-lowered module verifies exactly (max_rel 0.0).  The store
  object is alpha-equal to the shipped `affd_lift`, but that law
  lives in `SCAN_DIAG_LAWS`, an opt-in regime set the default never
  runs.

The honest caveats on the AdaLN win: the delivered term routes
*generic* (a single `applyd` has no batched-spine depth to exploit),
so −4.07 % is the executor model's op pricing — two elementwise ops
priced above one carrier pair — not a measured wall-clock claim.
Running the human opt-in set (`DEFAULT + SCAN_DIAG_LAWS`) does reach
the site — `affd_lift`, `affd_lift_unit`, `affd_unlift` all fire and
the carrier upgrade ships a *batched* spine — but under the uniform
generic-model accounting that extraction reads **+7.9 %**.  The
machine object wins the site in the regime the default actually
ships; the human library owns the same capability behind an opt-in.

Also note `SoftSlotMoE`: the *store* `om_lift` fires and routes the
term `batched` where the human `OM_LIFT` never fired — the literal
`dim=-1` pattern matched a site the metavar-dim guard declined.
Cost-neutral, verified, and a real coverage datum: the machine's
narrower spelling is not always weaker.

## The gap analysis — where machine loses to human

The machine repertoire misses every win in the *algebraic composition*
family:

| human-only win | law family | machine coverage gap |
|---|---|---|
| LoRAAdapter −20 % | `assoc_linear` — linear/matmul chain reassociation | no associativity or chain-composition object was ever admitted |
| GAU, SplineKAN −0.003 % | `mul_square`/`square_to_pow`/`square_expand` — the `x·x`/`x²`/`pow` spelling bridge | the store holds no elementwise self-product fold |
| MoSHead (fires, neutral) | `index_select_*` gather dedup | no gather/dedup family |

And every *silent* model (10 of 22) is silent for both arms — neither
side covers attention-score softmax families, routing, or recurrence
blocks on this holdout.

The machine side's distinctive coverage is exactly the families the
humans left thin in `DEFAULT`: the auto-cond view strips (pad/no-op
view identities) and the constructed carrier lifts (scan `applyd`,
`om_apply`).  On this zoo that trades 4 human wins for 3 machine wins
with only `CosineAttention` overlapping — the arm delta, not the name
count, is the measure.

## The human-bar verdict

1. **Yes, a machine-built ruleset alone wins real yield on held-out
   code** — 3/22 verified, including the zoo's second-largest drop
   (TimeCondConv −5 %) and one win the shipped default cannot produce
   (AdaLNBlock −4.07 %, an admitted-but-unshipped store object).
2. **No, it does not match the human set** — 3 vs 4 wins, and the
   human arm owns the biggest single win (LoRA −20 %).  The gap is
   coverage, not soundness: nothing the machine arm extracted failed
   verification anywhere (21/21 lowerable verified in every arm).
3. **The arms are complementary, not nested** — `full + store` wins
   6/22.  The machine repertoire extends the library (carrier lifts,
   the wider `*_id` strip spellings) more than it duplicates it; the
   promotion step added review and durability, not reach — the pure
   store arm reproduces every promoted-law win verbatim.
4. **The actionable residue**: `affd_step_lift`'s AdaLN win argues for
   either shipping `affd_lift` into `DEFAULT` (it is already shipped,
   just opt-in) or promoting the carrier lifts the store already
   holds — the yield was sitting admitted-but-unused.

## Caveats / limits

- **Cost-model fidelity, same caveat as the zoo retro**: `Δ%` is the
  executor-aware dag-cost model, not wall-clock; the AdaLN drop is a
  model-priced op-count win under generic delivery.
- **Store reconstruction**: the committed evidence.db `lemmas` table
  is empty; the store arm re-spells the 20 objects from the committed
  test constructors (`tests/test_discovery_{synthesis,gauntlet,
  autocond}.py`) and the promotion audit — the same code paths that
  produced the recorded `usable` verdicts.
- **Name-vs-object provenance**: `om_lift` collides (human carrier
  law / machine store object); the attribution above keys on rule
  *object* identity per arm.  `law_provenance` intentionally resolves
  the shipped name to `human` — see the module docstring.
- **Untrained weights**, same as the zoo retro; the wins that matter
  (`assoc_linear`, the pad strips, the affd lift) are structural.
- **Probe**: `/tmp/human_bar_probe.py` (ephemeral, per the project's
  probe convention); `dag_cost(res.term, res.cost_fn)` per arm; the
  ablation arms and store constructors are recomputable from
  `tests/test_law_provenance.py` (`_store_strips`, `_affd_step_lift`,
  `_promoted_ruleset`).

## Files

- `packages/catopt-core/src/catopt_core/laws/provenance.py` —
  `MACHINE_SHIPPED` / `MACHINE_STORED` / `MACHINE_LAWS` +
  `law_provenance`; re-exported from `catopt_core.laws`.
- `tests/test_law_provenance.py` — registry totality, the promotion
  table pin, arm partitioning, and the four behavioral ablation pins
  (TimeCondConv both directions, AdaLNBlock exclusivity, LoRAAdapter
  human-only).
- Gates: `pytest tests/test_law_provenance.py` 10 passed (3 s);
  ruff/ty/vulture/radon clean.  Not committed.
