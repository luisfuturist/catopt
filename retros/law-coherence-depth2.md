# Law coherence, depth 2 — chains, reach, and the mediator table

`law-coherence-catalogue.md` enumerated the pair layer (~44 co-firing
critical pairs, every relation between two laws).  The open question
was whether useful structure exists at depth > 1 — derivation chains
`A ⇒ D ⇒ C`, coherences between coherences — and whether enumeration
still suffices there (~22–26 k triples if done naively).

`tools/law_coherence2.py` measures it, on the same instance-level,
budgeted machinery (`law_verifier.verify_law` — one bounded e-graph
saturation plus a replayable certificate per question, counted as one
oracle call).  Three probes plus honest cost accounting:

Reproduce:

    .venv/bin/python tools/law_coherence2.py            # ~2 min, CPU
    .venv/bin/python tools/law_coherence2.py --with-layout
    .venv/bin/python tools/law_coherence2.py --skip-reach --json d2.json

`--skip-reach` keeps the run torch-free (the corpus probe is the only
torch user — lazy import, model export only).

## 1. The probes

* **Stratification** — a derivable law has *rank 1* when the
  *primitive rules alone* merge its instance sides, *rank k* when
  primitives plus rank-(<k) derivables do.  Laws that never stratify
  are cyclic: every derivation fires another unstratified law
  (inverse-pair twins are the expected shape).  Effective basis =
  `#primitives + #unstratified-SCCs`, checked by grounding: seed each
  cyclic class once and verify every unseeded law derives.
* **Reach under removal** — for each law `L`, saturate a 7-model
  corpus (SwiGLU, ResidualMLP, GatedResidualBlock,
  ManualSoftmaxAttention, MatrixChain, ParallelLinear, NormLinear)
  under `ALL − {L}`; compare e-node/e-class counts to the baseline.
  Derivability is an *equality* verdict; this is the bounded-search
  *reach* verdict — the two need not agree.
* **Mediator table** — for every co-firing pair that fails to rejoin
  under `{A, B}`, the full census: every `C` with `{A, B, C}`
  sufficient (one oracle call per library rule), which candidates are
  *essential* (`universe − {C}` fails to rejoin — pair members are
  candidates too, since a rejoin can re-fire `A` or `B`), and a
  minimal mediating subset greedily reduced from the census or, when
  no single rule suffices, from the found witness.

## 2. Stratification — the derived layer is flat, not deep

The headline measurement of the base universe:

| universe | primitives | rank 1 | rank 2 | unstratified (cyclic) | eff. basis |
|---|---|---|---|---|---|
| `ALL_RULES` (54) | 29 | **0** | **0** | 25 (11 SCCs + 1 downstream) | 40 |
| `+LAYOUT` (131) | 38 | 1 | 2 | 90 (43 SCCs + 2 downstream) | 82 |

**Not one law in `ALL_RULES` is derivable from the primitives alone.**
All 25 "derivable" verdicts are self-supporting cycles — ten
inverse/duplicate pairs plus the 4-member matmul class — and one
downstream singleton (`silu_mul_form`, provable only once the
`{silu_expand, silu_fold}` class is seeded).  Grounding confirms it:
seed one member per SCC (29 + 11 seeds) and **every** unseeded law
re-derives; zero uncovered.  So the "23 rules of redundancy" the pair
catalogue reported is entirely *equivalence-class* redundancy — the
same 2-cell carried twice — plus exactly one emergent law.  There are
no `A ⇒ D ⇒ C` chains inside `ALL_RULES` at all: the derivation graph
has depth 1 (primitives → nothing; cyclic classes → themselves).

Under `--with-layout` the first chains appear:

* `linear_to_matmul_t` is the **only rank-1 derivable law** — the NT
  bridge's "fold" direction is provable from the primitives alone.
* `matmul_transpose_rev_bb` and `matmul_transpose_rev_db` are **rank
  2** — the first measured `A ⇒ D ⇒ C` chains in the library: their
  derivations need `linear_to_matmul_t`, itself only derivable.
* `right_factor_linear` is an anomaly worth its own line: *derivable*
  under `ALL − {self}` but with an **e-graph-dependent witness** — no
  standalone certificate replayed even under the full library, so it
  records no premises at all — and it stays underivable under the
  seeds (the one `uncovered` row).  It is the weakest "derivable" in
  the library: a merge the e-graph sees but no bounded derivation
  carries.

## 3. Reach under removal — derivable ≠ disposable

Removing each law and re-saturating the corpus (7 exports, 5 it /
40 k nodes, EXPANSIVE budgeted):

* `ALL_RULES`: 14 of 54 laws change bounded reach; **6 are
  derivable** — `assoc_matmul_rev` (−8 enodes, MatrixChain),
  `linear_channel_scale`, `linear_row_scale`, `weight_factor_linear`,
  `pow_to_square`, and `silu_expand` (−1 enode / +1 class, a small
  budget-reshuffle anomaly).  All are inverse-pair members: the twin
  proves the *equality*, but only this direction's LHS fires on these
  spellings.  **Derivable-as-equality is not derivable-as-reach** —
  measured justification for shipping both directions.
* `+LAYOUT`: `linear_to_matmul_t` — the rank-1 law — is the **largest
  derivable-law reach cost in the table** (−73 enodes / −24 classes
  across 4 of 7 cases), behind only `comm_mul`/`assoc_mul` overall.
  It is provable in principle and load-bearing in practice: without
  the bridge, the whole linear-level equivalence neighbourhood is
  unreachable within budget.  `linear_from_matmul_t` is *primitive*
  (the de-facto inverse pair is not derivability-symmetric).
* 40 of 54 laws (112 of 131 under layout) are inert on this corpus —
  a 7-model slice, not a claim of uselessness.

## 4. The mediator table — the first triples, and two beyond-triples

`ALL_RULES`: all 7 non-pair-confluent pairs rejoin under **exactly
one** single-rule mediator each, and the mediator is always a pair
member's structural twin — `silu_fold` (2 pairs), `pow_to_square`
(1), `naturality_scalar` (4).  The essential column adds a nuance the
pair layer couldn't see: the rejoin usually needs a **pair member to
re-fire** — `swiglu_fuse` is essential to its own pairs (the re-folded
term is its redex), `square_expand` and the `sdpa_fold_*` member are
essential to theirs.  "The coherence of `A × B` requires `C`" — the
new 3-cell — in practice reads "requires `C` *and* `B` again".

`+LAYOUT`: 40 lib-mediated pairs, zero divergent.  Mediator hubs:

    linear_from_matmul_t (13)   silu_fold (6)   pow_to_square (5)
    naturality_scalar (4)       transpose_{push,pull}_{sub,silu,
                                mul,square}(_bare) twins (2 each)

This refines the catalogue retro's "linear_to_matmul_t mediates 15
coherences": `linear_to_matmul_t` is the common **pair member** of
~15 non-confluent pairs (every `X × linear_to_matmul_t` row), but the
rule that actually rejoins them is its twin `linear_from_matmul_t`
(13 singles) — the "unfold" direction is the mediating rewrite.
Two further patterns: **mediator redundancy** — `silu_expand ×
transpose_push/pull_silu` has *two* independent mediators
(`silu_fold` or the pull/push twin), so `essential` is empty even
though the join is real; and **pair-member re-firing** —
`sub_to_add × transpose_push_sub` lists `sub_to_add` itself as
essential.

And the first structure beyond triples:

    matmul_transpose_rev_db x linear_from_matmul_t_bare
        singles: (none)
        essential: linear_to_matmul_t, matmul_transpose_rev_dd
        minimal+pair: {linear_to_matmul_t, matmul_transpose_rev_dd}
    matmul_transpose_rev_bb x linear_from_matmul_t_bare
        → {linear_to_matmul_t, matmul_transpose_rev_bd}  (likewise)

Two pairs whose reducts rejoin only through a **two-rule derivation**
— no single third rule suffices.  These are genuine 4-rule
coherences, and they were found by enumeration because the witness
(certificate rules) pointed straight at them.

## 5. Where enumeration actually strains

Measured oracle calls and wall-time:

| phase | `ALL_RULES` (54) | `+LAYOUT` (131) |
|---|---|---|
| pair catalogue | 3,002 calls / 0.35 s | 17,392 calls / 1.53 s |
| stratification | 25 / 0.02 s | 275 / 0.17 s |
| grounding | 14 / 0.01 s | 47 / 0.07 s |
| mediator table | 406 / 0.26 s | 5,502 / 1.66 s |
| corpus reach | 385 saturations | 924 saturations |

* Oracle calls cost **~0.1–1 ms each** (law-side instances are tiny;
  every saturation stops at a fixed point).  The feared `C(54,3) ≈
  24.8 k` triple space at the observed per-call cost is **~4–15 s of
  oracle time** — naive triple enumeration of a *derivability-shaped*
  question is not the wall.
* The real cost concentration is elsewhere: the corpus-reach probe
  (385–924 *program* saturations — the run's dominant wall-time, ~1–2
  min) and, conceptually, the pair layer itself.  A triple is only a
  question worth asking when the pair probe already found a shared
  redex and a failed join — **the pair catalogue is the index that
  makes the triple space enumerable**.  Without it, "enumerate all
  triples" has no defined per-triple instance: co-firing-instance
  discovery is a structural search, not a `verify_law` call.
* Scaling honestly: oracle calls grow `O(n²)` in the catalogue
  (direct edges) plus `O(#nonconfluent × n)` in the singles census —
  the layout run's 5.5 k mediator calls came from 40 pairs × 131
  candidates.  Still linear-per-pair, and the measured totals (23 k
  calls, ~3.4 s oracle time, 131 rules) say pair-indexed enumeration
  holds at least an order of magnitude further.

## 6. Sanity

`[PASS]` — the 4-member matmul equivalence class; `silu_fold`
mediating both `silu_expand × swiglu_fuse` and `silu_mul_form ×
swiglu_fuse`; and under `--with-layout` the NT-bridge neighbourhood
(all `X × linear_to_matmul_t` pairs rejoin via `linear_from_matmul_t`
— the refined reading of the "15 coherences" claim, now measured as
13 single-mediator rows).

## 7. Honest limits

* Everything is instance-level and budgeted, exactly as the pair
  catalogue — "unstratified" means *no derivation found from the
  smaller premise set within budget*, not independence in theory.
  The rank-2 laws may have longer primitive-only derivations the 30
  it / 200 k cap doesn't reach; the rank structure is a
  bounded-saturation measurement by design.
* `essential` covers the found witness, the singles census, and the
  pair members — a rule essential only to a *different* derivation
  than any of those is unreported.
* The reach corpus is 7 small exports; inert ≠ useless, and the −1
  enode anomaly under `silu_expand` removal is budget reshuffling at
  noise level, reported rather than explained away.
* `minimal` is greedy, one solution — other minimal mediating sets
  may exist (the `singles` list *is* the full single-rule census; the
  beyond-triple rows give the witness-derived set, not all subsets).

## Verdict

Depth > 1 has real, thin structure — and enumeration suffices to see
all of it.  In `ALL_RULES` the derived layer is *flat*: zero
rank-1 laws, every derivation a self-supporting cycle; the effective
basis is exactly 29 primitives + 11 equivalence classes = **40**, and
the only emergent law (`silu_mul_form`) hangs one step below the silu
class.  The interesting depth-2 objects live at the boundary:
`linear_to_matmul_t` (rank-1 derivable yet the most load-bearing
derivable law under bounded reach), the two rank-2
`matmul_transpose_rev_*` chains it enables, `right_factor_linear`
(the lone derivable-but-unreplayable law), and two pairs that need a
*pair* of mediators.  The cost wall is not oracle calls (sub-ms each;
naive triples ≈ seconds) — it is that triples need the pair layer to
be questions at all.  No learned guide is warranted: the 3-cell layer
is small, sharply-structured, and fully enumerable for at least
another order of magnitude of library growth.
