# Law coherence catalogue — the 3-cell layer, made explicit

A rewrite law is a 2-cell `lhs ~[2]~> rhs`; a relation *between* two
laws is a 3-cell (ADR 0002).  The pipeline has always detected these
implicitly — `tools/law_pipeline.py`'s ranker flags proposals as
`duplicate` / `inverse` of shipped rules by alpha-normal key, and
`tools/law_verifier.py` asks whether one law is derivable from the
rest of the library.  What never existed is the *pairwise* catalogue:
for every ordered pair `(A, B)` of shipped laws, does `A` derive `B`?
Do they commute?  Are they the same 2-cell carried by two rules?

`tools/law_coherence.py` builds that catalogue.  Everything is
measured by the shipped machinery — `EGraph` saturation on each law's
concrete instance (`law_verifier.instance_of`, extended with a
generic pattern-driven fallback for the layout family) — never
asserted.

Reproduce: `.venv/bin/python tools/law_coherence.py` (~2 s, CPU-only;
`--with-layout` for the 130-rule universe, `--json PATH` for raw
data).

## 1. The relation tests

* **direct** `A ⇒ B` — the single rule `A` alone merges `B`'s two
  instance sides (`verify_law(inst_B, [A])`).  The strongest 3-cell:
  `B`'s equality is a one-rule consequence of `A`.  Inverse and
  duplicate rules always land here (the e-graph seeds both sides, so
  `A` proves `B` by rewriting the far side); the interesting edges
  are the *composite* ones.
* **derivable** — the library minus `B` proves `B`
  (`law_verifier.rediscover`).  A derivable law adds no basis
  element.
* **essential** — `B` is derivable, but removing *this* rule breaks
  the derivation (`verify_law(inst_B, ALL - {A, B})` fails).  `A` is
  a load-bearing premise, not merely a participant.  Candidates are
  the found witness plus the direct-edge sources — an essential rule
  occurs in *every* derivation, hence in the one already found.
* **confluent** — on a term where both fire, apply `A`-once and
  `B`-once and ask whether the one-step reducts rejoin under `{A,B}`
  (a bounded local-confluence / critical-pair probe), else under the
  whole universe (*library-mediated*).  Pairs with no shared firing
  instance are `no-cofire`, not guessed.
* **equivalent** — same alpha-normal `(lhs, rhs)` (duplicate) or
  mutually direct-derivable both ways: the same equality two ways.

## 2. The catalogue — 53 shipped rules

All 53 rules have a well-typed instance via
`law_verifier.instance_of` (the bench `LAW_CASES` registry plus the
SDPA-fold generic builder); the tool's own `_generic_instance`
fallback is exercised only by the `--with-layout` universe.

| measure | count |
|---|---|
| shipped rules | 53 |
| instanced | 53 |
| derivable from the others | 23 |
| primitive (independent basis elements) | 30 |
| equivalence classes of size > 1 | 10 |
| direct edges `A ⇒ B` | 31 (30 structural + 1 composite) |
| co-firing pairs probed | 44 |
| pair-confluent | 37 |
| library-mediated | 5 |
| divergent | 2 |

**Basis count: 30 independent elements of 53 shipped.**  Every
primitive rule is automatically independent of every other — if `P`
were derivable from the others it would not be primitive — so the
independence number is exact, not bounded.

### Equivalence classes — "the same law two ways"

Ten classes: the nine clean inverse pairs (`pow_to_square`/
`square_to_pow`, `assoc_*`/`_rev`, `naturality_scalar`/`_rev`,
`linear_channel_scale`/`_rev`, `linear_row_scale`/`_rev`,
`weight_factor_linear`/`weight_distribute_linear`,
`right_distribute_matmul`/`right_factor_matmul`) — and one
**four-member class**:

```
{ distribute_matmul_over_add, factor_matmul,
  weight_distribute_matmul, weight_factor_matmul }
```

Two duplicate pairs that are also mutually inverse — the same
equality carried by four rule objects.  This is the only
*accidental* redundancy in the library; the inverse pairs are kept
deliberately (eqsat needs both directions reachable).

### Essential premises

Every inverse-pair member has its twin as its *only* premise — the
structural echo is the whole derivation.  The two interesting
readings:

* `silu_mul_form`: essential premise `silu_expand`.  The library's
  one genuinely-emergent law is a one-premise consequence: expanding
  `silu` inside the `mul` is the entire proof.
* The four-member matmul class: **no** essential premise.  Each
  member has three independent proofs (its duplicate plus the two
  inverses) — maximally redundant.

## 3. Non-obvious derivations

* `silu_expand ⇒ silu_mul_form` — the only *composite* direct edge
  in `ALL_RULES`: a 1-step derivation by a different law applied
  inside a `mul` context (first seen by `law_verifier`; now recorded
  as a pairwise edge, not just a whole-library verdict).
* `linear_from_matmul_t ⇒ linear_to_matmul_t` (visible only under
  `--with-layout`) — a **de-facto inverse the alpha-key cannot
  see**: `linear_to_matmul_t` emits `transpose(W, dim0=-2, dim1=-1)`
  literally, while `linear_from_matmul_t` matches `transpose(W, D0,
  D1)` with attr metavariables.  Structurally the pair is a
  `composite`; the probe catches it because saturation, not the
  key, is the test.

## 4. Confluence — where order matters

Of 44 co-firing pairs, 37 rejoin under the pair alone, 5 need the
library (`square_expand × square_to_pow` rejoins via `pow_to_square`;
`naturality_scalar_rev` against four `sdpa_fold_*mul` spellings
rejoins via its forward twin), and **2 genuinely diverge**:

* `silu_expand × swiglu_fuse`
* `silu_mul_form × swiglu_fuse`

Both are the *same* critical pair, and it is real: `swiglu_fuse`'s
LHS needs `mul (silu …) (linear …)`.  Expanding `silu` inside the
`mul` destroys the fusion redex, and no rule rebuilds a `silu` or
un-fuses a `chunk`/`concat` — the two one-step reducts have no
common reduct even under the whole library.  The library is missing
the un-fuse 2-cell that would mediate this pair.

Read honestly: divergence is **not** a soundness defect.  The
e-graph holds both reducts in one e-class; equality saturation is
confluent by construction.  What the probe marks is where *ordered*
term rewriting is order-sensitive — a real coherence gap precisely
when the search is scheduled as a rewrite sequence rather than a
saturation.

## 5. Cluster skeleton — select_mul / softmax_fold / transpose_push_mul

The neighbourhood is almost disjoint: no direct derivations among
its `ALL_RULES` members, and `softmax_fold`, `select_mul`,
`sdpa_fold_*` are all primitive.  The only internal structure is the
layout push/pull inverse pairs and the commutation coherences —
`comm_mul`/`comm_add` commute with `select_mul`, the `sdpa_fold`s,
and the transpose pushes (all pair-confluent).  The folds are
genuine axioms: nothing derives them and they derive nothing else.

Under `--with-layout` (130 rules) the same run gives 91 derivable /
39 primitive — the 36 push/pull inverse pairs dominate — and
`linear_to_matmul_t` emerges as the **mediator of 15 library
coherences**: the NT bridge's "unfold" direction is what lets every
linear-level rule (`weight_factor_linear`, `assoc_linear*`,
`linear_*_scale`, the fusion laws) rejoin after the transpose is
exposed.

## 6. Honest limits

* **Instance-level, bounded.**  "Derivable" means the two sides of
  *one concrete instance* merged within budget (30 it / 200 k
  nodes); "primitive" means no derivation was *found*.  Every check
  reached a fixed point (no budget truncation observed), but a law
  derivable only at other shapes would read primitive.
* **Confluence is one-step, not a full critical-pair analysis.**  The
  probe takes the first firing of each rule on the other's
  LHS-instance; a pair with other overlap modes could diverge where
  this instance commutes, or vice versa.
* **No undecidable rows in `ALL_RULES`** — all 53 instanced via
  `law_verifier`'s registry + generic fallback.  The
  `_generic_instance` fallback exists for the layout family (all 77
  instanced under `--with-layout`); a rule whose metavars fall
  outside the known leaf/attr tables would report `no-instance`
  rather than a guessed term.
* **`essential` is per-found-derivation**: candidates are the
  witnessed rules plus direct-edge sources.  An essential rule must
  occur in every derivation, hence in the found one — the
  restriction is sound, but a rule essential only to a *different*
  derivation than the found one is still found, since it is tested.

## Verdict

The shipped library's coherence structure is: **30 independent
basis elements**, 18 rules in nine direction-pairs kept by design,
4 rules of
accidental redundancy (the matmul distribute/factor class, all
provably the same 2-cell), exactly one emergent law
(`silu_mul_form`), one alpha-invisible inverse
(`linear_from_matmul_t`/`linear_to_matmul_t`), and exactly one
missing coherence — the un-fuse rule that would mediate
`silu_expand × swiglu_fuse`.  The 3-cell layer is now enumerable,
and the answer to "does the library have redundancy?" is *yes, but
almost none of it is accidental*.
