# Axiom/lemma split — the minimal kernel, annotated not deleted

The coherence retros (`law-coherence-catalogue.md`,
`law-coherence-depth2.md`) measured the shipped library's structure:
54 laws, 29 primitives, 11 derivability cycles — an effective basis
of 40 — but nothing in the *code* said which rules are the kernel
and which are carried derivations.  This change makes the
distinction a field: every shipped rule now answers
``rule.kind ∈ {"axiom", "lemma", "redundant"}`` and every non-axiom
carries ``rule.derivation`` — the name(s) of the axiom(s) that prove
it.

Reproduce:

    .venv/bin/python tools/law_coherence.py --emit-basis   # ~30 s, CPU
    .venv/bin/python -m pytest tests/test_laws_structure.py -q

## 1. The mechanism

* ``Rewrite.derivation: tuple[str, ...] = ()``
  (`egraph/types.py`) — the names of shipped rules forming ONE
  recorded derivation of the rule's instance.  Every current entry
  is a *single* premise — a measured direct edge `{A} ⇒ rule` — so
  each lemma carries a one-step proof from the kernel.  Empty =
  designated kernel member.
* ``Rewrite.kind`` (property, derived not stored) — ``"axiom"`` when
  ``derivation`` is empty; ``"redundant"`` when it is non-empty and
  the rule also carries the new ``tags.REDUNDANT`` tag; else
  ``"lemma"``.  One field of truth (the premise tuple), one tag for
  the "same 2-cell twice" case the premise list alone cannot
  express.
* ``laws.base.R(..., derivation=...)`` forwards it.  Rules minted at
  runtime (`meta.synthesize_rules`, the witness-pair factory in
  `egraph/core.py`, `rulecache` reconstructions) default to
  `()` — they are not shipped-library members and keep their own
  provenance (`parents` / `guard_pats`), so the default reads
  "unmeasured", never "kernel".
* `tools/law_coherence.py --emit-basis` computes the whole table
  from the measured catalogue: primitives plus the
  alphabetically-first member of each equivalence class are axioms;
  a non-seed class member is `redundant` when an *earlier* classmate
  is already its alpha-duplicate, else `lemma`; `derivation` is the
  lexicographically-first axiom with a measured direct edge, else
  the found witness verbatim.  The table is also in `--json` output
  under `"basis"`.  The shipped annotations were generated from this
  flag, not hand-picked — a diff-checking run reports zero
  mismatches.

## 2. The measured table (fresh catalogue run, post-`silu_fold`)

| kind | count | members |
|---|---|---|
| axiom | 40 | 29 primitives + 11 cycle seeds |
| lemma | 12 | 10 inverse twins + `factor_matmul` + `silu_mul_form` |
| redundant | 2 | `weight_distribute_matmul`, `weight_factor_matmul` |

The 11 derivability cycles (each seeds its alphabetical-first
member): the ten inverse pairs `{assoc_linear*`, `assoc_matmul*`,
`linear_*_scale*`, `naturality_scalar*`, `pow_to_square`,
`right_*_matmul`, `silu_expand/fold`, `weight_*_linear}` plus the
4-member matmul class `{distribute_matmul_over_add, factor_matmul,
weight_distribute_matmul, weight_factor_matmul}`.

Every lemma's `derivation` names a **kernel axiom** — the premise
never routes through another lemma (`silu_mul_form ← silu_expand`,
each inverse twin ← its class seed, `factor_matmul ←
distribute_matmul_over_add`).  That was not forced: it is what the
direct-edge data showed.  `emit_basis` falls back to the found
witness if no axiom proves a rule directly, so a future law whose
proof needs a lemma will say so honestly.

Two judgement calls baked into `emit_basis`, documented rather than
measured:

* **Seed choice is a convention.**  `{silu_expand, silu_fold}`
  derives each other; calling `silu_expand` the axiom is the
  alphabetical tie-break the grounding check already used, not a
  theorem.  (It happens to match intuition — expand before fold,
  distribute before factor — everywhere except
  `weight_distribute_linear` seeding over `weight_factor_linear`.)
* **Inverse twins are lemmas, not redundant.**  A twin covers the
  *opposite* direction — the reach probe showed removals cost real
  enodes (`linear_to_matmul_t` −73; six ALL_RULES lemmas change
  bounded reach).  Only the two alpha-*duplicate* spellings —
  literally the same pattern under metavar renaming, same direction,
  same fires — are marked `redundant`.

## 3. Minimal kernel vs shipped set

The shipped `ALL_RULES` stays 54 — this is annotation, not deletion
(direction matters for firing; both members of an inverse pair are
what eqsat explores).  The *minimal kernel* the marks describe is
the 40 axioms: every lemma and redundant rule re-derives from it
(verified by the depth-2 grounding check — seed each class once and
all 14 unseeded laws re-derive within budget).

What the split buys: a defensible answer to "what is the core theory"
(40 axioms — the monoid/group laws, the bilinearity directions kept
as seeds, the product folds, the sdpa/silu/softmax kernel
recognitions); a record on each lemma of *why it needs nothing new*
(`derivation` = the premise); and a marked boundary for the two
rules the catalogue calls accidental redundancy.

## 4. The certificate upgrade path (report-only)

`derivation=("silu_expand",)` is currently a *name* — the measured
claim "`silu_expand` alone merges silu_mul_form's instance sides".
The emit machinery could turn it into an artifact: `law_verifier`
already returns replayable witnesses and `egraph/certs.py` already
has `Derivation`/`Certificate` — a proof-carrying derivation with
per-step rule+path records that `verify_certificate` replays.
Emitting `derivation` as a replayed `Derivation` (steps, not just
premise names) would make each lemma a certified theorem of the
kernel — and would catch drift: a lemma whose recorded proof stops
replaying after a library edit is a real finding, not a stale
comment.  Deliberately not implemented: the premise-name marker is
the honest cheap version, and instance-level certificates inherit
the catalogue's limits.

## 5. Honest limits

* Everything inherits the coherence probe's bounds: "axiom" means
  *no derivation found within budget on the law's instance*, not
  independence in theory; "lemma" means one measured derivation
  exists.
* The marks cover `ALL_RULES` (54).  `LAYOUT_RULES`, `ATTENTION`,
  `SCAN_*` and the carrier families are unmarked — `kind` reads
  "axiom" there only because the default is empty-derivation.  The
  layout universe is where derivable-from-primitives actually occurs
  (`linear_to_matmul_t` is rank-1); annotating it is a follow-up.
* `kind` is a property over (`derivation`, `REDUNDANT`-in-tags), so
  a rule tagged `REDUNDANT` with an empty derivation still reads
  "axiom" — the tag has no independent force; the structure test
  pins the intended correlation.
