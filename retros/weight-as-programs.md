# Retrospective — weights-as-programs (falsified)

The hypothesis: trained weights hide compressible structure; treat
the weight file as a program and let catopt search over it.

## What was tried

- Phase 0/0b — intra-matrix algebra probes (Toeplitz displacement,
  H-matrix, sparse, monarch ALS, INR coordinate manifold, Kronecker).
- Phase 5 — real perplexity on stories15M/TinyStories: structural
  compression destroys quality at every rank; only plain int8 RTN
  preserves ppl (4×, 5.16 vs 5.03 baseline).
- Relational probes — cross-layer bitwise duplicates, shared
  subspaces, Procrustes alignment: zero on stories15M + stories110M.
- Symmetry probes — equivariance (WG=GW), polynomial identities,
  learned displacement operators: all full-rank/generic.

## Verdict

Trained checkpoint weights are entropy-dense — no emergent exact
structure exists. Approximate structure (Kronecker, low-rank) exists
but norm bounds don't predict task quality, so it isn't usable for
compression either.

What survived and shipped in catopt: exact architectural structure
(tying, GQA replication, dead params), the certified-approximation
toolkit as an optional module (activation paths / verification —
where norm bounds ARE the contract), and the lesson that read like a
negative but is the actual finding: **the weight file is not a
program worth rewriting; the computation graph is.**

See `adrs/0001-weight-space-structure-falsified.md` for the decision
record and REPORT.md (main branch) §10 for the full measurements.
