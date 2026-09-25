# ADR 0001 — Weight-space structure hypothesis: falsified

**Status**: decided (negative) — closed.
**Date**: 2025-09 (measured on stories15M + stories110M, llama2.c)

## Context

The motivating extension of catopt: treat the weight file as a
program. If trained weights hid exact structure (duplicates,
subspaces, symmetries, generators), catopt's equivalence machinery
could shrink storage *losslessly* — a genuinely new capability.

## What was measured

Every hypothesis class for exact structure in trained checkpoints:

| probe | hypothesis | result |
|---|---|---|
| Phase 0 | Toeplitz / displacement rank | full rank — dead |
| Phase 0b | H-matrix, sparse, monarch/butterfly, INR manifold | dead or inconclusive; Kronecker-sum beat SVD 15–31% but… |
| Phase 5 | **real perplexity** (TinyStories, baseline 5.03) | structural compression kills quality at every rank (r192/288 → ppl 214); only int8 RTN survives (4×, ppl 5.16) |
| Relational | cross-layer bitwise dups / shared subspace / Procrustes alignment | **0 hits, full column rank, 0 aligned** on both checkpoints |
| Symmetry | equivariance `WG=GW` (cyclic/dihedral/block) | resid ≈ √2 — no symmetry |
| Polynomial | low-degree `p(W)=0` | minimal poly = full degree (288 distinct eigenvalues) |
| Learned displacement | rank(`AW−WB`) over structured A,B | disp-rank ~280/288 — full |

## Decision

**The door is closed.** Trained checkpoint weights are
information-dense at the bit level — no emergent exact structure
exists to exploit. The ε axis (bounded-error rewrites) survives as an
optional toolkit where a norm bound IS the contract (activation
paths, verification), but Phase 5 showed norm bounds don't predict
task quality — approximate structure ≠ quality-preserving.

## Consequences

- catopt's value is **graph-level**: semantic carriers, non-local
  pairing, omd executor, certificates — not weight compression.
- The exact corner that survives is architectural: tying, materialized
  GQA replication, dead params (measured 0–75% by archetype).
- README headline and claims now reflect this scope.
