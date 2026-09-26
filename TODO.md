# TODO

## Deferred

- [ ] `paper/` — a short paper-style write-up of the research claim:
      categorical semantics exposes transforms conventional tensor-level
      optimizers don't reach efficiently. Pull numbers from
      `bench/reassoc_scale.py` (head-to-head vs Inductor at scale) and
      `bench/search_efficiency.py` (alternatives-space vs e-graph size).
      Audience: compilers/PL reviewers. Frame around the fair-comparison
      pipeline (same backend, real checkpoints) — cite REPORT.md on this
      branch for the existing numbers and the falsified directions.
      Not started; blocked on the two benches landing.

- [ ] `plan 0002` leftovers — Phase E recognizer consolidation
      (law-side e-class bindings vs lowerer-side extracted terms —
      substrates genuinely differ, revisit only if a shared kernel
      emerges); plan-dict typing stays dict-of-any.

## Done

- [x] `bench/reassoc_scale.py` — landed. Post-grad FX capture proves
      Inductor keeps k left-assoc mms; CatOpt folds weights into one
      param → 1 mm. 8.93× vs Inductor at k=8, 16.12× at k=16 (CPU).
- [x] `bench/real_linear_attn.py` — landed. RetNet/GLA/delta-rule
      blocks: scan lift + value-chain fold verified fp64-exact;
      retnet 2.8–4.8× vs eager at the closed-form floor. Honest CPU
      negatives vs Inductor (pointwise fusion wins; launch-bound
      devices invert — run --device cuda for the real story).
- [x] `bench/search_efficiency.py` — landed. Saturated e-graph encodes
      all Catalan(k−1) bracketings (verified: 16,796 in 231 e-nodes at
      k=11); bounded saturation ≥10^5.6 programs in ~0.5s at k=24.

- Engineering: monorepo split (5 domain packages), 100% coverage
  line+branch, ruff/pyright clean, 1,585 tests, review pass fixed 5
  audit findings (incl. the omd_lower dense-fiber broadcast bug).
