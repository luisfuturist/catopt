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

## Active

- [ ] `bench/reassoc_scale.py` — head-to-head: deep matmul/linear-attn
      reassoc chain CatOpt finds, Inductor cannot express (post-grad
      graph capture as non-reachability evidence). In flight.
- [ ] `bench/search_efficiency.py` — alternatives space vs e-graph
      size + time-to-saturation scaling. In flight.

## Done

- Engineering: monorepo split (5 domain packages), 100% coverage
  line+branch, ruff/pyright clean, 1,585 tests, review pass fixed 5
  audit findings (incl. the omd_lower dense-fiber broadcast bug).
