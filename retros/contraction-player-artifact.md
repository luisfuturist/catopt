# The contraction player ships — bundled weights + a loader

`contraction-train-scale.md` settled the training regime (the
**curriculum**: scales 8, 12, 16, 20, 24, REINFORCE with the
greedy-completion critic, `--iterations 1800`, the einsum-valid bond
family), but every experiment still retrained from scratch — 4–6
minutes before a single board could be played.  This retro closes the
packaging gap: the player is now a **shipped artifact** —
`catopt_torch.contraction_policy` carries the game, the net, a usable
player wrapper, a lazy loader, and the pretrained curriculum weights
bundled as package data.

## 1. What shipped

* `packages/catopt-torch/src/catopt_torch/contraction_policy.py` — the
  contraction player, moved verbatim out of `tools/` (one canonical
  implementation): `ContractionGame` + the pinned scale-free features
  (`STATE_DIM` 7 + `PAIR_DIM` 9), `PairPolicyNet`, the lockstep rollout
  drivers (`run_policy_batch`, `rollout_orders`), the cost primitives
  (`pair_cost`, `greedy`, `cost_of_order`), the
  `random_bond_network` training family, the `ContractionPolicy`
  player (`order` — deterministic single pass; `sample_orders` /
  `best_order` — seeded sampled rollouts), and
  `save_contraction_policy` / `load_contraction_policy`.
* `packages/catopt-torch/src/catopt_torch/artifacts/
  contraction_policy_curriculum.pt` — **25 KiB**, `torch.save` of
  `{format, state_dict, meta}`: hidden 64, the `scale-free-v1` feature
  contract, trainer `rl`, scales `[8,12,16,20,24]`, iterations 1800,
  batch 24, seed 0, family `random_bond_network`, git sha `4311e457`,
  torch `2.14.0+cu130`.  The loader uses `weights_only=True` and
  rejects wrong-format / feature-drifted payloads with `ValueError`.
* `tools/train_contraction_artifact.py` — the reproducible trainer:
  curriculum regime, `_on_family` patch (no opt_einsum needed —
  training needs only the generator), save, reload-through-the-shipped-
  loader, and a held-out sanity table.
* `tests/test_contraction_policy_artifact.py` — 33 tests: artifact
  loads (default + custom path), valid orders, seed determinism,
  error paths, and the quality bar (beats our greedy on held-out
  boards).  Scoped coverage of the new module: **100 %** (355 stmts,
  60 branches).
* Wiring: `catopt_torch.__init__` lazy surface gains
  `ContractionPolicy` / `load_contraction_policy`; `catopt-torch`
  declares `numpy>=1.24` (already needed at runtime by `export.py`'s
  `Tensor.numpy()` calls; the game's feature tables are NumPy) and
  `package-data = artifacts/*.pt`; `.gitignore` gets a narrow
  `!artifacts/*.pt` exception (first bundled-weights precedent);
  `uv.lock` re-resolved.
* Tools refactor: `tools/contraction_policy.py` now imports the game /
  net / rollout machinery from the package (the old names stay as
  aliases, so `contraction_einsum`, `contraction_scale_probe`,
  `contraction_az`, `contraction_value_probe` are untouched);
  `contraction_einsum.py` aliases `random_bond_network` and delegates
  `_policy_rollouts` to `rollout_orders`.

## 2. How to load

```python
from catopt_torch import load_contraction_policy

policy = load_contraction_policy()          # bundled weights, CPU
order  = policy.order(tensors, sizes)       # deterministic single pass
cost   = policy.cost(tensors, sizes)
best   = policy.best_order(tensors, sizes, samples=64, seed=0)
```

`path=` loads a user checkpoint written by
`save_contraction_policy` / `tools/train_contraction_artifact.py`;
`device="cuda"` gives the throughput the experiments measured.
Nothing loads at import — the `torch.load` happens on call.

Reproduce the weights:

    .venv/bin/python tools/train_contraction_artifact.py   # ~265 s, CUDA

## 3. Capability statement (honest)

Trained on the `random_bond_network` einsum-valid family only.
Measured by `contraction-train-scale.md`: at n = 40 the single-pass
policy **ties, does not beat**, best-of-N randomised greedy
(~1.04–1.5× at ≥200 ms budgets), **beats the deterministic external
greedy by ~25–30 %**, has no small-board regression, degrades
gracefully when starved, and is **throughput-starved under ~50 ms
budgets** (a forward pass per decision).  Other tensor-network
distributions are out of scope.

Post-ship sanity (this run, held-out seeds, deterministic rollout /
our greedy, 8 boards): n = 12 → 0.253, n = 16 → 0.222, n = 20 →
0.164.

## 4. Boundary — what this is *not*

A **research artifact behind a documented API**, not a pipeline wire.
`Optimizer.optimize()` has no contraction-ordering hook for this
player (the contraction diagram search in
`catopt_orchestrator.diagram` is a different game — `ReorderCompose`
moves on a lifted diagram, not board play on `(tensors, sizes)`).
The wall-clock-budgeted anytime player
(`contraction_einsum.policy_best_order`) stays in tools: equal-budget
measurement is an experiment protocol, not an API.

## 5. Notes

* `random_bond_network` was split into helpers to stay under the
  radon threshold; the refactor is **verified bit-identical** to the
  tools version it replaces (7 scales × 20 seeds) — same RNG call
  order, same boards, same training distribution.
* `torch.__version__` is a `TorchVersion` (a `str` subclass) and is
  **rejected by `weights_only=True` loads** — the meta stores
  `str(torch.__version__)`.
* The tools-side re-export aliases keep `cp.*`/`ce.*` working, so the
  published ladders rerun unchanged.

## Gates

* `.venv/bin/ruff check` / `ruff format --check` on all touched files — pass
* `.venv/bin/ty check` (whole `packages`) — 0 errors
* New test file standalone: **33/33 pass** (~0.2 s)
* Scoped coverage (`--include=contraction_policy.py`): **100 %**
* `.venv/bin/vulture`, `lint-imports` (4 contracts kept), `bandit`,
  `semgrep --config .semgrep.yml` — clean
* `tools/radon_ratchet.py --update` regenerated the baseline: 43 new
  entries for the module plus the committed `meta_eval` fix (7→8)
* `uv lock` — resolved (numpy declared for catopt-torch)
* Training: CUDA, 265.2 s, `p_train_artifact` converged; artifact
  25 601 bytes; reload-through-loader sanity table above
* NOT run: full `pytest` / repo-wide `coverage` (11 GB machine)
