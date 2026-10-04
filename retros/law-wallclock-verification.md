# `select_mul` / `softmax_fold` under real wall-clock — does the modeled dispatch-count claim survive?

Every prior "pays" claim for the two shipped machine-discovered laws
was made under the pipeline's *own* cost model —
`executor_cost_for("generic")`'s roofline-plus-per-dispatch proxy —
plus one `law_bench` timing on a *registered synthetic term* per law.
Nothing had timed the real models end-to-end.  `tools/law_wallclock.py`
closes that gap: for each model a law fires on it builds three arms —
the raw `nn.Module`, the pipeline-optimized module
(`Optimizer(backend=TorchBackend())`, composed `default_rules()`), and
a **matched ablation** — the identical pipeline minus the one law —
and times all three on CUDA, eager and under CUDA-graph replay.

**The honest headline: the modeled 17.9–25.9 % `select_mul` drop does
NOT appear as end-to-end wall-clock on the SSM models — the optimized
modules are *slower* than raw there (0.36–0.86×), because the pipeline
lowers them to `BatchedScanModule`, which does more kernel work than
an unrolled 4–8-step eager loop at these dims.  But the law's
*marginal* contribution inside the pipeline — opt vs the
`select_mul`-less ablation, same executor family — is real and
positive almost everywhere.  `softmax_fold` is the clean win: the one
model it fires on is genuinely faster end-to-end.**

## 1. Method

* Device: RTX 2050, fp64 (consistent with every prior measurement in
  the repo — all exports/verifies run `.double()`).
* Per case: `opt.search(model, x, rules=…)` then
  `opt.lower(res, x, verify=True)` — the real pipeline, certificates
  and all; both arms verified `pass`.
* Ablation: `default_rules()` minus the law under test (a `RuleSet`
  `replace`).  `rule_fires` confirms the law fired in the full run
  (4/8/16 firings for `select_mul`, 1 for `softmax_fold`) and never in
  the ablated run.
* Timing: warmup 50, timed 200, per-iteration
  `torch.cuda.synchronize`, median + IQR — two modes:
  * `eager` — the deployed wall-clock (includes the lowered executor's
    Python dispatch; `IdentityRunner` is the default delivery);
  * `graph` — the same call captured into a `torch.cuda.CUDAGraph`
    manually, identically for all three arms; replay removes launch
    overhead, isolating kernel work.
* Sizes: `small` = the dims the impact/shape retros exported;
  `large` = doubled sequence/dims.
* Cost: `dag_cost` of the export root, the ablated extraction, and
  the full extraction under the search's own `cost_fn`.

## 2. Results — `select_mul` (fires on all five SSM models)

Median ms; speedup > 1 is faster.

| case | modeled law-marginal | eager raw | eager opt | eager abl | opt/raw | opt/abl |
|---|---|---|---|---|---|---|
| SelectiveSSM small | −18.9 % | 0.127 | 0.318 | 0.395 | **0.40×** | 1.24× |
| SelectiveSSM large | −21.1 % | 0.229 | 0.421 | 0.580 | **0.54×** | 1.38× |
| DiagDenseSSM small | −21.2 % | 0.114 | 0.280 | 0.342 | **0.41×** | 1.22× |
| DiagDenseSSM large | −23.8 % | 0.195 | 0.414 | 0.572 | **0.47×** | 1.38× |
| DiagonalSSM small | −28.0 % | 0.081 | 0.228 | 0.303 | **0.36×** | 1.33× |
| DiagonalSSM large | −31.9 % | 0.129 | 0.334 | 0.507 | **0.39×** | 1.52× |
| HybridBlock small | −15.9 % | 0.158 | 0.207 | 0.227 | **0.76×** | 1.10× |
| HybridBlock large | −20.3 % | 0.222 | 0.300 | 0.357 | **0.74×** | 1.19× |
| TwoLayerHybrid small | −18.5 % | 0.250 | 0.329 | 0.382 | **0.76×** | 1.16× |
| TwoLayerHybrid large | −22.1 % | 0.341 | 0.513 | 0.619 | **0.67×** | 1.21× |

Same arms under CUDA-graph replay (kernel time only):

| case | graph raw | graph opt | graph abl | opt/raw | opt/abl |
|---|---|---|---|---|---|
| SelectiveSSM small | 0.092 | 0.124 | 0.125 | **0.74×** | 1.01× |
| SelectiveSSM large | 0.161 | 0.186 | 0.244 | **0.86×** | 1.31× |
| DiagDenseSSM small | 0.065 | 0.088 | 0.112 | **0.74×** | 1.28× |
| DiagDenseSSM large | 0.146 | 0.171 | 0.229 | **0.85×** | 1.34× |
| DiagonalSSM small | 0.035 | 0.079 | 0.103 | **0.45×** | 1.31× |
| DiagonalSSM large | 0.049 | 0.134 | 0.193 | **0.37×** | 1.44× |
| HybridBlock small | 0.088 | 0.072 | 0.076 | **1.21×** | 1.05× |
| HybridBlock large | 0.102 | 0.096 | 0.104 | **1.07×** | 1.09× |
| TwoLayerHybrid small | 0.158 | 0.105 | 0.102 | **1.51×** | 0.98× |
| TwoLayerHybrid large | 0.195 | 0.142 | 0.158 | **1.37×** | 1.11× |

* Executor routing splits the family: the three pure SSMs lower to
  `BatchedScanModule`, the two hybrids to generic `IRModule`.
* On the `BatchedScanModule` models the batched-scan formulation is
  slower than the raw unrolled loop *at kernel level* — DiagonalSSM
  large is 2.7× slower (0.134 ms vs 0.049 ms).  The dispatch-proxy
  cost model priced it 32 % *cheaper*.  That inversion is **not** the
  law's fault: the ablated pipeline (still `BatchedScanModule`) is
  slower again — `select_mul` itself measurably *helps* inside the
  executor family the pipeline chose (opt/abl 1.0–1.44×).
* On the `IRModule` hybrids the pipeline genuinely beats raw at
  kernel level (+7…+51 %) — though its Python dispatch overhead eats
  that win and more in eager wall-clock.
* One inversion worth noting: TwoLayerHybrid small under graph has
  the ablated module *faster* than the full one (0.98×) while the
  model says the full extraction is 18.5 % cheaper — a real
  model/measurement disagreement, small but honest.

## 3. Results — `softmax_fold` (fires on ManualSoftmaxAttention)

| case | modeled law-marginal | eager raw | eager opt | eager abl | opt/raw | opt/abl |
|---|---|---|---|---|---|---|
| small | −20.0 % | 0.089 | 0.077 | 0.095 | **1.15×** | 1.23× |
| large | −19.4 % | 0.258 | 0.204 | 0.214 | **1.27×** | 1.05× |

| case | graph raw | graph opt | graph abl | opt/raw | opt/abl |
|---|---|---|---|---|---|
| small | 0.051 | 0.035 | 0.036 | **1.48×** | 1.04× |
| large | 0.244 | 0.189 | 0.196 | **1.29×** | 1.04× |

The manual-softmax fold is the clean win: `div(exp, sum)` → one fused
`softmax` kernel is a real kernel-count and kernel-work reduction, and
it shows up end-to-end — **+13–27 % eager, +29–48 % graph vs raw**.
The law's marginal within the pipeline (opt vs abl) is +4 % at kernel
level — most of the pipeline win comes from other rewrites — and up to
+23 % in eager (one whole `exp+sum+div` chain of dispatches removed).

## 4. The verdict, plainly

* **Modeled cost does not predict whole-pipeline wall-clock vs raw.**
  A −19…−32 % modeled drop coexists with measured −174 %…+51 % vs
  raw.  The dominant term is *executor routing*: the dispatch proxy
  cannot see that `BatchedScanModule`'s formulation costs more kernel
  time than the unrolled loop it replaces at these dims, nor that the
  generic `IRModule` eval pays more Python dispatch than a plain
  `nn.Module.forward`.
* **The laws themselves do pay — inside the pipeline they live in.**
  `select_mul`: opt beats the law-less ablation in 9/10 kernel-level
  and 10/10 eager measurements.  `softmax_fold`: positive marginal in
  all four.  The ablation is the right comparison for "does the law
  help", and it says yes — modestly (1.0–1.4×).
* **Ablating a rule is not a local delta.**  Removing `select_mul`
  changed the extracted term non-locally: on HybridBlock/TwoLayerHybrid
  the ablated extraction is *more* modeled-cost-expensive than the
  unoptimized root — the law unlocks cheaper forms elsewhere, not just
  at its own sites.  The measured opt/abl delta is "the law's marginal
  contribution within this extractor", not "one rewrite's kernel
  delta".
* **`softmax_fold` is the flagship evidence**: a machine-discovered
  law that pays in real wall-clock on a real model, verified by
  certificate — the strongest single number is +48 % graph /
  +27 % eager vs raw at the larger size.

## 5. Caveats

* Tiny fp64 models on a consumer GPU — every arm is launch-bound;
  at production dims/dtypes the balance shifts (batched scans are
  designed for long sequences, and `IRModule` dispatch amortises).
  The claims above are scoped to exactly what was measured.
* `eager` is the honest `IdentityRunner` delivery; a compiled runner
  would compress the Python-overhead story toward the `graph` column.
* The graph arms were captured manually and identically for raw and
  lowered modules — no arm benefits from a different mechanism.

## 6. Gates

* `.venv/bin/ruff check tools/law_wallclock.py` — pass
* `.venv/bin/ruff format --check tools/law_wallclock.py` — pass
* `tools/law_wallclock.py` runs to completion; `--json` writes the
  machine-readable table.
* Full pytest/coverage intentionally not run: measurement task, and
  this host cannot afford two heavy jobs (RAM).
* No `packages/` changes — measurement tool plus this retro only.
