# bench/

Systematic, presentable measurement of the catopt engine.  This is a
package, not a pile of scripts: one canonical `Report` per suite, one
CLI, and several renderers off the same data.

```bash
python -m bench list                        # the catalog
python -m bench run reassoc_scale --quick   # one suite
python -m bench run-all --quick             # every harnessed suite
python -m bench dashboard                   # cross-suite HTML index
python -m bench report bench/results/reassoc_scale.json  # re-render
python -m bench compare reassoc_scale       # vs the pinned baseline
```

Run any suite directly with its own typed flags:

```bash
python -m bench.suites.speedup.reassoc_scale --depths 4,8 --rows 16384
python -m bench.suites.correctness.law_bench --laws assoc --sizes 256
```

`--quick` shrinks a sweep (the per-suite `QUICK` dict, applied by the
CLI) and shortens the timing window.

## Layout

```
bench/
  __main__.py  cli.py  registry.py     # entry point, subcommands, catalog
  benchkit/                            # the harness
    model.py      # Variant, Case, Cell, Finding, Verdict
    runner.py     # torch Timer medians + IQR (the only clock)
    report.py     # Report: cells + findings + provenance + renderers
    stats.py      # formatting + polars flattening
    env.py        # provenance (torch/python/git/device)
    ledger.py     # append-only run ledger + baselines
    compare.py    # regression + expectation comparison
    render/       # markdown · html · plots · quarto · slidev · dashboard · catalog
    templates/    # jinja2 sources for the HTML surfaces
  suites/{correctness,search,cost,structure,speedup,e2e,bounded,integration}/
  common/                              # llama2c.py loader, fetch.py
  baselines/                           # committed golden results
  results/                             # working output (gitignored)
```

Suites live under `suites/<intent>/` (the directory is the intent)
and expose `run_bench(args) -> Report`.  They never time anything
themselves — they describe `Case`/`Variant` data and hand it to
`Runner`.

## The Report model

A suite states its conclusion as typed `Finding`s, not console text:

```python
Finding(
    claim="catopt reaches a weight-folded form Inductor cannot express",
    verdict=Verdict.WIN,  # win|parity|regression|negative|inconclusive
    headline="N× vs Inductor at (k,d,B·T)=(…)",  # measured, not hardcoded
    metric="catopt+inductor / inductor",
    value=...,  # from the timed cells
    evidence={"inductor_mm": 8, "catopt_mm": 1},
)
```

`Report` also carries a `title`, a `summary`, and the environment
provenance (git sha, device, versions, argv).  Raw per-cell diagnostics
live in `Cell.aux` and are rendered as collapsible pretty JSON — never
a truncated `repr`.

## Outputs

Every surface is rendered from the same `Report`, so numbers cannot
disagree between formats:

| Surface | Command flag / renderer | Use |
|---|---|---|
| JSON | always | machine-readable, schema-versioned (`benchkit/report.py`) |
| Markdown | always | GitHub tables, findings, aux in `<details>` |
| HTML | `--html` | single-file dashboard per suite (plotly embedded) |
| plots | `--plots` | plotly `.html` (interactive) + `.svg` (static) |
| Quarto | `--quarto` | `.qmd` → PDF/HTML (`quarto render …`) |
| Slidev | `--slidev` | assets + a slide fragment for `pitch/liquid` |
| dashboard | `python -m bench dashboard` | cross-suite `index.html` |

Plots use plotly (interactive + `kaleido` static export); the ledger
uses polars; the CLI is tyro (dataclass → typed flags); console output
is rich.

## Checkpoints

```bash
python -m bench.common.fetch             # both models, ~500 MB
python -m bench.common.fetch --models 15M
```

Downloads `stories15M.bin` / `stories110M.bin` into
`$XDG_CACHE_HOME/catopt` (default `~/.cache/catopt`), verifies the
7-int32 header and size, and skips valid copies.  Resolution order:
`--ckpt <path>` → `~/.cache/catopt/<name>` → `/tmp/<name>` → error.

## Adding a suite

1. Add a module under `suites/<intent>/` exposing
   `run_bench(args) -> Report` and (optionally) `QUICK = {...}`.
2. Register it in `bench/registry.py` with a `SuiteSpec` (intent,
   question, expected verdict).
3. Add a builder in `LAW_CASES` if it is a law bench.
4. Regenerate the catalog: `python -m bench catalog --write`.

Gates: `uv run pytest` (the harness has `tests/test_benchkit.py`),
`.venv/bin/ruff check packages tools bench`, and
`.venv/bin/ruff format --check packages tools bench`.  `bench/` is
linted and formatted but **not** type-checked (`[tool.ty.src] include`
is `packages` only) and sits outside the 100% coverage floor — it is
measurement code, not a shipped API.

## Suite catalog

Generated from `bench/registry.py` — the registry is the single
source of truth for each suite's intent, question and expected
verdict.  `python -m bench catalog --write` regenerates the block
below; `python -m bench catalog --check` fails if it drifts (and
`tests/test_benchkit.py` enforces the same in CI).

<!-- BEGIN GENERATED CATALOG -->
### correctness

| suite | tier | question | expected |
|---|---|---|---|
| `law_bench` | micro | Does each registered rewrite law fire, get picked by extraction, and lower to a verified term? | WIN — every registered law fires/picks/verifies; non-firing laws report honestly. |
| `laws_effect` | block | Do the law families pay off at runtime on realistic model families? | WIN on launch-bound families; honest negatives where Inductor's fused pointwise kernel wins on CPU. |
| `morphism_coverage` | model | Which morphism laws match and fire on real checkpoints? | Coverage map — matches/fires per law, declines with reasons. |

### search

| suite | tier | question | expected |
|---|---|---|---|
| `search_efficiency` | micro | How expensive is saturation relative to the program space it represents? | WIN — the e-graph encodes an exponential (Catalan) program space in polynomially many live e-nodes; exact saturation fragments at large k (honest limit). |

### cost

| suite | tier | question | expected |
|---|---|---|---|
| `cost_fidelity` | micro | Does the pipeline cost model rank candidates like measured latency? | High rank correlation (ρ) and pick accuracy — term-level cost tracks the backend. |

### evaluation

| suite | tier | question | expected |
|---|---|---|---|
| `policy_value` | micro | Does a learned search policy pick better rules than random, declaration-order, or a cost-model greedy? | NEGATIVE vs the cost-model greedy — the learned policy ranks rules better than random/declaration-order but does not match the evaluator greedy (associativity-direction confusion). |
| `eval_axis` | micro | Does plugging a different evaluator in change the program the engine extracts, and does the equivalence class expose a genuine multi-axis trade-off? | NEGATIVE on all three: an accurate profiler kills the bandwidth story (the bandwidth pick never differs from the launch pick), the root-class frontier yields ties and float noise only, and the residual target-sensitivity is the additive marginal decomposition of the non-additive PredictedCriterion — the true roofline value ranks the same form first under every target (7/7). |

### structure

| suite | tier | question | expected |
|---|---|---|---|
| `structure_census` _(ad-hoc)_ | model | How much catopt-exploitable structure do real trained weights carry? | Exact mode: ~zero bitwise structure on dense LLMs, real on structured ones. |
| `bound_amplification` _(ad-hoc)_ | block | How does a weight-space error bound propagate to the output? | Measured output error exceeds the certified bound — the bound is conservative (honest). |

### speedup

| suite | tier | question | expected |
|---|---|---|---|
| `reassoc_scale` | block | Can the e-graph find a form Inductor's post-grad graph cannot express? | WIN — the e-graph reaches a weights-first form Inductor's post-grad graph cannot express (see docs/results.md). |
| `real_win_hunt` | block | Which structured block topologies admit an autotuned win? | WIN on exploitable topologies; parity where structure is absent. |
| `real_linear_attn` | block | Does the affine-monoid scan lift pay on real linear-attention blocks? | WIN where the affine-monoid scan lift fires (fp64-exact); honest negatives where Inductor's pointwise fusion wins on CPU. |
| `morphism_e2e` | block | Do morphism windows convert term-flops into wall time? | WIN — term-FLOP reduction converts to measured wall time at GEMM-bound sizes. |
| `decode_scan_bench` _(cuda)_ | block | Does the chunked scan carrier beat the best non-carrier decode schedule? | WIN on launch-bound devices; CUDA needed for the graph leg. |
| `decode_bench` _(ad-hoc)_ | block | Does fewer GEMM launches pay off where launch overhead dominates? | NEGATIVE — the launch-bound hypothesis is falsified (losses where launch overhead already dominates). |
| `killer_demo` | block | Does per-model lowering autotune pick the measured-fastest variant? | WIN — the reported pick is the measured winner, never a static choice. |
| `bench_omd2` _(ad-hoc)_ | block | Does the cross-carrier omd lift survive a transformer-shaped attention? | Exploratory — fires on the mqa case; not a certified path. |

### e2e

| suite | tier | question | expected |
|---|---|---|---|
| `model_bench` | model | Do whole multi-block models beat Inductor under the autotuned lowering? | Latency/peak-memory/compile per model, verified; wins where structure exists. |
| `e2e_model` | model | Does composition (pairing + fold) beat plain Inductor end-to-end? | PARITY — pairing fires per block, verified fp64-exact; wall time is ~parity. |
| `e2e_models2` | model | Does composition hold across architecture families? | PARITY — in-repo replicas (minilm/vit/conv/llama/moe), verified per cell. |
| `e2e_llm` | model | Does composition help prefill + KV-cache decode on a llama-scale model? | Measured c+i/ind band; honest per-cell verdicts. |
| `stories15m_bench` _(ad-hoc)_ | model | Does the whole-model pipeline transform and verify a real checkpoint? | PARITY — all blocks transform+verify, but the mechanism doesn't pay at 15M/110M. |
| `bench_e2e` _(ad-hoc)_ | model | Does a MiniGPT optimize and verify at all? | Smoke — sanity only; use the rigorous sweeps for numbers. |

### bounded

| suite | tier | question | expected |
|---|---|---|---|
| `bounded_e2e` | model | Do error-budget rewrites buy wall-time on a real checkpoint? | WIN — error-budget rewrites deliver bounded members and a measured wall-time win vs Inductor on stories15M (see docs/results.md). |
| `structured_models` | block | Do LoRA / pruned / low-rank families admit bounded rewrites? | WIN where structure exists — params shrink / speedups, verified. |

### integration

| suite | tier | question | expected |
|---|---|---|---|
| `vllm_compare` _(cuda)_ | model | Can vLLM serve a catopt-optimized model token-for-token? | WIN — token-for-token agreement; the wins are complementary, not competing. |

## By mechanism

| mechanism | suites |
|---|---|
| laws | `law_bench`, `laws_effect` |
| egraph | `search_efficiency`, `reassoc_scale` |
| cost | `cost_fidelity`, `eval_axis` |
| policy | `policy_value` |
| pairing | `reassoc_scale`, `real_win_hunt`, `killer_demo`, `model_bench`, `e2e_model`, `e2e_models2`, `e2e_llm`, `stories15m_bench`, `bench_e2e` |
| autotune | `real_win_hunt`, `killer_demo`, `model_bench` |
| morphism | `morphism_coverage`, `morphism_e2e` |
| scan | `real_linear_attn` |
| carriers | `real_linear_attn`, `decode_scan_bench`, `bench_omd2` |
| decode | `decode_scan_bench`, `decode_bench` |
| cuda-graph | `decode_scan_bench` |
| omd | `bench_omd2` |
| bounded | `bound_amplification`, `bounded_e2e`, `structured_models` |
| weights | `structure_census`, `bound_amplification` |
| checkpoint | `morphism_coverage`, `stories15m_bench`, `bounded_e2e`, `vllm_compare` |
| serving | `vllm_compare` |
<!-- END GENERATED CATALOG -->
