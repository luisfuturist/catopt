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
python -m bench.suites.algebra.reassoc_scale --depths 4,8 --rows 16384
python -m bench.suites.core.law_bench --laws assoc --sizes 256
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
    compare.py    # regression comparison
    render/       # markdown · html · plots · quarto · slidev · dashboard
    templates/    # jinja2 sources for the HTML surfaces
  suites/{core,algebra,models}/        # the benchmarks
  common/                              # llama2c.py loader, fetch.py
  baselines/                           # committed golden results
  results/                             # working output (gitignored)
```

Suites live under `suites/` grouped by category and expose
`run_bench(args) -> Report`.  They never time anything themselves —
they describe `Case`/`Variant` data and hand it to `Runner`.

## The Report model

A suite states its conclusion as typed `Finding`s, not console text:

```python
Finding(
    claim="catopt reaches a weight-folded form Inductor cannot express",
    verdict=Verdict.WIN,                 # win|parity|regression|negative|inconclusive
    headline="8.93× vs Inductor at (k,d,B·T)=(8,512,4096)",
    metric="catopt+inductor / inductor",
    value=8.93,
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

1. Add a module under `suites/<category>/` exposing
   `run_bench(args) -> Report` and (optionally) `QUICK = {...}`.
2. Register it in `bench/registry.py` with a `SuiteSpec`.
3. Add a builder in `LAW_CASES` if it is a law bench.

Gates: `uv run pytest` (the harness has `tests/test_benchkit.py`),
`.venv/bin/ruff check`, `.venv/bin/ruff format --check`.

## Suite catalog — protocols and expected verdicts

| Suite | Expected verdict |
|---|---|
| `reassoc_scale` | Deep `x @ W1 @ … @ Wk` chains: the e-graph finds the weights-first form (one runtime GEMM); the post-grad FX capture **proves Inductor can't reach it**. Measured 8.93× vs Inductor at (k,d,B·T)=(8,512,4096), 16.12× at k=16. |
| `search_efficiency` | The e-graph encodes an exponential (Catalan) program space in O(k³) live e-nodes. Honest about the limits: exact saturation fragments past k≈11; production uses `rule_budgets`/`canonicalize`. |
| `real_linear_attn` | RetNet/GLA/delta-rule blocks: the affine-monoid scan lift fires and verifies fp64-exact. Measured retnet T=128 ≈ 2.9× vs eager; honest negatives where Inductor's fused pointwise kernel wins on CPU. |
| `real_win_hunt` | Autotuned wins on realistic block topologies (~1.1–10× vs Inductor). |
| `decode_scan_bench` | Carrier + CUDA-graph decode: chunked scan carriers amortize to zero launches (1.65–3.4× vs Inductor / best non-carrier). Needs CUDA for the graph leg. |
| `decode_bench` | The launch-bound hypothesis is **falsified**: launch-bound cells (B=1, T≤64) lose 4–15%; large cells land at parity. Losses included. |
| `model_bench` / `e2e_model` / `e2e_llm` / `e2e_models2` | Whole-model E2E: llama-toy, ~0.4B prefill+decode, non-decoder shapes; ~1.05× over plain Inductor, pairing fires per block, verified fp64-exact. |
| `cost_fidelity` | Predicted-cost vs measured-latency rank correlation (ρ); term-level cost tracks the backend. |
| `killer_demo` | `Autotuned` per-model lowering selection (eager/generic/batched/compiled), verified. |
| `law_bench` | Per-rewrite-law value harness: fired / rhs-member / picked / verified / cost & ms before→after. |
| `morphism_e2e` | Term-flops → wall-time conversion for the morphism laws (12.4–12.5× measured wall on 4-block chains). |
| `bounded_e2e` | `error_budget` sweep on a real checkpoint. Measured: bounded rewrites buy **nothing** on stories15M (0 accepted at every budget) — the honest negative. |
| `structured_models` | Structured-model families (LoRA / pruned / low-rank): bounded rewrites where structure exists. |
| `vllm_compare` | catopt vs vLLM and through it: HF export verified (~1.6e-5 max|Δ|), vLLM serves the exported dir token-for-token. Needs CUDA. |
| `structure_census` | How much catopt-exploitable structure real trained weights carry (ad-hoc). |
| `bound_amplification` | Weight-bound → output-error propagation on real activations (ad-hoc). |
| `stories15m_bench` | Real llama2.c checkpoints through `strategy=Compositional()` — parity at these sizes (ad-hoc). |
| `bench_e2e` / `bench_omd2` | Quick whole-model smoke / omd executor on an attention stack (ad-hoc research). |
