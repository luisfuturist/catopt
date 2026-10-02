# catopt — measured results

Generated from the pinned baselines in `bench/baselines/` —
regenerate with `python -m bench results --out docs/results.md`.
Each entry states what the suite asked, what it expected, what
it measured, and the machine it measured on.  The full
per-suite reports (timings, plots, raw metrics) are produced by
`python -m bench report <baseline.json>`.

## Compositional pairing demo

*Does per-model lowering autotune pick the measured-fastest variant?*

**Expected** — WIN — the reported pick is the measured winner, never a static choice.

| verdict | finding | headline |
|---|---|---|
| **WIN** | the Autotuned lowering beats eager on at least one model | best 2.55× vs eager on matrix_chain |
| **WIN** | every model's optimized lowering verifies | 4/4 models verified |

Measured 2026-10-02T00:19:09+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/killer_demo.json`](../bench/baselines/killer_demo.json).

## Rewrite-law effect bench

*Does each registered rewrite law fire, get picked by extraction, and lower to a verified term?*

**Expected** — WIN — every registered law fires/picks/verifies; non-firing laws report honestly.

| verdict | finding | headline |
|---|---|---|
| **WIN** | every registered law fires, is picked by extraction, and lowers to a verified term | 11 verified / 11 picked / 13 fired across 13 cells |

Measured 2026-10-02T00:18:07+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/law_bench.json`](../bench/baselines/law_bench.json).

## Weight-chain reassociation vs Inductor

*Can the e-graph find a form Inductor's post-grad graph cannot express?*

**Expected** — WIN — 8.9–16.1× vs Inductor on k-deep weight chains.

| verdict | finding | headline |
|---|---|---|
| **WIN** | catopt reaches a weight-folded form Inductor's post-grad graph cannot express | 17.12× vs Inductor at (k,d,B·T)=(16,512,4096) (18.63× vs eager) |
| **WIN** | every timed cell passes the fp32 equivalence gate | 12/12 cells verified |

Measured 2026-10-02T00:22:06+00:00 on 12th Gen Intel(R) Core(TM) i5-12500H (git `ba38bd9` (dirty)) — baseline [`bench/baselines/reassoc_scale.json`](../bench/baselines/reassoc_scale.json).

