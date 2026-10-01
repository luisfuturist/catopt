# bench/

Real-checkpoint benchmarks against Karpathy's llama2.c TinyStories
weights (`karpathy/tinyllamas` on Hugging Face).

## Fetch the checkpoints

```bash
python bench/fetch.py                  # both models, ~500 MB
python bench/fetch.py --models 15M     # just one
```

Downloads `stories15M.bin` / `stories110M.bin` into
`$XDG_CACHE_HOME/catopt` (default `~/.cache/catopt`), verifies the
7-int32 header and file size, and skips files already present and
valid. A valid copy already in `/tmp` is copied in instead of
re-downloaded.

Both benches resolve checkpoints in this order: `--ckpt <path>` →
`~/.cache/catopt/<name>` → `/tmp/<name>` → error pointing here.

## stories15m_bench.py — whole-model pipeline

```bash
python bench/stories15m_bench.py --device cuda --seq 128
python bench/stories15m_bench.py --device cpu  --seq 32
python bench/stories15m_bench.py --ckpt ~/.cache/catopt/stories110M.bin
```

Wraps the real checkpoint as an exportable `nn.Module`, runs
`optimize_compositional` block-by-block, verifies the optimized model
against eager, and times eager / catopt / Inductor / catopt+Inductor
with `torch.utils.benchmark`.

**Expected verdict: parity.** All blocks transform and verify
(~2e-5 max output delta; QKV + gate·up fuse, ~37% fewer GEMM
launches) but at stories15M/110M dimensions the mechanism doesn't pay
— ~3.0 ms / ~17.0 ms both ways on an RTX 2050. The transform is real;
the win isn't, at these sizes.

## decode_bench.py — launch-bound decode sweep

```bash
python bench/decode_bench.py --device cuda --quick    # B{1,4} x T{16,64}
python bench/decode_bench.py --device cuda --large    # B{8..32} x T{128..512}
python bench/decode_bench.py --device cpu  --quick
```

Sweeps (batch, seq) cells asking whether fewer GEMM launches pay off
where kernel-launch overhead dominates. Per cell: optimize once,
verify rel diff < 1e-4 (a wrong optimized module is a SKIP, never a
timed number), then median ± IQR for all four variants, CUDA-synced.
Verdict per cell: catopt+inductor/inductor < 0.97 → WIN, ≤ 1.03 →
parity, > 1.03 → REGRESSION.

**Expected verdict: the launch-bound hypothesis is falsified.**
Launch-bound cells (B=1, T≤64) lose 4–15% — split-view copies after
the fused GEMM cost more than the saved launches. Large cells
(B≥8, T≥128, and stories110M) land at parity within ~1%.

## bench_e2e.py — quick whole-model smoke

```bash
python bench/bench_e2e.py     # CUDA if available, else CPU
```

One-cell sanity run: a MiniGPT of stacked `ParallelBlock`s, optimize
once, report eager / Inductor / catopt / catopt+Inductor lower-quartile
ms. Naive single-config timing — for the rigorous sweep use
`decode_bench.py`.

## bench_omd2.py — omd executor on a realistic attention stack

```bash
python bench/bench_omd2.py --sizes 64,128 --skip-gap
```

Research bench: does the cross-carrier omd lift survive a
transformer-shaped attention (real q/k/v projections, multi-head,
causal mask), and is `BatchedOmdModule` still fast when it does?
Variants `mqa` (the firing case), `mha`, `mha-chunk`, `sdpa`; times
torch-eager / IR / best / omd / omd-batched / CUDA-graph / Inductor /
omd-direct where applicable. Bounded saturation (300k-node cap) —
exploratory, not a certified path.

## reassoc_scale.py — the head-to-head (the research claim)

```bash
python bench/reassoc_scale.py --device cpu
python bench/reassoc_scale.py --device cuda --depths 4,8,16 --dims 512 --rows 65536
```

Deep linear-attention-style chain `x @ W1 @ … @ Wk`: the equivalent
space is `k` bracketings of matmul; CatOpt's e-graph finds the
weights-first form (all k−1 weight products folded into one parameter
at compile time → one runtime GEMM), and the script **proves Inductor
can't reach it** by capturing Inductor's post-grad FX graph — all `k`
`aten.mm` nodes left-assoc, zero weight×weight mms — alongside
CatOpt's lowered graph (1 mm). Times eager / Inductor / manual ref /
catopt / catopt+inductor with `Timer`; verifies fp32 equivalence.

**Measured (CPU, torch 2.14):** 8.93× vs Inductor at (k,d,R)=(8,512,4096);
16.12× at k=16 — matching the modelled k× runtime-FLOP ratio.

## search_efficiency.py — the "finds it cheaply" half

```bash
python bench/search_efficiency.py --device cpu                # k=4..24
python bench/search_efficiency.py --exact-max 11              # exact saturation
```

For the k-chain the equivalent space is Catalan(k−1) (exact) /
k!·Catalan under comm. Sweeps saturation stats vs k; reports enodes
(bounded ~O(k³) live) vs the exponential program space, rule fires,
wall/extract times, and the `Optimizer.optimize` end-to-end path.
Honest about the limits: exact saturation re-enumerates substitutions
over fragmented classes past k≈11 (k=11 ≈ 17s); production uses
`rule_budgets`/`meta.canonicalize`, which is precisely why the
codebase ships them — the bench says so rather than claiming exact
eqsat scales.

## real_linear_attn.py — the real-model case

```bash
python bench/real_linear_attn.py --device cpu
python bench/real_linear_attn.py --device cuda --families retnet,gla,delta
```

Gated linear-attention blocks in the RetNet / GLA / delta-rule
family (Sun et al. 2023 shape): ``h_t = γ ⊙ h_{t−1} + u_t`` (retnet,
fixed decay), data-dependent decay (gla), or
``A_t = I − β k_t k_t^T`` (delta-rule), each with a k-deep value
chain — a real architecture where TWO transforms fire: the
affine-monoid scan lift (``applyd``/``affd`` carriers → O(log T)
level-batched schedule, certified fp64-exact via a saturated e-graph)
and the weights-first fold of the value chain.

**Measured (CPU):** retnet T=128 ``catopt_scan`` ≈ 2.9× vs eager, at
the closed-form geometric-series floor; honest negatives —
Inductor's fused pointwise kernel wins vs-eager on CPU (the launch
count inversion matters on launch-bound devices), and data-dependent
leaves (gla/delta) don't amortise per-leaf eval on CPU. The fp32
gate attributes which variant fails (`gate_checks` in aux).

## bounded_e2e.py — certified bounded rewrites on a real checkpoint

```bash
python bench/bounded_e2e.py --device cpu        # ~20 min (the 32k-row
                                                # head's bounded analysis dominates)
python bench/bounded_e2e.py --quick             # none + 1e-3 only
```

Sweeps ``search(..., error_budget=B)`` with ``detect_specials=True``
(B ∈ {None, 1e-4, 1e-3, 3e-3, 1e-2}) on the real stories15M
checkpoint through the ``Compositional`` pipeline; times eager /
inductor / catopt / catopt+inductor and measures held-out next-token
agreement (mean/max KL, top-1/top-5) on seeded prompts.  Also probes
the morphism lane (``KVLatentShare(budget=)``).

**Measured (CPU, T=128):** bounded rewrites buy **nothing** on this
model — 0 bounded members accepted at every budget.  The one real
candidate is the tied LM head: >50% of vocab rows have a
near-duplicate partner within 1e-4 (median nearest-row Chebyshev
distance ≈2.4e-6), so ``elide_bounded`` fires and is *delivered* by
extraction — then declined by the honest bound gate at every budget
(measured output rel exceeds the weight-space certified bound by
~2–8×).  Cost of asking: pipeline wall time 19 s → 134–365 s (the
O(rows²) near-dup clustering in ``specials._bounded_elide`` on the
32000×288 head).  Plumbing gaps documented in the report:
``error_budget`` does not reach ``MorphismSearch``'s per-block
searches (TypeError through ``optimize``), and ``KVLatentShare``
cannot match stories15M's 3-input ``Block(h, cos, sin)`` at any
budget.

## vllm_compare.py — catopt vs vLLM, and through it

```bash
# all-in-one env (torch-cuda + catopt packages + vllm + transformers):
/home/luis/vllm-probe/bin/python bench/vllm_compare.py --device cuda --quick

# split envs: repo .venv runs catopt/export legs; the serving leg is
# delegated to any interpreter that has vllm:
.venv/bin/python bench/vllm_compare.py --device cpu \
    --vllm-python ~/vllm-probe/bin/python --quick
```

vLLM's wins (paged KV cache, continuous batching, fused attention
kernels) are orthogonal to catopt's compile-time structural rewrites —
so the bench measures both AND runs the composition.  Three legs:

1. **HF export (verified).** Writes the stories15M checkpoint — or
   the catopt-*optimized* module — as a standard HF
   `LlamaForCausalLM` directory (`config.json` + `model.safetensors`).
   The optimized module's fused executors are split back to HF layout
   (`fused_10` → q|k|v, `fused_11` → gate|up), and q/k rows are
   permuted from llama2.c's interleaved RoPE to HF rotate-half (the
   llama.cpp `permute`); a pure-torch check proves attention scores
   are preserved (~1e-5 fp32 noise), and when `transformers` is
   importable the exported model's logits are compared against the
   catopt module (measured: 1.6e-5 max|Δ|).
2. **torch decode.** Greedy decode at a FIXED context window (catopt
   executors are shape-specialized — measured: optimizing at
   `(1,64)` bakes `[1,64,6,48]` reshapes).  Variants
   eager/inductor/catopt/catopt+inductor share one loop; a separate
   untimed eager *true* decode (unbounded window, real positions —
   vLLM-equivalent semantics) supplies reference tokens.
3. **vLLM serve.** `LLM(model=<hf_dir>, dtype='float32',
   enforce_eager=True)` — in-process when `import vllm` works, else
   via `--vllm-python` subprocess, else skipped with the full
   integration recipe in the report.  Reports init/prefill
   (~TTFT)/decode tok/s and **token agreement vs torch eager greedy**.

**Measured (RTX 2050 4GB, vllm 0.30.0+cu130, B=1 ctx=48 gen=16):**
vLLM serves the catopt-exported HF dir at ~245 tok/s decode with
*token-for-token* agreement (1.0) against torch eager greedy — the
integration chain is real end-to-end.  At this size torch's KV-free
Python decode loop (~410 tok/s fixed-window on the same GPU)
out-serves the engine — vLLM's overhead dominates below batch/scale
where paging and batching pay; the wins are complementary, not
competing.  Note vLLM's `QKVParallelLinear` already performs the QKV
fusion catopt finds — executor-level wins (eval tapes, batched
carriers) remain torch-side; weight-level wins (folded chains) map
cleanly through the HF checkpoint.

vLLM leg needs CUDA + toolkit-less-host fixes (handled:
`CUDA_HOME`→pip-wheel nvcc, `VLLM_USE_FLASHINFER_SAMPLER=0`).
Probe env used for the numbers above: `uv venv ~/vllm-probe
--python 3.12 && uv pip install --python ~/vllm-probe/bin/python vllm
-e packages/catopt-core -e packages/catopt-orchestrator
-e packages/catopt-carriers -e packages/catopt-torch`.

## benchkit.py — the shared harness + reports

```bash
python bench/run_all.py --device cpu --quick          # all harnessed suites
python bench/run_all.py --suites reassoc_scale --quick
python bench/reassoc_scale.py --out bench/results    # single suite + artifacts
```

`benchkit` is the shared machinery: `Case`/`Variant`/`Runner`
(torch.utils.benchmark medians + IQR, CUDA-synced) → `Report` (env
provenance: torch/python/git/device/timestamp) → JSON + Markdown +
matplotlib PNGs under `bench/results/` (gitignored — regenerate any
run). Harnessed suites expose `run_bench(args) -> Report`;
`run_all.py` drives them and writes the run-level `REPORT_<ts>.md`
index. matplotlib/pandas live in the `bench` dependency group —
`uv sync --group bench`.
