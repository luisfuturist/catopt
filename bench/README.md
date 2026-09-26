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
