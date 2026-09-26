"""Launch-bound regime benchmark: does catopt+Inductor beat Inductor?

Hypothesis under test: catopt's pairing pass (shared-input linears →
one fused GEMM + split views) reduces GEMM *count*, which should pay
off where kernel-launch overhead dominates — small batch, short
sequences, decode-style.  Prior evidence is mixed: a toy ParallelBlock
at b=4 gave 1.19×, but on full transformer blocks at T=128 the
consolidation was inside the noise (README: "parity, not a win").
This bench sweeps the launch-bound corner honestly and reports
whatever it finds.

Method per (B, T) cell, all on the SAME fixed input ``idx``:

  1. ``optimize_compositional`` runs ONCE on the real-checkpoint model;
     its output is verified against eager (rel diff < 1e-4) — a cell
     whose optimized module is wrong is marked SKIP, not timed.
  2. Four variants are timed with ``torch.utils.benchmark``
     ``Timer.blocked_autorange``: eager, inductor (``torch.compile`` of
     the ORIGINAL module), catopt-uncompiled, catopt+inductor.
     Compiled variants get >= 3 un-timed warmup calls so compilation is
     never inside the measurement.  Every timed call ends in
     ``torch.cuda.synchronize()`` on CUDA — wall time of an async
     launch stream is submission time, not execution time.
  3. Verdict per cell: ratio = median(catopt+inductor)/median(inductor)
       < 0.97  → WIN      (catopt+inductor meaningfully faster)
       ≤ 1.03  → parity
       > 1.03  → REGRESSION
     IQR is reported for every variant.

Batch semantics — honesty note: ``stories15m_bench.Stories15M.forward``
silently drops the batch dimension (``h = h[0]`` on 3-D embeddings), so
B > 1 on that class would measure batch-1 work under a batch-N label.
``BatchedBlock``/``BatchedStories`` below keep the exact same math and
weights (``load_state_dict`` copy) but run a true (B, T, D) forward;
B=1 output is verified bitwise against the stock model at load time.

    python bench/decode_bench.py --device cuda --quick
    python bench/decode_bench.py --device cuda \
        --ckpt ~/.cache/catopt/stories110M.bin
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_weights import load_llama2c                      # noqa: E402
from stories15m_bench import (Block, Stories15M,              # noqa: E402
                              resolve_ckpt)


# ---------------------------------------------------------------------------
# Batched model — same weights, same math, real (B, T, D) forward.
# ---------------------------------------------------------------------------
class BatchedBlock(Block):
    """Block.forward generalized from (T, D) to (B, T, D)."""

    def forward(self, h, cos, sin):
        B, T, D = h.shape

        def rope(x):
            x = x.reshape(B, T, self.nh, self.hd)
            x1, x2 = x[..., ::2], x[..., 1::2]
            c, s = cos[None, :, None, :], sin[None, :, None, :]
            out = torch.stack([x1 * c - x2 * s,
                               x1 * s + x2 * c], -1)
            return out.reshape(B, T, self.nh * self.hd)

        xn = F.rms_norm(h, (D,), self.rms_att, 1e-5)
        q = rope(self.wq(xn)).reshape(B, T, self.nh, self.hd)
        k = rope(self.wk(xn)).reshape(B, T, self.nh, self.hd)
        v = self.wv(xn).reshape(B, T, self.nh, self.hd)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True)
        att = att.transpose(1, 2).reshape(B, T, D)
        h = h + self.wo(att)
        xn = F.rms_norm(h, (D,), self.rms_ffn, 1e-5)
        return h + self.w2(F.silu(self.w1(xn)) * self.w3(xn))


class BatchedStories(Stories15M):
    """Stories15M with real batching — weights identical via
    ``load_state_dict``; only the forwards differ."""

    def __init__(self, w, cfg):
        super().__init__(w, cfg)
        for i, b in enumerate(self.blocks):
            nb = BatchedBlock(cfg["dim"], cfg["hidden"], cfg["n_heads"],
                              cfg["dim"] // cfg["n_heads"])
            nb.load_state_dict(b.state_dict())
            self.blocks[i] = nb

    def forward(self, idx):
        T = idx.shape[-1]
        h = self.emb(idx)                       # (B, T, D) — kept
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            h = b(h, cos, sin)
        h = F.rms_norm(h, (h.shape[-1],), self.rms_final, 1e-5)
        return self.head(h)


# ---------------------------------------------------------------------------
def load_model(ckpt: str, device: str):
    """Header-read pattern identical to stories15m_bench.main()."""
    w = load_llama2c(ckpt)
    with open(ckpt, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
    dim, hidden, L, nh = int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3])
    vocab, seq = w["token_embedding"].shape[0], int(hdr[6])
    cfg = dict(dim=dim, hidden=hidden, n_layers=L, n_heads=nh,
               vocab=vocab, seq_len=seq)
    model = BatchedStories(w, cfg).eval().to(device)
    return model, cfg


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max().item()
            / (a.abs().max().item() + 1e-8))


def gemm_kernel_counts(fn, device: str, iters: int = 5):
    """torch.profiler: total CUDA kernels + GEMM-family kernel launches.

    GEMM-family = kernel name matching gemm/gemv/xmma/cutlass/splitK/
    dot — what cuBLAS actually launches for linear/matmul sites.
    Returns (total_kernels, gemm_kernels) per iter, or None on CPU.
    """
    if device != "cuda":
        return None
    import re
    pat = re.compile(r"gemm|gemv|xmma|cutlass|splitk|dot|nvjet", re.I)
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
        dev = torch.autograd.DeviceType.CUDA
        evts = [e for e in prof.events() if e.device_type == dev]
    except Exception:                   # noqa: BLE001 — evidence only
        return None
    total = len(evts) / iters
    gemms = sum(1 for e in evts if pat.search(e.name)) / iters
    return total, gemms


def bench_cell(model, opt, idx, ref, device, min_run_time, warmup=3):
    """Time all four variants on one (B, T) cell.

    Every variant's output is checked against ``ref`` (rel diff < 1e-4)
    after warmup — a wrong program is never benchmarked silently.
    Returns dict tag -> {"median", "iqr"} or {"error"}.
    """
    cuda = device == "cuda"

    def make_fn(mod):
        def f():
            with torch.no_grad():
                mod(idx)
            if cuda:
                torch.cuda.synchronize()
        return f

    variants = [("eager", model), ("inductor", None),
                ("catopt", opt), ("catopt+inductor", None)]
    try:
        variants[1] = ("inductor", torch.compile(model))
        variants[3] = ("catopt+inductor", torch.compile(opt))
    except Exception as e:  # torch.compile itself raised
        variants[1] = ("inductor", e)
        variants[3] = ("catopt+inductor", e)

    out = {}
    for tag, mod in variants:
        if isinstance(mod, Exception):
            out[tag] = {"error": f"compile: {mod}"}
            continue
        fn = make_fn(mod)
        try:
            for _ in range(warmup):     # compile + autotune land here
                fn()
            with torch.no_grad():
                rd = rel_diff(ref, mod(idx))
            if not (rd < 1e-4):
                out[tag] = {"error": f"output rel diff {rd:.2e} ≥ 1e-4"}
                continue
        except Exception as e:          # noqa: BLE001 — honest SKIP
            out[tag] = {"error": f"{type(e).__name__}: {e}"}
            continue
        t = Timer("fn()", globals={"fn": fn})
        r = t.blocked_autorange(min_run_time=min_run_time)
        out[tag] = {"median": r.median, "iqr": r.iqr}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path — default resolves "
                         "$XDG_CACHE_HOME/catopt/stories15M.bin then "
                         "/tmp/stories15M.bin (see bench/fetch.py)")
    ap.add_argument("--quick", action="store_true",
                    help="B{1,4} x T{16,64}, min_run_time=0.5")
    ap.add_argument("--large", action="store_true",
                    help="B{8,16,32} x T{128,256,512} — the crossover sweep")
    ap.add_argument("--min-run-time", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    args.ckpt = resolve_ckpt(args.ckpt)

    if args.quick:
        batches, seqs = [1, 4], [16, 64]
        min_run = args.min_run_time or 0.5
    elif args.large:
        batches, seqs = [8, 16, 32], [128, 256, 512]
        min_run = args.min_run_time or 1.0
    else:
        batches, seqs = [1, 4, 16], [16, 64, 256]
        min_run = args.min_run_time or 1.0

    t0 = time.time()
    model, cfg = load_model(args.ckpt, args.device)
    print(f"ckpt={args.ckpt}  dim={cfg['dim']} hidden={cfg['hidden']} "
          f"L={cfg['n_layers']} vocab={cfg['vocab']}  "
          f"device={args.device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if args.device == "cuda" else ""))
    print(f"torch {torch.__version__}  min_run_time={min_run}s  "
          f"load {time.time()-t0:.1f}s")

    # Honesty gate: B=1 batched forward must reproduce the stock model.
    ref_stock = Stories15M(                     # stock forward, same ckpt
        load_llama2c(args.ckpt), cfg).eval().to(args.device)
    idx1 = torch.randint(0, cfg["vocab"], (1, 8), device=args.device)
    with torch.no_grad():
        d = (ref_stock(idx1) - model(idx1)[0]).abs().max().item()
    del ref_stock
    print(f"batched-vs-stock B=1 max|Δ| = {d:.2e}"
          + ("" if d == 0.0 else "  <-- NOT bitwise, investigate"))
    if args.device == "cuda":
        torch.cuda.empty_cache()

    from catopt.optimize import optimize_compositional

    rows = []
    gemm_evidence = None
    for B in batches:
        for T in seqs:
            g = torch.Generator(device="cpu").manual_seed(
                args.seed + B * 1000 + T)
            idx = torch.randint(0, cfg["vocab"], (B, T),
                                generator=g).to(args.device)
            tag = f"B={B:<2} T={T:<3}"
            # -- optimize once, verify, or SKIP --------------------------
            try:
                with torch.no_grad():
                    ref = model(idx)
                t_opt = time.time()
                opt, rep = optimize_compositional(model, idx,
                                                  verbose=False)
                t_opt = time.time() - t_opt
                with torch.no_grad():
                    rd = rel_diff(ref, opt(idx))
                paired = sum(1 for e in rep["blocks"].values()
                             if (e.get("stats") or {}).get(
                                 "paired_extract"))
                if not (rd < 1e-4):
                    rows.append((tag, None,
                                 f"SKIP verify rel diff {rd:.2e}"))
                    print(f"{tag}  SKIP — opt rel diff {rd:.2e} "
                          f"≥ 1e-4")
                    continue
            except Exception as e:      # noqa: BLE001
                rows.append((tag, None,
                             f"SKIP opt: {type(e).__name__}: {e}"))
                print(f"{tag}  SKIP — optimize: {e}")
                continue

            # -- kernel-count evidence, once, on the smallest cell -----
            if gemm_evidence is None and args.device == "cuda":
                def _eager_fwd():
                    with torch.no_grad():
                        model(idx)

                def _opt_fwd():
                    with torch.no_grad():
                        opt(idx)
                gemm_evidence = (
                    tag,
                    gemm_kernel_counts(_eager_fwd, args.device),
                    gemm_kernel_counts(_opt_fwd, args.device))

            # -- bench all four variants -------------------------------
            res = bench_cell(model, opt, idx, ref, args.device, min_run)
            rows.append((tag, res,
                         f"{rep['n_optimized']}/{rep['n_blocks']} blocks"
                         f" paired={paired} rel={rd:.1e} "
                         f"opt={t_opt:.0f}s"))
            # Free per-cell artifacts — deepcopied + compiled modules
            # pile up fast on a 4 GB card and distort later cells.
            del opt
            gc.collect()
            try:
                torch._dynamo.reset()
            except Exception:           # noqa: BLE001
                pass
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # -- table ---------------------------------------------------------
    print("\n=== medians (ms) — same idx per cell, no_grad, "
          "CUDA-synced ===")
    hdr = (f"{'cell':<11} {'eager':>16} {'inductor':>16} "
           f"{'catopt':>16} {'cat+ind':>16} {'c+i/ind':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    wins = parity = regs = skips = 0
    for tag, res, note in rows:
        if res is None:
            skips += 1
            print(f"{tag:<11} {note}")
            continue

        def fmt(tag_):
            e = res.get(tag_)
            if not e:
                return f"{'—':>16}"
            if "error" in e:
                return f"{'SKIP':>16}"
            return f"{e['median']*1e3:>8.3f} ±{e['iqr']*1e3:<6.3f}"

        mi = res.get("inductor") or {}
        mc = res.get("catopt+inductor") or {}
        if "median" in mi and "median" in mc:
            ratio = mc["median"] / mi["median"]
            verdict = ("WIN" if ratio < 0.97
                       else "REGRESSION" if ratio > 1.03 else "parity")
            wins += verdict == "WIN"
            regs += verdict == "REGRESSION"
            parity += verdict == "parity"
            rstr, vstr = f"{ratio:>8.3f}", verdict
        else:
            errs = "; ".join(f"{t}: {e['error']}" for t, e in res.items()
                             if "error" in (e or {}))
            rstr, vstr = f"{'—':>8}", f"SKIP ({errs})"
            skips += 1
        print(f"{tag:<11} {fmt('eager')} {fmt('inductor')} "
              f"{fmt('catopt')} {fmt('catopt+inductor')} {rstr}  {vstr}")
        print(f"{'':<11} └ {note}")

    print("-" * len(hdr))
    print(f"cells: {wins} WIN / {parity} parity / {regs} REGRESSION "
          f"/ {skips} SKIP")

    if gemm_evidence:
        tag, orig, optd = gemm_evidence
        print(f"\n=== kernel-count evidence ({tag}, 5 fwd iters) ===")
        if orig and optd:
            print(f"eager : {orig[0]:.0f} CUDA kernels/fwd, "
                  f"{orig[1]:.0f} GEMM-family")
            print(f"catopt: {optd[0]:.0f} CUDA kernels/fwd, "
                  f"{optd[1]:.0f} GEMM-family")
        else:
            print("  profiler unavailable — counts skipped")

    print("\nVerdict scale: cat+ind/ind <0.97 WIN, ≤1.03 parity, "
          ">1.03 REGRESSION.  Median ± IQR in ms.")


if __name__ == "__main__":
    main()
