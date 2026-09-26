"""End-to-end whole-model benchmark: eager vs Inductor vs catopt.

The missing headline number: not per-rewrite speedups but the
optimized forward pass of a whole stacked model.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from catopt.models import ParallelBlock
from catopt.optimize import optimize_model


class MiniGPT(nn.Module):
    """N stacked PaLM-style parallel blocks + head — every layer has
    five projections reading one normed activation (the pairing
    closure case)."""

    def __init__(self, dim=256, n_heads=8, n_layers=4, vocab=512):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ParallelBlock(dim, n_heads, 4) for _ in range(n_layers)]
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def bench(fn, x, reps=20, warmup=4):
    dev = x.device
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 4]  # lower quartile


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device: {dev}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if dev.type == "cuda"
            else ""
        )
    )
    torch.manual_seed(0)

    for n_layers, T, B in [(2, 128, 8)]:
        m = MiniGPT(256, 8, n_layers).to(dev).eval().float()
        x = torch.randn(B, T, 256, device=dev)
        n_params = sum(p.numel() for p in m.parameters())
        print(
            f"\n=== MiniGPT layers={n_layers} d=256 T={T} B={B} "
            f"params={n_params / 1e6:.1f}M ==="
        )

        t0 = time.perf_counter()
        print("  optimizing...", flush=True)
        opt, stats = optimize_model(
            m, x, verbose=False, max_iterations=8, max_enodes=200_000
        )
        opt = opt.to(dev).eval()
        pipe_s = time.perf_counter() - t0

        with torch.no_grad():
            d = (m(x) - opt(x)).abs().max().item()

        eager_ms = bench(m, x)
        ind_ms = bench(torch.compile(m), x, reps=15)
        opt_ms = bench(opt, x)
        opti_ms = bench(torch.compile(opt), x, reps=15)

        print(f"  pipeline time: {pipe_s:.1f}s | equiv: {d:.2e}")
        top = dict(
            sorted(
                stats.get("rule_fires", {}).items(),
                key=lambda kv: -kv[1],
            )[:8]
        )
        print(f"  rule fires (top): {top}")
        print(
            f"  {'eager':>14} {'inductor':>14} {'catopt':>14} "
            f"{'catopt+ind':>14}"
        )
        print(
            f"  {eager_ms:>12.3f}ms {ind_ms:>12.3f}ms "
            f"{opt_ms:>12.3f}ms {opti_ms:>12.3f}ms"
        )
        print(
            f"  speedup vs eager: {eager_ms / opt_ms:.2f}x | "
            f"vs inductor: {ind_ms / opti_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
