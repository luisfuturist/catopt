"""GPU benchmark for catopt — run after enabling the NVIDIA driver.

The dGPU on this machine (RTX 2050, GA107/Ampere) is currently blocked by
/etc/modprobe.d/blacklist-nvidia.conf.  To enable it:

    sudo mv /etc/modprobe.d/blacklist-nvidia.conf{,.disabled}
    sudo modprobe nvidia nvidia_modeset nvidia_uvm nvidia_drm
    python3 -c "import torch; print(torch.cuda.is_available())"

(If a tool like envycontrol/optimus-manager/system76-power created that
file it may restore it on reboot — disable it via that tool instead.)

Then:  python bench_gpu.py
"""

from __future__ import annotations

import sys

import torch

from catopt.benchmark import benchmark_model
from catopt.models import (
    AttentionBlock,
    DeepParallel,
    GQAAttention,
    MatrixChain,
    NormLinear,
    ParallelBlock,
    ParallelLinear,
    SwiGLU,
    TransformerBlock,
)
from catopt.optimize import optimize_model


def main() -> None:
    if not torch.cuda.is_available():
        print(__doc__)
        sys.exit(1)

    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"{'case':<24} {'equiv':>10} {'inductor ms':>12}"
        f" {'catopt ms':>10} {'speedup':>9}"
    )
    print("-" * 70)

    cases = [
        (
            "MatrixChain b=4096",
            MatrixChain(128, 64, 32, 8),
            torch.randn(4096, 128),
        ),
        (
            "ParallelLinear b=4096",
            ParallelLinear(512, 2),
            torch.randn(4096, 512),
        ),
        (
            "DeepParallel b=4096",
            DeepParallel(512, 512, 512),
            torch.randn(4096, 512),
        ),
        ("SwiGLU b=4096", SwiGLU(512, 4), torch.randn(4096, 512)),
        ("SwiGLU b=128", SwiGLU(512, 4), torch.randn(128, 512)),
        (
            "Attention b=64 T=256",
            AttentionBlock(512, 8),
            torch.randn(64, 256, 512),
        ),
        (
            "NormLinear b=256",
            NormLinear(512, 512),
            torch.randn(256, 64, 512),
        ),
        (
            "GQA b=64 T=256",
            GQAAttention(512, 8, 2),
            torch.randn(64, 256, 512),
        ),
        (
            "TBlock b=64 T=512",
            TransformerBlock(512, 8, 4),
            torch.randn(64, 512, 512),
        ),
        (
            "TBlock b=16 T=256",
            TransformerBlock(512, 8, 4),
            torch.randn(16, 256, 512),
        ),
        (
            "ParallelBlock b=64 T=256",
            ParallelBlock(512, 8, 4),
            torch.randn(64, 256, 512),
        ),
        (
            "ParallelBlock b=4 T=64",
            ParallelBlock(512, 8, 4),
            torch.randn(4, 64, 512),
        ),
    ]

    for name, model, x in cases:
        model = model.to(dev).eval()
        x = x.to(dev)
        opt, _ = optimize_model(model, x, verbose=False)
        opt = opt.to(dev).eval()
        with torch.no_grad():
            d = (model(x) - opt(x)).abs().max().item()
        r_o = benchmark_model(model, x, name, use_compile=True)
        r_p = benchmark_model(opt, x, name, use_compile=True)
        print(
            f"{name:<24} {d:>10.2e} {r_o.mean_ms:>12.4f}"
            f" {r_p.mean_ms:>10.4f} {r_o.mean_ms / r_p.mean_ms:>8.2f}x"
        )


if __name__ == "__main__":
    main()
