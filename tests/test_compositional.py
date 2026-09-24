"""Compositional optimization: per-block eqsat + recompose.

Whole-model eqsat on a deep stack of structured blocks is monolithic —
the e-graph grows with the product of the blocks' alternatives, so a
4-layer MiniGPT saturates for minutes while each block alone takes
seconds.  ``optimize_compositional`` walks the module tree, captures each
block's real input with forward hooks, optimizes blocks independently,
and grafts the lowered IRModules back into a clone of the model.
"""

import time

import pytest
import torch
import torch.nn as nn

from catopt.models import ParallelBlock, ParallelLinear, DeepParallel
from catopt.optimize import optimize_compositional


class MiniGPT(nn.Module):
    """Minimal PaLM/GPT-J-style stack: ModuleList of ParallelBlocks."""

    def __init__(self, dim: int = 64, n_heads: int = 4,
                 depth: int = 4, hidden_mult: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelBlock(dim, n_heads, hidden_mult) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def test_compositional_parallel_block_stack():
    """4 ParallelBlocks optimize independently in seconds and recompose
    into a numerically equivalent model — where the monolithic eqsat
    saturates for minutes."""
    torch.manual_seed(0)
    model = MiniGPT(dim=64, n_heads=4, depth=4, hidden_mult=2).eval()
    x = torch.randn(2, 16, 64)

    t0 = time.time()
    opt, stats = optimize_compositional(model, x, verbose=False)
    elapsed = time.time() - t0
    assert elapsed < 60, f"compositional pass took {elapsed:.1f}s"

    # Every block was optimized, and the product law (pairing) fired in
    # each: q/k/v/gate/up share one normed input → one fused GEMM.
    for i in range(4):
        e = stats["blocks"][f"blocks.{i}"]
        assert e["status"] == "optimized"
        assert e["stats"].get("pairing_groups", 0) >= 1
    assert stats["n_optimized"] == 4
    assert stats["n_failed"] == 0

    # Aggregated parameter report: per-block weight diffs prefixed by
    # block name; fused GEMM weights are derived tensors.
    pr = stats["param_report"]
    assert pr["eliminated"]
    assert any("fused" in n for n in pr["derived"])
    assert all(n.startswith("blocks.") for n in pr["eliminated"])

    # End-to-end equivalence (fp32).
    assert stats["end_to_end"]["max_rel_diff"] < 1e-4
    model.eval(); opt.eval()
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-4


def test_compositional_sequential_stack_fp64():
    """A plain ``nn.Sequential`` stack optimizes block-by-block too, and
    the recomposed model is fp64-exact vs the original."""
    torch.manual_seed(0)

    class MLPStack(nn.Module):
        def __init__(self, dim: int = 32, depth: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                *[DeepParallel(dim, dim, dim) for _ in range(depth)])

        def forward(self, x):
            return self.net(x)

    model = MLPStack().eval().double()
    x = torch.randn(64, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)
    assert stats["n_optimized"] == 4
    for i in range(4):
        assert stats["blocks"][f"net.{i}"]["status"] == "optimized"
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9


class _DataDependentBlock(nn.Module):
    """torch.export cannot trace a data-dependent branch — this block
    always fails export_to_ir and must fall back to the original."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.lin(x)
        return -self.lin(x)


def test_compositional_fallback_keeps_original():
    """A block that fails optimize_model keeps its original submodule;
    the rest of the stack still optimizes and the model stays correct."""
    torch.manual_seed(0)
    dim, depth = 32, 3

    class MixedStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mods = [ParallelLinear(dim, n_experts=2) for _ in range(depth)]
            mods.insert(1, _DataDependentBlock(dim))
            self.blocks = nn.ModuleList(mods)

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    model = MixedStack().eval()
    x = torch.randn(64, dim)

    opt, stats = optimize_compositional(model, x, verbose=False)

    bad = stats["blocks"]["blocks.1"]
    assert bad["status"] == "failed"
    assert "error" in bad
    # The original (unmodified) block object was kept.
    assert opt.blocks[1] is not None
    assert isinstance(opt.blocks[1], _DataDependentBlock)
    assert opt.blocks[1] is not model.blocks[1]  # clone, same weights
    assert torch.equal(opt.blocks[1].lin.weight, model.blocks[1].lin.weight)

    # The healthy blocks still optimized — pairing fires on each
    # ParallelLinear (two linears sharing one input).
    for i in (0, 2, 3):
        e = stats["blocks"][f"blocks.{i}"]
        assert e["status"] == "optimized"
        assert e["stats"].get("pairing_groups", 0) >= 1
    assert stats["n_failed"] == 1
    assert stats["n_optimized"] == 3

    # End-to-end output is still correct: the failed block runs its
    # original implementation inside the recomposed model.
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-4
    assert stats["end_to_end"]["max_rel_diff"] < 1e-4
