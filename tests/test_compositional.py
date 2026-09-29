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
from catopt.models import DeepParallel, ParallelBlock, ParallelLinear
from catopt.optimize import optimize_compositional


class MiniGPT(nn.Module):
    """Minimal PaLM/GPT-J-style stack: ModuleList of ParallelBlocks."""

    def __init__(
        self,
        dim: int = 64,
        n_heads: int = 4,
        depth: int = 4,
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelBlock(dim, n_heads, hidden_mult)
            for _ in range(depth)
        )

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

    # The recomposed model is a real clone sharing tensor storage —
    # and the caller's model was never mutated.
    assert stats["shared_params"] is True
    assert stats["in_place"] is False
    assert opt is not model
    assert all(isinstance(b, ParallelBlock) for b in model.blocks)

    # End-to-end equivalence (fp32).
    assert stats["end_to_end"]["max_rel_diff"] < 1e-4
    model.eval()
    opt.eval()
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
                *[DeepParallel(dim, dim, dim) for _ in range(depth)]
            )

        def forward(self, x):
            return self.net(x)

    model = MLPStack().eval().double()
    x = torch.randn(64, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)
    assert stats["n_optimized"] == 4
    assert stats["shared_params"] is True
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
            mods = [
                ParallelLinear(dim, n_experts=2) for _ in range(depth)
            ]
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
    # The original (unmodified) block object was kept — a fresh module
    # shell sharing the caller's parameter storage (param-sharing
    # clone: no second copy of the weights).
    assert stats["shared_params"] is True
    assert stats["in_place"] is False
    assert opt.blocks[1] is not None
    assert isinstance(opt.blocks[1], _DataDependentBlock)
    assert opt.blocks[1] is not model.blocks[1]  # clone, same weights
    assert opt.blocks[1].lin.weight is model.blocks[1].lin.weight
    assert torch.equal(
        opt.blocks[1].lin.weight, model.blocks[1].lin.weight
    )

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


def test_compositional_in_place_clone_failure_is_reported(monkeypatch):
    """When the recompose clone fails (e.g. an unpicklable non-tensor
    attr), the returned model is the INPUT unmodified — the report must
    say so, not run a degenerate self-comparison verify."""
    import catopt_torch.composer as C
    from catopt.optimize import optimize_compositional

    torch.manual_seed(0)
    model = MiniGPT(dim=32, n_heads=2, depth=1, hidden_mult=2).eval()
    x = torch.randn(1, 8, 32)

    def boom(*_a, **_k):
        raise RuntimeError("cannot pickle this attribute")

    monkeypatch.setattr(C.copy, "deepcopy", boom)
    opt, stats = optimize_compositional(model, x, verbose=False)
    assert opt is model  # same object — nothing grafted
    assert stats["in_place"] is True
    assert stats["shared_params"] is False
    assert stats["n_optimized"] == 0
    assert stats["end_to_end"]["skipped"] == "in_place"


class _RootParam(nn.Module):
    """Stack plus weights OUTSIDE any selected block: a Parameter and a
    buffer on the root, and an unregistered tensor attribute — the
    three tensor categories the sharing clone must alias."""

    def __init__(self, dim: int = 32, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelLinear(dim, n_experts=2) for _ in range(depth)
        )
        self.root_w = nn.Parameter(torch.randn(dim, dim))
        self.register_buffer("root_buf", torch.randn(dim))
        self.plain_tensor = torch.randn(dim)  # __dict__, unregistered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x @ self.root_w + self.root_buf + self.plain_tensor


def test_compositional_recompose_shares_tensor_storage():
    """The recompose clone copies module STRUCTURE but aliases every
    tensor — parameter, buffer, and unregistered __dict__ attr — so
    recomposing costs no second copy of the weights, and the caller's
    model is never mutated."""
    torch.manual_seed(0)
    model = _RootParam(dim=32, depth=2).eval()
    x = torch.randn(8, 32)

    opt, stats = optimize_compositional(model, x, verbose=False)

    assert stats["n_optimized"] == 2
    assert stats["in_place"] is False
    assert stats["shared_params"] is True
    assert opt is not model

    # Fresh shells, shared tensors — same objects, same storage.  The
    # blocks were replaced by IRModules (their params are new fused
    # weights); the weights outside every selected block alias.
    assert opt.blocks is not model.blocks
    assert opt.root_w is model.root_w
    assert opt.root_buf is model.root_buf
    assert opt.plain_tensor is model.plain_tensor
    assert opt.root_w.data_ptr() == model.root_w.data_ptr()

    # Grafting rebound only the clone's _modules — the caller's model
    # still holds its original blocks and forwards unchanged.
    assert all(isinstance(b, ParallelLinear) for b in model.blocks)
    assert stats["end_to_end"]["max_rel_diff"] < 1e-4


def test_shared_param_clone_preserves_structure_and_training():
    """Unit-level check of the clone helper: every module shell and
    dict is fresh, every tensor is aliased, non-tensor attrs copy."""
    import catopt_optimize.optimize as O

    torch.manual_seed(0)
    model = MiniGPT(dim=32, n_heads=2, depth=2, hidden_mult=2)
    model.plain_buf = torch.randn(4)
    model.tag = [1, 2, 3]  # non-tensor attr — must copy, not share
    model.train()

    clone = O._shared_param_clone(model)

    assert clone is not model
    assert clone.blocks is not model.blocks
    assert clone.blocks[0] is not model.blocks[0]
    # Structure copies; tensors share.
    assert clone.tag == model.tag and clone.tag is not model.tag
    assert clone.training is True
    assert clone.plain_buf is model.plain_buf
    for p_orig, p_new in zip(
        model.parameters(), clone.parameters(), strict=True
    ):
        assert p_new is p_orig
        assert p_new.data_ptr() == p_orig.data_ptr()
    # Rebinding on the clone leaves the original's _modules intact.
    O._replace_submodule(clone, "blocks.0", nn.Identity())
    assert isinstance(clone.blocks[0], nn.Identity)
    assert isinstance(model.blocks[0], ParallelBlock)


@pytest.mark.requires_cuda
def test_shared_param_clone_cuda_no_param_copy():
    """On GPU the clone allocates no second copy of the weights: device
    memory grows only by the (host-side) module shells."""
    import catopt_optimize.optimize as O

    torch.manual_seed(0)
    model = MiniGPT(dim=512, n_heads=8, depth=4, hidden_mult=4)
    model = model.half().cuda().eval()
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    before = torch.cuda.memory_allocated()

    clone = O._shared_param_clone(model)

    grown = torch.cuda.memory_allocated() - before
    assert grown < param_bytes // 10, (
        f"clone allocated {grown} B for {param_bytes} B of params"
    )
    for p_orig, p_new in zip(
        model.parameters(), clone.parameters(), strict=True
    ):
        assert p_new.data_ptr() == p_orig.data_ptr()
