"""Example neural-network blocks used as benchmark targets.

These are the "killer experiment" cases: neural-network computations
where conventional compilers (TorchInductor) reportedly struggle.

* :class:`SwiGLU` — SwiGLU activation + dual projections (the polyhedral RFC case).
* :class:`RMSNorm` — root-mean-square normalization.
* :class:`AttentionBlock` — scaled dot-product attention.
* :class:`ResidualMLP` — a residual MLP block where matmul associativity
  and distributivity can matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation: ``x * silu(gate) * up``."""

    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        h = dim * hidden_mult
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        u = self.up(x)
        h = F.silu(g) * u
        return self.down(h)


class RMSNorm(nn.Module):
    """Root-Mean-Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-token RMS normalization
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class AttentionBlock(nn.Module):
    """Multi-head attention using SDPA.

    The interesting optimization opportunity is around the QKV
    projections and the attention scaling.
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ResidualMLP(nn.Module):
    """A residual MLP with two linear layers and a Swish activation.

    The matmul associativity and distributivity of the residual
    connection create an optimization opportunity.
    """

    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        h = dim * hidden_mult
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(h, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = fc2(silu(fc1(norm(x)))) + x   — the residual is an 'add'
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return h + x


class MatrixChain(nn.Module):
    """Three sequential matmuls — demonstrates associativity optimization.

    The model computes ``((x @ W1) @ W2) @ W3`` (left-associative).

    **The categorical insight:** matrix multiplication is associative
    (composition in a category).  So ``((x @ W1) @ W2) @ W3`` can be
    rewritten as ``x @ (W1 @ (W2 @ W3))`` — i.e., pre-compute the fused
    weight ``W_fused = W1 @ W2 @ W3`` at compile time, then do a single
    matmul at runtime.

    With a funnel-shaped dimension schedule (e.g. 128→64→32→8), the
    right-associative form is dramatically cheaper in total FLOPs
    because the intermediate weights are small.

    TorchInductor does **not** discover this transformation — it has no
    matrix-chain reordering pass.
    """

    def __init__(self, d0: int, d1: int, d2: int, d3: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d0, d1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(d1, d2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(d2, d3) * 0.02)
        self.dims = (d0, d1, d2, d3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-associative: ((x @ W1) @ W2) @ W3
        h1 = torch.matmul(x, self.W1)
        h2 = torch.matmul(h1, self.W2)
        out = torch.matmul(h2, self.W3)
        return out

    @staticmethod
    def flops(dims: tuple[int, int, int, int], batch: int) -> int:
        """Total FLOP count for the left-associative form."""
        d0, d1, d2, d3 = dims
        return 2 * batch * (d0 * d1 + d1 * d2 + d2 * d3)

    @staticmethod
    def fused_flops(dims: tuple[int, int, int, int], batch: int) -> int:
        """Total FLOP count for the right-associative (fused-weight) form.

        Includes compile-time weight pre-computation.
        """
        d0, d1, d2, d3 = dims
        precompute = 2 * (d1 * d2 * d3 + d0 * d1 * d3)
        runtime = 2 * batch * d0 * d3
        return precompute + runtime

