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


class ParallelLinear(nn.Module):
    """Two projections of the SAME input, summed: ``x@W1 + x@W2``.

    This is a real, non-degenerate pattern: LoRA / parallel adapters /
    model-soup ensembles.  Deployment tooling merges the weights by hand
    (``peft``'s ``merge_and_unload``), but the compiler does not — Inductor
    keeps two separate matmuls because folding weight matrices together
    requires compile-time arithmetic on *parameters*, not kernel fusion.

    The categorical law is bilinearity of linear maps in the second slot:

        x @ W1 + x @ W2  =  x @ (W1 + W2)

    which needs the e-graph matcher to enforce that both matmuls share the
    SAME input (a repeated metavariable).
    """

    def __init__(self, dim: int, n_experts: int = 2, expert_dim: int | None = None) -> None:
        super().__init__()
        out = expert_dim or dim
        self.linears = nn.ModuleList(
            nn.Linear(dim, out, bias=False) for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum of parallel projections — NOT a stack, so it cannot be
        # reduced by trivially concatenating weights without summing them.
        out = self.linears[0](x)
        for lin in self.linears[1:]:
            out = out + lin(x)
        return out

    @staticmethod
    def flops(dims: tuple[int, int], batch: int) -> int:
        """Two (or more) matmuls: 2 * batch * dim * out * n_experts."""
        d, o = dims
        return 2 * batch * d * o  # per expert; caller multiplies


class DeepParallel(nn.Module):
    """Composed case: ``(x@W1 + x@W2) @ W3``.

    Requires TWO categorical laws in sequence plus compile-time folding:

    1. ``weight_factor_matmul``  — merge to ``x @ (W1+W2)``
    2. ``assoc_matmul``          — reassociate to ``x @ ((W1+W2) @ W3)``
    3. IRModule weight folding   — materialise ``(W1+W2) @ W3`` once

    No single pattern-matching pass finds this: it needs the distributive
    merge AND the reassociation to interact before folding is even legal.
    """

    def __init__(self, dim: int, mid: int, out: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(dim, mid, bias=False)
        self.W2 = nn.Linear(dim, mid, bias=False)
        self.W3 = nn.Linear(mid, out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(self.W1(x) + self.W2(x))


class NormLinear(nn.Module):
    """RMSNorm followed by a linear projection — the norm-folding target.

    ``x * rms * w_norm @ W.T`` has two commutable diagonals:
    the per-row scale ``rms`` (left diagonal — hoists out) and the
    per-channel gain ``w_norm`` (right diagonal — folds into W).
    """

    def __init__(self, dim: int, out: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(dim) * 0.5 + 1.0)
        self.proj = nn.Linear(dim, out or dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.proj(x * rms * self.norm_weight)


class TransformerBlock(nn.Module):
    """Decoder block: ``x + attn(norm(x)); x + swiglu(norm(x))``.

    Stacks every product-structure opportunity in one graph:

    * channel-gain fold into the q/k/v weights,
    * fused QKV (one GEMM → three head projections),
    * channel-gain fold into the gate/up weights,
    * fused SwiGLU gate/up,
    * residual adds (monoid structure).

    The two norms are RMSNorm-style (``x * rms * w``).
    """

    def __init__(self, dim: int, n_heads: int = 8,
                 hidden_mult: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.norm1_w = nn.Parameter(torch.ones(dim))
        self.norm2_w = nn.Parameter(torch.ones(dim))
        self.attn = AttentionBlock(dim, n_heads)
        h = dim * hidden_mult
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)

    def _rms(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return t * rms * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._rms(x, self.norm1_w))
        n = self._rms(x, self.norm2_w)
        x = x + self.down(F.silu(self.gate(n)) * self.up(n))
        return x


class ParallelBlock(nn.Module):
    """PaLM/GPT-J-style parallel block: ``x + attn(norm(x)) + mlp(norm(x))``.

    All five projections — q, k, v, gate, up — read the SAME normed
    activation ``n``.  The product law pairs them transitively into ONE
    GEMM whose output splits into five uneven sections; the downstream
    structure (head views + SDPA for q/k/v, silu·mul for gate/up) is
    preserved.  This is the closure property of the categorical product:
    pairing composes, and the fused weight is ``cat(Wq,Wk,Wv,Wg,Wu)``.
    """

    def __init__(self, dim: int, n_heads: int = 8,
                 hidden_mult: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.norm_w = nn.Parameter(torch.ones(dim))
        self.attn = AttentionBlock(dim, n_heads)
        h = dim * hidden_mult
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        n = x * rms * self.norm_w
        return x + self.attn(n) + self.down(F.silu(self.gate(n)) * self.up(n))


class GQAAttention(nn.Module):
    """Grouped-query attention: q has ``n_heads``, k/v have ``n_kv_heads``.

    Fused QKV here requires an *asymmetric* split — the fused GEMM output
    is ``[n_heads*d | n_kv*d | n_kv*d]`` — which is what real inference
    stacks implement (vLLM's ``QKVParallelLinear``).  ``torch.chunk``
    cannot express it; the IR needs a ``split`` with explicit sizes.
    """

    def __init__(self, dim: int, n_heads: int = 8, n_kv_heads: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                           enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.out_proj(out)


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

