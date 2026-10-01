"""Contraction-diagram tests — plan 0013 stages 1+2.

Sibling of ``test_morphism.py`` / ``test_morphism_kv.py``.  Pins the
diagram object layer end to end:

* ``lift_diagram`` — hyperedge topology over the morphism lift: leaf
  nodes for model args / bound tables / model-level intermediates,
  the ``<output>`` sink, per-end signature roles (``activation`` /
  ``const_table`` / ``unresolved``), wire-verdict annotations, and
  multi-arity shared-input edges.  Opaque nodes keep their place as
  boundary objects.
* the four moves — ``MergeProjs`` / ``FactorShared`` /
  ``ReorderCompose`` / ``SplitLeaf`` — each pairing a diagram-level
  legality check with a ``ReifySpec`` onto the certified machinery:
  candidacy, honest declines, and verified grafts (fp64 e2e).
* the greedy :class:`ContractionSearch` driver — consumed-node
  claims, per-candidate stats, the per-block fallback, param-sharing
  clone delivery, and the end-to-end verify.
"""

import catopt_orchestrator.diagram as D
import catopt_orchestrator.morphisms as M
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.cost import flops_cost, launch_aware_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_orchestrator import (
    ContractionSearch,
    DiagramMove,
    MorphismMatch,
    Optimizer,
    ReifySpec,
    optimize_diagram,
)
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer
from catopt_torch.models import ParallelLinear
from catopt_torch.report import VerifyReport

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    """fp64 probe input."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*dims, generator=g, dtype=torch.float64)


def _lift(model: nn.Module, x=None) -> D.Diagram:
    """Lift *model* through the torch ports to a Diagram."""
    return D.lift_diagram(
        model.eval().double(),
        _x() if x is None else x,
        source=TorchSource(),
        composer=TorchComposer(),
    )


def _reify(move, match, diagram, **kw):
    """Run a move's reify with the standard knobs."""
    args = dict(
        sink=TorchSink(),
        cost_fn=launch_aware_cost,
        verify_tol=1e-4,
        max_iterations=6,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    args.update(kw)
    return move.reify(match, diagram.graph, **args)


class _Lin(nn.Module):
    """``x @ Wᵀ`` — a bare projection block."""

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        """Seed the weight deterministically."""
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.lin.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project."""
        return self.lin(x)


class _Chain(nn.Module):
    """``x -> b0 -> b1 -> …`` — a pure chain."""

    def __init__(self, dim: int = 16, depth: int = 4) -> None:
        """``depth`` chained linears."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 100 + i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""
        for b in self.blocks:
            x = b(x)
        return x


class _ChainWrap(nn.Module):
    """``y = b0(x); return y + b1(y)`` — the ``chain_wrapped`` verdict.

    A's output feeds B AND the parent's wrap-add — the boundary is a
    chain that the residual wrap closes.
    """

    def __init__(self, dim: int = 16) -> None:
        """Two blocks under the wrap."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 110 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``y + b1(y)`` with ``y = b0(x)``."""
        y = self.blocks[0](x)
        return y + self.blocks[1](y)


class _ChainWrapTail(nn.Module):
    """``b0 -> b1 -> (z + b2(z))`` — a run ending chain_wrapped."""

    def __init__(self, dim: int = 16) -> None:
        """Three blocks; the last pair wraps."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 115 + i) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain two, wrap the third."""
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        return x + self.blocks[2](x)


class _Resid(nn.Module):
    """``x = x + b_i(x)`` — a residual stream."""

    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        """``depth`` residual linears."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 120 + i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accumulate the stream."""
        for b in self.blocks:
            x = x + b(x)
        return x


class _ResidTail(nn.Module):
    """``x += b0(x); return b1(x)`` — the tail consumes the sum."""

    def __init__(self, dim: int = 16) -> None:
        """A wrapped block plus a plain tail block."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 130 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wrap b0, then read the stream plainly."""
        return self.blocks[1](x + self.blocks[0](x))


class _ResidClose(nn.Module):
    """Two wrapped blocks then a plain tail — ``rw`` run + ``residual``
    close."""

    def __init__(self, dim: int = 16) -> None:
        """Two stream addends plus the tail reader."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 135 + i) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x += b0(x); x += b1(x); return b2(x)``."""
        x = x + self.blocks[0](x)
        x = x + self.blocks[1](x)
        return self.blocks[2](x)


class _SiluBlock(nn.Module):
    """``silu(x)`` — no projections at all."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply silu."""
        return F.silu(x)


class _ResidNoProj(nn.Module):
    """Residual stream of projection-free blocks — no commutes."""

    def __init__(self, dim: int = 16) -> None:
        """Two silu blocks on the stream."""
        super().__init__()
        self.blocks = nn.ModuleList([_SiluBlock(), _SiluBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accumulate."""
        x = x + self.blocks[0](x)
        return x + self.blocks[1](x)


class _DataDep(nn.Module):
    """A block whose forward reads ``x.data`` — export fails."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read .data — the exporter refuses."""
        return x + torch.ones_like(x.data)


class _ResidOpaque(nn.Module):
    """``x = x + b0(x); return b1(x)`` where b1 cannot export."""

    def __init__(self, dim: int = 16) -> None:
        """Linear head plus the opaque tail."""
        super().__init__()
        self.blocks = nn.ModuleList([_Lin(dim, 140), _DataDep()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wrap the linear, feed the opaque block."""
        return self.blocks[1](x + self.blocks[0](x))


class _Parallel(nn.Module):
    """``y = b0(x) + b1(x)`` — same-input family of bare projections."""

    def __init__(self, dim: int = 16) -> None:
        """Two sibling linears."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 150 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the sibling outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _Deep(nn.Module):
    """``up(silu(down(x)))`` — projections buried under activations."""

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        """Down projection + up projection."""
        super().__init__()
        self.down = nn.Linear(dim, dim, bias=False).double()
        self.up = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.down.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )
            self.up.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``up(silu(down(x)))``."""
        return self.up(F.silu(self.down(x)))


class _ParallelDeep(nn.Module):
    """Deep members — the compose recipe cannot factor their adds."""

    def __init__(self, dim: int = 16) -> None:
        """Two deep siblings."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Deep(dim, 160 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the deep outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _GeluLin(nn.Module):
    """``lin(gelu(x))`` — the projection reads a *transformed* input."""

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        """Linear over a gelu'd input."""
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.lin.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``lin(gelu(x))``."""
        return self.lin(F.gelu(x))


class _ParallelMismatch(nn.Module):
    """``lin0(x) + lin1(gelu(x))`` — shared input, unshared operands."""

    def __init__(self, dim: int = 16) -> None:
        """One plain linear + one gelu'd linear."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Lin(dim, 170), _GeluLin(dim, 171)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the mismatched outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _ParallelNonProj(nn.Module):
    """``lin(x) + silu(x)`` — one member carries no projection."""

    def __init__(self, dim: int = 16) -> None:
        """Linear + activation members."""
        super().__init__()
        self.blocks = nn.ModuleList([_Lin(dim, 180), _SiluBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum linear + activation."""
        return self.blocks[0](x) + self.blocks[1](x)


class _MulOut(nn.Module):
    """``b0(x) * b1(x)`` — shared input, non-additive consumption."""

    def __init__(self, dim: int = 16) -> None:
        """Two linears multiplied at the model level."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 190 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply — the additive slot evidence must fail."""
        return self.blocks[0](x) * self.blocks[1](x)


class _DoubleCall(nn.Module):
    """``b0(x) + b1(x) + b1(x)`` — b1 runs twice on the same input."""

    def __init__(self, dim: int = 16) -> None:
        """b0 once, b1 twice."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 200 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b1 is re-entered — calls==2."""
        return self.blocks[0](x) + self.blocks[1](x) + self.blocks[1](x)


class _ParallelIntra(nn.Module):
    """One ``ParallelLinear`` block — the intra fan-in arm."""

    def __init__(self, dim: int = 16) -> None:
        """Wrap the stock parallel-projection block."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [ParallelLinear(dim, n_experts=2).double()]
        )
        torch.manual_seed(210)
        with torch.no_grad():
            for lin in self.blocks[0].linears:
                lin.weight.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the intra block."""
        return self.blocks[0](x)


class _SharedNormBlock(nn.Module):
    """``proj(norm(x))`` on a shared ``nn.LayerNorm`` — the recompute."""

    def __init__(self, norm: nn.Module, dim: int, seed: int) -> None:
        """Hold the shared norm plus a per-block projection."""
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm then project."""
        return self.proj(self.norm(x))


class _SharedNormStack(nn.Module):
    """``y = Σ b_i(x)`` — siblings recompute one shared norm."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        """``depth`` blocks over one shared LayerNorm."""
        super().__init__()
        self.norm = nn.LayerNorm(dim).double()
        self.blocks = nn.ModuleList(
            _SharedNormBlock(self.norm, dim, 220 + i)
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


def _latent(r: int, dim: int, seed: int = 0) -> torch.Tensor:
    """The shared right-factor basis ``U`` (r, dim)."""
    return torch.randn(
        r,
        dim,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


class _KVBlock(nn.Module):
    """``sdpa(q,k,v) → out_proj`` — k/v built low-rank through ``U``."""

    def __init__(
        self, dim: int, d_kv: int, U: torch.Tensor, seed: int
    ) -> None:
        """Build projections; k/v factored through ``U``."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Linear(dim, d_kv, bias=False)
        self.k_proj = nn.Linear(dim, d_kv, bias=False)
        self.v_proj = nn.Linear(dim, d_kv, bias=False)
        self.out_proj = nn.Linear(d_kv, dim, bias=False)
        self.double()
        dk = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        dv = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        with torch.no_grad():
            self.k_proj.weight.copy_(dk @ U)
            self.v_proj.weight.copy_(dv @ U)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """sdpa then output projection."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return self.out_proj(o)


class _KVStack(nn.Module):
    """``y = Σ b_i(x)`` — parallel low-rank KV blocks."""

    def __init__(
        self,
        dim: int = 16,
        d_kv: int = 8,
        rank: int = 4,
        depth: int = 2,
    ) -> None:
        """``depth`` KV blocks sharing the basis ``U``."""
        super().__init__()
        U = _latent(rank, dim, seed=41)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 230 + i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


class _SliceUser(nn.Module):
    """``linear(x, W[i])`` — reads one slice of a shared stacked W."""

    def __init__(self, W: torch.Tensor, i: int) -> None:
        """Hold the shared Parameter and the slice index."""
        super().__init__()
        self.W = W
        self.i = i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through slice i."""
        return F.linear(x, self.W[self.i])


class _Stacked(nn.Module):
    """``y = Σ b_i(x)`` — siblings each slice one shared stacked W."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        """One stacked Parameter, sliced per block."""
        super().__init__()
        g = torch.Generator().manual_seed(240)
        self.W = nn.Parameter(
            torch.randn(
                depth, dim, dim, generator=g, dtype=torch.float64
            )
        )
        self.blocks = nn.ModuleList(
            _SliceUser(self.W, i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the members."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


class _WholeUser(nn.Module):
    """``x @ W.sum(0)`` — reads the shared leaf plainly, no view."""

    def __init__(self, W: torch.Tensor) -> None:
        """Hold the shared Parameter."""
        super().__init__()
        self.W = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through the stacked leaf reduced over dim 0."""
        return x @ self.W.sum(dim=0)


class _StackedMixed(nn.Module):
    """Two slicing members + one member reading W differently.

    The third member shares the leaf VALUE but holds no view site —
    the candidate must name only the slicing members.
    """

    def __init__(self, dim: int = 16) -> None:
        """Two slicers plus a plain whole-W reader."""
        super().__init__()
        g = torch.Generator().manual_seed(250)
        self.W = nn.Parameter(
            torch.randn(2, dim, dim, generator=g, dtype=torch.float64)
        )
        self.blocks = nn.ModuleList(
            [
                _SliceUser(self.W, 0),
                _SliceUser(self.W, 1),
                _WholeUser(self.W),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum all three members."""
        return self.blocks[0](x) + self.blocks[1](x) + self.blocks[2](x)


class _IntraSlice(nn.Module):
    """One block reading ``W[0]`` and ``W[1]`` — intra-leaf split."""

    def __init__(self, dim: int = 16) -> None:
        """A stacked Parameter inside the block itself."""
        super().__init__()
        g = torch.Generator().manual_seed(260)
        self.blocks = nn.ModuleList([self._Body(dim, g)])

    class _Body(nn.Module):
        """``linear(x, W[0]) + linear(x, W[1])``."""

        def __init__(self, dim: int, g: torch.Generator) -> None:
            """Hold the stacked Parameter."""
            super().__init__()
            self.W = nn.Parameter(
                torch.randn(
                    2, dim, dim, generator=g, dtype=torch.float64
                )
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sum two slices."""
            return F.linear(x, self.W[0]) + F.linear(x, self.W[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""
        return self.blocks[0](x)


class _TiedBlock(nn.Module):
    """``x @ w`` — reads a shared param plainly (no view site)."""

    def __init__(self, w: torch.Tensor) -> None:
        """Hold the shared weight."""
        super().__init__()
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ wᵀ``."""
        return x @ self.w.T


class _Tied(nn.Module):
    """Two blocks sharing one Parameter object, read plainly."""

    def __init__(self, dim: int = 16) -> None:
        """Both blocks reference the same Parameter."""
        super().__init__()
        g = torch.Generator().manual_seed(270)
        w = nn.Parameter(
            torch.randn(dim, dim, generator=g, dtype=torch.float64)
        )
        self.blocks = nn.ModuleList([_TiedBlock(w), _TiedBlock(w)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the members."""
        return self.blocks[0](x) + self.blocks[1](x)


class _CtxBlock(nn.Module):
    """``lin(x) * t`` — a two-arg block reading a bound table."""

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        """Own linear."""
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.lin.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x, t):
        """``lin(x) * t``."""
        return self.lin(x) * t


class _CtxStack(nn.Module):
    """Two blocks on one input sharing a bound-table context."""

    def __init__(self, dim: int = 16) -> None:
        """Register the shared table and the two blocks."""
        super().__init__()
        g = torch.Generator().manual_seed(290)
        self.register_buffer(
            "t", torch.randn(8, dim, generator=g, dtype=torch.float64)
        )
        self.blocks = nn.ModuleList(
            [_CtxBlock(dim, 291), _CtxBlock(dim, 292)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Both blocks read the same x AND the same bound table."""
        return self.blocks[0](x, self.t) + self.blocks[1](x, self.t)


class _FanoutCtx(nn.Module):
    """``b0(x)`` feeds ``b1``'s context slot AND ``b2``'s activation."""

    def __init__(self, dim: int = 16) -> None:
        """Producer + two consumers on one value."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Lin(dim, 280), self._Ctx(), _Lin(dim, 281)]
        )

    class _Ctx(nn.Module):
        """``lin(x) + table`` — reads the fan-out value as context."""

        def __init__(self) -> None:
            """Own linear plus the context read."""
            super().__init__()
            self.lin = nn.Linear(16, 16, bias=False).double()

        def forward(self, x, table):
            """``lin(x) + table``."""
            return self.lin(x) + table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed b0's output to b1's context and b2's activation."""
        t = self.blocks[0](x)
        return self.blocks[1](x, t) + self.blocks[2](t)


class _Kwarg(nn.Module):
    """A block only ever called with kwargs — never lifts."""

    def __init__(self, dim: int = 16) -> None:
        """Own a linear."""
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False).double()

    def forward(self, x=None):
        """Keyword-only forward."""
        return self.lin(x)


class _KwargStack(nn.Module):
    """``b0(x) -> b1(x=..)`` — the second block stays opaque."""

    def __init__(self, dim: int = 16) -> None:
        """Linear head plus kwargs-only tail."""
        super().__init__()
        self.blocks = nn.ModuleList([_Lin(dim, 290), _Kwarg(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain then kwargs call."""
        return self.blocks[1](x=self.blocks[0](x))


class _SkipMid(nn.Module):
    """``b0 -> b2`` — the middle block never runs."""

    def __init__(self, dim: int = 16) -> None:
        """Three children, one never executed."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 300 + i) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip blocks.1."""
        return self.blocks[2](self.blocks[0](x))


class _TupleOut(nn.Module):
    """A model returning a tuple — no tensor output object."""

    def __init__(self, dim: int = 16) -> None:
        """Two chained linears."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 310 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tuple."""
        y = self.blocks[1](self.blocks[0](x))
        return y, x


class _TupleBlock(nn.Module):
    """A block returning a TUPLE — its out_obj is not a tensor."""

    def __init__(self, dim: int = 16) -> None:
        """Own a linear."""
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False).double()

    def forward(self, x: torch.Tensor):
        """Return ``(lin(x), x)`` — a non-tensor output object."""
        return self.lin(x), x


class _TupleBlockStack(nn.Module):
    """``b0 -> b1`` where b0's output is a tuple — opaque producer."""

    def __init__(self, dim: int = 16) -> None:
        """Tuple block then a consumer of element 0."""
        super().__init__()
        self.blocks = nn.ModuleList([_TupleBlock(dim), _Lin(dim, 320)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the tuple's first element."""
        t = self.blocks[0](x)
        return self.blocks[1](t[0] + t[1])


class _TwoArg(nn.Module):
    """``forward(x, y)`` — two model-input leaves."""

    def __init__(self, dim: int = 16) -> None:
        """Two single-arg blocks."""
        super().__init__()
        self.blocks = nn.ModuleList(
            _Lin(dim, 320 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply b0 to x and b1 to y."""
        return self.blocks[0](x) + self.blocks[1](y)


class _PassThrough(nn.Module):
    """``return x`` — no blocks at all, input flows to output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity."""
        return x


# ---------------------------------------------------------------------------
#  Stage 1 — diagram construction
# ---------------------------------------------------------------------------


def test_lift_chain_shape():
    """A plain chain: block nodes, a model_input leaf, the output sink."""
    dg = _lift(_Chain(depth=3))
    assert repr(dg).startswith("Diagram(nodes=")
    kinds = {n.name: n.kind for n in dg.nodes}
    assert kinds["blocks.0"] == "block"
    assert kinds["<in:0>"] == "input"
    assert dg.node("<in:0>").role == "model_input"
    assert kinds["<output>"] == "output"
    # Three chain edges annotated by their boundary verdict + the two
    # boundary edges (leaf -> first block, last block -> output).
    assert len(dg.edges) == 4
    by_src = {e.src: e for e in dg.edges}
    assert by_src["blocks.0"].wire == "chain"
    assert by_src["blocks.1"].wire == "chain"
    assert by_src["<in:0>"].wire == ""
    assert by_src["blocks.2"].dsts == ("<output>",)
    assert by_src["blocks.2"].arity == 1
    # accessor coverage
    assert dg.sig("blocks.0") is not None
    assert dg.sig("<in:0>") is None
    assert dg.record("blocks.0").ir is not None
    assert dg.incoming("blocks.1")[0].src == "blocks.0"
    assert dg.outgoing("blocks.0")[0].dsts == ("blocks.1",)
    assert dg.consumers("<in:0>") == ("blocks.0",)
    assert dg.shared() == ()
    assert dg.activation_families() == []
    assert len(dg.wires) == 2


def test_lift_residual_leaves():
    """Residual sums are honest ``intermediate`` leaf nodes."""
    dg = _lift(_Resid(depth=2))
    leaves = [n for n in dg.nodes if n.kind == "input"]
    roles = {n.name: n.role for n in leaves}
    assert roles["<in:0>"] == "model_input"
    # x + b0(x) — the model-level sum no block produced.
    assert roles["<in:1>"] == "intermediate"
    assert roles["<in:2>"] == "intermediate"
    # the last stream value flows straight to the output sink
    out_edges = [e for e in dg.edges if "<output>" in e.dsts]
    assert len(out_edges) == 1 and out_edges[0].src == "<in:2>"
    assert [w.kind for w in dg.wires] == ["residual_wrapped"]


def test_lift_parallel_hyperedge():
    """Two blocks on one input object: ONE arity-2 hyperedge."""
    dg = _lift(_Parallel())
    fams = dg.activation_families()
    assert len(fams) == 1
    edge, members = fams[0]
    assert edge.arity == 2
    assert members == ("blocks.0", "blocks.1")
    assert edge.src == "<in:0>"
    assert len(dg.shared()) == 1
    assert dg.consumers("<in:0>") == ("blocks.0", "blocks.1")


def test_lift_multiinput_context():
    """Context ends carry ``const_table``; the bound table is a leaf.

    The shared ``self.t`` buffer is ONE hyperedge with two
    ``const_table`` ends — a multi-arity edge that is not an
    activation family.
    """
    dg = _lift(_CtxStack())
    n = dg.node("blocks.0")
    assert n.kind == "block"
    sig = n.sig
    assert [s.kind for s in sig.inputs] == ["activation", "const_table"]
    ctx = [
        en
        for e in dg.edges
        for en in e.ends
        if en.role == "const_table"
    ]
    assert len(ctx) == 2
    bound = [
        n for n in dg.nodes if n.kind == "input" and n.role == "bound"
    ]
    assert len(bound) == 1
    # the bound leaf's edge has arity 2 and no activation ends
    be = next(e for e in dg.edges if e.src == bound[0].name)
    assert be.arity == 2 and all(
        en.role == "const_table" for en in be.ends
    )
    # context ends never carry a wire verdict
    assert all(en.wire == "" for en in ctx)
    # the shared activation edge IS a family
    assert [m for _, m in dg.activation_families()] == [
        ("blocks.0", "blocks.1")
    ]


def test_lift_context_fanout():
    """One value feeding a state slot and an activation slot — a
    mixed-role arity-2 edge.

    The shared-input family on ``x`` (blocks.0 + blocks.1) is real,
    but ``blocks.0``'s own output edge carries its true per-end roles
    — it is not an activation family.
    """
    dg = _lift(_FanoutCtx())
    e = next(e for e in dg.edges if e.src == "blocks.0")
    assert e.arity == 2
    roles = {(en.node, en.role) for en in e.ends}
    assert ("blocks.2", "activation") in roles
    assert ("blocks.1", "state") in roles
    fams = dg.activation_families()
    assert [m for _, m in fams] == [("blocks.0", "blocks.1")]
    # the fanout edge itself is not an activation family
    assert all(e2 is not e for e2, _ in fams)


def test_lift_opaque_boundary():
    """Opaque nodes keep their place; their ends are ``unresolved``."""
    dg = _lift(_ResidOpaque())
    n = dg.node("blocks.1")
    assert n.kind == "opaque" and n.sig is None
    assert n.note  # the honest export decline
    # the opaque block DID run — its tensor arg is an edge end
    e = next(
        e
        for e in dg.edges
        if any(en.node == "blocks.1" for en in e.ends)
    )
    end = next(en for en in e.ends if en.node == "blocks.1")
    assert end.role == "unresolved"
    # its input is the model-level sum — an intermediate leaf
    assert dg.node(e.src).role == "intermediate"


def test_lift_output_sink_and_tuple():
    """The <output> sink appears iff a tensor model out object exists."""
    dg = _lift(_Chain(depth=2))
    out = dg.node("<output>")
    assert out.kind == "output"
    dg_t = _lift(_TupleOut())
    assert "<output>" not in {n.name for n in dg_t.nodes}
    assert all("<output>" not in e.dsts for e in dg_t.edges)


def test_lift_non_tensor_producer():
    """A block returning a tuple never becomes an edge's producer —
    the consumed elements are honest intermediates."""
    dg = _lift(_TupleBlockStack())
    # the tuple object produces no edge
    assert dg.outgoing("blocks.0") == ()
    e = next(
        e
        for e in dg.edges
        if any(en.node == "blocks.1" for en in e.ends)
    )
    # b1's input is t[0]+t[1] — a model-level intermediate leaf,
    # not the tuple-producing block
    assert e.src.startswith("<in:")
    assert dg.node(e.src).role == "intermediate"


def test_lift_passthrough():
    """No blocks at all: the input leaf feeds <output> directly."""
    dg = _lift(_PassThrough())
    names = {n.name for n in dg.nodes}
    assert names == {"<in:0>", "<output>"}
    e = dg.edges[0]
    assert e.src == "<in:0>" and e.dsts == ("<output>",)


def test_lift_two_model_args():
    """A tuple ``x`` marks BOTH arg tensors as model inputs."""
    model = _TwoArg().eval().double()
    dg = D.lift_diagram(
        model,
        (_x(), _x(seed=7)),
        source=TorchSource(),
        composer=TorchComposer(),
    )
    leaves = [n for n in dg.nodes if n.kind == "input"]
    assert sum(n.role == "model_input" for n in leaves) == 2


def test_diagram_of_graph_unknown_roles():
    """Without ``model`` provenance, leaves are honestly ``unknown``."""
    g = M.lift_graph(
        _Chain(depth=2).eval().double(),
        _x(),
        source=TorchSource(),
        composer=TorchComposer(),
    )
    dg = D.diagram_of_graph(g)  # no model, no x
    leaves = [n for n in dg.nodes if n.kind == "input"]
    assert all(n.role == "unknown" for n in leaves)
    # topology is still complete
    assert len(dg.edges) == 3


def test_diagram_of_graph_non_tensor_and_past_sig():
    """Ends skip non-tensor args; args past the signature are
    ``unresolved`` — hand-built record coverage."""
    x = _x()
    g = M.lift_graph(
        _Chain(depth=1).eval().double(),
        x,
        source=TorchSource(),
        composer=TorchComposer(),
    )
    rec = g.record("blocks.0")
    rec.in_objs = (*rec.in_objs, "not-a-tensor", x)
    dg = D.diagram_of_graph(g, model=_Chain(depth=1).double(), x=x)
    e = next(
        e
        for e in dg.edges
        if any(en.node == "blocks.0" for en in e.ends)
    )
    ends = [en for en in e.ends if en.node == "blocks.0"]
    # the str arg produced no end; the extra tensor arg is unresolved
    assert len(ends) == 2
    assert ends[1].role == "unresolved" and ends[1].pos == 2
    assert ends[1].wire == ""


def test_dedge_properties():
    """DEdge accessors: dsts / arity / wire on both wire kinds."""
    dg = _lift(_Chain(depth=2))
    e = next(e for e in dg.edges if e.src == "blocks.0")
    assert (
        e.dsts == ("blocks.1",) and e.arity == 1 and e.wire == "chain"
    )
    dg2 = _lift(_Parallel())
    e2 = next(e for e in dg2.edges if e.arity == 2)
    assert e2.wire == ""


# ---------------------------------------------------------------------------
#  merge_projs
# ---------------------------------------------------------------------------


def test_merge_candidates_family():
    """A shared-input family of projections emits one candidate."""
    dg = _lift(_Parallel())
    ms = D.MergeProjs().candidates(dg)
    assert len(ms) == 1
    m = ms[0]
    assert m.law == "merge_projs"
    assert m.nodes == ("blocks.0", "blocks.1")
    assert m.boundary == "family"
    assert m.reify.mode == "pair"
    assert "share one input hyperedge" in m.detail


def test_merge_candidates_intra():
    """A single block with >=2 fan-in projections is an intra match."""
    dg = _lift(_ParallelIntra())
    ms = D.MergeProjs().candidates(dg)
    intra = [m for m in ms if m.boundary == "intra"]
    assert len(intra) == 1
    assert intra[0].nodes == ("blocks.0",)
    assert "fan-in" in intra[0].detail


def test_merge_no_candidates_on_chain():
    """No shared leaf → no family; one-proj blocks → no intra."""
    assert D.MergeProjs().candidates(_lift(_Chain(depth=2))) == []


def test_merge_members_gate():
    """Families filtered by usability: calls!=1 drops the member."""
    dg = _lift(_DoubleCall())
    assert D.MergeProjs().candidates(dg) == []


def test_merge_proj_gate():
    """A family with <2 projecting members is no candidate."""
    dg = _lift(_ParallelNonProj())
    assert D.MergeProjs().candidates(dg) == []


def test_merge_evidence_gate():
    """Non-additive consumption fails the family evidence."""
    dg = _lift(_MulOut())
    assert D.MergeProjs().candidates(dg) == []


def test_merge_reify_grafted():
    """Two shared-input linears fuse — fp64 verified through the sink."""
    dg = _lift(_Parallel())
    ms = D.MergeProjs().candidates(dg)
    res = _reify(D.MergeProjs(), ms[0], dg)
    assert res["status"] == "grafted"
    assert set(res["reps"]) == {"blocks.0", "blocks.1"}
    assert res["rel_diff"] < 1e-12
    assert "joint" in res and "reified" in res


def test_merge_reify_intra():
    """The intra arm: two projections inside one block fuse."""
    dg = _lift(_ParallelIntra())
    ms = [
        m
        for m in D.MergeProjs().candidates(dg)
        if m.boundary == "intra"
    ]
    res = _reify(D.MergeProjs(), ms[0], dg)
    assert res["status"] == "grafted"
    assert set(res["reps"]) == {"blocks.0"}


def test_merge_reify_forced_wins():
    """Deep members: coordinated extraction beats the greedy form."""
    dg = _lift(_ParallelDeep())
    ms = D.MergeProjs().candidates(dg)
    res = _reify(D.MergeProjs(), ms[0], dg)
    # whether the greedy or the coordinated extraction won, the graft
    # is certified — and the stats name which path took it
    assert res["status"] == "grafted"
    assert "paired_extract" in res
    assert res["rel_diff"] < 1e-12


def test_merge_reify_no_groups():
    """Members project *different* operands — no shared group."""
    dg = _lift(_ParallelMismatch())
    ms = D.MergeProjs().candidates(dg)
    res = _reify(D.MergeProjs(), ms[0], dg)
    assert res["status"] == "declined"
    assert res["reason"] == "no shared-input projections"


def test_merge_reify_prep_declined():
    """A match on an opaque member declines at the prep stage."""
    dg = _lift(_ResidOpaque())
    m = MorphismMatch(
        law="merge_projs",
        nodes=("blocks.0", "blocks.1"),
        boundary="family",
        reify=ReifySpec(mode="pair", rules="compose"),
    )
    res = _reify(D.MergeProjs(), m, dg)
    assert res["status"] == "declined"


def test_merge_reify_forced_none(monkeypatch):
    """``extract_paired`` returning None falls back to greedy extract."""
    dg = _lift(_Parallel())
    ms = D.MergeProjs().candidates(dg)
    monkeypatch.setattr(
        EGraph, "extract_paired", lambda self, *a, **k: None
    )
    res = _reify(D.MergeProjs(), ms[0], dg)
    assert res["paired_extract"] is False
    assert res["status"] == "grafted"


def test_merge_reify_no_budget():
    """``symmetry_budget=None`` runs the re-saturation unbudgeted."""
    dg = _lift(_Parallel())
    ms = D.MergeProjs().candidates(dg)
    res = _reify(D.MergeProjs(), ms[0], dg, symmetry_budget=None)
    assert res["status"] == "grafted"


def test_merge_reify_no_improvement():
    """Under pure flops the paired form ties — honest decline.

    ``_ParallelDeep``'s members bury the shared-input linears under
    activations, so the compose recipe cannot factor the outer add —
    the only option left is the paired extract, whose fused GEMM does
    exactly the same MACs: equal flops → ``no_improvement``.
    """
    dg = _lift(_ParallelDeep())
    ms = D.MergeProjs().candidates(dg)
    res = _reify(D.MergeProjs(), ms[0], dg, cost_fn=flops_cost)
    assert res["status"] == "declined"
    assert res["reason"] == "no_improvement"


def test_merge_reify_verify_fail():
    """A failing sink verify declines — never an unchecked graft."""
    dg = _lift(_Parallel())
    ms = D.MergeProjs().candidates(dg)

    class _BadSink(TorchSink):
        def verify(self, *a, **k):
            return VerifyReport(passed=False, max_abs=1.0, max_rel=1.0)

    res = _reify(D.MergeProjs(), ms[0], dg, sink=_BadSink())
    assert res["status"] == "declined"
    assert "verify" in res["reason"]


# ---------------------------------------------------------------------------
#  factor_shared
# ---------------------------------------------------------------------------


def test_factor_cse_candidates():
    """The cse arm fires on the shared-norm recompute family."""
    dg = _lift(_SharedNormStack())
    ms = D.FactorShared().candidates(dg)
    assert len(ms) == 1
    m = ms[0]
    assert m.law == "factor_shared"
    assert m.nodes == ("blocks.0", "blocks.1")
    assert m.reify.mode == "cse"
    assert "cross_block_cse" in m.detail


def test_factor_kv_candidates():
    """The kv arm fires on the low-rank shared-input family — the
    cross-block family plus each member's intra k/v share."""
    dg = _lift(_KVStack())
    ms = D.FactorShared().candidates(dg)
    kv = [m for m in ms if m.reify.mode == "family"]
    assert len(kv) == 3
    assert all("kv_latent_share" in m.detail for m in kv)
    fam = [m for m in kv if m.boundary == "family"]
    assert len(fam) == 1 and fam[0].nodes == ("blocks.0", "blocks.1")


def test_factor_arm_switches():
    """The arms switch independently."""
    dg = _lift(_SharedNormStack())
    assert D.FactorShared(cse=False, kv=False).candidates(dg) == []
    assert D.FactorShared(cse=False).candidates(dg) == []
    dg_kv = _lift(_KVStack())
    only_cse = D.FactorShared(kv=False).candidates(dg_kv)
    assert all(m.reify.mode != "family" for m in only_cse)


def test_factor_reify_grafted_cse():
    """The cse reify grafts — one shared layer_norm computed once."""
    dg = _lift(_SharedNormStack())
    ms = D.FactorShared().candidates(dg)
    res = _reify(D.FactorShared(), ms[0], dg)
    assert res["status"] == "grafted"
    assert res["rel_diff"] < 1e-12
    assert set(res["reps"]) == {"blocks.0", "blocks.1"}


def test_factor_reify_grafted_kv():
    """The kv reify grafts the latent-share factorisation."""
    dg = _lift(_KVStack())
    ms = [
        m
        for m in D.FactorShared().candidates(dg)
        if m.reify.mode == "family"
    ]
    res = _reify(D.FactorShared(), ms[0], dg)
    assert res["status"] == "grafted"
    assert res["rel_diff"] < 1e-10


def test_factor_reify_declined():
    """A match on an opaque member declines through the machinery."""
    dg = _lift(_ResidOpaque())
    m = MorphismMatch(
        law="factor_shared",
        nodes=("blocks.0", "blocks.1"),
        boundary="family",
        reify=ReifySpec(mode="family", rules="compose"),
    )
    res = _reify(D.FactorShared(), m, dg)
    assert res["status"] == "declined"


def test_factor_move_protocol():
    """Moves satisfy the DiagramMove protocol."""
    for mv in D.DEFAULT_MOVES:
        assert isinstance(mv, DiagramMove)
        assert mv.name


# ---------------------------------------------------------------------------
#  reorder_compose
# ---------------------------------------------------------------------------


def test_reorder_chain_candidates():
    """A depth-4 chain yields every contiguous subwindow, wide-first."""
    dg = _lift(_Chain(depth=4))
    ms = D.ReorderCompose().candidates(dg)
    assert ms[0].nodes == (
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    )
    assert ms[0].boundary == "chain+chain+chain"
    node_sets = {m.nodes for m in ms}
    assert ("blocks.0", "blocks.1") in node_sets
    assert ("blocks.1", "blocks.2", "blocks.3") in node_sets
    # 6 windows: sizes 4/3/3/2/2/2
    assert len(ms) == 6
    assert all(m.reify.kinds for m in ms)


def test_reorder_chain_wrapped_close():
    """``chain_wrapped`` verdicts close runs — bare or after chains."""
    dg = _lift(_ChainWrapTail())
    ms = D.ReorderCompose().candidates(dg)
    kinds = {m.reify.kinds for m in ms}
    assert ("chain", "chain_wrapped") in kinds
    assert ("chain",) in kinds
    assert ("chain_wrapped",) in kinds
    # the wrap is only ever the LAST wire of a window
    assert all(
        m.reify.kinds[-1] == "chain_wrapped"
        or "chain_wrapped" not in m.reify.kinds
        for m in ms
    )


def test_reorder_bare_wrapped_pair():
    """``y = b0(x); return y + b1(y)`` — a one-wire wrapped window."""
    dg = _lift(_ChainWrap())
    ms = D.ReorderCompose().candidates(dg)
    assert len(ms) == 1
    assert ms[0].reify.kinds == ("chain_wrapped",)
    res = _reify(D.ReorderCompose(), ms[0], dg)
    assert res["status"] in {"grafted", "declined"}


def test_reorder_wrapped_opaque():
    """A wrapped wire to an opaque consumer is not composable."""

    class _WrapOpaque(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Lin(dim, 145), _DataDep()])

        def forward(self, x):
            y = self.blocks[0](x)
            return y + self.blocks[1](y)

    dg = _lift(_WrapOpaque())
    assert D.ReorderCompose().candidates(dg) == []


def test_reorder_residual_candidates():
    """Residual stream: the stream window plus the absorb pairs."""
    dg = _lift(_Resid(depth=3))
    ms = D.ReorderCompose().candidates(dg)
    assert ms[0].boundary == "residual_wrapped+residual_wrapped"
    assert {m.nodes for m in ms} == {
        ("blocks.0", "blocks.1", "blocks.2"),
        ("blocks.0", "blocks.1"),
        ("blocks.1", "blocks.2"),
    }
    assert all(m.reify.distribute for m in ms)


def test_reorder_residual_grafted():
    """The full residual window reifies + verifies fp64."""
    dg = _lift(_Resid(depth=3))
    ms = D.ReorderCompose().candidates(dg)
    res = _reify(D.ReorderCompose(), ms[0], dg)
    assert res["status"] == "grafted"
    assert res["rel_diff"] < 1e-12


def test_reorder_residual_close():
    """A wrapped block followed by a plain consumer: ``residual``."""
    dg = _lift(_ResidTail())
    ms = D.ReorderCompose().candidates(dg)
    assert any(m.reify.kinds == ("residual",) for m in ms)


def test_reorder_residual_run_close():
    """``rw`` run closed by a plain ``residual`` wire — the full
    window's kinds end ``residual``."""
    dg = _lift(_ResidClose())
    ms = D.ReorderCompose().candidates(dg)
    kinds = {m.reify.kinds for m in ms}
    assert ("residual_wrapped", "residual") in kinds
    assert ("residual_wrapped",) in kinds
    assert ("residual",) in kinds


def test_reorder_no_commute():
    """Projection-free residual blocks: no commute — no candidates."""
    dg = _lift(_ResidNoProj())
    assert D.ReorderCompose().candidates(dg) == []


def test_reorder_residual_opaque():
    """A residual wire into an opaque node is not legal material."""
    dg = _lift(_ResidOpaque())
    assert all(
        "blocks.1" not in m.nodes
        for m in D.ReorderCompose().candidates(dg)
    )


def test_reorder_no_wires():
    """No composable/residual wires — no candidates."""
    dg = _lift(_CtxStack())
    assert D.ReorderCompose().candidates(dg) == []


def test_reorder_pair_grafted():
    """The single-wire chain window is the OutInCompose contraction."""
    dg = _lift(_Chain(depth=2))
    ms = D.ReorderCompose().candidates(dg)
    res = _reify(D.ReorderCompose(), ms[0], dg)
    assert res["status"] == "grafted"
    assert res["rel_diff"] < 1e-12


# ---------------------------------------------------------------------------
#  split_leaf
# ---------------------------------------------------------------------------


def test_split_candidates_stacked():
    """Two members slicing one shared Parameter — one candidate."""
    dg = _lift(_Stacked())
    ms = D.SplitLeaf().candidates(dg)
    assert len(ms) == 1
    m = ms[0]
    assert m.law == "split_leaf"
    assert m.nodes == ("blocks.0", "blocks.1")
    assert m.boundary == "split"
    assert "view site" in m.detail
    assert m.reify.extra["leaf_names"] == {
        "blocks.0": ("p_w",),
        "blocks.1": ("p_w",),
    }


def test_split_candidates_mixed_member():
    """A member sharing the leaf value without a view site is left
    out of the candidate."""
    dg = _lift(_StackedMixed())
    ms = D.SplitLeaf().candidates(dg)
    assert len(ms) == 1
    assert ms[0].nodes == ("blocks.0", "blocks.1")


def test_split_candidates_intra():
    """One block slicing the same leaf twice splits in place."""
    dg = _lift(_IntraSlice())
    ms = D.SplitLeaf().candidates(dg)
    assert len(ms) == 1
    assert ms[0].nodes == ("blocks.0",)


def test_split_no_candidates():
    """No shared leaf values → no candidates (plain chain)."""
    dg = _lift(_Chain(depth=2))
    assert D.SplitLeaf().candidates(dg) == []


def test_split_no_view_sites():
    """A shared leaf read plainly by both members stays put."""
    dg = _lift(_Tied())
    assert D.SplitLeaf().candidates(dg) == []


def test_split_reify_grafted():
    """Both members' slices materialise — fp64 verified."""
    dg = _lift(_Stacked())
    ms = D.SplitLeaf().candidates(dg)
    res = _reify(D.SplitLeaf(), ms[0], dg)
    assert res["status"] == "grafted"
    assert set(res["reps"]) == {"blocks.0", "blocks.1"}
    members = res["members"]
    assert all(m["status"] == "grafted" for m in members.values())
    assert all(
        m["cost_after"] < m["cost_before"] for m in members.values()
    )
    # e2e check: the materialised modules are the originals
    torch.manual_seed(0)
    model = _Stacked().eval().double()
    sink = TorchSink()
    x = _x()
    with torch.no_grad():
        for name, rep in res["reps"].items():
            vr = sink.verify(
                dict(model.named_modules())[name],
                rep,
                (x,),
                rtol=1e-9,
            )
            assert vr.passed


def test_split_member_opaque():
    """A member whose record lost its IR declines honestly."""
    dg = _lift(_Stacked())
    dg.graph.record("blocks.1").ir = None
    ms = D.SplitLeaf().candidates(dg)
    assert ms == []  # the group degenerated — no sites left at all
    m = MorphismMatch(
        law="split_leaf",
        nodes=("blocks.0", "blocks.1"),
        boundary="split",
        reify=ReifySpec(
            mode="split",
            extra={"leaf_names": {"blocks.1": ("p_w",)}},
        ),
    )
    res = _reify(D.SplitLeaf(), m, dg)
    assert res["status"] == "declined"
    assert res["members"]["blocks.1"]["reason"] == "opaque node"
    assert (
        res["members"]["blocks.0"]["reason"]
        == "no view sites on the shared leaf"
    )


def test_split_no_leaf_names():
    """A crafted match with no leaf_names declines per member."""
    dg = _lift(_Stacked())
    m = MorphismMatch(
        law="split_leaf",
        nodes=("blocks.0",),
        boundary="split",
        reify=ReifySpec(mode="split"),  # no extra at all
    )
    res = _reify(D.SplitLeaf(), m, dg)
    assert res["status"] == "declined"
    assert res["reason"] == "no member split"
    assert (
        res["members"]["blocks.0"]["reason"]
        == "no view sites on the shared leaf"
    )


def test_split_unmaterialisable():
    """A site whose leaf value vanished cannot materialise."""
    dg = _lift(_Stacked())
    del dg.graph.record("blocks.0").leaves["p_w"]
    m = MorphismMatch(
        law="split_leaf",
        nodes=("blocks.0",),
        boundary="split",
        reify=ReifySpec(
            mode="split", extra={"leaf_names": {"blocks.0": ("p_w",)}}
        ),
    )
    res = _reify(D.SplitLeaf(), m, dg)
    assert res["status"] == "declined"
    assert (
        res["members"]["blocks.0"]["reason"]
        == "no materialisable view site"
    )


def test_split_no_improvement():
    """A flat cost model sees no op-count win — the member declines."""
    dg = _lift(_Stacked())
    ms = D.SplitLeaf().candidates(dg)
    res = _reify(D.SplitLeaf(), ms[0], dg, cost_fn=lambda t: 0.0)
    assert res["status"] == "declined"
    assert all(
        m["reason"] == "no_improvement" for m in res["members"].values()
    )


def test_split_verify_fail():
    """A failing sink declines the member — partial grafts stay exact."""
    dg = _lift(_Stacked())
    ms = D.SplitLeaf().candidates(dg)

    class _BadSink(TorchSink):
        def verify(self, *a, **k):
            return VerifyReport(passed=False, max_abs=1.0, max_rel=1.0)

    res = _reify(D.SplitLeaf(), ms[0], dg, sink=_BadSink())
    assert res["status"] == "declined"
    assert any(
        "verify failed" in m["reason"] for m in res["members"].values()
    )


def test_split_no_inputs():
    """A member IR with no input vars cannot lower — decline."""
    dg = _lift(_Stacked())
    rec = dg.graph.record("blocks.0")
    rec.ir = IR(
        root=rec.ir.root,
        inputs=[],
        input_names=set(),
        params=rec.ir.params,
    )
    m = MorphismMatch(
        law="split_leaf",
        nodes=("blocks.0",),
        boundary="split",
        reify=ReifySpec(
            mode="split", extra={"leaf_names": {"blocks.0": ("p_w",)}}
        ),
    )
    res = _reify(D.SplitLeaf(), m, dg)
    assert res["status"] == "declined"
    assert res["members"]["blocks.0"]["reason"] == "no input vars"


def test_split_sig_none_path():
    """A member with IR but no signature still lowers (act=0)."""
    dg = _lift(_Stacked())
    # strip the signature on the *morphism* node — split only needs
    # the record's IR and leaf table
    graph = dg.graph
    nodes = [
        M.MorphismNode(
            n.name, None if n.name == "blocks.0" else n.sig, n.opaque
        )
        for n in graph.nodes
    ]
    g2 = M.MorphismGraph(nodes, list(graph.wires), graph._records)
    d2 = D.diagram_of_graph(g2)
    ms = D.SplitLeaf().candidates(d2)
    assert ms and ms[0].nodes == ("blocks.0", "blocks.1")
    res = _reify(D.SplitLeaf(), ms[0], d2)
    assert res["members"]["blocks.0"]["status"] == "grafted"


def test_split_sites_unit():
    """_split_sites guards: view ops only, arg0 must be the Param."""
    w = Param("p_w", TensorType((2, 4, 4)))
    x = Var("x", TensorType((8, 4)))
    site = Op.make("select", w, dim=0, index=1)
    on_var = Op.make("select", x, dim=0, index=1)
    other = Op.make("reshape", w, shape=(8, 4))
    body = Op.make("add", site, on_var, other)
    names = frozenset({"p_w"})
    sites = D._split_sites(body, names)
    assert sites == [site]
    # Param arg under a different name — not the shared leaf
    body2 = Op.make(
        "select",
        Param("p_other", TensorType((2, 4, 4))),
        dim=0,
        index=0,
    )
    assert D._split_sites(body2, names) == []
    # a select with no args at all
    assert D._split_sites(Op.make("select"), names) == []


def test_slice_value_unit():
    """_slice_value covers each view spelling + honest ``None``."""
    t = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    p = Param("p", TensorType((2, 3, 4)))
    sel = Op.make("select", p, dim=0, index=1)
    assert torch.equal(D._slice_value(sel, t), t[1])
    nar = Op.make("narrow", p, dim=1, start=1, length=2)
    assert torch.equal(D._slice_value(nar, t), t[:, 1:3])
    slc = Op.make("slice", p, dim=2, start=0, end=2, step=1)
    assert torch.equal(D._slice_value(slc, t), t[:, :, 0:2])
    get = Op.make("getitem", p, index=0)
    assert torch.equal(D._slice_value(get, t), t[0])
    # a tensor-like without .select falls back to indexing
    fake = np.arange(24.0).reshape(2, 3, 4)
    assert np.array_equal(D._slice_value(sel, fake), fake[1])
    # narrow without .narrow → None; exceptions → None; unknown op → None
    assert D._slice_value(nar, fake) is None
    bad = Op.make("select", p, dim=0, index=9)
    assert D._slice_value(bad, t) is None
    assert D._slice_value(Op.make("add", p, p), t) is None


def test_materialise_unit():
    """_materialise: tensors get an owning copy; plain values pass."""
    t = torch.randn(4, dtype=torch.float64)
    v = D._materialise(t.select(0, 1))
    assert v.shape == () and float(v) == float(t[1])
    assert v.data_ptr() != t.data_ptr() or t[1].is_contiguous()
    assert D._materialise(5) == 5


def test_leaf_groups():
    """The leaf-table grouping: shared object vs equal values."""
    dg = _lift(_Stacked())
    groups = D._leaf_groups(dg.graph)
    assert len(groups) == 1
    g = groups[0]
    assert set(g) == {"blocks.0", "blocks.1"}
    assert g["blocks.0"][1] == ["p_w"]
    # non-tensor leaf entries are skipped; ir-less records are skipped
    dg2 = _lift(_Stacked())
    rec0 = dg2.graph.record("blocks.0")
    rec0.leaves["extra"] = 1.5  # not a tensor
    dg2.graph.record("blocks.1").ir = None
    groups2 = D._leaf_groups(dg2.graph)
    assert len(groups2) == 1 and set(groups2[0]) == {"blocks.0"}


# ---------------------------------------------------------------------------
#  The driver — ContractionSearch / optimize_diagram
# ---------------------------------------------------------------------------


def test_driver_family_graft():
    """End-to-end: the merge fires, split is consumed, e2e verifies."""
    torch.manual_seed(0)
    model = _Parallel().eval().double()
    mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=ContractionSearch(optimize_rest=False)
    )
    assert stats["contraction"] is True
    assert stats["n_hyperedges"] == 1
    assert stats["move_fires"] == {"merge_projs": 1}
    assert stats["n_rewritten"] == 2
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12
    with torch.no_grad():
        d = (model(_x()) - mod(_x())).abs().max().item()
    assert d < 1e-12


def test_driver_claims_and_skips():
    """Widest candidates claim their nodes — later overlaps skip."""
    torch.manual_seed(0)
    model = _Stacked().eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=ContractionSearch(optimize_rest=False)
    )
    fires = stats["move_fires"]
    assert fires  # something grafted
    skips = [
        v for v in stats["moves"].values() if v["status"] == "skipped"
    ]
    assert all(v["reason"] == "node already rewritten" for v in skips)
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12


def test_driver_split_end_to_end():
    """The stacked-W model: split_leaf delivers + verifies e2e."""
    torch.manual_seed(0)
    model = _Stacked().eval().double()
    strat = ContractionSearch(
        moves=(D.SplitLeaf(),), optimize_rest=False
    )
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=strat
    )
    assert stats["move_fires"] == {"split_leaf": 1}
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12


def test_driver_optimize_rest():
    """Unconsumed blocks fall back to the per-block search."""
    torch.manual_seed(0)

    class _Mix(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_Lin(dim, 330), _Lin(dim, 331), _DataDep()]
            )

        def forward(self, x):
            # shared-input pair, then an opaque tail (never optimises)
            return self.blocks[2](self.blocks[0](x) + self.blocks[1](x))

    model = _Mix().eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=ContractionSearch()
    )
    reps = stats["blocks"]
    assert reps["blocks.0"]["status"] == "rewritten"
    assert reps["blocks.1"]["status"] == "rewritten"
    # the opaque tail has no IR — the fallback reports it skipped
    assert reps["blocks.2"]["status"] == "skipped"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_driver_no_composer():
    """No composer port → an honest TypeError."""
    opt = Optimizer(
        source=TorchSource(),
        sink=TorchSink(),
        meter=TorchBackend().meter,
    )
    with pytest.raises(TypeError, match="Composer"):
        opt.optimize(
            _Chain(depth=2).eval().double(),
            _x(),
            strategy=ContractionSearch(),
        )


def test_driver_decline_then_next():
    """A declined move does not consume nodes — later candidates run."""

    class _Decliner:
        name = "decliner"

        def candidates(self, diagram):
            return [
                MorphismMatch(
                    law="decliner",
                    nodes=("blocks.0", "blocks.1", "blocks.2"),
                    boundary="x",
                    reify=ReifySpec(mode="pair", rules="compose"),
                )
            ]

        def reify(self, *a, **k):
            return {"status": "declined", "reason": "nope"}

    torch.manual_seed(0)
    model = _Chain(depth=3).eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        _x(),
        strategy=ContractionSearch(
            moves=(_Decliner(), D.ReorderCompose()), optimize_rest=False
        ),
    )
    by_law = {k.split(":")[0]: v for k, v in stats["moves"].items()}
    assert by_law["decliner"]["status"] == "declined"
    assert by_law["decliner"]["reason"] == "nope"
    reorder = [
        v
        for k, v in stats["moves"].items()
        if k.startswith("reorder_compose")
    ]
    assert any(v["status"] == "grafted" for v in reorder)


def test_driver_rest_optimized():
    """An unconsumed lifted block gets the per-block ``optimized``."""

    class _ParTail(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_Lin(dim, 400), _Lin(dim, 401), _Lin(dim, 402)]
            )

        def forward(self, x):
            return self.blocks[2](self.blocks[0](x) + self.blocks[1](x))

    torch.manual_seed(0)
    model = _ParTail().eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=ContractionSearch()
    )
    assert stats["blocks"]["blocks.2"]["status"] == "optimized"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_driver_move_error():
    """A move whose reify raises is recorded as an ``error`` decline."""
    torch.manual_seed(0)

    class _Boom:
        name = "boom"

        def candidates(self, diagram):
            return [
                MorphismMatch(
                    law="boom",
                    nodes=("blocks.0",),
                    boundary="x",
                    reify=ReifySpec(mode="intra", rules="compose"),
                )
            ]

        def reify(self, *a, **k):
            raise RuntimeError("bang")

    model = _Chain(depth=2).eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        _x(),
        strategy=ContractionSearch(
            moves=(_Boom(),), optimize_rest=False
        ),
    )
    (v,) = stats["moves"].values()
    assert v["status"] == "declined"
    assert v["reason"] == "error"
    assert "bang" in v["error"]


def test_driver_rest_verify_fail():
    """A block whose fallback verify fails is reported, not grafted."""
    torch.manual_seed(0)
    model = _ResidOpaque().eval().double()
    opt = Optimizer(backend=TorchBackend())

    class _BadSink(TorchSink):
        def verify(self, *a, **k):
            return VerifyReport(passed=False, max_abs=1.0, max_rel=1.0)

    opt.sink = _BadSink()
    _mod, stats = opt.optimize(
        model, _x(), strategy=ContractionSearch()
    )
    fails = [
        v for v in stats["blocks"].values() if v["status"] == "failed"
    ]
    assert fails and "verify" in fails[0]["reason"]


def test_driver_rest_exception():
    """A search exception on a fallback block records ``failed``."""

    class _BoomOpt(Optimizer):
        def search(self, *a, **k):
            raise RuntimeError("boom")

    torch.manual_seed(0)
    model = _ResidOpaque().eval().double()
    opt = _BoomOpt(backend=TorchBackend())
    _mod, stats = opt.optimize(
        model, _x(), strategy=ContractionSearch()
    )
    fails = [
        v
        for v in stats["blocks"].values()
        if v["status"] == "failed" and "boom" in v.get("error", "")
    ]
    assert fails


def test_driver_in_place_fallback():
    """A composer that cannot clone delivers in-place and says so."""
    torch.manual_seed(0)

    class _NoClone(TorchComposer):
        def clone_sharing(self, model):
            raise RuntimeError("no clone")

    model = _Parallel().eval().double()
    opt = Optimizer(
        source=TorchSource(),
        sink=TorchSink(),
        composer=_NoClone(),
        meter=TorchBackend().meter,
    )
    mod, stats = opt.optimize(
        model, _x(), strategy=ContractionSearch(optimize_rest=False)
    )
    assert stats["in_place"] is True
    assert stats["end_to_end"]["skipped"] == "in_place"
    assert mod is model


def test_driver_e2e_verify_error():
    """An e2e verify exception is recorded, not raised."""
    torch.manual_seed(0)

    class _ExplodeSink(TorchSink):
        def verify(self, ref, opt, args, **k):
            if ref.__class__ is _Parallel:
                raise RuntimeError("e2e boom")
            return super().verify(ref, opt, args, **k)

    model = _Parallel().eval().double()
    opt = Optimizer(
        source=TorchSource(),
        sink=_ExplodeSink(),
        composer=TorchComposer(),
        meter=TorchBackend().meter,
    )
    _mod, stats = opt.optimize(
        model, _x(), strategy=ContractionSearch(optimize_rest=False)
    )
    assert "e2e boom" in stats["end_to_end"]["error"]


def test_driver_verbose(caplog):
    """Verbose mode logs the diagram shape and candidate claims."""
    import logging as _logging

    torch.manual_seed(0)
    model = _Parallel().eval().double()
    with caplog.at_level(_logging.INFO, "catopt_orchestrator.diagram"):
        Optimizer(backend=TorchBackend()).optimize(
            model,
            _x(),
            strategy=ContractionSearch(optimize_rest=False),
            verbose=True,
        )
    recs = [r.getMessage() for r in caplog.records]
    assert any("[Diagram]" in r and "candidates" in r for r in recs)
    assert any("grafted" in r for r in recs)


def test_driver_stats_shape():
    """Stats carry the diagram: nodes/edges/hyperedges/wires/sigs."""
    torch.manual_seed(0)
    model = _Resid(depth=2).eval().double()
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=ContractionSearch(optimize_rest=False)
    )
    assert stats["n_blocks"] == 2
    assert stats["n_edges"] == 3
    assert set(stats) >= {
        "n_nodes",
        "n_hyperedges",
        "n_lifted",
        "sigs",
        "wires",
        "edges",
        "moves",
        "blocks",
        "wall_time_s",
    }


def test_optimize_diagram_entry():
    """The function entry: backend bundle and explicit ports."""
    torch.manual_seed(0)
    r = optimize_diagram(
        _Parallel().eval().double(),
        _x(),
        backend=TorchBackend(),
        strategy=ContractionSearch(optimize_rest=False),
    )
    assert r.stats["contraction"] is True
    torch.manual_seed(0)
    r2 = optimize_diagram(
        _Chain(depth=2).eval().double(),
        _x(),
        source=TorchSource(),
        sink=TorchSink(),
        composer=TorchComposer(),
        meter=TorchBackend().meter,
        strategy=ContractionSearch(optimize_rest=False),
        cost_fn=flops_cost,
    )
    assert r2.stats["contraction"] is True


def test_move_set_default_and_override():
    """DEFAULT_MOVES is the documented claim order; moves override."""
    names = [m.name for m in D.DEFAULT_MOVES]
    assert names == [
        "factor_shared",
        "merge_projs",
        "reorder_compose",
        "split_leaf",
    ]
    strat = ContractionSearch(moves=(D.SplitLeaf(),))
    assert [m.name for m in strat.moves] == ["split_leaf"]


def test_public_exports():
    """The package __init__ exposes the diagram surface."""
    import catopt_orchestrator as co

    for name in (
        "DEFAULT_MOVES",
        "ContractionSearch",
        "DEdge",
        "DEnd",
        "DNode",
        "Diagram",
        "DiagramMove",
        "FactorShared",
        "MergeProjs",
        "ReorderCompose",
        "SplitLeaf",
        "diagram_of_graph",
        "lift_diagram",
        "optimize_diagram",
    ):
        assert getattr(co, name) is getattr(D, name)
