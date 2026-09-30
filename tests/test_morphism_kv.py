"""KV-latent morphism law tests — shared right-factors, verified reify.

Plan-0011 follow-on.  Pins the ``kv_latent_share`` law end to end:
detection of same-input block families carrying ``k_proj``/``v_proj``
weights, the Gram-Schmidt common-factor certification, the wiring
evidence (additive output consumption + fan-out guard), and the
reified programs — every grafted rewrite verified fp64 against the
original.
"""

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_orchestrator import (
    MorphismLaw,
    MorphismSearch,
    Optimizer,
    optimize_morphisms,
)
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _latent(rank: int, dim: int, seed: int = 0) -> torch.Tensor:
    """A shared latent basis — fp64 rows of R^{dim}."""
    return torch.randn(
        rank,
        dim,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


class _KVBlock(nn.Module):
    """Attention-ish block: ``sdpa(q,k,v) → out_proj`` on ``x``.

    ``k_proj`` / ``v_proj`` are built low-rank through the shared
    basis ``U``; ``q_proj`` stays full-rank (it is not a KV token).
    The SDPA nonlinearity keeps the projections from folding into
    ``out_proj`` — the latent share is the *only* restructure left.
    """

    def __init__(
        self,
        dim: int,
        d_kv: int,
        U: torch.Tensor,
        seed: int,
        *,
        inexact: float = 0.0,
        full_rank: bool = False,
    ) -> None:
        """Initialise projections; k/v factored through ``U``."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Linear(dim, d_kv, bias=False)
        self.k_proj = nn.Linear(dim, d_kv, bias=False)
        self.v_proj = nn.Linear(dim, d_kv, bias=False)
        self.out_proj = nn.Linear(d_kv, dim, bias=False)
        self.double()
        if not full_rank:
            dk = torch.randn(
                d_kv, U.shape[0], generator=g, dtype=torch.float64
            )
            dv = torch.randn(
                d_kv, U.shape[0], generator=g, dtype=torch.float64
            )
            with torch.no_grad():
                self.k_proj.weight.copy_(
                    dk @ U
                    + inexact
                    * torch.randn(
                        d_kv, dim, generator=g, dtype=torch.float64
                    )
                )
                self.v_proj.weight.copy_(
                    dv @ U
                    + inexact
                    * torch.randn(
                        d_kv, dim, generator=g, dtype=torch.float64
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run sdpa(q,k,v) then the output projection."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return self.out_proj(o)


class _SharedKVStack(nn.Module):
    """``y = Σ b_i(x)`` — parallel consumers of one input."""

    def __init__(
        self,
        dim: int = 16,
        d_kv: int = 8,
        rank: int = 4,
        depth: int = 2,
        **kw,
    ) -> None:
        """Build ``depth`` low-rank KV blocks sharing one basis."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 10 + i, **kw) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


class _ResidualKVStack(nn.Module):
    """``y = x + (b0(x) + b1(x))`` — parallel sum inside a residual."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """Two shared-input KV blocks on the residual stream."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 20 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual around the member sum."""
        return x + (self.blocks[0](x) + self.blocks[1](x))


class _DownstreamKV(nn.Module):
    """``y = c(b0(x) + b1(x))`` — the family sum feeds another block."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """Two KV blocks plus a downstream consumer block."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 30 + i) for i in range(2)
        )
        self.c = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the member sum downstream."""
        return self.c(self.blocks[0](x) + self.blocks[1](x))


class _MulKVStack(nn.Module):
    """``y = b0(x) * b1(x)`` — shared input, non-additive outputs."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """Two shared-input KV blocks, multiplied."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 40 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the member outputs — no additive slot."""
        return self.blocks[0](x) * self.blocks[1](x)


class _FanoutKV(nn.Module):
    """One member's output escapes to a non-additive consumer."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """Two shared-input KV blocks; b1's output feeds c."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 50 + i) for i in range(2)
        )
        self.c = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b1's output fans out: sum AND a separate consumer."""
        b1 = self.blocks[1](x)
        return self.blocks[0](x) + b1 + self.c(b1)


class _SequentialKV(nn.Module):
    """``x = x + b_i(x)`` — each block sees a DIFFERENT stream value."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """Residual-chained KV blocks (no shared input)."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 60 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain the blocks on the evolving stream."""
        for b in self.blocks:
            x = x + b(x)
        return x


class _SingleKV(nn.Module):
    """One low-rank KV block — the intra-block MLA fold."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """One block whose k/v share both data and right-factor."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_KVBlock(dim, d_kv, _latent(rank, dim), 70)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single block."""
        return self.blocks[0](x)


class _QKVBlock(nn.Module):
    """Fused-qkv block: one ``qkv_proj`` weight (3·d_kv, d_in).

    The k/v row-sections (middle/last thirds — the SDPA convention)
    are built through the shared basis ``U``; the q section is
    full-rank.
    """

    def __init__(
        self,
        dim: int,
        d_kv: int,
        U: torch.Tensor,
        seed: int,
        *,
        bias: bool = False,
    ) -> None:
        """Initialise the fused projection through ``U``."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv_proj = nn.Linear(dim, 3 * d_kv, bias=bias)
        self.out_proj = nn.Linear(d_kv, dim, bias=False)
        self.d_kv = d_kv
        self.double()
        dk = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        dv = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        with torch.no_grad():
            self.qkv_proj.weight[d_kv : 2 * d_kv].copy_(dk @ U)
            self.qkv_proj.weight[2 * d_kv :].copy_(dv @ U)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused projection → sdpa → out."""
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return self.out_proj(o)


class _SingleQKV(nn.Module):
    """One fused-qkv block — intra MLA fold through the sections."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4, **kw
    ) -> None:
        """One block whose qkv k/v sections share a right-factor."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_QKVBlock(dim, d_kv, _latent(rank, dim), 80, **kw)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single block."""
        return self.blocks[0](x)


class _MatmulKVBlock(nn.Module):
    """KV block spelled with raw ``x @ W`` matmuls (right weights)."""

    def __init__(
        self, dim: int, d_kv: int, U: torch.Tensor, seed: int
    ) -> None:
        """Initialise right-side matmul weights through ``U``."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Parameter(
            torch.randn(dim, d_kv, generator=g, dtype=torch.float64)
        )
        dpk = torch.randn(
            U.shape[0], d_kv, generator=g, dtype=torch.float64
        )
        dpv = torch.randn(
            U.shape[0], d_kv, generator=g, dtype=torch.float64
        )
        self.k_proj = nn.Parameter(U.T @ dpk)
        self.v_proj = nn.Parameter(U.T @ dpv)
        self.out_proj = nn.Parameter(
            torch.randn(d_kv, dim, generator=g, dtype=torch.float64)
        )
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W`` spellings — the matmul-right orientation."""
        q = x @ self.q_proj
        k = x @ self.k_proj
        v = x @ self.v_proj
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return o @ self.out_proj


class _SingleMatmulKV(nn.Module):
    """One matmul-spelled block — the right-weight orientation."""

    def __init__(
        self, dim: int = 16, d_kv: int = 8, rank: int = 4
    ) -> None:
        """One block whose right-side k/v weights share a factor."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MatmulKVBlock(dim, d_kv, _latent(rank, dim), 90)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single block."""
        return self.blocks[0](x)


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    """Lift *model* through the torch ports."""
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    """fp64 probe input."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*dims, generator=g, dtype=torch.float64)


def _reify(match, graph, **kw):
    """Call the family reify with the standard knobs."""
    args = dict(
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    args.update(kw)
    return K._reify_family(match, graph, **args)


# ---------------------------------------------------------------------------
#  Units — sites, grouping, value helpers
# ---------------------------------------------------------------------------


def test_kv_named():
    """Token substring matching."""
    assert K._kv_named("p_attn_k_proj_weight", ("k_proj",))
    assert not K._kv_named("p_q_proj_weight", ("k_proj", "v_proj"))


def test_kv_proj_sites_orientations():
    """linear / matmul-right / qkv sites; everything else skipped."""
    x = Var("x", TensorType((8, 16)))
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wv = Param("p_v_proj_weight", TensorType((8, 16)))
    wq = Param("p_q_proj_weight", TensorType((8, 16)))
    wqkv = Param("p_qkv_proj_weight", TensorType((24, 16)))
    wqkv_bad = Param("p_qkv_proj_weight", TensorType((22, 16)))
    wqkv_noshp = Param("p_qkv_proj_weight", TensorType((None, 16)))
    wp = Param("p_w", TensorType((16, 8)))
    wb = Param("p_b", TensorType((8,)))
    expr = Op.make("add", wk, wv)

    # linear sites: kv tokens only, Param weights only.
    body = Op.make(
        "add",
        Op.make("linear", x, wk),
        Op.make("linear", x, wv, wb),
    )
    ss = K._kv_proj_sites(body, "b0", K._KV_TOKENS, ())
    assert len(ss) == 2 and all(s.orient == "linear" for s in ss)
    assert all(s.data is x for s in ss)

    # q_proj is not a token; compound weight terms are not sites.
    body = Op.make(
        "add",
        Op.make("linear", x, wq),
        Op.make("linear", x, expr),
    )
    assert K._kv_proj_sites(body, "b0", K._KV_TOKENS, ()) == []

    # param-only data operand is not a projection site.
    body = Op.make("linear", Param("p_y", TensorType((8, 16))), wk)
    assert K._kv_proj_sites(body, "b0", K._KV_TOKENS, ()) == []

    # qkv: divisible thirds gives two section sites; bad rows skipped;
    # unknown rows skipped.
    body = Op.make("linear", x, wqkv)
    ss = K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS)
    assert [s.span for s in ss] == [(8, 8), (16, 8)]
    assert all(s.orient == "qkv" for s in ss)
    body = Op.make("linear", x, wqkv_bad)
    assert K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS) == []
    body = Op.make("linear", x, wqkv_noshp)
    assert K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS) == []

    # matmul: right-side Param weight is a site; left-side is not.
    body = Op.make("matmul", x, wp)
    ss = K._kv_proj_sites(body, "b0", ("w",), ())
    assert [s.orient for s in ss] == ["matmul_r"]
    body = Op.make("matmul", wp, x)
    assert K._kv_proj_sites(body, "b0", ("w",), ()) == []
    body = Op.make("matmul", x, Var("y", TensorType((16, 8))))
    assert K._kv_proj_sites(body, "b0", ("y",), ()) == []

    # odd arities pass through untouched.
    odd_lin = Op("linear", (x,), {})
    odd_mm = Op("matmul", (x, wp, wb), {})
    assert K._kv_proj_sites(odd_lin, "b0", K._KV_TOKENS, ()) == []
    assert K._kv_proj_sites(odd_mm, "b0", ("w",), ()) == []


def test_latent_groups():
    """Sites group by their shared data term; singletons drop."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wv = Param("p_v_proj_weight", TensorType((8, 16)))
    w2 = Param("p_k2_proj_weight", TensorType((8, 16)))
    body = Op.make(
        "add",
        Op.make("linear", x, wk),
        Op.make(
            "add", Op.make("linear", x, wv), Op.make("linear", y, w2)
        ),
    )
    groups = K._latent_groups({"b": body}, K._KV_TOKENS, ())
    # The y-sited projection is a singleton — only the x group stays.
    ((data, ss),) = groups.items()
    assert data is x and len(ss) == 2


def test_input_families():
    """Only same-input-object, single-call, non-opaque blocks group."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    fams = K._input_families(g)
    assert sorted(fams.values()) == [["blocks.0", "blocks.1"]]

    # Sequential blocks see different stream values — no family.
    g2 = _lift(_SequentialKV().eval().double(), _x())
    assert K._input_families(g2) == {}

    # A block called twice cannot join a consumable family.
    class Twice(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_KVBlock(16, 8, _latent(4, 16), 1)]
            )

        def forward(self, x):
            return self.blocks[0](x) + self.blocks[0](x)

    g3 = _lift(Twice().eval().double(), _x())
    assert K._input_families(g3) == {}


def test_vals_close():
    """Exact-or-tolerant value equality."""
    a = torch.ones(4, 4, dtype=torch.float64)
    assert K._vals_close(a, a)
    assert not K._vals_close(a, 5)  # non-tensor
    assert not K._vals_close(a, torch.ones(2, 2))
    assert K._vals_close(a, a.clone())
    near = a.clone()
    near[0, 0] += 1e-9  # within tolerance
    assert K._vals_close(a, near)
    far = a.clone()
    far[0, 0] += 1e-2
    assert not K._vals_close(a, far)


def test_sum_vals_and_detach_fnorm():
    """Value-sum fold + the small numeric helpers."""
    assert K._sum_vals([]) is None
    assert K._sum_vals([None, 3]) is None
    a = torch.ones(4, dtype=torch.float64)
    assert K._sum_vals([a]) is a
    s = K._sum_vals([a, a, None])
    assert torch.equal(s, a * 2)
    t = torch.ones(2, 2, requires_grad=True)
    assert K._detach(t).requires_grad is False
    assert K._detach(5) == 5
    assert K._fnorm(torch.ones(2, 2)) == 2.0


def test_gs_row_basis_and_stack_cols():
    """Gram-Schmidt coverage: exact low-rank, full-rank, zero."""
    U = _latent(3, 12)
    d = torch.randn(5, 3, dtype=torch.float64)
    w = d @ U
    basis = K._gs_row_basis([w], 1e-8)
    assert len(basis) == 3
    Ut = K._stack_cols(basis)
    assert tuple(Ut.shape) == (12, 3)
    resid = w - (w @ Ut) @ Ut.T
    assert K._fnorm(resid) < 1e-10
    # Full-rank input: the whole row space is recovered.
    full = torch.randn(6, 12, dtype=torch.float64)
    assert len(K._gs_row_basis([full], 1e-8)) == 6
    # An all-zero matrix contributes nothing — empty basis.
    assert (
        K._gs_row_basis([torch.zeros(4, 8, dtype=torch.float64)], 1e-8)
        == []
    )


def test_factor_sites_paths():
    """The certification: every decline path plus the ok triple."""
    x = Var("x", TensorType((8, 16)))
    U = _latent(3, 16)
    dk = torch.randn(8, 3, dtype=torch.float64)
    dv = torch.randn(8, 3, dtype=torch.float64)
    wk, wv = dk @ U, dv @ U

    def site(name, orient="linear", span=None, data=None):
        return K._KVSite(
            "b",
            Op.make("linear", x, Param(name, TensorType((8, 16)))),
            data if data is not None else x,
            Param(name, TensorType((8, 16))),
            orient,
            span,
        )

    skew = torch.zeros(16, 16, dtype=torch.float64)
    skew[0] = 1e4 * torch.randn(16, dtype=torch.float64)
    for i in range(1, 16):
        skew[i] = 5e-5 * torch.randn(16, dtype=torch.float64)
    leaves = {
        "p_k": wk,
        "p_v": wv,
        "p_full": torch.randn(8, 16, dtype=torch.float64),
        "p_3d": torch.randn(2, 8, 16, dtype=torch.float64),
        "p_wr": (U.T @ dk.T).contiguous(),  # (16, 8) right-weight
        "p_mis": torch.randn(8, 12, dtype=torch.float64),
        "p_zero": torch.zeros(8, 16, dtype=torch.float64),
        "p_num": 3.14,  # non-tensor leaf
        "p_skew": skew,  # dropped directions sum past tolerance
        "p_qkv": torch.cat(
            [torch.randn(8, 16, dtype=torch.float64), dk @ U, dv @ U]
        ),
    }
    good = [site("p_k"), site("p_v")]
    fac = K._factor_sites(good, leaves, 1e-8)
    assert fac is not None
    Ut, d_in, err, kept = fac
    assert d_in == 16 and tuple(Ut.shape) == (16, 3) and err < 1e-10
    assert len(kept) == 2

    # missing / non-tensor / non-2D / d_in mismatch / empty basis /
    # residual beyond tolerance — each declines None.
    assert K._factor_sites([site("p_none")], leaves, 1e-8) is None
    assert K._factor_sites([site("p_num")], leaves, 1e-8) is None
    assert K._factor_sites([site("p_3d")], leaves, 1e-8) is None
    assert (
        K._factor_sites([site("p_k"), site("p_mis")], leaves, 1e-8)
        is None
    )
    assert K._factor_sites([site("p_zero")], leaves, 1e-8) is None
    assert K._factor_sites([site("p_skew")], leaves, 1e-8) is None

    # A full-rank member DOES factor — valid, useless: the stacked
    # row space is what it is; the cost gate declines downstream.
    fac_full = K._factor_sites(
        [site("p_k"), site("p_full")], leaves, 1e-8
    )
    assert fac_full is not None and fac_full[0].shape[1] > 3

    # matmul-right and qkv orientations factor the same way.
    s_r = site("p_wr", orient="matmul_r")
    s_q = site("p_qkv", orient="qkv", span=(8, 8))
    fac2 = K._factor_sites([s_r, s_q], leaves, 1e-8)
    assert fac2 is not None and fac2[0].shape[1] == 3


# ---------------------------------------------------------------------------
#  Units — term rewrite + filler + tables
# ---------------------------------------------------------------------------


def test_replace_nodes_and_add_chain():
    """Node-keyed rebuild + the add fold."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    a = Op.make("linear", x, w)
    b = Op.make("silu", a)
    new = Op.make("matmul", x, w)
    got = K._replace_nodes(b, {a: new})
    assert got == Op.make("silu", new)
    assert K._replace_nodes(x, {a: new}) is x
    assert K._add_chain([a]) is a
    two = K._add_chain([a, b])
    assert two == Op.make("add", a, b)
    three = K._add_chain([a, b, x])
    assert three == Op.make("add", a, Op.make("add", b, x))


def test_kv_rewrite_orientations():
    """The latent-factored substitutions per site spelling."""
    x = Var("x", TensorType((8, 16)))
    ut = Param("p_kv_latent_ut", TensorType((16, 4)))
    C = Op.make("matmul", x, ut)
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wb = Param("p_b", TensorType((8,)))
    n_lin = Op.make("linear", x, wk)
    n_bias = Op.make("linear", x, wk, wb)
    wr = Param("p_wr_weight", TensorType((16, 8)))
    n_mm = Op.make("matmul", x, wr)
    wqkv = Param("p_qkv_proj_weight", TensorType((24, 16)))
    n_qkv = Op.make("linear", x, wqkv)
    n_qkvb = Op.make(
        "linear", x, wqkv, Param("p_qb", TensorType((24,)))
    )

    def drv(shp, nm):
        return Param(nm, TensorType(shp))

    s_lin = K._KVSite("b", n_lin, x, wk, "linear")
    s_bias = K._KVSite("b", n_bias, x, wk, "linear")
    s_mm = K._KVSite("b", n_mm, x, wr, "matmul_r")
    derived = {
        s_lin: drv((8, 4), "p_kv_latent_d0"),
        s_bias: drv((8, 4), "p_kv_latent_db"),
        s_mm: drv((4, 8), "p_kv_latent_d1"),
    }
    body = Op.make(
        "add", n_lin, Op.make("add", n_mm, Op.make("silu", x))
    )
    out = K._kv_rewrite(body, [s_lin, s_mm], C, derived, {}, {}, {})
    lin_new = Op.make("linear", C, derived[s_lin])
    mm_new = Op.make("matmul", C, derived[s_mm])
    ops = M._iter_ops(out)
    assert lin_new in ops and mm_new in ops
    # the silu arm is untouched
    assert Op.make("silu", x) in ops
    # bias rides the recovery untouched
    out2 = K._kv_rewrite(n_bias, [s_bias], C, derived, {}, {}, {})
    assert out2.args[2] is wb and out2.args[1] is derived[s_bias]

    # qkv: two section sites -> one concat node, derived leaves only.
    ss = [
        K._KVSite("b", n_qkv, x, wqkv, "qkv", (8, 8)),
        K._KVSite("b", n_qkv, x, wqkv, "qkv", (16, 8)),
    ]
    dq = {s: drv((8, 4), f"p_d{i}") for i, s in enumerate(ss)}
    out3 = K._kv_rewrite(
        n_qkv,
        ss,
        C,
        dq,
        {n_qkv: drv((8, 16), "p_wq")},
        {},
        {},
    )
    assert out3.op == "concat" and len(out3.args) == 3
    assert out3.args[0].op == "linear"
    assert all(
        a.op == "linear" and a.args[0] is C for a in out3.args[1:]
    )
    ssb = [
        K._KVSite("b", n_qkvb, x, wqkv, "qkv", sp)
        for sp in ((16, 8), (8, 8))
    ]
    dqb = {s: drv((8, 4), f"p_b{i}") for i, s in enumerate(ssb)}
    sb = {s: drv((8,), f"p_sb{i}") for i, s in enumerate(ssb)}
    out4 = K._kv_rewrite(
        n_qkvb,
        ssb,
        C,
        dqb,
        {n_qkvb: drv((8, 16), "p_wqb")},
        {n_qkvb: drv((8,), "p_qbb")},
        sb,
    )
    assert out4.op == "concat" and len(out4.args[0].args) == 3
    assert all(len(a.args) == 3 for a in out4.args)


def test_zero_slot_and_family_tables():
    """The exact-zero filler and the namespaced tables."""
    sink = TorchSink()
    v = Var("z", TensorType((4, 8)))
    mod = K._zero_slot(sink, v)
    with torch.no_grad():
        out = mod(torch.randn(4, 8, dtype=torch.float64))
    assert torch.equal(out, torch.zeros(4, 8, dtype=torch.float64))


def test_kv_numbers():
    """The decode-path stats: flops + bytes deltas."""
    x = Var("x", TensorType((8, 16)))
    eff = torch.zeros(8, 16, dtype=torch.float64)
    site = K._KVSite(
        "b",
        Op.make("linear", x, Param("p", TensorType((8, 16)))),
        x,
        Param("p", TensorType((8, 16))),
        "linear",
    )
    kept = [
        (site, torch.zeros(8, 16, dtype=torch.float64), eff),
        (site, object(), eff),
    ]
    nums = K._kv_numbers(kept, 16, 4, x)
    assert nums["kv_flops_before"] == 2 * 8 * 16 * 8 * 2
    assert nums["kv_flops_after"] == 2 * 8 * 16 * 4 + 2 * (
        2 * 8 * 4 * 8
    )
    assert nums["kv_bytes_before"] == 2 * 8 * 16 * 8
    # unknown-shape data → tok folds to 1
    y = Var("y", TensorType((None, 16)))
    nums2 = K._kv_numbers(kept, 16, 4, y)
    assert nums2["kv_flops_before"] == 2 * 1 * 16 * 8 * 2


# ---------------------------------------------------------------------------
#  Match surface
# ---------------------------------------------------------------------------


def test_match_shared_input_family():
    """The shared-input family + intra candidates, family first."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    law = K.KVLatentShare()
    ms = law.match(g)
    assert isinstance(law, MorphismLaw)
    assert ms[0].nodes == ("blocks.0", "blocks.1")
    assert ms[0].boundary == "family"
    assert "shared-input kv family" in ms[0].detail
    intra = [m for m in ms if m.boundary == "intra"]
    assert len(intra) == 2
    assert "shared-data kv" in intra[0].detail
    # Spec payload carries the latent group's data term + config.
    ex = ms[0].reify.extra
    assert ex["tokens"] == law.tokens and ex["data"] is not None


def test_match_residual_parallel_family():
    """``x + (b0 + b1)`` — the sum-plus-input evidence form."""
    torch.manual_seed(0)
    g = _lift(_ResidualKVStack().eval().double(), _x())
    ms = K.KVLatentShare().match(g)
    fam = [m for m in ms if m.boundary == "family"]
    assert len(fam) == 1 and fam[0].nodes == ("blocks.0", "blocks.1")


def test_match_downstream_sum_consumer():
    """``c(b0 + b1)`` — the sum feeds a downstream block's input."""
    torch.manual_seed(0)
    g = _lift(_DownstreamKV().eval().double(), _x())
    ms = K.KVLatentShare().match(g)
    fam = [m for m in ms if m.boundary == "family"]
    assert len(fam) == 1


def test_match_no_fire_cases():
    """All the honest no-fires: product outputs, sequential inputs,
    fan-out to a plain consumer, single-member families."""
    torch.manual_seed(0)
    law = K.KVLatentShare()
    for mk in (_MulKVStack, _SequentialKV, _FanoutKV):
        g = _lift(mk().eval().double(), _x())
        fam = [m for m in law.match(g) if m.boundary == "family"]
        assert fam == [], mk.__name__


def test_match_single_member_family_skipped():
    """A same-input group where only ONE block has KV sites is not
    a cross-block family (the intra pass still sees the block)."""
    torch.manual_seed(0)

    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    _KVBlock(16, 8, _latent(4, 16), 5),
                    nn.Linear(16, 16, bias=False),
                ]
            )

        def forward(self, x):
            return self.blocks[0](x) + self.blocks[1](x)

    g = _lift(Mixed().eval().double(), _x())
    ms = K.KVLatentShare().match(g)
    assert [m for m in ms if m.boundary == "family"] == []
    assert any(m.nodes == ("blocks.0",) for m in ms)


def test_match_law_flags():
    """cross=False / intra=False disable each detection side."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    only_intra = K.KVLatentShare(cross=False).match(g)
    assert all(m.boundary == "intra" for m in only_intra)
    only_cross = K.KVLatentShare(intra=False).match(g)
    assert all(m.boundary == "family" for m in only_cross)


def test_match_evidence_probe_failure(monkeypatch):
    """A failed second-probe capture keeps single-capture evidence."""
    torch.manual_seed(0)
    model = _SharedKVStack().eval().double()
    comp = TorchComposer()

    def boom(_x):
        raise RuntimeError("no second probe")

    monkeypatch.setattr(comp, "perturbed", boom)
    g = M.lift_graph(model, _x(), source=TorchSource(), composer=comp)
    assert g._probe2 is False
    fam = [
        m for m in K.KVLatentShare().match(g) if m.boundary == "family"
    ]
    assert len(fam) == 1


def test_match_probe2_contradiction():
    """Evidence on capture 1 but NOT the probe declines the family."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    # Corrupt the probe-side output clone so the sum relation breaks.
    rec = g.record("blocks.1")
    rec.out_val2 = torch.zeros_like(rec.out_val2) + 99.0
    fam = [
        m for m in K.KVLatentShare().match(g) if m.boundary == "family"
    ]
    assert fam == []


# ---------------------------------------------------------------------------
#  Reify — declines and grafts
# ---------------------------------------------------------------------------


def _family_match(graph, idx=0):
    """The first (widest) family match on a lifted graph."""
    ms = [m for m in K.KVLatentShare().match(graph)]
    return ms[idx]


def test_reify_cross_graft_fp64():
    """The family graft: fused slot + zero filler, verified fp64."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9
    assert out["latent_rank"] == 4 and out["n_kv_sites"] == 4
    assert out["kv_flops_after"] < out["kv_flops_before"]
    assert out["kv_bytes_after"] < out["kv_bytes_before"]
    assert out["factor_max_err"] < 1e-9
    assert set(out["reps"]) == {"blocks.0", "blocks.1"}
    with torch.no_grad():
        z = out["reps"]["blocks.1"](_x((4, 16)))
    assert torch.equal(z, torch.zeros(4, 16, dtype=torch.float64))


def test_reify_intra_graft_fp64():
    """Single-block MLA fold: k+v through one latent, verified."""
    torch.manual_seed(0)
    g = _lift(_SingleKV().eval().double(), _x())
    m = _family_match(g)
    assert m.nodes == ("blocks.0",) and m.boundary == "intra"
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9
    assert "p_kv_latent_ut" in out["reified"]
    assert set(out["reps"]) == {"blocks.0"}


def test_reify_qkv_intra_graft_fp64():
    """Fused qkv: the k/v sections rewrite to concat pieces."""
    torch.manual_seed(0)
    g = _lift(_SingleQKV().eval().double(), _x())
    m = _family_match(g)
    assert m.nodes == ("blocks.0",)
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9
    assert "concat" in out["reified"]


def test_reify_matmul_orientation_graft_fp64():
    """Right-matmul weights factor the same way."""
    torch.manual_seed(0)
    g = _lift(_SingleMatmulKV().eval().double(), _x())
    m = _family_match(g)
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9


def test_reify_decline_paths():
    """Every honest decline: opaque, multi-input, lost sites, no
    factor, no savings, wrong shapes, failed verify."""
    torch.manual_seed(0)

    # --- opaque node
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    graph_rec = g.record("blocks.1")
    graph_rec.ir = None
    out = _reify(m, g)
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )

    # --- multi-input IR
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    r1 = g.record("blocks.1")
    two = IR(
        root=Var("y", TensorType((8, 16))),
        inputs=[
            Var("a", TensorType((8, 16))),
            Var("y", TensorType((8, 16))),
        ],
        input_names={"a", "y"},
    )
    r1.ir = two
    out = _reify(m, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "multi-input block"
    )

    # --- fewer than two sites on the group's data term
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    m = M.MorphismMatch(
        law="probe",
        nodes=m.nodes,
        boundary="family",
        reify=M.ReifySpec(
            mode="family",
            rules="compose",
            extra={
                **m.reify.extra,
                "data": Var("none", TensorType((8, 16))),
            },
        ),
    )
    out = _reify(m, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "no shared-data kv sites"
    )

    # --- no certified common factor (a leaf is not a tensor)
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").leaves["p_k_proj_weight"] = 3.0
    out = _reify(m, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "no certified common factor"
    )


def test_reify_declines_inexact_and_full_rank():
    """Noisy and full-rank weights both decline — the law only fires
    on real factorisations that actually save compute."""
    torch.manual_seed(0)
    # Inexact: the noise directions enter the basis — a common
    # factor exists, only at a useless rank; the cost gate drops it.
    g = _lift(_SharedKVStack(inexact=1e-3).eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "no_improvement"
    )
    assert out["latent_rank"] > 4

    g = _lift(_SharedKVStack(full_rank=True).eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "no_improvement"
    )
    # Honest record: the common factor existed trivially at r = d_in.
    assert out["latent_rank"] == 16
    assert out["kv_flops_after"] > out["kv_flops_before"]


def test_reify_decline_shape_guards():
    """The additive-sum slot preconditions, checked before graft."""
    torch.manual_seed(0)

    # member output not a tensor
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").out_val = "not-a-tensor"
    out = _reify(m, g)
    assert out["status"] == "declined" and "outputs" in out["reason"]

    # member outputs different shapes — can't sum honestly

    # (craft: shrink one record's out clone)
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").out_val = torch.zeros(
        8, 8, dtype=torch.float64
    )
    out = _reify(m, g)
    assert out["status"] == "declined" and "same shape" in out["reason"]

    # consumed slot needs in-shape == out-shape
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").example = torch.zeros(
        8, 32, dtype=torch.float64
    )
    out = _reify(m, g)
    assert out["status"] == "declined" and "in-shape" in out["reason"]


def test_reify_decline_verify_fail(monkeypatch):
    """A failing pair verify drops the rewrite — never grafts."""
    import catopt_torch.adapters as A

    def fail_verify(*_a, **_k):
        return A.VerifyReport(max_abs=1.0, max_rel=1.0, passed=False)

    monkeypatch.setattr(A, "verify_module", fail_verify)
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert out["status"] == "declined"
    assert "verify failed" in out["reason"]


# ---------------------------------------------------------------------------
#  End to end
# ---------------------------------------------------------------------------


def test_e2e_shared_kv_grafts_fp64():
    """Full pipeline: the family grafts, fp64-exact end to end, the
    stats carry the decode-path KV reduction numbers."""
    torch.manual_seed(0)
    model = _SharedKVStack().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[K.KVLatentShare()], optimize_rest=False
        ),
    )
    m = stats["matches"]["kv_latent_share:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["rel_diff"] < 1e-9
    assert m["latent_rank"] == 4
    # Bench numbers: decode-path KV compute + memory cut ~62%.
    assert m["kv_flops_after"] < m["kv_flops_before"] * 0.5
    assert m["kv_bytes_after"] < m["kv_bytes_before"] * 0.5
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    assert stats["n_rewritten"] == 2
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
    # A fresh input too — the fused family is the real restructure.
    x2 = _x(seed=7)
    with torch.no_grad():
        d2 = (model(x2) - opt(x2)).abs().max().item()
    assert d2 < 1e-9
    # The consumed slot contributes an exact zero.
    with torch.no_grad():
        z = opt.blocks[1](x2)
    assert torch.equal(z, torch.zeros_like(x2))
    # Original model untouched.
    assert model.blocks[1] is not opt.blocks[1]
    assert isinstance(model.blocks[1], _KVBlock)


def test_e2e_residual_parallel_and_downstream():
    """Both other wiring forms graft end to end."""
    torch.manual_seed(0)
    for mk in (_ResidualKVStack, _DownstreamKV):
        model = mk().eval().double()
        x = _x()
        opt, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            x,
            strategy=MorphismSearch(
                laws=[K.KVLatentShare()], optimize_rest=False
            ),
        )
        fam = [
            v
            for k, v in stats["matches"].items()
            if "family" in v.get("boundary", "")
        ]
        assert fam and fam[0]["status"] == "grafted", mk.__name__
        assert stats["end_to_end"]["max_rel_diff"] < 1e-9
        with torch.no_grad():
            d = (model(x.clone()) - opt(x.clone())).abs().max().item()
        assert d < 1e-9


def test_e2e_three_member_family():
    """Depth-3 family: one latent serves three blocks."""
    torch.manual_seed(0)
    model = _SharedKVStack(depth=3).eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[K.KVLatentShare()], optimize_rest=False
        ),
    )
    m = stats["matches"]["kv_latent_share:blocks.0+blocks.1+blocks.2"]
    assert m["status"] == "grafted"
    assert m["n_kv_sites"] == 6
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_intra_only_graft():
    """The single-block MLA fold grafts through the strategy too."""
    torch.manual_seed(0)
    model = _SingleKV().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[K.KVLatentShare()], optimize_rest=False
        ),
    )
    m = stats["matches"]["kv_latent_share:blocks.0"]
    assert m["status"] == "grafted"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_optimize_morphisms_entry_kv():
    """The function entry point accepts the law via strategy."""
    torch.manual_seed(0)
    model = _SingleKV().eval().double()
    res = optimize_morphisms(
        model,
        _x(),
        backend=TorchBackend(),
        strategy=MorphismSearch(
            laws=[K.KVLatentShare()], optimize_rest=False
        ),
    )
    assert (
        res.stats["matches"]["kv_latent_share:blocks.0"]["status"]
        == "grafted"
    )


# ---------------------------------------------------------------------------
#  morphisms.py plumbing coverage
# ---------------------------------------------------------------------------


def test_morphism_graph_aux_io():
    """Aux (model-level) io entries populate the evidence targets."""
    nodes = [M.MorphismNode("b0", None, False)]
    recs = {"b0": M._BlockRecord("b0", None)}
    io = {
        "b0": {"out": torch.ones(2, 2)},
        "<model>": {"out": torch.ones(2, 2), "out_obj": object()},
        "weird": None,
        "<other>": {"out_obj": object()},
    }
    g = M.MorphismGraph(nodes, [], recs, io=io)
    assert len(g._model_outs) == 1 and len(g._model_out_objs) == 2
    g2 = M.MorphismGraph(nodes, [], recs)
    assert g2._model_outs == () and g2._probe2 is False


def test_saturate_offer_kwargs():
    """The third offer element carries witness kwargs."""
    from catopt_core.laws import RuleSet

    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    t = Op.make("linear", x, w)
    t2 = Op.make("linear", x, Op.make("add", w, w))
    eg, eid = M._saturate(
        t,
        RuleSet("e", ()),
        max_iterations=2,
        max_enodes=100,
        symmetry_budget=None,
        offers=[
            (t2, "offer with kwargs", {"error_bound": 0.5, "note": "n"})
        ],
    )
    assert eg is not None and eid is not None


def test_morphisms_reexport():
    """The sibling-module law re-exports through ``morphisms``."""
    assert M.KVLatentShare is K.KVLatentShare


def test_package_getattr():
    """The package-level lazy export resolves the KV law."""
    import catopt_orchestrator as O

    assert O.KVLatentShare is K.KVLatentShare


def test_family_shapes_check_unit():
    """_family_shape_check: the additive-sum preconditions."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    recs = [g.record(n) for n in ("blocks.0", "blocks.1")]
    assert K._family_shape_check(recs) is None


# ---------------------------------------------------------------------------
#  Coverage — the remaining edge paths
# ---------------------------------------------------------------------------


class _DataDep(nn.Module):
    """Data-dependent branch — export fails, an opaque node."""

    def __init__(self, dim: int) -> None:
        """A block the exporter cannot trace."""
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on data — untraceable."""
        if x.sum() > 0:
            return self.lin(x)
        return -self.lin(x)


class _OpaqueKVStack(nn.Module):
    """KV siblings plus an unexportable sibling on the same input."""

    def __init__(self, dim: int = 16, d_kv: int = 8, rank: int = 4) -> None:
        """Two KV blocks flanking an opaque one."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            [
                _KVBlock(dim, d_kv, U, 95),
                _DataDep(dim),
                _KVBlock(dim, d_kv, U, 96),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum all three — the opaque out is an extra addend."""
        return (
            self.blocks[0](x) + self.blocks[1](x) + self.blocks[2](x)
        )


class _MemberOutModel(nn.Module):
    """A member's output IS the model's own return value."""

    def __init__(self, dim: int = 16, d_kv: int = 8, rank: int = 4) -> None:
        """b0's output is discarded; b1's IS the return."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 97 + i) for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return member-1's output object at top level."""
        self.blocks[0](x)
        return self.blocks[1](x)


def test_opaque_sibling_node_skipped():
    """An opaque node is skipped everywhere — family grouping, the
    intra loop — and the clean pair still factors."""
    torch.manual_seed(0)
    g = _lift(_OpaqueKVStack().eval().double(), _x())
    fams = K._input_families(g)
    assert sorted(fams.values()) == [["blocks.0", "blocks.2"]]
    ms = K.KVLatentShare().match(g)
    fam = [m for m in ms if m.boundary == "family"]
    assert fam and fam[0].nodes == ("blocks.0", "blocks.2")


def test_member_out_is_model_return():
    """Fan-out guard: a member whose output is the model return is
    a live non-additive consumer — no family fires."""
    torch.manual_seed(0)
    g = _lift(_MemberOutModel().eval().double(), _x())
    fam = [
        m
        for m in K.KVLatentShare().match(g)
        if m.boundary == "family"
    ]
    assert fam == []


def test_fanout_clean_none_out_obj():
    """A member with no captured out_obj is simply skipped."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    g.record("blocks.1").out_obj = None
    members = ("blocks.0", "blocks.1")
    assert K._fanout_clean(g, members) is True


def test_family_evidence_edge_cases():
    """s-None and non-tensor-x paths inside the evidence check."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    fam = ["blocks.0", "blocks.1"]
    members = ("blocks.0", "blocks.1")
    g.record("blocks.0").out_val = None
    g.record("blocks.1").out_val = None
    assert K._family_evidence(g, members, fam) is False
    # _sum_bases: non-tensor x adds nothing
    s = torch.ones(4, 4, dtype=torch.float64)
    assert K._sum_bases(s, None, None, None) == [s]


def test_match_cross_member_ir_edges():
    """A member whose ir is missing / multi-input: usable < 2."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    two = IR(
        root=Var("y", TensorType((8, 16))),
        inputs=[
            Var("a", TensorType((8, 16))),
            Var("y", TensorType((8, 16))),
        ],
        input_names={"a", "y"},
    )
    g.record("blocks.1").ir = two
    law = K.KVLatentShare()
    ms = law.match(g)
    assert [m for m in ms if m.boundary == "family"] == []
    # the intra pass also skips the multi-input record
    g.record("blocks.0").ir = None
    assert law.match(g) == []


def test_qkv_leaves_unit_edges():
    """_qkv_leaves: non-Param bias arg and a missing bias leaf."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_qkv_proj_weight", TensorType((24, 16)))
    wval = torch.randn(24, 16, dtype=torch.float64)
    site_nobias = K._KVSite(
        "b",
        Op.make("linear", x, w, Op.make("add", w, w)),
        x,
        w,
        "qkv",
        (8, 8),
    )
    params, leaves = {}, {}
    K._qkv_leaves(0, site_nobias, wval, 16, params, leaves, {}, {}, {})
    # non-Param bias arg -> early return after the q-section leaf
    assert "p_kv_latent_wq0" in params and len(params) == 1

    site_bias = K._KVSite(
        "b",
        Op.make("linear", x, w, Param("p_b", TensorType((24,)))),
        x,
        w,
        "qkv",
        (8, 8),
    )
    params2, leaves2 = {}, {}
    K._qkv_leaves(0, site_bias, wval, 16, params2, leaves2, {}, {}, {})
    # bias Param exists but its leaf is missing -> early return
    assert "p_kv_latent_wq0" in params2 and len(params2) == 1


def test_reify_qkv_biased_graft_fp64():
    """A biased fused-qkv block grafts with derived bias leaves."""
    torch.manual_seed(0)
    g = _lift(_SingleQKV(bias=True).eval().double(), _x())
    m = _family_match(g)
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9
    assert "p_kv_latent_qb" in out["reified"] or "qb" in out["reified"]


def test_family_shape_check_not_tensor():
    """oshape None: first member's captured out is not a tensor."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.0").out_val = 3.0
    out = _reify(m, g)
    assert out["status"] == "declined"
    assert "not tensors" in out["reason"]


def test_kv_numbers_edge_shapes():
    """Unknown-shape data and weights without element_size."""
    x = Var("x", TensorType((8, 16)))
    eff = torch.zeros(8, 16, dtype=torch.float64)
    site = K._KVSite(
        "b",
        Op.make("linear", x, Param("p", TensorType((8, 16)))),
        x,
        Param("p", TensorType((8, 16))),
        "linear",
    )
    # an unshapeable data term: tok folds to 1
    kept = [(site, object(), eff)]
    nums = K._kv_numbers(kept, 16, 4, None)
    assert nums["kv_flops_before"] == 2 * 1 * 16 * 8
    # no element_size anywhere -> the 8-byte default stays
    assert nums["kv_bytes_before"] == 8 * 16 * 8


def test_aux_outs_trailing_entries():
    """An aux entry missing ``out`` mid-iteration, then more."""
    nodes = [M.MorphismNode("b0", None, False)]
    recs = {"b0": M._BlockRecord("b0", None)}
    t = torch.ones(2, 2)
    io = {
        "<model>": {"out": t, "out_obj": object()},
        "<extra>": {"out_obj": object()},
        "<more>": {"out": t},
    }
    g = M.MorphismGraph(nodes, [], recs, io=io)
    assert len(g._model_outs) == 2 and len(g._model_out_objs) == 2
