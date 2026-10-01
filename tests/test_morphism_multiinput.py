"""Multi-input morphism lifting — attention blocks on ``(x, cos, sin)``.

Real attention blocks are not ``f(x)``: they take ``(x, cos, sin)``
rope tables or ``(x, kv_cache)`` state alongside the activation.
This suite pins the multi-input lift: the :class:`InputSig`
classification, the position-aware boundary fallback, composition
over the activation input with context passed through unchanged,
``kv_latent_share`` on the multi-input arm — and every honest
decline (multi-activation, mutation, opaque, unshared context).
All grafted paths verify fp64 against the original model.
"""

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_orchestrator import MorphismSearch, Optimizer
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _x() -> torch.Tensor:
    return torch.randn(
        8,
        16,
        generator=torch.Generator().manual_seed(0),
        dtype=torch.float64,
    )


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


def _optimize(
    model: nn.Module,
    x: torch.Tensor,
    laws: list | None = None,
    **kw,
):
    return Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=laws if laws is not None else M.DEFAULT_MORPHISM_LAWS,
            **kw,
        ),
    )


def _latent(rank: int, dim: int, seed: int = 0) -> torch.Tensor:
    """A shared latent basis — fp64 rows of R^{dim}."""
    return torch.randn(
        rank,
        dim,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


class _RopeAttn(nn.Module):
    """Real-shape attention block taking ``(x, cos, sin)`` tables.

    ``cos``/``sin`` arrive positionally (the rope convention), so the
    call is genuinely three-argument.  ``k_proj``/``v_proj`` can be
    built low-rank through a shared basis ``U`` for the latent-share
    tests.
    """

    def __init__(
        self,
        dim: int = 16,
        U: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.double()
        if U is not None:
            g = torch.Generator().manual_seed(seed)
            dk = torch.randn(
                dim, U.shape[0], generator=g, dtype=torch.float64
            )
            dv = torch.randn(
                dim, U.shape[0], generator=g, dtype=torch.float64
            )
            with torch.no_grad():
                self.k_proj.weight.copy_(dk @ U)
                self.v_proj.weight.copy_(dv @ U)

    def _rope(
        self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        d2 = t.shape[-1] // 2
        rot = torch.cat([-t[..., d2:], t[..., :d2]], dim=-1)
        return t * cos + rot * sin

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q = self._rope(self.q_proj(x), cos, sin)
        k = self._rope(self.k_proj(x), cos, sin)
        v = self.v_proj(x)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return self.o_proj(o)


class _RopeStack(nn.Module):
    """A chain of rope-attention blocks sharing buffer tables.

    ``shared=False`` gives each block its own (still constant) tables
    — the unshared-context honest decline.  ``wrap=True`` wraps each
    block's consumption in the parent's residual add
    (``residual_wrapped`` wires).
    """

    def __init__(
        self,
        dim: int = 16,
        depth: int = 2,
        *,
        shared: bool = True,
        wrap: bool = False,
        U: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _RopeAttn(dim, U=U, seed=50 + i) for i in range(depth)
        )
        self.wrap = wrap
        fr = torch.randn(
            8,
            dim // 2,
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
        )
        self.register_buffer(
            "cos", torch.cat([fr.cos(), fr.cos()], dim=-1)
        )
        self.register_buffer(
            "sin", torch.cat([fr.sin(), fr.sin()], dim=-1)
        )
        self.shared = shared
        if not shared:
            fr2 = torch.randn(
                8,
                dim // 2,
                generator=torch.Generator().manual_seed(8),
                dtype=torch.float64,
            )
            self.register_buffer(
                "cos2", torch.cat([fr2.cos(), fr2.cos()], dim=-1)
            )
            self.register_buffer(
                "sin2", torch.cat([fr2.sin(), fr2.sin()], dim=-1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, s2 = (
            (self.cos, self.sin)
            if self.shared
            else (self.cos2, self.sin2)
        )
        for i, b in enumerate(self.blocks):
            c, s = (self.cos, self.sin) if i == 0 else (c2, s2)
            x = x + b(x, c, s) if self.wrap else b(x, c, s)
        return x


class _RopeParallel(nn.Module):
    """Two rope blocks sharing one input — the cross-family shape."""

    def __init__(
        self, dim: int = 16, U: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _RopeAttn(dim, U=U, seed=60 + i) for i in range(2)
        )
        fr = torch.randn(
            8,
            dim // 2,
            generator=torch.Generator().manual_seed(9),
            dtype=torch.float64,
        )
        self.register_buffer(
            "cos", torch.cat([fr.cos(), fr.cos()], dim=-1)
        )
        self.register_buffer(
            "sin", torch.cat([fr.sin(), fr.sin()], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, self.cos, self.sin) + self.blocks[1](
            x, self.cos, self.sin
        )


class _StoriesAttn(nn.Module):
    """stories15M-shaped attention block taking ``(h, cos, sin)``.

    The k/v projections read the *normalised* stream — their shared
    data operand is the ``rms_norm`` term, not the bare input var —
    and the weights carry the llama2.c ``wk``/``wv`` names (the law
    is invoked with ``tokens=("wk", "wv")``).  ``U`` builds both
    projections through one shared latent basis.
    """

    def __init__(
        self,
        dim: int = 16,
        U: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.rms_att = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.double()
        if U is not None:
            g = torch.Generator().manual_seed(seed)
            dk = torch.randn(
                dim, U.shape[0], generator=g, dtype=torch.float64
            )
            dv = torch.randn(
                dim, U.shape[0], generator=g, dtype=torch.float64
            )
            with torch.no_grad():
                self.wk.weight.copy_(dk @ U)
                self.wv.weight.copy_(dv @ U)

    def _rope(
        self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        d2 = t.shape[-1] // 2
        rot = torch.cat([-t[..., d2:], t[..., :d2]], dim=-1)
        return t * cos + rot * sin

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        xn = F.rms_norm(h, (h.shape[-1],), self.rms_att, 1e-5)
        q = self._rope(self.wq(xn), cos, sin)
        k = self._rope(self.wk(xn), cos, sin)
        v = self.wv(xn)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return h + self.wo(o)


class _StoriesTiny(nn.Module):
    """One stories15M block behind shared rope-table buffers."""

    def __init__(
        self, dim: int = 16, U: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_StoriesAttn(dim, U=U)])
        fr = torch.randn(
            8,
            dim // 2,
            generator=torch.Generator().manual_seed(21),
            dtype=torch.float64,
        )
        self.register_buffer(
            "cos", torch.cat([fr.cos(), fr.cos()], dim=-1)
        )
        self.register_buffer(
            "sin", torch.cat([fr.sin(), fr.sin()], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, self.cos, self.sin)


class _CacheRead(nn.Module):
    """``(x, kv_cache)``-style block — a read-only positional cache."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        return self.lin(x) + cache[0]


class _CacheWrite(nn.Module):
    """Write-only state mutator — invisible to the exported IR."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        cache[0] = x[0]
        return self.lin(x)


class _CacheReadWrite(nn.Module):
    """Write-then-read mutator — visible to the export as a write op."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        cache[0] = x[0]
        return self.lin(x) + cache


class _CacheStack(nn.Module):
    """A chain of cache-reading blocks sharing one cache buffer."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _CacheRead(dim) for _ in range(depth)
        )
        self.register_buffer(
            "cache",
            torch.randn(
                8,
                dim,
                generator=torch.Generator().manual_seed(3),
                dtype=torch.float64,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, self.cache)
        return x


class _TwoStream(nn.Module):
    """Two data inputs — the multi-activation decline."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.lin(x) + self.lin(y)


class _RuntimeWeight(nn.Module):
    """A runtime-supplied weight — the opaque-input decline."""

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        return x @ w.T


class _ScalarScale(nn.Module):
    """A non-tensor positional arg — the arg-position decline."""

    def forward(self, x: torch.Tensor, k):
        return x * k


class _VaryingCtx(nn.Module):
    """An input-derived context table — ``state``/``read_only``."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.lin(x) * t


class _VaryingStack(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_VaryingCtx(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, x.abs().clamp(min=0.5))


class _IdxTable(nn.Module):
    """``lin(x) + table[idx]`` — a non-float tensor context arg."""

    def __init__(self, dim: int = 16, rows: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, table: torch.Tensor, idx):
        return self.lin(x) + table[idx]


class _IdxStack(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_IdxTable(dim)])
        self.register_buffer(
            "table",
            torch.randn(
                4,
                dim,
                generator=torch.Generator().manual_seed(4),
                dtype=torch.float64,
            ),
        )
        self.idx = torch.zeros(8, dtype=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, self.table, self.idx)


class _PreNormCtx(nn.Module):
    """``lin(norm(x) * mask)`` — a prenorm with a table mask."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.lin(self.norm(x) * mask)


class _PreNormStack(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_PreNormCtx(dim)])
        self.register_buffer(
            "mask",
            torch.randn(
                8,
                dim,
                generator=torch.Generator().manual_seed(5),
                dtype=torch.float64,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, self.mask)


class _SdpaCache(nn.Module):
    """``sdpa(x, k_cache, v_cache)`` — cache reads at sdpa pos ≥ 1."""

    def forward(self, x, k_cache, v_cache):
        return F.scaled_dot_product_attention(
            x.unsqueeze(0), k_cache, v_cache
        ).squeeze(0)


class _SdpaCacheStack(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_SdpaCache()])
        self.register_buffer(
            "kc",
            torch.randn(
                1,
                8,
                dim,
                generator=torch.Generator().manual_seed(6),
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "vc",
            torch.randn(
                1,
                8,
                dim,
                generator=torch.Generator().manual_seed(61),
                dtype=torch.float64,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, self.kc, self.vc)


class _FanoutCtx(nn.Module):
    """``lin(x) * ctx`` — the second arg is a benign context read."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        return self.lin(x) * ctx


class _FanoutModel(nn.Module):
    """b1 consumes b0's output as its *context* — not the stream."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim).double(), _FanoutCtx(dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks[0](x)
        return self.blocks[1](x, y)


class _DataDep(nn.Module):
    """Unexportable — a data-dependent guard fails ``torch.export``."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.lin(x)
        return -self.lin(x)


class _OpaqueThenRope(nn.Module):
    """An opaque A feeding a transformed input into a multi-input B."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_DataDep(dim), _RopeAttn(dim)])
        self.register_buffer(
            "cos", torch.ones(8, dim, dtype=torch.float64)
        )
        self.register_buffer(
            "sin", torch.ones(8, dim, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[1](
            self.blocks[0](x) * 2.0, self.cos, self.sin
        )


class _WrapTail(nn.Module):
    """``x + b0(x, c, s)`` then a plain single-input tail — ``residual``."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_RopeAttn(dim), nn.Linear(dim, dim, bias=False).double()]
        )
        self.register_buffer(
            "cos", torch.ones(8, dim, dtype=torch.float64)
        )
        self.register_buffer(
            "sin", torch.ones(8, dim, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[1](x + self.blocks[0](x, self.cos, self.sin))


class _SplitFan(nn.Module):
    """b0's output fans to *two* blocks — ambiguous activation edge."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Linear(dim, dim).double(),
                _FanoutCtx(dim),
                nn.Linear(dim, dim).double(),
            ]
        )
        self.register_buffer(
            "tab",
            torch.randn(
                8,
                dim,
                generator=torch.Generator().manual_seed(14),
                dtype=torch.float64,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks[0](x)
        return self.blocks[1](y, self.tab) + self.blocks[2](y)


_GLOBAL_TABLE = torch.randn(
    8,
    16,
    generator=torch.Generator().manual_seed(11),
    dtype=torch.float64,
)


class _GlobalCtx(nn.Module):
    """Context bound to a module-external global — invariant, unbound."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.double()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.lin(x) * t


class _GlobalStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_GlobalCtx()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x, _GLOBAL_TABLE)


# ---------------------------------------------------------------------------
#  Crafted-IR unit tests — the classifier's terminal roles
# ---------------------------------------------------------------------------

_T = TensorType((8, 16))
_TL = TensorType((16, 16))


def _v(name: str) -> Var:
    return Var(name, _T)


def _ir(root, *vs: Var) -> IR:
    return IR(
        root=root,
        inputs=list(vs),
        input_names={v.name for v in vs},
        params={},
    )


def _kinds(root, *names: str) -> tuple:
    """``_classify_inputs`` kinds for a root over named input vars."""
    vs = tuple(_v(n) for n in names)
    return tuple(i.kind for i in M._classify_inputs(_ir(root, *vs)))


def test_classify_dead_and_stream():
    """An unused context input is ``const_table``/``dead``."""
    x, c = _v("x"), _v("c")
    w = Param("w", _TL)
    kinds = M._classify_inputs(_ir(Op.make("linear", x, w), x, c))
    assert [(i.kind, i.role) for i in kinds] == [
        ("activation", "stream"),
        ("const_table", "dead"),
    ]


def test_classify_runtime_weight_opaque():
    """A var in a projection's *weight* position is opaque."""
    x, w = _v("x"), _v("w")
    kinds = _kinds(Op.make("linear", x, w), "x", "w")
    assert kinds == ("activation", "opaque")
    kinds = _kinds(Op.make("matmul", x, w), "x", "w")
    assert kinds == ("activation", "opaque")
    # …and the var-weight matmul reverses the roles.
    kinds = _kinds(Op.make("matmul", w, x), "x", "w")
    assert kinds == ("opaque", "activation")


def test_classify_addend_and_sdpa():
    """Bare addends and sdpa k/v positions are context reads."""
    x, c = _v("x"), _v("c")
    assert _kinds(Op.make("add", x, c), "x", "c") == (
        "const_table",
        "const_table",
    )
    k, v2 = _v("k"), _v("v")
    kinds = _kinds(Op.make("sdpa", x, k, v2), "x", "k", "v")
    assert kinds == ("activation", "const_table", "const_table")


def test_classify_norm_and_tables():
    """Norm pos>0 and table ops mark context; a norm subject is stream."""
    x, w = _v("x"), _v("w")
    n = Op.make("layer_norm", x, w, normalized_shape=(16,))
    assert _kinds(n, "x", "w") == ("activation", "opaque")
    p = Param("pw", _TL)
    t, idx = _v("t"), _v("i")
    root = Op.make(
        "add",
        Op.make("linear", x, p),
        Op.make("index_select", t, 0, idx),
    )
    assert _kinds(root, "x", "t", "i") == (
        "activation",
        "const_table",
        "const_table",
    )


def test_classify_pointwise():
    """Pointwise roles hinge on whether siblings carry vars."""
    x, c = _v("x"), _v("c")
    # var * Const → stream (the sibling carries no var)
    assert _kinds(Op.make("mul", x, M.Const(2.0)), "x", "c") == (
        "activation",
        "const_table",
    )
    # var * var → both context (a two-stream product is ambiguous)
    assert _kinds(Op.make("mul", x, c), "x", "c") == (
        "const_table",
        "const_table",
    )


def test_classify_mutation_and_unknown():
    """Write destinations are mutated state; unknown ops are opaque."""
    x, c = _v("x"), _v("c")
    sel = Op.make("select", x, dim=0, index=0)
    w = Param("w", _TL)
    root = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make("select_scatter", c, sel, dim=0, index=0),
    )
    kinds = M._classify_inputs(_ir(root, x, c))
    # x's select→select_scatter-src use is opaque (an opaque use
    # outranks its stream use); the write target is mutated — the IR
    # sees the mutation, no probe needed.
    assert (kinds[0].kind, kinds[0].role) == ("opaque", "unhandled")
    assert (kinds[1].kind, kinds[1].role) == ("state", "mutated")
    # An unclassifiable op marks its operand opaque.
    kinds = _kinds(Op.make("frobnicate", x, c), "x", "c")
    assert kinds == ("opaque", "opaque")


def test_classify_view_root_and_pos():
    """Views are transparent: a var ending at the root is stream."""
    x, c = _v("x"), _v("c")
    # root == the var itself → stream; the other input is dead.
    kinds = M._classify_inputs(_ir(x, x, c))
    assert (kinds[0].kind, kinds[0].role) == ("activation", "stream")
    assert (kinds[1].kind, kinds[1].role) == ("const_table", "dead")
    # A var at a non-arg0 position of a view op is a context read.
    kinds = _kinds(
        Op.make(
            "add", Op.make("mul", x, x), Op.make("expand_as", c, x)
        ),
        "x",
        "c",
    )
    assert kinds == ("const_table", "const_table")
    # A var under a view at arg0 keeps walking to the terminal op.
    kinds = _kinds(
        Op.make("add", Op.make("linear", x, Param("w", _TL)), c),
        "x",
        "c",
    )
    assert kinds == ("activation", "const_table")
    # A view op that IS the root: the var flows to the output — stream.
    kinds = _kinds(Op.make("transpose", x, 0, 1), "x", "c")
    assert kinds == ("activation", "const_table")


def test_classify_dedup_uses():
    """A var feeding the same op twice visits that op once."""
    x, c = _v("x"), _v("c")
    kinds = _kinds(Op.make("add", Op.make("mul", x, x), c), "x", "c")
    # x*x: both positions see a var sibling → context (not activation).
    assert kinds == ("const_table", "const_table")


def _sig(inputs: tuple) -> M.BlockSig:
    return M.BlockSig(
        in_projs=(),
        out_proj=(),
        norm=M.NormSig(kind="none", affine=False, pre=False),
        act=(),
        residual=False,
        shape=(None, None),
        inputs=inputs,
    )


def test_sig_liftable_verdicts():
    """The lift verdict reads the InputSig kinds."""
    assert M._sig_liftable(_sig(())) is None
    a = M.InputSig(0, "x", "activation", "stream", (8, 16))
    t = M.InputSig(1, "t", "const_table", "table", (8, 16))
    st = M.InputSig(1, "s", "state", "read_only", (8, 16))
    mu = M.InputSig(1, "m", "state", "mutated", (8, 16))
    op = M.InputSig(1, "w", "opaque", "unhandled", (16, 16))
    assert M._sig_liftable(_sig((a, t))) is None
    assert M._sig_liftable(_sig((a, st))) is None
    assert "mutates" in M._sig_liftable(_sig((a, mu)))
    assert "no activation" in M._sig_liftable(_sig((t,)))
    assert "multi-activation" in M._sig_liftable(_sig((a, a)))
    assert "opaque" in M._sig_liftable(_sig((a, op)))
    # _act_index returns the sig position, 0 when no activation exists.
    assert M._act_index((t,)) == 0
    assert M._act_index((t, a)) == 0
    assert (
        M._act_index(
            (t, M.InputSig(1, "x2", "activation", "stream", (8, 16)))
        )
        == 1
    )


def test_block_signature_infers_shapes():
    """``block_signature`` pulls shapes from the activation var."""
    x, c = _v("x"), _v("c")
    p = Param("w", _TL)
    ir = _ir(Op.make("add", Op.make("linear", x, p), c), x, c)
    sig = M.block_signature(ir)
    assert sig.shape == ((8, 16), (8, 16))
    assert [w.name for w in sig.in_projs] == ["w"]


# ---------------------------------------------------------------------------
#  Probe helpers — mutation + refinement
# ---------------------------------------------------------------------------


def test_probe_value_variants():
    """``_probe_value`` perturbs floats, clones others, passes non-tensors."""
    a = torch.ones(4, dtype=torch.float64)
    p = M._probe_value(a, 1)
    assert not torch.equal(p, a) and p.dtype == a.dtype
    i = torch.ones(4, dtype=torch.int64)
    pi = M._probe_value(i, 0)
    assert torch.equal(pi, i) and pi is not i
    assert M._probe_value(3.0, 0) == 3.0


def test_mutates_arg_probe_failure():
    """A block that cannot be replayed declines honestly."""

    class _Raises(nn.Module):
        def forward(self, x):
            raise RuntimeError("nope")

    assert "probe failed" in M._mutates_arg(
        _Raises(), (torch.randn(4),)
    )


def test_param_bound_ids():
    """Params, buffers and tensor attrs all count as bound."""
    m = nn.Module()
    m.p = nn.Parameter(torch.randn(4))
    m.register_buffer("b", torch.randn(4))
    m.t = torch.randn(4)
    ids = M._param_bound_ids(m)
    assert {id(m.p), id(m.b), id(m.t)} <= ids


def test_refine_inputs_edges():
    """``_refine_inputs``'s early exits: non-context passthrough, args gap."""
    a = M.InputSig(0, "x", "activation", "stream", (8, 16))
    t = M.InputSig(1, "t", "const_table", "table", (8, 16))
    sig = _sig((a, t))
    rec = M._BlockRecord(name="b0", module=None)
    # No captured in_objs → index out of range → passthrough.
    out = M._refine_inputs(sig, rec, None, set())
    assert out.inputs[1].kind == "const_table"
    # Varying unbound context → state/read_only.
    rec.in_objs = (torch.randn(4), torch.randn(4))
    rec.args = (rec.in_objs[0], rec.in_objs[1])
    out = M._refine_inputs(
        sig, rec, ((rec.in_objs[0], torch.randn(4)), {}), set()
    )
    assert out.inputs[1].kind == "state"
    # Bound object → stays const_table.
    out = M._refine_inputs(sig, rec, None, {id(rec.in_objs[1])})
    assert out.inputs[1].kind == "const_table"


# ---------------------------------------------------------------------------
#  Lift: signatures, wires, matches
# ---------------------------------------------------------------------------


def test_lift_rope_signature():
    """``(x, cos, sin)`` blocks lift: activation + const tables."""
    torch.manual_seed(0)
    g = _lift(_RopeStack(depth=2).eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    kinds = [(i.name, i.kind, i.role) for i in n0.sig.inputs]
    assert kinds == [
        ("x", "activation", "stream"),
        ("cos", "const_table", "table"),
        ("sin", "const_table", "table"),
    ]
    assert g.wires[0].kind == "chain"


def test_lift_cache_state_read():
    """``(x, cache)`` with a read-only positional read lifts."""
    torch.manual_seed(0)
    g = _lift(_CacheStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    assert [(i.name, i.kind) for i in n0.sig.inputs] == [
        ("x", "activation"),
        ("cache", "const_table"),
    ]
    assert g.wires[0].kind == "chain"


def test_lift_sdpa_cache():
    """``sdpa(x, k_cache, v_cache)`` — query is the stream."""
    torch.manual_seed(0)
    g = _lift(_SdpaCacheStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    assert [i.kind for i in n0.sig.inputs] == [
        "activation",
        "const_table",
        "const_table",
    ]


def test_lift_index_table():
    """A ``(x, table, idx)`` block — int-tensor arg, table context."""
    torch.manual_seed(0)
    g = _lift(_IdxStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    kinds = [(i.name, i.kind) for i in n0.sig.inputs]
    assert kinds == [
        ("x", "activation"),
        ("table", "const_table"),
        ("idx", "const_table"),
    ]


def test_lift_state_readonly_refinement():
    """A context that varies across probes refines to state/read_only."""
    torch.manual_seed(0)
    g = _lift(_VaryingStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    assert [(i.kind, i.role) for i in n0.sig.inputs] == [
        ("activation", "stream"),
        ("state", "read_only"),
    ]


def test_lift_global_table():
    """An unbound but invariant context stays a const_table."""
    torch.manual_seed(0)
    g = _lift(_GlobalStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque
    assert [i.kind for i in n0.sig.inputs] == [
        "activation",
        "const_table",
    ]


def test_lift_declines():
    """Honest declines: multi-activation, mutation, opaque, non-tensor."""
    torch.manual_seed(0)

    class _Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_TwoStream()])

        def forward(self, x):
            return self.blocks[0](x, x * 2)

    g = _lift(_Two().eval().double(), _x())
    assert g.node("blocks.0").opaque
    assert "multi-activation" in g.record("blocks.0").note

    class _Mut(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_CacheWrite()])
            self.register_buffer(
                "cache", torch.zeros(8, 16, dtype=torch.float64)
            )

        def forward(self, x):
            return self.blocks[0](x, self.cache)

    g = _lift(_Mut().eval().double(), _x())
    # Write-only mutation: the functionalised export cannot see it —
    # the perturbed probe catches it.
    assert g.node("blocks.0").opaque
    assert "mutates input" in g.record("blocks.0").note

    class _MutRW(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_CacheReadWrite()])
            self.register_buffer(
                "cache", torch.zeros(8, 16, dtype=torch.float64)
            )

        def forward(self, x):
            return self.blocks[0](x, self.cache)

    g = _lift(_MutRW().eval().double(), _x())
    # Write-then-read: the export itself carries the write op.
    assert g.node("blocks.0").opaque
    assert "mutates" in g.record("blocks.0").note

    class _RW(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_RuntimeWeight()])
            self.w = torch.randn(
                16,
                16,
                generator=torch.Generator().manual_seed(2),
                dtype=torch.float64,
            )

        def forward(self, x):
            return self.blocks[0](x, self.w)

    g = _lift(_RW().eval().double(), _x())
    assert g.node("blocks.0").opaque
    assert "opaque input" in g.record("blocks.0").note

    class _NT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_ScalarScale()])

        def forward(self, x):
            return self.blocks[0](x, 2.0)

    g = _lift(_NT().eval().double(), _x())
    assert g.node("blocks.0").opaque
    assert "non-tensor" in g.record("blocks.0").note


# ---------------------------------------------------------------------------
#  Laws: reach, reify, graft — all verified fp64
# ---------------------------------------------------------------------------


def _match(stats: dict, law: str) -> dict:
    """The law's report — the widest (``+``-joined) key on overlaps."""
    keys = [k for k in stats["matches"] if k.startswith(law)]
    assert keys
    key = max(keys, key=lambda k: (k.count("+"), k))
    return stats["matches"][key]


def test_out_in_compose_grafts_multi_input():
    """The headline: out_proj∘in_projs composes over the activation.

    ``b1(x, cos, sin)`` composes with ``b0``'s out-projection; the
    rope tables pass through — verified fp64.
    """
    torch.manual_seed(0)
    m = _RopeStack(depth=2).eval().double()
    x = _x()
    opt, stats = _optimize(
        m, x, laws=[M.OutInCompose()], optimize_rest=False
    )
    rep = _match(stats, "out_in_compose")
    assert rep["status"] == "grafted"
    assert rep["cost_after"] < rep["cost_before"]
    assert rep["rel_diff"] < 1e-12
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10
    # The fused term keeps the context vars — the call signature is
    # still (x, cos, sin).
    assert "cos" in rep["reified"] and "sin" in rep["reified"]
    with torch.no_grad():
        diff = (m(x.clone()) - opt(x.clone())).abs().max().item()
    assert diff < 1e-12


def test_window_compose_grafts_multi_input():
    """The ≥3-block chain-family window fuses over the activation."""
    torch.manual_seed(0)
    m = _RopeStack(depth=3).eval().double()
    x = _x()
    _, stats = _optimize(
        m, x, laws=[M.WindowCompose()], optimize_rest=False
    )
    rep = _match(stats, "window_compose")
    assert rep["status"] == "grafted"
    assert rep["rel_diff"] < 1e-12
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_residual_wire_multi_input():
    """``x = x + b(x, c, s)`` wires classify as residual_wrapped."""
    torch.manual_seed(0)
    g = _lift(_RopeStack(depth=2, wrap=True).eval().double(), _x())
    assert g.wires[0].kind == "residual_wrapped"
    ms = M.ResidualAbsorb().match(g)
    assert ms and ms[0].boundary == "residual_wrapped"


def test_norm_cascade_matches_multi_input():
    """An intra law sees the norm through the context inputs."""
    torch.manual_seed(0)
    g = _lift(_PreNormStack().eval().double(), _x())
    n0 = g.node("blocks.0")
    assert not n0.opaque and n0.sig.norm.kind == "layer_norm"
    ms = M.NormCascade().match(g)
    assert ms and ms[0].nodes == ("blocks.0",)


def test_kv_latent_intra_multi_input():
    """``kv_latent_share`` factors the k/v projections on (x,cos,sin)."""
    torch.manual_seed(0)
    m = _RopeStack(depth=1, U=_latent(4, 16)).eval().double()
    x = _x()
    _, stats = _optimize(
        m, x, laws=[K.KVLatentShare()], optimize_rest=False
    )
    rep = _match(stats, "kv_latent_share")
    assert rep["status"] == "grafted"
    assert rep["factor_max_err"] < 1e-8
    assert rep["rel_diff"] < 1e-10
    assert stats["end_to_end"]["max_rel_diff"] < 1e-8


def test_kv_latent_cross_multi_input():
    """The shared-input family works with multi-input members."""
    torch.manual_seed(0)
    m = _RopeParallel(U=_latent(4, 16)).eval().double()
    x = _x()
    opt, stats = _optimize(
        m, x, laws=[K.KVLatentShare()], optimize_rest=False
    )
    rep = _match(stats, "kv_latent_share")
    assert rep["status"] == "grafted"
    assert rep["boundary"] == "family"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-8
    with torch.no_grad():
        diff = (m(x.clone()) - opt(x.clone())).abs().max().item()
    assert diff < 1e-10


def test_kv_latent_stories_shape_intra_graft():
    """The stories15M shape: wk/wv share the ``rms_norm`` data term.

    The real-checkpoint sweep's block shape — multi-input
    ``(h, cos, sin)``, ``wk``/``wv`` weight names, and a shared data
    operand that is a norm *term*, not the bare input var.  With a
    factorable k/v pair the latent grafts and verifies.
    """
    torch.manual_seed(0)
    m = _StoriesTiny(U=_latent(4, 16)).eval().double()
    x = _x()
    opt, stats = _optimize(
        m,
        x,
        laws=[K.KVLatentShare(tokens=("wk", "wv"))],
        optimize_rest=False,
    )
    rep = _match(stats, "kv_latent_share")
    assert rep["status"] == "grafted"
    assert rep["boundary"] == "intra"
    assert rep["factor_max_err"] < 1e-8
    assert rep["rel_diff"] < 1e-10
    assert stats["end_to_end"]["max_rel_diff"] < 1e-8
    with torch.no_grad():
        diff = (m(x.clone()) - opt(x.clone())).abs().max().item()
    assert diff < 1e-10


def test_kv_latent_stories_shape_fp32_declines_clean():
    """fp32 full-rank weights decline with a reason, not a crash.

    In fp32 the Gram-Schmidt sweep over-covers: rounding leaves
    above-threshold residuals past the ambient dimension (stories15M
    produced 570 "directions" in R^288), which used to crash
    ``_stack_cols``'s one-hot indexing with a raw ``IndexError``.
    The capped basis now reaches the certify gate, which declines
    honestly — fp32 weights cannot certify at ``factor_tol=1e-8``.
    """
    torch.manual_seed(0)
    m = _StoriesTiny().eval().float()
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    _, stats = _optimize(
        m,
        x,
        laws=[K.KVLatentShare(tokens=("wk", "wv"))],
        optimize_rest=False,
    )
    rep = _match(stats, "kv_latent_share")
    assert rep["status"] == "declined"
    assert rep["reason"] == "no certified common factor"
    assert "error" not in rep


def test_optimize_rest_multi_input():
    """The per-block fallback searches on the full args tuple."""
    torch.manual_seed(0)
    m = _RopeStack(depth=1).eval().double()
    x = _x()
    _, stats = _optimize(m, x, laws=[], optimize_rest=True)
    assert stats["blocks"]["blocks.0"]["status"] == "optimized"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_sigs_serialized():
    """``stats['sigs']`` carries the input schema."""
    torch.manual_seed(0)
    m = _RopeStack(depth=2).eval().double()
    _, stats = _optimize(
        m, _x(), laws=[M.OutInCompose()], optimize_rest=False
    )
    sig = stats["sigs"]["blocks.0"]
    assert [i["kind"] for i in sig["inputs"]] == [
        "activation",
        "const_table",
        "const_table",
    ]


# ---------------------------------------------------------------------------
#  Honest declines on the multi-input arm
# ---------------------------------------------------------------------------


def test_unshared_context_declines():
    """Match fires at signature level; reify declines without sharing."""
    torch.manual_seed(0)
    m = _RopeStack(depth=2, shared=False).eval().double()
    g = _lift(m, _x())
    assert g.wires[0].kind == "chain"
    ms = M.OutInCompose().match(g)
    assert ms and ms[0].nodes == ("blocks.0", "blocks.1")
    _, stats = _optimize(
        m, _x(), laws=[M.OutInCompose()], optimize_rest=False
    )
    rep = _match(stats, "out_in_compose")
    assert rep["status"] == "declined"
    assert "not shared" in rep["reason"]


def test_ctx_consumed_a_out_declines():
    """A's output as B's *context* arg is not an activation edge."""
    torch.manual_seed(0)
    g = _lift(_FanoutModel().eval().double(), _x())
    assert g.wires[0].kind == "opaque"
    assert M.OutInCompose().match(g) == []


def test_split_fan_declines():
    """A's output fanned to two blocks is not a chain edge."""
    torch.manual_seed(0)
    g = _lift(_SplitFan().eval().double(), _x())
    assert all(w.kind == "opaque" for w in g.wires)


def test_opaque_upstream_no_edge():
    """An opaque A with no captured edge declines the residual arm."""
    torch.manual_seed(0)
    g = _lift(_OpaqueThenRope().eval().double(), _x())
    assert g.node("blocks.0").opaque
    assert not g.node("blocks.1").opaque
    assert g.wires[0].kind == "opaque"


def test_residual_wire_multi_to_single():
    """``b1(x + b0(x, c, s))`` — a residual wire into a plain tail."""
    torch.manual_seed(0)
    g = _lift(_WrapTail().eval().double(), _x())
    assert not g.node("blocks.0").opaque
    # The plain residual wire (no wrap on B's own consumption).
    assert g.wires[0].kind == "residual"


def test_residual_wrapped_match_and_reify():
    """A residual_wrapped multi-input pair reaches reify."""
    torch.manual_seed(0)
    m = _RopeStack(depth=2, wrap=True).eval().double()
    g = _lift(m, _x())
    assert g.wires[0].kind == "residual_wrapped"
    ms = M.ResidualAbsorb().match(g)
    assert ms and ms[0].boundary == "residual_wrapped"
    _, stats = _optimize(
        m, _x(), laws=[M.ResidualAbsorb()], optimize_rest=False
    )
    rep = _match(stats, "residual_absorb")
    # Cost may or may not win — the honest gate decides; either way the
    # joint was built, saturated, and verified through the sink.
    assert rep["status"] in {"declined", "grafted"}
    if rep["status"] == "grafted":
        assert rep["rel_diff"] < 1e-10


def test_window_ctx_decline():
    """A window over members with unshared context declines at reify."""
    torch.manual_seed(0)
    m = _RopeStack(depth=3, shared=False).eval().double()
    g = _lift(m, _x())
    ms = M.WindowCompose().match(g)
    assert ms
    _, stats = _optimize(
        m, _x(), laws=[M.WindowCompose()], optimize_rest=False
    )
    rep = _match(stats, "window_compose")
    assert rep["status"] == "declined"
    assert "not shared" in rep["reason"]


def test_lift_without_second_probe():
    """A composer that cannot perturb still lifts — probe evidence
    (``in_objs2``/``cap2``) is simply absent; invariance checks skip."""

    class _NoProbe(TorchComposer):
        def perturbed(self, x):
            raise RuntimeError("no second capture")

    torch.manual_seed(0)
    m = _RopeStack(depth=2).eval().double()
    g = M.lift_graph(m, _x(), source=TorchSource(), composer=_NoProbe())
    assert not g.node("blocks.0").opaque
    # Without a second capture the context vars stay bound-const via
    # the param/buffer identity refinement.
    assert [i.kind for i in g.node("blocks.0").sig.inputs] == [
        "activation",
        "const_table",
        "const_table",
    ]
    _, stats = Optimizer(
        backend=TorchBackend(), composer=_NoProbe()
    ).optimize(
        m,
        _x(),
        strategy=MorphismSearch(
            laws=[M.OutInCompose()], optimize_rest=False
        ),
    )
    rep = _match(stats, "out_in_compose")
    assert rep["status"] == "grafted"
    assert rep["rel_diff"] < 1e-12


# ---------------------------------------------------------------------------
#  ``_mi_boundary`` / ``_ctx_var_map`` — crafted-evidence unit tests
# ---------------------------------------------------------------------------


def _live_pair():
    """Lift a real chain pair, returning (graph, call args)."""
    torch.manual_seed(0)
    g = _lift(_RopeStack(depth=2).eval().double(), _x())
    return g


def test_mi_boundary_guards():
    """``_mi_boundary``'s early exits on absent/malformed evidence."""
    g = _live_pair()
    nodes = {n.name: n for n in g.nodes}
    rec_a, rec_b = g.record("blocks.0"), g.record("blocks.1")
    # Both single-input → the composer's call.
    assert M._mi_boundary("a", "b", {}, {}, {}, {}, {}) is None
    # Missing captures/io → None.
    assert (
        M._mi_boundary("blocks.0", "blocks.1", {}, {}, {}, {}, nodes)
        is None
    )
    capt = {
        "blocks.0": (rec_a.args, {}),
        "blocks.1": (rec_b.args, {}),
    }
    io = {
        "blocks.0": {
            "calls": 1,
            "in_objs": rec_a.in_objs,
            "out": rec_a.out_val,
            "out_obj": rec_a.out_obj,
        },
        "blocks.1": {
            "calls": 2,
            "in_objs": rec_b.in_objs,
            "out": rec_b.out_val,
            "out_obj": rec_b.out_obj,
        },
    }
    # calls != 1 → ambiguous → None.
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt, io, {}, {}, nodes)
        is None
    )
    io["blocks.1"]["calls"] = 1
    # kwargs → None.
    capt2 = dict(capt)
    capt2["blocks.1"] = (rec_b.args, {"k": 1})
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt2, io, {}, {}, nodes)
        is None
    )
    # Non-tensor activation value → None.
    capt3 = dict(capt)
    capt3["blocks.1"] = ((None, None, None), {})
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt3, io, {}, {}, nodes)
        is None
    )
    # A's output IS the model's return → escapes → None.
    io2 = dict(io)
    io2["<model>"] = {"out": rec_a.out_val, "out_obj": rec_a.out_obj}
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt, io2, {}, {}, nodes)
        is None
    )
    # B's captured objects too short for its activation position → None.
    io3 = dict(io)
    io3["blocks.1"] = dict(io["blocks.1"], in_objs=())
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt, io3, {}, {}, nodes)
        is None
    )
    # B's output not a tensor → None.
    io4 = dict(io)
    io4["blocks.1"] = dict(io["blocks.1"], out=None, out_obj=object())
    io4["<model>"] = {"out": torch.randn(1), "out_obj": object()}
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt, io4, {}, {}, nodes)
        is None
    )
    # B's output consumed neither plainly nor wrapped → ambiguous → None.
    io5 = dict(io)
    stray = torch.randn(8, 16, dtype=torch.float64)
    io5["blocks.1"] = dict(io["blocks.1"], out=stray, out_obj=object())
    io5["<model>"] = {"out": torch.randn(1), "out_obj": object()}
    assert (
        M._mi_boundary("blocks.0", "blocks.1", capt, io5, {}, {}, nodes)
        is None
    )


def test_mi_boundary_residual_probe_guards():
    """The second-probe residual check declines on missing evidence."""
    out = M._mi_residual_probe("a", "b", {}, {}, 0, 0)
    assert out is False
    t = torch.randn(4, dtype=torch.float64)
    capt = {"a": ((t,), {}), "b": ((t,), {})}
    # A repeated probe call is ambiguous → False.
    io_many = {"a": {"calls": 2, "out": t}}
    assert M._mi_residual_probe("a", "b", capt, io_many, 0, 0) is False
    io2 = {"a": {"calls": 1, "out": t}}
    # Index overflow / non-tensor → False.
    assert M._mi_residual_probe("a", "b", capt, io2, 5, 0) is False
    capt2 = {"a": ((t,), {}), "b": ((1.0,), {})}
    assert M._mi_residual_probe("a", "b", capt2, io2, 0, 0) is False
    # Shape/equality match → True.
    a_in = torch.randn(4, dtype=torch.float64)
    a_out = torch.randn(4, dtype=torch.float64)
    b_in = a_in + a_out
    capt3 = {"a": ((a_in,), {}), "b": ((b_in,), {})}
    io3 = {"a": {"calls": 1, "out": a_out}}
    assert M._mi_residual_probe("a", "b", capt3, io3, 0, 0) is True


def test_mi_io_helpers():
    """``_mi_consumers``/``_mi_io_has_value`` on crafted captures."""
    g = _live_pair()
    rec_a = g.record("blocks.0")
    capt = {
        "blocks.0": (rec_a.args, {}),
        "blocks.1": (g.record("blocks.1").args, {}),
    }
    io = {
        "blocks.0": {
            "calls": 1,
            "in_objs": rec_a.in_objs,
            "out": rec_a.out_val,
            "out_obj": rec_a.out_obj,
        },
        "blocks.1": {
            "calls": 1,
            "in_objs": g.record("blocks.1").in_objs,
            "out": g.record("blocks.1").out_val,
            "out_obj": g.record("blocks.1").out_obj,
        },
    }
    hits = M._mi_consumers(io, capt, rec_a.out_obj, rec_a.out_val)
    assert hits == ["blocks.1"]
    # A value nowhere in the flow → no consumers.
    miss = torch.randn(8, 16, dtype=torch.float64)
    assert M._mi_consumers(io, capt, rec_a.out_obj, miss) == []
    # model_out == val → the early True; a value nowhere in the flow
    # → False; an arg value → True.
    assert M._mi_io_has_value(capt, miss, miss) is True
    other = torch.randn(8, 16, dtype=torch.float64)
    assert M._mi_io_has_value(capt, miss, other) is False
    assert M._mi_io_has_value(capt, miss, rec_a.out_val) is True


def test_ctx_var_map_branches():
    """``_ctx_var_map``'s decline and probe-confirmation arms."""
    g = _live_pair()
    rec_a, rec_b = g.record("blocks.0"), g.record("blocks.1")
    cmap, why = M._ctx_var_map(rec_a, rec_b, 0)
    assert why is None and len(cmap) == 2
    # Missing IR → "opaque node".
    rec_none = M._BlockRecord(name="x", module=None)
    _, why = M._ctx_var_map(rec_none, rec_b, 0)
    assert why == "opaque node"
    _, why = M._ctx_var_map(rec_a, rec_none, 0)
    assert why == "opaque node"
    # Truncated capture → "not captured".
    rec_short = M._BlockRecord(
        name="s", module=None, ir=rec_b.ir, in_objs=(rec_b.in_objs[0],)
    )
    _, why = M._ctx_var_map(rec_a, rec_short, 0)
    assert "not captured" in why
    # Probe-2 objects diverge → not confirmed → not shared.
    rec_p = M._BlockRecord(
        name="p",
        module=None,
        ir=rec_a.ir,
        in_objs=rec_a.in_objs,
        in_objs2=(object(), object(), object()),
    )
    _, why = M._ctx_var_map(rec_p, rec_b, 0)
    assert "not shared" in why


# ---------------------------------------------------------------------------
#  KV family helpers — multi-input arms
# ---------------------------------------------------------------------------


def test_usable_member_branches():
    """``_usable_member``'s guards: ir, sig, arity, liftability."""
    g = _live_pair()
    assert K._usable_member(g, "blocks.0") is True
    rec = g.record("blocks.0")
    keep = rec.ir
    keep_node = g.node("blocks.0")
    rec.ir = None
    assert K._usable_member(g, "blocks.0") is False
    rec.ir = keep
    rec2 = M._BlockRecord(name="z", module=None, ir=keep)
    g._records["z"] = rec2
    g._by_name["z"] = M.MorphismNode(name="z", sig=None, opaque=True)
    assert K._usable_member(g, "z") is False  # no sig at all
    g._records.pop("z")
    g._by_name.pop("z")
    # A sig that fails the lift verdict → not usable.
    sig0 = g.node("blocks.0").sig
    a = M.InputSig(0, "x", "activation", "stream", (8, 16))
    g._by_name["blocks.0"] = M.MorphismNode(
        name="blocks.0",
        sig=M._dc_replace(sig0, inputs=(a, *sig0.inputs[1:])),
        opaque=False,
    )
    # (still liftable — activation + context)
    assert K._usable_member(g, "blocks.0") is True
    op = M.InputSig(1, "w", "opaque", "unhandled", (16, 16))
    g._by_name["blocks.0"] = M.MorphismNode(
        name="blocks.0",
        sig=M._dc_replace(sig0, inputs=(a, op, sig0.inputs[2])),
        opaque=False,
    )
    assert K._usable_member(g, "blocks.0") is False  # opaque input
    g._by_name["blocks.0"] = keep_node


def test_family_bodies_ctx_decline():
    """Unshared member context returns the honest reason."""
    torch.manual_seed(0)
    g = _lift(_RopeStack(depth=2, shared=False).eval().double(), _x())
    recs = [g.record("blocks.0"), g.record("blocks.1")]
    members = [(r.name, r.ir) for r in recs]
    bodies, why = K._family_bodies_ctx(members, recs, g)
    assert bodies is None and "not shared" in why
    # Shared context → bodies on the member-0 activation var.
    g = _live_pair()
    recs = [g.record("blocks.0"), g.record("blocks.1")]
    members = [(r.name, r.ir) for r in recs]
    bodies, why = K._family_bodies_ctx(members, recs, g)
    assert why is None and len(bodies) == 2


def test_kv_cross_declines_unshared_context():
    """A shared-input family with unshared context inputs skips."""
    torch.manual_seed(0)
    U = _latent(4, 16)

    class _Par(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                _RopeAttn(16, U=U, seed=70 + i) for i in range(2)
            )
            fr = torch.randn(
                8,
                8,
                generator=torch.Generator().manual_seed(12),
                dtype=torch.float64,
            )
            self.register_buffer(
                "cos", torch.cat([fr.cos(), fr.cos()], -1)
            )
            self.register_buffer(
                "sin", torch.cat([fr.sin(), fr.sin()], -1)
            )
            fr2 = torch.randn(
                8,
                8,
                generator=torch.Generator().manual_seed(13),
                dtype=torch.float64,
            )
            self.register_buffer(
                "cos2", torch.cat([fr2.cos(), fr2.cos()], -1)
            )
            self.register_buffer(
                "sin2", torch.cat([fr2.sin(), fr2.sin()], -1)
            )

        def forward(self, x):
            return self.blocks[0](x, self.cos, self.sin) + self.blocks[
                1
            ](x, self.cos2, self.sin2)

    g = _lift(_Par().eval().double(), _x())
    fams = K._input_families(g)
    assert sorted(fams.values()) == [["blocks.0", "blocks.1"]]
    ms = K.KVLatentShare().match(g)
    assert [m for m in ms if m.boundary == "family"] == []
    # Forcing a family match declines at prep — the context tables are
    # per-block, so the members cannot share one call's vars.
    match = M.MorphismMatch(
        law="kv_latent_share",
        nodes=("blocks.0", "blocks.1"),
        boundary="family",
        reify=M.ReifySpec(mode="family", kinds=(), extra={}),
        detail="test",
    )
    prep = K._family_prep(match, g)
    assert (
        prep["status"] == "declined" and "not shared" in prep["reason"]
    )


def test_family_prep_sig_guards():
    """``_family_prep``'s per-member sig checks, in order."""
    g = _live_pair()
    # Craft a cross match; then break each member guard.
    match = M.MorphismMatch(
        law="kv_latent_share",
        nodes=("blocks.0", "blocks.1"),
        boundary="family",
        reify=M.ReifySpec(mode="family", kinds=(), extra={}),
        detail="test",
    )
    # A record whose sig is missing → opaque.
    keep = g._by_name["blocks.1"]
    g._by_name["blocks.1"] = M.MorphismNode(
        name="blocks.1", sig=None, opaque=True
    )
    out = K._family_prep(match, g)
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )
    g._by_name["blocks.1"] = keep
    # sig/ir arity mismatch → multi-input block.
    sig1 = keep.sig
    g._by_name["blocks.1"] = M.MorphismNode(
        name="blocks.1",
        sig=M._dc_replace(sig1, inputs=sig1.inputs[:1]),
        opaque=False,
    )
    out = K._family_prep(match, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "multi-input block"
    )
    g._by_name["blocks.1"] = keep
    # Unliftable sig → the classifier's own reason.
    a = M.InputSig(0, "x", "activation", "stream", (8, 16))
    op = M.InputSig(1, "w", "opaque", "unhandled", (16, 16))
    dead = M.InputSig(2, "d", "const_table", "dead", (8, 16))
    g._by_name["blocks.1"] = M.MorphismNode(
        name="blocks.1",
        sig=M._dc_replace(sig1, inputs=(a, op, dead)),
        opaque=False,
    )
    out = K._family_prep(match, g)
    assert out["status"] == "declined" and "opaque" in out["reason"]
    g._by_name["blocks.1"] = keep


def test_zero_slot_multi_input():
    """A zero filler takes the member's full input list."""
    g = _live_pair()
    ir = g.record("blocks.0").ir
    sink = TorchSink()
    z = K._zero_slot(sink, ir.inputs[0], tuple(ir.inputs))
    out = z(
        torch.randn(8, 16, dtype=torch.float64),
        torch.randn(8, 16, dtype=torch.float64),
        torch.randn(8, 16, dtype=torch.float64),
    )
    assert torch.equal(out, torch.zeros(8, 16, dtype=torch.float64))


# ---------------------------------------------------------------------------
#  Reify internals — slot arities + var selection
# ---------------------------------------------------------------------------


def test_rec_act_and_lower_term():
    """``_rec_act``/``_lower_term``/``_slot_filler`` honour inputs."""
    g = _live_pair()
    assert M._rec_act(g, "blocks.0") == 0
    # An unlifted/unknown node falls back to position 0.
    g._by_name["nope"] = M.MorphismNode(
        name="nope", sig=None, opaque=True
    )
    assert M._rec_act(g, "nope") == 0
    g._by_name.pop("nope")
    sink = TorchSink()
    ir = g.record("blocks.0").ir
    # _lower_term with the full input list lowers a multi-arg module.
    mod = M._lower_term(
        ir.root,
        ir.inputs[0],
        ir.params,
        g.record("blocks.0").leaves,
        sink,
        tuple(ir.inputs),
    )
    args = g.record("blocks.0").args
    ref = g.record("blocks.0").module(*args)
    out = mod(*args)
    assert torch.allclose(out, ref)


def test_pair_joint_ctx_mapping():
    """``_pair_joint`` rebinds B's context vars onto A's inputs."""
    g = _live_pair()
    match = M.MorphismMatch(
        law="out_in_compose",
        nodes=("blocks.0", "blocks.1"),
        boundary="chain",
        reify=M.ReifySpec(mode="chain", rules=("a", "b", "c")),
        detail="test",
    )
    resolved, why = M._pair_joint(match, g)
    assert why is None
    inputs = resolved[6]
    assert inputs == tuple(g.record("blocks.0").ir.inputs)
    # The joint still references the context vars (passed through).
    names = {v.name for v in inputs}
    assert names == {"x", "cos", "sin"}


def test_window_reps_arity():
    """``_window_reps`` fills each slot over its own input list."""
    torch.manual_seed(0)
    g = _lift(_RopeStack(depth=3).eval().double(), _x())
    ms = M.WindowCompose().match(g)
    assert ms
    match = ms[0]
    var = g.record("blocks.0").ir.inputs[0]
    reps = M._window_reps(
        var,  # best = the input itself (any term works for arity)
        match,
        var,
        tuple(g.record("blocks.0").ir.inputs),
        g,
        {},
        {},
        TorchSink(),
    )
    assert set(reps) == set(match.nodes)
    # Later slots accept the member's own (x, cos, sin) args.
    for name in match.nodes[1:]:
        out = reps[name](*g.record(name).args)
        assert out.shape == g.record(name).args[0].shape


def test_resolved_joint_intra_inputs():
    """An intra match resolves with the block's full input list."""
    g = _live_pair()
    match = M.MorphismMatch(
        law="norm_cascade",
        nodes=("blocks.0",),
        boundary="intra",
        reify=M.ReifySpec(mode="intra", rules=()),
        detail="test",
    )
    resolved, why = M._intra_joint(match, g)
    assert why is None
    assert resolved[6] == tuple(g.record("blocks.0").ir.inputs)
    assert resolved[1] is g.record("blocks.0").ir.inputs[0]


def test_ctx_hit_no_host_ir():
    """``_ctx_hit`` declines when the host never exported."""
    obj = torch.randn(4, dtype=torch.float64)
    member = M._BlockRecord(name="m", module=None, in_objs=(obj,))
    host = M._BlockRecord(name="h", module=None)
    assert M._ctx_hit(host, member, 0) is None


def test_lift_block_empty_in_objs():
    """``_lift_block`` with no captured live objects: ``in_obj`` stays None.

    The activation re-anchor's ``act < len(in_objs)`` guard's False
    arm — a capture carrying the args but no live-object entry.
    """
    torch.manual_seed(0)
    m = _RopeStack(depth=1).eval().double()
    mod = m.blocks[0]
    x = _x()
    cap = ((x, m.cos, m.sin), {})
    rec = M._BlockRecord(name="b", module=mod)
    sig = M._lift_block(
        rec, mod, cap, None, {}, {}, TorchSource(), set()
    )
    assert sig is not None and rec.note is None
    assert rec.in_obj is None and rec.in_objs == ()
