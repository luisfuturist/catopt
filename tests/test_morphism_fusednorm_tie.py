"""Morphism coverage for the real-export spellings: fused ``rms_norm``
and the embedding-table weight.

The llama2.c checkpoints spell RMSNorm as the fused ``rms_norm`` op —
``rms_norm(x, w)`` ≡ ``x·rms⁻¹·w`` with the gain carried as an operand
rather than a pointwise ``mul`` — and the emb/head tie as an
``embedding`` table plus a ``linear`` head.  This suite pins:

* signature detection: ``norm.kind == "rms"`` for the fused spelling
  (affine iff a param-only weight operand is present, ``pre`` iff the
  norm feeds an in-projection) and ``sig.tables`` carrying the
  ``embedding`` gather source;
* ``NormCascade``'s verified fold of the fused gain into the block's
  in-projection weights (the unfuse offer +
  ``linear_channel_scale``);
* ``WeightTie``'s emb↔head share — fired by name+shape candidacy,
  decided by bitwise values at reify;
* the honest declines: weightless/post norms, untied tables, and
  guard arms (empty arity, param-only subjects, var tables).
"""

import catopt_orchestrator.morphisms as M
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_orchestrator import MorphismSearch, Optimizer
from catopt_torch.adapters import TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    return torch.randn(
        *dims, generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


def _sig_of(mod: nn.Module, x: torch.Tensor) -> M.BlockSig:
    """Export *mod* and extract its block signature."""
    ir, _ = TorchSource().to_ir(mod, x)
    return M.block_signature(ir)


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


def _optimize(model: nn.Module, x: torch.Tensor, laws: list, **kw):
    return Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(laws=laws, **kw)
    )


class _FusedRMSLin(nn.Module):
    """The llama2.c spelling: fused ``F.rms_norm`` feeding a Linear."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(
            F.rms_norm(x, (x.shape[-1],), self.w, 1e-5)
        )


class _FusedRMSStack(nn.Module):
    """A chain of fused RMSNorm→Linear blocks (stories-style)."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _FusedRMSLin(dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _FusedRMSFan(nn.Module):
    """Shared fused rms_norm feeding two projections — wq/wv shape."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(dim))
        self.wa = nn.Linear(dim, dim, bias=False)
        self.wb = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = F.rms_norm(x, (x.shape[-1],), self.norm_w, 1e-5)
        return self.wa(xn) + self.wb(xn)


class _FusedRMSPost(nn.Module):
    """``F.rms_norm`` AFTER the projection — not a pre-norm."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(self.lin(x), (x.shape[-1],), self.w, 1e-5)


class _RMSWeightless(nn.Module):
    """``F.rms_norm`` with ``weight=None`` — RMS structure, no gain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), None, 1e-5)


class _EmbHead(nn.Module):
    """``emb(idx) -> head(·)`` — the tied-embedding/head pair."""

    def __init__(
        self, vocab: int = 32, dim: int = 16, *, tied: bool = True
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tied:
            with torch.no_grad():
                self.head.weight.copy_(self.emb.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(idx))


class _EmbAlone(nn.Module):
    """A bare embedding block — the table weight without a head."""

    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Embedding(vocab, dim)])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](idx)


# ---------------------------------------------------------------------------
#  Signature detection — fused rms_norm
# ---------------------------------------------------------------------------


def test_sig_fused_rms_norm():
    """``F.rms_norm`` before a Linear reads as affine pre-norm rms."""
    torch.manual_seed(0)
    sig = _sig_of(_FusedRMSLin().eval().double(), _x())
    assert sig.norm.kind == "rms"
    assert sig.norm.affine and sig.norm.pre
    assert [w.name for w in sig.in_projs] == ["p_proj_weight"]
    assert [w.name for w in sig.out_proj] == ["p_proj_weight"]


def test_sig_fused_rms_norm_weightless():
    """``F.rms_norm`` with ``weight=None`` — rms, not affine."""
    sig = _sig_of(_RMSWeightless().eval().double(), _x())
    assert sig.norm.kind == "rms" and not sig.norm.affine
    assert not sig.norm.pre


def test_sig_fused_rms_post_norm_not_pre():
    """A fused norm after the projection is post, not pre."""
    torch.manual_seed(0)
    sig = _sig_of(_FusedRMSPost().eval().double(), _x())
    assert sig.norm.kind == "rms" and sig.norm.affine
    assert sig.norm.pre is False


def test_norm_nodes_fused_edges():
    """``_norm_nodes`` guard arms on the fused op."""
    x = Var("x", TensorType((8, 16)))
    p = Param("p_w", TensorType((16,)))
    p2 = Param("p_w2", TensorType((16,)))

    # Bare 0-arg node (raw construction) — arity guard.
    assert M._norm_nodes(Op("rms_norm", (), {})) == []
    # Param-only subject — compile-time data, not a stream norm.
    assert M._norm_nodes(Op.make("rms_norm", p, p2)) == []
    # Weightless form — rms structure, not affine.
    t = Op.make("rms_norm", x)
    assert M._norm_nodes(t) == [(t, "rms", False)]
    # Weighted form — affine.
    t = Op.make("rms_norm", x, p)
    assert M._norm_nodes(t) == [(t, "rms", True)]


def test_norm_unfolds_guards():
    """``_norm_unfolds`` only unfuses weighted fused norms."""
    x = Var("x", TensorType((8, 16)))
    p = Param("p_w", TensorType((16,)))

    # Non-norm ops and the weightless spelling offer nothing.
    assert M._norm_unfolds(Op.make("linear", x, p)) == []
    assert M._norm_unfolds(Op.make("rms_norm", x)) == []
    # A var-carrying "weight" operand is not a foldable gain.
    t = Op.make(
        "rms_norm", x, Op.make("mul", x, p), dim=(16,), eps=1e-5
    )
    assert M._norm_unfolds(t) == []
    # The weighted form unfolds to the pointwise-gain member.
    t = Op.make("rms_norm", x, p, dim=(16,), eps=1e-5)
    (host, expanded, law) = M._norm_unfolds(t)[0]
    assert host is t and "rms_norm" in law
    assert expanded == Op.make(
        "mul", Op.make("rms_norm", x, dim=(16,), eps=1e-5), p
    )


# ---------------------------------------------------------------------------
#  Signature detection — embedding table weights
# ---------------------------------------------------------------------------


def test_sig_embedding_table_weight():
    """``embedding(W, idx)``'s gather source lands in ``sig.tables``."""
    sig = _sig_of(nn.Embedding(32, 16).eval().double(),
                  torch.randint(0, 32, (1, 8)))
    assert [w.name for w in sig.tables] == ["p_weight"]
    assert not sig.in_projs and not sig.out_proj


def test_sig_table_edges():
    """Table collection guards: empty arity and var-carried tables."""
    ir = IR(root=Op("embedding", (), {}), inputs=[], input_names=set())
    assert M.block_signature(ir).tables == ()
    v, tab = Var("idx", TensorType((8,))), Var("tab", TensorType((32, 16)))
    ir = IR(
        root=Op.make("embedding", tab, v),
        inputs=[tab, v],
        input_names={"tab", "idx"},
    )
    assert M.block_signature(ir).tables == ()


# ---------------------------------------------------------------------------
#  Verified folds — NormCascade on the fused spelling
# ---------------------------------------------------------------------------


def test_e2e_norm_cascade_fused_rms_grafts_fp64():
    """The fused gain folds into the in-projection weight, verified."""
    torch.manual_seed(0)
    model = _FusedRMSStack(dim=16, depth=2).eval().double()
    x = _x()
    opt, stats = _optimize(
        model, x, laws=[M.NormCascade()], optimize_rest=False
    )
    for i in range(2):
        m = stats["matches"][f"norm_cascade:blocks.{i}"]
        assert m["status"] == "grafted"
        assert m["boundary"] == "intra"
        assert m["rel_diff"] < 1e-9
        # The gain moved into the weight: weightless rms_norm + a
        # param-side ``mul(p_w, p_proj_weight)``.
        assert "rms_norm x," in m["reified"]
        assert "mul p_w" in m["reified"]
        assert m["cost_after"] < m["cost_before"]
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_norm_cascade_fused_rms_shared_fan():
    """One shared fused norm feeding two projections folds once."""
    torch.manual_seed(0)

    class Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_FusedRMSFan(16)])

        def forward(self, x):
            return self.blocks[0](x)

    model = Wrap().eval().double()
    x = _x()
    opt, stats = _optimize(
        model, x, laws=[M.NormCascade()], optimize_rest=False
    )
    m = stats["matches"]["norm_cascade:blocks.0"]
    assert m["status"] == "grafted" and m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_norm_cascade_fused_post_norm_declines():
    """A post-position fused norm produces no intra match."""
    torch.manual_seed(0)

    class Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_FusedRMSPost(16)])

        def forward(self, x):
            return self.blocks[0](x)

    g = _lift(Wrap().eval().double(), _x())
    intra = [
        m for m in M.NormCascade().match(g) if m.boundary == "intra"
    ]
    assert intra == []


# ---------------------------------------------------------------------------
#  Verified shares — WeightTie on the emb↔head pair
# ---------------------------------------------------------------------------


def test_e2e_weight_tie_emb_head_grafts():
    """Bitwise-tied emb/head — the llama2.c wcls share — grafts."""
    torch.manual_seed(0)
    model = _EmbHead(tied=True).eval().double()
    idx = torch.randint(0, 32, (1, 8))
    opt, stats = _optimize(
        model, idx, laws=[M.WeightTie()], optimize_rest=False
    )
    m = stats["matches"]["weight_tie:emb+head"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "tie"
    assert m["tied"] == [["emb__p_weight", "head__p_weight"]]
    assert m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(idx.clone()) - opt(idx.clone())).abs().max().item()
    assert d < 1e-9


def test_weight_tie_emb_head_declines_untied():
    """Same shape + stem but different values — the value gate speaks."""
    torch.manual_seed(0)
    model = _EmbHead(tied=False).eval().double()
    idx = torch.randint(0, 32, (1, 8))
    _, stats = _optimize(
        model, idx, laws=[M.WeightTie()], optimize_rest=False
    )
    m = stats["matches"]["weight_tie:emb+head"]
    assert m["status"] == "declined"
    assert m["reason"] == "no_tied_values"


def test_weight_tie_no_candidate_without_table():
    """An embedding alone carries a table ref but no cross match."""
    torch.manual_seed(0)
    g = _lift(_EmbAlone().eval().double(), torch.randint(0, 32, (1, 8)))
    assert [w.name for w in g.sig("blocks.0").tables] == ["p_weight"]
    assert M.WeightTie().match(g) == []
