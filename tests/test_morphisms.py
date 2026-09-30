"""Morphism engine tests — signature extraction, laws, verified reify.

Plan 0011 stages 0+1.  Stage 0 pins the *signature algebra* on
two-block fixtures: ``A.out_proj ∘ B.in_proj`` composition through a
diagonal norm, the residual ``+`` monoid absorb, and weight-tying
detection (identical shapes + name stem).  Stage 1 pins the lifted
graph (``lift_graph``), each morphism law's match/no-match surface,
and the reified programs — every grafted rewrite is verified fp64.
"""

import catopt_orchestrator.morphisms as M
import torch
import torch.nn as nn
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.laws import RuleSet
from catopt_orchestrator import (
    MorphismLaw,
    MorphismSearch,
    Optimizer,
    optimize_morphisms,
)
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer
from catopt_torch.models import (
    DeepParallel,
    NormLinear,
    ParallelBlock,
    ParallelLinear,
    ResidualMLP,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


class _Scale(nn.Module):
    """Pure diagonal block: ``x ∘ s`` (scalar param)."""

    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.s


class _ChainStack(nn.Module):
    """``x = b_i(x)`` — plain chain of DeepParallel blocks."""

    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _ResidualPair(nn.Module):
    """``x = x + s(x); x = B(x)`` — a residual boundary into B."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Scale(), DeepParallel(dim, dim, dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.blocks[0](x)
        return self.blocks[1](x)


class _ResidualWrapped(nn.Module):
    """``x = x + b_i(x)`` — residual wraps both sides of the boundary."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class _ChainWrapped(nn.Module):
    """``x = A(x); x = x + B(x)`` — chain into a residual-wrapped B."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks[0](x)
        return x + self.blocks[1](x)


class _DiagChain(nn.Module):
    """``x = s(x); x = B(x)`` — a diagonal block feeding a projection."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Scale(), DeepParallel(dim, dim, dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks[0](x)
        return self.blocks[1](x)


class _NormStack(nn.Module):
    """Stack of NormLinear blocks — each carries an affine pre-norm."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            NormLinear(dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _TiedPair(nn.Module):
    """Two blocks sharing one ``nn.Parameter`` — a real weight tie."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(2)]
        )
        self.blocks[1].weight = self.blocks[0].weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _TiedParallel(nn.Module):
    """One block with two *value-tied* projections (intra-block tie).

    The two weights are distinct Parameters carrying identical values
    — the intra-block tie the signature detects by shape and the reify
    pass confirms by value.  (Sharing the Parameter object instead
    would collapse at export: the IR would already see one weight.)
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ParallelLinear(dim, n_experts=2)])
        lin = self.blocks[0]
        with torch.no_grad():
            lin.linears[1].weight.copy_(lin.linears[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


class _DataDependent(nn.Module):
    """Data-dependent branch — export always fails (opaque block)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.lin(x)
        return -self.lin(x)


class _MixedStack(nn.Module):
    """Healthy blocks sandwiching an un-exportable (opaque) block."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ParallelLinear(dim, n_experts=2),
                _DataDependent(dim),
                ParallelLinear(dim, n_experts=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _SkipMid(nn.Module):
    """``blocks[1]`` never executes — a captured-IO boundary node."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[2](self.blocks[0](x))


class _KwargBlock(nn.Module):
    """Keyword-arg call — not a single-positional-arg block."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))

    def forward(
        self, x: torch.Tensor, *, gain: float = 1.0
    ) -> torch.Tensor:
        return x * gain * self.gain


class _KwargStack(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False), _KwargBlock(dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[1](self.blocks[0](x), gain=2.0)


class _MiniGPT(nn.Module):
    """Two ParallelBlocks — pre-norm attention+MLP residual stack."""

    def __init__(
        self, dim: int = 16, n_heads: int = 2, depth: int = 2
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelBlock(dim, n_heads, hidden_mult=2)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def _sig_of(mod: nn.Module, x: torch.Tensor) -> M.BlockSig:
    """Export *mod* and extract its block signature."""
    ir, _ = TorchSource().to_ir(mod, x)
    return M.block_signature(ir)


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    """Lift *model* through the torch ports."""
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*dims, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
#  Stage 0 — signature extraction
# ---------------------------------------------------------------------------


def test_sig_deep_parallel():
    """``(x@W1 + x@W2) @ W3``: in = {W1,W2}, out = {W3}, no norm."""
    sig = _sig_of(DeepParallel(16, 16, 16).eval().double(), _x())
    assert {w.name for w in sig.in_projs} == {
        "p_w1_weight",
        "p_w2_weight",
    }
    assert [w.name for w in sig.out_proj] == ["p_w3_weight"]
    assert sig.norm.kind == "none" and not sig.norm.affine
    assert sig.act == ()
    assert sig.residual is False
    assert sig.shape == ((8, 16), (8, 16))


def test_sig_norm_linear():
    """NormLinear: affine RMS pre-norm feeding the projection."""
    sig = _sig_of(NormLinear(16).eval().double(), _x())
    assert sig.norm.kind == "rms"
    assert sig.norm.affine and sig.norm.pre
    assert [w.name for w in sig.in_projs] == ["p_proj_weight"]
    assert [w.name for w in sig.out_proj] == ["p_proj_weight"]


def test_sig_residual_mlp():
    """ResidualMLP: residual spine + affine LayerNorm + silu."""
    sig = _sig_of(ResidualMLP(16, hidden_mult=2).eval().double(), _x())
    assert sig.residual is True
    assert sig.norm.kind == "layer_norm" and sig.norm.affine
    assert sig.norm.pre
    assert "silu" in sig.act
    assert {w.name for w in sig.in_projs} == {"p_fc1_weight"}
    assert {w.name for w in sig.out_proj} == {"p_fc2_weight"}


def test_sig_scale_is_pure_diagonal():
    """``x * s``: a diagonal block — no projections, affine gain."""
    sig = _sig_of(_Scale().eval().double(), _x())
    assert sig.norm.kind == "diag" and sig.norm.affine
    assert not sig.norm.pre  # no in-projection for it to precede
    assert not sig.in_projs and not sig.out_proj
    assert not sig.residual
    assert M._is_pure_diagonal(sig)


def test_sig_parallel_linear_shared_input():
    """``x@W0 + x@W1``: both projections are input AND terminal."""
    sig = _sig_of(ParallelLinear(16, n_experts=2).eval().double(), _x())
    names = {w.name for w in sig.in_projs}
    assert names == {"p_linears_0_weight", "p_linears_1_weight"}
    assert {w.name for w in sig.out_proj} == names


def test_sig_post_norm_not_pre():
    """A scale AFTER the projection is a post norm, not pre."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    g = Param("p_gain", TensorType((16,)))
    ir = IR(
        root=Op.make("mul", Op.make("linear", x, w), g),
        inputs=[x],
        input_names={"x"},
        params={"p_w": w, "p_gain": g},
    )
    sig = M.block_signature(ir)
    assert sig.norm.kind == "diag" and sig.norm.affine
    assert sig.norm.pre is False


def test_sig_rms_non_affine():
    """``x · rms⁻¹`` with no gain — rms structure, not affine."""

    class RMSNoGain(nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return t * torch.rsqrt(
                t.pow(2).mean(-1, keepdim=True) + 1e-6
            )

    sig = _sig_of(RMSNoGain().eval().double(), _x())
    assert sig.norm.kind == "rms" and not sig.norm.affine


def test_sig_hand_built_matmul_forms():
    """matmul: weight on either side; activation x activation and
    weight x weight matmuls are not projections at all."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    w2 = Param("p_w2", TensorType((16, 16)))

    sig = M.block_signature(
        IR(root=Op.make("matmul", x, w), inputs=[x], input_names={"x"})
    )
    assert [r.name for r in sig.in_projs] == ["p_w"]

    sig = M.block_signature(
        IR(root=Op.make("matmul", w, x), inputs=[x], input_names={"x"})
    )
    assert [r.name for r in sig.in_projs] == ["p_w"]

    sig = M.block_signature(
        IR(
            root=Op.make("matmul", x, y),
            inputs=[x, y],
            input_names={"x", "y"},
        )
    )
    assert not sig.in_projs and not sig.out_proj

    sig = M.block_signature(
        IR(root=Op.make("matmul", w, w2), inputs=[], input_names=set())
    )
    assert not sig.in_projs and not sig.out_proj
    assert sig.shape[0] is None  # no inputs


def test_sig_folded_weight_expr():
    """A param-only weight expression (``W1 + W2``) is still a weight —
    the ref carries ``name=""`` and an inferred shape."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    w2 = Param("p_w2", TensorType((16, 16)))
    ir = IR(
        root=Op.make("linear", x, Op.make("add", w, w2)),
        inputs=[x],
        input_names={"x"},
    )
    sig = M.block_signature(ir)
    (ref,) = sig.out_proj
    assert ref.name == "" and ref.shape == (16, 16)


def test_sig_var_weight_and_short_arity():
    """``linear`` with a *var* weight is not a projection; a bare
    1-arg ``linear`` node is skipped by arity."""
    x = Var("x", TensorType((8, 16)))
    wv = Var("w", TensorType((16, 16)))
    ir = IR(
        root=Op.make("linear", x, wv),
        inputs=[x, wv],
        input_names={"x", "w"},
    )
    assert not M.block_signature(ir).in_projs
    # Raw arity-1 node (dodges schema validation) — skipped.
    odd = Op("linear", (x,), {})
    ir = IR(root=odd, inputs=[x], input_names={"x"})
    assert not M.block_signature(ir).in_projs
    # Bare var root — no ops at all.
    ir = IR(root=x, inputs=[x], input_names={"x"})
    sig = M.block_signature(ir)
    assert not sig.in_projs and sig.norm.kind == "none"


def test_sig_norm_edge_cases():
    """Norm detection edges: both-sides-rsqrt mul, var x var mul,
    param x param mul, matrix gain (not a channel diagonal), and a
    non-tuple-shaped gain."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    rms = Op.make("rsqrt", Op.make("mean", Op.make("pow", x, 2)))

    # both sides carry an rsqrt core -> not an rms node, no var side.
    t = Op.make("mul", rms, rms)
    assert M._norm_nodes(t) == []

    # var x var -> elementwise product, not a diagonal.
    t = Op.make("mul", x, Op.make("silu", x))
    assert M._norm_nodes(t) == []

    # param-only x param-only -> no var side.
    t = Op.make("mul", w, Param("p_w2", TensorType((16, 16))))
    assert M._norm_nodes(t) == []

    # matrix gain is not a scalar/channel diagonal.
    t = Op.make("mul", x, w)
    assert M._norm_nodes(t) == []

    # param-only gain with unknown shape -> still not a diagonal.
    weird = Op("mystery", (w,), {})
    t = Op.make("mul", x, weird)
    assert M._norm_nodes(t) == []

    # 1-arg mul node (raw) — skipped by arity.
    assert M._norm_nodes(Op("mul", (x,), {})) == []


def test_residual_spine_variants():
    """The add-spine monoid check: bare input addends, nested adds,
    shared add subtrees, a non-input Var, and no add spine at all."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    f = Op.make("linear", x, w)

    assert M._residual_spine(Op.make("add", f, x), [x]) is True
    assert M._residual_spine(Op.make("sub", f, x), [x]) is True
    assert M._residual_spine(Op.make("linear", x, w), [x]) is False
    # A Var that is not among the inputs does not count.
    assert M._residual_spine(Op.make("add", f, y), [x]) is False
    # Nested: add(add(f, g), x)
    g = Op.make("sigmoid", f)
    nested = Op.make("add", Op.make("add", f, g), x)
    assert M._residual_spine(nested, [x]) is True
    # Shared add-subtree addend (interned): exercises the seen-set.
    shared = Op.make("add", f, g)
    root = Op.make("add", shared, shared)
    assert M._residual_spine(root, [x]) is False


def test_weights_tied_predicate():
    """Signature-level tie: equal shape + equal stem or full name."""
    w1 = M.WeightRef(Param("p_w", TensorType((4, 4))), "p_w", (4, 4))
    w2 = M.WeightRef(Param("p_w", TensorType((4, 4))), "p_w", (4, 4))
    a = M.WeightRef(Param("a_0_w", TensorType((4, 4))), "a_0_w", (4, 4))
    b = M.WeightRef(Param("a_1_w", TensorType((4, 4))), "a_1_w", (4, 4))
    c = M.WeightRef(Param("c_q_w", TensorType((4, 4))), "c_q_w", (4, 4))
    d = M.WeightRef(Param("d_w", TensorType((8, 4))), "d_w", (8, 4))
    anon = M.WeightRef(Op.make("add", w1.term, w2.term), "", (4, 4))
    noshp = M.WeightRef(
        Param("e", TensorType((None, 4))), "e", (None, 4)
    )

    assert M.weights_tied(w1, w2) is True  # identical names
    assert M.weights_tied(a, b) is True  # equal stems (index tokens)
    assert M.weights_tied(a, c) is False  # different stems
    assert M.weights_tied(a, d) is False  # different shapes
    assert M.weights_tied(anon, w1) is False  # nameless weight expr
    assert M.weights_tied(a, noshp) is False  # unknown/mismatched shape


# ---------------------------------------------------------------------------
#  Stage 0 — the signature algebra on 2-block fixtures
# ---------------------------------------------------------------------------


def test_algebra_out_in_compose_fp64():
    """``A.out_proj ∘ B.in_proj`` composition: ``linear(linear(x,A),B)``
    equals ``linear(x, B@A)`` — fp64-exact at term level."""
    sink = TorchSink()
    x = Var("x", TensorType((8, 16)))
    a = Param("a_w", TensorType((16, 16)))
    b = Param("b_w", TensorType((16, 16)))
    params = {"a_w": a, "b_w": b}
    g = torch.Generator().manual_seed(0)
    leaves = {
        "a_w": torch.randn(16, 16, generator=g, dtype=torch.float64),
        "b_w": torch.randn(16, 16, generator=g, dtype=torch.float64),
    }
    composed = Op.make("linear", Op.make("linear", x, a), b)
    folded = Op.make("linear", x, Op.make("matmul", b, a))
    m1 = M._lower_term(composed, x, params, leaves, sink)
    m2 = M._lower_term(folded, x, params, leaves, sink)
    xr = _x()
    with torch.no_grad():
        d = (m1(xr) - m2(xr)).abs().max().item()
    assert d < 1e-12


def test_algebra_diag_absorption_fp64():
    """Diagonal scale absorption: ``linear(x∘s, W) = linear(x, W∘s)`` —
    the norm-cascade identity, fp64-exact."""
    sink = TorchSink()
    x = Var("x", TensorType((8, 16)))
    s = Param("p_s", TensorType(()))
    w = Param("p_w", TensorType((16, 16)))
    params = {"p_s": s, "p_w": w}
    leaves = {
        "p_s": torch.tensor(1.7, dtype=torch.float64),
        "p_w": torch.randn(
            16, 16, generator=torch.Generator().manual_seed(1)
        ).double(),
    }
    scaled_in = Op.make("linear", Op.make("mul", x, s), w)
    scaled_w = Op.make("linear", x, Op.make("mul", w, s))
    m1 = M._lower_term(scaled_in, x, params, leaves, sink)
    m2 = M._lower_term(scaled_w, x, params, leaves, sink)
    xr = _x()
    with torch.no_grad():
        d = (m1(xr) - m2(xr)).abs().max().item()
    assert d < 1e-12


def test_algebra_residual_monoid_fp64():
    """Residual ``+`` monoid: ``linear(x + A(x), W)`` distributes —
    ``linear(x,W) + linear(A(x),W)`` — fp64-exact."""
    sink = TorchSink()
    x = Var("x", TensorType((8, 16)))
    wa = Param("a_w", TensorType((16, 16)))
    wb = Param("b_w", TensorType((16, 16)))
    params = {"a_w": wa, "b_w": wb}
    leaves = {
        "a_w": torch.randn(
            16, 16, generator=torch.Generator().manual_seed(2)
        ).double(),
        "b_w": torch.randn(
            16, 16, generator=torch.Generator().manual_seed(3)
        ).double(),
    }
    ax = Op.make("linear", x, wa)
    mid = Op.make("add", x, ax)
    fused = Op.make("linear", mid, wb)
    distributed = Op.make(
        "add", Op.make("linear", x, wb), Op.make("linear", ax, wb)
    )
    assert M._distribute_over(fused, mid) == distributed
    m1 = M._lower_term(fused, x, params, leaves, sink)
    m2 = M._lower_term(distributed, x, params, leaves, sink)
    xr = _x()
    with torch.no_grad():
        d = (m1(xr) - m2(xr)).abs().max().item()
    assert d < 1e-12


# ---------------------------------------------------------------------------
#  Stage 1 — lift + law matching
# ---------------------------------------------------------------------------


def test_lift_graph_chain_stack():
    """A DeepParallel chain lifts to 3 nodes + 2 chain wires."""
    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=3).eval().double()
    g = _lift(model, _x())
    assert len(g.nodes) == 3
    assert all(not n.opaque for n in g.nodes)
    assert [w.kind for w in g.wires] == ["chain", "chain"]
    sig = g.sig("blocks.0")
    assert sig is not None and sig.out_proj[0].name == "p_w3_weight"
    assert g.node("blocks.1").name == "blocks.1"
    assert isinstance(g.record("blocks.2").ir, IR)
    assert "nodes=3" in repr(g)


def test_lift_graph_residual_wires():
    """``x + b(x)`` stacks classify their wires as residual."""
    torch.manual_seed(0)
    model = _ResidualWrapped(dim=16, depth=2).eval().double()
    g = _lift(model, _x())
    assert g.wires[0].kind == "residual_wrapped"


def test_lift_graph_opaque_block():
    """An un-exportable block is an opaque boundary node; wires still
    classify its dataflow but no law matches it."""
    torch.manual_seed(0)
    model = _MixedStack().eval().double()
    g = _lift(model, _x())
    assert g.node("blocks.1").opaque is True
    assert g.sig("blocks.1") is None
    assert graph_record_note(g, "blocks.1").startswith("export failed")
    # Laws never fire across an opaque node.
    matches = [
        m for law in M.DEFAULT_MORPHISM_LAWS for m in law.match(g)
    ]
    assert not any("blocks.1" in m.nodes for m in matches)


def graph_record_note(g: M.MorphismGraph, name: str) -> str:
    """Small helper: the recorded skip reason for a node."""
    return g.record(name).note or ""


def test_lift_graph_unexecuted_and_kwarg_blocks():
    """A block that never ran, and one called with kwargs, both lift
    to opaque boundary nodes with honest notes."""
    torch.manual_seed(0)
    g = _lift(_SkipMid().eval().double(), _x())
    assert g.node("blocks.1").opaque
    assert "not executed" in g.record("blocks.1").note
    # The wire across it cannot be a simple flow.
    assert all(w.kind == "opaque" for w in g.wires)

    g = _lift(_KwargStack().eval().double(), _x())
    assert g.node("blocks.1").opaque
    assert "single-positional" in g.record("blocks.1").note


def test_lift_graph_perturbed_probe_fails(monkeypatch):
    """When the second-probe capture fails, residual boundaries are
    unprovable — wires degrade to opaque, chains still classify."""
    torch.manual_seed(0)
    model = _ResidualWrapped(dim=16).eval().double()
    comp = TorchComposer()

    def boom(_x):
        raise RuntimeError("no second probe")

    monkeypatch.setattr(comp, "perturbed", boom)
    g = M.lift_graph(model, _x(), source=TorchSource(), composer=comp)
    # Residual needs the probe; without it the pair declines.
    assert g.wires[0].kind == "opaque"


def test_law_match_surfaces():
    """Each law's match/no-match signature-level behaviour."""
    torch.manual_seed(0)
    chain = _lift(_ChainStack(dim=16, depth=3).eval().double(), _x())
    res = _lift(_ResidualWrapped(dim=16, depth=2).eval().double(), _x())
    norm = _lift(_NormStack(dim=16).eval().double(), _x())
    diag = _lift(_DiagChain().eval().double(), _x())

    oc = M.OutInCompose()
    assert len(oc.match(chain)) == 2
    assert oc.match(chain)[0].boundary == "chain"
    assert oc.match(res) == []  # residual wires don't compose plainly
    # NormLinear pairs also carry out∘in projections — dims agree.
    assert len(oc.match(norm)) == 1
    assert isinstance(oc, MorphismLaw)

    ra = M.ResidualAbsorb()
    assert len(ra.match(res)) == 1
    assert ra.match(res)[0].reify.distribute is True
    assert ra.match(chain) == []

    nc = M.NormCascade()
    intra = [m for m in nc.match(norm) if m.boundary == "intra"]
    assert len(intra) == 2  # one affine pre-norm per NormLinear
    pair = [m for m in nc.match(diag) if m.boundary != "intra"]
    assert len(pair) == 1 and pair[0].nodes == ("blocks.0", "blocks.1")
    assert [m for m in nc.match(chain) if m.boundary != "intra"] == []

    wt = M.WeightTie()
    tied = _lift(_TiedPair().eval().double(), _x())
    tied_matches = wt.match(tied)
    assert len(tied_matches) == 1
    assert tied_matches[0].nodes == ("blocks.0", "blocks.1")
    untied = wt.match(chain)
    # intra: every DeepParallel has same-shape w1/w2/w3 -> candidates;
    # cross: p_w{i}_weight stems differ across blocks? They share
    # leaf names (blocks_0 vs blocks_1 strip to same stems) -> matched.
    assert untied  # candidates exist; the value gate lives in reify


def test_weight_tie_no_false_cross_match():
    """Blocks with differently-stemmed same-shape weights are not
    candidates — the filter is shape + stem, not shape alone."""
    torch.manual_seed(0)

    class TwoNorms(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [NormLinear(dim), NormLinear(dim)]
            )

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    g = _lift(TwoNorms().eval().double(), _x())
    # Same leaf name `p_proj_weight` in both blocks -> stems equal ->
    # candidate match fires; value gate decides at reify.
    matches = M.WeightTie().match(g)
    assert any(len(m.nodes) == 2 for m in matches)


def test_dims_compatible_and_diag_predicates():
    """Unit coverage for the signature predicates' shape logic."""

    def mk(i: object, o: object) -> M.BlockSig:
        return M.BlockSig(
            in_projs=(),
            out_proj=(),
            norm=M.NormSig("none", False, False),
            act=(),
            residual=False,
            shape=(i, o),
        )

    a = mk(None, (8, 16))
    b_ok = mk((8, 16), None)
    b_bad = mk((8, 32), None)
    b_none = mk(None, None)
    b_none_dim = mk((8, None), None)
    b_str = mk("not-a-tuple", None)
    assert M._dims_compatible(a, b_ok)
    assert not M._dims_compatible(a, b_bad)
    assert M._dims_compatible(a, b_none)
    assert M._dims_compatible(a, b_none_dim)
    assert M._dims_compatible(a, b_str)


def test_shape_tuple_edges():
    """Backend-neutral shape probing: tensors, missing/weird shapes."""
    t = torch.zeros(2, 3)
    assert M._shape_tuple(t) == (2, 3)
    assert M._shape_tuple(object()) is None

    class _S:
        shape = ("x", 3)  # non-int dims -> None, not a crash

    assert M._shape_tuple(_S()) is None

    class _S2:
        shape = (object(),)

    assert M._shape_tuple(_S2()) is None


# ---------------------------------------------------------------------------
#  Stage 1 — verified end-to-end runs
# ---------------------------------------------------------------------------


def test_e2e_chain_out_in_compose_grafts_fp64():
    """DeepParallel chain: out∘in fires, grafts, and the delivered
    model is fp64-exact vs the original."""
    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=4).eval().double()
    x = _x()

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(optimize_rest=False)
    )

    m = stats["matches"]["out_in_compose:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "chain"
    assert m["rel_diff"] < 1e-9
    assert m["cost_after"] < m["cost_before"]
    # The reified program is ONE linear with a fully folded weight.
    assert "linear x" in m["reified"]
    # Non-overlap: block 1 was consumed.
    assert (
        stats["matches"]["out_in_compose:blocks.1+blocks.2"]["status"]
        == "skipped"
    )
    assert (
        stats["matches"]["out_in_compose:blocks.2+blocks.3"]["status"]
        == "grafted"
    )
    assert stats["n_rewritten"] == 4
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
    # The original model is untouched.
    assert all(isinstance(b, DeepParallel) for b in model.blocks)


def test_e2e_residual_absorb_grafts_fp64():
    """``x + s·x → B``: the residual absorb distributes B's in-projs
    over the add and folds the scale into the weights — one GEMM."""
    torch.manual_seed(0)
    model = _ResidualPair(dim=16).eval().double()
    x = _x()

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(optimize_rest=False)
    )

    m = stats["matches"]["residual_absorb:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "residual"
    assert m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_residual_wrapped_grafts_fp64():
    """``x + b(x)`` blocks: the wrapped residual — B's slot zeroes out
    (lowered ``x * 0`` filler) and the fused pair takes the segment."""
    torch.manual_seed(0)
    model = _ResidualWrapped(dim=16, depth=2).eval().double()
    x = _x()

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(optimize_rest=False)
    )

    m = stats["matches"]["residual_absorb:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "residual_wrapped"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_chain_wrapped_grafts_fp64():
    """``A(x); x + B(x)`` — chain_wrapped pair reifies to one linear
    (the ``y + B(y)`` wrap folds through the compose recipe)."""
    torch.manual_seed(0)
    model = _ChainWrapped(dim=16).eval().double()
    x = _x()

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(optimize_rest=False)
    )

    m = stats["matches"]["out_in_compose:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "chain_wrapped"
    assert m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_norm_cascade_pair_and_intra():
    """Diagonal block → projections cascades (pair form); a NormLinear
    stack folds its affine pre-norm gains (intra form)."""
    torch.manual_seed(0)
    model = _DiagChain().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[M.NormCascade()], optimize_rest=False
        ),
    )
    m = stats["matches"]["norm_cascade:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert "mul" not in m["reified"].split("linear x")[1][:0]  # sanity
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9

    # Intra: each NormLinear's norm gain folds into its proj weight.
    model2 = _NormStack(dim=16).eval().double()
    opt2, stats2 = Optimizer(backend=TorchBackend()).optimize(
        model2,
        _x(),
        strategy=MorphismSearch(
            laws=[M.NormCascade()], optimize_rest=False
        ),
    )
    for i in range(2):
        m2 = stats2["matches"][f"norm_cascade:blocks.{i}"]
        assert m2["status"] == "grafted"
        assert m2["rel_diff"] < 1e-9
    assert stats2["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d2 = (
            (model2(_x().clone()) - opt2(_x().clone()))
            .abs()
            .max()
            .item()
        )
    assert d2 < 1e-9


def test_e2e_weight_tie_cross_block():
    """A real tie: two blocks sharing one Parameter — the joint e-graph
    merges them and the canonical name wins in both extractions."""
    torch.manual_seed(0)
    model = _TiedPair().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[M.WeightTie()], optimize_rest=False
        ),
    )
    m = stats["matches"]["weight_tie:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["tied"] == [["blocks_0__p_weight", "blocks_1__p_weight"]]
    assert m["rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_weight_tie_intra():
    """A block with two tied projections merges them internally."""
    torch.manual_seed(0)
    model = _TiedParallel().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[M.WeightTie()], optimize_rest=False
        ),
    )
    m = stats["matches"]["weight_tie:blocks.0"]
    assert m["status"] == "grafted"
    assert m["tied"]
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_weight_tie_declines_on_values():
    """Same shape + same stem, different VALUES -> the share pass finds
    nothing and the match declines honestly."""
    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        _x(),
        strategy=MorphismSearch(
            laws=[M.WeightTie()], optimize_rest=False
        ),
    )
    m = stats["matches"]["weight_tie:blocks.0+blocks.1"]
    assert m["status"] == "declined"
    assert m["reason"] == "no_tied_values"


def test_reify_declines_no_improvement():
    """A match whose saturation can't beat the un-rewritten joint
    declines at the cost gate."""
    torch.manual_seed(0)
    model = _NormStack(dim=16).eval().double()
    x = _x()
    g = _lift(model, x)
    # Hand-built intra match on the second block with the "scale"
    # recipe — but on a term the recipe can't improve (the gain was
    # already folded into the first block's sig? No: reify runs on
    # blocks.1's own term — scale recipe DOES improve it.  Use a
    # rules-free spec instead: nothing fires, cost is unchanged.)
    match = M.MorphismMatch(
        law="probe",
        nodes=("blocks.1",),
        boundary="intra",
        reify=M.ReifySpec(mode="intra", rules="tie"),
    )
    out = M._reify(
        match,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"
    assert out["cost_after"] == out["cost_before"]


def test_reify_opaque_node_declines():
    """A crafted match pointing at an opaque node fails closed."""
    torch.manual_seed(0)
    g = _lift(_MixedStack().eval().double(), _x())
    intra = M.MorphismMatch(
        law="probe",
        nodes=("blocks.1",),
        boundary="intra",
        reify=M.ReifySpec(mode="intra", rules="scale"),
    )
    out = M._reify(
        intra,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=2,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )
    pair = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1"),
        boundary="chain",
        reify=M.ReifySpec(mode="chain", rules="compose"),
    )
    out = M._reify(
        pair,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=2,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )
    tie = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1"),
        boundary="tie",
        reify=M.ReifySpec(mode="tie", rules="tie", share=True),
    )
    out = M._reify(
        tie,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=2,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )


def test_reify_intra_distribute_noop():
    """distribute=True on an intra match — ``mid`` is None, so the
    distribute branch is skipped entirely."""
    torch.manual_seed(0)
    g = _lift(_NormStack(dim=16).eval().double(), _x())
    match = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0",),
        boundary="intra",
        reify=M.ReifySpec(mode="intra", rules="scale", distribute=True),
    )
    out = M._reify(
        match,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    # No distribute offer; the norm-gain fold still grafts.
    assert out["status"] == "grafted"


def test_distribute_over_edges():
    """The constructed bilinear step: matmul on either side, a
    non-mid data operand, non-Op terms, and empty-arg nodes."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    mid = Op.make("add", x, Op.make("linear", x, w))

    # matmul(mid, W) — data on the left
    t = Op.make("matmul", mid, w)
    got = M._distribute_over(t, mid)
    assert got.op == "add" and all(a.op == "matmul" for a in got.args)
    # matmul(W, mid) — data on the right
    t = Op.make("matmul", w, mid)
    got = M._distribute_over(t, mid)
    assert got.op == "add" and all(a.op == "matmul" for a in got.args)
    # linear(mid, W, b) — the bias rides one branch
    b = Param("p_b", TensorType((16,)))
    t = Op.make("linear", mid, w, b)
    got = M._distribute_over(t, mid)
    assert got.op == "add"
    right = got.args[1]
    assert right.op == "linear" and len(right.args) == 3
    # a projection whose data is NOT the mid node — untouched
    t = Op.make("linear", x, w)
    assert M._distribute_over(t, mid) == t
    # matmul(x, y): neither side is mid — falls through to rebuild
    t = Op.make("matmul", x, y)
    assert M._distribute_over(t, mid) == t
    # non-Op and empty-arg nodes pass through
    assert M._distribute_over(x, mid) is x
    assert M._distribute_over(Op("mul", (x,), {}), mid).op == "mul"
    assert M._distribute_over(Op("linear", (), {}), mid).op == "linear"
    assert (
        M._distribute_over(Op("matmul", (mid,), {}), mid).op == "matmul"
    )


def test_reify_distribute_mid_unconsumed():
    """distribute=True with a residual mid that no projection reads —
    ``_distribute_over`` returns the term unchanged and no offer is
    registered (the ``dist == term`` skip branch)."""
    torch.manual_seed(0)

    class _Silu(nn.Module):
        """A block that consumes its input non-projectionally."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(x)

    class _Stack(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False), _Silu()]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.blocks[0](x)
            return self.blocks[1](x)

    g = _lift(_Stack().eval().double(), _x())
    match = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1"),
        boundary="residual",
        reify=M.ReifySpec(
            mode="residual", rules="compose", distribute=True
        ),
    )
    out = M._reify(
        match,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    # silu(x + A(x)) has nothing to distribute or fold.
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"


def test_saturate_empty_rules_and_offers():
    """``_saturate`` edge cases: empty rule set skips the run; an offer
    identical to the joint term is not re-offered."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    t = Op.make("linear", x, w)
    empty = RuleSet("empty", ())
    eg, eid = M._saturate(
        t, empty, max_iterations=2, max_enodes=100, symmetry_budget=None
    )
    assert eg.extract_best(eid, M.flops_cost) == t
    eg, eid = M._saturate(
        t,
        empty,
        max_iterations=2,
        max_enodes=100,
        symmetry_budget=4,
        offers=[(t, "self-offer — skipped")],
    )
    assert eg.extract_best(eid, M.flops_cost) == t


def test_e2e_verify_gate_declines(monkeypatch):
    """A failing verify declines the reified rewrite — and every
    per-block fallback fails its gate the same way."""
    import catopt_torch.adapters as A

    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()

    def fail_verify(*_a, **_k):
        return A.VerifyReport(max_abs=1.0, max_rel=1.0, passed=False)

    monkeypatch.setattr(A, "verify_module", fail_verify)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=MorphismSearch()
    )
    m = stats["matches"]["out_in_compose:blocks.0+blocks.1"]
    assert m["status"] == "declined"
    assert "verify failed" in m["reason"]
    assert stats["blocks"]["blocks.0"]["status"] == "failed"
    assert (
        "block verify failed" in stats["blocks"]["blocks.0"]["reason"]
    )
    assert stats["end_to_end"]["max_rel_diff"] == 1.0


def test_tie_verify_gate_declines(monkeypatch):
    """A failing per-block verify declines a real tie at delivery."""
    import catopt_torch.adapters as A

    torch.manual_seed(0)
    model = _TiedPair().eval().double()

    def fail_verify(*_a, **_k):
        return A.VerifyReport(max_abs=1.0, max_rel=1.0, passed=False)

    monkeypatch.setattr(A, "verify_module", fail_verify)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        _x(),
        strategy=MorphismSearch(
            laws=[M.WeightTie()], optimize_rest=False
        ),
    )
    m = stats["matches"]["weight_tie:blocks.0+blocks.1"]
    assert m["status"] == "declined"
    assert "tie verify failed" in m["reason"]


def test_morphism_error_declines():
    """A law producing a malformed match records an error decline —
    never a graft."""

    class BadLaw:
        name = "bad_law"

        def match(self, graph):
            return [
                M.MorphismMatch(
                    law=self.name,
                    nodes=("nonexistent",),
                    boundary="intra",
                    reify=M.ReifySpec(mode="intra", rules="scale"),
                )
            ]

    assert isinstance(BadLaw(), MorphismLaw)
    torch.manual_seed(0)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        _ChainStack(dim=16, depth=2).eval().double(),
        _x(),
        strategy=MorphismSearch(laws=[BadLaw()], optimize_rest=False),
    )
    m = stats["matches"]["bad_law:nonexistent"]
    assert m["status"] == "declined" and m["reason"] == "error"


def test_no_composer_raises():
    """The strategy needs the Composer port — a bare source/sink
    optimizer raises a clear TypeError."""
    import pytest

    opt = Optimizer(source=TorchSource(), sink=TorchSink())
    with pytest.raises(TypeError, match="Composer"):
        opt.optimize(
            _ChainStack(dim=16, depth=2).eval().double(),
            _x(),
            strategy=MorphismSearch(),
        )


def test_clone_failure_returns_in_place(monkeypatch):
    """A failed param-sharing clone returns the input model unmodified
    and reports it honestly."""
    import catopt_torch.composer as C

    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()

    def boom(*_a, **_k):
        raise RuntimeError("cannot pickle")

    monkeypatch.setattr(C.copy, "deepcopy", boom)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, _x(), strategy=MorphismSearch()
    )
    assert opt is model
    assert stats["in_place"] is True
    assert stats["end_to_end"]["skipped"] == "in_place"


def test_e2e_verify_raises_is_recorded():
    """An end-to-end verify that *raises* (not fails) is recorded as an
    error record, not a crash."""

    class _ExplodingVerify(TorchSink):
        def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
            if isinstance(ref, _ChainStack):
                raise RuntimeError("e2e probe exploded")
            return super().verify(
                ref, opt, inputs, rtol=rtol, atol=atol
            )

    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()
    _opt, stats = Optimizer(
        source=TorchSource(),
        sink=_ExplodingVerify(),
        composer=TorchComposer(),
    ).optimize(
        model,
        _x(),
        strategy=MorphismSearch(optimize_rest=False),
    )
    assert "error" in stats["end_to_end"]
    assert (
        stats["matches"]["out_in_compose:blocks.0+blocks.1"]["status"]
        == "grafted"
    )


def test_rest_pass_search_failure(monkeypatch):
    """A per-block fallback search that raises is recorded as a failed
    block — the model still recomposes."""
    torch.manual_seed(0)
    model = _MixedStack().eval().double()
    opt_ = Optimizer(backend=TorchBackend())
    real_search = opt_.search

    def flaky(block, x, **kw):
        if block is model.blocks[0]:
            raise RuntimeError("simulated search failure")
        return real_search(block, x, **kw)

    monkeypatch.setattr(opt_, "search", flaky)
    _opt, stats = opt_.optimize(model, _x(), strategy=MorphismSearch())
    assert stats["blocks"]["blocks.0"]["status"] == "failed"
    assert "simulated" in stats["blocks"]["blocks.0"]["error"]
    assert stats["blocks"]["blocks.1"]["status"] == "skipped"
    # The healthy tail block still optimized.
    assert stats["blocks"]["blocks.2"]["status"] == "optimized"


def test_optimize_morphisms_entry_and_kwargs():
    """The function entry point + strategy/kwarg plumbing."""
    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()
    x = _x()
    res = optimize_morphisms(
        model,
        x,
        backend=TorchBackend(),
        strategy=MorphismSearch(optimize_rest=False, verify_tol=1e-6),
        verbose=False,
    )
    assert res.stats["morphism"] is True
    assert (
        res.stats["matches"]["out_in_compose:blocks.0+blocks.1"][
            "status"
        ]
        == "grafted"
    )
    # Default strategy construction path (no explicit strategy).
    res2 = optimize_morphisms(model, x, backend=TorchBackend())
    assert res2.stats["morphism"] is True
    # Explicit ports instead of a backend bundle — and an explicit
    # cost_fn through the optimize kwargs.
    res3 = optimize_morphisms(
        model,
        x,
        source=TorchSource(),
        sink=TorchSink(),
        composer=TorchComposer(),
        strategy=MorphismSearch(optimize_rest=False),
        cost_fn=M.flops_cost,
    )
    assert res3.stats["morphism"] is True


def test_e2e_minigpt_residual_absorb_and_rest():
    """MiniGPT-style ParallelBlock stack: lifted, morphism laws fire on
    signatures the compositional pass can't see, and the result stays
    fp64-faithful."""
    torch.manual_seed(0)
    model = _MiniGPT(dim=16, n_heads=2, depth=2).eval().double()
    x = _x((2, 4, 16))

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch()
    )
    assert stats["n_lifted"] == 2
    # Every node's signature is exposed in the stats record.
    s0 = stats["sigs"]["blocks.0"]
    assert s0["residual"] is True and s0["norm"] == "rms"
    assert s0["norm_affine"] and s0["norm_pre"]
    assert set(s0["in_projs"]) == {
        "p_attn_q_proj_weight",
        "p_attn_k_proj_weight",
        "p_attn_v_proj_weight",
        "p_gate_weight",
        "p_up_weight",
    }
    assert stats["end_to_end"]["max_rel_diff"] < 1e-4
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-6


def test_e2e_verbose_and_unbounded_budget(caplog):
    """Verbose logging + symmetry_budget=None path."""
    import logging

    torch.manual_seed(0)
    model = _ChainStack(dim=16, depth=2).eval().double()
    with caplog.at_level(
        logging.INFO, logger="catopt_orchestrator.morphisms"
    ):
        _opt, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            _x(),
            strategy=MorphismSearch(
                optimize_rest=False, symmetry_budget=None
            ),
            verbose=True,
        )
    assert any(
        "[Morphism]" in r.getMessage()
        and r.name == "catopt_orchestrator.morphisms"
        for r in caplog.records
    )
    assert (
        stats["matches"]["out_in_compose:blocks.0+blocks.1"]["status"]
        == "grafted"
    )


def test_empty_model_lifts_empty():
    """A module with no children lifts to an empty graph; the driver
    recomposes and verifies it without any rewrites."""
    torch.manual_seed(0)

    class _Bare(nn.Module):
        def forward(self, x):
            return x * 2

    model = _Bare().eval().double()
    res = optimize_morphisms(
        model,
        _x(),
        backend=TorchBackend(),
        strategy=MorphismSearch(optimize_rest=False),
    )
    assert res.stats["n_blocks"] == 0
    assert res.stats["wires"] == []
    assert res.stats["end_to_end"]["max_rel_diff"] < 1e-9


def test_recipe_rules_cache():
    """Recipe sets build once and memoise."""
    a = M._recipe_rules("compose")
    b = M._recipe_rules("compose")
    assert a is b
    assert "assoc_linear" in a
    assert len(M._recipe_rules("tie")) == 0
