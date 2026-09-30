"""Cross-block CSE morphism law tests — shared subterms, verified reify.

Sibling of ``test_morphism_kv.py``.  Pins the ``cross_block_cse`` law
end to end: same-input family detection (``in_obj`` object identity),
structural equality modulo leaf renaming (Param leaves certified by
bitwise-equal values), the additive-consumption wiring evidence on
both capture probes, and the reified fused program — every grafted
rewrite verified fp64 against the original.

The honest non-reach is pinned too: sequential residual/chain blocks
see different stream objects, so no intermediate of block i is
consumable at block j — no match is ever emitted there.
"""

import copy

import catopt_orchestrator.crossblock_cse as C
import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import numpy as np
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
            _SharedNormBlock(self.norm, dim, 10 + i)
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


class _SharedWBlock(nn.Module):
    """``up(gelu(x @ w))`` — shares the ``gelu(x @ w)`` recompute."""

    def __init__(self, w: torch.Tensor, dim: int, seed: int) -> None:
        """Clone *w* (equal VALUES, distinct object) + own ``up``."""
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.up = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.up.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through the shared weight, activate, upsample."""
        return self.up(F.gelu(x @ self.w))


class _SharedWStack(nn.Module):
    """``y = Σ b_i(x)`` — value-equal (not object-equal) weights."""

    def __init__(self, dim: int = 16, shared: bool = True) -> None:
        """Two blocks through the same-values projection weight."""
        super().__init__()
        g = torch.Generator().manual_seed(99)
        w = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        w2 = w if shared else torch.randn(
            dim, dim, generator=g, dtype=torch.float64
        )
        self.blocks = nn.ModuleList(
            [_SharedWBlock(w, dim, 30), _SharedWBlock(w2, dim, 31)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _RecomputeBlock(nn.Module):
    """Block j recomputing a sibling's whole body as a subterm.

    ``proj(norm(x)) + extra(x)`` where ``(norm, proj)`` is an
    equal-valued clone of the sibling's — the producer's entire body
    appears as a *subterm* here.
    """

    def __init__(
        self, twin: _SharedNormBlock, dim: int, seed: int
    ) -> None:
        """Clone the sibling's norm+proj (equal values); own extra."""
        super().__init__()
        self.norm = twin.norm
        self.proj = copy.deepcopy(twin.proj)
        self.extra = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.extra.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recompute the sibling body, add an extra projection."""
        return self.proj(self.norm(x)) + self.extra(x)


class _RecomputeStack(nn.Module):
    """``y = b0(x) + b1(x)`` — b1 contains b0's whole computation."""

    def __init__(self, dim: int = 16) -> None:
        """One plain block plus the recomputing clone."""
        super().__init__()
        norm = nn.LayerNorm(dim).double()
        b0 = _SharedNormBlock(norm, dim, 40)
        self.blocks = nn.ModuleList([b0, _RecomputeBlock(b0, dim, 41)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _TwinStack(nn.Module):
    """Two value-identical blocks — the whole body is shared."""

    def __init__(self, dim: int = 16) -> None:
        """Deepcopy twins — equal values, distinct param objects."""
        super().__init__()
        norm = nn.LayerNorm(dim).double()
        b0 = _SharedNormBlock(norm, dim, 50)
        self.blocks = nn.ModuleList([b0, copy.deepcopy(b0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the twin outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


class _SeqNormStack(nn.Module):
    """``x = x + b_i(x)`` — sequential blocks see different inputs."""

    def __init__(self, dim: int = 16) -> None:
        """Residual-chained blocks over one shared norm."""
        super().__init__()
        self.norm = nn.LayerNorm(dim).double()
        self.blocks = nn.ModuleList(
            _SharedNormBlock(self.norm, dim, 60 + i)
            for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain the blocks on the evolving stream."""
        for b in self.blocks:
            x = x + b(x)
        return x


class _MulNormStack(nn.Module):
    """``y = b0(x) * b1(x)`` — shared input, non-additive outputs."""

    def __init__(self, dim: int = 16) -> None:
        """Two shared-norm blocks, multiplied."""
        super().__init__()
        self.norm = nn.LayerNorm(dim).double()
        self.blocks = nn.ModuleList(
            _SharedNormBlock(self.norm, dim, 70 + i)
            for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the member outputs — no additive slot."""
        return self.blocks[0](x) * self.blocks[1](x)


class _FanoutNormStack(nn.Module):
    """One member's output escapes to a non-additive consumer."""

    def __init__(self, dim: int = 16) -> None:
        """Two shared-norm blocks; b1's output also feeds ``c``."""
        super().__init__()
        self.norm = nn.LayerNorm(dim).double()
        self.blocks = nn.ModuleList(
            _SharedNormBlock(self.norm, dim, 80 + i)
            for i in range(2)
        )
        self.c = nn.Linear(dim, dim, bias=False).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b1's output fans out: the sum AND a separate consumer."""
        b1 = self.blocks[1](x)
        return self.blocks[0](x) + b1 + self.c(b1)


class _AddXBlock(nn.Module):
    """``proj(x + x)`` — a param-free shared subterm only."""

    def __init__(self, dim: int, seed: int) -> None:
        """Own projection; the ``x + x`` subterm is leaf-free."""
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False).double()
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.randn(dim, dim, generator=g, dtype=torch.float64)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double the input, then project."""
        return self.proj(x + x)


class _AddXStack(nn.Module):
    """Two ``x + x`` blocks — the shared subterm is already interned."""

    def __init__(self, dim: int = 16) -> None:
        """Same param-free subterm, different weights."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_AddXBlock(dim, 90), _AddXBlock(dim, 91)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        return self.blocks[0](x) + self.blocks[1](x)


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
    """Call the cse reify with the standard knobs."""
    args = dict(
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    args.update(kw)
    return C._reify_cse(match, graph, **args)


def _match(model: nn.Module) -> list:
    """Lift and match — the law's firings on *model*."""
    torch.manual_seed(0)
    g = _lift(model.eval().double(), _x())
    return C.CrossBlockCSE().match(g)


# ---------------------------------------------------------------------------
#  Units — leaf equality and same-computation
# ---------------------------------------------------------------------------


def test_leaf_same():
    """Every branch of the leaf-equality gate."""
    leaves = {}
    x = Var("x", TensorType((8, 16)))
    pa = Param("a__p", TensorType((4, 4)))
    pb = Param("b__p", TensorType((4, 4)))
    pc = Param("c__p", TensorType((4, 4)))
    t = torch.ones(4, 4, dtype=torch.float64)
    leaves.update({"a__p": t, "b__p": t.clone()})
    # identical object / structurally equal leaves
    assert C._leaf_same(x, x, leaves)
    assert C._leaf_same(
        Var("x", TensorType((8, 16))), x, leaves
    )
    # Var vs Param — different kinds never match
    assert not C._leaf_same(x, pa, leaves)
    # missing leaf values are unprovable
    assert not C._leaf_same(pa, Param("absent", TensorType((4, 4))), leaves)
    assert not C._leaf_same(Param("absent", TensorType((4, 4))), pa, leaves)
    # same leaf object (shared parameter read through two blocks)
    assert C._leaf_same(pa, pa, leaves)
    pbb = Param("bb__p", TensorType((4, 4)))
    leaves["bb__p"] = t  # the same tensor object under another name
    assert C._leaf_same(pa, pbb, leaves)
    assert C._leaf_same(pa, pb, leaves)  # bitwise-equal values
    # unequal tensor values
    leaves["c__p"] = torch.zeros(4, 4, dtype=torch.float64)
    assert not C._leaf_same(pa, pc, leaves)
    # non-tensor leaf values: plain equality (distinct objects)
    pe = Param("e__p", TensorType(()))
    pf = Param("f__p", TensorType(()))
    pz = Param("z__p", TensorType(()))
    leaves.update(
        {"e__p": float("1.5"), "f__p": 2.5, "z__p": float("1.5")}
    )
    assert C._leaf_same(pe, pz, leaves)  # equal non-tensor values
    assert not C._leaf_same(pe, pf, leaves)
    # non-bool ``==`` result (array-likes that are not _is_tensor)
    pg = Param("g__p", TensorType((2,)))
    ph = Param("h__p", TensorType((2,)))
    leaves.update({"g__p": np.array([1.0]), "h__p": np.array([1.0])})
    assert not C._leaf_same(pg, ph, leaves)


def test_same_computation():
    """Structural equality modulo leaf renaming, memoised."""
    x = Var("x", TensorType((8, 16)))
    wa = Param("a__w", TensorType((16, 16)))
    wb = Param("b__w", TensorType((16, 16)))
    wc = Param("c__w", TensorType((16, 16)))
    wv = torch.ones(16, 16, dtype=torch.float64)
    leaves = {"a__w": wv, "b__w": wv.clone(), "c__w": wv * 2}
    memo: dict = {}
    ta = Op.make("linear", x, wa)
    tb = Op.make("linear", x, wb)
    assert C._same_computation(ta, tb, leaves, memo)
    assert C._same_computation(ta, tb, leaves, {})  # fresh memo
    # memo hit path — second identical query is cached
    assert C._same_computation(ta, tb, leaves, memo)
    assert (ta, tb) in memo
    # identical term object
    assert C._same_computation(ta, ta, leaves, {})
    # unequal leaf values decline
    tc = Op.make("linear", x, wc)
    assert not C._same_computation(ta, tc, leaves, {})
    # op-name / attrs / arity mismatches
    assert not C._same_computation(
        ta, Op.make("gelu", x), leaves, {}
    )
    assert not C._same_computation(
        Op.make("transpose", x, dim0=0, dim1=1),
        Op.make("transpose", x, dim0=1, dim1=0),
        leaves,
        {},
    )
    bias = Param("a__b", TensorType((16,)))
    assert not C._same_computation(
        ta, Op.make("linear", x, wa, bias), leaves, {}
    )
    # child-level divergence under equal ops
    assert not C._same_computation(
        Op.make("mul", ta, wa),
        Op.make("mul", ta, wc),
        leaves,
        {},
    )
    # leaf mismatch inside an arg position
    assert not C._same_computation(
        Op.make("add", x, x),
        Op.make("add", x, wa),
        leaves,
        {},
    )


def test_family_shares_and_components():
    """The share scan: maximal outer match, identity-share skip,
    param-only skip, producer ordering."""
    x = Var("x", TensorType((8, 16)))
    wa = Param("a__w", TensorType((16, 16)))
    wb = Param("b__w", TensorType((16, 16)))
    wv = torch.ones(16, 16, dtype=torch.float64)
    leaves = {"a__w": wv, "b__w": wv.clone()}
    inner_a = Op.make("gelu", Op.make("matmul", x, wa))
    inner_b = Op.make("gelu", Op.make("matmul", x, wb))
    up_a = Param("a__u", TensorType((16, 16)))
    up_b = Param("b__u", TensorType((16, 16)))
    leaves.update(
        {
            "a__u": torch.randn(16, 16, dtype=torch.float64),
            "b__u": torch.randn(16, 16, dtype=torch.float64),
        }
    )
    body_a = Op.make("linear", inner_a, up_a)
    body_b = Op.make("linear", inner_b, up_b)
    shares, links = C._family_shares(
        {"a": body_a, "b": body_b}, ["a", "b"], leaves
    )
    # one maximal site: the whole ``gelu(matmul(x, w))`` subterm
    assert list(shares) == ["b"] and len(shares["b"]) == 1
    assert links == [("a", "b")]

    # Reversed order: the later member is the consumer, never the
    # producer — order guards the i<j reading.
    shares, links = C._family_shares(
        {"a": body_b, "b": body_a}, ["a", "b"], leaves
    )
    assert list(shares) == ["b"] and links == [("a", "b")]

    # Identical term objects (param-free subterm) are already shared
    # in the joint DAG — recording them would save nothing.
    free_a = Op.make("mul", Op.make("add", x, x), wa)
    free_b = Op.make("mul", Op.make("add", x, x), wb)
    # force wb's value to differ so the outer mul cannot match
    leaves["b__w"] = wv * 3
    shares, links = C._family_shares(
        {"a": free_a, "b": free_b}, ["a", "b"], leaves
    )
    assert shares == {} and links == []

    # A param-only Op in the body is skipped but descended past.
    pp = Op.make("matmul", wa, wa)
    body_c = Op.make("mul", Op.make("add", x, x), pp)
    shares, links = C._family_shares(
        {"a": free_a, "b": body_c}, ["a", "b"], leaves
    )
    assert shares == {} and links == []


def test_share_components():
    """Union-find over share links, ordered by execution position."""
    order = ["a", "b", "c", "d"]
    comps = C._share_components(
        [("a", "b"), ("b", "c"), ("c", "a")], order
    )
    assert comps == [("a", "b", "c")]
    comps = C._share_components([("b", "c"), ("a", "d")], order)
    assert comps == [("a", "d"), ("b", "c")]
    assert C._share_components([], order) == []


# ---------------------------------------------------------------------------
#  Match surface
# ---------------------------------------------------------------------------


def test_match_shared_norm_family():
    """The shared-norm recompute matches — one maximal site."""
    ms = _match(_SharedNormStack())
    law = C.CrossBlockCSE()
    assert isinstance(law, MorphismLaw)
    assert len(ms) == 1
    m = ms[0]
    assert m.nodes == ("blocks.0", "blocks.1")
    assert m.boundary == "family"
    assert m.reify.mode == "cse"
    assert "shared subterm" in m.detail
    assert "1" in m.detail


def test_match_value_equal_weights():
    """Equal-valued (not object-shared) weights share the subterm."""
    ms = _match(_SharedWStack())
    assert len(ms) == 1 and ms[0].nodes == ("blocks.0", "blocks.1")


def test_match_whole_body_subterm():
    """Block j recomputing block i's whole body as a subterm."""
    ms = _match(_RecomputeStack())
    assert len(ms) == 1
    assert ms[0].nodes == ("blocks.0", "blocks.1")


def test_match_twins():
    """Value-identical twins: the whole body shares."""
    ms = _match(_TwinStack())
    assert len(ms) == 1


def test_match_depth3():
    """Three siblings on one input: one fused family."""
    ms = _match(_SharedNormStack(depth=3))
    assert len(ms) == 1
    assert ms[0].nodes == ("blocks.0", "blocks.1", "blocks.2")


def test_match_no_fire_cases():
    """Honest non-reach: sequential inputs, product outputs, fan-out,
    distinct leaf values, already-interned param-free subterms."""
    for mk in (
        _SeqNormStack,
        _MulNormStack,
        _FanoutNormStack,
        _AddXStack,
        lambda: _SharedWStack(shared=False),
    ):
        assert _match(mk()) == [], getattr(mk, "__name__", mk)


def test_match_family_unusable_members():
    """A family member whose IR is missing or multi-input is dropped;
    fewer than two usable members is no match."""
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    g.record("blocks.1").ir = None  # post-lift mutation
    assert C.CrossBlockCSE().match(g) == []

    g = _lift(_SharedNormStack().eval().double(), _x())
    g.record("blocks.1").ir = IR(
        root=Var("y", TensorType((8, 16))),
        inputs=[
            Var("a", TensorType((8, 16))),
            Var("y", TensorType((8, 16))),
        ],
        input_names={"a", "y"},
    )
    assert C.CrossBlockCSE().match(g) == []


def test_match_evidence_probe_failure(monkeypatch):
    """A failed second-probe capture keeps single-capture evidence."""
    torch.manual_seed(0)
    model = _SharedNormStack().eval().double()
    comp = TorchComposer()

    def boom(_x):
        raise RuntimeError("no second probe")

    monkeypatch.setattr(comp, "perturbed", boom)
    g = M.lift_graph(model, _x(), source=TorchSource(), composer=comp)
    assert g._probe2 is False
    assert len(C.CrossBlockCSE().match(g)) == 1


def test_match_probe2_contradiction():
    """Evidence on capture 1 but not the probe declines the family."""
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    rec = g.record("blocks.1")
    rec.out_val2 = torch.zeros_like(rec.out_val2) + 99.0
    assert C.CrossBlockCSE().match(g) == []


# ---------------------------------------------------------------------------
#  Reify — declines and grafts
# ---------------------------------------------------------------------------


def _family_match(graph):
    """The law's single match on a lifted graph."""
    return C.CrossBlockCSE().match(graph)[0]


def test_reify_graft_fp64():
    """The family graft: fused slot + zero filler, verified fp64."""
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert out["status"] == "grafted"
    assert out["rel_diff"] < 1e-9
    assert out["cost_after"] < out["cost_before"]
    assert out["shared_sites"] == 1
    assert out["shared_ops"] == ["layer_norm"]
    assert set(out["reps"]) == {"blocks.0", "blocks.1"}
    with torch.no_grad():
        z = out["reps"]["blocks.1"](_x((4, 16)))
    assert torch.equal(z, torch.zeros(4, 16, dtype=torch.float64))


def test_reify_twins_and_recompute():
    """Whole-body shares graft too."""
    torch.manual_seed(0)
    for mk in (_TwinStack, _RecomputeStack):
        g = _lift(mk().eval().double(), _x())
        out = _reify(_family_match(g), g)
        assert out["status"] == "grafted", mk.__name__
        assert out["rel_diff"] < 1e-9


def test_reify_decline_paths():
    """Every honest decline: opaque, multi-input, lost shares."""
    torch.manual_seed(0)

    # --- opaque node
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").ir = None
    out = _reify(m, g)
    assert (
        out["status"] == "declined" and out["reason"] == "opaque node"
    )

    # --- multi-input IR
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").ir = IR(
        root=Var("y", TensorType((8, 16))),
        inputs=[
            Var("a", TensorType((8, 16))),
            Var("y", TensorType((8, 16))),
        ],
        input_names={"a", "y"},
    )
    out = _reify(m, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "multi-input block"
    )

    # --- the certified leaf equality vanished between match and reify
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    w = g.record("blocks.1").leaves["p_norm_weight"]
    g.record("blocks.1").leaves["p_norm_weight"] = w * 3
    out = _reify(m, g)
    assert (
        out["status"] == "declined"
        and out["reason"] == "no shared subterm"
    )


def test_reify_decline_no_improvement():
    """A flat cost model sees no DAG saving — declined on cost."""
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    out = _reify(
        _family_match(g), g, cost_fn=lambda t, memo=None: 0.0
    )
    assert (
        out["status"] == "declined" and out["reason"] == "no_improvement"
    )


def test_reify_decline_shape_guards():
    """The additive-sum slot preconditions, checked before graft."""
    torch.manual_seed(0)

    # member output not a tensor
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").out_val = "not-a-tensor"
    out = _reify(m, g)
    assert out["status"] == "declined" and "outputs" in out["reason"]

    # member outputs different shapes — can't sum honestly
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    g.record("blocks.1").out_val = torch.zeros(
        8, 8, dtype=torch.float64
    )
    out = _reify(m, g)
    assert out["status"] == "declined" and "same shape" in out["reason"]

    # consumed slot needs in-shape == out-shape
    g = _lift(_SharedNormStack().eval().double(), _x())
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
    g = _lift(_SharedNormStack().eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert out["status"] == "declined"
    assert "verify failed" in out["reason"]


# ---------------------------------------------------------------------------
#  End to end
# ---------------------------------------------------------------------------


def test_e2e_shared_norm_grafts_fp64():
    """Full pipeline: the family grafts, fp64-exact end to end, and
    the fused slot computes the shared subterm once."""
    torch.manual_seed(0)
    model = _SharedNormStack().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[C.CrossBlockCSE()], optimize_rest=False
        ),
    )
    m = stats["matches"]["cross_block_cse:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["rel_diff"] < 1e-9
    assert m["cost_after"] < m["cost_before"]
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
    # The shared norm is one DAG node inside the fused executor.
    fused = opt.blocks[0]
    root = getattr(fused, "_root", fused)
    n_norm = sum(
        1 for n in M._iter_ops(root) if n.op == "layer_norm"
    )
    assert n_norm == 1
    # The consumed slot contributes an exact zero.
    with torch.no_grad():
        z = opt.blocks[1](x2)
    assert torch.equal(z, torch.zeros_like(x2))
    # Original model untouched.
    assert model.blocks[1] is not opt.blocks[1]
    assert isinstance(model.blocks[1], _SharedNormBlock)


def test_e2e_recompute_graft():
    """The whole-body recompute also grafts end to end."""
    torch.manual_seed(0)
    model = _RecomputeStack().eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[C.CrossBlockCSE()], optimize_rest=False
        ),
    )
    m = stats["matches"]["cross_block_cse:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_sequential_declines():
    """A sequential residual stack produces no match — the produced
    intermediate is never consumable downstream."""
    torch.manual_seed(0)
    model = _SeqNormStack().eval().double()
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        _x(),
        strategy=MorphismSearch(
            laws=[C.CrossBlockCSE()], optimize_rest=False
        ),
    )
    assert stats["matches"] == {}
    assert stats["n_rewritten"] == 0


def test_optimize_morphisms_entry_cse():
    """The function entry point accepts the law via strategy."""
    torch.manual_seed(0)
    model = _SharedNormStack().eval().double()
    res = optimize_morphisms(
        model,
        _x(),
        backend=TorchBackend(),
        strategy=MorphismSearch(
            laws=[C.CrossBlockCSE()], optimize_rest=False
        ),
    )
    assert (
        res.stats["matches"]["cross_block_cse:blocks.0+blocks.1"][
            "status"
        ]
        == "grafted"
    )


def test_exports():
    """The law re-exports through morphisms and the package surface."""
    import catopt_orchestrator as O

    assert M.CrossBlockCSE is C.CrossBlockCSE
    assert O.CrossBlockCSE is C.CrossBlockCSE


def test_match_via_default_laws_unchanged():
    """The law is opt-in: the default law tuple does not contain it."""
    assert all(
        not isinstance(law, C.CrossBlockCSE)
        for law in M.DEFAULT_MORPHISM_LAWS
    )
    # And the default laws alone do not emit a cse match on a
    # shared-norm stack.
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    ms = [m for law in M.DEFAULT_MORPHISM_LAWS for m in law.match(g)]
    assert all(m.law != "cross_block_cse" for m in ms)


def test_reify_dispatch_through_morphisms():
    """``M._reify`` routes ``mode="cse"`` to the sibling module."""
    torch.manual_seed(0)
    g = _lift(_SharedNormStack().eval().double(), _x())
    m = _family_match(g)
    out = M._reify(
        m,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    assert out["status"] == "grafted"


def test_kv_machinery_reused():
    """The pass reuses the morphisms_kv evidence/family helpers."""
    assert C._input_families is K._input_families
    assert C._family_prep is K._family_prep
    assert C._family_evidence is K._family_evidence
