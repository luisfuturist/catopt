"""Morphism window laws — ``WindowCompose`` + ``ResidualReassoc``.

Extends the pair-level morphism algebra to ≥3-block windows:

* :class:`~catopt_orchestrator.morphisms.WindowCompose` merges a
  maximal run of ``out∘in``-composable chain pairs into ONE reify —
  one joint term, one e-graph, one verify — where
  :class:`~catopt_orchestrator.morphisms.OutInCompose` needed k−1
  pairwise passes.
* :class:`~catopt_orchestrator.morphisms.ResidualReassoc` distributes
  a residual stream's additive structure so a later block's input
  projections compose with *non-adjacent* contributions — the ``+``
  monoid is the legal commute path; bilinearity absorbs them.

Every grafted rewrite is verified fp64 end to end and cost-gated;
opaque blocks are boundaries no window crosses.
"""

import catopt_orchestrator.morphisms as M
import torch
import torch.nn as nn
from catopt_core.ir import op_repr, op_repr_dag
from catopt_orchestrator import MorphismLaw, MorphismSearch, Optimizer
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer
from catopt_torch.models import (
    DeepParallel,
    ParallelLinear,
    ResidualMLP,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


class _ChainStack(nn.Module):
    """``x = b_i(x)`` — plain chain of DeepParallel blocks."""

    def __init__(self, dim: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _ChainWrapTail(nn.Module):
    """``x = A(x); x = B(x); x + C(x)`` — chain run + wrapped tail."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        return x + self.blocks[2](x)


class _ResidualWrapped(nn.Module):
    """``x = x + b_i(x)`` — the residual-stream stack."""

    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class _ResidualTail(nn.Module):
    """``x + A(x); x + B(x); C(·)`` — stream + plain receiver tail."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.blocks[0](x)
        x = x + self.blocks[1](x)
        return self.blocks[2](x)


class _PreNormResidual(nn.Module):
    """ResidualMLP stack — receivers carry a pre-norm (``norm.pre``).

    The LayerNorm sits between the stream and the block's
    in-projections, so bilinearity cannot cross it: no commute
    opportunity, hence no window match.
    """

    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualMLP(dim, hidden_mult=2) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class _Scale(nn.Module):
    """Pure diagonal block: ``x ∘ s`` — no projections at all."""

    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.s


class _DiagFirstResidual(nn.Module):
    """Residual stream whose first block carries no projections."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _Scale(),
                DeepParallel(dim, dim, dim),
                DeepParallel(dim, dim, dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class _DataDependent(nn.Module):
    """Data-dependent branch — export always fails (opaque block)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.lin(x)
        return -self.lin(x)


class _MixedResidual(nn.Module):
    """Residual stream with an un-exportable middle block."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DeepParallel(dim, dim, dim),
                _DataDependent(dim),
                DeepParallel(dim, dim, dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class _FanoutStream(nn.Module):
    """``x = x + b_i(x)`` over high-fanout blocks.

    Each ``ParallelLinear`` block reads the stream ``n_experts``
    times, so a depth-5 window joint references every stream node
    ~9x per level: the ``op_repr`` *tree* expansion is ~10^3x the
    node count — the same shape as the stories15M window whose
    stats rendering OOMed ``_reify``.
    """

    def __init__(
        self, dim: int = 16, depth: int = 5, n: int = 8
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelLinear(dim, n_experts=n) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    """Lift *model* through the torch ports."""
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*dims, generator=g, dtype=torch.float64)


def _opt(
    model: nn.Module, x: torch.Tensor, laws: list
) -> tuple[nn.Module, dict]:
    """Run the morphism pipeline with an explicit law list."""
    return Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(laws=laws, optimize_rest=False),
    )


# ---------------------------------------------------------------------------
#  Match surfaces
# ---------------------------------------------------------------------------


def test_window_compose_match_surfaces():
    """WindowCompose: maximal composable chain runs, ≥3 blocks only."""
    torch.manual_seed(0)
    g4 = _lift(_ChainStack(depth=4).eval().double(), _x())
    matches = M.WindowCompose().match(g4)
    assert len(matches) == 1
    m = matches[0]
    assert m.law == "window_compose"
    assert m.nodes == (
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    )
    assert m.boundary == "chain+chain+chain"
    assert m.reify.kinds == ("chain", "chain", "chain")
    assert m.reify.mode == "chain"
    assert m.reify.rules == "compose"
    assert not m.reify.distribute
    assert isinstance(M.WindowCompose(), MorphismLaw)

    # A chain of two is a pair — OutInCompose's job, not a window.
    g2 = _lift(_ChainStack(depth=2).eval().double(), _x())
    assert M.WindowCompose().match(g2) == []

    # A residual stack carries no plain chain wires at all.
    gr = _lift(_ResidualWrapped().eval().double(), _x())
    assert M.WindowCompose().match(gr) == []


def test_window_compose_wrapped_tail_match():
    """``chain`` interior + ``chain_wrapped`` tail closes a window."""
    torch.manual_seed(0)
    g = _lift(_ChainWrapTail().eval().double(), _x())
    assert [w.kind for w in g.wires] == ["chain", "chain_wrapped"]
    (m,) = M.WindowCompose().match(g)
    assert m.nodes == ("blocks.0", "blocks.1", "blocks.2")
    assert m.reify.kinds == ("chain", "chain_wrapped")
    assert m.reify.mode == "chain_wrapped"
    assert m.boundary == "chain+chain_wrapped"


def test_residual_reassoc_match_surfaces():
    """ResidualReassoc: residual_wrapped runs + optional plain tail."""
    torch.manual_seed(0)
    g = _lift(_ResidualWrapped(depth=3).eval().double(), _x())
    (m,) = M.ResidualReassoc().match(g)
    assert m.law == "residual_reassoc"
    assert m.nodes == ("blocks.0", "blocks.1", "blocks.2")
    assert m.boundary == "residual_wrapped+residual_wrapped"
    assert m.reify.kinds == ("residual_wrapped", "residual_wrapped")
    assert m.reify.mode == "residual_wrapped"
    assert m.reify.distribute
    assert isinstance(M.ResidualReassoc(), MorphismLaw)

    # The plain-receiver tail variant.
    gt = _lift(_ResidualTail().eval().double(), _x())
    (mt,) = M.ResidualReassoc().match(gt)
    assert mt.reify.kinds == ("residual_wrapped", "residual")
    assert mt.reify.mode == "residual"

    # Two blocks make a pair — ResidualAbsorb's job, not a window.
    g2 = _lift(_ResidualWrapped(depth=2).eval().double(), _x())
    assert M.ResidualReassoc().match(g2) == []

    # Chain stacks carry no residual wires.
    gc = _lift(_ChainStack(depth=4).eval().double(), _x())
    assert M.ResidualReassoc().match(gc) == []


def test_residual_reassoc_no_match_cases():
    """No commute through pre-norms, and never across opaque nodes."""
    torch.manual_seed(0)
    # Every ResidualMLP receiver is pre-normed — bilinearity cannot
    # cross it, so the window has no commute opportunity.
    gp = _lift(_PreNormResidual().eval().double(), _x())
    assert M.ResidualReassoc().match(gp) == []

    # An opaque middle block breaks the stream coverage entirely.
    gm = _lift(_MixedResidual().eval().double(), _x())
    assert gm.node("blocks.1").opaque
    assert M.ResidualReassoc().match(gm) == []

    # Unit-level predicate checks.
    assert not M._stream_commutes(gm, ("blocks.0", "blocks.1"))
    sigless = M.MorphismNode("ghost", None, True)
    g = M.MorphismGraph(
        [sigless], [], {"ghost": M._BlockRecord("ghost", None)}
    )
    assert not M._stream_commutes(g, ("ghost", "ghost"))

    # A contributor with no out-projection is skipped (the diag
    # block feeds the stream but offers nothing to compose), yet the
    # later DeepParallel pair still commutes.
    gd = _lift(_DiagFirstResidual().eval().double(), _x())
    assert M._stream_commutes(gd, ("blocks.0", "blocks.1", "blocks.2"))
    assert M.ResidualReassoc().match(gd)


def test_window_signature_predicates():
    """Unit coverage for the wire/window signature helpers."""
    torch.manual_seed(0)
    g = _lift(_ChainStack(depth=2).eval().double(), _x())
    w = g.wires[0]
    assert M._wire_composes(g, w, frozenset({"chain"}))
    assert not M._wire_composes(g, w, frozenset({"chain_wrapped"}))
    a, b = g.sig("blocks.0"), g.sig("blocks.1")
    assert M._compose_pair_ok(a, b)
    assert not M._compose_pair_ok(None, b)
    assert not M._compose_pair_ok(a, None)
    assert M._window_nodes(g.wires, 0, 0) == ("blocks.0", "blocks.1")


# ---------------------------------------------------------------------------
#  Verified end-to-end runs
# ---------------------------------------------------------------------------


def test_e2e_window_compose_grafts_fp64():
    """4-block chain: one window match grafts the whole stack — one
    fused linear, fp64-exact end to end."""
    torch.manual_seed(0)
    model = _ChainStack(depth=4).eval().double()
    x = _x()
    opt, stats = _opt(model, x, [M.WindowCompose()])

    m = stats["matches"][
        "window_compose:blocks.0+blocks.1+blocks.2+blocks.3"
    ]
    assert m["status"] == "grafted"
    assert m["boundary"] == "chain+chain+chain"
    assert m["rel_diff"] < 1e-9
    assert m["cost_after"] < m["cost_before"]
    assert "linear x" in m["reified"]
    assert stats["n_rewritten"] == 4
    assert stats["morphism_fires"] == {"window_compose": 1}
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_window_compose_wrapped_tail_grafts_fp64():
    """Chain window closed by a wrapped tail: the last slot takes the
    exact-zero filler, the fused segment rides the first block."""
    torch.manual_seed(0)
    model = _ChainWrapTail().eval().double()
    x = _x()
    opt, stats = _opt(model, x, [M.WindowCompose()])

    m = stats["matches"]["window_compose:blocks.0+blocks.1+blocks.2"]
    assert m["status"] == "grafted"
    assert m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_residual_reassoc_grafts_fp64():
    """Residual stream ×3: one window match distributes every
    stream-reading projection and folds contributions — including the
    NON-adjacent ``blocks.0.out ∘ blocks.2.in`` composition."""
    torch.manual_seed(0)
    model = _ResidualWrapped(depth=3).eval().double()
    x = _x()
    opt, stats = _opt(model, x, [M.ResidualReassoc()])

    m = stats["matches"]["residual_reassoc:blocks.0+blocks.1+blocks.2"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "residual_wrapped+residual_wrapped"
    assert m["rel_diff"] < 1e-9
    assert m["cost_after"] < m["cost_before"]
    assert stats["n_rewritten"] == 3
    assert stats["morphism_fires"] == {"residual_reassoc": 1}
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_residual_reassoc_plain_tail_grafts_fp64():
    """``x + A(x); x + B(x); C(·)``: the unwrapped receiver absorbs
    the whole stream — last slot is an identity passthrough."""
    torch.manual_seed(0)
    model = _ResidualTail().eval().double()
    x = _x()
    opt, stats = _opt(model, x, [M.ResidualReassoc()])

    m = stats["matches"]["residual_reassoc:blocks.0+blocks.1+blocks.2"]
    assert m["status"] == "grafted"
    assert m["boundary"] == "residual_wrapped+residual"
    assert m["rel_diff"] < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_e2e_windows_claim_before_pair_laws():
    """Under the default family, windows claim their nodes first —
    pair matches on interior boundaries skip honestly."""
    torch.manual_seed(0)
    model = _ChainStack(depth=4).eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=MorphismSearch(optimize_rest=False)
    )
    w = stats["matches"][
        "window_compose:blocks.0+blocks.1+blocks.2+blocks.3"
    ]
    assert w["status"] == "grafted"
    for k, v in stats["matches"].items():
        if k.startswith("out_in_compose"):
            assert v["status"] == "skipped"
            assert v["reason"] == "node already rewritten"
    assert stats["morphism_fires"]["window_compose"] == 1

    model2 = _ResidualWrapped(depth=3).eval().double()
    opt2, stats2 = Optimizer(backend=TorchBackend()).optimize(
        model2, _x(), strategy=MorphismSearch(optimize_rest=False)
    )
    assert (
        stats2["matches"][
            "residual_reassoc:blocks.0+blocks.1+blocks.2"
        ]["status"]
        == "grafted"
    )
    for k, v in stats2["matches"].items():
        if k.startswith("residual_absorb"):
            assert v["status"] == "skipped"
    with torch.no_grad():
        d = (model2(_x().clone()) - opt2(_x().clone())).abs().max()
    assert d.item() < 1e-9
    with torch.no_grad():
        d0 = (model(x.clone()) - opt(x.clone())).abs().max()
    assert d0.item() < 1e-9


# ---------------------------------------------------------------------------
#  Reify internals — window path
# ---------------------------------------------------------------------------


def test_joint_parts_window_structure():
    """The n-ary joint builder: chain nests, residual accumulates a
    stream with one mid per later block, wrapped tails add in+body."""
    torch.manual_seed(0)
    # Chain family — no mids.
    g = _lift(_ChainStack(depth=3).eval().double(), _x())
    recs = [g.record(n.name) for n in g.nodes]
    joint, var, mids, params, leaves = M._joint_parts_window(
        [r.ir for r in recs], recs, ("chain", "chain")
    )
    assert mids == ()
    assert joint.op == "linear"  # DeepParallel root, nested
    assert var is recs[0].ir.inputs[0]
    # Every block's params are namespaced into the joint tables.
    assert "blocks_2__p_w3_weight" in params
    assert "blocks_2__p_w3_weight" in leaves

    # Chain + wrapped tail — joint is in+body.
    gw = _lift(_ChainWrapTail().eval().double(), _x())
    recs_w = [gw.record(n.name) for n in gw.nodes]
    jw, _, mids_w, _, _ = M._joint_parts_window(
        [r.ir for r in recs_w], recs_w, ("chain", "chain_wrapped")
    )
    assert mids_w == ()
    assert jw.op == "add"

    # Residual family — one stream mid per later block.
    gr = _lift(_ResidualWrapped(depth=3).eval().double(), _x())
    recs_r = [gr.record(n.name) for n in gr.nodes]
    jr, _, mids_r, _, _ = M._joint_parts_window(
        [r.ir for r in recs_r],
        recs_r,
        ("residual_wrapped", "residual_wrapped"),
    )
    assert len(mids_r) == 2
    assert all(m.op == "add" for m in mids_r)
    assert jr.op == "add"  # the outermost stream node s_2

    # Plain residual tail — the joint is the receiver's own body.
    gt = _lift(_ResidualTail().eval().double(), _x())
    recs_t = [gt.record(n.name) for n in gt.nodes]
    jt, _, mids_t, _, _ = M._joint_parts_window(
        [r.ir for r in recs_t], recs_t, ("residual_wrapped", "residual")
    )
    assert len(mids_t) == 2
    assert jt.op == "linear"


def test_reify_window_direct_and_slot_fillers():
    """A hand-built window match reifies with per-slot fillers:
    delta at the first slot (residual family), zeros mid-stream, an
    identity at the unwrapped receiver."""
    torch.manual_seed(0)
    g = _lift(_ResidualTail().eval().double(), _x())
    match = M.MorphismMatch(
        law="residual_reassoc",
        nodes=("blocks.0", "blocks.1", "blocks.2"),
        boundary="residual_wrapped+residual",
        reify=M.ReifySpec(
            mode="residual",
            rules="compose",
            distribute=True,
            kinds=("residual_wrapped", "residual"),
        ),
    )
    out = M._reify(
        match,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=8,
        max_enodes=50_000,
        symmetry_budget=512,
    )
    assert out["status"] == "grafted"
    assert set(out["reps"]) == {"blocks.0", "blocks.1", "blocks.2"}
    x = _x()
    with torch.no_grad():
        # Middle slot is the exact-zero stream filler.
        z = out["reps"]["blocks.1"](x)
        assert z.abs().max().item() == 0.0
        # The unwrapped receiver slot is the identity passthrough.
        i = out["reps"]["blocks.2"](x)
        assert torch.equal(i, x)


def test_reify_window_stats_render_dag_bounded():
    """The ``joint``/``reified`` stats strings are the DAG-aware
    rendering — on a sharing-heavy window joint the ``op_repr`` tree
    expansion explodes (the bench-measured ``_reify`` MemoryError)."""
    torch.manual_seed(0)
    g = _lift(_FanoutStream().eval().double(), _x())
    kinds = ("residual_wrapped",) * 4
    match = M.MorphismMatch(
        law="probe",
        nodes=tuple(f"blocks.{i}" for i in range(5)),
        boundary="+".join(kinds),
        reify=M.ReifySpec(
            mode="residual_wrapped",
            rules="compose",
            distribute=True,
            kinds=kinds,
        ),
    )
    resolved, why = M._window_joint(match, g)
    assert why is None
    joint = resolved[0]
    # The fixture really is sharing-heavy: tree repr ~10^3x the DAG.
    tree = op_repr(joint)
    assert len(op_repr_dag(joint)) * 100 < len(tree)

    out = M._reify(
        match,
        g,
        sink=TorchSink(),
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=8,
        max_enodes=50_000,
        symmetry_budget=512,
    )
    assert out["status"] in ("grafted", "declined")
    # The stats path emits the bounded let-form, linear in nodes.
    assert out["joint"] == op_repr_dag(joint)
    assert out["joint"].startswith("(let ")
    assert len(out["reified"]) < len(tree)


def test_reify_window_malformed_and_opaque():
    """Malformed specs and opaque members decline — never graft."""
    torch.manual_seed(0)
    g = _lift(_ChainStack(depth=3).eval().double(), _x())
    sink = TorchSink()
    kw = dict(
        sink=sink,
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    bad_kinds = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1", "blocks.2"),
        boundary="chain+chain",
        reify=M.ReifySpec(
            mode="chain", rules="compose", kinds=("chain",)
        ),
    )
    out = M._reify(bad_kinds, g, **kw)
    assert out["status"] == "declined"
    assert out["reason"] == "malformed window spec"

    # Opaque member: a window crafted across the un-exportable block.
    gm = _lift(_MixedResidual().eval().double(), _x())
    opaque = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1", "blocks.2"),
        boundary="residual_wrapped+residual_wrapped",
        reify=M.ReifySpec(
            mode="residual_wrapped",
            rules="compose",
            distribute=True,
            kinds=("residual_wrapped", "residual_wrapped"),
        ),
    )
    out = M._reify(opaque, gm, **kw)
    assert out["status"] == "declined"
    assert out["reason"] == "opaque node"


def test_reify_window_no_improvement_and_no_distribution():
    """A window spec with no rules declines on cost; a distribute
    offer on a window no projection can unfold registers no offers."""
    torch.manual_seed(0)
    g = _lift(_ChainStack(depth=3).eval().double(), _x())
    sink = TorchSink()
    kw = dict(
        sink=sink,
        cost_fn=M.flops_cost,
        verify_tol=1e-4,
        max_iterations=4,
        max_enodes=10_000,
        symmetry_budget=128,
    )
    no_rules = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1", "blocks.2"),
        boundary="chain+chain",
        reify=M.ReifySpec(
            mode="chain", rules="tie", kinds=("chain", "chain")
        ),
    )
    out = M._reify(no_rules, g, **kw)
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"

    # distribute=True but no projection reads the stream — the
    # offers stay empty and nothing improves.
    class _Silu(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(x)

    class _SiluStream(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Silu(), _Silu(), _Silu()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.blocks[0](x)
            x = x + self.blocks[1](x)
            return self.blocks[2](x)

    gs = _lift(_SiluStream().eval().double(), _x())
    dist = M.MorphismMatch(
        law="probe",
        nodes=("blocks.0", "blocks.1", "blocks.2"),
        boundary="residual_wrapped+residual",
        reify=M.ReifySpec(
            mode="residual",
            rules="compose",
            distribute=True,
            kinds=("residual_wrapped", "residual"),
        ),
    )
    out = M._reify(dist, gs, **kw)
    # silu(stream) distributes nothing and cannot improve.
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"


def test_window_verify_gate_declines(monkeypatch):
    """A failing pair verify declines a grafted window honestly."""
    import catopt_torch.adapters as A

    torch.manual_seed(0)
    model = _ChainStack(depth=4).eval().double()

    def fail_verify(*_a, **_k):
        return A.VerifyReport(max_abs=1.0, max_rel=1.0, passed=False)

    monkeypatch.setattr(A, "verify_module", fail_verify)
    _, stats = _opt(model, _x(), [M.WindowCompose()])
    m = stats["matches"][
        "window_compose:blocks.0+blocks.1+blocks.2+blocks.3"
    ]
    assert m["status"] == "declined"
    assert "verify failed" in m["reason"]
    assert stats["morphism_fires"] == {}


def test_lazy_sibling_reexport_and_family_dispatch():
    """The lazy ``morphisms_kv`` re-export resolves and misses raise
    AttributeError; a ``family``-mode spec dispatches to the sibling
    reify (and declines honestly on a non-family match)."""
    import pytest

    import catopt_orchestrator.morphisms_kv as kv

    assert M.KVLatentShare is kv.KVLatentShare
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(M, "NoSuchLaw")

    torch.manual_seed(0)
    g = _lift(_ChainStack(depth=2).eval().double(), _x())
    match = M.MorphismMatch(
        law="kv_latent_share",
        nodes=("blocks.0", "blocks.1"),
        boundary="shared_input",
        reify=M.ReifySpec(mode="family", rules="compose"),
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


def test_window_error_declines():
    """A window match over a nonexistent node records an error."""

    class BadWindow:
        name = "bad_window"

        def match(self, graph):
            return [
                M.MorphismMatch(
                    law=self.name,
                    nodes=("nope.0", "nope.1", "nope.2"),
                    boundary="chain+chain",
                    reify=M.ReifySpec(
                        mode="chain",
                        rules="compose",
                        kinds=("chain", "chain"),
                    ),
                )
            ]

    torch.manual_seed(0)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        _ChainStack(depth=3).eval().double(),
        _x(),
        strategy=MorphismSearch(
            laws=[BadWindow()], optimize_rest=False
        ),
    )
    m = stats["matches"]["bad_window:nope.0+nope.1+nope.2"]
    assert m["status"] == "declined"
    assert m["reason"] == "error"


def test_e2e_residual_reassoc_no_fallthrough_to_absorb():
    """When the window declines, the pair absorb law still applies —
    graceful decomposition, not all-or-nothing."""
    torch.manual_seed(0)
    model = _ResidualWrapped(depth=3).eval().double()
    x = _x()

    # A crippled window spec — no rules, so it always declines — yet
    # ResidualAbsorb still gets its shot at the pairs.
    class _NoOpReassoc(M.ResidualReassoc):
        name = "residual_reassoc"

        def match(self, graph):
            out = super().match(graph)
            for m in out:
                object.__setattr__(
                    m,
                    "reify",
                    M.ReifySpec(
                        mode=m.reify.mode,
                        rules="tie",
                        distribute=True,
                        kinds=m.reify.kinds,
                    ),
                )
            return out

    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[_NoOpReassoc(), M.ResidualAbsorb()],
            optimize_rest=False,
        ),
    )
    w = stats["matches"]["residual_reassoc:blocks.0+blocks.1+blocks.2"]
    assert w["status"] == "declined"
    assert w["reason"] == "no_improvement"
    pair = stats["matches"]["residual_absorb:blocks.0+blocks.1"]
    assert pair["status"] == "grafted"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
