"""Pairwise cross-block pass: joint optimization across boundaries.

``optimize_compositional`` optimizes each selected block in isolation;
the cross-block pass then tries each *adjacent* pair whose boundary is
a simple value flow — ``B(A(x))`` chain, or the ``x + A(x)`` residual —
as one joint micro-model through the ordinary ``optimize_model``.  The
joint result is grafted only when it verifies against the eager pair
AND its delivered FLOPs beat the sum of the separate results; every
other outcome is a silent decline recorded in ``stats["cross_pairs"]``.
"""

import catopt_optimize.optimize as O
import torch
import torch.nn as nn
from catopt.models import DeepParallel, ParallelLinear
from catopt.optimize import optimize_compositional


class _Scale(nn.Module):
    """Elementwise scalar multiply — the shared affine a residual add
    (or a downstream ``y + s·y``) can absorb into a neighbour's weight."""

    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.s


class _ChainStack(nn.Module):
    """``x = b_i(x)`` — a plain chain: each block's input IS the
    previous block's output object."""

    def __init__(self, dim: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _ResidualStack(nn.Module):
    """``x = x + b_i(x)`` — the residual wraps both sides of every
    boundary: B's input is ``a_in + a_out`` and B's own output is
    residual-added again."""

    def __init__(self, dim: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


def test_cross_pair_chain_grafts_jointly():
    """Adjacent DeepParallel blocks fuse pairwise: A's output projection
    composes with B's input projections — ``(x@W1+x@W2)@W3`` then the
    same again — into ONE folded weight; verified fp64-exact."""
    torch.manual_seed(0)
    model = _ChainStack(dim=32, depth=4).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    pairs = stats["cross_pairs"]
    e01 = pairs["blocks.0+blocks.1"]
    assert e01["status"] == "grafted"
    assert e01["boundary"] == "chain"
    assert e01["rel_diff"] < 1e-9
    assert e01["joint_cost"] < e01["separate_cost"]
    # Non-overlap: block 1 was consumed by the first graft.
    assert pairs["blocks.1+blocks.2"]["status"] == "skipped"
    assert (
        pairs["blocks.1+blocks.2"]["reason"] == "member already fused"
    )
    e23 = pairs["blocks.2+blocks.3"]
    assert e23["status"] == "grafted"
    assert e23["boundary"] == "chain"

    # The joint module replaces BOTH slots: fused wrapper at A's slot,
    # Identity at B's.
    assert isinstance(opt.blocks[0], O._FusedPair)
    assert isinstance(opt.blocks[1], nn.Identity)
    assert isinstance(opt.blocks[2], O._FusedPair)
    assert isinstance(opt.blocks[3], nn.Identity)
    assert stats["blocks"]["blocks.0"]["cross_pair"] == (
        "blocks.0+blocks.1"
    )

    # Aggregate param report reflects the delivered modules: two fused
    # weights total instead of four.
    pr = stats["param_report"]
    assert pr["optimized_params"] == 2
    assert any(
        n.startswith("blocks.0+blocks.1:") for n in pr["eliminated"]
    )
    assert not any(n.startswith("blocks.0:") for n in pr["eliminated"])

    # Original model unmutated; recomposed model fp64-exact.
    assert stats["shared_params"] is True
    assert all(isinstance(b, DeepParallel) for b in model.blocks)
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9


def test_cross_pair_residual_wrapped_graft():
    """``x = x + b(x)`` stacks: B's input is ``a_in + a_out`` AND B's
    output is residual-wrapped again — the joint absorbs both adds and
    the fused pair takes over the whole two-block segment."""
    torch.manual_seed(0)
    model = _ResidualStack(dim=32, depth=3).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "grafted"
    assert e["boundary"] == "residual_wrapped"
    assert e["joint_cost"] < e["separate_cost"]
    assert stats["cross_pairs"]["blocks.1+blocks.2"]["reason"] == (
        "member already fused"
    )

    # A's slot returns the delta (the parent's ``x + ·`` reconstructs
    # the segment); B's slot contributes an exact zero addend.
    fused = opt.blocks[0]
    assert isinstance(fused, O._FusedPair)
    assert fused.delta is True
    assert isinstance(opt.blocks[1], O._Zero)
    assert all(isinstance(b, DeepParallel) for b in model.blocks)

    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9


def test_cross_pair_residual_plain_graft():
    """``x = x + A(x); x = B(x)`` — the residual-only boundary: the add
    folds into B's input projection (scale absorbed as a diagonal)."""
    torch.manual_seed(0)

    class ResidualPlain(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    _Scale(),
                    DeepParallel(dim, dim, dim),
                    DeepParallel(dim, dim, dim),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.blocks[0](x)
            x = self.blocks[1](x)
            x = self.blocks[2](x)
            return x

    model = ResidualPlain().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "grafted"
    assert e["boundary"] == "residual"
    assert isinstance(opt.blocks[0], O._FusedPair)
    assert opt.blocks[0].delta is True
    assert isinstance(opt.blocks[1], nn.Identity)

    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_chain_wrapped_graft():
    """``x = A(x); x = x + B(x)`` — chain into a residual-WRAPPED B:
    the joint covers ``A(x) + B(A(x))`` and B's slot zeroes out."""
    torch.manual_seed(0)

    class ChainWrapped(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    DeepParallel(dim, dim, dim),
                    _Scale(),
                    DeepParallel(dim, dim, dim),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks[0](x)
            x = x + self.blocks[1](x)
            x = self.blocks[2](x)
            return x

    model = ChainWrapped().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "grafted"
    assert e["boundary"] == "chain_wrapped"
    fused = opt.blocks[0]
    assert isinstance(fused, O._FusedPair)
    assert fused.delta is False
    assert isinstance(opt.blocks[1], O._Zero)

    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_declined_no_cost_improvement():
    """A pair whose joint is not cheaper keeps the separate results —
    recorded as a ``declined`` outcome, not a failure."""
    torch.manual_seed(0)

    class SigStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sigmoid(), nn.Sigmoid()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for b in self.blocks:
                x = b(x)
            return x

    model = SigStack().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=True)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "declined"
    assert e["reason"] == "no cost improvement"
    assert e["joint_cost"] == e["separate_cost"]
    # The individually-optimized modules were grafted instead.
    assert not isinstance(opt.blocks[0], O._FusedPair)
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_non_adjacent_declines():
    """Blocks selected adjacently but executed out of order have no
    simple boundary — the pair is skipped without an optimize call."""
    torch.manual_seed(0)

    class Interleaved(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                ParallelLinear(dim, n_experts=2) for _ in range(3)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks[0](x)
            x = self.blocks[2](x)
            x = self.blocks[1](x)
            return x

    model = Interleaved().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    pairs = stats["cross_pairs"]
    # blocks.0's output is consumed by blocks.2, not blocks.1.
    assert pairs["blocks.0+blocks.1"]["status"] == "skipped"
    assert pairs["blocks.0+blocks.1"]["reason"] == "no simple boundary"
    # blocks.1 runs last: its output escapes as the model output.
    assert pairs["blocks.1+blocks.2"]["status"] == "skipped"
    assert pairs["blocks.1+blocks.2"]["reason"] == "no simple boundary"
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_member_not_optimized():
    """A failed block breaks the chain — both adjacent pairs skip."""
    torch.manual_seed(0)

    class _DataDependent(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.sum() > 0:
                return self.lin(x)
            return -self.lin(x)

    class Mixed(nn.Module):
        def __init__(self, dim: int = 32) -> None:
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

    model = Mixed().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    assert stats["blocks"]["blocks.1"]["status"] == "failed"
    for k in ("blocks.0+blocks.1", "blocks.1+blocks.2"):
        assert stats["cross_pairs"][k]["status"] == "skipped"
        assert stats["cross_pairs"][k]["reason"] == (
            "block not optimized"
        )
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_verify_decline(monkeypatch):
    """A joint that fails the eager-equivalence verify is declined —
    the individually-optimized modules stay grafted."""
    torch.manual_seed(0)
    orig = O.optimize_model

    def fake(m, ex, **kw):
        if isinstance(m, O._JointPair):
            return nn.Identity(), {}
        return orig(m, ex, **kw)

    monkeypatch.setattr(O, "optimize_model", fake)

    model = _ChainStack(dim=32, depth=2).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "declined"
    assert "verify failed" in e["reason"]
    assert not isinstance(opt.blocks[0], O._FusedPair)
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_error_decline(monkeypatch):
    """Any exception inside a pair attempt is a silent decline."""
    torch.manual_seed(0)
    orig = O.optimize_model

    def boom(m, ex, **kw):
        if isinstance(m, O._JointPair):
            raise RuntimeError("export boom")
        return orig(m, ex, **kw)

    monkeypatch.setattr(O, "optimize_model", boom)

    model = _ChainStack(dim=32, depth=2).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "declined"
    assert e["reason"] == "error"
    assert "export boom" in e["error"]
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_budget_cap():
    """``max_cross_pairs`` bounds the number of joint optimize runs."""
    torch.manual_seed(0)
    model = _ChainStack(dim=32, depth=4).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(
        model, x, verbose=False, max_cross_pairs=1
    )

    pairs = stats["cross_pairs"]
    assert pairs["blocks.0+blocks.1"]["status"] == "grafted"
    assert (
        pairs["blocks.1+blocks.2"]["reason"] == "member already fused"
    )
    assert pairs["blocks.2+blocks.3"]["status"] == "skipped"
    assert pairs["blocks.2+blocks.3"]["reason"] == "max_cross_pairs=1"
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_cross_pair_disabled():
    """``max_cross_pairs=0`` disables the pass; blocks still optimize."""
    torch.manual_seed(0)
    model = _ChainStack(dim=32, depth=2).eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(
        model, x, verbose=False, max_cross_pairs=0
    )

    assert stats["cross_pairs"] == {}
    assert stats["n_optimized"] == 2
    assert not isinstance(opt.blocks[0], O._FusedPair)
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


# ---------------------------------------------------------------------------
#  Unit-level: boundary classifier + graft wrappers
# ---------------------------------------------------------------------------


def _synth_io(
    a_in,
    a_out,
    b_in,
    b_out,
    model_out=None,
    extra_consumers=(),
    calls_a=1,
    calls_b=1,
    kw_a=None,
    kw_b=None,
    b_in_obj=None,
):
    """Fabricate ``(captured, io)`` for blocks ``"a"``, ``"b"``.

    Mirrors ``_capture_block_inputs``: ``captured`` holds detached
    clones, ``io`` holds the retained live objects plus call counts.
    ``b_in_obj`` overrides b's recorded arg object — passing ``a_out``
    models B consuming A's live output object.
    """
    captured = {"a": ((a_in,), kw_a or {}), "b": ((b_in,), kw_b or {})}
    io = {
        "a": {
            "calls": calls_a,
            "in_objs": (a_in,),
            "out_obj": a_out,
            "out": a_out,
        },
        "b": {
            "calls": calls_b,
            "in_objs": (b_in if b_in_obj is None else b_in_obj,),
            "out_obj": b_out,
            "out": b_out,
        },
        O._MODEL_KEY: {"out_obj": model_out, "out": model_out},
    }
    for n, t in extra_consumers:
        captured[n] = ((t.clone(),), {})
        io[n] = {
            "calls": 1,
            "in_objs": (t,),
            "out_obj": None,
            "out": None,
        }
    return captured, io


def _probe(captured, io):
    """A perturbed-input twin of a synthetic capture.

    Scales every tensor by 1.5 — linear relations (``b_in == a_in +
    a_out``, value-consumption) are preserved, matching what
    ``_perturbed_input`` produces on a real model.
    """
    cap2 = {
        n: (
            tuple(
                a * 1.5 if isinstance(a, torch.Tensor) else a
                for a in args
            ),
            kw,
        )
        for n, (args, kw) in captured.items()
    }
    io2 = {
        n: {
            **m,
            "out": (
                m["out"] * 1.5
                if isinstance(m.get("out"), torch.Tensor)
                else m.get("out")
            ),
        }
        for n, m in io.items()
    }
    return cap2, io2


def _boundary(name_a, name_b, captured, io, probe=True):
    """``_pair_boundary`` with a second-probe capture.

    ``probe=True`` reuses the same dicts — the fabricated relations
    hold by construction, so a real residual confirms.  ``probe=False``
    passes empty dicts (missing probe).  A ``(cap2, io2)`` tuple
    supplies an explicit second capture.
    """
    if probe is True:
        c2, i2 = captured, io
    elif probe is False:
        c2, i2 = {}, {}
    else:
        c2, i2 = probe
    return O._pair_boundary(name_a, name_b, captured, io, c2, i2)


def test_pair_boundary_missing_or_reentered():
    """No capture / no io / re-entered blocks → no boundary."""
    x = torch.randn(4, 4)
    assert O._pair_boundary("a", "b", {}, {}, {}, {}) is None
    cap, io = _synth_io(x, x.clone(), x.clone(), x.clone())
    assert _boundary("a", "zz", cap, io) is None
    cap, io = _synth_io(x, x.clone(), x.clone(), x.clone(), calls_a=2)
    assert _boundary("a", "b", cap, io) is None
    cap, io = _synth_io(x, x.clone(), x.clone(), x.clone(), calls_b=0)
    assert _boundary("a", "b", cap, io) is None


def test_pair_boundary_arg_shapes():
    """Kwargs, multi-arg calls and non-tensor inputs are declined."""
    x = torch.randn(4, 4)
    cap, io = _synth_io(
        x, x.clone(), x.clone(), x.clone(), kw_b={"k": 1}
    )
    assert _boundary("a", "b", cap, io) is None
    cap, io = _synth_io(
        x, x.clone(), x.clone(), x.clone(), kw_a={"k": 1}
    )
    assert _boundary("a", "b", cap, io) is None
    # Two positional args — not a single-value boundary.
    cap, io = _synth_io(x, x.clone(), x.clone(), x.clone())
    cap["b"] = ((x.clone(), x.clone()), {})
    io["b"]["in_objs"] = (x, x)
    assert _boundary("a", "b", cap, io) is None
    # Non-tensor pieces.
    cap, io = _synth_io("not-a-tensor", x.clone(), x.clone(), x.clone())
    assert _boundary("a", "b", cap, io) is None
    # A non-tensor block OUTPUT declines only after the A-side boundary
    # was accepted — chain into it, then the B-side check bails.
    a_out = torch.randn(4, 4)
    cap, io = _synth_io(
        x, a_out, a_out.clone(), "not-a-tensor", b_in_obj=a_out
    )
    assert _boundary("a", "b", cap, io) is None


def test_pair_boundary_chain_and_fanout():
    """Chain needs B to consume A's output object, unmodified, alone."""
    x = torch.randn(4, 4)
    a_out = torch.randn(4, 4)
    b_out = torch.randn(4, 4)
    # B consumed a_out (same live object); a third block reads b_out.
    other_in = b_out  # plainly consumed
    cap, io = _synth_io(
        x,
        a_out,
        a_out.clone(),
        b_out,
        extra_consumers=(("c", other_in),),
        b_in_obj=a_out,
    )
    assert _boundary("a", "b", cap, io) == "chain"
    # Same live object but its captured value differs — an in-place op
    # ran between the calls → not a simple edge.
    cap, io = _synth_io(
        x,
        a_out,
        a_out + 1.0,
        b_out,
        extra_consumers=(("c", other_in),),
        b_in_obj=a_out,
    )
    assert _boundary("a", "b", cap, io) is None
    # Fan-out: another block also consumed a_out → not a simple edge.
    cap, io = _synth_io(
        x,
        a_out,
        a_out.clone(),
        b_out,
        extra_consumers=(("c", a_out),),
        b_in_obj=a_out,
    )
    assert _boundary("a", "b", cap, io) is None
    # A's output escapes as the model output.
    cap, io = _synth_io(x, a_out, a_out.clone(), b_out, model_out=a_out)
    assert _boundary("a", "b", cap, io) is None
    # b's input is a different object AND a_out fans out → decline via
    # the ``elif a_fans`` arm.
    cap, io = _synth_io(
        x,
        a_out,
        torch.randn(4, 4),
        b_out,
        extra_consumers=(("c", a_out),),
    )
    assert _boundary("a", "b", cap, io) is None


def test_pair_boundary_residual_modes():
    """``b_in = a_in + a_out``; B's own consumption picks the suffix."""
    x = torch.randn(4, 4)
    a_out = torch.randn(4, 4)
    b_in = x + a_out
    b_out = torch.randn(4, 4)
    # B plainly consumed by a following block.
    cap, io = _synth_io(
        x, a_out, b_in, b_out, extra_consumers=(("c", b_out),)
    )
    assert _boundary("a", "b", cap, io) == "residual"
    # ... or B's output IS the model output.
    cap, io = _synth_io(x, a_out, b_in, b_out, model_out=b_out)
    assert _boundary("a", "b", cap, io) == "residual"
    # Residual-wrapped: ``b_in + b_out`` is a later block's input...
    cap, io = _synth_io(
        x,
        a_out,
        b_in,
        b_out,
        extra_consumers=(("c", b_in + b_out),),
    )
    assert _boundary("a", "b", cap, io) == "residual_wrapped"
    # ... or the model output itself.
    cap, io = _synth_io(x, a_out, b_in, b_out, model_out=b_in + b_out)
    assert _boundary("a", "b", cap, io) == "residual_wrapped"
    # Ambiguous: both plain and wrapped evidence → decline.
    cap, io = _synth_io(
        x,
        a_out,
        b_in,
        b_out,
        extra_consumers=(("c", b_out), ("d", b_in + b_out)),
    )
    assert _boundary("a", "b", cap, io) is None
    # Neither evidence: b_out's downstream is opaque → decline.
    cap, io = _synth_io(x, a_out, b_in, b_out, model_out=b_out * 2)
    assert _boundary("a", "b", cap, io) is None
    # Not a residual sum at all.
    cap, io = _synth_io(x, a_out, torch.randn(4, 4), b_out)
    assert _boundary("a", "b", cap, io) is None
    # Shape-mismatched candidates cannot form the residual sum.
    cap, io = _synth_io(
        torch.randn(4, 8), a_out, torch.randn(4, 8), b_out
    )
    assert _boundary("a", "b", cap, io) is None


def test_pair_boundary_chain_wrapped():
    """Chain boundary + residual-wrapped B → ``chain_wrapped``."""
    x = torch.randn(4, 4)
    a_out = torch.randn(4, 4)
    b_out = torch.randn(4, 4)
    wrapped = a_out + b_out
    cap, io = _synth_io(
        x,
        a_out,
        a_out.clone(),
        b_out,
        extra_consumers=(("c", wrapped),),
        b_in_obj=a_out,
    )
    assert _boundary("a", "b", cap, io) == "chain_wrapped"


def test_pair_boundary_residual_probe_required():
    """A first-probe residual match must be confirmed by the second
    (perturbed) capture — missing or inconsistent probe data declines."""
    x = torch.randn(4, 4)
    a_out = torch.randn(4, 4)
    b_in = x + a_out
    b_out = torch.randn(4, 4)
    cap, io = _synth_io(
        x, a_out, b_in, b_out, extra_consumers=(("c", b_out),)
    )
    # No second capture at all → the residual can't be confirmed.
    assert _boundary("a", "b", cap, io, probe=False) is None
    # Probe where the relation doesn't re-hold → declined.
    cap2, io2 = _probe(cap, io)
    cap2["b"] = ((torch.randn(4, 4),), {})
    assert O._pair_boundary("a", "b", cap, io, cap2, io2) is None


def test_residual_probe_decline_branches():
    """``_residual_probe`` fails closed on every malformed probe shape."""
    x = torch.randn(4, 4)
    a_out = torch.randn(4, 4)
    b_in = x + a_out
    cap, io = _synth_io(x, a_out, b_in, torch.randn(4, 4))
    # Missing pieces.
    assert not O._residual_probe("a", "b", {}, io)
    assert not O._residual_probe("a", "b", cap, {})
    assert not O._residual_probe("zz", "b", cap, io)
    # Re-entered producer on the probe.
    io2 = {**io, "a": {**io["a"], "calls": 2}}
    assert not O._residual_probe("a", "b", cap, io2)
    # Multi-arg / non-tensor probe captures.
    cap2 = {**cap, "a": ((x, x), {})}
    assert not O._residual_probe("a", "b", cap2, io)
    cap2 = {**cap, "b": (("not-a-tensor",), {})}
    assert not O._residual_probe("a", "b", cap2, io)
    io2 = {**io, "a": {**io["a"], "out": "not-a-tensor"}}
    assert not O._residual_probe("a", "b", cap, io2)
    # Relation doesn't re-hold.
    cap2 = {**cap, "b": ((torch.randn(4, 4),), {})}
    assert not O._residual_probe("a", "b", cap2, io)
    # Happy path.
    assert O._residual_probe("a", "b", cap, io)


def test_perturbed_input_variants():
    """The probe perturbs float tensors only and preserves structure."""
    f = torch.randn(4, 4)
    p = O._perturbed_input(f)
    assert torch.equal(p, f * 1.5 + 0.01)
    idx = torch.arange(4)  # integer tensors pass through unchanged
    assert O._perturbed_input(idx) is idx
    got = O._perturbed_input((f, idx, "meta"))
    assert isinstance(got, tuple) and got[1] is idx and got[2] == "meta"
    assert torch.equal(got[0], p)


def test_cross_pair_probe_failure_declines(monkeypatch):
    """If the perturbed second capture crashes, the pass fails closed:
    residual boundaries decline; the separately-optimized blocks still
    graft and the model stays correct."""
    torch.manual_seed(0)

    class ResidualWrapped(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                DeepParallel(dim, dim, dim) for _ in range(2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for b in self.blocks:
                x = x + b(x)
            return x

    monkeypatch.setattr(
        O, "_perturbed_input", lambda ex: torch.randn(3)
    )
    model = ResidualWrapped().eval().double()
    x = torch.randn(8, 32, dtype=torch.float64)

    opt, stats = optimize_compositional(model, x, verbose=False)

    e = stats["cross_pairs"]["blocks.0+blocks.1"]
    assert e["status"] == "skipped"
    assert e["reason"] == "no simple boundary"
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-9


def test_joint_pair_modes_and_wrappers():
    """The four micro-model modes compute their segment semantics."""
    a = _Scale().double()  # x -> 1.3 x
    lin = nn.Linear(4, 4, bias=False).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    for mode in (
        "chain",
        "chain_wrapped",
        "residual",
        "residual_wrapped",
    ):
        j = O._JointPair(a, lin, mode)
        with torch.no_grad():
            got = j(x)
            y = a(x)
            z = x + y if mode.startswith("residual") else y
            want = lin(z)
            if mode.endswith("_wrapped"):
                want = z + want
        assert torch.equal(got, want)

    inner = O._JointPair(a, lin, "chain")
    fused = O._FusedPair(inner, delta=False)
    delta_fused = O._FusedPair(inner, delta=True)
    zero = O._Zero()
    with torch.no_grad():
        assert torch.equal(fused(x), inner(x))
        assert torch.equal(delta_fused(x), inner(x) - x)
        assert torch.equal(zero(x), torch.zeros_like(x))


def test_executor_flops_paths():
    """``_executor_flops`` prices ``_root`` directly, via ``eval_mod``,
    and returns inf when nothing is priceable."""
    import types

    from catopt.optimize import optimize_model

    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    mod, _st = optimize_model(
        nn.Linear(8, 8, bias=False).double(), x, verbose=False
    )
    direct = O._executor_flops(mod)
    assert 0.0 < direct < float("inf")
    # Carrier-style wrapper: root reachable through eval_mod only.
    wrapped = types.SimpleNamespace(eval_mod=mod)
    assert O._executor_flops(wrapped) == direct
    # Unpriceable module → inf.
    assert O._executor_flops(nn.Identity()) == float("inf")
