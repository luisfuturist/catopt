"""KV-latent bounded mode — certified-approximate family grafts.

Plan-0012 follow-on.  ``KVLatentShare(budget=…)`` trades
certified-exact for certified-bound on *approximately* low-rank
families: the common basis truncates to the directions whose
summed per-site Frobenius residual fits the budget, the witness
offers ``error_bound = residual`` (measured, never claimed
smaller), the pair verify runs at the propagated bound — bounded
is verified-with-tolerance, not unverified — and the graft record
carries the bound into the match stats.
"""

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import Op, Param, TensorType, Var
from catopt_orchestrator import MorphismSearch, Optimizer
from catopt_torch.adapters import TorchSink, TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer

# ---------------------------------------------------------------------------
#  Fixtures — exact low-rank + small perturbation (the xKV case)
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
    """Attention-ish block whose k/v project through ``U`` + noise."""

    def __init__(
        self,
        dim: int,
        d_kv: int,
        U: torch.Tensor,
        seed: int,
        *,
        inexact: float = 0.0,
    ) -> None:
        """k/v = D·U + ``inexact``·randn — approximately low-rank."""
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
        inexact: float = 0.0,
    ) -> None:
        """Build ``depth`` approximately-low-rank KV blocks."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 10 + i, inexact=inexact)
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the member outputs."""
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


class _SingleKV(nn.Module):
    """One approximately-low-rank KV block — the intra MLA fold."""

    def __init__(
        self,
        dim: int = 16,
        d_kv: int = 8,
        rank: int = 4,
        inexact: float = 0.0,
    ) -> None:
        """One block whose k/v share data and an inexact factor."""
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            [_KVBlock(dim, d_kv, U, 70, inexact=inexact)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single block."""
        return self.blocks[0](x)


def _x(dims: tuple = (8, 16), seed: int = 0) -> torch.Tensor:
    """fp64 probe input."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*dims, generator=g, dtype=torch.float64)


def _lift(model: nn.Module, x: torch.Tensor) -> M.MorphismGraph:
    """Lift *model* through the torch ports."""
    return M.lift_graph(
        model, x, source=TorchSource(), composer=TorchComposer()
    )


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


def _family_match(graph, law=None):
    """The widest family match from *law* (default exact)."""
    law = K.KVLatentShare() if law is None else law
    return next(
        m for m in law.match(graph) if m.boundary == "family"
    )


def _site(name: str, orient: str = "linear"):
    """A minimal site record for the factorisation units."""
    x = Var("x", TensorType((8, 16)))
    return K._KVSite(
        "b",
        Op.make("linear", x, Param(name, TensorType((8, 16)))),
        x,
        Param(name, TensorType((8, 16))),
        orient,
    )


# ---------------------------------------------------------------------------
#  Units — truncation, residuals, the propagated bound
# ---------------------------------------------------------------------------


def test_truncate_basis_and_masses():
    """Budget truncation: cheapest directions drop while the running
    summed residual stays within budget; never below one vector."""
    U = _latent(3, 16)
    dk = torch.randn(8, 3, dtype=torch.float64)
    eps = 1e-2
    wk = dk @ U + eps * torch.randn(
        8,
        16,
        generator=torch.Generator().manual_seed(1),
        dtype=torch.float64,
    )
    dv = torch.randn(8, 3, dtype=torch.float64) @ U + eps * torch.randn(
        8,
        16,
        generator=torch.Generator().manual_seed(2),
        dtype=torch.float64,
    )
    mats = [wk, dv]
    basis = K._gs_row_basis(mats, 1e-8)
    assert len(basis) > 3  # the noise directions enter the cover

    # Degenerate calls pass the cover through untouched.
    assert K._truncate_basis([], [], 1.0) == []
    assert K._truncate_basis(mats, basis, 0.0) == list(basis)

    tb = K._truncate_basis(mats, basis, 0.5)
    assert 1 <= len(tb) < len(basis)
    # The measured summed residual honours the spend.
    Ut = K._stack_cols(tb)
    kept = [(_site("p_k"), wk, wk), (_site("p_v"), dv, dv)]
    resid, _mx = K._family_residual(kept, Ut)
    assert resid <= 0.5
    # A generous budget compresses harder or equal.
    tb2 = K._truncate_basis(mats, basis, 5.0)
    assert len(tb2) <= len(tb)


def test_factor_sites_bounded():
    """Bounded ``_factor_sites``: truncated basis, summed residual —
    and the coarse-cover case where even the base residual exceeds
    the budget."""
    U = _latent(3, 16)
    dk = torch.randn(8, 3, dtype=torch.float64)
    dv = torch.randn(8, 3, dtype=torch.float64)
    eps = 1e-2
    wk = dk @ U + eps * torch.randn(
        8,
        16,
        generator=torch.Generator().manual_seed(3),
        dtype=torch.float64,
    )
    wv = dv @ U + eps * torch.randn(
        8,
        16,
        generator=torch.Generator().manual_seed(4),
        dtype=torch.float64,
    )
    leaves = {"p_k": wk, "p_v": wv, "p_z": torch.zeros(8, 16)}
    good = [_site("p_k"), _site("p_v")]

    # Exact mode on the same leaves: the cover balloons to full rank.
    exact = K._factor_sites(good, leaves, 1e-8)
    assert exact is not None and exact[0].shape[1] > 4

    # Bounded: the basis truncates and err is the summed residual.
    fac = K._factor_sites(good, leaves, 1e-8, budget=0.5)
    assert fac is not None
    Ut, d_in, err, kept = fac
    assert d_in == 16 and int(Ut.shape[1]) < int(exact[0].shape[1])
    resid, mx = K._family_residual(kept, Ut)
    assert err == resid and err <= 0.5 and mx <= err

    # Empty cover still declines under a budget.
    assert (
        K._factor_sites([_site("p_z")], leaves, 1e-8, budget=1.0)
        is None
    )


def test_tmax_and_propagated_bound_edges():
    """The output-space estimate: every short-circuit leg."""
    x = _x()
    resid = 0.25

    class R:
        def __init__(self, out_val):
            self.out_val = out_val

    # A non-tensor input carries no sensitivity — both legs None.
    assert K._propagated_bound([R(x)], "x", resid) == (None, None)
    # No captured tensor output: the abs leg still reports.
    out_abs, rel = K._propagated_bound([R(None), R("s")], x, resid)
    assert out_abs == K._fnorm(x) * resid and rel is None
    # A duck-typed "tensor" lacking abs()/max() lands the same leg.
    fake = type(
        "T",
        (),
        {
            "shape": (2, 2),
            "dtype": "f8",
            "dim": 2,
            "abs": lambda self: self,
        },
    )()
    out_abs2, rel2 = K._propagated_bound([R(fake)], x, resid)
    assert out_abs2 == out_abs and rel2 is None
    # Real tensors give both legs; rel is abs normalised by |out|max.
    out = torch.ones(8, 16, dtype=torch.float64)
    oa, rl = K._propagated_bound([R(out), R(out)], x, resid)
    assert oa == out_abs and rl == oa / (2.0 + 1e-8)
    # _tmax directly: no abs, abs-without-max, and the real thing.
    assert K._tmax(object()) is None
    assert K._tmax(fake) is None
    assert K._tmax(out) == 1.0


# ---------------------------------------------------------------------------
#  Match surface — the budget rides the reify payload
# ---------------------------------------------------------------------------


def test_match_carries_budget():
    """``budget`` packs into ``reify.extra``; ``None`` by default."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack(inexact=1e-3).eval().double(), _x())
    m = _family_match(g, K.KVLatentShare(budget=0.25))
    assert m.reify.extra["budget"] == 0.25
    m_exact = _family_match(g)
    assert m_exact.reify.extra["budget"] is None


# ---------------------------------------------------------------------------
#  Reify — certified-bound grafts and honest declines
# ---------------------------------------------------------------------------


def test_bounded_reify_graft_fp64():
    """Within-bound perturbation: the bounded law grafts, bound =
    measured residual ≤ budget, verified at the propagated bound."""
    torch.manual_seed(0)
    x = _x()
    model = _SharedKVStack(inexact=1e-3).eval().double()
    g = _lift(model, x)
    out = _reify(_family_match(g, K.KVLatentShare(budget=0.2)), g)
    assert out["status"] == "grafted"
    # Honest bound: measured, ≤ budget, ≥ each per-site residual.
    assert 0.0 < out["error_bound"] <= 0.2
    assert out["error_bound"] >= out["factor_max_err"]
    assert out["bound_norm"] == "frobenius"
    assert out["error_budget"] == 0.2 and out["bounded"] is True
    # Verified-with-tolerance: the propagated bound, ≥ the fp gate.
    assert out["verify_tol"] >= 1e-4
    assert out["rel_diff"] < out["verify_tol"]
    assert out["latent_rank"] < 16
    # The delivered fused module sits inside the propagated bound.
    with torch.no_grad():
        d = (
            (model(x.clone()) - out["reps"]["blocks.0"](x.clone()))
            .abs()
            .max()
            .item()
        )
    assert d <= out["error_bound_out"]


def test_bounded_over_budget_declines():
    """Over-bound perturbations decline — never silently lossy.

    A perturbation needing more spend than the budget allows keeps
    a useless rank (the cost gate drops it); a cover coarser than
    the budget reports ``bound exceeds budget`` outright.
    """
    torch.manual_seed(0)
    # inexact=1e-2 needs ~0.6 of summed residual at rank 6 — a
    # budget of 0.05 buys almost nothing: the decline is honest.
    g = _lift(_SharedKVStack(inexact=1e-2).eval().double(), _x())
    out = _reify(_family_match(g, K.KVLatentShare(budget=0.05)), g)
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"
    assert out["latent_rank"] == 16

    # A factor tolerance looser than the budget: the measured
    # residual itself reports over-bound.
    g = _lift(_SharedKVStack(inexact=1e-3).eval().double(), _x())
    law = K.KVLatentShare(budget=0.01, factor_tol=0.4)
    out = _reify(_family_match(g, law), g)
    assert out["status"] == "declined"
    assert "bound exceeds budget" in out["reason"]


def test_bounded_intra_graft_fp64():
    """The single-block MLA fold grafts bounded too."""
    torch.manual_seed(0)
    x = _x()
    model = _SingleKV(inexact=1e-3).eval().double()
    g = _lift(model, x)
    ms = K.KVLatentShare(budget=0.1).match(g)
    m = next(mm for mm in ms if mm.boundary == "intra")
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["error_bound"] <= 0.1
    assert out["rel_diff"] < out["verify_tol"]
    with torch.no_grad():
        d = (model(x) - out["reps"]["blocks.0"](x)).abs().max().item()
    assert d <= out["error_bound_out"]


def test_bounded_exact_mode_unchanged():
    """``budget=None`` keeps today's gate byte-for-byte: the same
    inexact family still declines and no bound keys leak in."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack(inexact=1e-3).eval().double(), _x())
    out = _reify(_family_match(g), g)
    assert out["status"] == "declined"
    assert out["reason"] == "no_improvement"
    assert "bounded" not in out and "error_budget" not in out

    # And the exact family still grafts exactly under a budget —
    # the truncation only spends what the residual justifies.
    g = _lift(_SharedKVStack().eval().double(), _x())
    out = _reify(_family_match(g, K.KVLatentShare(budget=0.1)), g)
    assert out["status"] == "grafted"
    assert out["error_bound"] < 0.1 and out["rel_diff"] < 1e-9


def test_bounded_propagated_fallback():
    """A non-tensor captured input falls back to the weight-space
    bound as the verify tolerance — still verified, still grafted."""
    torch.manual_seed(0)
    g = _lift(_SharedKVStack(inexact=1e-3).eval().double(), _x())
    m = _family_match(g, K.KVLatentShare(budget=0.2))
    g.record("blocks.0").example = "not-a-tensor"
    out = _reify(m, g)
    assert out["status"] == "grafted"
    assert out["error_bound_out"] is None
    assert out["verify_tol"] == max(1e-4, out["error_bound"])


# ---------------------------------------------------------------------------
#  End to end — the bound surfaces in stats
# ---------------------------------------------------------------------------


def test_bounded_e2e_stats_report():
    """Full pipeline: the match stats carry the certified bound —
    never a silent lossy rewrite."""
    torch.manual_seed(0)
    model = _SharedKVStack(inexact=1e-3).eval().double()
    x = _x()
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(
            laws=[K.KVLatentShare(budget=0.2)], optimize_rest=False
        ),
    )
    m = stats["matches"]["kv_latent_share:blocks.0+blocks.1"]
    assert m["status"] == "grafted"
    assert m["bounded"] is True
    assert 0.0 < m["error_bound"] <= m["error_budget"] == 0.2
    assert m["bound_norm"] == "frobenius"
    assert m["error_bound_out"] is not None
    assert m["rel_diff"] < m["verify_tol"]
    # The fp64-vs-original diff on real inputs sits within the
    # propagated bound — and the end-to-end stat records it too.
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d <= m["error_bound_out"]
    assert stats["end_to_end"]["max_abs_diff"] == d
    assert stats["end_to_end"]["max_rel_diff"] == m["rel_diff"]
    # The consumed slot still contributes an exact zero.
    with torch.no_grad():
        z = opt.blocks[1](x.clone())
    assert torch.equal(z, torch.zeros_like(x))
