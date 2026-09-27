"""Tests for :mod:`catopt_optimize.autotune` — measured autotuning
over the lowering paths."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn
from catopt_optimize.autotune import (
    CandidateUnavailableError,
    optimize_model_autotuned,
)
from catopt_torch.adapters import TorchSink
from catopt_torch.report import verify_module


class _MLP(nn.Module):
    """relu(linear(x)) — small enough to saturate in a few iters."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.w(x))


class _TwoInput(nn.Module):
    """``add(linear(a), b)`` — exercises tuple ``example_input``."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(8, 8)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.w(a) + b


def _make():
    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(2, 8)


def _autotune(m, x, **kw):
    kw.setdefault("n_calls", 5)
    kw.setdefault("warmup", 2)
    kw.setdefault("max_iterations", 3)
    return optimize_model_autotuned(m, x, **kw)


def test_returns_verified_module_and_winner():
    m, x = _make()
    mod, stats = _autotune(
        m, x, candidates=("generic", "batched"), verbose=True
    )
    at = stats["autotune"]
    assert at["winner"] in ("generic", "batched")
    assert at["fallback"] is False
    assert at["ir_recovered"] is True
    # The returned module is verified-equivalent to the original.
    assert verify_module(m, mod, x).passed
    # optimize_model's own stats still ride along.
    assert "lowering" in stats
    assert at["search_s"] >= 0 and at["elapsed_s"] >= at["search_s"]


def test_per_candidate_timings_recorded():
    m, x = _make()
    _mod, stats = _autotune(
        m,
        x,
        candidates=("generic", "batched", "eager"),
        n_calls=2,  # <4: the no-IQR timing branch
    )
    cands = stats["autotune"]["candidates"]
    for name in ("generic", "batched", "eager"):
        rec = cands[name]
        assert rec["status"] == "timed"
        assert rec["verified"] is True
        assert rec["median_s"] > 0
        assert rec["n_calls"] == 2
        assert "iqr_s" in rec and "max_rel" in rec
    assert (
        stats["autotune"]["winner_median_s"]
        == cands[stats["autotune"]["winner"]]["median_s"]
    )


def test_batched_route_on_scan_model():
    """LinearRecurrence lowers to the batched carrier — the
    ``batched`` candidate reuses it and ``generic`` re-lowers the
    same term serially."""
    from catopt.models import LinearRecurrence

    torch.manual_seed(0)
    m = LinearRecurrence(dim=4, steps=8).eval()
    x = torch.randn(8, 4)
    mod, stats = _autotune(m, x, candidates=("generic", "batched"))
    at = stats["autotune"]
    assert stats["lowering"] == "batched"
    assert at["winner"] in ("generic", "batched")
    assert verify_module(m, mod, x).passed
    for name in ("generic", "batched"):
        assert at["candidates"][name]["status"] == "timed"
        assert at["candidates"][name]["verified"] is True


def test_tuple_example_input():
    torch.manual_seed(0)
    m = _TwoInput().eval()
    x = (torch.randn(2, 8), torch.randn(2, 8))
    mod, stats = _autotune(m, x, candidates=("generic",))
    assert stats["autotune"]["winner"] == "generic"
    assert verify_module(m, mod, x).passed


def test_failed_and_wrong_candidates_excluded():
    m, x = _make()

    def boom(ctx):
        raise RuntimeError("cannot build")

    def wrong(ctx):
        class W(nn.Module):
            def forward(self, x):
                return ctx.delivered(x) * 2

        return W()

    def explodes_in_verify(ctx):
        class W(nn.Module):
            def forward(self, x):
                raise RuntimeError("boom at call")

        return W()

    _mod, stats = _autotune(
        m,
        x,
        candidates=(
            "generic",
            ("boom", boom),
            ("wrong", wrong),
            ("vbad", explodes_in_verify),
            "no_such_candidate",
        ),
    )
    cands = stats["autotune"]["candidates"]
    assert cands["boom"]["status"] == "build_failed"
    assert cands["wrong"]["status"] == "verify_failed"
    assert cands["vbad"]["status"] == "verify_error"
    assert cands["no_such_candidate"]["status"] == "unknown"
    # None of the failed candidates may win; the timed one does.
    assert stats["autotune"]["winner"] == "generic"


def test_slow_candidate_loses():
    m, x = _make()

    def slow(ctx):
        inner = ctx.delivered

        class S(nn.Module):
            def forward(self, x):
                deadline = time.perf_counter() + 0.005
                while time.perf_counter() < deadline:
                    pass
                return inner(x)

        return S()

    mod, stats = _autotune(m, x, candidates=("generic", ("slow", slow)))
    assert stats["autotune"]["winner"] == "generic"
    assert stats["autotune"]["candidates"]["slow"]["status"] == "timed"
    assert verify_module(m, mod, x).passed


def test_time_failed_candidate_excluded():
    m, x = _make()

    def flaky(ctx):
        inner = ctx.delivered

        class F(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                if self.calls > 1:  # verify passes, timing explodes
                    raise RuntimeError("timed-call failure")
                return inner(x)

        return F()

    _mod, stats = _autotune(
        m, x, candidates=("generic", ("flaky", flaky))
    )
    cands = stats["autotune"]["candidates"]
    assert cands["flaky"]["status"] == "time_failed"
    assert stats["autotune"]["winner"] == "generic"


def test_fallback_when_everything_fails():
    m, x = _make()

    def boom(ctx):
        raise RuntimeError("cannot build")

    mod, stats = _autotune(m, x, candidates=(("boom", boom),))
    at = stats["autotune"]
    assert at["winner"] is None
    assert at["fallback"] is True
    # Fallback = the plain optimize_model output, verified honestly.
    assert at["candidates"]["_pipeline_fallback"]["verified"] is True
    assert verify_module(m, mod, x).passed


def test_budget_skips_candidates():
    m, x = _make()
    _mod, stats = _autotune(
        m,
        x,
        candidates=("generic", "batched"),
        budget_s=0.0,  # search already over budget
    )
    at = stats["autotune"]
    assert at["fallback"] is True
    for name in ("generic", "batched"):
        assert at["candidates"][name]["status"] == "skipped"


def test_cuda_graph_unavailable_on_cpu():
    m, x = _make()
    _mod, stats = _autotune(m, x, candidates=("generic", "cuda_graph"))
    rec = stats["autotune"]["candidates"]["cuda_graph"]
    assert rec["status"] == "unavailable"  # CPU input


def test_compiled_candidates_real_compile():
    """The ``compiled`` candidates really wrap in torch.compile —
    a compile failure must surface as a recorded exclusion, never
    as a silent win."""
    torch.manual_seed(0)
    m = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    _mod, stats = _autotune(
        m, x, candidates=("generic", "compiled", "compiled_generic")
    )
    for name in ("compiled", "compiled_generic"):
        rec = stats["autotune"]["candidates"][name]
        assert rec["status"] in (
            "timed",
            "verify_error",  # compile surfaces at first call
            "build_failed",
        )
        if rec["status"] == "timed":
            assert rec["verified"] is True


def test_unknown_and_unavailable_do_not_break_winner():
    m, x = _make()

    def unavailable(ctx):
        raise CandidateUnavailableError("not on this device")

    mod, stats = _autotune(
        m, x, candidates=(("nope_dev", unavailable), "generic")
    )
    assert (
        stats["autotune"]["candidates"]["nope_dev"]["status"]
        == "unavailable"
    )
    assert stats["autotune"]["winner"] == "generic"
    assert verify_module(m, mod, x).passed


def test_opaque_sink_recovers_via_fallback():
    """A sink whose executor exposes no IR internals: recovery
    fails, lowerer candidates report unavailable, verify errors are
    recorded, and the delivered module is returned via fallback —
    with ``verified=False`` honestly recorded."""

    class OpaqueSink(TorchSink):
        def lower(self, ir, params=None):
            return nn.Identity()  # no _root/_inputs/_param_map

        def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
            raise RuntimeError("verify unavailable")

    m, x = _make()
    mod, stats = optimize_model_autotuned(
        m,
        x,
        candidates=("generic", "batched"),
        sink=OpaqueSink(),
        n_calls=5,
        warmup=2,
        max_iterations=3,
    )
    at = stats["autotune"]
    assert at["ir_recovered"] is False
    assert at["fallback"] is True
    assert at["candidates"]["generic"]["status"] == "verify_error"
    assert at["candidates"]["batched"]["status"] == "unavailable"
    assert at["candidates"]["_pipeline_fallback"]["verified"] is False
    assert mod is not None


@pytest.mark.requires_cuda
def test_cuda_candidates():
    """CUDA input: ``cuda_graph`` should either time (batched
    carrier captured) or record an honest status — never a silent
    win."""
    from catopt.models import LinearRecurrence

    torch.manual_seed(0)
    m = LinearRecurrence(dim=8, steps=4).cuda().eval()
    x = torch.randn(4, 8, device="cuda")
    mod, stats = optimize_model_autotuned(
        m,
        x,
        candidates=("generic", "batched", "cuda_graph"),
        n_calls=5,
        warmup=2,
        max_iterations=4,
    )
    at = stats["autotune"]
    rec = at["candidates"]["cuda_graph"]
    assert rec["status"] in ("timed", "unavailable", "build_failed")
    if at["fallback"]:
        assert verify_module(m, mod, x).passed
    else:
        assert at["candidates"][at["winner"]]["verified"] is True
