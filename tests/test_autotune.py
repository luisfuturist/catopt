"""Tests for :mod:`catopt_optimize.autotune` — measured autotuning
over the lowering paths."""

from __future__ import annotations

import time
from dataclasses import dataclass

import catopt_optimize.autotune as at_mod
import catopt_optimize.calibrate as cal_mod
import pytest
import torch
import torch.nn as nn
from catopt.calibrate import TargetProfile, shape_bucket
from catopt.cost import fused_cost_for
from catopt.ir import Op, Param, TensorType, Var
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


def test_torch_compile_candidates_real_compile():
    """The ``torch_compile`` candidates really wrap in torch.compile —
    a compile failure must surface as a recorded exclusion, never
    as a silent win."""
    torch.manual_seed(0)
    m = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    _mod, stats = _autotune(
        m, x, candidates=("generic", "torch_compile", "torch_compile_generic")
    )
    for name in ("torch_compile", "torch_compile_generic"):
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


# ---------------------------------------------------------------------------
# Measured feedback — profile channel, corrected pricing, graph overhead
# ---------------------------------------------------------------------------


def _term():
    x = Var("x", TensorType((64,)))
    w = Param("W", TensorType((64, 64)))
    return Op.make("matmul", x, w)


def test_measured_feedback_writes_profile_dict():
    """profile= persists every timed candidate's median under
    (candidate, shape_bucket) — the dict updates in place and rides
    back out as stats["autotune"]["profile"]."""
    m, x = _make()
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
    }
    _mod, stats = _autotune(
        m, x, candidates=("generic", "batched", "eager"), profile=prof
    )
    at = stats["autotune"]
    bucket = at["shape_bucket"]
    assert bucket == shape_bucket(x)
    assert at["profile"] is prof
    written = at["measured_ns"]
    assert set(written) == {"generic", "batched", "eager"}
    for name in ("generic", "batched", "eager"):
        ent = prof["measured_ns"][name][bucket]
        assert ent["median_ns"] == written[name]["median_ns"] > 0
    # priced candidates record the model price + residual; eager has
    # no priced lowering — its median substitutes outright.
    assert written["generic"]["model_ns"] > 0
    assert "residual_ns" in written["generic"]
    assert "model_ns" not in written["eager"]
    rec = at["candidates"]["generic"]
    assert rec["model_ns"] > 0
    # no prior entries → the corrected prediction IS the model price
    assert rec["predicted_ns"] == rec["model_ns"]
    # write-back also learned corrections: priced candidates got a
    # first ratio observation; unpriced eager learned nothing
    for name in ("generic", "batched"):
        ent = prof["corrections"][name][bucket]
        meas = prof["measured_ns"][name][bucket]
        ratio = meas["median_ns"] / meas["model_ns"]
        assert ent["n"] == 1
        # factor = the observed ratio, clamped to the sane range
        assert ent["factor"] == pytest.approx(
            min(max(ratio, 0.1), 10.0)
        )
    assert "eager" not in prof["corrections"]


def test_measured_feedback_targetprofile_replaced():
    """A frozen TargetProfile is NOT mutated — the updated profile
    comes back under stats["autotune"]["profile"]."""
    m, x = _make()
    prof = TargetProfile("t", 2.5, 89.0, 8.7, "cpu", "t0")
    _mod, stats = _autotune(m, x, candidates=("generic",), profile=prof)
    at = stats["autotune"]
    out = at["profile"]
    assert out is not prof
    assert prof.measured_ns == {}  # original untouched
    assert prof.corrections == {}
    ent = out.measured_ns["generic"][at["shape_bucket"]]
    assert ent["median_ns"] > 0 and ent["model_ns"] > 0
    corr = out.corrections["generic"][at["shape_bucket"]]
    assert corr["n"] == 1 and 0.1 <= corr["factor"] <= 10.0
    assert out.graph_overhead_us == prof.graph_overhead_us
    # the learned table serialises with the profile
    assert TargetProfile.from_json(out.to_json()) == out


def test_measured_entries_shift_candidate_order():
    """A fake profile whose measured_ns says a custom candidate ran at
    1 ns flips the ATTEMPT order — the correction is consumed before
    any timing runs."""
    m, x = _make()
    bucket = shape_bucket(x)

    def cheap(ctx):
        return ctx.delivered

    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "measured_ns": {"cheap": {bucket: {"median_ns": 1.0}}},
    }
    _mod, stats = _autotune(
        m, x, candidates=("generic", ("cheap", cheap)), profile=prof
    )
    at = stats["autotune"]
    # cheapest-predicted-first: the 1 ns entry leads the attempt order
    assert next(iter(at["candidates"])) == "cheap"
    assert at["predicted_winner"] == "cheap"
    assert at["candidates"]["cheap"]["predicted_ns"] == 1.0


def test_learned_corrections_shift_candidate_order():
    """A learned ``corrections`` factor is consumed by the attempt
    ordering: the corrected ``predicted_ns`` is ``model * factor``,
    and this run's timing grows the table's ``n``."""
    m, x = _make()
    bucket = shape_bucket(x)
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        # learned on a previous run: generic is a 10th of its model
        "corrections": {"generic": {bucket: {"factor": 0.1, "n": 3}}},
    }
    _mod, stats = _autotune(
        m, x, candidates=("batched", "generic"), profile=prof
    )
    at = stats["autotune"]
    g = at["candidates"]["generic"]
    assert g["predicted_ns"] == pytest.approx(g["model_ns"] * 0.1)
    # cheapest-predicted-first: the corrected generic leads the
    # attempt order despite being declared second
    assert next(iter(at["candidates"])) == "generic"
    assert at["predicted_winner"] == "generic"
    # this run's measurement folded into the learned entry (n: 3 → 4)
    ent = prof["corrections"]["generic"][bucket]
    assert ent["n"] == 4
    assert 0.1 <= ent["factor"] <= 10.0


def test_below_min_samples_falls_back_to_residual():
    """One observation is a measured_ns residual, not a learned
    factor — the delivered prediction is the measurement itself, and
    ``model_ns`` stays the raw model price."""
    m, x = _make()
    bucket = shape_bucket(x)
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        # n=1 < min_samples → the factor does NOT apply; the direct
        # measurement (median 1 ns) transfers instead
        "corrections": {"generic": {bucket: {"factor": 9.9, "n": 1}}},
        "measured_ns": {
            "generic": {bucket: {"median_ns": 1.0, "model_ns": 5.0}}
        },
    }
    _mod, stats = _autotune(
        m, x, candidates=("batched", "generic"), profile=prof
    )
    g = stats["autotune"]["candidates"]["generic"]
    assert g["model_ns"] > 0
    # residual transfer, not model * 9.9
    assert g["predicted_ns"] == pytest.approx(
        g["model_ns"] + (1.0 - 5.0)
    )


def test_ordering_uses_measured_truth_after_one_run():
    """One run's write-back reaches the next run's ordering: a
    5 ms custom candidate goes from unpriced to last-predicted."""
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

    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
    }
    _autotune(
        m, x, candidates=("generic", ("slow", slow)), profile=prof
    )
    bucket = shape_bucket(x)
    assert prof["measured_ns"]["slow"][bucket]["median_ns"] > 1e6
    _mod, stats = _autotune(
        m, x, candidates=("generic", ("slow", slow)), profile=prof
    )
    at = stats["autotune"]
    # slow has no model price but its measured median prices it —
    # predicted ordering now matches the measured truth
    assert at["predicted_winner"] == "generic"
    assert (
        at["candidates"]["slow"]["predicted_ns"]
        > at["candidates"]["generic"]["predicted_ns"]
    )
    assert next(iter(at["candidates"])) == "generic"


def test_predicted_ns_without_profile_is_model_only():
    """Without profile= nothing is persisted and nothing reorders —
    but predicted prices are still recorded for fidelity."""
    m, x = _make()
    _mod, stats = _autotune(m, x, candidates=("generic", "eager"))
    at = stats["autotune"]
    c = at["candidates"]
    assert c["generic"]["model_ns"] > 0
    assert c["generic"]["predicted_ns"] == c["generic"]["model_ns"]
    assert "model_ns" not in c["eager"]
    assert "predicted_ns" not in c["eager"]
    assert at["predicted_winner"] == "generic"
    assert "measured_ns" not in at and "profile" not in at
    # second call ordering is the declared order — verify via records
    assert list(c)[:2] == ["generic", "eager"]


def test_candidate_pricing_failure_is_soft(monkeypatch):
    """A cost model that raises on a candidate must never break the
    measurement run — the candidate simply goes unpriced."""
    monkeypatch.setattr(
        at_mod,
        "executor_cost_for",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    m, x = _make()
    _mod, stats = _autotune(m, x, candidates=("generic",))
    rec = stats["autotune"]["candidates"]["generic"]
    assert rec["status"] == "timed"
    assert "model_ns" not in rec
    assert stats["autotune"]["predicted_ns"] is None


def test_compiled_price_charges_graph_overhead():
    """``graph_overhead_us`` widens the fused per-graph term to
    ``max(dispatch_us, graph_overhead_us)``: larger constants price
    the same term higher; below dispatch it is a no-op."""
    t = _term()
    base = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
    }
    # fused_cost_for now consumes graph_overhead_us itself
    # (per-graph term = max(dispatch, overhead)) — the baseline pins
    # it to the dispatch floor so the deltas below are pure overhead.
    raw = fused_cost_for({**base, "graph_overhead_us": 1.0})(t)
    low = at_mod._compiled_model_ns(
        t, {**base, "graph_overhead_us": 0.5}
    )
    assert low == pytest.approx(raw)  # below the dispatch floor
    high = at_mod._compiled_model_ns(
        t, {**base, "graph_overhead_us": 200.0}
    )
    assert high == pytest.approx(raw + (200.0 - 1.0) * 1e3)
    # absent → the calibrated fallback applies
    fb = cal_mod._FALLBACK_GRAPH_OVERHEAD_US
    none = at_mod._compiled_model_ns(t, dict(base))
    assert none == pytest.approx(raw + (fb - 1.0) * 1e3)
    # a TargetProfile consumes identically
    tp = TargetProfile(
        "t",
        2.5,
        89.0,
        8.7,
        "cpu",
        "t0",
        dispatch_us=1.0,
        graph_overhead_us=150.0,
    )
    assert at_mod._compiled_model_ns(t, tp) == pytest.approx(
        raw + (150.0 - 1.0) * 1e3
    )


def test_fused_charges_graph_overhead_detection(monkeypatch):
    """When the installed fused model already consumes the field the
    shim must not double-charge — detected behaviourally."""

    def fake_fused(profile):
        ov = (
            profile.get("graph_overhead_us", 0.0)
            if isinstance(profile, dict)
            else 0.0
        )

        def cost(t, memo=None):
            return 100.0 + ov * 1e3

        return cost

    monkeypatch.setattr(at_mod, "fused_cost_for", fake_fused)
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        "graph_overhead_us": 50.0,
    }
    assert at_mod._fused_charges_graph_overhead(prof) is True
    # native consumption detected → the price carries it, no surplus
    assert at_mod._compiled_model_ns(_term(), prof) == pytest.approx(
        100.0 + 50e3
    )


def test_graph_overhead_detection_exotic_profiles():
    """Unpriceable / uncarryable profile types report not-native."""
    # plain object() — clone accepts no attribute → None → False
    assert at_mod._fused_charges_graph_overhead(object()) is False

    @dataclass(frozen=True)
    class Frozen:  # dataclass WITHOUT the field → replace() raises
        x: int = 0

    assert at_mod._fused_charges_graph_overhead(Frozen()) is False

    class Bare:  # clones fine, but has no tflops — pricing raises
        pass

    assert at_mod._fused_charges_graph_overhead(Bare()) is False

    class Obj:  # priceable attribute profile — and the installed
        # fused model now natively consumes graph_overhead_us, so
        # the behavioural probe reports True on it.
        tflops = 2.5
        gbps = 89.0
        launch_us = 8.7
        dispatch_us = 1.0

    assert at_mod._fused_charges_graph_overhead(Obj()) is True
    assert at_mod._compiled_model_ns(_term(), Obj()) > 0.0
