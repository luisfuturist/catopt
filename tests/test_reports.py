"""Typed reports + the uniform verify gate (plan 0001 phase 3a/3b).

The stats dicts stay dicts at the public boundary; ``OptReport`` /
``CompositionalReport`` / ``BlockReport`` are the same schema as code
with an exact ``to_dict`` round-trip, and ``verify_equiv`` /
``verify_module`` replace the three near-duplicate diff+tolerance
implementations with one bit-identical computation.
"""

import torch
import torch.nn as nn

from catopt.models import ParallelLinear
from catopt.optimize import optimize_compositional, optimize_model
from catopt.report import (
    BlockReport,
    CompositionalReport,
    OptReport,
    rel_diff,
    verify_equiv,
    verify_module,
)


def _small_mlp(seed=0):
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(16, 16, bias=False)
            s.l2 = nn.Linear(16, 8, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    return M().eval().double()


# ---------------------------------------------------------------------------
#  OptReport — real optimize_model stats round-trip
# ---------------------------------------------------------------------------


def test_optreport_roundtrips_real_stats():
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    _opt, stats = optimize_model(m, x, verbose=False, max_iterations=3)

    rep = OptReport.from_stats(stats)
    assert rep.to_dict() == stats

    # typed access to the always-present saturation keys
    assert rep.iterations == stats["iterations"]
    assert rep.n_enodes == stats["n_enodes"]
    assert rep.n_classes == stats["n_classes"]
    assert rep.rule_fires == stats["rule_fires"]

    # the grouped view carries the EGraph.run payload
    sat = rep.saturation_stats
    assert sat.n_enodes == rep.n_enodes
    assert sat.iterations == rep.iterations
    assert sat.rule_budgets == rep.rule_budgets


def test_optreport_path_dependent_keys():
    x = torch.randn(2, 16, dtype=torch.float64)

    # absent by path → None field, omitted on serialize
    m = _small_mlp()
    _opt, stats = optimize_model(m, x, verbose=False, max_iterations=3)
    rep = OptReport.from_stats(stats)
    assert rep.eps_offers is None
    assert rep.paired_extract is None
    assert rep.causal_specialized is None
    assert "eps_offers" not in rep.to_dict()
    assert "paired_extract" not in rep.to_dict()

    # eps_rtol opt-in records eps_offers; round-trip still exact
    _opt, stats = optimize_model(
        m, x, eps_rtol=0.9, verbose=False, max_iterations=3
    )
    rep = OptReport.from_stats(stats)
    assert rep.eps_offers == stats["eps_offers"]
    assert rep.to_dict() == stats

    # the pairing pass sets pairing_groups / paired_extract
    pm = ParallelLinear(16, n_experts=2).eval().double()
    _opt, stats = optimize_model(pm, x, verbose=False)
    rep = OptReport.from_stats(stats)
    assert rep.pairing_groups == stats.get("pairing_groups")
    assert rep.pairing_groups is not None and rep.pairing_groups >= 1
    assert rep.to_dict() == stats


def test_optreport_extra_keys_preserved():
    """Keys outside the schema round-trip verbatim through ``extra`` —
    forward-compat for fields added before the dataclass learns them."""
    stats = {
        "iterations": 2,
        "n_enodes": 42,
        "some_future_key": {"nested": [1, 2]},
    }
    rep = OptReport.from_stats(stats)
    assert rep.extra == {"some_future_key": {"nested": [1, 2]}}
    assert rep.to_dict() == stats


# ---------------------------------------------------------------------------
#  BlockReport — exact dict shape per status
# ---------------------------------------------------------------------------


def test_blockreport_serializes_exact_shape():
    cases = [
        {"status": "not_executed"},
        {
            "status": "skipped",
            "reason": "non-positional kwargs ['scale']",
        },
        {
            "status": "failed",
            "error": "RuntimeError: export blew up",
            "time_s": 0.5,
        },
        {
            "status": "failed",
            "rel_diff": 0.3,
            "error": "RuntimeError: block verification failed",
            "time_s": 1.25,
        },
        {
            "status": "failed",
            "error": "OptimizationResourceError: e-graph reached",
            "reason": "resource_limit",
            "time_s": 0.1,
        },
        {
            "status": "optimized",
            "rel_diff": 1e-9,
            "stats": {"iterations": 2, "n_enodes": 30},
            "param_report": {"original_params": 2, "ratio": 0.5},
            "time_s": 0.2,
        },
        # unknown keys pass through untouched
        {"status": "optimized", "custom_field": "kept", "time_s": 0.0},
    ]
    for entry in cases:
        rep = BlockReport.from_dict(entry, name="blocks.0")
        assert rep.name == "blocks.0"
        assert rep.to_dict() == entry, entry


def test_blockreport_name_not_serialized():
    """``name`` is the dict key of rep["blocks"], not an entry key."""
    rep = BlockReport(name="blocks.3", status="not_executed")
    assert rep.to_dict() == {"status": "not_executed"}


# ---------------------------------------------------------------------------
#  CompositionalReport — real optimize_compositional stats round-trip
# ---------------------------------------------------------------------------


def test_compositional_report_roundtrips_real_stats():
    torch.manual_seed(0)
    model = nn.Sequential(
        ParallelLinear(16, n_experts=2),
        ParallelLinear(16, n_experts=2),
    ).double()
    x = torch.randn(4, 16, dtype=torch.float64)

    _opt, stats = optimize_compositional(
        model, x, verbose=False, max_iterations=5
    )
    rep = CompositionalReport.from_stats(stats)
    assert rep.to_dict() == stats

    assert rep.compositional is True
    assert rep.n_blocks == stats["n_blocks"]
    assert rep.n_optimized == stats["n_optimized"]
    assert rep.in_place is False
    assert rep.param_report == stats["param_report"]
    assert rep.end_to_end == stats["end_to_end"]
    assert rep.wall_time_s == stats["wall_time_s"]

    for name, blk in rep.blocks.items():
        entry = stats["blocks"][name]
        assert blk.name == name
        assert blk.status == entry["status"]
        if blk.stats is not None:
            assert blk.stats.n_enodes == entry["stats"]["n_enodes"]
            assert blk.stats.to_dict() == entry["stats"]


# ---------------------------------------------------------------------------
#  verify_equiv — bit-identical with the replaced inline computation
# ---------------------------------------------------------------------------


def test_verify_equiv_matches_inline_computation():
    torch.manual_seed(0)
    ref = torch.randn(4, 8, dtype=torch.float64)
    cases = [
        (ref, ref.clone()),  # identical
        (ref, ref + 1e-7),  # within tolerance
        (ref, ref + 1e-2),  # outside tolerance
        (ref, -ref),  # sign flip
        (  # zero reference — the 1e-8 denominator floor decides
            torch.zeros(4, 8, dtype=torch.float64),
            torch.full((4, 8), 1e-9, dtype=torch.float64),
        ),
    ]
    for ref_t, out_t in cases:
        # the exact computation the three call sites used to inline
        max_diff = (ref_t - out_t).abs().max().item()
        rd = max_diff / (ref_t.abs().max().item() + 1e-8)
        passed = rd < 1e-4

        vr = verify_equiv(ref_t, out_t, rtol=1e-4)
        assert vr.max_abs == max_diff
        assert vr.max_rel == rd
        assert vr.passed == passed
        assert rel_diff(ref_t, out_t) == rd


def test_verify_equiv_atol_and_rtol():
    ref = torch.ones(4, dtype=torch.float64)
    out = ref + 1e-7
    # rel gate alone passes; an atol tighter than max|Δ| fails
    assert verify_equiv(ref, out, rtol=1e-4).passed
    assert not verify_equiv(ref, out, rtol=1e-4, atol=1e-12).passed
    # a loose atol passes again
    assert verify_equiv(ref, out, rtol=1e-4, atol=1e-3).passed


def test_verify_module_single_and_tuple_inputs():
    m1 = _small_mlp()
    m2 = _small_mlp()  # same seed — bitwise identical weights
    x = torch.randn(2, 16, dtype=torch.float64)

    vr = verify_module(m1, m2, x)
    assert vr.passed and vr.max_abs == 0.0

    class TwoIn(nn.Module):
        def forward(s, a, b):
            return a + b

    t = TwoIn().eval()
    a, b = torch.randn(2, 2), torch.randn(2, 2)
    vr = verify_module(t, t, (a, b))
    assert vr.passed

    # a genuinely different module fails the gate
    m3 = _small_mlp(seed=7)
    vr = verify_module(m1, m3, x, rtol=1e-9)
    assert not vr.passed
    assert vr.max_abs > 0.0
