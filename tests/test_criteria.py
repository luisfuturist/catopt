"""Criterion-based selection — ``criteria_cost`` blends and the
``optimize_model(criteria=...)`` param.

The blend is a weighted sum of named cost axes
(:func:`catopt_optimize.criteria.criteria_cost`).  Sum of
additive-per-node models is additive, so extraction's subtractive
local-cost recovery stays exact for the additive axes — the tests
below check that contract, the selection flip when two axes
disagree, the validation surface, and the stats record.
"""

import math

import pytest
import torch
from catopt.cost import (
    _LAUNCH_S,
    _local_roofline,
    executor_cost_for,
    flops_cost,
    param_bytes_cost_for,
)
from catopt.egraph import EGraph
from catopt.ir import Op, Param, TensorType, Var
from catopt.ports import CostFn, signature_conforms
from catopt_optimize.criteria import AXES, criteria_cost


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _member_storage() -> Op:
    """Compute-light, memory-heavy: one GEMM on a *stored* weight —
    2048 FLOPs, 16 stored values."""
    x = _v("x", 64, 4)
    return Op.make("matmul", x, _p("p_w", 4, 4))


def _member_compute() -> Op:
    """Compute-heavy, memory-free: a larger GEMM on a *data* operand
    plus a neg — 4608 FLOPs, nothing stored."""
    x = _v("x", 64, 4)
    y = _v("y", 4, 8)
    return Op.make("neg", Op.make("matmul", x, y))


def _two_member_graph():
    """One e-class holding the storage member and the compute member
    (the union asserts their equality — the same mechanism a
    witnessed rewrite uses)."""
    a = _member_storage()
    b = _member_compute()
    eg = EGraph()
    ea = eg.add_term(a)
    eb = eg.add_term(b)
    eg.union(ea, eb)
    eg.rebuild()
    return eg, eg.find(ea), a, b


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------


def test_criteria_unknown_axis_rejected():
    with pytest.raises(ValueError, match="unknown criteria axes"):
        criteria_cost({"latency": 1.0, "speed": 1.0})
    # every documented axis is accepted
    for axis in AXES:
        criteria_cost({axis: 1.0})


@pytest.mark.parametrize(
    "bad", [-1.0, float("nan"), float("inf"), "x", None]
)
def test_criteria_bad_weight_rejected(bad):
    with pytest.raises(ValueError, match="non-negative weight"):
        criteria_cost({"latency": bad})


def test_criteria_no_positive_weight_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost({})
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost({"latency": 0.0, "flops": 0.0})


# ---------------------------------------------------------------------------
#  The blend itself
# ---------------------------------------------------------------------------


def test_criteria_none_is_the_shipped_default():
    """criteria=None blends {"latency": 1.0} — exactly
    ``executor_cost_for(lowering="generic")``."""
    t = Op.make("add", _v("x", 4, 8), _p("w", 4, 8))
    assert criteria_cost()(t) == pytest.approx(
        executor_cost_for(lowering="generic")(t)
    )


def test_criteria_weights_normalise_to_one():
    """Only relative weights matter — a uniform rescale is the same
    blend, recorded normalised."""
    t = Op.make("mul", _v("x", 8, 8), _v("y", 8, 8))
    a = criteria_cost({"flops": 2.0, "latency": 6.0})
    b = criteria_cost({"flops": 1.0, "latency": 3.0})
    assert a(t) == pytest.approx(b(t))
    assert a.criteria == {"flops": 0.25, "latency": 0.75}
    assert b.criteria == a.criteria
    assert sum(b.criteria.values()) == pytest.approx(1.0)
    # and the blend value is the weighted sum of the axis models
    assert a(t) == pytest.approx(
        0.25 * flops_cost(t)
        + 0.75 * executor_cost_for(lowering="generic")(t)
    )


def test_criteria_zero_weight_axis_is_dropped():
    c = criteria_cost({"flops": 1.0, "memory": 0.0})
    assert "memory" not in c.criteria
    assert c.charges_param_only is False


def test_criteria_memory_axis_sets_charges_param_only():
    """Storage billing must survive the param-only discount — the
    blend carries param_bytes_cost's marker through."""
    c = criteria_cost({"memory": 1.0})
    assert c.charges_param_only is True
    t = Op.make("linear", _v("x", 4, 64), _p("W", 64, 64))
    assert c(t) == pytest.approx(param_bytes_cost_for()(t))


def test_criteria_conforms_to_costfn_port():
    assert signature_conforms(criteria_cost({"flops": 1.0}), CostFn)


def test_criteria_profile_binds_time_axes():
    """profile threads into the ns-priced axes and rides the closure
    for reporting / extraction's memo pre-seeding."""
    prof = {"tflops": 10.0, "gbps": 100.0, "launch_us": 1.0}
    c = criteria_cost({"latency": 1.0, "depth": 1.0}, profile=prof)
    assert c.profile is prof
    t = Op.make("matmul", _v("x", 32, 32), _p("W", 32, 32))
    assert math.isfinite(c(t))


def test_criteria_all_axes_price_finite():
    """Every axis yields a callable blend — depth/compiled included
    (their whole-DAG contribution is the documented approximation)."""
    c = criteria_cost(
        {a: 1.0 for a in AXES},
        profile={"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7},
    )
    t = Op.make(
        "add",
        Op.make("matmul", _v("x", 16, 16), _p("W", 16, 16)),
        _v("y", 16, 16),
    )
    assert math.isfinite(c(t))


# ---------------------------------------------------------------------------
#  Additive safety — the extract_best contract
# ---------------------------------------------------------------------------


def test_criteria_blend_recovers_exact_local():
    """For additive axes the recovered local
    ``blend(t) - sum(blend(children))`` is exactly the weighted sum of
    the node-level contributions — what extract_best bills."""
    x = _v("x", 16, 32)
    y = _v("y", 16, 32)
    t = Op.make("add", Op.make("mul", x, y), Op.make("neg", x))
    wf, wl = 0.25, 0.75
    blend = criteria_cost({"flops": wf, "latency": wl})
    local = blend(t) - sum(blend(a) for a in t.args)
    # flops: add bills 1 * numel(out) = 512; latency: the node's local
    # roofline plus one generic dispatch (per = launch_s in ns).
    expected = wf * 512.0 + wl * (_local_roofline(t) + _LAUNCH_S * 1e9)
    assert local == pytest.approx(expected)


def test_criteria_extraction_picks_true_blend_min():
    """A saturated class with two members: extraction returns the
    member minimizing the blend — checked against direct prices."""
    eg, root, a, b = _two_member_graph()
    blend = criteria_cost({"flops": 1.0, "memory": 1.0})
    best = eg.extract_best(root, blend)
    assert best == (a if blend(a) < blend(b) else b)


# ---------------------------------------------------------------------------
#  Weighted blend changes selection — two axes disagree
# ---------------------------------------------------------------------------


def test_criteria_blend_flips_selection_between_axes():
    """The storage member wins on FLOPs; the compute member wins on
    memory (it stores nothing).  The blend crosses over as the
    memory weight grows."""
    eg, root, storage, compute = _two_member_graph()
    assert flops_cost(storage) < flops_cost(compute)
    assert param_bytes_cost_for()(compute) < param_bytes_cost_for()(
        storage
    )

    assert (
        eg.extract_best(root, criteria_cost({"flops": 1.0})) == storage
    )
    assert (
        eg.extract_best(root, criteria_cost({"memory": 1.0})) == compute
    )
    # flops-dominant blend: storage still wins
    assert (
        eg.extract_best(
            root, criteria_cost({"flops": 1.0, "memory": 0.001})
        )
        == storage
    )
    # memory-dominant blend: the crossover has passed
    assert (
        eg.extract_best(
            root, criteria_cost({"flops": 1.0, "memory": 1000.0})
        )
        == compute
    )


# ---------------------------------------------------------------------------
#  optimize_model(criteria=...) — precedence + stats record
# ---------------------------------------------------------------------------


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()


def test_optimize_model_criteria_records_stats():
    """criteria= builds the blend, stats records the normalised
    weights actually priced, and the module stays equivalent."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m, x, verbose=False, criteria={"latency": 2.0}
        )
    assert stats["criteria"] == {"latency": 1.0}
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_default_run_records_no_criteria():
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        _mod, stats = optimize_model(m, x, verbose=False)
    assert stats["criteria"] is None


def test_optimize_model_explicit_cost_fn_beats_criteria():
    """Precedence: explicit cost_fn > criteria > default — a given
    cost_fn leaves the criteria record empty (none was priced)."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m,
            x,
            verbose=False,
            cost_fn=flops_cost,
            criteria={"memory": 1.0},
        )
    assert stats["criteria"] is None
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_bad_criteria_fails_loud():
    """An unknown axis surfaces as ValueError from the optimizer —
    validation isn't deferred to extraction."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with pytest.raises(ValueError, match="unknown criteria axes"):
        optimize_model(m, x, verbose=False, criteria={"speed": 1.0})
