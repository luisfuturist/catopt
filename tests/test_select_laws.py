"""Tests for the ``select_mul`` law — elementwise ``mul`` through
``select`` (:data:`catopt_core.laws.tensor.SELECT_MUL`).

``select`` is a stride view (``torch.select``), the same kind of free
re-layout as ``transpose``.  Elementwise ``mul`` commutes with it:

    mul(select(u, dim=D, index=I), select(v, dim=D, index=I))
        -> select(mul(u, v), dim=D, index=I)

It is the ``select`` analogue of the layout family's
``transpose_pull_mul``.  The shared ``dim``/``index`` attribute
metavariables make the *matcher* enforce the dim/index precondition —
and a ``dim-eq-attr`` cond guards the residual: ``u``/``v`` must agree
along ``D``, since broadcasting ``mul(u, v)`` along the *selected*
axis changes the result (the view-oracle's counterexample:
``u=(4,), v=(2,4)``).

Covered surface:

* registration/tagging;
* a term-level match/instantiate round-trip;
* numeric soundness on real fp64 tensors (lhs == rhs);
* the matcher precondition — a mismatched ``dim`` or ``index`` declines;
* an end-to-end run on a real ``SelectiveSSM`` export: the law fires,
  the extracted cost drops, the certificate replays, and the lowered
  before/after modules agree under ``sink.verify``;
* the law firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.laws import (
    DEFAULT,
    SELECT_MUL,
    SIMPLIFICATION_RULES,
    tags,
)
from catopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink
from catopt_torch.backend import TorchBackend
from catopt_torch.models.ssm import SelectiveSSM
from catopt_torch.torch_bridge import export_to_ir

_SINK = TorchSink()


def _cost_fn():
    """The pipeline's selection model (roofline + per-dispatch term)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), _SINK.supported_ops
    )


def _saturate(term, rules, cost_fn, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root, eg.extract_best(root, cost_fn)


def _params_of(term):
    """Every ``Param`` leaf of *term* (DAG-aware)."""
    seen: set[int] = set()
    out: dict[str, Param] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term, ir):
    """Wrap *term* as an ``IR`` reusing *ir*'s inputs."""
    return IR(
        root=term,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=_params_of(term),
    )


def _select_lhs(dim, index, u=None, v=None):
    u = u if u is not None else Var("u", TensorType((4, 3, 8)))
    v = v if v is not None else Var("v", TensorType((4, 3, 8)))
    return Op.make(
        "mul",
        Op.make("select", u, dim=dim, index=index),
        Op.make("select", v, dim=dim, index=index),
    )


# ---------------------------------------------------------------------------
#  Registration / tagging
# ---------------------------------------------------------------------------


def test_select_mul_registered_and_tagged():
    """``select_mul`` is a shipped simplification, in the default set."""
    assert SELECT_MUL in SIMPLIFICATION_RULES
    assert SELECT_MUL.tags == {tags.SIMPLIFICATION}
    assert "select_mul" in {r.name for r in DEFAULT}


# ---------------------------------------------------------------------------
#  Match / instantiate round-trip
# ---------------------------------------------------------------------------


def test_select_mul_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    the shared ``dim``/``index`` metavariables bind consistently."""
    u = Var("u", TensorType((4, 3, 8)))
    v = Var("v", TensorType((4, 3, 8)))
    subst = {"u": u, "v": v, "$attr:D": 1, "$attr:I": 2}
    assert pattern_metavars(SELECT_MUL.lhs) == {
        "u",
        "v",
        "$attr:D",
        "$attr:I",
    }
    term = instantiate_pattern(SELECT_MUL.lhs, subst)
    found = match_pattern(SELECT_MUL.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SELECT_MUL.lhs, found) == term
    # the RHS instantiates to the fused select-of-product form.
    rhs = instantiate_pattern(SELECT_MUL.rhs, subst)
    assert rhs == Op.make(
        "select", Op.make("mul", u, v), dim=1, index=2
    )


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_select_mul_sound_fp64():
    """``sel(u) ⊙ sel(v) == sel(u ⊙ v)`` on real fp64 tensors."""
    u = torch.randn(4, 3, 8, dtype=torch.float64)
    v = torch.randn(4, 3, 8, dtype=torch.float64)
    lhs = u.select(1, 2) * v.select(1, 2)
    rhs = (u * v).select(1, 2)
    assert torch.equal(lhs, rhs)


def test_select_mul_broadcast_axis_guard():
    """``dim-eq-attr`` declines when ``u``/``v`` disagree along ``D``.

    Broadcasting ``mul(u, v)`` along the *selected* axis is the false
    region: ``u=(4,), v=(2,4)`` gives ``u[0]·v[0]`` (scalar×vector) vs
    ``(u·v)[0] = u·v[0]`` (elementwise) — genuinely different tensors.
    Broadcasting on any *other* axis stays allowed.
    """
    from catopt_core.laws.cond import eval_cond

    unsafe = {
        "u": Var("u", TensorType((4,))),
        "v": Var("v", TensorType((2, 4))),
        "$attr:D": 0,
        "$attr:I": 0,
    }
    assert not eval_cond(SELECT_MUL.cond, unsafe)
    assert not SELECT_MUL.check(unsafe)
    # other-axis broadcast (dim 1: 1 vs 4) — still safe.
    other_axis = {
        "u": Var("u", TensorType((2, 1))),
        "v": Var("v", TensorType((2, 4))),
        "$attr:D": 0,
        "$attr:I": 0,
    }
    assert SELECT_MUL.check(other_axis)
    # the numeric counterexample itself.
    ut = torch.arange(4.0)
    vt = torch.arange(8.0).reshape(2, 4)
    assert not torch.equal(ut[0] * vt[0], (ut * vt)[0])
    # and on the safe shapes the sides really are equal.
    u2 = torch.randn(2, 4, dtype=torch.float64)
    v2 = torch.randn(2, 4, dtype=torch.float64)
    assert torch.equal(u2[0] * v2[0], (u2 * v2)[0])


# ---------------------------------------------------------------------------
#  Firing + the matcher precondition
# ---------------------------------------------------------------------------


def test_select_mul_fires_and_rhs_is_member():
    eg, root, _best = _saturate(
        _select_lhs(1, 2), [SELECT_MUL], _cost_fn()
    )
    assert eg.rule_fires.get("select_mul", 0) > 0
    u = Var("u", TensorType((4, 3, 8)))
    v = Var("v", TensorType((4, 3, 8)))
    rhs = Op.make("select", Op.make("mul", u, v), dim=1, index=2)
    assert list(eg.matches(rhs, eg.find(root)))


def test_select_mul_declines_on_dim_mismatch():
    """Different ``dim`` on the two selects — the matcher vetoes."""
    u = Var("u", TensorType((4, 3, 8)))
    v = Var("v", TensorType((4, 3, 8)))
    term = Op.make(
        "mul",
        Op.make("select", u, dim=1, index=0),
        Op.make("select", v, dim=2, index=0),
    )
    eg, _root, _best = _saturate(term, [SELECT_MUL], _cost_fn())
    assert eg.rule_fires.get("select_mul", 0) == 0


def test_select_mul_declines_on_index_mismatch():
    """Different ``index`` on the two selects — the matcher vetoes."""
    u = Var("u", TensorType((4, 3, 8)))
    v = Var("v", TensorType((4, 3, 8)))
    term = Op.make(
        "mul",
        Op.make("select", u, dim=1, index=0),
        Op.make("select", v, dim=1, index=1),
    )
    eg, _root, _best = _saturate(term, [SELECT_MUL], _cost_fn())
    assert eg.rule_fires.get("select_mul", 0) == 0


# ---------------------------------------------------------------------------
#  End-to-end on a real SSM export
# ---------------------------------------------------------------------------


def test_select_mul_end_to_end_ssm():
    """The law fires on a real ``SelectiveSSM`` export, the extracted
    cost drops, the certificate replays, and the lowered before/after
    modules agree under ``sink.verify``."""
    torch.manual_seed(0)
    model = SelectiveSSM(8, 8, 4).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    ir, leaves = export_to_ir(model, x)
    cost_fn = _cost_fn()

    without = [r for r in DEFAULT if r.name != "select_mul"]
    with_law = list(DEFAULT)

    eg0, _r0, best0 = _saturate(ir.root, without, cost_fn)
    eg1, _r1, best1 = _saturate(ir.root, with_law, cost_fn)

    assert eg1.rule_fires.get("select_mul", 0) > 0
    assert eg0.rule_fires.get("select_mul", 0) == 0
    c0, c1 = dag_cost(best0, cost_fn), dag_cost(best1, cost_fn)
    assert c1 < c0
    assert best1 != best0

    # the certificate replays on the real export.
    cert = eg1.certificate(ir.root, best1, cost_fn=cost_fn)
    verify_certificate(ir.root, cert)

    # the lowered before/after modules are numerically equal.
    m0 = _lower_extracted(best0, _ir_of(best0, ir), leaves, _SINK)
    m1 = _lower_extracted(best1, _ir_of(best1, ir), leaves, _SINK)
    assert _SINK.verify(m0, m1, (x,), rtol=1e-4).passed


def test_select_mul_fires_in_default_pipeline():
    """The public default pipeline sees the law fire and verifies."""
    torch.manual_seed(0)
    model = SelectiveSSM(8, 8, 4).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("select_mul", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed
