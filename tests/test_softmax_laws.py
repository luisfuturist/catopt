"""Tests for the ``softmax_fold`` law — the manual-softmax fold
(:data:`catopt_core.laws.tensor.SOFTMAX_FOLD`).

The SECOND machine-discovered law admitted to the library (after
``select_mul``) — proposed by the law pipeline's pattern-recognition
pass over the model census:

    div(exp(u), sum(exp(u), dim=RD, keepdim=RK))
        -> softmax(u, dim=SD)

Unlike ``select_mul`` the precondition is NOT structural: ``keepdim``
must be True (a dropped dim broadcasts wrongly — or not at all —
against the numerator) and the reduce must cover exactly one axis
(softmax has no multi-axis image).  ``check`` carries that side
condition; ``derive`` translates the sum's ``dim`` tuple ``(-1,)``
into softmax's scalar ``dim=-1``.

Covered surface:

* registration/tagging;
* a term-level match/instantiate round-trip, derive included;
* the check/derive hooks unit-tested branch by branch;
* numeric soundness on real fp64 tensors (lhs == rhs);
* firing + RHS-is-member, on both the tuple- and int-``dim`` spellings;
* the side-condition declines — ``keepdim=False``, a multi-axis sum —
  and the matcher precondition decline (the numerator and denominator
  ``exp`` terms differ);
* an end-to-end run on a real ``ManualSoftmaxAttention`` export: the
  law fires, the extracted cost drops ~19%, the certificate replays,
  and the lowered before/after modules agree under ``sink.verify``;
* the law firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.laws import (
    DEFAULT,
    SIMPLIFICATION_RULES,
    SOFTMAX_FOLD,
    tags,
)
from catopt_core.laws.tensor import (
    _check_sum_keepdim,
    _derive_softmax_dim,
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
from catopt_torch.models import ManualSoftmaxAttention
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


def _softmax_lhs(dim, keepdim, u=None, v=None):
    """``div(exp(u), sum(exp(v or u), dim, keepdim))`` — *v* defaults
    to *u* (a legal LHS instance; pass a different var to build the
    declined operand-mismatch shape)."""
    u = u if u is not None else Var("u", TensorType((4, 8)))
    v = v if v is not None else u
    return Op.make(
        "div",
        Op.make("exp", u),
        Op.make("sum", Op.make("exp", v), dim=dim, keepdim=keepdim),
    )


# ---------------------------------------------------------------------------
#  Registration / tagging
# ---------------------------------------------------------------------------


def test_softmax_fold_registered_and_tagged():
    """``softmax_fold`` is a shipped simplification, in the default set."""
    assert SOFTMAX_FOLD in SIMPLIFICATION_RULES
    assert SOFTMAX_FOLD.tags == {tags.SIMPLIFICATION}
    assert "softmax_fold" in {r.name for r in DEFAULT}


# ---------------------------------------------------------------------------
#  Match / instantiate round-trip (derive supplies the RHS-only attr)
# ---------------------------------------------------------------------------


def test_softmax_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    ``derive`` produces the ``$attr:SD`` the RHS needs."""
    u = Var("u", TensorType((4, 8)))
    subst = {"u": u, "$attr:RD": (-1,), "$attr:RK": True}
    assert pattern_metavars(SOFTMAX_FOLD.lhs) == {
        "u",
        "$attr:RD",
        "$attr:RK",
    }
    term = instantiate_pattern(SOFTMAX_FOLD.lhs, subst)
    found = match_pattern(SOFTMAX_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SOFTMAX_FOLD.lhs, found) == term
    # the RHS instantiates to the folded softmax — the sum's dim tuple
    # unwrapped to the scalar softmax dim.
    rhs_subst = {**found, **SOFTMAX_FOLD.derive(found)}
    rhs = instantiate_pattern(SOFTMAX_FOLD.rhs, rhs_subst)
    assert rhs == Op.make("softmax", u, dim=-1)


# ---------------------------------------------------------------------------
#  check / derive hooks — branch by branch
# ---------------------------------------------------------------------------


def test_check_sum_keepdim_accepts_legal_shapes():
    """keepdim=True with a single reduce axis — tuple or int spelling."""
    assert _check_sum_keepdim({"$attr:RK": True, "$attr:RD": (-1,)})
    assert _check_sum_keepdim({"$attr:RK": True, "$attr:RD": 1})


def test_check_sum_keepdim_declines_illegal_shapes():
    """No keepdim, a multi-axis reduce, or a dimless (full) reduce."""
    assert not _check_sum_keepdim({"$attr:RK": False, "$attr:RD": (-1,)})
    assert not _check_sum_keepdim(
        {"$attr:RK": True, "$attr:RD": (-2, -1)}
    )
    assert not _check_sum_keepdim({"$attr:RK": True, "$attr:RD": None})


def test_derive_softmax_dim_unwraps_the_tuple():
    """The sum's ``dim`` tuple becomes softmax's scalar ``dim``; an int
    passes through unchanged."""
    assert _derive_softmax_dim({"$attr:RD": (-1,)}) == {"$attr:SD": -1}
    assert _derive_softmax_dim({"$attr:RD": 1}) == {"$attr:SD": 1}


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_softmax_fold_sound_fp64():
    """``exp(u) / Σ exp(u) == softmax(u)`` on real fp64 tensors — to
    ~1e-16, not bitwise: the kernel's max-subtraction rounds
    differently."""
    u = torch.randn(4, 8, dtype=torch.float64)
    e = torch.exp(u)
    lhs = e / e.sum(dim=-1, keepdim=True)
    rhs = torch.softmax(u, dim=-1)
    assert torch.allclose(lhs, rhs, atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
#  Firing + the side-condition declines
# ---------------------------------------------------------------------------


def test_softmax_fold_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        _softmax_lhs((-1,), True, u=u), [SOFTMAX_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("softmax_fold", 0) > 0
    rhs = Op.make("softmax", u, dim=-1)
    assert list(eg.matches(rhs, eg.find(root)))


def test_softmax_fold_fires_on_int_dim_spelling():
    """A hand-minted ``sum(u, dim=-1)`` (int, not tuple) still folds —
    ``check`` accepts a scalar axis and ``derive`` passes it through."""
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        _softmax_lhs(-1, True, u=u), [SOFTMAX_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("softmax_fold", 0) > 0
    rhs = Op.make("softmax", u, dim=-1)
    assert list(eg.matches(rhs, eg.find(root)))


def test_softmax_fold_declines_on_no_keepdim():
    """``keepdim=False`` — the sum's dropped dim cannot broadcast the
    softmax denominator; ``check`` vetoes."""
    eg, _root, _best = _saturate(
        _softmax_lhs((-1,), False), [SOFTMAX_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("softmax_fold", 0) == 0


def test_softmax_fold_declines_on_multi_dim_reduce():
    """A sum over two axes has no single-``dim`` softmax image —
    ``check`` vetoes."""
    eg, _root, _best = _saturate(
        _softmax_lhs((-2, -1), True), [SOFTMAX_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("softmax_fold", 0) == 0


def test_softmax_fold_declines_on_operand_mismatch():
    """``div(exp(u), sum(exp(v)))`` with u ≠ v — the shared ``u``
    metavariable makes the matcher veto, no check needed."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _softmax_lhs((-1,), True, u=u, v=v), [SOFTMAX_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("softmax_fold", 0) == 0


# ---------------------------------------------------------------------------
#  End-to-end on a real attention export
# ---------------------------------------------------------------------------


def test_softmax_fold_end_to_end_manual_attention():
    """The law fires on a real ``ManualSoftmaxAttention`` export, the
    extracted cost drops, the certificate replays, and the lowered
    before/after modules agree under ``sink.verify``."""
    torch.manual_seed(0)
    model = ManualSoftmaxAttention(8).eval().double()
    x = torch.randn(2, 8, 8, dtype=torch.float64)
    ir, leaves = export_to_ir(model, x)
    cost_fn = _cost_fn()

    without = [r for r in DEFAULT if r.name != "softmax_fold"]
    with_law = list(DEFAULT)

    eg0, _r0, best0 = _saturate(ir.root, without, cost_fn)
    eg1, _r1, best1 = _saturate(ir.root, with_law, cost_fn)

    assert eg1.rule_fires.get("softmax_fold", 0) > 0
    assert eg0.rule_fires.get("softmax_fold", 0) == 0
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


def test_softmax_fold_fires_in_default_pipeline():
    """The public default pipeline sees the law fire and verifies."""
    torch.manual_seed(0)
    model = ManualSoftmaxAttention(8).eval().double()
    x = torch.randn(2, 8, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("softmax_fold", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed
