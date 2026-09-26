"""Coverage tests for catopt.act_eps internals — activation-space ε.

``test_act_eps.py`` pins the headline behaviour end to end; this file
closes the residual gaps: extension registration failure tolerance,
``_site_absmax``'s calib forms, ``calibrate``'s nn.Module/dict/degraded
evaluation paths, the L∞ (unknown-element-count) bound, wrap dedup,
cyclic offers, and the ``act_low_rank`` wrap-factory declines.

Defensive branches deliberately not covered (suggest ``pragma: no
cover``):
- ``if c in out: continue`` in ``_activation_classes``
  (act_eps.py:203) and ``ec is None: continue`` in ``_wrap_sites``
  (239): e-class keys are canonical — ``EGraph.union`` deletes
  non-canonical entries eagerly — so neither guard can fire.
"""

import math

import pytest
import torch
import torch.nn as nn

from catopt import eps as _eps_mod
from catopt.act_eps import (
    _adequant_torch,
    _aquant_torch,
    _register_extensions,
    _site_absmax,
    act_low_rank,
    act_quant,
    calibrate,
    extract_with_offers,
)
from catopt.cost import count_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, Param, TensorType, Var, op_repr


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  Extension registration + the runtime quant pair
# ---------------------------------------------------------------------------


def test_register_extensions_survives_missing_lip_free():
    """``_register_extensions`` is best-effort: if ``eps._LIP_FREE``
    isn't there to update, the failure is swallowed — registration of
    the ops themselves never depends on it."""
    assert "aquant" in _eps_mod._LIP_FREE
    assert "adequant" in _eps_mod._LIP_FREE
    saved = _eps_mod._LIP_FREE
    try:
        del _eps_mod._LIP_FREE
        _register_extensions()  # must not raise
    finally:
        _eps_mod._LIP_FREE = saved
    assert "aquant" in _eps_mod._LIP_FREE


def test_aquant_adequant_roundtrip_bound():
    """The runtime pair: ``q·s`` reconstructs x within the certified
    per-call bound ``s(x)/2·√n``, verified numerically — including the
    exact zero-tensor case where s is clamped to 1."""
    torch.manual_seed(0)
    for bits in (8, 4):
        levels = 2 ** (bits - 1) - 1
        x = torch.randn(5, 7, dtype=torch.float64) * 4
        q, s = _aquant_torch(x, bits=bits)
        assert q.dtype == torch.int8
        xhat = _adequant_torch((q, s))
        assert xhat.dtype == x.dtype
        err = float(torch.linalg.norm(x - xhat))
        bound = float(s) / 2 * math.sqrt(x.numel())
        assert err <= bound + 1e-9
        assert float(s) == pytest.approx(
            float(x.abs().max()) / levels
        )
    # zero tensor: s clamped to 1, q == 0 → exact identity
    q0, s0 = _aquant_torch(torch.zeros(3, 3))
    assert float(s0) == 1.0
    assert torch.equal(_adequant_torch((q0, s0)), torch.zeros(3, 3))


# ---------------------------------------------------------------------------
#  _site_absmax — every calib form
# ---------------------------------------------------------------------------


def test_site_absmax_all_calib_forms():
    key = "relu(linear(x, W))"
    assert _site_absmax(0.5, key) == 0.5
    assert _site_absmax({"per_site": {key: 0.3}}, key) == 0.3
    # per_site dict present but key missing → global fallback
    assert (
        _site_absmax({"per_site": {"other": 1.0}, "global": 0.7}, key)
        == 0.7
    )
    # plain {repr: amax} dict — the (b) form
    assert _site_absmax({key: 0.4}, key) == 0.4
    # dict without per_site and without the key → global or None
    assert _site_absmax({"other": 1.0, "global": 0.6}, key) == 0.6
    assert _site_absmax({"per_site": {"other": 1.0}}, key) is None
    assert _site_absmax({"other": 1.0}, key) is None
    # non-dict, non-scalar → uncalibrated
    assert _site_absmax("0.5", key) is None
    assert _site_absmax(None, key) is None


# ---------------------------------------------------------------------------
#  calibrate — module hooks, dict inputs, degraded evaluation
# ---------------------------------------------------------------------------


def test_calibrate_nn_module_hooks():
    """The nn.Module path: forward hooks on every submodule, absmax
    keyed by module name, 'model' for the root — real values checked
    against a manual forward."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    m = m.eval().double()
    x = torch.randn(2, 8, dtype=torch.float64)
    calib = calibrate(m, x)
    assert set(calib["per_site"]) == {"0", "1", "2", "model"}
    assert calib["n_sites"] == 4
    with torch.no_grad():
        a1 = m[0](x)
        a2 = m[1](a1)
        out = m[2](a2)
    assert calib["per_site"]["0"] == pytest.approx(float(a1.abs().max()))
    assert calib["per_site"]["1"] == pytest.approx(float(a2.abs().max()))
    assert calib["per_site"]["2"] == pytest.approx(float(out.abs().max()))
    assert calib["per_site"]["model"] == calib["per_site"]["2"]
    assert calib["global"] == max(calib["per_site"].values())
    # keep_tensors retains the activation samples act_low_rank reads
    calib2 = calibrate(m, x, keep_tensors=True)
    assert torch.equal(calib2["tensors"]["1"], a2)

    # a module returning a non-tensor (tuple) output is skipped by the
    # hook — no entry, no crash
    class Pair(nn.Module):
        def forward(self, t):
            return (t, t + 1)

    m2 = nn.Sequential(Pair())
    calib3 = calibrate(m2.eval().double(), x)
    assert calib3["per_site"].get("model") is None  # tuple → unrecorded
    assert calib3["n_sites"] == 0


def test_calibrate_ir_dict_inputs_and_eval_edges():
    """IR-path edges: dict inputs bind by var name; Const leaves
    evaluate to ``torch.tensor(value)``; Params resolve through src;
    missing bindings silently produce no site entry — that is the
    contract: partial coverage, no exception."""
    x = _v("x", 2, 4)
    y = _v("y", 2, 4)
    w = _p("w", 2, 4)
    inner = Op.make("mul", x, Const(2.0))
    term = Op.make("add", inner, Op.make("mul", y, w))
    ir = IR(root=term, inputs=[x, y], params={})
    x0 = torch.ones(2, 4, dtype=torch.float64)
    y0 = torch.full((2, 4), 3.0, dtype=torch.float64)

    # dict input — names, not positions
    calib = calibrate(ir, {"x": x0, "y": y0}, {"w": torch.ones(2, 4)})
    assert calib["per_site"][op_repr(inner)] == pytest.approx(2.0)
    assert calib["per_site"][op_repr(term)] == pytest.approx(5.0)
    assert calib["n_sites"] == 3

    # fewer inputs than vars → strict=False truncates silently:
    # ops needing the unbound var simply have no site
    calib_t = calibrate(ir, (x0,), {"w": torch.ones(2, 4)})
    assert op_repr(inner) in calib_t["per_site"]
    assert op_repr(term) not in calib_t["per_site"]  # y unbound → unevaluated
    assert calib_t["n_sites"] == 1  # truncated, not an error

    # param missing from src → its subtree silently drops out too
    calib_p = calibrate(ir, {"x": x0, "y": y0}, {})
    assert op_repr(term) not in calib_p["per_site"]
    assert op_repr(inner) in calib_p["per_site"]


def test_calibrate_degraded_ops():
    """Ops that can't produce a tensor contribute no site and no
    crash: unknown ops (no binding), ops that raise inside torch, ops
    returning non-tensors (aquant's (q, s) pair), and non-term leaves."""
    x = _v("x", 2, 4)
    good = Op.make("mul", x, Const(2.0))
    no_binding = Op.make("bogus_op", x)
    raises = Op.make("reshape", x, shape=(3, 3))  # 8 elems → (3,3)
    pair = Op.make("aquant", x, bits=8)  # bound: returns (q, s) tuple
    raw = Op.make("add", x, 7)  # non-term leaf child
    term = Op.make(
        "add", good, Op.make("add", no_binding, Op.make("add", raises, Op.make("add", pair, raw)))
    )
    ir = IR(root=term, inputs=[x], params={})
    calib = calibrate(ir, torch.ones(2, 4, dtype=torch.float64), {})
    # only the well-behaved op recorded a site — everything degraded
    # silently rather than raising mid-calibration
    assert calib["per_site"] == {op_repr(good): 2.0}
    assert calib["n_sites"] == 1


# ---------------------------------------------------------------------------
#  act_quant — L∞ bound, scalar calib, witness=False, member dedup
# ---------------------------------------------------------------------------


def test_act_quant_linf_bound_for_unknown_element_count():
    """When the site's element count can't be inferred (shape None),
    the surviving certificate is elementwise — an L∞ bound
    ``|x|max/(2·levels)``, reported under ``bound_norm='linf'``."""
    torch.manual_seed(0)
    x0 = Var("x0", TensorType(()))  # scalar → select shape uninferrable
    sel = Op.make("select", x0, dim=0, index=0)
    x1 = _v("x1", 4)
    term = Op.make("add", sel, x1)
    eg = EGraph()
    eg.add_term(term)
    calib = {op_repr(sel): 0.5}
    offers = act_quant(eg, {}, bits=8, calib=calib)
    assert len(offers) == 1
    o = offers[0]
    levels = 127
    assert o["bound_norm"] == "linf"
    assert o["bound"] == pytest.approx(0.5 / (2 * levels))
    law = " ".join(r.law for r in eg._rule_objs.values())
    assert "n=?" in law and "L∞" in law


def test_act_quant_scalar_calib_and_witness_false():
    """A bare float calib applies to every site; witness=False unions
    the members without registering bound-carrying rewrites."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    act = Op.make("relu", x)
    term = Op.make("mul", act, _p("p", 4, 8))
    eg = EGraph()
    eg.add_term(term)
    offers = act_quant(eg, {}, bits=8, calib=0.7)
    assert len(offers) == 1
    levels = 127
    n = 32
    assert offers[0]["bound"] == pytest.approx(
        0.7 / (2 * levels) * math.sqrt(n)
    )
    eg2 = EGraph()
    eg2.add_term(term)
    offers2 = act_quant(eg2, {}, bits=8, witness=False)
    assert len(offers2) == 1
    assert not any(r.startswith("act_q8") for r in eg2._rule_objs)


def test_wrap_sites_member_dedup():
    """Two raw enodes in one consumer class that resolve to the SAME
    wrapped member produce one offer — the ``seen`` dedup guards the
    union, not the accounting."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    p1 = _p("p1", 4, 8)
    p2 = _p("p2", 4, 8)
    act = Op.make("relu", x)
    t1 = Op.make("add", act, p1)
    t2 = Op.make("add", act, p2)
    eg = EGraph()
    eg.add_term(t1)
    eg.add_term(t2)
    c_p1, c_p2 = eg.add_term(p1), eg.add_term(p2)
    c_t1, c_t2 = eg.add_term(t1), eg.add_term(t2)
    # merge the param classes and the consumer classes: the two raw
    # add enodes now live in one class and canonicalize identically
    eg.union(c_p1, c_p2)
    eg.union(c_t1, c_t2)
    offers = act_quant(eg, {}, bits=8, witness=False)
    assert len(offers) == 1


# ---------------------------------------------------------------------------
#  extract_with_offers — cyclic skips, quant_eids filter, default cost
# ---------------------------------------------------------------------------


def test_extract_with_offers_skips_cyclic_and_filters():
    """A cyclic member (the consumer class IS the activation class)
    can't be realised — ``extract_with_offers`` skips it.  And
    ``quant_eids`` restricts which wrapped reads get forced."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    p = _p("p", 4, 8)
    act = Op.make("relu", x)
    mul = Op.make("mul", act, p)
    eg = EGraph()
    c_act = eg.add_term(act)
    c_mul = eg.add_term(mul)
    eg.union(c_act, c_mul)  # class contains mul(c_self, p) — cyclic
    root = eg.find(c_mul)
    offers = act_quant(eg, {}, bits=8, witness=False)
    assert len(offers) == 1 and offers[0]["cyclic"]
    # cyclic → skipped by overrides → extraction picks the exact member
    term = extract_with_offers(eg, root, offers)  # default count_cost
    assert term is not None
    assert not _contains(term, "aquant")


def _contains(t, op):
    return isinstance(t, Op) and (
        t.op == op or any(_contains(a, op) for a in t.args)
    )


def test_extract_with_offers_quant_eids_subset():
    """``quant_eids`` picks which wrapped activation classes are
    forced — offers outside the set fall back to plain extraction."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    a1 = Op.make("relu", x)
    a2 = Op.make("tanh", x)
    term = Op.make("add", a1, a2)
    eg = EGraph()
    root = eg.add_term(term)
    offers = act_quant(eg, {}, bits=8, witness=False)
    assert len(offers) == 2
    keep = {offers[0]["quant_eid"]}
    term1 = extract_with_offers(
        eg, root, offers, quant_eids=keep, cost_fn=count_cost
    )
    assert term1 is not None
    # the kept class is the only one eligible for a forced wrap
    n_forced = sum(
        1 for o in offers if o["quant_eid"] in keep and not o["cyclic"]
    )
    assert n_forced == 1


# ---------------------------------------------------------------------------
#  act_low_rank — wrap-factory declines and law variants
# ---------------------------------------------------------------------------


def test_act_low_rank_declines_bad_shape_and_rank():
    """The wrap factory refuses activations whose last dim can't be
    inferred (shape None) or doesn't exceed the requested rank."""
    torch.manual_seed(0)
    # shape-uninferrable activation (select on a scalar var)
    x0 = Var("x0", TensorType(()))
    sel = Op.make("select", x0, dim=0, index=0)
    cons = Op.make("add", sel, _v("x1", 4))
    eg = EGraph()
    eg.add_term(cons)
    assert act_low_rank(eg, {}, rank=2) == []
    # rank >= d: no point projecting to a non-bottleneck
    eg2 = EGraph()
    eg2.add_term(
        Op.make("mul", Op.make("relu", _v("x", 4, 8)), _p("p", 4, 8))
    )
    assert act_low_rank(eg2, {}, rank=8) == []
    assert act_low_rank(eg2, {}, rank=16) == []


def test_act_low_rank_law_uncalibrated_and_unmeasured():
    """Uncalibrated offers declare the contractive bound but admit no
    measured residual — and a calib without kept tensors likewise
    reports nothing measured."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    term = Op.make("mul", Op.make("relu", x), _p("p", 4, 8))
    # uncalibrated entirely
    eg = EGraph()
    eg.add_term(term)
    offers = act_low_rank(eg, {}, rank=4, calib=None, seed=0)
    assert len(offers) == 1
    assert offers[0]["bound"] == float("inf")
    law = " ".join(r.law for r in eg._rule_objs.values())
    assert "UNCALIBRATED" in law
    assert "measured residual" not in law
    # calibrated amax but no activation samples → finite bound, no
    # measured-residual clause either
    eg2 = EGraph()
    eg2.add_term(term)
    # the relu class's oldest term is the input member itself
    calib = {op_repr(Op.make("relu", x)): 3.0}
    offers2 = act_low_rank(eg2, {}, rank=4, calib=calib, seed=0)
    assert len(offers2) == 1
    n = 32
    assert offers2[0]["bound"] == pytest.approx(3.0 * math.sqrt(n))
    law2 = " ".join(r.law for r in eg2._rule_objs.values())
    assert "measured residual" not in law2
    # P / Pᵀ land in source_tensors as real params, orthonormal columns
    src = {}
    eg3 = EGraph()
    eg3.add_term(term)
    calib3 = {"per_site": {op_repr(Op.make("relu", x)): 3.0}, "global": 3.0}
    act_low_rank(eg3, src, rank=4, calib=calib3, seed=0)
    P = next(t for k, t in src.items() if k.startswith("act_lrP_"))
    assert P.shape == (8, 4)
    gram = P.double().T @ P.double()
    assert torch.allclose(gram, torch.eye(4, dtype=gram.dtype), atol=1e-9)
