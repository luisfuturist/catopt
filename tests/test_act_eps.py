# ruff: noqa: RUF003
"""Activation-space ε: certified bounded-error rewrites on ACTIVATIONS.

Where :mod:`catopt.eps` certifies *weight* substitutions (compile-time
constants), :mod:`catopt.act_eps` certifies activation substitutions —
the bound is a runtime contract: ``aquant`` computes the per-call scale
``s(x) = |x|max/levels`` and the witnessed rewrite carries the
calibrated worst case ``(|x|max_cal/levels)/2·√n``.

These tests pin the mechanics on a 2-layer model: the dynamic
``f(x) → f(adequant(aquant(x)))`` offers exist at activation edges,
the executed member honours its certified bound end-to-end, and the
certificate is non-exact, replayable, and verifiable.
"""

import math

import torch
import torch.nn as nn

from catopt.act_eps import (
    act_low_rank,
    act_quant,
    calibrate,
    extract_with_offers,
)
from catopt.cost import count_cost
from catopt.egraph import EGraph, verify_certificate
from catopt.eps import model_bound
from catopt.ir import IR, Op, op_repr
from catopt.rules import all_rules
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _two_layer(seed=0):
    """linear → relu → linear, fp64: the smallest model with a real
    intermediate activation."""
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 64, bias=False)
            s.l2 = nn.Linear(64, 32, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    return M().eval().double()


def _build(m, x, iters=3):
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=iters)
    return ir, src, eg, root


def _has_op(term, name):
    return isinstance(term, Op) and (
        term.op == name or any(_has_op(a, name) for a in term.args)
    )


def test_calibrate_collects_per_site_absmax():
    """calibrate(ir, x, src) keys activation sites by op_repr — the key
    act_quant uses to bind e-classes to calibrated bounds."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    calib = calibrate(ir, x, src)
    assert calib["n_sites"] == 3  # linear1, relu, linear2 outputs
    assert calib["global"] == max(calib["per_site"].values())

    # every op subterm of the root recorded
    def ops(t):
        if not isinstance(t, Op):
            return set()
        return {op_repr(t)} | set().union(*(ops(a) for a in t.args))

    assert ops(ir.root) <= set(calib["per_site"])
    # and the values are the true absmaxes
    with torch.no_grad():
        a1 = torch.nn.functional.linear(x, src["p_l1_weight"])
        key = op_repr(ir.root.args[0].args[0])
        assert calib["per_site"][key] == float(a1.abs().max())


def test_act_quant_offers_at_activation_edges():
    """Every consumer edge of an activation class gets a
    quantize/dequantize member; Var inputs and Param leaves get none."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, _root = _build(m, x)
    calib = calibrate(ir, x, src)
    offers = act_quant(eg, src, bits=8, calib=calib)
    # two activation edges: relu's input (l1 out), l2's input (relu out)
    assert len(offers) == 2
    wrapped = {eg.find(o["quant_eid"]) for o in offers}
    assert len(wrapped) == 2
    for o in offers:
        assert o["bound"] > 0 and math.isfinite(o["bound"])
        # the offered member sits in the consumer's e-class
        assert o["member"] in eg.get_class(o["site_eid"]).nodes
    assert any(r.startswith("act_q8#") for r in eg._rule_objs)
    # no offer wraps the input Var or a Param leaf class
    for o in offers:
        rep = eg._oldest_term(eg.find(o["quant_eid"]))
        assert isinstance(rep, Op)


def test_act_quant_executes_within_certified_bound():
    """End-to-end: extract a term carrying both quant members, run it,
    and the measured output error stays under the certified model bound."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, root = _build(m, x)
    calib = calibrate(ir, x, src)
    offers = act_quant(eg, src, bits=8, calib=calib)

    term = extract_with_offers(eg, root, offers, cost_fn=count_cost)
    assert term is not None
    assert _has_op(term, "aquant") and _has_op(term, "adequant")

    mod = ir_to_torch_module(
        IR(root=term, inputs=ir.inputs, params=ir.params), src
    )
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    assert err > 0  # a real, nonzero approximation
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert not cert.exact
    assert cert.error_bound > 0
    # certificate aggregates the site bounds of the forced members —
    # both activation edges were quantized, both steps contribute
    site_bounds = sorted(o["bound"] for o in offers)
    step_bounds = sorted(
        cert.rules[s.rule].error_bound
        for s in cert.steps
        if cert.rules.get(s.rule) and cert.rules[s.rule].error_bound
    )
    assert step_bounds == site_bounds
    # whole-model bound: site bounds × Lipschitz path sensitivities
    mb = model_bound(
        term,
        cert,
        src,
        input_norm=float(torch.linalg.norm(x, dim=-1).max()),
    )
    assert mb["bound"] != float("inf")
    assert mb["bound"] >= err


def test_act_quant_certificate_replayable_and_verified():
    """The bounded derivation replays standalone: verify_certificate
    reconstructs the extracted term step by step."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, root = _build(m, x)
    offers = act_quant(eg, src, bits=8, calib=calibrate(ir, x, src))
    term = extract_with_offers(eg, root, offers, cost_fn=count_cost)
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert cert.replayable
    assert "act_q8" in " ".join(cert.rules_used)
    out = verify_certificate(ir.root, cert, strict=True)
    assert op_repr(out) == op_repr(term)
    # the bound formula is recorded in the witness law text
    laws = " ".join(r.law for r in cert.rules.values() if r.error_bound)
    assert "s(x)/2" in laws and "runtime" in laws


def test_act_quant_uncalibrated_is_honestly_unbounded():
    """Without calibration data the offer still lands — carrying
    error_bound=inf rather than a fabricated constant."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, root = _build(m, x)
    offers = act_quant(eg, src, bits=8)  # no calib
    assert offers
    assert all(o["bound"] == float("inf") for o in offers)
    term = extract_with_offers(eg, root, offers, cost_fn=count_cost)
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert cert.error_bound == float("inf")
    assert not cert.exact


def test_act_quant_bounded_extraction_prefers_exact():
    """Under a zero ε budget, bounded extraction stays on the exact
    member — the quant offers are bound-carrying and get banned."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, root = _build(m, x)
    act_quant(eg, src, bits=8, calib=calibrate(ir, x, src))
    term = eg.extract_best_bounded(
        root, count_cost, max_error=0.0, src_term=ir.root
    )
    assert term is not None
    assert not _has_op(term, "aquant")
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert cert.exact


def test_act_quant_shared_activation_wraps_every_read():
    """One activation read by two consumers: both consumer classes get
    offers, and both wrap the same shared (q, s) pair — an int8 buffer
    read twice, not two quantizers."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 64, bias=False)
            s.l2 = nn.Linear(64, 64, bias=False)

        def forward(s, x):
            y = s.l1(x)
            return s.l2(y) + y

    m = M().eval().double()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, _root = _build(m, x)
    offers = act_quant(eg, src, bits=8, calib=calibrate(ir, x, src))
    by_site = {}
    for o in offers:
        by_site.setdefault(eg.find(o["quant_eid"]), []).append(o)
    # the l1 output class is wrapped at (at least) the l2 edge and the
    # add edge — two offers on one activation class
    multi = [v for v in by_site.values() if len(v) >= 2]
    assert multi, "expected ≥2 consumer offers on the shared activation"
    # both members reference the same adequant(aquant(·)) subtree —
    # the wrapped positions resolve to one shared adequant class
    for grp in multi:
        dq_ids = {eg.find(o["member"].children[o["pos"]]) for o in grp}
        assert len(dq_ids) == 1


def test_act_low_rank_offer_and_bound():
    """Stretch: activation random-projection offers carry the certified
    contractive bound ‖Δx‖ ≤ ‖x‖ ≤ √n·|x|max, with the measured
    residual reported alongside."""
    m = _two_layer()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src, eg, root = _build(m, x)
    calib = calibrate(ir, x, src, keep_tensors=True)
    offers = act_low_rank(eg, src, rank=8, calib=calib, seed=0)
    assert offers
    assert any(n.startswith("act_lrP_") for n in src)
    o = offers[0]
    assert o["bound"] > 0
    term = extract_with_offers(eg, root, offers, cost_fn=count_cost)
    assert term is not None and _has_op(term, "matmul")
    mod = ir_to_torch_module(
        IR(root=term, inputs=ir.inputs, params=ir.params), src
    )
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert not cert.exact and cert.error_bound > 0
    # measured residual recorded in the law text
    laws = " ".join(r.law for r in cert.rules.values() if r.error_bound)
    assert "measured residual" in laws
    mb = model_bound(
        term,
        cert,
        src,
        input_norm=float(torch.linalg.norm(x, dim=-1).max()),
    )
    assert mb["bound"] >= err or mb["bound"] == float("inf")
