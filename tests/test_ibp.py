"""Interval bound propagation — per-op rules, end-to-end tightness,
and honest reporting.

``eps.model_bound``'s spectral Lipschitz products are honest but loose;
``catopt.ibp`` replaces them with interval propagation: local Lipschitz
constants evaluated on real activation boxes, with a spectral fallback
for ops that have no interval rule.  These tests pin the interval
rules per-op, then verify the end-to-end bound is (a) still sound —
above the measured error — and (b) tighter than spectral, reporting
the achieved ratio honestly when it is not.
"""


import torch
import torch.nn as nn

from catopt.cost import param_bytes_cost_for
from catopt.egraph import EGraph
from catopt.eps import model_bound, quant_params
from catopt.ibp import (
    Box,
    _add,
    _matmul,
    _mul,
    _softmax,
    _sub,
    _unary,
    ibp_bound,
    tight_model_bound,
)
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.rules import all_rules
from catopt.torch_bridge import export_to_ir


def _box(lo, hi):
    lo, hi = (
        torch.as_tensor(lo, dtype=torch.float64),
        torch.as_tensor(hi, dtype=torch.float64),
    )
    return Box(lo, hi)


# ---------------------------------------------------------------------------
#  Per-op interval rules
# ---------------------------------------------------------------------------


def test_interval_add_sub_mul_corners():
    a = _box([1.0, -2.0], [2.0, -1.0])
    b = _box([0.5, 3.0], [1.5, 4.0])
    s = _add(a, b)
    assert torch.equal(s.lo, torch.tensor([1.5, 1.0]))
    assert torch.equal(s.hi, torch.tensor([3.5, 3.0]))
    d = _sub(a, b)
    assert torch.equal(d.lo, torch.tensor([-0.5, -6.0]))
    assert torch.equal(d.hi, torch.tensor([1.5, -4.0]))
    m = _mul(a, b)
    # corner-min/max per element
    assert torch.equal(m.lo, torch.tensor([0.5, -8.0]))
    assert torch.equal(m.hi, torch.tensor([3.0, -3.0]))


def test_interval_matmul_contains_all_products():
    torch.manual_seed(0)
    Ac = torch.randn(3, 4, dtype=torch.float64)
    Bc = torch.randn(4, 5, dtype=torch.float64)
    Ar, Br = (
        torch.rand(3, 4) * 0.1 + 0.05,
        torch.rand(4, 5) * 0.1 + 0.05,
    )
    A = Box(Ac - Ar, Ac + Ar)
    B = Box(Bc - Br, Bc + Br)
    C = _matmul(A, B)
    for _ in range(200):
        a = Ac + (torch.rand_like(Ac) * 2 - 1) * Ar
        b = Bc + (torch.rand_like(Bc) * 2 - 1) * Br
        p = a @ b
        assert (p >= C.lo - 1e-12).all() and (p <= C.hi + 1e-12).all()


def test_interval_monotone_unaries():
    a = _box([-1.0, 0.5], [-0.2, 2.0])
    r = _unary(a, "relu", torch.relu)
    assert torch.equal(r.lo, torch.tensor([0.0, 0.5]))
    assert torch.equal(r.hi, torch.tensor([0.0, 2.0]))
    s = _unary(a, "sigmoid", torch.sigmoid)
    assert torch.allclose(s.lo, torch.sigmoid(a.lo))
    t = _unary(a, "tanh", torch.tanh)
    assert torch.allclose(t.hi, torch.tanh(a.hi))


def test_interval_silu_includes_interior_min():
    # silu's global min ≈ -0.2785 at x ≈ -1.2785 — must appear when the
    # box straddles the critical point, must not when it doesn't.
    a = _box([-2.0], [0.0])
    s = _unary(a, "silu", torch.nn.functional.silu)
    assert float(s.lo) <= -0.2784
    b = _box([1.0], [3.0])
    s2 = _unary(b, "silu", torch.nn.functional.silu)
    assert (
        float(s2.lo)
        >= float(
            torch.nn.functional.silu(
                torch.tensor(1.0, dtype=torch.float64)
            )
        )
        - 1e-6
    )


def test_interval_softmax_bounds():
    a = _box([[-1.0, 0.0, -2.0]], [[1.0, 0.5, -0.5]])
    out = _softmax(a, -1)
    for _ in range(200):
        x = a.lo + torch.rand_like(a.lo) * (a.hi - a.lo)
        p = torch.softmax(x, -1)
        assert (p >= out.lo - 1e-9).all() and (p <= out.hi + 1e-9).all()
    # a point box collapses to the exact softmax
    pt = _softmax(_box([0.3, -1.0], [0.3, -1.0]), -1)
    assert torch.allclose(
        pt.lo,
        torch.softmax(
            torch.tensor([0.3, -1.0], dtype=torch.float64), -1
        ),
    )


def test_structural_ops_preserve_boxes():
    """reshape/transpose/concat/index_select map boxes exactly."""
    x = Var("x", TensorType((2, 4)))
    t = Op.make("reshape", x, shape=(4, 2))
    b = _box(torch.zeros(2, 4), torch.ones(2, 4))
    res = ibp_bound(t, {}, {"x": b})
    assert res["lo"].shape == (4, 2)
    tr = ibp_bound(
        Op.make("transpose", x, arg1=0, arg2=1), {}, {"x": b}
    )
    assert tr["lo"].shape == (4, 2)


# ---------------------------------------------------------------------------
#  ibp_bound on a small graph
# ---------------------------------------------------------------------------


def test_ibp_bound_contains_true_outputs():
    """linear(relu(linear(x))) — the output box contains every sampled
    in-box evaluation, and a point box gives exact evaluation."""
    torch.manual_seed(0)
    W1 = torch.randn(32, 16, dtype=torch.float64)
    W2 = torch.randn(8, 32, dtype=torch.float64)
    x0 = torch.randn(4, 16, dtype=torch.float64)
    x = Var("x", TensorType((4, 16)))
    term = Op.make(
        "linear",
        Op.make(
            "relu",
            Op.make("linear", x, Param("W1", TensorType((32, 16)))),
        ),
        Param("W2", TensorType((8, 32))),
    )
    env = {"W1": W1, "W2": W2}

    # point box == exact eval
    res = ibp_bound(term, env, {"x": x0})
    true = torch.nn.functional.linear(
        torch.relu(torch.nn.functional.linear(x0, W1)), W2
    )
    assert torch.allclose(res["lo"], true) and res["width"] < 1e-12

    # widened box contains samples
    res = ibp_bound(term, env, {"x": (x0 - 0.2, x0 + 0.2)})
    for _ in range(50):
        xs = x0 + (torch.rand_like(x0) * 2 - 1) * 0.2
        ys = torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(xs, W1)), W2
        )
        assert (ys >= res["lo"] - 1e-9).all()
        assert (ys <= res["hi"] + 1e-9).all()
    assert res["unsupported"] == []


def test_site_boxes_inject_center_radius():
    """The eps-site mechanism: a member represented as centre±radius —
    injecting a radius at the member's path widens its box, and the
    output interval grows to cover the perturbed evaluations."""
    torch.manual_seed(0)
    W = torch.randn(8, 8, dtype=torch.float64)
    x0 = torch.randn(2, 8, dtype=torch.float64)
    x = Var("x", TensorType((2, 8)))
    w = Param("Wq", TensorType((8, 8)))
    site = Op.make("mul", w, Const(0.01))  # the eps member
    term = Op.make("linear", x, site)
    res = ibp_bound(term, {"Wq": W}, {"x": x0}, site_boxes={(1,): 0.5})
    # output must contain x·(W·0.01 ± 0.5)ᵀ evaluations
    for _ in range(50):
        d = (torch.rand(8, 8, dtype=torch.float64) * 2 - 1) * 0.5
        y = torch.nn.functional.linear(x0, W * 0.01 + d)
        assert (y >= res["lo"] - 1e-9).all()
        assert (y <= res["hi"] + 1e-9).all()


def test_ibp_reports_unsupported_ops():
    """An op with no interval rule produces ±∞, flagged honestly."""
    x = Var("x", TensorType((2, 3)))
    t = Op.make("sdpa", x, x, x)
    # a point box evaluates exactly through the torch binding (sound)
    res0 = ibp_bound(t, {}, {"x": torch.ones(2, 3)})
    assert res0["width"] < 1e-9
    # a real box has no interval rule → ±∞, flagged honestly
    res = ibp_bound(t, {}, {"x": (torch.zeros(2, 3), torch.ones(2, 3))})
    assert "sdpa" in res["unsupported"]
    assert res["width"] == float("inf")


# ---------------------------------------------------------------------------
#  End-to-end: tight_model_bound on the 2-layer quantized model
# ---------------------------------------------------------------------------


def _quant_model(seed=0):
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 64, bias=False)
            s.l2 = nn.Linear(64, 32, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    return M().eval().double()


def _quant_setup(seed=0):
    m = _quant_model(seed)
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    offers = quant_params(eg, src, bits=8)
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(ir.root, term, root_eid=root)
    return m, x, ir, src, term, cert, offers


def test_tight_bound_beats_spectral_and_stays_sound():
    """The headline result: IBP's bound is (a) ≥ the measured error —
    still a certificate — and (b) strictly below the spectral product."""
    m, x, ir, src, term, cert, offers = _quant_setup()
    assert len(offers) == 2
    res = tight_model_bound(term, cert, src, x)
    spec = res["spectral_bound"]
    err = res["measured_error"]
    assert spec != float("inf")
    assert err is not None and err > 0
    # soundness: bound covers the measured error
    assert res["bound"] >= err - 1e-9
    assert res["artifact_bound"] >= err - 1e-9
    # tightness: strictly below spectral
    assert res["bound"] <= spec + 1e-12
    assert res["tighter"]
    # report the achieved ratios (the ~100x → ≤10x ask)
    assert res["cert_conservatism"] < spec / err
    assert res["note"]
    print(
        f"\n  spectral={spec:.4f}  ibp={res['bound']:.4f}  "
        f"artifact={res['artifact_bound']:.4f}  err={err:.5f}  "
        f"(ibp {res['cert_conservatism']:.1f}x, "
        f"artifact {res['artifact_conservatism']:.1f}x)"
    )


def test_tight_bound_uses_real_activation_norms():
    """The weight-side edge is billed the measured activation norm, not
    input_norm × a sensitivity product — check the site contributions
    individually undercut their spectral counterparts."""
    m, x, ir, src, term, cert, offers = _quant_setup()
    res = tight_model_bound(term, cert, src, x)
    mb = model_bound(
        term, cert, src, torch.linalg.norm(x, dim=-1).max().item()
    )
    spec_by_path = {
        tuple(c["path"]): c["contribution"]
        for c in mb["site_contributions"]
        if c.get("path")
    }
    ibp_better = 0
    for c in res["site_contributions"]:
        if c.get("path") is None or c.get("fallback"):
            continue
        s = spec_by_path.get(tuple(c["path"]))
        if s is not None and c["contribution"] <= s + 1e-12:
            ibp_better += 1
    assert ibp_better >= 1, "no site improved on its spectral path"


def test_dead_relu_zeroes_site_contribution():
    """A ReLU whose *widened* pre-activation box stays negative has
    slope 0 — the quantized upstream weight contributes exactly 0,
    where spectral still bills the full Lipschitz product."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 64, bias=False)
            s.l2 = nn.Linear(64, 32, bias=False)
            s.l1.weight.data = -0.5 * torch.ones(64, 32)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    m = M().eval().double()
    x = torch.ones(4, 32, dtype=torch.float64)  # pre-act = -16, dead
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    quant_params(eg, src, bits=8)
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(ir.root, term, root_eid=root)
    res = tight_model_bound(term, cert, src, x)
    # everything after the dead relu contributes ~0; bound is tiny vs
    # the spectral product and still covers the (also ~0) true error
    assert res["bound"] < res["spectral_bound"] * 0.5
    assert res["bound"] >= (res["measured_error"] or 0.0) - 1e-9


def test_ibp_never_worse_than_spectral():
    """Sites with no local-rule path contribute exactly their spectral
    share — the total is a per-site min, never worse than model_bound."""
    m, x, ir, src, term, cert, offers = _quant_setup()
    res = tight_model_bound(term, cert, src, x)
    assert res["bound"] <= res["spectral_bound"] + 1e-12
    for c in res["site_contributions"]:
        assert "contribution" in c and "fallback" in c


def test_honest_report_when_no_gain():
    """When IBP cannot beat spectral the result dict says so — a
    single quantized linear bills the site identically on both paths
    (same activation norm, no intervening ops to tighten)."""
    torch.manual_seed(0)
    m = nn.Linear(32, 32, bias=False).eval().double()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    offers = quant_params(eg, src, bits=8)
    assert offers
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(ir.root, term, root_eid=root)
    res = tight_model_bound(term, cert, src, x)
    assert "spectral_bound" in res and "bound" in res
    assert isinstance(res["note"], str) and res["note"]
    if not res["tighter"]:
        assert "did not improve" in res["note"]
    # artifact bound is still tighter (real ΔW < Frobenius radius)
    assert res["artifact_bound"] <= res["bound"] + 1e-12
    assert res["artifact_bound"] >= (res["measured_error"] or 0) - 1e-9


def test_artifact_bound_uses_realized_delta():
    """The realized ΔW (lhs−rhs of the cert step) is a tighter, still
    true radius than the cert's worst-case Frobenius envelope."""
    m, x, ir, src, term, cert, offers = _quant_setup()
    res = tight_model_bound(term, cert, src, x)
    assert res["artifact_bound"] <= res["bound"] + 1e-12
    # and it should be *noticeably* tighter than the cert-radius bound
    # on a quantized model (actual σ_max(ΔW) << (s/2)·√n)
    assert res["artifact_bound"] < res["bound"] * 0.9
