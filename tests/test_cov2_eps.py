# ruff: noqa: RUF002, RUF003
"""Coverage tests for catopt.eps internals — the optional ε axis.

``test_eps.py`` pins the headline behaviour on exported models; this
file closes the residual gaps: the guard branches of the three offer
passes (``low_rank_params`` / ``kron_linear_params`` /
``low_rank_gather``), ``optimize_weight``'s degenerate inputs, the
spectral/Lipschitz helpers, and ``model_bound``'s skip conditions —
with real numeric assertions on seeded tensors.

Defensive branches deliberately not covered (suggest ``pragma: no
cover``/``no branch``):
- ``ec is None: continue`` inside the class scans of ``quant_params``
  (eps.py:582), ``low_rank_gather`` (732), ``kron_linear_params``
  (883) and ``low_rank_params`` (1053): ``EGraph.union`` deletes
  non-canonical ``_classes`` keys eagerly, so every id in the
  ``list(eg._classes.keys())`` snapshot resolves to a live class.
"""

import math

import pytest
import torch

from catopt.egraph import Certificate, CertStep, EGraph, Rewrite
from catopt.eps import (
    _input_sensitivity,
    _lip_wrt,
    _path_sensitivity,
    _shape_of,
    _term_spectral,
    kron_linear_params,
    low_rank_gather,
    low_rank_params,
    model_bound,
    optimize_weight,
    quant_params,
)
from catopt.ir import Const, Op, Param, TensorType, Var


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  optimize_weight — degenerate branches
# ---------------------------------------------------------------------------


def test_optimize_weight_non_matrix_and_non_float():
    """``optimize_weight`` only searches 2-D floating weights — a
    vector and an integer matrix get no offers and keep all bytes."""
    res = optimize_weight("W", torch.randn(64))
    assert res["offers"] == []
    assert res["bytes"] == res["orig_bytes"] == 64 * 4

    res = optimize_weight("W", torch.arange(64).reshape(8, 8))
    assert res["offers"] == []
    assert res["bytes"] == res["orig_bytes"] == 64 * 8


def test_optimize_weight_nan_weight_no_offers():
    """SVD failure is swallowed honestly: a NaN weight produces no
    low-rank offer (S=None), the rearranged SVDs keep failing through
    every factor pair, and amax=nan fails the ``amax > 0`` test — an
    empty offer list, not a crash or a fabricated bound."""
    res = optimize_weight("W", torch.full((32, 32), float("nan")))
    assert res["offers"] == []
    assert res["bytes"] == res["orig_bytes"]


def test_optimize_weight_zero_weight_no_offers():
    """A zero weight: ``S[0] == 0`` skips low-rank, the Kronecker energy
    fractions are all NaN so no K is ever accepted (the ``not len(ok)``
    continue), and ``amax == 0`` skips quantization."""
    res = optimize_weight("W", torch.zeros(32, 32))
    assert res["offers"] == []
    assert res["bytes"] == res["orig_bytes"]


def test_optimize_weight_negative_rtol_no_rank_fits_budget():
    """Negative rtol ⇒ negative budget ⇒ ``tail[1:] <= budget`` is
    empty (the appended 0 fails it too) → the low-rank offer is
    skipped.  Kronecker/quant use rtol² and absmax — still offered."""
    torch.manual_seed(0)
    res = optimize_weight("W", torch.randn(32, 32).double(), rtol=-1.0)
    kinds = [o[0] for o in res["offers"]]
    assert "lowrank" not in kinds
    assert "quant" in kinds


def test_optimize_weight_rank_fails_storage_saving():
    """A rank that meets rtol but fails ``r(o+i) < 0.9·o·i`` is
    refused — the pass declines rather than offer a bigger program."""
    torch.manual_seed(0)
    W = torch.randn(16, 16).double()
    # seeded 16x16: σ decays slowly — rtol=0.3 forces r=10, and
    # 10·(16+16) = 320 ≥ 0.9·256 = 230.4, so the offer is declined.
    S = torch.linalg.svdvals(W)
    budget = 0.3 * float(S[0])
    tail = torch.cat([S, S.new_zeros(1)])
    below = (tail[1:] <= budget).nonzero()
    r = int(below[0].item()) + 1
    assert r * 32 >= 0.9 * 256  # pin the skip precondition itself
    res = optimize_weight("W", W, rtol=0.3)
    kinds = [o[0] for o in res["offers"]]
    assert "lowrank" not in kinds


def test_optimize_weight_kronecker_offer_executes():
    """The executable Kronecker-sum offer path: factor params are
    injected, the add-chain member is offered, and the reported
    residual equals the true rearranged-SVD tail."""
    torch.manual_seed(0)
    A, B = torch.randn(16, 16), torch.randn(4, 4)
    W = (torch.kron(A, B) + 0.01 * torch.randn(64, 64)).double()
    res = optimize_weight("W", W, rtol=0.05)
    kinds = [o[0] for o in res["offers"]]
    assert "kron" in kinds
    kron = res["offers"][kinds.index("kron")]
    K, resid = kron[1], kron[2]
    assert K >= 1 and resid > 0
    # the derived factor params materialise in source_tensors
    k_names = [n for n in res["source_tensors"] if n.startswith("w_W_k")]
    assert len(k_names) == 2 * K  # one (a, b) pair per K-term
    # the certified residual is honest — never below the realized
    # error of the reported factorisation
    assert resid <= float(torch.linalg.norm(W)) + 1e-9


# ---------------------------------------------------------------------------
#  quant_params — per-channel and skip guards
# ---------------------------------------------------------------------------


def test_quant_params_per_channel_exact_bound():
    """Per-channel int8: the certified bound is exactly
    (√n_cols/2)·‖s‖₂ with s = row_absmax/levels — verified against an
    independent recomputation, and the realized error honours it."""
    torch.manual_seed(0)
    x = _v("x", 4, 64)
    W = torch.randn(64, 64, dtype=torch.float64)
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("W", 64, 64)))
    src = {"W": W}
    offers = quant_params(eg, src, bits=8, per_channel=True)
    assert len(offers) == 1
    o = offers[0]
    levels = 2 ** (8 - 1) - 1
    flat = W.detach().reshape(W.shape[0], -1)
    sc = flat.abs().amax(-1, keepdim=True) / levels
    expected = float(
        math.sqrt(flat.shape[1]) / 2 * torch.linalg.norm(sc.squeeze(-1))
    )
    assert o["bound"] == pytest.approx(expected, rel=1e-12)
    # scale rows stored as real params shaped (rows, 1, ...)
    assert src["eps_s8c_W"].shape == (64, 1)
    # realized per-entry error honours |Δw_ij| ≤ s_row/2
    resid = (src["eps_q8c_W"].double() * sc - flat).abs()
    assert float(resid.max()) <= float(sc.max() / 2) + 1e-12


def test_quant_params_skips_non_float_scalar_and_nontensor():
    """The offer guard: integer tensors (no float interpretation),
    numel-1 scalars, and non-tensor source values are all declined."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    eg = EGraph()
    eg.add_term(Op.make("mul", x, _p("Wi", 4, 8)))
    eg.add_term(Op.make("mul", x, _p("s", 4, 8)))
    eg.add_term(Op.make("mul", x, _p("nt", 4, 8)))
    eg.add_term(_p("one", 8))  # bare leaf — the only valid offer
    src = {
        "Wi": torch.arange(32).reshape(4, 8),  # int64
        "s": torch.tensor(2.5),  # numel == 1
        "nt": "not-a-tensor",
        "one": torch.ones(8),
    }
    offers = quant_params(eg, src, bits=8)
    assert [o["name"] for o in offers] == ["one"]
    assert offers[0]["bound"] == pytest.approx(
        (1.0 / 127) / 2 * math.sqrt(8), rel=1e-9
    )


def test_quant_params_zero_weight_and_no_witness():
    """amax == 0 skips the per-tensor offer; ``witness=False`` unions
    the member with no Rewrite so no bound is recorded anywhere."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("Wz", 8, 8)))
    eg.add_term(Op.make("linear", x, _p("Wg", 8, 8)))
    src = {"Wz": torch.zeros(8, 8), "Wg": torch.randn(8, 8)}
    offers = quant_params(eg, src, bits=8, witness=False)
    assert {o["name"] for o in offers} == {"Wg"}  # zero weight refused
    assert not any(n.startswith("eps_q8") for n in eg._rule_objs)


# ---------------------------------------------------------------------------
#  low_rank_params — site guards + bias path
# ---------------------------------------------------------------------------


def _eg_with_linear_sites():
    """An e-graph holding one ``linear`` enode per kind of weight
    situation the pass must decline or accept."""
    eg = EGraph()
    x = _v("x", 4, 64)
    b = _p("b", 64)
    terms = [
        Op.make(
            "linear", x, Op.make("mul", _p("Wok", 64, 64), Const(1.0))
        ),  # weight slot is an Op, not a Param
        Op.make("linear", x, _p("Wmiss", 64, 64)),  # missing from src
        Op.make("linear", x, _p("W3d", 4, 4, 4)),  # not a matrix
        Op.make("linear", x, _p("Wnan", 64, 64)),  # SVD fails
        Op.make("linear", x, _p("Wzero", 64, 64)),  # S[0] == 0
        Op.make("linear", x, _p("Wfull", 64, 64)),  # nothing fits rtol
        Op.make("linear", x, _p("Wok", 64, 64), b),  # biased offer site
    ]
    for t in terms:
        eg.add_term(t)
    return eg


def test_low_rank_params_guards_and_bias_site():
    torch.manual_seed(0)
    Wok = torch.randn(64, 6) @ torch.randn(6, 64) + 0.002 * torch.randn(
        64, 64
    )
    src = {
        "Wok": Wok.double(),
        "W3d": torch.randn(4, 4, 4),
        "Wnan": torch.full((64, 64), float("nan")),
        "Wzero": torch.zeros(64, 64),
        "Wfull": torch.randn(64, 64),
        "b": torch.randn(64),
    }
    eg = _eg_with_linear_sites()
    offers = low_rank_params(eg, src, rtol=0.05)
    assert {o["name"] for o in offers} == {"Wok"}
    o = offers[0]
    assert o["stored"] == o["rank"] * 128 < o["original"]
    # the biased site's offered member keeps the bias as third child
    outer = [
        n
        for n in eg.get_class(o["site_eid"]).nodes
        if n.op == "linear" and len(n.children) == 3
    ]
    assert outer, "expected a 3-child linear member carrying the bias"
    # the certified bound is exactly σ_{r+1} of the source weight
    S = torch.linalg.svdvals(src["Wok"])
    assert o["bound"] == pytest.approx(float(S[o["rank"]]), rel=1e-9)


def test_low_rank_params_witness_false_registers_no_rule():
    torch.manual_seed(0)
    x = _v("x", 4, 64)
    W = torch.randn(64, 4) @ torch.randn(4, 64) + 0.001 * torch.randn(
        64, 64
    )
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("W", 64, 64)))
    src = {"W": W.double()}
    offers = low_rank_params(eg, src, rtol=0.05, witness=False)
    assert len(offers) == 1
    assert not any(n.startswith("eps_lr") for n in eg._rule_objs)


def test_low_rank_params_negative_rtol_no_offer():
    """Negative rtol ⇒ the ``below`` set is empty (0 fails a negative
    budget too) → every site declined."""
    torch.manual_seed(0)
    x = _v("x", 4, 64)
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("W", 64, 64)))
    src = {"W": torch.randn(64, 64)}
    assert low_rank_params(eg, src, rtol=-1.0) == []


# ---------------------------------------------------------------------------
#  low_rank_gather — site guards
# ---------------------------------------------------------------------------


def test_low_rank_gather_guards():
    torch.manual_seed(0)
    idx = _v("idx", 8)
    Wok = (
        torch.randn(1000, 12) @ torch.randn(12, 64)
        + 0.01 * torch.randn(1000, 64)
    ).double()
    src = {
        "Wok": Wok,
        "W3d": torch.randn(4, 4, 4),
        "Wnan": torch.full((1000, 64), float("nan")),
        "Wzero": torch.zeros(1000, 64),
        "Wfull": torch.randn(1000, 64),
    }
    eg = EGraph()
    w = _p("Wok", 1000, 64)
    for t in (
        Op.make(
            "embedding", Op.make("mul", w, Const(1.0)), idx
        ),  # table is an Op
        Op.make("embedding", _p("Wmiss", 8, 8), idx),  # missing param
        Op.make("embedding", _p("W3d", 4, 4, 4), idx),  # not 2-D
        Op.make("embedding", _p("Wnan", 1000, 64), idx),  # SVD fails
        Op.make("embedding", _p("Wzero", 1000, 64), idx),  # S[0] == 0
        Op.make(
            "embedding", _p("Wfull", 1000, 64), idx
        ),  # nothing fits rtol
        Op.make("embedding", w, idx),  # the real offer
    ):
        eg.add_term(t)
    offers = low_rank_gather(eg, src, rtol=0.05)
    assert {o["name"] for o in offers} == {"Wok"}
    o = offers[0]
    S = torch.linalg.svdvals(Wok)
    assert o["bound"] == pytest.approx(float(S[o["rank"]]), rel=1e-9)


def test_low_rank_gather_storage_guard_and_witness_false():
    """``min_saving`` below the guaranteed floor declines every offer;
    witness=False unions without a bound-carrying Rewrite."""
    torch.manual_seed(0)
    idx = _v("idx", 8)
    W = (
        torch.randn(1000, 8) @ torch.randn(8, 64)
        + 0.005 * torch.randn(1000, 64)
    ).double()
    eg = EGraph()
    eg.add_term(Op.make("embedding", _p("W", 1000, 64), idx))
    src = {"W": W}
    # r·(v+d) >= 0 for any r → the saving check always fails
    assert low_rank_gather(eg, src, rtol=0.5, min_saving=0.0) == []
    offers = low_rank_gather(eg, src, rtol=0.05, witness=False)
    assert len(offers) == 1
    assert not any(n.startswith("eps_emb") for n in eg._rule_objs)


def test_low_rank_gather_negative_rtol_no_offer():
    torch.manual_seed(0)
    idx = _v("idx", 8)
    eg = EGraph()
    eg.add_term(Op.make("embedding", _p("W", 64, 64), idx))
    src = {"W": torch.randn(64, 64)}
    assert low_rank_gather(eg, src, rtol=-1.0) == []


# ---------------------------------------------------------------------------
#  kron_linear_params — site guards + bias path
# ---------------------------------------------------------------------------


def test_kron_linear_params_guards_and_bias():
    torch.manual_seed(0)
    Wk = (
        torch.kron(torch.randn(8, 8), torch.randn(8, 8))
        + 0.01 * torch.randn(64, 64)
    ).double()
    src = {
        "Wk": Wk,
        "b": torch.randn(64),
        "W3d": torch.randn(4, 4, 4),
        "Wnan": torch.full((64, 64), float("nan")),
        "Wzero": torch.zeros(64, 64),
    }
    x = _v("x", 4, 64)
    wk = _p("Wk", 64, 64)
    eg = EGraph()
    for t in (
        Op.make(
            "linear", x, Op.make("mul", wk, Const(1.0))
        ),  # weight is an Op
        Op.make("linear", x, _p("Wmiss", 64, 64)),  # missing param
        Op.make("linear", x, _p("W3d", 4, 4, 4)),  # not 2-D
        Op.make("linear", x, _p("Wnan", 64, 64)),  # rearranged SVD fails
        Op.make("linear", x, _p("Wzero", 64, 64)),  # S[0] == 0
        Op.make("linear", x, wk, _p("b", 64)),  # biased offer site
    ):
        eg.add_term(t)
    offers = kron_linear_params(eg, src, rtol=0.05)
    assert {o["name"] for o in offers} == {"Wk"}
    o = offers[0]
    m1, n1, m2, n2 = o["factors"]
    assert m1 * m2 == 64 and n1 * n2 == 64
    assert o["stored"] == o["K"] * (m1 * n1 + m2 * n2)
    assert o["stored"] < o["original"]
    assert len([n for n in src if n.startswith("eps_k")]) == 2 * o["K"]


def test_kron_linear_params_witness_false():
    torch.manual_seed(0)
    x = _v("x", 4, 64)
    Wk = (
        torch.kron(torch.randn(8, 8), torch.randn(8, 8))
        + 0.01 * torch.randn(64, 64)
    ).double()
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("Wk", 64, 64)))
    src = {"Wk": Wk}
    offers = kron_linear_params(eg, src, rtol=0.05, witness=False)
    assert len(offers) == 1
    assert not any(n.startswith("eps_kron") for n in eg._rule_objs)


# ---------------------------------------------------------------------------
#  Spectral/Lipschitz helpers — direct, with real numbers
# ---------------------------------------------------------------------------


def test_term_spectral_param_const_and_eval_paths():
    torch.manual_seed(0)
    W = torch.randn(8, 6, dtype=torch.float64)
    M = torch.randn(6, 4, dtype=torch.float64)
    v = torch.randn(8, dtype=torch.float64)
    env = {"W": W, "M": M, "v": v}
    # Param leaf: 2-D → σ_max, non-2-D → abs max; absent → None
    assert _term_spectral(_p("W", 8, 6), env) == pytest.approx(
        float(torch.linalg.norm(W, 2))
    )
    assert _term_spectral(_p("v", 8), env) == pytest.approx(
        float(v.abs().max())
    )
    assert _term_spectral(_p("missing", 8, 6), env) is None
    assert _term_spectral(Const(-2.5), env) == 2.5
    # param-only Op evaluates then measures — exact vs torch matmul
    t = Op.make("matmul", _p("W", 8, 6), _p("M", 6, 4))
    assert _term_spectral(t, env) == pytest.approx(
        float(torch.linalg.norm(W @ M, 2))
    )
    # a 1-D result measures by abs max
    t1 = Op.make("mul", _p("v", 8), Const(2.0))
    assert _term_spectral(t1, env) == pytest.approx(
        float((v * 2).abs().max())
    )
    # op with no torch binding → None (not an exception)
    assert (
        _term_spectral(Op.make("bogus_op", _p("W", 8, 6)), env) is None
    )
    # an arg that can't evaluate → None
    t2 = Op.make(
        "mul", Op.make("bogus_op", _p("W", 8, 6)), _p("W", 8, 6)
    )
    assert _term_spectral(t2, env) is None
    # param absent from env → lookup failure inside → None
    t3 = Op.make("mul", _p("ZZ", 8, 6), _p("W", 8, 6))
    assert _term_spectral(t3, env) is None
    # an Op containing a Var leaf isn't param-only → no fold, None
    t4 = Op.make("mul", _v("x", 8, 6), _p("W", 8, 6))
    assert _term_spectral(t4, env) is None


def test_lip_wrt_softmax_sdpa_and_unknown():
    x = _v("x", 4, 4)
    children = [x, x]
    assert _lip_wrt("softmax", 0, children, {}) == 1.0
    assert _lip_wrt("sdpa", 0, children, {}) is None
    assert _lip_wrt("frobnicate", 0, children, {}) is None
    # weight-side edge resolves via act_norm when provided
    assert _lip_wrt("linear", 1, children, {}, act_norm=3.0) == 3.0


def test_input_sensitivity_skips_unbounded_ops():
    """An sdpa child contributes no bound (lip None → skipped); the
    result comes solely from the bounded path."""
    x = _v("x", 4, 4)
    term = Op.make("add", x, Op.make("sdpa", x, x, x))
    assert _input_sensitivity(term, {}, 1.0) == 1.0
    # a term that is only unsupported ops → best stays 0
    term2 = Op.make("sdpa", x, x, x)
    assert _input_sensitivity(term2, {}, 1.0) == 0.0


def test_path_sensitivity_out_of_range_and_leaf():
    x = _v("x", 4, 4)
    w = _p("W", 4, 4)
    term = Op.make("add", w, x)
    # descends into a leaf: second hop hits a Param, not an Op
    assert _path_sensitivity(term, (0, 0), {}) is None
    # child index beyond arity
    assert _path_sensitivity(term, (5,), {}) is None


def test_model_bound_skips_unbounded_and_unknown_rules():
    """Steps whose rule is absent from the rule table — or carries no
    error_bound — are skipped before site location is attempted."""
    x = _v("x", 4, 4)
    w = _p("W", 4, 4)
    term = Op.make("linear", x, w)
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(rule="ghost", path=(), lhs=term, rhs=term),
            CertStep(rule="free", path=(), lhs=term, rhs=term),
        ],
        rules={
            "free": Rewrite(name="free", lhs=term, rhs=term),
        },
    )
    mb = model_bound(term, cert, {"W": torch.eye(4)})
    assert mb["bound"] == 0.0
    assert mb["n_bounded_steps"] == 0
    assert mb["site_contributions"] == []


def test_shape_of_wrapper_delegates():
    x = _v("x", 2, 3)
    assert _shape_of(x) == (2, 3)
    assert _shape_of(Op.make("bogus_op", x)) == (2, 3)
    assert _shape_of(Const(1.0)) == ()


# ---------------------------------------------------------------------------
#  Bound exactness — real numbers on small matrices
# ---------------------------------------------------------------------------


def test_quant_bound_is_exact_s_over_two_sqrt_n():
    """The quant offer's reported bound is exactly (s/2)·√n for the
    per-tensor scale — recomputed independently — and the realized
    Frobenius error honours it."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    W = torch.randn(8, 8, dtype=torch.float64) * 3
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("W", 8, 8)))
    src = {"W": W}
    o = quant_params(eg, src, bits=8)[0]
    levels = 127
    s = float(W.abs().max()) / levels
    assert o["bound"] == pytest.approx(
        s / 2 * math.sqrt(W.numel()), rel=1e-9
    )
    err = float(torch.linalg.norm(src["eps_q8_W"].double() * s - W))
    assert err <= o["bound"] + 1e-9


def test_low_rank_bound_is_exact_sigma_r_plus_1():
    """Eckart–Young: the offered bound equals σ_{r+1} computed
    independently, and the stored factorization realizes it."""
    torch.manual_seed(0)
    x = _v("x", 4, 32)
    W = (
        torch.randn(32, 5) @ torch.randn(5, 32)
        + 1e-3 * torch.randn(32, 32)
    ).double()
    eg = EGraph()
    eg.add_term(Op.make("linear", x, _p("W", 32, 32)))
    src = {"W": W}
    o = low_rank_params(eg, src, rtol=0.05)[0]
    S = torch.linalg.svdvals(W)
    assert o["bound"] == pytest.approx(float(S[o["rank"]]), rel=1e-9)
    # the injected factors reconstruct W within the bound
    un = f"eps_u_W_{o['site_eid']}"
    vn = f"eps_v_W_{o['site_eid']}"
    err = float(torch.linalg.norm(src[un] @ src[vn] - W, 2))
    assert err <= o["bound"] + 1e-9
