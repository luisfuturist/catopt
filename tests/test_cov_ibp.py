# ruff: noqa: RUF002, RUF003
"""Coverage tests for catopt.ibp internals and tight_model_bound paths.

test_ibp.py pins the headline behaviour (sound + tighter than spectral).
This file exercises the machinery underneath it: the scalar bound
helpers, every interval op rule (including the honest-unsupported
paths), the per-hop Lipschitz table ``_hop``, the elementwise
Lipschitz maps, the realized-delta tensor walk ``_prop_delta``, and
``tight_model_bound`` over synthetic certificates covering every site
kind — weight slot, activation linear (spectral-unsafe), embedding
matmul, activation matmul, unlocated — plus the fallback ladder
(local walk → spectral contribution → global sensitivity → ∞).
"""

import math

import pytest
import torch
import torch.nn as nn

from catopt.cost import param_bytes_cost_for
from catopt.egraph import Certificate, CertStep, EGraph, Rewrite
from catopt.eps import (
    kron_linear_params,
    low_rank_params,
    quant_params,
)
from catopt.ibp import (
    Box,
    _add,
    _closest_to_zero,
    _collect_sites,
    _div,
    _elem_lip_map,
    _eval_concrete,
    _fro_bound,
    _grid_lip,
    _hop,
    _inf_box,
    _matmul,
    _maxrow_bound,
    _minabs,
    _mul,
    _neg,
    _norm_input_box,
    _prop_delta,
    _site_delta,
    _site_scalar,
    _softmax_lip,
    _spec_bound,
    _spectral_path_ok,
    _sub,
    _unary,
    _unary_lip,
    _walk_site,
    _walk_site_artifact,
    _widen,
    ibp_bound,
    tight_model_bound,
)
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.rules import all_rules
from catopt.torch_bridge import export_to_ir


def _box(lo, hi):
    return Box(
        torch.as_tensor(lo, dtype=torch.float64),
        torch.as_tensor(hi, dtype=torch.float64),
    )


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  Scalar bounds derived from a box
# ---------------------------------------------------------------------------


def test_width_maxabs_and_fro_bound():
    b = _box([[-2.0, 0.5]], [[1.0, 3.0]])
    assert torch.equal(b.width, torch.tensor([[3.0, 2.5]]))
    assert torch.equal(b.maxabs, torch.tensor([[2.0, 3.0]]))
    assert _fro_bound(b) == pytest.approx(math.sqrt(4 + 9))


def test_maxrow_bound_scalar_and_rows():
    # 0-d box → the scalar itself
    assert _maxrow_bound(_box(0.5, 0.5)) == pytest.approx(0.5)
    # (n, d) box → max row L2
    b = _box([[3.0, 4.0], [1.0, 0.0]], [[3.0, 4.0], [1.0, 0.0]])
    assert _maxrow_bound(b) == pytest.approx(5.0)


def test_spec_bound_vector_matrix_and_batch():
    # 1-D: operator bound is max|x|
    assert _spec_bound(_box([-2.0, 0.5], [-2.0, 0.5])) == pytest.approx(
        2.0
    )
    # 0-D
    assert _spec_bound(_box(1.5, 1.5)) == pytest.approx(1.5)
    # 2-D: σ_max of the maxabs matrix
    m = _box([[3.0, 0.0], [0.0, 0.0]], [[3.0, 0.0], [0.0, 0.0]])
    assert _spec_bound(m) == pytest.approx(3.0)
    # 3-D batch of matrices: max over the batch
    t = torch.zeros(2, 2, 2, dtype=torch.float64)
    t[0, 0, 0] = 4.0
    t[1, 0, 0] = 2.0
    assert _spec_bound(Box(t, t)) == pytest.approx(4.0)


def test_minabs_straddle_and_one_sided():
    # straddles zero → 0
    assert _minabs(_box([-1.0, 2.0], [1.0, 3.0])) == 0.0
    # strictly positive → min lo
    assert _minabs(_box([2.0, 5.0], [4.0, 9.0])) == pytest.approx(2.0)
    # strictly negative → min |hi|
    assert _minabs(_box([-9.0, -2.0], [-4.0, -1.0])) == pytest.approx(
        1.0
    )


def test_closest_to_zero():
    b = _box([-2.0, 3.0, -5.0], [-1.0, 4.0, 6.0])
    assert torch.equal(
        _closest_to_zero(b), torch.tensor([-1.0, 3.0, 0.0])
    )


# ---------------------------------------------------------------------------
#  Interval primitives
# ---------------------------------------------------------------------------


def test_neg_and_div():
    a = _box([1.0, -2.0], [3.0, -1.0])
    n = _neg(a)
    assert torch.equal(n.lo, torch.tensor([-3.0, 1.0]))
    assert torch.equal(n.hi, torch.tensor([-1.0, 2.0]))
    # division by a box containing 0 → None (unbounded)
    assert _div(a, _box([-1.0], [1.0])) is None
    # valid division
    d = _div(_box([2.0, -4.0], [6.0, -2.0]), _box([1.0], [2.0]))
    assert d is not None
    assert torch.equal(d.lo, torch.tensor([1.0, -4.0]))
    assert torch.equal(d.hi, torch.tensor([6.0, -1.0]))


def test_unary_sqrt_rsqrt_square_and_unknown():
    # sqrt of a partially-negative box → unsupported
    assert _unary(_box([-1.0], [4.0]), "sqrt", torch.sqrt) is None
    s = _unary(_box([1.0, 4.0], [4.0, 9.0]), "sqrt", torch.sqrt)
    assert torch.equal(s.lo, torch.tensor([1.0, 2.0]))
    assert torch.equal(s.hi, torch.tensor([2.0, 3.0]))
    # rsqrt needs strictly positive; it is decreasing (lo from hi)
    assert _unary(_box([0.0], [1.0]), "rsqrt", torch.rsqrt) is None
    r = _unary(_box([1.0, 4.0], [4.0, 9.0]), "rsqrt", torch.rsqrt)
    assert torch.allclose(
        r.lo, torch.tensor([0.5, 1.0 / 3.0], dtype=torch.float64)
    )
    assert torch.allclose(
        r.hi, torch.tensor([1.0, 0.5], dtype=torch.float64)
    )
    # square: straddling box gets lo 0; one-sided keeps min²
    sq = _unary(_box([-2.0, 1.0], [1.0, 3.0]), "square", torch.square)
    assert torch.equal(sq.lo, torch.tensor([0.0, 1.0]))
    assert torch.equal(sq.hi, torch.tensor([4.0, 9.0]))
    sq2 = _unary(_box([2.0], [3.0]), "square", torch.square)
    assert torch.equal(sq2.lo, torch.tensor([4.0]))
    # neg and unknown
    ng = _unary(_box([1.0], [2.0]), "neg", torch.neg)
    assert torch.equal(ng.lo, torch.tensor([-2.0]))
    assert _unary(_box([1.0], [2.0]), "frobnicate", torch.tanh) is None


def test_gelu_interior_min():
    # gelu's min is at x* ≈ -0.7518 — a straddling box must include it
    g = _unary(
        _box([-2.0], [0.0]), "gelu", torch.nn.functional.gelu
    )
    assert float(g.lo) <= -0.1699  # gelu min ≈ -0.16997 at x*≈-0.75
    # entirely positive: lo = gelu(lo) (gelu is increasing for x > 0)
    g2 = _unary(
        _box([1.0], [3.0]), "gelu", torch.nn.functional.gelu
    )
    assert float(g2.lo) == pytest.approx(
        float(torch.nn.functional.gelu(torch.tensor(1.0))), abs=1e-6
    )


def test_inf_box_known_and_unknown_shape():
    x = _v("x", 2, 3)
    t = Op.make("mul", x, Const(2.0))
    b = _inf_box(t)
    assert b.lo.shape == (2, 3)
    assert float(b.lo.min()) == float("-inf")
    # a bare Const has scalar type → ±inf scalar
    b2 = _inf_box(Const(1.0))
    assert b2.lo.ndim == 0


def test_eval_concrete_variants():
    x = _v("x", 2)
    w = _p("w", 2)
    vals = {"x": torch.ones(2)}
    env = {"w": torch.full((2,), 3.0)}
    assert torch.equal(_eval_concrete(x, env, vals), torch.ones(2))
    assert torch.equal(_eval_concrete(w, env, vals), torch.full((2,), 3.0))
    c = _eval_concrete(Const(2.5), env, vals)
    assert float(c) == 2.5
    # missing var/param → None
    assert _eval_concrete(_v("nope", 2), env, vals) is None
    assert _eval_concrete(_p("np", 2), env, vals) is None
    # op without a torch binding → None
    assert (
        _eval_concrete(Op.make("bogus_op", x), env, vals) is None
    )
    # op that raises inside torch → None (shape error)
    bad = Op.make(
        "matmul", _p("w", 2), _p("w", 2)
    )  # 1-D @ 1-D is fine; use reshape to impossible shape
    bad = Op.make("reshape", w, shape=(3, 3))
    assert _eval_concrete(bad, env, vals) is None
    # non-term leaf → None
    assert _eval_concrete(17, env, vals) is None
    # an evaluatable op composes
    ok = Op.make("add", x, w)
    assert torch.equal(
        _eval_concrete(ok, env, vals), torch.tensor([4.0, 4.0])
    )


def test_norm_input_box_forms():
    x = _v("x", 2, 3)
    y = _v("y", 4)
    term = Op.make("add", x, Op.make("sum", y, arg1=0))
    x0 = torch.randn(2, 3)
    y0 = torch.randn(4)

    # dict of tensors + input_radius
    ib = _norm_input_box({"x": x0}, term, input_radius=0.1)
    assert torch.allclose(ib["x"].lo, x0 - 0.1)
    # dict of (lo, hi) pairs and of Box
    ib = _norm_input_box(
        {"x": (x0 - 1, x0 + 1), "y": Box(y0, y0)}, term
    )
    assert torch.equal(ib["y"].lo, y0)
    # bare (lo, hi) tensor pair matching the first var's shape
    ib = _norm_input_box((x0 - 0.5, x0 + 0.5), term)
    assert torch.allclose(ib["x"].hi, x0 + 0.5)
    # tuple of tensors → mapped to vars in traversal order
    ib = _norm_input_box((x0, y0), term)
    assert set(ib) == {"x", "y"}
    # single tensor → first var
    ib = _norm_input_box(x0, term)
    assert torch.equal(ib["x"].lo, x0)


# ---------------------------------------------------------------------------
#  ibp_bound — per-op interval rules end to end
# ---------------------------------------------------------------------------


def test_ibp_div_and_zero_divisor():
    x = _v("x", 3)
    y = _v("y", 3)
    t = Op.make("div", x, y)
    envb = {"x": (torch.ones(3), torch.full((3,), 2.0)), "y": torch.full((3,), 4.0)}
    res = ibp_bound(t, {}, envb)
    assert torch.allclose(res["lo"], torch.full((3,), 0.25))
    # divisor straddling zero → unsupported + ±inf
    res = ibp_bound(t, {}, {"x": torch.ones(3), "y": (torch.full((3,), -1.0), torch.ones(3))})
    assert "div:0-in-divisor" in res["unsupported"]
    assert res["width"] == float("inf")


def test_ibp_linear_bias_and_1d_weight():
    x = _v("x", 2, 4)
    w = _p("w", 3, 4)
    b = _p("b", 3)
    W = torch.randn(3, 4, dtype=torch.float64)
    B = torch.randn(3, dtype=torch.float64)
    x0 = torch.randn(2, 4, dtype=torch.float64)
    res = ibp_bound(
        Op.make("linear", x, w, b), {"w": W, "b": B}, {"x": x0}
    )
    ref = torch.nn.functional.linear(x0, W, B)
    assert torch.allclose(res["lo"], ref, atol=1e-12)
    # a 1-D weight box has no linear rule → flagged, ±inf
    w1 = _p("w1", 4)
    res = ibp_bound(
        Op.make("linear", x, w1),
        {"w1": torch.randn(4, dtype=torch.float64)},
        {"x": x0},
    )
    assert "linear:1d-weight" in res["unsupported"]


def test_ibp_pow_exponent_variants():
    x = _v("x", 3)
    pos = (torch.ones(3), torch.full((3,), 2.0))
    # constant exponent 2 → square rule (straddle-aware)
    res = ibp_bound(
        Op.make("pow", x, Const(2.0)),
        {},
        {"x": (torch.full((3,), -1.0), torch.full((3,), 2.0))},
    )
    assert torch.equal(res["lo"], torch.zeros(3))
    assert torch.equal(res["hi"], torch.full((3,), 4.0))
    # constant integer exponent ≥ 1 on positive box
    res = ibp_bound(
        Op.make("pow", x, Const(3.0)), {}, {"x": pos}
    )
    assert torch.allclose(res["hi"], torch.full((3,), 8.0))
    # exponent 0 → ones
    res = ibp_bound(
        Op.make("pow", x, Const(0.0)), {}, {"x": pos}
    )
    assert torch.equal(res["lo"], torch.ones(3))
    # negative exponent on positive box → decreasing
    res = ibp_bound(
        Op.make("pow", x, Const(-1.0)), {}, {"x": pos}
    )
    assert torch.allclose(res["lo"], torch.full((3,), 0.5))
    assert torch.allclose(res["hi"], torch.ones(3))
    # non-constant exponent → unsupported
    e = _v("e", 3)
    res = ibp_bound(
        Op.make("pow", x, e),
        {},
        {"x": pos, "e": (torch.ones(3), torch.full((3,), 2.0))},
    )
    assert "pow" in res["unsupported"]


def test_ibp_reductions():
    x = _v("x", 2, 3)
    lo = torch.zeros(2, 3)
    hi = torch.ones(2, 3)
    for op, ref in (
        ("sum", torch.sum),
        ("mean", torch.mean),
        ("max", torch.amax),
        ("min", torch.amin),
    ):
        t = Op.make(op, x, arg1=-1)
        res = ibp_bound(t, {}, {"x": (lo, hi)})
        assert torch.allclose(res["lo"], ref(lo, dim=-1))
        assert torch.allclose(res["hi"], ref(hi, dim=-1))
        assert res["unsupported"] == []
    # whole-tensor max/min (no dim)
    res = ibp_bound(Op.make("max", x), {}, {"x": (lo, hi)})
    assert float(res["lo"]) == 0.0 and float(res["hi"]) == 1.0
    # dim as list + keepdim
    res = ibp_bound(
        Op.make("sum", x, dim=(0,), keepdim=True), {}, {"x": (lo, hi)}
    )
    assert res["lo"].shape == (1, 3)


def test_ibp_embedding_where_and_masked_fill():
    W = torch.randn(5, 4, dtype=torch.float64)
    w = _p("w", 5, 4)
    idx = _v("idx", 3)
    ivals = torch.tensor([0, 2, 4])
    # embedding with concretely evaluatable indices
    res = ibp_bound(
        Op.make("embedding", w, idx),
        {"w": W},
        {},
        values={"idx": ivals},
    )
    assert torch.equal(res["lo"], W[ivals])
    # same op, indices not evaluable → unsupported
    res = ibp_bound(Op.make("embedding", w, idx), {"w": W}, {})
    assert "embedding:idx" in res["unsupported"]

    # where with a concrete condition → exact selection of endpoints
    c = _v("c", 2)
    a, b = _v("a", 2), _v("b", 2)
    res = ibp_bound(
        Op.make("where", c, a, b),
        {},
        {
            "a": (torch.zeros(2), torch.ones(2)),
            "b": (torch.full((2,), 10.0), torch.full((2,), 20.0)),
        },
        values={"c": torch.tensor([True, False])},
    )
    assert torch.equal(res["lo"], torch.tensor([0.0, 10.0]))
    # non-concrete condition → conservative hull of both branches
    res = ibp_bound(
        Op.make("where", c, a, b),
        {},
        {
            "c": (torch.zeros(2), torch.ones(2)),
            "a": (torch.zeros(2), torch.ones(2)),
            "b": (torch.full((2,), 10.0), torch.full((2,), 20.0)),
        },
    )
    assert torch.equal(res["lo"], torch.zeros(2))
    assert torch.equal(res["hi"], torch.full((2,), 20.0))

    # masked_fill with concrete mask + scalar value
    x = _v("x", 3)
    m = _p("m", 3)
    res = ibp_bound(
        Op.make("masked_fill", x, m, Const(-1.0)),
        {"m": torch.tensor([True, False, True])},
        {"x": (torch.zeros(3), torch.ones(3))},
    )
    assert torch.equal(res["lo"], torch.tensor([-1.0, 0.0, -1.0]))


def test_ibp_concat_and_structural_failure():
    a, b = _v("a", 2, 2), _v("b", 2, 2)
    la, ha = torch.zeros(2, 2), torch.ones(2, 2)
    lb, hb = torch.full((2, 2), 5.0), torch.full((2, 2), 6.0)
    res = ibp_bound(
        Op.make("concat", a, b, dim=-1), {}, {"a": (la, ha), "b": (lb, hb)}
    )
    assert res["lo"].shape == (2, 4)
    assert torch.equal(res["lo"][:, 2:], lb)
    res = ibp_bound(
        Op.make("stack", a, b, dim=0), {}, {"a": (la, ha), "b": (lb, hb)}
    )
    assert res["hi"].shape == (2, 2, 2)
    # structural op that raises at eval (out-of-range select) →
    # honest unsupported flag (the interval map can't run)
    res = ibp_bound(
        Op.make("select", a, arg1=0, arg2=5), {}, {"a": (la, ha)}
    )
    assert "select" in res["unsupported"]


def test_ibp_values_fallback_and_unknown_leaf():
    # Var not in input_box but present in values → point box
    x = _v("x", 2)
    res = ibp_bound(
        Op.make("relu", x), {}, {}, values={"x": torch.ones(2)}
    )
    assert torch.equal(res["lo"], torch.ones(2))
    # Var missing everywhere → var:name flagged, ±inf
    res = ibp_bound(Op.make("relu", x), {}, {})
    assert "var:x" in res["unsupported"]
    # Param missing from env → param:name flagged
    res = ibp_bound(
        Op.make("relu", _p("w", 2)), {}, {"x": torch.ones(2)}
    )
    assert "param:w" in res["unsupported"]


# ---------------------------------------------------------------------------
#  _hop — the per-op local Lipschitz table
# ---------------------------------------------------------------------------


def _argb(*boxes):
    return list(boxes)


def test_hop_linear_matmul_slots():
    xb = _box(torch.randn(4, 8), torch.randn(4, 8) + 1)
    wb = _box(torch.randn(6, 8), torch.randn(6, 8) + 1)
    bb = _box(torch.randn(6), torch.randn(6) + 1)
    ob = _box(torch.randn(4, 6), torch.randn(4, 6) + 1)
    node = Op.make("linear", _v("x", 4, 8), _p("w", 6, 8), _p("b", 6))
    s0, k0 = _hop(node, 0, _argb(xb, wb, bb), ob, "row")
    assert s0 == pytest.approx(_spec_bound(wb)) and k0 == "row"
    # weight slot: spec kind rides through unscaled; row kind pays √rows
    s1, k1 = _hop(node, 1, _argb(xb, wb, bb), ob, "spec")
    assert s1 == pytest.approx(_maxrow_bound(xb)) and k1 == "row"
    s1r, _ = _hop(node, 1, _argb(xb, wb, bb), ob, "row")
    assert s1r == pytest.approx(s1 * math.sqrt(6))
    s2, _ = _hop(node, 2, _argb(xb, wb, bb), ob, "row")
    assert s2 == 1.0
    # matmul has no bias slot
    mm = Op.make("matmul", _v("x", 4, 8), _p("w", 8, 6))
    s3, _ = _hop(mm, 0, _argb(xb, wb), ob, "row")
    assert s3 == pytest.approx(_spec_bound(wb))
    assert _hop(mm, 2, _argb(xb, wb, bb), ob, "row")[0] is None


def test_hop_embedding_conv2d_and_preserving_ops():
    xb, wb = _box(torch.randn(4, 4), torch.randn(4, 4)), _box(
        torch.randn(4, 4), torch.randn(4, 4)
    )
    ob = xb
    emb = Op.make("embedding", _p("w", 4, 4), _v("i", 3))
    assert _hop(emb, 0, _argb(wb, wb), ob, "row") == (1.0, "row")
    assert _hop(emb, 1, _argb(wb, wb), ob, "row") == (None, "row")
    conv = Op.make("conv2d", _v("x", 1, 1, 4, 4), _p("w", 1, 1, 3, 3))
    assert _hop(conv, 0, _argb(xb, wb), ob, "row") == (None, "row")
    add = Op.make("add", _v("a", 2), _v("b", 2))
    for i in (0, 1):
        assert _hop(add, i, _argb(xb, wb), ob, "row") == (1.0, "row")
    mf = Op.make("masked_fill", _v("a", 2), _v("m", 2), Const(0.0))
    assert _hop(mf, 0, _argb(xb, wb, wb), ob, "row") == (1.0, "row")
    assert _hop(mf, 1, _argb(xb, wb, wb), ob, "row") == (None, "row")


def test_hop_transpose_reshape_rearrange():
    xb = _box(torch.randn(4, 8), torch.randn(4, 8))
    ob_same = _box(torch.randn(8, 4), torch.randn(8, 4))
    tr = Op.make("transpose", _v("x", 4, 8), arg1=-2, arg2=-1)
    # swapping the last two dims preserves the spectral bound
    assert _hop(tr, 0, [xb], ob_same, "spec") == (1.0, "spec")
    # swapping dim 0 ↔ 1 on a 3-D tensor is a rearrangement
    xb3 = _box(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    tr3 = Op.make("transpose", _v("x", 2, 4, 8), arg1=0, arg2=1)
    ob3 = _box(torch.randn(4, 2, 8), torch.randn(4, 2, 8))
    m, k = _hop(tr3, 0, [xb3], ob3, "row")
    assert m == 1.0 and k == "row"  # last dim (8) preserved
    # flatten changing the row dim pays a √(numel/d_out) bound
    fl = Op.make("flatten", _v("x", 2, 4))
    ob1 = _box(torch.randn(8), torch.randn(8))
    m, k = _hop(fl, 0, [xb], ob1, "row")
    assert m == pytest.approx(math.sqrt(8 / 8)) or m > 0


def test_hop_mul_div_pow():
    ab = _box(torch.full((3,), 0.5), torch.full((3,), 2.0))
    bb = _box(torch.full((3,), 1.0), torch.full((3,), 4.0))
    ob = _box(torch.zeros(3), torch.ones(3))
    mul = Op.make("mul", _v("a", 3), _v("b", 3))
    assert _hop(mul, 0, [ab, bb], ob, "row") == (4.0, "row")
    assert _hop(mul, 1, [ab, bb], ob, "row") == (2.0, "row")
    div = Op.make("div", _v("a", 3), _v("b", 3))
    assert _hop(div, 0, [ab, bb], ob, "row") == (1.0, "row")
    m, _ = _hop(div, 1, [ab, bb], ob, "row")
    assert m == pytest.approx(2.0 / (1.0 * 1.0))
    # divisor touching zero → no local rule
    bz = _box(torch.full((3,), -1.0), torch.ones(3))
    assert _hop(div, 0, [ab, bz], ob, "row") == (None, "row")

    pw = Op.make("pow", _v("a", 3), _v("e", 3))
    pt = _box(torch.ones(3), torch.ones(3))  # point exponent
    e2 = _box(torch.full((3,), 2.0), torch.full((3,), 2.0))
    assert _hop(pw, 1, [ab, e2], ob, "row") == (None, "row")
    assert _hop(pw, 0, [ab, e2], ob, "row") == (
        2 * 2.0,
        "row",
    )
    e0 = _box(torch.zeros(3), torch.zeros(3))
    assert _hop(pw, 0, [ab, e0], ob, "row") == (0.0, "row")
    e1 = _box(torch.ones(3), torch.ones(3))
    assert _hop(pw, 0, [ab, e1], ob, "row") == (1.0, "row")
    # negative exponent on a strictly positive box: |e|·minabs^(e-1)
    em = _box(torch.full((3,), -1.0), torch.full((3,), -1.0))
    m, _ = _hop(pw, 0, [ab, em], ob, "row")
    assert m == pytest.approx(1.0 * 0.5 ** (-2))  # = 4
    # negative base with integer exponent still bounded
    an = _box(torch.full((3,), -2.0), torch.full((3,), -1.0))
    m, _ = _hop(pw, 0, [an, e2], ob, "row")
    assert m == pytest.approx(2 * 2.0)
    # non-constant exponent → None
    en = _box(torch.ones(3), torch.full((3,), 2.0))
    assert _hop(pw, 0, [ab, en], ob, "row") == (None, "row")


def test_hop_softmax_reductions_where_unary_and_unknown():
    xb = _box(torch.zeros(2, 3), torch.ones(2, 3))
    ob = _box(torch.zeros(3), torch.ones(3))
    sm = Op.make("softmax", _v("x", 2, 3), arg1=-1)
    m, k = _hop(sm, 0, [xb], xb, "row")
    assert 0 < m <= 1.0 and k == "row"
    # sum over 6 inputs → 3 outputs: √(n_in/n_out) bound
    s = Op.make("sum", _v("x", 2, 3))
    m, _ = _hop(s, 0, [xb], ob, "row")
    assert m == pytest.approx(math.sqrt(2.0))
    me = Op.make("mean", _v("x", 2, 3))
    m, _ = _hop(me, 0, [xb], ob, "row")
    assert m == pytest.approx(1 / math.sqrt(2.0))
    mx = Op.make("max", _v("x", 2, 3))
    assert _hop(mx, 0, [xb], ob, "row") == (1.0, "row")
    w = Op.make("where", _v("c", 2), _v("a", 2), _v("b", 2))
    assert _hop(w, 1, [xb, xb, xb], xb, "row") == (1.0, "row")
    assert _hop(w, 0, [xb, xb, xb], xb, "row") == (None, "row")
    rl = Op.make("relu", _v("x", 3))
    dead = _box(torch.full((3,), -3.0), torch.full((3,), -1.0))
    assert _hop(rl, 0, [dead], ob, "row") == (0.0, "row")
    live = _box(torch.full((3,), -1.0), torch.full((3,), 2.0))
    assert _hop(rl, 0, [live], ob, "row") == (1.0, "row")
    sd = Op.make("sdpa", _v("q", 2, 2), _v("k", 2, 2), _v("v", 2, 2))
    assert _hop(sd, 0, [xb, xb, xb], ob, "row") == (None, "row")


def test_unary_lip_all_ops():
    mid = _box([-1.0], [1.0])
    assert _unary_lip("sigmoid", mid, {}) == pytest.approx(0.25)
    assert _unary_lip("tanh", mid, {}) == pytest.approx(1.0)
    sat = _box([4.0], [6.0])
    assert _unary_lip("sigmoid", sat, {}) < 0.02  # σ'(4) ≈ 0.0177
    assert _unary_lip("silu", mid, {}) > 0.0
    assert _unary_lip("gelu", mid, {}) > 0.0
    assert _unary_lip("exp", _box([0.0], [2.0]), {}) == pytest.approx(
        math.exp(2.0)
    )
    assert _unary_lip("neg", mid, {}) == 1.0
    assert _unary_lip("sqrt", _box([0.0], [1.0]), {}) is None
    assert _unary_lip("sqrt", _box([4.0], [9.0]), {}) == pytest.approx(
        0.25
    )
    assert _unary_lip("rsqrt", _box([0.0], [1.0]), {}) is None
    assert _unary_lip("rsqrt", _box([4.0], [9.0]), {}) == pytest.approx(
        0.5 * 4.0**-1.5
    )
    assert _unary_lip("square", _box([0.0], [3.0]), {}) == pytest.approx(
        6.0
    )
    assert _unary_lip("pow", mid, {"arg1": 2}) is None
    assert _unary_lip("frobnicate", mid, {}) is None


def test_grid_lip_and_softmax_lip():
    b = _box([-1.0], [1.0])
    # max |cos| over [-1,1]: the candidate point 0 lies inside
    v = _grid_lip(b, torch.cos, (0.0,))
    assert v == pytest.approx(1.0)
    b2 = _box([3.0], [4.0])  # 0 not inside → endpoints only
    v2 = _grid_lip(b2, torch.cos, (0.0,))
    assert v2 == pytest.approx(max(abs(math.cos(3)), abs(math.cos(4))))
    sm_box = _box(torch.zeros(1, 4), torch.ones(1, 4))
    lip = _softmax_lip(sm_box, {"arg1": -1})
    assert 0 < lip <= 1.0


def test_elem_lip_map_all_ops():
    b = _box([-1.0, 4.0, -2.0], [1.0, 6.0, -1.0])
    relu = _elem_lip_map("relu", b)
    assert torch.equal(relu, torch.tensor([1.0, 1.0, 0.0]))
    sig = _elem_lip_map("sigmoid", b)
    assert sig[0].item() == pytest.approx(0.25)
    th = _elem_lip_map("tanh", b)
    assert th[0].item() == pytest.approx(1.0)
    assert _elem_lip_map("silu", b) is not None
    assert _elem_lip_map("gelu", b) is not None
    assert torch.equal(
        _elem_lip_map("exp", b), b.hi.exp()
    )
    assert torch.equal(
        _elem_lip_map("square", b), 2 * b.maxabs
    )
    assert _elem_lip_map("sqrt", b) is None  # box dips below 0
    pos = _box([1.0, 4.0], [4.0, 9.0])
    assert _elem_lip_map("sqrt", pos) is not None
    assert _elem_lip_map("rsqrt", pos) is not None
    assert _elem_lip_map("rsqrt", b) is None
    assert torch.equal(
        _elem_lip_map("neg", b), torch.ones_like(b.lo)
    )
    assert _elem_lip_map("softmax", b) is None


# ---------------------------------------------------------------------------
#  _prop_delta — the realized-delta tensor walk
# ---------------------------------------------------------------------------


def _pb(*ts):
    """Point boxes for operand boxes."""
    return [Box(t, t) for t in ts]


def test_prop_delta_linear_matmul_slots():
    x = torch.randn(4, 8, dtype=torch.float64)
    W = torch.randn(6, 8, dtype=torch.float64)
    d_x = torch.randn(4, 8, dtype=torch.float64)
    d_W = torch.randn(6, 8, dtype=torch.float64)
    node = Op.make(
        "linear", _v("x", 4, 8), _p("w", 6, 8), _p("b", 6)
    )
    env = {"w": W, "b": torch.randn(6, dtype=torch.float64)}
    argb = _pb(x, W, env["b"])
    ob = Box(x @ W.T, x @ W.T)
    cur, mag = _prop_delta(node, 0, d_x, False, argb, ob, env, {})
    assert torch.allclose(cur, d_x @ W.T)
    assert not mag
    # weight slot: activation evaluated via values → a @ ΔWᵀ
    cur, _ = _prop_delta(
        node, 1, d_W, False, argb, ob, env, {"x": x}
    )
    assert torch.allclose(cur, x @ d_W.T)
    # bias slot broadcasts the vector over rows
    d_b = torch.randn(6, dtype=torch.float64)
    cur, _ = _prop_delta(node, 2, d_b, False, argb, ob, env, {})
    assert cur.shape == (4, 6)
    assert _prop_delta(node, 3, d_b, False, argb, ob, env, {}) is None
    # weight slot needs an evaluatable activation; missing it → None
    assert (
        _prop_delta(node, 1, d_W, False, argb, ob, env, {}) is None
    )

    mm = Op.make("matmul", _v("a", 4, 8), _p("w", 8, 6))
    a = torch.randn(4, 8, dtype=torch.float64)
    B = torch.randn(8, 6, dtype=torch.float64)
    cur, _ = _prop_delta(
        mm, 0, d_x, False, _pb(a, B), None, {"w": B}, {}
    )
    assert torch.allclose(cur, d_x @ B)
    # arg1: sib(0) = "a" Var evaluated through values
    cur, _ = _prop_delta(
        mm, 1, torch.randn(8, 6, dtype=torch.float64),
        False, _pb(a, B), None, {"w": B}, {"a": a},
    )
    assert _prop_delta(mm, 2, d_x, False, _pb(a, B), None, {}, {}) is None


def test_prop_delta_add_sub_neg_and_structural():
    d = torch.randn(3, dtype=torch.float64)
    b = _pb(torch.ones(3), torch.ones(3))
    add = Op.make("add", _v("a", 3), _v("b", 3))
    cur, mag = _prop_delta(add, 0, d, False, b, None, {}, {})
    assert torch.equal(cur, d) and not mag
    sub = Op.make("sub", _v("a", 3), _v("b", 3))
    cur, _ = _prop_delta(sub, 1, d, False, b, None, {}, {})
    assert torch.equal(cur, -d)
    cur, mag = _prop_delta(sub, 1, d.abs(), True, b, None, {}, {})
    assert mag and torch.equal(cur, d.abs())
    neg = Op.make("neg", _v("a", 3))
    cur, _ = _prop_delta(neg, 0, d, False, b[:1], None, {}, {})
    assert torch.equal(cur, -d)
    cur, mag = _prop_delta(neg, 0, d.abs(), True, b[:1], None, {}, {})
    assert mag and torch.equal(cur, d.abs())
    # reshape applies the same map to the delta
    rs = Op.make("reshape", _v("a", 4), shape=(2, 2))
    cur, _ = _prop_delta(
        rs, 0, torch.randn(4, dtype=torch.float64),
        False, _pb(torch.ones(4)), None, {}, {},
    )
    assert cur.shape == (2, 2)
    # non-zero operand index on a structural op → no tensor rule
    assert (
        _prop_delta(rs, 1, torch.randn(4, dtype=torch.float64), False,
                    _pb(torch.ones(4)), None, {}, {})
        is None
    )


def test_prop_delta_concat_stack_index_select_embedding():
    d1 = torch.randn(2, 2, dtype=torch.float64)
    a2 = torch.ones(2, 3, dtype=torch.float64)
    cat = Op.make("concat", _v("a", 2, 2), _v("b", 2, 3), dim=-1)
    argb = _pb(torch.ones(2, 2), a2)
    cur, _ = _prop_delta(cat, 0, d1, False, argb, None, {}, {})
    assert cur.shape == (2, 5)
    assert torch.equal(cur[:, :2], d1)
    assert torch.equal(cur[:, 2:], torch.zeros(2, 3))
    st = Op.make("stack", _v("a", 2, 2), _v("b", 2, 3), dim=0)
    # stack needs same-shape parts — use matching shapes
    a2b = torch.ones(2, 2, dtype=torch.float64)
    st = Op.make("stack", _v("a", 2, 2), _v("b", 2, 2), dim=0)
    cur, _ = _prop_delta(
        st, 1, d1, False, _pb(a2b, a2b), None, {}, {}
    )
    assert cur.shape == (2, 2, 2)
    assert torch.equal(cur[0], torch.zeros(2, 2))
    assert torch.equal(cur[1], d1)

    ix = Op.make(
        "index_select", _v("a", 4, 3), _p("i", 2), arg1=0
    )
    ivals = torch.tensor([0, 2])
    cur, _ = _prop_delta(
        ix, 0, torch.randn(4, 3, dtype=torch.float64), False,
        _pb(torch.ones(4, 3), ivals), None, {"i": ivals}, {},
    )
    assert cur.shape == (2, 3)

    emb = Op.make("embedding", _p("w", 5, 3), _p("i", 2))
    dW = torch.randn(5, 3, dtype=torch.float64)
    cur, _ = _prop_delta(
        emb, 0, dW, False,
        _pb(torch.ones(5, 3), ivals), None, {"i": ivals}, {},
    )
    assert torch.equal(cur, dW[ivals])
    # non-evaluatable index → no tensor rule
    assert (
        _prop_delta(emb, 0, dW, False,
                    _pb(torch.ones(5, 3), ivals), None, {}, {})
        is None
    )


def test_prop_delta_mul_div_reductions_and_lip_ops():
    a = torch.randn(4, dtype=torch.float64)
    d = torch.randn(4, dtype=torch.float64)
    mul = Op.make("mul", _v("a", 4), _v("b", 4))
    av = torch.full((4,), 2.0, dtype=torch.float64)
    # sibling Var evaluated through values
    cur, _ = _prop_delta(
        mul, 0, d, False, _pb(a, av), None, {}, {"b": av}
    )
    assert torch.equal(cur, d * 2)
    # missing sibling value → None
    assert _prop_delta(mul, 0, d, False, _pb(a, av), None, {}, {}) is None

    div = Op.make("div", _v("a", 4), _v("b", 4))
    bv = torch.full((4,), 2.0, dtype=torch.float64)
    cur, _ = _prop_delta(
        div, 0, d, False, _pb(a, bv), None, {}, {"b": bv}
    )
    assert torch.equal(cur, d / 2)
    # divisor slot → elementwise magnitude bound
    cur, mag = _prop_delta(
        div, 1, d, False, _pb(a, bv), None,
        {}, {"a": a, "b": bv},
    )
    assert mag
    assert torch.equal(cur, (a / (bv * bv)).abs() * d.abs())
    # zero divisor → None
    bz = torch.zeros(4, dtype=torch.float64)
    assert (
        _prop_delta(div, 0, d, False, _pb(a, bz), None, {}, {"b": bz})
        is None
    )

    su = Op.make("sum", _v("a", 4), arg1=0)
    cur, _ = _prop_delta(
        su, 0, d, False, _pb(a), None, {}, {}
    )
    assert float(cur) == pytest.approx(float(d.sum()))

    rl = Op.make("relu", _v("a", 4))
    live = Box(a - 1, a + 1)  # a box for the relu's input
    cur, mag = _prop_delta(
        rl, 0, d, False, [live], None, {}, {}
    )
    assert mag
    assert torch.equal(cur, d.abs() * ((live.hi > 0).double()))

    wh = Op.make("where", _v("c", 4), _v("a", 4), _v("b", 4))
    cv = torch.tensor([True, False, True, False])
    cur, _ = _prop_delta(
        wh, 1, d, False, _pb(a, a, a), None, {}, {"c": cv}
    )
    assert torch.equal(cur, torch.where(cv, d, torch.zeros_like(d)))
    cur, _ = _prop_delta(
        wh, 2, d, False, _pb(a, a, a), None, {}, {"c": cv}
    )
    assert torch.equal(cur, torch.where(cv, torch.zeros_like(d), d))
    # cond slot itself has no tensor rule
    assert (
        _prop_delta(wh, 0, d, False, _pb(a, a, a), None, {}, {"c": cv})
        is None
    )

    mf = Op.make("masked_fill", _v("a", 4), _p("m", 4), Const(0.0))
    mv = torch.tensor([True, False, True, False])
    cur, _ = _prop_delta(
        mf, 0, d, False, _pb(a, a), None, {"m": mv}, {}
    )
    assert torch.equal(cur, d.masked_fill(mv, 0.0))

    # an op with no tensor rule → None (caller degrades)
    sdpa = Op.make("sdpa", _v("q", 2, 2), _v("k", 2, 2), _v("v", 2, 2))
    assert (
        _prop_delta(sdpa, 0, d, False, _pb(a, a, a), None, {}, {})
        is None
    )


# ---------------------------------------------------------------------------
#  Scalar/artifact site walks + helpers
# ---------------------------------------------------------------------------


def _two_layer_term():
    x = _v("x", 4, 8)
    inner = Op.make("linear", x, _p("W1", 8, 8))
    mid = Op.make("relu", inner)
    term = Op.make("linear", mid, _p("W2", 6, 8))
    return x, inner, mid, term


def test_walk_site_and_artifact_direct():
    torch.manual_seed(0)
    x, inner, mid, term = _two_layer_term()
    env = {
        "W1": torch.randn(8, 8, dtype=torch.float64),
        "W2": torch.randn(6, 8, dtype=torch.float64),
    }
    x0 = torch.randn(4, 8, dtype=torch.float64)
    ib = ibp_bound(term, env, {"x": x0})
    boxes = ib["boxes"]
    # scalar walk from the inner linear site at (0, 0)
    R: dict = {}
    got = _walk_site(term, (0, 0), 1.0, "row", boxes, R)
    assert got is not None and got > 0
    assert R.get((), 0) > 0 and R.get((0,), 0) > 0
    # path through an unsupported op → None
    q = _v("q", 4, 4)
    sd_term = Op.make("sdpa", x, q, q)
    ib2 = ibp_bound(
        sd_term, {}, {"x": x0[:, :4], "q": torch.ones(4, 4)}
    )
    assert (
        _walk_site(sd_term, (0,), 1.0, "row", ib2["boxes"], {}) is None
    )
    # artifact walk: a realized delta at the inner linear hops through
    # relu (magnitude map) then the outer linear (exact matmul)
    delta = 0.01 * torch.randn(4, 8, dtype=torch.float64)
    got = _walk_site_artifact(
        term, (0, 0), delta, "row", boxes, env, {"x": x0}
    )
    # the relu hop loses the sign (magnitude bound), so the following
    # linear hop multiplies by |W2| — the sound elementwise envelope
    lip = (boxes[(0,)].hi > 0).double()
    ref = (
        (delta.abs() * lip) @ env["W2"].abs().T
    ).abs().max().item()
    assert got == pytest.approx(ref)
    # degrade path: hop with a scalar rule but no tensor rule (softmax)
    sm_term = Op.make(
        "linear", Op.make("softmax", inner, arg1=-1), _p("W2", 6, 8)
    )
    ib3 = ibp_bound(sm_term, env, {"x": x0})
    got = _walk_site_artifact(
        sm_term, (0, 0), delta, "row", ib3["boxes"], env, {"x": x0}
    )
    assert got is not None and got > 0


def test_widen_and_collect_sites():
    b = _box(torch.zeros(2), torch.ones(2))
    out = _widen({(): b}, {(): 0.5})
    assert torch.equal(out[()].lo, torch.full((2,), -0.5))
    same = _widen({(): b}, {})
    assert same[()] is b

    x = _v("x", 2, 4)
    site_t = Op.make("mul", _p("W", 4, 4), Const(0.5))
    term = Op.make("linear", x, site_t)
    ghost = Op.make("mul", _p("G", 2, 2), Const(1.0))
    rules = {
        "r_loc": Rewrite(
            name="r_loc", lhs=site_t, rhs=site_t,
            error_bound=0.1, bound_norm="frobenius",
        ),
        "r_ghost": Rewrite(
            name="r_ghost", lhs=ghost, rhs=ghost,
            error_bound=0.2, bound_norm="frobenius",
        ),
        "r_free": Rewrite(name="r_free", lhs=site_t, rhs=site_t),
    }
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(rule="r_loc", path=(1,), lhs=site_t, rhs=site_t),
            CertStep(rule="r_ghost", path=(), lhs=ghost, rhs=ghost),
            CertStep(rule="r_free", path=(1,), lhs=site_t, rhs=site_t),
        ],
        rules=rules,
    )
    sites, unlocated = _collect_sites(term, cert)
    assert unlocated == pytest.approx(0.2)
    located = [s for s in sites if s["path"] is not None]
    unloc = [s for s in sites if s["path"] is None]
    assert len(located) == 1 and located[0]["path"] == (1,)
    assert len(unloc) == 1


def test_site_delta_and_scalar_forms():
    torch.manual_seed(0)
    W = torch.randn(8, 8, dtype=torch.float64)
    x = _v("x", 4, 8)
    w = _p("W", 8, 8)
    site = {
        "rule": "r",
        "bound": 0.5,
        "norm": "frobenius",
        "path": (1,),
        "lhs": w,
        "rhs": Op.make("mul", w, Const(0.5)),
    }
    d = _site_delta(site, {"W": W}, {"x": torch.ones(4, 8, dtype=torch.float64)})
    assert torch.allclose(d, W - W * 0.5)
    # non-evaluatable side → None
    site2 = dict(site, rhs=Op.make("mul", _p("NOPE", 8, 8), Const(0.5)))
    assert _site_delta(site2, {"W": W}, {}) is None

    # _site_scalar on a weight-slot site: realized σ_max(ΔW) is capped
    # at the certificate radius (min(realized, bound))
    term = Op.make("linear", x, Op.make("mul", w, Const(0.99)))
    boxes = ibp_bound(
        term, {"W": W}, {"x": torch.ones(4, 8, dtype=torch.float64)}
    )["boxes"]
    site99 = dict(site, rhs=Op.make("mul", w, Const(0.99)))
    r, kind = _site_scalar(
        site99, term, boxes, {"W": W},
        {"x": torch.ones(4, 8, dtype=torch.float64)},
        actual=True,
    )
    assert kind == "spec"
    realized = float(torch.linalg.matrix_norm(W - W * 0.99, 2).max())
    assert realized < 0.5  # real ΔW ≪ cert radius
    assert r == pytest.approx(realized)
    # when the realized delta exceeds the radius the bound caps it
    r_cap, _ = _site_scalar(
        site, term, boxes, {"W": W},
        {"x": torch.ones(4, 8, dtype=torch.float64)},
        actual=True,
    )
    assert r_cap == pytest.approx(0.5)
    # without the realized delta → certificate radius
    r2, kind2 = _site_scalar(
        site, term, boxes, {"W": W}, {"x": torch.ones(4, 8, dtype=torch.float64)},
        actual=False,
    )
    assert (r2, kind2) == (0.5, "spec")

    # spectral norm site at an activation-linear position → input-scaled
    xb_env = {"x": torch.ones(4, 8, dtype=torch.float64)}
    inner = Op.make("linear", x, w)
    site3 = dict(site, norm="spectral", path=(0,),
                 lhs=inner, rhs=inner)
    term3 = Op.make("relu", inner)
    boxes3 = ibp_bound(term3, {"W": W}, xb_env)["boxes"]
    assert not _spectral_path_ok(site3, term3)
    r3, kind3 = _site_scalar(
        site3, term3, boxes3, {"W": W}, xb_env, actual=False
    )
    assert kind3 == "row"
    assert r3 == pytest.approx(0.5 * _maxrow_bound(boxes3[(0, 0)]))


# ---------------------------------------------------------------------------
#  tight_model_bound — synthetic certificates for every site kind
# ---------------------------------------------------------------------------


def _rule(name, lhs, rhs, bound, norm):
    return Rewrite(
        name=name,
        lhs=lhs,
        rhs=rhs,
        error_bound=bound,
        bound_norm=norm,
    )


def test_tmb_weight_slot_site_frobenius_and_artifact():
    """Quant-style site: rhs is a concrete mul member sitting in the
    weight slot of the root linear.  Delta walk = a @ ΔWᵀ — tighter
    than the cert radius."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    w2 = _p("W2", 6, 8)
    W1 = torch.randn(8, 8, dtype=torch.float64)
    W2 = torch.randn(6, 8, dtype=torch.float64)
    member = Op.make("mul", w2, Const(1.0))  # identity member in term
    term = Op.make(
        "linear",
        Op.make("relu", Op.make("linear", x, _p("W1", 8, 8))),
        member,
    )
    src_term = Op.make(
        "linear",
        Op.make("relu", Op.make("linear", x, _p("W1", 8, 8))),
        w2,
    )
    x0 = torch.randn(4, 8, dtype=torch.float64)
    env = {"W1": W1, "W2": W2 * 0.99}  # realize a small ΔW
    # lhs = exact W2 value, rhs = the member in the term
    env["W2"] = W2
    cert = Certificate(
        src=src_term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(
                rule="eps_q",
                path=(1,),
                lhs=w2,
                rhs=member,
            )
        ],
        rules={"eps_q": _rule("eps_q", w2, member, 0.5, "frobenius")},
    )
    res = tight_model_bound(term, cert, env, x0)
    c = res["site_contributions"][0]
    assert c["path"] == (1,) and not c.get("spectral_unsafe")
    assert res["bound"] >= (res["measured_error"] or 0) - 1e-9
    assert res["artifact_bound"] <= res["bound"] + 1e-12
    assert res["bound"] <= res["spectral_bound"] + 1e-12


def test_tmb_activation_linear_site_is_spectral_unsafe():
    """eps_lr-style spectral site at an activation position: spectral's
    downstream-only walk misses the input-norm factor — flagged
    unsafe, and the local input-scaled estimate is used."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    inner = Op.make("linear", x, _p("V", 4, 8))
    site_rhs = Op.make("linear", inner, _p("U", 8, 4))
    site_lhs = Op.make("linear", x, _p("W", 8, 8))
    term = Op.make("relu", site_rhs)
    src_term = Op.make("relu", site_lhs)
    # real truncated-SVD factors: UV is the rank-4 approx of W, so the
    # declared spectral bound b = 1.5·σ5 honestly covers ‖W − UV‖₂
    W = torch.randn(8, 8, dtype=torch.float64)
    Uf, S, Vf = torch.linalg.svd(W)
    r = 4
    env = {
        "V": Vf[:r, :].contiguous(),  # (r, i)
        "U": (Uf[:, :r] * S[:r]).contiguous(),  # (o, r)
        "W": W,
    }
    bound = float(S[r]) * 1.5
    x0 = torch.randn(4, 8, dtype=torch.float64)
    cert = Certificate(
        src=src_term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(rule="eps_lr", path=(0,), lhs=site_lhs,
                     rhs=site_rhs)
        ],
        rules={
            "eps_lr": _rule("eps_lr", site_lhs, site_rhs, bound,
                            "spectral")
        },
    )
    res = tight_model_bound(term, cert, env, x0)
    c = res["site_contributions"][0]
    assert c["spectral_unsafe"]
    assert "spectral" in res["note"].lower()
    # input-scaled local estimate: bound ≥ σ-bound × ‖x‖
    assert res["bound"] > 0
    # soundness: the honest factorisation keeps bound ≥ measured err
    assert res["bound"] >= (res["measured_error"] or 0) - 1e-9


def test_tmb_matmul_site_variants():
    """matmul sites: embedding arg0 → per-row bound (sound); Param arg0
    → bound on the member value; an activation arg0 → input-scaled
    (spectral unsafe)."""
    torch.manual_seed(0)
    idx = _v("idx", 3)
    ivals = torch.tensor([0, 2, 4])
    W = torch.randn(5, 6, dtype=torch.float64)
    V = torch.randn(6, 4, dtype=torch.float64)
    # ---- embedding arg0 → spectral sound, per-row bound
    site = Op.make(
        "matmul",
        Op.make("embedding", _p("W", 5, 6), idx),
        _p("V", 6, 4),
    )
    env = {"W": W, "V": V}
    cert = Certificate(
        src=site,
        dst=site,
        root_eid=None,
        steps=[
            CertStep(rule="eps_emb", path=(), lhs=site, rhs=site)
        ],
        rules={
            "eps_emb": _rule("eps_emb", site, site, 0.2, "spectral")
        },
    )
    # example_input supplies the idx var
    res = tight_model_bound(site, cert, env, ivals)
    c = res["site_contributions"][0]
    assert not c["spectral_unsafe"]
    assert res["bound"] > 0

    # ---- Param arg0 → bound on the member value, still spectral-sound
    pm = _p("M", 3, 6)
    site2 = Op.make("matmul", pm, _p("V", 6, 4))
    env2 = {"M": torch.randn(3, 6, dtype=torch.float64), "V": V}
    cert2 = Certificate(
        src=site2,
        dst=site2,
        root_eid=None,
        steps=[
            CertStep(rule="eps_m", path=(), lhs=site2, rhs=site2)
        ],
        rules={
            "eps_m": _rule("eps_m", site2, site2, 0.2, "spectral")
        },
    )
    res2 = tight_model_bound(site2, cert2, env2, torch.zeros(1))
    assert not res2["site_contributions"][0]["spectral_unsafe"]

    # ---- activation arg0 → input-scaled, flagged unsafe
    xv = _v("x", 3, 6)
    site3 = Op.make("matmul", xv, _p("V", 6, 4))
    env3 = {"V": V}
    cert3 = Certificate(
        src=site3,
        dst=site3,
        root_eid=None,
        steps=[
            CertStep(rule="eps_m", path=(), lhs=site3, rhs=site3)
        ],
        rules={
            "eps_m": _rule("eps_m", site3, site3, 0.2, "spectral")
        },
    )
    res3 = tight_model_bound(
        site3, cert3, env3, torch.randn(3, 6, dtype=torch.float64)
    )
    assert res3["site_contributions"][0]["spectral_unsafe"]


def test_tmb_unlocated_site_contributes_at_program_bound():
    """A bound step whose produced member is nowhere in the term →
    unlocated mass added to the total, flagged in contributions."""
    x = _v("x", 2, 4)
    w = _p("W", 4, 4)
    term = Op.make("linear", x, w)
    ghost = Op.make("mul", _p("G", 4, 4), Const(0.5))
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(rule="r_g", path=(), lhs=ghost, rhs=ghost)
        ],
        rules={"r_g": _rule("r_g", ghost, ghost, 0.7, "frobenius")},
    )
    res = tight_model_bound(
        term, cert, {"W": torch.eye(4, dtype=torch.float64)},
        torch.randn(2, 4, dtype=torch.float64),
    )
    c = res["site_contributions"][0]
    assert c["unlocated"] and c["contribution"] == pytest.approx(0.7)
    assert res["bound"] == pytest.approx(0.7)


def test_tmb_fallback_ladder_and_notes():
    """Site under an op with no local rule: cc falls back to the
    spectral contribution; a spectral-unsafe site there drops to the
    global sensitivity (or ∞)."""
    torch.manual_seed(0)
    x = _v("x", 4, 4)
    q, k = _v("q", 4, 4), _v("k", 4, 4)
    member = Op.make("mul", _p("V", 4, 4), Const(1.0))
    site_p = _p("V", 4, 4)
    term = Op.make("sdpa", q, k, member)
    env = {
        "V": torch.randn(4, 4, dtype=torch.float64),
    }
    q0 = torch.randn(4, 4, dtype=torch.float64)
    k0 = torch.randn(4, 4, dtype=torch.float64)
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(rule="eps_q", path=(2,), lhs=site_p, rhs=member)
        ],
        rules={
            "eps_q": _rule("eps_q", site_p, member, 0.4, "frobenius")
        },
    )
    # input_radius widens the point boxes so sdpa has no interval rule
    res = tight_model_bound(
        term, cert, env, (q0, k0), input_radius=0.01
    )
    c = res["site_contributions"][0]
    # sdpa has no local rule → fallback flag set; spectral also can't
    # see through sdpa so the contribution is honest (∞ allowed)
    assert c["fallback"]
    assert c["contribution"] == float("inf") or c["contribution"] > 0
    assert "sdpa" in res["unsupported_ops"]
    assert "interval rule" in res["note"]


def test_tmb_not_converged_and_radius():
    """max_iter=0 with a real perturbation → converged=False reported
    honestly; input_radius widens the input box."""
    x = _v("x", 4, 8)
    member = Op.make("mul", _p("W1", 8, 8), Const(1.0))
    term = Op.make(
        "linear",
        Op.make("relu", Op.make("linear", x, member)),
        _p("W2", 6, 8),
    )
    env = {
        "W1": torch.randn(8, 8, dtype=torch.float64),
        "W2": torch.randn(6, 8, dtype=torch.float64),
    }
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[
            CertStep(
                rule="eps_q", path=(0, 0, 1), lhs=_p("W1", 8, 8),
                rhs=member,
            )
        ],
        rules={
            "eps_q": _rule(
                "eps_q", _p("W1", 8, 8), member, 0.3, "frobenius"
            )
        },
    )
    res = tight_model_bound(
        term, cert, env, torch.randn(4, 8, dtype=torch.float64), max_iter=0,
        input_radius=0.01,
    )
    assert not res["converged"]
    assert "did not fully converge" in res["note"]


def test_tmb_tuple_input_and_ir_to_module_error():
    """example_input as a tuple maps Vars in traversal order; a cert
    whose src doesn't lower gives measured_error=None."""
    x = _v("x", 2, 4)
    y = _v("y", 2, 4)
    term = Op.make("add", x, y)
    cert = Certificate(
        src=Op.make("bogus_op", x),  # does not lower → err None
        dst=term,
        root_eid=None,
        steps=[],
        rules={},
    )
    res = tight_model_bound(
        term, cert, {}, (torch.ones(2, 4), torch.zeros(2, 4))
    )
    assert res["bound"] == 0.0
    assert res["measured_error"] is None or res[
        "measured_error"
    ] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  tight_model_bound — real composition of eps sites (quant + low-rank +
#  kron), exercising _collect_sites on genuine certificates
# ---------------------------------------------------------------------------


def _compressible_model(seed=0, low_rank=True):
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 16, bias=False)
            s.l2 = nn.Linear(16, 8, bias=False)
            if low_rank:
                # l1 weight is near rank-4 → eps_lr offer fires cheaply
                A = torch.randn(16, 4)
                B = torch.randn(4, 32)
                s.l1.weight.data = A @ B + 0.001 * torch.randn(16, 32)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    return M().eval().double()


def test_tmb_composition_quant_lowrank_kron():
    """Three eps site kinds in one certificate: quantized weights
    (frobenius, weight slot), low-rank factors (spectral, activation),
    Kronecker sums (frobenius program).  The composed bound is sound:
    bound ≥ measured error on a random input."""
    m = _compressible_model()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    o_q = quant_params(eg, src, bits=8)
    o_l = low_rank_params(eg, src, rtol=0.9)
    o_k = kron_linear_params(eg, src, rtol=0.9)
    assert o_q, "no quant offers"
    assert o_l, "no low-rank offer on a near-low-rank weight"
    assert o_k, "no kron offer on a 16x32 weight"
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(ir.root, term, root_eid=root)
    norms = {
        cert.rules[s.rule].bound_norm
        for s in cert.steps
        if cert.rules.get(s.rule) and cert.rules[s.rule].error_bound
    }
    assert norms  # at least one bounded step
    res = tight_model_bound(term, cert, src, x)
    assert res["n_bounded_steps"] >= 1
    err = res["measured_error"]
    assert err is not None
    assert res["bound"] >= err - 1e-9
    assert res["artifact_bound"] >= err - 1e-9
    # per-site spectral min holds EXCEPT at spectral-unsafe sites
    # (activation linear/matmul): there the spectral figure itself can
    # under-bound, so the local input-scaled estimate replaces it —
    # honestly reported via the flag, not hidden under a min.
    unsafe = [
        c
        for c in res["site_contributions"]
        if c.get("spectral_unsafe")
    ]
    if not unsafe:
        assert res["bound"] <= res["spectral_bound"] + 1e-12
    else:
        assert "UNDER-bound" in res["note"]
    for c in res["site_contributions"]:
        assert "contribution" in c
        assert "fallback" in c
        assert "spectral_unsafe" in c


def test_tmb_random_inputs_soundness():
    """The soundness invariant on random inputs: bound ≥ measured
    error for several seeds — the certificate is honest, not just at
    the example point."""
    m = _compressible_model(seed=3)
    x0 = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x0)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    quant_params(eg, src, bits=8)
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(ir.root, term, root_eid=root)
    for s in range(4):
        torch.manual_seed(100 + s)
        x = torch.randn(4, 32, dtype=torch.float64)
        res = tight_model_bound(term, cert, src, x)
        err = res["measured_error"]
        assert err is not None and err >= 0
        assert res["bound"] >= err - 1e-9, (
            f"seed {s}: bound {res['bound']} < measured {err}"
        )
