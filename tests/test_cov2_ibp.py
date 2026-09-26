# ruff: noqa: RUF002, RUF003
"""Second-wave coverage for catopt.ibp — the last ~50 lines.

``test_ibp.py`` pins the end-to-end contract and ``test_cov_ibp.py``
the machinery tables; this file finishes the module: ``Box``/scalar
edge cases (NaN spectral fallback, the ``_INVALID``/unknown-shape
``_inf_box`` sentinel — a regression pin for a fixed crash), the
interval evaluator's remaining op edges (``sub``, ``pow`` domain
falls, unbound ops, non-tensor bindings, unknown leaves), both
``transpose`` attr spellings in ``_hop``, the residual ``_prop_delta``
op cases, ``_site_scalar``/``_walk_site`` degenerate paths, and
``tight_model_bound``'s fallback ladder (spectral-unsafe → global
sensitivity → ∞) plus the artifact-clamp branch.

Defensive branches deliberately not covered (suggest ``pragma: no
cover``/``no branch``):
- ``_ibp_eval`` unary dispatch with a missing torch binding
  (ibp.py:434→436): every op in the unary set ships a binding — only
  reachable by deleting one (done here via mutation, so the line is
  actually exercised).
- ``elif op in ("concat", "stack", "cat")`` with ``fn is None``
  (451→500): reachable only for ``cat``, which has no binding
  (canonical name is ``concat``) — covered.
- ``_prop_delta`` ``index_select`` ``fn is None`` (1250-1251):
  reachable only with the binding removed — covered via mutation.
- ``c0 = (s["bound"], "row")`` in ``tight_model_bound`` (1574):
  ``_site_scalar`` returns None only when the site's input box is
  absent from the recorded boxes, but ``ibp_bound`` records a box for
  every subterm — unreachable end to end.
"""

import math

import pytest
import torch

import catopt.act_eps  # noqa: F401 — makes aquant resolvable
from catopt import torch_bridge
from catopt.egraph import Certificate, CertStep, Rewrite
from catopt.ibp import (
    Box,
    _fro_bound,
    _hop,
    _inf_box,
    _matmul,
    _maxrow_bound,
    _prop_delta,
    _rearrange_mult,
    _site_delta,
    _site_scalar,
    _spec_bound,
    _spectral_path_ok,
    _walk_site,
    _walk_site_artifact,
    ibp_bound,
    tight_model_bound,
)
from catopt.ir import Const, Op, Param, TensorType, Var


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _box(lo, hi):
    return Box(
        torch.as_tensor(lo, dtype=torch.float64),
        torch.as_tensor(hi, dtype=torch.float64),
    )


def _pb(*ts):
    return [Box(t, t) for t in ts]


def _rule(name, lhs, rhs, bound, norm):
    return Rewrite(
        name=name,
        lhs=lhs,
        rhs=rhs,
        error_bound=bound,
        bound_norm=norm,
    )


# ---------------------------------------------------------------------------
#  Scalar bounds + the _inf_box sentinel (regression pin)
# ---------------------------------------------------------------------------


def test_spec_bound_nan_matrix_falls_back():
    """A NaN box makes ``matrix_norm`` raise — the fallback returns the
    Frobenius norm of the maxabs (NaN propagates honestly, no crash)."""
    m = torch.full((3, 3), float("nan"), dtype=torch.float64)
    b = Box(m, m)
    assert math.isnan(_spec_bound(b))


def test_spec_bound_exactness_on_small_matrices():
    """σ_max over the box is exactly ‖maxabs‖₂ — compared to a direct
    SVD of the corner element."""
    torch.manual_seed(0)
    lo = torch.randn(4, 5, dtype=torch.float64) - 1
    hi = lo + torch.rand(4, 5, dtype=torch.float64) + 0.5
    b = Box(lo, hi)
    expected = float(torch.linalg.norm(b.maxabs.double(), 2))
    assert _spec_bound(b) == pytest.approx(expected, rel=1e-12)
    # _fro_bound is the plain Frobenius norm of the corner
    assert _fro_bound(b) == pytest.approx(
        float(torch.linalg.norm(b.maxabs.double())), rel=1e-12
    )


def test_inf_box_invalid_and_none_dim_shapes():
    """Regression pin: ``_inf_box`` on a provably ill-typed term
    (``_INVALID``) or a dim carrying ``None`` must return the scalar
    ±inf box — not index into a non-tuple shape."""
    # _INVALID: broadcast-incompatible add
    bad = Op.make("add", _v("a", 2, 3), _v("b", 4))
    box = _inf_box(bad)
    assert box.lo.ndim == 0
    assert float(box.lo) == float("-inf")
    assert float(box.hi) == float("inf")
    # a declared None dim
    nv = _v("n", None, 4)
    box2 = _inf_box(nv)
    assert box2.lo.ndim == 0 and float(box2.hi) == float("inf")


def test_inf_box_sentinel_through_ibp():
    """The sentinel propagates end-to-end: an ill-typed div whose
    divisor box straddles zero reports the scalar ±inf box, flagged in
    ``unsupported`` — this is the path that used to crash."""
    x = _v("x", 2, 3)
    y = _v("y", 4)  # broadcast-invalid divisor shape → _INVALID
    res = ibp_bound(
        Op.make("div", x, y),
        {},
        {"x": torch.ones(2, 3), "y": (torch.full((4,), -1.0), torch.ones(4))},
    )
    assert "div:0-in-divisor" in res["unsupported"]
    assert res["lo"].shape == torch.Size([])  # scalar ±inf, not (2,4)
    assert res["width"] == float("inf")


def test_inf_box_var_with_none_dim_unbound():
    """A Var whose declared shape carries ``None`` and has no input
    box: flagged ``var:n`` and given the scalar ±inf box."""
    nv = _v("n", None, 4)
    res = ibp_bound(Op.make("relu", nv), {}, {})
    assert "var:n" in res["unsupported"]
    assert res["lo"].shape == torch.Size([])


# ---------------------------------------------------------------------------
#  _ibp_eval — remaining op edges
# ---------------------------------------------------------------------------


def test_ibp_sub_and_pow_domain_falls():
    x = _v("x", 3)
    y = _v("y", 3)
    # sub op — exact interval subtraction
    res = ibp_bound(
        Op.make("sub", x, y),
        {},
        {
            "x": (torch.ones(3), torch.full((3,), 2.0)),
            "y": (torch.zeros(3), torch.ones(3)),
        },
    )
    assert torch.equal(res["lo"], torch.zeros(3))
    assert torch.equal(res["hi"], torch.full((3,), 2.0))
    # pow: constant non-integer exponent > 0 with a straddling base —
    # no valid interval rule → unsupported
    res = ibp_bound(
        Op.make("pow", x, Const(2.5)),
        {},
        {"x": (torch.full((3,), -1.0), torch.ones(3))},
    )
    assert "pow" in res["unsupported"]
    # negative integer exponent with a straddling base → same fall
    res = ibp_bound(
        Op.make("pow", x, Const(-1.0)),
        {},
        {"x": (torch.full((3,), -1.0), torch.ones(3))},
    )
    assert "pow" in res["unsupported"]


def test_ibp_unary_unsupported_paths():
    """A unary op returning None (domain violation) and a unary op
    with no registered binding both land on the honest-unsupported
    path."""
    x = _v("x", 3)
    # sqrt domain: lo < 0 → _unary None → flagged, ±inf
    res = ibp_bound(
        Op.make("sqrt", x),
        {},
        {"x": (torch.full((3,), -1.0), torch.ones(3))},
    )
    assert "sqrt" in res["unsupported"]
    assert res["width"] == float("inf")
    # binding absent: reachable only if the op has no torch lowering —
    # simulate by removing sqrt's binding (restored after)
    saved = torch_bridge._CORE_TORCH_BINDINGS["sqrt"]
    try:
        del torch_bridge._IR_TO_TORCH["sqrt"]
        res = ibp_bound(
            Op.make("sqrt", x), {}, {"x": torch.ones(3)}
        )
        assert "sqrt" in res["unsupported"]
    finally:
        torch_bridge._IR_TO_TORCH["sqrt"] = saved


def test_ibp_cat_spelling_has_no_binding():
    """``cat`` (the non-canonical spelling) has no torch binding — the
    concat branch sees ``fn is None`` and falls through to the honest
    unsupported path."""
    a, b = _v("a", 2), _v("b", 2)
    res = ibp_bound(
        Op.make("cat", a, b, dim=0),
        {},
        {"a": torch.ones(2), "b": torch.ones(2)},
    )
    assert "cat" in res["unsupported"]
    assert res["width"] == float("inf")


def test_ibp_masked_fill_nonscalar_value_skips():
    """masked_fill's fill value evaluated to a non-scalar tensor —
    ``float(v)`` fails → vv None → the rule abstains honestly."""
    x = _v("x", 3)
    m = _v("m", 3)
    res = ibp_bound(
        Op.make("masked_fill", x, m, _v("vv", 3)),
        {},
        {"x": (torch.zeros(3), torch.ones(3))},
        values={
            "m": torch.tensor([True, False, True]),
            "vv": torch.ones(3),  # multi-element — float(v) raises
        },
    )
    assert "masked_fill" in res["unsupported"]
    assert res["width"] == float("inf")


def test_ibp_non_tensor_binding_and_unknown_leaf():
    """A binding that returns a non-tensor (``aquant``'s (q, s) pair)
    can't be wrapped in a point box — flagged unsupported.  A leaf
    that is none of Var/Param/Const/Op is reported by type name."""
    x = _v("x", 2, 3)
    res = ibp_bound(
        Op.make("aquant", x, bits=8), {}, {"x": torch.ones(2, 3)}
    )
    assert "aquant" in res["unsupported"]
    assert res["width"] == float("inf")
    # raw int as an Op arg → the else-leaf path
    res = ibp_bound(
        Op.make("add", x, 7), {}, {"x": torch.ones(2, 3)}
    )
    assert "leaf:int" in res["unsupported"]


# ---------------------------------------------------------------------------
#  _hop — transpose spellings, rearrangement, arg-count edges
# ---------------------------------------------------------------------------


def test_hop_transpose_canonical_spelling():
    """``transpose`` reads the canonical ``dim0``/``dim1`` spelling — a
    last-two-dim swap preserves the bound; anything else pays the
    rearrangement multiplier."""
    xb = _box(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    ob = _box(torch.randn(2, 8, 4), torch.randn(2, 8, 4))
    x = _v("x", 2, 4, 8)
    tr = Op.make("transpose", x, dim0=1, dim1=2)
    assert _hop(tr, 0, [xb], ob, "spec") == (1.0, "spec")
    # swapping the row dim itself → rearrangement multiplier
    ob2 = _box(torch.randn(8, 4, 2), torch.randn(8, 4, 2))
    tr = Op.make("transpose", x, dim0=0, dim1=2)
    m, k = _hop(tr, 0, [xb], ob2, "row")
    assert k == "row"
    assert m == pytest.approx(math.sqrt(64 / 2))


def test_rearrange_mult_changes_last_dim():
    """A reshape that alters the row (last) dim pays √(numel/d_out)."""
    inb = _box(torch.randn(4, 8), torch.randn(4, 8))
    outb = _box(torch.randn(2, 16), torch.randn(2, 16))
    assert _rearrange_mult(inb, outb) == pytest.approx(math.sqrt(2))
    rs = Op.make("reshape", _v("x", 4, 8), shape=(2, 16))
    m, k = _hop(rs, 0, [inb], outb, "row")
    assert (m, k) == (pytest.approx(math.sqrt(2)), "row")


def test_hop_mul_div_pow_arity_and_domain_edges():
    xb = _box(torch.ones(3), torch.full((3,), 2.0))
    ob = _box(torch.zeros(3), torch.ones(3))
    # mul/div with a non-binary arity → no local rule
    mul1 = Op.make("mul", _v("a", 3))
    assert _hop(mul1, 0, [xb], ob, "row") == (None, "row")
    div1 = Op.make("div", _v("a", 3))
    assert _hop(div1, 0, [xb], ob, "row") == (None, "row")
    # pow: non-integer exponent > 1 on a base straddling zero → None
    pw = Op.make("pow", _v("a", 3), _v("e", 3))
    a_str = _box(torch.full((3,), -1.0), torch.ones(3))
    e25 = _box(torch.full((3,), 2.5), torch.full((3,), 2.5))
    assert _hop(pw, 0, [a_str, e25], ob, "row") == (None, "row")
    # fractional exponent < 1 on a straddling base → minabs ≤ 0 → None
    e05 = _box(torch.full((3,), 0.5), torch.full((3,), 0.5))
    assert _hop(pw, 0, [a_str, e05], ob, "row") == (None, "row")


# ---------------------------------------------------------------------------
#  _site_delta / _site_scalar / _spectral_path_ok — degenerate paths
# ---------------------------------------------------------------------------


def test_site_delta_incompatible_shapes():
    """Both sides evaluate, but their difference can't be formed —
    ``_site_delta`` returns None rather than raise."""
    site = {
        "rule": "r",
        "bound": 1.0,
        "norm": "frobenius",
        "path": (0,),
        "lhs": _p("A", 4),
        "rhs": _p("B", 5),
    }
    env = {"A": torch.ones(4), "B": torch.ones(5)}
    assert _site_delta(site, env, {}) is None


def test_spectral_path_ok_weight_slot_and_leaf_site():
    """A spectral site in a weight slot is sound (bound rows); a
    spectral site whose subterm is neither linear nor matmul keeps the
    default sound verdict."""
    x = _v("x", 4, 8)
    term = Op.make("linear", x, Op.make("mul", _p("W", 8, 8), Const(0.5)))
    site = {"norm": "spectral", "path": (1,), "bound": 0.1}
    assert _spectral_path_ok(site, term) is True
    # leaf (Param) site at a non-weight position
    term2 = Op.make("add", _v("y", 4), _p("P", 4))
    site2 = {"norm": "spectral", "path": (1,), "bound": 0.1}
    assert _spectral_path_ok(site2, term2) is True


def test_site_scalar_weight_slot_vector_delta():
    """Weight-slot site whose realized delta is a vector: the spectral
    radius is max|Δ| capped by the cert bound."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    w = _p("w", 8)
    W = torch.randn(8, dtype=torch.float64)
    member = Op.make("mul", w, Const(0.9))
    term = Op.make("matmul", x, member)
    boxes = ibp_bound(term, {"w": W}, {"x": torch.ones(4, 8)})["boxes"]
    site = {
        "rule": "r",
        "bound": 0.5,
        "norm": "frobenius",
        "path": (1,),
        "lhs": w,
        "rhs": member,
    }
    vals = {"x": torch.ones(4, 8, dtype=torch.float64)}
    r, kind = _site_scalar(site, term, boxes, {"w": W}, vals, actual=True)
    realized = float((W - 0.9 * W).abs().max())
    assert kind == "spec"
    assert r == pytest.approx(min(realized, 0.5))


def test_site_scalar_activation_actual_and_missing_boxes():
    """A frobenius site at an activation position uses the realized
    max-row delta when finite; spectral linear/matmul sites with no
    input box recorded degrade to None."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    w = _p("W", 4, 8)
    member = Op.make("mul", w, Const(0.9))
    term = Op.make("relu", member)
    env = {"W": torch.randn(4, 8, dtype=torch.float64)}
    vals = {"x": torch.ones(4, 8, dtype=torch.float64)}
    boxes = ibp_bound(term, env, vals)["boxes"]
    site = {
        "rule": "r",
        "bound": 0.5,
        "norm": "frobenius",
        "path": (0,),
        "lhs": w,
        "rhs": member,
    }
    r, kind = _site_scalar(site, term, boxes, env, vals, actual=True)
    expected = _maxrow_bound(Box(env["W"] * 0.1, env["W"] * 0.1))
    assert (kind == "row") and r == pytest.approx(expected)

    # spectral site at a linear activation, but no input box → None
    U = _p("U", 8, 8)
    lin = Op.make("linear", x, U)
    term3 = Op.make("relu", lin)
    site3 = {
        "rule": "r",
        "bound": 0.5,
        "norm": "spectral",
        "path": (0,),
        "lhs": lin,
        "rhs": lin,
    }
    assert _site_scalar(site3, term3, {}, {"U": torch.eye(8)}, {}, actual=False) is None

    # spectral site at a matmul activation with no input box → None
    mm = Op.make("matmul", x, _p("V", 8, 4))
    term4 = Op.make("relu", mm)
    site4 = {
        "rule": "r",
        "bound": 0.5,
        "norm": "spectral",
        "path": (0,),
        "lhs": mm,
        "rhs": mm,
    }
    assert _site_scalar(site4, term4, {}, {"V": torch.eye(8)}, {}, actual=False) is None

    # spectral site whose subterm is neither linear nor matmul → b,row
    add_t = Op.make("add", Op.make("relu", x), _p("P", 4, 8))
    site5 = {
        "rule": "r",
        "bound": 0.25,
        "norm": "spectral",
        "path": (0,),
        "lhs": add_t.args[0],
        "rhs": add_t.args[0],
    }
    r5, k5 = _site_scalar(
        site5, add_t, boxes, {"P": torch.zeros(4, 8)}, vals, actual=False
    )
    assert (r5, k5) == (0.25, "row")

    # realized delta non-finite (inf) → falls through to the norm path
    env_inf = {"W": torch.full((4, 8), float("inf"), dtype=torch.float64)}
    boxes_inf = ibp_bound(term, env_inf, vals)["boxes"]
    r6, k6 = _site_scalar(
        site, term, boxes_inf, env_inf, vals, actual=True
    )
    assert (r6, k6) == (0.5, "row")  # cert radius, not the inf delta


# ---------------------------------------------------------------------------
#  _walk_site / _walk_site_artifact — degenerate paths
# ---------------------------------------------------------------------------


def test_walk_site_leaf_and_missing_boxes():
    """A path descending into a leaf returns None (no hop possible);
    so does a walk whose operand boxes were never recorded."""
    x = _v("x", 4, 4)
    term = Op.make("relu", x)
    boxes = ibp_bound(term, {}, {"x": torch.ones(4, 4)})["boxes"]
    # path (0,0) descends into the Var → not an Op → None
    assert _walk_site(term, (0, 0), 1.0, "row", boxes, {}) is None
    # linear term but the weight child's box is missing
    w = _p("W", 4, 4)
    term2 = Op.make("linear", x, w)
    sparse = {
        (): Box(torch.ones(4, 4), torch.ones(4, 4)),
        (0,): Box(torch.ones(4, 4), torch.ones(4, 4)),
    }
    assert _walk_site(term2, (0,), 1.0, "row", sparse, {}) is None


def test_walk_site_artifact_scalar_and_leaf_paths():
    """``_walk_site_artifact`` accepts a scalar (delegates to the
    scalar walk), returns None on a missing delta, and None on a path
    descending into a leaf."""
    x = _v("x", 4, 4)
    w = _p("W", 4, 4)
    term = Op.make("linear", x, w)
    env = {"W": torch.randn(4, 4, dtype=torch.float64)}
    boxes = ibp_bound(term, env, {"x": torch.ones(4, 4)})["boxes"]
    assert (
        _walk_site_artifact(term, (0,), None, "row", boxes, env, {})
        is None
    )
    # scalar delta → _walk_site
    got = _walk_site_artifact(term, (0,), 0.5, "row", boxes, env, {})
    assert got is not None and got >= 0
    # path into a leaf → None
    assert (
        _walk_site_artifact(
            Op.make("relu", x), (0, 0),
            torch.ones(4, 4), "row", {}, {}, {},
        )
        is None
    )


# ---------------------------------------------------------------------------
#  _prop_delta — remaining op cases
# ---------------------------------------------------------------------------


def test_prop_delta_linear_bias_broadcast_failure():
    """Bias delta that can't broadcast to the output shape degrades to
    the unbroadcast delta — still a bound, just unshaped."""
    x = _v("x", 4, 6)
    node = Op.make("linear", x, _p("w", 6, 6), _p("b", 6))
    cur = torch.randn(4, 6, dtype=torch.float64)
    argb = _pb(
        torch.ones(4, 6),
        torch.ones(6, 6),
        torch.ones(6),
    )
    outb = _pb(torch.zeros(6))[0]  # incompatible shape
    res, _ = _prop_delta(node, 2, cur, False, argb, outb, {}, {})
    assert torch.equal(res, cur)


def test_prop_delta_cast_ops_success_and_failure():
    """``float``/cast family apply their torch binding to the delta;
    a binding that raises (type_as missing its type arg) degrades to
    the double delta."""
    d = torch.randn(3, dtype=torch.float64)
    b = _pb(torch.ones(3))
    fl = Op.make("float", _v("a", 3), dtype="float64")
    cur, mag = _prop_delta(fl, 0, d, False, b, None, {}, {})
    assert torch.equal(cur, d.double()) and not mag
    ta = Op.make("type_as", _v("a", 3), _v("t", 3))
    cur2, _ = _prop_delta(
        ta, 0, d, False, _pb(torch.ones(3), torch.ones(3)), None, {}, {}
    )
    assert torch.equal(cur2, d.double())
    # structural op whose binding raises → no tensor rule
    rs = Op.make("reshape", _v("a", 4), shape=(3, 3))
    assert (
        _prop_delta(
            rs, 0, torch.randn(4, dtype=torch.float64),
            False, _pb(torch.ones(4)), None, {}, {},
        )
        is None
    )


def test_prop_delta_index_select_missing_and_bad_index():
    """``index_select`` without a binding → None; a binding whose eval
    raises (non-integer index tensor) → None."""
    ix = Op.make("index_select", _v("a", 4, 3), _p("i", 2), dim=0)
    cur = torch.randn(4, 3, dtype=torch.float64)
    argb = _pb(torch.ones(4, 3), torch.tensor([0.5, 1.5]))
    saved = torch_bridge._CORE_TORCH_BINDINGS["index_select"]
    try:
        del torch_bridge._IR_TO_TORCH["index_select"]
        assert (
            _prop_delta(ix, 0, cur, False, argb, None, {"i": torch.tensor([0, 2])}, {})
            is None
        )
    finally:
        torch_bridge._IR_TO_TORCH["index_select"] = saved
    # index evaluates but isn't valid for index_select (float dtype)
    assert (
        _prop_delta(
            ix, 0, cur, False, argb, None,
            {"i": torch.tensor([0.5, 1.5])}, {},
        )
        is None
    )


def test_prop_delta_div_sum_and_where_missing_sibling():
    d = torch.randn(4, dtype=torch.float64)
    div = Op.make("div", _v("a", 4), _v("b", 4))
    bv = torch.full((4,), 2.0, dtype=torch.float64)
    # divisor slot: numerator can't be evaluated → None
    assert (
        _prop_delta(
            div, 1, d, False, _pb(torch.ones(4), bv), None,
            {}, {"b": bv},
        )
        is None
    )
    # sum with an out-of-range dim → binding raises → None
    su = Op.make("sum", _v("a", 4), dim=9)
    assert (
        _prop_delta(su, 0, d, False, _pb(torch.ones(4)), None, {}, {})
        is None
    )
    # where with a non-evaluatable condition → None
    wh = Op.make("where", _v("c", 4), _v("a", 4), _v("b", 4))
    assert (
        _prop_delta(
            wh, 1, d, False, _pb(torch.ones(4), torch.ones(4), torch.ones(4)),
            None, {}, {},
        )
        is None
    )


# ---------------------------------------------------------------------------
#  tight_model_bound — scalar input, fallback ladder, artifact clamp
# ---------------------------------------------------------------------------


def test_tmb_scalar_example_input():
    """A 0-D example input feeds ``_input_norm``'s scalar branch; the
    bound over an empty cert is 0."""
    x = _v("x")
    x = Var("x", TensorType(()))
    term = Op.make("mul", x, _p("s", 1))
    cert = Certificate(
        src=term, dst=term, root_eid=None, steps=[], rules={}
    )
    res = tight_model_bound(
        term, cert, {"s": torch.ones(1)}, torch.tensor(2.5)
    )
    assert res["bound"] == 0.0
    assert res["n_bounded_steps"] == 0


def test_tmb_fallback_ladder_finite_and_infinite():
    """Spectral-unsafe site with no interval rule on its path:
    conv2d's weight slot resolves via the global Lipschitz product
    (finite); sdpa has no Lipschitz constant at all → ∞, honestly."""
    torch.manual_seed(0)
    x = _v("x", 1, 1, 6, 6)
    z = _v("z", 4, 8)
    site = Op.make("linear", z, _p("U", 8, 8))
    term = Op.make(
        "conv2d", x, Op.make("reshape", site, shape=(1, 1, 3, 3))
    )
    env = {"U": torch.randn(8, 8, dtype=torch.float64)}
    x0 = torch.randn(1, 1, 6, 6, dtype=torch.float64)
    z0 = torch.randn(4, 8, dtype=torch.float64)
    cert = Certificate(
        src=term,
        dst=term,
        root_eid=None,
        steps=[CertStep(rule="eps_lr", path=(1, 0), lhs=site, rhs=site)],
        rules={"eps_lr": _rule("eps_lr", site, site, 0.4, "spectral")},
    )
    res = tight_model_bound(term, cert, env, (x0, z0))
    c = res["site_contributions"][0]
    assert c["fallback"] and c["spectral_unsafe"]
    assert math.isfinite(c["contribution"]) and c["contribution"] > 0

    q, k = _v("q", 4, 4), _v("k", 4, 4)
    site2 = Op.make("linear", z, _p("U", 4, 8))
    term2 = Op.make("sdpa", q, k, site2)
    cert2 = Certificate(
        src=term2,
        dst=term2,
        root_eid=None,
        steps=[
            CertStep(rule="eps_lr", path=(2,), lhs=site2, rhs=site2)
        ],
        rules={
            "eps_lr": _rule("eps_lr", site2, site2, 0.4, "spectral")
        },
    )
    res2 = tight_model_bound(
        term2, cert2, env, (x0[:, :, :4, :4].reshape(4, 4), x0[:, :, :4, :4].reshape(4, 4), z0)
    )
    c2 = res2["site_contributions"][0]
    assert c2["fallback"] and c2["spectral_unsafe"]
    assert c2["contribution"] == float("inf")
    assert res2["bound"] == float("inf")


def test_tmb_artifact_clamped_to_cert_radius():
    """When the realized |lhs−rhs| propagates LARGER than the cert
    radius allows (a deliberately dishonest certificate), the artifact
    contribution clamps down to the radius — never wider than proven."""
    torch.manual_seed(0)
    x = _v("x", 4, 8)
    w = _p("W", 4, 8)
    member = Op.make("mul", w, Const(0.5))
    term = Op.make("relu", Op.make("mul", x, member))
    src_term = Op.make("relu", Op.make("mul", x, w))
    env = {"W": torch.randn(4, 8, dtype=torch.float64)}
    cert = Certificate(
        src=src_term,
        dst=term,
        root_eid=None,
        steps=[CertStep(rule="r", path=(0, 1), lhs=w, rhs=member)],
        rules={
            # radius 0.001 far below the real |0.5·W| → artifact > cert
            "r": _rule("r", w, member, 0.001, "frobenius")
        },
    )
    res = tight_model_bound(
        term, cert, env, torch.randn(4, 8, dtype=torch.float64)
    )
    c = res["site_contributions"][0]
    assert c["artifact_contribution"] == pytest.approx(
        c["contribution"]
    )


# ---------------------------------------------------------------------------
#  Mixed-dtype boxes — regression pin for the dtype-promoting _matmul
# ---------------------------------------------------------------------------


def test_matmul_box_mixed_dtype():
    """fp32 input box × fp64 param box: the centre/radius product must
    promote dtypes instead of erroring — regression pin."""
    alo = torch.zeros(2, 3)  # float32
    ahi = torch.ones(2, 3)
    blo = torch.zeros(3, 4, dtype=torch.float64)
    bhi = torch.ones(3, 4, dtype=torch.float64)
    c = _matmul(Box(alo, ahi), Box(blo, bhi))
    assert c.lo.dtype == torch.float64
    # contains every fp32 sample × fp64 param product
    torch.manual_seed(0)
    for _ in range(50):
        a = alo + torch.rand(2, 3) * (ahi - alo)
        b = blo + torch.rand(3, 4, dtype=torch.float64) * (bhi - blo)
        p = a.double() @ b
        assert (p >= c.lo - 1e-9).all() and (p <= c.hi + 1e-9).all()


def test_ibp_mixed_dtype_term_end_to_end():
    """Same promotion through ``ibp_bound``: fp32 input box through a
    linear with fp64 weights yields a finite fp64 output box."""
    torch.manual_seed(0)
    x = _v("x", 2, 3)
    W = torch.randn(4, 3, dtype=torch.float64)
    res = ibp_bound(
        Op.make("linear", x, _p("W", 4, 3)),
        {"W": W},
        {"x": (torch.zeros(2, 3), torch.ones(2, 3))},
    )
    assert res["lo"].dtype == torch.float64
    assert math.isfinite(res["width"])
    for _ in range(50):
        xs = torch.rand(2, 3)
        ys = torch.nn.functional.linear(xs.double(), W)
        assert (ys >= res["lo"] - 1e-9).all()
        assert (ys <= res["hi"] + 1e-9).all()
