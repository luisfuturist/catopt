"""Coverage-gap tests for catopt.cost.

``tests/test_cost.py`` covers the flagship semantics — broadcasting,
weight dedup, the eps storage axis, the e-graph end-to-end.  This file
drives the per-op ``_flops_of`` arms and the structural edges those
never reach:

* ``_flops_of`` fallbacks: matmul with an unshapeable RHS, conv2d with
  ``groups > 1`` and with non-int weight dims, ``aff_compose`` /
  ``apply`` on short-rank map parts, ``sdpa`` on a rank-2 query;
* ``depth_cost``/``depth_cost_for`` on a provably ill-typed term (the
  launch-not-veto policy) and ``_local_roofline``'s ``_INVALID`` arm;
* ``param_bytes_cost`` storage semantics: numpy ``.size`` and missing
  ``element_size`` fallbacks in ``_param_numel``, ``_folds_to_param``
  on non-``Op`` terms and the ``allow_const=False`` matmul/concat
  rule, ``_fold_ewidth``'s unresolvable-leaf 4.0 default and its
  non-tensor ``None`` leaves, param-only folds billed per
  materialisation (``concat`` re-stores copies; nested folds bill only
  the outermost output), ``by_bytes`` dtype widths;
* ``dag_cost`` on a shared-subterm DAG, the param-only compile-time
  discount and its ``charges_param_only`` opt-out, the ``dag_exact``
  early return (direct and through a ``functools.partial``), and a
  cost fn without a ``memo`` parameter;
* ``_is_strided`` on a non-tuple shape, ``count_cost``/``depth_cost``
  on carrier + view ops;
* ``roofline_cost_for`` keyword calibration and ``CostModel``'s
  matmul/linear arms including the scalar-operand fallbacks.

The one branch listed as unreachable by construction is in
``test_defensive_branch_inventory``.
"""

import functools
import types

import numpy as np
import pytest
import torch

from catopt.cost import (
    _INVALID_COST,
    _LAUNCH_S,
    CostModel,
    _flops_of,
    _fold_ewidth,
    _folds_to_param,
    _is_strided,
    _local_roofline,
    _param_index,
    count_cost,
    dag_cost,
    depth_cost,
    depth_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost,
    param_bytes_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.typing import _INVALID, _shape_of


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


#: An op whose inferred shape is ``None`` — a carrier-internal ``()``
#: member read as a tensor.  Used wherever an unshapeable operand is
#: needed to reach a fallback arm.
UNKNOWN = Op.make("transpose", Const(1.0))


# ---------------------------------------------------------------------------
#  _flops_of — per-op arms and their fallbacks
# ---------------------------------------------------------------------------


def test_flops_matmul_unshapeable_rhs():
    """matmul(x, <None-shaped>) bills 2*n_out with no K to read."""
    t = Op.make("matmul", _v("x", 4, 8), UNKNOWN)
    assert _shape_of(t) == (4, 8)  # first-operand propagation
    assert _flops_of(t) == pytest.approx(2 * 4 * 8)
    assert flops_cost(t) == pytest.approx(2 * 4 * 8)


def test_flops_conv2d_groups_and_fallback():
    x = _v("x", 2, 4, 8, 8)
    w = _p("w", 8, 4, 3, 3)
    # groups=2 halves the kernel work per output channel.
    g1 = Op.make("conv2d", x, w)
    g2 = Op.make("conv2d", x, w, groups=2)
    assert _flops_of(g2) == pytest.approx(_flops_of(g1) / 2)
    # groups as a non-int / <=1 attr changes nothing.
    g1b = Op.make("conv2d", x, w, groups=1)
    assert _flops_of(g1b) == pytest.approx(_flops_of(g1))
    # Non-int kernel dims can't form a K — flat 2*n_out fallback.
    w_holed = _p("wh", 8, 4, None, 3)
    t = Op.make("conv2d", x, w_holed)
    assert _flops_of(t) == pytest.approx(2 * _numel_of(t))


def _numel_of(t) -> int:
    from catopt.typing import _numel

    return _numel(_shape_of(t))


def test_flops_aff_family_fallbacks():
    # aff_compose on a rank-1 map part: no d axis to price against.
    f1 = _v("f1", 4)
    g = _v("g", 4, 4)
    t = Op.make("aff_compose", f1, g)
    assert _shape_of(t) == (4,)
    assert _flops_of(t) == pytest.approx(2 * 4)
    # Same when the last dim's extent is unknown (not an int).
    t2 = Op.make("aff_compose", _v("f2", 4, None), g)
    assert _flops_of(t2) == pytest.approx(2 * 4)
    # apply on a rank-1 map part -> flat 2*n_out.
    t3 = Op.make("apply", f1, _v("h", 4))
    assert _flops_of(t3) == pytest.approx(2 * _numel_of(t3))
    # aff/aff_diag/om packaging is free.
    assert _flops_of(Op.make("aff", g, f1)) == 0.0
    assert _flops_of(Op.make("aff_diag", f1, f1)) == 0.0
    # affd/applyd elementwise carrier arithmetic.
    assert _flops_of(Op.make("affd_compose", f1, f1)) == pytest.approx(
        3 * 4
    )
    assert _flops_of(
        Op.make("applyd", f1, _v("h2", 4))
    ) == pytest.approx(2 * 4)


def test_flops_sdpa_rank2_query():
    """sdpa needs (…, T, d) to price T — a rank-2 q falls back flat."""
    q = _v("q", 4, 8)
    t = Op.make("sdpa", q, _v("k", 4, 8), _v("v", 4, 8))
    assert _flops_of(t) == pytest.approx(4 * _numel_of(t))


def test_flops_default_weight_ops():
    """Ops absent from _OP_FLOPS bill 1 flop per output element, and
    the view-indexed ops (select/getitem/narrow-style) infer real
    shapes for it."""
    x = _v("x", 4, 8)
    for op, expected_shape in (
        ("rms_norm", (4, 8)),
        ("layer_norm", (4, 8)),
        ("narrow", (4, 8)),  # default arm: first-operand shape
        ("max", (4, 8)),
        ("min", (4, 8)),
    ):
        t = Op.make(op, x)
        assert _flops_of(t) == pytest.approx(1.0 * 4 * 8), op
        assert _shape_of(t) == expected_shape, op
    # select removes the indexed axis; getitem passes through.
    sel = Op.make("select", x, dim=0, index=1)
    assert _shape_of(sel) == (8,)
    assert _flops_of(sel) == pytest.approx(8.0)
    sp = Op.make("split", _v("s", 8), sizes=(3, 5), index=0)
    gi = Op.make("getitem", sp, index=0)
    assert _shape_of(gi) == (3,)
    # flatten merges a dim range into one.
    fl = Op.make("flatten", _v("f", 2, 3, 4), start_dim=1)
    assert _shape_of(fl) == (2, 12)
    assert _flops_of(fl) == pytest.approx(24.0)


def test_flops_invalid_term_is_poison():
    bad = Op.make("add", _v("a", 4, 8), _v("b", 4, 7))
    assert _shape_of(bad) is _INVALID
    assert _flops_of(bad) >= _INVALID_COST
    assert flops_cost(bad) >= _INVALID_COST


# ---------------------------------------------------------------------------
#  view / carrier ops under the structural models
# ---------------------------------------------------------------------------


def test_count_and_depth_on_view_and_carrier_ops():
    x = _v("x", 4, 8)
    # View ops and carrier packaging cost nothing under count_cost.
    for op_term in (
        Op.make("transpose", x, arg1=0, arg2=1),
        Op.make("reshape", x, shape=(4, 8)),
        Op.make("aff", _v("A", 8, 8), _v("b", 8)),
        Op.make("om", _v("m", 4, 8), _v("l", 4, 8), _v("a", 4, 8)),
        Op.make("eye", dim=4),
        Op.make("cswap", d1=2, d2=2),
    ):
        assert count_cost(op_term) == 0.0, op_term
    # A view on top of real work counts the work only.
    t = Op.make("transpose", Op.make("neg", x), arg1=0, arg2=1)
    assert count_cost(t) == 1.0
    # depth: a view adds no latency over its child.
    assert depth_cost(t) == pytest.approx(depth_cost(t.args[0]))


def test_launch_aware_view_ops_skip_penalty():
    x = _v("x", 4, 8)
    neg = Op.make("neg", x)
    assert launch_aware_cost(neg) == pytest.approx(
        flops_cost(neg) + 1.0
    )
    tr = Op.make("transpose", neg, arg1=0, arg2=1)
    # The transpose itself launches nothing.
    assert launch_aware_cost(tr) == pytest.approx(
        launch_aware_cost(neg)
    )
    # Memo reuse returns the cached value.
    memo = {}
    first = launch_aware_cost(tr, memo)
    assert launch_aware_cost(tr, memo) == first


# ---------------------------------------------------------------------------
#  depth_cost / _local_roofline — invalid terms
# ---------------------------------------------------------------------------


def test_depth_cost_invalid_term_charges_launch_not_veto():
    bad = Op.make("add", _v("a", 4, 8), _v("b", 4, 7))
    assert _local_roofline(bad) == _INVALID_COST
    # depth is structural: one launch, not the poison price.
    assert depth_cost(bad) == pytest.approx(_LAUNCH_S * 1e9)
    # …and the profile-calibrated closure uses its own launch constant.
    df = depth_cost_for(
        {"tflops": 2.5, "gbps": 89.0, "launch_us": 2.0}
    )
    assert df(bad) == pytest.approx(2e-6 * 1e9)
    assert df.__name__ == "depth_cost_for"
    assert df.profile == {"tflops": 2.5, "gbps": 89.0, "launch_us": 2.0}


def test_depth_cost_zero_arg_op():
    """A childless op has depth == its own latency (default=0.0)."""
    eye = Op.make("eye", dim=4)
    assert depth_cost(eye) == pytest.approx(
        _local_roofline(eye, {}), rel=0.01
    )
    assert depth_cost(eye) >= 0.0


def test_roofline_cost_for_kwarg_calibration():
    """Keyword overrides re-price each op's compute/memory/launch."""
    # A large GEMM is compute-bound at the default profile; a big
    # elementwise op is memory-bound.  Each override moves the price
    # of the term its regime dominates.
    x = _v("x", 512, 512)
    t = Op.make("matmul", x, _p("W", 512, 512))
    ew = Op.make("neg", x)
    default = roofline_cost(t)
    default_ew = roofline_cost(ew)
    assert roofline_cost_for(peak_flops=1e15)(t) != default
    assert roofline_cost_for(peak_bw=1e15)(ew) != default_ew
    assert roofline_cost_for(launch_s=0.0)(t) != default
    # All three overrides at once: pure compute term only.
    only_compute = roofline_cost_for(
        peak_flops=1e12, peak_bw=1e18, launch_s=0.0
    )
    flops = _flops_of(t)
    assert only_compute(t) == pytest.approx(flops / 1e12 * 1e9)
    # A profile object (attrs) and a dict both work.
    prof = types.SimpleNamespace(tflops=2.5, gbps=89.0, launch_us=8.7)
    by_obj = roofline_cost_for(prof)
    by_dict = roofline_cost_for(
        {"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7}
    )
    assert by_obj(t) == pytest.approx(by_dict(t))
    assert by_obj.profile is prof
    # roofline_cost_for() with no profile IS the default.
    assert roofline_cost_for()(t) == pytest.approx(default)


# ---------------------------------------------------------------------------
#  param_bytes_cost — storage semantics
# ---------------------------------------------------------------------------


def test_param_numel_source_tensor_variants():
    x = _v("x", 4, 8)
    W = _p("W", 8, 8)
    t = Op.make("linear", x, W)
    # numpy arrays: .size is an int attribute (no .numel method).
    assert param_bytes_cost(t, {"W": np.zeros((3, 4))}) == 12.0
    # objects exposing numel() but no element_size(): by_bytes uses
    # the fp32 default of 4 bytes.
    class FakeTensor:
        def numel(self):
            return 10

    assert param_bytes_cost(t, {"W": FakeTensor()}) == 10.0
    assert param_bytes_cost(
        t, {"W": FakeTensor()}, by_bytes=True
    ) == 40.0
    # A source object with neither a callable numel() nor a numeric
    # .size falls back to the declared TensorType.
    odd = types.SimpleNamespace(numel=None, size="not-a-number")
    assert param_bytes_cost(t, {"W": odd}) == 64.0
    # torch dtype widths under by_bytes.
    assert param_bytes_cost(
        t, {"W": torch.zeros(8, dtype=torch.float16)}, by_bytes=True
    ) == 16.0
    assert param_bytes_cost(
        t, {"W": torch.zeros(8, dtype=torch.int8)}, by_bytes=True
    ) == 8.0


def test_folds_to_param_non_op_and_const_rules():
    """_folds_to_param's leaf and Const-arities mirror the lowerer."""
    memo: dict = {}
    # Non-Op terms never fold.
    assert _folds_to_param(_v("x", 4), None, memo) is False
    assert _folds_to_param(_p("P", 4), None, memo) is False
    assert _folds_to_param(Const(1.0), None, memo) is False
    # Elementwise ops accept Const operands; matmul/concat do not.
    P = _p("P", 4)
    assert _folds_to_param(
        Op.make("mul", P, Const(2.0)), None, memo
    ) is True
    assert _folds_to_param(
        Op.make("matmul", _p("A", 4, 4), Const(2.0)), None, memo
    ) is False
    assert _folds_to_param(
        Op.make("concat", P, Const(2.0), dim=0), None, memo
    ) is False
    # concat of two resolvable Params folds.
    assert _folds_to_param(
        Op.make("concat", P, _p("Q", 4), dim=0), None, memo
    ) is True
    # Non-foldable ops (reshape) never fold even param-only.
    assert _folds_to_param(
        Op.make("reshape", P, shape=(2, 2)), None, memo
    ) is False
    # Bound source_tensors: a leaf absent from it cannot fold.
    # (Fresh memo per call — the cache keys on the term, not the
    # source dict.)
    A, B = _p("A", 4, 4), _p("B", 4, 4)
    assert (
        _folds_to_param(Op.make("matmul", A, B), {"A": 1.0}, {})
        is False
    )
    assert _folds_to_param(Op.make("matmul", A, B), None, {}) is True


def test_param_bytes_bills_folds_at_output_size():
    """A materialised fold stores its OUTPUT — concat re-stores every
    copy; a folded weight matmul stores the dense product, not the
    operands."""
    x = _v("x", 4, 8)
    W = _p("W", 4, 4)
    # concat(W, W) lowers to one stored (8,4) — occurrences are copies.
    cat = Op.make("concat", W, W, dim=0)
    t = Op.make("matmul", x, cat)
    assert param_bytes_cost(t) == 8 * 4
    # A folded weight product stores only numel(A@B).
    A, B = _p("A", 4, 8), _p("B", 8, 2)
    prod = Op.make("matmul", A, B)
    assert param_bytes_cost(Op.make("matmul", x, prod)) == 4 * 2
    # …while an unfoldable param-only subtree still stores its leaves.
    t2 = Op.make("reshape", _p("R", 4, 5), shape=(4, 5))
    assert param_bytes_cost(t2) == 20.0


def test_param_bytes_shared_fold_billed_once():
    """The same fold object reached twice is one stored tensor —
    ``_fold_memo``/``_build_params`` materialise it a single time.
    (A non-foldable wrapper keeps the two folds as separate reads —
    elementwise parents would themselves fold into one storage.)"""
    W, Q = _p("W", 4, 4), _p("Q", 4, 4)
    cat = Op.make("concat", W, Q, dim=0)  # interned: one object
    shared = Op.make("zwrap", cat, cat)
    assert param_bytes_cost(shared) == 8 * 4
    # Two DISTINCT folds each materialise — billed per occurrence.
    cat2 = Op.make("concat", W, _p("R", 4, 4), dim=0)
    two = Op.make("zwrap", cat, cat2)
    assert param_bytes_cost(two) == 2 * 8 * 4


def test_param_bytes_index_keys():
    """_param_index exposes the two storage entry kinds by key."""
    W = _p("W", 4, 4)
    cat = Op.make("concat", W, W, dim=0)
    idx = _param_index(cat, None, {})
    assert list(idx) == [("\x00fold", cat)]
    assert idx[("\x00fold", cat)] == 32.0
    leaf = _param_index(W, None, {})
    assert leaf == {"W": 16.0}


def test_param_bytes_fold_ewidth():
    """_fold_ewidth: widest resolvable leaf dtype; 4.0 default; Const
    leaves contribute no width."""
    P = _p("P", 4)
    Q = _p("Q", 4)
    # No source_tensors -> fp32 default per leaf.
    assert _fold_ewidth(P, None) == 4.0
    assert _fold_ewidth(Op.make("add", P, Q), None) == 4.0
    # Consts and other non-tensor leaves report None.
    assert _fold_ewidth(Const(1.0), None) is None
    assert _fold_ewidth("raw_leaf", None) is None
    mul = Op.make("mul", P, Const(2.0))
    assert _fold_ewidth(mul, None) == 4.0
    # All-Const fold: no widths at all -> None -> caller defaults.
    cc = Op.make("add", Const(1.0), Const(2.0))
    assert _fold_ewidth(cc, None) is None
    # ...and by_bytes still prices it at fp32 width.
    assert param_bytes_cost(cc, None, by_bytes=True) == pytest.approx(
        _numel_of(cc) * 4.0
    )
    # The widest leaf wins (fp16's 2 bytes over int8's 1).
    src = {
        "P": torch.zeros(4, dtype=torch.float16),
        "Q": torch.zeros(4, dtype=torch.int8),
    }
    cat = Op.make("concat", P, Q, dim=0)
    assert param_bytes_cost(cat, src, by_bytes=True) == 8 * 2.0
    # A leaf whose tensor has no element_size() (numpy's attr is
    # named ``itemsize``) defaults to the fp32 width — and that leaf
    # still resolves, so the subtree still folds.
    mul = Op.make("mul", P, Const(2.0))
    assert param_bytes_cost(
        mul, {"P": np.zeros(4)}, by_bytes=True
    ) == 4.0 * 4.0


def test_param_bytes_for_closure_markers():
    fn = param_bytes_cost_for({"W": torch.zeros(4, 4)})
    assert fn.__name__ == "param_bytes_cost_for"
    assert fn.charges_param_only is True
    assert fn.dag_exact is True
    W = _p("W", 4, 4)
    assert fn(Op.make("mul", W, Const(2.0))) == 16.0


# ---------------------------------------------------------------------------
#  dag_cost — shared DAG, param-only discount, exact models
# ---------------------------------------------------------------------------


def test_dag_cost_shared_subtree_billed_once():
    x = _v("x", 4, 8)
    W = _p("W", 8, 8)
    m = Op.make("mul", x, W)  # interned — the SAME object twice
    t = Op.make("add", m, m)
    tree = flops_cost(t)
    assert dag_cost(t, flops_cost) == pytest.approx(
        _flops_of(t) + flops_cost(m)
    )
    assert dag_cost(t, flops_cost) < tree


def test_dag_cost_param_only_discount():
    """Param-only subtrees fold at compile time: runtime cost 0."""
    x = _v("x", 4, 8)
    A, B = _p("A", 4, 8), _p("B", 8, 8)
    fold = Op.make("matmul", A, B)
    assert dag_cost(fold, flops_cost) == 0.0
    # …while the whole term still bills the input-dependent part:
    # dag_cost = tree cost minus the compile-time fold.
    t = Op.make("add", Op.make("neg", x), fold)
    assert dag_cost(t, flops_cost) == pytest.approx(
        flops_cost(t) - flops_cost(fold)
    )
    # param_bytes_cost opts out: folded storage is real storage
    # (the materialised (4,8) product, billed at output numel).
    assert dag_cost(t, param_bytes_cost) == pytest.approx(
        param_bytes_cost(t)
    )
    assert dag_cost(t, param_bytes_cost) == pytest.approx(4 * 8)


def test_dag_cost_dag_exact_partial_and_markers():
    """functools.partial bindings unwrap .func for both markers."""
    W = _p("W", 4, 4)
    cat = Op.make("concat", W, W, dim=0)
    t = Op.make("add", cat, cat)
    src = {"W": torch.zeros(4, 4)}
    bound = functools.partial(param_bytes_cost, source_tensors=src)
    assert dag_cost(t, bound) == pytest.approx(
        param_bytes_cost(t, src)
    )
    # charges_param_only via .func: a param-only subtree stays billed.
    only = Op.make("matmul", _p("A", 4, 4), _p("B", 4, 4))
    assert dag_cost(only, bound) == pytest.approx(
        param_bytes_cost(only, src)
    )


def test_dag_cost_cost_fn_without_memo():
    """A bare ``fn(term)`` cost model works — memo is only forwarded
    when the signature takes it."""
    x = _v("x", 4, 8)
    W = _p("W", 8, 8)

    def leaf_cost(t):
        return 1.0 if isinstance(t, (Var, Param)) else 0.0

    t = Op.make("add", Op.make("mul", x, W), Op.make("neg", x))
    # Two distinct leaves (x, W): ops contribute 0 local.
    assert dag_cost(t, leaf_cost) == pytest.approx(2.0)

    # A custom model charging param-only work too.
    def unit(t, memo=None):
        return 1.0

    unit.charges_param_only = True
    only = Op.make("matmul", _p("A", 4, 4), _p("B", 4, 4))
    # bill_params -> not skipped: local = 1 - (1 + 1) clamped at 0.
    assert dag_cost(only, unit) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
#  _is_strided / _bytes_of edges
# ---------------------------------------------------------------------------


def test_is_strided_edges():
    # Non-tuple (scalar) chunk shape -> not strided.
    assert _is_strided(Op.make("chunk", Const(1.0), chunks=2)) is False
    # Non-view op -> not strided.
    assert _is_strided(_v("x", 4, 8)) is False
    x = _v("x", 4, 8)
    # chunk on the LAST dim yields strided consumers' views.
    assert (
        _is_strided(Op.make("chunk", x, chunks=2, dim=-1)) is True
    )
    assert (
        _is_strided(Op.make("chunk", x, chunks=2, dim=0)) is False
    )


# ---------------------------------------------------------------------------
#  CostModel — configurable weighted model
# ---------------------------------------------------------------------------


def test_cost_model_linear_and_fallback_arms():
    cm = CostModel()
    x = _v("x", 4, 16)
    W = _p("W", 8, 16)
    # linear: 2 * n_out * in_features = 2 * 32 * 16.
    assert cm(Op.make("linear", x, W)) == pytest.approx(2 * 32 * 16)
    # scalar x: no in-features axis -> flat 2*n fallback.
    t = Op.make("linear", Const(1.0), W)
    assert cm(t) == pytest.approx(2.0)
    # matmul with unshapeable RHS -> coeff * n.
    m = Op.make("matmul", _v("v", 4, 8), UNKNOWN)
    assert cm(m) == pytest.approx(cm.matmul_coeff * 4 * 8)
    # Custom weights: op_weights override, weight_coeff is the default.
    cm2 = CostModel(op_weights={"neg": 5.0}, weight_coeff=3.0)
    assert cm2(Op.make("neg", x)) == pytest.approx(5.0 * 4 * 16)
    assert cm2(Op.make("frobnicate", x)) == pytest.approx(3.0 * 4 * 16)
    # Non-Op leaves cost nothing.
    assert cm(x) == 0.0 and cm(Const(1.0)) == 0.0


def test_cost_model_matmul_rank1_rhs():
    """CostModel's matmul arm reads k from dim -2 only: a rank-1 RHS
    gets k = 1 (a known CostModel quirk vs ``_flops_of``, which does
    recover the vector's axis)."""
    cm = CostModel()
    x = _v("x", 4, 8)
    v = _p("v", 8)
    # (4,8) @ (8,) -> (4,), and CostModel prices it as 2*n.
    assert cm(Op.make("matmul", x, v)) == pytest.approx(2 * 4)


# ---------------------------------------------------------------------------
#  defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------


def test_defensive_branch_inventory():
    """Unreachable-by-construction code — ``# pragma: no cover``
    candidates rather than contorted tests:

    * ``_local_roofline`` lines ~799-800 —
      ``if flops >= _INVALID_COST: return _INVALID_COST``.
      ``_flops_of`` only returns ``_INVALID_COST`` when the inferred
      shape is ``_INVALID`` (line ~122), but that case already
      returned at ~796-797.  With a non-invalid shape the flops value
      is always finite, so the guard cannot fire.
    """
    # Demonstration of why: every non-_INVALID shape yields finite
    # flops, so the second check in _local_roofline can never trigger.
    x = _v("x", 4, 8)
    for t in (
        Op.make("matmul", x, UNKNOWN),  # fallback arm
        Op.make("frobnicate", x),  # unknown op: shapes[0] passthrough
    ):
        assert _shape_of(t) is not _INVALID
        assert _flops_of(t) < _INVALID_COST
