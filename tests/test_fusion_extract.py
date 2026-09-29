"""Fusion-preferred extraction — near-tie members resolve by kernels.

``EGraph.extract_best(..., fusion_epsilon=...)`` (and the
``extract_fused`` wrapper, epsilon 5%) add a secondary key among
members priced within epsilon of the class minimum:
``(len(fusion_regions(term)), not exposes_pointwise, nops)`` via
:func:`catopt_core.cost.fusion_member_key`.  These tests pin the band
semantics (near-tie only), the key ordering (fewer regions, then a
root a pointwise consumer could absorb), determinism on exact ties,
and reuse of the shared cost memo for the region probes.
"""

from catopt.egraph import EGraph
from catopt.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.cost import (
    count_cost,
    fusion_member_key,
    fusion_regions,
)


def _t(*shape):
    return TensorType(tuple(shape))


def _x():
    return Var("x", _t(4, 4))


def _w():
    return Param("W", _t(4, 4))


def _class_of(*terms):
    """EGraph holding *terms* unioned into one e-class -> root eid."""
    eg = EGraph()
    eids = [eg.add_term(t) for t in terms]
    root = eids[0]
    for e in eids[1:]:
        eg.union(root, e)
    return eg, eg.find(root)


def test_fused_prefers_pointwise_over_boundary():
    """Equal-cost members, equal region counts: a sigmoid member and a
    matmul member each occupy one kernel, but only the pointwise root
    can merge into a consumer's region — the boundary flag decides."""
    x, w = _x(), _w()
    pointwise = Op.make("sigmoid", x)
    boundary = Op.make("matmul", x, w)
    eg, root = _class_of(pointwise, boundary)
    # count_cost ties at 1 op each; member sort order has "matmul"
    # first, so plain extraction keeps the boundary member.
    default = eg.extract_best(root, count_cost)
    assert default.op == "matmul"
    fused = eg.extract_fused(root, count_cost)
    assert fused.op == "sigmoid"
    # Both members really are single-region terms — it is the
    # exposes-pointwise flag, not the count, that flips the pick.
    assert len(fusion_regions(pointwise)) == 1
    assert len(fusion_regions(boundary)) == 1


def test_fused_prefers_fewer_regions():
    """Same boundary flag on both members: fewer predicted kernels wins."""
    x = _x()
    one_region = Op.make("sigmoid", Op.make("sigmoid", x))
    two_region = Op.make("neg", Op.make("sum", x))
    eg, root = _class_of(one_region, two_region)
    # Both are 2 ops -> count tie; "neg" sorts before "sigmoid", so
    # plain extraction keeps the 2-region member.
    default = eg.extract_best(root, count_cost)
    assert default.op == "neg"
    fused = eg.extract_fused(root, count_cost)
    assert fused.op == "sigmoid"
    assert len(fusion_regions(one_region)) == 1
    assert len(fusion_regions(two_region)) == 2


def test_fused_declines_outside_band():
    """A fusible member priced beyond epsilon loses: the band is a
    near-tie concession, not a fusion override of the cost model."""
    x, w = _x(), _w()
    cheap_boundary = Op.make("matmul", x, w)  # count_cost 1
    costly_pointwise = Op.make("sigmoid", Op.make("sigmoid", x))  # 2
    eg, root = _class_of(cheap_boundary, costly_pointwise)
    fused = eg.extract_fused(root, count_cost)
    assert fused.op == "matmul"  # 2 > 1 * (1 + 0.05) -> out of band


def test_fused_wider_epsilon_widens_band():
    """Same class as the decline test, but a wide band admits the
    pointwise member — epsilon is the band's width control."""
    x, w = _x(), _w()
    cheap_boundary = Op.make("matmul", x, w)
    costly_pointwise = Op.make("sigmoid", Op.make("sigmoid", x))
    eg, root = _class_of(cheap_boundary, costly_pointwise)
    fused = eg.extract_fused(root, count_cost, fusion_epsilon=1.5)
    assert fused.op == "sigmoid"


def test_fused_exact_tie_deterministic():
    """Fully equal keys (regions, flag, nops) fall back to the
    canonical member order — stable across passes and rebuilds."""
    x = _x()
    neg = Op.make("neg", x)
    sig = Op.make("sigmoid", x)
    eg, root = _class_of(neg, sig)
    t1 = eg.extract_best(root, count_cost, fusion_epsilon=0.05)
    t2 = eg.extract_best(root, count_cost, fusion_epsilon=0.05)
    assert op_repr(t1) == op_repr(t2)
    # "neg" sorts before "sigmoid"; equal keys keep the earlier member.
    assert t1.op == "neg"
    eg2, root2 = _class_of(sig, neg)  # union order must not matter
    t3 = eg2.extract_fused(root2, count_cost)
    assert op_repr(t3) == op_repr(t1)


def test_fused_region_probes_use_shared_memo():
    """Region counts land in the per-cost_fn shared memo (``("fr",
    term)`` keys) — priced once ever, not per pass per member."""
    x, w = _x(), _w()
    eg, root = _class_of(Op.make("sigmoid", x), Op.make("matmul", x, w))
    fused = eg.extract_fused(root, count_cost)
    memo = eg._cost_memos[id(count_cost)][1]
    assert ("fr", fused) in memo
    fr_keys = {k for k in memo if k[0] == "fr"}
    assert len(fr_keys) == 2  # one per in-band member
    size = len(memo)
    again = eg.extract_fused(root, count_cost)
    assert op_repr(again) == op_repr(fused)
    assert len(memo) == size  # second pass prices nothing new


def test_fused_leaf_members_pick_sorted_first():
    """Two leaf members tie at zero regions; key equality defers to
    member order.  Covers the leaf branch of candidate collection."""
    x = _x()
    eg = EGraph()
    xeid = eg.add_term(x)
    ceid = eg.add_term(Const(0))
    eg.union(xeid, ceid)
    best = eg.extract_fused(eg.find(xeid), count_cost)
    # repr("0") < repr("x") in the canonical member order.
    assert best == Const(0)


def test_fused_param_fold_is_not_pointwise():
    """A param-only pointwise root folds to a materialised Param — a
    kernel INPUT, not a fusible member — so exposes_pointwise must
    see through it to ``False`` (as must a transparent root whose
    args are leaves/boundary ops)."""
    w = _w()
    z = Param("Z", _t(4, 4))
    fold = Op.make("add", w, w)  # param-only fold, 0 regions
    plumbing = Op.make("transpose", w, dim0=0, dim1=1)  # transparent
    _ = z
    eg, root = _class_of(fold, plumbing)
    # Both are param-only -> both price 0 -> both in the band;
    # keys (0 regions, boundary flag, 1 op) tie -> member order wins:
    # "add" < "transpose".
    fused = eg.extract_fused(root, count_cost)
    assert fused.op == "add"
    m = eg._cost_memos[id(count_cost)][1]
    assert fusion_member_key(fold, m) == (0, 1)
    assert fusion_member_key(plumbing, m) == (0, 1)


def test_fused_transparent_root_exposes_pointwise_child():
    """Transparent plumbing forwards fusion: a transpose root over a
    pointwise child still exposes a fusible top to its consumer."""
    x, w = _x(), _w()
    wrapped = Op.make("transpose", Op.make("neg", x), dim0=0, dim1=1)
    boundary = Op.make("matmul", x, w)
    eg, root = _class_of(wrapped, boundary)
    # count tie: transpose is a view (0) + neg (1) vs matmul (1).
    fused = eg.extract_fused(root, count_cost)
    assert fused.op == "transpose"
    m = eg._cost_memos[id(count_cost)][1]
    # One region (the neg), root flagged fusible-through-plumbing.
    assert fusion_member_key(wrapped, m) == (1, 0)
    assert fusion_member_key(boundary, m) == (1, 1)


def test_fused_single_member_classes_untouched():
    """Classes with a single valid candidate skip the band entirely —
    fusion mode must not perturb a forced extraction."""
    x, w = _x(), _w()
    eg = EGraph()
    root = eg.add_term(Op.make("matmul", x, w))
    plain = eg.extract_best(root, count_cost)
    fused = eg.extract_fused(root, count_cost)
    assert op_repr(fused) == op_repr(plain)


def test_fused_overrides_and_bans_pass_through():
    """``extract_fused`` forwards the coordinated-extraction knobs."""
    x, w = _x(), _w()
    pointwise = Op.make("sigmoid", x)
    boundary = Op.make("matmul", x, w)
    eg = EGraph()
    sp = eg.add_term(pointwise)
    mm = eg.add_term(boundary)
    eg.union(sp, mm)
    root = eg.find(sp)
    # Ban the pointwise member -> the boundary member is the only
    # candidate regardless of fusion preference.
    ban_node = next(
        n for n in eg.get_class(root).nodes if n.op == "sigmoid"
    )
    fused = eg.extract_fused(root, count_cost, bans={root: {ban_node}})
    assert fused.op == "matmul"
    # Or override directly to it.
    ov_node = next(
        n for n in eg.get_class(root).nodes if n.op == "matmul"
    )
    forced = eg.extract_fused(
        root, count_cost, overrides={root: ov_node}
    )
    assert forced.op == "matmul"


def test_optimize_model_fusion_epsilon_kwarg():
    """``optimize_model(fusion_epsilon=…)`` arms the near-tie band and
    records it in stats; 0 keeps byte-identical selection."""
    import torch
    import torch.nn as nn
    from catopt.optimize import optimize_model

    class M(nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.sigmoid(x))

    m = M().eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    mod, stats = optimize_model(
        m, x, fusion_epsilon=0.05, verbose=False
    )
    assert stats["fusion_epsilon"] == 0.05
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x))
    _mod2, stats2 = optimize_model(m, x, verbose=False)
    assert "fusion_epsilon" not in stats2
