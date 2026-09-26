"""Coverage wave-2 tests for ``catopt.laws`` — the rewrite side
conditions exercised in BOTH directions, and the non-local pairing
passes over adversarial e-graph shapes.

The check/derive hooks in ``laws.tensor`` take ``bound`` dicts mapping
metavariables to terms and ``"$attr:NAME"`` keys to the matched node's
concrete attribute values — these tests build real bindings and prove
each guard accepts exactly the shapes it claims and vetoes the rest.
``laws.pairing``'s passes run over whole e-graphs, so their tests build
e-graphs with the degenerate structures the guards exist for (cyclic
e-classes, concat-valued weights, var-containing weights, uneven and
unknown weight shapes)."""

import torch

from catopt.egraph import EGraph
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.laws.base import (
    _is_channel_scale,
    _is_row_scale,
    _is_scalar,
    _shape_of,
)
from catopt.laws.pairing import (
    _pair_shared_input,
    _term_has_var,
    _wshape,
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from catopt.laws.tensor import (
    _QKV_CAT,
    _REPEAT_KV,
    _REPEAT_V,
    GQA_ABSORB,
    QKV_FUSE_ASYM,
    _check_gqa_absorb,
    _check_linear_bias_compose,
    _check_repeat_chain,
    _check_score_transpose,
    _check_sdpa_base,
    _check_sdpa_mf,
    _check_sdpa_mf_scaled,
    _check_sdpa_scaled,
    _check_softmax_dim,
    _derive_scale_div,
    _derive_scale_mul,
    _derive_scale_one,
    _derive_split_sizes,
    _head,
    _head_v,
    _scale_of,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _ill_typed():
    """A term whose shape is provably invalid (not a tuple)."""
    return Op.make("add", _v("ia", 2), _v("ib", 3))


# ---------------------------------------------------------------------------
#  laws/base — scalar / row / channel scale predicates on non-tuple shapes
# ---------------------------------------------------------------------------


def test_base_scale_predicates_on_non_tuple_shapes():
    # An ill-typed term's shape is the _INVALID marker (not a tuple):
    # both row-scale and channel-scale must answer False, not crash.
    bad = _ill_typed()
    assert not isinstance(_shape_of(bad), tuple)
    assert not _is_row_scale(bad)
    assert not _is_channel_scale({"c": bad, "W": _p("W", 4, 4)})
    # ...and the healthy directions still hold.
    assert _is_scalar(Const(1))
    assert _is_scalar(_p("s"))
    assert _is_row_scale(_p("r", 2, 1))
    assert not _is_row_scale(_p("r", 2, 4))
    assert _is_channel_scale({"c": _p("c", 4), "W": _p("W", 4, 4)})
    assert _is_channel_scale({"c": Const(2), "W": _p("W", 4, 4)})
    assert not _is_channel_scale({"c": _p("c", 5), "W": _p("W", 4, 4)})
    # a weight shape that isn't a concrete tuple → not a channel scale
    assert not _is_channel_scale({"c": _p("c", 4), "W": bad})
    assert not _is_channel_scale({"c": _p("c", 4)})


# ---------------------------------------------------------------------------
#  laws/tensor — _check_linear_bias_compose shape guards
# ---------------------------------------------------------------------------


def test_check_linear_bias_compose_guards():
    # A (h,i) B (o,h) b1 (h,) b2 (o,)|scalar x (...,i)
    good = {
        "x": _v("x", 2, 3),
        "A": _p("A", 5, 3),
        "B": _p("B", 7, 5),
        "b1": _p("b1", 5),
        "b2": _p("b2", 7),
    }
    assert _check_linear_bias_compose(good)
    # scalar outer bias is allowed
    assert _check_linear_bias_compose({**good, "b2": Const(1)})

    def veto(**kw):
        return not _check_linear_bias_compose({**good, **kw})

    # A or B not rank-2
    assert veto(A=_p("A", 5))
    assert veto(B=_p("B", 7, 5, 2))
    # B's input dim != A's output dim
    assert veto(B=_p("B", 7, 4))
    # inner bias must be exactly (h,)
    assert veto(b1=_p("b1", 4))
    assert veto(b1=_p("b1", 5, 1))
    # outer bias must be scalar or (o,)
    assert veto(b2=_p("b2", 6))
    # x's last dim must feed A's input dim
    assert veto(x=_v("x", 2, 4))


# ---------------------------------------------------------------------------
#  laws/tensor — _derive_split_sizes (asymmetric fused QKV)
# ---------------------------------------------------------------------------


def test_derive_split_sizes_uneven_qkv():
    bound = {"Q": _p("Q", 6, 4), "K": _p("K", 2, 4), "V": _p("V", 2, 4)}
    assert _derive_split_sizes(bound) == {"$attr:SZ": (6, 2, 2)}
    # a None dim anywhere in a weight declines the derive
    assert (
        _derive_split_sizes(
            {
                "Q": _p("Q", None, 4),
                "K": _p("K", 2, 4),
                "V": _p("V", 2, 4),
            }
        )
        is None
    )
    # a bound leaf with no tensor type declines too
    assert (
        _derive_split_sizes(
            {"Q": Const(1), "K": _p("K", 2, 4), "V": _p("V", 2, 4)}
        )
        is None
    )


def test_qkv_fuse_asym_fires_and_vetoes_via_egraph():
    """The derive hook runs inside a real ``apply_rule`` — an uneven
    (6,2,2) QKV match produces the split member; an unknown-width Q is
    vetoed before instantiation."""
    x = _v("x", 2, 4)
    Q, K, V = _p("Q", 6, 4), _p("K", 2, 4), _p("V", 2, 4)

    def sdpa(wq, wk, wv):
        return Op.make(
            "sdpa",
            _head_v(Op.make("linear", x, wq), (2, 2, 3, 2)),
            _head_v(Op.make("linear", x, wk), (2, 2, 1, 2)),
            _head_v(Op.make("linear", x, wv), (2, 2, 1, 2)),
            scale=0.5,
            enable_gqa=False,
        )

    eg = EGraph()
    root = eg.add_term(sdpa(Q, K, V))
    assert eg.apply_rule(QKV_FUSE_ASYM, root)
    # the fused member was materialised through split enodes
    assert any(
        n.op == "split" for ec in eg._classes.values() for n in ec.nodes
    )

    # A weight with an unknown out-dim vetoes the derive → no merge.
    eg2 = EGraph()
    bad = eg2.add_term(sdpa(_p("Qb", None, 4), K, V))
    assert not eg2.apply_rule(QKV_FUSE_ASYM, bad)
    assert not any(
        n.op == "split"
        for ec in eg2._classes.values()
        for n in ec.nodes
    )


# ---------------------------------------------------------------------------
#  laws/tensor — repeat_kv chain + GQA absorption guards
# ---------------------------------------------------------------------------


def _repeat_bound(r=4, *, ud=3, es=None, rs=None, k_shape=(1, 4, 2, 3)):
    """A well-formed unsqueeze→expand→reshape binding: repeat r along
    the inserted axis of ``k`` — exactly repeat_interleave semantics."""
    k = _p("k", *k_shape)
    us = (*k_shape[:ud], 1, *k_shape[ud:])
    if es is None:
        es = (*us[:ud], r, *us[ud + 1 :])
    if rs is None:
        rs = (*us[: ud - 1], us[ud - 1] * r, *us[ud + 1 :])
    return {"k": k, "$attr:UDk": ud, "$attr:ESk": es, "$attr:RSk": rs}


def test_check_repeat_chain_guards():
    ok = _repeat_bound()
    assert _check_repeat_chain(ok, "k")
    # non-int dim / non-tuple shapes / unresolvable base → veto
    assert not _check_repeat_chain({**ok, "$attr:UDk": "3"}, "k")
    assert not _check_repeat_chain({**ok, "$attr:ESk": 5}, "k")
    assert not _check_repeat_chain(
        {**ok, "k": _ill_typed(), "base": None}, "k"
    )
    # symbolic dims in the base → veto
    assert not _check_repeat_chain(
        _repeat_bound(k_shape=(None, 4, 2, 3)), "k"
    )
    # unsqueeze dim 0 is not a repeat_interleave pattern
    assert not _check_repeat_chain(_repeat_bound(ud=0), "k")
    # expand rank mismatch → veto
    assert not _check_repeat_chain(
        {**ok, "$attr:ESk": (1, 4, 2, 4)}, "k"
    )
    # repeat factor must be an int > 1
    assert not _check_repeat_chain(
        {**ok, "$attr:ESk": (1, 4, 2, 1, 3)}, "k"
    )
    # expand may only grow the inserted dim
    assert not _check_repeat_chain(
        {**ok, "$attr:ESk": (1, 8, 2, 4, 3)}, "k"
    )
    # reshape that doesn't merge the repeated dims → veto
    assert not _check_repeat_chain(
        {**ok, "$attr:RSk": (1, 4, 2, 4, 3)}, "k"
    )


def test_check_gqa_absorb_both_directions():
    """q heads = kv heads * repeat, k/v chains with the same repeat."""
    q = _p("q", 1, 4, 8, 3)  # 8 q-heads, 2 kv-heads, r = 4
    b = _repeat_bound()
    bound = {
        "q": q,
        "k": b["k"],
        "v": _p("v", 1, 4, 2, 3),
        "$attr:UDk": 3,
        "$attr:ESk": (1, 4, 2, 4, 3),
        "$attr:RSk": (1, 4, 8, 3),
        "$attr:UDv": 3,
        "$attr:ESv": (1, 4, 2, 4, 3),
        "$attr:RSv": (1, 4, 8, 3),
    }
    assert _check_gqa_absorb(bound)
    # k/v repeat factors must agree — a valid-but-different v chain
    # (r=3 with its own consistent reshape) is rejected here
    assert not _check_gqa_absorb(
        {
            **bound,
            "$attr:ESv": (1, 4, 2, 3, 3),
            "$attr:RSv": (1, 4, 6, 3),
        }
    )
    # a malformed repeat chain vetoes
    assert not _check_gqa_absorb({**bound, "$attr:UDk": 0})
    # q/k shapes unresolvable → veto
    assert not _check_gqa_absorb({**bound, "q": _ill_typed()})
    # symbolic dims in q/k → veto
    assert not _check_gqa_absorb(
        {**bound, "q": _p("q2", None, 4, 8, 3)}
    )
    # q head count must equal kv_heads * r
    assert not _check_gqa_absorb({**bound, "q": _p("q3", 1, 4, 7, 3)})


def test_gqa_absorb_real_firing_and_veto():
    """End-to-end through ``apply_rule``: a real repeat_kv sdpa term
    gains the ``enable_gqa`` member; a mismatched head count does not."""
    q = _p("q", 1, 4, 8, 3)
    k, v = _p("k", 1, 4, 2, 3), _p("v", 1, 4, 2, 3)

    def rep(t):
        return Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make(
                    "expand",
                    Op.make("unsqueeze", t, arg1=3),
                    shape=(1, 4, 2, 4, 3),
                ),
                shape=(1, 4, 8, 3),
            ),
            arg1=1,
            arg2=2,
        )

    term = Op.make(
        "sdpa",
        Op.make("transpose", q, arg1=1, arg2=2),
        rep(k),
        rep(v),
        arg4=0.0,
        arg5=False,
    )
    eg = EGraph()
    root = eg.add_term(term)
    assert eg.apply_rule(GQA_ABSORB, root)
    assert any(
        dict(n.attrs).get("arg7") is True
        for n in eg.get_class(root).nodes
    )

    # q with 7 heads (2 kv * 4 = 8 required) → the check vetoes.
    bad = Op.make(
        "sdpa",
        Op.make("transpose", _p("qb", 1, 4, 7, 3), arg1=1, arg2=2),
        rep(k),
        rep(v),
        arg4=0.0,
        arg5=False,
    )
    eg2 = EGraph()
    root2 = eg2.add_term(bad)
    assert not eg2.apply_rule(GQA_ABSORB, root2)
    assert all(
        dict(n.attrs).get("arg7") is not True
        for n in eg2.get_class(root2).nodes
    )


# ---------------------------------------------------------------------------
#  laws/tensor — sdpa fold guards (score transpose / softmax dim / fill)
# ---------------------------------------------------------------------------


def _sdpa_bound(**kw):
    b = {
        "Q": _p("Q", 1, 4, 8, 6),
        "K": _p("K", 1, 4, 8, 6),
        "V": _p("V", 1, 4, 8, 6),
        "M": _p("M", 8, 8),
        "MK": _p("MK", 8, 8),
        "S": Const(0.125),
        "F": Const(float("-inf")),
        "$attr:TD1": -2,
        "$attr:TD2": -1,
        "$attr:SD": -1,
    }
    b.update(kw)
    return b


def test_check_score_transpose_and_softmax_dim():
    assert _check_score_transpose(_sdpa_bound())
    # dims must be ints on the last two axes of a concrete K
    assert not _check_score_transpose(_sdpa_bound(**{"$attr:TD1": 0}))
    assert not _check_score_transpose(
        _sdpa_bound(**{"$attr:TD1": "-2"})
    )
    assert not _check_score_transpose(
        _sdpa_bound(K=_p("K2", None, 4, 8, 6))
    )
    # swapped order on the same axes is still a transpose
    assert _check_score_transpose(
        _sdpa_bound(**{"$attr:TD1": -1, "$attr:TD2": -2})
    )

    assert _check_softmax_dim(_sdpa_bound())
    assert _check_softmax_dim(_sdpa_bound(**{"$attr:SD": 3}))
    assert not _check_softmax_dim(_sdpa_bound(**{"$attr:SD": 0}))
    assert not _check_softmax_dim(_sdpa_bound(**{"$attr:SD": "last"}))
    assert not _check_softmax_dim(_sdpa_bound(Q=_p("Q2")))


def test_check_sdpa_helpers_and_scale_derives():
    assert _check_sdpa_base(_sdpa_bound())
    # base check fails → every derived check fails
    assert not _check_sdpa_mf(_sdpa_bound(**{"$attr:SD": 0}))
    assert not _check_sdpa_scaled(_sdpa_bound(S=Const("s")))
    assert not _check_sdpa_mf_scaled(_sdpa_bound(F=Const(-1.0)))
    # the masked_fill value must be a large negative
    assert _check_sdpa_mf(_sdpa_bound())
    assert not _check_sdpa_mf(_sdpa_bound(F=Const(-1.0)))
    assert _check_sdpa_mf_scaled(_sdpa_bound())
    # scale plumbing
    assert _scale_of(_sdpa_bound()) == 0.125
    assert _scale_of(_sdpa_bound(S=Const("x"))) is None
    assert _derive_scale_mul(_sdpa_bound()) == {"$attr:SC": 0.125}
    assert _derive_scale_div(_sdpa_bound()) == {"$attr:SC": 8.0}
    assert _derive_scale_div(_sdpa_bound(S=Const("x"))) is None
    assert _derive_scale_one(_sdpa_bound()) == {"$attr:SC": 1.0}


def test_head_and_repeat_template_terms():
    """The shared pattern builders mint the expected structure."""
    t = _head("w")
    assert t.op == "transpose" and t.args[0].op == "reshape"
    assert dict(t.attrs) == {"arg1": 1, "arg2": 2}
    assert dict(t.args[0].attrs) == {"shape": "S"}
    tv = _head_v("w", "S9")
    assert dict(tv.args[0].attrs) == {"shape": "S9"}
    assert _QKV_CAT.op == "concat" and _QKV_CAT.args[0].op == "concat"
    for rep in (_REPEAT_KV, _REPEAT_V):
        assert rep.op == "transpose"
        rs = rep.args[0]
        assert rs.op == "reshape" and rs.args[0].op == "expand"
        assert rs.args[0].args[0].op == "unsqueeze"


# ---------------------------------------------------------------------------
#  laws/pairing — small helpers
# ---------------------------------------------------------------------------


def test_term_has_var_and_wshape():
    x, w = _v("x", 4), _p("w", 4, 4)
    assert _term_has_var(x)
    assert _term_has_var(Op.make("add", w, Op.make("mul", x, w)))
    assert not _term_has_var(w)
    assert not _term_has_var(Const(0))
    assert not _term_has_var(Op.make("add", w, Const(1)))
    assert _wshape(w) == (4, 4)
    assert _wshape(Const(0)) == ()


# ---------------------------------------------------------------------------
#  laws/pairing — _pair_shared_input guard paths
# ---------------------------------------------------------------------------


def _cyclic_eclass(eg):
    """An e-class whose every member is self-referential.

    Built by unioning a computed class into a leaf class, then
    dropping the acyclic member — the degenerate shape the pairing
    pass's var-reachability memo must cut rather than loop on."""
    leaf = eg.add_term(_p(f"cyc{len(eg._classes)}", 2))
    f = eg.add_enode("neg", (leaf,))
    eg.union(f, leaf)
    cid = eg.find(f)
    for n in list(eg._classes[cid].nodes):
        if n.op == "leaf":
            eg._classes[cid].nodes.discard(n)
    return cid


def test_pair_shared_input_happy_group():
    """Two linears on one input pair into one GEMM + split views; the
    merge is witnessed so certificates can replay it."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 4))
    w1 = eg.add_term(_p("w1", 8, 4))
    w2 = eg.add_term(_p("w2", 8, 4))
    m1 = eg.add_enode("linear", (x, w1))
    m2 = eg.add_enode("linear", (x, w2))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    grp = groups[0]
    assert set(grp) == {eg.find(m1), eg.find(m2)}
    en = next(iter(grp.values()))
    assert en.op == "split" and dict(en.attrs)["sizes"] == (8, 8)
    # members were unioned to their split views
    for cid in grp:
        assert any(n.op == "split" for n in eg.get_class(cid).nodes)


def test_pair_shared_input_guard_paths():
    """Every 'continue' in the pass corresponds to a real degenerate
    structure — each is built and verified to leave no offer."""
    # (a) input e-class with no reachable Var → pairing a compile-time
    # chain is noise.  A cyclic-only class also exercises the
    # memoised reachability walk's cycle cut.
    eg = EGraph()
    cyc = _cyclic_eclass(eg)
    w = eg.add_term(_p("w", 8, 4))
    eg.add_enode("linear", (cyc, w))
    # weight class that cannot resolve a term → skipped member
    x = eg.add_term(_v("x", 2, 4))
    wcyc = _cyclic_eclass(eg)
    eg.add_enode("linear", (x, wcyc))
    # weight that IS a concat (already fused) → excluded member
    wcat = eg.add_term(
        Op.make("concat", _p("wa", 8, 4), _p("wb", 8, 4), dim=0)
    )
    eg.add_enode("linear", (x, wcat))
    # lone surviving member → no pair
    w1 = eg.add_term(_p("w1", 8, 4))
    eg.add_enode("linear", (x, w1))
    assert pair_shared_input_linears(eg) == []

    # (b) two DISTINCT enodes sharing input AND weight — the weight set
    # has one element → nothing to fuse.
    eg2 = EGraph()
    x2 = eg2.add_term(_v("x2", 2, 4))
    w2 = eg2.add_term(_p("w2", 8, 4))
    eg2.add_enode("linear", (x2, w2), {"variant": 1})
    eg2.add_enode("linear", (x2, w2), {"variant": 2})
    assert pair_shared_input_linears(eg2) == []

    # (c) a weight reachable from a Var → the fused GEMM would read
    # runtime data — declined.
    eg3 = EGraph()
    x3 = eg3.add_term(_v("x3", 2, 4))
    wv = eg3.add_term(Op.make("mul", _v("g", 8, 4), _p("wb", 8, 4)))
    w3 = eg3.add_term(_p("w3", 8, 4))
    eg3.add_enode("linear", (x3, wv))
    eg3.add_enode("linear", (x3, w3))
    assert pair_shared_input_linears(eg3) == []

    # (d) a weight whose out-dim is unknown — sizes can't be proven →
    # the whole cluster is skipped.
    eg4 = EGraph()
    x4 = eg4.add_term(_v("x4", 2, 4))
    wn = eg4.add_term(_p("wn", None, 4))
    w4 = eg4.add_term(_p("w4", 8, 4))
    eg4.add_enode("linear", (x4, wn))
    eg4.add_enode("linear", (x4, w4))
    assert pair_shared_input_linears(eg4) == []


def test_pair_shared_input_offer_without_witness():
    """When no representative term exists for a member's class the
    union proceeds witness-free (the merge is still recorded)."""
    eg = EGraph()
    # Force the "oldest term" resolution to fail so the pass cannot
    # synthesise a pointwise witness — the offer must still land.
    eg._oldest_term = lambda *a, **k: None
    x = eg.add_term(_v("x", 2, 4))
    w1 = eg.add_term(_p("w1", 8, 4))
    w2 = eg.add_term(_p("w2", 8, 4))
    eg.add_enode("linear", (x, w1))
    eg.add_enode("linear", (x, w2))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1


def test_pair_shared_input_conv_groups_excluded():
    """Grouped convolutions cannot concat on out-channels — the key
    returns None and grouped members are excluded from pairing."""
    eg = EGraph()
    x = eg.add_term(_v("x", 1, 8, 16, 16))
    w1 = eg.add_term(_p("w1", 8, 8, 3, 3))
    w2 = eg.add_term(_p("w2", 8, 8, 3, 3))
    wg = eg.add_term(_p("wg", 8, 4, 3, 3))
    wflat = eg.add_term(_p("wflat", 8, 8))  # not a 4-D conv weight
    eg.add_enode("conv2d", (x, w1), {"stride": 1, "padding": 1})
    eg.add_enode("conv2d", (x, w2), {"stride": 1, "padding": 1})
    eg.add_enode(
        "conv2d", (x, wg), {"stride": 1, "padding": 1, "groups": 2}
    )
    eg.add_enode("conv2d", (x, wflat), {"stride": 1, "padding": 1})
    groups = pair_shared_input_convs(eg)
    assert len(groups) == 1
    grp = groups[0]
    # only the two same-geometry members paired; sizes = out channels
    assert len(grp) == 2
    assert dict(next(iter(grp.values())).attrs)["sizes"] == (8, 8)


def test_pair_shared_input_conv_incompatible_shapes():
    """Convs on one input that share no compatible geometry (kernel or
    stride) produce no group."""
    eg = EGraph()
    x = eg.add_term(_v("x", 1, 8, 16, 16))
    w1 = eg.add_term(_p("w1", 8, 8, 3, 3))
    w2 = eg.add_term(_p("w2", 8, 8, 1, 1))
    eg.add_enode("conv2d", (x, w1), {"stride": 1, "padding": 1})
    eg.add_enode("conv2d", (x, w2), {"stride": 1, "padding": 0})
    assert pair_shared_input_convs(eg) == []


def test_pair_shared_input_arbitrary_op():
    """The pass is op-parametric: running it on 'matmul'-style members
    shows the same domain-sharing mechanics for a custom signature."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 4))
    w1 = eg.add_term(_p("w1", 8, 4))
    w2 = eg.add_term(_p("w2", 8, 4))
    eg.add_enode("matmul", (x, w1))
    eg.add_enode("matmul", (x, w2))
    groups = _pair_shared_input(
        eg,
        op="matmul",
        split_dim=-1,
        cluster_key=lambda n, wt: (
            ("mm",)
            if isinstance(_wshape(wt), tuple) and len(_wshape(wt)) == 2
            else None
        ),
    )
    assert len(groups) == 1


# ---------------------------------------------------------------------------
#  laws/pairing — share_duplicate_params edges
# ---------------------------------------------------------------------------


def test_share_duplicate_params_edges():
    torch.manual_seed(0)
    a = torch.randn(4, 4, dtype=torch.float64)
    src = {
        "p1": a.clone(),
        "p2": a.clone(),  # exact tie with p1
        "p3": torch.randn(4, 4, dtype=torch.float64),
        "odd": "not-a-tensor",  # non-tensor entry is skipped
    }
    eg = EGraph()
    for n in ("p1", "p2", "p3"):
        eg.add_term(Param(n, _T(4, 4)))
    groups = share_duplicate_params(eg, src)
    assert sorted(map(sorted, groups)) == [["p1", "p2"]]
    # the merge was witnessed with a named pointwise rule
    assert any(r.startswith("share#") for r in eg._rule_objs)

    # Re-running over the already-merged classes is a no-op —
    # the same groups are reported and nothing breaks.
    groups2 = share_duplicate_params(eg, src)
    assert sorted(map(sorted, groups2)) == [["p1", "p2"]]

    # witness=False merges without a synthesised rule.
    eg2 = EGraph()
    for n in ("p1", "p2"):
        eg2.add_term(Param(n, _T(4, 4)))
    groups3 = share_duplicate_params(
        eg2, {"p1": a.clone(), "p2": a.clone()}, witness=False
    )
    assert groups3 == [["p1", "p2"]]
    assert not any(r.startswith("share#") for r in eg2._rule_objs)


def _dup_head_setup():
    torch.manual_seed(0)
    h, d, i = 4, 3, 5
    blk = torch.randn(d, i, dtype=torch.float64)
    other = torch.randn(d, i, dtype=torch.float64)
    W = torch.cat([blk, other, blk, blk], dim=0)  # heads 0,2,3 tied
    return h, d, i, W


def test_share_duplicate_param_slices_name_collision():
    """A colliding dedup name gets suffixed — the planted tensor is
    never clobbered and the offer carries the renamed param."""
    h, d, i, W = _dup_head_setup()
    source = {"W": W}
    # a source tensor absent from the graph is skipped outright
    source["ghost"] = W.clone()
    # pre-register a DIFFERENT tensor under the name the pass wants —
    # it must rename to W__heads4_1 rather than clobber it.
    source["W__heads4"] = torch.zeros(2, d, i, dtype=torch.float64)

    eg = EGraph()
    eg.add_term(Param("W", _T(h * d, i)))
    offers = share_duplicate_param_slices(eg, source)
    assert len(offers) == 1
    off = offers[0]
    assert off["dedup_param"] == "W__heads4_1"
    assert tuple(source[off["dedup_param"]].shape) == (2, d, i)
    assert torch.equal(source["W__heads4"], torch.zeros(2, d, i))


def test_share_duplicate_param_slices_rerun_reuses_name():
    """A second run over the same source sees the registered dedup
    tensor already value-equal → registration is skipped and the
    offer reuses the same name."""
    h, d, i, W = _dup_head_setup()
    source = {"W": W}
    eg = EGraph()
    eg.add_term(Param("W", _T(h * d, i)))
    offers = share_duplicate_param_slices(eg, source)
    assert len(offers) == 1
    assert offers[0]["dedup_param"] == "W__heads4"
    offers2 = share_duplicate_param_slices(eg, source)
    assert len(offers2) == 1
    assert offers2[0]["dedup_param"] == "W__heads4"


def test_share_duplicate_param_slices_witness_off():
    """witness=False unions without a synthesised rule."""
    h, d, i, W = _dup_head_setup()
    eg = EGraph()
    eg.add_term(Param("W2", _T(h * d, i)))
    offers = share_duplicate_param_slices(
        eg, {"W2": W.clone()}, witness=False
    )
    assert len(offers) == 1
    assert not any(r.startswith("share_slices#") for r in eg._rule_objs)
