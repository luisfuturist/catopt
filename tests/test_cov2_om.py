"""Coverage tests for catopt.om — the guard branches behind the
online-softmax monoid laws.

Every check/derive hook is driven BOTH ways: a bound substitution that
satisfies the contract (the rule must fire and the lifted carrier must
evaluate correctly in fp64) and bound substitutions violating each
clause (shape ``None``/``_INVALID``, wrong concat axes, projection-vs-
domain mismatches, non-numeric sdpa flags, mask extents that are
neither sliceable nor broadcastable).

``bound`` dicts are minted by hand — the same convention the e-graph
uses (metavariable -> concrete term, ``"$attr:X"`` -> attr value) —
because several vetoes cannot be reached through a well-typed minted
term (e.g. an ``_INVALID``-shaped operand bound to a metavar).
Each law also gets at least one real ``apply_rewrite_at`` end-to-end
fire whose result is evaluated against a serial fp64 reference.
"""

import torch

import catopt.om as OM
from catopt import meta
from catopt.ir import Const, Op, TensorType, Var

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(shape, seed: int = 0):
    g = torch.Generator().manual_seed(9000 + seed)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _ill() -> Op:
    """A provably ill-typed term — its shape is ``_INVALID``."""
    return Op.make("add", _v("zz_a", 2, 3), _v("zz_b", 5, 4))


#: A leaf that is not a term at all — ``_shape_of`` reports ``None``.
RAW = "raw_leaf"


def _check(rule, bound: dict) -> bool:
    return rule.check(bound)


def _derive(rule, bound: dict):
    return rule.derive(bound)


def _fire(rule, term):
    return meta.apply_rewrite_at(rule, term, ())


def _eval(term, env):
    return meta._eval_term(term, env)


def _find(rules, name):
    for r in rules:
        if r.name == name:
            return r
    raise KeyError(name)


MFC = {r.name: r for r in OM.MASKED_FILL_CAT}
AMC = {r.name: r for r in OM.ADD_MASK_CAT}
WRC = {r.name: r for r in OM.WHERE_CAT}
CH = {r.name: r for r in OM.CAT_HOM}
SDPA = {r.name: r for r in OM.SDPA_CAT_LAWS}


# ---------------------------------------------------------------------------
#  om_lift — softmax must be over the LAST dim; s[-1] contracts v[-2]
# ---------------------------------------------------------------------------


def test_om_lift_fires_all_spellings_and_evals():
    """matmul(softmax(s,-1), v) lifts to om_apply(om_elem(s,v)) under
    each attr spelling; fp64-equal to dense softmax @ v.  T=1, K=3
    (non-power-of-2) are deliberate."""
    T, K, d = 1, 3, 5
    s, v = _v("s", T, K), _v("v", K, d)
    env = {s: _rand((T, K), 1), v: _rand((K, d), 2)}
    ref = torch.softmax(env[s], dim=-1) @ env[v]
    for rule, sm in (
        (OM.OM_LIFT, Op.make("softmax", s, arg1=-1)),
        (OM.OM_LIFT_DIM, Op.make("softmax", s, dim=-1)),
        (OM.OM_LIFT_PLAIN, Op.make("softmax", s)),
    ):
        t0 = Op.make("matmul", sm, v)
        out = _fire(rule, t0)
        assert out is not None, rule.name
        assert out.op == "om_apply" and out.args[0].op == "om_elem"
        got = _eval(out, env)
        assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_om_lift_check_accepts_unknown_dims_and_batched():
    """None extents are wildcards: (T,None) scores with (None,d) values
    still satisfy s[-1] == v[-2]; batched ranks pass."""
    assert OM.OM_LIFT.check(
        {"s": _v("s", 4, None), "v": _v("v", None, 5), "$attr:SD": -1}
    )
    assert OM.OM_LIFT.check(
        {
            "s": _v("s", 2, 4, 7),
            "v": _v("v", 2, 7, 3),
            "$attr:SD": 2,
        }
    )
    # negative-dim spelling of the last axis also passes
    assert OM.OM_LIFT_DIM.check(
        {"s": _v("s", 2, 4, 7), "v": _v("v", 2, 7, 3), "$attr:SD": -1}
    )


def test_om_lift_check_vetoes():
    c = OM._check_om_lift
    # non-tuple shapes: unknown leaf / _INVALID term → line 126
    assert not c({"s": RAW, "v": _v("v", 7, 3), "$attr:SD": -1})
    assert not c({"s": _ill(), "v": _v("v", 7, 3), "$attr:SD": -1})
    # ranks < 2 on either operand
    assert not c({"s": _v("s", 7), "v": _v("v", 7, 3), "$attr:SD": -1})
    assert not c({"s": _v("s", 4, 7), "v": _v("v", 7), "$attr:SD": -1})
    # scalar () scores
    assert not c({"s": Const(1.0), "v": _v("v", 7, 3), "$attr:SD": -1})
    # softmax dim not an int, or not the last axis
    assert not c({"s": _v("s", 4, 7), "v": _v("v", 7, 3), "$attr:SD": "x"})
    assert not c({"s": _v("s", 4, 7), "v": _v("v", 7, 3), "$attr:SD": 0})
    assert not c(
        {"s": _v("s", 2, 4, 7), "v": _v("v", 2, 7, 3), "$attr:SD": 1}
    )
    # contraction mismatch s[-1] != v[-2]
    assert not c({"s": _v("s", 4, 7), "v": _v("v", 9, 3), "$attr:SD": -1})


def test_om_lift_no_fire_on_wrong_softmax_axis_term():
    """End-to-end: a matched term whose softmax is on dim 0 must not
    rewrite (the check vetoes, not the matcher)."""
    t0 = Op.make(
        "matmul",
        Op.make("softmax", _v("s", 4, 7), arg1=0),
        _v("v", 7, 3),
    )
    assert _fire(OM.OM_LIFT, t0) is None


def test_om_unlift_round_trips():
    t0 = Op.make(
        "om_apply",
        Op.make("om_elem", _v("s", 4, 7), _v("v", 7, 3)),
    )
    out = _fire(OM.OM_UNLIFT, t0)
    assert out is not None
    assert out.op == "matmul"


# ---------------------------------------------------------------------------
#  om_split / om_merge — the homomorphism, both directions
# ---------------------------------------------------------------------------


def _split_bound(s1, s2, v1, v2, sd=-1, vd=-2):
    return {
        "s1": s1,
        "s2": s2,
        "v1": v1,
        "v2": v2,
        "$attr:SD": sd,
        "$attr:VD": vd,
    }


def test_om_split_fires_and_evals_uneven_blocks():
    """elem(cat(s1,s2), cat(v1,v2)) = elem(s1,v1) ⊕ elem(s2,v2), fp64 —
    uneven blocks K1=3, K2=5."""
    T, K1, K2, d = 4, 3, 5, 6
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    v1, v2 = _v("v1", K1, d), _v("v2", K2, d)
    env = {
        s1: _rand((T, K1), 10),
        s2: _rand((T, K2), 11),
        v1: _rand((K1, d), 12),
        v2: _rand((K2, d), 13),
    }
    t0 = Op.make(
        "om_elem",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", v1, v2, dim=-2),
    )
    out = _fire(OM.OM_SPLIT, t0)
    assert out is not None
    got = _eval(out, env)
    ref = _eval(t0, env)
    assert meta._eval_allclose(got, ref, tol=1e-12)


def test_om_split_arg1_spelling_fires():
    s1, s2 = _v("s1", 4, 3), _v("s2", 4, 5)
    v1, v2 = _v("v1", 3, 6), _v("v2", 5, 6)
    t0 = Op.make(
        "om_elem",
        Op.make("concat", s1, s2, arg1=-1),
        Op.make("concat", v1, v2, arg1=-2),
    )
    assert _fire(OM.OM_SPLIT_ARG1, t0) is not None


def test_om_split_check_vetoes():
    c = OM.OM_SPLIT.check
    good = _split_bound(
        _v("s1", 4, 3), _v("s2", 4, 5), _v("v1", 3, 6), _v("v2", 5, 6)
    )
    assert c(good)
    # non-tuple members → line 196
    assert not c({**good, "s2": RAW})
    assert not c({**good, "v1": _ill()})
    # concat dims not ints → line 198
    assert not c({**good, "$attr:SD": "x"})
    assert not c({**good, "$attr:VD": None})
    # wrong axes → lines 199-202
    assert not c({**good, "$attr:SD": 0})
    assert not c({**good, "$attr:VD": -1})
    # _chunks_compatible vetoes:
    #   rank < 2 / rank mismatch → line 174
    assert not c(
        _split_bound(_v("s1", 3), _v("s2", 5), _v("v1", 3, 6), _v("v2", 5, 6))
    )
    assert not c(
        _split_bound(
            _v("s1", 4, 3), _v("s2", 4, 5), _v("v1", 2, 3, 6), _v("v2", 5, 6)
        )
    )
    #   scores off-axis mismatch → line 176
    assert not c(
        _split_bound(
            _v("s1", 4, 3), _v("s2", 9, 5), _v("v1", 3, 6), _v("v2", 5, 6)
        )
    )
    #   values off-axis mismatch → line 178
    assert not c(
        _split_bound(
            _v("s1", 4, 3), _v("s2", 4, 5), _v("v1", 3, 6), _v("v2", 5, 7)
        )
    )
    #   contraction mismatch s[-1] != v[-2] → line 180
    assert not c(
        _split_bound(
            _v("s1", 4, 3), _v("s2", 4, 5), _v("v1", 8, 6), _v("v2", 5, 6)
        )
    )
    #   batch dims not broadcastable s_i[:-2] vs v_i[:-2] → line 183
    assert not c(
        _split_bound(
            _v("s1", 2, 4, 3),
            _v("s2", 2, 4, 5),
            _v("v1", 3, 3, 6),
            _v("v2", 3, 5, 6),
        )
    )


def test_om_merge_derive_and_eval():
    """om_compose(elem,elem) → elem(cat s, cat v): the derive computes
    the cat dims from bound shapes; vetoes return None."""
    good = _split_bound(
        _v("s1", 4, 3), _v("s2", 4, 5), _v("v1", 3, 6), _v("v2", 5, 6)
    )
    assert _derive(OM.OM_MERGE, good) == {"$attr:SD": 1, "$attr:VD": 0}
    # non-tuple member → line 237
    assert _derive(OM.OM_MERGE, {**good, "s1": RAW}) is None
    # incompatible chunks → line 239
    assert (
        _derive(
            OM.OM_MERGE,
            _split_bound(
                _v("s1", 4, 3), _v("s2", 9, 5), _v("v1", 3, 6), _v("v2", 5, 6)
            ),
        )
        is None
    )
    # end-to-end: fire on a real term, fp64-check both sides
    T, K1, K2, d = 4, 3, 5, 6
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    v1, v2 = _v("v1", K1, d), _v("v2", K2, d)
    env = {
        s1: _rand((T, K1), 20),
        s2: _rand((T, K2), 21),
        v1: _rand((K1, d), 22),
        v2: _rand((K2, d), 23),
    }
    t0 = Op.make(
        "om_compose",
        Op.make("om_elem", s1, v1),
        Op.make("om_elem", s2, v2),
    )
    out = _fire(OM.OM_MERGE, t0)
    assert out is not None and out.op == "om_elem"
    assert meta._eval_allclose(_eval(out, env), _eval(t0, env), tol=1e-12)


def test_om_assoc_both_directions_eval():
    f = Op.make("om_elem", _v("s1", 4, 3), _v("v1", 3, 5))
    g = Op.make("om_elem", _v("s2", 4, 4), _v("v2", 4, 5))
    h = Op.make("om_elem", _v("s3", 4, 2), _v("v3", 2, 5))
    left = Op.make("om_compose", Op.make("om_compose", f, g), h)
    right = Op.make("om_compose", f, Op.make("om_compose", g, h))
    assert _fire(OM.OM_ASSOC, left) == right
    assert _fire(OM.OM_ASSOC_REV, right) == left


def test_concat_binarize_fires_both_spellings():
    for r in OM.CONCAT_BINARIZE:
        n = int(r.name.split("_")[2])
        ak = "dim" if r.name.endswith("_dim") else "arg1"
        xs = [_v(f"x{n}{ak}{i}", 4, 3 + i) for i in range(n)]
        t0 = Op.make("concat", *xs, **{ak: -1})
        out = _fire(r, t0)
        assert out is not None, r.name


# ---------------------------------------------------------------------------
#  matmul_t_concat — q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T)
# ---------------------------------------------------------------------------


def _mtc_bound(q, k1, k2, kd=-2, t1=-2, t2=-1):
    return {
        "q": q,
        "k1": k1,
        "k2": k2,
        "$attr:KD": kd,
        "$attr:T1": t1,
        "$attr:T2": t2,
    }


def test_matmul_t_concat_fires_and_evals():
    T, E, K1, K2 = 4, 6, 3, 5
    q, k1, k2 = _v("q", T, E), _v("k1", K1, E), _v("k2", K2, E)
    env = {
        q: _rand((T, E), 30),
        k1: _rand((K1, E), 31),
        k2: _rand((K2, E), 32),
    }
    t0 = Op.make(
        "matmul",
        q,
        Op.make(
            "transpose",
            Op.make("concat", k1, k2, dim=-2),
            arg1=-2,
            arg2=-1,
        ),
    )
    out = _fire(OM.MATMUL_T_CONCAT, t0)
    assert out is not None and out.op == "concat"
    got = _eval(out, env)
    ref = _eval(t0, env)
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_matmul_t_concat_check_vetoes():
    c = OM.MATMUL_T_CONCAT.check
    good = _mtc_bound(_v("q", 4, 6), _v("k1", 3, 6), _v("k2", 5, 6))
    assert c(good)
    # dims not ints → line 327
    assert not c({**good, "$attr:KD": "x"})
    assert not c({**good, "$attr:T1": None})
    # non-tuple shapes → line 329
    assert not c({**good, "k1": RAW})
    assert not c({**good, "q": _ill()})
    # ranks: k1 rank<2, k2 rank mismatch, q rank<2 → line 332
    assert not c(_mtc_bound(_v("q", 4, 6), _v("k1", 6), _v("k2", 5, 6)))
    assert not c(_mtc_bound(_v("q", 4, 6), _v("k1", 3, 6), _v("k2", 2, 5, 6)))
    assert not c(_mtc_bound(_v("q", 6), _v("k1", 3, 6), _v("k2", 5, 6)))
    # key concat not on the sequence axis → line 334
    assert not c({**good, "$attr:KD": -1})
    # transpose not exactly .T → line 336 (same dim twice, or a
    # non-adjacent pair on a rank-3 key block)
    assert not c({**good, "$attr:T1": 0, "$attr:T2": 0})
    assert not c(
        _mtc_bound(
            _v("q", 4, 6),
            _v("k1", 2, 3, 6),
            _v("k2", 2, 5, 6),
            t1=0,
            t2=1,
        )
    )
    # k1 vs k2 off-axis mismatch → line 338
    assert not c(_mtc_bound(_v("q", 4, 6), _v("k1", 3, 6), _v("k2", 5, 7)))
    # q's feature axis != k's → line 341
    assert not c(_mtc_bound(_v("q", 4, 8), _v("k1", 3, 6), _v("k2", 5, 6)))
    # batch dims not broadcastable → line 342
    assert not c(
        _mtc_bound(
            _v("q", 2, 4, 6), _v("k1", 3, 3, 6), _v("k2", 3, 5, 6)
        )
    )


def test_matmul_t_concat_derive():
    good = _mtc_bound(_v("q", 4, 6), _v("k1", 3, 6), _v("k2", 5, 6))
    assert _derive(OM.MATMUL_T_CONCAT, good) == {"$attr:SD": 1}
    # non-tuple → line 351
    assert _derive(OM.MATMUL_T_CONCAT, {**good, "q": RAW}) is None


# ---------------------------------------------------------------------------
#  _cat_axis_plan — the mask-distribution planner behind every
#  masked_fill/add/where-over-concat law
# ---------------------------------------------------------------------------


def _mf_bound(s1, s2, m, v, d=-1):
    return {"s1": s1, "s2": s2, "m": m, "v": v, "$attr:D": d}


def test_masked_fill_cat_slice_fires_and_evals():
    """Mask extent == K1+K2 on the cat axis → slice mode: block i gets
    split(m, (K1,K2), -1, i).  fp64 vs serial masked_fill."""
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    m = _v("m", T, K1 + K2)
    ninf = Const(float("-inf"))
    env = {
        s1: _rand((T, K1), 40),
        s2: _rand((T, K2), 41),
        m: torch.rand(T, K1 + K2, dtype=torch.float64) > 0.5,
    }
    t0 = Op.make(
        "masked_fill", Op.make("concat", s1, s2, dim=-1), m, ninf
    )
    rule = MFC["masked_fill_cat_slice_dim"]
    out = _fire(rule, t0)
    assert out is not None and out.op == "concat"
    got = _eval(out, env)
    ref = _eval(t0, env)
    # identical computation: bitwise equal, -inf positions included
    assert torch.equal(got, ref)
    # the reuse-mode sibling must NOT fire on a sliceable mask
    assert _fire(MFC["masked_fill_cat_reuse_dim"], t0) is None


def test_masked_fill_cat_reuse_scalar_and_rank1_masks():
    """Scalar () mask (cat axis absent → md<0, line 511) and extent-1
    mask both take the reuse branch."""
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    sl, ru = MFC["masked_fill_cat_slice_dim"], MFC["masked_fill_cat_reuse_dim"]
    for m in (_v("m1"), _v("m2", 1)):
        b = _mf_bound(s1, s2, m, Const(0.0))
        assert ru.check(b) and not sl.check(b)
    # reuse derive emits DO only — no SZ/MD
    assert _derive(ru, _mf_bound(s1, s2, _v("m", 1), Const(0.0))) == {
        "$attr:DO": 1
    }
    # slice derive emits DO + SZ + MD
    assert _derive(
        sl, _mf_bound(s1, s2, _v("m", T, K1 + K2), Const(0.0))
    ) == {"$attr:DO": 1, "$attr:SZ": (K1, K2), "$attr:MD": 1}


def test_cat_axis_plan_vetoes():
    """Every veto branch of _cat_axis_plan via the masked_fill rules."""
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    sl = MFC["masked_fill_cat_slice_dim"]
    # s1 () scalar → ``and s1`` fails → line 464
    assert not sl.check(_mf_bound(Const(0.0), s2, _v("m", T, K1 + K2), Const(0.0)))
    # rank mismatch s1 vs s2 → line 464
    assert not sl.check(
        _mf_bound(s1, _v("s2", 2, T, K2), _v("m", T, K1 + K2), Const(0.0))
    )
    # non-tuple score block → line 464
    assert not sl.check(_mf_bound(RAW, s2, _v("m", T, K1 + K2), Const(0.0)))
    # D not an int → line 466
    assert not sl.check(
        _mf_bound(s1, s2, _v("m", T, K1 + K2), Const(0.0), d="x")
    )
    # ill-typed cat (off-axis dims differ) → line 469
    assert not sl.check(
        _mf_bound(s1, _v("s2", 9, K2), _v("m", T, K1 + K2), Const(0.0))
    )
    # unknown block extents → line 471-472
    assert not sl.check(
        _mf_bound(
            _v("s1", T, None), s2, _v("m", T, K1 + K2), Const(0.0)
        )
    )
    # mask shape not a tuple → line 479
    assert not sl.check(
        _mf_bound(s1, s2, _ill(), Const(0.0))
    )
    # fixed operand (v) off-axis incompatible → line 498
    assert not sl.check(
        _mf_bound(s1, s2, _v("m", T, K1 + K2), _v("v", 7, 3))
    )
    # fixed operand covering the whole cat axis needs its own slice
    # → lines 501-503
    assert not sl.check(
        _mf_bound(
            _v("s1", T, K1),
            _v("s2", T, K1),
            _v("m", T, 2 * K1),
            _v("v", T, 2 * K1),
        )
    )
    # …but exactly the block size is fine when K1 == K2 (equal blocks
    # reuse the operand) → plan exists, reuse mode
    ru = MFC["masked_fill_cat_reuse_dim"]
    assert ru.check(
        _mf_bound(
            _v("s1", T, K1),
            _v("s2", T, K1),
            _v("m", T, K1),
            _v("v", T, K1),
        )
    )
    # mask extent None → line 514
    assert not sl.check(
        _mf_bound(s1, s2, _v("m", T, None), Const(0.0))
    )
    # mask extent neither K1+K2, 1, nor the equal-block size → line 519
    assert not sl.check(
        _mf_bound(s1, s2, _v("m", T, K1 + K2 + 2), Const(0.0))
    )
    # the SLICED operand itself failing off-axis broadcast → line 507
    # (mask has the right K1+K2 extent but a wrong T dim)
    assert not sl.check(
        _mf_bound(s1, s2, _v("m", T + 3, K1 + K2), Const(0.0))
    )
    # derive returns None when the plan fails → line 534
    assert (
        _derive(sl, _mf_bound(s1, s2, _v("m", T, None), Const(0.0)))
        is None
    )


def test_add_mask_cat_both_orders_and_modes():
    """add(cat,m) and add(m,cat) — the mask distributes sliced or
    reused; the check has no fixed operand (fixed_keys=())."""
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    m_sl, m_ru = _v("m", T, K1 + K2), _v("m", 1)
    for ak in ("dim", "arg1"):
        for tag in ("cat_m", "m_cat"):  # mask-second / mask-first
            sl = AMC[f"add_{tag}_slice_{ak}"]
            ru = AMC[f"add_{tag}_reuse_{ak}"]
            assert sl.check(
                {"s1": s1, "s2": s2, "m": m_sl, "$attr:D": -1}
            )
            assert not sl.check(
                {"s1": s1, "s2": s2, "m": m_ru, "$attr:D": -1}
            )
            assert ru.check(
                {"s1": s1, "s2": s2, "m": m_ru, "$attr:D": -1}
            )
    # fp64: add(cat s, m) = cat(add blocks) — uneven blocks, additive
    # 0/-inf style mask
    env = {
        s1: _rand((T, K1), 50),
        s2: _rand((T, K2), 51),
        m_sl: _rand((T, K1 + K2), 52) * 20 - 10,
    }
    t0 = Op.make(
        "add", Op.make("concat", s1, s2, dim=-1), m_sl
    )
    out = _fire(AMC["add_cat_m_slice_dim"], t0)
    assert out is not None
    assert torch.allclose(
        _eval(out, env), _eval(t0, env), atol=1e-12, rtol=1e-12
    )
    t0r = Op.make(
        "add", m_sl, Op.make("concat", s1, s2, arg1=-1)
    )
    outr = _fire(AMC["add_m_cat_slice_arg1"], t0r)
    assert outr is not None
    assert torch.allclose(
        _eval(outr, env), _eval(t0r, env), atol=1e-12, rtol=1e-12
    )


def test_where_cat_both_positions():
    """where(m, cat, v) and where(m, v, cat): v is a fixed operand —
    it must broadcast along the cat axis."""
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    m = _v("m", T, K1 + K2)
    for ak in ("dim", "arg1"):
        for tag in ("x", "y"):  # cat in the then- / else-branch
            sl = WRC[f"where_cat_{tag}_slice_{ak}"]
            b = {
                "s1": s1,
                "s2": s2,
                "m": m,
                "v": Const(float("-inf")),
                "$attr:D": -1,
            }
            assert sl.check(b), sl.name
            # v covering the whole axis vetoes (needs its own slice)
            assert not sl.check({**b, "v": _v("v", T, K1 + K2)})
            # …but a block-sized v reuses when K1 == K2
            assert sl.check(
                {
                    "s1": _v("s1", T, K1),
                    "s2": _v("s2", T, K1),
                    "m": _v("m", T, 2 * K1),
                    "v": _v("v", T, K1),
                    "$attr:D": -1,
                }
            )


def test_where_cat_evals_fp64():
    T, K1, K2 = 4, 3, 5
    s1, s2 = _v("s1", T, K1), _v("s2", T, K2)
    m = _v("m", T, K1 + K2)
    ninf = Const(float("-inf"))
    env = {
        s1: _rand((T, K1), 60),
        s2: _rand((T, K2), 61),
        m: torch.rand(T, K1 + K2, dtype=torch.float64) > 0.5,
    }
    t0 = Op.make(
        "where", m, Op.make("concat", s1, s2, dim=-1), ninf
    )
    out = _fire(WRC["where_cat_x_slice_dim"], t0)
    assert out is not None
    # identical computation: bitwise equal, -inf positions included
    assert torch.equal(_eval(out, env), _eval(t0, env))


# ---------------------------------------------------------------------------
#  cat_hom — both operands concat'd (the chunked-mask case)
# ---------------------------------------------------------------------------


def _pair_bound(a1, a2, b1, b2, da=-1, db=-1):
    return {
        "a1": a1,
        "a2": a2,
        "b1": b1,
        "b2": b2,
        "$attr:DA": da,
        "$attr:DB": db,
    }


def test_cat_hom_add_fires_and_evals():
    T, K1, K2 = 4, 3, 5
    a1, a2 = _v("a1", T, K1), _v("a2", T, K2)
    b1, b2 = _v("b1", T, K1), _v("b2", T, K2)
    env = {
        a1: _rand((T, K1), 70),
        a2: _rand((T, K2), 71),
        b1: _rand((T, K1), 72),
        b2: _rand((T, K2), 73),
    }
    t0 = Op.make(
        "add",
        Op.make("concat", a1, a2, dim=-1),
        Op.make("concat", b1, b2, dim=-1),
    )
    out = _fire(CH["cat_hom_add_dim"], t0)
    assert out is not None and out.op == "concat"
    assert torch.allclose(
        _eval(out, env), _eval(t0, env), atol=1e-12, rtol=1e-12
    )


def test_check_cat_pair_vetoes():
    c = CH["cat_hom_add_dim"].check
    good = _pair_bound(
        _v("a1", 4, 3), _v("a2", 4, 5), _v("b1", 4, 3), _v("b2", 4, 5)
    )
    assert c(good)
    # () scalar or non-tuple members → line 657
    assert not c({**good, "a1": Const(0.0)})
    assert not c({**good, "b2": RAW})
    # cat dims not ints → line 659
    assert not c({**good, "$attr:DA": "x"})
    assert not c({**good, "$attr:DB": None})
    # rank mismatch inside a pair → line 662
    assert not c(
        _pair_bound(
            _v("a1", 4, 3), _v("a2", 2, 4, 5), _v("b1", 4, 3), _v("b2", 4, 5)
        )
    )
    # the two cats land on different axes of the result → line 667
    assert not c(
        _pair_bound(
            _v("a1", 4, 3), _v("a2", 4, 5), _v("b1", 3, 4), _v("b2", 5, 4),
            da=-1, db=0,
        )
    )
    # a pair off-axis incompatible → line 669
    assert not c(
        _pair_bound(
            _v("a1", 4, 3), _v("a2", 9, 5), _v("b1", 4, 3), _v("b2", 4, 5)
        )
    )
    # b pair off-axis incompatible → line 671
    assert not c(
        _pair_bound(
            _v("a1", 4, 3), _v("a2", 4, 5), _v("b1", 4, 3), _v("b2", 9, 5)
        )
    )
    # per-block broadcast INVALID → line 676
    assert not c(
        _pair_bound(
            _v("a1", 4, 8), _v("a2", 4, 3), _v("b1", 5, 8), _v("b2", 5, 3)
        )
    )
    # NOTE: the final ``_dim_eq(ba[i], bb[i])`` guard (line 678) is
    # unreachable after the earlier guards — every off-axis dim is
    # already pairwise equal per operand, so the two broadcast results
    # can only differ where earlier checks veto first.  Defensive —
    # suggest ``# pragma: no cover``.
    # derive vetoes propagate None
    assert _derive(CH["cat_hom_add_dim"], {**good, "$attr:DA": "x"}) is None
    assert _derive(CH["cat_hom_add_dim"], good) == {"$attr:DO": 1}


def test_cat_hom_masked_fill_fires():
    a1, a2 = _v("a1", 4, 3), _v("a2", 4, 5)
    b1, b2 = _v("b1", 4, 3), _v("b2", 4, 5)
    ninf = Const(float("-inf"))
    t0 = Op.make(
        "masked_fill",
        Op.make("concat", a1, a2, arg1=-1),
        Op.make("concat", b1, b2, arg1=-1),
        ninf,
    )
    assert _fire(CH["cat_hom_masked_fill_arg1"], t0) is not None


# ---------------------------------------------------------------------------
#  sdpa-over-concat — the flag unfold, the unmasked companion, the
#  explicit attn_mask form
# ---------------------------------------------------------------------------


def _sdpa_bound(q, k1, k2, v1, v2, kd=-2, vd=-2, **extra):
    b = {
        "q": q,
        "k1": k1,
        "k2": k2,
        "v1": v1,
        "v2": v2,
        "$attr:KD": kd,
        "$attr:VD": vd,
    }
    b.update(extra)
    return b


def test_sdpa_cat_check_vetoes():
    c = OM._check_sdpa_cat
    good = _sdpa_bound(
        _v("q", 4, 6),
        _v("k1", 3, 6),
        _v("k2", 5, 6),
        _v("v1", 3, 8),
        _v("v2", 5, 8),
    )
    assert c(good)
    # non-tuple member → line 966
    assert not c({**good, "q": RAW})
    # rank/arity mismatches → line 974
    assert not c({**good, "q": _v("q", 6)})
    assert not c({**good, "k1": _v("k1", 6)})
    assert not c({**good, "v1": _v("v1", 8)})
    assert not c({**good, "k2": _v("k2", 2, 5, 6)})
    assert not c({**good, "v2": _v("v2", 2, 5, 8)})
    # cat dims not ints → line 977
    assert not c({**good, "$attr:KD": "x"})
    assert not c({**good, "$attr:VD": None})
    # wrong cat axes → lines 979-982
    assert not c({**good, "$attr:KD": -1})
    assert not c({**good, "$attr:VD": -1})
    # k1 vs k2 off-axis mismatch → line 986
    assert not c({**good, "k2": _v("k2", 5, 7)})
    # v1 vs v2 off-axis mismatch → line 990
    assert not c({**good, "v2": _v("v2", 5, 9)})
    # q's head dim must equal both k head dims → line 993
    assert not c({**good, "q": _v("q", 4, 7)})
    assert not c({**good, "k2": _v("k2", 5, 7)})
    # per-block key/value counts must pair → line 994
    assert not c({**good, "v1": _v("v1", 4, 8)})
    # per-block batch broadcast failure → line 1003
    assert not c(
        _sdpa_bound(
            _v("q", 2, 4, 6),
            _v("k1", 3, 3, 6),
            _v("k2", 3, 5, 6),
            _v("v1", 3, 3, 8),
            _v("v2", 3, 5, 8),
        )
    )
    # nonzero dropout → line 1006
    assert not c({**good, "$attr:DP": 0.1})
    # non-numeric (or bool) scale → line 1011
    assert not c({**good, "$attr:SC": "x"})
    assert not c({**good, "$attr:SC": True})
    # no scale given and head dim unknown → line 1013 tail
    assert not c({**good, "q": _v("q", 4, None)})


def test_sdpa_cat_derive_scale():
    # bound scale passes through as float
    assert _derive(
        SDPA["sdpa_cat_5_dim"],
        _sdpa_bound(
            _v("q", 4, 6),
            _v("k1", 3, 6),
            _v("k2", 5, 6),
            _v("v1", 3, 8),
            _v("v2", 5, 8),
            **{"$attr:SC": 0.5},
        ),
    ) == {"$attr:SC": 0.5}
    # default: 1/sqrt(E) from q's head dim
    out = _derive(
        SDPA["sdpa_cat_0_dim"],
        _sdpa_bound(
            _v("q", 4, 16),
            _v("k1", 3, 16),
            _v("k2", 5, 16),
            _v("v1", 3, 8),
            _v("v2", 5, 8),
        ),
    )
    assert out == {"$attr:SC": 16.0**-0.5}
    # head dim unknown → None (line 1025)
    assert (
        _derive(
            SDPA["sdpa_cat_0_dim"],
            _sdpa_bound(
                _v("q", 4, None),
                _v("k1", 3, None),
                _v("k2", 5, None),
                _v("v1", 3, 8),
                _v("v2", 5, 8),
            ),
        )
        is None
    )


def test_sdpa_cat_fires_and_evals_fp64():
    """sdpa(q, cat k, cat v) → om_apply(om_elem(cat scaled-scores,
    cat v)) — fp64 vs torch.nn.functional.scaled_dot_product_attention."""
    import torch.nn.functional as F

    T, E, K1, K2, Dv = 4, 6, 3, 5, 8
    q, k1, k2 = _v("q", T, E), _v("k1", K1, E), _v("k2", K2, E)
    v1, v2 = _v("v1", K1, Dv), _v("v2", K2, Dv)
    env = {
        q: _rand((T, E), 80),
        k1: _rand((K1, E), 81),
        k2: _rand((K2, E), 82),
        v1: _rand((K1, Dv), 83),
        v2: _rand((K2, Dv), 84),
    }
    t0 = Op.make(
        "sdpa",
        q,
        Op.make("concat", k1, k2, dim=-2),
        Op.make("concat", v1, v2, dim=-2),
    )
    out = _fire(SDPA["sdpa_cat_0_dim"], t0)
    assert out is not None and out.op == "om_apply"
    ref = F.scaled_dot_product_attention(
        env[q],
        torch.cat([env[k1], env[k2]], dim=-2),
        torch.cat([env[v1], env[v2]], dim=-2),
    )
    got = _eval(out, env)
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_sdpa_cat_causal_fires_and_evals_fp64():
    """is_causal unfolds into a materialised cmask; the whole thing
    still equals torch's fused causal attention in fp64."""
    import torch.nn.functional as F

    T, E, K1, K2, Dv = 4, 6, 3, 5, 8
    q, k1, k2 = _v("q", T, E), _v("k1", K1, E), _v("k2", K2, E)
    v1, v2 = _v("v1", K1, Dv), _v("v2", K2, Dv)
    env = {
        q: _rand((T, E), 90),
        k1: _rand((K1, E), 91),
        k2: _rand((K2, E), 92),
        v1: _rand((K1, Dv), 93),
        v2: _rand((K2, Dv), 94),
    }
    t0 = Op.make(
        "sdpa",
        q,
        Op.make("concat", k1, k2, dim=-2),
        Op.make("concat", v1, v2, dim=-2),
        is_causal=True,
    )
    out = _fire(SDPA["sdpa_cat_causal_0_dim"], t0)
    assert out is not None and out.op == "om_apply"
    ref = F.scaled_dot_product_attention(
        env[q],
        torch.cat([env[k1], env[k2]], dim=-2),
        torch.cat([env[v1], env[v2]], dim=-2),
        is_causal=True,
    )
    got = _eval(out, env)
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_sdpa_mask_cat_check_and_eval():
    """Explicit attn_mask: it must broadcast against (…,Tq,K1+K2) with
    the key axis last.  Both float-bias and bool keep-mask eval."""
    import torch.nn.functional as F

    c = OM._check_sdpa_mask_cat
    T, E, K1, K2, Dv = 4, 6, 3, 5, 8
    base = _sdpa_bound(
        _v("q", T, E),
        _v("k1", K1, E),
        _v("k2", K2, E),
        _v("v1", K1, Dv),
        _v("v2", K2, Dv),
    )
    # the inner _check_sdpa_cat veto propagates → line 1116
    assert not c({**base, "q": _v("q", 7), "m": _v("m", T, K1 + K2)})
    # mask shape not a tuple → line 1121
    assert not c({**base, "m": RAW})
    # non-int key extents: split sizes underivable → line 1124
    assert not c({**base, "k1": _v("k1", None, E), "m": _v("m", T, K1 + K2)})
    # NOTE line 1129 (``bb is _INVALID``) is unreachable: the inner
    # _check_sdpa_cat already proved the same broadcast — defensive,
    # suggest ``# pragma: no cover``.
    # good masks: key-axis extent K1+K2 or 1
    assert c({**base, "m": _v("m", T, K1 + K2)})
    assert c({**base, "m": _v("m", 1)})
    # mask not broadcastable against scores → line 1132-1133
    assert not c({**base, "m": _v("m", T + 3, K1 + K2)})
    # mask whose last dim is neither Tk nor broadcasts to it is still
    # rejected via b[-1] — extent 1 broadcasts, so use extent Tq+1:
    assert not c({**base, "m": _v("m", T, K1 + K2 + 1)})

    # fp64 end-to-end through the mask rule
    q, k1, k2 = _v("q", T, E), _v("k1", K1, E), _v("k2", K2, E)
    v1, v2 = _v("v1", K1, Dv), _v("v2", K2, Dv)
    m = _v("m", T, K1 + K2)
    env = {
        q: _rand((T, E), 100),
        k1: _rand((K1, E), 101),
        k2: _rand((K2, E), 102),
        v1: _rand((K1, Dv), 103),
        v2: _rand((K2, Dv), 104),
        m: _rand((T, K1 + K2), 105) * 8 - 4,
    }
    t0 = Op.make(
        "sdpa",
        q,
        Op.make("concat", k1, k2, dim=-2),
        Op.make("concat", v1, v2, dim=-2),
        m,
    )
    out = _fire(SDPA["sdpa_cat_mask_0_dim"], t0)
    assert out is not None and out.op == "om_apply"
    ref = F.scaled_dot_product_attention(
        env[q],
        torch.cat([env[k1], env[k2]], dim=-2),
        torch.cat([env[v1], env[v2]], dim=-2),
        attn_mask=env[m],
    )
    got = _eval(out, env)
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_attnbias_and_cmask_bindings():
    """The generator bindings behave as documented: bool keep-masks
    become 0/-inf biases, float masks pass through; cmask is the
    strict upper triangle of the score plane."""
    b = OM.TORCH_BINDINGS["attnbias"]
    keep = torch.tensor([[True, False], [True, True]])
    out = b(keep)
    assert torch.equal(
        out,
        torch.tensor([[0.0, float("-inf")], [0.0, 0.0]]),
    )
    fm = torch.randn(3, 4, dtype=torch.float64)
    assert b(fm) is fm
    cm = OM.TORCH_BINDINGS["cmask"]
    x = torch.zeros(2, 3)
    bad = cm(x)
    assert bad.dtype == torch.bool
    assert torch.equal(bad, ~torch.tril(torch.ones(2, 3, dtype=torch.bool)))
    # off>0 shrinks the allowed triangle: keep = tril(-off)
    bad1 = cm(x, off=1)
    assert torch.equal(
        bad1, ~torch.tril(torch.ones(2, 3, dtype=torch.bool), diagonal=-1)
    )
    fl = OM.TORCH_BINDINGS["fill"]
    assert torch.equal(fl(x, value=2.5), torch.full_like(x, 2.5))
