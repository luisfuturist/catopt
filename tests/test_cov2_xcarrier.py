"""Coverage tests for catopt.xcarrier — the cross-carrier guards,
the carrier-aware ``_xshape`` resolver, the view-commute check/derive
pairs, and the non-local passes' veto paths.

Each check is driven BOTH ways: a bound satisfying the contract (and a
real ``apply_rewrite_at`` fire whose result is fp64-evaluated against
a serial reference) plus bounds violating every clause — ``None`` /
``_INVALID`` shapes, ``()`` scalar ranks, non-leaf scales, feature-
axis slices, projection-vs-domain mismatches, and every documented
veto in the non-local gather/lift passes (cyclic carriers, shared
sub-trees, non-pack map members, feature-axis stacks).

``bound`` dicts are minted by hand — the e-graph's convention
(metavar -> concrete term, ``"$attr:X"`` -> attr value) — since the
interesting veto inputs (ill-typed operands, non-int attrs) cannot
always be produced by a well-typed minted term.
"""

import torch

import catopt.xcarrier as XC
from catopt import meta
from catopt.egraph import EGraph
from catopt.ir import Const, Op, TensorType, Var
from catopt.typing import _shape_of

torch.manual_seed(0)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(shape, seed: int = 0):
    g = torch.Generator().manual_seed(7000 + seed)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _ill() -> Op:
    """Provably ill-typed term — its shape is ``_INVALID``."""
    return Op.make("add", _v("zz_a", 2, 3), _v("zz_b", 5, 4))


#: A non-term leaf — ``_shape_of`` reports ``None`` for it.
RAW = "raw_leaf"


def _fire(rule, term):
    return meta.apply_rewrite_at(rule, term, ())


def _eval(term, env):
    return meta._eval_term(term, env)


def _law(rule, t0, env, tol=1e-10):
    """Fire *rule* at the root and fp64-check both sides."""
    out = _fire(rule, t0)
    assert out is not None, f"{rule.name} did not fire"
    assert meta._eval_allclose(_eval(t0, env), _eval(out, env), tol=tol)
    return out


# ---------------------------------------------------------------------------
#  _xshape — the true value shape through carrier ops
# ---------------------------------------------------------------------------


def test_xshape_applyd_variants():
    h = _v("h", 4)
    # applyd(aff_diag(a,b),h) → broadcast of all factors: (5,4)
    t = Op.make(
        "applyd", Op.make("aff_diag", _v("a", 5, 4), _v("b", 5, 4)), h
    )
    assert XC._xshape(t) == (5, 4)
    # broadcasting case: a (1,4) against b (5,4)
    t = Op.make(
        "applyd", Op.make("aff_diag", _v("a", 1, 4), _v("b", 5, 4)), h
    )
    assert XC._xshape(t) == (5, 4)
    # opaque map leaf → broadcast(map-shape, h-shape): line 233-234
    t = Op.make("applyd", _v("f", 5, 4), h)
    assert XC._xshape(t) == (5, 4)


def test_xshape_apply_variants():
    h, c = _v("h", 4), _v("c", 6)
    # apply(aff(A,c),h) → broadcast(A[:-1], c): line 242
    t = Op.make("apply", Op.make("aff", _v("A", 6, 4), c), h)
    assert XC._xshape(t) == (6,)
    # A resolves to a non-rank-≥1 shape (scalar) → return cs: line 243
    t = Op.make("apply", Op.make("aff", Const(1.0), c), h)
    assert XC._xshape(t) == (6,)
    # opaque map rank ≥ 2 → fs[:-1]: lines 245-246
    t = Op.make("apply", _v("f", 6, 4), h)
    assert XC._xshape(t) == (6,)
    # opaque map rank < 2 → cost-model fallback: line 247
    t = Op.make("apply", _v("f", 6), h)
    assert XC._xshape(t) == _shape_of(t)


def test_xshape_projection_ops():
    f = _v("f", 5, 4)
    # leaf map → the map's own shape: lines 252 / 257-260
    assert XC._xshape(Op.make("affd_a", f)) == (5, 4)
    assert XC._xshape(Op.make("affd_b", f)) == (5, 4)
    assert XC._xshape(Op.make("aff_A", f)) == (5, 4)
    # aff_b of a rank-≥2 leaf reports the OUTPUT shape fs[:-1]
    assert XC._xshape(Op.make("aff_b", f)) == (5,)
    # aff_b of a rank-1 leaf reports fs unchanged (line 260)
    assert XC._xshape(Op.make("aff_b", _v("g", 6))) == (6,)
    # op maps → the projected component's shape
    a, b, A = _v("a", 5, 4), _v("b", 5, 4), _v("A", 5, 4, 4)
    assert XC._xshape(Op.make("affd_a", Op.make("aff_diag", a, b))) == (5, 4)
    assert XC._xshape(Op.make("affd_b", Op.make("aff_diag", a, b))) == (5, 4)
    assert XC._xshape(Op.make("aff_A", Op.make("aff", A, b))) == (5, 4, 4)
    assert XC._xshape(Op.make("aff_b", Op.make("aff", A, b))) == (5, 4)


def test_xshape_om_family():
    s = _v("s", 4, 7)
    # om_elem_affd concrete → score prefix ++ value dim: line 266
    t = Op.make(
        "om_elem_affd",
        s, _v("a", 7, 3), _v("b", 7, 3), _v("h", 3),
    )
    assert XC._xshape(t) == (4, 3)
    # non-concrete member → fallback: line 267
    t = Op.make(
        "om_elem_affd",
        _v("s", 4, None), _v("a", 7, 3), _v("b", 7, 3), _v("h", 3),
    )
    assert XC._xshape(t) == _shape_of(t)
    # om_elem_aff dense: (…,Tq,d) = ss[:-1] ++ A[-2]: line 271
    t = Op.make(
        "om_elem_aff",
        s, _v("A", 7, 3, 4), _v("b", 7, 3), _v("h", 4),
    )
    assert XC._xshape(t) == (4, 3)
    # A rank < 2 or non-concrete → fallback: line 272
    t = Op.make(
        "om_elem_aff",
        s, _v("A", 7), _v("b", 7, 3), _v("h", 4),
    )
    assert XC._xshape(t) == _shape_of(t)
    t = Op.make(
        "om_elem_aff",
        s, _v("A", 7, None, 4), _v("b", 7, 3), _v("h", 4),
    )
    assert XC._xshape(t) == _shape_of(t)
    # omd_elem dense fiber: sa[-2] is the value axis (line 277)
    t = Op.make("omd_elem", s, _v("a", 7, 3, 4), _v("b", 7, 3))
    assert XC._xshape(t) == (4, 3)
    # omd_elem diagonal fiber: sa[-1]
    t = Op.make("omd_elem", s, _v("a", 7, 3), _v("b", 7, 3))
    assert XC._xshape(t) == (4, 3)
    # non-concrete → line 279
    t = Op.make("omd_elem", _v("s", 4, None), _v("a", 7, 3), _v("b", 7, 3))
    assert XC._xshape(t) == _shape_of(t)
    # omd_compose / omd_apply(m) report the carrier's own shape
    f = _v("f", 4, 3)
    assert XC._xshape(
        Op.make("omd_compose", f, _v("g", 4, 3), validate=False)
    ) == (4, 3)
    assert XC._xshape(
        Op.make("omd_apply", f, _v("h", 3), validate=False)
    ) == (4, 3)
    assert XC._xshape(
        Op.make("omd_applym", f, _v("h", 3), validate=False)
    ) == (4, 3)
    # omd(m,l,fa,fb) reports fb's shape: line 285
    t = Op.make(
        "omd",
        _v("m", 4, 1), _v("l", 4, 1), _v("fa", 4, 3), _v("fb", 4, 3),
        validate=False,
    )
    assert XC._xshape(t) == (4, 3)


def test_xshape_generic_op_surrogate_path():
    """A carrier member nested in an ordinary op: the child is replaced
    by a surrogate leaf carrying its corrected (concrete) shape; a
    non-concrete child is left in place (the 293->290 branch)."""
    good = Op.make("omd_elem", _v("s", 4, 7), _v("a", 7, 3), _v("b", 7, 3))
    bad = Op.make(
        "omd_elem", _v("s", 4, None), _v("a", 7, 3), _v("b", 7, 3)
    )
    # one concrete + one non-concrete carrier child under add
    t = Op.make("add", bad, good)
    assert XC._xshape(t) == (4, None)
    # non-carrier ordinary terms still go through _shape_of
    t2 = Op.make("matmul", _v("x", 3, 4), _v("y", 4, 5))
    assert XC._xshape(t2) == _shape_of(t2)
    # a Var/leaf passes straight through
    assert XC._xshape(_v("z", 2, 2)) == (2, 2)


def test_omd_torch_binding():
    m = _rand((4, 1), 1)
    l_ = _rand((4, 1), 2)
    fa = _rand((4, 3), 3)
    fb = _rand((4, 3), 4)
    out = XC.TORCH_BINDINGS["omd"](m, l_, fa, fb)
    assert out == (m, l_, fa, fb)


def test_xshape_memo_shared_subterm():
    """A DAG node visited twice hits the id-memo (line 210) and
    returns the SAME result."""
    t = Op.make("omd_elem", _v("s", 4, 7), _v("a", 7, 3), _v("b", 7, 3))
    dag = Op.make("add", t, t)
    memo = {}
    assert XC._xshape(dag, memo) == (4, 3)
    assert memo[id(t)] == (4, 3)


# ---------------------------------------------------------------------------
#  Readout / promotion guards
# ---------------------------------------------------------------------------


def test_matmul_applyd_vec_guard():
    """W (o,d) contracting the FEATURE axis of a (d,) diagonal map:
    the dense promotion."""
    o, d = 6, 4
    W, a, b, h = _v("W", o, d), _v("a", d), _v("b", d), _v("h", d)
    bound = {"W": W, "a": a, "b": b, "h": h}
    assert XC.XC_MATMUL_APPLYD_VEC.check(bound)
    env = {
        W: _rand((o, d), 10), a: _rand((d,), 11),
        b: _rand((d,), 12), h: _rand((d,), 13),
    }
    t0 = Op.make(
        "matmul", W, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    _law(XC.XC_MATMUL_APPLYD_VEC, t0, env)
    # non-concrete / non-tuple members → line 446
    assert not XC.XC_MATMUL_APPLYD_VEC.check({**bound, "W": RAW})
    assert not XC.XC_MATMUL_APPLYD_VEC.check({**bound, "a": _v("a", None)})
    # rank/equality vetoes
    assert not XC.XC_MATMUL_APPLYD_VEC.check({**bound, "W": _v("W", o, d, 2)})
    assert not XC.XC_MATMUL_APPLYD_VEC.check({**bound, "b": _v("b", d + 1)})
    assert not XC.XC_MATMUL_APPLYD_VEC.check({**bound, "W": _v("W", o, d + 1)})


def test_matmul_apply_guard():
    """W (o,i) through the dense carrier: A (i,i), c=h=(i,)."""
    o, i = 6, 4
    W, A, c, h = _v("W", o, i), _v("A", i, i), _v("c", i), _v("h", i)
    bound = {"W": W, "A": A, "c": c, "h": h}
    assert XC.XC_MATMUL_APPLY.check(bound)
    env = {
        W: _rand((o, i), 20), A: _rand((i, i), 21),
        c: _rand((i,), 22), h: _rand((i,), 23),
    }
    t0 = Op.make("matmul", W, Op.make("apply", Op.make("aff", A, c), h))
    _law(XC.XC_MATMUL_APPLY, t0, env)
    # non-concrete → line 458
    assert not XC.XC_MATMUL_APPLY.check({**bound, "A": _v("A", None, i)})
    assert not XC.XC_MATMUL_APPLY.check({**bound, "h": RAW})
    # inner condition failures
    assert not XC.XC_MATMUL_APPLY.check({**bound, "A": _v("A", i, i + 1)})
    assert not XC.XC_MATMUL_APPLY.check({**bound, "c": _v("c", i + 1)})
    assert not XC.XC_MATMUL_APPLY.check({**bound, "W": _v("W", o)})


def test_linear_applyd_guard():
    """linear(applyd(aff_diag(a,b),h), W): a,b (…,i) equal, h (i,),
    W (o,i) — the batched (T,i) promotion."""
    T, o, i = 5, 6, 4
    W, a, b, h = (
        _v("W", o, i), _v("a", T, i), _v("b", T, i), _v("h", i),
    )
    bound = {"W": W, "a": a, "b": b, "h": h}
    assert XC.XC_LINEAR_APPLYD.check(bound)
    env = {
        W: _rand((o, i), 30), a: _rand((T, i), 31),
        b: _rand((T, i), 32), h: _rand((i,), 33),
    }
    t0 = Op.make(
        "linear", Op.make("applyd", Op.make("aff_diag", a, b), h), W
    )
    _law(XC.XC_LINEAR_APPLYD, t0, env)
    # non-concrete → line 475
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "W": RAW})
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "a": _v("a", T, None)})
    # condition failures: W rank, a!=b, h not (i,), feature mismatch
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "W": _v("W", o)})
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "b": _v("b", T, i + 1)})
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "h": _v("h", i, i)})
    assert not XC.XC_LINEAR_APPLYD.check({**bound, "h": _v("h", i + 1)})


def test_linear_apply_guard():
    """linear(apply(aff(A,c),h), W): A (…,i,i), W (o,i), h (i,),
    c broadcastable to (…,i)."""
    T, o, i = 3, 5, 4
    W, A, c, h = (
        _v("W", o, i), _v("A", T, i, i), _v("c", i), _v("h", i),
    )
    bound = {"W": W, "A": A, "c": c, "h": h}
    assert XC.XC_LINEAR_APPLY.check(bound)
    env = {
        W: _rand((o, i), 40), A: _rand((T, i, i), 41),
        c: _rand((i,), 42), h: _rand((i,), 43),
    }
    t0 = Op.make(
        "linear", Op.make("apply", Op.make("aff", A, c), h), W
    )
    _law(XC.XC_LINEAR_APPLY, t0, env)
    # non-concrete → line 490
    assert not XC.XC_LINEAR_APPLY.check({**bound, "c": RAW})
    # inner condition → line 499
    assert not XC.XC_LINEAR_APPLY.check({**bound, "W": _v("W", o)})
    assert not XC.XC_LINEAR_APPLY.check({**bound, "A": _v("A", T, i, i + 1)})
    assert not XC.XC_LINEAR_APPLY.check({**bound, "h": _v("h", i, i)})
    assert not XC.XC_LINEAR_APPLY.check({**bound, "h": _v("h", i + 1)})
    # c not broadcastable against A[:-1] → line 500
    assert not XC.XC_LINEAR_APPLY.check({**bound, "c": _v("c", 7, 2, i)})


# ---------------------------------------------------------------------------
#  Scalar scale / same-state add guards
# ---------------------------------------------------------------------------


def test_scale_applyd_guard():
    """mul(applyd(aff_diag(a,b),h), r): r a leaf broadcasting against
    the state shape — scalar (), (d,), or the full a-shape."""
    d = 4
    a, b, h = _v("a", d), _v("b", d), _v("h", d)
    # leaf scalar / (d,) / full shape all accepted
    for r in (_v("r"), _v("r", d), Const(2.0)):
        bound = {"r": r, "a": a, "b": b, "h": h}
        assert XC.XC_SCALE_APPLYD.check(bound)
    env = {
        a: _rand((d,), 50), b: _rand((d,), 51), h: _rand((d,), 52),
    }
    t0 = Op.make(
        "mul",
        Op.make("applyd", Op.make("aff_diag", a, b), h),
        Const(3.0),
    )
    _law(XC.XC_SCALE_APPLYD, t0, env)
    t0 = Op.make(
        "mul",
        Const(3.0),
        Op.make("applyd", Op.make("aff_diag", a, b), h),
    )
    _law(XC.XC_SCALE_APPLYD_PRE, t0, env)
    # a computed (Op) scale is vetoed by _named_scale — recurrences
    # route through affd_compose instead
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": Op.make("mul", a, b), "a": a, "b": b, "h": h}
    )
    # non-concrete members → line 539
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": Const(1.0), "a": _v("a", None), "b": b, "h": h}
    )
    # a!=b / h not (d,) / h[-1]!=a[-1] → line 541
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": Const(1.0), "a": a, "b": _v("b", d + 1), "h": h}
    )
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": Const(1.0), "a": a, "b": b, "h": _v("h", d, d)}
    )
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": Const(1.0), "a": a, "b": b, "h": _v("h", d + 1)}
    )
    # r of an unrelated shape (not scalar/(d,)/full) → _scalar_or_broadcast
    assert not XC.XC_SCALE_APPLYD.check(
        {"r": _v("r", d + 1), "a": a, "b": b, "h": h}
    )


def test_scale_apply_guard():
    """mul(apply(aff(A,c),h), r): r scalar, A[:-1]-shaped, or a
    (…,1) row scale."""
    o, i = 5, 4
    A, c, h = _v("A", o, i, i), _v("c", o, i), _v("h", i)
    bound = {"r": Const(2.0), "A": A, "c": c, "h": h}
    assert XC.XC_SCALE_APPLY.check(bound)          # scalar → line 571
    assert XC.XC_SCALE_APPLY.check({**bound, "r": _v("r", o, i)})  # == A[:-1]
    assert XC.XC_SCALE_APPLY.check({**bound, "r": _v("r", o, 1)})  # row scale
    env = {
        A: _rand((o, i, i), 60), c: _rand((o, i), 61),
        h: _rand((i,), 62),
    }
    t0 = Op.make(
        "mul", Op.make("apply", Op.make("aff", A, c), h), Const(2.0)
    )
    _law(XC.XC_SCALE_APPLY, t0, env)
    # non-concrete → line 552
    assert not XC.XC_SCALE_APPLY.check({**bound, "A": _v("A", o, None, i)})
    # non-leaf r → line 554
    assert not XC.XC_SCALE_APPLY.check({**bound, "r": Op.make("mul", c, h)})
    # A/h contract failure → line 561
    assert not XC.XC_SCALE_APPLY.check({**bound, "A": _v("A", o, i, i + 1)})
    assert not XC.XC_SCALE_APPLY.check({**bound, "h": _v("h", i, i)})
    # c not broadcastable to A[:-1] → line 563
    assert not XC.XC_SCALE_APPLY.check({**bound, "c": _v("c", o + 2, i)})
    # r unknown / not broadcastable → line 566
    assert not XC.XC_SCALE_APPLY.check({**bound, "r": RAW})
    assert not XC.XC_SCALE_APPLY.check({**bound, "r": _v("r", 9)})
    # r of an unrelated full rank that isn't scalar/out-shape/(…,1)
    # → the trailing return at line 574-578 → False
    assert not XC.XC_SCALE_APPLY.check({**bound, "r": _v("r", o, 2)})


def test_add_applyd_guard():
    T, d = 5, 4
    f, g, h = _v("f", T, d), _v("g", T, d), _v("h", d)
    bound = {"f": f, "g": g, "h": h}
    assert XC.XC_ADD_APPLYD.check(bound)
    fa, fb = _v("fa", T, d), _v("fb", T, d)
    ga, gb = _v("ga", T, d), _v("gb", T, d)
    env = {
        fa: _rand((T, d), 70), fb: _rand((T, d), 71),
        ga: _rand((T, d), 72), gb: _rand((T, d), 73),
        h: _rand((d,), 74),
    }
    t0 = Op.make(
        "add",
        Op.make("applyd", Op.make("aff_diag", fa, fb), h),
        Op.make("applyd", Op.make("aff_diag", ga, gb), h),
    )
    _law(XC.XC_ADD_APPLYD, t0, env)
    # non-concrete → line 587
    assert not XC.XC_ADD_APPLYD.check({**bound, "f": RAW})
    # f!=g or h not the broadcast vector → line 588
    assert not XC.XC_ADD_APPLYD.check({**bound, "g": _v("g", T, d + 1)})
    assert not XC.XC_ADD_APPLYD.check({**bound, "h": _v("h", d, d)})


def test_add_apply_guard():
    T, i = 3, 4
    f, g, h = _v("f", T, i, i), _v("g", T, i, i), _v("h", i)
    bound = {"f": f, "g": g, "h": h}
    assert XC.XC_ADD_APPLY.check(bound)
    # non-concrete → line 596
    assert not XC.XC_ADD_APPLY.check({**bound, "g": RAW})
    # non-square / h mismatch → the inner condition
    assert not XC.XC_ADD_APPLY.check({**bound, "g": _v("g", T, i, i + 1)})
    assert not XC.XC_ADD_APPLY.check({**bound, "h": _v("h", i + 1)})
    fa, fb = _v("fa", T, i, i), _v("fb", T, i)
    ga, gb = _v("ga", T, i, i), _v("gb", T, i)
    env = {
        fa: _rand((T, i, i), 80), fb: _rand((T, i), 81),
        ga: _rand((T, i, i), 82), gb: _rand((T, i), 83),
        h: _rand((i,), 84),
    }
    t0 = Op.make(
        "add",
        Op.make("apply", Op.make("aff", fa, fb), h),
        Op.make("apply", Op.make("aff", ga, gb), h),
    )
    _law(XC.XC_ADD_APPLY, t0, env)


# ---------------------------------------------------------------------------
#  chunk/split guards
# ---------------------------------------------------------------------------


def test_chunk_applyd_guard():
    T, d = 6, 4
    a, b, h = _v("a", T, d), _v("b", T, d), _v("h", d)
    bound = {"a": a, "b": b, "h": h, "$attr:D": 0}
    assert XC.XC_CHUNK_APPLYD.check(bound)
    env = {
        a: _rand((T, d), 90), b: _rand((T, d), 91), h: _rand((d,), 92),
    }
    t0 = Op.make(
        "chunk",
        Op.make("applyd", Op.make("aff_diag", a, b), h),
        chunks=3, dim=0, index=1,
    )
    _law(XC.XC_CHUNK_APPLYD, t0, env)
    # non-concrete or non-int D → line 614
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "a": RAW})
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "$attr:D": "x"})
    # a!=b / rank<2 / bad h → line 616
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "b": _v("b", T, d + 1)})
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "a": _v("a", d)})
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "h": _v("h", d, d)})
    # feature-axis slice → line 617 (D normalises to last axis)
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "$attr:D": -1})
    assert not XC.XC_CHUNK_APPLYD.check({**bound, "$attr:D": 1})


def test_split_applyd_guard_sizes():
    """split needs int sizes on top of the chunk contract."""
    T, d = 6, 4
    a, b, h = _v("a", T, d), _v("b", T, d), _v("h", d)
    good = {"a": a, "b": b, "h": h, "$attr:D": 0, "$attr:SZ": (2, 4)}
    assert XC.XC_SPLIT_APPLYD.check(good)
    # non-int size members → _check_split_sizes_int
    assert not XC.XC_SPLIT_APPLYD.check({**good, "$attr:SZ": (2, "x")})
    assert not XC.XC_SPLIT_APPLYD.check({**good, "$attr:SZ": "nope"})


def test_chunk_apply_guard_and_derive():
    """chunk(apply(aff(A,c),h)): A = c ++ (i,); the slice dim in c's
    coords maps to the same leading index of A (derive DA)."""
    T, d, i = 6, 4, 3
    A, c, h = _v("A", T, d, i), _v("c", T, d), _v("h", i)
    bound = {"A": A, "c": c, "h": h, "$attr:D": 1}
    assert XC.XC_CHUNK_APPLY.check(bound)
    assert XC.XC_CHUNK_APPLY.derive(bound) == {"$attr:DA": 1}
    # slice on the last (still a value) axis is sound for the dense map
    bound2 = {"A": A, "c": c, "h": h, "$attr:D": -1}
    assert XC.XC_CHUNK_APPLY.check(bound2)
    assert XC.XC_CHUNK_APPLY.derive(bound2) == {"$attr:DA": 1}
    env = {
        A: _rand((T, d, i), 100), c: _rand((T, d), 101),
        h: _rand((i,), 102),
    }
    t0 = Op.make(
        "chunk",
        Op.make("apply", Op.make("aff", A, c), h),
        chunks=2, dim=0, index=1,
    )
    _law(XC.XC_CHUNK_APPLY, t0, env)
    # non-concrete / non-int D → line 629 (returns (False, None))
    assert not XC.XC_CHUNK_APPLY.check({**bound, "A": RAW})
    assert not XC.XC_CHUNK_APPLY.check({**bound, "$attr:D": "x"})
    assert XC.XC_CHUNK_APPLY.derive({**bound, "A": RAW}) is None
    # A != c++(i) / bad h → line 637
    assert not XC.XC_CHUNK_APPLY.check({**bound, "A": _v("A", T, d, i + 1)})
    assert not XC.XC_CHUNK_APPLY.check({**bound, "A": _v("A", T + 1, d, i)})
    assert not XC.XC_CHUNK_APPLY.check({**bound, "h": _v("h", i + 1)})
    assert not XC.XC_CHUNK_APPLY.check({**bound, "c": _v("c", T, d, i)})
    # split variant: same contract + int sizes
    sg = {**bound, "$attr:SZ": (3, 3)}
    assert XC.XC_SPLIT_APPLY.check(sg)
    assert not XC.XC_SPLIT_APPLY.check({**sg, "$attr:SZ": (3, None)})
    # uneven split fp64
    env2 = {
        A: _rand((T, d, i), 103), c: _rand((T, d), 104),
        h: _rand((i,), 105),
    }
    t0 = Op.make(
        "split",
        Op.make("apply", Op.make("aff", A, c), h),
        sizes=(2, 4), dim=0, index=1,
    )
    _law(XC.XC_SPLIT_APPLY, t0, env2)


# ---------------------------------------------------------------------------
#  om element / omd guards
# ---------------------------------------------------------------------------


def test_om_elem_affd_evaluates_vs_serial():
    """The fused element's (m,l,a) triple equals the serial
    om_elem(s, a⊙h+b) — fp64."""
    Tq, K, d = 4, 7, 3
    s, a, b, h = (
        _v("s", Tq, K), _v("a", K, d), _v("b", K, d), _v("h", d),
    )
    env = {
        s: _rand((Tq, K), 190), a: _rand((K, d), 191),
        b: _rand((K, d), 192), h: _rand((d,), 193),
    }
    fused = Op.make("om_elem_affd", s, a, b, h)
    serial = Op.make(
        "om_elem", s, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    out = _fire(XC.XC_OM_ELEM_AFFD, serial)
    assert out is not None and out.op == "om_elem_affd"
    assert meta._eval_allclose(
        _eval(fused, env), _eval(serial, env), tol=1e-12
    )
    # and the reverse law unfolds back
    assert _fire(XC.XC_OM_ELEM_AFFD_REV, fused) is not None


def test_om_elem_aff_evaluates_vs_serial():
    """Dense-fiber fused element: numerator = reshape(e@A)@h + e@b."""
    Tq, K, d, i = 4, 7, 3, 2
    s, A, b, h = (
        _v("s", Tq, K), _v("A", K, d, i), _v("b", K, d), _v("h", i),
    )
    env = {
        s: _rand((Tq, K), 200), A: _rand((K, d, i), 201),
        b: _rand((K, d), 202), h: _rand((i,), 203),
    }
    fused = Op.make("om_elem_aff", s, A, b, h)
    serial = Op.make(
        "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
    )
    out = _fire(XC.XC_OM_ELEM_AFF, serial)
    assert out is not None
    assert meta._eval_allclose(
        _eval(fused, env), _eval(serial, env), tol=1e-12
    )
    assert _fire(XC.XC_OM_ELEM_AFF_REV, fused) is not None


def test_omd_lift_checks_and_dense_eval():
    """XC_OMD_LIFT/DENSE .check delegate to the elem guards (lines
    708, 712); the dense fiber's deferred element applies h through
    omd_applym — fp64 vs serial om_apply."""
    Tq, K, d, i = 4, 7, 3, 2
    bound_d = {
        "s": _v("s", Tq, K), "a": _v("a", K, d), "b": _v("b", K, d),
        "h": _v("h", d),
    }
    assert XC.XC_OMD_LIFT.check(bound_d)
    assert not XC.XC_OMD_LIFT.check({**bound_d, "h": _v("h", d + 1)})
    bound_m = {
        "s": _v("s", Tq, K), "A": _v("A", K, d, i),
        "b": _v("b", K, d), "h": _v("h", i),
    }
    assert XC.XC_OMD_LIFT_DENSE.check(bound_m)
    assert not XC.XC_OMD_LIFT_DENSE.check({**bound_m, "A": _v("A", K, d)})
    # fp64: omd_applym(omd_elem(s,A,b), h) — the dense-fiber element
    # evaluates its (…,Tq,d,i) coefficient then contracts h
    s, A, b, h = bound_m["s"], bound_m["A"], bound_m["b"], bound_m["h"]
    env = {
        s: _rand((Tq, K), 210), A: _rand((K, d, i), 211),
        b: _rand((K, d), 212), h: _rand((i,), 213),
    }
    t0 = Op.make(
        "om_apply",
        Op.make("om_elem", s, Op.make("apply", Op.make("aff", A, b), h)),
    )
    out = _fire(XC.XC_OMD_LIFT_DENSE, t0)
    assert out is not None and out.op == "omd_applym"
    got = _eval(out, env)
    ref = _eval(t0, env)
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)
    # the diag-fiber lift too: omd_apply(omd_elem, h)
    s, a, b, h = bound_d["s"], bound_d["a"], bound_d["b"], bound_d["h"]
    env = {
        s: _rand((Tq, K), 214), a: _rand((K, d), 215),
        b: _rand((K, d), 216), h: _rand((d,), 217),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_elem", s, Op.make("applyd", Op.make("aff_diag", a, b), h)
        ),
    )
    out = _fire(XC.XC_OMD_LIFT, t0)
    assert out is not None and out.op == "omd_apply"
    assert torch.allclose(
        _eval(out, env), _eval(t0, env), atol=1e-12, rtol=1e-12
    )


def test_om_elem_affd_guard():
    Tq, K, d = 4, 7, 3
    s, a, b, h = (
        _v("s", Tq, K), _v("a", K, d), _v("b", K, d), _v("h", d),
    )
    bound = {"s": s, "a": a, "b": b, "h": h}
    assert XC.XC_OM_ELEM_AFFD.check(bound)
    # non-concrete → line 676
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "s": RAW})
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "a": _v("a", K, None)})
    # arity/equality conditions → line 678
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "s": _v("s", K)})
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "a": _v("a", d)})
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "b": _v("b", K, d + 1)})
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "h": _v("h", d, d)})
    # contraction mismatches → lines 679-680
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "s": _v("s", Tq, K + 1)})
    assert not XC.XC_OM_ELEM_AFFD.check({**bound, "h": _v("h", d + 1)})
    # batch dims not broadcastable → line 681
    assert not XC.XC_OM_ELEM_AFFD.check(
        {
            "s": _v("s", 2, Tq, K),
            "a": _v("a", 3, K, d),
            "b": _v("b", 3, K, d),
            "h": h,
        }
    )


def test_om_elem_aff_guard():
    Tq, K, d, i = 4, 7, 3, 2
    s, A, b, h = (
        _v("s", Tq, K), _v("A", K, d, i), _v("b", K, d), _v("h", i),
    )
    bound = {"s": s, "A": A, "b": b, "h": h}
    assert XC.XC_OM_ELEM_AFF.check(bound)
    # non-concrete → line 692
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "A": _ill()})
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "s": _v("s", Tq, None)})
    # inner conditions → line 701
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "s": _v("s", K)})
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "A": _v("A", K, d)})
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "A": _v("A", K + 1, d, i)})
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "b": _v("b", K, d + 1)})
    assert not XC.XC_OM_ELEM_AFF.check({**bound, "h": _v("h", i + 1)})
    # leading broadcast failure → line 702
    assert not XC.XC_OM_ELEM_AFF.check(
        {
            "s": _v("s", 2, Tq, K),
            "A": _v("A", 3, K, d, i),
            "b": _v("b", 3, K, d),
            "h": h,
        }
    )


def test_omd_pair_guard():
    Tq, K1, K2, d = 4, 3, 5, 4
    bound = {
        "s1": _v("s1", Tq, K1), "s2": _v("s2", Tq, K2),
        "a1": _v("a1", K1, d), "b1": _v("b1", K1, d),
        "a2": _v("a2", K2, d), "b2": _v("b2", K2, d),
        "h": _v("h", d),
    }
    assert XC.XC_OMD_PAIR_LIFT.check(bound)
    # one leaf failing the elem contract → line 736
    assert not XC.XC_OMD_PAIR_LIFT.check({**bound, "a2": _v("a2", K2, d + 1)})
    assert not XC.XC_OMD_PAIR_LIFT.check({**bound, "s1": RAW})
    # score blocks disagreeing off the key axis → line 739-741
    assert not XC.XC_OMD_PAIR_LIFT.check({**bound, "s2": _v("s2", Tq + 1, K2)})
    # a maps disagreeing off the key axis → line 740
    assert not XC.XC_OMD_PAIR_LIFT.check({**bound, "a2": _v("a2", K2, d + 1)})


def test_omd_split_guard_and_eval():
    Tq, K1, K2, d = 4, 3, 5, 4
    bound = {
        "s1": _v("s1", Tq, K1), "s2": _v("s2", Tq, K2),
        "a1": _v("a1", K1, d), "a2": _v("a2", K2, d),
        "b1": _v("b1", K1, d), "b2": _v("b2", K2, d),
        "$attr:SD": -1, "$attr:AD": -2, "$attr:BD": -2,
    }
    assert XC.XC_OMD_SPLIT.check(bound)
    # non-concrete member → line 756
    assert not XC.XC_OMD_SPLIT.check({**bound, "s1": RAW})
    # dims not ints → line 758
    assert not XC.XC_OMD_SPLIT.check({**bound, "$attr:SD": "x"})
    assert not XC.XC_OMD_SPLIT.check({**bound, "$attr:AD": None})
    # rank/equality failures → line 760
    assert not XC.XC_OMD_SPLIT.check({**bound, "s2": _v("s2", Tq, K2, 1)})
    assert not XC.XC_OMD_SPLIT.check({**bound, "a2": _v("a2", K2, d, 1)})
    assert not XC.XC_OMD_SPLIT.check({**bound, "b1": _v("b1", K1, d + 1)})
    # wrong axes → lines 761-764
    assert not XC.XC_OMD_SPLIT.check({**bound, "$attr:SD": 0})
    assert not XC.XC_OMD_SPLIT.check({**bound, "$attr:AD": -1})
    assert not XC.XC_OMD_SPLIT.check({**bound, "$attr:BD": 1})
    # scores disagree off the key axis → line 766
    assert not XC.XC_OMD_SPLIT.check({**bound, "s2": _v("s2", Tq + 1, K2)})
    # a maps disagree off the key axis, or s[-1] != a[-2] → line 772
    assert not XC.XC_OMD_SPLIT.check({**bound, "a2": _v("a2", K2, d + 1)})
    assert not XC.XC_OMD_SPLIT.check({**bound, "s1": _v("s1", Tq, K1 + 1)})
    # batch broadcast failure → line 773-775
    bad = dict(bound)
    bad["s1"], bad["s2"] = _v("s1", 2, Tq, K1), _v("s2", 2, Tq, K2)
    bad["a1"], bad["b1"] = _v("a1", 3, K1, d), _v("b1", 3, K1, d)
    bad["a2"], bad["b2"] = _v("a2", 3, K2, d), _v("b2", 3, K2, d)
    assert not XC.XC_OMD_SPLIT.check(bad)
    # fp64 end-to-end through XC_OMD_SPLIT: deferred tree vs serial
    s1, s2 = bound["s1"], bound["s2"]
    a1, a2, b1, b2 = bound["a1"], bound["a2"], bound["b1"], bound["b2"]
    env = {
        s1: _rand((Tq, K1), 110), s2: _rand((Tq, K2), 111),
        a1: _rand((K1, d), 112), a2: _rand((K2, d), 113),
        b1: _rand((K1, d), 114), b2: _rand((K2, d), 115),
    }
    t0 = Op.make(
        "omd_elem",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", a1, a2, dim=-2),
        Op.make("concat", b1, b2, dim=-2),
    )
    out = _fire(XC.XC_OMD_SPLIT, t0)
    assert out is not None and out.op == "omd_compose"
    assert meta._eval_allclose(_eval(t0, env), _eval(out, env), tol=1e-12)


def test_omd_split_arg1_spelling():
    bound = {
        "s1": _v("s1", 4, 3), "s2": _v("s2", 4, 5),
        "a1": _v("a1", 3, 4), "a2": _v("a2", 5, 4),
        "b1": _v("b1", 3, 4), "b2": _v("b2", 5, 4),
        "$attr:SD": -1, "$attr:AD": -2, "$attr:BD": -2,
    }
    assert XC.XC_OMD_SPLIT_ARG1.check(bound)


def test_matmul_applyd_rows_guard():
    """E contracts the ROW axis: E (P,K), the map's a-part f (K,d),
    h (d,)."""
    P, K, d = 5, 7, 4
    E, f, h = _v("E", P, K), _v("f", K, d), _v("h", d)
    bound = {"E": E, "f": f, "h": h}
    assert XC.XC_MATMUL_APPLYD_ROWS.check(bound)
    # non-concrete → line 790
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "E": RAW})
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "f": _v("f", K, None)})
    # rank failures → line 792
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "E": _v("E", K)})
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "f": _v("f", d)})
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "h": _v("h", d, d)})
    # h not the feature vector → line 794
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "h": _v("h", d + 1)})
    # E[-1] != f[-2] → line 797-798
    assert not XC.XC_MATMUL_APPLYD_ROWS.check({**bound, "E": _v("E", P, K + 1)})
    # batch broadcast failure → line 799
    assert not XC.XC_MATMUL_APPLYD_ROWS.check(
        {"E": _v("E", 2, P, K), "f": _v("f", 3, K, d), "h": h}
    )


# ---------------------------------------------------------------------------
#  View helpers — _resolve_view_shape, _view_dims, value-shape contracts
# ---------------------------------------------------------------------------


def test_resolve_view_shape():
    r = XC._resolve_view_shape
    assert r((4, 6), 24) == (4, 6)
    assert r((4, -1), 24) == (4, 6)
    assert r((-1,), 24) == (24,)
    # not a tuple/list, or empty → line 1265
    assert r("x", 24) is None
    assert r((), 24) is None
    assert r(24, 24) is None
    # dims that are not int/-1/positive → line 1267
    assert r((4, 0), 24) is None
    assert r((4, -2), 24) is None
    assert r((4, "x"), 24) is None
    # more than one -1 → line 1270
    assert r((-1, -1), 24) is None
    # -1 cannot be inferred consistently → line 1277
    assert r((-1, 5), 24) is None
    # numel mismatch without -1 → line 1282
    assert r((4, 5), 24) is None


def test_view_dims():
    bound = {"$attr:D1": 0, "$attr:D2": -1}
    assert XC._view_dims(bound, 3) == (0, 2)
    # non-int dims → line 1291
    assert XC._view_dims({"$attr:D1": "x", "$attr:D2": 0}, 3) is None
    assert XC._view_dims({"$attr:D1": 0, "$attr:D2": None}, 3) is None


def test_apply_value_shapes():
    A, b, h = _v("A", 5, 6, 4), _v("b", 5, 6), _v("h", 4)
    bound = {"A": A, "b": b, "h": h}
    assert XC._apply_value_shapes(bound) == (5, 6, 4)
    # non-concrete members → line 1301
    assert XC._apply_value_shapes({**bound, "A": RAW}) is None
    assert XC._apply_value_shapes({**bound, "b": _v("b", 5, None)}) is None
    # contract failures: b != A[:-1], bad h → lines 1302-1305
    assert XC._apply_value_shapes({**bound, "b": _v("b", 5, 7)}) is None
    assert XC._apply_value_shapes({**bound, "h": _v("h", 5)}) is None
    assert XC._apply_value_shapes({**bound, "h": _v("h", 4, 4)}) is None
    assert XC._apply_value_shapes({**bound, "A": _v("A", 6)}) is None


def test_applyd_value_shapes():
    a, b, h = _v("a", 5, 4), _v("b", 5, 4), _v("h", 4)
    bound = {"a": a, "b": b, "h": h}
    assert XC._applyd_value_shapes(bound) == (5, 4)
    # non-concrete → line 1314
    assert XC._applyd_value_shapes({**bound, "a": RAW}) is None
    # a!=b / bad h → line 1316
    assert XC._applyd_value_shapes({**bound, "b": _v("b", 5, 5)}) is None
    assert XC._applyd_value_shapes({**bound, "h": _v("h", 5)}) is None
    assert XC._applyd_value_shapes({**bound, "h": _v("h", 4, 4)}) is None


def test_reshape_apply_guard_and_derive():
    """reshape(apply(aff(A,b),h), S): S must resolve against the value
    numel; the map gets S+(i,) — the MHA head-split."""
    T, D, i = 4, 12, 3
    A, b, h = _v("A", T, D, i), _v("b", T, D), _v("h", i)
    bound = {"A": A, "b": b, "h": h, "$attr:S": (T, 3, 4)}
    assert XC.XC_RESHAPE_APPLY.check(bound)
    assert XC.XC_RESHAPE_APPLY.derive(bound) == {
        "$attr:SA": (T, 3, 4, i)
    }
    # a -1 in S is carried through verbatim into SA
    b2 = {**bound, "$attr:S": (T, -1, 4)}
    assert XC.XC_RESHAPE_APPLY.check(b2)
    assert XC.XC_RESHAPE_APPLY.derive(b2) == {"$attr:SA": (T, -1, 4, i)}
    env = {
        A: _rand((T, D, i), 120), b: _rand((T, D), 121),
        h: _rand((i,), 122),
    }
    t0 = Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", A, b), h),
        shape=(T, 3, 4),
    )
    _law(XC.XC_RESHAPE_APPLY, t0, env)
    # A contract fails → check False (line 1323) & derive None (1337)
    assert not XC.XC_RESHAPE_APPLY.check({**bound, "A": RAW})
    assert XC.XC_RESHAPE_APPLY.derive({**bound, "A": RAW}) is None
    # ill-formed S → check False; derive None via unresolvable S (1337)
    assert not XC.XC_RESHAPE_APPLY.check({**bound, "$attr:S": (T, 5)})
    assert (
        XC.XC_RESHAPE_APPLY.derive({**bound, "$attr:S": (T, 5)}) is None
    )


def test_transpose_apply_guard_and_derive():
    """transpose(apply(aff(A,b),h), d1, d2): dims normalised mod the
    VALUE rank; the map's last (input) axis is never permuted — the
    MHA (T,nh,hd)->(nh,T,hd) swap."""
    T, nh, hd, i = 4, 3, 2, 5
    A, b, h = (
        _v("A", T, nh, hd, i), _v("b", T, nh, hd), _v("h", i),
    )
    bound = {"A": A, "b": b, "h": h, "$attr:D1": 0, "$attr:D2": 1}
    assert XC.XC_TRANSPOSE_APPLY.check(bound)
    assert XC.XC_TRANSPOSE_APPLY.derive(bound) == {
        "$attr:DA1": 0, "$attr:DA2": 1
    }
    env = {
        A: _rand((T, nh, hd, i), 130), b: _rand((T, nh, hd), 131),
        h: _rand((i,), 132),
    }
    t0 = Op.make(
        "transpose",
        Op.make("apply", Op.make("aff", A, b), h),
        arg1=0, arg2=1,
    )
    _law(XC.XC_TRANSPOSE_APPLY, t0, env)
    # negative dims normalise mod value rank
    b2 = {**bound, "$attr:D1": -3, "$attr:D2": -1}
    assert XC.XC_TRANSPOSE_APPLY.derive(b2) == {
        "$attr:DA1": 0, "$attr:DA2": 2
    }
    # contract failure → line 1344
    assert not XC.XC_TRANSPOSE_APPLY.check({**bound, "A": _v("A", i)})
    assert not XC.XC_TRANSPOSE_APPLY.check({**bound, "b": _v("b", T, nh, hd + 1)})
    # derive: A contract fail → line 1354; non-int dims → line 1357
    assert XC.XC_TRANSPOSE_APPLY.derive({**bound, "A": RAW}) is None
    assert (
        XC.XC_TRANSPOSE_APPLY.derive({**bound, "$attr:D1": "x"}) is None
    )
    # a non-int dim also vetoes the check (dd None → False)
    assert not XC.XC_TRANSPOSE_APPLY.check({**bound, "$attr:D2": "x"})


def test_reshape_applyd_guard():
    """Diagonal reshape keeps the FEATURE axis last: head-PACKING
    views ((T,d)->(T,1,d)) pass, head-SPLITTING (d -> nh*hd) veto."""
    T, d = 4, 6
    a, b, h = _v("a", T, d), _v("b", T, d), _v("h", d)
    bound = {"a": a, "b": b, "h": h}
    # packing view keeps last dim == d
    assert XC.XC_RESHAPE_APPLYD.check({**bound, "$attr:S": (T, 1, d)})
    # splitting d across axes vetoed
    assert not XC.XC_RESHAPE_APPLYD.check({**bound, "$attr:S": (T, 2, 3)})
    # numel-mismatched S vetoed
    assert not XC.XC_RESHAPE_APPLYD.check({**bound, "$attr:S": (T, 7)})
    # contract fail
    assert not XC.XC_RESHAPE_APPLYD.check({**bound, "a": RAW, "$attr:S": (T, 1, d)})
    env = {
        a: _rand((T, d), 140), b: _rand((T, d), 141), h: _rand((d,), 142),
    }
    t0 = Op.make(
        "reshape",
        Op.make("applyd", Op.make("aff_diag", a, b), h),
        shape=(T, 1, d),
    )
    _law(XC.XC_RESHAPE_APPLYD, t0, env)


def test_transpose_applyd_guard_and_derive():
    """Only permutations leaving the feature axis LAST are sound."""
    T, nh, d = 4, 3, 5
    a, b, h = _v("a", T, nh, d), _v("b", T, nh, d), _v("h", d)
    bound = {"a": a, "b": b, "h": h, "$attr:D1": 0, "$attr:D2": 1}
    assert XC.XC_TRANSPOSE_APPLYD.check(bound)
    assert XC.XC_TRANSPOSE_APPLYD.derive(bound) == {
        "$attr:DA1": 0, "$attr:DA2": 1
    }
    env = {
        a: _rand((T, nh, d), 150), b: _rand((T, nh, d), 151),
        h: _rand((d,), 152),
    }
    t0 = Op.make(
        "transpose",
        Op.make("applyd", Op.make("aff_diag", a, b), h),
        arg1=0, arg2=1,
    )
    _law(XC.XC_TRANSPOSE_APPLYD, t0, env)
    # contract fail / rank < 2 → line 1368
    assert not XC.XC_TRANSPOSE_APPLYD.check({**bound, "a": RAW})
    assert not XC.XC_TRANSPOSE_APPLYD.check({**bound, "a": _v("a", d), "b": _v("b", d)})
    # feature-axis permutations → line 1376 tail (dd fine, axis last)
    assert not XC.XC_TRANSPOSE_APPLYD.check({**bound, "$attr:D2": -1})
    assert not XC.XC_TRANSPOSE_APPLYD.check({**bound, "$attr:D1": 2})
    # non-int dims → dd None → False (line 1376 first clause)
    assert not XC.XC_TRANSPOSE_APPLYD.check({**bound, "$attr:D1": "x"})
    # derive: contract fail → 1386; dd None → 1389
    assert XC.XC_TRANSPOSE_APPLYD.derive({**bound, "a": RAW}) is None
    assert (
        XC.XC_TRANSPOSE_APPLYD.derive({**bound, "$attr:D2": "x"}) is None
    )


def test_transpose_apply_rev_dims():
    """The pushed-through form re-fuses only when A's and b's
    transposes agree on the same value axes and never touch the
    input axis."""
    o1, o2, i = 6, 5, 4
    A, b, h = _v("A", o1, o2, i), _v("b", o1, o2), _v("h", i)
    bound = {
        "A": A, "b": b, "h": h,
        "$attr:P1": 0, "$attr:P2": 1, "$attr:Q1": 0, "$attr:Q2": -1,
    }
    # 0/-1 normalise to the same value axes → accepted pair
    assert XC.XC_TRANSPOSE_APPLY_REV.check(bound)
    assert XC.XC_TRANSPOSE_APPLY_REV.derive(bound) is not None
    # A contract fails → line 1512
    assert not XC.XC_TRANSPOSE_APPLY_REV.check({**bound, "A": RAW})
    # dims not ints → line 1517
    assert not XC.XC_TRANSPOSE_APPLY_REV.check({**bound, "$attr:P1": "x"})
    assert not XC.XC_TRANSPOSE_APPLY_REV.check({**bound, "$attr:Q2": None})
    # permuting the map's INPUT axis → lines 1519-1520
    assert not XC.XC_TRANSPOSE_APPLY_REV.check({**bound, "$attr:P1": 1, "$attr:P2": -1})
    # A and b permuting different axes → line 1523
    assert not XC.XC_TRANSPOSE_APPLY_REV.check({**bound, "$attr:Q1": 1})
    # derive veto → line 1534
    assert (
        XC.XC_TRANSPOSE_APPLY_REV.derive({**bound, "A": RAW}) is None
    )
    env = {
        A: _rand((o1, o2, i), 160), b: _rand((o1, o2), 161),
        h: _rand((i,), 162),
    }
    t0 = Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("transpose", A, arg1=0, arg2=1),
            Op.make("transpose", b, arg1=0, arg2=1),
        ),
        h,
    )
    _law(XC.XC_TRANSPOSE_APPLY_REV, t0, env)


def test_transpose_applyd_rev_dims():
    T, nh, d = 4, 3, 5
    a, b, h = _v("a", T, nh, d), _v("b", T, nh, d), _v("h", d)
    bound = {
        "a": a, "b": b, "h": h,
        "$attr:P1": 0, "$attr:P2": 1, "$attr:Q1": 0, "$attr:Q2": 1,
    }
    assert XC.XC_TRANSPOSE_APPLYD_REV.check(bound)
    assert XC.XC_TRANSPOSE_APPLYD_REV.derive(bound) is not None
    # contract fail / rank < 2 → line 1565
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check({**bound, "a": RAW})
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check(
        {**bound, "a": _v("a", d), "b": _v("b", d)}
    )
    # dims not ints → line 1570
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check({**bound, "$attr:P1": "x"})
    # a/b permuting different axes → line 1573
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check({**bound, "$attr:Q2": 0})
    # feature axis moved → line 1575 (the stored dim pair AGREES
    # across a and b, so the veto is the feature-axis check itself)
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check(
        {**bound, "$attr:P2": 2, "$attr:Q2": 2}
    )
    assert not XC.XC_TRANSPOSE_APPLYD_REV.check(
        {**bound, "$attr:P1": -1, "$attr:Q1": -1}
    )
    # derive veto → line 1586
    assert (
        XC.XC_TRANSPOSE_APPLYD_REV.derive({**bound, "$attr:P1": "x"})
        is None
    )


def test_reshape_apply_rev_guard():
    """apply(aff(reshape(A,SA), reshape(b,SB)),h) refolds only when
    SA == SB+(i,) literally."""
    T, D, i = 4, 12, 3
    A, b, h = _v("A", T, D, i), _v("b", T, D), _v("h", i)
    bound = {
        "A": A, "b": b, "h": h,
        "$attr:SA": (T, 3, 4, i), "$attr:SB": (T, 3, 4),
    }
    assert XC.XC_RESHAPE_APPLY_REV.check(bound)
    # contract fail → line 1623
    assert not XC.XC_RESHAPE_APPLY_REV.check({**bound, "A": RAW})
    # non-tuple attrs → line 1628
    assert not XC.XC_RESHAPE_APPLY_REV.check({**bound, "$attr:SA": "x"})
    assert not XC.XC_RESHAPE_APPLY_REV.check({**bound, "$attr:SB": 7})
    # SA not literally SB+(i,) → line 1630
    assert not XC.XC_RESHAPE_APPLY_REV.check(
        {**bound, "$attr:SA": (T, 3, 4)}
    )
    assert not XC.XC_RESHAPE_APPLY_REV.check(
        {**bound, "$attr:SA": (T, 4, 3, i)}
    )
    # the tucked-on axis must be the map's input dim → line 1632
    assert not XC.XC_RESHAPE_APPLY_REV.check(
        {**bound, "$attr:SA": (T, 3, 4, i + 1)}
    )
    # ill-formed SB → line 1633
    assert not XC.XC_RESHAPE_APPLY_REV.check(
        {**bound, "$attr:SB": (T, 5)}
    )
    env = {
        A: _rand((T, D, i), 170), b: _rand((T, D), 171),
        h: _rand((i,), 172),
    }
    t0 = Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("reshape", A, shape=(T, 3, 4, i)),
            Op.make("reshape", b, shape=(T, 3, 4)),
        ),
        h,
    )
    _law(XC.XC_RESHAPE_APPLY_REV, t0, env)


def test_reshape_applyd_rev_guard():
    T, d = 4, 6
    a, b, h = _v("a", T, d), _v("b", T, d), _v("h", d)
    bound = {
        "a": a, "b": b, "h": h,
        "$attr:SA": (T, 1, d), "$attr:SB": (T, 1, d),
    }
    assert XC.XC_RESHAPE_APPLYD_REV.check(bound)
    # contract fail → line 1664
    assert not XC.XC_RESHAPE_APPLYD_REV.check({**bound, "a": RAW})
    # non-tuple or mismatched attrs → line 1671
    assert not XC.XC_RESHAPE_APPLYD_REV.check({**bound, "$attr:SA": "x"})
    assert not XC.XC_RESHAPE_APPLYD_REV.check(
        {**bound, "$attr:SB": (T, d, 1)}
    )
    # resolved S must keep the feature axis last → line 1673
    assert not XC.XC_RESHAPE_APPLYD_REV.check(
        {**bound, "$attr:SA": (T, 2, 3), "$attr:SB": (T, 2, 3)}
    )


# ---------------------------------------------------------------------------
#  Non-local passes — the veto paths inside _gather_stack and
#  omd_tree_lift / _omd_convert / _elem_affine_options
# ---------------------------------------------------------------------------


def _affd_leaf(a, b, h):
    return Op.make("applyd", Op.make("aff_diag", a, b), h)


def test_map_out_shape():
    # diag: broadcast(map, h)
    assert XC._map_out_shape((5, 4), (4,), "diag") == (5, 4)
    # dense: square last two → fs[:-1]
    assert XC._map_out_shape((5, 4, 4), (4,), "dense") == (5, 4)
    # dense non-square or rank < 2 → line 1981
    assert XC._map_out_shape((5, 4, 3), (3,), "dense") is None
    assert XC._map_out_shape((4,), (4,), "dense") is None
    assert XC._map_out_shape("raw", (4,), "dense") is None


def test_gather_applyd_stack_veto_rank_of_h():
    """h must be a (d,) vector — a rank-2 or shapeless state class
    skips the offer entirely (line 2031)."""
    eg = EGraph()
    d = 4
    h = _v("h", 2, 2)  # rank-2 state: cannot broadcast onto features
    t0 = Op.make(
        "stack",
        *(_affd_leaf(_v(f"a{i}", d), _v(f"b{i}", d), h) for i in range(3)),
        dim=0,
    )
    eg.add_term(t0)
    assert XC.gather_applyd_stack(eg) == []


def test_gather_applyd_stack_veto_feature_axis():
    """stacking onto the feature axis (dim=-1 of a (T,d) value) is
    vetoed — h could not broadcast (line 2063)."""
    eg = EGraph()
    T, d = 5, 4
    h = _v("h", d)
    t0 = Op.make(
        "stack",
        *(
            _affd_leaf(_v(f"a{i}", T, d), _v(f"b{i}", T, d), h)
            for i in range(3)
        ),
        dim=-1,
    )
    eg.add_term(t0)
    assert XC.gather_applyd_stack(eg) == []
    # a positive out-of-range axis is vetoed the same way
    t1 = Op.make(
        "stack",
        *(
            _affd_leaf(_v(f"a{i}", T, d), _v(f"b{i}", T, d), h)
            for i in range(3)
        ),
        dim=2,
    )
    eg = EGraph()
    eg.add_term(t1)
    assert XC.gather_applyd_stack(eg) == []


def test_gather_apply_stack_dense_vetoes():
    i, o = 4, 5
    h = _v("h", i)
    # non-square map: value shape underivable → lines 1981/2043
    eg = EGraph()
    t0 = Op.make(
        "stack",
        *(
            Op.make("apply", Op.make("aff", _v(f"A{k}", 3, i), _v(f"c{k}", 3)), h)
            for k in range(3)
        ),
        dim=0,
    )
    eg.add_term(t0)
    assert XC.gather_apply_stack(eg) == []
    # map's input dim != h's dim → fs != vs+(i,) → line 2052
    eg = EGraph()
    t0 = Op.make(
        "stack",
        *(
            Op.make("apply", Op.make("aff", _v(f"A{k}", i, i), _v(f"c{k}", i)), _v("h", 7))
            for k in range(3)
        ),
        dim=0,
    )
    eg.add_term(t0)
    assert XC.gather_apply_stack(eg) == []
    # square map over a DIFFERENT dim than h → the fs != vs+(hs,)
    # alignment check vetoes first (line 2052 — its sibling continue
    # at 2060 is unreachable: both kinds force S[-1] == hs[0];
    # defensive, suggest ``# pragma: no cover``)
    eg = EGraph()
    t0 = Op.make(
        "stack",
        *(
            Op.make("apply", Op.make("aff", _v(f"A{k}", o, o), _v(f"c{k}", o)), h)
            for k in range(3)
        ),
        dim=0,
    )
    eg.add_term(t0)
    assert XC.gather_apply_stack(eg) == []


def test_gather_applyd_stack_duplicate_value_shapes():
    """Two aff members of the same map class giving the SAME value
    shape — the duplicate is skipped (``vs in m``, line 2043) and the
    offer still lands on the deduplicated map."""
    eg = EGraph(track_proofs=True)
    d = 4
    h = _v("h", d)
    kids = []
    for k in range(3):
        f1 = eg.add_term(Op.make("aff_diag", _v(f"a{k}", d), _v(f"b{k}", d)))
        f2 = eg.add_term(Op.make("aff_diag", _v(f"a{k}x", d), _v(f"b{k}x", d)))
        eg.union(f1, f2)
        kids.append(eg.add_enode("applyd", (f1, eg.add_term(h))))
    eg.add_enode("stack", tuple(kids), {"dim": 0})
    offers = XC.gather_applyd_stack(eg)
    assert len(offers) == 1
    assert offers[0]["term"].op == "applyd"


def test_elem_affine_options_skips_non_pack_members():
    """A map class holding non-aff members (leaves, other ops): the
    scan skips them (the 2176->2175 loop branch) and still finds the
    aff member."""
    eg = EGraph()
    d = 4
    h = _v("h", d)
    s = _v("s", 4, 3)
    a, b = _v("a", 3, d), _v("b", 3, d)
    f = eg.add_term(Op.make("aff_diag", a, b))
    g = eg.add_term(_v("g", 3, d))          # leaf member in map class
    eg.union(f, g)
    m = eg.add_term(Op.make("mul", a, b))   # non-pack op member
    eg.union(f, m)
    v = eg.add_enode("applyd", (f, eg.add_term(h)))
    opts = XC._elem_affine_options(eg, eg.find(v))
    assert len(opts) == 1
    h_eid = next(iter(opts))
    assert h_eid == eg.find(eg.add_term(h))
    kind, a_eid, b_eid = opts[h_eid][0]
    assert kind == "diag"
    assert a_eid == eg.find(eg.add_term(a))
    assert b_eid == eg.find(eg.add_term(b))
    # the whole-tree lift still fires through the noisy map class
    e = eg.add_enode("om_elem", (eg.add_term(s), v))
    eg.add_enode("om_apply", (e,))
    offers = XC.omd_tree_lift(eg)
    assert len(offers) == 1 and offers[0]["kind"] == "diag"


def test_omd_convert_shared_subtree_memo():
    """om_compose(C, C) — the shared subtree is converted once; the
    second visit hits the memo (line 2204) and the lift completes."""
    eg = EGraph(track_proofs=True)
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _v("s1", Tq, K1), _v("s2", Tq, K2)
    a1, b1 = _v("a1", K1, d), _v("b1", K1, d)
    a2, b2 = _v("a2", K2, d), _v("b2", K2, d)
    h = _v("h", d)
    e1 = eg.add_term(Op.make("om_elem", s1, _affd_leaf(a1, b1, h)))
    e2 = eg.add_term(Op.make("om_elem", s2, _affd_leaf(a2, b2, h)))
    c1 = eg.add_enode("om_compose", (e1, e2))
    c2 = eg.add_enode("om_compose", (c1, c1))
    eg.add_enode("om_apply", (c2,))
    offers = XC.omd_tree_lift(eg)
    assert len(offers) == 1 and offers[0]["kind"] == "diag"
    env = {
        s1: _rand((Tq, K1), 180), s2: _rand((Tq, K2), 181),
        a1: _rand((K1, d), 182), b1: _rand((K1, d), 183),
        a2: _rand((K2, d), 184), b2: _rand((K2, d), 185),
        h: _rand((d,), 186),
    }
    # the lifted deferred tree evaluates to chunked attention —
    # compose(x,x) duplicates block x, so the serial reference is the
    # two-block cat of (s1,s2) with itself
    got = _eval(offers[0]["term"], env)
    s_cat = torch.cat([env[s1], env[s2]], dim=-1)
    v_cat = torch.cat(
        [env[a1] * env[h] + env[b1], env[a2] * env[h] + env[b2]], dim=-2
    )
    # serial reference: om_compose(T,T) of the merged block =
    # softmax over duplicated keys
    e = torch.exp(s_cat - s_cat.amax(dim=-1, keepdim=True))
    ref = (e @ v_cat) / e.sum(dim=-1, keepdim=True)
    # omd_apply of the converted tree applies h at the end — the
    # deferred numerator is (e@a)⊙h + e@b on the SAME concatenated
    # blocks
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_omd_tree_lift_cyclic_carrier():
    """A carrier class that contains an om_compose over ITSELF (a
    self-loop formed by union): the recursive collect/convert honour
    their seen/stack guards (lines 2269, 2206) instead of diverging."""
    eg = EGraph(track_proofs=True)
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _v("s1", Tq, K1), _v("s2", Tq, K2)
    a1, b1 = _v("a1", K1, d), _v("b1", K1, d)
    a2, b2 = _v("a2", K2, d), _v("b2", K2, d)
    h = _v("h", d)
    e1 = eg.add_term(Op.make("om_elem", s1, _affd_leaf(a1, b1, h)))
    e2 = eg.add_term(Op.make("om_elem", s2, _affd_leaf(a2, b2, h)))
    comp = eg.add_enode("om_compose", (e1, e2))
    eg.union(comp, e1)  # class now: {om_elem, om_compose(M, e2)}
    M = eg.find(e1)
    eg.add_enode("om_apply", (M,))
    offers = XC.omd_tree_lift(eg)
    # the acyclic om_elem member still converts; the cyclic compose
    # member is skipped by the stack guard
    assert len(offers) == 1 and offers[0]["kind"] == "diag"
    assert offers[0]["term"].op == "omd_apply"


def test_gather_stack_child_without_apply_member():
    """A stack child class with no applyd member at all: the per-child
    scan skips its non-apply nodes (the 2018->2017 loop branch) and
    the empty option-set intersection yields no offer."""
    eg = EGraph()
    d = 4
    h = _v("h", d)
    t0 = Op.make(
        "stack",
        _affd_leaf(_v("a0", d), _v("b0", d), h),
        _v("plain", d),  # a leaf child — no apply member
        _affd_leaf(_v("a2", d), _v("b2", d), h),
        dim=0,
    )
    eg.add_term(t0)
    assert XC.gather_applyd_stack(eg) == []


def test_elem_affine_options_dense_member():
    """The apply/aff (dense) arm of _elem_affine_options — lines
    2170-2173 — collects dense options under their h."""
    eg = EGraph()
    o, i = 5, 4
    h = _v("h", i)
    A, c = _v("A", o, i), _v("c", o)
    f = eg.add_term(Op.make("aff", A, c))
    v = eg.add_enode("apply", (f, eg.add_term(h)))
    opts = XC._elem_affine_options(eg, eg.find(v))
    assert len(opts) == 1
    h_eid = next(iter(opts))
    kind, a_eid, b_eid = opts[h_eid][0]
    assert kind == "dense"
    assert a_eid == eg.find(eg.add_term(A))
    assert b_eid == eg.find(eg.add_term(c))


def test_omd_tree_lift_partial_affine_carrier_vetoes():
    """A compose tree where the leaves are affine under DIFFERENT h's
    (or not affine at all): every candidate h fails the whole-tree
    conversion — exercising the no-opts / wrong-kind / partial-compose
    continue branches (lines 2215/2217/2223->2209, 2209->2233,
    2283->2282, 2291)."""
    Tq, K, d = 4, 3, 4
    s1, s2, s3 = _v("s1", Tq, K), _v("s2", Tq, K), _v("s3", Tq, K)
    a1, b1 = _v("a1", K, d), _v("b1", K, d)
    a2, b2 = _v("a2", K, d), _v("b2", K, d)
    h1, h2 = _v("h1", d), _v("h2", d)
    v = _v("v", K, d)
    eg = EGraph()
    e1 = eg.add_term(Op.make("om_elem", s1, _affd_leaf(a1, b1, h1)))
    e2 = eg.add_term(Op.make("om_elem", s2, _affd_leaf(a2, b2, h2)))
    e3 = eg.add_term(Op.make("om_elem", s3, v))  # non-affine leaf
    c1 = eg.add_enode("om_compose", (e1, e2))
    c2 = eg.add_enode("om_compose", (c1, e3))
    eg.add_enode("om_apply", (c2,))
    assert XC.omd_tree_lift(eg) == []


def test_collect_skips_non_om_members_in_carrier_cone():
    """A non-om enode inside the carrier cone (a stray leaf member in
    a constituent class) is skipped by _collect — line 2276->2271 —
    while the real om tree still lifts."""
    eg = EGraph(track_proofs=True)
    Tq, K, d = 4, 3, 4
    s1, s2 = _v("s1", Tq, K), _v("s2", Tq, K)
    a1, b1 = _v("a1", K, d), _v("b1", K, d)
    a2, b2 = _v("a2", K, d), _v("b2", K, d)
    h = _v("h", d)
    e1 = eg.add_term(Op.make("om_elem", s1, _affd_leaf(a1, b1, h)))
    e2 = eg.add_term(Op.make("om_elem", s2, _affd_leaf(a2, b2, h)))
    # give the carrier class an unrelated member — a bare leaf —
    # that _collect must skip over
    stray = eg.add_term(_v("stray", Tq, K))
    c1 = eg.add_enode("om_compose", (e1, e2))
    eg.union(c1, stray)
    eg.add_enode("om_apply", (c1,))
    offers = XC.omd_tree_lift(eg)
    assert len(offers) == 1 and offers[0]["kind"] == "diag"


#: NOTE on unreachable lines in the passes: ``_offer_witness``'s
#: ``src is None`` return (line 1955) requires an e-class whose EVERY
#: member resolves cyclically through ``_oldest_term`` — but every
#: class is born with at least one acyclic enode and unions only ever
#: ADD members, so it cannot arise.  Likewise ``ec is None`` at lines
#: 2008/2254 — the id comes from ``_classes`` itself, so its canonical
#: find() is always present.  And the ``S[-1] != hs[0]`` continue at
#: line 2060 is dead on both kinds: diag requires fs == broadcast(fs,
#: hs) hence fs[-1] == hs[0]; dense requires fs == vs+(hs[0],) with
#: fs[-1] == fs[-2] (from _map_out_shape), hence S[-1] == fs[-2] ==
#: hs[0].  All defensive — suggest ``# pragma: no cover``.
