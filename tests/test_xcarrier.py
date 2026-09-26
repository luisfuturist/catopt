# ruff: noqa: RUF002, RUF003
"""Cross-carrier laws — the scan↔softmax seam, verified on fp64.

This test file is the concrete companion to ``catopt/xcarrier.py``.
It answers, on real tensors:

WHAT CROSSES (positive, exact):
  * A linear readout exits either carrier — the scan analogue of the
    traced category's tightening axiom:

        matmul(E, applyd(aff_diag(a,b),h)) = applyd(aff_diag(Ea,Eb),h)
        matmul(W, apply(aff(A,c),h))     = apply(aff(WA,Wc),h)
        linear(applyd(aff_diag(a,b),h),W)
            = apply(aff(unsqueeze(a,-2)⊙W, linear(b,W)),h)   (promotion)

  * The om numerator IS such a readout — ``om_elem(s, a⊙h+b)`` is one
    fused element ``om_elem_affd(s,a,b,h)`` evaluating to the same
    (m,l,A) triple; it composes under the ordinary om homomorphism.

  * Attention over scanned values can be lifted whole into the
    deferred ``omd`` carrier — a single recurrence whose step state is
    (m, l, affine numerator); ``omd_apply(f,h)`` materialises h at the
    end.  Elem-level: ``XC_OMD_LIFT``/``XC_OMD_PAIR_LIFT``; tree-level:
    the non-local ``omd_tree_lift`` pass (the shared-h condition is
    global — no lhs→rhs rule can see it).

  * The sequence a scan emits is ONE application:
    ``stack(applyd(f_i,h)) = applyd(aff_diag(stack a_i, stack b_i),h)``
    — the non-local ``gather_applyd_stack``/``gather_apply_stack``
    passes (stack is variadic; no lhs→rhs rule can be variadic).

  * Carrier-preserving value arithmetic: scalar scale, same-state add,
    chunk/split on non-feature axes.

WHAT DOES NOT CROSS (negative, documented):
  * Scores: q,k affine in h ⇒ s = q·k is QUADRATIC in h — no affine
    carrier captures it and exp∘quadratic has no finite carrier.  The
    quadratic residue is measured concretely.
  * Softmax attention ≠ linear attention — the linear-attention
    reassociation (QKᵀ)V = Q(KᵀV) is exact but is NOT softmax; the gap
    is measured, not rewritten away.
  * Guards veto the unsound instances: feature-axis contraction in the
    row law, feature-axis chunk, non-shared-h trees, concrete-valued
    leaves inside an omd lift.

Everything is float64; carrier law tolerances sit at 1e-10..1e-14 —
reassociation error only.
"""

from __future__ import annotations

import torch

import catopt.xcarrier as XC
from catopt import meta
from catopt.egraph import EGraph
from catopt.ir import Op, TensorType, Var
from catopt.torch_bridge import _IR_TO_TORCH

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _V(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _rand(shape, seed_shift: int = 0):
    g = torch.Generator().manual_seed(1234 + seed_shift)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _law(rule, t0, env, tol=1e-10):
    """Fire *rule* at the root of *t0* and fp64-check both sides."""
    applied = meta.apply_rewrite_at(rule, t0, ())
    assert applied is not None, f"{rule.name} did not fire"
    a = meta._eval_term(t0, env)
    b = meta._eval_term(applied, env)
    assert meta._eval_allclose(a, b, tol=tol), (
        f"{rule.name}: fp64 mismatch {_max_diff(a, b):.3e}"
    )
    return applied


def _max_diff(a, b):
    if isinstance(a, tuple):
        return max(_max_diff(x, y) for x, y in zip(a, b, strict=True))
    return (a - b).abs().max().item()


def _no_fire(rule, t0):
    assert meta.apply_rewrite_at(rule, t0, ()) is None


def _class_ops(eg: EGraph, eid: int) -> set:
    return {n.op for n in eg.get_class(eid).nodes}


# ---------------------------------------------------------------------------
#  A. Readout / tightening laws — term-level fp64 verification
# ---------------------------------------------------------------------------


def test_matmul_applyd_rows():
    """E contracts the ROW axis of a (K,d) diagonal-affine block:
    E@(a⊙h+b) = (E@a)⊙h + E@b — stays in the diagonal carrier."""
    P, K, d = 5, 7, 4
    E, a, b, h = (
        _V("E", (P, K)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    env = {
        E: _rand((P, K), 1),
        a: _rand((K, d), 2),
        b: _rand((K, d), 3),
        h: _rand((d,), 4),
    }
    f = Op.make("aff_diag", a, b)
    t0 = Op.make("matmul", E, Op.make("applyd", f, h))
    _law(XC.XC_MATMUL_APPLYD_ROWS, t0, env)
    # reverse: the pulled-through form re-fuses
    t1 = Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("matmul", E, Op.make("affd_a", f)),
            Op.make("matmul", E, Op.make("affd_b", f)),
        ),
        h,
    )
    _law(XC.XC_MATMUL_APPLYD_ROWS_REV, t1, env)


def test_matmul_applyd_rows_veto_feature_axis():
    """The row law must NOT fire when E contracts the FEATURE axis
    (E[-1] == d, not K): that case is the *promotion* law instead."""
    P, K, d = 5, 7, 4
    E, a, b, h = (
        _V("E", (P, d)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    t0 = Op.make(
        "matmul", E, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    _no_fire(XC.XC_MATMUL_APPLYD_ROWS, t0)
    # also vetoed: E[-1] matching neither axis
    E2 = _V("E", (P, 3))
    t0 = Op.make(
        "matmul", E2, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    _no_fire(XC.XC_MATMUL_APPLYD_ROWS, t0)


def test_matmul_applyd_vec_promotion():
    """W (o,d) contracting the feature axis promotes the diagonal map
    to a dense one: W(a⊙h+b) = (W·diag a)h + Wb = mul(W,a)@h + W@b."""
    o, d = 6, 4
    W, a, b, h = (
        _V("W", (o, d)),
        _V("a", (d,)),
        _V("b", (d,)),
        _V("h", (d,)),
    )
    env = {
        W: _rand((o, d), 5),
        a: _rand((d,), 6),
        b: _rand((d,), 7),
        h: _rand((d,), 8),
    }
    t0 = Op.make(
        "matmul", W, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    _law(XC.XC_MATMUL_APPLYD_VEC, t0, env)
    # ...and the promoted form de-promotes back
    t1 = Op.make(
        "apply",
        Op.make("aff", Op.make("mul", W, a), Op.make("matmul", W, b)),
        h,
    )
    _law(XC.XC_MATMUL_APPLYD_VEC_REV, t1, env)


def test_matmul_apply_dense():
    """matmul(W, apply(aff(A,c),h)) = apply(aff(WA, Wc), h)."""
    o, i = 6, 4
    W, A, c, h = (
        _V("W", (o, i)),
        _V("A", (i, i)),
        _V("c", (i,)),
        _V("h", (i,)),
    )
    env = {
        W: _rand((o, i), 9),
        A: _rand((i, i), 10),
        c: _rand((i,), 11),
        h: _rand((i,), 12),
    }
    t0 = Op.make("matmul", W, Op.make("apply", Op.make("aff", A, c), h))
    _law(XC.XC_MATMUL_APPLY, t0, env)
    t1 = Op.make(
        "apply",
        Op.make(
            "aff", Op.make("matmul", W, A), Op.make("matmul", W, c)
        ),
        h,
    )
    _law(XC.XC_MATMUL_APPLY_REV, t1, env)


def test_linear_applyd_promotion():
    """linear(applyd(aff_diag(a,b),h), W) — the torch.export spelling of
    the promotion, batched a (T,i) → dense (T,o,i) maps."""
    T, o, i = 8, 6, 4
    W, a, b, h = (
        _V("W", (o, i)),
        _V("a", (T, i)),
        _V("b", (T, i)),
        _V("h", (i,)),
    )
    env = {
        W: _rand((o, i), 13),
        a: _rand((T, i), 14),
        b: _rand((T, i), 15),
        h: _rand((i,), 16),
    }
    t0 = Op.make(
        "linear", Op.make("applyd", Op.make("aff_diag", a, b), h), W
    )
    out = _law(XC.XC_LINEAR_APPLYD, t0, env)
    # the promoted term really is a dense `apply`
    assert out.op == "apply"
    # vector-state form: a,b (i,)
    a1, b1 = _V("a", (i,)), _V("b", (i,))
    env1 = {
        W: env[W],
        a1: _rand((i,), 17),
        b1: _rand((i,), 18),
        h: env[h],
    }
    t0 = Op.make(
        "linear", Op.make("applyd", Op.make("aff_diag", a1, b1), h), W
    )
    _law(XC.XC_LINEAR_APPLYD, t0, env1)


def test_linear_apply_dense():
    """linear(apply(aff(A,c),h), W) = apply(aff(WA, linear(c,W)), h)."""
    T, o, i = 8, 6, 4
    W, A, c, h = (
        _V("W", (o, i)),
        _V("A", (T, i, i)),
        _V("c", (T, i)),
        _V("h", (i,)),
    )
    env = {
        W: _rand((o, i), 19),
        A: _rand((T, i, i), 20),
        c: _rand((T, i), 21),
        h: _rand((i,), 22),
    }
    t0 = Op.make("linear", Op.make("apply", Op.make("aff", A, c), h), W)
    _law(XC.XC_LINEAR_APPLY, t0, env)


def test_scale_and_add_laws():
    """Scalar scale commutes into both carriers; same-state adds merge."""
    d = 4
    a1, b1, a2, b2, h = (
        _V("a1", (d,)),
        _V("b1", (d,)),
        _V("a2", (d,)),
        _V("b2", (d,)),
        _V("h", (d,)),
    )
    A1, c1, A2, c2 = (
        _V("A1", (d, d)),
        _V("c1", (d,)),
        _V("A2", (d, d)),
        _V("c2", (d,)),
    )
    r = _V("r", ())
    env = {
        a1: _rand((d,), 30),
        b1: _rand((d,), 31),
        a2: _rand((d,), 32),
        b2: _rand((d,), 33),
        A1: _rand((d, d), 34),
        c1: _rand((d,), 35),
        A2: _rand((d, d), 36),
        c2: _rand((d,), 37),
        h: _rand((d,), 38),
        r: _rand((), 39),
    }
    f1 = Op.make("aff_diag", a1, b1)
    f2 = Op.make("aff_diag", a2, b2)
    t0 = Op.make("mul", Op.make("applyd", f1, h), r)
    _law(XC.XC_SCALE_APPLYD, t0, env)
    t0 = Op.make("mul", r, Op.make("applyd", f1, h))
    _law(XC.XC_SCALE_APPLYD_PRE, t0, env)
    g1 = Op.make("aff", A1, c1)
    g2 = Op.make("aff", A2, c2)
    t0 = Op.make("mul", Op.make("apply", g1, h), r)
    _law(XC.XC_SCALE_APPLY, t0, env)
    t0 = Op.make(
        "add", Op.make("applyd", f1, h), Op.make("applyd", f2, h)
    )
    _law(XC.XC_ADD_APPLYD, t0, env)
    t0 = Op.make(
        "add", Op.make("apply", g1, h), Op.make("apply", g2, h)
    )
    _law(XC.XC_ADD_APPLY, t0, env)


def test_chunk_split_applyd():
    """chunk/split of a scanned value block = applyd of the sliced map —
    any axis except the feature axis."""
    K, d = 8, 4
    a, b, h = _V("a", (K, d)), _V("b", (K, d)), _V("h", (d,))
    env = {
        a: _rand((K, d), 40),
        b: _rand((K, d), 41),
        h: _rand((d,), 42),
    }
    base = Op.make("applyd", Op.make("aff_diag", a, b), h)
    t0 = Op.make("chunk", base, chunks=2, dim=0, index=1)
    _law(XC.XC_CHUNK_APPLYD, t0, env)
    t0 = Op.make("split", base, sizes=(3, 5), dim=0, index=0)
    _law(XC.XC_SPLIT_APPLYD, t0, env)


def test_chunk_applyd_veto_feature_axis():
    """Chunking the FEATURE axis is unsound — h is not chunked."""
    K, d = 8, 4
    a, b, h = _V("a", (K, d)), _V("b", (K, d)), _V("h", (d,))
    base = Op.make("applyd", Op.make("aff_diag", a, b), h)
    _no_fire(
        XC.XC_CHUNK_APPLYD,
        Op.make("chunk", base, chunks=2, dim=-1, index=0),
    )
    _no_fire(
        XC.XC_CHUNK_APPLYD,
        Op.make("chunk", base, chunks=2, dim=1, index=0),
    )


def test_chunk_apply_dense():
    """chunk(apply(aff(A,c),h)) slices A and c on the same leading
    axis — A's map-input axis is last, never sliced."""
    T, i = 8, 4
    A, c, h = _V("A", (T, i, i)), _V("c", (T, i)), _V("h", (i,))
    env = {
        A: _rand((T, i, i), 50),
        c: _rand((T, i), 51),
        h: _rand((i,), 52),
    }
    base = Op.make("apply", Op.make("aff", A, c), h)
    t0 = Op.make("chunk", base, chunks=2, dim=0, index=1)
    _law(XC.XC_CHUNK_APPLY, t0, env)
    t0 = Op.make("split", base, sizes=(3, 5), dim=0, index=0)
    _law(XC.XC_SPLIT_APPLY, t0, env)


# ---------------------------------------------------------------------------
#  B. The heterogeneous om element — the scan folded inside the om leaf
# ---------------------------------------------------------------------------


def test_om_elem_affd():
    """om_elem(s, a⊙h+b) = (m, l, (e@a)⊙h + e@b) — one fused element,
    fp64-identical to the nested form."""
    Tq, K, d = 5, 7, 4
    s, a, b, h = (
        _V("s", (Tq, K)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    env = {
        s: _rand((Tq, K), 60),
        a: _rand((K, d), 61),
        b: _rand((K, d), 62),
        h: _rand((d,), 63),
    }
    t0 = Op.make(
        "om_elem", s, Op.make("applyd", Op.make("aff_diag", a, b), h)
    )
    _law(XC.XC_OM_ELEM_AFFD, t0, env)
    # unfold direction
    t1 = Op.make("om_elem_affd", s, a, b, h)
    _law(XC.XC_OM_ELEM_AFFD_REV, t1, env)


def test_om_elem_affd_veto_bad_axes():
    """The element law requires s[-1]==a[-2] (the KEY axis) and
    h[-1]==a[-1] (the FEATURE axis) — a wrong pairing must veto."""
    Tq, K, d = 5, 7, 4
    s, a, b, h = (
        _V("s", (Tq, d)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    t0 = Op.make(
        "om_elem",
        s,  # s keys = d ≠ K — wrong
        Op.make("applyd", Op.make("aff_diag", a, b), h),
    )
    _no_fire(XC.XC_OM_ELEM_AFFD, t0)


def test_om_elem_aff_dense():
    """Dense fiber: v = A@h+b with per-key A (K,d,i);
    numerator = reshape(e@A)@h + e@b."""
    Tq, K, d, i = 5, 7, 4, 3
    s, A, b, h = (
        _V("s", (Tq, K)),
        _V("A", (K, d, i)),
        _V("b", (K, d)),
        _V("h", (i,)),
    )
    env = {
        s: _rand((Tq, K), 64),
        A: _rand((K, d, i), 65),
        b: _rand((K, d), 66),
        h: _rand((i,), 67),
    }
    t0 = Op.make(
        "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
    )
    _law(XC.XC_OM_ELEM_AFF, t0, env)
    _law(XC.XC_OM_ELEM_AFF_REV, Op.make("om_elem_aff", s, A, b, h), env)


def test_fused_elem_composes_under_om():
    """The fused element is an ordinary om triple: om_compose of two
    om_elem_affd leaves equals the concrete two-block compose."""
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _V("s1", (Tq, K1)), _V("s2", (Tq, K2))
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    h = _V("h", (d,))
    env = {
        s1: _rand((Tq, K1), 70),
        s2: _rand((Tq, K2), 71),
        a1: _rand((K1, d), 72),
        b1: _rand((K1, d), 73),
        a2: _rand((K2, d), 74),
        b2: _rand((K2, d), 75),
        h: _rand((d,), 76),
    }
    lhs = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make("om_elem_affd", s1, a1, b1, h),
            Op.make("om_elem_affd", s2, a2, b2, h),
        ),
    )
    rhs = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make(
                "om_elem",
                s1,
                Op.make("applyd", Op.make("aff_diag", a1, b1), h),
            ),
            Op.make(
                "om_elem",
                s2,
                Op.make("applyd", Op.make("aff_diag", a2, b2), h),
            ),
        ),
    )
    a = meta._eval_term(lhs, env)
    b = meta._eval_term(rhs, env)
    assert meta._eval_allclose(a, b, tol=1e-12)


# ---------------------------------------------------------------------------
#  C. The deferred omd carrier — attention output stays affine in h
# ---------------------------------------------------------------------------


def test_omd_lift_single():
    """om_apply(om_elem(s, applyd(aff_diag(a,b),h)))
    = omd_apply(omd_elem(s,a,b), h): single-block attention over
    scanned values is ONE diagonal-affine map of the initial state."""
    Tq, K, d = 5, 7, 4
    s, a, b, h = (
        _V("s", (Tq, K)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    env = {
        s: _rand((Tq, K), 80),
        a: _rand((K, d), 81),
        b: _rand((K, d), 82),
        h: _rand((d,), 83),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            s,
            Op.make("applyd", Op.make("aff_diag", a, b), h),
        ),
    )
    _law(XC.XC_OMD_LIFT, t0, env)
    _law(
        XC.XC_OMD_UNLIFT,
        Op.make("omd_apply", Op.make("omd_elem", s, a, b), h),
        env,
    )


def test_omd_lift_dense():
    """Dense-fiber omd: the deferred numerator pair (e@A, e@b) with
    per-key A (K,d,i); omd_applym contracts the last axis."""
    Tq, K, d, i = 5, 7, 4, 3
    s, A, b, h = (
        _V("s", (Tq, K)),
        _V("A", (K, d, i)),
        _V("b", (K, d)),
        _V("h", (i,)),
    )
    env = {
        s: _rand((Tq, K), 84),
        A: _rand((K, d, i), 85),
        b: _rand((K, d), 86),
        h: _rand((i,), 87),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_elem", s, Op.make("apply", Op.make("aff", A, b), h)
        ),
    )
    _law(XC.XC_OMD_LIFT_DENSE, t0, env)
    _law(
        XC.XC_OMD_UNLIFT_DENSE,
        Op.make("omd_applym", Op.make("omd_elem", s, A, b), h),
        env,
    )


def test_omd_pair_lift():
    """Two-block compose under one om_apply lifts to an omd_compose —
    the matcher enforces the shared h."""
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _V("s1", (Tq, K1)), _V("s2", (Tq, K2))
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    h = _V("h", (d,))
    env = {
        s1: _rand((Tq, K1), 90),
        s2: _rand((Tq, K2), 91),
        a1: _rand((K1, d), 92),
        b1: _rand((K1, d), 93),
        a2: _rand((K2, d), 94),
        b2: _rand((K2, d), 95),
        h: _rand((d,), 96),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make(
                "om_elem",
                s1,
                Op.make("applyd", Op.make("aff_diag", a1, b1), h),
            ),
            Op.make(
                "om_elem",
                s2,
                Op.make("applyd", Op.make("aff_diag", a2, b2), h),
            ),
        ),
    )
    _law(XC.XC_OMD_PAIR_LIFT, t0, env)


def test_omd_pair_lift_veto_different_states():
    """Different initial states in the two leaves must NOT lift — the
    whole point of the shared-h metavariable."""
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _V("s1", (Tq, K1)), _V("s2", (Tq, K2))
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    h, h2 = _V("h", (d,)), _V("h2", (d,))
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make(
                "om_elem",
                s1,
                Op.make("applyd", Op.make("aff_diag", a1, b1), h),
            ),
            Op.make(
                "om_elem",
                s2,
                Op.make("applyd", Op.make("aff_diag", a2, b2), h2),
            ),
        ),
    )
    _no_fire(XC.XC_OMD_PAIR_LIFT, t0)


def test_omd_split():
    """omd_elem(cat s, cat a, cat b) = omd_elem ⊕ omd_elem — the om
    homomorphism holds on the deferred carrier (scores cat on the key
    axis, a/b cat on dim -2)."""
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _V("s1", (Tq, K1)), _V("s2", (Tq, K2))
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    env = {
        s1: _rand((Tq, K1), 100),
        s2: _rand((Tq, K2), 101),
        a1: _rand((K1, d), 102),
        b1: _rand((K1, d), 103),
        a2: _rand((K2, d), 104),
        b2: _rand((K2, d), 105),
    }
    t0 = Op.make(
        "omd_elem",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", a1, a2, dim=-2),
        Op.make("concat", b1, b2, dim=-2),
    )
    _law(XC.XC_OMD_SPLIT, t0, env)


def test_omd_split_veto_wrong_axes():
    """Scores concat'd on a non-key axis, or values concat'd on the
    feature axis, must veto."""
    Tq, K1, K2, d = 4, 3, 5, 4
    s1, s2 = _V("s1", (Tq, K1)), _V("s2", (Tq, K2))
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    # scores concat on the QUERY axis — wrong
    t0 = Op.make(
        "omd_elem",
        Op.make("concat", s1, s2, dim=-2),
        Op.make("concat", a1, a2, dim=-2),
        Op.make("concat", b1, b2, dim=-2),
    )
    _no_fire(XC.XC_OMD_SPLIT, t0)
    # values concat on the FEATURE axis — wrong
    t0 = Op.make(
        "omd_elem",
        Op.make("concat", s1, s2, dim=-1),
        Op.make("concat", a1, a2, dim=-1),
        Op.make("concat", b1, b2, dim=-1),
    )
    _no_fire(XC.XC_OMD_SPLIT, t0)


def test_omd_compose_boundary_cases():
    """All-(-inf) score blocks, and the rescaling invariance: the
    deferred carrier survives the degenerate cases like om does."""
    Tq, K, d = 3, 4, 4
    s_neg = torch.full((Tq, K), float("-inf"), dtype=torch.float64)
    a, b = _rand((K, d), 110), _rand((K, d), 111)
    h = _rand((d,), 112)
    f = _IR_TO_TORCH["omd_elem"](s_neg, a, b)
    # fully-masked block: m = -inf and l, fa, fb are NaN — exactly like
    # om_elem's exp(s − −inf); the compose where-guards absorb it.
    assert torch.isinf(f[0]).all() and torch.isnan(f[1]).all()
    out = _IR_TO_TORCH["omd_apply"](f, h)
    assert torch.isnan(out).all()  # same NaN semantics as om_apply
    # -inf block absorbed by a live one
    s_live = _rand((Tq, K), 113)
    g = _IR_TO_TORCH["omd_elem"](s_live, a, b)
    fg = _IR_TO_TORCH["omd_compose"](f, g)
    ref = _IR_TO_TORCH["omd_apply"](g, h)
    got = _IR_TO_TORCH["omd_apply"](fg, h)
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
#  D. Non-local passes — stack gather and whole-tree omd lift
# ---------------------------------------------------------------------------


def _add(eg: EGraph, t):
    return eg.add_term(t)


def test_gather_applyd_stack():
    """stack(applyd(f_i,h)) → applyd(aff_diag(stack a_i, stack b_i),h):
    the emitted sequence is ONE map applied to the initial state."""
    n, d = 5, 4
    a = [_V(f"a{i}", (d,)) for i in range(n)]
    b = [_V(f"b{i}", (d,)) for i in range(n)]
    h = _V("h", (d,))
    env = {
        **{a[i]: _rand((d,), 120 + i) for i in range(n)},
        **{b[i]: _rand((d,), 130 + i) for i in range(n)},
        h: _rand((d,), 140),
    }
    t0 = Op.make(
        "stack",
        *(
            Op.make("applyd", Op.make("aff_diag", a[i], b[i]), h)
            for i in range(n)
        ),
        dim=0,
    )
    eg = EGraph(track_proofs=True)
    root = _add(eg, t0)
    offers = XC.gather_applyd_stack(eg)
    assert len(offers) == 1
    # the offered member is a single applyd of a stacked aff_diag
    offered = offers[0]["term"]
    assert offered.op == "applyd"
    assert offered.args[0].op == "aff_diag"
    assert offered.args[0].args[0].op == "stack"
    # fp64: the union is a real equality
    a_val = meta._eval_term(t0, env)
    b_val = meta._eval_term(offered, env)
    assert meta._eval_allclose(a_val, b_val, tol=1e-12)
    # the merge landed in the stack's class
    assert "applyd" in _class_ops(eg, root)


def test_gather_applyd_stack_veto_mismatched_states():
    """No shared h → no offer."""
    n, d = 3, 4
    a = [_V(f"a{i}", (d,)) for i in range(n)]
    b = [_V(f"b{i}", (d,)) for i in range(n)]
    hs = [_V(f"h{i}", (d,)) for i in range(n)]
    t0 = Op.make(
        "stack",
        *(
            Op.make("applyd", Op.make("aff_diag", a[i], b[i]), hs[i])
            for i in range(n)
        ),
        dim=0,
    )
    eg = EGraph()
    _add(eg, t0)
    assert XC.gather_applyd_stack(eg) == []


def test_gather_apply_stack_dense():
    """Dense analog: stack(apply(aff(A_i,c_i),h)) →
    apply(aff(stack A_i, stack c_i), h)."""
    n, i = 4, 3
    A = [_V(f"A{k}", (i, i)) for k in range(n)]
    c = [_V(f"c{k}", (i,)) for k in range(n)]
    h = _V("h", (i,))
    env = {
        **{A[k]: _rand((i, i), 150 + k) for k in range(n)},
        **{c[k]: _rand((i,), 160 + k) for k in range(n)},
        h: _rand((i,), 170),
    }
    t0 = Op.make(
        "stack",
        *(
            Op.make("apply", Op.make("aff", A[k], c[k]), h)
            for k in range(n)
        ),
        dim=0,
    )
    eg = EGraph(track_proofs=True)
    root = _add(eg, t0)
    offers = XC.gather_apply_stack(eg)
    assert len(offers) == 1
    a_val = meta._eval_term(t0, env)
    b_val = meta._eval_term(offers[0]["term"], env)
    assert meta._eval_allclose(a_val, b_val, tol=1e-12)
    assert "apply" in _class_ops(eg, root)


def test_omd_tree_lift_pass():
    """The whole om tree lifts when EVERY leaf is affine in the same h:
    om_apply(om_compose(elem1, om_compose(elem2, elem3))) offers
    omd_apply(omd-tree, h)."""
    Tq, K1, K2, K3, d = 4, 3, 5, 2, 4
    s1, s2, s3 = (
        _V("s1", (Tq, K1)),
        _V("s2", (Tq, K2)),
        _V("s3", (Tq, K3)),
    )
    a1, b1 = _V("a1", (K1, d)), _V("b1", (K1, d))
    a2, b2 = _V("a2", (K2, d)), _V("b2", (K2, d))
    a3, b3 = _V("a3", (K3, d)), _V("b3", (K3, d))
    h = _V("h", (d,))
    env = {
        s1: _rand((Tq, K1), 180),
        s2: _rand((Tq, K2), 181),
        s3: _rand((Tq, K3), 182),
        a1: _rand((K1, d), 183),
        b1: _rand((K1, d), 184),
        a2: _rand((K2, d), 185),
        b2: _rand((K2, d), 186),
        a3: _rand((K3, d), 187),
        b3: _rand((K3, d), 188),
        h: _rand((d,), 189),
    }
    def leaf(s, a, b):
        return Op.make(
            "om_elem",
            s,
            Op.make("applyd", Op.make("aff_diag", a, b), h),
        )
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            leaf(s1, a1, b1),
            Op.make("om_compose", leaf(s2, a2, b2), leaf(s3, a3, b3)),
        ),
    )
    eg = EGraph(track_proofs=True)
    root = _add(eg, t0)
    offers = XC.omd_tree_lift(eg)
    assert len(offers) == 1
    assert offers[0]["kind"] == "diag"
    offered = offers[0]["term"]
    assert offered.op == "omd_apply"
    assert offered.args[0].op == "omd_compose"
    a_val = meta._eval_term(t0, env)
    b_val = meta._eval_term(offered, env)
    assert meta._eval_allclose(a_val, b_val, tol=1e-12)
    assert "omd_apply" in _class_ops(eg, root)


def test_omd_tree_lift_vetoes():
    """(i) one leaf with a DIFFERENT h → no lift; (ii) one leaf with a
    concrete (non-affine) value block → no lift."""
    Tq, K, d = 4, 3, 4
    s1, s2 = _V("s1", (Tq, K)), _V("s2", (Tq, K))
    a1, b1 = _V("a1", (K, d)), _V("b1", (K, d))
    a2, b2 = _V("a2", (K, d)), _V("b2", (K, d))
    h, h2, v = _V("h", (d,)), _V("h2", (d,)), _V("v", (K, d))
    def leaf(s, a, b, hh):
        return Op.make(
            "om_elem",
            s,
            Op.make("applyd", Op.make("aff_diag", a, b), hh),
        )
    # (i) mixed states
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_compose", leaf(s1, a1, b1, h), leaf(s2, a2, b2, h2)
        ),
    )
    eg = EGraph()
    _add(eg, t0)
    assert XC.omd_tree_lift(eg) == []
    # (ii) one leaf is a concrete value block — the carrier can't
    # express a partially-deferred numerator
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_compose", leaf(s1, a1, b1, h), Op.make("om_elem", s2, v)
        ),
    )
    eg = EGraph()
    _add(eg, t0)
    assert XC.omd_tree_lift(eg) == []


# ---------------------------------------------------------------------------
#  E. The seam in the e-graph — rules fire and the class really gains
#     the cross-carrier member
# ---------------------------------------------------------------------------


def test_egraph_xcarrier_fires():
    """In an e-graph, the elem laws and the omd lift land real members
    in the carrier classes — the seam is crossed by a named rule, not
    by construction."""
    Tq, K, d = 5, 7, 4
    s, a, b, h = (
        _V("s", (Tq, K)),
        _V("a", (K, d)),
        _V("b", (K, d)),
        _V("h", (d,)),
    )
    env = {
        s: _rand((Tq, K), 200),
        a: _rand((K, d), 201),
        b: _rand((K, d), 202),
        h: _rand((d,), 203),
    }
    t0 = Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            s,
            Op.make("applyd", Op.make("aff_diag", a, b), h),
        ),
    )
    eg = EGraph(track_proofs=True)
    root = _add(eg, t0)
    assert eg.apply_rule(XC.XC_OM_ELEM_AFFD, root)
    assert eg.apply_rule(XC.XC_OMD_LIFT, root)
    ops = _class_ops(eg, root)
    assert "omd_apply" in ops
    # the elem class gained the fused member
    assert any(n.op == "om_elem_affd" for n in eg._node_to_class)
    # the fired rules are recorded by name
    assert "xc_om_elem_affd" in eg.rule_fires
    assert "xc_omd_lift" in eg.rule_fires
    # evaluate via any_term on the class (may pick either member —
    # all members of a class are equal, so this is a real check)
    t_any = eg.any_term(root)
    assert meta._eval_allclose(
        meta._eval_term(t0, env), meta._eval_term(t_any, env), tol=1e-12
    )


def test_egraph_stack_then_omd_chain():
    """The two passes compose: gather the stack into ONE applyd, then
    attention over it lifts to omd — scan→attention is a single
    deferred carrier."""
    n, Tq, d = 4, 3, 4
    a = [_V(f"a{i}", (d,)) for i in range(n)]
    b = [_V(f"b{i}", (d,)) for i in range(n)]
    s = _V("s", (Tq, n))
    h = _V("h", (d,))
    env = {
        **{a[i]: _rand((d,), 210 + i) for i in range(n)},
        **{b[i]: _rand((d,), 220 + i) for i in range(n)},
        s: _rand((Tq, n), 230),
        h: _rand((d,), 231),
    }
    vseq = Op.make(
        "stack",
        *(
            Op.make("applyd", Op.make("aff_diag", a[i], b[i]), h)
            for i in range(n)
        ),
        dim=0,
    )
    t0 = Op.make("om_apply", Op.make("om_elem", s, vseq))
    eg = EGraph(track_proofs=True)
    _root = _add(eg, t0)
    st_offers = XC.gather_applyd_stack(eg)
    assert len(st_offers) == 1
    # after the union, the om_elem's value class has the fused applyd
    # member — the tree lift sees it
    om_offers = XC.omd_tree_lift(eg)
    assert len(om_offers) == 1
    a_val = meta._eval_term(t0, env)
    b_val = meta._eval_term(om_offers[0]["term"], env)
    assert meta._eval_allclose(a_val, b_val, tol=1e-12)


# ---------------------------------------------------------------------------
#  F. The wall — scores are quadratic, softmax is not linear attention
# ---------------------------------------------------------------------------


def test_score_side_is_quadratic_not_affine():
    """If q_i = M_i h + c_i and k_j = N_j h + d_j then s_ij = q_i·k_j
    carries hᵀ(M_iᵀN_j)h — measured as the non-affine residue
    s(x+y) − s(x) − s(y) + s(0) ≠ 0.  No affine carrier — and no om
    element — can hold it."""
    Tq, K, i = 4, 5, 6
    g = torch.Generator().manual_seed(777)
    M = torch.randn(Tq, i, dtype=torch.float64, generator=g)
    N = torch.randn(K, i, dtype=torch.float64, generator=g)
    cq = torch.randn(Tq, dtype=torch.float64, generator=g)
    dk = torch.randn(K, dtype=torch.float64, generator=g)

    def s(h):
        return torch.outer(M @ h + cq, N @ h + dk)  # (Tq,K) quadratic

    x = torch.randn(i, dtype=torch.float64, generator=g)
    y = torch.randn(i, dtype=torch.float64, generator=g)
    z = torch.zeros(i, dtype=torch.float64)
    residue = (s(x + y) - s(x) - s(y) + s(z)).abs().max().item()
    assert residue > 0.1  # genuinely quadratic — no affine law exists


def test_softmax_attention_is_not_linear_attention():
    """The linear-attention identity (QKᵀ)V = Q(KᵀV) is exact — and it
    is NOT softmax attention.  Any rule equating them would be
    approximate; we measure the gap instead of writing the law."""
    Tq, K, d = 4, 5, 3
    g = torch.Generator().manual_seed(888)
    q = torch.randn(Tq, d, dtype=torch.float64, generator=g)
    k = torch.randn(K, d, dtype=torch.float64, generator=g)
    v = torch.randn(K, d, dtype=torch.float64, generator=g)
    s = q @ k.T
    softmax_out = torch.softmax(s, dim=-1) @ v
    linear_out = (q @ k.T) @ v  # unnormalised "linear attn"
    kv_out = q @ (k.T @ v)  # the exact reassociation
    # the EXACT law: the two linear forms agree
    assert torch.allclose(linear_out, kv_out, atol=1e-12, rtol=1e-12)
    # the UNSOUND law nobody may write: softmax vs linear differ by O(1)
    gap = (softmax_out - linear_out).abs().max().item()
    assert gap > 1e-2
    # ...and even normalised linear attention differs from softmax:
    # softmax ≠ (s/l) @ v with l = s-sum — the weights are exp(s−m).
    lv = s.sum(dim=-1, keepdim=True)
    normed_linear = (s / lv.clamp_min(1e-9)) @ v
    assert not torch.allclose(normed_linear, softmax_out, atol=1e-6)


def test_om_compose_still_exact_on_affine_values():
    """The om homomorphism itself never needed to know the values were
    affine: om_elem(cat) == om_elem⊕om_elem holds for v = a⊙h+b —
    i.e. the seam crossing preserves the existing carrier laws."""
    Tq, K1, K2, d = 4, 3, 5, 4
    g = torch.Generator().manual_seed(999)
    s1 = torch.randn(Tq, K1, dtype=torch.float64, generator=g)
    s2 = torch.randn(Tq, K2, dtype=torch.float64, generator=g)
    a1 = torch.randn(K1, d, dtype=torch.float64, generator=g)
    b1 = torch.randn(K1, d, dtype=torch.float64, generator=g)
    a2 = torch.randn(K2, d, dtype=torch.float64, generator=g)
    b2 = torch.randn(K2, d, dtype=torch.float64, generator=g)
    h = torch.randn(d, dtype=torch.float64, generator=g)
    om_elem = _IR_TO_TORCH["om_elem"]
    om_compose = _IR_TO_TORCH["om_compose"]
    om_apply = _IR_TO_TORCH["om_apply"]
    whole = om_apply(
        om_elem(
            torch.cat([s1, s2], -1),
            torch.cat([a1 * h + b1, a2 * h + b2], -2),
        )
    )
    blocked = om_apply(
        om_compose(om_elem(s1, a1 * h + b1), om_elem(s2, a2 * h + b2))
    )
    assert torch.allclose(whole, blocked, atol=1e-13, rtol=1e-13)


# ---------------------------------------------------------------------------
#  End-to-end: export → carrier laws → non-local lifts → omd member
# ---------------------------------------------------------------------------


class _ScanAttn(torch.nn.Module):
    """h_t = a_t⊙h + x_t (diagonal scan); out = softmax(qkᵀ) @ stack(h).

    The attention VALUES are the scan's emitted sequence — the exact
    shape ``omd_tree_lift`` was built for.  h0 is a parameter so the
    first step does not degenerate to x_0."""

    def __init__(self, T: int, D: int):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(T, D) * 0.1)
        self.h0 = torch.nn.Parameter(torch.randn(D) * 0.1)
        self.wq = torch.nn.Linear(D, D, bias=False)
        self.wk = torch.nn.Linear(D, D, bias=False)

    def forward(self, x):
        h = self.h0
        outs = []
        for t in range(x.shape[0]):
            h = self.a[t] * h + x[t]
            outs.append(h)
        v = torch.stack(outs)
        s = self.wq(x) @ self.wk(x).transpose(-1, -2)
        return torch.softmax(s, dim=-1) @ v


def test_scan_to_attention_lifts_into_omd_end_to_end():
    """build_egraph on an exported scan→attention model produces an
    ``omd_apply`` member at the root class — chunked attention over
    scanned values as ONE affine-in-h0 recurrence — and the member
    evaluates fp64-exact."""
    from catopt.ir import IR
    from catopt.regime import build_egraph
    from catopt.torch_bridge import ir_to_torch_module

    torch.manual_seed(0)
    m = _ScanAttn(8, 8).eval().double()
    x = torch.randn(8, 8, dtype=torch.float64)
    eg, root, ir, src, stats = build_egraph(m, x)
    assert stats.get("nonlocal_lifts", 0) > 0

    # locate the omd_apply enode in the root class and extract its term
    term = None
    for n in eg.get_class(eg.find(root)).nodes:
        if n.op == "omd_apply":
            args = [eg.any_term(eg.find(c)) for c in n.children]
            term = Op.make("omd_apply", *args, **dict(n.attrs))
            break
    assert term is not None, "omd_tree_lift did not fire end-to-end"

    mod = ir_to_torch_module(
        IR(root=term, params=ir.params, inputs=ir.inputs), src
    )
    ref = m(x)
    with torch.no_grad():
        out = mod(x)
    assert (out - ref).abs().max().item() < 1e-12
