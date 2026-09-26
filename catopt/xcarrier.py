"""Cross-carrier laws — what can and cannot pass between the scan
(``aff``/``aff_diag``) and online-softmax (``om``) monoid domains.

THE QUESTION
    Every existing carrier lives behind its own seam: an e-class never
    holds enodes from two carrier families, because no rewrite has ever
    asserted an equality between them.  This module asks whether a
    *sound, exact* law crosses the seam — concretely: attention over a
    scanned sequence, where the scan emits states ``h_t`` that attention
    reads as queries, keys, or values.

THE POSITIVE RESULT — the value/readout side is crossable
    The observation that unlocks everything is that a carrier
    application is affine in the initial state, and **linear maps push
    through affine evaluation** (the scan-carrier analogue of the traced
    category's *tightening* axiom — a post-context exits the carrier):

        matmul(E, applyd(aff_diag(a,b), h))          (a,b : (K,d), h : (d,))
            = applyd(aff_diag(E@a, E@b), h)
        matmul(W, apply(aff(A,c), h))                (W : (o,i), A : (i,i))
            = apply(aff(W@A, W@c), h)
        linear(applyd(aff_diag(a,b),h), W)           (a,b : (…,i), h : (i,))
            = apply(aff(mul(unsqueeze(a,-2), W), linear(b,W)), h)
            — the diagonal carrier is *promoted* to the dense carrier:
            W·diag(a_t) is a dense per-row matrix.

    Stacking, scaling, slicing, and adding same-state applications are
    likewise exact:

        stack_i(applyd(f_i, h)) = applyd(aff_diag(stack affd_a f_i,
                                                stack affd_b f_i), h)
            — "the sequence a scan emits is ONE map applied to h0".
            The stack op is variadic, so this is a non-local pass
            (:func:`gather_applyd_stack`), not a lhs→rhs rule; the new
            projection ops ``affd_a``/``affd_b`` (and dense ``aff_A``/
            ``aff_b``) expose carrier components as tensor children so
            the stacked map is an ordinary ``aff_diag`` leaf.
        mul(applyd(f,h), r) = applyd((r⊙a, r⊙b), h)     (scalar r)
        add(applyd(f,h), applyd(g,h))                   (shared h)
            = applyd((affd_a f + affd_a g, affd_b f + affd_b g), h)
        chunk(applyd(aff_diag(a,b),h), ·, D, i)
            = applyd(aff_diag(chunk a, chunk b), h)     (D ≠ feature axis)
        reshape(apply(aff(A,b),h), S)
            = apply(aff(reshape(A, S+(i,)), reshape(b,S)), h)
        transpose(apply(aff(A,b),h), d1, d2)
            = apply(aff(transpose(A), transpose(b)), h) (dims mod
                value rank — the map's input axis is always last)
            — the dense view laws are exact for ANY well-typed view:
            a multi-head ``view(T,nh,hd).transpose(0,1)`` over a
            scanned value block keeps the affine member reachable
            under the view wrapper.  Diagonal variants exist but are
            restricted to views that keep the feature axis last
            (h is never permuted).

    THE HETEROGENEOUS ELEMENT
        The om numerator is ``e @ v`` — a matmul in the value slot, and
        therefore a case of the readout law:

            om_elem(s, applyd(aff_diag(a,b),h))
                = om_elem_affd(s, a, b, h)                    [fused elem]

        ``om_elem_affd`` evaluates to the SAME (m,l,a) triple, with the
        numerator computed as ``(e@a)⊙h + e@b`` — the scan is folded
        *inside* the om element.  The triple then flows through the
        ordinary ``om_compose``/``om_apply`` machinery: chunked
        attention over scanned values is a standard om tree with
        heterogeneous leaves.  ``om_elem_aff`` is the dense counterpart
        (v = A@h+b with per-key A : (K,d,i)).

    THE DEFERRED CARRIER — attention whose output stays affine in h
        Stronger: keep the numerator *symbolic* in h through the whole
        compose tree.  The ``omd`` carrier is the om triple over the
        diagonal-affine fiber:

            omd_elem(s, a, b) = (m, l, e@a, e@b)   — a 4-tuple
            omd_compose rescales BOTH numerator factors by the
                FlashAttention e-weights — distributing because the
                combine is bilinear in the numerator.
            omd_apply(f, h)  = (fa⊙h + fb) / l      (diagonal fiber)
            omd_applym(f, h) = (fa@h + fb) / l      (dense fiber)

        ``omd_apply(omd-tree, h)`` equals ``om_apply`` of the
        corresponding concrete tree — the whole of chunked attention
        over scanned values is ONE diagonal-affine map of the initial
        state, coefficientised by the softmax data.  This is the
        exact "attention-with-recurrent-state" (S4/RWKV-style) carrier:
        a single recursion whose step state is (m, l, affine numerator).

        Elem-level lifting into ``omd`` is NOT a local law: the
        ``om_elem`` value is a concrete 3-tuple while the ``omd_elem``
        value is a deferred 4-tuple — different types, unifiable only
        under the enclosing ``omd_apply(·, h)``.  So the local rules
        lift ``om_apply(om_elem(s, applyd …))`` directly (h is bound),
        a pair rule covers binary composes, and the general tree lift
        is the non-local pass :func:`omd_tree_lift` (all leaves must be
        affine in the SAME h — a global condition no lhs→rhs rule can
        see).

THE NEGATIVE RESULT — the score side is a wall
    If ``q_i = M_i h + c_i`` and ``k_j = N_j h + d_j`` are both affine
    in the shared state, then

        s_ij = q_i · k_j = hᵀ(M_iᵀ N_j)h + (affine terms)

    is QUADRATIC in h — no affine carrier captures it, and softmax of
    a quadratic form has no finite carrier at all (exp∘quadratic is
    transcendental in h).  With only ONE side affine (say k scanned,
    q external) the scores are affine with a rank-3 coefficient
    ``A_s[i,j,:] = M_iᵀ k_j`` — algebraically fine but inexpressible:
    the contraction ``Σ_o M[i,o,p]·k[j,o]`` is an einsum over a
    non-adjacent pair of axes, and this IR has no einsum/tensordot op.
    Other candidates likewise fail:

    * softmax-attention → linear-attention: approximate, not exact.
      (What IS exact: ``(QKᵀ)V → Q(KᵀV)`` by matmul associativity —
      linear attention's KV state is then plain accumulation, already
      reachable through ``affd_lift_unit``.)
    * trace ↔ om: ``Tr(f)`` is a LINEAR fixpoint (resolvent); the om
      recurrence is nonlinear (max).  No exact law.
    * scan-of-softmax: softmax is nonlinear, no fold law exists.

VERIFICATION
    Every rewrite here is fp64-verified on concrete tensors in
    tests/test_xcarrier.py; the passes assert equality by construction
    and (optionally) attach pointwise witnesses like ``trace_lift``.

Torch bindings for the new ops are registered into
:data:`catopt.torch_bridge._IR_TO_TORCH` at import time — additive,
the same mechanism :mod:`catopt.trace` uses.
"""

from __future__ import annotations

from typing import Any

import torch

from catopt.egraph import EGraph, Rewrite
from catopt.ir import Op, TensorType, Var
from catopt.rules import R
from catopt.torch_bridge import _IR_TO_TORCH

__all__ = [
    "XC_LAWS",
    "gather_apply_stack",
    "gather_applyd_stack",
    "omd_tree_lift",
]


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.cost import _shape_of as _so

    return _so(t)


def _broadcast_ok(a, b) -> bool:
    from catopt.cost import _INVALID, _broadcast

    return _broadcast(a, b) is not _INVALID


def _bc(a, b):
    """Broadcast result, or None when incompatible/unknown."""
    from catopt.cost import _INVALID, _broadcast

    r = _broadcast(a, b)
    return None if r is _INVALID else r


def _concrete(s) -> bool:
    return (
        isinstance(s, tuple)
        and len(s) > 0
        and all(isinstance(d, int) for d in s)
    )


def _vec(t) -> bool:
    s = _shape_of(t)
    return _concrete(s) and len(s) == 1


def _xshape(t: Any, _memo: dict | None = None):
    """The TRUE value shape of a bound term — for the XC guards.

    ``catopt.cost._shape_of`` prices the carriers by convention:
    ``apply``/``applyd`` report h's (state) shape, ``aff``/``aff_diag``
    report the map's linear part.  That convention is exact on the
    recurrence spine, where state and value shapes coincide — but it
    lies for cross-carrier members: a stacked or promoted map's value
    (…,o) differs from the state (i,).  ``EGraph.any_term`` resolves a
    metavariable to an ARBITRARY class member, so a tensor e-class may
    resolve through a carrier member whose convention shape vetoes a
    legal rewrite (observed: ``xc_om_elem_aff`` on HybridBlock — the
    score block's matmul reads through a promoted ``apply`` and
    reports ``(d,)`` instead of ``(T,d)``).

    This resolver computes the *value* shape for the carrier-
    application ops and the fused/deferred om family (absent from the
    cost model's dispatch), then re-dispatches ``_shape_of`` over
    surrogate leaves carrying the corrected child shapes — so carrier
    members nested inside an ordinary tensor term cannot poison the
    result.  Map terms (``aff``/``aff_diag``/the composes) keep the
    cost convention — the checks rely on it.
    """
    if _memo is None:
        _memo = {}
    key = id(t)
    if key in _memo:
        return _memo[key]
    out = _xshape_rec(t, _memo)
    _memo[key] = out
    return out


def _xshape_rec(t: Any, memo: dict):
    if not isinstance(t, Op):
        return _shape_of(t)
    op, args = t.op, t.args
    if op == "applyd" and len(args) == 2:
        # value = a⊙h + b — broadcast of all three factors
        f, h = args
        hs = _xshape(h, memo)
        if (
            isinstance(f, Op)
            and f.op == "aff_diag"
            and len(f.args) == 2
        ):
            return _bc(
                _bc(_xshape(f.args[0], memo), hs),
                _xshape(f.args[1], memo),
            )
        fs = _shape_of(f)  # opaque map: a-part convention
        return _bc(fs, hs)
    if op == "apply" and len(args) == 2:
        # value = A@h + c — A (…,o,i) contracts h's axis, + c
        f, _h = args
        if isinstance(f, Op) and f.op == "aff" and len(f.args) == 2:
            As = _xshape(f.args[0], memo)
            cs = _xshape(f.args[1], memo)
            if isinstance(As, tuple) and len(As) >= 1:
                return _bc(tuple(As[:-1]), cs)
            return cs
        fs = _shape_of(f)  # dense map convention: (…,o,i)
        if isinstance(fs, tuple) and len(fs) >= 2:
            return tuple(fs[:-1])
        return _shape_of(t)
    if op in ("affd_a", "aff_A") and len(args) == 1:
        f = args[0]
        if isinstance(f, Op) and f.args:
            return _xshape(f.args[0], memo)
        return _shape_of(f)
    if op in ("affd_b", "aff_b") and len(args) == 1:
        f = args[0]
        if isinstance(f, Op) and len(f.args) >= 2:
            return _xshape(f.args[1], memo)
        fs = _shape_of(f)
        if op == "aff_b" and isinstance(fs, tuple) and len(fs) >= 2:
            return tuple(fs[:-1])  # dense b-part ≈ output shape
        return fs  # diag b-part ≈ a-part shape
    # The fused/deferred om family — the applied VALUE shape, the same
    # convention om_elem uses: score prefix ++ value's last dim.
    if op == "om_elem_affd" and len(args) == 4:
        ss, sa = _xshape(args[0], memo), _xshape(args[1], memo)
        if _concrete(ss) and _concrete(sa):
            return tuple(ss[:-1]) + (sa[-1],)
        return _shape_of(t)
    if op == "om_elem_aff" and len(args) == 4:
        ss, sa = _xshape(args[0], memo), _xshape(args[1], memo)
        if _concrete(ss) and _concrete(sa) and len(sa) >= 2:
            return tuple(ss[:-1]) + (sa[-2],)
        return _shape_of(t)
    if op == "omd_elem" and len(args) == 3:
        ss, sa = _xshape(args[0], memo), _xshape(args[1], memo)
        if _concrete(ss) and _concrete(sa):
            # dense fiber a (…,K,d,i) → applied value (…,Tq,d)
            last = sa[-2] if len(sa) >= 3 else sa[-1]
            return tuple(ss[:-1]) + (last,)
        return _shape_of(t)
    if op == "omd_compose" and len(args) == 2:
        return _xshape(args[0], memo)
    if op in ("omd_apply", "omd_applym") and len(args) == 2:
        return _xshape(args[0], memo)
    if op == "omd" and len(args) == 4:
        return _xshape(args[3], memo)  # fb — applied numerator shape
    # generic op: re-dispatch _shape_of on surrogate leaves carrying
    # the corrected child shapes.
    fixed = list(args)
    touched = False
    for i, a in enumerate(fixed):
        if isinstance(a, Op):
            s = _xshape(a, memo)
            if _concrete(s):
                fixed[i] = Var("_xs", TensorType(tuple(s)))
                touched = True
    if touched:
        return _shape_of(Op.make(op, *fixed, **t.attrs))
    return _shape_of(t)


def _shape(t):
    s = _xshape(t)
    return s if isinstance(s, tuple) else None


# ---------------------------------------------------------------------------
#  Torch bindings — registered into the bridge's op table (additive).
# ---------------------------------------------------------------------------


def _affd_a(f, *a, **kw):
    return f[0]


def _affd_b(f, *a, **kw):
    return f[1]


def _om_elem_affd(
    s: torch.Tensor, a: torch.Tensor, b: torch.Tensor, h: torch.Tensor
):
    """om_elem whose value block is diagonal-affine in h:
    elem(s, a⊙h+b) = (m, l, (e@a)⊙h + e@b).  Same triple as
    ``om_elem(s, applyd(aff_diag(a,b),h))`` — fp64-identical modulo
    reassociation."""
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    return m, e.sum(dim=-1, keepdim=True), (e @ a) * h + e @ b


def _om_elem_aff(
    s: torch.Tensor, A: torch.Tensor, b: torch.Tensor, h: torch.Tensor
):
    """om_elem with dense-affine values: v = A@h + b with per-key
    A : (…,K, d, i).  numerator = (e@A)@h + e@b where e@A contracts the
    key axis — computed via reshape (the IR has no einsum).  Leading
    axes of A (e.g. attention heads) are batch dims for the matmul and
    are preserved through the flattening reshape."""
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    d, i = A.shape[-2], A.shape[-1]
    ea = e @ A.reshape(*A.shape[:-2], d * i)  # (…,Tq,d·i)
    ea = ea.reshape(*ea.shape[:-1], d, i)  # (…,Tq,d,i)
    return m, e.sum(dim=-1, keepdim=True), ea @ h + e @ b


def _omd_elem(s: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Deferred-affine om element: the om triple of (s, M(h)+b) with h
    left symbolic — a 4-tuple (m, l, e@M, e@b) whose numerator is the
    affine PAIR.

    Diagonal fiber: a is (…,K,d) → e@a is the (…,Tq,d) diagonal
    coefficient (consumed by ``omd_apply``).  Dense fiber: a is
    (…,K,d,i) → e contracts the key axis via a (d·i)-flattened reshape
    (the IR has no einsum), giving the (…,Tq,d,i) dense coefficient
    (consumed by ``omd_applym``)."""
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    l = e.sum(dim=-1, keepdim=True)
    if a.dim() >= 3:
        d, i = a.shape[-2], a.shape[-1]
        ea = e @ a.reshape(*a.shape[:-2], d * i)
        ea = ea.reshape(*ea.shape[:-1], d, i)
        return m, l, ea, e @ b
    return m, l, e @ a, e @ b


def _omd(m, l, fa, fb, *a, **kw):
    return (m, l, fa, fb)


def _omd_compose(f, g):
    """Combine two deferred-affine triples.  The rescaling is linear in
    each numerator component, so the pair stays a pair:
    e1·(fa1⊙h+fb1) + e2·(fa2⊙h+fb2) = (e1 fa1 + e2 fa2)⊙h
                                    + (e1 fb1 + e2 fb2)."""
    m1, l1, fa1, fb1 = f
    m2, l2, fa2, fb2 = g
    mx = torch.maximum(m1, m2)
    fin1, fin2 = torch.isfinite(m1), torch.isfinite(m2)
    e1 = torch.where(fin1, torch.exp(m1 - mx), torch.zeros_like(mx))
    e2 = torch.where(fin2, torch.exp(m2 - mx), torch.zeros_like(mx))
    # fa may carry extra trailing axes past e's (…,Tq,1) — the dense
    # fiber's coefficient is (…,Tq,d,i) — so the row weight needs
    # trailing 1-dims to broadcast (diag fiber: a no-op reshape).
    e1f = e1.reshape(*e1.shape, *([1] * (fa1.dim() - e1.dim())))
    e2f = e2.reshape(*e2.shape, *([1] * (fa2.dim() - e2.dim())))
    fin1f = fin1.reshape(*fin1.shape, *([1] * (fa1.dim() - fin1.dim())))
    fin2f = fin2.reshape(*fin2.shape, *([1] * (fa2.dim() - fin2.dim())))
    zl1, zl2 = torch.zeros_like(l1), torch.zeros_like(l2)
    za1, za2 = torch.zeros_like(fa1), torch.zeros_like(fa2)
    zb1, zb2 = torch.zeros_like(fb1), torch.zeros_like(fb2)
    l = torch.where(fin1, l1 * e1, zl1) + torch.where(
        fin2, l2 * e2, zl2
    )
    fa = torch.where(fin1f, fa1 * e1f, za1) + torch.where(
        fin2f, fa2 * e2f, za2
    )
    fb = torch.where(fin1, fb1 * e1, zb1) + torch.where(
        fin2, fb2 * e2, zb2
    )
    return mx, l, fa, fb


def _omd_apply(f, h, *a, **kw):
    """Apply a deferred-diagonal carrier: (fa⊙h + fb) / l."""
    return (f[2] * h + f[3]) / f[1]


def _omd_applym(f, h, *a, **kw):
    """Apply a deferred-dense carrier: (fa@h + fb) / l.  fa's last dim
    is the map's input axis — (…,Tq,o,i) @ (i,) → (…,Tq,o)."""
    return (f[2] @ h + f[3]) / f[1]


_IR_TO_TORCH["affd_a"] = _affd_a
_IR_TO_TORCH["affd_b"] = _affd_b
_IR_TO_TORCH["aff_A"] = _affd_a
_IR_TO_TORCH["aff_b"] = _affd_b
_IR_TO_TORCH["om_elem_affd"] = _om_elem_affd
_IR_TO_TORCH["om_elem_aff"] = _om_elem_aff
_IR_TO_TORCH["omd"] = _omd
_IR_TO_TORCH["omd_elem"] = _omd_elem
_IR_TO_TORCH["omd_compose"] = _omd_compose
_IR_TO_TORCH["omd_apply"] = _omd_apply
_IR_TO_TORCH["omd_applym"] = _omd_applym


# ---------------------------------------------------------------------------
#  Guards
# ---------------------------------------------------------------------------


def _check_matmul_applyd_vec(bound: dict) -> bool:
    """matmul(W, applyd(aff_diag(a,b),h)) — W contracts the FEATURE
    axis: a,b,h all (d,), W (o,d).  The dense promotion:
    W·(a⊙h+b) = (W·diag a)h + Wb."""
    W, a, b, h = (_shape(bound.get(k)) for k in ("W", "a", "b", "h"))
    if not all(_concrete(s) for s in (W, a, b, h)):
        return False
    return (
        len(W) == 2 and len(a) == 1 and a == b == h and W[-1] == a[-1]
    )


def _check_matmul_apply(bound: dict) -> bool:
    """matmul(W, apply(aff(A,c),h)) = apply(aff(WA,Wc),h) — post-compose
    of a linear map with a dense affine map.  W (o,i), A (i,i),
    c,h (i,)."""
    W, A, c, h = (_shape(bound.get(k)) for k in ("W", "A", "c", "h"))
    if not all(_concrete(s) for s in (W, A, c, h)):
        return False
    return (
        len(W) == 2
        and len(A) == 2
        and A[0] == A[1] == W[-1]
        and len(c) == 1
        and c == h
        and c[-1] == A[-1]
    )


def _check_linear_applyd(bound: dict) -> bool:
    """linear(applyd(aff_diag(a,b),h), W) — the diagonal→dense
    promotion: W·diag(a_t) per row = mul(unsqueeze(a,-2), W).
    a,b : (…,i) equal shapes, h : (i,), W : (o,i)."""
    W, a, b, h = (_shape(bound.get(k)) for k in ("W", "a", "b", "h"))
    if not all(_concrete(s) for s in (W, a, b, h)):
        return False
    return (
        len(W) == 2
        and len(a) >= 1
        and a == b
        and len(h) == 1
        and h[-1] == a[-1] == W[-1]
    )


def _check_linear_apply(bound: dict) -> bool:
    """linear(apply(aff(A,c),h), W) = apply(aff(WA, linear(c,W)),h):
    A (…,i,i), W (o,i), h (i,), c broadcastable to (…,i)."""
    W, A, c, h = (_shape(bound.get(k)) for k in ("W", "A", "c", "h"))
    if not all(_concrete(s) for s in (W, A, c, h)):
        return False
    if not (
        len(W) == 2
        and len(A) >= 2
        and A[-1] == A[-2] == W[-1]
        and len(h) == 1
        and h[-1] == A[-1]
        and len(c) >= 1
    ):
        return False
    return _broadcast_ok(c, A[:-1])


def _scalar_or_broadcast(bound: dict, rkey: str, tshape) -> bool:
    rs = _shape(bound.get(rkey))
    return (
        rs is not None
        and _broadcast_ok(rs, tshape)
        and (len(rs) == 0 or rs == tshape or rs == (tshape[-1],))
    )


def _named_scale(bound: dict, rkey: str) -> bool:
    """The scale operand must resolve to a LEAF — a named scalar or
    gain (Var/Param/Const), not a computed factor.

    A *computed* r (any Op term — e.g. the per-step decay a_t on a
    recurrence spine) makes this law replay the carrier's own step
    composition as a concrete map: the minted ``mul(f.a, r)`` factors
    unfold via ``affd_unlift`` and re-match this same rule against
    EVERY applyd member of the state class — a cross-product per step,
    hence exponential e-graph growth.  A leaf r cannot create that
    loop: the factors a firing mints are always Op terms, which this
    guard then vetoes.  Computed scales are not lost coverage either —
    ``r ⊙ applyd(f, h)`` is already reachable through the carrier's
    own ``affd_compose``/``aff_compose`` machinery; the leaf case
    (readout scales like 1/√d, learned gains) is the one that needs a
    concrete map."""
    return not isinstance(bound.get(rkey), Op)


def _check_scale_applyd(bound: dict) -> bool:
    """mul(applyd(aff_diag(a,b),h), r) = applyd((r⊙a, r⊙b),h): r must
    be a leaf (named scalar/gain) broadcasting against the state shape
    (scalar, (d,), or full shape)."""
    if not _named_scale(bound, "r"):
        return False
    a, b, h = (_shape(bound.get(k)) for k in ("a", "b", "h"))
    if not all(_concrete(s) for s in (a, b, h)):
        return False
    if a != b or len(h) != 1 or h[-1] != a[-1]:
        return False
    return _scalar_or_broadcast(bound, "r", a)


def _check_scale_apply(bound: dict) -> bool:
    """mul(apply(aff(A,c),h), r) = apply(aff(rA, rc),h): r broadcasts
    against the OUTPUT shape A[:-1] (scalar or row scale (…,1) also
    fine since mul broadcasts).  Keep it strict: scalar or A[:-1] or
    (…,1)-shaped broadcastable."""
    A, c, h = (_shape(bound.get(k)) for k in ("A", "c", "h"))
    if not all(_concrete(s) for s in (A, c, h)):
        return False
    if not _named_scale(bound, "r"):
        return False
    if not (
        len(A) >= 2
        and A[-1] == A[-2]
        and len(h) == 1
        and h[-1] == A[-1]
    ):
        return False
    if not _broadcast_ok(c, A[:-1]):
        return False
    rs = _shape(bound.get("r"))
    if rs is None or not _broadcast_ok(rs, A[:-1]):
        return False
    # r must also broadcast against A (…,o,i): only safe when r is a
    # scalar or matches A[:-1] extended correctly — restrict to scalar
    # or exact output shape or (…,1) row scale.
    if len(rs) == 0:
        return True
    if rs == A[:-1]:
        return True
    return (
        len(rs) == len(A) - 1
        and rs[-1] == 1
        and _broadcast_ok(rs[:-1], A[:-2])
    )


def _check_add_applyd(bound: dict) -> bool:
    """add(applyd(f,h), applyd(g,h)) with the SAME h: componentwise add
    of the two maps.  f,g's shapes are their a-parts (cost convention);
    require equal shapes and h a (d,) broadcast vector."""
    f, g, h = (_shape(bound.get(k)) for k in ("f", "g", "h"))
    if not all(_concrete(s) for s in (f, g, h)):
        return False
    return f == g and len(h) == 1 and h[-1] == f[-1]


def _check_add_apply(bound: dict) -> bool:
    """add(apply(f,h), apply(g,h)) — dense analog; f's shape is A's:
    require equal (…,i,i) shapes and h (i,)."""
    f, g, h = (_shape(bound.get(k)) for k in ("f", "g", "h"))
    if not all(_concrete(s) for s in (f, g, h)):
        return False
    return (
        f == g
        and len(f) >= 2
        and f[-1] == f[-2]
        and len(h) == 1
        and h[-1] == f[-1]
    )


def _check_chunk_applyd(bound: dict, split: bool = False) -> bool:
    """chunk/split of applyd(aff_diag(a,b),h): the slice axis must not
    be the FEATURE axis (h is not chunked), i.e. D % rank < rank-1."""
    a, b, h = (_shape(bound.get(k)) for k in ("a", "b", "h"))
    D = bound.get("$attr:D")
    if not (
        all(_concrete(s) for s in (a, b, h)) and isinstance(D, int)
    ):
        return False
    if a != b or len(a) < 2 or len(h) != 1 or h[-1] != a[-1]:
        return False
    return D % len(a) < len(a) - 1


def _check_chunk_apply(bound: dict) -> tuple[bool, int | None]:
    """chunk(apply(aff(A,c),h)): A = c's shape ++ (i,); the slice dim D
    (in c's coords) maps to the same leading index of A (A's extra axis
    is LAST).  Any axis of c is sound.  Returns (ok, DA)."""
    A, c, h = (_shape(bound.get(k)) for k in ("A", "c", "h"))
    D = bound.get("$attr:D")
    if not (
        all(_concrete(s) for s in (A, c, h)) and isinstance(D, int)
    ):
        return False, None
    if not (
        len(c) >= 1
        and len(A) == len(c) + 1
        and A[:-1] == c
        and len(h) == 1
        and h[-1] == A[-1]
    ):
        return False, None
    return True, D % len(c)


def _check_split_sizes_int(bound: dict) -> bool:
    sz = bound.get("$attr:SZ")
    return isinstance(sz, (list, tuple)) and all(
        isinstance(x, int) for x in sz
    )


def _check_chunk_applyd_ok(bound: dict) -> bool:
    return _check_chunk_applyd(bound)


def _check_split_applyd(bound: dict) -> bool:
    return _check_chunk_applyd(bound) and _check_split_sizes_int(bound)


def _check_chunk_apply_ok(bound: dict) -> bool:
    return _check_chunk_apply(bound)[0]


def _derive_chunk_apply(bound: dict):
    ok, da = _check_chunk_apply(bound)
    return {"$attr:DA": da} if ok else None


def _check_split_apply(bound: dict) -> bool:
    return _check_chunk_apply_ok(bound) and _check_split_sizes_int(
        bound
    )


def _check_om_elem_affd(bound: dict) -> bool:
    """om_elem(s, applyd(aff_diag(a,b),h)) — v = a⊙h+b must be a valid
    value block: s (…,Tq,K), a,b (…,K,d) equal, s[-1]==a[-2], h (d,)."""
    s, a, b, h = (_shape(bound.get(k)) for k in ("s", "a", "b", "h"))
    if not all(_concrete(x) for x in (s, a, b, h)):
        return False
    if not (len(s) >= 2 and len(a) >= 2 and a == b and len(h) == 1):
        return False
    if s[-1] != a[-2] or h[-1] != a[-1]:
        return False
    return _broadcast_ok(s[:-2], a[:-2])


def _check_om_elem_aff(bound: dict) -> bool:
    """om_elem(s, apply(aff(A,b),h)) — v = A@h+b, A (…,K,d,i) rank-≥3,
    b == A[:-1], h (i,), s (…,Tq,K).  Leading axes of A beyond
    (K,d,i) are BATCH dims — a per-head map (nh,K,d,i) is fine
    because the torch bindings matmul over them; they only have to
    broadcast against s's leading dims."""
    s, A, b, h = (_shape(bound.get(k)) for k in ("s", "A", "b", "h"))
    if not all(_concrete(x) for x in (s, A, b, h)):
        return False
    if not (
        len(s) >= 2
        and len(A) >= 3
        and s[-1] == A[-3]
        and b == A[:-1]
        and len(h) == 1
        and h[-1] == A[-1]
    ):
        return False
    return _broadcast_ok(s[:-2], A[:-3])


def _check_omd_lift(bound: dict) -> bool:
    """Same shape requirements as om_elem_affd — plus h stays bound
    through to the omd_apply."""
    return _check_om_elem_affd(bound)


def _check_omd_lift_dense(bound: dict) -> bool:
    return _check_om_elem_aff(bound)


def _check_omd_pair(bound: dict) -> bool:
    """Two om_elem-applyd leaves composed under om_apply — each leaf
    must satisfy the elem-applyd shape relation and the two score
    blocks must agree off the key axis."""
    ok1 = _check_om_elem_affd(
        {
            "s": bound.get("s1"),
            "a": bound.get("a1"),
            "b": bound.get("b1"),
            "h": bound.get("h"),
        }
    )
    ok2 = _check_om_elem_affd(
        {
            "s": bound.get("s2"),
            "a": bound.get("a2"),
            "b": bound.get("b2"),
            "h": bound.get("h"),
        }
    )
    if not (ok1 and ok2):
        return False
    s1, s2 = _shape(bound.get("s1")), _shape(bound.get("s2"))
    a1, a2 = _shape(bound.get("a1")), _shape(bound.get("a2"))
    return all(s1[i] == s2[i] for i in range(len(s1) - 1)) and all(
        a1[i] == a2[i] for i in range(len(a1)) if i != len(a1) - 2
    )


def _check_omd_split(bound: dict) -> bool:
    """omd_elem(cat s, cat a, cat b) splits like om_elem: scores cat on
    the LAST dim (keys), a/b cat on dim -2 (the same key axis)."""
    sd, ad, bd = (
        bound.get("$attr:SD"),
        bound.get("$attr:AD"),
        bound.get("$attr:BD"),
    )
    s1, s2 = _shape(bound.get("s1")), _shape(bound.get("s2"))
    a1, a2 = _shape(bound.get("a1")), _shape(bound.get("a2"))
    b1, b2 = _shape(bound.get("b1")), _shape(bound.get("b2"))
    if not all(_concrete(x) for x in (s1, s2, a1, a2, b1, b2)):
        return False
    if not all(isinstance(d, int) for d in (sd, ad, bd)):
        return False
    if len(s1) != len(s2) or len(a1) != len(a2) or a1 != b1 or a2 != b2:
        return False
    if sd % len(s1) != len(s1) - 1:
        return False
    if ad % len(a1) != len(a1) - 2 or bd % len(b1) != len(b1) - 2:
        return False
    if not all(s1[i] == s2[i] for i in range(len(s1) - 1)):
        return False
    if not (
        all(a1[i] == a2[i] for i in range(len(a1)) if i != len(a1) - 2)
        and s1[-1] == a1[-2]
        and s2[-1] == a2[-2]
    ):
        return False
    return _broadcast_ok(s1[:-2], a1[:-2]) and _broadcast_ok(
        s2[:-2], a2[:-2]
    )


def _matmul_applyd_rows_via_f(bound: dict) -> bool:
    """Check for the row-contraction law when only f is bound.

    The LHS ``matmul(E, applyd(f, h))`` binds the map f opaquely; the
    components enter the RHS through the ``affd_a``/``affd_b``
    projections, so the check verifies the *application's* shape
    contract: the value v = applyd(f,h) must be a rank-≥2 block whose
    second-to-last dim matches E[-1], and f's own shape (its a-part,
    by the cost model's convention) is v's shape, with h a vector
    matching the last dim."""
    E, f, h = (_shape(bound.get(k)) for k in ("E", "f", "h"))
    if not all(_concrete(s) for s in (E, f, h)):
        return False
    if len(E) < 2 or len(f) < 2 or len(h) != 1:
        return False
    if h[-1] != f[-1]:
        return False
    # v's shape = broadcast(f, h) = f's shape here (h broadcasts on the
    # feature dim).  The contraction is E[-1] vs v[-2] = f[-2].
    if E[-1] != f[-2]:
        return False
    return _broadcast_ok(E[:-2], f[:-2])


# ---------------------------------------------------------------------------
#  Laws — carrier readout / promotion (the tightening analogue)
# ---------------------------------------------------------------------------

#: E contracts the ROW axis of a diagonal-affine value block:
#: E @ (a⊙h + b) = (E@a)⊙h + E@b — stays in the diagonal carrier.
XC_MATMUL_APPLYD_ROWS = R(
    "xc_matmul_applyd_rows",
    Op.make("matmul", "E", Op.make("applyd", "f", "h")),
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("matmul", "E", Op.make("affd_a", "f")),
            Op.make("matmul", "E", Op.make("affd_b", "f")),
        ),
        "h",
    ),
    law="A left linear map on a diagonal-affine vector block contracts "
    "the ROW axis and stays diagonal: E@(a⊙h+b) = (E@a)⊙h + E@b.  "
    "This is the om numerator's shape (e@v) — the readout law that "
    "crosses into the softmax carrier.",
    check=_matmul_applyd_rows_via_f,
)

XC_MATMUL_APPLYD_ROWS_REV = R(
    "xc_matmul_applyd_rows_rev",
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("matmul", "E", Op.make("affd_a", "f")),
            Op.make("matmul", "E", Op.make("affd_b", "f")),
        ),
        "h",
    ),
    Op.make("matmul", "E", Op.make("applyd", "f", "h")),
    law="reverse of xc_matmul_applyd_rows",
)

#: W contracts the FEATURE axis of a vector state: promotes the
#: diagonal carrier into the dense one (W·diag(a) is dense).
XC_MATMUL_APPLYD_VEC = R(
    "xc_matmul_applyd_vec",
    Op.make(
        "matmul",
        "W",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
    ),
    Op.make(
        "apply",
        Op.make(
            "aff", Op.make("mul", "W", "a"), Op.make("matmul", "W", "b")
        ),
        "h",
    ),
    law="Feature-axis contraction of a diagonal-affine vector is a "
    "DENSE affine map: W(a⊙h+b) = (W·diag a)h + Wb.  Carrier "
    "promotion aff_diag → aff.",
    check=_check_matmul_applyd_vec,
)

XC_MATMUL_APPLYD_VEC_REV = R(
    "xc_matmul_applyd_vec_rev",
    Op.make(
        "apply",
        Op.make(
            "aff", Op.make("mul", "W", "a"), Op.make("matmul", "W", "b")
        ),
        "h",
    ),
    Op.make(
        "matmul",
        "W",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
    ),
    law="reverse promotion (only matches the exact promoted form)",
    check=_check_matmul_applyd_vec,
)

#: Post-composition through the dense carrier (tightening):
#: matmul(W, apply(aff(A,c),h)) = apply(aff(WA, Wc), h).
XC_MATMUL_APPLY = R(
    "xc_matmul_apply",
    Op.make(
        "matmul", "W", Op.make("apply", Op.make("aff", "A", "c"), "h")
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("matmul", "W", "A"),
            Op.make("matmul", "W", "c"),
        ),
        "h",
    ),
    law="Linear post-context exits the carrier: W·(Ah+c) = (WA)h + Wc.",
    check=_check_matmul_apply,
)

XC_MATMUL_APPLY_REV = R(
    "xc_matmul_apply_rev",
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("matmul", "W", "A"),
            Op.make("matmul", "W", "c"),
        ),
        "h",
    ),
    Op.make(
        "matmul", "W", Op.make("apply", Op.make("aff", "A", "c"), "h")
    ),
    law="reverse tightening",
    check=_check_matmul_apply,
)

#: F.linear forms — what torch.export actually emits.
XC_LINEAR_APPLYD = R(
    "xc_linear_applyd",
    Op.make(
        "linear",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        "W",
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("mul", Op.make("unsqueeze", "a", dim=-2), "W"),
            Op.make("linear", "b", "W"),
        ),
        "h",
    ),
    law="linear readout of a diagonal-affine state is a DENSE affine "
    "map in h: W(a⊙h+b) = (W·diag a)h + Wb, and W·diag(a_t) for a "
    "batched (T,i) a is mul(unsqueeze(a,-2), W) : (T,o,i).",
    check=_check_linear_applyd,
)

XC_LINEAR_APPLYD_REV = R(
    "xc_linear_applyd_rev",
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("mul", Op.make("unsqueeze", "a", dim=-2), "W"),
            Op.make("linear", "b", "W"),
        ),
        "h",
    ),
    Op.make(
        "linear",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        "W",
    ),
    law="reverse promotion",
    check=_check_linear_applyd,
)

XC_LINEAR_APPLY = R(
    "xc_linear_apply",
    Op.make(
        "linear", Op.make("apply", Op.make("aff", "A", "c"), "h"), "W"
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("matmul", "W", "A"),
            Op.make("linear", "c", "W"),
        ),
        "h",
    ),
    law="F.linear post-composition through the dense carrier: "
    "(Ah+c)Wᵀ = (WA)h + cWᵀ — batched A (…,i,i) broadcasts W.",
    check=_check_linear_apply,
)

XC_LINEAR_APPLY_REV = R(
    "xc_linear_apply_rev",
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("matmul", "W", "A"),
            Op.make("linear", "c", "W"),
        ),
        "h",
    ),
    Op.make(
        "linear", Op.make("apply", Op.make("aff", "A", "c"), "h"), "W"
    ),
    law="reverse",
    check=_check_linear_apply,
)


# ---------------------------------------------------------------------------
#  Scalar scale / same-state add — carrier arithmetic on the value side
# ---------------------------------------------------------------------------


def _scale_applyd(name, r_first):
    mul = lambda x, y: Op.make("mul", x, y)
    lhs = (
        mul("r", Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"))
        if r_first
        else mul(
            Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"), "r"
        )
    )
    return R(
        name,
        lhs,
        Op.make(
            "applyd",
            Op.make(
                "aff_diag",
                Op.make("mul", "a", "r"),
                Op.make("mul", "b", "r"),
            ),
            "h",
        ),
        law="A named (leaf) scale commutes into the carrier: "
        "r⊙(a⊙h+b) = (r⊙a)⊙h + r⊙b.  r must resolve to a leaf — a "
        "computed factor (e.g. a per-step decay on a recurrence "
        "spine) is already reachable through affd_compose and "
        "would loop with affd_unlift.",
        check=_check_scale_applyd,
    )


XC_SCALE_APPLYD = _scale_applyd("xc_scale_applyd", r_first=False)
XC_SCALE_APPLYD_PRE = _scale_applyd("xc_scale_applyd_pre", r_first=True)


def _scale_apply(name, r_first):
    base = Op.make("apply", Op.make("aff", "A", "c"), "h")
    lhs = (
        Op.make("mul", "r", base)
        if r_first
        else Op.make("mul", base, "r")
    )
    return R(
        name,
        lhs,
        Op.make(
            "apply",
            Op.make(
                "aff",
                Op.make("mul", "A", "r"),
                Op.make("mul", "c", "r"),
            ),
            "h",
        ),
        law="Scalar/row scale commutes into the dense carrier: "
        "r·(Ah+c) = (rA)h + rc.  Same leaf restriction as the "
        "diagonal variant.",
        check=_check_scale_apply,
    )


XC_SCALE_APPLY = _scale_apply("xc_scale_apply", r_first=False)
XC_SCALE_APPLY_PRE = _scale_apply("xc_scale_apply_pre", r_first=True)

#: add of two same-state applications — residual stream around a scan.
XC_ADD_APPLYD = R(
    "xc_add_applyd",
    Op.make(
        "add", Op.make("applyd", "f", "h"), Op.make("applyd", "g", "h")
    ),
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make(
                "add", Op.make("affd_a", "f"), Op.make("affd_a", "g")
            ),
            Op.make(
                "add", Op.make("affd_b", "f"), Op.make("affd_b", "g")
            ),
        ),
        "h",
    ),
    law="Same-state affine applications add componentwise — residual "
    "addition around a scan stays in the carrier.",
    check=_check_add_applyd,
)

XC_ADD_APPLY = R(
    "xc_add_apply",
    Op.make(
        "add", Op.make("apply", "f", "h"), Op.make("apply", "g", "h")
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make(
                "add", Op.make("aff_A", "f"), Op.make("aff_A", "g")
            ),
            Op.make(
                "add", Op.make("aff_b", "f"), Op.make("aff_b", "g")
            ),
        ),
        "h",
    ),
    law="Dense analog of xc_add_applyd.",
    check=_check_add_apply,
)


# ---------------------------------------------------------------------------
#  Slice (chunk/split) commutes through a carrier application
# ---------------------------------------------------------------------------


def _chunk_applyd(name, op, extra_check):
    lhs = (
        Op.make(
            "chunk",
            Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
            chunks="N",
            dim="D",
            index="I",
        )
        if op == "chunk"
        else Op.make(
            "split",
            Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
            sizes="SZ",
            dim="D",
            index="I",
        )
    )
    inner = (
        (
            Op.make("chunk", "a", chunks="N", dim="D", index="I"),
            Op.make("chunk", "b", chunks="N", dim="D", index="I"),
        )
        if op == "chunk"
        else (
            Op.make("split", "a", sizes="SZ", dim="D", index="I"),
            Op.make("split", "b", sizes="SZ", dim="D", index="I"),
        )
    )
    return R(
        name,
        lhs,
        Op.make("applyd", Op.make("aff_diag", inner[0], inner[1]), "h"),
        law=f"{op} distributes over a diagonal-affine application on "
        "any non-feature axis — block i of a scanned value stream "
        "is applyd of the block-sliced map.",
        check=extra_check,
    )


XC_CHUNK_APPLYD = _chunk_applyd(
    "xc_chunk_applyd", "chunk", _check_chunk_applyd_ok
)
XC_SPLIT_APPLYD = _chunk_applyd(
    "xc_split_applyd", "split", _check_split_applyd
)


def _chunk_apply(name, op, check):
    """chunk(apply(aff(A,c),h)) → apply(aff(chunk A, chunk c),h): the
    slice axis lives in c's coordinates; A carries the same leading
    axes plus one extra LAST (the map's input dim), so A's slice dim is
    the same normalized index — derived as DA = D % rank(c)."""
    base = Op.make("apply", Op.make("aff", "A", "c"), "h")
    if op == "chunk":
        lhs = Op.make("chunk", base, chunks="N", dim="D", index="I")
        ia = Op.make("chunk", "A", chunks="N", dim="DA", index="I")
        ib = Op.make("chunk", "c", chunks="N", dim="D", index="I")
    else:
        lhs = Op.make("split", base, sizes="SZ", dim="D", index="I")
        ia = Op.make("split", "A", sizes="SZ", dim="DA", index="I")
        ib = Op.make("split", "c", sizes="SZ", dim="D", index="I")
    return R(
        name,
        lhs,
        Op.make("apply", Op.make("aff", ia, ib), "h"),
        law=f"{op} distributes over a dense-affine application — sound "
        "on EVERY axis of c (the map's input axis is last, never "
        "sliced).",
        check=check,
        derive=_derive_chunk_apply,
    )


XC_CHUNK_APPLY = _chunk_apply(
    "xc_chunk_apply", "chunk", _check_chunk_apply_ok
)
XC_SPLIT_APPLY = _chunk_apply(
    "xc_split_apply", "split", _check_split_apply
)


# ---------------------------------------------------------------------------
#  View ops (reshape/transpose) commute through a carrier application
# ---------------------------------------------------------------------------
#
#  A carrier application's VALUE is the map's output: apply(aff(A,b),h)
#  is A@h+b with A (…, out axes …, i) — the map's input axis is LAST,
#  so every value axis is a leading axis of A.  A view op that only
#  relabels/permutates VALUE axes therefore sinks through the
#  application and lands on the map unchanged in kind:
#
#      reshape(apply(aff(A,b),h), S)
#          = apply(aff(reshape(A, S+(i,)), reshape(b, S)), h)
#      transpose(apply(aff(A,b),h), d1, d2)
#          = apply(aff(transpose(A, d1', d2'),
#                      transpose(b, d1', d2')), h)   (d' = dims mod value rank)
#
#  EXACT for the dense carrier on ANY well-typed view: flat order is
#  preserved, the contracted axis stays last, so (A@h+b).view equals
#  (A.view)@h + b.view pointwise.  THIS IS THE MHA FORM — a natural
#  ``wv(scan).view(T,nh,hd).transpose(0,1)`` leaves the affine member
#  one view below the om leaf's value class; these two laws surface it.
#
#  The diagonal carrier is more delicate: ``applyd``'s h broadcasts
#  onto the value's LAST axis, which is the feature axis — a view may
#  not touch it:
#
#      transpose(applyd(aff_diag(a,b),h), d1, d2)  — only when neither
#          dim is the last (feature) axis; h is NOT permuted.
#      reshape(applyd(aff_diag(a,b),h), S)        — only when the
#          resolved S[-1] stays the feature dim d (head-PACKING views
#          like (T,d)->(T,1,d) pass; head-SPLITTING (T,d)->(T,nh,hd)
#          does NOT — d splits across axes and h can no longer
#          broadcast; that case must go through the dense promotion).
#
#  LIMITATIONS (vetoed, documented):
#    * b must be exactly value-shaped (b == A[:-1] / b == a).  A
#      broadcast-smaller b would need broadcast-then-view on the
#      coefficient — the IR has broadcast, but the fused form is not
#      minted; rare in practice (projections produce full-shaped b).
#    * No view may touch the map's INPUT axis (it is never a value
#      axis for apply; for applyd the last-axis restriction above).
#    * Non-sound directions are not minted: e.g. transpose of the
#      feature axis under applyd is a DIFFERENT affine map (h would
#      have to permute too), not expressible here.


def _numel_of(s) -> int:
    n = 1
    for d in s:
        n *= d
    return n


def _resolve_view_shape(S, numel_in: int):
    """Concrete resolution of a reshape ``shape`` attr against the
    input's numel — the same convention ``cost._shape_of`` uses (one
    literal -1 may appear in exported graphs and is inferred).
    Returns the resolved tuple, or None when the shape is ill-formed
    or numel-inconsistent (then the view — and the law — is not
    well-typed)."""
    if not isinstance(S, (tuple, list)) or len(S) == 0:
        return None
    if not all(isinstance(d, int) and (d == -1 or d > 0) for d in S):
        return None
    negs = sum(1 for d in S if d == -1)
    if negs > 1:
        return None
    known = 1
    for d in S:
        if d != -1:
            known *= d
    if negs:
        if known <= 0 or numel_in % known:
            return None
        return tuple(
            numel_in // known if d == -1 else int(d) for d in S
        )
    if known != numel_in:
        return None
    return tuple(int(d) for d in S)


def _view_dims(bound: dict, rank: int):
    """The transpose's two dims normalised mod *rank* — the value's
    rank, so the same numbers also index the map's leading axes."""
    d1, d2 = bound.get("$attr:D1"), bound.get("$attr:D2")
    if not (isinstance(d1, int) and isinstance(d2, int)):
        return None
    return d1 % rank, d2 % rank


def _apply_value_shapes(bound: dict):
    """Shared contract for the dense view laws: A (…, value, i) with
    b exactly value-shaped (b == A[:-1]) and h a matching (i,) vector.
    Returns A's shape, or None."""
    A, b, h = (_shape(bound.get(k)) for k in ("A", "b", "h"))
    if not all(_concrete(s) for s in (A, b, h)):
        return None
    if not (
        len(A) >= 2 and b == A[:-1] and len(h) == 1 and h[-1] == A[-1]
    ):
        return None
    return A


def _applyd_value_shapes(bound: dict):
    """Same for the diagonal carrier: a == b value-shaped, h (d,)
    matching the feature (last) axis."""
    a, b, h = (_shape(bound.get(k)) for k in ("a", "b", "h"))
    if not all(_concrete(s) for s in (a, b, h)):
        return None
    if not (a == b and len(a) >= 1 and len(h) == 1 and h[-1] == a[-1]):
        return None
    return a


def _check_reshape_apply(bound: dict) -> bool:
    A = _apply_value_shapes(bound)
    if A is None:
        return False
    return (
        _resolve_view_shape(bound.get("$attr:S"), _numel_of(A[:-1]))
        is not None
    )


def _derive_reshape_apply(bound: dict):
    """The map's input axis stays LAST: A reshapes to S+(i,).  A -1 in
    S is carried through verbatim — numel(A) = numel(v)·i so the
    inferred dim resolves identically on the map."""
    A = _apply_value_shapes(bound)
    S = bound.get("$attr:S")
    if A is None or _resolve_view_shape(S, _numel_of(A[:-1])) is None:
        return None
    return {"$attr:SA": tuple(S) + (A[-1],)}


def _check_transpose_apply(bound: dict) -> bool:
    A = _apply_value_shapes(bound)
    if A is None:
        return False
    return _view_dims(bound, len(A) - 1) is not None


def _derive_transpose_apply(bound: dict):
    """Transpose dims are bound on the VALUE (rank len(A)-1) but are
    stored on the map (rank len(A)) — normalise mod the value rank so
    the map's last (input) axis can never be permuted."""
    A = _apply_value_shapes(bound)
    if A is None:
        return None
    dd = _view_dims(bound, len(A) - 1)
    if dd is None:
        return None
    return {"$attr:DA1": dd[0], "$attr:DA2": dd[1]}


def _check_reshape_applyd(bound: dict) -> bool:
    """Diagonal reshape: sound iff the resolved view keeps the LAST
    axis at size d — numel preservation plus S[-1] == V[-1] means the
    flat-order relabel never mixes a feature component into a leading
    axis, so h still multiplies the right components."""
    V = _applyd_value_shapes(bound)
    if V is None:
        return False
    rs = _resolve_view_shape(bound.get("$attr:S"), _numel_of(V))
    return rs is not None and rs[-1] == V[-1]


def _check_transpose_applyd(bound: dict) -> bool:
    a = _applyd_value_shapes(bound)
    if a is None or len(a) < 2:
        return False
    dd = _view_dims(bound, len(a))
    # h broadcasts onto the last (feature) axis — a permutation that
    # moves it would permute the h-components themselves; vetoed.
    return dd is not None and dd[0] < len(a) - 1 and dd[1] < len(a) - 1


def _derive_transpose_applyd(bound: dict):
    a = _applyd_value_shapes(bound)
    if a is None:
        return None
    dd = _view_dims(bound, len(a))
    if dd is None:
        return None
    return {"$attr:DA1": dd[0], "$attr:DA2": dd[1]}


#: Dense forward laws — the view sinks into the map, the apply rises
#: to the value class (this is what makes the affine member visible
#: to ``_elem_affine_options`` under MHA's reshape/transpose head
#: split).
XC_RESHAPE_APPLY = R(
    "xc_reshape_apply",
    Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", "A", "b"), "h"),
        shape="S",
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("reshape", "A", shape="SA"),
            Op.make("reshape", "b", shape="S"),
        ),
        "h",
    ),
    law="A view on the VALUE axes of a dense-affine application sinks "
    "into the map: reshape(A@h+b, S) = reshape(A, S+(i,))@h + "
    "reshape(b, S) — the contracted axis is last and never "
    "relabelled, so this is exact for ANY well-typed S (the MHA "
    "head-split: (T,nh·hd) -> (T,nh,hd)).  Requires b == A[:-1] "
    "exactly (a broadcast-smaller b is vetoed — documented).",
    check=_check_reshape_apply,
    derive=_derive_reshape_apply,
)

XC_TRANSPOSE_APPLY = R(
    "xc_transpose_apply",
    Op.make(
        "transpose",
        Op.make("apply", Op.make("aff", "A", "b"), "h"),
        arg1="D1",
        arg2="D2",
    ),
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("transpose", "A", arg1="DA1", arg2="DA2"),
            Op.make("transpose", "b", arg1="DA1", arg2="DA2"),
        ),
        "h",
    ),
    law="transpose(A@h+b, d1, d2) = transpose(A, d1', d2')@h + "
    "transpose(b, d1', d2') — a permutation of the value axes "
    "permutes the map's output axes (dims are normalised mod the "
    "value rank so the stored map dims never touch the input "
    "axis).  Exact — the MHA (T,nh,hd)->(nh,T,hd) head swap.",
    check=_check_transpose_apply,
    derive=_derive_transpose_apply,
)

#: Diagonal forward laws — restricted to views that keep the feature
#: axis LAST (h is not permuted).
XC_RESHAPE_APPLYD = R(
    "xc_reshape_applyd",
    Op.make(
        "reshape",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        shape="S",
    ),
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("reshape", "a", shape="S"),
            Op.make("reshape", "b", shape="S"),
        ),
        "h",
    ),
    law="reshape(a⊙h+b, S) = reshape(a,S)⊙h + reshape(b,S) — sound "
    "ONLY when the resolved S[-1] stays the feature dim d: then "
    "the flat-order relabel never mixes feature components into "
    "leading axes and h still broadcasts correctly.  Head-PACKING "
    "views pass; head-SPLITTING ones (d -> nh×hd) are vetoed — "
    "use the dense promotion for those.",
    check=_check_reshape_applyd,
)

XC_TRANSPOSE_APPLYD = R(
    "xc_transpose_applyd",
    Op.make(
        "transpose",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        arg1="D1",
        arg2="D2",
    ),
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("transpose", "a", arg1="DA1", arg2="DA2"),
            Op.make("transpose", "b", arg1="DA1", arg2="DA2"),
        ),
        "h",
    ),
    law="transpose(a⊙h+b, d1, d2) = transpose(a,d1',d2')⊙h + "
    "transpose(b,d1',d2') — the per-element diagonal map commutes "
    "with any permutation that leaves the feature axis last "
    "(h is NOT permuted; a transpose touching it is vetoed).",
    check=_check_transpose_applyd,
    derive=_derive_transpose_applyd,
)


# --- reverse forms: the pushed-through member re-fuses ---------------


def _transpose_apply_rev_dims(bound: dict):
    """For the dense reverse: the dims stored on the map's transpose
    (rank len(A)) and on b's transpose (rank len(b)) must agree as
    VALUE axes — i.e. normalise to the same pair, never touching A's
    last (input) axis.  Returns the value-rank dims, or None."""
    A = _apply_value_shapes(bound)
    if A is None:
        return None
    n1, n = len(A), len(A) - 1
    p1, p2 = bound.get("$attr:P1"), bound.get("$attr:P2")
    q1, q2 = bound.get("$attr:Q1"), bound.get("$attr:Q2")
    if not all(isinstance(d, int) for d in (p1, p2, q1, q2)):
        return None
    d1, d2 = p1 % n1, p2 % n1
    if d1 >= n or d2 >= n:  # permutes the map's input axis
        return None
    if {d1, d2} != {q1 % n, q2 % n}:  # A and b permute different axes
        # (transpose(x,a,b) == transpose(x,b,a) — unordered compare)
        return None
    return d1, d2


def _check_transpose_apply_rev(bound: dict) -> bool:
    return _transpose_apply_rev_dims(bound) is not None


def _derive_transpose_apply_rev(bound: dict):
    dd = _transpose_apply_rev_dims(bound)
    if dd is None:
        return None
    return {"$attr:RD1": dd[0], "$attr:RD2": dd[1]}


XC_TRANSPOSE_APPLY_REV = R(
    "xc_transpose_apply_rev",
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("transpose", "A", arg1="P1", arg2="P2"),
            Op.make("transpose", "b", arg1="Q1", arg2="Q2"),
        ),
        "h",
    ),
    Op.make(
        "transpose",
        Op.make("apply", Op.make("aff", "A", "b"), "h"),
        arg1="RD1",
        arg2="RD2",
    ),
    law="reverse of xc_transpose_apply — the pushed-through form "
    "re-fuses when both transposes agree on the same value axes.",
    check=_check_transpose_apply_rev,
    derive=_derive_transpose_apply_rev,
)


def _transpose_applyd_rev_dims(bound: dict):
    a = _applyd_value_shapes(bound)
    if a is None or len(a) < 2:
        return None
    n = len(a)
    p1, p2 = bound.get("$attr:P1"), bound.get("$attr:P2")
    q1, q2 = bound.get("$attr:Q1"), bound.get("$attr:Q2")
    if not all(isinstance(d, int) for d in (p1, p2, q1, q2)):
        return None
    d1, d2 = p1 % n, p2 % n
    if {d1, d2} != {q1 % n, q2 % n}:
        return None
    if d1 >= n - 1 or d2 >= n - 1:  # feature axis moved — unsound
        return None
    return d1, d2


def _check_transpose_applyd_rev(bound: dict) -> bool:
    return _transpose_applyd_rev_dims(bound) is not None


def _derive_transpose_applyd_rev(bound: dict):
    dd = _transpose_applyd_rev_dims(bound)
    if dd is None:
        return None
    return {"$attr:RD1": dd[0], "$attr:RD2": dd[1]}


XC_TRANSPOSE_APPLYD_REV = R(
    "xc_transpose_applyd_rev",
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("transpose", "a", arg1="P1", arg2="P2"),
            Op.make("transpose", "b", arg1="Q1", arg2="Q2"),
        ),
        "h",
    ),
    Op.make(
        "transpose",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        arg1="RD1",
        arg2="RD2",
    ),
    law="reverse of xc_transpose_applyd — same feature-axis "
    "restriction: the stored dims must agree and stay off the "
    "last axis.",
    check=_check_transpose_applyd_rev,
    derive=_derive_transpose_applyd_rev,
)


def _check_reshape_apply_rev(bound: dict) -> bool:
    """apply(aff(reshape(A,SA), reshape(b,SB)), h) refolds to
    reshape(apply(aff(A,b),h), SB) — requires SA == SB+(i,) literally
    (the spelling the forward law mints; a semantically-equal but
    differently-spelled SA is conservatively vetoed), plus the usual
    map contract and a well-typed SB."""
    A = _apply_value_shapes(bound)
    if A is None:
        return False
    SA, SB = bound.get("$attr:SA"), bound.get("$attr:SB")
    if not (
        isinstance(SA, (tuple, list)) and isinstance(SB, (tuple, list))
    ):
        return False
    if len(SA) != len(SB) + 1 or tuple(SA[:-1]) != tuple(SB):
        return False
    if SA[-1] != A[-1]:
        return False
    return _resolve_view_shape(SB, _numel_of(A[:-1])) is not None


XC_RESHAPE_APPLY_REV = R(
    "xc_reshape_apply_rev",
    Op.make(
        "apply",
        Op.make(
            "aff",
            Op.make("reshape", "A", shape="SA"),
            Op.make("reshape", "b", shape="SB"),
        ),
        "h",
    ),
    Op.make(
        "reshape",
        Op.make("apply", Op.make("aff", "A", "b"), "h"),
        shape="SB",
    ),
    law="reverse of xc_reshape_apply — only matches when the map's "
    "reshape is literally S+(i,), the spelling the forward law "
    "produces.",
    check=_check_reshape_apply_rev,
)


def _check_reshape_applyd_rev(bound: dict) -> bool:
    """Diagonal reverse: both coefficients must carry the same shape
    attr, keeping the feature axis last."""
    a = _applyd_value_shapes(bound)
    if a is None:
        return False
    SA, SB = bound.get("$attr:SA"), bound.get("$attr:SB")
    if not (
        isinstance(SA, (tuple, list))
        and isinstance(SB, (tuple, list))
        and tuple(SA) == tuple(SB)
    ):
        return False
    rs = _resolve_view_shape(SA, _numel_of(a))
    return rs is not None and rs[-1] == a[-1]


XC_RESHAPE_APPLYD_REV = R(
    "xc_reshape_applyd_rev",
    Op.make(
        "applyd",
        Op.make(
            "aff_diag",
            Op.make("reshape", "a", shape="SA"),
            Op.make("reshape", "b", shape="SB"),
        ),
        "h",
    ),
    Op.make(
        "reshape",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        shape="SA",
    ),
    law="reverse of xc_reshape_applyd — same resolved-S[-1]==d "
    "restriction.",
    check=_check_reshape_applyd_rev,
)


# ---------------------------------------------------------------------------
#  The heterogeneous om element — scan folded INSIDE the om leaf
# ---------------------------------------------------------------------------

XC_OM_ELEM_AFFD = R(
    "xc_om_elem_affd",
    Op.make(
        "om_elem",
        "s",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
    ),
    Op.make("om_elem_affd", "s", "a", "b", "h"),
    law="The om numerator e@v is a matmul in v — the readout law: "
    "om_elem(s, a⊙h+b) = (m, l, (e@a)⊙h + e@b).  An om leaf whose "
    "value block is a scan image folds the scan inside the "
    "carrier; the result is an ordinary om triple and composes "
    "freely under om_compose.",
    check=_check_om_elem_affd,
)

XC_OM_ELEM_AFFD_REV = R(
    "xc_om_elem_affd_rev",
    Op.make("om_elem_affd", "s", "a", "b", "h"),
    Op.make(
        "om_elem",
        "s",
        Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
    ),
    law="unfold the fused element",
    check=_check_om_elem_affd,
)

XC_OM_ELEM_AFF = R(
    "xc_om_elem_aff",
    Op.make(
        "om_elem", "s", Op.make("apply", Op.make("aff", "A", "b"), "h")
    ),
    Op.make("om_elem_aff", "s", "A", "b", "h"),
    law="Dense counterpart of xc_om_elem_affd: v = A@h+b with per-key "
    "A (K,d,i); numerator = reshape(e@A)@h + e@b.",
    check=_check_om_elem_aff,
)

XC_OM_ELEM_AFF_REV = R(
    "xc_om_elem_aff_rev",
    Op.make("om_elem_aff", "s", "A", "b", "h"),
    Op.make(
        "om_elem", "s", Op.make("apply", Op.make("aff", "A", "b"), "h")
    ),
    law="unfold",
    check=_check_om_elem_aff,
)


# ---------------------------------------------------------------------------
#  The deferred carrier — omd: attention output stays affine in h
# ---------------------------------------------------------------------------

XC_OMD_LIFT = R(
    "xc_omd_lift",
    Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            "s",
            Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        ),
    ),
    Op.make("omd_apply", Op.make("omd_elem", "s", "a", "b"), "h"),
    law="Single-block attention over diagonal-affine values IS a "
    "diagonal-affine map of the shared state: the softmax data "
    "becomes the coefficient.  (m,l,e@a,e@b) defers h to the "
    "apply.",
    check=_check_omd_lift,
)

XC_OMD_UNLIFT = R(
    "xc_omd_unlift",
    Op.make("omd_apply", Op.make("omd_elem", "s", "a", "b"), "h"),
    Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            "s",
            Op.make("applyd", Op.make("aff_diag", "a", "b"), "h"),
        ),
    ),
    law="unfold the deferred carrier",
    check=_check_omd_lift,
)

XC_OMD_LIFT_DENSE = R(
    "xc_omd_lift_dense",
    Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            "s",
            Op.make("apply", Op.make("aff", "A", "b"), "h"),
        ),
    ),
    Op.make("omd_applym", Op.make("omd_elem", "s", "A", "b"), "h"),
    law="Dense-fiber variant: the numerator pair (e@A : (…,Tq,d,i), "
    "e@b) defers h; omd_applym contracts the last axis.",
    check=_check_omd_lift_dense,
)

XC_OMD_UNLIFT_DENSE = R(
    "xc_omd_unlift_dense",
    Op.make("omd_applym", Op.make("omd_elem", "s", "A", "b"), "h"),
    Op.make(
        "om_apply",
        Op.make(
            "om_elem",
            "s",
            Op.make("apply", Op.make("aff", "A", "b"), "h"),
        ),
    ),
    law="unfold",
    check=_check_omd_lift_dense,
)

XC_OMD_PAIR_LIFT = R(
    "xc_omd_pair_lift",
    Op.make(
        "om_apply",
        Op.make(
            "om_compose",
            Op.make(
                "om_elem",
                "s1",
                Op.make("applyd", Op.make("aff_diag", "a1", "b1"), "h"),
            ),
            Op.make(
                "om_elem",
                "s2",
                Op.make("applyd", Op.make("aff_diag", "a2", "b2"), "h"),
            ),
        ),
    ),
    Op.make(
        "omd_apply",
        Op.make(
            "omd_compose",
            Op.make("omd_elem", "s1", "a1", "b1"),
            Op.make("omd_elem", "s2", "a2", "b2"),
        ),
        "h",
    ),
    law="Two-block case of the deferred lift — the matcher enforces "
    "the SHARED h across leaves.  Deeper trees are non-local (the "
    "shared-h condition is global): see omd_tree_lift.",
    check=_check_omd_pair,
)


def _omd_split(name, ak):
    return R(
        name,
        Op.make(
            "omd_elem",
            Op.make("concat", "s1", "s2", **{ak: "SD"}),
            Op.make("concat", "a1", "a2", **{ak: "AD"}),
            Op.make("concat", "b1", "b2", **{ak: "BD"}),
        ),
        Op.make(
            "omd_compose",
            Op.make("omd_elem", "s1", "a1", "b1"),
            Op.make("omd_elem", "s2", "a2", "b2"),
        ),
        law="The om homomorphism holds on the deferred carrier: "
        "omd_elem(cat s, cat a, cat b) = omd_elem ⊕ omd_elem — "
        "chunked attention over scanned values decomposes while "
        "staying affine in h.",
        check=_check_omd_split,
    )


XC_OMD_SPLIT = _omd_split("xc_omd_split", "dim")
XC_OMD_SPLIT_ARG1 = _omd_split("xc_omd_split_arg1", "arg1")


#: The whole cross-carrier law set.
XC_LAWS: list[Rewrite] = [
    XC_MATMUL_APPLYD_ROWS,
    XC_MATMUL_APPLYD_ROWS_REV,
    XC_MATMUL_APPLYD_VEC,
    XC_MATMUL_APPLYD_VEC_REV,
    XC_MATMUL_APPLY,
    XC_MATMUL_APPLY_REV,
    XC_LINEAR_APPLYD,
    XC_LINEAR_APPLYD_REV,
    XC_LINEAR_APPLY,
    XC_LINEAR_APPLY_REV,
    XC_SCALE_APPLYD,
    XC_SCALE_APPLYD_PRE,
    XC_SCALE_APPLY,
    XC_SCALE_APPLY_PRE,
    XC_ADD_APPLYD,
    XC_ADD_APPLY,
    XC_CHUNK_APPLYD,
    XC_SPLIT_APPLYD,
    XC_CHUNK_APPLY,
    XC_SPLIT_APPLY,
    XC_RESHAPE_APPLY,
    XC_TRANSPOSE_APPLY,
    XC_RESHAPE_APPLYD,
    XC_TRANSPOSE_APPLYD,
    XC_RESHAPE_APPLY_REV,
    XC_TRANSPOSE_APPLY_REV,
    XC_RESHAPE_APPLYD_REV,
    XC_TRANSPOSE_APPLYD_REV,
    XC_OM_ELEM_AFFD,
    XC_OM_ELEM_AFFD_REV,
    XC_OM_ELEM_AFF,
    XC_OM_ELEM_AFF_REV,
    XC_OMD_LIFT,
    XC_OMD_UNLIFT,
    XC_OMD_LIFT_DENSE,
    XC_OMD_UNLIFT_DENSE,
    XC_OMD_PAIR_LIFT,
    XC_OMD_SPLIT,
    XC_OMD_SPLIT_ARG1,
]


# ---------------------------------------------------------------------------
#  Pass 1 — the sequence a scan emits is ONE carrier application
# ---------------------------------------------------------------------------
#
#  ``stack(applyd(f_0,h), …, applyd(f_{n-1},h))`` — the SSM's emitted
#  sequence — equals ``applyd(aff_diag(stack a_i, stack b_i), h)`` when
#  every row shares the initial-state e-class h.  Stack is variadic so
#  this is a non-local offer (like pair_shared_input_linears), not a
#  lhs→rhs rule.  The components come out through the affd_a/affd_b
#  projections, which work for ANY map term (leaf or compose tree).
# ---------------------------------------------------------------------------


def _stack_dim(attrs: dict) -> int:
    d = attrs.get("dim", attrs.get("arg1", 0))
    return d if isinstance(d, int) else 0


def _offer_witness(
    eg: EGraph,
    cid: int,
    offered: Any,
    offered_eid: int,
    provenance: str,
):
    """Pointwise Rewrite certifying a non-local offer — the same
    convention as ``trace_lift._lift_witness``: lhs is the oldest
    member of the class, rhs the offered term; the merge replays as a
    named rule step in certificates."""
    src = eg._oldest_term(eg.find(cid))
    if src is None:
        return None
    return Rewrite(
        name=f"{provenance}#{offered_eid}",
        lhs=src,
        rhs=offered,
        law=(
            "pointwise witness for a non-local offer: stack of "
            "same-state carrier applications is one application of "
            "the stacked map"
        ),
    )


def _map_out_shape(f_shape, h_shape, kind: str):
    """Shape of ``apply_op(f, h)``'s VALUE given the map's shape.

    diag:  v = a⊙h+b  → broadcast(f_shape, h_shape)
    dense: v = A@h+b  → A[:-1] (h contracts the last axis)."""
    if kind == "diag":
        return _bc(f_shape, h_shape)
    if (
        isinstance(f_shape, tuple)
        and len(f_shape) >= 2
        and f_shape[-1] == f_shape[-2]
    ):
        return f_shape[:-1]
    return None


def _gather_stack(
    eg: EGraph,
    apply_op: str,
    proj_a: str,
    proj_b: str,
    pack_op: str,
    kind: str,
    witness: bool,
    provenance: str,
) -> list[dict]:
    """Shared engine for gather_applyd_stack / gather_apply_stack.

    Finds ``stack`` enodes whose children ALL expose an ``apply_op``
    member over one shared state e-class h, then offers
    ``apply_op(pack_op(stack(proj_a f_i), stack(proj_b f_i)), h)``.
    Soundness guards: children must share a concrete shape S, h must be
    a (d,) vector with d == S[-1], and the stack axis must not be the
    new last axis (h broadcasts along the feature dim).
    """
    offers: list[dict] = []
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "stack" or not node.children:
                continue
            kids = [eg.find(ch) for ch in node.children]
            # per-child apply members: {h_eid: [f_eids]}
            opts = []
            for ch in kids:
                seen_h: dict[int, list[int]] = {}
                for n in eg.get_class(ch).nodes:
                    if n.op == apply_op and len(n.children) == 2:
                        seen_h.setdefault(
                            eg.find(n.children[1]), []
                        ).append(eg.find(n.children[0]))
                opts.append(seen_h)
            common = set(opts[0])
            for o in opts[1:]:
                common &= set(o)
            D = _stack_dim(dict(node.attrs))
            for h_eid in sorted(common):
                ht = eg.any_term(h_eid)
                hs = _shape(ht) if ht is not None else None
                if not (_concrete(hs) and len(hs) == 1):
                    continue
                # per child: {true value shape: chosen map eid}.
                # applyd's reported shape is h's — the TRUE shape is
                # broadcast(a_i, h) (diag) or A_i[:-1] (dense).
                vs_map: list[dict] = []
                for o in opts:
                    m: dict = {}
                    for f in sorted(o[h_eid]):
                        ft = eg.any_term(f)
                        fs = _shape(ft) if ft is not None else None
                        vs = _map_out_shape(fs, hs, kind)
                        if vs is None or not _concrete(vs) or vs in m:
                            continue
                        # require the map's shape to align exactly —
                        # no broadcast-shrinkage inside the pack:
                        # diag: fs == vs (a_i is vs-shaped);
                        # dense: fs == vs + (i,) and h fills axis -1.
                        if (kind == "diag" and fs != vs) or (
                            kind == "dense"
                            and fs != tuple(vs) + (hs[0],)
                        ):
                            continue
                        m[vs] = f
                    vs_map.append(m)
                shared = set(vs_map[0])
                for m in vs_map[1:]:
                    shared &= set(m)
                for S in sorted(shared):
                    if S[-1] != hs[0]:
                        continue
                    d_norm = D % (len(S) + 1)
                    if d_norm >= len(S):
                        continue  # stacking onto the feature axis
                    f_ids = [m[S] for m in vs_map]
                    attrs = dict(node.attrs)
                    sa = eg.add_enode(
                        "stack",
                        tuple(
                            eg.add_enode(proj_a, (f,)) for f in f_ids
                        ),
                        attrs,
                    )
                    sb = eg.add_enode(
                        "stack",
                        tuple(
                            eg.add_enode(proj_b, (f,)) for f in f_ids
                        ),
                        attrs,
                    )
                    fmap = eg.add_enode(pack_op, (sa, sb))
                    out = eg.add_enode(apply_op, (fmap, h_eid))
                    offered_term = eg.any_term(out)
                    w = (
                        _offer_witness(
                            eg, c, offered_term, out, provenance
                        )
                        if witness and offered_term is not None
                        else None
                    )
                    eg.union(
                        c,
                        out,
                        witness=w,
                        note=(
                            f"{provenance}: stack of "
                            f"{len(kids)} {apply_op} members "
                            "over shared h"
                        ),
                    )
                    offers.append(
                        {
                            "stack_eid": c,
                            "h_eid": h_eid,
                            "out_eid": out,
                            "term": offered_term,
                        }
                    )
    return offers


def gather_applyd_stack(
    eg: EGraph,
    *,
    witness: bool = True,
    provenance: str = "xc_stack_diag",
) -> list:
    """Offer ``applyd(aff_diag(stack affd_a f_i, stack affd_b f_i), h)``
    for every ``stack`` enode whose children are applyd applications
    over a shared initial state — the scan sequence as ONE map."""
    return _gather_stack(
        eg,
        "applyd",
        "affd_a",
        "affd_b",
        "aff_diag",
        "diag",
        witness,
        provenance,
    )


def gather_apply_stack(
    eg: EGraph,
    *,
    witness: bool = True,
    provenance: str = "xc_stack_dense",
) -> list:
    """Dense analog: ``apply(aff(stack aff_A f_i, stack aff_b f_i), h)``."""
    return _gather_stack(
        eg,
        "apply",
        "aff_A",
        "aff_b",
        "aff",
        "dense",
        witness,
        provenance,
    )


# ---------------------------------------------------------------------------
#  Pass 2 — lift a whole om tree into the deferred omd carrier
# ---------------------------------------------------------------------------
#
#  ``om_apply(F)`` where F's class contains an om tree whose EVERY leaf
#  is ``om_elem(s_i, applyd(aff_diag(a_i,b_i),h))`` (or the dense
#  ``apply(aff(A_i,b_i),h)``) with the SAME h —
#  offers ``omd_apply(omd-tree, h)`` / ``omd_applym(omd-tree, h)``.
#  The shared-h condition spans the whole tree, so this is non-local.
# ---------------------------------------------------------------------------


def _elem_affine_options(eg: EGraph, v_cid: int) -> dict[int, list]:
    """For a value-block e-class: ``{h_eid: [(kind, a_eid, b_eid)]}`` —
    every applyd/apply member whose map exposes aff_diag/aff parts."""
    out: dict[int, list] = {}
    for n in eg.get_class(v_cid).nodes:
        if n.op == "applyd" and len(n.children) == 2:
            kind, pack = "diag", "aff_diag"
        elif n.op == "apply" and len(n.children) == 2:
            kind, pack = "dense", "aff"
        else:
            continue
        fcid, hcid = eg.find(n.children[0]), eg.find(n.children[1])
        for fn in eg.get_class(fcid).nodes:
            if fn.op == pack and len(fn.children) == 2:
                out.setdefault(hcid, []).append(
                    (
                        kind,
                        eg.find(fn.children[0]),
                        eg.find(fn.children[1]),
                    )
                )
    return out


def _omd_convert(
    eg: EGraph,
    cid: int,
    h_eid: int,
    kind: str,
    memo: dict,
    stack: frozenset,
):
    """Convert an om-carrier e-class to an omd-carrier eid, or None.

    Tries each member: om_elem leaves convert when the value class has
    an affine member of ``kind`` over ``h_eid``; om_compose members
    convert when both sides do.  ``om`` packaging leaves (concrete
    numerators) cannot convert."""
    cid = eg.find(cid)
    key = (cid, h_eid, kind)
    if key in memo:
        return memo[key]
    if cid in stack:
        return None
    stack = stack | {cid}
    res = None
    for n in sorted(
        eg.get_class(cid).nodes,
        key=lambda x: (x.op, x.children, repr(x.attrs)),
    ):
        if n.op == "om_elem" and len(n.children) == 2:
            opts = _elem_affine_options(eg, n.children[1]).get(h_eid)
            if opts:
                pick = next((o for o in opts if o[0] == kind), None)
                if pick is not None:
                    res = eg.add_enode(
                        "omd_elem",
                        (eg.find(n.children[0]), pick[1], pick[2]),
                    )
                    break
        elif n.op == "om_compose" and len(n.children) == 2:
            f = _omd_convert(
                eg, n.children[0], h_eid, kind, memo, stack
            )
            g = _omd_convert(
                eg, n.children[1], h_eid, kind, memo, stack
            )
            if f is not None and g is not None:
                res = eg.add_enode("omd_compose", (f, g))
                break
    memo[key] = res
    return res


def omd_tree_lift(
    eg: EGraph, *, witness: bool = True, provenance: str = "omd_lift"
) -> list[dict]:
    """Lift ``om_apply`` members whose carrier tree is entirely
    scan-valued into the deferred ``omd`` carrier.

    For each ``om_apply`` enode, collect the h-options of every
    reachable ``om_elem`` leaf; for each candidate h and each fiber
    kind (diag first, then dense) try a whole-tree conversion; on
    success offer ``omd_apply(tree, h)`` / ``omd_applym(tree, h)``
    into the om_apply's class.  Returns a list of offer dicts.
    """
    offers: list[dict] = []
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "om_apply" or len(node.children) != 1:
                continue
            carrier = eg.find(node.children[0])
            # candidate h's: union of leaf options under the carrier
            h_cands: set[int] = set()

            def _collect(ccid: int, seen: frozenset = frozenset()):
                ccid = eg.find(ccid)
                if ccid in seen:
                    return
                seen = seen | {ccid}
                for n in eg.get_class(ccid).nodes:
                    if n.op == "om_elem" and len(n.children) == 2:
                        h_cands.update(
                            _elem_affine_options(eg, n.children[1])
                        )
                    elif n.op == "om_compose" and len(n.children) == 2:
                        _collect(n.children[0], seen)
                        _collect(n.children[1], seen)

            _collect(carrier)
            memo: dict = {}
            for h_eid in sorted(h_cands):
                for kind, apply_op in (
                    ("diag", "omd_apply"),
                    ("dense", "omd_applym"),
                ):
                    root = _omd_convert(
                        eg, carrier, h_eid, kind, memo, frozenset()
                    )
                    if root is None:
                        continue
                    out = eg.add_enode(apply_op, (root, h_eid))
                    offered = eg.any_term(out)
                    w = (
                        _offer_witness(eg, c, offered, out, provenance)
                        if witness and offered is not None
                        else None
                    )
                    eg.union(
                        c,
                        out,
                        witness=w,
                        note=(
                            f"{provenance}: om tree over "
                            f"{kind}-affine values, shared h"
                        ),
                    )
                    offers.append(
                        {
                            "om_apply_eid": c,
                            "h_eid": h_eid,
                            "out_eid": out,
                            "term": offered,
                            "kind": kind,
                        }
                    )
                    break  # one kind per h is enough
    return offers
