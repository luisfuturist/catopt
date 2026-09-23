"""Online-softmax monoid — the nonlinear analogue of the scan carrier.

The affine-map monoid (``SCAN_LAWS`` in :mod:`catopt.rules`) showed that
lifting a recurrence into a monoid carrier lets plain associativity
discover parallel bracketings.  The same trick works for the softmax
recurrence — FlashAttention's running (max, sum, numerator) triple.

CARRIER
    ``om(m, l, a)`` is a triple value: running row-max ``m``, running
    denominator ``l = Σ exp(s − m)``, running numerator
    ``a = Σ exp(s − m) · v``.

COMPOSITION (associative *and* commutative)
    ``(m1,l1,a1) ⊕ (m2,l2,a2)``
        = (mx, l1·e^(m1−mx) + l2·e^(m2−mx), a1·e^(m1−mx) + a2·e^(m2−mx))
    with mx = max(m1, m2).  We deliberately do NOT add a commutativity
    rule — comm is the explosive law (permutation space), exactly as in
    SCAN_LAWS; OM_ASSOC/OM_ASSOC_REV suffice for reassociation.

ELEMENT
    ``om_elem(s, v)`` for score block ``s (…,T,K)`` and value block
    ``v (…,K,d)``: ``(rowmax(s), rowsum(exp(s−rowmax)), exp(s−rowmax)@v)``.

HOMOMORPHISM
    ``elem(cat(s1,s2), cat(v1,v2)) = elem(s1,v1) ⊕ elem(s2,v2)`` —
    and ``softmax(s) @ v = a / l``.  This is the law tensor algebra
    cannot state: softmax does not distribute over concat, but its
    monoid carrier does.

``om_apply(f)`` evaluates a carrier back to tensor-land: ``a / l``
(unclamped — a fully-masked row yields NaN exactly like dense softmax).

The rules:
* ``OM_LIFT`` / ``OM_UNLIFT``   — enter / leave the carrier.
* ``OM_SPLIT`` / ``OM_MERGE``   — the homomorphism, both directions.
* ``OM_ASSOC`` / ``OM_ASSOC_REV`` — reassociating the combine gives the
  chunked/blocked attention schedules (FlashAttention tiling is a
  bracketing of this monoid product).
* ``CONCAT_BINARIZE_*``         — n-ary concat → nested binary (the IR's
  concat is variadic; the homomorphism is binary).
* ``MATMUL_T_CONCAT``           — ``q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T)``:
  matmul distributes over concat in the transposed operand, exposing
  per-block score tensors for OM_SPLIT.

Note on concat attrs: torch.export emits ``cat`` with the dim as the
positional ``arg1`` attribute, while rule-produced concats (and
hand-built terms, following rules.py convention) use ``dim``.  LHS
patterns are generated in both spellings; all RHS-produced concats use
``dim``.
"""

from typing import Any

from catopt.egraph import Rewrite
from catopt.ir import Op
from catopt.rules import R


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.cost import _shape_of as _so
    return _so(t)


def _dim_eq(a: Any, b: Any) -> bool:
    """cat-compatibility on a non-concatenated axis (None = unknown)."""
    return a is None or b is None or a == b


def _broadcast_ok(a, b) -> bool:
    from catopt.cost import _broadcast, _INVALID
    return _broadcast(a, b) is not _INVALID


# ---------------------------------------------------------------------------
#  Lift / unlift — enter and leave the carrier
# ---------------------------------------------------------------------------

def _check_om_lift(bound: dict) -> bool:
    """softmax must be over the scores' LAST dim (the key axis), and the
    matmul must contract s[...,K] with v[...,K,d].  A softmax over any
    other axis is well-typed but a *different* program — the check is
    load-bearing."""
    sd = bound.get("$attr:SD", -1)
    ss, vs = _shape_of(bound.get("s")), _shape_of(bound.get("v"))
    if not (isinstance(ss, tuple) and isinstance(vs, tuple)
            and len(ss) >= 2 and len(vs) >= 2):
        return False
    if not isinstance(sd, int) or sd % len(ss) != len(ss) - 1:
        return False
    return _dim_eq(ss[-1], vs[-2])


def _om_lift(name: str, attr_key: str | None) -> Rewrite:
    sm = (Op.make("softmax", "s", **{attr_key: "SD"})
          if attr_key is not None else Op.make("softmax", "s"))
    return R(
        name,
        Op.make("matmul", sm, "v"),
        Op.make("om_apply", Op.make("om_elem", "s", "v")),
        law="matmul(softmax(s), v) lifts into the online-softmax monoid: "
            "softmax does not distribute over concat, but elem does — "
            "the carrier (m, l, a) is what distributes.",
        check=_check_om_lift)


#: torch.export emits softmax(x, dim) with the dim in ``arg1``.
OM_LIFT = _om_lift("om_lift", "arg1")
#: Same law for the ``dim=`` kwarg spelling and a bare softmax(-1).
OM_LIFT_DIM = _om_lift("om_lift_dim", "dim")
OM_LIFT_PLAIN = _om_lift("om_lift_plain", None)

OM_UNLIFT = R(
    "om_unlift",
    Op.make("om_apply", Op.make("om_elem", "s", "v")),
    Op.make("matmul", Op.make("softmax", "s", arg1=-1), "v"),
    law="om_apply(elem(s,v)) unfolds back to dense softmax(s) @ v.",
)


# ---------------------------------------------------------------------------
#  The homomorphism — elem distributes over concat
# ---------------------------------------------------------------------------

def _chunks_compatible(s1, s2, v1, v2) -> bool:
    """Chunk pair i must contract s_i[...,K_i] with v_i[...,K_i,d]; the
    two chunks must be cat-compatible off the concatenated axis, and
    each chunk's batch dims must broadcast s_i against v_i."""
    ns, nv = len(s1), len(v1)
    if ns < 2 or nv < 2 or len(s2) != ns or len(v2) != nv:
        return False
    if not all(_dim_eq(s1[i], s2[i]) for i in range(ns - 1)):
        return False
    if not all(_dim_eq(v1[i], v2[i])
               for i in range(nv) if i != nv - 2):
        return False
    if not (_dim_eq(s1[-1], v1[nv - 2])
            and _dim_eq(s2[-1], v2[nv - 2])):
        return False
    return (_broadcast_ok(s1[:-2], v1[:-2])
            and _broadcast_ok(s2[:-2], v2[:-2]))


def _check_om_concat_dims(bound: dict) -> bool:
    """OM_SPLIT fires only when scores concat on the LAST dim (keys)
    and values concat on dim -2 (the same key axis, pre-contraction).
    A cat along any other axis is well-typed but WRONG."""
    sd, vd = bound.get("$attr:SD"), bound.get("$attr:VD")
    s1, s2 = _shape_of(bound.get("s1")), _shape_of(bound.get("s2"))
    v1, v2 = _shape_of(bound.get("v1")), _shape_of(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return False
    if not (isinstance(sd, int) and isinstance(vd, int)):
        return False
    if sd % len(s1) != len(s1) - 1:
        return False  # scores must concat along the key axis
    if vd % len(v1) != len(v1) - 2:
        return False  # values concat along the key axis (-2 before @)
    return _chunks_compatible(s1, s2, v1, v2)


def _om_split(name: str, attr_key: str) -> Rewrite:
    return R(
        name,
        Op.make("om_elem",
                Op.make("concat", "s1", "s2", **{attr_key: "SD"}),
                Op.make("concat", "v1", "v2", **{attr_key: "VD"})),
        Op.make("om_compose",
                Op.make("om_elem", "s1", "v1"),
                Op.make("om_elem", "s2", "v2")),
        law="Homomorphism: elem(cat(s1,s2), cat(v1,v2)) = "
            "elem(s1,v1) ⊕ elem(s2,v2) — the law tensor algebra cannot "
            "state (softmax does not distribute over concat).",
        check=_check_om_concat_dims)


OM_SPLIT = _om_split("om_split", "dim")
OM_SPLIT_ARG1 = _om_split("om_split_arg1", "arg1")


def _derive_om_concat_dims(bound: dict) -> dict | None:
    """Concat dims for the merged element, computed from bound shapes:
    scores join on their last dim, values on dim -2.  Vetoes (returns
    None) when the two chunks cannot form a well-typed cat."""
    s1, s2 = _shape_of(bound.get("s1")), _shape_of(bound.get("s2"))
    v1, v2 = _shape_of(bound.get("v1")), _shape_of(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return None
    if not _chunks_compatible(s1, s2, v1, v2):
        return None
    return {"$attr:SD": len(s1) - 1, "$attr:VD": len(v1) - 2}


OM_MERGE = R(
    "om_merge",
    Op.make("om_compose",
            Op.make("om_elem", "s1", "v1"),
            Op.make("om_elem", "s2", "v2")),
    Op.make("om_elem",
            Op.make("concat", "s1", "s2", dim="SD"),
            Op.make("concat", "v1", "v2", dim="VD")),
    law="Reverse homomorphism: two block elements merge into the "
        "element of the concatenated block — chunking is reversible.",
    derive=_derive_om_concat_dims)


# ---------------------------------------------------------------------------
#  Associativity — the monoid law that generates blocked schedules
# ---------------------------------------------------------------------------

OM_ASSOC = R(
    "om_assoc",
    Op.make("om_compose",
            Op.make("om_compose", "f", "g"), "h"),
    Op.make("om_compose", "f",
            Op.make("om_compose", "g", "h")),
    law="⊕ is associative: (f ⊕ g) ⊕ h = f ⊕ (g ⊕ h).  Every "
        "bracketing of the block product is a legal attention schedule.")

OM_ASSOC_REV = R(
    "om_assoc_rev",
    Op.make("om_compose", "f",
            Op.make("om_compose", "g", "h")),
    Op.make("om_compose",
            Op.make("om_compose", "f", "g"), "h"),
    law="⊕ is associative: f ⊕ (g ⊕ h) = (f ⊕ g) ⊕ h.")


# ---------------------------------------------------------------------------
#  concat is variadic in the IR; the homomorphism is binary.
# ---------------------------------------------------------------------------

def _concat_binarize(n: int, attr_key: str) -> Rewrite:
    xs = [f"x{i}" for i in range(n)]
    rhs = Op.make("concat", xs[0], xs[1], dim="D")
    for x in xs[2:]:
        rhs = Op.make("concat", rhs, x, dim="D")
    return R(
        f"concat_binarize_{n}_{attr_key}",
        Op.make("concat", *xs, **{attr_key: "D"}),
        rhs,
        law="n-ary concat is iterated binary concat — cat is a binary "
            "tensor product applied repeatedly.")


#: 3..6-ary concats, both attr spellings.  Structural only — needs no
#: shape check (binarisation preserves well-typedness identically).
CONCAT_BINARIZE: list[Rewrite] = [
    _concat_binarize(n, ak)
    for n in range(3, 7) for ak in ("dim", "arg1")
]


# ---------------------------------------------------------------------------
#  matmul distributes over concat in the transposed operand
# ---------------------------------------------------------------------------

def _check_matmul_t_concat(bound: dict) -> bool:
    """q @ cat(k1,k2,dim).T splits only when the key concat is on the
    SEQUENCE axis (dim -2 of k, i.e. keys) and the transpose is exactly
    .T on the last two dims — so that after the transpose the concat
    lands on the scores' last dim."""
    kd, t1, t2 = (bound.get("$attr:KD"), bound.get("$attr:T1"),
                  bound.get("$attr:T2"))
    k1, k2 = _shape_of(bound.get("k1")), _shape_of(bound.get("k2"))
    q = _shape_of(bound.get("q"))
    if not all(isinstance(x, int) for x in (kd, t1, t2)):
        return False
    if not all(isinstance(s, tuple) for s in (k1, k2, q)):
        return False
    nk = len(k1)
    if nk < 2 or len(k2) != nk or len(q) < 2:
        return False
    if kd % nk != nk - 2:
        return False  # keys concat on the sequence axis, not features
    if {t1 % nk, t2 % nk} != {nk - 2, nk - 1}:
        return False  # transpose must be exactly .T (last two dims)
    if not all(_dim_eq(k1[i], k2[i])
               for i in range(nk) if i != nk - 2):
        return False
    # q[...,d] contracts with k.T[...,d,K] — k's LAST dim is d.
    if not _dim_eq(q[-1], k1[-1]):
        return False
    return _broadcast_ok(q[:-2], k1[:-2])


def _derive_score_concat_dim(bound: dict) -> dict | None:
    """The concat dim through the transpose: after .T the key axis is
    last in k.T and lands on the scores' last dim, at index
    rank(out)-1 = max(rank q, rank k) - 1."""
    q, k1 = _shape_of(bound.get("q")), _shape_of(bound.get("k1"))
    if not (isinstance(q, tuple) and isinstance(k1, tuple)):
        return None
    return {"$attr:SD": max(len(q), len(k1)) - 1}


def _matmul_t_concat(name: str, attr_key: str) -> Rewrite:
    return R(
        name,
        Op.make("matmul", "q",
                Op.make("transpose",
                        Op.make("concat", "k1", "k2", **{attr_key: "KD"}),
                        arg1="T1", arg2="T2")),
        Op.make("concat",
                Op.make("matmul", "q",
                        Op.make("transpose", "k1",
                                arg1="T1", arg2="T2")),
                Op.make("matmul", "q",
                        Op.make("transpose", "k2",
                                arg1="T1", arg2="T2")),
                dim="SD"),
        law="matmul distributes over concat in the transposed operand: "
            "q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T) — the score blocks "
            "surface as one concat so OM_SPLIT can chunk the softmax.",
        check=_check_matmul_t_concat,
        derive=_derive_score_concat_dim)


MATMUL_T_CONCAT = _matmul_t_concat("matmul_t_concat", "dim")
MATMUL_T_CONCAT_ARG1 = _matmul_t_concat("matmul_t_concat_arg1", "arg1")


# ---------------------------------------------------------------------------
#  Law set
# ---------------------------------------------------------------------------

#: Minimal law set for chunked-attention discovery.  Deliberately
#: excludes a commutativity rule for om_compose — comm is the explosive
#: law, same as SCAN_LAWS; block order is preserved by the concat
#: structure itself.
OM_LAWS: list[Rewrite] = [
    OM_LIFT, OM_LIFT_DIM, OM_LIFT_PLAIN, OM_UNLIFT,
    OM_SPLIT, OM_SPLIT_ARG1, OM_MERGE,
    OM_ASSOC, OM_ASSOC_REV,
    *CONCAT_BINARIZE,
    MATMUL_T_CONCAT, MATMUL_T_CONCAT_ARG1,
]
