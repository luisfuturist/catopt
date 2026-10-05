"""Declarative side conditions for rewrite laws — pure data, no lambdas.

A law is a 2-cell: a term pair plus a side condition.  Historically the
condition was an opaque Python callable (``check=``), which is the one
thing keeping laws from being serializable data — patterns encode fine
(:mod:`catopt_core.rulecache` already round-trips ``Op`` trees), but a
``lambda`` does not.  This module is the small declarative DSL that
closes the gap: a condition is a nested tuple tree (lists after
``json.loads``), interpreted against the same ``bound`` environment the
``check`` hook sees — ``{metavar: resolved_term, "$attr:NAME": value}``.

Grammar
-------
::

    cond := bool
          | ("and", c1, ...) | ("or", c1, ...) | ("not", c)
          | PRED

    PRED := ("shaped", T)            # _shape_of(T) is a tuple
          | ("concrete", T)          # tuple, every dim an int
          | ("scalar", T)            # shape == ()
          | ("uniform", T)           # tuple, every dim == 1 (scalar ok)
          | ("ones-but-last", T)     # tuple, s[:-1] all == 1
          | ("rank", T, CMP, k)      # len(shape) CMP k
          | ("rank-eq", A, B)        # equal ranks, both shaped
          | ("shape-eq", A, B)       # sa == sb exactly (sa shaped;
                                     #   None dims compare equal)
          | ("shape-compat", A, B)   # same rank, None dims are wildcards
          | ("dim-eq", A, i, B, j)   # sa[i] == sb[j]  (None==None ok)
          | ("dim-compat", A, i, B, j)  # sa[i]/sb[j] None-wildcard eq
          | ("dim-eq-const", T, i, k)   # s[i] == k
          | ("dim-eq-attr", A, K, B, K2)  # sa[K]==sb[K2], Ks are attr metavars
          | ("bcast-into", T, U)     # _broadcast(s_T, s_U) == s_U
          | ("mm-shape-ok", A, B)    # _matmul_shape(sa, sb) not None
          | ("axes-last2", T, D0, D1)   # transpose pair = last-two swap
          | ("axes-distinct", T, D0, D1) # pair normalizes, d0 != d1
          | ("axes-eq", T, A0,A1,B0,B1)  # two distinct pairs, same set
          | ("axis", T, NAME, k)     # attr % rank == k % rank
          | ("op-in", T, ops)        # term is an Op with op in ops
          | ("leaf", T)              # term is not an Op
          | ("const", T)             # term is a Const leaf
          | ("term-eq", A, B)        # bound terms equal
          | ("const-num", T)         # t.value is int|float
          | ("const-cmp", T, CMP, v) # t.value CMP v (numeric)
          | ("attr-is", NAME, v)     # bound attr is v (identity)
          | ("attr-eq", NAME, v)     # bound attr == v
          | ("attr-in", NAME, vs)    # bound attr in vs
          | ("attr-type", NAME, K)   # isinstance per K:
                                     #   int|float|number|bool|str|tuple
          | ("attr-len", NAME, CMP, k)   # tuple attr, len CMP k
          | ("attr-cmp-dim", NAME, CMP, T, K)
                                     # $attr:NAME CMP shape(T)[$attr:K]
          | ("attr-eq-attr", A, B)   # bound attrs equal (both bound)
          | ("dim-mod", T, K, m, r)  # sa[K mod rank] % m == r
          | ("bcast-eq", A,B,C,D)    # bcast(sA,sB) == bcast(sC,sD)
          | ("ones-before", T, K)    # s[i]==1 for i < K mod (rank+1)
          | ("axes-noop", T, D0, D1) # transpose pair is a semantic
                                     #   no-op on T (same axis, or both
                                     #   swapped extents are 1)
          | ("flat-pair-unsq", T,K,G,S)  # flat-read pairing of T's dims
                                     #   agrees between broadcast-into-G
                                     #   and unsq-into-S (the wr guard)
          | ("flat-map-unsq", T,K,V,S)   # the wl naturality: grid of
                                     #   unsq(mul(u,v),K) equals grid of
                                     #   mul(unsq(u,K),reshape(v,S)) AND
                                     #   both operands' flat read maps
                                     #   agree across the two orders
          | ("repeat-chain", T,UD,ES,RS)  # unsq→expand→reshape is
                                     #   repeat_interleave on T's dim
          | ("repeat-heads", A,B,UD,ES)  # sa[-2] == sb[-2] * the repeat
                                     #   factor $attr:ES[UD mod rank+1]

    T/A/B := metavar name (str) | ("mm-out", T, T)   — a *shape spec*:
             a bound term's inferred shape, or the matmul output shape
             of two specs (accepted by every shape-reading predicate).
    G/S   := the same — any shape spec resolves here too.
    Additional shape specs (accepted everywhere a spec is read):
          | ("bcast", T, T)          # broadcast of two specs
          | ("unsq-out", T, K)       # insert a 1 at $attr:K mod (rank+1)
          | ("reshape-out", T, NAME) # resolved reshape target of T
                                     #   under $attr:NAME (-1 folded,
                                     #   numel checked — invalid→None)
          | ("getitem-out", T)       # tensor-index output: s[1:]
          | ("tail-block", T, NAME)  # the trailing block attr NAME's
                                     #   reduce dims prove — su[-k:]
                                     #   when they name exactly T's
                                     #   last k axes (the *_norm
                                     #   normalized_shape), else None
    D*/NAME := attribute metavar names — looked up under "$attr:NAME".
    CMP    := "==" | "!=" | "<" | "<=" | ">" | ">=".

Strictness contract: any predicate that cannot be *proven* from the
binding declines (returns False) — unknown/non-tuple shapes, missing
attrs, non-int dims.  That is the same posture the hand-written checks
take (``None``-dim wildcards only where an op says ``*-compat``), so a
law migrated verbatim keeps its decline behaviour.

The interpreter never calls back into user code: a ``cond`` is a
finite tree of the atoms above and ``eval_cond`` is a total predicate
over it — the ``lemma store`` can ship ``{lhs, rhs, cond}`` records as
JSON without shipping a callable.

Declarative derives
-------------------

The same gap exists on the ``derive=`` side: a derive hook computes the
RHS attributes the LHS cannot bind (``{"$attr:NAME": value}``, or
``None`` to veto), and it was an opaque callable too.  The second half
of this module is the twin DSL — a ``dspec`` is a mapping
``{NAME: expr}`` (or a tuple of ``(NAME, expr)`` pairs) of declarative
value expressions evaluated against the same ``bound`` environment:

::

    dspec := {NAME: expr, ...} | ((NAME, expr), ...)
    expr  := int | float | bool | None         # literal scalar
           | ("lit", v)                        # explicit literal
           | ("attr", NAME)                    # bound $attr value
           | ("attr0", NAME)                   # v[0] if tuple else v
           | ("const", T)                      # bound leaf's ``.value``
           | ("shape", T)                      # inferred shape tuple
           | ("dim", T, i)                     # int dim of shape(T)
           | ("leaf-dim", T, i)                # dim of declared .typ
           | ("len", e)                        # len of a tuple/list
           | ("tuple", e1, ...)                # tuple literal
           | ("concat", e1, ...)               # tuple concatenation
           | ("add"|"sub"|"mul"|"fdiv"|"floordiv", e, e)
           | ("neg"|"recip"|"float"|"int", e)
           | ("bcast", T, T)                   # broadcast two shapes —
                                               # concrete or veto

    NAME := attribute metavar name — the result maps under
            ``"$attr:NAME"``; a ``"$attr:"``-prefixed spec key is
            accepted too.
    T    := a *shape spec* — metavar name or ``("mm-out", T, T)``, the
            same refs the cond predicates resolve.  ``leaf-dim`` takes
            a metavar name directly (a bound leaf's declared ``.typ``
            shape, every dim non-``None`` — the strict read the
            asymmetric-QKV derive used).

Strictness mirrors ``check``'s: any expr that cannot be computed — a
missing attr, an unknown or non-concrete shape, a non-numeric const, a
division by zero, an out-of-range index — vetoes the *whole* derive
(:func:`eval_derive` returns ``None``), the same posture the
hand-written hooks took.  A malformed node (an unknown op, a bare
string where an expr belongs) raises ``ValueError`` — a bug, not a
decline.
"""

from __future__ import annotations

import functools
from typing import Any

from catopt_core.ir import Const, Op
from catopt_core.typing import (
    _axis_pair,
    _broadcast,
    _matmul_shape,
    _numel,
    _shape_of,
)

__all__ = [
    "as_check",
    "as_derive",
    "compile_derive",
    "compile_guard",
    "cond_from_data",
    "cond_to_data",
    "derive_from_data",
    "derive_to_data",
    "eval_cond",
    "eval_derive",
]

#: Content-keyed shape memo (same contract as ``laws.base._SHAPE_MEMO``:
#: bound terms are interned Op/leaf objects, so one shared dict turns
#: repeated per-binding shape inference into a DAG walk).
_MEMO: dict = {}

_CMPS: dict = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

_ATTR_TYPES: dict = {
    "int": int,
    "float": float,
    "number": (int, float),
    "bool": bool,
    "str": str,
    "tuple": tuple,
}


# ---------------------------------------------------------------------------
#  Shape-spec resolution — metavar names and the one computed shape
# ---------------------------------------------------------------------------


def _unsq_shape(shape: Any, dim: Any) -> Any:
    """Return ``unsqueeze``'s output shape: a 1 at ``dim % (rank+1)``.

    Mirrors the ``unsqueeze`` branch of ``_infer_op_shape`` — a
    non-tuple *shape* or non-int *dim* resolves to ``None``.
    """
    if not isinstance(shape, tuple) or not isinstance(dim, int):
        return None
    nd = dim % (len(shape) + 1)
    return (*shape[:nd], 1, *shape[nd:])


def _fold_minus1(shape: tuple, base_n: int) -> tuple:
    """Fold ``-1`` dims of a reshape target through the base numel.

    Mirrors the ``-1`` resolution in ``_infer_op_shape``'s ``reshape``
    branch — an unresolvable fold leaves ``None`` in that slot.
    """
    if -1 not in shape:
        return shape
    known = 1
    for d in shape:
        if d != -1:
            known *= d if isinstance(d, int) and d > 0 else 1
    inferred = (
        base_n // known if known and base_n % known == 0 else None
    )
    return tuple(inferred if d == -1 else d for d in shape)


def _nonneg(shape: Any) -> bool:
    return all(isinstance(d, int) and d >= 0 for d in shape)


def _reshape_shape(base: Any, target: Any) -> Any:
    """Return the resolved ``reshape`` output shape, or ``None``.

    Mirrors the ``reshape`` branch of ``_infer_op_shape``: ``-1`` dims
    fold through numel, and a numel-mismatched target is invalid —
    reported here as ``None`` (the cond strictness contract declines;
    it does not need the ``_INVALID`` poison marker).
    """
    if not isinstance(base, tuple):
        return None
    if not isinstance(target, (tuple, list)):
        return None
    shape = _fold_minus1(tuple(target), _numel(base))
    if (
        _nonneg(shape)
        and _nonneg(base)
        and _numel(shape) != _numel(base)
    ):
        return None
    return shape


def _s_mm_out(args: tuple, bound: dict) -> Any:
    """``("mm-out", a, b)`` — matmul output shape of two specs."""
    if len(args) != 2:
        return None
    return _matmul_shape(_shape(bound, args[0]), _shape(bound, args[1]))


def _s_bcast(args: tuple, bound: dict) -> Any:
    """``("bcast", a, b)`` — broadcast two specs (None reads wildcard)."""
    if len(args) != 2:
        return None
    a, b = _shape(bound, args[0]), _shape(bound, args[1])
    return _broadcast(
        a if isinstance(a, tuple) else None,
        b if isinstance(b, tuple) else None,
    )


def _s_unsq_out(args: tuple, bound: dict) -> Any:
    """``("unsq-out", T, K)`` — the ``unsqueeze`` output shape."""
    if len(args) != 2:
        return None
    return _unsq_shape(
        _shape(bound, args[0]), bound.get(f"$attr:{args[1]}")
    )


def _s_reshape_out(args: tuple, bound: dict) -> Any:
    """``("reshape-out", T, NAME)`` — the resolved reshape target."""
    if len(args) != 2:
        return None
    return _reshape_shape(
        _shape(bound, args[0]), bound.get(f"$attr:{args[1]}")
    )


def _s_getitem_out(args: tuple, bound: dict) -> Any:
    """``("getitem-out", T)`` — tensor-index output: drop dim 0."""
    if len(args) != 1:
        return None
    s = _shape(bound, args[0])
    return s[1:] if isinstance(s, tuple) and s else None


def _attr_dims(bound: dict, name: str) -> tuple | None:
    """Return the bound reduce-dims attr as a tuple, or ``None``.

    Accepts the exported ``(-1,)`` tuple, a list, or a hand-minted
    bare int; every entry must be a non-``bool`` int — the
    normalization the procedural rms hook used.
    """
    dims = bound.get(f"$attr:{name}")
    if isinstance(dims, int) and not isinstance(dims, bool):
        dims = (dims,)
    if not (
        isinstance(dims, (tuple, list))
        and dims
        and all(
            isinstance(d, int) and not isinstance(d, bool) for d in dims
        )
    ):
        return None
    return tuple(dims)


def _s_tail_block(args: tuple, bound: dict) -> Any:
    """``("tail-block", T, NAME)`` — the trailing block NAME proves.

    The ``F.*_norm`` ``normalized_shape`` contract: resolves to
    ``shape(T)[-k:]`` (``k = len(dims)``) iff the bound
    ``$attr:NAME`` reduce dims — a bare non-``bool`` int or a
    non-empty tuple/list of non-``bool`` ints — name *exactly* T's
    last ``k`` axes: each in range, no duplicates, covering
    ``{rank-k … rank-1}`` (``k`` distinct in-range axes whose minimum
    is ``rank - k`` IS that block).  T's shape must be a tuple of
    concrete ints — the resolved value IS the trailing block, so an
    unshaped or ``None``-dim operand cannot mint it.  Every failure
    resolves to ``None``, which both the ``shape-eq``/``shaped``
    guards and the ``("shape", spec)`` derive read as a decline —
    one spec serves the whole normalized-shape precondition.
    """
    if len(args) != 2:
        return None
    dims = _attr_dims(bound, args[1])
    su = _shape(bound, args[0])
    if (
        dims is None
        or not isinstance(su, tuple)
        or not all(isinstance(d, int) for d in su)
    ):
        return None
    return _trailing_block(su, dims)


def _trailing_block(su: tuple, dims: tuple) -> tuple | None:
    """Return ``su[-k:]`` iff *dims* names exactly su's last ``k`` axes.

    ``k`` distinct in-range axes whose minimum is ``rank - k`` IS the
    trailing block ``{rank-k … rank-1}``.
    """
    rank, k = len(su), len(dims)
    if not 1 <= k <= rank or not all(-rank <= d < rank for d in dims):
        return None
    norm = {d % rank for d in dims}
    if len(norm) != k or min(norm) != rank - k:
        return None
    return tuple(su[rank - k :])


#: Tag-dispatch for tuple shape specs — each handler takes
#: ``(args, bound)`` and validates its own arity.
_SPEC_OPS: dict = {
    "mm-out": _s_mm_out,
    "bcast": _s_bcast,
    "unsq-out": _s_unsq_out,
    "reshape-out": _s_reshape_out,
    "getitem-out": _s_getitem_out,
    "tail-block": _s_tail_block,
}


def _shape(bound: dict, ref: Any) -> Any:
    """Resolve a shape spec to an inferred shape (tuple | None | str).

    ``ref`` is a metavar name (looked up in ``bound`` and inferred via
    ``_shape_of``) or a tuple spec dispatched through ``_SPEC_OPS`` —
    ``("mm-out", a, b)``, ``("bcast", a, b)``, ``("unsq-out", a, K)``,
    ``("reshape-out", a, NAME)`` and ``("getitem-out", a)``.
    Anything else resolves to ``None`` (unknown), which every
    predicate treats as a decline.
    """
    if isinstance(ref, str):
        return _shape_of(bound.get(ref), _MEMO)
    if isinstance(ref, (tuple, list)) and ref:
        handler = _SPEC_OPS.get(ref[0])
        if handler is not None:
            return handler(tuple(ref[1:]), bound)
    return None


def _tshape(bound: dict, ref: Any) -> tuple | None:
    """Resolve a spec to a tuple shape, or ``None`` when not a tuple."""
    s = _shape(bound, ref)
    return s if isinstance(s, tuple) else None


def _dim_at(shape: tuple, i: int) -> tuple[bool, Any]:
    """Return ``(in_range, shape[i])`` — ``i`` may be negative."""
    n = len(shape)
    if not (-n <= i < n):
        return False, None
    return True, shape[i]


def _axes_pair(bound: dict, k0: str, k1: str, rank: int) -> Any:
    """Normalize a bound ``transpose`` axis pair against ``rank``.

    Mirrors :func:`catopt_core.laws.layout._bound_axes`: unbound attrs
    mean the *bare* ``t()`` spelling — the implicit last-two swap;
    bound attrs normalize through ``_axis_pair``.  Rank < 2 has no
    swap to move.
    """
    if rank < 2:
        return None
    if f"$attr:{k0}" not in bound and f"$attr:{k1}" not in bound:
        return (rank - 2, rank - 1)
    return _axis_pair(
        bound.get(f"$attr:{k0}"), bound.get(f"$attr:{k1}"), rank
    )


def _num_value(bound: dict, ref: Any) -> Any:
    """Return the bound term's leaf ``.value`` (``Const``), or None."""
    t = bound.get(ref)
    return getattr(t, "value", None) if t is not None else None


# ---------------------------------------------------------------------------
#  Predicate handlers — one tiny function per op (the _OPS table below)
# ---------------------------------------------------------------------------


def _p_shaped(args: tuple, bound: dict) -> bool:
    return _tshape(bound, args[0]) is not None


def _p_concrete(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    return s is not None and all(isinstance(d, int) for d in s)


def _p_scalar(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    return s is not None and len(s) == 0


def _p_uniform(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    return s is not None and all(d == 1 for d in s)


def _p_ones_but_last(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    return s is not None and all(d == 1 for d in s[:-1])


def _p_rank(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    return s is not None and _CMPS[args[1]](len(s), args[2])


def _p_rank_eq(args: tuple, bound: dict) -> bool:
    sa, sb = _tshape(bound, args[0]), _tshape(bound, args[1])
    return sa is not None and sb is not None and len(sa) == len(sb)


def _p_shape_eq(args: tuple, bound: dict) -> bool:
    sa, sb = _tshape(bound, args[0]), _shape(bound, args[1])
    return sa is not None and sa == sb


def _p_shape_compat(args: tuple, bound: dict) -> bool:
    sa, sb = _tshape(bound, args[0]), _tshape(bound, args[1])
    return (
        sa is not None
        and sb is not None
        and len(sa) == len(sb)
        and all(
            da is None or db is None or da == db
            for da, db in zip(sa, sb, strict=True)
        )
    )


def _p_dim_eq(args: tuple, bound: dict) -> bool:
    sa, sb = _tshape(bound, args[0]), _tshape(bound, args[2])
    if sa is None or sb is None:
        return False
    ok_a, da = _dim_at(sa, args[1])
    ok_b, db = _dim_at(sb, args[3])
    return ok_a and ok_b and da == db


def _p_dim_compat(args: tuple, bound: dict) -> bool:
    sa, sb = _tshape(bound, args[0]), _tshape(bound, args[2])
    if sa is None or sb is None:
        return False
    ok_a, da = _dim_at(sa, args[1])
    ok_b, db = _dim_at(sb, args[3])
    return ok_a and ok_b and (da is None or db is None or da == db)


def _p_dim_eq_const(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    if s is None:
        return False
    ok, d = _dim_at(s, args[1])
    return ok and d == args[2]


def _p_dim_eq_attr(args: tuple, bound: dict) -> bool:
    """``sa[K] == sb[K2]`` — dims named by bound attr metavars.

    ``K``/``K2`` resolve through ``$attr:`` keys to bound ints
    (negative dims are honored by ``_dim_at``).  Needed for view-
    commutation guards where the axis is itself pattern-bound —
    e.g. ``mul(select(u,D,I), select(v,D,I)) == select(mul(u,v),D,I)``
    only when ``u``/``v`` agree along ``D``, since broadcasting the
    product along the *selected* axis changes the result.
    """
    sa, sb = _tshape(bound, args[0]), _tshape(bound, args[2])
    if sa is None or sb is None:
        return False
    ia = bound.get(f"$attr:{args[1]}")
    ib = bound.get(f"$attr:{args[3]}")
    if not isinstance(ia, int) or not isinstance(ib, int):
        return False
    ok_a, da = _dim_at(sa, ia)
    ok_b, db = _dim_at(sb, ib)
    return ok_a and ok_b and da == db


def _p_dim_mod(args: tuple, bound: dict) -> bool:
    """``("dim-mod", T, K, m, r)`` — ``sa[K % rank] % m == r``.

    For parity-style guards on an attr-named axis — the ``glu_fold``
    split axis (``("dim-mod", "u", "D", 2, 0)`` requires the dim the
    bound ``$attr:D`` names to be a known even int, since ``chunk``
    splits an odd axis first-big while ``glu`` halves exactly).
    ``K`` resolves like ``dim-eq-attr``'s indices: the bound attr
    must be an int in range; the named dim must itself be an int —
    ``None`` dims off the axis are fine, on it they decline.  A
    non-int or zero modulus declines rather than raising.
    """
    s = _tshape(bound, args[0])
    d = bound.get(f"$attr:{args[1]}")
    m = args[2]
    if (
        s is None
        or not isinstance(d, int)
        or not isinstance(m, int)
        or m == 0
        or not (-len(s) <= d < len(s))
    ):
        return False
    n = s[d % len(s)]
    return isinstance(n, int) and n % m == args[3]


def _p_bcast_into(args: tuple, bound: dict) -> bool:
    small, big = _tshape(bound, args[0]), _shape(bound, args[1])
    return (
        small is not None
        and isinstance(big, tuple)
        and _broadcast(small, big) == big
    )


def _p_mm_shape_ok(args: tuple, bound: dict) -> bool:
    return isinstance(
        _matmul_shape(_shape(bound, args[0]), _shape(bound, args[1])),
        tuple,
    )


def _p_axes_last2(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    if s is None:
        return False
    pair = _axes_pair(bound, args[1], args[2], len(s))
    return pair is not None and set(pair) == {len(s) - 2, len(s) - 1}


def _p_axes_distinct(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    if s is None:
        return False
    pair = _axes_pair(bound, args[1], args[2], len(s))
    return pair is not None and pair[0] != pair[1]


def _p_axes_eq(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    if s is None:
        return False
    p1 = _axes_pair(bound, args[1], args[2], len(s))
    p2 = _axes_pair(bound, args[3], args[4], len(s))
    return (
        p1 is not None
        and p2 is not None
        and p1[0] != p1[1]
        and p2[0] != p2[1]
        and set(p1) == set(p2)
    )


def _p_axis(args: tuple, bound: dict) -> bool:
    s = _tshape(bound, args[0])
    v = bound.get(f"$attr:{args[1]}")
    return (
        s is not None
        and len(s) > 0
        and isinstance(v, int)
        and v % len(s) == args[2] % len(s)
    )


def _p_op_in(args: tuple, bound: dict) -> bool:
    t = bound.get(args[0])
    return isinstance(t, Op) and t.op in args[1]


def _p_leaf(args: tuple, bound: dict) -> bool:
    return not isinstance(bound.get(args[0]), Op)


def _p_const(args: tuple, bound: dict) -> bool:
    return isinstance(bound.get(args[0]), Const)


def _p_term_eq(args: tuple, bound: dict) -> bool:
    return bound.get(args[0]) == bound.get(args[1])


def _p_const_num(args: tuple, bound: dict) -> bool:
    return isinstance(_num_value(bound, args[0]), (int, float))


def _p_const_cmp(args: tuple, bound: dict) -> bool:
    v = _num_value(bound, args[0])
    return isinstance(v, (int, float)) and _CMPS[args[1]](v, args[2])


def _p_attr_is(args: tuple, bound: dict) -> bool:
    return bound.get(f"$attr:{args[0]}") is args[1]


def _p_attr_eq(args: tuple, bound: dict) -> bool:
    return bound.get(f"$attr:{args[0]}") == args[1]


def _p_attr_in(args: tuple, bound: dict) -> bool:
    return bound.get(f"$attr:{args[0]}") in args[1]


def _p_attr_type(args: tuple, bound: dict) -> bool:
    return isinstance(
        bound.get(f"$attr:{args[0]}"), _ATTR_TYPES[args[1]]
    )


def _p_attr_len(args: tuple, bound: dict) -> bool:
    v = bound.get(f"$attr:{args[0]}")
    return isinstance(v, tuple) and _CMPS[args[1]](len(v), args[2])


def _p_attr_cmp_dim(args: tuple, bound: dict) -> bool:
    """``$attr:NAME CMP shape(T)[$attr:K]`` — attr-vs-dim comparison.

    For slice-covering guards: ``end >= shape(u)[dim]`` spells
    ``("attr-cmp-dim", "A_end", ">=", "U", "A_dim")``.  Both sides
    must resolve to ints — a missing attr, an unshaped term, a
    non-int index or a ``None`` dim all decline.
    """
    v = bound.get(f"$attr:{args[0]}")
    s = _tshape(bound, args[2])
    k = bound.get(f"$attr:{args[3]}")
    if not isinstance(v, int) or s is None or not isinstance(k, int):
        return False
    ok, d = _dim_at(s, k)
    return ok and isinstance(d, int) and _CMPS[args[1]](v, d)


def _p_attr_eq_attr(args: tuple, bound: dict) -> bool:
    """``("attr-eq-attr", A, B)`` — two bound attrs compare equal.

    The attr-to-attr sibling of ``attr-eq``: for guards where two
    pattern-bound attrs must coincide (``gqa_absorb_repeat``'s k/v
    expand shapes).  An unbound side declines — the strict posture.
    """
    a = bound.get(f"$attr:{args[0]}", _MISSING)
    b = bound.get(f"$attr:{args[1]}", _MISSING)
    return a is not _MISSING and a == b


def _p_bcast_eq(args: tuple, bound: dict) -> bool:
    """``bcast(sa, sb) == bcast(sc, sd)`` — output grids coincide.

    The ``id:out_shape_eq`` half of a strip guard: every spec must
    resolve to a tuple and both broadcasts must succeed — an unknown
    or ill-typed side declines, it does not wildcard.
    """
    specs = [_shape(bound, a) for a in args]
    if not all(isinstance(s, tuple) for s in specs):
        return False
    g1 = _broadcast(specs[0], specs[1])
    g2 = _broadcast(specs[2], specs[3])
    return isinstance(g1, tuple) and isinstance(g2, tuple) and g1 == g2


def _p_ones_before(args: tuple, bound: dict) -> bool:
    """``s[i] == 1`` for every dim before the normalized attr axis.

    ``("ones-before", T, K)``: *K* is the bound ``unsqueeze`` axis
    (normalized ``mod rank+1``).  This is the ``id:same_pairing``
    half of the unsqueeze strip guard — broadcasting ``unsq(u, K)``
    to the common grid reads ``u`` through the same index map as
    broadcasting ``u`` exactly when every dim shifted by the
    insertion is a broadcast-1.
    """
    s = _tshape(bound, args[0])
    k = bound.get(f"$attr:{args[1]}")
    if s is None or not isinstance(k, int):
        return False
    nd = k % (len(s) + 1)
    return all(d == 1 for d in s[:nd])


def _p_axes_noop(args: tuple, bound: dict) -> bool:
    """``transpose(u, D0, D1)`` is a semantic no-op on ``u``.

    True when the normalized axis pair is the same axis (``d0 == d1``
    — the literal ``tr:noop``) or when both swapped extents are 1 —
    swapping two size-1 axes neither changes the shape tuple nor the
    broadcast pairing.  Anything else declines (an unknown shape or
    unbound pair cannot be proven a no-op).  A rank-1 operand has only
    the one axis — every valid pair is (0, 0).
    """
    s = _tshape(bound, args[0])
    if s is None or not s:
        return False
    if len(s) == 1:
        d0 = bound.get(f"$attr:{args[1]}")
        d1 = bound.get(f"$attr:{args[2]}")
        return (
            isinstance(d0, int)
            and isinstance(d1, int)
            and -1 <= d0 < 1
            and -1 <= d1 < 1
        )
    pair = _axes_pair(bound, args[1], args[2], len(s))
    if pair is None:
        return False
    d0, d1 = pair
    return d0 == d1 or (s[d0] == 1 and s[d1] == 1)


def _all_int(shape: tuple) -> bool:
    return all(isinstance(d, int) for d in shape)


def _concrete_specs(bound: dict, refs: tuple) -> list | None:
    """Resolve every spec in *refs*; ``None`` unless all concrete."""
    out = []
    for r in refs:
        s = _tshape(bound, r)
        if s is None or not _all_int(s):
            return None
        out.append(s)
    return out


def _stride(shape: tuple, i: int) -> int:
    """Flat-index stride of ``shape[i]``: product of trailing dims."""
    n = 1
    for d in shape[i + 1 :]:
        n *= d
    return n


def _p_flat_pair_unsq(args: tuple, bound: dict) -> bool:
    """Flat-read pairing of ``u`` survives ``unsq→S`` vs ``bcast→G``.

    ``("flat-pair-unsq", T, K, G, S)``: the *wr* naturality's u-side —
    ``reshape(mul(u,v), S)`` reads ``u`` through its broadcast into
    ``G = bcast(u,v)``; ``mul(unsq(u,K), reshape(v,S))`` reads it
    through ``unsq``'s broadcast into ``S``.  For the elements to
    coincide for all ``u`` each non-1 dim's flat-index digit must sit
    at the same stride: ``u``-dim ``k`` is read at grid position
    ``len(G)-r+k`` in ``G`` and at ``len(S)-(r+1)+j`` in ``S`` (``j``
    is ``k`` or ``k+1`` past the inserted axis ``nd = K mod r+1``).
    Requires every spec to resolve to a concrete-int shape.
    """
    k = bound.get(f"$attr:{args[1]}")
    specs = _concrete_specs(bound, (args[0], args[2], args[3]))
    if specs is None or not isinstance(k, int):
        return False
    su, g, s = specs
    r, n, n2 = len(su), len(g), len(s)
    nd = k % (r + 1)
    for kk in range(r):
        if su[kk] == 1:
            continue
        j = kk if kk < nd else kk + 1
        p1, p2 = n - r + kk, n2 - (r + 1) + j
        if not (0 <= p1 < n and 0 <= p2 < n2):
            return False
        if g[p1] != su[kk] or s[p2] != su[kk]:
            return False
        if _stride(g, p1) != _stride(s, p2):
            return False
    return True


def _viewed_map(src: tuple, grid: tuple) -> dict:
    """Flat-read coeff map of *src* broadcast into *grid*.

    Returns ``{grid position: flat stride}`` for each non-1 dim —
    the index map ``I ↦ Σ_p I[p]·c_p`` into the operand's flat
    storage under right-aligned broadcasting.
    """
    off = len(grid) - len(src)
    return {
        off + j: _stride(src, j) for j, dj in enumerate(src) if dj != 1
    }


def _shifted_map(src: tuple, grid: tuple, nd: int) -> dict:
    """Flat-read coeff map of *src* through an axis insertion.

    The operand broadcasts into *grid*, then an axis is inserted at
    ``nd`` — positions at or past it read one slot further right.
    """
    off = len(grid) - len(src)
    out = {}
    for j, dj in enumerate(src):
        if dj != 1:
            p = off + j
            out[p + (1 if p >= nd else 0)] = _stride(src, j)
    return out


def _flat_map_ok(lmap: dict, rmap: dict, w: tuple) -> bool:
    """Compare flat-read coefficient maps on the shared grid *w*.

    Two maps read the same elements for every value iff their
    coefficients agree at every position the grid actually varies
    (``w[p] > 1``; size-1 positions read the constant 0 either way).
    """
    for p, wdim in enumerate(w):
        if wdim == 1:
            continue
        if lmap.get(p, 0) != rmap.get(p, 0):
            return False
    return True


def _p_flat_map_unsq(args: tuple, bound: dict) -> bool:
    """Decide the *wl* naturality: ``unsq∘mul`` under a reshape partner.

    ``("flat-map-unsq", T, K, V, S)`` decides
    ``mul(unsq(u,K), reshape(v,S)) == unsq(mul(u,v), K)`` at shape
    level: the output grids must coincide —
    ``bcast(unsq(u), S) == unsq(bcast(u,v), K)`` (note *K* normalizes
    against each operand's own rank) — and each operand's flat-read
    coefficient map must agree across the two orders: ``v`` read via
    ``reshape(v,S)``'s broadcast vs its broadcast in ``mul(u,v)``,
    ``u`` read via ``unsq``'s broadcast vs its broadcast in
    ``mul(u,v)`` shifted by the inserted axis.  All shapes must
    resolve to concrete ints.
    """
    k = bound.get(f"$attr:{args[1]}")
    specs = _concrete_specs(bound, (args[0], args[2], args[3]))
    if specs is None or not isinstance(k, int):
        return False
    su, sv, s = specs
    su_p = _unsq_shape(su, k)
    gu = _broadcast(su, sv)
    gl = _broadcast(su_p, s)
    if not isinstance(gu, tuple) or not isinstance(gl, tuple):
        return False
    nd = k % (len(gu) + 1)
    w = (*gu[:nd], 1, *gu[nd:])
    if w != gl:
        return False
    return _flat_map_ok(
        _viewed_map(su_p, w), _shifted_map(su, gu, nd), w
    ) and _flat_map_ok(_viewed_map(s, w), _shifted_map(sv, gu, nd), w)


def _repeat_merge_ok(bs: tuple, d: int, es: tuple, rs: tuple) -> bool:
    """Decide the chain's shape equation at the normalized axis ``d``.

    Expand must be exactly the unsqueezed shape with an int factor
    ``r > 1`` at ``d`` — and reshape exactly
    ``us[:d-1] + (us[d-1]*r,) + us[d+1:]``.
    """
    nd = len(bs)
    d = d % (nd + 1)
    us = _unsq_shape(bs, d)
    if len(es) != len(us) or len(rs) != nd or d == 0:
        return False
    r = es[d]
    if not isinstance(r, int) or r <= 1:
        return False
    if any(es[i] != us[i] for i in range(len(us)) if i != d):
        return False
    merged = (*us[: d - 1], (us[d - 1] or 0) * r, *us[d + 1 :])
    return rs == merged


def _p_repeat_chain(args: tuple, bound: dict) -> bool:
    """``("repeat-chain", T, UD, ES, RS)`` — the copy-map chain.

    Decides that ``reshape(expand(unsqueeze(T, UD), ES), RS)`` is
    ``repeat_interleave`` on T's dim ``d-1``: unsqueeze inserts a 1
    at ``d = UD % (rank+1)``, expand broadcasts *only* that dim by
    an int factor ``r > 1``, and the reshape merges dims ``d-1`` and
    ``d`` back into one.  The base shape must be a tuple of concrete
    dims — every unprovable piece declines (the strict posture; a
    law never admits a chain it cannot verify).
    """
    d = bound.get(f"$attr:{args[1]}")
    es = bound.get(f"$attr:{args[2]}")
    rs = bound.get(f"$attr:{args[3]}")
    bs = _shape(bound, args[0])
    if not (
        isinstance(d, int)
        and isinstance(es, tuple)
        and isinstance(rs, tuple)
        and isinstance(bs, tuple)
    ):
        return False
    if any(x is None for x in bs):
        return False
    return _repeat_merge_ok(bs, d, es, rs)


def _concrete_rank2(bound: dict, ref: Any) -> tuple | None:
    """Resolve *ref* to a rank-≥2 shape of concrete dims, or None."""
    s = _tshape(bound, ref)
    if s is None or len(s) < 2 or any(x is None for x in s):
        return None
    return s


def _rep_factor(es: tuple, d: int, rank: int) -> int | None:
    """Return the expand factor at the normalized axis, or None."""
    dd = d % (rank + 1)
    if not (0 <= dd < len(es)):
        return None
    r = es[dd]
    return r if isinstance(r, int) else None


def _p_repeat_heads(args: tuple, bound: dict) -> bool:
    """``("repeat-heads", A, B, UD, ES)`` — the repeated-head relation.

    ``sa[-2] == sb[-2] * r`` where ``r = es[d]`` is the expand
    factor at the normalized unsqueeze axis ``d = UD %
    (len(sb)+1)`` — the head-count equation a repeat-absorption law
    needs (query heads = kv heads times the repeat factor).  Both
    shapes must resolve, be rank ≥ 2, and carry concrete dims
    throughout; the axis must be a bound int and ``es`` a tuple
    covering it with an int at the factor position — anything
    unprovable declines.
    """
    sa = _concrete_rank2(bound, args[0])
    sb = _concrete_rank2(bound, args[1])
    d = bound.get(f"$attr:{args[2]}")
    es = bound.get(f"$attr:{args[3]}")
    if (
        sa is None
        or sb is None
        or not isinstance(d, int)
        or not isinstance(es, tuple)
    ):
        return False
    r = _rep_factor(es, d, len(sb))
    return r is not None and sa[-2] == sb[-2] * r


_OPS: dict = {
    "and": None,  # combinators are handled in eval_cond directly
    "or": None,
    "not": None,
    "shaped": _p_shaped,
    "concrete": _p_concrete,
    "scalar": _p_scalar,
    "uniform": _p_uniform,
    "ones-but-last": _p_ones_but_last,
    "rank": _p_rank,
    "rank-eq": _p_rank_eq,
    "shape-eq": _p_shape_eq,
    "shape-compat": _p_shape_compat,
    "dim-eq": _p_dim_eq,
    "dim-compat": _p_dim_compat,
    "dim-eq-const": _p_dim_eq_const,
    "dim-eq-attr": _p_dim_eq_attr,
    "dim-mod": _p_dim_mod,
    "bcast-into": _p_bcast_into,
    "mm-shape-ok": _p_mm_shape_ok,
    "axes-last2": _p_axes_last2,
    "axes-distinct": _p_axes_distinct,
    "axes-eq": _p_axes_eq,
    "axis": _p_axis,
    "op-in": _p_op_in,
    "leaf": _p_leaf,
    "const": _p_const,
    "term-eq": _p_term_eq,
    "const-num": _p_const_num,
    "const-cmp": _p_const_cmp,
    "attr-is": _p_attr_is,
    "attr-eq": _p_attr_eq,
    "attr-in": _p_attr_in,
    "attr-type": _p_attr_type,
    "attr-len": _p_attr_len,
    "attr-cmp-dim": _p_attr_cmp_dim,
    "attr-eq-attr": _p_attr_eq_attr,
    "bcast-eq": _p_bcast_eq,
    "ones-before": _p_ones_before,
    "axes-noop": _p_axes_noop,
    "flat-pair-unsq": _p_flat_pair_unsq,
    "flat-map-unsq": _p_flat_map_unsq,
    "repeat-chain": _p_repeat_chain,
    "repeat-heads": _p_repeat_heads,
}


def eval_cond(cond: Any, bound: dict) -> bool:
    """Evaluate a declarative condition against a ``bound`` dict.

    ``bound`` is the same environment the ``check`` hook sees —
    metavar names to resolved member terms plus ``"$attr:NAME"`` keys
    to matched attribute values.  Returns the boolean verdict; an
    unknown operator or malformed node raises ``ValueError`` (a
    malformed condition is a bug, not a decline).
    """
    if isinstance(cond, bool):
        return cond
    if not isinstance(cond, (tuple, list)) or not cond:
        raise ValueError(f"malformed cond node: {cond!r}")
    op = cond[0]
    if op == "and":
        return all(eval_cond(c, bound) for c in cond[1:])
    if op == "or":
        return any(eval_cond(c, bound) for c in cond[1:])
    if op == "not":
        if len(cond) != 2:
            raise ValueError(f"malformed not node: {cond!r}")
        return not eval_cond(cond[1], bound)
    handler = _OPS.get(op)
    if handler is None:
        raise ValueError(f"unknown cond op: {op!r}")
    return bool(handler(tuple(cond[1:]), bound))


# ---------------------------------------------------------------------------
#  Guard composition + the check-shaped view
# ---------------------------------------------------------------------------


def compile_guard(cond: Any, check: Any):
    """Return the callable side condition for ``cond`` + ``check``.

    ``cond`` evaluates first (pure data — cheap and total), then the
    procedural ``check`` when present; the conjunction IS the rule's
    side condition.  ``Rewrite.__post_init__`` folds a rule's ``cond``
    into ``check`` through here, so every evaluation site — apply_rule,
    certificate replay, term-level matching, meta's composite guards —
    stays on the single ``rule.check`` convention.
    """
    if check is None:

        def guard(bound: dict) -> bool:
            return eval_cond(cond, bound)

    else:

        def guard(bound: dict) -> bool:
            return eval_cond(cond, bound) and bool(check(bound))

    return guard


def as_check(cond: Any):
    """View a declarative ``cond`` as a ``check``-shaped callable.

    For the test-facing ``_check_*`` helpers the law modules keep:
    the helper IS the same data the rule's ``cond`` field carries,
    so the two can never drift.  (``functools.partial`` —
    ``rulecache._hook_sig`` fingerprints partials explicitly.)
    """
    return functools.partial(eval_cond, cond)


# ---------------------------------------------------------------------------
#  Serialization — JSON-safe canonical form
# ---------------------------------------------------------------------------


def cond_to_data(cond: Any) -> Any:
    """Return the JSON-canonical form of *cond* (lists, not tuples).

    ``json.dumps`` already serializes a tuple tree as arrays; this
    normalizes to lists explicitly so the dumped form is identical
    whether the author wrote tuples or loaded lists.
    """
    if isinstance(cond, (tuple, list)):
        return [cond_to_data(x) for x in cond]
    return cond


def cond_from_data(data: Any) -> Any:
    """Rebuild the canonical tuple-tree cond from parsed data.

    Inverse of :func:`cond_to_data` — lists become tuples again, so a
    rule rebuilt from a store record carries the same canonical form
    as one written in source.
    """
    if isinstance(data, (tuple, list)):
        return tuple(cond_from_data(x) for x in data)
    return data


# ---------------------------------------------------------------------------
#  Declarative derives — the ``derive=`` twin of the cond DSL
#
#  A ``dspec`` is ``{NAME: expr}`` (or ``((NAME, expr), ...)``) — pure
#  data, canonicalised to a sorted tuple of pairs so a rule stays
#  hashable and store-loaded specs compare equal to source-spelled
#  ones.  ``eval_derive`` walks each expr and produces the
#  ``{"$attr:NAME": value}`` map the ``derive`` hook contract expects,
#  or ``None`` when any expr declines.
# ---------------------------------------------------------------------------


class _Decline(Exception):
    """Internal control flow: one uncomputable expr vetoes the spec."""


#: Sentinel for "the binding has no such entry" — distinct from a
#: legitimately ``None`` bound value.
_MISSING = object()


def _d_lit(args: tuple, bound: dict) -> Any:
    """``("lit", v)`` — an explicit literal (string, tuple, …)."""
    return args[0]


def _d_attr(args: tuple, bound: dict) -> Any:
    """``("attr", NAME)`` — the bound attribute metavar's value."""
    v = bound.get(f"$attr:{args[0]}", _MISSING)
    if v is _MISSING:
        raise _Decline(f"$attr:{args[0]} unbound")
    return v


def _d_attr0(args: tuple, bound: dict) -> Any:
    """``("attr0", NAME)`` — ``v[0]`` for a tuple attr, else ``v``."""
    v = bound.get(f"$attr:{args[0]}", _MISSING)
    if v is _MISSING:
        raise _Decline(f"$attr:{args[0]} unbound")
    if isinstance(v, tuple):
        if not v:
            raise _Decline("empty attr tuple")
        return v[0]
    return v


def _d_const(args: tuple, bound: dict) -> Any:
    """``("const", T)`` — the bound term's leaf ``.value``."""
    v = getattr(bound.get(args[0]), "value", _MISSING)
    if v is _MISSING:
        raise _Decline(f"{args[0]} has no .value")
    return v


def _d_shape(args: tuple, bound: dict) -> tuple:
    """``("shape", T)`` — the inferred shape of a bound term."""
    s = _shape(bound, args[0])
    if not isinstance(s, tuple):
        raise _Decline(f"{args[0]} unshaped")
    return s


def _d_dim(args: tuple, bound: dict) -> int:
    """``("dim", T, i)`` — a concrete dim of the inferred shape."""
    s = _tshape(bound, args[0])
    i = args[1]
    if s is None or not isinstance(i, int):
        raise _Decline("dim: unshaped term or non-int index")
    ok, d = _dim_at(s, i)
    if not ok or not isinstance(d, int):
        raise _Decline("dim: index out of range or dim unknown")
    return d


def _d_leaf_dim(args: tuple, bound: dict) -> Any:
    """``("leaf-dim", T, i)`` — a dim of the leaf's declared shape.

    Reads ``term.typ.shape`` — never inferred: an ``Op`` binding has no
    ``.typ`` and declines, matching the strict read the asymmetric-QKV
    derive used (every declared dim must be non-``None``).
    """
    t = bound.get(args[0])
    sh = getattr(getattr(t, "typ", None), "shape", None)
    i = args[1]
    if (
        not isinstance(sh, tuple)
        or not sh
        or any(d is None for d in sh)
        or not isinstance(i, int)
        or not (-len(sh) <= i < len(sh))
    ):
        raise _Decline("leaf-dim: no concrete declared shape")
    return sh[i]


def _d_len(args: tuple, bound: dict) -> int:
    """``("len", e)`` — length of a tuple/list/string value."""
    v = _dexpr(args[0], bound)
    if not isinstance(v, (tuple, list, str)):
        raise _Decline("len: not a sized value")
    return len(v)


def _d_tuple(args: tuple, bound: dict) -> tuple:
    """``("tuple", e1, ...)`` — a tuple of the evaluated elements."""
    return tuple(_dexpr(a, bound) for a in args)


def _d_concat(args: tuple, bound: dict) -> tuple:
    """``("concat", e1, ...)`` — concatenation of tuple/list values."""
    parts = [_dexpr(a, bound) for a in args]
    if not all(isinstance(p, (tuple, list)) for p in parts):
        raise _Decline("concat: a non-tuple operand")
    out: list = []
    for p in parts:
        out.extend(p)
    return tuple(out)


def _d_bcast(args: tuple, bound: dict) -> tuple:
    """``("bcast", T, T)`` — broadcast two shape specs, concrete dims.

    The operands resolve through ``_shape`` — a non-tuple result
    (``None`` unknown, the ``_INVALID`` marker) reads as the ``None``
    wildcard, exactly what ``_broadcast`` does with an unshaped side.
    The result must be a tuple of ints or the whole derive vetoes.
    """
    sa = _shape(bound, args[0])
    sb = _shape(bound, args[1])
    out = _broadcast(
        sa if isinstance(sa, tuple) else None,
        sb if isinstance(sb, tuple) else None,
    )
    if not (
        isinstance(out, tuple) and all(isinstance(d, int) for d in out)
    ):
        raise _Decline("bcast: non-concrete or ill-typed result")
    return tuple(out)


_DBINOPS: dict = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "fdiv": lambda a, b: a / b,
    "floordiv": lambda a, b: a // b,
}

_DUNOPS: dict = {
    "neg": lambda a: -a,
    "recip": lambda a: 1.0 / a,
    "float": lambda a: float(a),
    "int": lambda a: int(a),
}


def _binop(op: str):
    """Build the ``(add|sub|mul|fdiv|floordiv)`` handler for *op*."""

    def handler(args: tuple, bound: dict) -> Any:
        try:
            return _DBINOPS[op](
                _dexpr(args[0], bound), _dexpr(args[1], bound)
            )
        except (ArithmeticError, TypeError, ValueError) as e:
            raise _Decline(f"{op}: {e}") from None

    return handler


def _unop(op: str):
    """Build the ``(neg|recip|float|int)`` handler for *op*."""

    def handler(args: tuple, bound: dict) -> Any:
        try:
            return _DUNOPS[op](_dexpr(args[0], bound))
        except (ArithmeticError, TypeError, ValueError) as e:
            raise _Decline(f"{op}: {e}") from None

    return handler


_DOPS: dict = {
    "lit": _d_lit,
    "attr": _d_attr,
    "attr0": _d_attr0,
    "const": _d_const,
    "shape": _d_shape,
    "dim": _d_dim,
    "leaf-dim": _d_leaf_dim,
    "len": _d_len,
    "tuple": _d_tuple,
    "concat": _d_concat,
    "bcast": _d_bcast,
    **{k: _binop(k) for k in _DBINOPS},
    **{k: _unop(k) for k in _DUNOPS},
}


def _dexpr(node: Any, bound: dict) -> Any:
    """Evaluate one derive expr to its value (``_Decline`` vetoes).

    Bare scalars (``int``/``float``/``bool``/``None``) are literals;
    every other node is an ``(op, arg, ...)`` tuple dispatched through
    ``_DOPS``.  A bare string or unknown op is malformed — ``ValueError``,
    a bug rather than a decline.
    """
    if node is None or isinstance(node, (bool, int, float)):
        return node
    if not isinstance(node, (tuple, list)) or not node:
        raise ValueError(f"malformed derive expr: {node!r}")
    handler = _DOPS.get(node[0])
    if handler is None:
        raise ValueError(f"unknown derive op: {node[0]!r}")
    return handler(tuple(node[1:]), bound)


def derive_from_data(data: Any) -> Any:
    """Canonicalise a derive spec to sorted ``((NAME, expr), ...)``.

    Accepts a mapping ``{NAME: expr}`` or a sequence of ``(NAME, expr)``
    pairs — the forms ``json.loads`` hands back and the ones a law
    spells in source.  ``"$attr:"``-prefixed keys lose the prefix (the
    output keys are bare names; ``eval_derive`` re-prefixes).  Expr
    trees canonicalise through ``cond_from_data`` — the same
    list-to-tuple normalisation — so a store-loaded spec is equal and
    hashable alongside its source-spelled twin.  ``None`` passes
    through; anything else raises ``ValueError``.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, (tuple, list)) and all(
        isinstance(p, (tuple, list)) and len(p) == 2 for p in data
    ):
        items = [tuple(p) for p in data]
    else:
        raise ValueError(f"malformed derive spec: {data!r}")
    out = []
    for k, v in items:
        if not isinstance(k, str):
            raise ValueError(f"derive spec key not a name: {k!r}")
        name = k[len("$attr:") :] if k.startswith("$attr:") else k
        out.append((name, cond_from_data(v)))
    return tuple(sorted(out, key=lambda p: p[0]))


def derive_to_data(spec: Any) -> Any:
    """Return the JSON-canonical form of *spec*: ``{NAME: expr-lists}``.

    Inverse of :func:`derive_from_data` — canonical pairs become a
    ``dict`` with the expr trees' tuples normalised to lists (through
    ``cond_to_data``), so the dumped form is identical whether the
    author wrote a dict or the canonical pairs.  ``None`` passes
    through.
    """
    if spec is None:
        return None
    return {
        name: cond_to_data(expr)
        for name, expr in derive_from_data(spec)
    }


def eval_derive(dspec: Any, bound: dict) -> dict | None:
    """Evaluate a declarative derive spec against a ``bound`` dict.

    ``bound`` is the same environment the ``derive`` hook sees —
    metavar names to resolved member terms plus ``"$attr:NAME"`` keys
    to matched attribute values.  Returns the extra-substitution map
    ``{"$attr:NAME": value}`` (possibly empty); the first expr that
    declines vetoes the whole spec and ``None`` comes back — the same
    ``derive`` contract the e-graph and certificate replay honour.
    A malformed expr raises ``ValueError`` — a bug, not a decline.
    """
    out: dict = {}
    for name, expr in derive_from_data(dspec):
        try:
            out[f"$attr:{name}"] = _dexpr(expr, bound)
        except _Decline:
            return None
    return out


def compile_derive(dspec: Any, derive: Any):
    """Return the callable derive for ``dspec`` + ``derive``.

    The spec evaluates first (pure data — cheap and total), then the
    procedural ``derive`` when present; a ``None`` from either vetoes,
    and both maps merge (``derive``'s keys win a collision — it is the
    deliberate override).  ``Rewrite.__post_init__`` folds a rule's
    ``dspec`` into ``derive`` through here, so every evaluation site —
    apply_rule, certificate replay, meta's composite guards — stays on
    the single ``rule.derive`` convention.  The two-closure split
    mirrors ``compile_guard``: ``laws.serialize._proc_derive`` tells a
    spec-only fold from a hand-written hook by the closure's code
    object.
    """
    spec = derive_from_data(dspec)
    if derive is None:

        def derived(bound: dict) -> dict | None:
            return eval_derive(spec, bound)

    else:

        def derived(bound: dict) -> dict | None:
            out = eval_derive(spec, bound)
            if out is None:
                return None
            extra = derive(bound)
            if extra is None:
                return None
            return {**out, **extra}

    return derived


def as_derive(dspec: Any):
    """View a declarative ``dspec`` as a ``derive``-shaped callable.

    For the test-facing ``_derive_*`` helpers the law modules keep:
    the helper IS the same data the rule's ``dspec`` field carries,
    so the two can never drift.  (``functools.partial`` —
    ``rulecache._hook_sig`` fingerprints partials explicitly, and
    ``Rewrite.__post_init__`` recovers the spec from this shape, so a
    rule built ``derive=as_derive(spec)`` is still full data.)
    """
    return functools.partial(eval_derive, derive_from_data(dspec))
