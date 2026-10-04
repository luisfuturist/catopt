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

    T/A/B := metavar name (str) | ("mm-out", T, T)   — a *shape spec*:
             a bound term's inferred shape, or the matmul output shape
             of two specs (accepted by every shape-reading predicate).
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
"""

from __future__ import annotations

import functools
from typing import Any

from catopt_core.ir import Const, Op
from catopt_core.typing import (
    _axis_pair,
    _broadcast,
    _matmul_shape,
    _shape_of,
)

__all__ = [
    "as_check",
    "compile_guard",
    "cond_from_data",
    "cond_to_data",
    "eval_cond",
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


def _shape(bound: dict, ref: Any) -> Any:
    """Resolve a shape spec to an inferred shape (tuple | None | str).

    ``ref`` is a metavar name (looked up in ``bound`` and inferred via
    ``_shape_of``) or ``("mm-out", a, b)`` — the matmul output shape of
    two nested specs.  Anything else resolves to ``None`` (unknown),
    which every predicate treats as a decline.
    """
    if isinstance(ref, str):
        return _shape_of(bound.get(ref), _MEMO)
    if (
        isinstance(ref, (tuple, list))
        and len(ref) == 3
        and ref[0] == "mm-out"
    ):
        return _matmul_shape(
            _shape(bound, ref[1]), _shape(bound, ref[2])
        )
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
