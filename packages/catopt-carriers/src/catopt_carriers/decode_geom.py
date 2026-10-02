# ruff: noqa: RUF003 -- law strings and docstrings use
# mathematical notation (≡, Δ, −, ∘) deliberately; ASCII would misstate
# it.
"""Decode-memory geometry laws — head packing and KV layout moves.

``decode_laws`` covers the bandwidth *policy* rewrites (append-as-
scatter, index dedup, repeat-as-gather).  This module covers the
*geometry*: how per-head tensors pack into one contiguous KV buffer
and how a shared gather/index slides across the views.  Everything
here is an EXACT equational law over the existing op table — pure
layout, no arithmetic, no value-level reasoning.

OP-TABLE AUDIT — what the IR can express
----------------------------------------
* **Head packing** — ``stack`` (insert axis) / ``concat`` (extend
  axis) are the packing constructors; ``select`` / ``unbind`` /
  ``getitem`` / ``slice`` / ``narrow`` are the per-head views.
  The exact laws: adjacent slices of ONE base re-join
  (``cat(t[a:b], t[b:c], d) ≡ t[a:c]``), a view covering exactly one
  operand of a cat recovers it (``cat(x,y,d)[0:ex] ≡ x``,
  ``[ex:] ≡ y``), and a ``select``/``unbind``/``getitem`` on a
  ``stack`` returns the stacked operand.  ``cat(unsqueeze(a,d),
  unsqueeze(b,d), d) ≡ stack(a,b,d)`` packs two heads materialised
  apart.  Binary concat only — an n-way cat is a nested binary term.
* **Shared index across heads** — ``index_select`` on an axis OTHER
  than the view axis commutes with ``slice`` / ``narrow`` /
  ``select`` / ``unbind`` / ``stack`` / ``concat`` (the axis index
  shifts where a view removes/inserts one — ``derive`` computes it).
  Per-head gathers ``cat(K_h0[I], K_h1[I], h)`` collapse to one
  gather over the packed table plus views — and gathers on the cat
  axis itself batch as ``cat(x[Ix], y[Iy]) ≡ cat(x,y)[Ix + (ex+Iy)]``.
* **Shared tables (RoPE cos/sin)** — rope factors arrive as
  ``mul(x, C)`` / ``mul(x, S)`` term metavariables (params or
  computed tables).  Term-identical tables are already shared by
  interning; the geometry law lifts them over the pack:
  ``cat(x·C, y·C, d) ≡ cat(x,y,d)·C`` — exact when ``C`` broadcasts
  against the cat (checked, not assumed: a per-head table that
  does not broadcast on the packed axis declines).  Value-level
  const dedup (bitwise-equal but distinct params) is NOT here — it
  is a value property, owned by ``share_duplicate_params``'s
  ``_offer_witness`` ritual; see the audit in ``decode_laws``.

LAWS (:data:`DECODE_GEOM_LAWS`)
-------------------------------
1. ``cat_slice_merge[_step]`` / ``cat_narrow_merge`` — adjacent
   slices of one base re-join into the bigger view.
2. ``cat_head_*`` — a view covering exactly one cat operand
   recovers it (``slice`` and ``narrow`` spellings, left and
   right pieces); ``stack_select_*`` / ``stack_unbind_*`` /
   ``stack_getitem_*`` — a select on a stack returns the stacked
   head.
3. ``stack_from_cat_unsqueeze`` — ``cat(a[d], b[d], d) ≡
   stack(a,b,d)``: heads materialised apart pack into one buffer.
4. ``slice_full[_step]`` / ``slice_bare_id`` / ``narrow_full`` —
   a full-extent read is the base (the merge's cleanup).
5. ``gather_<view>_{out,in}[_t]`` — ``index_select`` slides across
   ``slice`` / ``narrow`` / ``select`` / ``unbind`` / ``stack`` on
   a different axis, both directions and both index spellings
   (``index=`` attr and index-tensor operand).
6. ``gather_cat_{out,in}[_t]`` — a gather slides across a cat on a
   different axis; ``gather_cat_batch`` — per-piece gathers on the
   cat axis itself batch into one gather with an offset index.
7. ``unary_cat_<op>`` / ``binary_cat_<op>`` — pointwise maps lift
   over a cat; the binary family is the shared-table lift
   (``cat(x·C, y·C) ≡ cat(x,y)·C`` for rope tables, broadcast-
   checked).

Not implemented (documented limits): n-way cat patterns (binary
only — n-way cats are nested binary terms and merge pairwise);
gather/gather axis arithmetic on the SAME axis through ``select``;
``transpose``/``permute`` axis tracking for gathers (the layout
laws own transpose mobility); slice-of-slice bound arithmetic;
value-level param/const tying (``share_duplicate_params`` owns it).
"""

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws import R as _R
from catopt_core.laws import RuleSet
from catopt_core.laws import tags as _tags
from catopt_core.typing import broadcast


def R(name: str, lhs, rhs, **kw) -> Rewrite:
    """Module-local law constructor — decode laws carry ``CARRIER``+``DECODE``."""
    kw.setdefault("tags", (_tags.CARRIER, _tags.DECODE))
    return _R(name, lhs, rhs, **kw)


__all__ = [
    "DECODE_GEOM_LAWS",
    "DECODE_GEOM_RULES",
    "GATHER_CAT_LAWS",
    "GATHER_VIEW_LAWS",
    "HEAD_PACK_LAWS",
    "TABLE_LIFT_LAWS",
    "VIEW_ID_LAWS",
]


def _dshape(t: Any):
    """Return the value shape of a bound term — carrier-aware.

    Same resolver ``decode_laws`` uses: a metavariable may resolve
    to a carrier member whose cost-convention shape is not the
    tensor value shape.
    """
    from catopt_carriers.xcarrier import _xshape

    return _xshape(t)


def _shape(t: Any):
    """Return the bound term's shape tuple, or ``None`` if unresolvable."""
    s = _dshape(t)
    return s if isinstance(s, tuple) else None


def _dim_eq(a: Any, b: Any) -> bool:
    """Axis compatibility: equal, or either side unknown (None)."""
    return a is None or b is None or a == b


def _axis(v: Any, n: int) -> int | None:
    """Normalise a raw dim into ``[0, n)`` — ``None`` when not an int."""
    return v % n if isinstance(v, int) and n > 0 else None


def _off_axis_eq(a: tuple, b: tuple, axis: int) -> bool:
    """Off-axis compatibility of a same-rank pair (None = wildcard)."""
    return all(
        _dim_eq(x, y)
        for i, (x, y) in enumerate(zip(a, b, strict=True))
        if i != axis
    )


def _cat_pair(bound: dict):
    """Return ``(axis, xs, ys)`` for a well-typed ``cat(x,y,d)`` pair.

    ``None`` when the operands' ranks differ or the cat axis (``CD``,
    or the view axis ``VD`` which must be the same axis) does not
    normalise.
    """
    xs, ys = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (xs and ys and len(xs) == len(ys)):
        return None
    n = len(xs)
    axis = _axis(bound.get("$attr:CD"), n)
    vd = _axis(bound.get("$attr:VD"), n)
    if axis is None or vd != axis or not _off_axis_eq(xs, ys, axis):
        return None
    return axis, xs, ys


# ---------------------------------------------------------------------------
#  1. Head packing: adjacent slices of one base re-join
#
#  cat(t[a:b], t[b:c], d) ≡ t[a:c] — the piecewise-read spelling of a
#  packed buffer reconstitutes the bigger view.  Exact under Python
#  slicing for any ints a≤b≤c (clamped effective bounds are monotone),
#  so no extent knowledge is needed — only contiguity and the shared
#  axis.  ``narrow`` is the start+length spelling: contiguity is
#  s2 == s1 + l1 and ``torch.narrow`` rejects out-of-range/negative
#  starts, so s1 ≥ 0 is required.
# ---------------------------------------------------------------------------


def _merge_axis(bound: dict) -> int | None:
    """Return the shared axis of a merge's three dims, or ``None``.

    The cat dim (``CD``) and both piece dims (``D1``/``D2``) must all
    normalise to the same axis of *t*.
    """
    ts = _shape(bound.get("t"))
    if not ts:
        return None
    n = len(ts)
    cd = _axis(bound.get("$attr:CD"), n)
    d1 = _axis(bound.get("$attr:D1"), n)
    d2 = _axis(bound.get("$attr:D2"), n)
    if cd is None or cd != d1 or cd != d2:
        return None
    return cd


def _check_slice_merge(bound: dict) -> bool:
    """cat(slice(t,d,a,b), slice(t,d,b,c), d) → slice(t,d,a,c) guards.

    All three dims must normalise to the same axis of *t*, and the
    piece bounds must be contiguous: ``a ≤ b(=e1=s2) ≤ c``.  ``None``
    starts read as 0 (the ``t[:b]`` spelling).
    """
    if _merge_axis(bound) is None:
        return False
    s1, e1 = bound.get("$attr:S1"), bound.get("$attr:E1")
    s2, e2 = bound.get("$attr:S2"), bound.get("$attr:E2")
    s1 = 0 if s1 is None else s1
    if not (
        isinstance(s1, int)
        and isinstance(e1, int)
        and isinstance(s2, int)
        and isinstance(e2, int)
    ):
        return False
    return s1 <= e1 == s2 <= e2


def _check_slice_merge_step(bound: dict) -> bool:
    """Require unit steps on the merge — strided pieces do not tile."""
    p1, p2 = bound.get("$attr:P1"), bound.get("$attr:P2")
    return (
        p1 in (None, 1)
        and p2 in (None, 1)
        and _check_slice_merge(bound)
    )


CAT_SLICE_MERGE = R(
    "cat_slice_merge",
    Op.make(
        "concat",
        Op.make("slice", "t", dim="D1", start="S1", end="E1"),
        Op.make("slice", "t", dim="D2", start="S2", end="E2"),
        dim="CD",
    ),
    Op.make("slice", "t", dim="D1", start="S1", end="E2"),
    check=_check_slice_merge,
    law="Adjacent slices of one base re-join: cat(t[a:b], t[b:c], d) "
    "≡ t[a:c] — per-head KV reads of a packed buffer reconstitute "
    "the wider view.  Exact under slice-clamp semantics for a≤b≤c.",
)

CAT_SLICE_MERGE_STEP = R(
    "cat_slice_merge_step",
    Op.make(
        "concat",
        Op.make(
            "slice", "t", dim="D1", start="S1", end="E1", step="P1"
        ),
        Op.make(
            "slice", "t", dim="D2", start="S2", end="E2", step="P2"
        ),
        dim="CD",
    ),
    Op.make("slice", "t", dim="D1", start="S1", end="E2"),
    check=_check_slice_merge_step,
    law="Unit-step spelling of the slice merge — a strided piece is "
    "not a contiguous tile and declines.",
)


def _check_narrow_merge(bound: dict) -> bool:
    """cat(narrow(t,d,a,l1), narrow(t,d,a+l1,l2), d) → narrow(a, l1+l2).

    ``torch.narrow`` raises on negative starts and over-long lengths,
    so contiguity is checked on non-negative bounds only.
    """
    if _merge_axis(bound) is None:
        return False
    s1, l1 = bound.get("$attr:S1"), bound.get("$attr:L1")
    s2, l2 = bound.get("$attr:S2"), bound.get("$attr:L2")
    if not (
        isinstance(s1, int)
        and isinstance(l1, int)
        and isinstance(s2, int)
        and isinstance(l2, int)
    ):
        return False
    return s1 >= 0 and l1 >= 0 and s2 == s1 + l1 and l2 >= 0


def _derive_narrow_merge(bound: dict) -> dict | None:
    """``length`` = l1 + l2 — the merged extent."""
    l1, l2 = bound.get("$attr:L1"), bound.get("$attr:L2")
    if not (isinstance(l1, int) and isinstance(l2, int)):
        return None
    return {"$attr:LT": l1 + l2}


CAT_NARROW_MERGE = R(
    "cat_narrow_merge",
    Op.make(
        "concat",
        Op.make("narrow", "t", dim="D1", start="S1", length="L1"),
        Op.make("narrow", "t", dim="D2", start="S2", length="L2"),
        dim="CD",
    ),
    Op.make("narrow", "t", dim="D1", start="S1", length="LT"),
    check=_check_narrow_merge,
    derive=_derive_narrow_merge,
    law="Narrow spelling of the merge: cat(narrow(t,a,l1), "
    "narrow(t,a+l1,l2), d) ≡ narrow(t,a,l1+l2).",
)


# ---------------------------------------------------------------------------
#  2. Packed → per-head recovery: a view covering exactly one cat
#     operand recovers it; a select on a stack returns its head.
# ---------------------------------------------------------------------------


def _check_cat_head_left(bound: dict) -> bool:
    """slice(cat(x,y,d), d, 0, ex) ≡ x — the read IS the left piece."""
    got = _cat_pair(bound)
    if got is None:
        return False
    axis, xs, _ys = got
    s, e = bound.get("$attr:S"), bound.get("$attr:E")
    ex = xs[axis]
    return s in (None, 0) and isinstance(ex, int) and e == ex


def _check_cat_head_right(bound: dict) -> bool:
    """slice(cat(x,y,d), d, ex, e) ≡ y — the tail read IS the right piece.

    The start must be exactly the left extent; the end may be the
    combined extent or beyond (``x[ex:]`` spells a to-end read — the
    int64 sentinel ≥ any extent counts too).
    """
    got = _cat_pair(bound)
    if got is None:
        return False
    axis, xs, ys = got
    s, e = bound.get("$attr:S"), bound.get("$attr:E")
    ex, ey = xs[axis], ys[axis]
    if not (isinstance(ex, int) and s == ex):
        return False
    if e is None:
        return True  # to-end read: always exactly the tail piece
    return isinstance(ey, int) and isinstance(e, int) and e >= ex + ey


def _check_cat_narrow_left(bound: dict) -> bool:
    """narrow(cat(x,y,d), d, 0, ex) ≡ x — narrow must be in-range."""
    got = _cat_pair(bound)
    if got is None:
        return False
    axis, xs, _ys = got
    s, le = bound.get("$attr:S"), bound.get("$attr:L")
    ex = xs[axis]
    return s == 0 and isinstance(ex, int) and le == ex


def _check_cat_narrow_right(bound: dict) -> bool:
    """narrow(cat(x,y,d), d, ex, ey) ≡ y — exact in-range tail."""
    got = _cat_pair(bound)
    if got is None:
        return False
    axis, xs, ys = got
    s, le = bound.get("$attr:S"), bound.get("$attr:L")
    ex, ey = xs[axis], ys[axis]
    return (
        isinstance(ex, int)
        and isinstance(ey, int)
        and s == ex
        and le == ey
    )


_CAT = Op.make("concat", "x", "y", dim="CD")
_SLICE_V = Op.make("slice", _CAT, dim="VD", start="S", end="E")
_NARROW_V = Op.make("narrow", _CAT, dim="VD", start="S", length="L")

CAT_HEAD_LEFT = R(
    "cat_head_left",
    _SLICE_V,
    "x",
    check=_check_cat_head_left,
    law="The left head of a packed buffer read back: cat(x,y,d)"
    "[:, :ex] ≡ x when the view covers exactly x's extent.",
)

CAT_HEAD_RIGHT = R(
    "cat_head_right",
    _SLICE_V,
    "y",
    check=_check_cat_head_right,
    law="The tail head of a packed buffer: cat(x,y,d)[:, ex:] ≡ y "
    "when the view starts exactly at x's extent and runs to the end.",
)

CAT_HEAD_LEFT_NARROW = R(
    "cat_head_left_narrow",
    _NARROW_V,
    "x",
    check=_check_cat_narrow_left,
    law="Narrow spelling of the left-head recovery.",
)

CAT_HEAD_RIGHT_NARROW = R(
    "cat_head_right_narrow",
    _NARROW_V,
    "y",
    check=_check_cat_narrow_right,
    law="Narrow spelling of the tail-head recovery.",
)


def _check_stack_axis(bound: dict):
    """Return ``(n+1, stack_axis)`` for a well-typed ``stack(a,b)``."""
    a, b = _shape(bound.get("a")), _shape(bound.get("b"))
    if not (a and b and len(a) == len(b)):
        return None
    if not all(_dim_eq(x, y) for x, y in zip(a, b, strict=True)):
        return None
    n1 = len(a) + 1
    sd = _axis(bound.get("$attr:SD"), n1)
    return (n1, sd) if sd is not None else None


def _check_stack_view(bound: dict) -> bool:
    """select/unbind(stack(a,b,sd), dd, i) — the view axis is sd."""
    got = _check_stack_axis(bound)
    if got is None:
        return False
    n1, sd = got
    return _axis(bound.get("$attr:DD"), n1) == sd


def _check_stack_getitem(bound: dict) -> bool:
    """getitem(stack(a,b,sd), i) — only a dim-0 stack indexes directly."""
    got = _check_stack_axis(bound)
    return got is not None and got[1] == 0


_STACK = Op.make("stack", "a", "b", dim="SD")
_STACK_HEAD_LAWS: list[Rewrite] = []
for _sel_op in ("select", "unbind"):
    for _i in (0, 1):
        _STACK_HEAD_LAWS.append(
            R(
                f"stack_{_sel_op}_head{_i}",
                Op.make(_sel_op, _STACK, dim="DD", index=_i),
                "a" if _i == 0 else "b",
                check=_check_stack_view,
                law=f"{_sel_op}(stack(a,b,d), d, {_i}) ≡ "
                f"{'a' if _i == 0 else 'b'} — a per-head read of a "
                "packed stack recovers the head.",
            )
        )
for _i in (0, 1):
    _STACK_HEAD_LAWS.append(
        R(
            f"stack_getitem_head{_i}",
            Op.make("getitem", _STACK, index=_i),
            "a" if _i == 0 else "b",
            check=_check_stack_getitem,
            law=f"stack(a,b,0)[{_i}] ≡ {'a' if _i == 0 else 'b'} — "
            "the dim-0 indexing spelling of the head recovery.",
        )
    )


def _check_cat_unsqueeze(bound: dict) -> bool:
    """cat(unsqueeze(a,d), unsqueeze(b,d), d) ≡ stack(a,b,d) guards.

    All three dims live on the rank-(n+1) unsqueezed tensors and must
    normalise to the same (inserted) axis; the operands must be
    stack-compatible.
    """
    a, b = _shape(bound.get("a")), _shape(bound.get("b"))
    if not (a is not None and b is not None and len(a) == len(b)):
        return False
    n1 = len(a) + 1
    d = _axis(bound.get("$attr:D"), n1)
    d2 = _axis(bound.get("$attr:D2"), n1)
    cd = _axis(bound.get("$attr:CD"), n1)
    if d is None or d != d2 or d != cd:
        return False
    return all(_dim_eq(x, y) for x, y in zip(a, b, strict=True))


STACK_FROM_CAT_UNSQUEEZE = R(
    "stack_from_cat_unsqueeze",
    Op.make(
        "concat",
        Op.make("unsqueeze", "a", dim="D"),
        Op.make("unsqueeze", "b", dim="D2"),
        dim="CD",
    ),
    Op.make("stack", "a", "b", dim="D"),
    check=_check_cat_unsqueeze,
    law="Two heads materialised apart pack into one buffer: "
    "cat(a.unsqueeze(d), b.unsqueeze(d), d) ≡ stack(a,b,d) — "
    "extent-2 along the inserted axis either way.",
)


# ---------------------------------------------------------------------------
#  3. Full-read identities — the merge's cleanup.
#
#  slice(t, d, 0, e) ≡ t when e covers the extent (a `x[:]` to-end
#  read or an over-long end both clamp to the whole axis);
#  narrow(t, d, 0, len) ≡ t only when len IS the extent (narrow
#  raises out of range).
# ---------------------------------------------------------------------------


def _check_slice_full(bound: dict) -> bool:
    """slice(t,d,s,e) ≡ t — a full-extent read is the base."""
    ts = _shape(bound.get("t"))
    d = _axis(bound.get("$attr:D"), len(ts) if ts else 0)
    if ts is None or d is None:
        return False
    s, e = bound.get("$attr:S"), bound.get("$attr:E")
    if s not in (None, 0):
        return False
    if e is None:
        return True
    ext = ts[d]
    return isinstance(ext, int) and isinstance(e, int) and e >= ext


def _check_slice_full_step(bound: dict) -> bool:
    """Require a unit step on the full-read spelling."""
    p = bound.get("$attr:P")
    return p in (None, 1) and _check_slice_full(bound)


def _check_slice_bare(bound: dict) -> bool:
    """slice(t,d) with no bounds is the identity on any real axis."""
    ts = _shape(bound.get("t"))
    return (
        ts is not None
        and len(ts) > 0
        and isinstance(bound.get("$attr:D"), int)
    )


def _check_narrow_full(bound: dict) -> bool:
    """narrow(t,d,0,ext) ≡ t — narrow must be exactly in-range."""
    ts = _shape(bound.get("t"))
    d = _axis(bound.get("$attr:D"), len(ts) if ts else 0)
    if ts is None or d is None:
        return False
    s, le = bound.get("$attr:S"), bound.get("$attr:L")
    ext = ts[d]
    return s == 0 and isinstance(ext, int) and le == ext


SLICE_FULL = R(
    "slice_full",
    Op.make("slice", "t", dim="D", start="S", end="E"),
    "t",
    check=_check_slice_full,
    law="A full-extent slice is the base: slice(t,d,0,e) ≡ t for "
    "e ≥ t.shape[d] (clamped) — the packed buffer read whole.",
)

SLICE_FULL_STEP = R(
    "slice_full_step",
    Op.make("slice", "t", dim="D", start="S", end="E", step="P"),
    "t",
    check=_check_slice_full_step,
    law="Unit-step spelling of the full-read identity.",
)

SLICE_BARE = R(
    "slice_bare_id",
    Op.make("slice", "t", dim="D"),
    "t",
    check=_check_slice_bare,
    law="slice(t,d) with no bounds is a one-axis ``[:]`` — the "
    "identity.",
)

NARROW_FULL = R(
    "narrow_full",
    Op.make("narrow", "t", dim="D", start="S", length="L"),
    "t",
    check=_check_narrow_full,
    law="narrow(t,d,0,ext) ≡ t — the whole axis is the base.",
)


# ---------------------------------------------------------------------------
#  4. Gather mobility: index_select slides across head views on a
#     DIFFERENT axis.
#
#  A gather on axis s commutes with any view that leaves s untouched:
#     index_select(slice(t,d,a,b), s, I) ≡ slice(index_select(t,s,I), d,a,b)
#  and likewise for narrow/select/unbind/stack/concat.  Views that
#  remove (select/unbind) or insert (stack) an axis shift the
#  coordinate — ``derive`` maps it.  Both directions are offered (the
#  shared-gather fold AND the per-head unroll) so extraction prices
#  them; both index spellings are covered (``index=`` attr and the
#  index-tensor operand).
# ---------------------------------------------------------------------------


def _check_gather_axes(bound: dict) -> bool:
    """Gather and view must act on DIFFERENT axes of the same rank."""
    ts = _shape(bound.get("t"))
    if not ts:
        return False
    n = len(ts)
    g = _axis(bound.get("$attr:GD"), n)
    v = _axis(bound.get("$attr:VD"), n)
    return g is not None and v is not None and g != v


def _gather(t: Any, dim: str, spec: str) -> Op:
    """Build an ``index_select`` pattern over *t*.

    ``spec="attr"`` binds the static ``index`` tuple; ``"tensor"``
    binds the index as the second tensor operand.
    """
    if spec == "attr":
        return Op.make("index_select", t, dim=dim, index="I")
    return Op.make("index_select", t, "idx", dim=dim)


def _check_gather_select(bound: dict) -> bool:
    """index_select(select(t,sd,i), s, I): s is a kept axis of t."""
    ts = _shape(bound.get("t"))
    if not (ts and len(ts) >= 2):
        return False
    n = len(ts)
    sd = _axis(bound.get("$attr:SD"), n)
    gd = _axis(bound.get("$attr:GD"), n - 1)
    return sd is not None and gd is not None


def _derive_gather_select(bound: dict) -> dict | None:
    """Map the post-removal gather axis into *t*'s coordinates."""
    ts = _shape(bound.get("t"))
    if not (ts and len(ts) >= 2):
        return None
    n = len(ts)
    sd = _axis(bound.get("$attr:SD"), n)
    gd = _axis(bound.get("$attr:GD"), n - 1)
    if sd is None or gd is None:
        return None
    return {"$attr:GT": gd if gd < sd else gd + 1}


def _check_gather_select_rev(bound: dict) -> bool:
    """select(index_select(t,g,I), sd, i): the gather axis survives."""
    ts = _shape(bound.get("t"))
    if not (ts and len(ts) >= 2):
        return False
    n = len(ts)
    sd = _axis(bound.get("$attr:SD"), n)
    gt = _axis(bound.get("$attr:GT"), n)
    return sd is not None and gt is not None and gt != sd


def _derive_gather_select_rev(bound: dict) -> dict | None:
    """Map the gather axis into the selected tensor's coordinates."""
    ts = _shape(bound.get("t"))
    if not (ts and len(ts) >= 2):
        return None
    n = len(ts)
    sd = _axis(bound.get("$attr:SD"), n)
    gt = _axis(bound.get("$attr:GT"), n)
    if sd is None or gt is None or gt == sd:
        return None
    return {"$attr:GD2": gt if gt < sd else gt - 1}


def _check_gather_stack(bound: dict) -> bool:
    """index_select(stack(x,y,sd), g, I): g is not the stacked axis."""
    a, b = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (a and b and len(a) == len(b)):
        return False
    n1 = len(a) + 1
    sd = _axis(bound.get("$attr:SD"), n1)
    gd = _axis(bound.get("$attr:GD"), n1)
    return (
        sd is not None
        and gd is not None
        and sd != gd
        and all(_dim_eq(u, v) for u, v in zip(a, b, strict=True))
    )


def _derive_gather_stack(bound: dict) -> dict | None:
    """Map the stacked gather axis into the operands' coordinates."""
    a = _shape(bound.get("x"))
    if not a:
        return None
    n1 = len(a) + 1
    sd = _axis(bound.get("$attr:SD"), n1)
    gd = _axis(bound.get("$attr:GD"), n1)
    if sd is None or gd is None or gd == sd:
        return None
    return {"$attr:GA": gd if gd < sd else gd - 1}


def _check_gather_stack_rev(bound: dict) -> bool:
    """stack(index_select(x,g,I), index_select(y,g,I), sd) — shared g."""
    a, b = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (a and b and len(a) == len(b)):
        return False
    if not all(_dim_eq(u, v) for u, v in zip(a, b, strict=True)):
        return False
    n1 = len(a) + 1
    sd = _axis(bound.get("$attr:SD"), n1)
    gt = _axis(bound.get("$attr:GT"), len(a))
    return sd is not None and gt is not None


def _derive_gather_stack_rev(bound: dict) -> dict | None:
    """Map the operand gather axis into the stacked coordinates."""
    a = _shape(bound.get("x"))
    if not a:
        return None
    n1 = len(a) + 1
    sd = _axis(bound.get("$attr:SD"), n1)
    gt = _axis(bound.get("$attr:GT"), len(a))
    if sd is None or gt is None:
        return None
    return {"$attr:GD2": gt if gt < sd else gt + 1}


def _check_gather_cat(bound: dict) -> bool:
    """index_select(cat(x,y,cd), g, I): g is not the cat axis."""
    xs, ys = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (xs and ys and len(xs) == len(ys)):
        return False
    n = len(xs)
    cd = _axis(bound.get("$attr:CD"), n)
    gd = _axis(bound.get("$attr:GD"), n)
    return (
        cd is not None
        and gd is not None
        and cd != gd
        and _off_axis_eq(xs, ys, cd)
    )


def _gather_view_laws() -> list[Rewrite]:
    """Build the gather/view commutations, both index spellings.

    The plain views (``slice`` ±step, ``narrow``) keep the raw dim —
    they preserve rank, so the bound axis metavariable is already a
    valid axis on the gathered tensor.  ``select``/``unbind`` drop an
    axis and ``stack`` inserts one — their derived coordinate maps
    live in the check/derive pair above.
    """
    rules: list[Rewrite] = []
    for spec, tag in (("attr", ""), ("tensor", "_t")):
        idx = "an index attr" if spec == "attr" else "an index tensor"
        # -- slice (unstrided / strided) ---------------------------------
        for step in (False, True):
            st = "_step" if step else ""
            sl_kw = {"step": "P"} if step else {}
            lhs_v = Op.make(
                "slice", "t", dim="VD", start="A", end="B", **sl_kw
            )
            rhs_v = Op.make(
                "slice",
                _gather("t", "GD", spec),
                dim="VD",
                start="A",
                end="B",
                **sl_kw,
            )
            rules += [
                R(
                    f"gather_slice{st}_out{tag}",
                    _gather(lhs_v, "GD", spec),
                    rhs_v,
                    check=_check_gather_axes,
                    law="A gather commutes with a slice on another "
                    f"axis: t[a:b,d][s,I] ≡ t[s,I][a:b,d] ({idx}).",
                ),
                R(
                    f"gather_slice{st}_in{tag}",
                    rhs_v,
                    _gather(lhs_v, "GD", spec),
                    check=_check_gather_axes,
                    law="Reverse: push the gather under the slice "
                    "(the per-head unroll).",
                ),
            ]
        # -- narrow ------------------------------------------------------
        lhs_v = Op.make("narrow", "t", dim="VD", start="A", length="L")
        rhs_v = Op.make(
            "narrow",
            _gather("t", "GD", spec),
            dim="VD",
            start="A",
            length="L",
        )
        rules += [
            R(
                f"gather_narrow_out{tag}",
                _gather(lhs_v, "GD", spec),
                rhs_v,
                check=_check_gather_axes,
                law="Gather commutes with narrow on another axis.",
            ),
            R(
                f"gather_narrow_in{tag}",
                rhs_v,
                _gather(lhs_v, "GD", spec),
                check=_check_gather_axes,
                law="Reverse narrow commutation.",
            ),
        ]
        # -- select / unbind (axis removed — derive maps the coord) ------
        for op in ("select", "unbind"):
            lhs_v = Op.make(op, "t", dim="SD", index="SI")
            rhs_v = Op.make(
                op,
                _gather("t", "GT", spec),
                dim="SD",
                index="SI",
            )
            rules += [
                R(
                    f"gather_{op}_out{tag}",
                    _gather(lhs_v, "GD", spec),
                    rhs_v,
                    check=_check_gather_select,
                    derive=_derive_gather_select,
                    law=f"Gather commutes with {op} on another axis "
                    "— the head view reads the shared packed gather.",
                ),
                R(
                    f"gather_{op}_in{tag}",
                    rhs_v,
                    _gather(lhs_v, "GD2", spec),
                    check=_check_gather_select_rev,
                    derive=_derive_gather_select_rev,
                    law=f"Reverse {op} commutation — the gather "
                    "pushes under the head view.",
                ),
            ]
        # -- stack (axis inserted — derive maps both ways) ---------------
        lhs_v = Op.make("stack", "x", "y", dim="SD")
        rhs_v = Op.make(
            "stack",
            _gather("x", "GA", spec),
            _gather("y", "GA", spec),
            dim="SD",
        )
        in_lhs = Op.make(
            "stack",
            _gather("x", "GT", spec),
            _gather("y", "GT", spec),
            dim="SD",
        )
        rules += [
            R(
                f"gather_stack_out{tag}",
                _gather(lhs_v, "GD", spec),
                rhs_v,
                check=_check_gather_stack,
                derive=_derive_gather_stack,
                law="A gather off the stacked axis distributes to "
                "the heads — one packed gather spelled per head.",
            ),
            R(
                f"gather_stack_in{tag}",
                in_lhs,
                _gather(lhs_v, "GD2", spec),
                check=_check_gather_stack_rev,
                derive=_derive_gather_stack_rev,
                law="Heads sharing one index batch the gather on "
                "the packed stack.",
            ),
        ]
        # -- concat (off-axis) -------------------------------------------
        lhs_v = Op.make("concat", "x", "y", dim="CD")
        rhs_v = Op.make(
            "concat",
            _gather("x", "GD", spec),
            _gather("y", "GD", spec),
            dim="CD",
        )
        rules += [
            R(
                f"gather_cat_out{tag}",
                _gather(lhs_v, "GD", spec),
                rhs_v,
                check=_check_gather_cat,
                law="A gather off the cat axis distributes into the "
                "pieces — per-head gathers share the packed table.",
            ),
            R(
                f"gather_cat_in{tag}",
                rhs_v,
                _gather(lhs_v, "GD", spec),
                check=_check_gather_cat,
                law="Per-piece gathers sharing one index batch "
                "into one gather over the packed buffer.",
            ),
        ]
    return rules


def _static_index(v: Any) -> bool:
    """Whether *v* is a non-empty static index list of ints."""
    return (
        isinstance(v, (tuple, list))
        and len(v) >= 1
        and all(
            isinstance(i, int) and not isinstance(i, bool) for i in v
        )
    )


def _check_gather_same_axis(bound: dict) -> bool:
    """cat(x[Ix], y[Iy], d) → cat(x,y,d)[Ix + ex+Iy] guards.

    Both index lists must be static int tuples, the gather axis must
    be the cat axis (metavar-equality already binds them to the same
    spelling), and the left extent must be concrete for the offset.
    """
    xs, ys = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (xs and ys and len(xs) == len(ys)):
        return False
    n = len(xs)
    cd = _axis(bound.get("$attr:CD"), n)
    if cd is None or not _off_axis_eq(xs, ys, cd):
        return False
    return (
        _static_index(bound.get("$attr:IX"))
        and _static_index(bound.get("$attr:IY"))
        and isinstance(xs[cd], int)
    )


def _derive_gather_same_axis(bound: dict) -> dict | None:
    """J = Ix + (ex + Iy) — the packed-axis index list."""
    xs = _shape(bound.get("x"))
    ix, iy = bound.get("$attr:IX"), bound.get("$attr:IY")
    if not (
        xs
        and isinstance(ix, (tuple, list))
        and isinstance(iy, (tuple, list))
    ):
        return None
    cd = _axis(bound.get("$attr:CD"), len(xs))
    if cd is None or not isinstance(xs[cd], int):
        return None
    ex = xs[cd]
    return {"$attr:J": tuple(ix) + tuple(ex + v for v in iy)}


GATHER_CAT_BATCH = R(
    "gather_cat_batch",
    Op.make(
        "concat",
        Op.make("index_select", "x", dim="CD", index="IX"),
        Op.make("index_select", "y", dim="CD", index="IY"),
        dim="CD",
    ),
    Op.make(
        "index_select",
        Op.make("concat", "x", "y", dim="CD"),
        dim="CD",
        index="J",
    ),
    check=_check_gather_same_axis,
    derive=_derive_gather_same_axis,
    law="The multi-head gather batch: cat(x[Ix], y[Iy], d) ≡ "
    "cat(x,y,d)[Ix + (ex+Iy)] — per-piece gathers on the packed "
    "axis fold into ONE gather whose index offsets the right "
    "piece's rows by the left extent.",
)


# ---------------------------------------------------------------------------
#  5. Shared-table lift: pointwise maps slide over a cat.
#
#  cat(f(x), f(y)) ≡ f(cat(x,y)) for elementwise f — the rope cos/sin
#  multiply happens ONCE over the packed KV instead of per head.  The
#  unary family is unconditional (a well-typed cat); the binary
#  family requires the shared operand to broadcast against BOTH
#  pieces AND the cat — a per-position table C that only fits the
#  halves is not the shared-buffer factor.
# ---------------------------------------------------------------------------


def _check_cat_pair(bound: dict) -> bool:
    """Check the cat pair is well-typed: same rank, off-axis dims."""
    xs, ys = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (xs and ys and len(xs) == len(ys)):
        return False
    axis = _axis(bound.get("$attr:CD"), len(xs))
    return axis is not None and _off_axis_eq(xs, ys, axis)


def _check_binary_cat(bound: dict) -> bool:
    """``cat(x⊙c, y⊙c) ≡ cat(x,y)⊙c`` — c broadcasts on the cat too.

    ``c`` must broadcast into both pieces AND the packed result: a
    per-head table whose cat-axis extent only covers one piece
    cannot multiply the packed buffer — that class declines.
    """
    xs, ys = _shape(bound.get("x")), _shape(bound.get("y"))
    if not (xs and ys and len(xs) == len(ys)):
        return False
    n = len(xs)
    axis = _axis(bound.get("$attr:CD"), n)
    if axis is None or not _off_axis_eq(xs, ys, axis):
        return False
    cs = _shape(bound.get("c"))
    if not isinstance(cs, tuple):
        return False
    ex, ey = xs[axis], ys[axis]
    merged = (
        ex + ey if isinstance(ex, int) and isinstance(ey, int) else None
    )
    cat_shape = (*xs[:axis], merged, *xs[axis + 1 :])
    return (
        broadcast(cs, xs) == xs
        and broadcast(cs, ys) == ys
        and broadcast(cs, cat_shape) == cat_shape
    )


_POINTWISE_UNARY: tuple[str, ...] = (
    "neg",
    "abs",
    "silu",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "exp",
    "sqrt",
    "rsqrt",
    "square",
    "log",
)

_POINTWISE_BINARY: tuple[str, ...] = ("add", "mul", "sub", "div")


def _table_lift_laws() -> list[Rewrite]:
    """Build the pointwise-into-cat lifts (unary + shared-operand)."""
    rules: list[Rewrite] = []
    for op in _POINTWISE_UNARY:
        rules.append(
            R(
                f"unary_cat_{op}",
                Op.make(
                    "concat",
                    Op.make(op, "x"),
                    Op.make(op, "y"),
                    dim="CD",
                ),
                Op.make(op, Op.make("concat", "x", "y", dim="CD")),
                check=_check_cat_pair,
                law=f"{op} is elementwise: cat({op}(x), {op}(y)) ≡ "
                f"{op}(cat(x,y)) — the map runs once over the "
                "packed buffer.",
            )
        )
    for op in _POINTWISE_BINARY:
        rules.append(
            R(
                f"binary_cat_{op}",
                Op.make(
                    "concat",
                    Op.make(op, "x", "c"),
                    Op.make(op, "y", "c"),
                    dim="CD",
                ),
                Op.make(
                    op,
                    Op.make("concat", "x", "y", dim="CD"),
                    "c",
                ),
                check=_check_binary_cat,
                law=f"Shared factor lifts over the pack: cat(x{op}c, "
                f"y{op}c) ≡ cat(x,y){op}c — a shared table (RoPE "
                "cos/sin, scales) materialises once against the "
                "packed buffer.  c must broadcast on the cat axis "
                "too.",
            )
        )
    return rules


# ---------------------------------------------------------------------------
#  Rule collections
# ---------------------------------------------------------------------------

#: Slice/narrow merges, the pack/recover views, and the stack pack.
HEAD_PACK_LAWS: list[Rewrite] = [
    CAT_SLICE_MERGE,
    CAT_SLICE_MERGE_STEP,
    CAT_NARROW_MERGE,
    CAT_HEAD_LEFT,
    CAT_HEAD_RIGHT,
    CAT_HEAD_LEFT_NARROW,
    CAT_HEAD_RIGHT_NARROW,
    *_STACK_HEAD_LAWS,
    STACK_FROM_CAT_UNSQUEEZE,
]

#: Full-read identities — the merge's cleanup.
VIEW_ID_LAWS: list[Rewrite] = [
    SLICE_FULL,
    SLICE_FULL_STEP,
    SLICE_BARE,
    NARROW_FULL,
]

#: index_select mobility across head views (both directions, both
#: index spellings).
GATHER_VIEW_LAWS: list[Rewrite] = _gather_view_laws()

#: Gather over/across a cat — the off-axis commute and the
#: same-axis batch fold.
GATHER_CAT_LAWS: list[Rewrite] = [
    r for r in GATHER_VIEW_LAWS if "gather_cat" in r.name
] + [GATHER_CAT_BATCH]

#: Shared-table lifts — rope cos/sin and friends over the pack.
TABLE_LIFT_LAWS: list[Rewrite] = _table_lift_laws()

#: The whole decode-geometry slice.
DECODE_GEOM_LAWS: list[Rewrite] = [
    *HEAD_PACK_LAWS,
    *VIEW_ID_LAWS,
    *GATHER_VIEW_LAWS,
    GATHER_CAT_BATCH,
    *TABLE_LIFT_LAWS,
]

#: The same set as a composable :class:`~catopt_core.laws.RuleSet`.
DECODE_GEOM_RULES = RuleSet(
    "decode_geom",
    tuple(DECODE_GEOM_LAWS),
    description=(
        "decode-memory geometry laws — head packing, shared-index "
        "gather mobility, and shared-table lifts"
    ),
)
