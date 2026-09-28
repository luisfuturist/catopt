# ruff: noqa: RUF001, RUF002, RUF003 -- law strings and docstrings use
# mathematical notation (≡, Δ, −, ∘) deliberately; ASCII would misstate
# it.
"""Decode-time memory-traffic laws — the bandwidth-bound rewrite slice.

A decode step is bandwidth-bound: every generated token reads the whole
KV cache and every weight exactly once.  The rewrites here expose the
traffic-saving moves a decode scheduler wants — a cache append spelled
as a WRITE into a preallocated buffer instead of a re-concatenation, a
gather over a deduplicated row set, and the ``repeat_kv`` head-copy
seen as a reindexing — all as *exact* equational laws over the
existing op table.  No new ops, no new bindings: every RHS mints terms
the 252-binding ``OpTable.full()`` already lowers.

OP-TABLE AUDIT — what the IR can express
----------------------------------------
* **Buffer writes** — no in-place ``index_copy_``/``copy_`` generator
  exists (the IR is functional), but the scatter family is complete:
  ``slice_scatter`` / ``select_scatter`` / ``index_put`` / ``scatter``
  / ``copy``.  The export boundary already functionalizes
  ``copy_(slice(buf, …), src)`` into ``slice_scatter``
  (``torch_bridge._handle_copy_``), so the expressible law is the
  *bridge*: ``cat(prefix_view(buf), new)`` ≡
  ``slice_scatter(buf, new, n, n+m)`` — family 1, both directions.
* **Attention over a concatenated cache (read side)** — the ``sdpa``
  binding takes a fixed ``(q, k, v[, mask])`` signature; there is no
  multi-tensor key/value operand.  The split-concat transform already
  exists as ``catopt_carriers.om.SDPA_CAT_LAWS``: the online-softmax
  carrier chunks the attention across the k/v blocks, which IS the
  decode read split.  Not duplicated here.
* **Token dedup** — ``unique`` / inverse-index primitives are ABSENT
  from the table, and ``embedding``'s ``idx`` arrives as a tensor
  operand whose *values* a rewrite cannot inspect, so
  ``embed(unique(idx))[inv]`` is unexpressible as stated.  The
  expressible fragment is the static index: ``index_select``'s
  ``index`` attr is an int tuple dedup-able at match time (law 2) —
  the same gather-of-gather identity, proven statically.
* **repeat_interleave / repeat_kv** — absent as a primitive, but its
  expansion chain ``reshape∘expand∘unsqueeze`` is the matched form
  (``catopt_core.laws.tensor._check_repeat_chain`` proves the chain is
  a copy map), and ``index_select`` expresses the same reindex (law
  3).  ``GQA_ABSORB`` already handles the sdpa-consumer case; law 3
  exposes the copy as a gather for every other consumer.
* **Weight sharing** — ``share_duplicate_params`` /
  ``share_duplicate_param_slices`` already discover EXACT duplicates
  (bitwise/elementwise-equal params and per-head slices); near-
  duplicates are certified approximations, not equalities, and stay
  out of a semantics-preserving law set by design.

LAWS (:data:`DECODE_LAWS`)
--------------------------
1. ``kv_append_scatter_*`` / ``scatter_to_cat_*`` —
   ``cat(narrow(buf,d,0,n) | slice(buf,d,0,n[,1]), new, d)`` ≡
   ``slice_scatter(buf, new, d, n, n+m)``.  Exact iff the buffer's tail
   IS the write region (``buf.shape[d] == n + m``): a longer buffer
   declines, since ``slice_scatter`` would keep live tail cells the
   cat never carried.  Both directions are offered so eqsat weighs the
   read spelling against the write spelling.
2. ``index_select_dedup`` — ``t[I] ≡ t[U][inv]`` for a repeated static
   index (the gather-of-gather homomorphism, ``U`` first-occurrence
   uniques, ``inv`` the position map).
3. ``repeat_kv_as_gather`` — the copy diagonal Δ_r
   (unsqueeze→expand→reshape) as the index map ``i ↦ i // r``.
4. ``index_select_id`` — an identity gather is the identity (the
   cleanup the dedup/repeat compositions need to converge).

Not implemented (documented limits): in-place ``copy_`` semantics per
se — the IR is pure; the buffer-write law covers its functional
image.  Dynamic token dedup — needs a ``unique`` op and inspectable
index values.  ``select_scatter`` single-row append — expressible via
``unsqueeze``+``slice_scatter`` but left out of this verified slice.
"""

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws import R
from catopt_core.laws.tensor import _check_repeat_chain

__all__ = [
    "DECODE_LAWS",
    "INDEX_SELECT_DEDUP",
    "INDEX_SELECT_ID",
    "KV_APPEND_SCATTER_LAWS",
    "REPEAT_AS_GATHER",
    "SCATTER_TO_CAT_LAWS",
]


def _dshape(t: Any):
    """Return the value shape of a bound term — carrier-aware.

    Delegates to :func:`catopt_carriers.xcarrier._xshape`, the same
    resolver ``catopt_carriers.om``'s checks use: a metavariable may
    resolve to a carrier member whose *cost-convention* shape is not
    the tensor value shape.
    """
    from catopt_carriers.xcarrier import _xshape

    return _xshape(t)


def _dim_eq(a: Any, b: Any) -> bool:
    """Axis compatibility: equal, or either side unknown (None)."""
    return a is None or b is None or a == b


# ---------------------------------------------------------------------------
#  1. KV-cache append:  cat(prefix_view(buf), new) ≡ slice_scatter
#
#  A decode append reads the cache as a PREFIX view of a preallocated
#  slot buffer and concatenates the new K/V — ``cat(buf[:n], new, d)``.
#  The functional write-back spelling is ``slice_scatter(buf, new, d,
#  n, n+m)``: positions < n keep buf (the cache), positions [n, n+m)
#  take the new rows.  The two agree EXACTLY iff the buffer's extent on
#  the cat axis is n + m — the write must run to the buffer's end.
# ---------------------------------------------------------------------------


def _check_cat_prefix(bound: dict) -> bool:
    """cat(prefix_view(buf, d), new, d) → slice_scatter conditions.

    The view must be an unstrided prefix read (start 0) along the cat
    axis, the view axis must BE the cat axis, and the buffer's tail
    must equal ``new``'s extent on that axis — otherwise
    ``slice_scatter`` preserves live tail cells the cat never carries.
    """
    cd, vd = bound.get("$attr:CD"), bound.get("$attr:VD")
    start = bound.get("$attr:VS")
    n = bound.get("$attr:VL")  # narrow's length / slice's end
    step = bound.get("$attr:VST")  # absent on the no-step slice
    if step not in (None, 1):
        return False  # a strided read is not a prefix read
    if start not in (None, 0):
        return False  # only a prefix view spells the cache read
    if not (
        isinstance(cd, int)
        and isinstance(vd, int)
        and isinstance(n, int)
        and n >= 0
    ):
        return False
    bs, ns = _dshape(bound.get("buf")), _dshape(bound.get("new"))
    if not (isinstance(bs, tuple) and isinstance(ns, tuple)):
        return False
    if not bs or len(ns) != len(bs):
        return False
    dn = cd % len(bs)
    if vd % len(bs) != dn:
        return False  # the view must read along the cat axis
    bd, m = bs[dn], ns[dn]
    if not (isinstance(bd, int) and isinstance(m, int)):
        return False  # cannot prove the write reaches the buffer end
    if bd != n + m:
        return False  # the buffer tail must be exactly the write region
    return all(_dim_eq(bs[i], ns[i]) for i in range(len(bs)) if i != dn)


def _derive_scatter_end(bound: dict) -> dict | None:
    """``end`` = the buffer's extent on the cat axis (== n + m).

    ``_check_cat_prefix`` already proved ``buf.shape[d] == n + m``;
    the derive re-verifies so a standalone replay stays safe.
    """
    bs, cd = _dshape(bound.get("buf")), bound.get("$attr:CD")
    if not (isinstance(bs, tuple) and bs and isinstance(cd, int)):
        return None
    bd = bs[cd % len(bs)]
    if not isinstance(bd, int):
        return None
    return {"$attr:SE": bd}


#: The shared RHS — write ``new`` into buf's tail [n, n+m).
_SCATTER_RHS = Op.make(
    "slice_scatter",
    "buf",
    "new",
    dim="CD",
    start="VL",
    end="SE",
    step=1,
)

#: LHS spellings: the cache prefix as ``narrow`` or ``slice`` (the
#: step attr may be spelled or defaulted at the boundary).
_CAT_PREFIX_LHS: tuple[tuple[str, Op], ...] = (
    (
        "narrow",
        Op.make(
            "concat",
            Op.make("narrow", "buf", dim="VD", start="VS", length="VL"),
            "new",
            dim="CD",
        ),
    ),
    (
        "slice",
        Op.make(
            "concat",
            Op.make(
                "slice",
                "buf",
                dim="VD",
                start="VS",
                end="VL",
                step="VST",
            ),
            "new",
            dim="CD",
        ),
    ),
    (
        "slice_ns",
        Op.make(
            "concat",
            Op.make("slice", "buf", dim="VD", start="VS", end="VL"),
            "new",
            dim="CD",
        ),
    ),
)

KV_APPEND_SCATTER_LAWS: list[Rewrite] = [
    R(
        f"kv_append_scatter_{tag}",
        lhs,
        _SCATTER_RHS,
        check=_check_cat_prefix,
        derive=_derive_scatter_end,
        law="KV append as a buffer write: cat(buf[:n], new, d) ≡ "
        "slice_scatter(buf, new, d, n, n+m) when buf.shape[d] == n+m — "
        "the cache prefix is preserved, the tail IS the write region.",
    )
    for tag, lhs in _CAT_PREFIX_LHS
]


def _check_scatter_cat(bound: dict) -> bool:
    """slice_scatter(buf, src, d, s, e) → cat(buf[:s], src) conditions.

    Only a TAIL write collapses to a cat: ``e`` must be the buffer's
    extent (positions past ``e`` keep buf's values and no cat can
    express them), ``src`` must fill the slice exactly
    (``src.shape[d] == e − s``), and the off-axis extents must agree.
    """
    cd = bound.get("$attr:CD")
    s0, se = bound.get("$attr:S0"), bound.get("$attr:SE")
    st = bound.get("$attr:ST")  # absent on the no-step spelling
    if st not in (None, 1):
        return False  # a strided write is not a contiguous append
    if not (
        isinstance(cd, int)
        and isinstance(s0, int)
        and isinstance(se, int)
        and 0 <= s0 <= se
    ):
        return False
    bs, ss = _dshape(bound.get("buf")), _dshape(bound.get("src"))
    if not (
        isinstance(bs, tuple)
        and isinstance(ss, tuple)
        and bs
        and len(ss) == len(bs)
    ):
        return False
    dn = cd % len(bs)
    bd, m = bs[dn], ss[dn]
    if not (isinstance(bd, int) and isinstance(m, int)):
        return False  # cannot prove the write reaches the buffer end
    if se != bd:
        return False  # the slice must run to the buffer's end
    if m != se - s0:
        return False  # src must fill the slice exactly
    return all(_dim_eq(bs[i], ss[i]) for i in range(len(bs)) if i != dn)


#: The shared RHS — the write read back as prefix-view concat.
_CAT_TAIL_RHS = Op.make(
    "concat",
    Op.make("slice", "buf", dim="CD", start=0, end="S0"),
    "src",
    dim="CD",
)

_SCATTER_LHS: tuple[tuple[str, Op], ...] = (
    (
        "step",
        Op.make(
            "slice_scatter",
            "buf",
            "src",
            dim="CD",
            start="S0",
            end="SE",
            step="ST",
        ),
    ),
    (
        "nostep",
        Op.make(
            "slice_scatter",
            "buf",
            "src",
            dim="CD",
            start="S0",
            end="SE",
        ),
    ),
)

SCATTER_TO_CAT_LAWS: list[Rewrite] = [
    R(
        f"scatter_to_cat_{tag}",
        lhs,
        _CAT_TAIL_RHS,
        check=_check_scatter_cat,
        law="The buffer write read back as views: slice_scatter(buf, "
        "src, d, s, e) ≡ cat(buf[:s], src, d) when e == buf.shape[d] — "
        "the tail append re-spelled for read-side consumers.",
    )
    for tag, lhs in _SCATTER_LHS
]


# ---------------------------------------------------------------------------
#  2. Gather dedup:  index_select(t, d, I) ≡ index_select(t, d, U)[inv]
#
#  Reindexing composes: t[U[inv[i]]] = t[I[i]] whenever U[inv[i]] == I[i]
#  — take U as I's first-occurrence uniques and inv the position map.
#  The inner gather touches each surviving row once; the outer expands
#  the |U|-row table back to |I|.  The dynamic-token analogue
#  (embedding over unique(idx)) needs a ``unique`` op the table lacks
#  AND index values a rewrite cannot see — only the static-index form
#  is expressible.
# ---------------------------------------------------------------------------


def _check_dedup_index(bound: dict) -> bool:
    """Require the static index to be a repeated int sequence.

    Dedup is only worth offering when it strictly shrinks the gathered
    axis; a repeat-free index would add an op for nothing.
    """
    idx = bound.get("$attr:I")
    return (
        isinstance(idx, (tuple, list))
        and len(idx) >= 2
        and all(
            isinstance(i, int) and not isinstance(i, bool) for i in idx
        )
        and len(set(idx)) < len(idx)
    )


def _derive_dedup_index(bound: dict) -> dict:
    """U = first-occurrence uniques; inv[i] = I[i]'s slot in U."""
    pos: dict[int, int] = {}
    uniq: list[int] = []
    inv: list[int] = []
    for i in tuple(bound["$attr:I"]):
        j = pos.get(i)
        if j is None:
            j = len(uniq)
            pos[i] = j
            uniq.append(i)
        inv.append(j)
    return {"$attr:U": tuple(uniq), "$attr:VI": tuple(inv)}


INDEX_SELECT_DEDUP = R(
    "index_select_dedup",
    Op.make("index_select", "t", dim="D", index="I"),
    Op.make(
        "index_select",
        Op.make("index_select", "t", dim="D", index="U"),
        dim="D",
        index="VI",
    ),
    check=_check_dedup_index,
    derive=_derive_dedup_index,
    law="Gather-of-gather: t[I] ≡ t[U][inv] — deduplicate the gathered "
    "axis to its first-occurrence uniques, then re-expand.  The decode "
    "reading: the inner select touches |U| rows of the table where the "
    "repeated index would have recomputed |I|.",
)


# ---------------------------------------------------------------------------
#  3. repeat_kv as a gather:  reshape(expand(unsqueeze(k, d), ES), RS)
#     ≡ index_select(k, d−1, (i // r))
#
#  The unsqueeze→expand→reshape chain is the diagonal Δ_r — a copy
#  map, i.e. repeat_interleave on the merged axis.  Reindexing IS the
#  same map: out[…, n, …] = k[…, n // r, …].  Offering the index_select
#  spelling makes the head copy visible to every gather-aware law
#  (dedup, identity elimination) and consumer — not only to
#  GQA_ABSORB's sdpa site.
# ---------------------------------------------------------------------------


def _check_repeat_gather(bound: dict) -> bool:
    """Require the chain to be repeat_interleave on dim UDk − 1."""
    return _check_repeat_chain(bound, "k")


def _derive_repeat_gather(bound: dict) -> dict | None:
    """Derive index = (i // r) on the merged axis; dim = that axis.

    The merged axis is d−1 of k (the check proves d ≥ 1); its extent h
    and the repeat factor r = ES[d] must be concrete to mint the map.
    """
    ks = _dshape(bound.get("k"))
    d, es = bound.get("$attr:UDk"), bound.get("$attr:ESk")
    if not (
        isinstance(ks, tuple)
        and isinstance(d, int)
        and isinstance(es, tuple)
    ):
        return None
    dn = d % (len(ks) + 1)
    if dn == 0:
        return None
    h, r = ks[dn - 1], es[dn]
    if not (isinstance(h, int) and isinstance(r, int) and r > 1):
        return None
    return {
        "$attr:GD": dn - 1,
        "$attr:GI": tuple(i // r for i in range(h * r)),
    }


REPEAT_AS_GATHER = R(
    "repeat_kv_as_gather",
    Op.make(
        "reshape",
        Op.make(
            "expand",
            Op.make("unsqueeze", "k", dim="UDk"),
            shape="ESk",
        ),
        shape="RSk",
    ),
    Op.make("index_select", "k", dim="GD", index="GI"),
    check=_check_repeat_gather,
    derive=_derive_repeat_gather,
    law="The copy diagonal Δ_r as a reindex: repeat_interleave(k, r, "
    "d−1) ≡ index_select(k, d−1, (i // r)).  The gather spelling "
    "exposes the head duplication outside GQA_ABSORB's sdpa site.",
)


# ---------------------------------------------------------------------------
#  4. Identity gather:  index_select(t, d, (0..n−1)) ≡ t
#
#  The cleanup law — dedup's inner select on an already-distinct axis
#  is an identity gather, and so is the repeat-chain inner select
#  after Δ_r is factored through it.
# ---------------------------------------------------------------------------


def _check_identity_index(bound: dict) -> bool:
    """Require the index to be range(t.shape[d]) — a provable identity."""
    idx, d = bound.get("$attr:I"), bound.get("$attr:D")
    ts = _dshape(bound.get("t"))
    if not (
        isinstance(idx, (tuple, list))
        and isinstance(d, int)
        and isinstance(ts, tuple)
        and ts
    ):
        return False
    n = ts[d % len(ts)]
    return isinstance(n, int) and tuple(idx) == tuple(range(n))


INDEX_SELECT_ID = R(
    "index_select_id",
    Op.make("index_select", "t", dim="D", index="I"),
    "t",
    check=_check_identity_index,
    law="An identity index map is the identity morphism: "
    "index_select(t, d, (0..n−1)) ≡ t.",
)


#: The whole decode slice — buffer-append bridging, gather dedup,
#: repeat-as-gather, and the identity-gather cleanup.
DECODE_LAWS: list[Rewrite] = [
    *KV_APPEND_SCATTER_LAWS,
    *SCATTER_TO_CAT_LAWS,
    INDEX_SELECT_DEDUP,
    REPEAT_AS_GATHER,
    INDEX_SELECT_ID,
]
