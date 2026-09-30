"""Structurally-special weight offers — the exact-elision pass.

A stored weight can be *structurally* special in ways no equational
law can see: dead rows (output neurons that always produce zeros),
dead columns (input features never read), duplicated output slices
(the same neuron computed twice), a diagonal or identity matrix, or
a block-diagonal stack.  Each is an EXACT fact — zeros compute zeros,
a gather of a duplicated row reproduces it bitwise — so the offers
carry ``error_bound=0.0``, not a residual.

:func:`offer_weight_specials` is a non-local pass in the
:mod:`catopt_core.laws.pairing` family, a sibling of
:func:`catopt_core.laws.factored.offer_low_rank_factors`: it scans
projection e-nodes (``matmul(data, W)`` / ``linear(data, W[, b])``),
inspects the weight's *stored value*, and offers specialised members
into the consumer's e-class under a pointwise witness:

* **identity** — ``W = I`` collapses the projection to the input
  itself (a present bias re-added as ``add(x, b)``);
* **diagonal** — ``W = diag(d)`` collapses to ``mul(x, d)``;
* **zero** — ``W = 0`` collapses to a zeros member
  (``mul(index_select(x, -1, 0…0), 0)`` — exact regardless of the
  data's rank or leading shape);
* **elide** — bitwise-dead input slices are gathered away on the
  data side (``index_select(x, -1, keep)``) and bitwise-duplicate
  output slices are computed once then re-expanded
  (``index_select(y', -1, imap)``), around a shrunk derived weight
  ``W' = W[firsts][:, keep]`` — dead rows are just the all-zero
  duplicate group, so one mechanism covers pruning dead neurons AND
  ValueTie-style row duplication;
* **block_diag** — a contiguous diagonal-ordered block partition
  splits the projection into per-block projections on ``split``
  input slices, concatenated.

Derived parameters are registered into ``source_tensors`` (the
``share_duplicate_param_slices`` convention) so lowering materialises
them like any weight.  Extraction decides: every offer is strictly
flop-cheaper than the dense projection by construction, and under
launch-aware pricing the extra gathers/splits honestly lose when the
weight is too small to bother — the offer is never forced.

Exactness is *value-level*: a slice is dead iff every element is
exactly ``0.0``, and slices deduplicate on bitwise content — no
tolerance, no "small".  Out of scope (same boundaries as the
low-rank pass): left-weight ``matmul(W, data)``, non-leaf weight
expressions, and non-2-D weights.
"""

# ruff: noqa: RUF003 -- comments/docstrings use
# mathematical notation (×, ∘, ≠) deliberately.

from typing import Any

from catopt_core.ir import Const, Param, TensorType
from catopt_core.laws.factored import (
    _cls_has_var,
    _detach,
    _fresh_name,
    _leaf_params,
    _site_orient,
)
from catopt_core.laws.pairing import _exact_equal, _is_tensor

__all__ = ["offer_weight_specials"]

_PROV = "weight_special"


# ---------------------------------------------------------------------------
#  Value analysis — bitwise signatures, spans, dedup, block structure
# ---------------------------------------------------------------------------


def _sig(s: Any) -> bytes:
    """Bitwise content signature of one weight slice.

    Same convention as ``share_duplicate_param_slices``: raw bytes
    are stricter than ``torch.equal`` (``-0.0``/``+0.0`` and NaN
    payloads stay distinct) — dedup is therefore always exact.
    """
    return _detach(s).cpu().contiguous().numpy().tobytes()


def _row_spans(w: Any, o: int, i: int) -> list[tuple[int, int] | None]:
    """First/last nonzero column per row; ``None`` for dead rows.

    A row is dead iff every element is exactly ``0.0`` (IEEE — a
    ``-0.0`` row counts, and still reproduces bitwise through the
    gather member).  The span is computed by cumsum position
    bookkeeping, not a Python element scan.
    """
    spans: list[tuple[int, int] | None] = []
    for r in range(o):
        m = _detach(w[r]) != 0
        if not bool(m.any()):
            spans.append(None)
            continue
        c = m.cumsum(0)
        lo = int((c == 0).sum())
        # positions with cumsum == total are the trailing run from
        # the last nonzero on: hi = i - len(run) + 1.
        hi = i - int((c == int(m.sum())).sum()) + 1
        spans.append((lo, hi))
    return spans


def _dedup(
    sigs: list[bytes],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """First-occurrence index map + representative row positions."""
    uniq: list[bytes] = []
    imap: list[int] = []
    for s in sigs:
        try:
            imap.append(uniq.index(s))
        except ValueError:
            imap.append(len(uniq))
            uniq.append(s)
    firsts = tuple(imap.index(u) for u in range(len(uniq)))
    return tuple(imap), firsts


def _components(spans: list) -> list[list]:
    """Merge rows with overlapping nonzero spans into components.

    Interval-merge on spans sorted by their left edge: a row joins the
    running component while its span overlaps it.  The result is the
    connected components of the row/col bipartite support graph —
    over-merging is harmless (a merged "block" is still exact, just
    coarser) and under-merging cannot happen: two rows sharing a
    column have overlapping spans by definition.
    """
    live = sorted(
        (r for r, s in enumerate(spans) if s is not None),
        key=lambda r: spans[r][0],
    )
    comps: list[list] = []  # [row_indices, lo, running_hi]
    for r in live:
        lo, hi = spans[r]
        if comps and lo < comps[-1][2]:
            comps[-1][0].append(r)
            comps[-1][2] = max(comps[-1][2], hi)
        else:
            comps.append([[r], lo, hi])
    return comps


def _blocks(
    spans: list, o: int, i: int
) -> list[tuple[int, int, int, int]] | None:
    """Diagonal-ordered contiguous block partition, or ``None``.

    Returns ``(r0, r1, c0, c1)`` per block when ≥2 components tile the
    rows of ``[0, o)`` contiguously and in order.  Column slices are
    derived from component left edges — leading/interior dead columns
    fold into a neighbouring block's slice (they multiply by zero
    either way), so the col ranges tile ``[0, i)``.  Dead *rows*
    break the tiling and decline the offer — the elide member covers
    that case instead.  The construction is self-certifying: every
    row's span lies inside its block's col slice, so the per-block
    submatrices reassemble the weight exactly.
    """
    comps = _components(spans)
    if len(comps) < 2:
        return None
    blocks: list[tuple[int, int, int, int]] = []
    r_edge = 0
    for rows, lo, hi in comps:
        if sorted(rows) != list(range(r_edge, r_edge + len(rows))):
            return None
        blocks.append((r_edge, r_edge + len(rows), lo, hi))
        r_edge += len(rows)
    if r_edge != o:
        return None
    bounds = [0] + [b[2] for b in blocks[1:]] + [i]
    return [
        (b[0], b[1], bounds[j], bounds[j + 1])
        for j, b in enumerate(blocks)
    ]


def _analyse(wn: Any, o: int, i: int) -> dict:
    """Slice-level structure of a normalized ``(out, in)`` weight.

    ``wn`` is the weight with output slices as rows and input slices
    as columns — ``w`` for ``linear``, ``w.T`` for ``matmul``.  One
    analysis serves every member builder: output-slice byte
    signatures dedupe duplicate/dead rows, per-column nonzero checks
    find dead inputs, and row spans drive the diagonal/block tests.
    """
    spans = _row_spans(wn, o, i)
    sigs = [_sig(wn[r : r + 1]) for r in range(o)]
    imap, firsts = _dedup(sigs)
    keep = tuple(
        c for c in range(i) if bool((_detach(wn[:, c]) != 0).any())
    )
    diag = o == i and all(s == (j, j + 1) for j, s in enumerate(spans))
    return {
        "o": o,
        "i": i,
        "imap": imap,
        "firsts": firsts,
        "keep": keep,
        "diag": diag,
        "ident": diag and bool((_detach(wn.diagonal()) == 1).all()),
        "zero": all(s is None for s in spans),
        # A diagonal weight IS 1×1-block-diagonal, but the pointwise
        # mul member strictly dominates o tiny GEMMs — don't compute.
        "blocks": _blocks(spans, o, i) if not diag else None,
    }


# ---------------------------------------------------------------------------
#  Member builders — each returns (kind, eid, extra) or None
# ---------------------------------------------------------------------------


def _op(orient: str) -> str:
    """Projection op name for an orientation tag."""
    return "linear" if orient == "linear" else "matmul"


def _reg(eg: Any, source_tensors: dict, want: str, value: Any):
    """Register one derived param value; return ``(eid, name)``.

    A name already carrying bitwise-identical values is reused (the
    ``_register_factor`` convention); a poisoned name takes a ``_n``
    suffix.  The stored tensor is cloned out of any view so the
    derived parameter owns its storage.
    """
    name = want
    if not (
        want in source_tensors
        and _exact_equal(source_tensors[want], value)
    ):
        name = _fresh_name(source_tensors, want)
        source_tensors[name] = _detach(value).clone()
    eid = eg.add_term(
        Param(name, TensorType(tuple(int(d) for d in value.shape))),
        provenance=_PROV,
    )
    return eid, name


def _gather(eg: Any, cid: int, index: tuple) -> int:
    """``index_select(., -1, index)`` — last-axis gather/reindex."""
    return eg.add_enode(
        "index_select",
        (cid,),
        {"dim": -1, "index": tuple(index)},
        provenance=_PROV,
    )


def _bias_add(eg: Any, eid: int, node: Any) -> int:
    """Re-add a present ``linear`` bias outside the rewritten member."""
    if node.op == "linear" and len(node.children) == 3:
        return eg.add_enode(
            "add",
            (eid, eg.find(node.children[2])),
            {},
            provenance=_PROV,
        )
    return eid


def _m_identity(eg: Any, node: Any, data_c: int) -> tuple:
    """``W = I``: the projection IS the input (plus any bias)."""
    return (
        "identity",
        _bias_add(eg, eg.find(data_c), node),
        {},
    )


def _m_diag(
    eg: Any, node: Any, data_c: int, p: Param, wn: Any, tensors: dict
) -> tuple:
    """``W = diag(d)``: the projection is a pointwise scale."""
    d_eid, d_name = _reg(
        eg, tensors, f"{p.name}__diag", _detach(wn.diagonal())
    )
    core = eg.add_enode(
        "mul", (eg.find(data_c), d_eid), {}, provenance=_PROV
    )
    return ("diagonal", _bias_add(eg, core, node), {"d_param": d_name})


def _m_zero(eg: Any, node: Any, data_c: int, o: int) -> tuple:
    """``W = 0``: exact zeros, spelled shape-agnostically.

    ``index_select`` of the first input column ``o`` times gives a
    ``(…, o)`` tensor of the data's own shape and dtype without
    knowing its leading dims; ``mul(·, 0)`` zeroes it.
    """
    g = _gather(eg, eg.find(data_c), (0,) * o)
    c0 = eg.add_term(Const(0.0), provenance=_PROV)
    core = eg.add_enode("mul", (g, c0), {}, provenance=_PROV)
    return ("zero", _bias_add(eg, core, node), {})


def _m_elide(
    eg: Any,
    node: Any,
    orient: str,
    data_c: int,
    p: Param,
    wn: Any,
    a: dict,
    tensors: dict,
) -> tuple | None:
    """Dead-input gather + unique-output gather around a shrunk weight.

    ``None`` when the weight has nothing to elide — no dead input
    slices and no duplicate output slices.
    """
    o, i = a["o"], a["i"]
    keep, firsts, imap = a["keep"], a["firsts"], a["imap"]
    if len(keep) == i and len(firsts) == o:
        return None
    sub = wn[list(firsts)][:, list(keep)]
    w_eid, w_name = _reg(
        eg,
        tensors,
        f"{p.name}__el{len(firsts)}x{len(keep)}",
        sub if orient == "linear" else sub.T,
    )
    g = eg.find(data_c)
    if len(keep) < i:
        g = _gather(eg, g, keep)
    core = eg.add_enode(_op(orient), (g, w_eid), {}, provenance=_PROV)
    if len(firsts) < o:
        core = _gather(eg, core, imap)
    return ("elide", _bias_add(eg, core, node), {"w_param": w_name})


def _m_blocks(
    eg: Any,
    node: Any,
    orient: str,
    data_c: int,
    p: Param,
    wn: Any,
    a: dict,
    tensors: dict,
) -> tuple | None:
    """Per-block projections on split inputs, concatenated — or None."""
    blocks = a["blocks"]
    if not blocks:
        return None
    op = _op(orient)
    sizes = tuple(c1 - c0 for _, _, c0, c1 in blocks)
    outs: list[int] = []
    names: list[str] = []
    for j, (r0, r1, c0, c1) in enumerate(blocks):
        wb = wn[r0:r1, c0:c1]
        we, wn_ = _reg(
            eg,
            tensors,
            f"{p.name}__bd{j}",
            wb if orient == "linear" else wb.T,
        )
        names.append(wn_)
        xj = eg.add_enode(
            "split",
            (eg.find(data_c),),
            {"sizes": sizes, "dim": -1, "index": j},
            provenance=_PROV,
        )
        outs.append(eg.add_enode(op, (xj, we), {}, provenance=_PROV))
    cat = outs[0]
    for e in outs[1:]:
        cat = eg.add_enode(
            "concat", (cat, e), {"dim": -1}, provenance=_PROV
        )
    return (
        "block_diag",
        _bias_add(eg, cat, node),
        {"w_params": names, "sizes": sizes},
    )


def _members(
    eg: Any,
    node: Any,
    orient: str,
    data_c: int,
    p: Param,
    wn: Any,
    a: dict,
    tensors: dict,
) -> list[tuple]:
    """Candidate ``(kind, eid, extra)`` members for one site.

    Strongest structure wins outright — identity, diagonal and zero
    dominate every coarser reading of the same weight.  ``elide`` and
    ``block_diag`` can coexist (dup rows inside real blocks, blocks
    beside dead inputs), so both are offered and extraction picks.
    """
    if a["ident"]:
        ms = [_m_identity(eg, node, data_c)]
    elif a["diag"]:
        ms = [_m_diag(eg, node, data_c, p, wn, tensors)]
    elif a["zero"]:
        ms = [_m_zero(eg, node, data_c, a["o"])]
    else:
        ms = [
            _m_elide(eg, node, orient, data_c, p, wn, a, tensors),
            _m_blocks(eg, node, orient, data_c, p, wn, a, tensors),
        ]
    return [m for m in ms if m is not None]


# ---------------------------------------------------------------------------
#  The pass
# ---------------------------------------------------------------------------

_LAW = {
    "identity": (
        "pointwise witness for identity elision: the weight's stored "
        "value is exactly the identity matrix, so the projection is "
        "the identity map — bound 0, established by the specials pass"
    ),
    "diagonal": (
        "pointwise witness for diagonal elision: the weight's stored "
        "value is exactly diagonal, so the projection is a pointwise "
        "scale — bound 0, established by the specials pass"
    ),
    "zero": (
        "pointwise witness for zero-weight elision: the weight's "
        "stored value is exactly zero, so the projection is the zero "
        "map — bound 0, established by the specials pass"
    ),
    "elide": (
        "pointwise witness for dead/duplicate-slice elision: the "
        "weight's stored value has bitwise-dead input slices and "
        "bitwise-duplicate output slices, so the projection equals a "
        "narrower projection plus a gather — bound 0, established by "
        "the specials pass"
    ),
    "block_diag": (
        "pointwise witness for block-diagonal splitting: the "
        "weight's stored value is exactly block-diagonal, so the "
        "projection is the concatenation of per-block projections — "
        "bound 0, established by the specials pass"
    ),
}


def _offer_one_site(
    eg: Any,
    cid: int,
    node: Any,
    orient: str,
    data_c: int,
    p: Param,
    source_tensors: dict,
    cache: dict,
    witness: bool,
) -> list[dict]:
    """Analyse one leaf weight's value and offer its special members."""
    w = source_tensors.get(p.name)
    if w is None or not _is_tensor(w) or len(w.shape) != 2:
        return []
    wn = _detach(w if orient == "linear" else w.T)
    o, i = int(wn.shape[0]), int(wn.shape[1])
    if not o or not i:
        return []  # degenerate axis — nothing safe to spell
    key = (p.name, orient)
    a = cache.get(key)
    if a is None:
        a = _analyse(wn, o, i)
        cache[key] = a
    recs: list[dict] = []
    for kind, eid, extra in _members(
        eg, node, orient, data_c, p, wn, a, source_tensors
    ):
        merged = eg._offer_witness(
            cid,
            eid,
            rhs_term=eg.any_term(eid),
            provenance=_PROV,
            law=_LAW[kind],
            witness=witness,
            error_bound=0.0,
            bound_norm="frobenius",
            note=(f"offer_weight_specials: {p.name} [{orient}] {kind}"),
        )
        if merged:
            recs.append(
                {
                    "param": p.name,
                    "orient": orient,
                    "kind": kind,
                    "in_dim": i,
                    "out_dim": o,
                    "eid": eid,
                    **extra,
                }
            )
    return recs


def offer_weight_specials(
    eg: Any,
    source_tensors: dict,
    *,
    witness: bool = True,
) -> list[dict]:
    """Offer exact members for structurally-special weight params.

    Scans every ``matmul(data, W)`` and ``linear(data, W[, b])``
    e-node whose weight argument's e-class holds a ``Param`` leaf
    backed by a 2-D ``source_tensors`` value, inspects the stored
    values, and offers the applicable exact members — identity,
    diagonal, zero, dead/duplicate elision, block-diagonal split —
    into the consumer's e-class under pointwise witnesses carrying a
    certified zero bound.

    The members compete as ordinary alternatives: extraction selects
    one only when the active cost model honestly prefers it (under
    flop pricing always — each member does strictly less work; under
    the roofline/executor models only when the saving beats the extra
    gather/split launches).  ``data`` must reach a ``Var`` — a
    param-only consumer folds at compile time either way.  The
    derived parameters are registered into ``source_tensors`` so
    lowering materialises them like any weight.

    Returns one record per offered member: ``{param, orient, kind,
    in_dim, out_dim, <derived param names>, eid}``.
    """
    offers: list[dict] = []
    has_var = _cls_has_var(eg, {})
    cache: dict = {}
    # Snapshot: the offered unions mutate the class table mid-walk —
    # find() always lands on the live canonical id for a snapshot key.
    for cid in list(eg._classes):
        cid = eg.find(cid)
        for node in tuple(eg._classes[cid].nodes):
            site = _site_orient(node)
            if site is None:
                continue
            orient, data_c, w_c = site
            if not has_var(data_c, frozenset()):
                continue  # param-only consumer folds either way
            for p in _leaf_params(eg, w_c):
                recs = _offer_one_site(
                    eg,
                    cid,
                    node,
                    orient,
                    data_c,
                    p,
                    source_tensors,
                    cache,
                    witness,
                )
                if recs:
                    offers.extend(recs)
                    break  # one leaf weight per consumer site
    return offers
