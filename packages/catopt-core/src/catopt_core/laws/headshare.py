"""HeadShare — compute-level dedup of bitwise-equal attention heads.

A non-local pass in the :mod:`catopt_core.laws.pairing` family — a
sibling of :func:`~catopt_core.laws.specials.offer_weight_specials`,
which already dedups duplicate *rows* inside one projection weight.
This pass dedups whole *heads*: when heads ``i`` and ``j`` of an
``sdpa`` site are provably identical — every (Wq, Wk, Wv) head-block
pair bitwise-equal under a resolved head structure — the kernel runs
once on the ``k`` unique heads and the output re-expands by a gather:

    sdpa(q, k, v)  ≡  index_select(sdpa(q', k', v'), ha, imap)
    o' = index_select(o, ha, uniq)

``sdpa`` is headwise-independent along every leading axis (all dims
before the ``(T, d)`` tail are batch-like), so equal inputs on equal
positions give equal outputs — an exact, value-level equality, the
``share_duplicate_param_slices`` convention: raw-byte slices, stricter
than ``torch.equal`` (``-0.0``/``+0.0`` and NaN payloads stay
distinct).

Per-head resolution accepts an operand whose members reduce to a
*site* — a ``linear``/``matmul`` projection (or a leaf parameter)
whose last dim is the flat feature axis — lifted to heads by a
``reshape`` splitting the trailing features into ``(h, d)``, plus
head-uniform chains on top:

* axis ops — ``transpose`` / ``permute`` / ``movedim`` /
  ``unsqueeze`` / ``squeeze``;
* head-axis cuts and gathers — ``slice`` / ``narrow`` / ``split`` /
  ``chunk`` / ``index_select`` on the head axis;
* broadcast repeats — ``unsqueeze`` + ``expand`` + merge-``reshape``,
  the exported GQA kv-expansion (head ``m`` reads base head
  ``m // g``);
* pointwise ops whose other operand is constant along the head axis
  (RoPE tables and friends — ``add``/``sub``/``mul``/``div``/``pow``,
  ``concat`` off the head axis, unary elementwise, ``softmax`` off the
  head axis);
* per-head ``stack`` spellings, where each child is compared as a
  param-erased skeleton plus whole-tensor bytes.

Fused-QKV ``split``/``slice`` on the feature axis before the
head-split tracks the byte offset, so the q/k/v sections of one packed
weight attribute correctly.

Declines (no offer): 4+-argument sdpa (attn_mask children),
``dropout_p != 0``, ``enable_gqa``, operands whose chain hits an
unrecognised op, per-head factor tensors, and any head pair whose
weight blocks differ by even one byte.  Equal *weights* are rare in
trained models; they are real in shared-head architectures,
quantization-induced ties, GQA kv replication, and after
``WeightTie``-style merges — the CHAI observation specialised to
*exact* equality.  Nothing is approximated: when no head pair is
bitwise-equal the pass simply does not fire.

The offered member competes on cost like every witness offer — a
gather-``sdpa``-gather member strictly shrinks the ``sdpa`` work
(``sdpa`` flops scale with ``h``) while ``index_select`` is memory
traffic only; under flop pricing it always wins when ``k < h``, under
launch-aware pricing only when the kernel saving beats the extra
gathers.

Every offer lands under a pointwise witness (``error_bound=0``); each
record carries a ``recheck`` recipe — the per-head source keys — so
the compositional cache can re-verify byte equality against a
replayed block's own tensors via :func:`headshare_keys_hold`.
"""

from typing import Any

from catopt_core.attrs import attr_of
from catopt_core.egraph.types import _LeafRegistry
from catopt_core.ir import Const, Op, Param, Var, _attr_key
from catopt_core.laws.base import _shape_of
from catopt_core.laws.factored import _leaf_params
from catopt_core.laws.pairing import _is_tensor
from catopt_core.laws.specials import _sig

__all__ = ["headshare_keys_hold", "share_duplicate_attention_heads"]

_PROV = "head_share"

# ---------------------------------------------------------------------------
#  Resolution states
#
#  ("s", shape, sources, bcast) — a feature-space value: its last dim
#      is the flat (h·d) feature axis; sources entries (name, kind,
#      off) say feature f draws from parameter `name`'s block
#      [off+f), where kind is "rows" (linear weight (o,i)), "cols"
#      (matmul weight (i,o)) or "vec" (any tensor's last-dim slice).
#      `bcast` marks axes provably made of broadcast copies.
#  ("h", ha, shape, sigs, bcast) — a head-structured value: `sigs[j]`
#      is the comparable per-head signature entry (marker, keys);
#      markers: "blk" for site-derived blocks, ("skel", ...) for
#      per-head pieces, ("sel", entry) for head-axis selects.
# ---------------------------------------------------------------------------

#: Shape-preserving coercions and pointwise unary ops — they keep the
#: head/feature axes positional and uniform.
_PASSTHRU = frozenset(
    {
        "contiguous",
        "alias",
        "to",
        "type_as",
        "float",
        "dropout",
        "clone",
        "detach",
        "detach_",
        "copy",
        "neg",
        "square",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "expm1",
        "sigmoid",
        "silu",
        "tanh",
        "gelu",
        "relu",
        "abs",
        "sign",
        "sin",
        "cos",
        "log",
        "floor",
        "ceil",
        "round",
        "reciprocal",
        "erf",
    }
)

#: Pointwise binary ops — position-preserving in both states.
_ELEMWISE = frozenset({"add", "sub", "mul", "div", "pow"})

#: Dim-arg unary ops that are head-uniform off the head axis and
#: feature-block-mixing on the feature axis (a softmax over positions
#: treats every head identically; over the flat feature axis it would
#: mix head blocks).
_DIM_OPS = frozenset({"softmax", "log_softmax"})


class _Ctx:
    """Per-pass context: tensor table plus resolution/byte memos."""

    def __init__(self, eg: Any, src: dict) -> None:
        self.eg = eg
        self.src = src
        self.memo: dict[int, tuple] = {}
        self.pmemo: dict[int, Any] = {}
        self.busy: set[int] = set()
        self.bcache: dict[tuple, bytes | None] = {}


# ---------------------------------------------------------------------------
#  Small helpers — class shapes, state fields, products
# ---------------------------------------------------------------------------


def _cls_shape(eg: Any, cid: int):
    """Canonical member's inferred shape for *cid* (None-safe)."""
    t = eg._any_term_cached(cid)
    if t is None:
        return None
    s = _shape_of(t)
    return s if isinstance(s, tuple) else None


def _rank(st: tuple) -> int:
    """Rank carried by a resolution state."""
    return len(st[1]) if st[0] == "s" else len(st[2])


def _bcast_axes(st: tuple) -> frozenset:
    """Return the provably-broadcast axes of a resolution state."""
    return st[3] if st[0] == "s" else st[4]


def _prod(dims: Any) -> int | None:
    """Product of a dim tuple — ``None`` if any extent is unknown."""
    p = 1
    for d in dims:
        if not isinstance(d, int):
            return None
        p *= d
    return p


def _src_params(ctx: _Ctx, cid: int) -> list[tuple[str, Any]]:
    """``(name, tensor)`` for Param leaves of *cid* backed by src."""
    out = []
    for p in _leaf_params(ctx.eg, cid):
        t = ctx.src.get(p.name)
        if t is not None and _is_tensor(t):
            out.append((p.name, t))
    return sorted(out)


def _heads_of(states: tuple) -> list:
    """Return the ``("h", ...)`` states among *states*."""
    return [s for s in states if s[0] == "h"]


def _sites_of(states: tuple) -> list:
    """Return the ``("s", ...)`` states among *states*."""
    return [s for s in states if s[0] == "s"]


# ---------------------------------------------------------------------------
#  Uniformity / broadcast predicates
# ---------------------------------------------------------------------------


def _const_along(ctx: _Ctx, cid: int, p: int, rp: int) -> bool:
    """Whether *cid*'s value is constant along parent axis *p*.

    Shape says so (the axis is absent or extent-1), or some resolved
    state marks it broadcast — e.g. a RoPE table ``expand``ed to the
    full head count keeps extent ``h`` but is constant along it.
    """
    cs = _cls_shape(ctx.eg, cid)
    if not isinstance(cs, tuple) or len(cs) > rp:
        return False
    pc = p - (rp - len(cs))
    if pc < 0:
        return True
    if cs[pc] == 1:
        return True
    return any(pc in _bcast_axes(s) for s in _resolve(ctx, cid))


def _uniform_f(ctx: _Ctx, cid: int) -> bool:
    """Feature-uniform — scalar or last-dim-1 (position-safe factors)."""
    cs = _cls_shape(ctx.eg, cid)
    return isinstance(cs, tuple) and (not cs or cs[-1] == 1)


def _bcast_of(ctx: _Ctx, node: Any, rp: int) -> frozenset:
    """Broadcast axes of a binary op — both children constant there."""
    c0, c1 = node.children[0], node.children[1]
    return frozenset(
        p
        for p in range(rp)
        if _const_along(ctx, c0, p, rp) and _const_along(ctx, c1, p, rp)
    )


# ---------------------------------------------------------------------------
#  Leaf / projection sites
# ---------------------------------------------------------------------------


def _bias_sources(ctx: _Ctx, b_c: int, o: int) -> list | None:
    """Return sources for a ``linear`` bias — [] uniform, None veto.

    A scalar ``Const`` or all-dims-1 bias adds the same value to every
    head block — feature-uniform, contributes nothing.  A 1-D ``(o,)``
    leaf joins the source list as a ``vec`` block source.  Anything
    else (a Var, a computed term, a wider param) cannot be attributed
    per head — the whole site declines.
    """
    for n in ctx.eg._classes[ctx.eg.find(b_c)].nodes:
        if n.op != "leaf":
            continue
        leaf = _LeafRegistry.decode(n.attrs[0][1])
        if isinstance(leaf, Const):
            return []
        if isinstance(leaf, Param):
            t = ctx.src.get(leaf.name)
            if t is None or not _is_tensor(t):
                continue
            sh = tuple(t.shape)
            if len(sh) == 1 and sh[0] == o:
                return [(leaf.name, "vec", 0)]
            if all(d == 1 for d in sh):
                return []
    return None


def _h_leaf(ctx: _Ctx, node: Any, shape: Any) -> list:
    """Return site entries for a Param leaf (feature-space, last dim)."""
    if not node.attrs:
        return []
    leaf = _LeafRegistry.decode(node.attrs[0][1])
    if not isinstance(leaf, Param):
        return []
    t = ctx.src.get(leaf.name)
    if t is None or not _is_tensor(t) or len(t.shape) < 1:
        return []
    sh = tuple(int(d) for d in t.shape)
    return [("s", sh, ((leaf.name, "vec", 0),), frozenset())]


def _h_proj(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``linear(x, W[, b])`` / ``matmul(x, W)`` — the projection site."""
    if not isinstance(shape, tuple) or not shape:
        return []
    if node.op == "linear" and len(node.children) in (2, 3):
        kind = "rows"
    elif node.op == "matmul" and len(node.children) == 2:
        kind = "cols"
    else:
        return []
    for name, w in _src_params(ctx, node.children[1]):
        if len(w.shape) != 2:
            continue
        o = int(w.shape[0]) if kind == "rows" else int(w.shape[1])
        if shape[-1] is not None and shape[-1] != o:
            continue
        sources = [(name, kind, 0)]
        if len(node.children) == 3:
            bias = _bias_sources(ctx, node.children[2], o)
            if bias is None:
                continue
            sources += bias
        return [("s", shape, tuple(sources), frozenset())]
    return []


# ---------------------------------------------------------------------------
#  reshape — the head-creating split and the broadcast merge
# ---------------------------------------------------------------------------


def _site_to_heads(st: tuple, out: tuple) -> tuple | None:
    """Split a site's flat feature axis into ``(h, d)`` trailing dims."""
    _, cs, sources, _bc = st
    if len(out) < 2 or not cs or not isinstance(cs[-1], int):
        return None
    hb, d = out[-2], out[-1]
    if hb * d != cs[-1] or tuple(cs[:-1]) != tuple(out[:-2]):
        return None
    sigs = tuple(
        (
            "blk",
            tuple(
                (n, k, off + j * d, off + (j + 1) * d)
                for n, k, off in sources
            ),
        )
        for j in range(hb)
    )
    return ("h", len(out) - 2, out, sigs, frozenset())


def _site_regroup(st: tuple, out: tuple) -> tuple | None:
    """Non-head reshape of a site — the feature axis stays last."""
    _, cs, sources, bcast = st
    if (
        not out
        or not cs
        or out[-1] != cs[-1]
        or _prod(out[:-1]) != _prod(cs[:-1])
    ):
        return None
    nb = (
        frozenset({len(out) - 1})
        if len(cs) - 1 in bcast
        else frozenset()
    )
    return ("s", out, sources, nb)


def _heads_merge(st: tuple, out: tuple) -> tuple | None:
    """Merge the head axis with a following broadcast (repeat) axis.

    The exported GQA kv-expansion spells ``reshape(expand(unsqueeze
    (…)))`` — the ``unsqueeze``/``expand`` pair plants a size-1 axis
    right after the head axis and broadcasts it to ``g``; the final
    ``reshape`` merges ``(h, g)`` into one ``h·g`` axis.  Head ``m``
    of the merged axis reads base head ``m // g``.  Also accepts the
    trivial ``g == 1`` merge, which needs no broadcast proof.
    """
    _, ha, cs, sigs, bcast = st
    if len(out) != len(cs) - 1 or ha + 1 >= len(cs):
        return None
    g, hb = cs[ha + 1], cs[ha]
    if not (isinstance(g, int) and isinstance(hb, int)):
        return None
    if g != 1 and ha + 1 not in bcast:
        return None
    if (
        out[ha] != hb * g
        or tuple(out[:ha]) != tuple(cs[:ha])
        or tuple(out[ha + 1 :]) != tuple(cs[ha + 2 :])
    ):
        return None
    nb = frozenset(
        (a - 1 if a > ha + 1 else a) for a in bcast if a != ha + 1
    )
    return (
        "h",
        ha,
        out,
        tuple(sigs[m // g] for m in range(hb * g)),
        nb,
    )


def _heads_regroup(st: tuple, out: tuple) -> tuple | None:
    """Rewrite a generic head-preserving reshape — the ``(A, h, B)`` rule.

    A reshape is head-safe whenever the output splits as ``(A', h, B')``
    with ``prod(A') == prod(A)`` and ``prod(B') == prod(B)``: row-major
    relayout then keeps ``out[..., j, ...] == in[..., j, ...]`` along
    that axis — the constraint is self-certifying, so any matching
    position *is* the head axis.  Batch-dim merges ``(B,T,h,d) ->
    (B·T,h,d)`` and within-head splits ``(B,h,T,d) -> (B,h,T,d1,d2)``
    both fall out of it.
    """
    _, ha, cs, sigs, bcast = st
    h = cs[ha]
    if not isinstance(h, int):
        return None
    pre, post = _prod(cs[:ha]), _prod(cs[ha + 1 :])
    if pre is None or post is None:
        return None
    for i in range(len(out)):
        if (
            out[i] == h
            and _prod(out[:i]) == pre
            and _prod(out[i + 1 :]) == post
        ):
            nb = frozenset({i}) if ha in bcast else frozenset()
            return ("h", i, out, sigs, nb)
    return None


def _h_reshape(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``reshape`` — head-split of a site, site regroup, head merge."""
    out = dict(node.attrs).get("shape")
    if not isinstance(out, (tuple, list)) or not all(
        isinstance(d, int) for d in out
    ):
        return []
    out = tuple(int(d) for d in out)
    res = []
    for st in _resolve(ctx, node.children[0]):
        rs = (
            (_site_to_heads(st, out), _site_regroup(st, out))
            if st[0] == "s"
            else (_heads_merge(st, out), _heads_regroup(st, out))
        )
        res.extend(r for r in rs if r is not None)
    return res


# ---------------------------------------------------------------------------
#  Axis bookkeeping — transpose / permute / movedim / unsqueeze /
#  squeeze / expand
# ---------------------------------------------------------------------------


def _axis_perm(node: Any, r: int) -> tuple | None:
    """Return the parent-axis -> child-axis map for the axis op."""
    a = dict(node.attrs)
    if node.op == "transpose":
        d0, d1 = attr_of(a, "dim0"), attr_of(a, "dim1")
        if not isinstance(d0, int) or not isinstance(d1, int):
            return None
        f = list(range(r))
        f[d0 % r], f[d1 % r] = d1 % r, d0 % r
        return tuple(f)
    if node.op == "permute":
        dims = attr_of(a, "dim", "dims", "order")
        if not isinstance(dims, (tuple, list)) or not all(
            isinstance(d, int) for d in dims
        ):
            return None
        f = tuple(d % r for d in dims)
        return (
            f if len(f) == r and sorted(f) == list(range(r)) else None
        )
    src = attr_of(a, "source")
    dst = attr_of(a, "destination")
    src = src if isinstance(src, (tuple, list)) else (src,)
    dst = dst if isinstance(dst, (tuple, list)) else (dst,)
    if len(src) != len(dst) or not all(
        isinstance(v, int) for v in (*src, *dst)
    ):
        return None
    srcn = [s % r for s in src]
    order = [i for i in range(r) if i not in set(srcn)]
    for s, d in sorted(
        zip(srcn, [d % r for d in dst], strict=True),
        key=lambda p: p[1],
    ):
        order.insert(d, s)
    return tuple(order) if sorted(order) == list(range(r)) else None


def _h_perm(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``transpose``/``permute``/``movedim`` — remap the head axis."""
    if not isinstance(shape, tuple):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        f = _axis_perm(node, r)
        if f is None:
            continue
        if st[0] == "s":
            if f[r - 1] == r - 1:  # feature axis stays last
                res.append(("s", shape, st[2], st[3]))
        else:
            bc = frozenset(p for p in range(r) if f[p] in st[4])
            res.append(("h", f.index(st[1]), shape, st[3], bc))
    return res


def _remap_bcast(bcast: frozenset, dn: int) -> frozenset:
    """Return the axis set after dropping axis *dn*.

    The inverse of :func:`_shift_bcast`.
    """
    return frozenset(a - (a > dn) for a in bcast if a != dn)


def _shift_bcast(bcast: frozenset, dn: int) -> frozenset:
    """Axis set after inserting an axis at *dn*."""
    return frozenset(a + (a >= dn) for a in bcast)


def _h_unsq(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``unsqueeze`` — insert a size-1 axis (the GQA repeat seed)."""
    if not isinstance(shape, tuple):
        return []
    d = attr_of(dict(node.attrs), "dim")
    if not isinstance(d, int):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % (r + 1)
        if st[0] == "s":
            if dn <= r - 1:  # the feature axis must stay last
                res.append(("s", shape, st[2], _shift_bcast(st[3], dn)))
        else:
            ha = st[1] + (dn <= st[1])
            res.append(("h", ha, shape, st[3], _shift_bcast(st[4], dn)))
    return res


def _h_squeeze(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``squeeze(dim)`` — drop one axis; dim-less squeeze declines."""
    if not isinstance(shape, tuple):
        return []
    d = attr_of(dict(node.attrs), "dim")
    if not isinstance(d, int):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % r
        if st[0] == "s":
            if dn != r - 1:
                res.append(("s", shape, st[2], _remap_bcast(st[3], dn)))
        elif dn != st[1]:
            ha = st[1] - (dn < st[1])
            res.append(("h", ha, shape, st[3], _remap_bcast(st[4], dn)))
    return res


def _h_expand(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``expand``/``broadcast_to`` — broadcast axes become copies.

    A broadcast axis is the only axis a ``(h, g) -> h·g`` merge may
    consume: each of its ``g`` positions is the same base position.
    Expanding the head axis itself (``1 -> h``) makes every head a
    copy of head 0 — a legal, degenerate share.  On a site the
    feature axis must stay put; grown axes are marked broadcast so a
    later ``mul`` recognises the operand as head-uniform.
    """
    out = attr_of(dict(node.attrs), "shape")
    if not isinstance(out, (tuple, list)) or not all(
        isinstance(v, int) for v in out
    ):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        cs = st[1] if st[0] == "s" else st[2]
        pad = len(out) - len(cs)
        if pad < 0 or not all(
            out[pad + i] in (cs[i], -1) or cs[i] == 1
            for i in range(len(cs))
        ):
            continue
        nb = {
            p
            for p in range(len(out))
            if (p < pad and out[p] != 1)
            or (
                p >= pad
                and cs[p - pad] == 1
                and (cs[p - pad] if out[p] == -1 else out[p]) != 1
            )
        }
        if st[0] == "s":
            eff_last = cs[-1] if out[-1] == -1 else out[-1]
            if eff_last == cs[-1]:
                res.append(("s", tuple(out), st[2], frozenset(nb)))
            continue
        ha, sigs = st[1], st[3]
        hp = pad + ha
        eff = cs[ha] if out[hp] == -1 else out[hp]
        if cs[ha] == 1 and eff > 1:
            sigs = tuple(sigs[0] for _ in range(eff))
        nb |= {a + pad for a in st[4]}
        res.append(("h", hp, tuple(out), sigs, frozenset(nb)))
    return res


# ---------------------------------------------------------------------------
#  Axis-local ops — slice / narrow / select / split / chunk /
#  index_select / softmax
# ---------------------------------------------------------------------------


def _bounds(a: dict, node: Any, extent: Any) -> tuple[int, int] | None:
    """``(lo, hi)`` slice bounds for slice/narrow."""
    if not isinstance(extent, int):
        return None
    if node.op == "narrow":
        lo, ln = a.get("start"), a.get("length")
        if not isinstance(lo, int) or not isinstance(ln, int):
            return None
        return (lo, min(lo + ln, extent))
    if a.get("step", 1) not in (None, 1):
        return None
    lo = a.get("start", 0)
    if not isinstance(lo, int):
        return None
    hi = a.get("end")
    hi = extent if not isinstance(hi, int) or hi > extent else hi
    return None if hi <= lo else (lo, hi)


def _part_span(
    node: Any, a: dict, extent: Any
) -> tuple[int, int] | None:
    """``(lo, width)`` of section ``index`` for split/chunk."""
    idx = a.get("index")
    if not isinstance(extent, int) or not isinstance(idx, int):
        return None
    if node.op == "split":
        sizes = a.get("sizes")
        if (
            not isinstance(sizes, (tuple, list))
            or not all(isinstance(s, int) for s in sizes)
            or sum(sizes) != extent
            or idx >= len(sizes)
        ):
            return None
        return (int(sum(sizes[:idx])), sizes[idx])
    n = a.get("chunks")
    if not isinstance(n, int) or n < 1 or extent % n or idx >= n:
        return None
    return (idx * (extent // n), extent // n)


def _h_cut(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``slice``/``narrow``/``select`` — axis-local, off-head or on."""
    if not isinstance(shape, tuple):
        return []
    d = attr_of(dict(node.attrs), "dim", default=0)
    if not isinstance(d, int):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % r
        if st[0] == "s":
            if dn != r - 1:
                bc = (
                    _remap_bcast(st[3], dn)
                    if node.op == "select"
                    else st[3]
                )
                res.append(("s", shape, st[2], bc))
            elif node.op != "select":
                sp = _bounds(dict(node.attrs), node, st[1][-1])
                if sp is not None:
                    srcs = tuple(
                        (n, k, off + sp[0]) for n, k, off in st[2]
                    )
                    res.append(("s", shape, srcs, st[3]))
        elif dn == st[1]:
            if node.op != "select":
                sp = _bounds(dict(node.attrs), node, len(st[3]))
                if sp is not None:
                    res.append(
                        ("h", st[1], shape, st[3][sp[0] : sp[1]], st[4])
                    )
        else:
            ha = st[1] - (node.op == "select" and dn < st[1])
            bc = (
                _remap_bcast(st[4], dn)
                if node.op == "select"
                else st[4]
            )
            res.append(("h", ha, shape, st[3], bc))
    return res


def _h_split(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``split``/``chunk`` — feature-axis sections shift site offsets."""
    if not isinstance(shape, tuple):
        return []
    a = dict(node.attrs)
    d = attr_of(a, "dim", default=0)
    if not isinstance(d, int):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % r
        if st[0] == "s":
            if dn != r - 1:
                res.append(("s", shape, st[2], st[3]))
                continue
            sp = _part_span(node, a, st[1][-1])
            if sp is None:
                continue
            srcs = tuple((n, k, off + sp[0]) for n, k, off in st[2])
            res.append(("s", shape, srcs, st[3]))
        elif dn == st[1]:
            sp = _part_span(node, a, len(st[3]))
            if sp is not None:
                res.append(
                    (
                        "h",
                        st[1],
                        shape,
                        st[3][sp[0] : sp[0] + sp[1]],
                        st[4],
                    )
                )
        else:
            res.append(("h", st[1], shape, st[3], st[4]))
    return res


def _h_gather(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``index_select`` — a head-axis gather remaps signatures."""
    if not isinstance(shape, tuple):
        return []
    a = dict(node.attrs)
    d = attr_of(a, "dim", default=0)
    if not isinstance(d, int):
        return []
    idx = a.get("index")
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % r
        if st[0] == "s":
            if dn != r - 1:
                res.append(("s", shape, st[2], st[3]))
        elif dn != st[1]:
            res.append(("h", st[1], shape, st[3], st[4]))
        elif isinstance(idx, (tuple, list)) and all(
            isinstance(i, int) and 0 <= i < len(st[3]) for i in idx
        ):
            bc = frozenset(x for x in st[4] if x != st[1])
            res.append(
                ("h", st[1], shape, tuple(st[3][i] for i in idx), bc)
            )
    return res


def _h_dimop(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``softmax``/``log_softmax`` — uniform off the tracked axes."""
    if not isinstance(shape, tuple):
        return []
    d = attr_of(dict(node.attrs), "dim")
    if not isinstance(d, int):
        return []
    res = []
    for st in _resolve(ctx, node.children[0]):
        r = _rank(st)
        dn = d % r
        if st[0] == "s":
            if dn != r - 1:
                res.append(("s", shape, st[2], st[3]))
        elif dn != st[1]:
            res.append(("h", st[1], shape, st[3], st[4]))
    return res


# ---------------------------------------------------------------------------
#  Elementwise binary + concat — the RoPE-generality case
# ---------------------------------------------------------------------------


def _merge_entry(e0: tuple, e1: tuple) -> tuple:
    """Merge two per-head signature entries ``(marker, keys)``."""
    return (("m", e0[0], e1[0]), e0[1] + e1[1])


def _merge_one(
    ctx: _Ctx, node: Any, shape: tuple, h0: tuple, st1: tuple
) -> tuple | None:
    """Merge two heads states under broadcasting."""
    rp = len(shape)
    ha0 = h0[1] + (rp - len(h0[2]))
    ha1 = st1[1] + (rp - len(st1[2]))
    if ha0 != ha1:
        return None
    s1 = st1[3]
    if len(s1) == 1 and len(h0[3]) > 1:
        s1 = s1 * len(h0[3])
    if len(s1) != len(h0[3]):
        return None
    sigs = tuple(
        _merge_entry(a, b) for a, b in zip(h0[3], s1, strict=True)
    )
    return ("h", ha0, shape, sigs, _bcast_of(ctx, node, rp))


def _merge_heads(
    ctx: _Ctx, node: Any, shape: tuple, sts0: tuple, sts1: tuple
) -> list:
    """Merge child states under a broadcasting pointwise op.

    A non-merged child must be provably constant along the parent's
    head axis — otherwise the node's heads are not attributable and
    the state declines.
    """
    rp = len(shape)
    out = []
    merged1: set[int] = set()
    hs0, hs1 = _heads_of(sts0), _heads_of(sts1)
    for h0 in hs0:
        ha0 = h0[1] + (rp - len(h0[2]))
        merged = False
        for i1, h1 in enumerate(hs1):
            r = _merge_one(ctx, node, shape, h0, h1)
            if r is not None:
                merged = True
                merged1.add(i1)
                out.append(r)
        if not merged and _const_along(ctx, node.children[1], ha0, rp):
            out.append(
                ("h", ha0, shape, h0[3], _bcast_of(ctx, node, rp))
            )
    for i1, h1 in enumerate(hs1):
        ha1 = h1[1] + (rp - len(h1[2]))
        if i1 not in merged1 and _const_along(
            ctx, node.children[0], ha1, rp
        ):
            out.append(
                ("h", ha1, shape, h1[3], _bcast_of(ctx, node, rp))
            )
    return out


def _merge_sites(
    ctx: _Ctx, node: Any, shape: tuple, sts0: tuple, sts1: tuple
) -> list:
    """Merge feature-space sites under a pointwise op.

    Equal-width sites contribute both source lists; a width-1 site is
    feature-uniform and contributes nothing; a non-site child must be
    scalar or last-dim-1 — otherwise the site's blocks are not
    attributable and the state declines.
    """
    bc = _bcast_of(ctx, node, len(shape))
    out = []
    for s0 in _sites_of(sts0):
        w0 = s0[1][-1] if s0[1] else None
        for s1 in _sites_of(sts1):
            w1 = s1[1][-1] if s1[1] else None
            if w0 == w1:
                out.append(
                    (
                        "s",
                        shape,
                        tuple(dict.fromkeys(s0[2] + s1[2])),
                        bc,
                    )
                )
            elif w1 == 1:
                out.append(("s", shape, s0[2], bc))
            elif w0 == 1:
                out.append(("s", shape, s1[2], bc))
        if _uniform_f(ctx, node.children[1]):
            out.append(("s", shape, s0[2], bc))
    if _uniform_f(ctx, node.children[0]):
        for s1 in _sites_of(sts1):
            out.append(("s", shape, s1[2], bc))
    return out


def _merge_elem(ctx: _Ctx, node: Any, shape: Any) -> list:
    """Pointwise binary — heads or sites merge; others must uniform."""
    if not isinstance(shape, tuple) or not shape:
        return []
    sts0 = _resolve(ctx, node.children[0])
    sts1 = _resolve(ctx, node.children[1])
    return _merge_heads(ctx, node, shape, sts0, sts1) + _merge_sites(
        ctx, node, shape, sts0, sts1
    )


def _h_concat(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``concat`` — head-axis concat extends the signature list."""
    if not isinstance(shape, tuple) or not shape:
        return []
    d = attr_of(dict(node.attrs), "dim", default=0)
    if not isinstance(d, int):
        return []
    rp = len(shape)
    dn = d % rp
    out = []
    for h0 in _heads_of(_resolve(ctx, node.children[0])):
        for h1 in _heads_of(_resolve(ctx, node.children[1])):
            if len(h0[2]) != len(h1[2]) or h0[1] != h1[1]:
                continue
            if dn == h0[1]:
                bc = _bcast_of(ctx, node, rp) - {dn}
                out.append(("h", dn, shape, h0[3] + h1[3], bc))
            elif len(h0[3]) == len(h1[3]):
                sigs = tuple(
                    _merge_entry(a, b)
                    for a, b in zip(h0[3], h1[3], strict=True)
                )
                bc = _bcast_of(ctx, node, rp)
                out.append(("h", dn, shape, sigs, bc))
    return out


# ---------------------------------------------------------------------------
#  stack — the per-head spelling
# ---------------------------------------------------------------------------


def _skel(t: Any, keys: list) -> tuple:
    """Param-erased term skeleton; *keys* collects whole-tensor keys."""
    if isinstance(t, Param):
        keys.append((t.name, "whole", 0, 0))
        return ("p",)
    if isinstance(t, Var):
        return ("v", t.name)
    if isinstance(t, Const):
        return ("c", t.value)
    if isinstance(t, Op):
        return (
            "op",
            t.op,
            _attr_key(t.attrs),
            tuple(_skel(a, keys) for a in t.args),
        )
    return ("l", repr(t))


def _piece_select(ctx: _Ctx, term: Any) -> tuple | None:
    """``select``/single-``index_select`` on a head axis → its entry."""
    if not isinstance(term, Op) or not term.args:
        return None
    if term.op == "select":
        d, i = attr_of(term, "dim"), attr_of(term, "index")
    elif term.op == "index_select":
        d = attr_of(term, "dim")
        idx = attr_of(term, "index")
        i = (
            idx[0]
            if isinstance(idx, (tuple, list)) and len(idx) == 1
            else None
        )
    else:
        return None
    if not isinstance(d, int) or not isinstance(i, int):
        return None
    base = ctx.eg.add_term(term.args[0])
    for st in _resolve(ctx, base):
        if (
            st[0] == "h"
            and st[1] == d % len(st[2])
            and 0 <= i < len(st[3])
        ):
            marker, keys = st[3][i]
            return (("sel", marker), keys)
    return None


def _piece(ctx: _Ctx, cid: int) -> tuple | None:
    """Per-head signature entry of *cid*'s canonical term.

    ``("sel", head_entry)`` when the class is a head-axis select of a
    resolvable base; else ``(("skel", skeleton), keys)`` — equal iff
    the term structures match with params erased AND the erased
    params' whole-tensor bytes match pairwise.
    """
    cid = ctx.eg.find(cid)
    if cid in ctx.pmemo:
        return ctx.pmemo[cid]
    if cid in ctx.busy:
        return None
    ctx.busy.add(cid)
    entry = None
    term = ctx.eg._any_term_cached(cid)
    if term is not None:
        entry = _piece_select(ctx, term)
        if entry is None:
            keys: list = []
            entry = (("skel", _skel(term, keys)), tuple(keys))
    ctx.busy.discard(cid)
    ctx.pmemo[cid] = entry
    return entry


def _h_stack(ctx: _Ctx, node: Any, shape: Any) -> list:
    """``stack`` — per-head terms become per-head signature entries."""
    if not isinstance(shape, tuple) or not shape or not node.children:
        return []
    d = attr_of(dict(node.attrs), "dim", default=0)
    if not isinstance(d, int):
        return []
    ha = d % len(shape)
    entries = []
    for c in node.children:
        e = _piece(ctx, c)
        if e is None:
            return []
        entries.append(e)
    return [("h", ha, shape, tuple(entries), frozenset())]


# ---------------------------------------------------------------------------
#  The resolver — per-class states, memoised, cycle-guarded
# ---------------------------------------------------------------------------

_OP_HANDLERS = {
    "leaf": _h_leaf,
    "linear": _h_proj,
    "matmul": _h_proj,
    "reshape": _h_reshape,
    "transpose": _h_perm,
    "permute": _h_perm,
    "movedim": _h_perm,
    "unsqueeze": _h_unsq,
    "squeeze": _h_squeeze,
    "expand": _h_expand,
    "broadcast_to": _h_expand,
    "slice": _h_cut,
    "narrow": _h_cut,
    "select": _h_cut,
    "split": _h_split,
    "chunk": _h_split,
    "index_select": _h_gather,
    "concat": _h_concat,
    "stack": _h_stack,
}


def _interpret(ctx: _Ctx, node: Any, shape: Any) -> list:
    """All resolution states one enode produces (usually ≤1)."""
    op = node.op
    if len(node.children) == 1:
        if op in _PASSTHRU:
            return list(_resolve(ctx, node.children[0]))
        if op in _DIM_OPS:
            return _h_dimop(ctx, node, shape)
    if op in _ELEMWISE and len(node.children) == 2:
        return _merge_elem(ctx, node, shape)
    h = _OP_HANDLERS.get(op)
    return h(ctx, node, shape) if h is not None else []


def _resolve(ctx: _Ctx, cid: int) -> tuple:
    """Every provable resolution state of an e-class.

    A class may carry both a feature-space ``("s")`` and a
    head-structured ``("h")`` reading (different members); each
    consumer picks the one it needs.  Memoised per pass; a class
    revisited on a DFS cycle contributes nothing.
    """
    cid = ctx.eg.find(cid)
    hit = ctx.memo.get(cid)
    if hit is not None:
        return hit
    if cid in ctx.busy:
        return ()
    ctx.busy.add(cid)
    out: list = []
    shape = _cls_shape(ctx.eg, cid)
    for node in sorted(ctx.eg._classes[cid].nodes, key=repr):
        for st in _interpret(ctx, node, shape):
            if st not in out:
                out.append(st)
    ctx.busy.discard(cid)
    ctx.memo[cid] = tuple(out)
    return ctx.memo[cid]


# ---------------------------------------------------------------------------
#  Byte signatures, the offer, and the replay recheck
# ---------------------------------------------------------------------------


def _slice_bytes(t: Any, kind: str, a: int, b: int) -> bytes:
    """Raw bytes of the key's slice — the bitwise signature unit."""
    if kind == "rows":
        s = t[a:b]
    elif kind == "cols":
        s = t[:, a:b]
    elif kind == "vec":
        s = t[..., a:b]
    else:
        s = t
    return _sig(s)


def _key_bytes(ctx: _Ctx, key: tuple) -> bytes | None:
    """Resolve ``(name, kind, a, b)`` to slice bytes (memoised)."""
    hit = ctx.bcache.get(key, ...)
    if hit is not ...:
        return hit
    t = ctx.src.get(key[0])
    val = (
        _slice_bytes(t, key[1], key[2], key[3])
        if t is not None and _is_tensor(t)
        else None
    )
    ctx.bcache[key] = val
    return val


def _joint_sig(ctx: _Ctx, entries: list) -> tuple | None:
    """One head's joint signature — ``(marker, key-bytes)`` per operand."""
    out = []
    for marker, keys in entries:
        bs = []
        for k in keys:
            b = _key_bytes(ctx, k)
            if b is None:
                return None
            bs.append(b)
        out.append((marker, tuple(bs)))
    return tuple(out)


def _offer_sdpa(
    ctx: _Ctx, cid: int, node: Any, witness: bool
) -> dict | None:
    """Resolve one ``sdpa`` node's operands and offer the shared member."""
    attrs = dict(node.attrs)
    if attrs.get("dropout_p") or attrs.get("enable_gqa"):
        return None
    parts = []
    for c in node.children:
        hs = _heads_of(_resolve(ctx, c))
        if not hs:
            return None
        parts.append(hs[0])
    ha, h, rank = parts[0][1], len(parts[0][3]), len(parts[0][2])
    if (
        rank < 3
        or ha > rank - 3
        or any(
            p[1] != ha or len(p[3]) != h or len(p[2]) != rank
            for p in parts[1:]
        )
    ):
        return None
    uniq: list = []
    imap: list = []
    for j in range(h):
        s = _joint_sig(ctx, [p[3][j] for p in parts])
        if s is None:
            return None
        try:
            imap.append(uniq.index(s))
        except ValueError:
            imap.append(len(uniq))
            uniq.append(s)
    if len(uniq) == h:
        return None
    firsts = tuple(imap.index(u) for u in range(len(uniq)))
    gathered = tuple(
        ctx.eg.add_enode(
            "index_select",
            (ctx.eg.find(c),),
            {"dim": ha, "index": firsts},
            provenance=_PROV,
        )
        for c in node.children
    )
    core = ctx.eg.add_enode("sdpa", gathered, attrs, provenance=_PROV)
    member = ctx.eg.add_enode(
        "index_select",
        (core,),
        {"dim": ha, "index": tuple(imap)},
        provenance=_PROV,
    )
    merged = ctx.eg._offer_witness(
        cid,
        member,
        rhs_term=ctx.eg.any_term(member),
        provenance=_PROV,
        law=(
            "pointwise witness for head sharing: the sdpa operands' "
            "head blocks are bitwise-equal under index_map, so the "
            "attention computes once per unique head and re-expands "
            "by gather (equality established by the headshare pass "
            "over source tensors)"
        ),
        witness=witness,
        note=(
            f"share_duplicate_attention_heads: {h} heads -> "
            f"{len(firsts)} unique (axis {ha})"
        ),
    )
    if not merged:
        return None
    keys = tuple(
        tuple(k for p in parts for k in p[3][j][1]) for j in range(h)
    )
    return {
        "eid": member,
        "axis": ha,
        "heads": h,
        "unique": len(firsts),
        "index_map": tuple(imap),
        "uniq": firsts,
        "params": sorted({k[0] for ks in keys for k in ks}),
        "recheck": {"imap": tuple(imap), "keys": keys},
    }


def share_duplicate_attention_heads(
    eg: Any, source_tensors: dict, *, witness: bool = True
) -> list[dict]:
    """Offer deduplicated-head members for bitwise-shared sdpa heads.

    Scans every ``sdpa`` enode, resolves each operand to a head
    structure (see the module docstring for the recognised chains),
    and — when the joint per-head byte signatures deduplicate to
    ``k < h`` — offers

    ``index_select(sdpa(index_select(o, ha, uniq) for o in q,k,v),
    ha, imap)``

    into the sdpa's e-class under a pointwise witness.  The member
    competes as an ordinary alternative: extraction selects it only
    when the active cost model honestly prefers it (under flop
    pricing always — the ``sdpa`` work shrinks by ``k/h`` while the
    gathers are views).

    Returns one record per offered member: ``{eid, axis, heads,
    unique, index_map, uniq, params, recheck}`` — ``recheck`` is the
    replay recipe :func:`headshare_keys_hold` consumes.
    """
    ctx = _Ctx(eg, source_tensors)
    offers: list[dict] = []
    for cid in list(eg._classes):
        cid = eg.find(cid)
        for node in tuple(eg._classes[cid].nodes):
            if node.op != "sdpa" or len(node.children) != 3:
                continue
            rec = _offer_sdpa(ctx, cid, node, witness)
            if rec is not None:
                offers.append(rec)
    return offers


def headshare_keys_hold(sites: Any, name_of: Any, tensors: Any) -> bool:
    """Re-verify recorded head equality on a replayed block's values.

    *sites* is the list of ``recheck`` records (``{"imap", "keys"}``,
    keys per head over ``(name, kind, a, b)``); *name_of* maps
    template-side param names to the current block's; *tensors* the
    current ``source_tensors``.  ``False`` — a violated assumption or
    an unmappable name — sends the caller down the ordinary search
    path.
    """
    for site in sites:
        imap, keys = site["imap"], site["keys"]
        firsts = {}
        for j, u in enumerate(imap):
            firsts.setdefault(u, j)
        for j, u in enumerate(imap):
            ref = firsts[u]
            if ref == j:
                continue
            kj, kr = keys[j], keys[ref]
            if len(kj) != len(kr):
                return False
            for ka, kb in zip(kj, kr, strict=True):
                if ka[1] != kb[1]:
                    return False
                na, nb = name_of(ka[0]), name_of(kb[0])
                ta = tensors.get(na) if na else None
                tb = tensors.get(nb) if nb else None
                if ta is None or tb is None:
                    return False
                if _slice_bytes(
                    ta, ka[1], ka[2], ka[3]
                ) != _slice_bytes(tb, kb[1], kb[2], kb[3]):
                    return False
    return True
