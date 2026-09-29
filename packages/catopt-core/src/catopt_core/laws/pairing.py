"""Non-local passes over the whole e-graph — NOT equational laws.

The functions in this module are *passes*, not ``lhs → rhs`` rewrite
rules: they walk every e-class, group morphisms by their shared domain
(the product universal property ``⟨f₁,…,f_k⟩ = (f₁ × … × f_k) ∘ Δ``,
applied non-locally), or deduplicate parameters by value, and offer
the fused/deduplicated alternative back into the e-graph with a
pointwise witness.  A term-local law can only fire when a specific
parent op consumes the projections — these passes cover the general
case pattern matching cannot see.

They live beside the laws because they *are* the same universal
properties — just evaluated over the diagram rather than a term.
"""

# ruff: noqa: RUF002, RUF003 -- comments/docstrings use
# mathematical notation (×, ∘, Δ, −) deliberately.

from typing import Any

from catopt_core.ir import Op


def _is_tensor(x: Any) -> bool:
    """Duck-typed "is a tensor" check.

    The sharing passes are called with the backend's leaf values
    (``source_tensors`` from ``export_to_ir``); core must not name the
    tensor type, so a tensor is recognised structurally — ``shape`` /
    ``dtype`` / ``dim`` (a ``str``/``dict``/``None`` entry, which a
    caller may leave in the mapping, has none of these).
    """
    return (
        hasattr(x, "shape")
        and hasattr(x, "dtype")
        and hasattr(x, "dim")
    )


def _exact_equal(a: Any, b: Any) -> bool:
    """Exact value equality of two tensor-like leaves.

    ``torch.equal`` semantics without importing torch: same shape AND
    every element equal, so ``NaN != NaN`` and ``-0.0 == +0.0`` (unlike
    a raw-bytes comparison).  Uses the objects' own ``==`` / ``.all()``,
    keeping core tensor-library-free.
    """
    if tuple(a.shape) != tuple(b.shape):
        return False
    return bool((a == b).all())


# ---------------------------------------------------------------------------
#  Diagram-level pairing pass — the product law in full generality.
#
#  ⟨f₁,…,f_k⟩ = (f₁ × … × f_k) ∘ Δ  is a NON-LOCAL rewrite: it pairs
#  morphisms by their shared domain, not by a consumer pattern.  A
#  term-local lhs→rhs rule can only fire when a specific parent op
#  (mul, sdpa) happens to consume the projections — which is exactly why
#  pattern-matching optimizers miss the general case.  Here we implement
#  it as a pass over the e-graph: group every `linear(x, Wᵢ)` e-node by
#  the e-class of x, then offer each member's class the alternative
#
#      splitᵢ( linear(x, cat(W₁,…,W_k)) )
#
#  so the group may be extracted as ONE GEMM plus k zero-cost views.
#  Subsumes swiglu_fuse / qkv_fuse / qkv_fuse_asym / parallel_mul_fuse.
#  Guards: the shared input must be runtime data (contain a Var), and
#  every weight must be param-only so the concat folds at compile time.
# ---------------------------------------------------------------------------


def _term_has_var(t: Any) -> bool:
    """Return True iff the term mentions a ``Var`` leaf.

    Re-exported from :mod:`catopt_core.laws` for backward compatibility;
    delegates to the single implementation,
    :func:`catopt_core.typing.has_var_leaf`.
    """
    from catopt_core.typing import has_var_leaf

    return has_var_leaf(t)


def _pair_shared_input(
    eg: Any, *, op: str, split_dim: int, cluster_key
) -> list[dict[int, Any]]:
    """Product law over an arbitrary projection signature.

    Groups ``op`` e-nodes by shared input e-class and offers each member
    ``split_i(op(x, cat(W_1..W_k)))`` — one fused kernel plus per-member
    views.  ``cluster_key(enode, weight_term)`` returns a hashable
    signature under which members can share one fused kernel (or None to
    exclude a member): for conv2d it captures stride/padding/dilation/
    groups and the trailing weight dims.
    """
    from catopt_core.egraph import ENode, _LeafRegistry
    from catopt_core.ir import Var as _Var
    from catopt_core.typing import _shape_of as _so

    # Per-class "can some representative reach a Var leaf" — memoized and
    # cycle-guarded.  Replaces materializing any_term + _term_has_var per
    # class (quadratic tree walks on deep graphs) and is *more* sound:
    # it detects var-reachability rather than trusting an arbitrary rep.
    hasvar: dict[int, bool] = {}

    def cls_has_var(cid: int, stack: frozenset = frozenset()) -> bool:
        cid = eg.find(cid)
        if cid in hasvar:
            return hasvar[cid]
        if cid in stack:
            return False
        res = False
        for n in eg._classes[cid].nodes:
            if n.op == "leaf":
                t = _LeafRegistry.decode(n.attrs[0][1])
                if isinstance(t, _Var):
                    res = True
                    break
            elif any(cls_has_var(c, stack | {cid}) for c in n.children):
                res = True
                break
        hasvar[cid] = res
        return res

    by_input: dict[int, list[tuple[ENode, int, int]]] = {}
    for cid in list(eg._classes.keys()):
        for node in eg._classes[cid].nodes:
            # Only bias-free projections: a fused bias would need a
            # second concat; keeping the pairing arity at (x, w).
            if node.op != op or len(node.children) != 2:
                continue
            by_input.setdefault(eg.find(node.children[0]), []).append(
                (node, cid, eg.find(node.children[1]))
            )

    groups: list[dict[int, Any]] = []
    for x_eid, members in by_input.items():
        if not cls_has_var(x_eid):
            continue  # pairing weight-only chains is compile-time noise
        # Cluster members by compat signature: convs differing only in
        # stride/kernel cannot share one fused conv, but each compatible
        # subset still pairs (e.g. two 1x1 heads pair; a 3x3 stays out).
        clusters: dict[Any, list[tuple[ENode, int, int]]] = {}
        for entry in members:
            node, cid, w = entry
            # ``_any_term_cached`` resolves to the class's minimum-size
            # member and returns a STABLE object across calls — the
            # content-keyed ``_SHAPE_MEMO`` in ``_so`` then dedupes shape
            # inference on repeated weights.
            wt = eg._any_term_cached(w)
            if wt is None:
                continue
            # An already-fused member (weight is itself a concat, e.g.
            # produced by swiglu_fuse or an earlier pairing) must not
            # join the group: it would widen the fused GEMM by its own
            # sub-members' outputs — computing them twice.
            if getattr(wt, "op", None) == "concat":
                continue
            k = cluster_key(node, wt)
            if k is None:
                continue
            clusters.setdefault(k, []).append(entry)

        for cluster in clusters.values():
            weights = sorted({w for _, _, w in cluster})
            if len(weights) < 2:
                continue
            if any(cls_has_var(w) for w in weights):
                continue  # fused weight must fold at compile time
            wts = [eg._any_term_cached(w) for w in weights]
            if any(t is None for t in wts):
                continue  # pragma: no cover — memoized _any_term_cached can't differ
            sizes: list[int] = []
            for t in wts:
                s = _so(t)
                if not (isinstance(s, tuple) and len(s) >= 1 and s[0]):
                    break
                sizes.append(s[0])
            if len(sizes) != len(weights):
                continue
            ordered_w, ordered_sizes, cat_args = _tile_fused_weight(
                eg, weights, sizes
            )
            cat = cat_args[0]
            for a in cat_args[1:]:
                cat = eg.add_enode("concat", (cat, a), {"dim": 0})
            fused = eg.add_enode(
                op, (x_eid, cat), dict(cluster[0][0].attrs)
            )
            index_of = {w: i for i, w in enumerate(ordered_w)}
            group: dict[int, Any] = {}
            for _, cid, w in cluster:
                enode = ENode(
                    "split",
                    (fused,),
                    (
                        ("dim", split_dim),
                        ("index", index_of[w]),
                        ("sizes", ordered_sizes),
                    ),
                )
                split_eid = eg.add_enode(
                    "split",
                    (fused,),
                    {
                        "sizes": ordered_sizes,
                        "dim": split_dim,
                        "index": index_of[w],
                    },
                )
                # Replayable witness: the member's own class term ->
                # its section of the fused GEMM.  Pointwise honesty —
                # asserts this instance, exactly what the pass proved.
                split_term = eg._any_term_cached(split_eid)
                eg._offer_witness(
                    cid,
                    split_eid,
                    rhs_term=split_term,
                    provenance="pair",
                    law=(
                        "pointwise witness for a non-local offer: "
                        "this member equals its split section of "
                        "the shared fused weight (equality "
                        "established by the pairing pass)"
                    ),
                )
                group.setdefault(cid, enode)
            groups.append(group)
    return groups


def _wshape(t: Any):
    from catopt_core.typing import _shape_of as _so

    return _so(t)


def _select_tile(eg: Any, w_eid: int):
    """Return ``(base_eid, index, base_shape)`` for dim-0 ``select`` tiles.

    ``None`` unless *w_eid*'s class carries a ``select(base, dim≡0,
    index=i)`` enode over a known-shape base.  Routed-MoE exports
    spell stacked expert parameters ``W (E, out, in)`` as
    ``select(W, 0, e)`` per expert; detected on the *enode* so a
    rule-introduced class representative cannot hide the tile.
    """
    from catopt_core.typing import _shape_of as _so

    for n in eg._classes[eg.find(w_eid)].nodes:
        if n.op != "select" or len(n.children) != 1:
            continue
        a = dict(n.attrs)
        idx = a.get("index")
        if not isinstance(idx, int):
            continue
        base = eg.find(n.children[0])
        bt = eg._any_term_cached(base)
        bs = _so(bt) if bt is not None else None
        if not (
            isinstance(bs, tuple)
            and len(bs) >= 2
            and a.get("dim", 0) % len(bs) == 0
            and all(isinstance(d, int) for d in bs)
        ):
            continue
        return (base, idx, bs)
    return None


def _tile_fused_weight(
    eg: Any, weights: list[int], sizes: list[int]
) -> tuple[list[int], tuple[int, ...], list[int]]:
    """Fused-weight arguments with stacked-parameter re-tiling.

    Member weights spelled ``select(W, dim=0, index=i)`` that jointly
    cover ALL ``W.shape[0]`` tiles of one stacked base concat to
    exactly that base flattened — emit ``reshape(W, (K·o, …))``, a
    free view the lowerer folds to one fused Param, instead of a
    runtime concat over the deliberately non-folding select views
    (which is what made routed-MoE pairing lose extraction: the gate
    side is honestly unroutable, but the fused weight was billed for
    re-assembling itself every call).  Members that only partially
    tile a base keep their own weight e-class as an argument.

    Returns ``(ordered_weights, ordered_sizes, cat_args)`` — the
    member weight e-class ids and their out-dim sizes in fused row
    order, plus the ordered concat-argument e-class ids.
    """
    tiles: dict[int, dict[int, int]] = {}  # base eid -> {index: pos}
    shapes: dict[int, tuple] = {}
    for pos, w in enumerate(weights):
        t = _select_tile(eg, w)
        if t is not None:
            tiles.setdefault(t[0], {})[t[1]] = pos
            shapes[t[0]] = t[2]
    ordered_w: list[int] = []
    ordered_s: list[int] = []
    cat_args: list[int] = []
    seen: set[int] = set()
    for pos, w in enumerate(weights):
        t = _select_tile(eg, w)
        if t is None:
            cat_args.append(w)
            ordered_w.append(w)
            ordered_s.append(sizes[pos])
            continue
        if t[0] in seen:
            continue  # this base's piece already covers the member
        base, _idx, bs = t
        idxs = tiles[base]
        if len(idxs) != bs[0] or sorted(idxs) != list(range(bs[0])):
            # partial tile — the selects stay runtime arguments
            cat_args.append(w)
            ordered_w.append(w)
            ordered_s.append(sizes[pos])
            continue
        seen.add(base)
        merged = (bs[0] * bs[1], *bs[2:])
        cat_args.append(
            eg.add_enode("reshape", (base,), {"shape": merged})
        )
        for i in sorted(idxs):
            ordered_w.append(weights[idxs[i]])
            ordered_s.append(sizes[idxs[i]])
    return ordered_w, tuple(ordered_s), cat_args


def pair_shared_input_linears(eg: Any) -> list[dict[int, Any]]:
    """Pair all `linear` e-nodes that share an input e-class.

    Returns one group per shared input: a dict mapping each member's
    canonical class id to the ``split`` ENode that reads its section of
    the shared fused GEMM.  The caller may feed the union of these dicts
    to ``extract_best`` as ``overrides`` — per-class greedy extraction
    cannot see that all members choosing a split share ONE fused GEMM
    (each split's subtree alone costs more than the member's own
    linear), so the coordinated choice must be forced globally.

    Also runs the expert-sum batching pass,
    :func:`batch_tiled_expert_sums`: the dual of pairing — the per-expert
    down projections of a routed MoE share no input but stack along the
    expert axis, so they batch as ONE grouped GEMM rather than fusing
    into one weight.  Its offers are plain e-class members (no
    overrides): extraction picks them only when honestly cheaper.

    Idempotent: re-running rebuilds the same (hash-consed) enodes.
    """

    def key(enode, wt):
        s = _wshape(wt)
        # 2-D weight only (a 1-D "weight" cannot cat along out-dim).
        return (
            ("lin",) if isinstance(s, tuple) and len(s) == 2 else None
        )

    groups = _pair_shared_input(
        eg, op="linear", split_dim=-1, cluster_key=key
    )
    batch_tiled_expert_sums(eg)
    return groups


# ---------------------------------------------------------------------------
#  Expert-sum batching — the dual of shared-input pairing.
#
#  A routed-MoE block in dense-eval form ends with
#
#      Σ_e g_e ⊙ linear(h_e, select(W, dim=0, e))        W : (E, o, i)
#
#  — per-expert projections on DISTINCT gated activations.  The product
#  law cannot pair them (no shared domain), but they are slices of one
#  stacked parameter, so the whole sum is ONE grouped GEMM:
#
#      reshape(sum(mul(bmm(stack h_e, W.mT), stack g_e), 0))
#
#  torch.matmul right-aligns batch dims, so the (E, …tokens…, i) stack
#  is flattened to rank 3 ``(E, N, i)`` for the bmm and reshaped back —
#  both reshapes are free views.  The weight side is ``transpose(W)``
#  on the stacked Param, which ``_fold_weight_chains`` folds at
#  lowering — zero runtime cost, unlike a concat over select tiles.
# ---------------------------------------------------------------------------


def _linear_tile(eg: Any, cid: int):
    """Return ``(base_eid, index, input_eid, base_shape)`` of a tile.

    ``None`` unless *cid*'s class carries a bias-free
    ``linear(h, select(base, dim=0, index=i))`` enode over a rank-3
    stacked base ``(E, out, in)`` — the per-expert projection spelling
    a routed-MoE export produces.
    """
    for n in eg._classes[eg.find(cid)].nodes:
        if n.op != "linear" or len(n.children) != 2:
            continue
        t = _select_tile(eg, n.children[1])
        if t is not None and len(t[2]) == 3:
            return (t[0], t[1], eg.find(n.children[0]), t[2])
    return None


def _expert_leaf(eg: Any, cid: int):
    """Classify an add-tree leaf of an expert sum.

    Returns ``(leaf_cid, base, index, h_eid, gate_eid | None)`` —
    ``gate_eid`` is set when the leaf is spelled
    ``mul(linear(h, tile), g)`` (either argument order).  The gated
    spelling is preferred over a bare ``linear`` member: rule
    enrichment can rewrite ``g ⊙ (h @ Wᵀ)`` as ``(g ⊙ h) @ Wᵀ`` inside
    the same class, and treating the leaf as gated stacks the gate
    factors for one broadcast multiply instead of leaving E serial
    gate muls inside the inputs.

    ``None`` when the class is neither a tiled linear nor a gated one.
    """
    for n in eg._classes[cid].nodes:
        if n.op != "mul" or len(n.children) != 2:
            continue
        for a, b in ((0, 1), (1, 0)):
            lt = _linear_tile(eg, n.children[a])
            if lt is not None:
                return (
                    cid,
                    lt[0],
                    lt[1],
                    lt[2],
                    eg.find(n.children[b]),
                )
    lt = _linear_tile(eg, cid)
    if lt is not None:
        return (cid, lt[0], lt[1], lt[2], None)
    return None


def _decompose_expert_sum(
    eg: Any,
    cid: int,
    visiting: frozenset,
    leaves: list,
    residuals: list,
) -> None:
    """Walk an ``add`` tree, splitting it into expert leaves and others.

    ``leaves`` collects :func:`_expert_leaf` records; ``residuals``
    collects every class that is neither an ``add`` member nor an
    expert leaf — residual terms (e.g. a stream skip connection) and
    classes revisited on a cycle.  ``visiting`` is the DFS path set —
    a cyclic class lands in ``residuals`` rather than recursing.
    """
    cid = eg.find(cid)
    if cid in visiting:
        residuals.append(cid)
        return
    for n in eg._classes[cid].nodes:
        if n.op == "add" and len(n.children) == 2:
            visiting = visiting | {cid}
            _decompose_expert_sum(
                eg, n.children[0], visiting, leaves, residuals
            )
            _decompose_expert_sum(
                eg, n.children[1], visiting, leaves, residuals
            )
            return
    leaf = _expert_leaf(eg, cid)
    if leaf is not None:
        leaves.append(leaf)
    else:
        residuals.append(cid)


def _has_add(eg: Any, cid: int) -> bool:
    """Return True iff *cid*'s class carries a binary ``add`` enode."""
    return any(
        n.op == "add" and len(n.children) == 2
        for n in eg._classes[eg.find(cid)].nodes
    )


def _uniform_input(eg: Any, lfs: list):
    """Return the shared expert-input shape — ``None`` when unstackable.

    The batched operand ``stack(h_0..h_{E-1})`` is spelled only when
    every ``h_e`` resolves to ONE fully-known shape of rank >= 2:
    ``torch.stack`` needs equal shapes and the flatten-to-rank-3
    reshape needs literal dims.
    """
    from catopt_core.typing import _shape_of as _so

    hs = [_so(eg._any_term_cached(lf[3])) for lf in lfs]
    h0 = hs[0]
    if not (
        isinstance(h0, tuple)
        and len(h0) >= 2
        and all(isinstance(d, int) for d in h0)
        and all(s == h0 for s in hs)
    ):
        return None
    return h0


def _gates_uniform(eg: Any, lfs: list, h0: tuple):
    """Uniform gating verdict — ``True``/``False`` gated, ``None`` to veto.

    Mixed gated/ungated leaves have no uniform batched spelling; gate
    factors must be per-token scalars ``(…, 1)`` — the only shape that
    broadcasts against the flattened ``(E, N, o)`` bmm output.
    """
    from catopt_core.typing import _shape_of as _so

    gated = all(lf[4] is not None for lf in lfs)
    if not gated and any(lf[4] is not None for lf in lfs):
        return None  # mixed gating — no uniform batched spelling
    if gated and any(
        _so(eg._any_term_cached(lf[4])) != (*h0[:-1], 1) for lf in lfs
    ):
        return None
    return gated


def _cover_check(eg: Any, base: int, lfs: list):
    """Guard one candidate base; return the offer facts or ``None``.

    On success returns
    ``(base_eid, base_shape, ordered_leaves, h_shape, gated)``.
    """
    from catopt_core.typing import _shape_of as _so

    idxs = sorted(lf[2] for lf in lfs)
    bs = _so(eg._any_term_cached(base))
    # ``bs`` is already an int tuple — ``_select_tile`` validated
    # the base; the isinstance narrows it for the typechecker.
    if not (
        isinstance(bs, tuple)
        and len(set(idxs)) == len(idxs)
        and idxs == list(range(bs[0]))
    ):
        return None
    h0 = _uniform_input(eg, lfs)
    if h0 is None:
        return None
    gated = _gates_uniform(eg, lfs, h0)
    if gated is None:
        return None
    return base, bs, sorted(lfs, key=lambda lf: lf[2]), h0, gated


def _gemm_cover(eg: Any, leaves: list):
    """Find a stacked base whose tiles *leaves* fully and safely cover.

    Groups the decomposed leaves by their stacked base and returns the
    first group's :func:`_cover_check` facts, else ``None``.
    """
    by_base: dict[int, list] = {}
    for lf in leaves:
        by_base.setdefault(lf[1], []).append(lf)
    for base, lfs in by_base.items():
        found = _cover_check(eg, base, lfs)
        if found is not None:
            return found
    return None


def _build_grouped_gemm(
    eg: Any, base: int, bs: tuple, ordered: list, h0: tuple, gated: bool
) -> int:
    """Assemble the batched member; return its e-class id.

    ``stack(h_0..h_{E-1}) -> reshape(E, N, i)`` — ``torch.matmul``
    right-aligns batch dims, so the expert stack must flatten to
    rank 3 — then ``matmul(., transpose(base))``, an optional
    broadcast gate multiply over ``reshape(stack g, (E, N, 1))``,
    ``sum(dim=0)``, and a reshape back to the leaf output shape.
    Children stay e-class ids so extraction keeps sharing each
    member's best subterm.
    """
    n_flat = 1
    for d in h0[:-1]:
        n_flat *= d
    hb = eg.add_enode(
        "stack", tuple(lf[3] for lf in ordered), {"dim": 0}
    )
    hbf = eg.add_enode(
        "reshape", (hb,), {"shape": (bs[0], n_flat, h0[-1])}
    )
    wt = eg.add_enode("transpose", (base,), {"dim0": -2, "dim1": -1})
    yb = eg.add_enode("matmul", (hbf, wt), {})
    if gated:
        gb = eg.add_enode(
            "stack", tuple(lf[4] for lf in ordered), {"dim": 0}
        )
        gb = eg.add_enode(
            "reshape", (gb,), {"shape": (bs[0], n_flat, 1)}
        )
        yb = eg.add_enode("mul", (yb, gb), {})
    acc = eg.add_enode("sum", (yb,), {"dim": 0})
    return eg.add_enode("reshape", (acc,), {"shape": (*h0[:-1], bs[1])})


def _offer_grouped_gemm(
    eg: Any, cid: int, leaves: list, residuals: list
) -> bool:
    """Offer the grouped-GEMM member for one decomposed sum class.

    Unguarded leftovers — residual classes plus leaves of bases this
    offer does not cover — rejoin through a rebuilt ``add`` chain, so
    the offered member is term-for-term equal to the decomposed sum.
    The member lands in *cid* under a pointwise witness.

    Returns True when an offer was made.
    """
    found = _gemm_cover(eg, leaves)
    if found is None:
        return False
    base, bs, ordered, h0, gated = found
    acc = _build_grouped_gemm(eg, base, bs, ordered, h0, gated)
    covered = {lf[0] for lf in ordered}
    for r in residuals + [
        lf[0] for lf in leaves if lf[0] not in covered
    ]:
        acc = eg.add_enode("add", (acc, eg.find(r)), {})
    eg._offer_witness(
        cid,
        acc,
        rhs_term=eg._any_term_cached(acc),
        provenance="batch",
        law=(
            "pointwise witness for expert-sum batching: the gated "
            "per-expert sum over one stacked parameter equals one "
            "grouped GEMM (stack -> bmm -> gate-mul -> sum) — "
            "equality established by the batching pass"
        ),
    )
    return True


def batch_tiled_expert_sums(eg: Any) -> int:
    """Offer grouped-GEMM members for sums of weight-tiled linears.

    Scans every e-class carrying an ``add`` enode, decomposes its
    add-tree into expert leaves (``linear`` or ``mul(linear, gate)``
    over ``select`` tiles of one rank-3 stacked base) and residuals,
    and — when some base's tiles are fully covered with compatible
    shapes — offers the class the batched spelling

    ``reshape(sum(mul(bmm(stack h, W.mT), stack g), dim=0), y_shape)``

    via :meth:`EGraph._offer_witness`.  The member competes as an
    ordinary alternative — extraction picks it only when the cost
    model honestly prefers one bmm + two stacks over E serial linears;
    there is nothing to force and nothing to roll back.

    Returns the number of offered members.
    """
    offers = 0
    for cid in list(eg._classes.keys()):
        cid = eg.find(cid)
        if not _has_add(eg, cid):
            continue
        leaves: list = []
        residuals: list = []
        _decompose_expert_sum(eg, cid, frozenset(), leaves, residuals)
        if len(leaves) < 2:
            continue
        # A residual still carrying an add enode is a cyclic re-entry
        # — folding it into the offer would build a self-referential
        # member extraction cannot term-ify.
        if any(_has_add(eg, r) for r in residuals):
            continue
        if _offer_grouped_gemm(eg, cid, leaves, residuals):
            offers += 1
    return offers


_CONV_ATTR_KEYS = ("stride", "padding", "dilation", "groups")


def pair_shared_input_convs(eg: Any) -> list[dict[int, Any]]:
    """Pair `conv2d` e-nodes sharing an input — the same product law.

    The fused weight is ``cat`` along out-channels (dim 0), valid only
    when members share stride/padding/dilation/groups and kernel dims;
    the projections split the output along the channel dim (1).
    """

    def key(enode, wt):
        a = dict(enode.attrs)
        if a.get("groups", 1) != 1:
            return None  # grouped conv: cat on O mixes groups wrongly
        s = _wshape(wt)
        if not (isinstance(s, tuple) and len(s) == 4):
            return None
        # same non-weight attrs AND same (in_ch, kh, kw) — cat on O
        # requires identical trailing weight dims.
        return (
            tuple(sorted((k, a[k]) for k in _CONV_ATTR_KEYS if k in a)),
            s[1:],
        )

    return _pair_shared_input(
        eg, op="conv2d", split_dim=1, cluster_key=key
    )


def share_duplicate_params(
    eg: Any, source_tensors: dict, *, witness: bool = True
) -> list[list[str]]:
    """Union the e-classes of Param leaves holding identical tensors.

    Exact weight *tying* discovered, not declared.

    If ``p_i`` and ``p_j`` contain equal values, substituting either for
    the other is an exact semantic equality.  After the merge,
    deterministic extraction keeps one representative everywhere; the
    dropped name never enters the extracted term, so ``_build_params``
    omits it from the optimized module's state dict — the stored
    weights file shrinks by the duplicate's size with zero cost.

    This catches the real-world cases: tied embeddings/classifiers
    (llama2.c ``wcls``), duplicated adapter or branch weights, and any
    GQA head sharing the checkpoint materialised as separate tensors.

    ``source_tensors`` maps Param names to their tensors (as
    ``export_to_ir`` returns).  Returns the groups of names merged.
    Each merge carries a pointwise witness, like the other non-local
    passes — certificates replay it as a rule step.
    """
    from catopt_core.ir import Param, TensorType

    by_sig: dict[tuple, list[str]] = {}
    for name, t in source_tensors.items():
        if not _is_tensor(t):
            continue
        key = (tuple(t.shape), str(t.dtype))
        by_sig.setdefault(key, []).append(name)

    groups: list[list[str]] = []
    for _key, names in by_sig.items():
        if len(names) < 2:
            continue
        # split by exact value equality (cheap: hash first, confirm)
        reps: list[str] = []
        clusters: list[list[str]] = []
        for name in names:
            placed = False
            for ci, rep in enumerate(reps):
                if _exact_equal(
                    source_tensors[name], source_tensors[rep]
                ):
                    clusters[ci].append(name)
                    placed = True
                    break
            if not placed:
                reps.append(name)
                clusters.append([name])
        for cluster in clusters:
            if len(cluster) > 1:
                groups.append(cluster)

    for cluster in groups:
        canon = cluster[0]
        ct = source_tensors[canon]
        canon_term = Param(
            canon, TensorType(tuple(int(d) for d in ct.shape))
        )
        canon_eid = eg.add_term(canon_term)
        for name in cluster[1:]:
            t = source_tensors[name]
            p = Param(name, TensorType(tuple(int(d) for d in t.shape)))
            eid = eg.add_term(p)
            eg._offer_witness(
                eid,
                canon_eid,
                rhs_term=canon_term,
                lhs_term=p,
                provenance="share",
                name_eid=eid,
                law=(
                    "pointwise witness for exact weight tying: "
                    "the two parameter leaves hold bitwise-equal "
                    "tensors (equality established by the sharing "
                    "pass over source tensors)"
                ),
                witness=witness,
                note=f"share_duplicate_params: {name} == {canon}",
            )
    return groups


def share_duplicate_param_slices(
    eg: Any,
    source_tensors: dict,
    *,
    witness: bool = True,
    head_counts: range = range(2, 65),
) -> list[dict]:
    """Slice-level weight sharing for one 2-D parameter.

    Deduplicates bitwise-equal row-blocks INSIDE a single 2-D parameter —
    the intra-tensor analogue of :func:`share_duplicate_params`.

    A projection weight W (o×i) in a multi-head architecture packs h
    per-head row-blocks of ``d_head = o / h`` rows (GQA/MHA QKV, fused
    per-head gates).  When two heads are bitwise equal — tied heads,
    duplicated branches, GQA replication baked into the checkpoint —
    they need only one storage copy:

        W = reshape(index_select(D, dim=0, index=map), (o, i))

    where ``D`` is a NEW ``(k, d_head, i)`` parameter stacking the
    ``k < h`` distinct head-blocks in first-occurrence order and
    ``map`` sends each head slot to its surviving block.  The member is
    offered to W's e-class with a pointwise witness — an exact equality
    (``error_bound=None``), so certificates replay it as a rule step.

    Two representation notes:

    * ``index_select`` gathers at HEAD granularity (dim 0 of the
      stacked ``(k, d_head, i)`` dedup tensor, index length h), not per
      row — the index map stays a readable per-head permutation.
    * The deduplicated tensor is REGISTERED INTO ``source_tensors``
      under a fresh ``{name}__heads{h}`` key.  Lowering passes the same
      dict as ``param_values`` to ``ir_to_torch_module``, so the
      offered member materialises with the correct values and the
      extracted module's state dict holds only the unique blocks.

    Guards: 2-D params only; only head counts in ``head_counts``
    dividing ``o``; fires only when the stored values strictly shrink
    (``k·d_head·i < o·i``, i.e. some head group has ≥ 2 members); the
    param must actually appear in the e-graph.  Equality is BITWISE —
    chunks are grouped by their raw bytes, stricter than
    ``torch.equal`` (which conflates −0.0/+0.0 and treats NaN payloads
    as never equal).

    Extraction note: flop/count cost models charge param-only subtrees
    0, so under them the plain leaf always ties and wins on size — the
    member is then selectable only via ``extract_best`` overrides (the
    same coordinated-extraction contract as the diagram pairing pass).
    Under the storage axis — ``catopt_core.cost.param_bytes_cost_for``
    (``charges_param_only``) — the member prices at k·d_head·i against
    the leaf's o·i and wins directly, no override needed.

    Returns one record per offered member:
    ``{param, heads, head_dim, unique, index_map, dedup_param,
      stored_before, stored_after, eid}``.
    """
    from catopt_core.egraph import ENode
    from catopt_core.ir import Param, TensorType

    offers: list[dict] = []
    for name, t in list(source_tensors.items()):
        if not _is_tensor(t) or t.dim() != 2:
            continue
        o, i = int(t.shape[0]), int(t.shape[1])
        leaf = ENode("leaf", (), (("key", name),))
        w_eid = eg._node_to_class.get(leaf)
        if w_eid is None:
            continue  # param not in the graph — nothing to offer to

        # Try each candidate head count; keep the factorisation with
        # the smallest surviving storage (k·d_head·i).
        best = None  # (stored, h, d_head, imap, unique_chunks)
        for h in head_counts:
            if h < 2 or h > o or o % h != 0:
                continue
            d = o // h
            # Group head-blocks by raw bytes — bitwise equality, not
            # torch.equal's value equality.
            sigs = [
                t[j * d : (j + 1) * d]
                .detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
                for j in range(h)
            ]
            uniq: list[bytes] = []
            imap_build: list[int] = []
            for s in sigs:
                try:
                    imap_build.append(uniq.index(s))
                except ValueError:
                    imap_build.append(len(uniq))
                    uniq.append(s)
            k = len(uniq)
            stored = k * d * i
            if k < h and (best is None or stored < best[0]):
                first = [imap_build.index(u) for u in range(k)]
                best = (
                    stored,
                    h,
                    d,
                    tuple(imap_build),
                    [t[f * d : (f + 1) * d] for f in first],
                )
        if best is None:
            continue
        stored, h, d, imap, uniques = best
        k = len(uniques)

        # Stack the unique head-blocks WITHOUT importing torch: a fresh
        # zero tensor of the right shape (dtype/device inherited from the
        # blocks via ``new_zeros``) filled in place, then detached — the
        # duck-typed equivalent of ``torch.stack(uniques).detach()``.
        dedup = uniques[0].new_zeros((k, d, i))
        for j, u in enumerate(uniques):
            dedup[j] = u
        dedup = dedup.detach()
        dedup_name = f"{name}__heads{h}"
        # Fresh registration; reuse an identical prior entry on re-run.
        if not (
            dedup_name in source_tensors
            and _exact_equal(source_tensors[dedup_name], dedup)
        ):
            base_name, n = dedup_name, 0
            while dedup_name in source_tensors:
                n += 1
                dedup_name = f"{base_name}_{n}"
            source_tensors[dedup_name] = dedup

        p_dedup = Param(dedup_name, TensorType((k, d, i)))
        member = Op.make(
            "reshape",
            Op.make("index_select", p_dedup, dim=0, index=imap),
            shape=(o, i),
        )
        member_eid = eg.add_term(
            member, provenance="share_duplicate_param_slices"
        )
        w_term = Param(name, TensorType((o, i)))
        eg._offer_witness(
            w_eid,
            member_eid,
            rhs_term=member,
            lhs_term=w_term,
            provenance="share_slices",
            law=(
                "pointwise witness for slice-level weight sharing: "
                "the parameter's head row-blocks are bitwise-equal "
                "to blocks of the deduplicated stack under "
                "index_map (equality established by the sharing "
                "pass over source tensors)"
            ),
            witness=witness,
            note=(
                f"share_duplicate_param_slices: {name} (o={o}) "
                f"= {h} heads x {d} rows -> {k} unique"
            ),
        )
        offers.append(
            {
                "param": name,
                "heads": h,
                "head_dim": d,
                "unique": k,
                "index_map": imap,
                "dedup_param": dedup_name,
                "stored_before": o * i,
                "stored_after": k * d * i,
                "eid": member_eid,
            }
        )
    return offers
