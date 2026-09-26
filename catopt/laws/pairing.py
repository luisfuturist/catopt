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

from catopt.egraph import Rewrite
from catopt.ir import Op

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
    from catopt.ir import Var

    if isinstance(t, Var):
        return True
    if isinstance(t, Op):
        return any(_term_has_var(a) for a in t.args)
    return False


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
    from catopt.egraph import ENode, _LeafRegistry
    from catopt.ir import Var as _Var
    from catopt.typing import _shape_of as _so

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
            cat = weights[0]
            for w in weights[1:]:
                cat = eg.add_enode("concat", (cat, w), {"dim": 0})
            fused = eg.add_enode(
                op, (x_eid, cat), dict(cluster[0][0].attrs)
            )
            index_of = {w: i for i, w in enumerate(weights)}
            group: dict[int, Any] = {}
            for _, cid, w in cluster:
                enode = ENode(
                    "split",
                    (fused,),
                    (
                        ("dim", split_dim),
                        ("index", index_of[w]),
                        ("sizes", tuple(sizes)),
                    ),
                )
                split_eid = eg.add_enode(
                    "split",
                    (fused,),
                    {
                        "sizes": tuple(sizes),
                        "dim": split_dim,
                        "index": index_of[w],
                    },
                )
                # Replayable witness: the member's own class term ->
                # its section of the fused GEMM.  Pointwise honesty —
                # asserts this instance, exactly what the pass proved.
                src = getattr(eg, "_oldest_term", eg.any_term)(cid)
                split_term = eg._any_term_cached(split_eid)
                wit = None
                if src is not None and split_term is not None:
                    wit = Rewrite(
                        name=f"pair#{split_eid}",
                        lhs=src,
                        rhs=split_term,
                        law=(
                            "pointwise witness for a non-local offer: "
                            "this member equals its split section of "
                            "the shared fused weight (equality "
                            "established by the pairing pass)"
                        ),
                    )
                eg.union(cid, split_eid, witness=wit)
                group.setdefault(cid, enode)
            groups.append(group)
    return groups


def _wshape(t: Any):
    from catopt.typing import _shape_of as _so

    return _so(t)


def pair_shared_input_linears(eg: Any) -> list[dict[int, Any]]:
    """Pair all `linear` e-nodes that share an input e-class.

    Returns one group per shared input: a dict mapping each member's
    canonical class id to the ``split`` ENode that reads its section of
    the shared fused GEMM.  The caller may feed the union of these dicts
    to ``extract_best`` as ``overrides`` — per-class greedy extraction
    cannot see that all members choosing a split share ONE fused GEMM
    (each split's subtree alone costs more than the member's own
    linear), so the coordinated choice must be forced globally.

    Idempotent: re-running rebuilds the same (hash-consed) enodes.
    """

    def key(enode, wt):
        s = _wshape(wt)
        # 2-D weight only (a 1-D "weight" cannot cat along out-dim).
        return (
            ("lin",) if isinstance(s, tuple) and len(s) == 2 else None
        )

    return _pair_shared_input(
        eg, op="linear", split_dim=-1, cluster_key=key
    )


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
    """Union the e-classes of Param leaves holding identical tensors —
    exact weight *tying* discovered, not declared.

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
    import torch as _t

    from catopt.ir import Param, TensorType

    by_sig: dict[tuple, list[str]] = {}
    for name, t in source_tensors.items():
        if not isinstance(t, _t.Tensor):
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
                if _t.equal(source_tensors[name], source_tensors[rep]):
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
            wit = None
            if witness:
                wit = Rewrite(
                    name=f"share#{eid}",
                    lhs=p,
                    rhs=canon_term,
                    law=(
                        "pointwise witness for exact weight tying: "
                        "the two parameter leaves hold bitwise-equal "
                        "tensors (equality established by the sharing "
                        "pass over source tensors)"
                    ),
                )
            eg.union(
                eid,
                canon_eid,
                witness=wit,
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
    """Slice-level weight sharing: deduplicate bitwise-equal row-blocks
    INSIDE a single 2-D parameter — the intra-tensor analogue of
    :func:`share_duplicate_params`.

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
    Under the storage axis — ``catopt.cost.param_bytes_cost_for``
    (``charges_param_only``) — the member prices at k·d_head·i against
    the leaf's o·i and wins directly, no override needed.

    Returns one record per offered member:
    ``{param, heads, head_dim, unique, index_map, dedup_param,
      stored_before, stored_after, eid}``.
    """
    import torch as _t

    from catopt.egraph import ENode
    from catopt.ir import Param, TensorType

    offers: list[dict] = []
    for name, t in list(source_tensors.items()):
        if not isinstance(t, _t.Tensor) or t.dim() != 2:
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
            imap: list[int] = []
            for s in sigs:
                try:
                    imap.append(uniq.index(s))
                except ValueError:
                    imap.append(len(uniq))
                    uniq.append(s)
            k = len(uniq)
            stored = k * d * i
            if k < h and (best is None or stored < best[0]):
                first = [imap.index(u) for u in range(k)]
                best = (
                    stored,
                    h,
                    d,
                    tuple(imap),
                    [t[f * d : (f + 1) * d] for f in first],
                )
        if best is None:
            continue
        stored, h, d, imap, uniques = best
        k = len(uniques)

        with _t.no_grad():
            dedup = _t.stack(uniques, dim=0).detach().clone()
        dedup_name = f"{name}__heads{h}"
        # Fresh registration; reuse an identical prior entry on re-run.
        if not (
            dedup_name in source_tensors
            and _t.equal(source_tensors[dedup_name], dedup)
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
        wit = None
        if witness:
            wit = Rewrite(
                name=f"share_slices#{member_eid}",
                lhs=w_term,
                rhs=member,
                law=(
                    "pointwise witness for slice-level weight sharing: "
                    "the parameter's head row-blocks are bitwise-equal "
                    "to blocks of the deduplicated stack under "
                    "index_map (equality established by the sharing "
                    "pass over source tensors)"
                ),
                error_bound=None,
            )
        eg.union(
            w_eid,
            member_eid,
            witness=wit,
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
