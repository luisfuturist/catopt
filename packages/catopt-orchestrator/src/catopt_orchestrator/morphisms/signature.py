"""Stage 0 — block signatures.

The structural signature of one block, read off its exported IR term:
projections, norms, activations, the residual spine, and the per-input
roles.  Signatures describe *structure* — never ``nn.Module`` objects —
so the morphism engine stays backend-neutral.  Split out of
:mod:`catopt_orchestrator.morphisms` (plan 0011); the whole surface is
re-exported from that package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catopt_core.ir import IR, Op, Param, Var
from catopt_core.laws.pairing import _is_tensor
from catopt_core.typing import has_var_leaf, shape_of

# ---------------------------------------------------------------------------
#  Stage 0 — block signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightRef:
    """One projection weight inside a block's IR — a signature term.

    ``term`` is the weight operand as it appears in the projection op:
    a :class:`~catopt_core.ir.Param` leaf in the common case, or a
    param-only (already-folded) weight expression.  ``name`` is the
    ``Param`` name for leaf weights (``""`` for compound terms) and
    ``shape`` its weight shape when known.  Weight refs are the atoms
    the morphism algebra composes: ``A.out_proj ∘ B.in_proj``.
    """

    term: Any
    name: str
    shape: tuple | None


@dataclass(frozen=True)
class NormSig:
    """The block's normalisation / diagonal-scale signature.

    ``kind`` ∈ ``{"none", "layer_norm", "rms", "diag"}`` — a LayerNorm
    op, an RMS-style ``x · rms⁻¹ · w`` pattern (either spelling: the
    pointwise ``mul`` chain or the fused ``rms_norm`` op), or a bare
    diagonal (elementwise gain) map.  ``affine`` records a *learnable* gain
    (a ``Param`` inside the scale term — a ``Const`` scalar scale is a
    diagonal map but not affine).  ``pre`` is True when the norm sits
    on the path from the block input to an in-projection — the
    pre-norm position whose gain cascades into the next weights.
    """

    kind: str
    affine: bool
    pre: bool


#: Op names counted as activations on a block's spine.
_ACT_OPS = frozenset(
    {
        "silu",
        "gelu",
        "tanh",
        "sigmoid",
        "relu",
        "softmax",
        "sdpa",
        "exp",
    }
)


@dataclass(frozen=True)
class InputSig:
    """One block input's role in the signature — the multi-input arm.

    A block's ``ir.inputs`` are its call's positional tensor args (in
    order).  Each classifies into a ``kind``:

    * ``"activation"`` — the computed stream the block transforms:
      an input that *enters* computation as data (a projection's data
      operand, a norm's subject, a bare ``add``/``sub`` stream
      addend, an ``sdpa`` operand).  Exactly one may exist for the
      block to lift.
    * ``"const_table"`` — context that rides alongside the stream in
      positions a ``Param``/``Const`` could fill (the pointwise
      factor/mask/table roles — rope ``cos``/``sin`` tables,
      embedding/gather/index sources, ``sdpa`` masks).  Passes
      through a composition unchanged.
    * ``"state"`` — call-varying context (``role`` ``"read_only"`` —
      the refined kind for a context input whose captured value
      differs across the two lift probes) or a mutable buffer the
      block *writes* (``role`` ``"mutated"`` — it is arg0 of a
      write op; an honest lift decline).
    * ``"opaque"`` — an unhandled use (a runtime-supplied weight,
      an op the signature cannot classify).  Honest decline.

    ``role`` is the finer structural descriptor (``"stream"`` /
    ``"table"`` / ``"dead"`` / ``"mutated"`` / ``"read_only"`` /
    ``"unhandled"``); ``index`` is the position in ``ir.inputs`` and
    in the captured call args (lift requires tensor-only args, so the
    two orders coincide).
    """

    index: int
    name: str
    kind: str
    role: str
    shape: tuple | None


#: Write ops — arg0 is the mutated base in the functionalised IR
#: spelling (``copy_`` threads as ``copy``/``*_scatter``).
_MUT_OPS = frozenset(
    {
        "copy",
        "index_put",
        "index_add",
        "slice_scatter",
        "select_scatter",
        "scatter",
        "scatter_add",
        "scatter_reduce",
    }
)

#: Transparent view/read ops — a var passing through arg0 keeps its
#: pending classification: the *terminal* (non-view) consumer decides
#: the role (``unsqueeze(cos) → mul`` is still a table read;
#: ``slice(x) → linear`` is still a stream entry).
_VIEW_OPS = frozenset(
    {
        "slice",
        "select",
        "narrow",
        "getitem",
        "unsqueeze",
        "squeeze",
        "reshape",
        "view",
        "expand",
        "expand_as",
        "broadcast_to",
        "permute",
        "transpose",
        "flatten",
        "unflatten",
        "movedim",
        "contiguous",
        "detach",
        "detach_",
        "clone",
        "to",
        "type_as",
        "float",
        "double",
        "half",
        "bfloat16",
        "repeat",
        "chunk",
        "split",
        "tensor_split",
        "unbind",
        "roll",
        "flip",
        "pad",
        "triu",
        "tril",
        "alias",
    }
)

#: Multi-operand pointwise ops — the "context position" test: a var
#: operand whose sibling args carry a var is a factor/table riding
#: the stream; one whose siblings are all var-free IS the stream.
_POINTWISE_OPS = frozenset(
    {
        "mul",
        "div",
        "pow",
        "fmod",
        "remainder",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "atan2",
        "xlogy",
        "heaviside",
        "isclose",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
        "logical_or",
        "logical_xor",
        "bitwise_and",
        "bitwise_or",
        "where",
        "lerp",
        "clamp",
        "clamp_min",
        "clamp_max",
        "addcmul",
        "addcdiv",
        "masked_fill",
    }
)

#: Ops whose operand positions are all table/index roles — the var
#: is read like a lookup table, never streamed.
_TABLE_OPS = frozenset(
    {
        "embedding",
        "index",
        "index_select",
        "gather",
        "take_along_dim",
        "searchsorted",
        "one_hot",
        "nonzero",
        "item",
        "numel",
    }
)

#: Ops where the var operand is a norm's *subject* at position 0 —
#: stream entries.  A var at a later position is a runtime weight —
#: unhandled.
_NORM_OPS = frozenset(
    {"layer_norm", "rms_norm", "batch_norm", "group_norm"}
)


def _use_roles_positional(op: str, pos: int) -> str | None:
    """Classify positional-stream ops — ``None`` when unhandled here.

    Projections, convs and norms take the stream at position 0 — a
    var anywhere else is a runtime weight (``opaque``).  ``sdpa``'s
    query is the stream; its k/v/mask operands are context reads
    (a ``(x, kv_cache)`` block's cache state).
    """
    if op in ("linear", "matmul", "einsum", "conv1d", "conv2d"):
        # A var operand is data — ``conv``/``matmul`` weights arrive
        # as params in normal exports; a var weight is still a data
        # read (the op computes on it), not a signature projection.
        return "stream" if pos == 0 else "opaque"
    if op == "sdpa":
        return "stream" if pos == 0 else "context"
    if op in _NORM_OPS:
        return "stream" if pos == 0 else "opaque"
    return None


def _use_roles(n: Op, pos: int) -> str:
    """Classify one terminal use of an input var: stream or context.

    ``"stream"`` — the var enters as computed data; ``"context"`` —
    it sits in a position a ``Param``/``Const`` table could fill;
    ``"mutated"`` — it is the destination of a write op;
    ``"opaque"`` — the use is one the signature cannot classify.
    """
    op = n.op
    if op in _MUT_OPS:
        return "mutated" if pos == 0 else "opaque"
    role = _use_roles_positional(op, pos)
    if role is not None:
        return role
    if op in ("add", "sub", "concat", "stack", "hstack", "vstack"):
        # A raw addend/concatenand is a consumed read — context; the
        # activation's stream role comes through a projection/norm/
        # attention use, not by being summed.
        return "context"
    if op in _POINTWISE_OPS:
        siblings = [a for j, a in enumerate(n.args) if j != pos]
        return (
            "context"
            if any(has_var_leaf(a) for a in siblings)
            else "stream"
        )
    if op in _TABLE_OPS:
        return "context"
    return "opaque"


def _var_uses(root: Any, v: Var, parents: dict[Any, list]) -> list[str]:
    """Collect the terminal use-roles of one input var.

    Follows transparent view ops (``_VIEW_OPS`` arg0) to the
    consuming terminal op; a var reaching the root through views —
    or being the root — counts as stream (the output IS the var,
    transformed).  Non-arg0 uses inside view ops (the index list of
    an ``index``…) are terminal ``context`` uses.
    """
    roles: list[str] = []
    work = [(p, pos) for p, pos in parents.get(v, ())]
    seen: set = set()
    while work:
        n, pos = work.pop()
        if n in seen:
            continue
        seen.add(n)
        if n.op in _VIEW_OPS and pos == 0:
            consumers = parents.get(n)
            if consumers is None:
                # The view result IS the block root — the var flows
                # straight to the output: a stream.
                roles.append("stream")
            else:
                work.extend(consumers)
        elif n.op in _VIEW_OPS:
            roles.append("context")
        else:
            roles.append(_use_roles(n, pos))
    return roles


def _classify_inputs(ir: IR) -> tuple[InputSig, ...]:
    """Classify each of the block's ``ir.inputs`` — term level only.

    Single-input IRs keep the historical semantics: the one input is
    the activation, unconditionally.  For multi-input IRs the
    per-var terminal uses vote: any ``mutated`` use marks the input
    ``"state"``/``"mutated"``; any ``opaque`` use makes it opaque;
    any ``stream`` use makes it an activation candidate; the rest
    are context — provisionally ``"const_table"`` (lift's probe
    evidence may refine the kind to ``"state"``/``"read_only"``).
    """
    out: list[InputSig] = []
    if len(ir.inputs) == 1:
        v = ir.inputs[0]
        return (
            InputSig(0, v.name, "activation", "stream", v.typ.shape),
        )
    parents: dict[Any, list] = {}
    for n in _iter_ops(root := ir.root):
        for pos, a in enumerate(n.args):
            parents.setdefault(a, []).append((n, pos))
    if not isinstance(root, Op):
        parents.setdefault(root, [])
    for i, v in enumerate(ir.inputs):
        roles = _var_uses(root, v, parents)
        if root == v:
            roles.append("stream")
        kind, role = _kind_of_roles(roles)
        out.append(InputSig(i, v.name, kind, role, v.typ.shape))
    return tuple(out)


def _kind_of_roles(roles: list[str]) -> tuple[str, str]:
    """Fold a var's use-roles into its ``(kind, role)``."""
    if any(r == "mutated" for r in roles):
        return "state", "mutated"
    if any(r == "opaque" for r in roles):
        return "opaque", "unhandled"
    if any(r == "stream" for r in roles):
        return "activation", "stream"
    if not roles:
        return "const_table", "dead"
    return "const_table", "table"


def _act_index(inputs: tuple[InputSig, ...]) -> int:
    """Return the activation input's position — ``0`` for legacy sigs."""
    for inp in inputs:
        if inp.kind == "activation":
            return inp.index
    return 0


def _sig_liftable(sig: BlockSig) -> str | None:
    """Give the honest lift verdict — ``None`` when the sig is composable.

    One activation input plus pass-through context is liftable;
    multiple/zero activations, mutated state, and opaque inputs are
    the honest declines.
    """
    if not sig.inputs:
        return None  # legacy hand-built sig — nothing to classify
    if any(
        i.kind == "state" and i.role == "mutated" for i in sig.inputs
    ):
        return "mutates an input"
    n_act = sum(1 for i in sig.inputs if i.kind == "activation")
    if n_act == 0:
        return "no activation input"
    if n_act > 1:
        return "multi-activation inputs"
    if any(i.kind == "opaque" for i in sig.inputs):
        return "opaque input kind"
    return None


@dataclass(frozen=True)
class BlockSig:
    """The structural signature of one block — the morphism object.

    * ``in_projs`` — refs of the projections whose *data* operand reads
      the block input without crossing another projection (the
      input-reading weights).
    * ``out_proj`` — refs of the *terminal* projections: those no
      other projection consumes (length-1 for a single-chain block;
      several for parallel-head blocks — the plan's ``Param|None``
      generalised to the multi-exit case).
    * ``norm`` — the norm signature; the affine ``pre`` form is what
      ``NORM_CASCADE`` folds into ``in_projs``.
    * ``act`` — sorted activation op names present (``"silu"``,
      ``"sdpa"``, …); empty for a purely linear block.
    * ``residual`` — the output spine adds a bare block input
      (``x + f(x)``).
    * ``shape`` — ``(in_shape, out_shape)`` tuples (elements may be
      ``None``); from the captured IO when lifted, else inferred.
      ``in_shape`` is the *activation* input's shape.
    * ``inputs`` — the per-input :class:`InputSig` roles, in
      ``ir.inputs`` order.  Single-input blocks carry exactly one
      ``"activation"`` entry (the historical spelling); a
      multi-input block lifts when exactly one input is the
      activation and the rest classify ``const_table`` /
      read-only ``state`` — rope ``(x, cos, sin)`` blocks and
      cache-carrying ``(x, kv_cache)`` blocks included.
    * ``tables`` — weights of *table* ops (the ``W`` operand of
      ``embedding(W, idx)``): param-only gather sources the block
      reads as a lookup, not a projection.  They are signature terms
      for tying (``WeightTie`` — the classic emb↔head share) but
      never projection endpoints.
    """

    in_projs: tuple[WeightRef, ...]
    out_proj: tuple[WeightRef, ...]
    norm: NormSig
    act: tuple[str, ...]
    residual: bool
    shape: tuple
    inputs: tuple[InputSig, ...] = ()
    tables: tuple[WeightRef, ...] = ()


def _iter_ops(term: Any) -> list[Op]:
    """All ``Op`` nodes in a term, deduplicated (terms are interned)."""
    seen: set[Any] = set()
    out: list[Op] = []
    work = [term]
    while work:
        t = work.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op):
            out.append(t)
            work.extend(t.args)
    return out


def _subtree_nodes(term: Any) -> set:
    """Every node (ops and leaves) in a term's subtree."""
    seen: set[Any] = set()
    work = [term]
    while work:
        t = work.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op):
            work.extend(t.args)
    return seen


def _param_only(term: Any) -> bool:
    """Check the subtree mentions no ``Var`` — a weight-side term."""
    return not has_var_leaf(term)


def _has_param(term: Any) -> bool:
    """Check the subtree contains a ``Param`` leaf (learnable)."""
    return any(isinstance(t, Param) for t in _subtree_nodes(term))


def _projections(root: Any) -> list[tuple[Op, Any, Any]]:
    """Collect ``(node, data, weight)`` projection sites in a term.

    ``linear(d, w[, b])`` takes the weight by position; ``matmul(a, b)``
    takes whichever operand is param-only (both param-only is a weight
    *expression* folded elsewhere; neither is an activation-activation
    product like attention scores — not a projection).
    """
    projs: list[tuple[Op, Any, Any]] = []
    for n in _iter_ops(root):
        if n.op == "linear" and len(n.args) >= 2:
            if _param_only(n.args[1]):
                projs.append((n, n.args[0], n.args[1]))
        elif n.op == "matmul" and len(n.args) == 2:
            a, b = n.args
            pa, pb = _param_only(a), _param_only(b)
            if pa == pb:
                continue
            projs.append((n, b if pa else a, a if pa else b))
    return projs


def _weight_ref(term: Any, memo: dict) -> WeightRef:
    """Build the :class:`WeightRef` for one weight term."""
    if isinstance(term, Param):
        return WeightRef(
            term=term, name=term.name, shape=term.typ.shape
        )
    shp = shape_of(term, memo)
    return WeightRef(
        term=term,
        name="",
        shape=shp if isinstance(shp, tuple) else None,
    )


def _residual_spine(root: Any, inputs: list[Var]) -> bool:
    """Check the output spine adds a bare block input: ``x + f(x)``."""
    ins = set(inputs)
    work = [root]
    seen: set[Any] = set()
    while work:
        t = work.pop()
        if not (isinstance(t, Op) and t.op in ("add", "sub")):
            continue
        for a in t.args:
            if isinstance(a, Var) and a in ins:
                return True
            if (
                isinstance(a, Op)
                and a.op in ("add", "sub")
                and a not in seen
            ):
                seen.add(a)
                work.append(a)
    return False


def _has_rsqrt(term: Any) -> bool:
    """Check the subtree applies ``rsqrt`` to a var-carrying term."""
    return any(
        n.op == "rsqrt" and has_var_leaf(n) for n in _iter_ops(term)
    )


def _rms_entry(n: Op) -> tuple[Op, str, bool] | None:
    """One fused ``rms_norm(x, w)`` node as an ``(n, "rms", affine)`` entry.

    ``None`` when the node has no operands or its subject is var-free
    (compile-time data, not a stream norm); affine iff the weight
    operand (``args[1]``) is present and param-only.
    """
    if not n.args or not has_var_leaf(n.args[0]):
        return None
    return (n, "rms", len(n.args) > 1 and _param_only(n.args[1]))


def _table_refs(root: Any, memo: dict) -> tuple[WeightRef, ...]:
    """Weights of table ops — ``embedding(W, idx)``'s gather source.

    A param-only ``args[0]`` is a stored table the block reads as a
    lookup — a signature term for tying (``WeightTie`` sees the
    emb↔head share through it), never a projection endpoint.
    """
    return tuple(
        _weight_ref(n.args[0], memo)
        for n in _iter_ops(root)
        if n.op == "embedding" and n.args and _param_only(n.args[0])
    )


def _norm_nodes(root: Any) -> list[tuple[Op, str, bool]]:
    """Normalisation/diagonal nodes: ``(node, kind, affine)`` triples.

    ``layer_norm`` ops are affine when they carry a weight arg; a
    ``mul`` whose one side holds the ``rsqrt`` core is an RMS node
    (affine iff the other side is param-only); the fused ``rms_norm``
    op — ``rms_norm(x, w)`` ≡ ``x·rms⁻¹·w`` (llama2.c's spelling) — is
    the same RMS signature with the gain folded into the op's second
    operand (affine iff that operand is param-only); any other ``mul``
    with exactly one var side and a scalar/rank-1 param-only other side
    is a bare diagonal (affine iff the gain contains a ``Param``).
    """
    out: list[tuple[Op, str, bool]] = []
    for n in _iter_ops(root):
        if n.op == "layer_norm":
            affine = len(n.args) > 1 and _param_only(n.args[1])
            out.append((n, "layer_norm", affine))
        elif n.op == "rms_norm":
            # Fused RMS norm — the weight operand (when present and
            # param-only) is the affine gain.
            rms = _rms_entry(n)
            if rms is not None:
                out.append(rms)
        elif n.op == "mul" and len(n.args) >= 2:
            a, b = n.args[0], n.args[1]
            ra, rb = _has_rsqrt(a), _has_rsqrt(b)
            if ra != rb:
                gain = b if ra else a
                out.append((n, "rms", _param_only(gain)))
            else:
                va, vb = has_var_leaf(a), has_var_leaf(b)
                if va == vb:
                    continue
                gain = b if va else a
                gs = shape_of(gain)
                if isinstance(gs, tuple) and len(gs) <= 1:
                    out.append((n, "diag", _has_param(gain)))
    return out


def block_signature(
    ir: IR,
    *,
    in_shape: tuple | None = None,
    out_shape: tuple | None = None,
) -> BlockSig:
    """Extract the :class:`BlockSig` of a block's IR — term level only.

    Works on the exported term, never on ``nn.Module`` objects.  A
    block with no recognised projections simply gets empty tuples —
    the signature is still meaningful (a pure diagonal block reads as
    ``norm.kind == "diag"`` with no projections, which is exactly what
    ``NORM_CASCADE``'s pair form matches).
    """
    root, inputs = ir.root, list(ir.inputs)
    input_sigs = _classify_inputs(ir)
    act_idx = _act_index(input_sigs)
    projs = _projections(root)
    proj_nodes = {n for n, _, _ in projs}
    memo: dict = {}
    # In-projections read the input without crossing another
    # projection; terminal projections feed no other projection's data.
    data_nodes = {n: _subtree_nodes(d) for n, d, _ in projs}
    in_datas = [
        d for n, d, _ in projs if not (data_nodes[n] & proj_nodes)
    ]
    in_refs = tuple(
        _weight_ref(w, memo)
        for n, d, w in projs
        if not (data_nodes[n] & proj_nodes)
    )
    inside: set = set()
    for dn in data_nodes.values():
        inside |= dn & proj_nodes
    out_refs = tuple(
        _weight_ref(w, memo) for n, _, w in projs if n not in inside
    )
    tbl_refs = _table_refs(root, memo)
    norm_nodes = _norm_nodes(root)
    # Pre-norm: the norm node sits on the path into an in-projection.
    in_scope: set = set()
    for d in in_datas:
        in_scope |= _subtree_nodes(d)
    kind_rank = {"layer_norm": 3, "rms": 2, "diag": 1}
    best = ("none", False, False)
    for node, kind, affine in norm_nodes:
        cand = (kind, affine, node in in_scope)
        if (
            kind_rank[kind],
            affine,
            cand[2],
        ) > (kind_rank.get(best[0], 0), best[1], best[2]):
            best = cand
    norm = NormSig(kind=best[0], affine=best[1], pre=best[2])
    acts = tuple(sorted({n.op for n in _iter_ops(root)} & _ACT_OPS))
    i_shp = (
        in_shape
        if in_shape is not None
        else (inputs[act_idx].typ.shape if inputs else None)
    )
    o_shp = out_shape if out_shape is not None else shape_of(root, memo)
    return BlockSig(
        in_projs=in_refs,
        out_proj=out_refs,
        norm=norm,
        act=acts,
        residual=_residual_spine(root, inputs[act_idx : act_idx + 1]),
        shape=(i_shp, o_shp),
        inputs=input_sigs,
        tables=tbl_refs,
    )


def _stem(name: str) -> str:
    """Return a weight name's structural stem — minus index tokens.

    ``p_linears_0_weight`` -> ``p_linears_weight``: digit-only
    underscore-segments are the index positions export assigns, so
    stripping them recovers the structural slot name (the "name prefix"
    of the signature-level tying rule).
    """
    leaf = name.split("__")[-1].rsplit(".", 1)[-1]
    return "_".join(s for s in leaf.split("_") if not s.isdigit())


def weights_tied(a: WeightRef, b: WeightRef) -> bool:
    """Signature-level weight-tying candidate: same shape, same stem.

    Detection is deliberately *structural*: identical shapes plus a
    shared name stem (or identical full names — a parameter object
    shared between two blocks exports under the same leaf name in
    both).  Value equality is the *reify-time* gate
    (:func:`share_duplicate_params` groups by bitwise equality) — a
    match here is a candidate, never a commitment.
    """
    return (
        a.shape is not None
        and a.shape == b.shape
        and bool(a.name)
        and bool(b.name)
        and (a.name == b.name or _stem(a.name) == _stem(b.name))
    )


# ---------------------------------------------------------------------------
#  Shape / probe helpers — shared with the lifted graph
# ---------------------------------------------------------------------------


def _shape_tuple(t: Any) -> tuple | None:
    """Return a tensor's shape as an int tuple — ``None`` if absent."""
    s = getattr(t, "shape", None)
    if s is None:
        return None
    try:
        return tuple(int(d) for d in s)
    except (TypeError, ValueError):
        return None


def _probe_value(a: Any, idx: int) -> Any:
    """Return an index-perturbed clone of a float tensor arg.

    Each arg position gets a distinct affine map — a write like
    ``cache[0] = x[0]`` copies a differently-perturbed value into the
    destination, so the post-run comparison cannot be satisfied
    idempotently.
    """
    if not _is_tensor(a):
        return a
    c = a.clone()
    is_flt = getattr(a, "is_floating_point", None)
    if callable(is_flt) and is_flt():
        return c * (1.5 + 0.25 * idx) + (0.01 + 0.003 * idx)
    return c
