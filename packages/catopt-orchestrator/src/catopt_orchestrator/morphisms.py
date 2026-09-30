"""The morphism engine — block-level optimization over signatures.

Plan 0011, stages 0+1.  Whole-model equality saturation is bounded by
the tensor-level enode cliff; the morphism engine lifts each *block*
to a structural signature and rewrites the tiny block-level graph
(N objects + wires, not N*ops enodes).

``lift_graph`` : model -> :class:`MorphismGraph` — objects are the
composer-selected blocks, arrows are the per-boundary value-flow
verdicts (``chain`` / ``residual`` / ``*_wrapped``), and every node
carries a :class:`BlockSig` read off the block's IR term — the
signature describes *structure* (projections, norms, residuals),
never torch modules.  Blocks that cannot be lifted (export failure, not
executed, non-tensor call signature) stay in the graph as **opaque
boundary nodes**: wires connect through them but no law matches them —
honest partial coverage, same convention as the carrier executors.

A :class:`MorphismLaw` matches on signatures only and returns
:class:`MorphismMatch` rewrites.  Each match carries a
:class:`ReifySpec` — the recipe that maps the morphism-level rewrite
back to concrete IR terms: the joint term is built by *substituting*
block IRs into each other (backend-neutral term composition, no
``nn.Module`` surgery), the term-level laws that already exist
(``assoc_linear`` / ``weight_factor_*`` / ``linear_channel_scale`` /
the exact-tying pass) re-derive the rewrite inside an e-graph, and the
resulting program is cost-gated and **verified** through the sink
before anything is grafted.  Reified steps either replay as witnessed
e-graph merges or ship under the numeric verify gate — a rewrite that
cannot be certified is a decline, never a graft.

Signature coverage, explicitly: signatures recognise ``linear`` /
``matmul`` projections with param-only weights, ``layer_norm`` and
RMS/diagonal-scale norms, known activation ops, and the residual
``x + f(x)`` spine.  Attention internals (``sdpa``), carrier ops, and
everything else flow through untouched — they make the block richer,
not opaque.  Only blocks whose *export itself* fails (or that never
executed on the captured input) are opaque.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from catopt_core import laws
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, Param, Var, op_repr
from catopt_core.laws import tags as _law_tags
from catopt_core.laws.pairing import share_duplicate_params
from catopt_core.pipeline import LowerResult
from catopt_core.ports import Composer, CostFn, Meter, Sink, Source
from catopt_core.typing import _shape_of, has_var_leaf

from catopt_orchestrator.morphisms_kv import KVLatentShare
from catopt_orchestrator.runners import IdentityRunner

log = logging.getLogger("catopt_orchestrator.morphisms")

__all__ = [
    "DEFAULT_MORPHISM_LAWS",
    "BlockSig",
    "KVLatentShare",
    "MorphismGraph",
    "MorphismLaw",
    "MorphismMatch",
    "MorphismNode",
    "MorphismSearch",
    "NormCascade",
    "NormSig",
    "OutInCompose",
    "ReifySpec",
    "ResidualAbsorb",
    "ResidualReassoc",
    "WeightRef",
    "WeightTie",
    "WindowCompose",
    "Wire",
    "block_signature",
    "lift_graph",
    "optimize_morphisms",
    "weights_tied",
]


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
    op, an RMS-style ``x · rms⁻¹ · w`` pattern, or a bare diagonal
    (elementwise gain) map.  ``affine`` records a *learnable* gain
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
    """

    in_projs: tuple[WeightRef, ...]
    out_proj: tuple[WeightRef, ...]
    norm: NormSig
    act: tuple[str, ...]
    residual: bool
    shape: tuple


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
    shp = _shape_of(term, memo)
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


def _norm_nodes(root: Any) -> list[tuple[Op, str, bool]]:
    """Normalisation/diagonal nodes: ``(node, kind, affine)`` triples.

    ``layer_norm`` ops are affine when they carry a weight arg; a
    ``mul`` whose one side holds the ``rsqrt`` core is an RMS node
    (affine iff the other side is param-only); any other ``mul`` with
    exactly one var side and a scalar/rank-1 param-only other side is a
    bare diagonal (affine iff the gain contains a ``Param``).
    """
    out: list[tuple[Op, str, bool]] = []
    for n in _iter_ops(root):
        if n.op == "layer_norm":
            affine = len(n.args) > 1 and _param_only(n.args[1])
            out.append((n, "layer_norm", affine))
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
                gs = _shape_of(gain)
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
        else (inputs[0].typ.shape if inputs else None)
    )
    o_shp = (
        out_shape if out_shape is not None else _shape_of(root, memo)
    )
    return BlockSig(
        in_projs=in_refs,
        out_proj=out_refs,
        norm=norm,
        act=acts,
        residual=_residual_spine(root, inputs),
        shape=(i_shp, o_shp),
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
#  Stage 1 — the lifted graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MorphismNode:
    """One block object in the lifted graph.

    ``opaque`` marks a *boundary node* — the block exported no usable
    signature (export failure, never executed on the probe, non-tensor
    call signature).  Opaque nodes stay wired in place (they are real
    objects in the composition) but no morphism law matches them.
    """

    name: str
    sig: BlockSig | None
    opaque: bool


@dataclass(frozen=True)
class Wire:
    """One arrow between adjacent blocks — the boundary verdict.

    ``kind`` is the composer's joint-mode classification
    (``"chain"`` / ``"chain_wrapped"`` / ``"residual"`` /
    ``"residual_wrapped"``); ``"opaque"`` records a boundary that is
    not a simple value flow (re-entry, fan-out, non-tensor pieces) —
    no morphism law crosses it.
    """

    src: str
    dst: str
    kind: str


@dataclass
class _BlockRecord:
    """The lift record for one node — IR, leaves, captured IO.

    Beyond the exported IR and leaf values, the record keeps the
    captured-IO evidence morphism laws read: ``in_obj`` / ``out_obj``
    are the *live* argument/output objects of the block's first call
    (object identity is the structural proof two blocks shared an
    input or that an output escaped), ``out_val`` its detached output
    clone, ``calls`` the invocation count, and ``example2`` /
    ``out_val2`` the perturbed-probe counterparts.
    """

    name: str
    module: Any
    ir: IR | None = None
    leaves: dict[str, Any] = field(default_factory=dict)
    args: tuple = ()
    example: Any = None
    note: str | None = None
    in_obj: Any = None
    out_obj: Any = None
    out_val: Any = None
    calls: int = 0
    example2: Any = None
    out_val2: Any = None


def _io_evidence(
    rec: _BlockRecord, ient: dict, cap2: Any, ient2: dict
) -> None:
    """Store the captured-IO evidence on the record.

    Live objects (``in_obj``/``out_obj``) carry identity evidence —
    two blocks sharing one input object consume literally the same
    tensor; detached clones (``out_val``/``out_val2``) carry the
    value evidence the additive-consumption checks compare.
    """
    in_objs = ient.get("in_objs")
    rec.in_obj = in_objs[0] if in_objs else None
    rec.out_obj = ient.get("out_obj")
    rec.out_val = ient.get("out")
    rec.calls = int(ient.get("calls", 0))
    if cap2 is not None:
        rec.example2 = cap2[0][0]
    rec.out_val2 = ient2.get("out")


def _aux_outs(io: Any, names: set) -> tuple[tuple, tuple]:
    """Return the non-block io entries' (out values, out objects).

    Entries keyed by something that is not a block name are
    model-level rows — their outputs are the "consumed value"
    targets the family laws check.
    """
    outs: list = []
    objs: list = []
    for k, e in (io or {}).items():
        if k in names or not isinstance(e, dict):
            continue
        if e.get("out") is not None:
            outs.append(e["out"])
        if e.get("out_obj") is not None:
            objs.append(e["out_obj"])
    return tuple(outs), tuple(objs)


class MorphismGraph:
    """The lifted graph: blocks as objects, wires as arrows.

    Built by :func:`lift_graph` from the Composer port's block list
    plus the captured IO (``capture_inputs`` + ``boundary`` — the same
    machinery the compositional strategy drives).  ``nodes`` are in
    execution order; ``wires`` carry the boundary verdict for each
    adjacent pair; :meth:`record` returns the per-block lift record
    the reify recipes read.
    """

    def __init__(
        self,
        nodes: list[MorphismNode],
        wires: list[Wire],
        records: dict[str, _BlockRecord],
        io: Any = None,
        io2: Any = None,
    ) -> None:
        """Store the graph — construction is :func:`lift_graph`'s job.

        ``io`` / ``io2`` are the composer's captured dataflow maps
        (first capture and perturbed probe); entries whose key is not
        a block name are model-level rows — their outputs are the
        "consumed value" targets the family laws check.
        """
        self.nodes = tuple(nodes)
        self.wires = tuple(wires)
        self._records = records
        self._by_name = {n.name: n for n in self.nodes}
        names = set(self._by_name)
        self._model_outs, self._model_out_objs = _aux_outs(io, names)
        self._model_outs2 = _aux_outs(io2, names)[0]
        self._probe2 = bool(io2)

    def node(self, name: str) -> MorphismNode:
        """Return the :class:`MorphismNode` for *name*."""
        return self._by_name[name]

    def sig(self, name: str) -> BlockSig | None:
        """Return the node's signature (``None`` when opaque)."""
        return self._by_name[name].sig

    def record(self, name: str) -> _BlockRecord:
        """Return the block's lift record (IR, leaves, captured args)."""
        return self._records[name]

    def __repr__(self) -> str:
        """Return a compact summary of the lifted graph."""
        return (
            f"MorphismGraph(nodes={len(self.nodes)}, "
            f"wires={len(self.wires)}, "
            f"opaque={sum(1 for n in self.nodes if n.opaque)})"
        )


def _shape_tuple(t: Any) -> tuple | None:
    """Return a tensor's shape as an int tuple — ``None`` if absent."""
    s = getattr(t, "shape", None)
    if s is None:
        return None
    try:
        return tuple(int(d) for d in s)
    except (TypeError, ValueError):
        return None


def lift_graph(
    model: Any,
    x: Any,
    *,
    source: Source,
    composer: Composer,
    block_pred: Any = None,
) -> MorphismGraph:
    """Lift a model to its :class:`MorphismGraph` — signature level.

    Blocks come from the composer's ``blocks`` port (the same
    selection the compositional strategy uses); each block's IR is
    exported through the ``source`` port on its captured input and
    summarised by :func:`block_signature`.  Wires between adjacent
    blocks come from ``composer.boundary`` over the two-capture IO
    evidence (the perturbed second probe included).  Blocks that fail
    export, never executed, or take a non-tensor call signature are
    kept as opaque boundary nodes.
    """
    blocks = composer.blocks(model, predicate=block_pred)
    captured, io = composer.capture_inputs(model, blocks, x)
    try:
        captured2, io2 = composer.capture_inputs(
            model, blocks, composer.perturbed(x)
        )
    except Exception:
        captured2, io2 = {}, {}

    nodes: list[MorphismNode] = []
    records: dict[str, _BlockRecord] = {}
    for name, mod in blocks:
        rec = _BlockRecord(name=name, module=mod)
        records[name] = rec
        cap = captured.get(name)
        if cap is None:
            rec.note = "not executed on the captured input"
        elif len(cap[0]) != 1 or cap[1]:
            rec.note = "not a single-positional-arg call"
        else:
            rec.args = cap[0]
            rec.example = cap[0][0]
            _io_evidence(
                rec,
                io.get(name, {}),
                captured2.get(name),
                io2.get(name, {}),
            )
            try:
                rec.ir, rec.leaves = source.to_ir(mod, rec.example)
            except Exception as e:
                rec.note = f"export failed: {type(e).__name__}: {e}"
        sig = (
            block_signature(
                rec.ir,
                in_shape=_shape_tuple(rec.example),
                out_shape=_shape_tuple(io.get(name, {}).get("out")),
            )
            if rec.ir is not None
            else None
        )
        nodes.append(
            MorphismNode(name=name, sig=sig, opaque=sig is None)
        )

    wires = [
        Wire(
            blocks[i][0],
            blocks[i + 1][0],
            composer.boundary(
                blocks[i][0],
                blocks[i + 1][0],
                captured,
                io,
                captured2,
                io2,
            )
            or "opaque",
        )
        for i in range(len(blocks) - 1)
    ]
    return MorphismGraph(nodes, wires, records, io=io, io2=io2)


# ---------------------------------------------------------------------------
#  Stage 1 — morphism laws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReifySpec:
    """The recipe mapping a morphism rewrite back to concrete terms.

    * ``mode`` — the boundary mode the joint term is built in
      (``"chain"`` / ``"residual"`` / their ``_wrapped`` forms), or
      ``"intra"`` (single-block rewrite on its own term) / ``"tie"``
      (exact weight sharing across the nodes' joint e-graph).
    * ``rules`` — the named term-level recipe saturated on the joint
      e-graph: ``"compose"`` (the bilinearity/associativity/factor
      family that folds ``A.out ∘ B.in``), ``"scale"`` (the diagonal
      naturality family folding norm gains into weights), ``"tie"``
      (no saturation — the exact-tying pass).
    * ``distribute`` — first offer the bilinear expansion of the
      residual sum into each in-projection
      (``linear(x + A(x), W) = linear(x,W) + linear(A(x),W)``) as a
      witnessed member — the data-slot distributivity step the rule
      set has no ``linear``-spelling law for, asserted at morphism
      level and gated by the pair verify.
    * ``share`` — run the value-exact weight-tying pass on the joint
      e-graph before extraction.
    * ``kinds`` — for *window* rewrites (``len(nodes) >= 3``), the
      per-wire boundary kinds in arrow order (``len(nodes) - 1``
      entries).  ``mode`` then mirrors ``kinds[-1]``: it still drives
      the first-slot delta and last-slot filler conventions.  Empty
      for pair / intra / tie matches.
    * ``extra`` — an opaque law-carried payload the mode's reify
      reads back: the ``"family"`` mode (KV latent sharing,
      :mod:`catopt_orchestrator.morphisms_kv`) packs the name-token
      sets, the factor tolerance and the shared data term
      identifying the latent group.
    """

    mode: str
    rules: str = "compose"
    distribute: bool = False
    share: bool = False
    kinds: tuple[str, ...] = ()
    extra: Any = None


@dataclass(frozen=True)
class MorphismMatch:
    """One law firing on the lifted graph.

    ``nodes`` are the block names in arrow order; ``boundary`` is the
    wire kind (or ``"intra"`` / ``"tie"``); ``reify`` is the
    :class:`ReifySpec` the engine executes; ``detail`` is the human
    record of *why* the law saw this match.
    """

    law: str
    nodes: tuple[str, ...]
    boundary: str
    reify: ReifySpec
    detail: str = ""


@runtime_checkable
class MorphismLaw(Protocol):
    """A rewrite law on the morphism graph — signatures only.

    ``match`` reads :class:`MorphismGraph` signatures/wires and returns
    :class:`MorphismMatch` rewrites; it never touches tensor-level
    terms (those arrive at *reify* time, through the spec the match
    carries).  Conformance is duck-typed: a ``name`` plus a
    ``match(graph) -> list`` method.
    """

    name: str

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Return the law's firings on *graph*."""
        ...


def _dims_compatible(a: BlockSig, b: BlockSig) -> bool:
    """Check A's output feeds B's input (last dims agree).

    Unknown shapes pass — the match is a candidate; the verify gate at
    reify time is authoritative.
    """
    sa, sb = a.shape[1], b.shape[0]
    if (
        isinstance(sa, tuple)
        and isinstance(sb, tuple)
        and sa
        and sb
        and sa[-1] is not None
        and sb[-1] is not None
    ):
        return sa[-1] == sb[-1]
    return True


def _is_pure_diagonal(sig: BlockSig) -> bool:
    """Check the whole block is one diagonal map (``x ∘ s``)."""
    return (
        sig.norm.kind == "diag"
        and not sig.in_projs
        and not sig.out_proj
        and not sig.residual
    )


def _pair_matches(
    graph: MorphismGraph,
    law: str,
    kinds: frozenset,
    pred: Any,
    spec: Any,
    detail: str,
) -> list[MorphismMatch]:
    """Emit a match for each wire whose boundary and sigs qualify."""
    out = []
    for w in graph.wires:
        if w.kind not in kinds:
            continue
        a, b = graph.sig(w.src), graph.sig(w.dst)
        if a is None or b is None or not pred(a, b):
            continue
        out.append(
            MorphismMatch(
                law=law,
                nodes=(w.src, w.dst),
                boundary=w.kind,
                reify=spec(w.kind),
                detail=detail.format(a=w.src, b=w.dst),
            )
        )
    return out


class OutInCompose:
    """``A.out_proj ∘ B.in_proj`` — compose projections across a chain.

    The signature-level form of the cross-pair weight fold: A's
    terminal projection composes with each of B's input projections.
    Matches chain boundaries (plain or wrapped) between lifted blocks
    that both carry projections; reify builds the joint term
    ``B(A(x))`` and saturates the compose recipe — the term-level
    ``assoc_linear`` / ``weight_factor_*`` laws fold the chain into a
    single weight.
    """

    name = "out_in_compose"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match chain wires with projections on both sides."""
        return _pair_matches(
            graph,
            self.name,
            frozenset({"chain", "chain_wrapped"}),
            lambda a, b: (
                bool(a.out_proj)
                and bool(b.in_projs)
                and _dims_compatible(a, b)
            ),
            lambda kind: ReifySpec(mode=kind, rules="compose"),
            "{a}.out_proj ∘ {b}.in_projs",
        )


class ResidualAbsorb:
    """``x + A(x) → B`` — the residual add is absorbable into B's projs.

    Matches residual boundaries where B has input projections: the
    reified program distributes each ``linear(x + A(x), W)`` over the
    add (bilinearity — the witnessed morphism step), then the compose
    recipe folds the ``A(x)`` side's ``A.out ∘ W`` composition.  What
    remains is ``linear(x, W) + linear(A_inner, W @ A_out)`` — B
    absorbing A's output projection into its input weights.
    """

    name = "residual_absorb"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match residual wires into blocks with input projections."""
        return _pair_matches(
            graph,
            self.name,
            frozenset({"residual", "residual_wrapped"}),
            lambda a, b: bool(b.in_projs),
            lambda kind: ReifySpec(
                mode=kind, rules="compose", distribute=True
            ),
            "{b} absorbs the residual add over {a}",
        )


class NormCascade:
    """Norm diagonals cascade through block chains into next weights.

    Two forms:

    * *pair* — a pure diagonal block (``x ∘ s`` — a standalone gain or
      scale) on a chain boundary: its diagonal commutes into B's input
      projections (``linear(x ∘ s, W) = linear(x, W ∘ s)``).
    * *node* — an affine *pre*-norm inside one block: the gain folds
      into the block's own in-projection weights (the RMSNorm→Linear
      fold the signature already sees).
    """

    name = "norm_cascade"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match diagonal-feeding wires and affine pre-norm nodes."""
        out = _pair_matches(
            graph,
            self.name,
            frozenset({"chain", "chain_wrapped"}),
            lambda a, b: _is_pure_diagonal(a) and bool(b.in_projs),
            lambda kind: ReifySpec(mode=kind, rules="scale"),
            "{a} diagonal cascades into {b}.in_projs",
        )
        for n in graph.nodes:
            sig = n.sig
            if sig is None:
                continue
            if sig.norm.affine and sig.norm.pre and sig.in_projs:
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=(n.name,),
                        boundary="intra",
                        reify=ReifySpec(mode="intra", rules="scale"),
                        detail=(
                            f"{n.name} pre-norm gain folds into "
                            "in_projs"
                        ),
                    )
                )
        return out


class WeightTie:
    """Weight tying: identical param shapes + name stem → share.

    Candidate generation is signature-level: two nodes (possibly the
    same node — duplicated branch weights inside one block) carrying
    weights with equal shapes, plus an equal name stem for the
    cross-block case (a shared ``nn.Parameter`` exports under the same
    leaf name in both blocks).  Reify interns the involved blocks'
    terms into one e-graph and runs :func:`share_duplicate_params` —
    the value-exact pass decides whether the candidate tie is real; a
    shape+name coincidence that does not share *values* declines.
    """

    name = "weight_tie"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match intra-block shape dups and cross-block stem+shape ties."""
        out = []
        lifted = [
            (n.name, s) for n in graph.nodes if (s := n.sig) is not None
        ]
        for name, sig in lifted:
            refs = tuple(dict.fromkeys(sig.in_projs + sig.out_proj))
            if len(refs) > 1 and any(
                x.shape is not None and x.shape == y.shape
                for i, x in enumerate(refs)
                for y in refs[i + 1 :]
            ):
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=(name,),
                        boundary="tie",
                        reify=ReifySpec(
                            mode="intra", rules="tie", share=True
                        ),
                        detail=f"{name}: same-shape weights",
                    )
                )
        for i, (name_a, sa) in enumerate(lifted):
            for name_b, sb in lifted[i + 1 :]:
                refs_a = sa.in_projs + sa.out_proj
                refs_b = sb.in_projs + sb.out_proj
                if any(
                    weights_tied(wa, wb)
                    for wa in refs_a
                    for wb in refs_b
                ):
                    out.append(
                        MorphismMatch(
                            law=self.name,
                            nodes=(name_a, name_b),
                            boundary="tie",
                            reify=ReifySpec(
                                mode="tie", rules="tie", share=True
                            ),
                            detail=(f"{name_a}~{name_b}: tied weights"),
                        )
                    )
        return out


def _compose_pair_ok(a: BlockSig | None, b: BlockSig | None) -> bool:
    """Check the out∘in signature predicate on one adjacent pair."""
    return (
        a is not None
        and b is not None
        and bool(a.out_proj)
        and bool(b.in_projs)
        and _dims_compatible(a, b)
    )


def _wire_composes(
    graph: MorphismGraph, w: Wire, kinds: frozenset[str]
) -> bool:
    """Check a wire's boundary kind and both sides' signatures."""
    return w.kind in kinds and _compose_pair_ok(
        graph.sig(w.src), graph.sig(w.dst)
    )


def _window_nodes(wires: tuple[Wire, ...], i: int, j: int) -> tuple:
    """Block names for wires ``i..j`` inclusive — arrow order."""
    return (
        *(wires[t].src for t in range(i, j + 1)),
        wires[j].dst,
    )


def _stream_commutes(
    graph: MorphismGraph, nodes: tuple[str, ...]
) -> bool:
    """Check a residual window carries a real commute opportunity.

    Every node must be lifted (an opaque block is a boundary, never
    crossed), and some earlier block's out-projection must compose
    into a *later* block's input projections.  The receiving block's
    projections must read the stream with no pre-norm in the way —
    ``norm.pre`` marks a nonlinear normaliser bilinearity cannot
    cross.
    """
    sigs = [graph.sig(n) for n in nodes]
    if any(s is None for s in sigs):
        return False
    lifted = [s for s in sigs if s is not None]
    return any(
        _stream_pair_ok(si, sj)
        for i, si in enumerate(lifted[:-1])
        for sj in lifted[i + 1 :]
    )


def _stream_pair_ok(a: BlockSig, b: BlockSig) -> bool:
    """One contributing pair: out-proj into a pre-norm-free receiver."""
    return (
        bool(a.out_proj)
        and bool(b.in_projs)
        and not b.norm.pre
        and _dims_compatible(a, b)
    )


class WindowCompose:
    """``A.out ∘ B.in ∘ C.in ∘ …`` — compose a whole chain window.

    Generalises :class:`OutInCompose` from a boundary pair to a
    maximal run of ≥3 blocks: interior wires must be plain ``chain``
    (each block's output is consumed by exactly the next input), and
    the final wire may additionally be ``chain_wrapped`` (the
    parent's ``y + B(y)`` wrap around the last block).  One
    :class:`MorphismMatch` per window carries one :class:`ReifySpec`
    — a single joint term, a single joint e-graph, a single verify —
    instead of k-1 pairwise passes that would consume the blocks two
    at a time.
    """

    name = "window_compose"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Grow maximal composable chain windows left-to-right."""
        out: list[MorphismMatch] = []
        wires = graph.wires
        i = 0
        while i < len(wires):
            if not _wire_composes(
                graph, wires[i], frozenset({"chain"})
            ):
                i += 1
                continue
            j = i
            while j + 1 < len(wires) and _wire_composes(
                graph, wires[j + 1], frozenset({"chain"})
            ):
                j += 1
            kinds = [wires[t].kind for t in range(i, j + 1)]
            # A wrapped tail may close the window (last block only).
            if j + 1 < len(wires) and _wire_composes(
                graph, wires[j + 1], frozenset({"chain_wrapped"})
            ):
                j += 1
                kinds.append("chain_wrapped")
            if j - i >= 1:  # >=2 wires -> >=3 blocks
                nodes = _window_nodes(wires, i, j)
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=nodes,
                        boundary="+".join(kinds),
                        reify=ReifySpec(
                            mode=kinds[-1],
                            rules="compose",
                            kinds=tuple(kinds),
                        ),
                        detail=(
                            f"{nodes[0]}->…->{nodes[-1]}: "
                            f"{len(nodes)}-block out∘in window"
                        ),
                    )
                )
            i = j + 1
        return out


class ResidualReassoc:
    """The residual ``+`` monoid commutes receivers past blocks.

    On a residual-stream run ``s = x + f0(x) + f1(·) + …`` the stream
    every block reads is a *sum* of all earlier contributions, so a
    later block's input projections may legally distribute over
    addends a non-adjacent block produced:
    ``linear(s, W) = linear(x, W) + Σ linear(f_i(·), W)`` — and each
    ``linear(f_i(·), W)`` is the ``f_i.out ∘ W`` composition the pair
    laws reach only for adjacent blocks.  The additive monoid
    (associativity + commutativity of ``+``) is the legal commute
    path; bilinearity does the absorption.

    Matches maximal windows of ≥3 blocks whose interior wires are
    ``residual_wrapped`` (the stream flows on), optionally closed by
    a plain ``residual`` receiver; emits one match whose reify offers
    the stream-distributed joint — a constructed equality asserted at
    morphism level and gated by the fp64 verify + the cost gate.
    """

    name = "residual_reassoc"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Grow maximal residual-stream windows left-to-right."""
        out: list[MorphismMatch] = []
        wires = graph.wires
        i = 0
        while i < len(wires):
            if wires[i].kind != "residual_wrapped":
                i += 1
                continue
            j = i
            while (
                j + 1 < len(wires)
                and wires[j + 1].kind == "residual_wrapped"
            ):
                j += 1
            # An unwrapped receiver may close the window.
            if j + 1 < len(wires) and wires[j + 1].kind == "residual":
                j += 1
            if j - i >= 1:  # >=2 wires -> >=3 blocks on the stream
                nodes = _window_nodes(wires, i, j)
                if _stream_commutes(graph, nodes):
                    kinds = [wires[t].kind for t in range(i, j + 1)]
                    out.append(
                        MorphismMatch(
                            law=self.name,
                            nodes=nodes,
                            boundary="+".join(kinds),
                            reify=ReifySpec(
                                mode=kinds[-1],
                                rules="compose",
                                distribute=True,
                                kinds=tuple(kinds),
                            ),
                            detail=(
                                f"{nodes[0]}->…->{nodes[-1]}: residual"
                                " stream reassociation over "
                                f"{len(nodes)} blocks"
                            ),
                        )
                    )
            i = j + 1
        return out


#: The default morphism law family, in application order — widest
#: spans first (a window claims its nodes before any pair law sees
#: them; a declined window leaves its pairs to the pair laws), then
#: pair-level transforms, then node-level and tying.
DEFAULT_MORPHISM_LAWS: tuple[MorphismLaw, ...] = (
    WindowCompose(),
    ResidualReassoc(),
    OutInCompose(),
    ResidualAbsorb(),
    NormCascade(),
    WeightTie(),
)


# ---------------------------------------------------------------------------
#  Reify — morphism rewrite -> concrete, verified IR
# ---------------------------------------------------------------------------


def _ns_prefix(name: str) -> str:
    """Namespace prefix for one block's params inside a joint term."""
    return name.replace(".", "_") + "__"


def _prefix_params(term: Any, pre: str) -> Any:
    """Rename every ``Param`` leaf in *term* with the block prefix."""
    if isinstance(term, Param):
        return Param(pre + term.name, term.typ)
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_prefix_params(a, pre) for a in term.args),
            **term.attrs,
        )
    return term


def _subst(term: Any, var: Var, repl: Any) -> Any:
    """Substitute *repl* for every occurrence of *var* in *term*."""
    if term == var:
        return repl
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_subst(a, var, repl) for a in term.args),
            **term.attrs,
        )
    return term


def _joint_parts(
    ira: IR,
    irb: IR,
    rec_a: _BlockRecord,
    rec_b: _BlockRecord,
    mode: str,
) -> tuple[Any, Var, Any, dict, dict]:
    """Compose the pair's IRs per the boundary mode — terms, not modules.

    Returns ``(joint, x, mid, params, leaves)``: the joint term over
    the shared input ``x`` (A's input variable), ``mid`` — the value B
    reads (``A(x)`` for chains, ``x + A(x)`` for residuals) — and the
    namespaced param/value tables (per-block ``p_*`` leaf names never
    collide).  The ``_wrapped`` modes add the outer wrap the parent's
    ``y + ·`` performs.
    """
    pa, pb = _ns_prefix(rec_a.name), _ns_prefix(rec_b.name)
    x, vb = ira.inputs[0], irb.inputs[0]
    y_a = _prefix_params(ira.root, pa)
    mid = Op.make("add", x, y_a) if mode.startswith("residual") else y_a
    body = _subst(_prefix_params(irb.root, pb), vb, mid)
    joint = (
        Op.make("add", mid, body) if mode.endswith("_wrapped") else body
    )
    params = {
        **{pa + k: Param(pa + k, p.typ) for k, p in ira.params.items()},
        **{pb + k: Param(pb + k, p.typ) for k, p in irb.params.items()},
    }
    leaves = {
        **{pa + k: v for k, v in rec_a.leaves.items()},
        **{pb + k: v for k, v in rec_b.leaves.items()},
    }
    return joint, x, mid, params, leaves


def _joint_parts_window(
    irs: list[IR],
    recs: list[_BlockRecord],
    kinds: tuple[str, ...],
) -> tuple[Any, Var, tuple, dict, dict]:
    """Compose a ≥3-block window's IRs — the n-ary :func:`_joint_parts`.

    Returns ``(joint, x, mids, params, leaves)``: ``joint`` is the
    whole window's function of the first block's input ``x``;
    ``mids`` is the tuple of residual-*stream* nodes the later blocks
    read (in creation order ``s_0..s_{k-2}`` — the distribute offer
    expands them outermost-first); empty for the chain family, which
    has no additive structure to distribute over.

    Construction is driven by ``kinds`` (one per interior boundary):
    a residual-family window accumulates the stream
    ``s_j = s_{j-1} + f_j(s_{j-1})`` every later block reads; a
    chain-family window nests ``f_j(f_{j-1}(·))``.  In both, the last
    wire's ``_wrapped`` mark decides whether the segment's value is
    the raw last body or ``in + body``.
    """
    pres = [_ns_prefix(r.name) for r in recs]
    roots = [
        _prefix_params(ir.root, pre)
        for ir, pre in zip(irs, pres, strict=True)
    ]
    params: dict[str, Param] = {}
    leaves: dict[str, Any] = {}
    for ir, rec, pre in zip(irs, recs, pres, strict=True):
        params.update(
            {
                pre + k: Param(pre + k, p.typ)
                for k, p in ir.params.items()
            }
        )
        leaves.update({pre + k: v for k, v in rec.leaves.items()})
    x = irs[0].inputs[0]
    first = _subst(roots[0], irs[0].inputs[0], x)
    mids: list[Any] = []
    if kinds[0].startswith("residual"):
        # Stream semantics: block j >= 1 reads s_{j-1}, the running
        # in+out sum; interior wires are residual_wrapped by the law's
        # own grammar (the stream must flow on).
        stream = Op.make("add", x, first)
        mids.append(stream)
        for j in range(1, len(irs) - 1):
            fj = _subst(roots[j], irs[j].inputs[0], stream)
            stream = Op.make("add", stream, fj)
            mids.append(stream)
        body = _subst(roots[-1], irs[-1].inputs[0], stream)
        joint = (
            Op.make("add", stream, body)
            if kinds[-1].endswith("_wrapped")
            else body
        )
    else:
        cur = first
        for j in range(1, len(irs) - 1):
            cur = _subst(roots[j], irs[j].inputs[0], cur)
        body = _subst(roots[-1], irs[-1].inputs[0], cur)
        joint = (
            Op.make("add", cur, body)
            if kinds[-1].endswith("_wrapped")
            else body
        )
    return joint, x, tuple(mids), params, leaves


def _distribute_over(term: Any, mid: Any) -> Any:
    """Distribute projections whose data operand IS the residual sum.

    ``linear(x + A(x), W[, b]) → linear(x, W) + linear(A(x), W[, b])``
    — bilinearity of the data slot (the bias rides one branch — exact).
    ``matmul`` gets the same treatment on whichever side carries the
    sum.  A *constructed* equality: the rule set has no ``linear``-slot
    distribute law, so this step is offered into the joint e-graph
    under a pointwise witness and gated by the pair's fp64 verify.
    ``mid`` is always an additive node by construction — the joint
    builders only ever pass residual-stream ``add`` nodes.
    """
    if not isinstance(term, Op):
        return term
    args = tuple(_distribute_over(a, mid) for a in term.args)
    if term.op == "linear" and args and args[0] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("linear", lhs, args[1]),
            Op.make("linear", rhs, *args[1:]),
        )
    if term.op == "matmul" and len(args) >= 2 and args[0] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("matmul", lhs, args[1]),
            Op.make("matmul", rhs, args[1]),
        )
    if term.op == "matmul" and len(args) >= 2 and args[1] == mid:
        lhs, rhs = mid.args
        return Op.make(
            "add",
            Op.make("matmul", args[0], lhs),
            Op.make("matmul", args[0], rhs),
        )
    return Op.make(term.op, *args, **term.attrs)


#: Term-level rule recipes the reify specs name.  Both are subsets of
#: the core ``FULL`` preset — the bilinearity / associativity / factor
#: family that composes projections (``compose``), and the diagonal
#: naturality family that folds scale gains into weights (``scale``).
_RECIPE_NAMES: dict[str, tuple[str, ...]] = {
    "compose": (
        "weight_factor_linear",
        "weight_distribute_linear",
        "right_factor_linear",
        "assoc_linear",
        "assoc_linear_bias",
        "assoc_linear_bias_rev",
        "weight_factor_matmul",
        "weight_distribute_matmul",
        "right_factor_matmul",
        "assoc_matmul",
        "assoc_matmul_rev",
        "linear_channel_scale",
        "linear_channel_scale_rev",
        "linear_row_scale",
        "linear_row_scale_rev",
        "naturality_scalar",
        "naturality_scalar_rev",
        "distribute_matmul_over_add",
        "factor_matmul",
        "right_distribute_matmul",
        "comm_add",
        "assoc_add",
        "comm_mul",
        "id_add",
        "id_mul",
        "sub_to_add",
        "double_neg",
    ),
    "scale": (
        "linear_channel_scale",
        "linear_channel_scale_rev",
        "linear_row_scale",
        "linear_row_scale_rev",
        "naturality_scalar",
        "naturality_scalar_rev",
        "weight_factor_linear",
        "assoc_linear",
        "comm_mul",
        "assoc_mul",
        "id_mul",
    ),
    "tie": (),
}

_RECIPE_RULESETS: dict[str, Any] = {}


def _recipe_rules(name: str) -> Any:
    """Materialise the named term-level recipe as a ``RuleSet``."""
    rs = _RECIPE_RULESETS.get(name)
    if rs is None:
        rs = laws.FULL.named(*_RECIPE_NAMES[name])
        _RECIPE_RULESETS[name] = rs
    return rs


def _witness_offer(
    eg: EGraph, eid: int, term: Any, offered: tuple
) -> None:
    """Merge one constructed equality into the joint's e-class.

    ``offered`` is ``(term, law_text)`` plus an optional kwargs dict
    (``note`` / ``error_bound`` / ``bound_norm`` — the KV-latent
    offer certifies its factorisation residual); identity offers are
    skipped.
    """
    term_o, law_text = offered[0], offered[1]
    if term_o is term:
        return
    kw = dict(offered[2]) if len(offered) > 2 else {}
    kw.setdefault(
        "note",
        "residual_absorb: data-slot bilinearity over the residual add",
    )
    eg._offer_witness(
        eid,
        rhs_term=term_o,
        lhs_term=term,
        provenance="morphism_reify",
        law=law_text,
        **kw,
    )


def _saturate(
    term: Any,
    rules: Any,
    *,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
    offers: list | None = None,
) -> tuple[EGraph, int]:
    """One joint e-graph: intern, offer constructed members, saturate.

    ``offers`` are ``(term, law_text)`` constructed equalities (the
    residual-distribute step) merged into the joint's e-class under a
    pointwise witness — the documented non-local-pass ritual (the
    pairing/tying passes use ``EGraph._offer_witness`` the same way);
    they replay in certificates like every other witnessed merge and
    the pair verify gates the assertion.  A third element, when
    present, is a kwargs dict for the witness (``note`` /
    ``error_bound`` / ``bound_norm`` — the KV-latent offer certifies
    its factorisation residual).  Returns ``(eg, root_eid)``.
    """
    eg = EGraph()
    eid = eg.add_term(term)
    for offered in offers or ():
        _witness_offer(eg, eid, term, offered)
    if len(rules):
        budgets = (
            {
                r.name: symmetry_budget
                for r in rules.tagged(_law_tags.EXPANSIVE)
            }
            if symmetry_budget is not None
            else None
        )
        eg.run(
            rules,
            eid,
            max_iterations=max_iterations,
            max_nodes=max_enodes,
            rule_budgets=budgets,
        )
    return eg, eid


def _lower_term(
    term: Any, var: Var, params: dict, leaves: dict, sink: Sink
) -> Any:
    """Lower one term over one input var through the sink."""
    ir = IR(
        root=term,
        inputs=[var],
        input_names={var.name},
        params=params,
    )
    return sink.lower(ir, leaves)


def _verify_pair(
    sink: Sink,
    ref_term: Any,
    opt_term: Any,
    var: Var,
    params: dict,
    leaves: dict,
    args: tuple,
    rtol: float,
) -> Any:
    """Verify the reified term against the un-rewritten joint."""
    ref = _lower_term(ref_term, var, params, leaves, sink)
    opt = _lower_term(opt_term, var, params, leaves, sink)
    return sink.verify(ref, opt, args, rtol=rtol)


def _slot_filler(sink: Sink, var: Var, mode: str) -> Any:
    """Build the B-slot filler for a consumed pair: id or exact zero.

    Both are lowered terms — backend-neutral: ``x`` evaluates to its
    input; ``x * 0`` evaluates to a zero of the input's shape (the
    composer's ``_Zero`` uses ``zeros_like`` — equivalent on finite
    inputs, which is what the verify gate runs).
    """
    root = (
        Op.make("mul", var, Const(0))
        if mode.endswith("_wrapped")
        else var
    )
    ir = IR(root=root, inputs=[var], input_names={var.name}, params={})
    return sink.lower(ir, {})


def _intra_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve an intra (single-node) match — its own IR as joint."""
    rec = graph.record(match.nodes[0])
    if rec.ir is None:
        return None, "opaque node"
    return (
        (
            rec.ir.root,
            rec.ir.inputs[0],
            (),
            rec.ir.params,
            rec.leaves,
            rec.args,
        ),
        None,
    )


def _window_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a ≥3-block window match — the n-ary joint term."""
    spec = match.reify
    recs = [graph.record(n) for n in match.nodes]
    if len(spec.kinds) != len(recs) - 1 or len(recs) < 2:
        return None, "malformed window spec"
    if any(r.ir is None for r in recs):
        return None, "opaque node"
    joint, var, mids, params, leaves = _joint_parts_window(
        [r.ir for r in recs if r.ir is not None],
        recs,
        spec.kinds,
    )
    return (
        (
            joint,
            var,
            mids if spec.distribute else (),
            params,
            leaves,
            recs[0].args,
        ),
        None,
    )


def _pair_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a boundary-pair match — the two-block joint term."""
    spec = match.reify
    rec_a, rec_b = (graph.record(n) for n in match.nodes)
    if rec_a.ir is None or rec_b.ir is None:
        return None, "opaque node"
    term, var, mid, params, leaves = _joint_parts(
        rec_a.ir, rec_b.ir, rec_a, rec_b, spec.mode
    )
    return (
        (
            term,
            var,
            (mid,) if spec.distribute else (),
            params,
            leaves,
            rec_a.args,
        ),
        None,
    )


def _resolve_joint(
    match: MorphismMatch, graph: MorphismGraph
) -> tuple[tuple | None, str | None]:
    """Resolve a match to its joint term plus lowering context.

    Returns ``((term, var, mids, params, leaves, args), None)`` —
    the joint program over the first block's input variable, the
    additive nodes to distribute over (already gated by
    ``spec.distribute``, so empty unless the spec asks), the
    namespaced param/leaf tables, and the first block's captured
    args — or ``(None, reason)`` for an honest decline.
    """
    spec = match.reify
    if spec.mode == "intra":
        return _intra_joint(match, graph)
    if spec.kinds:
        return _window_joint(match, graph)
    return _pair_joint(match, graph)


def _distribute_offers(term: Any, mids: tuple) -> list:
    """Build the constructed bilinear steps, progressive over each mid.

    Expands every projection whose data operand IS one of the
    additive mid nodes — outermost first, so a window's nested
    stream fully unfolds.  Each progressively-distributed form is a
    separate exact-equality offer merged at the joint's e-class;
    the pair verify gates the whole assertion once.
    """
    dist = term
    offers = []
    for m_node in reversed(mids):
        nxt = _distribute_over(dist, m_node)
        if nxt is not dist:
            offers.append(
                (
                    nxt,
                    "bilinearity of the projection's data slot "
                    "over the residual add — morphism-level "
                    "assertion, gated by the pair verify",
                )
            )
            dist = nxt
    return offers


def _window_reps(
    best: Any,
    match: MorphismMatch,
    var: Var,
    params: dict,
    leaves: dict,
    sink: Sink,
) -> dict[str, Any]:
    """Per-slot replacements for a grafted ≥3-block window.

    The fused term lands at the first slot — a ``best - x`` delta
    when the family wraps the first block's output in a residual
    add.  Each later slot's filler follows the wire INTO it: a
    wrapped consumption takes the exact-zero addend, a plain one
    the identity passthrough — same convention as the pair slots.
    """
    spec = match.reify
    a_term = (
        Op.make("sub", best, var)
        if spec.mode.startswith("residual")
        else best
    )
    reps = {
        match.nodes[0]: _lower_term(a_term, var, params, leaves, sink)
    }
    for j, name in enumerate(match.nodes[1:], start=1):
        reps[name] = _slot_filler(sink, var, spec.kinds[j - 1])
    return reps


def _reify(
    match: MorphismMatch,
    graph: MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
) -> dict[str, Any]:
    """Execute one morphism rewrite; return the graft record or a decline.

    Term-level reification: build the joint IR per the boundary mode,
    run the recipe (distribute offer → rule saturation → extraction),
    cost-gate against the un-rewritten joint, verify through the sink,
    and produce the per-slot replacement modules.
    """
    spec = match.reify
    if spec.mode == "tie" or spec.share:
        return _reify_tie(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
        )
    if spec.mode == "family":
        # The KV-latent rewrite lives in the sibling module (its
        # factorisation machinery is substantial); resolved lazily
        # to keep the modules acyclic.
        from catopt_orchestrator.morphisms_kv import _reify_family

        return _reify_family(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            symmetry_budget=symmetry_budget,
        )
    resolved, decline = _resolve_joint(match, graph)
    if resolved is None:
        return {"status": "declined", "reason": decline}
    term, var, mids, params, leaves, args = resolved

    offers = _distribute_offers(term, mids)
    eg, eid = _saturate(
        term,
        _recipe_rules(spec.rules),
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        symmetry_budget=symmetry_budget,
        offers=offers,
    )
    best = eg.extract_best(eid, cost_fn)
    base_cost = dag_cost(term, cost_fn)
    best_cost = dag_cost(best, cost_fn)
    info: dict[str, Any] = {
        "cost_before": base_cost,
        "cost_after": best_cost,
        "joint": op_repr(term),
        "reified": op_repr(best),
    }
    if not best_cost < base_cost:
        return {
            "status": "declined",
            "reason": "no_improvement",
            **info,
        }
    vr = _verify_pair(
        sink, term, best, var, params, leaves, args, verify_tol
    )
    info["rel_diff"] = vr.max_rel
    if not vr.passed:
        return {
            "status": "declined",
            "reason": f"reify verify failed: {vr.max_rel:.3e}",
            **info,
        }
    if spec.mode == "intra":
        reps = {
            match.nodes[0]: _lower_term(best, var, params, leaves, sink)
        }
    elif spec.kinds:
        reps = _window_reps(best, match, var, params, leaves, sink)
    else:
        a_name, b_name = match.nodes
        a_term = (
            Op.make("sub", best, var)
            if spec.mode.startswith("residual")
            else best
        )
        reps = {
            a_name: _lower_term(a_term, var, params, leaves, sink),
            b_name: _slot_filler(sink, var, spec.mode),
        }
    return {"status": "grafted", "reps": reps, **info}


def _reify_tie(
    match: MorphismMatch,
    graph: MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
) -> dict[str, Any]:
    """Reify a weight-tying match: joint e-graph, exact share, verify.

    Interns the involved blocks' (namespaced) terms into one e-graph
    and runs the value-exact :func:`share_duplicate_params` pass.  The
    candidate tie the signature detected is real only when the pass
    merges names — a shape+stem coincidence that does not share
    *values* declines before lowering.  Each surviving block re-lowers
    on its extracted term and verifies against its original module on
    the captured input.
    """
    eg = EGraph()
    recs = [graph.record(n) for n in match.nodes]
    leaves: dict[str, Any] = {}
    params: dict[str, Param] = {}
    vars_: dict[str, Var] = {}
    eids: dict[str, int] = {}
    for r in recs:
        if r.ir is None:
            return {"status": "declined", "reason": "opaque node"}
        pre = _ns_prefix(r.name)
        eids[r.name] = eg.add_term(_prefix_params(r.ir.root, pre))
        params.update(
            {
                pre + k: Param(pre + k, p.typ)
                for k, p in r.ir.params.items()
            }
        )
        leaves.update({pre + k: v for k, v in r.leaves.items()})
        vars_[r.name] = r.ir.inputs[0]
    groups = share_duplicate_params(eg, leaves)
    if not groups:
        return {"status": "declined", "reason": "no_tied_values"}
    reps: dict[str, Any] = {}
    rels: dict[str, float] = {}
    for r in recs:
        extracted = eg.extract_best(eids[r.name], cost_fn)
        # The merged param may be the *other* block's name — lower with
        # the joint tables so the shared canonical name resolves.
        opt = _lower_term(
            extracted, vars_[r.name], params, leaves, sink
        )
        vr = sink.verify(r.module, opt, r.args, rtol=verify_tol)
        rels[r.name] = vr.max_rel
        if not vr.passed:
            return {
                "status": "declined",
                "reason": f"tie verify failed on {r.name}: "
                f"{vr.max_rel:.3e}",
            }
        reps[r.name] = opt
    return {
        "status": "grafted",
        "reps": reps,
        "tied": groups,
        "rel_diff": max(rels.values()),
    }


# ---------------------------------------------------------------------------
#  The strategy + entry point
# ---------------------------------------------------------------------------


class MorphismSearch:
    """The morphism-level :class:`~catopt_core.ports.Strategy`.

    ``lift → match → reify → graft``: lift the model to a
    :class:`MorphismGraph`, fire the signature laws, reify each
    surviving match to a verified IR program, and graft the results
    into a parameter-sharing clone.  Blocks untouched by a grafted
    rewrite get the ordinary per-block search+lower pass
    (``optimize_rest``) — morphism transforms compose *over* the
    per-block pipeline, not instead of it.

    Configuration: ``laws`` (the morphism law family),
    ``block_pred`` (block selection), ``verify_tol`` (the fp gate),
    ``joint_max_iterations`` / ``joint_max_enodes`` /
    ``symmetry_budget`` (the joint e-graph bounds), and
    ``optimize_rest`` (per-block fallback for untouched blocks).
    ``optimize`` kwargs (``rules``, ``cost_fn``, ``verbose``, …)
    forward to the per-block searches.
    """

    name = "morphism"

    def __init__(
        self,
        *,
        laws: Iterable[MorphismLaw] | None = None,
        block_pred: Any = None,
        optimize_rest: bool = True,
        verify_tol: float = 1e-4,
        joint_max_iterations: int = 8,
        joint_max_enodes: int = 50_000,
        symmetry_budget: int | None = 512,
    ) -> None:
        """Store the strategy configuration."""
        self.laws = (
            tuple(laws) if laws is not None else DEFAULT_MORPHISM_LAWS
        )
        self.block_pred = block_pred
        self.optimize_rest = optimize_rest
        self.verify_tol = verify_tol
        self.joint_max_iterations = joint_max_iterations
        self.joint_max_enodes = joint_max_enodes
        self.symmetry_budget = symmetry_budget

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Lift, match, reify, graft — end to end through the ports."""
        mod, stats = _optimize_morphisms(
            model,
            x,
            optimizer=optimizer,
            laws=self.laws,
            block_pred=self.block_pred,
            optimize_rest=self.optimize_rest,
            verify_tol=self.verify_tol,
            joint_max_iterations=self.joint_max_iterations,
            joint_max_enodes=self.joint_max_enodes,
            symmetry_budget=self.symmetry_budget,
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


def _optimize_morphisms(
    model: Any,
    x: Any,
    *,
    optimizer: Any,
    laws: Iterable[MorphismLaw],
    block_pred: Any,
    optimize_rest: bool,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
    cost_fn: CostFn | None = None,
    rules: Any = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verbose: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Lift, match, reify, recompose — the morphism pipeline.

    Returns ``(model, stats)`` — ``stats["matches"]`` carries the
    per-match verdicts (``grafted`` / ``declined`` / ``skipped`` plus
    the cost/rel-diff record), ``stats["blocks"]`` the per-block
    fallback reports, ``stats["wires"]`` the boundary classification,
    and ``stats["end_to_end"]`` the whole-model equivalence check.
    """
    t_start = time.time()
    composer = optimizer.composer
    if composer is None:
        raise TypeError(
            "MorphismSearch needs a Composer port — pass composer= "
            "(or backend=) to Optimizer"
        )
    source = optimizer.source
    sink = optimizer.sink
    if cost_fn is None:
        cost_fn = flops_cost

    graph = lift_graph(
        model,
        x,
        source=source,
        composer=composer,
        block_pred=block_pred,
    )
    matches = [m for law in laws for m in law.match(graph)]
    if verbose:
        log.info(
            "[Morphism] %s; matches: %s",
            graph,
            [(m.law, m.nodes) for m in matches],
        )

    replacements: dict[str, Any] = {}
    match_stats: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    consumed_by: dict[str, str] = {}
    morphism_fires: dict[str, int] = {}
    for m in matches:
        key = f"{m.law}:{'+'.join(m.nodes)}"
        if any(n in consumed for n in m.nodes):
            match_stats[key] = {
                "status": "skipped",
                "reason": "node already rewritten",
            }
            continue
        try:
            res = _reify(
                m,
                graph,
                sink=sink,
                cost_fn=cost_fn,
                verify_tol=verify_tol,
                max_iterations=joint_max_iterations,
                max_enodes=joint_max_enodes,
                symmetry_budget=symmetry_budget,
            )
        except Exception as e:
            match_stats[key] = {
                "status": "declined",
                "reason": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            continue
        match_stats[key] = {
            "boundary": m.boundary,
            "detail": m.detail,
            **{k: v for k, v in res.items() if k != "reps"},
        }
        if res["status"] == "grafted":
            replacements.update(res["reps"])
            for n in m.nodes:
                consumed.add(n)
                consumed_by[n] = m.law
            morphism_fires[m.law] = morphism_fires.get(m.law, 0) + 1
            if verbose:
                log.info("[Morphism] %s: grafted (%s)", key, m.detail)

    block_reports: dict[str, dict[str, Any]] = {}
    if optimize_rest:
        for node in graph.nodes:
            name = node.name
            if name in consumed:
                block_reports[name] = {
                    "status": "rewritten",
                    "law": consumed_by[name],
                }
                continue
            rec = graph.record(name)
            rep: dict[str, Any] = {}
            block_reports[name] = rep
            if not rec.args or rec.ir is None:
                rep["status"] = "skipped"
                rep["reason"] = rec.note
                continue
            try:
                res_s = optimizer.search(
                    rec.module,
                    rec.example,
                    rules=rules,
                    max_iterations=max_iterations,
                    max_enodes=max_enodes,
                    max_memory_mb=max_memory_mb,
                    verbose=verbose,
                )
                lr = optimizer.lower(
                    res_s,
                    rec.example,
                    runner=IdentityRunner(),
                    verify=False,
                    verbose=verbose,
                )
                vr = sink.verify(
                    rec.module, lr.module, rec.args, rtol=verify_tol
                )
                rep["rel_diff"] = vr.max_rel
                if not vr.passed:
                    rep["status"] = "failed"
                    rep["reason"] = (
                        f"block verify failed: {vr.max_rel:.3e}"
                    )
                    continue
                replacements[name] = lr.module
                rep["status"] = "optimized"
            except Exception as e:
                rep["status"] = "failed"
                rep["error"] = f"{type(e).__name__}: {e}"

    in_place = False
    try:
        new_model = composer.clone_sharing(model)
    except Exception:
        new_model = model
        in_place = True
        replacements = {}
    composer.graft(new_model, replacements)

    stats: dict[str, Any] = {
        "morphism": True,
        "n_blocks": len(graph.nodes),
        "n_lifted": sum(1 for n in graph.nodes if not n.opaque),
        "sigs": {
            n.name: (
                {
                    "in_projs": [w.name for w in n.sig.in_projs],
                    "out_proj": [w.name for w in n.sig.out_proj],
                    "norm": n.sig.norm.kind,
                    "norm_affine": n.sig.norm.affine,
                    "norm_pre": n.sig.norm.pre,
                    "act": list(n.sig.act),
                    "residual": n.sig.residual,
                    "shape": n.sig.shape,
                }
                if n.sig is not None
                else None
            )
            for n in graph.nodes
        },
        "wires": [(w.src, w.dst, w.kind) for w in graph.wires],
        "matches": match_stats,
        "morphism_fires": morphism_fires,
        "n_rewritten": len(consumed),
        "blocks": block_reports,
        "in_place": in_place,
        "shared_params": not in_place,
    }
    if in_place:
        stats["end_to_end"] = {
            "skipped": "in_place",
            "reason": "param-sharing clone failed — model unmodified",
        }
    else:
        args2 = x if isinstance(x, tuple) else (x,)
        try:
            vr = sink.verify(model, new_model, args2)
            stats["end_to_end"] = {
                "max_abs_diff": vr.max_abs,
                "max_rel_diff": vr.max_rel,
            }
        except Exception as e:
            stats["end_to_end"] = {"error": f"{type(e).__name__}: {e}"}
    stats["wall_time_s"] = time.time() - t_start
    return new_model, stats


def optimize_morphisms(
    model: Any,
    x: Any,
    *,
    backend: Any = None,
    source: Source | None = None,
    sink: Sink | None = None,
    composer: Composer | None = None,
    meter: Meter | None = None,
    strategy: Any = None,
    **kw: Any,
) -> LowerResult:
    """Run the morphism pipeline on *model* — the function entry point.

    Assembles an :class:`~catopt_orchestrator.optimize.Optimizer` from
    the given ports (``backend`` bundle or explicit
    ``source``/``sink``/``composer``/``meter`` — the composer port is
    required for block lifting) and runs ``optimize`` under
    :class:`MorphismSearch`.  ``strategy`` overrides the search
    configuration entirely (``MorphismSearch(laws=..., ...)`` selects
    the morphism law family); remaining ``**kw`` (``rules``,
    ``cost_fn``, ``verbose``, …) reach the per-block fallback
    searches.
    """
    from catopt_orchestrator.optimize import Optimizer

    opt = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        composer=composer,
        meter=meter,
    )
    strat = strategy if strategy is not None else MorphismSearch()
    return opt.optimize(model, x, strategy=strat, **kw)
