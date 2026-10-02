"""The contraction diagram — plan 0013, stages 1+2.

The morphism engine (plan 0011) lifts a model to objects-and-arrows
at block granularity: :func:`~catopt_orchestrator.morphisms.lift_graph`
produces a :class:`~catopt_orchestrator.morphisms.MorphismGraph` whose
nodes carry structural signatures and whose wires classify the
adjacent-block boundaries.  Stage 1 re-presents that graph as the
object the plan's *contraction search* operates on: a hypergraph
where one tensor VALUE is one edge — a shared input read by ``k``
blocks is a single hyperedge with ``k`` consumer ends, not ``k``
wires.

Four node kinds cover the object layer: ``"block"`` (lifted, carries
its :class:`~catopt_orchestrator.morphisms.BlockSig`), ``"opaque"``
(boundary nodes — real objects in the flow no law may cross),
``"input"`` (leaf values no block produced — model args, bound
tensors, model-level intermediates such as residual-stream sums), and
``"output"`` (the model-output sink).  Edges carry per-end evidence:
the consumer's positional arg index, the signature role at that
position (``activation`` / ``const_table`` / ``state`` /
``opaque``), and the composer's boundary verdict when the end closes
a classified adjacent wire.  Multi-arity edges are respected, never
collapsed.

Stage 2 is the contraction move set — four moves, each pairing a
diagram-level legality check with a recipe onto the existing
certified machinery.  A move is certified iff its reified term
verifies through the sink; a move that cannot reify declines, never
grafts:

* :class:`MergeProjs` — the pairing law at diagram level: members of
  a shared-activation hyperedge that project the shared input fuse
  into one GEMM plus per-member ``split`` views
  (:func:`~catopt_core.laws.pairing.pair_shared_input_linears` on the
  joint family term, then coordinated ``extract_paired`` — the
  single-block fan-in arm included).
* :class:`FactorShared` — factor a computation or weight shared by
  >=2 branches into a shared intermediate; delegates candidacy to the
  existing family laws (:class:`CrossBlockCSE` for identical
  subterms, :class:`KVLatentShare` for the common right-factor / KV
  latent fold) relabelled as one diagram move.
* :class:`ReorderCompose` — contract a node chain differently:
  contiguous subwindows of composable chain wires and residual-stream
  runs, each reified through the window-joint machinery (the window
  laws generalised — a declined maximal window still yields its
  profitable pieces).
* :class:`SplitLeaf` — a leaf shared across member leaf tables and
  read only through slice sites (``select`` / ``slice`` / ``narrow``
  / ``getitem`` — the stacked-parameter spelling) splits into
  per-consumer materialised slice params.  The slice ops deliberately
  do NOT fold at lowering, so materialising is a strict op-count win
  the per-member verify still gates.

The driver (:class:`ContractionSearch`) has two modes: ``"greedy"``
— the stage-2 local pass where widest candidates claim their nodes
first and declines leave the nodes to the ordinary per-block
fallback — and ``"search"`` (stage 3), which explores move
*orderings* over diagram states: a grafted move writes its members'
reified bodies back and candidacy re-runs on the rewritten state, so
compositions no single candidate can reach become reachable.  The
search itself lives in
:mod:`catopt_orchestrator.diagram_search`; every delivered
transition still passes the move's own certified reify, and the
whole-model ``end_to_end`` verify gates the composition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import Any, Protocol, runtime_checkable

from catopt_core.cost import dag_cost, launch_aware_cost
from catopt_core.ir import Op, Param, TensorType, op_repr_dag
from catopt_core.laws.pairing import (
    _exact_equal,
    _is_tensor,
    pair_shared_input_convs,
    pair_shared_input_linears,
)
from catopt_core.pipeline import LowerResult
from catopt_core.ports import Composer, CostFn, Meter, Sink, Source

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
from catopt_orchestrator.crossblock_cse import CrossBlockCSE
from catopt_orchestrator.morphisms import (
    BlockSig,
    MorphismGraph,
    MorphismMatch,
    ReifySpec,
)
from catopt_orchestrator.morphisms_kv import KVLatentShare
from catopt_orchestrator.runners import IdentityRunner

log = logging.getLogger("catopt_orchestrator.diagram")

__all__ = [
    "DEFAULT_MOVES",
    "ContractionSearch",
    "DEdge",
    "DEnd",
    "DNode",
    "Diagram",
    "DiagramMove",
    "FactorShared",
    "MergeProjs",
    "ReorderCompose",
    "SplitLeaf",
    "diagram_of_graph",
    "lift_diagram",
    "optimize_diagram",
]


# ---------------------------------------------------------------------------
#  Stage 1 — the diagram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DEnd:
    """One consumer end of a hyperedge.

    ``pos`` is the consumer's positional arg index (``-1`` on the
    model-output sink end); ``role`` is the signature input kind at
    that position (``"activation"`` / ``"const_table"`` /
    ``"state"`` / ``"opaque"``), ``"unresolved"`` when the consumer
    carries no signature, and ``"output"`` on the sink end.
    ``wire`` is the composer's boundary verdict when this end closes
    a classified adjacent-block wire at the activation position —
    ``""`` otherwise (context reads and non-adjacent flows are never
    wires).
    """

    node: str
    pos: int
    role: str
    wire: str = ""


@dataclass(frozen=True)
class DEdge:
    """One hyperedge: a single tensor value and every consumer end.

    ``src`` is the producer node — the block whose captured output
    object IS this value, or the leaf ``<in:k>`` node when no block
    produced it.  ``len(ends) > 1`` is a true hyperedge: one shared
    tensor feeding several nodes.
    """

    src: str
    ends: tuple[DEnd, ...]

    @property
    def dsts(self) -> tuple[str, ...]:
        """The consumer node names, in first-seen order."""
        return tuple(e.node for e in self.ends)

    @property
    def arity(self) -> int:
        """Consumer count — ``> 1`` makes the edge multi-ary."""
        return len(self.ends)

    @property
    def wire(self) -> str:
        """Return the boundary verdict on an activation end.

        ``""`` when no end closes a classified wire.
        """
        for e in self.ends:
            if e.wire:
                return e.wire
        return ""


@dataclass(frozen=True)
class DNode:
    """One node of the contraction diagram.

    ``kind`` is ``"block"`` (lifted — ``sig`` carries the
    :class:`~catopt_orchestrator.morphisms.BlockSig`), ``"opaque"``
    (a boundary node: a real object in the composition whose export
    produced no usable signature — ``note`` records the honest
    decline), ``"input"`` (a leaf value no block produced) or
    ``"output"`` (the model-output sink).

    For ``input`` nodes ``role`` records the leaf's provenance:
    ``"model_input"`` (a model call arg), ``"bound"`` (a parameter /
    buffer / module-attr tensor — the shared tables), or
    ``"intermediate"`` (a model-level computed value — the
    residual-stream sums ``x + f(x)`` the blocks consume); ``model``
    absent at build time leaves it ``"unknown"``.
    """

    name: str
    kind: str
    sig: BlockSig | None = None
    role: str = ""
    note: str = ""


class Diagram:
    """The model's contraction diagram over the morphism layer.

    ``nodes`` covers the lifted blocks, opaque boundary nodes, leaf
    inputs and the output sink; ``edges`` are the hyperedges — one
    per distinct consumed tensor OBJECT (identity, not equality: two
    block consumers reading the same live tensor share one edge).
    The underlying :class:`MorphismGraph` stays reachable as
    ``diagram.graph`` — the lift records, captured IO and wire
    verdicts the moves and the reify machinery read.
    """

    def __init__(
        self,
        graph: MorphismGraph,
        nodes: list[DNode],
        edges: list[DEdge],
    ) -> None:
        """Store the diagram — construction is :func:`lift_diagram`'s."""
        self.graph = graph
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self._by_name = {n.name: n for n in self.nodes}
        self._in: dict[str, list[DEdge]] = {}
        self._out: dict[str, list[DEdge]] = {}
        for e in self.edges:
            self._out.setdefault(e.src, []).append(e)
            for end in e.ends:
                self._in.setdefault(end.node, []).append(e)

    @property
    def wires(self) -> tuple:
        """The morphism layer's adjacent-pair boundary verdicts."""
        return self.graph.wires

    def node(self, name: str) -> DNode:
        """Return the :class:`DNode` for *name*."""
        return self._by_name[name]

    def sig(self, name: str) -> BlockSig | None:
        """Return the node's signature — ``None`` when not a block."""
        return self._by_name[name].sig

    def record(self, name: str) -> Any:
        """Return the block's lift record (IR, leaves, captured args)."""
        return self.graph.record(name)

    def incoming(self, name: str) -> tuple[DEdge, ...]:
        """Return the hyperedges consumed by *name*."""
        return tuple(self._in.get(name, ()))

    def outgoing(self, name: str) -> tuple[DEdge, ...]:
        """Return the hyperedges *name* produces."""
        return tuple(self._out.get(name, ()))

    def consumers(self, name: str) -> tuple[str, ...]:
        """Distinct consumer node names across *name*'s outgoing edges."""
        seen: dict[str, None] = {}
        for e in self._out.get(name, ()):
            for end in e.ends:
                seen.setdefault(end.node, None)
        return tuple(seen)

    def shared(self) -> tuple[DEdge, ...]:
        """Return the true hyperedges — edges consumed by >=2 ends."""
        return tuple(e for e in self.edges if e.arity > 1)

    def activation_families(
        self,
    ) -> list[tuple[DEdge, tuple[str, ...]]]:
        """Shared-activation hyperedges and their member node names.

        An edge contributes when >=2 distinct nodes consume its value
        at an ``activation`` position — the same-``x`` families the
        family-level rewrites run over.
        """
        out: list[tuple[DEdge, tuple[str, ...]]] = []
        for e in self.edges:
            members = tuple(
                dict.fromkeys(
                    en.node for en in e.ends if en.role == "activation"
                )
            )
            if len(members) >= 2:
                out.append((e, members))
        return out

    def __repr__(self) -> str:
        """Return a compact summary of the diagram."""
        return (
            f"Diagram(nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}, "
            f"hyper={len(self.shared())}, "
            f"opaque={sum(1 for n in self.nodes if n.kind == 'opaque')})"
        )


def _leaf_roles(
    graph: MorphismGraph, model: Any, x: Any
) -> tuple[dict[int, str], set[int], set[int]]:
    """Build the producer map, model-arg ids and bound-tensor ids.

    ``producers`` maps ``id(out_obj)`` to the earliest block emitting
    it — a pass-through block returning its input does not *create*
    the value, so the earliest writer wins.  ``model_in`` marks the
    call args; ``bound_ids`` the params/buffers/tensor attrs.
    """
    producers: dict[int, str] = {}
    for n in graph.nodes:
        rec = graph.record(n.name)
        if _is_tensor(rec.out_obj):
            producers.setdefault(id(rec.out_obj), n.name)
    args = x if isinstance(x, tuple) else (x,)
    model_in = {id(a) for a in args if _is_tensor(a)}
    bound_ids = (
        M._param_bound_ids(model) if model is not None else set()
    )
    return producers, model_in, bound_ids


def _collect_ends(graph: MorphismGraph) -> dict[int, list[DEnd]]:
    """Group every consumed tensor object into consumer ends.

    Keyed on ``id(obj)`` — object identity, the same evidence the
    lift's wires use.  Non-tensor args produce no end; a signature
    gap (opaque node, arg past the signature) is ``unresolved``.
    """
    ends_of: dict[int, list[DEnd]] = {}
    for n in graph.nodes:
        rec = graph.record(n.name)
        sig = n.sig
        for j, obj in enumerate(rec.in_objs):
            if not _is_tensor(obj):
                continue
            role = (
                sig.inputs[j].kind
                if sig is not None and j < len(sig.inputs)
                else "unresolved"
            )
            ends_of.setdefault(id(obj), []).append(
                DEnd(n.name, j, role)
            )
    for obj in graph._model_out_objs:
        if _is_tensor(obj):
            ends_of.setdefault(id(obj), []).append(
                DEnd("<output>", -1, "output")
            )
    return ends_of


def _leaf_role(
    obj_id: int, model_in: set[int], bound_ids: set[int], model: Any
) -> str:
    """Classify one leaf object's provenance."""
    if obj_id in model_in:
        return "model_input"
    if obj_id in bound_ids:
        return "bound"
    return "unknown" if model is None else "intermediate"


def _annotate(
    ends: list[DEnd], src: str, wire_map: dict
) -> tuple[DEnd, ...]:
    """Mark each activation end with the ``(src, end.node)`` verdict.

    Only the activation end can close a wire — context reads and the
    output sink carry no boundary verdict.
    """
    return tuple(
        _dc_replace(e, wire=wire_map.get((src, e.node), ""))
        if e.role == "activation"
        else e
        for e in ends
    )


def diagram_of_graph(
    graph: MorphismGraph, *, model: Any = None, x: Any = None
) -> Diagram:
    """Build the hyperedge layer over an already-lifted graph.

    Edges are keyed on the consumed tensor's object identity — the
    same evidence :func:`lift_graph` used for its wires, generalised
    to every arg position and every consumer.  ``model``/``x`` only
    colour the leaf roles (bound / model_input / intermediate); the
    topology itself needs neither.
    """
    producers, model_in, bound_ids = _leaf_roles(graph, model, x)
    ends_of = _collect_ends(graph)
    wire_map = {(w.src, w.dst): w.kind for w in graph.wires}
    nodes: list[DNode] = [
        DNode(
            n.name,
            "opaque" if n.opaque else "block",
            sig=n.sig,
            note=graph.record(n.name).note or "",
        )
        for n in graph.nodes
    ]
    leaf_count = 0
    edges: list[DEdge] = []
    for obj_id, ends in ends_of.items():
        src = producers.get(obj_id)
        if src is None:
            src = f"<in:{leaf_count}>"
            leaf_count += 1
            nodes.append(
                DNode(
                    src,
                    "input",
                    role=_leaf_role(obj_id, model_in, bound_ids, model),
                )
            )
        edges.append(DEdge(src, _annotate(ends, src, wire_map)))
    if any(
        end.node == "<output>"
        for ends in ends_of.values()
        for end in ends
    ):
        nodes.append(DNode("<output>", "output"))
    return Diagram(graph, nodes, edges)


def lift_diagram(
    model: Any,
    x: Any,
    *,
    source: Source,
    composer: Composer,
    block_pred: Any = None,
) -> Diagram:
    """Lift *model* to its :class:`Diagram` — the object layer.

    Same lift as :func:`~catopt_orchestrator.morphisms.lift_graph`
    (composer-selected blocks, captured IO, two-probe evidence); the
    hyperedge layer is derived purely additively on the resulting
    :class:`MorphismGraph`, so every honesty guarantee of the lift
    carries over verbatim — opaque blocks stay boundary nodes, wires
    that are not simple value flows stay ``opaque`` verdicts, and a
    shared tensor is one multi-arity edge.
    """
    graph = M.lift_graph(
        model,
        x,
        source=source,
        composer=composer,
        block_pred=block_pred,
    )
    return diagram_of_graph(graph, model=model, x=x)


# ---------------------------------------------------------------------------
#  Stage 2 — the contraction moves
# ---------------------------------------------------------------------------


@runtime_checkable
class DiagramMove(Protocol):
    """A contraction move on the diagram — signatures + hyperedges only.

    ``candidates`` reads the :class:`Diagram` (the object layer) and
    returns :class:`MorphismMatch` candidates whose
    :class:`ReifySpec`s name the certified machinery each move's
    ``reify`` executes; a candidate that cannot reify declines —
    never a graft.
    """

    name: str

    def candidates(self, diagram: Diagram) -> list[MorphismMatch]:
        """Return the move's candidates on *diagram*."""
        ...

    def reify(
        self,
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
        """Reify one candidate; return the graft record or a decline."""
        ...


class _MorphismReify:
    """Moves whose candidates carry morphism-layer reify specs."""

    def reify(
        self,
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
        """Delegate to the morphism engine's certified reify."""
        return M._reify(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            symmetry_budget=symmetry_budget,
        )


def _usable(diagram: Diagram, name: str) -> bool:
    """Check a node is family material — the ``_input_families`` gates.

    Lifted (exportable, signature-consistent per
    :func:`~catopt_orchestrator.morphisms_kv._usable_member`) plus the
    captured-IO gates: the block ran exactly once and its activation
    arg object was captured.
    """
    rec = diagram.record(name)
    return (
        rec.in_obj is not None
        and rec.calls == 1
        and K._usable_member(diagram.graph, name)
    )


class MergeProjs:
    """``x @ W_i`` siblings on one leaf fuse to one GEMM + split views.

    The pairing law (:func:`pair_shared_input_linears`) expressed at
    diagram level: a shared-activation hyperedge whose member bodies
    project the shared input.  Two arms: *cross* (a >=2-block family
    on one input object — the additive family graft, evidence via
    :func:`~catopt_orchestrator.morphisms_kv._family_evidence`) and
    *intra* (one block with >=2 input projections — the inside-one-
    block fan-in).  Candidacy is signature-level; the value side is
    the reify's job — coordinated extraction picks the fused form
    only when its true DAG cost beats the members' (under
    ``launch_aware_cost`` / ``count_cost`` one fused GEMM wins
    outright; under pure ``flops_cost`` the forms tie and the move
    honestly declines).
    """

    name = "merge_projs"

    def candidates(self, diagram: Diagram) -> list[MorphismMatch]:
        """Match shared-input families with >=2 projecting members."""
        out = self._family_candidates(diagram)
        out.extend(self._intra_candidates(diagram))
        out.sort(key=lambda m: (-len(m.nodes), m.nodes))
        return out

    def _family_candidates(
        self, diagram: Diagram
    ) -> list[MorphismMatch]:
        """Shared-activation hyperedges with >=2 projecting members."""
        out: list[MorphismMatch] = []
        for _edge, fam in diagram.activation_families():
            members = tuple(n for n in fam if _usable(diagram, n))
            if len(members) < 2:
                continue
            proj = [
                n
                for n in members
                if (s := diagram.sig(n)) is not None and s.in_projs
            ]
            if len(proj) < 2:
                continue
            if not K._family_evidence(
                diagram.graph, members, list(fam)
            ):
                continue
            out.append(
                MorphismMatch(
                    law=self.name,
                    nodes=members,
                    boundary="family",
                    reify=ReifySpec(mode="pair", rules="compose"),
                    detail=(
                        f"{'+'.join(members)}: {len(proj)} members "
                        "share one input hyperedge's projections"
                    ),
                )
            )
        return out

    def _intra_candidates(
        self, diagram: Diagram
    ) -> list[MorphismMatch]:
        """Blocks carrying >=2 fan-in projections of their own input."""
        out: list[MorphismMatch] = []
        for n in diagram.nodes:
            sig = n.sig
            if (
                n.kind == "block"
                and sig is not None
                and len(sig.in_projs) >= 2
            ):
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=(n.name,),
                        boundary="intra",
                        reify=ReifySpec(mode="pair", rules="compose"),
                        detail=(
                            f"{n.name}: {len(sig.in_projs)} fan-in "
                            "projections on the block input"
                        ),
                    )
                )
        return out

    def reify(
        self,
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
        """Fuse the shared-input projections; certify + cost-gate."""
        return _reify_merge(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            symmetry_budget=symmetry_budget,
        )


def _reify_merge(
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
    """Reify a merge_projs match; return the graft record or a decline.

    Build the family joint (the additive ``Σ b_i(x)`` — the same
    machinery the KV/CSE family rewrites use), saturate the compose
    recipe, run the pairing passes over the shared input e-class, and
    compare the coordinated ``extract_paired`` term against the
    greedy one on true DAG cost — the fused form only wins when all
    members take it.  Then the shared ``_family_gate``: cost gate
    against the un-rewritten joint, additive-slot shape check, fp64
    verify, slot replacements.
    """
    spec = match.reify
    prep = K._family_prep(match, graph)
    if prep.get("status") == "declined":
        return prep
    joint, params, leaves = (
        prep["joint"],
        prep["params"],
        prep["leaves"],
    )
    rules = M._recipe_rules(spec.rules)
    eg, eid = M._saturate(
        joint,
        rules,
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        symmetry_budget=symmetry_budget,
    )
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    info: dict[str, Any] = {"joint": op_repr_dag(joint)}
    if not groups:
        return {
            "status": "declined",
            "reason": "no shared-input projections",
            **info,
        }
    eg.rebuild()
    # Brief re-saturation so the fused members see the recipe rules —
    # same ordering as the monolithic ``_pairing_and_lifts``.
    budgets = (
        {
            r.name: symmetry_budget
            for r in rules.tagged(M._law_tags.EXPANSIVE)
        }
        if symmetry_budget is not None
        else None
    )
    eg.run(
        rules,
        eid,
        max_iterations=5,
        max_nodes=max_enodes,
        rule_budgets=budgets,
    )
    memo: dict = {}
    best = eg.extract_best(eid, cost_fn)
    base = dag_cost(best, cost_fn, memo=memo)
    forced = eg.extract_paired(eid, cost_fn, groups)
    fc = (
        dag_cost(forced, cost_fn, memo=memo)
        if forced is not None
        else float("inf")
    )
    info["paired_groups"] = len(groups)
    if fc <= base:
        best = forced
        info["paired_extract"] = True
    else:
        info["paired_extract"] = False
        info["paired_delta"] = fc - base
    info["reified"] = op_repr_dag(best)
    return K._family_gate(
        match,
        prep["recs"],
        prep["irs"],
        graph,
        intra=prep["intra"],
        sink=sink,
        cost_fn=cost_fn,
        verify_tol=verify_tol,
        joint=joint,
        best=best,
        x=prep["x"],
        inputs=prep["inputs"],
        params=params,
        leaves=leaves,
        info=info,
    )


class FactorShared(_MorphismReify):
    """Factor a subtree shared by >=2 branches into one intermediate.

    The KV-latent-share generalisation at diagram level: members of a
    shared-activation hyperedge whose bodies recompute the identical
    subterm (identical op trees, leaf-certified params) fold to one
    computation — or whose shared-data projections admit a certified
    common right-factor fold through one latent.  Candidacy delegates
    to the two existing family laws and relabels them as one diagram
    move; the reify is the certified morphism machinery (``cse`` /
    ``family`` modes — additive family evidence, constructed-equality
    witness, cost gate, fp64 verify).

    Parameters mirror :class:`KVLatentShare` — ``cse``/``kv`` switch
    the arms independently; ``budget=None`` keeps the exact gate.
    """

    name = "factor_shared"

    def __init__(
        self,
        *,
        cse: bool = True,
        kv: bool = True,
        tokens: tuple[str, ...] = K._KV_TOKENS,
        qkv_tokens: tuple[str, ...] = K._QKV_TOKENS,
        factor_tol: float = 1e-8,
        budget: float | None = None,
    ) -> None:
        """Store the arm switches and the latent-law configuration."""
        self.cse = bool(cse)
        self.kv = bool(kv)
        self.tokens = tuple(tokens)
        self.qkv_tokens = tuple(qkv_tokens)
        self.factor_tol = float(factor_tol)
        self.budget = None if budget is None else float(budget)

    def candidates(self, diagram: Diagram) -> list[MorphismMatch]:
        """Match shared-subterm and shared-latent families."""
        out: list[MorphismMatch] = []
        if self.cse:
            out.extend(CrossBlockCSE().match(diagram.graph))
        if self.kv:
            out.extend(
                KVLatentShare(
                    tokens=self.tokens,
                    qkv_tokens=self.qkv_tokens,
                    factor_tol=self.factor_tol,
                    budget=self.budget,
                ).match(diagram.graph)
            )
        return [
            _dc_replace(
                m,
                law=self.name,
                detail=f"{m.law}: {m.detail}",
            )
            for m in out
        ]


#: Wire kinds carrying a plain ``B(A(x))`` contraction.
_CHAIN_KINDS = frozenset({"chain"})
#: The wrapped close a chain window may take (``y + B(y)``).
_CHAIN_WRAP = frozenset({"chain_wrapped"})
#: Residual-stream kinds — the additive monoid's window grammar.
_RESIDUAL_KINDS = frozenset({"residual", "residual_wrapped"})


class ReorderCompose(_MorphismReify):
    """Contract a block chain differently — the window laws' general.

    On a run of ``chain`` wires (``B(A(x))`` — plain function
    composition) any contiguous subwindow contracts to one morphism:
    reify builds the nested joint term and the compose recipe folds
    the ``out ∘ in`` projections.  On a residual stream
    (``s_j = s_{j-1} + f_j(s_{j-1})``) the same move reads as
    reassociation: the additive monoid commutes a receiver's
    in-projections over the earlier addends.  Unlike the window laws
    the move enumerates EVERY contiguous subwindow of a legal run —
    widest first — so a maximal window that declines on cost or
    verify still yields its profitable pieces (the greedy driver
    claims nodes, so the search is local, not exhaustive).
    """

    name = "reorder_compose"

    def candidates(self, diagram: Diagram) -> list[MorphismMatch]:
        """Emit legal chain/residual subwindows, widest first."""
        out: list[MorphismMatch] = []
        out.extend(self._chain_candidates(diagram.graph))
        out.extend(self._residual_candidates(diagram.graph))
        out.sort(key=lambda m: (-len(m.nodes), m.nodes))
        return out

    def _chain_candidates(self, graph: MorphismGraph) -> list:
        """All composable subwindows of the maximal chain runs.

        A run is ``chain`` wires plus an optional ``chain_wrapped``
        tail — the parent's ``y + B(y)`` wrap can only close a window,
        so a wrapped wire is always a run's last (a bare wrapped wire
        is a one-wire run: the OutInCompose wrapped pair).
        """
        wires = graph.wires
        n = len(wires)
        out: list[MorphismMatch] = []
        i = 0
        while i < n:
            if wires[i].kind == "chain_wrapped":
                if not M._wire_composes(graph, wires[i], _CHAIN_WRAP):
                    i += 1
                    continue
                j = i  # a bare wrap closes immediately
            elif not M._wire_composes(graph, wires[i], _CHAIN_KINDS):
                i += 1
                continue
            else:
                j = i
                while j + 1 < n and M._wire_composes(
                    graph, wires[j + 1], _CHAIN_KINDS
                ):
                    j += 1
                if j + 1 < n and M._wire_composes(
                    graph, wires[j + 1], _CHAIN_WRAP
                ):
                    j += 1
            self._emit_windows(wires, i, j, distribute=False, out=out)
            i = j + 1
        return out

    def _residual_candidates(self, graph: MorphismGraph) -> list:
        """Residual-stream subwindows + the single-wire absorb pairs."""
        wires = graph.wires
        n = len(wires)
        out: list[MorphismMatch] = []
        i = 0
        while i < n:
            if wires[i].kind not in _RESIDUAL_KINDS:
                i += 1
                continue
            j = i
            while j + 1 < n and wires[j + 1].kind == "residual_wrapped":
                j += 1
            if j + 1 < n and wires[j + 1].kind == "residual":
                j += 1
            self._emit_windows(
                wires,
                i,
                j,
                distribute=True,
                out=out,
                legal=lambda nodes: self._residual_legal(graph, nodes),
            )
            i = j + 1
        return out

    def _emit_windows(
        self,
        wires: tuple,
        i: int,
        j: int,
        *,
        distribute: bool,
        out: list,
        legal: Any = None,
    ) -> None:
        """Emit every contiguous subwindow of wire-run ``[i, j]``.

        ``legal`` is an optional per-window predicate (the residual
        stream's commute check); ``distribute`` marks the residual
        specs (the additive bilinear offers).
        """
        for a in range(i, j + 1):
            for b in range(a, j + 1):
                kinds = tuple(wires[t].kind for t in range(a, b + 1))
                nodes = M._window_nodes(tuple(wires), a, b)
                if legal is not None and not legal(nodes):
                    continue
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=nodes,
                        boundary="+".join(kinds),
                        reify=ReifySpec(
                            mode=kinds[-1],
                            rules="compose",
                            distribute=distribute,
                            kinds=kinds,
                        ),
                        detail=(
                            f"{nodes[0]}->…->{nodes[-1]}: contract "
                            f"{len(nodes)}-block window "
                            f"({'+'.join(kinds)})"
                        ),
                    )
                )

    @staticmethod
    def _residual_legal(graph: MorphismGraph, nodes: tuple) -> bool:
        """Check the residual precondition — stream commutes or B projects.

        A multi-block window needs a real commute opportunity
        (:func:`M._stream_commutes`); the single-wire pair only asks
        the receiver carry in-projections — the residual_absorb law's
        own predicate (``linear(x + A(x), W)`` distributes whether or
        not A projects).
        """
        if len(nodes) <= 2:
            sig = graph.sig(nodes[-1])
            return sig is not None and bool(sig.in_projs)
        return M._stream_commutes(graph, nodes)


#: View-site ops the split move materialises — the extraction family
#: that deliberately stays runtime at lowering (unlike foldable
#: views), so replacing a site by a fresh Param is a strict op win.
_SPLIT_VIEW_OPS = frozenset({"select", "slice", "narrow", "getitem"})


def _split_sites(body: Any, names: frozenset) -> list[Op]:
    """View ops reading a shared ``Param`` leaf directly.

    Conservative: the site's arg0 must be the leaf itself (a
    ``select(reshape(W), …)`` chain is an honest non-reach).
    """
    return [
        n
        for n in M._iter_ops(body)
        if n.op in _SPLIT_VIEW_OPS
        and n.args
        and isinstance(n.args[0], Param)
        and n.args[0].name in names
    ]


def _slice_value(node: Op, val: Any) -> Any | None:
    """Evaluate one view site on the shared leaf's captured value.

    Duck-typed indexing (the module is backend-neutral); an op the
    tensor cannot evaluate returns ``None`` — the site just does not
    split.
    """
    a = node.attrs
    try:
        if node.op == "select":
            sel = getattr(val, "select", None)
            if callable(sel):
                return sel(int(a.get("dim", 0)), int(a.get("index", 0)))
            return val[int(a.get("index", 0))]
        if node.op == "narrow":
            nar = getattr(val, "narrow", None)
            if callable(nar):
                return nar(
                    int(a.get("dim", 0)),
                    int(a.get("start", 0)),
                    int(a.get("length", 0)),
                )
            return None
        if node.op == "slice":
            dim = int(a.get("dim", 0))
            return val[
                (slice(None),) * dim
                + (slice(a.get("start"), a.get("end"), a.get("step")),)
            ]
        if node.op == "getitem":
            return val[a.get("index", 0)]
    except Exception:
        return None
    return None


def _leaf_groups(
    graph: MorphismGraph,
) -> list[dict[str, tuple[Any, list[str]]]]:
    """Leaf-table value groups — {member: (ir, [leaf names])}.

    Groups bitwise-equal tensor entries of the blocks' leaf tables
    (the ``share_duplicate_params`` value convention): one tensor
    object exported under two blocks' names — or equal-valued copies —
    is ONE leaf value in the diagram.  Whether the value is *shared*
    (>=2 view-site consumers, across members or within one) is the
    candidate gate's call.
    """
    reps: list[Any] = []
    groups: list[dict[str, tuple[Any, list[str]]]] = []
    for n in graph.nodes:
        rec = graph.record(n.name)
        if rec.ir is None:
            continue
        for lname, val in rec.leaves.items():
            if not _is_tensor(val):
                continue
            for gi, rep in enumerate(reps):
                if val is rep or _exact_equal(val, rep):
                    groups[gi].setdefault(n.name, (rec.ir, []))[
                        1
                    ].append(lname)
                    break
            else:
                reps.append(val)
                groups.append({n.name: (rec.ir, [lname])})
    return groups


class SplitLeaf:
    """A shared leaf splits into per-consumer views where profitable.

    Diagram reading: a leaf tensor shared across member leaf tables
    (bitwise-equal values — one logical leaf feeding >=2 consumers)
    that every consumer reads only through slice sites —
    ``select(W, dim, i)`` and kin, the stacked-parameter spelling of
    routed-MoE / fused-qkv weights.  The move materialises each
    consumer's slice as its own Param (the evaluated view value): the
    slice ops deliberately do not fold at lowering, so each
    materialised site removes a runtime gather — and the plain
    ``Param`` it leaves behind re-enters the pairing/tying passes a
    view term cannot.

    Per-member reify: cost-gate the rewritten body against the
    original, then verify the materialised module against the block
    on its captured args — members failing either gate keep their
    original module; the grafted subset is still exact.
    """

    name = "split_leaf"

    def candidates(self, diagram: Diagram) -> list[MorphismMatch]:
        """Match shared leaves read through >=2 slice sites."""
        out: list[MorphismMatch] = []
        for group in _leaf_groups(diagram.graph):
            members: list[str] = []
            leaf_names: dict[str, tuple] = {}
            n_sites = 0
            for name, (ir, lnames) in group.items():
                sites = _split_sites(ir.root, frozenset(lnames))
                if sites:
                    members.append(name)
                    leaf_names[name] = tuple(lnames)
                    n_sites += len(sites)
            if len(members) < 2 and n_sites < 2:
                continue
            out.append(
                MorphismMatch(
                    law=self.name,
                    nodes=tuple(members),
                    boundary="split",
                    reify=ReifySpec(
                        mode="split",
                        rules="compose",
                        extra={"leaf_names": leaf_names},
                    ),
                    detail=(
                        f"{'+'.join(members)}: shared leaf read "
                        f"through {n_sites} view site(s)"
                    ),
                )
            )
        return out

    def reify(
        self,
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
        """Split the shared leaf; certify + cost-gate per member."""
        return _reify_split(
            match,
            graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
        )


def _materialise(val: Any) -> Any:
    """Return an owning copy of the sliced value — contiguous, cloned."""
    cont = getattr(val, "contiguous", None)
    if callable(cont):
        val = cont()
    clone = getattr(val, "clone", None)
    if callable(clone):
        val = clone()
    return val


def _materialise_sites(
    name: str, sites: list, leaves: dict, params: dict, out: dict
) -> dict:
    """Materialise each view site into a fresh ``p_split_*`` Param.

    Evaluates the view on the captured leaf value
    (:func:`_slice_value` — duck-typed, backend-free), registers the
    materialised tensor in the leaf/param tables, and returns the
    ``{site_op: Param}`` replacement map.  Sites whose value cannot
    be evaluated are skipped — the caller's empty-map check is the
    honest decline.
    """
    repl: dict[Any, Any] = {}
    for site in sites:
        leaf_val = leaves.get(site.args[0].name)
        val = _slice_value(site, leaf_val)
        if val is None:
            continue
        val = _materialise(val)
        pname = f"p_split_{M._ns_prefix(name)}{len(repl)}"
        shape = getattr(val, "shape", None)
        ps = Param(
            pname,
            TensorType(
                tuple(int(d) for d in shape)
                if shape is not None
                else ()
            ),
        )
        params[pname] = ps
        out[pname] = val
        repl[site] = ps
    return repl


def _split_member(
    name: str,
    rec: Any,
    sig: BlockSig | None,
    leaf_names: tuple,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
) -> tuple[Any | None, dict[str, Any]]:
    """Split one member's view sites; return ``(rep, member stats)``.

    ``rep`` is the materialised module when the rewrite improves the
    member's own DAG cost AND verifies against the captured call —
    else ``None`` and the stats carry the honest verdict.
    """
    ir = rec.ir
    st: dict[str, Any] = {"leaf_names": list(leaf_names)}
    if ir is None:
        return None, {
            **st,
            "status": "declined",
            "reason": "opaque node",
        }
    sites = _split_sites(ir.root, frozenset(leaf_names))
    if not sites:
        return None, {
            **st,
            "status": "declined",
            "reason": "no view sites on the shared leaf",
        }
    params = dict(ir.params)
    leaves = dict(rec.leaves)
    repl = _materialise_sites(name, sites, rec.leaves, params, leaves)
    if not repl:
        return None, {
            **st,
            "status": "declined",
            "reason": "no materialisable view site",
        }
    body2 = K._replace_nodes(ir.root, repl)
    st["cost_before"] = dag_cost(ir.root, cost_fn)
    st["cost_after"] = dag_cost(body2, cost_fn)
    if not st["cost_after"] < st["cost_before"]:
        return None, {
            **st,
            "status": "declined",
            "reason": "no_improvement",
        }
    if not ir.inputs:
        return None, {
            **st,
            "status": "declined",
            "reason": "no input vars",
        }
    act = M._act_index(sig.inputs) if sig is not None else 0
    var = ir.inputs[act]
    opt = M._lower_term(
        body2,
        var,
        params,
        leaves,
        sink,
        tuple(ir.inputs),
    )
    vr = sink.verify(rec.module, opt, rec.args, rtol=verify_tol)
    st["rel_diff"] = vr.max_rel
    if not vr.passed:
        return None, {
            **st,
            "status": "declined",
            "reason": f"split verify failed: {vr.max_rel:.3e}",
        }
    st["_term"] = (body2, params, leaves)
    return opt, {**st, "status": "grafted", "sites": len(repl)}


def _reify_split(
    match: MorphismMatch,
    graph: MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
) -> dict[str, Any]:
    """Reify a split_leaf match — per-member materialisation + verify.

    No joint term: the members' inputs need not coincide (the leaf is
    shared, not the stream), so each block's own body is rewritten
    and verified independently — the grafted subset is exactly the
    members whose slice materialisation is cheaper and verified.
    """
    extra = match.reify.extra or {}
    leaf_names: dict[str, tuple] = extra.get("leaf_names", {})
    reps: dict[str, Any] = {}
    terms: dict[str, tuple] = {}
    members: dict[str, Any] = {}
    for name in match.nodes:
        rec = graph.record(name)
        rep, st = _split_member(
            name,
            rec,
            graph.sig(name),
            leaf_names.get(name, ()),
            sink,
            cost_fn,
            verify_tol,
        )
        members[name] = st
        if rep is not None:
            reps[name] = rep
            # grafted ⇒ ``_term`` is always present (``_split_member``
            # sets it on the verified path only)
            terms[name] = st.pop("_term")
    info: dict[str, Any] = {"members": members}
    if not reps:
        return {
            "status": "declined",
            "reason": "no member split",
            **info,
        }
    return {
        "status": "grafted",
        "reps": reps,
        "_terms": terms,
        **info,
    }


#: The default move set, in claim order: family-level structure first
#: (a factored/merged family claims its members before chain
#: contraction sees them), then chain windows, then leaf splits.
DEFAULT_MOVES: tuple[DiagramMove, ...] = (
    FactorShared(),
    MergeProjs(),
    ReorderCompose(),
    SplitLeaf(),
)


# ---------------------------------------------------------------------------
#  The driver — greedy local search over the move set
# ---------------------------------------------------------------------------


class ContractionSearch:
    """The diagram-level :class:`~catopt_core.ports.Strategy`.

    ``lift → candidates → reify → graft`` over the contraction
    diagram: the moves enumerate certified-machinery candidates on
    the hypergraph, each reify is cost-gated and sink-verified, and
    the grafted results land in a parameter-sharing clone — same
    delivery discipline as :class:`MorphismSearch` (untouched blocks
    get the ordinary per-block ``optimize_rest`` pass).

    Configuration: ``moves`` (the move set), ``mode`` —
    ``"greedy"`` (stage-2 behaviour: candidates claimed widest-first
    on the initial lift, the default while stage-4 verification work
    hardens the composed path) or ``"search"`` (stage 3: best-first
    search over diagram states — move orderings are explored, grafts
    rewrite the member bodies and new candidates can fire on the
    rewritten state; see
    :mod:`catopt_orchestrator.diagram_search`) — plus
    ``search_depth`` / ``search_states`` (the search bounds),
    ``block_pred`` (block selection), ``verify_tol`` (the fp gate),
    ``joint_max_iterations`` / ``joint_max_enodes`` /
    ``symmetry_budget`` (the joint e-graph bounds), ``optimize_rest``
    (per-block fallback).  ``cost_fn`` prices the reified terms —
    ``launch_aware_cost`` by default, since the fusion moves' wins
    are kernel-count wins pure FLOPs cannot see.
    """

    name = "contraction"

    def __init__(
        self,
        *,
        moves: Any = None,
        mode: str = "greedy",
        search_depth: int = 4,
        search_states: int = 64,
        block_pred: Any = None,
        optimize_rest: bool = True,
        verify_tol: float = 1e-4,
        joint_max_iterations: int = 8,
        joint_max_enodes: int = 50_000,
        symmetry_budget: int | None = 512,
    ) -> None:
        """Store the strategy configuration."""
        if mode not in ("greedy", "search"):
            raise ValueError(
                f"unknown diagram-search mode {mode!r} — "
                "expected 'greedy' or 'search'"
            )
        self.moves = (
            tuple(moves) if moves is not None else DEFAULT_MOVES
        )
        self.mode = mode
        self.search_depth = int(search_depth)
        self.search_states = int(search_states)
        self.block_pred = block_pred
        self.optimize_rest = optimize_rest
        self.verify_tol = verify_tol
        self.joint_max_iterations = joint_max_iterations
        self.joint_max_enodes = joint_max_enodes
        self.symmetry_budget = symmetry_budget

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Lift, move, reify, graft — end to end through the ports."""
        mod, stats = _optimize_diagram(
            model,
            x,
            optimizer=optimizer,
            moves=self.moves,
            mode=self.mode,
            search_depth=self.search_depth,
            search_states=self.search_states,
            block_pred=self.block_pred,
            optimize_rest=self.optimize_rest,
            verify_tol=self.verify_tol,
            joint_max_iterations=self.joint_max_iterations,
            joint_max_enodes=self.joint_max_enodes,
            symmetry_budget=self.symmetry_budget,
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


def _optimize_diagram(
    model: Any,
    x: Any,
    *,
    optimizer: Any,
    moves: tuple,
    mode: str,
    search_depth: int,
    search_states: int,
    block_pred: Any,
    optimize_rest: bool,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
    error_budget: float | None = None,
    detect_specials: bool = False,
    detect_factors: bool = False,
    cost_fn: CostFn | None = None,
    rules: Any = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verbose: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Lift to a diagram, run the move set per ``mode``, recompose.

    ``"greedy"`` claims candidates widest-first on the initial lift;
    ``"search"`` explores orderings over diagram states
    (:func:`catopt_orchestrator.diagram_search.search_moves`).  Either
    way a declined move leaves its nodes for the per-block fallback.
    Returns ``(model, stats)`` — ``stats["moves"]`` carries the
    per-candidate verdicts, ``stats["edges"]`` the hyperedge
    topology, ``stats["diagram_search"]`` the mode's cost/exploration
    record, and ``stats["end_to_end"]`` the whole-model equivalence
    check.
    """
    t_start = time.time()
    composer = optimizer.composer
    if composer is None:
        raise TypeError(
            "ContractionSearch needs a Composer port — pass composer= "
            "(or backend=) to Optimizer"
        )
    source = optimizer.source
    sink = optimizer.sink
    fallback_cost = M._fallback_cost_kw(cost_fn)
    if cost_fn is None:
        cost_fn = launch_aware_cost

    diagram = lift_diagram(
        model,
        x,
        source=source,
        composer=composer,
        block_pred=block_pred,
    )
    graph = diagram.graph
    if mode == "search":
        # Lazy sibling import — the same convention ``_rest_block``
        # uses for ``optimize``; keeps the modules acyclic.
        from catopt_orchestrator.diagram_search import search_moves

        (
            replacements,
            move_stats,
            consumed,
            consumed_by,
            move_fires,
            dsearch,
        ) = search_moves(
            moves,
            diagram,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            joint_max_iterations=joint_max_iterations,
            joint_max_enodes=joint_max_enodes,
            symmetry_budget=symmetry_budget,
            max_depth=search_depth,
            max_states=search_states,
            verbose=verbose,
        )
    else:
        (
            replacements,
            move_stats,
            consumed,
            consumed_by,
            move_fires,
            dsearch,
        ) = _apply_moves(
            moves,
            diagram,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            joint_max_iterations=joint_max_iterations,
            joint_max_enodes=joint_max_enodes,
            symmetry_budget=symmetry_budget,
            verbose=verbose,
        )

    block_reports: dict[str, dict[str, Any]] = {}
    if optimize_rest:
        for node in graph.nodes:
            rep = _rest_block(
                node.name,
                graph,
                optimizer=optimizer,
                sink=sink,
                consumed=consumed,
                consumed_by=consumed_by,
                rules=rules,
                fallback_cost=fallback_cost,
                verify_tol=verify_tol,
                max_iterations=max_iterations,
                max_enodes=max_enodes,
                max_memory_mb=max_memory_mb,
                error_budget=error_budget,
                detect_specials=detect_specials,
                detect_factors=detect_factors,
                verbose=verbose,
            )
            block_reports[node.name] = rep
            rep_mod = rep.pop("_module", None)
            if rep_mod is not None:
                replacements[node.name] = rep_mod

    new_model, in_place = _deliver(model, composer, replacements)
    stats = _diagram_stats(
        diagram,
        move_stats,
        move_fires,
        consumed,
        block_reports,
        in_place,
    )
    stats["diagram_search"] = dsearch
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


def _node_cost(graph: MorphismGraph, name: str, cost_fn: Any) -> float:
    """Price one node's current body — ``dag_cost`` over its root."""
    rec = graph.record(name)
    return dag_cost(rec.ir.root, cost_fn) if rec.ir is not None else 0.0


def _body_costs(graph: MorphismGraph, cost_fn: Any) -> dict[str, float]:
    """Per-node reified-body costs — the state price both modes share."""
    return {
        n.name: _node_cost(graph, n.name, cost_fn) for n in graph.nodes
    }


def _graft_contrib(
    contrib: dict[str, float], res: dict[str, Any], cost_fn: Any
) -> None:
    """Update per-node body costs after a grafted move, in place.

    Nodes the move re-expresses (``_terms``) are repriced on their new
    roots; a graft without ``_terms`` (a foreign ``DiagramMove``) is
    not term-representable — its ``cost_after`` is charged to the
    first rep and the rest booked at zero (the filler convention the
    built-in moves use), or the old costs stand when the result
    reports no number at all.  Heuristic only — it ranks search
    states, never gates correctness.
    """
    terms = res.get("_terms") or {}
    for name in res["reps"]:
        term = terms.get(name)
        if term is not None:
            contrib[name] = dag_cost(term[0], cost_fn)
    unpriced = [n for n in res["reps"] if n not in terms]
    if unpriced:
        after = res.get("cost_after")
        if isinstance(after, (int, float)):
            contrib[unpriced[0]] = float(after)
            for n in unpriced[1:]:
                contrib[n] = 0.0


def _apply_moves(
    moves: tuple,
    diagram: Diagram,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
    verbose: bool,
) -> tuple[dict, dict, set, dict, dict, dict]:
    """Collect candidates and claim them greedily, widest first.

    A grafted move consumes its nodes — overlapping candidates skip;
    a declined move leaves its nodes for later candidates (and
    ultimately for the per-block fallback).  Reify exceptions become
    honest ``error`` declines — never silent.

    The last return is the ``diagram_search`` stats record: the
    candidate count and the reified-body cost before/after the greedy
    pass, on the same per-node ``dag_cost`` measure the
    ``mode="search"`` driver uses.
    """
    cands: list[tuple[DiagramMove, MorphismMatch]] = [
        (mv, m) for mv in moves for m in mv.candidates(diagram)
    ]
    # Widest claims first, law name breaking ties.
    cands.sort(key=lambda cm: (-len(cm[1].nodes), cm[1].law))
    if verbose:
        log.info(
            "[Diagram] %s; candidates: %s",
            diagram,
            [(m.law, m.nodes) for _, m in cands],
        )

    contrib = _body_costs(diagram.graph, cost_fn)
    initial_cost = sum(contrib.values())
    replacements: dict[str, Any] = {}
    move_stats: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    consumed_by: dict[str, str] = {}
    move_fires: dict[str, int] = {}
    for k, (mv, m) in enumerate(cands):
        key = f"{m.law}:{'+'.join(m.nodes)}#{k}"
        res, entry = _one_move(
            mv,
            m,
            diagram,
            consumed,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            joint_max_iterations=joint_max_iterations,
            joint_max_enodes=joint_max_enodes,
            symmetry_budget=symmetry_budget,
        )
        move_stats[key] = entry
        if res is None or res["status"] != "grafted":
            continue
        replacements.update(res["reps"])
        _graft_contrib(contrib, res, cost_fn)
        for n in m.nodes:
            consumed.add(n)
            consumed_by[n] = m.law
        move_fires[m.law] = move_fires.get(m.law, 0) + 1
        if verbose:
            log.info("[Diagram] %s: grafted (%s)", key, m.detail)
    dsearch = {
        "mode": "greedy",
        "candidates": len(cands),
        "moves_applied": sum(move_fires.values()),
        "initial_cost": initial_cost,
        "final_cost": sum(contrib.values()),
    }
    return (
        replacements,
        move_stats,
        consumed,
        consumed_by,
        move_fires,
        dsearch,
    )


def _deliver(
    model: Any, composer: Composer, replacements: dict
) -> tuple[Any, bool]:
    """Clone the model (param-sharing) and graft the replacements.

    A composer that cannot clone fails honest: the original model is
    returned unmodified and ``in_place`` flags it — the caller's
    stats record the skip instead of silently mutating a shared
    parameter bank.
    """
    try:
        new_model = composer.clone_sharing(model)
    except Exception:
        return model, True
    composer.graft(new_model, replacements)
    return new_model, False


def _diagram_stats(
    diagram: Diagram,
    move_stats: dict,
    move_fires: dict,
    consumed: set,
    block_reports: dict,
    in_place: bool,
) -> dict[str, Any]:
    """Assemble the run stats — the diagram topology plus verdicts."""
    graph = diagram.graph
    return {
        "contraction": True,
        "n_blocks": len(graph.nodes),
        "n_nodes": len(diagram.nodes),
        "n_edges": len(diagram.edges),
        "n_hyperedges": len(diagram.shared()),
        "n_lifted": sum(1 for n in graph.nodes if not n.opaque),
        "sigs": {
            n.name: (M._sig_dict(n.sig) if n.sig is not None else None)
            for n in graph.nodes
        },
        "wires": [(w.src, w.dst, w.kind) for w in graph.wires],
        "edges": [(e.src, e.dsts, e.wire) for e in diagram.edges],
        "moves": move_stats,
        "move_fires": move_fires,
        "n_rewritten": len(consumed),
        "blocks": block_reports,
        "in_place": in_place,
        "shared_params": not in_place,
    }


def _one_move(
    mv: DiagramMove,
    m: MorphismMatch,
    diagram: Diagram,
    consumed: set,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
) -> tuple[dict | None, dict]:
    """Reify one candidate → ``(result | None, stats entry)``.

    ``None`` when the candidate's nodes are already claimed or its
    reify raised — the stats entry carries the honest verdict either
    way.
    """
    if any(n in consumed for n in m.nodes):
        return None, {
            "status": "skipped",
            "reason": "node already rewritten",
        }
    try:
        res = mv.reify(
            m,
            diagram.graph,
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            max_iterations=joint_max_iterations,
            max_enodes=joint_max_enodes,
            symmetry_budget=symmetry_budget,
        )
    except Exception as e:
        return None, {
            "status": "declined",
            "reason": "error",
            "error": f"{type(e).__name__}: {e}",
        }
    entry = {
        "boundary": m.boundary,
        "detail": m.detail,
        **M._res_stats(res),
    }
    return res, entry


def _rest_block(
    name: str,
    graph: MorphismGraph,
    *,
    optimizer: Any,
    sink: Sink,
    consumed: set,
    consumed_by: dict,
    rules: Any,
    fallback_cost: dict,
    verify_tol: float,
    max_iterations: int,
    max_enodes: int | None,
    max_memory_mb: float | None,
    error_budget: float | None,
    detect_specials: bool,
    detect_factors: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run the per-block fallback for one unconsumed node.

    Returns the report dict; a successful optimise stashes the
    replacement module under the private ``"_module"`` key for the
    caller to pop into the graft set.
    """
    # Same local-import convention as _optimize_morphisms.
    from catopt_orchestrator.optimize import (
        _block_bound_gate,
        _bounded_gate,
        _rel_bound,
    )

    if name in consumed:
        return {"status": "rewritten", "law": consumed_by[name]}
    rec = graph.record(name)
    rep: dict[str, Any] = {}
    if not rec.args or rec.ir is None:
        rep["status"] = "skipped"
        rep["reason"] = rec.note
        return rep
    try:
        res_s = optimizer.search(
            rec.module,
            M._rest_example(rec),
            rules=rules,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            max_memory_mb=max_memory_mb,
            error_budget=error_budget,
            detect_specials=detect_specials,
            detect_factors=detect_factors,
            verbose=verbose,
            **fallback_cost,
        )
        lr = optimizer.lower(
            res_s,
            M._rest_example(rec),
            runner=IdentityRunner(),
            verify=False,
            verbose=verbose,
        )
        vr, out_s = _block_bound_gate(
            sink,
            rec.module,
            lr.module,
            rec.args,
            verify_tol,
            lr.stats,
            res_s,
        )
        rep["rel_diff"] = vr.max_rel
        if not _bounded_gate(vr, out_s):
            rep["status"] = "failed"
            rep["reason"] = (
                f"block verify failed: {vr.max_rel:.3e} "
                f"(accepted bound {_rel_bound(out_s, vr):.3e})"
            )
            return rep
        rep["_module"] = lr.module
        rep["status"] = "optimized"
        rep["stats"] = lr.stats
        return rep
    except Exception as e:
        rep["status"] = "failed"
        rep["error"] = f"{type(e).__name__}: {e}"
        return rep


def optimize_diagram(
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
    """Run the contraction pipeline on *model* — the function entry point.

    Assembles an :class:`~catopt_orchestrator.optimize.Optimizer` from
    the ports (``backend`` bundle or explicit
    ``source``/``sink``/``composer``/``meter``) and runs ``optimize``
    under :class:`ContractionSearch`.  ``strategy`` overrides the
    search configuration; ``**kw`` (``cost_fn``, ``rules``,
    ``verbose``, …) reach the move reifies and the per-block
    fallback.
    """
    from catopt_orchestrator.optimize import Optimizer

    opt = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        composer=composer,
        meter=meter,
    )
    strat = strategy if strategy is not None else ContractionSearch()
    return opt.optimize(model, x, strategy=strat, **kw)
