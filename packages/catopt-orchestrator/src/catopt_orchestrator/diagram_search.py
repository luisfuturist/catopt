"""Diagram-state search — plan 0013, stage 3.

Stage 2's driver applies the contraction moves greedily on the
*initial* lift: candidates are enumerated once, the widest claims
win, and a grafted move only removes its nodes from further play.
That misses the moves' real interaction — a graft rewrites its
members' bodies, and the rewritten bodies change what candidacy sees
(a leaf split materialises plain ``Param``s the pairing pass can
fuse; a window fold leaves fused terms a family law can factor
further).  The moves are semi-commutative: ``factor_shared`` after
``merge_projs`` reaches a different composition than the reverse, so
the search must explore *orderings*, not single applications.

A *diagram state* is the per-block body assignment: the same node
set and the same wires (the delivered move set rewrites bodies, not
topology), each block's IR replaced by the term its last graft
verified.  A move maps state to state: candidates are read on the
state's rebuilt :class:`~catopt_orchestrator.diagram.Diagram`, the
reify runs against the state's *current* bodies — so its cost gate
and fp64 verify measure the composition honestly — and a grafted
result's ``_terms`` write the successor's bodies back.  A graft
without ``_terms`` cannot be re-represented: its nodes are *sealed*
(blanked to ``ir=None`` — an honest "body not representable" — and
excluded from further candidacy by the sealed prefilter).

Why best-first + exact merging rather than an e-graph or a beam:

* State equality is cheap and exact: terms are hash-consed
  (:meth:`~catopt_core.ir.Op.make` interns), so a state key is a
  tuple of interned roots plus leaf names — O(#blocks) hashing, no
  canonicalisation.  Two orderings reaching the same body map are
  literally the same state; the ``seen`` map unions them — the
  and-or-graph reading of the plan's "e-graph over diagram states",
  where the shared unit is the whole state, not an e-node.
* The morphism layer keeps the space polynomial: branching is the
  per-state candidate count (a handful), and productive chains are
  short (every real move's own gate demands strict local
  improvement).  A beam would approximate what the ``seen`` map
  already does exactly — so the bounds are ``max_depth`` (composed-
  move chain length) and ``max_states`` (expansions), not a width
  cap.
* Cost is reified-term cost, not a diagram heuristic: a state is
  priced by ``Σ dag_cost(body root)`` over its current per-node
  terms — exactly the terms each graft verified and the sink lowers.

Every delivered transition still passes its move's certified reify,
and the whole-model ``end_to_end`` verify at delivery gates the
composition itself.  Greedy remains the default
:class:`~catopt_orchestrator.diagram.ContractionSearch` mode until
stage 4's verification work hardens the composed path.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from itertools import count
from typing import Any

from catopt_core.ir import IR
from catopt_core.ports import CostFn, Sink

import catopt_orchestrator.morphisms as M
from catopt_orchestrator.diagram import (
    Diagram,
    DiagramMove,
    _body_costs,
    _graft_contrib,
    _one_move,
    diagram_of_graph,
)
from catopt_orchestrator.morphisms import MorphismMatch

log = logging.getLogger("catopt_orchestrator.diagram_search")


@dataclass
class _DState:
    """One searched diagram state — current bodies + delivery data.

    ``graph``/``diagram`` are the state objects the moves read;
    ``contrib`` is the per-node body cost under the search's
    ``cost_fn``; ``reps`` is the accumulated delivery map (the latest
    graft per node wins — each successive rep was verified against
    the body current at its own transition); ``sealed`` names nodes
    whose grafts could not be re-expressed as terms; ``applied`` /
    ``entries`` / ``consumed_by`` are the path record for stats;
    ``cost`` is the state's total reified cost; ``key`` is the
    equality fingerprint.
    """

    graph: M.MorphismGraph
    diagram: Diagram
    reps: dict[str, Any]
    sealed: frozenset[str]
    contrib: dict[str, float]
    applied: tuple[tuple[str, tuple[str, ...]], ...]
    consumed_by: dict[str, str]
    entries: dict[str, dict[str, Any]]
    cost: float
    key: tuple


def _state_key(graph: M.MorphismGraph) -> tuple:
    """Fingerprint a state: per-node interned root + leaf names.

    Interned terms make this exact and cheap — identical body maps
    hash identically regardless of the path that produced them, so
    the search's ``seen`` map unions identical reachable states (the
    and-or-graph merge).  A sealed node carries ``ir=None`` — its
    stale body is never shown to candidacy.
    """
    key = []
    for n in graph.nodes:
        rec = graph.record(n.name)
        root = rec.ir.root if rec.ir is not None else None
        key.append((n.name, root, tuple(sorted(rec.leaves))))
    return tuple(key)


def _initial_state(diagram: Diagram, cost_fn: CostFn) -> _DState:
    """Build the seed state — the lift's original bodies, priced."""
    graph = diagram.graph
    contrib = _body_costs(graph, cost_fn)
    return _DState(
        graph=graph,
        diagram=diagram,
        reps={},
        sealed=frozenset(),
        contrib=contrib,
        applied=(),
        consumed_by={},
        entries={},
        cost=sum(contrib.values()),
        key=_state_key(graph),
    )


def _candidate_key(mv: Any, m: Any, st: _DState) -> tuple:
    """Fingerprint a ``(move, candidate, state)`` reify for caching.

    A reify reads only the member records' bodies and leaf tables, so
    identical member roots + leaf objects give an identical result —
    reusing it across states is sound.  ``id(mv)`` keeps
    differently-configured move instances apart; ``id`` on leaf
    values is deliberately conservative (a materialised leaf is a
    fresh object, so a changed value always misses).
    """
    spec = getattr(m, "reify", None)
    parts: list[Any] = [
        id(mv),
        m.law,
        m.nodes,
        m.boundary,
        getattr(spec, "mode", ""),
        getattr(spec, "rules", ""),
        getattr(spec, "distribute", False),
        getattr(spec, "share", False),
        tuple(getattr(spec, "kinds", ())),
        repr(getattr(spec, "extra", None)),
    ]
    for n in m.nodes:
        rec = st.graph._records.get(n)
        if rec is None or rec.ir is None:
            parts.append((n, None))
            continue
        parts.append(
            (
                n,
                rec.ir.root,
                tuple(
                    (k, id(v)) for k, v in sorted(rec.leaves.items())
                ),
            )
        )
    return tuple(parts)


def _advance(
    state: _DState,
    m: MorphismMatch,
    res: dict[str, Any],
    entry: dict[str, Any],
    *,
    cost_fn: CostFn,
) -> _DState:
    """Build the successor state for one grafted candidate.

    Nodes the move re-expresses get new records (reified root over
    the node's unchanged input vars, plus the move's param/leaf
    tables); rep'd nodes without ``_terms`` are sealed to ``ir=None``
    so their stale bodies stay invisible to candidacy.  The wire
    structure, signatures and capture evidence carry over — the
    delivered move set rewrites bodies, not the topology the lift
    proved.
    """
    graph = state.graph
    terms = res.get("_terms") or {}
    records = dict(graph._records)
    sealed = set(state.sealed)
    for name in res["reps"]:
        rec = records.get(name)
        if rec is None:
            continue
        term = terms.get(name)
        if term is None or rec.ir is None:
            records[name] = _dc_replace(rec, ir=None)
            sealed.add(name)
            continue
        root, params, leaves = term
        records[name] = _dc_replace(
            rec,
            ir=IR(
                root=root,
                inputs=list(rec.ir.inputs),
                input_names=set(rec.ir.input_names),
                params=dict(params),
            ),
            leaves=dict(leaves),
        )
    contrib = dict(state.contrib)
    _graft_contrib(contrib, res, cost_fn)
    g2 = M.MorphismGraph(list(graph.nodes), list(graph.wires), records)
    # Model-level evidence is topology-fixed — carry it verbatim.
    g2._model_outs = graph._model_outs
    g2._model_out_objs = graph._model_out_objs
    g2._model_outs2 = graph._model_outs2
    g2._probe2 = graph._probe2
    dg2 = diagram_of_graph(g2)
    key = f"{m.law}:{'+'.join(m.nodes)}#{len(state.applied)}"
    return _DState(
        graph=g2,
        diagram=dg2,
        reps={**state.reps, **res["reps"]},
        sealed=frozenset(sealed),
        contrib=contrib,
        applied=(*state.applied, (m.law, m.nodes)),
        consumed_by={
            **state.consumed_by,
            **{n: m.law for n in res["reps"]},
        },
        entries={**state.entries, key: entry},
        cost=sum(contrib.values()),
        key=_state_key(g2),
    )


def _expand_one(
    mv: DiagramMove,
    m: MorphismMatch,
    st: _DState,
    cache: dict,
    stats: dict,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
) -> _DState | None:
    """Try one candidate on *st*; return the successor or ``None``.

    Candidates touching a sealed node are skipped outright (their
    stale bodies are not candidacy material); everything else goes
    through the move's certified reify — declines and errors simply
    produce no transition.
    """
    if any(n in st.sealed for n in m.nodes):
        stats["moves_sealed"] += 1
        return None
    stats["moves_tried"] += 1
    ckey = _candidate_key(mv, m, st)
    got = cache.get(ckey)
    if got is None:
        got = _one_move(
            mv,
            m,
            st.diagram,
            set(st.sealed),
            sink=sink,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            joint_max_iterations=joint_max_iterations,
            joint_max_enodes=joint_max_enodes,
            symmetry_budget=symmetry_budget,
        )
        cache[ckey] = got
    else:
        stats["cache_hits"] += 1
    res, entry = got
    if res is None or res.get("status") != "grafted":
        stats["moves_declined"] += 1
        return None
    return _advance(st, m, res, entry, cost_fn=cost_fn)


def _expand_state(
    st: _DState,
    moves: tuple[DiagramMove, ...],
    cache: dict,
    stats: dict,
    **kw: Any,
) -> Iterator[_DState]:
    """Yield the successors of one state — every legal transition.

    Re-enumerates candidacy on the state's diagram for each move, so
    orderings — not just single applications — are explored.
    """
    for mv in moves:
        for m in mv.candidates(st.diagram):
            succ = _expand_one(mv, m, st, cache, stats, **kw)
            if succ is not None:
                yield succ


def search_moves(
    moves: tuple[DiagramMove, ...],
    diagram: Diagram,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    joint_max_iterations: int,
    joint_max_enodes: int,
    symmetry_budget: int | None,
    max_depth: int,
    max_states: int,
    verbose: bool,
) -> tuple[dict, dict, set, dict, dict, dict]:
    """Best-first search over move-application states.

    Every frontier state expands all its candidates — so orderings,
    not just single applications, are explored — and identical
    reachable states merge on the interned-term key.  Returns the
    same tuple as the greedy ``_apply_moves`` plus the
    ``diagram_search`` stats record.
    """
    seed = _initial_state(diagram, cost_fn)
    best = seed
    seen = {seed.key}
    frontier: list = [(seed.cost, 0, seed)]
    seq = count(1)
    cache: dict[tuple, tuple] = {}
    stats: dict[str, Any] = {
        "mode": "search",
        "initial_cost": seed.cost,
        "states_explored": 0,
        "states_generated": 0,
        "states_merged": 0,
        "moves_tried": 0,
        "moves_declined": 0,
        "moves_sealed": 0,
        "cache_hits": 0,
        "bound_depth": max_depth,
        "bound_states": max_states,
        "bound_hit": False,
    }
    kw: dict[str, Any] = dict(
        sink=sink,
        cost_fn=cost_fn,
        verify_tol=verify_tol,
        joint_max_iterations=joint_max_iterations,
        joint_max_enodes=joint_max_enodes,
        symmetry_budget=symmetry_budget,
    )
    while frontier:
        if stats["states_explored"] >= max_states:
            stats["bound_hit"] = True
            break
        _, _, st = heapq.heappop(frontier)
        stats["states_explored"] += 1
        if len(st.applied) >= max_depth:
            continue
        for succ in _expand_state(st, moves, cache, stats, **kw):
            if succ.key in seen:
                stats["states_merged"] += 1
                continue
            seen.add(succ.key)
            stats["states_generated"] += 1
            if succ.cost < best.cost:
                best = succ
            heapq.heappush(frontier, (succ.cost, next(seq), succ))
            if verbose:
                last = succ.applied[-1]
                log.info(
                    "[DiagramSearch] depth=%d %s:%s -> cost %.1f",
                    len(succ.applied),
                    last[0],
                    "+".join(last[1]),
                    succ.cost,
                )
    stats["best_cost"] = best.cost
    stats["moves_applied"] = len(best.applied)
    stats["best_path"] = [
        f"{law}:{'+'.join(nodes)}" for law, nodes in best.applied
    ]
    move_fires: dict[str, int] = {}
    for law, _nodes in best.applied:
        move_fires[law] = move_fires.get(law, 0) + 1
    return (
        best.reps,
        best.entries,
        set(best.reps),
        best.consumed_by,
        move_fires,
        stats,
    )
