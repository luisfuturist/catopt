"""The search/lower seam — the result objects both phases share.

The pipeline's two phases are separate verbs (plan 0006):

* ``search : (Source + Capabilities) -> IR -> EGraph -> extracted term``
* ``lower  : (Sink + Runner)         -> term -> runnable -> verified``

:func:`catopt_optimize.optimize.search` produces a
:class:`SearchResult`; :func:`catopt_optimize.optimize.lower` consumes
one and returns a :class:`LowerResult`.  The *objects* live in
``catopt_core`` — torch-free, backend-neutral — because they are part
of the hexagonal contract, not of any one orchestrator: a caller can
hold a :class:`SearchResult`, inspect its saturated e-graph,
enumerate its frontier, prove an equivalence with
:meth:`SearchResult.certificate`, and lower it under several runners
without ever re-running the search.

Port needs by phase
-------------------
* :class:`SearchResult` is produced by a :class:`~catopt_core.ports.Source`
  (``model -> (IR, leaves)``) plus, optionally, a
  :class:`~catopt_core.ports.Capabilities` (backend-relative pricing
  and the const-fold registry).  ``model`` is the original model —
  kept as provenance; ``lower`` verifies against a fresh lowering of
  ``ir`` instead, so the reference is a runnable in the SINK's runtime
  (which the source-side model need not be).
* :class:`LowerResult` is produced by a :class:`~catopt_core.ports.Sink`
  plus a delivery runner.  ``stats`` is a FRESH dict — the search
  record plus the lowering keys; ``lower`` never mutates
  :attr:`SearchResult.stats`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catopt_core.egraph import Certificate, EGraph
    from catopt_core.ir import IR
    from catopt_core.ports import CostFn, Executor, VerifyResult

__all__ = ["LowerResult", "SearchResult"]


@dataclass(eq=False)
class SearchResult:
    """The record of a ``search`` run — inspectable mid-pipeline.

    ``ir`` / ``param_values`` are the source's export (the program and
    its concrete leaf values); ``eg`` / ``root_eid`` are the saturated
    e-graph and the root e-class; ``term`` is the member the search
    extracted under ``cost_fn``; ``stats`` is the *search* record
    (``eg.run`` stats plus ``rule_fires``, ``criteria``,
    ``pairing_groups`` / ``nonlocal_lifts`` / ``paired_extract`` /
    ``causal_specialized`` when they fired).  ``source`` and ``model``
    are provenance — which port produced the IR, and which model the
    result optimizes.

    ``eq=False``: identity semantics — a result is a handle into one
    concrete e-graph, not a value.
    """

    ir: IR
    eg: EGraph
    root_eid: int
    term: Any
    param_values: Mapping[str, Any]
    stats: dict[str, Any]
    cost_fn: CostFn | None = None
    source: Any = None
    model: Any = None

    def alternatives(
        self, top_k: int = 8, cost_fn: CostFn | None = None
    ) -> list[tuple[float, Any]]:
        """Enumerate the cheapest distinct members of the root class.

        Delegates to :meth:`EGraph.extract_alternatives` on
        ``self.eg``: for each non-leaf root enode, force extraction
        through it and record the DAG cost.  Returns ``(cost, term)``
        pairs, cheapest first — the discovery-engine frontier.

        ``cost_fn`` defaults to the pricing the search used (the
        backend-relative wrapped model when ``capabilities`` priced
        it); a result carrying no ``cost_fn`` falls back to
        :func:`~catopt_core.cost.flops_cost`.
        """
        cf = cost_fn if cost_fn is not None else self.cost_fn
        if cf is None:
            from catopt_core.cost import flops_cost

            cf = flops_cost
        return self.eg.extract_alternatives(self.root_eid, cf, top_k)

    def certificate(self, a: Any, b: Any = None) -> Certificate:
        """Build a proof-carrying derivation ``a`` -> ``b``.

        Delegates to :meth:`EGraph.certificate` pinned to this
        result's ``root_eid`` and ``cost_fn`` — the natural call is
        ``res.certificate(res.ir.root, res.term)``: the level-2
        certificate that the extracted term is provably equivalent to
        the exported program.  ``b=None`` extracts under ``cost_fn``.
        """
        return self.eg.certificate(
            a, b, root_eid=self.root_eid, cost_fn=self.cost_fn
        )


@dataclass(eq=False)
class LowerResult:
    """The record of a ``lower`` run — the delivered module.

    ``module`` is the runnable executor (a carrier-batched module when
    the extracted root plans one, the sink's generic lowering
    otherwise, then whatever the delivery runner wrapped it in).
    ``stats`` is a fresh dict: the *search* record plus the *lower*
    record (``lowering``, ``runner``, and whatever the runner wrote —
    ``compiled``, ``cuda_graph``); :attr:`SearchResult.stats` is never
    mutated.  ``verified`` is the sink's equivalence report, or
    ``None`` when ``lower`` ran with ``verify=False``.

    Iterates as ``(module, stats)`` so ``mod, stats = lower(...)``
    mirrors the legacy tuple return — ``eq=False`` keeps the result a
    handle, not a value.
    """

    module: Executor
    stats: dict[str, Any]
    verified: VerifyResult | None = None

    def __iter__(self) -> Iterator[Any]:
        """Unpack to ``(module, stats)`` — the legacy return shape."""
        yield self.module
        yield self.stats
