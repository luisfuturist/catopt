"""The ``Engine``-conforming adapter over the Rust search core.

``catopt_native._native.NativeEGraph`` is the compiled half — union-
find, enode storage, compiled matching, rebuild/congruence and the
dirty-frontier saturation loop, all a faithful port of
``catopt_core.egraph.core.EGraph``'s *search* surface.  This module is
the thin Python half:

* :func:`_flatten` / :func:`_ser_pattern` — the serialization layer
  (terms and patterns cross the FFI as plain tuples);
* :func:`_build_term` — the ``build_term`` callback the Rust side
  calls to materialise a bound metavariable's resolved member (so
  ``check``/``derive`` see real ``Op``/leaf objects, exactly like the
  Python engine);
* :class:`NativeEngine` — the :class:`~catopt_core.ports.Engine` the
  pipeline consumes: same ``add_term`` / ``find`` / ``run`` /
  ``extract_best`` surface, plus a lazily materialised ``_classes``
  view that lets the *reference* extraction mixin
  (:class:`catopt_core.egraph.extract._ExtractMixin`) run unmodified —
  extraction semantics are the Python implementation's *by
  construction*, so the extracted term and cost are identical.

Scope boundary (plan 0010, lever 3): no proof machinery — ``union``
accepts and drops ``rule``/``subst``/``witness``/``note``; there is no
``merge_log``/``applications``/``certificate``.  A run that needs a
certificate uses the Python engine.
"""

from __future__ import annotations

import sys
from typing import Any

from catopt_core.egraph.extract import _ExtractMixin
from catopt_core.egraph.types import (
    EClass,
    ENode,
    _LeafRegistry,
    _norm_attr_value,
)
from catopt_core.ir import Op

from catopt_native._native import NativeEGraph as _RustEGraph

__all__ = ["NativeEngine"]


def _flatten(term: Any, nodes: list, memo: dict) -> int:
    """Append *term*'s DAG to *nodes* post-order; return its index.

    Shared subtrees intern once through *memo* — exported IR terms are
    DAGs with heavy sharing, so walking without it re-expands shared
    cones exponentially.
    """
    if term in memo:
        return memo[term]
    if isinstance(term, Op):
        idxs = tuple(_flatten(a, nodes, memo) for a in term.args)
        attrs = tuple(
            sorted(
                (k, _norm_attr_value(v)) for k, v in term.attrs.items()
            )
        )
        nodes.append(("op", term.op, idxs, attrs))
    else:
        _LeafRegistry.register(term)
        nodes.append(("leaf", repr(term)))
    memo[term] = len(nodes) - 1
    return memo[term]


def _ser_pattern(p: Any) -> tuple:
    """Serialize a pattern subtree (metavariables → ``("var", n)``)."""
    if isinstance(p, str):
        return ("var", p)
    if isinstance(p, Op):
        return (
            "op",
            p.op,
            tuple(_ser_pattern(a) for a in p.args),
            tuple(
                (k, _norm_attr_value(v))
                for k, v in sorted(p.attrs.items())
            ),
        )
    _LeafRegistry.register(p)
    return ("leaf", repr(p))


def _build_term(ser: Any) -> Any:
    """Materialise a serialized term — the Rust ``build_term`` hook."""
    if ser[0] == "leaf":
        return _LeafRegistry.decode(ser[1])
    _tag, op, children, attrs = ser
    return Op.make(
        op, *(_build_term(c) for c in children), **dict(attrs)
    )


class NativeEngine(_ExtractMixin):
    """The native search engine — Rust core + reference extraction.

    Satisfies :class:`catopt_core.ports.Engine` and (except proof
    machinery) the ``EGraph`` surface ``optimize.search`` consumes:
    ``add_term`` / ``find`` / ``rebuild`` / ``run`` / ``extract_best``
    / ``rule_fires`` / ``n_enodes`` / ``n_classes`` / ``_classes``.

    ``engine_name = "native"`` is what ``stats["engine"]`` records.
    """

    engine_name = "native"
    #: Honest reporting: the native core stores no proof witnesses.
    truncation_level = 1

    def __init__(self) -> None:
        """Create the Rust core with the ``build_term`` hook."""
        self._g = _RustEGraph(_build_term)
        self._view: dict[int, EClass] | None = None

    # -- internals -----------------------------------------------------

    def _invalidate(self) -> None:
        """Drop the materialised ``_classes`` view after a mutation."""
        self._view = None

    # -- engine surface --------------------------------------------------

    @property
    def n_enodes(self) -> int:
        """Return the number of enodes."""
        return self._g.n_enodes

    @property
    def n_classes(self) -> int:
        """Return the number of e-classes."""
        return self._g.n_classes

    @property
    def rule_fires(self) -> dict[str, int]:
        """Return ``{rule_name: merge_count}``."""
        return dict(self._g.rule_fires())

    def certificate(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn: Any = None,
    ) -> Any:
        """Certificates are a Python-engine capability — raise.

        The native core records no merge log / applications
        (``truncation_level = 1``), so there is nothing to certify.
        A run needing a certificate uses the Python engine.
        """
        raise NotImplementedError(
            "certificates require the Python EGraph engine — "
            "the native core records no proof witnesses"
        )

    def find(self, eid: int) -> int:
        """Return the canonical e-class id of ``eid``."""
        return self._g.find(eid)

    def get_class(self, eid: int) -> EClass:
        """Return the e-class containing ``eid`` (from the view)."""
        return self._classes[self.find(eid)]

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        """Add a term (Var/Const/Param/Op); return its e-class id."""
        nodes: list = []
        _flatten(term, nodes, {} if _memo is None else _memo)
        self._invalidate()
        return self._g.add_term(nodes)

    def add_leaf(self, key: str, provenance: str | None = None) -> int:
        """Add a leaf by registry key."""
        self._invalidate()
        return self._g.add_leaf(key)

    def union(
        self,
        a: int,
        b: int,
        rule: str | None = None,
        subst: dict | None = None,
        note: str = "",
        witness: Any = None,
    ) -> bool:
        """Merge two e-classes.

        Proof metadata (``rule``/``subst``/``note``/``witness``) is
        accepted for signature parity and dropped — the native core is
        search-only; certificates stay a Python-engine concern.
        """
        self._invalidate()
        return self._g.union(a, b)

    def rebuild(self, classes: Any = None) -> bool:
        """Canonicalize children and merge duplicates."""
        self._invalidate()
        ids = None if classes is None else [int(c) for c in classes]
        return self._g.rebuild(ids)

    def _register_rules(self, rules: Any) -> list[str]:
        """Compile every rule into the Rust core; return ordered names.

        A ``priority_of`` map (a ``RuleSet``) schedules each iteration
        exactly as the Python engine does (stable sort).
        """
        ordered = list(rules)
        priority_of = getattr(rules, "priority_of", None)
        if priority_of is not None:
            ordered.sort(key=priority_of)
        for r in ordered:
            self._g.add_rule(
                r.name,
                _ser_pattern(r.lhs),
                _ser_pattern(r.rhs),
                check=r.check,
                derive=r.derive,
            )
        return [r.name for r in ordered]

    def _improving_cost(self, root_eid: int, cost_fn: Any) -> float:
        """Re-extract under the mid-run graph and price it."""
        self._invalidate()
        term = self.extract_best(root_eid, cost_fn)
        return cost_fn(term) if term is not None else float("inf")

    def _run_loop(
        self,
        names: list[str],
        root_eid: int,
        max_iterations: int,
        cap: int,
        budgets: dict | None,
        stop: str,
        patience: int,
        cost_fn: Any,
    ) -> tuple[str, int, int]:
        """Drive ``run_iteration``; return ``(stop, iterations, improved)``.

        The loop runs here rather than inside the extension: each
        ``run_iteration`` is a complete FFI call, so the "improving"
        policy can re-extract between iterations through the reference
        mixin without tripping PyO3's reentrant-borrow guard.
        """
        stop_reason = "max_iterations"
        best_cost = float("inf")
        stall = 0
        improved = 0
        iterations = 0
        for _ in range(max_iterations):
            iterations += 1
            self._invalidate()
            reason = self._g.run_iteration(
                names,
                max_nodes=cap,
                rule_budgets=budgets,
            )
            if reason != "continue":
                stop_reason = reason
                break
            if stop == "improving":
                cost = self._improving_cost(root_eid, cost_fn)
                if cost < best_cost:
                    best_cost = cost
                    stall = 0
                    improved += 1
                else:
                    stall += 1
                    if stall >= patience:
                        stop_reason = "improving"
                        break
        # Postcondition: canonicalised graph (unrestricted rebuild).
        self._invalidate()
        self._g.finish()
        return stop_reason, iterations, improved

    def _run_stats(
        self,
        budgets: dict | None,
        stop_reason: str,
        iterations: int,
        improved: int,
        improving: bool,
    ) -> dict[str, Any]:
        """Assemble the ``EGraph.run`` stats record."""
        spent = dict(self._g.budget_spent())
        budget_keys = budgets or {}
        stats: dict[str, Any] = {
            "iterations": iterations,
            "n_enodes": self.n_enodes,
            "n_classes": self.n_classes,
            # Stats-shape parity with EGraph.run — the native core
            # records no proof witnesses (Python-engine concern).
            "n_proof_edges": 0,
            "truncation_level": 1,
            "rule_budgets": {n: spent.get(n, 0) for n in budget_keys},
            "budget_suspended": [
                n
                for n, b in budget_keys.items()
                if spent.get(n, 0) >= b
            ],
            "stop": stop_reason,
        }
        if improving:
            stats["improved"] = improved
        return stats

    def run(
        self,
        rules: Any,
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: Any = None,
    ) -> dict[str, Any]:
        """Run equality saturation — mirrors ``EGraph.run``.

        ``rules`` is any ``RuleSetLike`` iterable of ``Rewrite``s;
        a ``priority_of`` map (a ``RuleSet``) schedules each iteration
        exactly as the Python engine does.  ``stop="improving"``
        extracts through the Python mixin between iterations (the
        reference extractor — identical terms and costs).
        """
        if stop not in ("fixed_point", "improving"):
            raise ValueError(
                f"stop must be 'fixed_point' or 'improving', "
                f"got {stop!r}"
            )
        if stop == "improving" and cost_fn is None:
            raise ValueError(
                "stop='improving' needs a cost_fn to extract with"
            )
        names = self._register_rules(rules)
        cap = max_nodes if max_nodes is not None else sys.maxsize
        budgets = dict(rule_budgets) if rule_budgets else None
        stop_reason, iterations, improved = self._run_loop(
            names,
            root_eid,
            max_iterations,
            cap,
            budgets,
            stop,
            patience,
            cost_fn,
        )
        return self._run_stats(
            budgets,
            stop_reason,
            iterations,
            improved,
            stop == "improving",
        )

    # -- the _classes view -------------------------------------------------

    @property
    def _classes(self) -> dict[int, EClass]:
        """Materialised ``eid -> EClass`` view for the extraction mixin.

        Rebuilt lazily and invalidated on every mutation — extraction
        always sees the post-saturation graph, and ``_ExtractMixin``
        (and ``_carrier_upgrade``, ``extract_alternatives``,
        ``extract_paired``, ``diverse_classes``) runs unmodified over
        real ``catopt_core`` ``EClass``/``ENode`` objects.
        """
        view = self._view
        if view is None:
            view = {}
            for eid, nodes in self._g.classes():
                ec = EClass(id=eid)
                ec.nodes = {
                    ENode(op, tuple(children), tuple(attrs))
                    for op, children, attrs in nodes
                }
                view[eid] = ec
            self._view = view
        return view
