"""The search game — state, action, rule book, transition, evaluator.

The formal mapping (ADR 0003, plan 0016 stage 5) onto the e-graph:

* **state** — the e-graph plus a search cursor (:class:`GameState`);
* **action** — one legal rewrite (:class:`Action`);
* **rule book** — the law library (:class:`RuleBook`);
* **transition** — applying one rewrite (:func:`transition`);
* **evaluator** — the extracted cost (:class:`Evaluator`).

The **referee is not here**: legality is enforced by the laws and the
certificate (ADR 0003).  This is a thin formalization over
:class:`catopt_core.egraph.EGraph`, not a second engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catopt_core.cost import dag_cost, flops_cost

__all__ = [
    "Action",
    "Evaluator",
    "GameState",
    "RuleBook",
    "transition",
]


@dataclass(frozen=True)
class Action:
    """One legal move: apply ``rule`` to the e-class at ``eclass``."""

    rule: str
    eclass: int = 0

    def __repr__(self) -> str:
        """Return ``rule@eclass``."""
        return f"{self.rule}@{self.eclass}"


@dataclass
class GameState:
    """The board: an e-graph plus the search cursor.

    ``features`` (a :class:`~catopt_core.features.ProgramFeatures` or
    ``None``) is the static description a learned policy reads; the
    engine itself never needs it.
    """

    eg: Any
    root_eid: int
    iteration: int = 0
    best_cost: float | None = None
    features: Any = None


class RuleBook:
    """The law library as a source of legal actions."""

    def __init__(self, rules: Any) -> None:
        """Index ``rules`` by name."""
        self.rules = tuple(rules)
        self._by_name = {r.name: r for r in self.rules}

    def actions(self, state: GameState) -> list[Action]:
        """Return one action per rule, targeting the root class."""
        return [Action(r.name, state.root_eid) for r in self.rules]

    def rule(self, name: str) -> Any:
        """Return the rule named ``name``.

        Raises :class:`KeyError` when the book has no such rule.
        """
        return self._by_name[name]


def transition(eg: Any, action: Action, rule_book: RuleBook) -> bool:
    """Apply ``action`` to ``eg``; return whether it changed the graph."""
    return bool(
        eg.apply_rule(rule_book.rule(action.rule), action.eclass)
    )


class Evaluator:
    """Score a state by the extracted cost of its cheapest member."""

    def __init__(self, cost_fn: Any = None) -> None:
        """Fix the pricing function (default ``flops_cost``)."""
        self.cost_fn = cost_fn if cost_fn is not None else flops_cost

    def evaluate(self, state: GameState) -> float:
        """Return the cost of ``state``'s cheapest extracted member."""
        term = state.eg.extract_best(state.root_eid, self.cost_fn)
        if term is None:
            return float("inf")
        return float(dag_cost(term, self.cost_fn))
