"""The search environment — one program, one episode (torch-free).

Plan 0016 stage 7, the RL half (ADR 0003).  An episode starts from a
program; each step applies one rule to its e-graph, and the reward is
the normalized cost improvement that step produced.  The environment
never decides equivalence — the laws do — so a policy trained on it
can only learn to *choose*, never to change the answer.

Torch-free: this is the engine side of the RL loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.features import ProgramFeatures, compute_features

__all__ = ["SearchEnv", "StepResult"]

#: Cost improvements smaller than this are treated as no improvement.
_EPS = 1e-12


@dataclass(frozen=True)
class StepResult:
    """One transition: the new state, its reward, and whether it ended."""

    features: ProgramFeatures
    reward: float
    done: bool
    cost: float


class SearchEnv:
    """A single-program search episode.

    ``horizon`` bounds the episode length; ``patience`` ends it early
    once that many consecutive steps fail to improve the cost.
    """

    def __init__(
        self,
        term: Any,
        rules: Any,
        cost_fn: Any = None,
        horizon: int = 6,
        patience: int = 2,
    ) -> None:
        """Bind the program, the rule set, and the episode bounds."""
        self.term = term
        self.rules = tuple(rules)
        self.by_name = {r.name: r for r in self.rules}
        self.cost_fn = cost_fn if cost_fn is not None else flops_cost
        self.horizon = horizon
        self.patience = patience
        self.eg: Any = None
        self.root = 0
        self._t = 0
        self._stall = 0
        self._cost = float("inf")

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return the legal rule names — the action set."""
        return tuple(r.name for r in self.rules)

    @property
    def progress(self) -> float:
        """Return the fraction of the horizon consumed."""
        return self._t / self.horizon

    @property
    def cost(self) -> float:
        """Return the current cheapest extracted cost."""
        return self._cost

    def _extract(self) -> Any:
        return self.eg.extract_best(self.root, self.cost_fn)

    def _price(self) -> float:
        term = self._extract()
        if term is None:
            return float("inf")
        return float(dag_cost(term, self.cost_fn))

    def state(self) -> ProgramFeatures:
        """Return the static features of the current cheapest program."""
        return compute_features(self._extract())

    def reset(self) -> ProgramFeatures:
        """Start an episode; return the initial state."""
        self.eg = EGraph()
        self.root = self.eg.add_term(self.term)
        self._t = 0
        self._stall = 0
        self._cost = self._price()
        return self.state()

    def step(self, rule_name: str) -> StepResult:
        """Apply ``rule_name``; return the resulting transition."""
        if self.eg is None:
            raise RuntimeError("reset() before step()")
        before = self._cost
        self.eg.apply_rule(self.by_name[rule_name], self.root)
        after = self._price()
        improved = after < before - _EPS
        self._stall = 0 if improved else self._stall + 1
        self._t += 1
        reward = 0.0
        if before not in (0.0, float("inf")):
            reward = (before - after) / before
        self._cost = after
        done = self._t >= self.horizon or self._stall >= self.patience
        return StepResult(self.state(), reward, done, after)
