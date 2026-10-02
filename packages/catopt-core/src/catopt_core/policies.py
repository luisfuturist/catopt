"""Search policies — the game's players (ADR 0003, plan 0016 stage 6).

A policy only *orders* the legal actions the law library already
licenses, so it can never change what is certified.  The non-ML
policies land first; a learned policy is one more conforming value
(stage 7).

A true beam search needs to clone e-graph state per branch, which the
engine does not support today — so ``BeamPolicy`` here is a single-step
*scoring* policy (argmax over a scorer), documented as such rather than
pretending to look ahead.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from catopt_core.game import Action

__all__ = [
    "BeamPolicy",
    "ExistingPolicy",
    "GreedyPolicy",
    "RandomPolicy",
]

#: A scorer ranks one action at one state; higher is better.
Scorer = Callable[[Any, Action], float]


class RandomPolicy:
    """Pick a legal action uniformly at random (seeded)."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        """Seed the generator (``None`` for system entropy)."""
        self._rng = random.Random(seed)

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return a uniformly random action."""
        return self._rng.choice(list(actions))


class ExistingPolicy:
    """Keep the engine's own order — the identity policy."""

    name = "existing"

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return the first action (declaration order)."""
        return next(iter(actions))


class GreedyPolicy:
    """Pick the highest-priority legal action, deterministically.

    Priority comes from ``priority_of`` (rule name -> number, lower is
    earlier); rules absent from the map default to ``0.0``.  Ties keep
    declaration order (``min`` is stable).
    """

    name = "greedy"

    def __init__(self, priority_of: Any = None) -> None:
        """Fix the priority map (empty by default)."""
        self.priority_of = priority_of or {}

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return the lowest-priority-number action."""
        return min(
            actions, key=lambda a: self.priority_of.get(a.rule, 0.0)
        )


class BeamPolicy:
    """Pick the action a ``scorer`` ranks highest (single step).

    ``scorer(state, action) -> float``, higher is better.  The default
    scorer ranks by rule declaration order, so the behaviour matches
    :class:`ExistingPolicy` until a real scorer (e.g. a learned value
    model) is supplied.  Named for its intended use as the beam's
    expansion oracle; it does not itself look ahead.
    """

    name = "beam"

    def __init__(self, scorer: Scorer | None = None) -> None:
        """Fix the scorer (defaults to declaration order)."""
        self.scorer = scorer

    def score(self, state: Any, action: Action, index: int) -> float:
        """Score ``action`` — the scorer's value, or ``-index``."""
        if self.scorer is None:
            return -float(index)
        return float(self.scorer(state, action))

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return the highest-scoring action."""
        acts = list(actions)
        best_i = 0
        best_s: float | None = None
        for i, a in enumerate(acts):
            s = self.score(state, a, i)
            if best_s is None or s > best_s:
                best_s, best_i = s, i
        return acts[best_i]
