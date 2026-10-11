"""The generic optimization game — one protocol, every board.

catopt's three boards — the search environment
(:class:`catopt_core.search_env.SearchEnv`), the construction arena
(:class:`catopt_discovery.arena.Arena`) and the meta-arena
(:class:`catopt_discovery.meta_arena.MetaArena`) — share one shape:
observe a state, play a move, get a report, repeat until done.  This
module names that shape once (:class:`Board`) and puts the episode
driver (:func:`run_episode`), the arm comparison (:func:`evaluate`)
and the training loop (:func:`train_policy`) over it — the two arena
``run_episode`` functions were already copies of each other.

A new *domain* is data + a small adapter: a factory ``case ->
Board`` and a featurizer for the learned player
(:mod:`catopt_discovery.players`).  :mod:`catopt_discovery.play`
keeps the registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Board",
    "Report",
    "evaluate",
    "run_episode",
    "train_policy",
]


@runtime_checkable
class Board(Protocol):
    """The optimization-game contract every domain implements.

    ``observe`` returns the read-only state a player decides on —
    it must carry ``done`` (the episode terminator) and everything
    the featurizer reads.  ``step`` applies a move and returns
    ``(state, report)`` where *report* carries ``reward`` (the
    learning signal) and ``terminal`` (the game over).  Both arenas
    already satisfy this; a foreign environment adapts by wrapping.
    """

    def observe(self) -> Any:
        """Return the read-only state snapshot (``state.done`` ends)."""
        ...  # pragma: no cover — protocol

    def step(self, action: Any) -> tuple[Any, Any]:
        """Apply *action*; return ``(state', report)``."""
        ...  # pragma: no cover — protocol


@dataclass(frozen=True)
class Report:
    """The generic step report — what the driver and player read.

    Boards may return richer reports (both arenas do); this is the
    floor: the reward a learner trains on and the terminal flag an
    episode ends on.  ``detail`` carries domain extras.
    """

    action: Any
    reward: float
    terminal: bool = False
    applied: bool = True
    note: str = ""
    cost: float = float("nan")
    certificate_ok: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """One episode's reports in play order — the generic edition.

    ``total`` is the sum of step rewards (the REINFORCE return);
    ``terminal`` the last report when it ended the episode.
    """

    reports: list[Any]

    @property
    def total(self) -> float:
        """Sum the episode's step rewards."""
        return sum(r.reward for r in self.reports)

    @property
    def terminal(self) -> Any | None:
        """Return the terminal report, or ``None``."""
        if self.reports and self.reports[-1].terminal:
            return self.reports[-1]
        return None


def _done(state: Any) -> bool:
    """Read the terminator — ``state.done`` is the contract."""
    return bool(getattr(state, "done", False))


def run_episode(
    board: Board,
    player: Callable[[Any], Any | None],
    budget: int,
) -> Trajectory:
    """Play *player* on *board* until *budget* actions are spent.

    A player is any callable ``state -> action | None`` — ``None``
    ends the episode early, as does ``state.done``.
    """
    reports: list = []
    for _ in range(budget):
        state = board.observe()
        if _done(state):
            break
        action = player(state)
        if action is None:
            break
        _state, rep = board.step(action)
        reports.append(rep)
    return Trajectory(reports)


def train_policy(
    player: Any,
    cases: Iterable,
    board_of: Callable[[Any], Board],
    *,
    budget: int,
) -> list[float]:
    """One episode per case; the player's learner gets each return.

    *player* must expose ``finish_episode(total, rewards=...)`` —
    :class:`~catopt_discovery.players.LinearPolicy` does; any object
    with that method (or none — a frozen arm) plugs in.
    """
    totals: list[float] = []
    for case in cases:
        traj = run_episode(board_of(case), player, budget)
        fin = getattr(player, "finish_episode", None)
        if fin is not None:
            fin(traj.total, rewards=[r.reward for r in traj.reports])
        totals.append(traj.total)
    return totals


def imitate_policy(
    player: Any,
    expert: Callable[[int], Any],
    cases: Iterable,
    board_of: Callable[[Any], Board],
    *,
    budget: int,
) -> dict[str, int]:
    """Warm-start *player* by replaying *expert* episodes.

    Each case runs one episode where the fresh ``expert(index)``
    picks every action and ``player.imitate(state, action)`` takes a
    cross-entropy step toward it; the board still executes the move,
    so the trace is real (the expert acts on true states, not a
    static log).  The factory is per-case, like ``evaluate``'s arms —
    a stateful expert (a draining playbook) resets per episode.
    Returns the per-run counts of demonstrated vs. successfully
    imitated moves — a gap means the expert's actions left the
    policy's live legal set, worth knowing.
    """
    shown = imitated = 0
    for i, case in enumerate(cases):
        board = board_of(case)
        exp = expert(i)
        for _ in range(budget):
            state = board.observe()
            if _done(state):
                break
            action = exp(state)
            if action is None:
                break
            shown += 1
            if player.imitate(state, action):
                imitated += 1
            _state, _rep = board.step(action)
    return {"shown": shown, "imitated": imitated}


def evaluate(
    arms: dict[str, Callable[[int], Any]],
    cases: list,
    board_of: Callable[[Any], Board],
    *,
    budget: int,
) -> dict[str, list[dict]]:
    """Play every arm on the same cases — paired boards.

    ``arms`` maps a name to a factory ``episode_index -> player``;
    every arm sees the same *cases* in the same order, so means
    compare like for like.  Rows carry reward, step count, and the
    terminal cost/certificate when the board reports them.
    """
    out: dict[str, list[dict]] = {}
    for name, mk in arms.items():
        rows: list[dict] = []
        for i, case in enumerate(cases):
            traj = run_episode(board_of(case), mk(i), budget)
            t = traj.terminal
            rows.append(
                {
                    "case": case[0]
                    if isinstance(case, (list, tuple))
                    else i,
                    "steps": len(traj.reports),
                    "reward": round(traj.total, 3),
                    "cost": getattr(t, "cost", None),
                    "cert": getattr(t, "certificate_ok", None),
                }
            )
        out[name] = rows
    return out
