"""``play`` — the optimization game as a tool.

One registry of *domains* (a board family plus its player
vocabulary), one entrypoint: pick a domain, pick arms, get the
paired-boards table.  Adding a domain is data — a case generator, a
``case -> Board`` factory, the legal-move enumerator and the
learned player's featurizer — no driver code:

.. code-block:: text

    DOMAINS[name] = Domain(
        cases=gen_cases,            # seed, n -> [case, ...]
        board_of=...,               # case -> Board
        legal=...,                  # state -> tuple[action, ...]
        featurizer=...,             # (state, action, hist) -> dict
        scripted=...,               # optional fixed-playbook arm
    )

Shipped domains:

* ``meta`` — the meta-arena: fire/saturate/declare/handle/extract
  over the mined-spec corpus (:func:`meta_player.gen_cases`);
* ``joint`` — training graphs: a forward term plus every derived
  gradient under one ``joint`` root, played on the same meta-arena
  board (the reverse handler's programs are ordinary terms);
* ``search`` — the core :class:`~catopt_core.search_env.SearchEnv`:
  one rule per step under a horizon/patience bound, adapted to the
  :class:`~catopt_discovery.engine.Board` contract.

Run it::

    python -m catopt_discovery.play --domain meta --train-cases 40 \
        --eval-cases 10
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Any

from catopt_core.cost.basic import count_cost
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from catopt_core.search_env import SearchEnv

from . import engine, lawdata, training
from . import meta_arena as ma
from . import meta_player as mp
from .meta_player import _bucket
from .players import LinearPolicy

__all__ = [
    "DOMAINS",
    "Domain",
    "SearchBoard",
    "play",
]


@dataclass(frozen=True)
class Domain:
    """One board family's registration — data, not code."""

    cases: Any  # (seed, n) -> [case, ...]
    board_of: Any  # case -> Board
    legal: Any  # state -> tuple[action, ...]
    featurizer: Any  # (state, action, hist) -> dict
    features: Any = ()  # the learned arm's schema
    scripted: Any = None  # () -> player, or None
    deliver: Any = None  # (board, case, traj) -> artifact | None


def play(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> engine.Trajectory:
    """Play one episode of *domain*/*case* under *player*."""
    d = DOMAINS[domain]
    return engine.run_episode(d.board_of(case), player, budget)


def deliver(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> dict:
    """Play an episode on *case*; return the lowered artifact.

    The product loop: the board's winning extraction resolved by
    ``MetaArena.deliverable`` (handled ops instantiate their
    kernels, unhandled abbreviations spell back), lowered through
    the domain's sink, verified against the source model.  Domains
    without a ``deliver`` hook raise — the game score needs no
    artifact; delivery does.
    """
    d = DOMAINS[domain]
    if d.deliver is None:
        raise ValueError(f"domain {domain!r} has no deliver hook")
    board = d.board_of(case)
    traj = engine.run_episode(board, player, budget)
    return d.deliver(board, case, traj)


# ---------------------------------------------------------------------------
#  Domain: meta — the mixed board
# ---------------------------------------------------------------------------


def _meta_board(case: Any) -> ma.MetaArena:
    """Probe-arity case tuple -> MetaArena (count_cost)."""
    return mp._board(case, count_cost, 20_000)


# ---------------------------------------------------------------------------
#  Domain: joint — forward + gradients on one board
# ---------------------------------------------------------------------------


def _joint_cases(seed: int, n: int) -> list:
    """Joint programs over shape-varied forward archetypes.

    Each case joints a forward (silu / softmax-spine / linear+tanh
    at a random width) with every derived gradient under one
    ``joint`` root — the reverse handler's programs are ordinary
    terms, so the meta-arena's move set applies unchanged.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d, b = 4 * rng.choice((1, 2)), 4 * rng.choice((1, 2))
        x = Var("x", TensorType((b, d)))
        w = Var("w", TensorType((d, d)))
        archetype = rng.randrange(4)
        if archetype == 0:
            fwd = Op.make("mul", x, Op.make("sigmoid", x))
        elif archetype == 1:
            fwd = Op.make("softmax", x, dim=-1)
        elif archetype == 2:
            fwd = Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=1, keepdim=True),
            )
        else:
            fwd = Op.make("tanh", Op.make("matmul", x, w))
        cot = training._cotangent_for(fwd)
        grads = training.backward(fwd, cotangent=cot)
        joint = Op.make(
            "joint",
            fwd,
            *[grads[k] for k in sorted(grads)],
            validate=False,
        )
        out.append((f"joint{i}:{archetype}", joint))
    return out


def _joint_board(case: Any) -> ma.MetaArena:
    """Board a joint program under the meta-arena's moves."""
    return ma.MetaArena(case[1], cost_fn=count_cost)


# ---------------------------------------------------------------------------
#  Domain: torch — real modules under the sink's bound
# ---------------------------------------------------------------------------


def _torch_cases(seed: int, n: int) -> list:
    """Small real ``nn.Module``s lifted to IR — the product board.

    Each case is a torch module exported through
    ``catopt_torch.adapters.TorchSource`` and played under the real
    sink's ``supported_ops`` bound — kernel claims price at sink
    costs, not ambient vocabulary.  Modules that fail to export are
    skipped honestly.  Torch is imported lazily — the registry
    itself stays torch-free.
    """
    import torch
    import torch.nn as nn
    from catopt_torch.adapters import TorchSink, TorchSource

    class _SpelledSiLU(nn.Module):
        """x·sigmoid(x) spelled out — the claim move's live site."""

        def forward(self, x):
            """Spell ``x·sigmoid(x)`` — no fused op named."""
            return x * torch.sigmoid(x)

    rng = random.Random(seed)
    supported = TorchSink().supported_ops
    src = TorchSource()
    builders = [
        ("spelled_silu", lambda d: _SpelledSiLU()),
        (
            "mlp",
            lambda d: nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
            ),
        ),
        (
            "relu_lin",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.ReLU()),
        ),
        ("lin", lambda d: nn.Linear(d, d)),
        (
            "softsign_mlp",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.Softsign()),
        ),
    ]
    out = []
    for i in range(n):
        d = 4 * rng.choice((1, 2, 4))
        kind, mk = builders[rng.randrange(len(builders))]
        torch.manual_seed(seed * 10_007 + i)
        model = mk(d).eval()
        example = torch.randn(4, d)
        ir, leaves = src.to_ir(model, example)
        out.append(
            (
                f"torch{i}:{kind}",
                ir.root,
                None,
                (),
                supported,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "input": example,
                    "model": model,
                },
            )
        )
    return out


def _torch_board(case: Any) -> ma.MetaArena:
    """Board a torch-lifted program under the real sink bound."""
    rules = case[2] if len(case) > 2 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=count_cost,
        supported=case[4] if len(case) > 4 else None,
    )


def _torch_deliver(board: ma.MetaArena, case: Any, traj: Any) -> dict:
    """Lower the winning extraction and verify it against the model.

    ``deliverable`` resolves handled claims to their kernels and
    unhandled abbreviations to spelled form; ``TorchSink.lower``
    rebuilds the module from it; ``verify`` runs both on the case's
    example input.  The game score becomes a delivered artifact.
    """
    from catopt_core.ir import IR
    from catopt_torch.adapters import TorchSink

    ex = case[5]
    best = board.eg.extract_best(board.root, board.feasible_cost)
    if best is None:
        return {"delivered": False, "reward": traj.total}
    term = board.deliverable(best)
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    sink = TorchSink()
    module = sink.lower(ir, params=ex["leaves"])
    rep = sink.verify(ex["model"], module, (ex["input"],))
    t = traj.terminal
    return {
        "delivered": True,
        "module": module,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }


# ---------------------------------------------------------------------------
#  Domain: search — SearchEnv adapted to the Board contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchState:
    """The search board's observable: rule names, cursor, terminator."""

    actions: tuple[str, ...]
    steps: int
    done: bool
    cost: float
    baseline: float


class SearchBoard:
    """``SearchEnv`` under the :class:`engine.Board` contract.

    Actions are rule names; ``observe`` returns a
    :class:`_SearchState`; ``step`` fires the rule once and reports
    the env's normalized cost improvement as reward.  Episode ends
    at the env's horizon/patience, or when the player drains.
    """

    def __init__(
        self,
        term: Any,
        rules: Any,
        *,
        horizon: int = 6,
        patience: int = 2,
    ) -> None:
        """Bind the program, rules, and episode bounds."""
        self.env = SearchEnv(term, rules, cost_fn=count_cost)
        self._h = horizon
        self._p = patience
        self._steps = 0
        self._done = True
        self._baseline = float("inf")

    def observe(self) -> _SearchState:
        """Return the board view — reset lazily on first look."""
        if self._done and self._steps == 0:
            self.env.reset()
            self._done = False
            self._baseline = self.env.cost
        return _SearchState(
            actions=self.env.action_names,
            steps=self._steps,
            done=self._done,
            cost=self.env.cost,
            baseline=self._baseline,
        )

    def step(self, action: Any) -> tuple[_SearchState, engine.Report]:
        """Fire one rule; report the cost improvement as reward."""
        self.observe()  # ensure reset
        name = (
            action if isinstance(action, str) else action.params["rule"]
        )
        res = self.env.step(name)
        self._steps += 1
        self._done = res.done
        return self.observe(), engine.Report(
            action, reward=res.reward, terminal=res.done, cost=res.cost
        )


def _search_cases(seed: int, n: int) -> list:
    """Rule-fire boards: the demo forwards plus small variants."""
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    pool = [
        Op.make("mul", x, Op.make("sigmoid", x)),
        Op.make("add", Op.make("mul", x, y), Op.make("matmul", x, y)),
        Op.make("mul", x, x),
        Op.make("div", x, Op.make("add", Op.make("abs", x), Const(1))),
    ]
    return [
        (f"search{i}", pool[rng.randrange(len(pool))]) for i in range(n)
    ]


def _search_board(case: Any) -> SearchBoard:
    """Board a search case under the rule-fire game."""
    return SearchBoard(case[1], DEFAULT_SEARCH_RULES)


def _search_legal(state: _SearchState) -> tuple:
    """Every rule name is a legal move until the env says done."""
    return state.actions


def _search_featurizer(
    state: _SearchState, action: Any, _hist: Any
) -> dict:
    """Minimal row: rule bucket plus the board summary."""
    return {
        "bias": 1.0,
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
        f"a:r:{_bucket(action, 'rule'):02d}": 1.0,
    }


SEARCH_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)


def _random_policy(legal: Any) -> Any:
    """Return a uniform player over any domain's legal set."""

    class _R:
        def __init__(self, seed: int) -> None:
            self._r = random.Random(seed)

        def __call__(self, state: Any) -> Any | None:
            acts = legal(state)
            return acts[self._r.randrange(len(acts))] if acts else None

    return _R


def _scripted_search() -> Any:
    """Fire every rule once in order — the search board's playbook."""
    it = {"i": 0}
    order = tuple(r.name for r in DEFAULT_SEARCH_RULES)

    def go(state: _SearchState) -> Any | None:
        if state.done or it["i"] >= len(order):
            return None
        name = order[it["i"]]
        it["i"] += 1
        return name

    return go


DOMAINS: dict[str, Domain] = {
    "meta": Domain(
        cases=mp.gen_cases,
        board_of=_meta_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
    ),
    "joint": Domain(
        cases=_joint_cases,
        board_of=_joint_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
    ),
    "torch": Domain(
        cases=_torch_cases,
        board_of=_torch_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "search": Domain(
        cases=_search_cases,
        board_of=_search_board,
        legal=_search_legal,
        featurizer=_search_featurizer,
        features=SEARCH_FEATURES,
        scripted=_scripted_search,
    ),
}


def _arms(domain: Domain, player: LinearPolicy, seed: int) -> dict:
    """Build the arm set for one domain: baselines + learned pair."""
    rnd = _random_policy(domain.legal)
    arms: dict[str, Any] = {
        "random": lambda i: rnd(seed + 10_000 + i),
        "learned": lambda i: player.frozen(seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            seed + 30_000 + i, greedy=True
        ),
    }
    if domain.scripted is not None:
        arms = {"scripted": lambda i: domain.scripted(), **arms}
    return arms


def _table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over eval rows — generic edition."""
    head = f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7} {'cert':>5} {'steps':>6}"
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train the learned arm on one case stream, eval on another."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="meta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument(
        "--deliver",
        action="store_true",
        help="lower + verify the first eval case's extraction",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    d = DOMAINS[args.domain]
    train = d.cases(args.seed, args.train_cases)
    ev = d.cases(args.seed + 7919, args.eval_cases)
    player = LinearPolicy(
        seed=args.seed,
        legal=d.legal,
        featurizer=d.featurizer,
        features=d.features,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    totals = engine.train_policy(
        player, train, d.board_of, budget=args.budget
    )
    table = engine.evaluate(
        _arms(d, player, args.seed), ev, d.board_of, budget=args.budget
    )
    print(f"== play — domain {args.domain} ==")  # stdout-compat
    print(_table(table))  # stdout-compat
    decade = max(len(totals) // 10, 1)
    means = [
        round(sum(totals[i : i + decade]) / decade, 2)
        for i in range(0, len(totals) - decade + 1, decade)
    ]
    print(f"train decade means: {means}")  # stdout-compat
    if args.deliver:
        arm = (
            d.scripted() if d.scripted is not None else player.frozen(0)
        )
        res = deliver(args.domain, ev[0], arm, budget=args.budget)
        shown = {k: v for k, v in res.items() if k != "module"}
        print(f"deliver[{ev[0][0]}]: {shown}")  # stdout-compat
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {"table": table, "train_totals": totals}, f, indent=1
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
