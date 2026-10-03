"""Train an RL search policy (REINFORCE) and race it against the rest.

Plan 0016 stage 7, the RL half.  Usage::

    python tools/train_rl_policy.py [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
from typing import Any

import torch
from catopt_core.egraph import EGraph
from catopt_core.game import Action, Evaluator, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.search_env import SearchEnv
from catopt_core.trajectories import rule_samples
from catopt_torch.rl import RLPolicy, train_reinforce


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain(d: int, k: int, m: int) -> Op:
    """Build the expensive bracketing ``a·(b·c)``."""
    a, b, c = _v("a", d, k), _v("b", k, m), _v("c", m, d)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def programs(n: int, seed: int, lo: int = 2, hi: int = 5) -> list[Op]:
    """Build a family of matmul chains with random shapes."""
    rng = random.Random(seed)
    return [
        _chain(
            rng.randint(lo, hi),
            rng.randint(lo, hi),
            rng.randint(lo, hi),
        )
        for _ in range(n)
    ]


def _greedy_rule(env: SearchEnv, rules: Any) -> str:
    """Return the one-step cost-model oracle from the current state."""
    term = env.eg.extract_best(env.root, env.cost_fn)
    samples = rule_samples(term, rules)
    return max(samples, key=lambda s: s.delta_cost).rule


def _rl_rule(env: SearchEnv, policy: RLPolicy) -> str:
    """Return the RL policy's greedy pick from the current state."""
    acts = [Action(n) for n in env.action_names]
    logits = policy.logits(env.state(), env.progress, acts)
    return acts[int(torch.argmax(logits))].rule


def _episode(env: SearchEnv, chooser: Any) -> float:
    """Run one episode under ``chooser``; return the final cost."""
    env.reset()
    for _ in range(env.horizon):
        if env.step(chooser(env)).done:
            break
    return env.cost


def main() -> None:
    """Train the RL policy, then race it on held-out programs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dev = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"device: {dev}  (torch {torch.__version__})")

    rules = all_rules()
    by_name = {r.name: r for r in rules}
    model = train_reinforce(
        programs(args.train, 0),
        rules,
        episodes=args.episodes,
        horizon=args.horizon,
        hidden=args.hidden,
        device=dev,
        log_every=max(1, args.episodes // 5),
    )
    policy = RLPolicy(model, by_name, device=dev)

    held = programs(8, 999, lo=2, hi=7)  # shapes unseen in training
    choosers = {
        "random": lambda env: random.choice(env.action_names),
        "declaration": lambda env: env.action_names[0],
        "greedy": lambda env: _greedy_rule(env, rules),
        "rl": lambda env: _rl_rule(env, policy),
    }
    print()
    for name, chooser in choosers.items():
        costs = [
            _episode(
                SearchEnv(p, rules, horizon=args.horizon, patience=2),
                chooser,
            )
            for p in held
        ]
        print(
            f"{name:12s} mean final cost {statistics.mean(costs):8.1f}"
        )

    base = []
    for p in held:
        eg = EGraph()
        base.append(Evaluator().evaluate(GameState(eg, eg.add_term(p))))
    print(f"{'no-op':12s} mean final cost {statistics.mean(base):8.1f}")


if __name__ == "__main__":
    main()
