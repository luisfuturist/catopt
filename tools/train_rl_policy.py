"""Train an RL search policy on a multi-family mixture and race it.

Plan 0016 stage 7, the RL half (ADR 0003).  REINFORCE over the search
env; training data is a mixture of three families (matmul chains,
elementwise duplication, a linear/relu block — see :mod:`families`).
The trained policy is then scored **per family**: the mean rank of its
greedy first pick (1 = best) and its hit rate, plus a race against
random / declaration-order / the one-step greedy oracle / no-op on the
mean final cost.  ``--train-families chain`` reproduces the one-family
baseline the mixture is compared against.

Usage::

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
from catopt_core.features import compute_features
from catopt_core.game import Action, Evaluator, GameState
from catopt_core.laws import all_rules
from catopt_core.search_env import SearchEnv
from catopt_core.trajectories import rule_samples
from catopt_discovery import families
from catopt_torch.rl import RLPolicy, train_reinforce


def _parse_families(spec: str) -> list[str]:
    """Parse a comma-separated family list; reject unknown names."""
    fams = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in fams if f not in families.FAMILIES]
    if bad:
        raise SystemExit(
            f"unknown families {bad}; pick from {families.FAMILIES}"
        )
    return fams


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


def _rl_pick(
    policy: RLPolicy, term: object, actions: list[Action]
) -> str:
    """Return the policy's greedy first pick for a bare program."""
    state = GameState(None, 0, features=compute_features(term))
    return policy.choose(state, actions).rule


def _rank_table(rows: list[tuple[str, dict]]) -> None:
    """Print the per-family rank / hit-rate table."""
    head = (
        f"{'family':<8} {'n':>3} {'mean_rank':>9} "
        f"{'hit_rate':>8} {'mean_delta':>11}"
    )
    print(head)
    print("-" * len(head))
    for fam, s in rows:
        print(
            f"{fam:<8} {s['n']:>3} {s['mean_rank']:>9.2f} "
            f"{s['hit_rate']:>8.2f} {s['mean_delta']:>11.1f}"
        )


def _race_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print the per-family mean-final-cost race."""
    players = ("random", "declaration", "greedy", "rl", "no-op")
    head = f"{'family':<8} " + " ".join(f"{p:>11}" for p in players)
    print(head)
    print("-" * len(head))
    for fam, costs in rows:
        row = " ".join(f"{costs[p]:>11.1f}" for p in players)
        print(f"{fam:<8} {row}")


def main() -> None:
    """Train the RL policy on the mixture, then score it per family."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--train-families", default=",".join(families.FAMILIES)
    )
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Keep the episode-progress logs, silence the egraph's per-extraction
    # INFO chatter (which would otherwise drown the report tables).
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("catopt_torch.rl").setLevel(logging.INFO)

    dev = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    train_fams = _parse_families(args.train_families)
    print(f"device: {dev}  (torch {torch.__version__})")
    print(f"train families: {','.join(train_fams)}")

    rules = all_rules()
    by_name = {r.name: r for r in rules}
    actions = [Action(r.name) for r in rules]

    progs = families.mixture_programs(train_fams, args.train, args.seed)
    print(f"train programs: {len(progs)}")
    model = train_reinforce(
        progs,
        rules,
        episodes=args.episodes,
        horizon=args.horizon,
        hidden=args.hidden,
        device=dev,
        seed=args.seed,
        log_every=max(1, args.episodes // 5),
    )
    policy = RLPolicy(model, by_name, device=dev)

    rank_rows: list[tuple[str, dict]] = []
    race_rows: list[tuple[str, dict[str, float]]] = []
    for fam in families.FAMILIES:
        held = families.family_programs(
            fam, args.eval, args.seed + 100, held_out=True
        )
        scores = [
            families.score_pick(p, rules, _rl_pick(policy, p, actions))
            for p in held
        ]
        rank_rows.append((fam, families.summarize(scores)))

        choosers: dict[str, Any] = {
            "random": lambda env: random.choice(env.action_names),
            "declaration": lambda env: env.action_names[0],
            "greedy": lambda env: _greedy_rule(env, rules),
            "rl": lambda env: _rl_rule(env, policy),
        }
        costs: dict[str, float] = {}
        for name, chooser in choosers.items():
            vals = [
                _episode(
                    SearchEnv(
                        p, rules, horizon=args.horizon, patience=2
                    ),
                    chooser,
                )
                for p in held
            ]
            costs[name] = statistics.fmean(vals)
        base = []
        for p in held:
            eg = EGraph()
            base.append(
                Evaluator().evaluate(GameState(eg, eg.add_term(p)))
            )
        costs["no-op"] = statistics.fmean(base)
        race_rows.append((fam, costs))

    print()
    _rank_table(rank_rows)
    print()
    _race_table(race_rows)


if __name__ == "__main__":
    main()
