"""Pretrain and post-train a search policy, then predict a program.

Plan 0016 stage 7 (ADR 0003).  Pretraining data is generated on
synthetic small programs; post-training fine-tunes on a small real
torch model exported to IR.  The trained policy then predicts which
rule improves a held-out program, and the engine confirms the
prediction against the best and mean rule.

Usage::

    python tools/train_search_policy.py [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import random
import statistics
from typing import Any

import torch
from catopt_core.egraph import EGraph
from catopt_core.features import compute_features
from catopt_core.game import Action, Evaluator, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.trajectories import RuleSample, rule_samples
from catopt_torch.learned_policy import LearnedPolicy, train_rule_value
from torch import nn


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain(d: int, k: int, m: int) -> Op:
    """Build the expensive bracketing ``a·(b·c)``."""
    a, b, c = _v("a", d, k), _v("b", k, m), _v("c", m, d)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def synthetic_programs(n: int, seed: int) -> list[Op]:
    """Small random matmul chains — the pretraining family."""
    rng = random.Random(seed)
    return [
        _chain(
            rng.choice([2, 3, 4]),
            rng.choice([2, 3, 4]),
            rng.choice([2, 3, 4]),
        )
        for _ in range(n)
    ]


def small_model_programs(n: int, seed: int) -> list[Op]:
    """Post-training data: small real torch models exported to IR."""
    from catopt_torch.adapters import TorchSource

    progs: list[Op] = []
    for i in range(n):
        torch.manual_seed(seed + i)
        model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8)
        )
        ir, _ = TorchSource().to_ir(model, torch.randn(4, 8))
        progs.append(ir.root)
    return progs


def _samples(programs: list[Op], rules: Any) -> list[RuleSample]:
    out: list[RuleSample] = []
    for p in programs:
        out.extend(rule_samples(p, rules))
    return out


def _cost(term: Any) -> float:
    eg = EGraph()
    return Evaluator().evaluate(GameState(eg, eg.add_term(term)))


def _rule_deltas(term: Any, rules: Any, before: float) -> list[float]:
    """Return the cost improvement each rule alone gives."""
    out: list[float] = []
    for r in rules:
        eg = EGraph()
        root = eg.add_term(term)
        eg.apply_rule(r, root)
        out.append(before - Evaluator().evaluate(GameState(eg, root)))
    return out


def main() -> None:
    """Run pretrain, post-train, then predict on a held-out program."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pretrain", type=int, default=40)
    ap.add_argument("--posttrain", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    dev = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"device: {dev}  (torch {torch.__version__})")

    rules = all_rules()
    by_name = {r.name: r for r in rules}

    pre = _samples(synthetic_programs(args.pretrain, 0), rules)
    print(f"pretrain samples: {len(pre)}")
    model = train_rule_value(
        pre, epochs=args.epochs, device=dev, hidden=args.hidden
    )

    post = _samples(small_model_programs(args.posttrain, 1), rules)
    print(f"posttrain samples: {len(post)} (+{len(pre)} replay)")
    # Replay the pretraining family: post-training on a different family
    # alone causes catastrophic forgetting (measured: the chain pick
    # drops from rank 1 to rank 2).
    model = train_rule_value(
        pre + post,
        epochs=args.epochs,
        device=dev,
        hidden=args.hidden,
        seed=1,
    )

    held = _chain(3, 5, 2)
    before = _cost(held)
    feats = compute_features(held)
    by_rule = {
        r.name: d
        for r, d in zip(
            rules, _rule_deltas(held, rules, before), strict=True
        )
    }
    policy = LearnedPolicy(model, by_name, device=dev)
    pick = policy.choose(
        GameState(None, 0, features=feats),
        [Action(r.name) for r in rules],
    )
    ordered = sorted(by_rule, key=by_rule.__getitem__, reverse=True)
    print(f"held-out cost before: {before:.0f}")
    print(
        f"policy picks: {pick.rule}  (delta {by_rule[pick.rule]:+.0f})"
    )
    print(
        f"best delta: {max(by_rule.values()):+.0f}; "
        f"mean {statistics.mean(by_rule.values()):+.0f}"
    )
    print(f"picked rank: {ordered.index(pick.rule) + 1}/{len(ordered)}")


if __name__ == "__main__":
    main()
