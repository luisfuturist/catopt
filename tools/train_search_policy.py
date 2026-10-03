"""Train a supervised search policy on a multi-family mixture.

Plan 0016 stage 7 (ADR 0003).  The policy is a ``RuleValueNet`` over
(program features ⊕ structural rule vector).  Training data is a
mixture of three families (matmul chains, elementwise duplication, a
linear/relu block — see :mod:`families`).  The trained policy then
picks one rule per held-out program and is scored **per family**: the
mean rank of its pick (1 = best) and its hit rate (a strictly-improving
rule).  ``--train-families chain`` reproduces the one-family baseline
the mixture is compared against.

Usage::

    python tools/train_search_policy.py [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse

import families
import torch
from catopt_core.features import compute_features
from catopt_core.game import Action, GameState
from catopt_core.laws import all_rules
from catopt_core.trajectories import rule_samples
from catopt_torch.learned_policy import LearnedPolicy, train_rule_value


def _parse_families(spec: str) -> list[str]:
    """Parse a comma-separated family list; reject unknown names."""
    fams = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in fams if f not in families.FAMILIES]
    if bad:
        raise SystemExit(
            f"unknown families {bad}; pick from {families.FAMILIES}"
        )
    return fams


def _pick(
    policy: LearnedPolicy, term: object, actions: list[Action]
) -> str:
    """Return the policy's rule choice for a bare program."""
    state = GameState(None, 0, features=compute_features(term))
    return policy.choose(state, actions).rule


def _table(rows: list[tuple[str, dict]]) -> None:
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


def main() -> None:
    """Train on the mixture, then score the policy per family."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--train-families", default=",".join(families.FAMILIES)
    )
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    progs = families.mixture_programs(
        train_fams, args.per_family, args.seed
    )
    samples = [s for p in progs for s in rule_samples(p, rules)]
    print(f"train programs: {len(progs)}  samples: {len(samples)}")
    model = train_rule_value(
        samples,
        epochs=args.epochs,
        hidden=args.hidden,
        device=dev,
        seed=args.seed,
    )
    policy = LearnedPolicy(model, by_name, device=dev)

    print()
    rows = []
    for fam in families.FAMILIES:
        held = families.family_programs(
            fam, args.eval, args.seed + 100, held_out=True
        )
        scores = [
            families.score_pick(p, rules, _pick(policy, p, actions))
            for p in held
        ]
        rows.append((fam, families.summarize(scores)))
    _table(rows)


if __name__ == "__main__":
    main()
