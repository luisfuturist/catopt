"""Diagnose whether the n = 40 gap is generalisation or inductive bias.

Plan 0016 follow-up.  ``contraction_policy.py`` is trained on n = 8-12 and
tested at n = 20/30/40; ``contraction-policy-einsum.md`` showed it loses
to ``opt_einsum``'s randomised greedy at n = 40 by ~1.8x, and
``contraction-policy-throughput.md`` showed the gap is a *quality* gap,
not throughput (quality saturates with rollout count).  The one lever
never pulled is the **training distribution**.  If training at scale
(``tools/contraction_policy.py --train-scales 16,20,24``) closes the gap,
the cause was a train/test scale mismatch (**generalisation**).  If it
does not, the question is whether the features can express what
distinguishes a good contraction at n = 40 (**inductive bias**).

This probe answers the second question directly, on the einsum-valid bond
family (the one ``opt_einsum`` can price), with two measurements:

1. **Decision agreement at n = 40.**  Walk the best cheap player's order
   (``opt_einsum``'s ``RandomGreedy``, the player we lose to) and, at
   every state along it, ask each policy which pair it would contract.
   Agreement is the fraction of the teacher's decisions the policy
   reproduces.  A policy that cannot choose like the strong player is
   either untrained at scale or cannot represent the choice.
2. **Feature sufficiency.**  Fit a supervised net — *the same features,
   the same net* — to the best cheap player's action at n = 40, with a
   held-out board split.  If the net reaches high held-out top-1
   accuracy, the features *can* express the choice, so the RL gap is a
   generalisation/optimisation issue, not inductive bias.  If it cannot
   fit even the training labels, the feature vector is the limit.

A third section checks whether each policy beats greedy *where it was
trained* (the REINFORCE objective), so "trained at scale" is measured
rather than assumed.

``opt_einsum`` lives in the opt-in ``einsum`` dependency group; the tool
exits with a sync hint when the group is absent.

Usage::

    uv sync --group einsum
    .venv/bin/python tools/contraction_scale_probe.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Any

import contraction_einsum as ce
import contraction_policy as cp
import contraction_scale as cs
import torch
from torch.nn import functional as F

__all__ = ["main"]

#: The best cheap player's order is the teacher; this is its budget.
_TEACHER_BUDGET = 1.0


def _samples(
    tensors: Any,
    sizes: dict[int, int],
    order: list[tuple[int, int]],
) -> list[tuple[Any, tuple[int, int]]]:
    """Replay a teacher order into ``(state, action)`` samples."""
    return cp._replay_samples(tensors, sizes, order, len(tensors))


def _policy_choice(
    model: Any, game: cp.ContractionGame, device: str
) -> tuple[int, int]:
    """Return the policy's argmax pair at ``game``'s state."""
    sf, pf = cp._batch_inputs([game], device)
    with torch.no_grad():
        logits = cp._logits(model, sf, pf)
    return game.pairs[int(torch.argmax(logits, dim=1)[0])]


def _agreement_section(
    models: dict[str, Any],
    device: str,
    n: int,
    boards: int,
    seed: int,
    budget: float,
) -> dict[str, float]:
    """Measure each player's agreement with the best cheap player."""
    print()
    print(
        f"== 1. decision agreement with the best cheap player (n={n}) =="
    )
    print(
        "  at each state along opt_einsum RandomGreedy's order, does the"
    )
    print("  player pick the same pair?  (mean over states and boards)")
    names = ["our-greedy", *models]
    acc: dict[str, list[float]] = {k: [] for k in names}
    outer: list[float] = []
    for k in range(boards):
        tensors, sizes = ce.random_bond_network(n, seed + k)
        e = ce.to_einsum(tensors, sizes)
        order = ce.oe_random_greedy_order(e, budget)
        ref = cs.greedy(tensors, sizes)
        ts = [frozenset(t) for t in tensors]
        for a, b in order:
            game = cp.ContractionGame(ts, sizes, ref)
            target = (min(a, b), max(a, b))
            cheapest = min(
                game.pairs,
                key=lambda p: cs.pair_cost(ts[p[0]], ts[p[1]], sizes),
            )
            acc["our-greedy"].append(cheapest == target)
            outer.append(not (ts[a] & ts[b]))
            for name, model in models.items():
                acc[name].append(
                    _policy_choice(model, game, device) == target
                )
            ts = cp._merge_ts(ts, a, b)
    print(f"  {'player':>15} {'agreement':>10}")
    print("-" * 27)
    out: dict[str, float] = {}
    for name in names:
        m = statistics.fmean(acc[name])
        out[name] = m
        print(f"  {name:>15} {m:>10.3f}")
    print(
        f"  (the teacher's order forms an outer product in "
        f"{statistics.fmean(outer):.3f} of its steps)"
    )
    return out


def _fit_section(
    device: str,
    n: int,
    boards: int,
    seed: int,
    budget: float,
    *,
    epochs: int,
    hidden: int,
    holdout: int,
) -> None:
    """Test whether the same features and net fit the best cheap player.

    Two teachers are fitted, both with the *same* features and net: the
    deterministic staged greedy (a clean, reproducible rule) and
    RandomGreedy's best order (the player we lose to, whose decisions
    are stochastic).  The deterministic fit is the calibration — if the
    net fits *it* but not RandomGreedy, the features are sufficient and
    the residual is the teacher's stochasticity.
    """
    print()
    print(
        f"== 2. feature sufficiency: fit the best cheap player (n={n}) =="
    )
    train_boards = [
        ce.random_bond_network(n, seed + k) for k in range(boards)
    ]
    test_boards = [
        ce.random_bond_network(n, seed + 10_000 + k)
        for k in range(holdout)
    ]
    teachers = {
        "oe-greedy": lambda e: ce.oe_greedy_order(e),
        "oe-rand-greedy": lambda e: ce.oe_random_greedy_order(
            e, budget
        ),
    }

    def _dataset(bds: list[Any], order_fn: Any) -> list[Any]:
        data: list[Any] = []
        for tensors, sizes in bds:
            e = ce.to_einsum(tensors, sizes)
            data.extend(_samples(tensors, sizes, order_fn(e)))
        return data

    print(
        f"  {len(train_boards)} train / {len(test_boards)} held-out "
        f"boards; {epochs} epochs"
    )
    print(
        f"  {'teacher':>15} {'train':>8} {'held-out':>9} "
        f"{'roll/teacher':>13}"
    )
    print("-" * 48)
    for tname, order_fn in teachers.items():
        train = _dataset(train_boards, order_fn)
        test = _dataset(test_boards, order_fn)
        rng = random.Random(seed)
        model = cp.PairPolicyNet(hidden).to(device)
        cp._fit_norm(model, (n,), rng, device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(epochs):
            rng.shuffle(train)
            for start in range(0, len(train), 64):
                chunk = train[start : start + 64]
                sf, pf, mask, target = cp._pack(chunk, device)
                logits = cp._logits(model, sf, pf).masked_fill(
                    mask == 0, -1e9
                )
                loss = F.cross_entropy(
                    logits, torch.tensor(target, device=device)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        roll = _rollout_vs_teacher(model, test_boards, order_fn, device)
        print(
            f"  {tname:>15} {_top1(model, train, device):>8.3f} "
            f"{_top1(model, test, device):>9.3f} {roll:>13.3f}"
        )
    print(
        "  (high held-out top-1 => the features can express the strong "
        "player's choice;"
    )
    print(
        "   roll/teacher < 1 => the net's own greedy rollout beats the "
        "teacher's order)"
    )


def _rollout_vs_teacher(
    model: Any,
    boards: list[Any],
    order_fn: Any,
    device: str,
) -> float:
    """Mean greedy-rollout cost / teacher-order cost on held-out boards."""
    vals = []
    for tensors, sizes in boards:
        ref = cs.greedy(tensors, sizes)
        got = cp.run_policy_batch(
            model,
            tensors,
            sizes,
            ref,
            samples=1,
            greedy=True,
            temperature=1.0,
            device=device,
        )[0]
        e = ce.to_einsum(tensors, sizes)
        teacher = ce.our_cost_of_order(tensors, sizes, order_fn(e))
        vals.append(got / teacher)
    return statistics.fmean(vals)


def _top1(
    model: Any,
    data: list[tuple[Any, tuple[int, int]]],
    device: str,
) -> float:
    """Masked top-1 accuracy of the net on a labelled dataset."""
    hits = 0
    for game, action in data:
        sf, pf = cp._batch_inputs([game], device)
        with torch.no_grad():
            logits = cp._logits(model, sf, pf)
        hits += int(torch.argmax(logits, dim=1)[0]) == game.pairs.index(
            action
        )
    return hits / max(len(data), 1)


def _train_fit_section(
    models: dict[str, Any],
    device: str,
    train_ns: dict[str, tuple[int, ...]],
    boards: int,
    seed: int,
) -> None:
    """Check whether each policy beats greedy where it was trained."""
    print()
    print(
        "== 3. does each policy beat greedy on its training board? =="
    )
    print("  ratio to greedy (mean over boards; <1 = beats greedy)")
    print(f"  {'policy':>15} {'train n':>10} {'ratio':>8}")
    print("-" * 35)
    for name, model in models.items():
        ns = train_ns[name]
        vals = []
        for n0 in ns:
            for k in range(boards):
                tensors, sizes = ce.random_bond_network(
                    n0, seed + 500 + k
                )
                ref = cs.greedy(tensors, sizes)
                got = cp.run_policy_batch(
                    model,
                    tensors,
                    sizes,
                    ref,
                    samples=1,
                    greedy=True,
                    temperature=1.0,
                    device=device,
                )[0]
                vals.append(got / ref)
        print(f"  {name:>15} {ns!s:>10} {statistics.fmean(vals):>8.3f}")


def main(argv: list[str] | None = None) -> int:
    """Train the policies, then run the generalisation probe."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--scales", type=int, default=40)
    ap.add_argument("--boards", type=int, default=12)
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument(
        "--only-fit",
        action="store_true",
        help="skip the RL policies; run only the feature-sufficiency fit",
    )
    ap.add_argument(
        "--teacher-budget",
        type=float,
        default=_TEACHER_BUDGET,
        help="seconds the best cheap player gets per board",
    )
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  (torch {torch.__version__})")

    if args.only_fit:
        _fit_section(
            dev,
            args.scales,
            args.boards,
            args.seed,
            args.teacher_budget,
            epochs=args.epochs,
            hidden=args.hidden,
            holdout=args.holdout,
        )
        return 0

    train_ns = {"base": (8, 10, 12), "scale": (16, 20, 24)}
    models: dict[str, Any] = {}
    with ce._on_family(ce.random_bond_network):
        for name, ns in train_ns.items():
            t0 = time.perf_counter()
            models[name] = cp.train_rl(
                ns=ns,
                iterations=args.iterations,
                batch=args.batch,
                hidden=args.hidden,
                device=dev,
                seed=args.seed,
            )
            print(
                f"trained {name} ({ns}) in "
                f"{time.perf_counter() - t0:.1f}s"
            )

    _agreement_section(
        models,
        dev,
        args.scales,
        args.boards,
        args.seed,
        args.teacher_budget,
    )
    _fit_section(
        dev,
        args.scales,
        args.boards,
        args.seed,
        args.teacher_budget,
        epochs=args.epochs,
        hidden=args.hidden,
        holdout=args.holdout,
    )
    _train_fit_section(models, dev, train_ns, 8, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
