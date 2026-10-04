"""Teacher-trajectory distillation — does the strong teacher transfer.

Plan 0016 follow-up.  ``contraction-train-scale.md`` §2 measured a
surprise: a supervised net fit to the *deterministic* teacher
(``oe-greedy``) rolls out **better than its teacher** (roll/teacher
0.82) — the smoothed learned policy beats the order it was trained to
copy.  The lever never pulled: distill the **strong** teacher —
``opt_einsum``'s ``RandomGreedy`` (``oe-rand-greedy``), the player the
learned policy still loses to at n = 40 (~1.04x best-case).

The catch is structural, and this tool measures it honestly:
``RandomGreedy``'s edge is best-of-N restarts — its *returned* path is
the argmin of many stochastic trials, so a single order's per-step
labels cannot carry the restart structure.  Two distillation arms probe
exactly that:

* ``oe-best`` — supervise on the *selected* order only (what
  ``oe_random_greedy_order`` returns), the experiment the retro
  proposed.
* ``oe-all`` — supervise on **every trial's** trajectory (collected via
  ``RandomGreedy.setup``'s trial function + ``ssa_to_linear``): the
  policy then learns the teacher's per-step *distribution*, the closest
  a single-pass policy can get to "sample like the teacher".

They are compared against the two existing regimes at equal eval:

* ``rl`` — REINFORCE curriculum (the bundled-artifact regime retrained
  against the current feature contract).
* ``dp`` — the existing imitation trainer's teacher: exact DP where
  affordable (``--dp-max``), our best-of-restart order beyond it.
  ``dp-small`` (opt-in) is the pure-DP control: only the scales where
  the DP is affordable, so every label is exact-optimal — it isolates
  teacher strength from the restart-fallback labels.

Eval is the einsum ladder (``contraction_einsum`` machinery) at n =
20/30/40 under **both** cost models, at three equalities: single-pass
(argmax rollout vs ``oe-greedy`` vs a 1-episode ``RandomGreedy``),
equal rollouts (best-of-N vs ``RandomGreedy(max_repeats=N)``), and
equal wall-clock (anytime players vs ``max_time=budget``).  A held-out
agreement section walks the teacher's order on unseen n = 40 boards and
reports top-1 agreement plus a Spearman rank correlation between the
policy's pair scores and the negated pairwise cost (the cost-surface
ranking both teachers optimise).

``opt_einsum`` lives in the opt-in ``einsum`` dependency group; the
tool exits with a sync hint (via ``contraction_einsum``) when absent.

Usage::

    uv sync --group einsum
    .venv/bin/python tools/contraction_distill.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import contraction_einsum as ce
import contraction_policy as cp
import contraction_scale as cs
import torch
from catopt_torch import contraction_policy as m
from catopt_torch.contraction_policy import (
    PairPolicyNet,
    random_bond_network,
    rollout_orders,
    save_contraction_policy,
)
from opt_einsum import paths as oe_paths
from opt_einsum.path_random import RandomGreedy
from torch import nn
from torch.nn import functional as F

__all__ = ["main"]

#: The settled curriculum (``contraction-train-scale.md`` §3).
_CURRICULUM = (8, 12, 16, 20, 24)

#: Boards per training scale for the supervised arms.
_PER_N = 24

#: Training-board seed offset (kept disjoint from eval seeds).
_TRAIN_SEED = 50_000

#: Held-out agreement-board seed offset.
_AGREE_SEED = 60_000


def _git_sha() -> str | None:
    """Best-effort checkout SHA for a saved artifact's provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
    except OSError:
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _save_arms(
    models: dict[str, nn.Module],
    stem: str,
    args: argparse.Namespace,
    train_ns: tuple[int, ...],
) -> None:
    """Write each trained arm to ``<stem>-<arm>.pt`` as an artifact.

    The payload is the shipped ``save_contraction_policy`` format, so
    ``load_contraction_policy`` (and every tool that takes a weights
    path, e.g. ``contraction_guided_restart --policy``) reloads it.
    """
    teachers = {
        "oe-best": "oe-rand-greedy (argmin order)",
        "oe-all": "oe-rand-greedy (all trial trajectories)",
        "dp": "exact DP + restart fallback",
        "dp-small": "exact DP",
        "rl": "self-play REINFORCE",
    }
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        out = base.with_name(f"{base.name}-{name}.pt")
        save_contraction_policy(
            model,
            out,
            meta={
                "trainer": f"distill-{name}",
                "train_scales": list(train_ns),
                "epochs": args.epochs,
                "teacher": teachers.get(name, name),
                "teacher_repeats": args.teacher_repeats,
                "teacher_budget": args.teacher_budget,
                "trial_cap": args.trial_cap,
                "seed": args.seed,
                "family": "random_bond_network",
                "git_sha": _git_sha(),
                "torch": str(torch.__version__),
                "created": time.strftime("%Y-%m-%d"),
            },
        )
        print(f"saved {out} ({out.stat().st_size / 1024:.1f} KiB)")


# ---------------------------------------------------------------------------
#  oe-rand-greedy trajectories: the selected order and every trial's
# ---------------------------------------------------------------------------


def _oe_trials(
    e: ce.Einsum, repeats: int, budget: float
) -> list[tuple[list[tuple[int, int]], float]]:
    """Run ``RandomGreedy``'s trial loop; return ``(order, cost)`` pairs.

    ``RandomGreedy.__call__`` keeps only the argmin trial; this drives
    the same trial function directly (same per-repeat seeding, same
    deterministic first trial) so *every* trajectory is available for
    the ``oe-all`` arm, plus the argmin for ``oe-best``.  ``repeats``
    caps trials, ``budget`` caps wall seconds — the same stop rule the
    real optimiser uses.
    """
    opt = RandomGreedy(max_repeats=repeats, max_time=budget)
    fn, args = opt.setup(list(e.inputs), e.output, e.size_dict)
    t0 = time.perf_counter()
    out = []
    for r in range(repeats):
        ssa, cost, _size = fn(r, *args)
        order = [tuple(p) for p in oe_paths.ssa_to_linear(ssa)]
        out.append((order, float(cost)))
        if time.perf_counter() - t0 >= budget:
            break
    return out


def _oe_rand_1_order(e: ce.Einsum) -> list[tuple[int, int]]:
    """Return a single ``RandomGreedy`` episode — its trial 0 (det.)."""
    opt = RandomGreedy(max_repeats=1)
    path = opt(e.inputs, e.output, e.size_dict)
    return [tuple(p) for p in path]


def _oe_rand_n_order(
    e: ce.Einsum, repeats: int
) -> list[tuple[int, int]]:
    """``RandomGreedy(max_repeats=N)`` — best of exactly N episodes."""
    opt = RandomGreedy(max_repeats=repeats)
    path = opt(e.inputs, e.output, e.size_dict)
    return [tuple(p) for p in path]


# ---------------------------------------------------------------------------
#  Datasets and the supervised trainer
# ---------------------------------------------------------------------------


def _boards(
    ns: tuple[int, ...], per_n: int, seed: int
) -> list[tuple[Any, dict[int, int]]]:
    """Draw the shared training boards — identical across distill arms."""
    out = []
    for n in ns:
        for k in range(per_n):
            out.append(random_bond_network(n, seed + 1000 * n + k))
    return out


def _replay_dataset(
    boards: list[tuple[Any, dict[int, int]]],
    orders_per_board: list[list[list[tuple[int, int]]]],
) -> list[tuple[Any, tuple[int, int]]]:
    """Replay per-board teacher orders into ``(state, action)`` rows."""
    data: list[tuple[Any, tuple[int, int]]] = []
    for (tensors, sizes), orders in zip(
        boards, orders_per_board, strict=True
    ):
        for order in orders:
            data.extend(
                cp._replay_samples(tensors, sizes, order, len(tensors))
            )
    return data


def _dp_dataset(
    boards: list[tuple[Any, dict[int, int]]],
    *,
    dp_max: int,
    restarts: int,
    top_k: int,
    seed: int,
) -> list[tuple[Any, tuple[int, int]]]:
    """Replay the existing imitation teacher's orders (DP + fallback)."""
    teacher = cp._make_teacher("dp", dp_max, restarts, top_k)
    rng = random.Random(seed)
    orders = [teacher(tensors, sizes, rng) for tensors, sizes in boards]
    return _replay_dataset(boards, [[o] for o in orders])


def _fit_supervised(
    data: list[tuple[Any, tuple[int, int]]],
    ns: tuple[int, ...],
    rng: random.Random,
    *,
    epochs: int,
    batch: int,
    hidden: int,
    lr: float,
    device: str,
    seed: int,
) -> nn.Module:
    """Masked cross-entropy on teacher labels — ``train_imitation``'s loss."""
    torch.manual_seed(seed)
    model = PairPolicyNet(hidden).to(device)
    cp._fit_norm(model, ns, rng, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        rng.shuffle(data)
        for start in range(0, len(data), batch):
            chunk = data[start : start + batch]
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
    return model.eval()


# ---------------------------------------------------------------------------
#  Held-out agreement: top-1 on the teacher's walk + cost-rank Spearman
# ---------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks of ``xs`` (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2.0
        i = j + 1
    return rk


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation; ``nan`` when a side is degenerate."""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da * db == 0:
        return float("nan")
    return num / (da * db)


def _policy_scores(
    model: nn.Module, game: Any, device: str
) -> list[float]:
    """Return the policy's per-pair scores at ``game``'s state."""
    sf, pf = cp._batch_inputs([game], device)
    with torch.no_grad():
        logits = cp._logits(model, sf, pf)
    return logits[0].tolist()


def _agreement_section(
    models: dict[str, nn.Module],
    device: str,
    n: int,
    boards: int,
    seed: int,
    teacher_budget: float,
) -> None:
    """Walk the teacher's order on held-out boards; score each policy.

    Per state: top-1 (did the policy argmax the teacher's pair) and the
    Spearman correlation between the policy's pair scores and the
    negated pairwise cost — the ranking the teacher optimises.  Also
    ``roll/teacher``: the policy's own greedy-rollout cost over the
    teacher order's cost (the retro's 0.82/2.43 metric).
    """
    print()
    print(
        f"== held-out agreement: walk the oe-rand-greedy order (n={n}) =="
    )
    print(
        "  top-1 = fraction of teacher pairs the policy argmaxes;  "
        "rho = Spearman(policy scores, -pair cost)"
    )
    names = ["our-greedy", *models]
    top1: dict[str, list[float]] = {k: [] for k in names}
    rho: dict[str, list[float]] = {k: [] for k in names}
    roll: dict[str, list[float]] = {k: [] for k in models}
    for k in range(boards):
        tensors, sizes = random_bond_network(n, seed + k)
        e = ce.to_einsum(tensors, sizes)
        order = ce.oe_random_greedy_order(e, teacher_budget)
        teacher_cost = ce.our_cost_of_order(tensors, sizes, order)
        ref = cs.greedy(tensors, sizes)
        ts = [frozenset(t) for t in tensors]
        for a, b in order:
            game = cp.ContractionGame(ts, sizes, ref, n0=n)
            target = (min(a, b), max(a, b))
            pairs = list(game.pairs)
            neg = [-cs.pair_cost(ts[x], ts[y], sizes) for x, y in pairs]
            cheapest = min(
                pairs,
                key=lambda p: cs.pair_cost(ts[p[0]], ts[p[1]], sizes),
            )
            top1["our-greedy"].append(cheapest == target)
            for name, model in models.items():
                scores = _policy_scores(model, game, device)
                pick = pairs[
                    max(range(len(pairs)), key=scores.__getitem__)
                ]
                top1[name].append(pick == target)
                # A 1- or 2-pair state has a trivial rank correlation
                # (undefined / always +-1); score only real choices.
                if len(pairs) >= 3:
                    rho[name].append(_spearman(scores, neg))
            ts = cp._merge_ts(ts, a, b)
        for name, model in models.items():
            _ords, costs = rollout_orders(
                model, tensors, sizes, ref, 1, device, 1.0, greedy=True
            )
            roll[name].append(costs[0] / teacher_cost)
    print(f"  {'player':>12} {'top-1':>7} {'rho':>7} {'roll/tch':>9}")
    print("-" * 39)
    for name in names:
        r = statistics.fmean(rho[name]) if rho[name] else float("nan")
        tail = (
            f"{statistics.fmean(roll[name]):>9.3f}"
            if name in roll
            else f"{'-':>9}"
        )
        print(
            f"  {name:>12} {statistics.fmean(top1[name]):>7.3f} "
            f"{r:>7.3f} {tail}"
        )
    print(
        "  (roll/tch < 1 => the policy's own rollout beats the order "
        "it imitates)"
    )


# ---------------------------------------------------------------------------
#  The ladder: single-pass, equal rollouts, equal wall-clock
# ---------------------------------------------------------------------------


@dataclass
class _Res:
    """One player's outcome on a board: order, both costs, work."""

    order: list[tuple[int, int]] | None
    our: float
    oe: float
    work: float


def _scored(
    tensors: Any,
    sizes: dict[int, int],
    e: ce.Einsum,
    order: list[tuple[int, int]] | None,
    work: float,
) -> _Res:
    """Score an order under both cost models (inf when missing)."""
    if order is None:
        return _Res(None, float("inf"), float("inf"), work)
    return _Res(
        order,
        ce.our_cost_of_order(tensors, sizes, order),
        ce.oe_cost_of_order(e, order),
        work,
    )


def _policy_order(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    ref: float,
    samples: int,
    device: str,
    *,
    greedy_flag: bool,
) -> list[tuple[int, int]]:
    """Best (or argmax, when ``greedy_flag``) policy-rollout order."""
    orders, costs = rollout_orders(
        model,
        tensors,
        sizes,
        ref,
        samples,
        device,
        cp._TEMP,
        greedy=greedy_flag,
    )
    return orders[min(range(len(costs)), key=costs.__getitem__)]


def _single_pass(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
) -> list[tuple[int, list[dict[str, _Res]]]]:
    """One rollout each: policy argmax vs oe-greedy vs 1-episode rand."""
    out: list[tuple[int, list[dict[str, _Res]]]] = []
    for n in scales:
        boards = []
        for k in range(instances):
            tensors, sizes = random_bond_network(n, seed + k)
            e = ce.to_einsum(tensors, sizes)
            ref = cs.greedy(tensors, sizes)
            found = {
                "our-greedy": _scored(
                    tensors,
                    sizes,
                    e,
                    ce.our_greedy_order(tensors, sizes),
                    n - 1,
                ),
                "oe-greedy": _scored(
                    tensors, sizes, e, ce.oe_greedy_order(e), n - 1
                ),
                "oe-rand-1": _scored(
                    tensors, sizes, e, _oe_rand_1_order(e), n - 1
                ),
            }
            for name, model in models.items():
                order = _policy_order(
                    model,
                    tensors,
                    sizes,
                    ref,
                    1,
                    device,
                    greedy_flag=True,
                )
                found[name] = _scored(tensors, sizes, e, order, 1)
            boards.append(found)
        out.append((n, boards))
    return out


def _equal_rollouts(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
    repeats: int,
) -> list[tuple[int, list[dict[str, _Res]]]]:
    """Best-of-N: sampled policy rollouts vs ``max_repeats=N`` trials."""
    out: list[tuple[int, list[dict[str, _Res]]]] = []
    for n in scales:
        boards = []
        for k in range(instances):
            tensors, sizes = random_bond_network(n, seed + k)
            e = ce.to_einsum(tensors, sizes)
            ref = cs.greedy(tensors, sizes)
            rng = random.Random(seed + k)
            found = {
                "our-greedy": _scored(
                    tensors,
                    sizes,
                    e,
                    ce.our_greedy_order(tensors, sizes),
                    1,
                ),
                "our-restart": _scored(
                    tensors,
                    sizes,
                    e,
                    cp._restart_order(tensors, sizes, rng, repeats, 3),
                    repeats,
                ),
                "oe-greedy": _scored(
                    tensors, sizes, e, ce.oe_greedy_order(e), 1
                ),
                "oe-rand-greedy": _scored(
                    tensors,
                    sizes,
                    e,
                    _oe_rand_n_order(e, repeats),
                    repeats,
                ),
            }
            for name, model in models.items():
                order = _policy_order(
                    model,
                    tensors,
                    sizes,
                    ref,
                    repeats,
                    device,
                    greedy_flag=False,
                )
                found[name] = _scored(tensors, sizes, e, order, repeats)
            boards.append(found)
        out.append((n, boards))
    return out


def _equal_clock(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
    budgets: tuple[float, ...],
) -> list[tuple[float, int, list[dict[str, _Res]]]]:
    """Anytime players under a per-instance wall-clock budget."""
    out: list[tuple[float, int, list[dict[str, _Res]]]] = []
    for budget in budgets:
        for n in scales:
            boards = []
            for k in range(instances):
                tensors, sizes = random_bond_network(n, seed + k)
                e = ce.to_einsum(tensors, sizes)
                ref = cs.greedy(tensors, sizes)
                order, _s, rolls = ce.our_restart_order(
                    tensors, sizes, budget, seed + k
                )
                found = {
                    "our-greedy": _scored(
                        tensors,
                        sizes,
                        e,
                        ce.our_greedy_order(tensors, sizes),
                        1,
                    ),
                    "our-restart": _scored(
                        tensors, sizes, e, order, rolls
                    ),
                    "oe-greedy": _scored(
                        tensors, sizes, e, ce.oe_greedy_order(e), 1
                    ),
                    "oe-rand-greedy": _scored(
                        tensors,
                        sizes,
                        e,
                        ce.oe_random_greedy_order(e, budget),
                        float("nan"),
                    ),
                }
                for name, model in models.items():
                    prior = ce._policy_prior(
                        model, tensors, sizes, device
                    )
                    order, _s, rolls = ce.policy_best_order(
                        model,
                        tensors,
                        sizes,
                        ref,
                        budget,
                        device,
                        prior,
                        seed + k,
                    )
                    found[name] = _scored(
                        tensors, sizes, e, order, rolls
                    )
                boards.append(found)
            out.append((budget, n, boards))
    return out


def _ladder_table(
    label: str,
    data: list[tuple[str, int, list[dict[str, _Res]]]],
    models: dict[str, nn.Module],
) -> None:
    """Print a ratio-to-best-found ladder under both cost models."""
    present = {p for _t, _n, bs in data for d in bs for p in d}
    players = [
        p
        for p in (
            "our-greedy",
            "our-restart",
            *models,
            "oe-greedy",
            "oe-rand-1",
            "oe-rand-greedy",
        )
        if p in present
    ]
    for tag in sorted({t for t, _n, _d in data}):
        print()
        print(f"== {label}: {tag} ==")
        print(
            f"  {'n':>3} {'player':>15} {'our-ratio':>10} "
            f"{'oe-ratio':>10} {'work':>9}"
        )
        print("-" * 51)
        for t, n, boards in data:
            if t != tag:
                continue
            for p in players:
                our = [
                    d[p].our / min(r.our for r in d.values())
                    for d in boards
                    if p in d
                ]
                oe = [
                    d[p].oe / min(r.oe for r in d.values())
                    for d in boards
                    if p in d
                ]
                work = [d[p].work for d in boards if p in d]
                print(
                    f"  {n:>3} {p:>15} "
                    f"{statistics.fmean(our):>10.3f} "
                    f"{statistics.fmean(oe):>10.3f} "
                    f"{ce._num(statistics.fmean(work), 9, '.1f')}"
                )


def _pairwise(
    label: str,
    data: list[tuple[str, int, list[dict[str, _Res]]]],
    models: dict[str, nn.Module],
) -> None:
    """Head-to-head learned/opt_einsum ratios under both cost models."""
    cols = ["oe-greedy", "oe-rand-1", "oe-rand-greedy"]
    cols = [
        c
        for c in cols
        if any(c in d for _t, _n, bs in data for d in bs)
    ]
    print()
    print(
        f"== {label}: learned / opt_einsum pairwise "
        "(mean; <1 = learned wins) =="
    )
    head = (
        f"  {'tag':>10} {'n':>3} {'player':>15} "
        + " ".join(f"{'our/' + c:>16}" for c in cols)
        + " "
        + " ".join(f"{'oe/' + c:>16}" for c in cols)
    )
    print(head)
    print("-" * len(head))

    def _cell(vals: list[float]) -> str:
        finite = [v for v in vals if v == v]
        return (
            ce._num(statistics.fmean(finite), 16)
            if finite
            else f"{'DNF':>16}"
        )

    for t, n, boards in data:
        for name in models:
            our_r: dict[str, list[float]] = {c: [] for c in cols}
            oe_r: dict[str, list[float]] = {c: [] for c in cols}
            for d in boards:
                learned = d[name]
                for c in cols:
                    our_r[c].append(
                        learned.our / d[c].our
                        if d[c].our < float("inf")
                        else float("nan")
                    )
                    oe_r[c].append(
                        learned.oe / d[c].oe
                        if d[c].oe < float("inf")
                        else float("nan")
                    )
            print(
                f"  {t:>10} {n:>3} {name:>15} "
                + " ".join(_cell(our_r[c]) for c in cols)
                + " "
                + " ".join(_cell(oe_r[c]) for c in cols)
            )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Train four policies on the curriculum, then compare at equal eval."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument(
        "--train-scales",
        default=",".join(str(n) for n in _CURRICULUM),
        help="comma-separated curriculum n (the settled regime)",
    )
    ap.add_argument("--per-n", type=int, default=_PER_N)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--iterations",
        type=int,
        default=1800,
        help="REINFORCE iterations for the rl arm",
    )
    ap.add_argument(
        "--rl-batch",
        type=int,
        default=24,
        help="episode batch for the rl arm",
    )
    ap.add_argument(
        "--teacher-repeats",
        type=int,
        default=128,
        help="RandomGreedy trials per training board (a real budget)",
    )
    ap.add_argument(
        "--teacher-budget",
        type=float,
        default=0.75,
        help="seconds cap for the training-board teacher (and the "
        "held-out agreement teacher)",
    )
    ap.add_argument(
        "--trial-cap",
        type=int,
        default=8,
        help="max trial trajectories per board for the oe-all arm",
    )
    ap.add_argument("--dp-max", type=int, default=14)
    ap.add_argument("--dp-restarts", type=int, default=32)
    ap.add_argument("--dp-top-k", type=int, default=3)
    ap.add_argument(
        "--arms",
        default="rl,oe-best,oe-all,dp",
        help="comma-separated subset of rl,oe-best,oe-all,dp,dp-small",
    )
    ap.add_argument(
        "--scales",
        default="20,30,40",
        help="comma-separated eval n",
    )
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--agree-n", type=int, default=40)
    ap.add_argument("--agree-boards", type=int, default=8)
    ap.add_argument(
        "--rollouts",
        type=int,
        default=64,
        help="N for the equal-rollouts ladder",
    )
    ap.add_argument(
        "--budgets",
        default="200,1000",
        help="comma-separated per-instance wall-clock budgets (ms)",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="artifact stem: each trained arm is serialised to "
        "<save>-<arm>.pt in the shipped artifact format (loadable "
        "via load_contraction_policy / --policy paths)",
    )
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_ns = tuple(
        int(x) for x in args.train_scales.split(",") if x.strip()
    )
    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    budgets = tuple(
        1e-3 * float(x) for x in args.budgets.split(",") if x.strip()
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"device: {dev}  (torch {torch.__version__})")
    print(
        f"feature contract: {m._FEATURE_CONTRACT} "
        f"(STATE_DIM={m.STATE_DIM}, PAIR_DIM={m.PAIR_DIM})"
    )
    print(f"train scales: {train_ns}  arms: {arms}")
    print(
        f"oe teacher: RandomGreedy repeats<={args.teacher_repeats}, "
        f"max_time={args.teacher_budget}s"
    )

    t0 = time.perf_counter()
    boards = _boards(train_ns, args.per_n, _TRAIN_SEED + args.seed)
    models: dict[str, nn.Module] = {}
    rng = random.Random(args.seed)

    with ce._on_family(random_bond_network):
        if "rl" in arms:
            t = time.perf_counter()
            models["rl"] = cp.train_rl(
                ns=train_ns,
                iterations=args.iterations,
                batch=args.rl_batch,
                hidden=args.hidden,
                device=dev,
                seed=args.seed,
            )
            print(f"trained rl in {time.perf_counter() - t:.1f}s")
        sup_args = {
            "epochs": args.epochs,
            "batch": args.batch,
            "hidden": args.hidden,
            "lr": args.lr,
            "device": dev,
            "seed": args.seed,
        }
        if "oe-best" in arms or "oe-all" in arms:
            t = time.perf_counter()
            trials = [
                _oe_trials(
                    ce.to_einsum(tensors, sizes),
                    args.teacher_repeats,
                    args.teacher_budget,
                )
                for tensors, sizes in boards
            ]
            counts = [len(tr) for tr in trials]
            print(
                f"oe teacher trials in {time.perf_counter() - t:.1f}s "
                f"(mean {statistics.fmean(counts):.0f}/board)"
            )
            datasets: dict[str, list[Any]] = {}
            if "oe-best" in arms:
                datasets["oe-best"] = _replay_dataset(
                    boards,
                    [[min(tr, key=lambda x: x[1])[0]] for tr in trials],
                )
            if "oe-all" in arms:
                datasets["oe-all"] = _replay_dataset(
                    boards,
                    [
                        [order for order, _c in tr[: args.trial_cap]]
                        for tr in trials
                    ],
                )
            for name, data in datasets.items():
                print(f"  {name} dataset: {len(data)} rows")
                t = time.perf_counter()
                models[name] = _fit_supervised(
                    data, train_ns, rng, **sup_args
                )
                print(
                    f"trained {name} in {time.perf_counter() - t:.1f}s"
                )
        if "dp" in arms:
            t = time.perf_counter()
            data = _dp_dataset(
                boards,
                dp_max=args.dp_max,
                restarts=args.dp_restarts,
                top_k=args.dp_top_k,
                seed=args.seed,
            )
            print(
                f"dp dataset in {time.perf_counter() - t:.1f}s: "
                f"{len(data)} rows"
            )
            t = time.perf_counter()
            models["dp"] = _fit_supervised(
                data, train_ns, rng, **sup_args
            )
            print(f"trained dp in {time.perf_counter() - t:.1f}s")
        if "dp-small" in arms:
            # Pure-DP control: only the scales where the teacher is the
            # exact DP (no restart fallback), i.e. the shipped imitation
            # recipe — isolates teacher strength from scale coverage.
            small_ns = tuple(n for n in train_ns if n <= args.dp_max)
            small = [b for b in boards if len(b[0]) <= args.dp_max]
            t = time.perf_counter()
            data = _dp_dataset(
                small,
                dp_max=args.dp_max,
                restarts=args.dp_restarts,
                top_k=args.dp_top_k,
                seed=args.seed,
            )
            print(
                f"dp-small dataset (ns={small_ns}) in "
                f"{time.perf_counter() - t:.1f}s: {len(data)} rows"
            )
            t = time.perf_counter()
            models["dp-small"] = _fit_supervised(
                data, small_ns, rng, **sup_args
            )
            print(f"trained dp-small in {time.perf_counter() - t:.1f}s")
    print(f"all training done in {time.perf_counter() - t0:.1f}s")

    if args.save:
        _save_arms(models, args.save, args, train_ns)

    _agreement_section(
        models,
        dev,
        args.agree_n,
        args.agree_boards,
        _AGREE_SEED + args.seed,
        args.teacher_budget,
    )

    single = _single_pass(
        scales, args.instances, args.seed, models, dev
    )
    _ladder_table(
        "equal single-pass",
        [("1 rollout", n, b) for n, b in single],
        models,
    )
    _pairwise(
        "equal single-pass",
        [("1 rollout", n, b) for n, b in single],
        models,
    )

    rolled = _equal_rollouts(
        scales, args.instances, args.seed, models, dev, args.rollouts
    )
    _ladder_table(
        "equal rollouts",
        [(f"best-of-{args.rollouts}", n, b) for n, b in rolled],
        models,
    )
    _pairwise(
        "equal rollouts",
        [(f"best-of-{args.rollouts}", n, b) for n, b in rolled],
        models,
    )

    clock = _equal_clock(
        scales, args.instances, args.seed, models, dev, budgets
    )
    _ladder_table(
        "equal wall-clock",
        [(f"{1e3 * b:.0f} ms", n, d) for b, n, d in clock],
        models,
    )
    _pairwise(
        "equal wall-clock",
        [(f"{1e3 * b:.0f} ms", n, d) for b, n, d in clock],
        models,
    )

    print()
    print("== verdict ==")
    print(
        "  compare each distilled arm's pairwise vs oe-rand-greedy "
        "with rl's: <1.04 means the stronger teacher transferred."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
