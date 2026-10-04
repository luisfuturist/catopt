"""Policy-guided restarts vs uniform restarts — become the baseline.

``project/retros/contraction-train-scale.md`` left the learned player
at ~1.04x of ``opt_einsum``'s randomised greedy at n = 40 (adequate
budget) and diagnosed the residual gap as *algorithmic*:
``oe-rand-greedy`` is best-of-N random restarts over a staged
heuristic, and no single forward pass matches a search procedure.  The
direct response is to **become the restart**: run rollout episodes
that sample each contraction pair from the policy's own softmax
distribution — a temperature knob — and keep the best order so far.

The shipped machinery already implements the move:
:func:`catopt_torch.contraction_policy.rollout_orders` with
``greedy=False`` samples from ``Categorical(logits / temperature)``,
so a "policy-guided restart" is exactly best-of-N sampled rollouts.
This tool adds what the earlier ladders did not isolate:

* a **temperature sweep** (default ``{0.5, 1.0, 2.0}``) on the
  budgeted player;
* an **affine batch scheduler** — lockstep rollouts cost
  ``fixed + marginal x batch`` (~190 ms + ~6 ms/rollout at n = 40 on
  this box), not the purely linear ``ms/rollout`` model
  ``contraction_einsum.policy_best_order`` sizes with, so the guided
  player fits the affine model per scale and fills the budget with
  maximal lockstep batches — a strictly better scheduling of the same
  algorithm (``guided-lin`` is the shipped linear-sized player, kept
  as the ablation);
* a **uniform-restart control** — the same best-of-N loop with a
  uniform top-k proposal (``our_restart_order``), which isolates the
  policy's contribution to the restart *algorithm*;
* an **equal-rollout-count arm** — every player re-run at exactly the
  rollout count the guided player achieved under the clock, which
  separates per-rollout *quality* from rollouts-per-ms *throughput*;
* the learned **single-pass** argmax order and ``oe-greedy`` as the
  deterministic references.

Instances are the einsum-valid ``random_bond_network`` family (the one
``opt_einsum`` prices) and every order is scored by
``opt_einsum.contract_path`` — the independent metric — with our
pairwise cost model alongside.  The bundled distilled artifact
(:func:`load_contraction_policy`'s default) is the policy under test.

**Honest caveat.**  A uniform/oe rollout is a heuristic scan; a guided
rollout costs a forward pass per contraction step, so the guided
player completes *fewer* restarts in the same milliseconds.  Whether
better rollouts beat more rollouts is precisely what is measured —
rollouts-per-ms is reported next to quality.  One asymmetry is kept
deliberately: the uniform and oe restarts count a free deterministic
greedy as episode 0 while the guided player is all-sampled, so any
guided win is *despite* forgoing the free seeding.

Usage::

    uv sync --group einsum
    .venv/bin/python tools/contraction_guided_restart.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Any

import contraction_einsum as ce
import contraction_scale as cs
import torch
from catopt_torch.contraction_policy import (
    MAX_BATCH,
    load_contraction_policy,
    random_bond_network,
    rollout_orders,
)

__all__ = ["main"]

#: Softmax temperatures swept by the guided-restart player.
_TEMPS = (0.5, 1.0, 2.0)


def _tname(t: float) -> str:
    """Player name for the budgeted guided restart at temperature t."""
    return f"guided-T{t:g}"


def _tfixed(t: float) -> str:
    """Player name for the equal-rollouts arm at temperature t."""
    return f"gfix-T{t:g}"


# ---------------------------------------------------------------------------
#  Players
# ---------------------------------------------------------------------------


def _oe_timed(e: ce.Einsum, budget: float) -> tuple[Any, int]:
    """opt_einsum rand-greedy filling ``budget``; return (order, repeats)."""
    opt = ce.RandomGreedy(max_repeats=ce._HUGE, max_time=budget)
    path = opt(e.inputs, e.output, e.size_dict)
    return [tuple(p) for p in path], len(opt.costs)


def _oe_fixed(e: ce.Einsum, rollouts: int) -> tuple[Any, int]:
    """opt_einsum rand-greedy for exactly ``rollouts`` trials.

    ``max_time=None`` runs the full repeat range; trial 0 is the plain
    deterministic greedy and each later trial is seeded by its index,
    so the run is deterministic for a given count.
    """
    opt = ce.RandomGreedy(max_repeats=max(1, rollouts), max_time=None)
    path = opt(e.inputs, e.output, e.size_dict)
    return [tuple(p) for p in path], len(opt.costs)


def _uniform_fixed(
    tensors: Any,
    sizes: dict[int, int],
    rng: random.Random,
    rollouts: int,
    top_k: int = 3,
) -> list[tuple[int, int]]:
    """Best of ``rollouts`` episodes: greedy + uniform top-k samples.

    Mirrors ``our_restart_order``'s accounting (the deterministic
    greedy is episode 0, the rest are uniform top-k samples) so the
    equal-rollout arm counts episodes the same way the clocked arm
    does — and the same way ``oe`` counts its trial 0.
    """
    best = ce.our_greedy_order(tensors, sizes)
    best_cost = ce.our_cost_of_order(tensors, sizes, best)
    for _ in range(max(0, rollouts - 1)):
        order = ce._random_greedy_order(tensors, sizes, rng, top_k)
        cost = ce.our_cost_of_order(tensors, sizes, order)
        if cost < best_cost:
            best, best_cost = order, cost
    return best


def _guided_fixed(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    rollouts: int,
    device: str,
    temperature: float,
    seed: int,
) -> list[tuple[int, int]]:
    """Best of exactly ``rollouts`` policy-sampled episodes."""
    torch.manual_seed(seed)
    best: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    done = 0
    while done < rollouts:
        batch = min(MAX_BATCH, rollouts - done)
        orders, costs = ce._policy_rollouts(
            model,
            tensors,
            sizes,
            greedy_ref,
            batch,
            device,
            temperature,
        )
        k = min(range(len(costs)), key=costs.__getitem__)
        if costs[k] < best_cost:
            best, best_cost = orders[k], costs[k]
        done += batch
    assert best is not None
    return best


def _single_pass_order(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    device: str,
) -> list[tuple[int, int]]:
    """One deterministic argmax pass — the single-pass player."""
    orders, _costs = rollout_orders(
        model, tensors, sizes, greedy_ref, 1, device, 1.0, greedy=True
    )
    return orders[0]


def _guided_prior(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    device: str,
) -> tuple[float, float]:
    """Estimate the affine batch cost ``f + m * B`` for this board.

    A lockstep batch of ``B`` episodes costs one fixed round of
    forward passes (the game steps) plus a per-episode marginal (the
    sequential Python ``step`` updates) — measured as ``f, m`` by
    timing two batch sizes.  ``policy_best_order``'s linear
    ``ms/rollout`` prior absorbs ``f`` into ``m`` and so under-fills
    every budget that is not already starving.  ``f`` is floored by
    what the timed batch-1 probe implies — the warmup call absorbs
    one-off CUDA costs, which otherwise leak into ``f`` as an
    underestimate and let the first sized batch overshoot the budget.
    """
    ce._policy_rollouts(
        model, tensors, sizes, greedy_ref, 4, device, 1.0
    )
    timed: list[tuple[int, float]] = []
    for b in (8, 48, 1):
        t0 = time.perf_counter()
        ce._policy_rollouts(
            model, tensors, sizes, greedy_ref, b, device, 1.0
        )
        timed.append((b, time.perf_counter() - t0))
    bs = [b for b, _t in timed]
    mb = statistics.fmean(bs)
    mt = statistics.fmean(t for _b, t in timed)
    var = statistics.fmean([(b - mb) ** 2 for b in bs])
    m = max(
        statistics.fmean([(b - mb) * (t - mt) for b, t in timed]) / var,
        1e-6,
    )
    f = max(max(t - m * b for b, t in timed), 0.0)
    return f, m


def _affine_fit(
    obs: list[tuple[int, float]], f0: float, m0: float
) -> tuple[float, float]:
    """Least-squares refit of ``f + m * B`` on observed batches."""
    bs = [b for b, _t in obs]
    if len(set(bs)) < 2:
        return f0, m0
    mb = statistics.fmean(bs)
    mt = statistics.fmean(t for _b, t in obs)
    var = statistics.fmean([(b - mb) ** 2 for b in bs])
    if var <= 0:
        return f0, m0
    m = statistics.fmean([(b - mb) * (t - mt) for b, t in obs]) / var
    f = mt - m * mb
    if m <= 0 or f < 0:
        return f0, m0
    return f, m


def _guided_best_order(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    budget: float,
    device: str,
    f0: float,
    m0: float,
    temperature: float,
    seed: int,
) -> tuple[list[tuple[int, int]], float, int]:
    """Anytime guided restart: best sampled order within ``budget``.

    Same contract as ``policy_best_order`` — fill the wall-clock
    budget with best-of-N sampled rollouts — but each lockstep batch
    is sized from a refitted affine model ``f + m * B`` of the true
    batch cost, so the whole remaining budget is spent on one maximal
    batch instead of a dribble of small ones.  When the fixed cost of
    a single batch already exceeds the remaining budget, one rollout
    is run anyway (the same first-batch overshoot the shipped protocol
    accepts) and the loop stops.
    """
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    best: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    rollouts = 0
    obs: list[tuple[int, float]] = []
    while True:
        remaining = budget - (time.perf_counter() - t0)
        f, m = _affine_fit(obs, f0, m0)
        if remaining >= f + m:
            batch = min(MAX_BATCH, max(1, int((remaining - f) / m)))
        elif rollouts == 0:
            batch = 1
        else:
            break
        tb = time.perf_counter()
        orders, costs = ce._policy_rollouts(
            model,
            tensors,
            sizes,
            greedy_ref,
            batch,
            device,
            temperature,
        )
        obs.append((batch, time.perf_counter() - tb))
        rollouts += batch
        k = min(range(len(costs)), key=costs.__getitem__)
        if costs[k] < best_cost:
            best, best_cost = orders[k], costs[k]
    assert best is not None
    return best, time.perf_counter() - t0, rollouts


# ---------------------------------------------------------------------------
#  Measurement
# ---------------------------------------------------------------------------


def _run_board(
    tensors: Any,
    sizes: dict[int, int],
    policy: Any,
    device: str,
    budget: float,
    temps: tuple[float, ...],
    seed: int,
    f0: float,
    m0: float,
) -> dict[str, ce._Res]:
    """Run every player on one board at an equal wall-clock budget."""
    e = ce.to_einsum(tensors, sizes)
    ref = cs.greedy(tensors, sizes)
    steps = max(len(tensors) - 1, 0)
    out: dict[str, ce._Res] = {}

    t0 = time.perf_counter()
    order = ce.our_greedy_order(tensors, sizes)
    out["our-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    order, secs, rolls = ce.our_restart_order(
        tensors, sizes, budget, seed
    )
    out["our-restart"] = ce._res(tensors, sizes, e, order, secs, rolls)

    t0 = time.perf_counter()
    order = ce.oe_greedy_order(e)
    out["oe-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    t0 = time.perf_counter()
    order, reps = _oe_timed(e, budget)
    out["oe-rand-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, reps
    )

    t0 = time.perf_counter()
    order = _single_pass_order(
        policy.model, tensors, sizes, ref, device
    )
    out["single-pass"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    prior_lin = ce._policy_prior(policy.model, tensors, sizes, device)
    order, secs, rolls = ce.policy_best_order(
        policy.model,
        tensors,
        sizes,
        ref,
        budget,
        device,
        prior_lin,
        seed,
    )
    out["guided-lin"] = ce._res(tensors, sizes, e, order, secs, rolls)

    for t in temps:
        order, secs, rolls = _guided_best_order(
            policy.model,
            tensors,
            sizes,
            ref,
            budget,
            device,
            f0,
            m0,
            t,
            seed,
        )
        out[_tname(t)] = ce._res(tensors, sizes, e, order, secs, rolls)
    return out


def _measure(
    budgets: tuple[float, ...],
    scales: tuple[int, ...],
    boards: int,
    seed: int,
    policy: Any,
    temps: tuple[float, ...],
    device: str,
) -> list[tuple[float, int, list[dict[str, ce._Res]]]]:
    """Run the wall-clock ladder; return (budget, n, boards) rows.

    The affine batch-cost prior ``(f, m)`` is estimated once per
    scale — it depends on ``n`` (steps per episode), not on the
    budget, the temperature or the particular board — and the online
    refit inside ``_guided_best_order`` absorbs the board-to-board
    drift.
    """
    data: list[tuple[float, int, list[dict[str, ce._Res]]]] = []
    for n in scales:
        t0b, s0b = random_bond_network(n, seed)
        f0, m0 = _guided_prior(
            policy.model, t0b, s0b, cs.greedy(t0b, s0b), device
        )
        for budget in budgets:
            rows = []
            for k in range(boards):
                tensors, sizes = random_bond_network(n, seed + k)
                rows.append(
                    _run_board(
                        tensors,
                        sizes,
                        policy,
                        device,
                        budget,
                        temps,
                        seed + k,
                        f0,
                        m0,
                    )
                )
            data.append((budget, n, rows))
    data.sort()
    return data


def _measure_fixed(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    seed: int,
    policy: Any,
    temps: tuple[float, ...],
    device: str,
) -> list[tuple[float, int, list[tuple[int, dict[str, ce._Res]]]]]:
    """Re-run every restart player at the guided-T1.0 rollout count.

    ``N`` is the number of episodes the guided player at temperature
    1.0 actually completed inside the wall-clock budget on that board
    — so the guided player's fixed arm reproduces its clocked run
    while the heuristic players are *held down* to the same count.
    The comparison isolates per-rollout quality from throughput.
    """
    out: list[
        tuple[float, int, list[tuple[int, dict[str, ce._Res]]]]
    ] = []
    for budget, n, rows in data:
        fixed: list[tuple[int, dict[str, ce._Res]]] = []
        for k, res in enumerate(rows):
            tensors, sizes = random_bond_network(n, seed + k)
            e = ce.to_einsum(tensors, sizes)
            ref = cs.greedy(tensors, sizes)
            rng = random.Random(seed + k)
            roll = max(1, round(res[_tname(1.0)].work))
            r: dict[str, ce._Res] = {}
            t0 = time.perf_counter()
            order = _uniform_fixed(tensors, sizes, rng, roll)
            r["unif-fixed"] = ce._res(
                tensors,
                sizes,
                e,
                order,
                time.perf_counter() - t0,
                roll,
            )
            t0 = time.perf_counter()
            order, reps = _oe_fixed(e, roll)
            r["oe-fixed"] = ce._res(
                tensors,
                sizes,
                e,
                order,
                time.perf_counter() - t0,
                reps,
            )
            for t in temps:
                t0 = time.perf_counter()
                order = _guided_fixed(
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    roll,
                    device,
                    t,
                    seed + k,
                )
                r[_tfixed(t)] = ce._res(
                    tensors,
                    sizes,
                    e,
                    order,
                    time.perf_counter() - t0,
                    roll,
                )
            fixed.append((roll, r))
        out.append((budget, n, fixed))
    return out


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _players(temps: tuple[float, ...]) -> tuple[str, ...]:
    """Return the wall-clock players in report order."""
    return (
        "our-greedy",
        "our-restart",
        "oe-greedy",
        "oe-rand-greedy",
        "single-pass",
        "guided-lin",
        *(_tname(t) for t in temps),
    )


def _ladder(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    temps: tuple[float, ...],
) -> None:
    """Equal-wall-clock ladder: cost ratios, rollouts and r/ms."""
    players = _players(temps)
    for budget in sorted({b for b, _n, _d in data}):
        print()
        print(
            f"== 1. equal wall-clock: {1e3 * budget:.0f} ms per "
            "instance =="
        )
        print(
            f"  {'n':>3} {'player':>15} {'our-ratio':>10} "
            f"{'oe-ratio':>10} {'work':>7} {'ms':>7} {'r/s':>7}"
        )
        print("-" * 68)
        for b, n, boards in data:
            if b != budget:
                continue
            for p in players:
                our = [
                    d[p].our / min(r.our for r in d.values())
                    for d in boards
                ]
                oe = [
                    d[p].oe / min(r.oe for r in d.values())
                    for d in boards
                ]
                work = [d[p].work for d in boards]
                secs = [d[p].secs for d in boards]
                rate = [
                    w / s for w, s in zip(work, secs, strict=True) if s
                ]
                print(
                    f"  {n:>3} {p:>15} "
                    f"{statistics.fmean(our):>10.3f} "
                    f"{statistics.fmean(oe):>10.3f} "
                    f"{statistics.fmean(work):>7.1f} "
                    f"{1e3 * statistics.fmean(secs):>7.1f} "
                    f"{statistics.fmean(rate):>7.2f}"
                )


def _pairwise(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    temps: tuple[float, ...],
) -> None:
    """Guided / baseline oe-cost ratios at equal wall-clock."""
    bases = (
        "oe-rand-greedy",
        "our-restart",
        "oe-greedy",
        "single-pass",
    )
    print()
    print("== 2. pairwise, oe cost (mean; <1 = row player wins) ==")
    head = f"  {'ms':>6} {'n':>3} {'player':>15} " + " ".join(
        f"{'/' + c:>16}" for c in bases
    )
    print(head)
    print("-" * len(head))
    for budget, n, boards in data:
        for p in (
            *(_tname(t) for t in temps),
            "our-restart",
            "single-pass",
        ):
            cells = []
            for c in bases:
                if c == p:
                    cells.append(f"{'-':>16}")
                    continue
                ratios = [
                    d[p].oe / d[c].oe
                    for d in boards
                    if d[c].oe < float("inf")
                ]
                cells.append(
                    f"{statistics.fmean(ratios):>16.3f}"
                    if ratios
                    else f"{'DNF':>16}"
                )
            print(
                f"  {1e3 * budget:>6.0f} {n:>3} {p:>15} "
                + " ".join(cells)
            )


def _fixed_section(
    fixed: list[
        tuple[float, int, list[tuple[int, dict[str, ce._Res]]]]
    ],
    temps: tuple[float, ...],
) -> None:
    """Equal-rollout-count arm: per-rollout quality, oe-cost ratios."""
    print()
    print(
        "== 3. equal rollout count (N = guided-T1 clocked rollouts) =="
    )
    print("  all players re-run at the same episode count per board;")
    print("  ratios are oe costs (mean; <1 = guided wins)")
    head = (
        f"  {'ms':>6} {'n':>3} {'N':>5} "
        + " ".join(f"{f'{t:g}/unif':>9}" for t in temps)
        + "  "
        + " ".join(f"{f'{t:g}/oe':>9}" for t in temps)
        + f"  {'unif/oe':>9}"
    )
    print(head)
    print("-" * len(head))
    for budget, n, rows in fixed:
        ns = [r for r, _d in rows]
        line = f"  {1e3 * budget:>6.0f} {n:>3} {statistics.fmean(ns):>5.0f} "
        cells_u = []
        cells_o = []
        for t in temps:
            ru = [
                d[_tfixed(t)].oe / d["unif-fixed"].oe for _r, d in rows
            ]
            ro = [d[_tfixed(t)].oe / d["oe-fixed"].oe for _r, d in rows]
            cells_u.append(f"{statistics.fmean(ru):>9.3f}")
            cells_o.append(f"{statistics.fmean(ro):>9.3f}")
        ctx = [d["unif-fixed"].oe / d["oe-fixed"].oe for _r, d in rows]
        print(
            line
            + " ".join(cells_u)
            + "  "
            + " ".join(cells_o)
            + f"  {statistics.fmean(ctx):>9.3f}"
        )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the guided-restart ladder on the bundled policy artifact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boards", type=int, default=4)
    ap.add_argument("--scales", default="20,30,40")
    ap.add_argument(
        "--budgets",
        default="50,200,1000",
        help="comma-separated per-instance budgets (ms)",
    )
    ap.add_argument(
        "--temps",
        default=",".join(f"{t:g}" for t in _TEMPS),
        help="comma-separated guided-restart softmax temperatures",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--policy",
        default=None,
        help="policy artifact path (default: bundled distilled weights)",
    )
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_contraction_policy(args.policy, device=dev)
    temps = tuple(float(x) for x in args.temps.split(",") if x.strip())
    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    budgets = tuple(
        1e-3 * float(x) for x in args.budgets.split(",") if x.strip()
    )
    print(f"device: {dev}  (torch {torch.__version__})")
    print(f"policy: {args.policy or 'bundled'}  meta={policy.meta}")
    print(f"boards/cell: {args.boards}  temps: {temps}")

    data = _measure(
        budgets, scales, args.boards, args.seed, policy, temps, dev
    )
    _ladder(data, temps)
    _pairwise(data, temps)
    fixed = _measure_fixed(data, args.seed, policy, temps, dev)
    _fixed_section(fixed, temps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
