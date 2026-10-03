"""Contraction ordering vs opt_einsum — the honest field baseline.

Plan 0016 follow-up.  ``contraction_policy.py`` measured a *learned*
contraction-ordering policy beating **our own** players (``greedy``,
``restart``, bounded ``search``) at equal wall-clock, generalising from
n = 8-12 to n = 20/30/40.  But every baseline was ours, so the win could
be an artefact of how we wrote them.  ``opt_einsum`` ships the players
that matter — its staged ``greedy`` (memory-removed, no outer product
until forced) and the exact ``optimal`` DP — so it is the honest
yardstick.

This tool is a **falsification attempt**.  It runs the learned policy
and ``opt_einsum``'s players on the **same** instances under the **same**
wall-clock budget, and scores *every* player's order with a single
independent metric — ``opt_einsum``'s own ``contract_path`` cost — so
the comparison cannot be tilted by our cost model.  Both cost models
are reported side by side.

Two instance families are used, both **valid einsums**:

* ``random_bond_network`` — a seeded random tensor network in which
  every index appears in at most two tensors (a *bond*) or once (an
  *open* leg), so it maps one-to-one onto an einsum expression.
  ``contraction_scale.random_network`` **cannot** be used: its indices
  repeat up to 11x (a hypergraph), which is not expressible as an
  einsum at all — a finding in itself, reported in section 0.
* ``attention`` / ``bilinear`` / ``mlp_stack`` — small but genuine
  einsums lifted from an attention core and an MLP block.

``opt_einsum`` lives in the opt-in ``einsum`` dependency group; the tool
exits with a sync hint when the group is absent, so the default env is
untouched.

Usage::

    uv sync --group einsum
    .venv/bin/python tools/contraction_einsum.py --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import random
import statistics
import string
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import contraction_policy as cp
import contraction_scale as cs
import numpy as np
import torch

try:
    import opt_einsum as oe
    from opt_einsum.path_random import RandomGreedy
except ImportError as exc:
    raise SystemExit(
        "opt_einsum is not installed — it lives in the opt-in "
        "`einsum` dependency group:\n"
        "    uv sync --group einsum\n"
        f"({exc})"
    ) from exc

__all__ = ["main"]

#: opt_einsum subscript alphabet (letters + digits: 62 distinct labels).
_ALPHA = string.ascii_letters + string.digits

#: Maximum tensor rank the bond-network generator will build.
_DEGREE_CAP = 4

#: Probability a bond-network tensor also carries an open (output) leg.
_OPEN_LEG_P = 0.2

#: An effectively-unbounded repeat count for the deadline-bounded
#: randomised-greedy player (its ``max_time`` is the real budget).
_HUGE = 10**9


# ---------------------------------------------------------------------------
#  Instances — valid einsums only
# ---------------------------------------------------------------------------


def random_bond_network(
    n: int, seed: int
) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    """Seeded random tensor network with every index shared <= 2 times.

    A spanning tree guarantees connectivity, a few extra bonds add
    cycles, and a fraction of tensors carry an open leg.  Every index
    therefore appears in exactly two tensors (a *bond*) or exactly one
    (an *open* leg) — the precondition for a faithful einsum mapping.
    """
    rng = random.Random(seed)
    tensors: list[set[int]] = [set() for _ in range(n)]
    sizes: dict[int, int] = {}
    nxt = 0

    def new_bond() -> int:
        nonlocal nxt
        sizes[nxt] = rng.randint(2, 8)
        bond = nxt
        nxt += 1
        return bond

    def link(i: int, j: int) -> None:
        bond = new_bond()
        tensors[i].add(bond)
        tensors[j].add(bond)

    for i in range(1, n):
        room = [j for j in range(i) if len(tensors[j]) < _DEGREE_CAP]
        link(i, rng.choice(room) if room else rng.randrange(i))
    extra = max(1, n // 4)
    added = 0
    tries = 0
    while added < extra and tries < 50 * extra + 50:
        tries += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or (tensors[i] & tensors[j]):
            continue
        if len(tensors[i]) >= _DEGREE_CAP:
            continue
        if len(tensors[j]) >= _DEGREE_CAP:
            continue
        link(i, j)
        added += 1
    for i in range(n):
        if rng.random() < _OPEN_LEG_P and len(tensors[i]) < _DEGREE_CAP:
            tensors[i].add(new_bond())
    return tuple(tuple(sorted(t)) for t in tensors), sizes


def _net_from_subs(
    subs: list[str], size_dict: dict[str, int]
) -> tuple[tuple[tuple[str, ...], ...], dict[str, int]]:
    """Build ``(tensors, sizes)`` from explicit subscripts and sizes."""
    tensors = tuple(tuple(s) for s in subs)
    sizes = {i: size_dict[i] for s in subs for i in s}
    return tensors, sizes


def attention_scores() -> Any:
    """Build a genuine attention einsum: the ``Q K^T`` score matrix."""
    return _net_from_subs(
        ["bqe", "bke"], {"b": 2, "q": 8, "k": 8, "e": 16}
    )


def attention_context() -> Any:
    """Build a genuine attention einsum: ``scores @ V``."""
    return _net_from_subs(
        ["bqk", "bkh"], {"b": 2, "q": 8, "k": 8, "h": 16}
    )


def bilinear_pool() -> Any:
    """Build a genuine bilinear block: ``X W Y^T``."""
    return _net_from_subs(
        ["btd", "de", "btf"],
        {"b": 2, "t": 8, "d": 32, "e": 16, "f": 32},
    )


def mlp_stack() -> Any:
    """Build a genuine MLP block folded into one einsum ``X W1..W5``."""
    return _net_from_subs(
        ["btd", "de", "ef", "fg", "gh", "hi"],
        {
            "b": 2,
            "t": 8,
            "d": 32,
            "e": 64,
            "f": 64,
            "g": 64,
            "h": 64,
            "i": 32,
        },
    )


# ---------------------------------------------------------------------------
#  The einsum view of a network
# ---------------------------------------------------------------------------


@dataclass
class Einsum:
    """A network as an einsum: expression, shapes and label tables."""

    expr: str
    shapes: list[tuple[int, ...]]
    inputs: list[set[str]]
    output: set[str]
    size_dict: dict[str, int]


def to_einsum(tensors: Any, sizes: dict[int, int]) -> Einsum:
    """Map a ``(tensors, sizes)`` network onto an einsum expression.

    Index labels become single-character subscripts and the output is
    the set of indices appearing exactly once, so the mapping is only
    faithful when every index appears at most twice.
    """
    labels = sorted(sizes)
    letters = {lab: _ALPHA[i] for i, lab in enumerate(labels)}
    counts = Counter(i for t in tensors for i in t)
    out = "".join(letters[i] for i in labels if counts[i] == 1)
    subs = ["".join(letters[i] for i in t) for t in tensors]
    return Einsum(
        expr=",".join(subs) + "->" + out,
        shapes=[tuple(sizes[i] for i in t) for t in tensors],
        inputs=[set(s) for s in subs],
        output=set(out),
        size_dict={letters[i]: sizes[i] for i in labels},
    )


def max_multiplicity(tensors: Any) -> int:
    """Largest number of tensors any single index appears in."""
    counts = Counter(i for t in tensors for i in t)
    return max(counts.values(), default=0)


# ---------------------------------------------------------------------------
#  Our players, as order producers
# ---------------------------------------------------------------------------


def our_greedy_order(
    tensors: Any, sizes: dict[int, int]
) -> list[tuple[int, int]]:
    """Cheapest-pair greedy order — the baseline the retro used."""
    ts = [frozenset(t) for t in tensors]
    order: list[tuple[int, int]] = []
    while len(ts) > 1:
        _c, a, b = min(
            (cs.pair_cost(ts[x], ts[y], sizes), x, y)
            for x, y in cs._pairs(ts)
        )
        order.append((a, b))
        ts = cs._merge(ts, a, b)
    return order


def _random_greedy_order(
    tensors: Any,
    sizes: dict[int, int],
    rng: random.Random,
    top_k: int,
) -> list[tuple[int, int]]:
    """One randomised-greedy episode; returns its contraction order."""
    ts = [frozenset(t) for t in tensors]
    order: list[tuple[int, int]] = []
    while len(ts) > 1:
        cands = sorted(
            (cs.pair_cost(ts[a], ts[b], sizes), a, b)
            for a, b in cs._pairs(ts)
        )
        _c, a, b = cands[rng.randrange(min(top_k, len(cands)))]
        order.append((a, b))
        ts = cs._merge(ts, a, b)
    return order


def our_cost_of_order(
    tensors: Any, sizes: dict[int, int], order: list[tuple[int, int]]
) -> float:
    """Replay an order and sum our pairwise cost model."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    for a, b in order:
        total += cs.pair_cost(ts[a], ts[b], sizes)
        ts = cs._merge(ts, a, b)
    return total


def our_restart_order(
    tensors: Any,
    sizes: dict[int, int],
    budget: float,
    seed: int,
    *,
    top_k: int = 3,
) -> tuple[list[tuple[int, int]], float, int]:
    """Anytime best-of-N randomised-greedy order within ``budget``.

    Starts from the deterministic greedy order (so it can never be worse
    than greedy) and keeps sampling until the budget is spent.  Returns
    the order, the elapsed seconds and the rollout count.
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best = our_greedy_order(tensors, sizes)
    best_cost = our_cost_of_order(tensors, sizes, best)
    rollouts = 1
    while True:
        order = _random_greedy_order(tensors, sizes, rng, top_k)
        cost = our_cost_of_order(tensors, sizes, order)
        rollouts += 1
        if cost < best_cost:
            best, best_cost = order, cost
        if time.perf_counter() - t0 >= budget:
            break
    return best, time.perf_counter() - t0, rollouts


def _policy_rollouts(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Sample ``samples`` lockstep policy rollouts; return orders + costs.

    The mirror of ``contraction_policy.run_policy_batch``, except it also
    records the contraction order each rollout chose, so the order can be
    re-scored with opt_einsum's independent cost model.
    """
    games = [
        cp.ContractionGame(tensors, sizes, greedy_ref)
        for _ in range(samples)
    ]
    orders: list[list[tuple[int, int]]] = [[] for _ in range(samples)]
    while not games[0].done:
        sf, pf = cp._batch_inputs(games, device)
        logits = cp._logits(model, sf, pf)
        idx = cp.Categorical(logits=logits / temperature).sample()
        for j, (g, i) in enumerate(
            zip(games, idx.tolist(), strict=True)
        ):
            a, b = g.pairs[i]
            orders[j].append((a, b))
            g.step(a, b)
    return orders, [g.cost for g in games]


def policy_best_order(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    budget: float,
    device: str,
    per_prior: float,
    seed: int,
) -> tuple[list[tuple[int, int]], float, int]:
    """Anytime learned policy: best sampled order within ``budget``.

    Lockstep batches are sized from a warm per-rollout prior and then
    re-sized from the *actual* elapsed time, so the policy fills the
    budget and nothing is timed outside it — the same protocol the
    equal-wall-clock retro used.
    """
    torch.manual_seed(seed)
    per = max(per_prior, 1e-6)
    batch = max(1, min(cp._MAX_BATCH, int(0.25 * budget / per)))
    t0 = time.perf_counter()
    best: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    rollouts = 0
    while True:
        orders, costs = _policy_rollouts(
            model, tensors, sizes, greedy_ref, batch, device, cp._TEMP
        )
        rollouts += batch
        k = min(range(len(costs)), key=costs.__getitem__)
        if costs[k] < best_cost:
            best, best_cost = orders[k], costs[k]
        elapsed = time.perf_counter() - t0
        per = elapsed / rollouts
        remaining = budget - elapsed
        if remaining < per:
            break
        batch = max(1, min(cp._MAX_BATCH, int(remaining / per)))
    assert best is not None
    return best, time.perf_counter() - t0, rollouts


def _policy_prior(
    model: Any, tensors: Any, sizes: dict[int, int], device: str
) -> float:
    """Warm per-rollout seconds for the policy on this board."""
    ref = cs.greedy(tensors, sizes)
    cp.run_policy_batch(
        model,
        tensors,
        sizes,
        ref,
        samples=4,
        greedy=False,
        temperature=cp._TEMP,
        device=device,
    )
    t0 = time.perf_counter()
    cp.run_policy_batch(
        model,
        tensors,
        sizes,
        ref,
        samples=8,
        greedy=False,
        temperature=cp._TEMP,
        device=device,
    )
    return (time.perf_counter() - t0) / 8


# ---------------------------------------------------------------------------
#  opt_einsum's players
# ---------------------------------------------------------------------------


def oe_cost_of_order(e: Einsum, order: list[tuple[int, int]]) -> float:
    """Score an order with opt_einsum's own ``contract_path`` FLOP cost."""
    _path, info = oe.contract_path(
        e.expr,
        *e.shapes,
        shapes=True,
        optimize=[tuple(p) for p in order],
    )
    return float(info.opt_cost)


def oe_greedy_order(e: Einsum) -> list[tuple[int, int]]:
    """opt_einsum's staged memory-removed greedy path."""
    path, _info = oe.contract_path(
        e.expr, *e.shapes, shapes=True, optimize="greedy"
    )
    return [tuple(p) for p in path]


def oe_optimal_order(
    e: Einsum, budget: float
) -> list[tuple[int, int]] | None:
    """opt_einsum's exact ``optimal`` DP, or ``None`` on timeout."""
    try:
        with cs._deadline(budget):
            path, _info = oe.contract_path(
                e.expr, *e.shapes, shapes=True, optimize="optimal"
            )
        return [tuple(p) for p in path]
    except cs._Timeout:
        return None


def oe_random_greedy_order(
    e: Einsum, budget: float
) -> list[tuple[int, int]]:
    """opt_einsum's randomised greedy, filled to ``budget`` seconds."""
    opt = RandomGreedy(max_repeats=_HUGE, max_time=budget)
    path = opt(e.inputs, e.output, e.size_dict)
    return [tuple(p) for p in path]


# ---------------------------------------------------------------------------
#  One board, every player
# ---------------------------------------------------------------------------


@dataclass
class _Res:
    """One player's outcome: its order, both costs, time and work."""

    order: list[tuple[int, int]] | None
    our: float
    oe: float
    secs: float
    work: float


def _res(
    tensors: Any,
    sizes: dict[int, int],
    e: Einsum,
    order: list[tuple[int, int]] | None,
    secs: float,
    work: float,
) -> _Res:
    """Package a player's order with both cost models (inf if DNF)."""
    if order is None:
        return _Res(None, float("inf"), float("inf"), secs, work)
    return _Res(
        order,
        our_cost_of_order(tensors, sizes, order),
        oe_cost_of_order(e, order),
        secs,
        work,
    )


def _run_board(
    tensors: Any,
    sizes: dict[int, int],
    models: dict[str, Any],
    device: str,
    budget: float,
    seed: int,
) -> dict[str, _Res]:
    """Run every player on one board at an equal wall-clock budget."""
    e = to_einsum(tensors, sizes)
    ref = cs.greedy(tensors, sizes)
    steps = max(len(tensors) - 1, 0)
    out: dict[str, _Res] = {}

    t0 = time.perf_counter()
    order = our_greedy_order(tensors, sizes)
    out["our-greedy"] = _res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    order, secs, rolls = our_restart_order(tensors, sizes, budget, seed)
    out["our-restart"] = _res(tensors, sizes, e, order, secs, rolls)

    for name, model in models.items():
        prior = _policy_prior(model, tensors, sizes, device)
        order, secs, rolls = policy_best_order(
            model, tensors, sizes, ref, budget, device, prior, seed
        )
        out[name] = _res(tensors, sizes, e, order, secs, rolls)

    t0 = time.perf_counter()
    order = oe_greedy_order(e)
    out["oe-greedy"] = _res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    t0 = time.perf_counter()
    order = oe_optimal_order(e, budget)
    out["oe-optimal"] = _res(
        tensors, sizes, e, order, time.perf_counter() - t0, 0.0
    )

    t0 = time.perf_counter()
    order = oe_random_greedy_order(e, budget)
    out["oe-rand-greedy"] = _res(
        tensors, sizes, e, order, time.perf_counter() - t0, float("nan")
    )
    return out


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

_PLAYERS = (
    "our-greedy",
    "our-restart",
    "oe-greedy",
    "oe-rand-greedy",
    "oe-optimal",
)


def _players(models: dict[str, Any]) -> tuple[str, ...]:
    """Return the players to report, with learned models in place."""
    out: list[str] = []
    for p in _PLAYERS:
        out.append(p)
        if p == "our-restart":
            out.extend(models)
    return tuple(out)


def _num(v: float, width: int, fmt: str = ".3f") -> str:
    """Render a float, or ``-`` when it is ``inf``/``nan``."""
    if v != v or v in (float("inf"), float("-inf")):
        return f"{'-':>{width}}"
    return f"{format(v, fmt):>{width}}"


def _family_section() -> None:
    """Show that the retro's family is not an einsum at all."""
    print("== 0. why a new instance family ==")
    print("  max index multiplicity of the retro's random_network")
    print("  (1 or 2 = a valid einsum; >2 = a hyperedge opt_einsum")
    print("  cannot represent):")
    for n in (8, 20, 30, 40):
        tensors, sizes = cs.random_network(n, 0)
        bond, bs = random_bond_network(n, 0)
        print(
            f"    n={n:>2}: hypergraph={max_multiplicity(tensors)}"
            f"  bond-network={max_multiplicity(bond)}"
            f"  (indices {len(sizes)} vs {len(bs)})"
        )


def _control_section(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, Any],
    device: str,
    budget: float,
) -> None:
    """Small-n control where opt_einsum's exact DP can finish."""
    players = _players(models)
    print()
    print("== 1. control — players / the exact optimum (mean) ==")
    print("  our dp is the exact subset DP; oe-opt is opt_einsum's")
    print("  exact DP (should agree with dp to a factor of 2 = FLOPs).")
    head = f"  {'n':>3} {'dp':>10} {'oe-opt/2dp':>10} " + " ".join(
        f"{p:>12}" for p in players
    )
    print(head)
    print("-" * len(head))
    for n in scales:
        acc: dict[str, list[float]] = {p: [] for p in players}
        for k in range(instances):
            tensors, sizes = random_bond_network(n, seed + k)
            e = to_einsum(tensors, sizes)
            dp = cs.dp(tensors, sizes)
            found = _run_board(
                tensors, sizes, models, device, budget, seed + k
            )
            for p in players:
                acc[p].append(found[p].our / dp)
            if k == 0:
                opt = oe_optimal_order(e, budget)
                ratio = (
                    oe_cost_of_order(e, opt) / (2.0 * dp)
                    if opt
                    else float("nan")
                )
                shown = f"{dp:>10.3g} {_num(ratio, 10)} "
            else:
                shown = ""
            print(
                f"  {n:>3} {shown}"
                + " ".join(
                    _num(statistics.fmean(acc[p]), 12) for p in players
                )
            )


def _outer_steps(tensors: Any, order: list[tuple[int, int]]) -> int:
    """Count the steps of an order that contract a disjoint pair."""
    ts = [frozenset(t) for t in tensors]
    n = 0
    for a, b in order:
        if not (ts[a] & ts[b]):
            n += 1
        ts = cs._merge(ts, a, b)
    return n


def _mechanism_section(
    scales: tuple[int, ...], seed: int, instances: int
) -> None:
    """Show *why* our greedy loses: it forms outer products eagerly."""
    print()
    print("== 2b. mechanism — outer-product steps per order (mean) ==")
    print("  cheapest-pair greedy grabs cheap outer products early;")
    print("  opt_einsum's staged greedy defers them until forced.")
    head = (
        f"  {'n':>3} {'our-greedy':>12} {'oe-greedy':>11} {'steps':>7}"
    )
    print(head)
    print("-" * len(head))
    for n in scales:
        ours: list[float] = []
        oes: list[float] = []
        for k in range(instances):
            tensors, sizes = random_bond_network(n, seed + k)
            ours.append(
                _outer_steps(tensors, our_greedy_order(tensors, sizes))
            )
            oes.append(
                _outer_steps(
                    tensors, oe_greedy_order(to_einsum(tensors, sizes))
                )
            )
        print(
            f"  {n:>3} {statistics.fmean(ours):>12.1f} "
            f"{statistics.fmean(oes):>11.1f} {n - 1:>7}"
        )


def _measure(
    budgets: tuple[float, ...],
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, Any],
    device: str,
) -> list[tuple[float, int, list[dict[str, _Res]]]]:
    """Run every player on every board once; return the raw results."""
    data: list[tuple[float, int, list[dict[str, _Res]]]] = []
    for budget in budgets:
        for n in scales:
            boards = []
            for k in range(instances):
                tensors, sizes = random_bond_network(n, seed + k)
                boards.append(
                    _run_board(
                        tensors, sizes, models, device, budget, seed + k
                    )
                )
            data.append((budget, n, boards))
    return data


def _ladder_table(
    data: list[tuple[float, int, list[dict[str, _Res]]]],
    models: dict[str, Any],
) -> None:
    """Print the equal-wall-clock ladder under both cost models."""
    players = _players(models)
    for budget in sorted({b for b, _n, _d in data}):
        print()
        print(
            f"== 2. equal wall-clock: {1e3 * budget:.0f} ms per instance =="
        )
        print(
            f"  {'n':>3} {'player':>15} {'our-ratio':>10} "
            f"{'oe-ratio':>10} {'work':>9} {'ms':>7}"
        )
        print("-" * 61)
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
                print(
                    f"  {n:>3} {p:>15} "
                    f"{statistics.fmean(our):>10.3f} "
                    f"{statistics.fmean(oe):>10.3f} "
                    f"{_num(statistics.fmean(work), 9, '.1f')} "
                    f"{1e3 * statistics.fmean(secs):>7.1f}"
                )


def _pairwise_table(
    data: list[tuple[float, int, list[dict[str, _Res]]]],
    models: dict[str, Any],
) -> None:
    """Head-to-head learned/opt_einsum ratios under both cost models."""
    if not models:
        return
    cols = ["oe-greedy", "oe-rand-greedy", "oe-optimal"]
    print()
    print(
        "== 3. learned / opt_einsum pairwise ratio "
        "(mean; <1 = learned wins) =="
    )
    head = (
        f"  {'ms':>6} {'n':>3} {'player':>15} "
        + " ".join(f"{'our/' + c:>16}" for c in cols)
        + " "
        + " ".join(f"{'oe/' + c:>16}" for c in cols)
    )
    print(head)
    print("-" * len(head))

    def _cell(vals: list[float]) -> str:
        finite = [v for v in vals if v == v]
        return (
            _num(statistics.fmean(finite), 16)
            if finite
            else f"{'DNF':>16}"
        )

    for budget, n, boards in data:
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
                f"  {1e3 * budget:>6.0f} {n:>3} {name:>15} "
                + " ".join(_cell(our_r[c]) for c in cols)
                + " "
                + " ".join(_cell(oe_r[c]) for c in cols)
            )


def _real_section(
    models: dict[str, Any], device: str, budget: float
) -> None:
    """Order small but genuine attention / MLP einsums with both tools."""
    print()
    print("== 4. real einsums (attention / MLP block) ==")
    cases = [
        ("attention QK^T", attention_scores),
        ("attention @V", attention_context),
        ("bilinear XWY^T", bilinear_pool),
        ("MLP stack XW1..W5", mlp_stack),
    ]
    head = (
        f"  {'case':>18} {'ops':>4} {'our-greedy':>11} "
        f"{'our-restart':>12} "
        + " ".join(f"{name:>12}" for name in models)
        + f" {'oe-greedy':>10} {'oe-optimal':>11} {'verified':>9}"
    )
    print(head)
    print("-" * len(head))
    for label, build in cases:
        tensors, sizes = build()
        e = to_einsum(tensors, sizes)
        found = _run_board(tensors, sizes, models, device, budget, 0)
        opt = oe_optimal_order(e, budget)
        orders = [r.order for r in found.values() if r.order]
        ok = _verify_real(e, orders)
        opt_cost = oe_cost_of_order(e, opt) if opt else float("nan")
        learned = " ".join(
            f"{found[name].oe:>12.4g}" for name in models
        )
        print(
            f"  {label:>18} {len(tensors):>4} "
            f"{found['our-greedy'].oe:>11.4g} "
            f"{found['our-restart'].oe:>12.4g} "
            f"{learned} "
            f"{found['oe-greedy'].oe:>10.4g} "
            f"{opt_cost:>11.4g} "
            f"{('yes' if ok else 'NO'):>9}"
        )


def _verify_real(
    e: Einsum, orders: list[list[tuple[int, int]]]
) -> bool:
    """Check every order contracts the real einsum to the same result."""
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal(sh) for sh in e.shapes]
    ref = oe.contract(e.expr, *arrays, optimize="greedy")
    for order in orders:
        got = oe.contract(
            e.expr, *arrays, optimize=[tuple(p) for p in order]
        )
        if not np.allclose(got, ref):
            return False
    return True


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _on_family(family: Any) -> Any:
    """Train on the tool's einsum-valid family, not the hypergraph one."""
    old = cs.random_network
    cs.random_network = family
    try:
        yield
    finally:
        cs.random_network = old


def main(argv: list[str] | None = None) -> int:
    """Train the policy, then falsify it against opt_einsum."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument(
        "--scales",
        default="20,30,40",
        help="comma-separated n at scale",
    )
    ap.add_argument(
        "--control",
        default="8,10",
        help="comma-separated n for the DP control",
    )
    ap.add_argument(
        "--budgets",
        default="50,200,1000",
        help="comma-separated per-instance budgets (ms)",
    )
    ap.add_argument("--trainer", default="rl", choices=("rl", "none"))
    ap.add_argument(
        "--train-scales",
        default="8,10,12",
        help="comma-separated n the 'learned' policy trains on",
    )
    ap.add_argument(
        "--compare-scales",
        default="",
        help="comma-separated n for a second 'learned-scale' policy",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device: {dev}  (torch {torch.__version__}, "
        f"opt_einsum {oe.__version__})"
    )
    base_ns = tuple(
        int(x) for x in args.train_scales.split(",") if x.strip()
    )
    cmp_ns = tuple(
        int(x) for x in args.compare_scales.split(",") if x.strip()
    )

    models: dict[str, Any] = {}
    if args.trainer != "none":
        with _on_family(random_bond_network):
            t0 = time.perf_counter()
            models["learned"] = cp.train_rl(
                ns=base_ns,
                iterations=args.iterations,
                batch=args.batch,
                hidden=args.hidden,
                device=dev,
                seed=args.seed,
            )
            print(
                f"trained learned ({base_ns}) in "
                f"{time.perf_counter() - t0:.1f}s"
            )
            if cmp_ns:
                t0 = time.perf_counter()
                models["learned-scale"] = cp.train_rl(
                    ns=cmp_ns,
                    iterations=args.iterations,
                    batch=args.batch,
                    hidden=args.hidden,
                    device=dev,
                    seed=args.seed,
                )
                print(
                    f"trained learned-scale ({cmp_ns}) in "
                    f"{time.perf_counter() - t0:.1f}s"
                )

    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    control = tuple(
        int(x) for x in args.control.split(",") if x.strip()
    )
    budgets = tuple(
        1e-3 * float(x) for x in args.budgets.split(",") if x.strip()
    )

    _family_section()
    _control_section(
        control, args.instances, args.seed, models, dev, budgets[-1]
    )
    _mechanism_section(scales, args.seed, args.instances)
    data = _measure(
        budgets, scales, args.instances, args.seed, models, dev
    )
    _ladder_table(data, models)
    _pairwise_table(data, models)
    _real_section(models, dev, budgets[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
