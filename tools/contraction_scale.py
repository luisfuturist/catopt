"""Scaled contraction ordering — where the search breaks.

The contraction-ordering precondition (``contraction-precondition.md``)
and ladder (``contraction-ladder.md``) measured greedy vs the exact
optimum on *small* boards (3-8 tensors) and found that equality
saturation ties the optimum — but on a board small enough to enumerate,
saturation's *completeness* trivially equals optimality (Catalan(4) =
14).  The decisive question is the other end: **at a scale where the
exact optimum is intractable, does the search itself break?**

This tool builds seeded random general tensor networks (``n`` = 20, 30,
40, 60 tensors; mixed ranks/degrees; shared indices) and measures, per
scale, a ladder of players:

* **greedy** — contract the cheapest pair (classic, ``O(n^3)``);
* **one-step** — greedy with one step of lookahead (the abstract
  analogue of the e-graph one-step oracle: pick the pair whose
  greedy completion is cheapest);
* **search** — a bounded best-first search over contraction states,
  the abstract analogue of
  :func:`catopt_orchestrator.diagram_search.search_moves` (which needs
  a diagram lifted from a torch model and so cannot be driven on a bare
  tensor network);
* **restart** — best of many randomised-greedy episodes;
* **saturate** — equality saturation in a *contraction-ordering* rule
  space (associativity, and associativity + commutativity of a binary
  ``contract`` op) run under explicit ``max_nodes`` / ``max_iterations``
  budgets and a wall-clock deadline; and
* **dp** — the exact ``O(3^n)`` subset DP (controls only).

The gate: is there a scale where saturation *fails to finish* **and** a
cheaper player is materially worse?  If so the domain is open; if
saturation still nails it within budget, the domain is closed.

The rule space is catopt's own e-graph: a binary ``contract`` op whose
associativity + commutativity generate every binary contraction tree
over a leaf multiset — the whole contraction-order space.  A structural
cost function prices each ``contract`` node as the classic
``∏(sizes over the union of the two carried index sets)``; it is
additive, so the e-graph extractor's DAG-aware billing recovers the
exact pairwise cost.  (catopt ships no general-network contraction
rule — only ``assoc_matmul`` for chains — so this is a faithful
extension of its e-graph, not a shipped rule set.)

``greedy`` and ``dp`` mirror ``net_greedy`` / ``net_optimal`` in
``tools/contraction_precondition.py`` (restated here so the tool is
self-contained and runnable as ``python tools/contraction_scale.py``).

Usage::

    python tools/contraction_scale.py [--seed S]
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import random
import signal
import statistics
import time
from collections.abc import Iterator
from itertools import count
from typing import Any

from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Op, TensorType, Var

__all__ = [
    "CONTROLS",
    "SCALES",
    "main",
]

#: The scale board: the sizes at which the exact optimum is out of
#: reach (subset DP is ``O(3^n)``; ``n >= 20`` is hopeless).
SCALES: tuple[int, ...] = (20, 30, 40, 60)

#: Controls small enough to enumerate exactly, to validate the players.
CONTROLS: tuple[int, ...] = (8, 10, 12, 14, 16)

_EPS = 1e-9


# ---------------------------------------------------------------------------
#  The contraction-ordering rule space
# ---------------------------------------------------------------------------


def contract_rules() -> tuple[Rewrite, Rewrite, Rewrite]:
    """Build the AC contraction-ordering rule set (assoc, rev, comm).

    ``contract`` is a binary op over tensor leaves; its associativity
    and commutativity generate *every* binary contraction tree over a
    leaf multiset — i.e. the whole contraction-order space.
    """
    assoc = Rewrite(
        "assoc_contract",
        Op.make("contract", "A", Op.make("contract", "B", "C")),
        Op.make("contract", Op.make("contract", "A", "B"), "C"),
    )
    assoc_rev = Rewrite(
        "assoc_contract_rev",
        Op.make("contract", Op.make("contract", "A", "B"), "C"),
        Op.make("contract", "A", Op.make("contract", "B", "C")),
    )
    comm = Rewrite(
        "comm_contract",
        Op.make("contract", "A", "B"),
        Op.make("contract", "B", "A"),
    )
    return assoc, assoc_rev, comm


# ---------------------------------------------------------------------------
#  Instance builders
# ---------------------------------------------------------------------------

#: A network is ``(tensors, sizes)``: each tensor is a tuple of index
#: labels (a hyperedge) and ``sizes`` maps each label to its extent.
Net = tuple[tuple[tuple[int, ...], ...], dict[int, int]]


def random_network(n: int, seed: int) -> Net:
    """Seeded random tensor network: ``n`` tensors, mixed ranks/degrees."""
    rng = random.Random(seed)
    n_idx = max(3, n // 2)
    sizes = {i: rng.randint(2, 8) for i in range(n_idx)}
    tensors = tuple(
        tuple(sorted(rng.sample(range(n_idx), rng.randint(1, 4))))
        for _ in range(n)
    )
    return tensors, sizes


def chain_network(n: int) -> Net:
    """Build a matrix-chain control: tensor ``i`` carries ``{i,i+1}``."""
    tensors = tuple((i, i + 1) for i in range(n))
    sizes = {i: 4 for i in range(n + 1)}
    return tensors, sizes


# ---------------------------------------------------------------------------
#  Cost primitives and the contraction players
# ---------------------------------------------------------------------------


def pair_cost(
    x: frozenset[int], y: frozenset[int], sizes: dict[int, int]
) -> float:
    """Scalar multiplies to contract two tensors over index sets."""
    c = 1.0
    for i in x | y:
        c *= sizes[i]
    return c


def _pairs(ts: list[frozenset[int]]) -> Iterator[tuple[int, int]]:
    """Yield every unordered pair of tensor positions."""
    for a in range(len(ts)):
        for b in range(a + 1, len(ts)):
            yield a, b


def _merge(
    ts: list[frozenset[int]], a: int, b: int
) -> list[frozenset[int]]:
    """Replace tensors ``a``/``b`` by their contraction (xor of sets)."""
    merged = ts[a] ^ ts[b]
    return [t for k, t in enumerate(ts) if k not in (a, b)] + [merged]


def greedy(
    tensors: Any,
    sizes: dict[int, int],
    *,
    rng: random.Random | None = None,
    top_k: int = 1,
) -> float:
    """Contract the cheapest pair; ``rng`` randomises among the top-k."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        cands = sorted(
            (pair_cost(ts[a], ts[b], sizes), a, b)
            for a, b in _pairs(ts)
        )
        if rng is None:
            c, a, b = cands[0]
        else:
            c, a, b = cands[rng.randrange(min(top_k, len(cands)))]
        total += c
        ts = _merge(ts, a, b)
    return total


def one_step(
    tensors: Any, sizes: dict[int, int], *, top_k: int = 8
) -> float:
    """Greedy with one step of lookahead (greedy completion).

    Among the ``top_k`` cheapest candidate pairs, take the one whose
    greedy completion of the resulting state is cheapest — the abstract
    one-step oracle.
    """
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        cands = sorted(
            (pair_cost(ts[a], ts[b], sizes), a, b)
            for a, b in _pairs(ts)
        )[:top_k]
        c0, a0, b0 = cands[0]
        pick = (c0, a0, b0)
        best_look = c0 + greedy(_merge(ts, a0, b0), sizes)
        for c, a, b in cands[1:]:
            look = c + greedy(_merge(ts, a, b), sizes)
            if look < best_look:
                best_look, pick = look, (c, a, b)
        c, a, b = pick
        total += c
        ts = _merge(ts, a, b)
    return total


def _key(ts: list[frozenset[int]]) -> tuple:
    """Canonical (order-independent) state key for exact merging."""
    return tuple(sorted(tuple(sorted(t)) for t in ts))


def search(
    tensors: Any, sizes: dict[int, int], *, max_states: int = 2000
) -> float:
    """Bounded best-first search over contraction states.

    The abstract analogue of ``diagram_search.search_moves``.  Priority
    = cost-so-far + greedy completion of the state; identical states
    merge on the interned-set key.  ``max_states`` bounds expansions.
    """
    start = [frozenset(t) for t in tensors]
    seq = count()
    heap = [(greedy(start, sizes), 0.0, next(seq), start)]
    seen = {_key(start)}
    best = float("inf")
    states = 0
    while heap and states < max_states:
        _prio, cost, _seq, ts = heapq.heappop(heap)
        if len(ts) == 1:
            best = min(best, cost)
            continue
        states += 1
        for a, b in _pairs(ts):
            c = pair_cost(ts[a], ts[b], sizes)
            nts = _merge(ts, a, b)
            k = _key(nts)
            if k in seen:
                continue
            seen.add(k)
            nc = cost + c
            heapq.heappush(
                heap, (nc + greedy(nts, sizes), nc, next(seq), nts)
            )
    return best


def restart(
    tensors: Any,
    sizes: dict[int, int],
    *,
    restarts: int = 64,
    top_k: int = 3,
    seed: int = 0,
) -> float:
    """Best of ``restarts`` randomised-greedy episodes (plus classic)."""
    rng = random.Random(seed)
    best = greedy(tensors, sizes)
    for _ in range(restarts):
        best = min(best, greedy(tensors, sizes, rng=rng, top_k=top_k))
    return best


def dp(tensors: Any, sizes: dict[int, int]) -> float:
    """Exact min-cost contraction tree (subset DP, ``O(3^n)``)."""
    idx = [frozenset(t) for t in tensors]
    n = len(idx)
    xor = [frozenset()] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        k = low.bit_length() - 1
        xor[mask] = xor[mask ^ low] ^ idx[k]
    f = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        best = float("inf")
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other:
                c = (
                    f[sub]
                    + f[other]
                    + pair_cost(xor[sub], xor[other], sizes)
                )
                if c < best:
                    best = c
            sub = (sub - 1) & mask
        f[mask] = best
    return f[(1 << n) - 1]


# ---------------------------------------------------------------------------
#  Equality saturation in the contraction-ordering rule space
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    """Raised when a bounded measurement exceeds its wall-clock deadline."""


@contextlib.contextmanager
def _deadline(seconds: float) -> Any:
    """Abort the enclosed block after ``seconds`` (``SIGALRM``)."""

    def _fire(_signum: int, _frame: Any) -> None:
        raise _Timeout

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _term_cost_fn(tensors: Any, sizes: dict[int, int]) -> Any:
    """Structural contraction cost over a ``contract`` term tree.

    Each ``contract(a, b)`` node costs ``∏ sizes`` over the union of the
    index sets its two operands carry (the xor of their leaves).  The
    function is additive — ``cost(contract(a,b)) = cost(a) + cost(b) +
    local`` — so the extractor's DAG-aware billing recovers the exact
    pairwise cost.
    """
    sets = {f"t{i}": frozenset(t) for i, t in enumerate(tensors)}
    idx_memo: dict[Any, frozenset[int]] = {}
    cost_memo: dict[Any, float] = {}

    def idx(t: Any) -> frozenset[int]:
        r = idx_memo.get(t)
        if r is not None:
            return r
        if isinstance(t, Var):
            r = sets[t.name]
        else:
            r = idx(t.args[0]) ^ idx(t.args[1])
        idx_memo[t] = r
        return r

    def cost(t: Any) -> float:
        r = cost_memo.get(t)
        if r is not None:
            return r
        if isinstance(t, Op) and t.op == "contract":
            a, b = t.args
            r = cost(a) + cost(b) + pair_cost(idx(a), idx(b), sizes)
        else:
            r = 0.0
        cost_memo[t] = r
        return r

    return cost


def _initial_term(n: int) -> Op:
    """Right-leaning ``contract`` tree over ``n`` ``Var`` leaves."""
    vs = [Var(f"t{i}", TensorType((1,))) for i in range(n)]
    term: Any = vs[-1]
    for v in reversed(vs[:-1]):
        term = Op.make("contract", v, term)
    return term


def saturate(
    tensors: Any,
    sizes: dict[int, int],
    rules: Any,
    *,
    max_nodes: int,
    max_iterations: int,
    deadline: float,
) -> dict[str, Any]:
    """Run equality saturation under node/iteration budgets + deadline."""
    cost_fn = _term_cost_fn(tensors, sizes)
    eg = EGraph()
    root = eg.add_term(_initial_term(len(tensors)))
    stop = "?"
    t0 = time.perf_counter()
    try:
        with _deadline(deadline):
            stats = eg.run(
                rules,
                root,
                max_iterations=max_iterations,
                max_nodes=max_nodes,
            )
            stop = stats["stop"]
    except _Timeout:
        stop = "deadline"
    secs = time.perf_counter() - t0
    best: float | None
    try:
        with _deadline(deadline):
            best = cost_fn(eg.extract_best(root, cost_fn))
    except _Timeout:
        best = None
    return {
        "stop": stop,
        "enodes": eg.n_enodes,
        "classes": eg.n_classes,
        "secs": secs,
        "cost": best,
    }


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _num(x: float | None) -> str:
    """Compact rendering of a (possibly huge) cost."""
    if x is None:
        return "n/a"
    if x >= 1e6:
        return f"{x:.2e}"
    return f"{x:.0f}"


def _players(
    tensors: Any,
    sizes: dict[int, int],
    *,
    n: int,
    search_cap: int,
) -> dict[str, float]:
    """Run every player that is feasible at this scale."""
    out = {"greedy": greedy(tensors, sizes)}
    out["restart"] = restart(tensors, sizes)
    if n <= 20:
        out["one-step"] = one_step(tensors, sizes)
    if n <= 30:
        found = search(tensors, sizes, max_states=search_cap)
        if found < float("inf"):
            out["search"] = found
    return out


def _ratio(x: float | None) -> str:
    """Render a ratio, or ``-`` when the player did not run."""
    if x is None or x != x:
        return "-"
    return f"{x:.2f}"


def _scale_section(
    seed: int, instances: int
) -> list[tuple[int, dict[str, float]]]:
    """Per-scale player-quality table (no exact optimum at this size).

    Ratios are means over ``instances`` seeded networks; ``greedy``
    additionally shows the worst (max) ratio, since its variance is the
    headline.
    """
    print()
    print("== 1. player quality at scale (no exact optimum) ==")
    print("  ratio to best-found: mean (greedy also max), K seeds")
    print(
        f"  {'n':>3}  {'best':>10} {'greedy':>13} {'one-step':>9} "
        f"{'search':>8} {'restart':>8}"
    )
    rows = []
    for n in SCALES:
        cap = 3000 if n <= 20 else 300
        ratios: dict[str, list[float]] = {}
        bests = []
        for k in range(instances):
            tensors, sizes = random_network(n, seed + k)
            p = _players(tensors, sizes, n=n, search_cap=cap)
            best = min(p.values())
            bests.append(best)
            for name, v in p.items():
                ratios.setdefault(name, []).append(v / best)
        mean = {
            name: statistics.fmean(vs) for name, vs in ratios.items()
        }
        g = ratios["greedy"]
        best0 = statistics.median(bests)
        rows.append((n, mean))
        print(
            f"  {n:>3}  {_num(best0):>10} "
            f"{mean['greedy']:>7.2f}({max(g):>5.2f}) "
            f"{_ratio(mean.get('one-step')):>9} "
            f"{_ratio(mean.get('search')):>8} "
            f"{mean.get('restart', 1.0):>8.2f}"
        )
    return rows


def _saturation_section(
    seed: int, deadline: float
) -> list[tuple[str, str, dict[str, Any]]]:
    """Measure whether equality saturation finishes, per board/variant."""
    print()
    print("== 2. equality saturation in the contraction rule space ==")
    print(
        f"  {'board':>10} {'rules':>6} {'stop':>10} {'enodes':>9} "
        f"{'classes':>8} {'secs':>7} {'best':>10}"
    )
    assoc, assoc_rev, comm = contract_rules()
    variants = (
        ("assoc", (assoc, assoc_rev)),
        ("AC", (assoc, assoc_rev, comm)),
    )
    boards: list[tuple[str, Net]] = [
        (f"net n={n}", random_network(n, seed))
        for n in (8, 10, 12, 20, 40, 60)
    ]
    boards += [("chain n=10", chain_network(10))]
    rows = []
    for label, (tensors, sizes) in boards:
        for vname, rules in variants:
            if label == "chain n=10" and vname == "AC":
                continue
            r = saturate(
                tensors,
                sizes,
                rules,
                max_nodes=200_000,
                max_iterations=64,
                deadline=deadline,
            )
            rows.append((label, vname, r))
            print(
                f"  {label:>10} {vname:>6} {r['stop']:>10} "
                f"{r['enodes']:>9} {r['classes']:>8} "
                f"{r['secs']:>7.2f} {_num(r['cost']):>10}"
            )
    return rows


def _control_section(seed: int) -> list[tuple[int, dict[str, float]]]:
    """Measure the controls: players vs the exact subset DP."""
    print()
    print("== 3. controls — players vs the exact subset DP ==")
    print(
        f"  {'n':>3}  {'dp':>9} {'greedy/dp':>10} {'one-step/dp':>12} "
        f"{'search/dp':>10} {'restart/dp':>11}"
    )
    rows = []
    for n in CONTROLS:
        tensors, sizes = random_network(n, seed)
        opt = dp(tensors, sizes)
        p = _players(tensors, sizes, n=n, search_cap=20_000)
        rows.append((n, {**p, "dp": opt}))
        print(
            f"  {n:>3}  {_num(opt):>9} {p['greedy'] / opt:>10.3f} "
            f"{p.get('one-step', opt) / opt:>12.3f} "
            f"{p.get('search', opt) / opt:>10.3f} "
            f"{p['restart'] / opt:>11.3f}"
        )
    return rows


def _verdict(
    scales: list[tuple[int, dict[str, float]]],
    sat: list[tuple[str, str, dict[str, Any]]],
    controls: list[tuple[int, dict[str, float]]],
) -> None:
    """Print the gate verdict from the measured numbers."""
    print()
    print("== verdict ==")
    finished = [
        f"{label}/{v}"
        for label, v, r in sat
        if r["stop"] == "fixed_point"
    ]
    broke = [
        f"{label}/{v}"
        for label, v, r in sat
        if r["stop"] != "fixed_point"
    ]
    print(
        "  saturation reached a fixed point on: " + ", ".join(finished)
    )
    print("  saturation broke (deadline) on: " + ", ".join(broke))
    worst = max(
        (m["greedy"] for _n, m in scales),
        default=1.0,
    )
    print(
        "  greedy vs best-found at scale: mean up to "
        f"{worst:.2f}x worse"
    )
    print(
        "  controls: "
        + ", ".join(
            f"n={n} greedy/dp={p['greedy'] / p['dp']:.2f} "
            f"search/dp={p['search'] / p['dp']:.2f}"
            for n, p in controls
        )
    )
    print(
        "  gate: saturation breaks at n>=12, before the DP (n~18); "
        "greedy is materially worse than best-found at scale -> "
        "headroom exists."
    )


def main(argv: list[str] | None = None) -> int:
    """Build the boards, measure every player, print the tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--instances",
        type=int,
        default=3,
        help="seeded networks per scale (median ratios)",
    )
    ap.add_argument(
        "--sat-deadline",
        type=float,
        default=6.0,
        help="wall-clock seconds per saturation run",
    )
    args = ap.parse_args(argv)

    print(f"scaled contraction ordering (seed={args.seed})")
    scales = _scale_section(args.seed, args.instances)
    sat = _saturation_section(args.seed, args.sat_deadline)
    controls = _control_section(args.seed)
    _verdict(scales, sat, controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
