"""Contraction-ordering precondition for a learned search policy.

Plan 0016 stage 7 asks whether a learned/RL search policy can add
value *beyond* the one-step greedy cost-model oracle.  That is only
possible where greedy is not already optimal.  Tensor-contraction
ordering is the textbook candidate: it is NP-hard, super-exponential,
and classic greedy ("contract the cheapest adjacent pair") is known to
lose on real instances.

This tool measures that precondition, honestly, in three layers:

1. **Matrix chains** — the classic parenthesization problem.  Greedy
   (cheapest adjacent pair) vs the exact ``O(n^3)`` interval DP.
2. **General tensor networks** — 3-6 tensors with mixed ranks and
   shared indices.  Greedy (cheapest pair) vs an exact subset DP,
   which is itself validated against exhaustive enumeration.
3. **catopt's own machinery** — does catopt's contraction search beat
   greedy on the same instances?

   * :func:`catopt_orchestrator.diagram_search.search_moves` (the
     bounded lookahead) vs the greedy ``_apply_moves`` driver, on
     ``nn.Linear`` chain models — the diagram-liftable spelling of a
     matrix chain.
   * the e-graph rule space (``assoc_matmul`` over ``Var``-leaf
     chains), where catopt actually *prices* the ordering: saturation
     reaches the exact DP optimum, so the one-step greedy oracle
     (:func:`catopt_core.trajectories.rule_samples`) can be scored
     against a true optimum.

Nothing here is tuned to make greedy lose: the random sweeps are
seeded and the ratio distributions (including ties) are reported as
measured.

Usage::

    python tools/contraction_precondition.py [--instances N] [--seed S]
"""

from __future__ import annotations

import argparse
import random
import statistics
from typing import Any

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.search_env import SearchEnv
from catopt_core.trajectories import rule_samples

__all__ = [
    "CHAIN_CASES",
    "NET_CASES",
    "chain_greedy",
    "chain_optimal",
    "main",
    "net_brute",
    "net_greedy",
    "net_optimal",
]

_EPS = 1e-9


# ---------------------------------------------------------------------------
#  Layer 1 — matrix-chain parenthesization
# ---------------------------------------------------------------------------


def chain_greedy(dims: tuple[int, ...]) -> float:
    """Contract the cheapest adjacent pair until one matrix remains.

    Ties break to the lowest index (the deterministic "classic greedy"
    of the CLRS exercise).
    """
    d = list(dims)
    total = 0.0
    while len(d) > 2:
        i = min(
            range(1, len(d) - 1),
            key=lambda k: (d[k - 1] * d[k] * d[k + 1], k),
        )
        total += d[i - 1] * d[i] * d[i + 1]
        del d[i]
    return total


def chain_optimal(dims: tuple[int, ...]) -> float:
    """Exact min-cost parenthesization (the O(n^3) interval DP)."""
    n = len(dims) - 1
    m = [[0.0] * (n + 1) for _ in range(n + 1)]
    for span in range(2, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            m[i][j] = min(
                m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                for k in range(i, j)
            )
    return m[1][n]


# ---------------------------------------------------------------------------
#  Layer 2 — general tensor networks
# ---------------------------------------------------------------------------


def _pair_cost(
    x: frozenset[int], y: frozenset[int], sizes: dict[int, int]
) -> float:
    """Scalar multiplications to contract two tensors over index sets."""
    cost = 1.0
    for i in x | y:
        cost *= sizes[i]
    return cost


def net_greedy(tensors: Any, sizes: dict[int, int]) -> float:
    """Repeatedly contract the cheapest pair (min union size)."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        best_c = float("inf")
        pair = (0, 1)
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                c = _pair_cost(ts[a], ts[b], sizes)
                if c < best_c:
                    best_c = c
                    pair = (a, b)
        a, b = pair
        total += best_c
        merged = ts[a] ^ ts[b]
        ts = [t for k, t in enumerate(ts) if k not in (a, b)]
        ts.append(merged)
    return total


def net_optimal(tensors: Any, sizes: dict[int, int]) -> float:
    """Exact min-cost contraction tree via the subset DP.

    A subset's intermediate carries exactly the indices appearing an
    odd number of times (each contraction cancels a shared index), so
    ``xor`` of the members' index sets gives the intermediate's
    indices and the DP is exact.
    """
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
                    + _pair_cost(xor[sub], xor[other], sizes)
                )
                if c < best:
                    best = c
            sub = (sub - 1) & mask
        f[mask] = best
    return f[(1 << n) - 1]


def net_brute(tensors: Any, sizes: dict[int, int]) -> float:
    """Exhaustive min over every contraction order — the DP's oracle."""
    best = [float("inf")]

    def rec(ts: list[frozenset[int]], total: float) -> None:
        if total >= best[0]:
            return
        if len(ts) == 1:
            best[0] = total
            return
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                c = _pair_cost(ts[a], ts[b], sizes)
                merged = ts[a] ^ ts[b]
                nxt = [t for k, t in enumerate(ts) if k not in (a, b)]
                nxt.append(merged)
                rec(nxt, total + c)

    rec([frozenset(t) for t in tensors], 0.0)
    return best[0]


# ---------------------------------------------------------------------------
#  Instance sets
# ---------------------------------------------------------------------------

#: Curated matrix chains.  ``clrs`` is the textbook example; the rest
#: are small instances where greedy is provably suboptimal.
CHAIN_CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("clrs", (30, 35, 15, 5, 10, 20, 25)),
    ("small-a", (20, 7, 5, 22, 31)),
    ("small-b", (26, 1, 40, 32, 22)),
    ("small-c", (11, 25, 31, 11, 32, 35, 40, 39)),
    ("flat", (10, 20, 30, 40, 50)),
)

#: Curated small networks: ``(tensors, sizes)`` with mixed ranks.
NET_CASES: tuple[tuple[str, Any, dict[int, int]], ...] = (
    (
        "spiky",
        ((0, 3, 4), (5,), (0, 5), (1,), (2, 3, 5), (2,)),
        {0: 11, 1: 4, 2: 11, 3: 11, 4: 4, 5: 10},
    ),
    (
        "chain-hyper",
        ((1,), (1, 3, 4), (1,), (1, 2), (1, 4), (0, 1, 3)),
        {0: 12, 1: 12, 2: 9, 3: 9, 4: 9},
    ),
    (
        "mixed-rank",
        ((2, 4, 5), (0, 4, 5), (3,), (5,), (6,)),
        {0: 8, 2: 8, 3: 10, 4: 5, 5: 4, 6: 2},
    ),
)


def random_chains(
    n: int, seed: int, lo: int = 4, hi: int = 8
) -> list[tuple[int, ...]]:
    """``n`` random matrix chains with 4-8 matrices, dims in 1..40."""
    rng = random.Random(seed)
    return [
        tuple(
            rng.randint(1, 40) for _ in range(rng.randint(lo, hi) + 1)
        )
        for _ in range(n)
    ]


def random_nets(n: int, seed: int) -> list[tuple[Any, dict[int, int]]]:
    """``n`` random 3-6 tensor networks over 3-7 shared indices."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        nt = rng.randint(3, 6)
        ni = rng.randint(3, 7)
        labels = list(range(ni))
        tensors = [
            tuple(
                sorted(rng.sample(labels, rng.randint(1, min(3, ni))))
            )
            for _ in range(nt)
        ]
        sizes = {i: rng.randint(2, 12) for i in range(ni)}
        out.append((tensors, sizes))
    return out


# ---------------------------------------------------------------------------
#  Layer 3a — catopt's diagram contraction-move space
# ---------------------------------------------------------------------------


def _chain_model(dims: tuple[int, ...]) -> Any:
    """Build an ``nn.Linear`` chain of the given ``dims``."""
    import torch
    import torch.nn as nn

    class _Lin(nn.Module):
        def __init__(self, din: int, dout: int, seed: int) -> None:
            super().__init__()
            self.lin = nn.Linear(din, dout, bias=False).double()
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.lin.weight.copy_(
                    torch.randn(
                        dout, din, generator=g, dtype=torch.float64
                    )
                )

        def forward(self, x: Any) -> Any:
            return self.lin(x)

    class _Chain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                _Lin(dims[i], dims[i + 1], 400 + i)
                for i in range(len(dims) - 1)
            )

        def forward(self, x: Any) -> Any:
            for b in self.blocks:
                x = b(x)
            return x

    return _Chain()


def diagram_chain_costs(dims: tuple[int, ...]) -> dict[str, float]:
    """Run catopt's contraction driver greedy and search on a chain.

    Returns ``initial``, ``greedy`` and ``search`` final body costs —
    the same per-node ``dag_cost`` measure both modes report.
    """
    import torch
    from catopt_orchestrator import ContractionSearch, Optimizer
    from catopt_torch.backend import TorchBackend

    torch.manual_seed(0)
    x = torch.randn(4, dims[0], dtype=torch.float64)
    out: dict[str, float] = {}
    for mode in ("greedy", "search"):
        model = _chain_model(dims).eval().double()
        _mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            x,
            strategy=ContractionSearch(
                mode=mode,
                optimize_rest=False,
                search_depth=8,
                search_states=256,
            ),
        )
        ds = stats["diagram_search"]
        out["initial"] = ds["initial_cost"]
        out[mode] = ds.get("final_cost", ds.get("best_cost"))
    return out


# ---------------------------------------------------------------------------
#  Layer 3b — catopt's e-graph rule space (the priced ordering)
# ---------------------------------------------------------------------------


def _var_chain(dims: tuple[int, ...]) -> Op:
    """Right-seeded ``m0·(m1·(…))`` over ``Var`` leaves of ``dims``."""
    mats = [
        Var(f"m{i}", TensorType((dims[i], dims[i + 1])))
        for i in range(len(dims) - 1)
    ]
    term = mats[-1]
    for m in reversed(mats[:-1]):
        term = Op.make("matmul", m, term)
    return term


def egraph_sat_opt(term: Op, rules: Any) -> float:
    """Saturated optimum — the cheapest member of the full closure."""
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root)
    best = eg.extract_best(root, flops_cost)
    return dag_cost(best, flops_cost)


def _best_rule(env: SearchEnv, rules: Any) -> str:
    """Return the one-step cost-model oracle's pick from the state."""
    term = env.eg.extract_best(env.root, env.cost_fn)
    samples = rule_samples(term, rules, flops_cost)
    return max(samples, key=lambda s: s.delta_cost).rule


def egraph_greedy(
    term: Op,
    rules: Any,
    *,
    horizon: int = 6,
    patience: int = 2,
) -> float:
    """Run the greedy oracle episode; return the final extracted cost."""
    env = SearchEnv(
        term,
        rules,
        cost_fn=flops_cost,
        horizon=horizon,
        patience=patience,
    )
    env.reset()
    for _ in range(horizon):
        if env.step(_best_rule(env, rules)).done:
            break
    return env.cost


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _ratio_stats(ratios: list[float]) -> dict[str, float]:
    """Summarise a greedy/optimal ratio distribution."""
    rs = sorted(ratios)
    n = len(rs)
    lose = [r for r in rs if r > 1.0 + _EPS]
    return {
        "n": float(n),
        "lose": float(len(lose)),
        "pct": 100.0 * len(lose) / n,
        "mean": statistics.fmean(rs),
        "p50": rs[n // 2],
        "p90": rs[min(n - 1, int(0.9 * n))],
        "worst": rs[-1],
    }


def _print_stats(label: str, s: dict[str, float]) -> None:
    """Print one ratio-distribution summary line."""
    print(
        f"  {label:<22} n={int(s['n']):>4}  "
        f"greedy-loses={int(s['lose']):>4} ({s['pct']:>4.1f}%)  "
        f"mean={s['mean']:.3f}  p50={s['p50']:.3f}  "
        f"p90={s['p90']:.3f}  worst={s['worst']:.2f}"
    )


def _chain_section(n_rand: int, seed: int) -> dict[str, Any]:
    """Curated + random matrix-chain greedy-vs-DP measurements."""
    print("== 1. matrix chains — greedy vs O(n^3) DP ==")
    print(f"  {'case':<22} {'greedy':>10} {'optimal':>10} {'ratio':>7}")
    curated = []
    for name, dims in CHAIN_CASES:
        g, o = chain_greedy(dims), chain_optimal(dims)
        curated.append((name, g, o, g / o))
        print(f"  {name:<22} {g:>10.0f} {o:>10.0f} {g / o:>7.3f}")
    ratios = [
        chain_greedy(d) / chain_optimal(d)
        for d in random_chains(n_rand, seed)
    ]
    stats = _ratio_stats(ratios)
    print("  random n=4..8, dims 1..40:")
    _print_stats("matrix chains", stats)
    return {"curated": curated, "stats": stats}


def _net_section(n_rand: int, seed: int) -> dict[str, Any]:
    """Curated + random tensor-network greedy-vs-DP measurements."""
    print()
    print("== 2. general tensor networks — greedy vs subset DP ==")
    rng = random.Random(seed + 1)
    checked = 0
    for _ in range(200):
        nt = rng.randint(3, 5)
        ni = rng.randint(2, 5)
        labels = list(range(ni))
        tensors = [
            tuple(
                sorted(rng.sample(labels, rng.randint(1, min(3, ni))))
            )
            for _ in range(nt)
        ]
        sizes = {i: rng.randint(2, 6) for i in range(ni)}
        assert (
            abs(net_optimal(tensors, sizes) - net_brute(tensors, sizes))
            < _EPS
        )
        checked += 1
    print(
        f"  DP validated vs exhaustive enumeration on {checked} cases"
    )
    print(f"  {'case':<22} {'greedy':>10} {'optimal':>10} {'ratio':>7}")
    curated = []
    for name, tensors, sizes in NET_CASES:
        g, o = net_greedy(tensors, sizes), net_optimal(tensors, sizes)
        curated.append((name, g, o, g / o))
        print(f"  {name:<22} {g:>10.0f} {o:>10.0f} {g / o:>7.3f}")
    ratios = [
        net_greedy(t, s) / net_optimal(t, s)
        for t, s in random_nets(n_rand, seed + 2)
    ]
    stats = _ratio_stats(ratios)
    print("  random 3-6 tensors, mixed ranks/degrees:")
    _print_stats("tensor networks", stats)
    return {"curated": curated, "stats": stats}


def _diagram_section() -> list[tuple[str, float, float, float]]:
    """Measure catopt's diagram search vs greedy on chain models."""
    print()
    print(
        "== 3a. catopt diagram search_moves vs greedy _apply_moves =="
    )
    print(
        f"  {'dims':<26} {'initial':>9} {'greedy':>9} "
        f"{'search':>9} {'search<greedy':>13}"
    )
    rows = []
    for dims in (
        (2, 3, 14, 14, 3, 8),
        (30, 35, 15, 5, 10, 20, 25),
        (26, 1, 40, 32, 22),
        (11, 25, 31, 11, 32, 35, 40, 39),
    ):
        c = diagram_chain_costs(dims)
        better = c["search"] < c["greedy"] - _EPS
        rows.append((str(dims), c["initial"], c["greedy"], c["search"]))
        print(
            f"  {dims!s:<26} {c['initial']:>9.0f} "
            f"{c['greedy']:>9.0f} {c['search']:>9.0f} "
            f"{better!s:>13}"
        )
    return rows


def _egraph_section(n_rand: int, seed: int) -> dict[str, Any]:
    """Measure catopt's e-graph: saturation optimum vs the greedy oracle."""
    print()
    print("== 3b. catopt e-graph rule space (the priced ordering) ==")
    rules = all_rules()
    print(
        f"  {'dims':<26} {'init':>9} {'sat-opt':>9} {'2*DP':>9} "
        f"{'1-step':>9} {'greedy':>9}"
    )
    curated = []
    for dims in (
        (2, 3, 14, 14, 3, 8),
        (30, 35, 15, 5, 10, 20, 25),
        (26, 1, 40, 32, 22),
        (11, 25, 31, 11, 32, 35, 40, 39),
    ):
        term = _var_chain(dims)
        sat = egraph_sat_opt(term, rules)
        o1 = egraph_greedy(term, rules, horizon=1, patience=1)
        ge = egraph_greedy(term, rules)
        dp = 2.0 * chain_optimal(dims)
        curated.append((dims, sat, o1, ge, dp))
        print(
            f"  {dims!s:<26} {dag_cost(term, flops_cost):>9.0f} "
            f"{sat:>9.0f} {dp:>9.0f} {o1:>9.0f} {ge:>9.0f}"
        )
    one_step = []
    greedy = []
    sat_vs_dp = 0
    for dims in random_chains(n_rand, seed + 3):
        term = _var_chain(dims)
        sat = egraph_sat_opt(term, rules)
        if abs(sat - 2.0 * chain_optimal(dims)) > _EPS:
            sat_vs_dp += 1
        one_step.append(
            egraph_greedy(term, rules, horizon=1, patience=1) / sat
        )
        greedy.append(egraph_greedy(term, rules) / sat)
    print(
        f"  saturation == 2*DP optimum on "
        f"{n_rand - sat_vs_dp}/{n_rand} random chains"
    )
    print("  random n=4..8 chains, ratio vs saturated optimum:")
    _print_stats("one-step oracle", _ratio_stats(one_step))
    _print_stats("greedy episode h6", _ratio_stats(greedy))
    return {
        "curated": curated,
        "one_step": _ratio_stats(one_step),
        "greedy": _ratio_stats(greedy),
        "sat_matches_dp": (n_rand - sat_vs_dp, n_rand),
    }


def _verdict(
    chain: dict[str, Any],
    net: dict[str, Any],
    eg: dict[str, Any],
    diagram: list[tuple[str, float, float, float]],
) -> None:
    """Print the precondition verdict from the measured numbers."""
    print()
    print("== verdict ==")
    print(
        f"  classic matrix chains : greedy loses "
        f"{chain['stats']['pct']:.0f}% of random instances, "
        f"worst {chain['stats']['worst']:.1f}x"
    )
    print(
        f"  general networks      : greedy loses "
        f"{net['stats']['pct']:.0f}%, worst "
        f"{net['stats']['worst']:.1f}x"
    )
    print(
        f"  catopt e-graph space  : one-step oracle loses "
        f"{eg['one_step']['pct']:.0f}% (worst "
        f"{eg['one_step']['worst']:.2f}x); greedy episode loses "
        f"{eg['greedy']['pct']:.0f}%"
    )
    wins = sum(1 for _, _i, g, s in diagram if s < g - _EPS)
    print(
        f"  catopt diagram space  : search beats greedy on "
        f"{wins}/{len(diagram)} chain models"
    )
    print(
        "  precondition: contraction ordering is a domain where a"
        " policy can beat one-step greedy."
    )


def main(argv: list[str] | None = None) -> int:
    """Build the instances, measure, and print the tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    chain = _chain_section(args.instances, args.seed)
    net = _net_section(args.instances, args.seed)
    diagram = _diagram_section()
    eg = _egraph_section(max(20, args.instances // 2), args.seed)
    _verdict(chain, net, eg, diagram)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
