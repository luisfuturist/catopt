"""Contraction-ordering ladder — does greedy lose, and what closes it?

Plan 0016 stage 7 / ADR 0003 (the EVALUATION dimension).  The
contraction-ordering precondition
(`tools/contraction_precondition.py`,
`project/retros/contraction-precondition.md`) showed that the one-step
greedy cost-model oracle is *provably* suboptimal on matrix chains and
general tensor networks.  This suite turns that precondition into a
scored **ladder**: it builds a seeded hard-instance set and ranks every
non-learned player by cost against the exact DP optimum.

Players on the **priced e-graph rule space** (matrix chains over ``Var``
leaves, where ``assoc_matmul`` closes every bracketing and the products
are charged):

* ``greedy-1step``   — the one-step rule-delta oracle
  (:func:`catopt_core.trajectories.rule_samples`): apply the
  largest-delta rule once.
* ``greedy-episode`` — the RL script's greedy player (horizon 6,
  patience 2): the same oracle driven until it stalls.
* ``saturate``       — full equality saturation: the ceiling this rule
  set can reach.
* ``optimal``        — the exact ``O(n^3)`` interval DP, on the
  ``flops_cost`` scale (a matmul costs ``2·M·N·K``, so the DP optimum
  is ``2 x`` the scalar-multiplication DP).

Players on **general tensor networks** (3-6 tensors, mixed ranks; no
e-graph spelling exists — the retro's documented limitation):

* ``greedy-classic`` — contract the cheapest pair (min union size).
* ``optimal``        — the exact subset DP over the parity
  characterisation, validated here against exhaustive enumeration.

The headline is the honest one: saturation reaches the DP optimum on
*every* chain (the priced rule space is exact here), the one-step
oracle loses badly, and the multi-step episode nearly closes it — so
the residual headroom a learned ``Policy`` could chase is **search
cost**, not quality.  A learned player drops into :func:`_chain_players`
unchanged: it drives the same :class:`catopt_core.search_env.SearchEnv`
episode, so its cost is on the same ``flops_cost`` scale and the
ratio/rank statistics need no change.

CPU-only, no network, deterministic (seeded instance draws).
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from catopt_core.laws import all_rules

from bench.benchkit import (
    Case,
    Cell,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)
from tools.contraction_precondition import (
    _var_chain,
    chain_optimal,
    egraph_greedy,
    egraph_sat_opt,
    net_brute,
    net_greedy,
    net_optimal,
    random_chains,
    random_nets,
)

#: ``--quick`` shrinks the sweep (the CLI applies this dict).
QUICK = {"chains": 16, "nets": 12, "repeats": 1}

#: Ratio slack — a ratio within this of 1.0 counts as "at the optimum".
_EPS = 1e-9

#: The Runner settings for the marker variants.  Every case's real
#: measurement is a one-shot wall-clock run (build the e-graph, drive
#: the player, price it) — not a repeatable per-call kernel — so the
#: variants carry instant markers and their ``medians`` are backfilled
#: with the single best-of-``repeats`` time (IQR 0; see
#: ``search_efficiency`` for the same pattern).  A ~1 ms autorange keeps
#: the marker timing itself negligible.
_RUNNER_WARMUP = 0
_RUNNER_MIN_RUN_TIME = 0.001

_ONESHOT_NOTE = (
    "one-shot wall-clock; median backfilled from the best-of-k run "
    "(IQR 0), quality stats in aux"
)


def _marker() -> None:
    """Instant stand-in for a one-shot measurement."""
    return None


# ---------------------------------------------------------------------------
#  Players
# ---------------------------------------------------------------------------


def _chain_players(
    rules: Any,
) -> list[tuple[str, Callable[[tuple[int, ...]], float]]]:
    """The chain ladder for a given rule set, in report order.

    Each player maps a ``dims`` tuple to a final cost on the
    ``flops_cost`` scale (``saturate``'s units).  A learned ``Policy``
    player slots in here as ``("learned", learned)``, where ``learned``
    drives the same :class:`~catopt_core.search_env.SearchEnv` episode
    and returns ``env.cost`` — the ratio/rank statistics below need no
    change.  Deliberately not populated today: this suite scores the
    non-learned players only (the learned reward is reworked
    elsewhere), and must not depend on it.
    """

    def one_step(dims: tuple[int, ...]) -> float:
        return egraph_greedy(
            _var_chain(dims), rules, horizon=1, patience=1
        )

    def episode(dims: tuple[int, ...]) -> float:
        return egraph_greedy(
            _var_chain(dims), rules, horizon=6, patience=2
        )

    def saturate(dims: tuple[int, ...]) -> float:
        return egraph_sat_opt(_var_chain(dims), rules)

    def optimal(dims: tuple[int, ...]) -> float:
        return 2.0 * chain_optimal(dims)

    return [
        ("greedy-1step", one_step),
        ("greedy-episode", episode),
        ("saturate", saturate),
        ("optimal", optimal),
    ]


def _net_players() -> list[
    tuple[str, Callable[[tuple[Any, Any]], float]]
]:
    """The network ladder — classic greedy vs the exact subset DP.

    There is no e-graph spelling of a general tensor network (the
    retro's documented limitation), so the priced players do not apply
    here; the net family measures the textbook baseline directly.
    """

    def greedy(inst: tuple[Any, Any]) -> float:
        return net_greedy(inst[0], inst[1])

    def optimal(inst: tuple[Any, Any]) -> float:
        return net_optimal(inst[0], inst[1])

    return [("greedy-classic", greedy), ("optimal", optimal)]


# ---------------------------------------------------------------------------
#  Scoring helpers
# ---------------------------------------------------------------------------


def _ranks_asc(values: list[float]) -> list[float]:
    """1-based average ranks, smallest value ranks first (ties share)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while (
            j + 1 < len(values)
            and values[order[j + 1]] == values[order[i]]
        ):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _measure(
    players: list[tuple[str, Callable[[Any], float]]],
    instance: Any,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score every player once; return ``(cost, best-of-k wall time)``."""
    costs: dict[str, float] = {}
    times: dict[str, float] = {}
    for name, fn in players:
        best = float("inf")
        cost = float("inf")
        for _ in range(max(repeats, 1)):
            t0 = time.perf_counter()
            cost = fn(instance)
            best = min(best, time.perf_counter() - t0)
        costs[name] = cost
        times[name] = best
    return costs, times


def _cell(
    runner: Runner,
    name: str,
    family: str,
    params: dict,
    players: list[tuple[str, Callable[[Any], float]]],
    instance: Any,
    repeats: int,
) -> Cell:
    """Build, time and score one instance as a benchkit ``Cell``."""
    costs, times = _measure(players, instance, repeats)
    opt = costs["optimal"]
    names = [p for p, _ in players]
    ratios = {p: costs[p] / opt for p in names}
    ranks = dict(
        zip(names, _ranks_asc([costs[p] for p in names]), strict=True)
    )
    aux = {
        "family": family,
        "players": {
            p: {
                "cost": costs[p],
                "ratio": ratios[p],
                "rank": ranks[p],
            }
            for p in names
        },
        "optimal": opt,
        **params,
    }
    case = Case(
        name=name,
        params=dict(params),
        variants=[
            Variant(p, _marker, note=_ONESHOT_NOTE) for p in names
        ],
        aux=aux,
    )
    cell = runner.run_case(case)
    for p in names:
        cell.medians[p] = times[p]
        cell.iqr[p] = 0.0
    return cell


# ---------------------------------------------------------------------------
#  Aggregation + findings
# ---------------------------------------------------------------------------


def _agg(cells: list[Cell], family: str, player: str) -> dict | None:
    """Aggregate one player's ratios/ranks over one instance family."""
    rows = [
        c.aux["players"][player]
        for c in cells
        if c.aux["family"] == family and player in c.aux["players"]
    ]
    if not rows:
        return None
    ratios = [r["ratio"] for r in rows]
    ranks = [r["rank"] for r in rows]
    n = len(ratios)
    at_opt = sum(1 for r in ratios if r <= 1.0 + _EPS)
    return {
        "n": n,
        "mean": statistics.fmean(ratios),
        "median": statistics.median(ratios),
        "at_opt": at_opt,
        "pct_opt": 100.0 * at_opt / n,
        "worst": max(ratios),
        "mean_rank": statistics.fmean(ranks),
    }


def _findings(cells: list[Cell]) -> list[Finding]:
    """The typed ladder conclusions, all read off the measured cells."""
    one = _agg(cells, "chain", "greedy-1step")
    epi = _agg(cells, "chain", "greedy-episode")
    sat = _agg(cells, "chain", "saturate")
    net_g = _agg(cells, "net", "greedy-classic")
    net_o = _agg(cells, "net", "optimal")
    assert one and epi and sat and net_g and net_o

    sat_exact = sat["at_opt"] == sat["n"]
    headroom = (
        "no — the priced rule space already reaches the DP optimum: "
        f"saturate ties optimal on {sat['at_opt']}/{sat['n']} chains; "
        f"the residual headroom above greedy is search cost, not "
        "quality"
        if sat_exact
        else f"saturate misses optimal on "
        f"{sat['n'] - sat['at_opt']}/{sat['n']} chains"
    )
    return [
        Finding(
            claim=(
                "a learned search policy can beat greedy on "
                "contraction-ordering *quality*"
            ),
            verdict=Verdict.NEGATIVE if sat_exact else Verdict.WIN,
            headline=headroom,
            metric="saturate chains at the DP optimum",
            value=round(sat["at_opt"] / sat["n"], 3),
            evidence={
                "chain": {
                    "greedy-1step": one,
                    "greedy-episode": epi,
                    "saturate": sat,
                },
                "net": {"greedy-classic": net_g, "optimal": net_o},
            },
        ),
        Finding(
            claim=(
                "full equality saturation reaches the exact "
                "contraction-ordering DP optimum on matrix chains"
            ),
            verdict=Verdict.WIN if sat_exact else Verdict.NEGATIVE,
            headline=(
                f"saturate == 2·DP optimum on {sat['at_opt']}/"
                f"{sat['n']} chains (worst ratio {sat['worst']:.3f})"
            ),
            metric="saturate chains at optimum / n",
            value=round(sat["at_opt"] / sat["n"], 3),
            evidence={"saturate": sat},
        ),
        Finding(
            claim=(
                "the one-step greedy cost-model oracle is optimal on "
                "contraction ordering"
            ),
            verdict=(
                Verdict.PARITY
                if one["at_opt"] == one["n"]
                else Verdict.NEGATIVE
            ),
            headline=(
                f"one-step oracle at the optimum on {one['at_opt']}/"
                f"{one['n']} chains ({one['pct_opt']:.0f}%); mean "
                f"ratio {one['mean']:.2f}, worst {one['worst']:.2f}x"
            ),
            metric="one-step oracle chains at optimum / n",
            value=round(one["at_opt"] / one["n"], 3),
            evidence={"greedy-1step": one},
        ),
        Finding(
            claim=(
                "the multi-step greedy episode is optimal on "
                "contraction ordering"
            ),
            verdict=(
                Verdict.PARITY
                if epi["at_opt"] == epi["n"]
                else Verdict.NEGATIVE
            ),
            headline=(
                f"greedy episode (h6/p2) at the optimum on "
                f"{epi['at_opt']}/{epi['n']} chains vs one-step "
                f"{one['at_opt']}/{one['n']}; worst {epi['worst']:.2f}x"
            ),
            metric="greedy episode chains at optimum / n",
            value=round(epi["at_opt"] / epi["n"], 3),
            evidence={"greedy-episode": epi, "greedy-1step": one},
        ),
        Finding(
            claim=(
                "classic greedy is optimal on general tensor networks"
            ),
            verdict=(
                Verdict.PARITY
                if net_g["at_opt"] == net_g["n"]
                else Verdict.NEGATIVE
            ),
            headline=(
                f"classic greedy at the optimum on {net_g['at_opt']}/"
                f"{net_g['n']} networks ({net_g['pct_opt']:.0f}%); "
                f"mean ratio {net_g['mean']:.2f}, worst "
                f"{net_g['worst']:.1f}x"
            ),
            metric="classic greedy networks at optimum / n",
            value=round(net_g["at_opt"] / net_g["n"], 3),
            evidence={"greedy-classic": net_g, "optimal": net_o},
        ),
    ]


# ---------------------------------------------------------------------------
#  Console reporting
# ---------------------------------------------------------------------------

_CHAIN_ORDER = (
    "greedy-1step",
    "greedy-episode",
    "saturate",
    "optimal",
)
_NET_ORDER = ("greedy-classic", "optimal")


def _print_ladder(cells: list[Cell], family: str, order: tuple) -> None:
    """Print one family's per-player ratio/rank ladder."""
    print(f"\n  {family} ladder:")
    head = (
        f"  {'player':<16} {'n':>4} {'mean':>7} {'median':>7} "
        f"{'%opt':>6} {'worst':>7} {'rank':>6}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for p in order:
        s = _agg(cells, family, p)
        if s is None:
            continue
        print(
            f"  {p:<16} {s['n']:>4} {s['mean']:>7.3f} "
            f"{s['median']:>7.3f} {s['pct_opt']:>5.0f}% "
            f"{s['worst']:>7.3f} {s['mean_rank']:>6.2f}"
        )


def _print_ties(cells: list[Cell], family: str, player: str) -> None:
    """State how many instances the given player ties the optimum."""
    s = _agg(cells, family, player)
    if s is None:
        return
    print(
        f"  {family}/{player}: ties the optimum on {s['at_opt']}/"
        f"{s['n']} instances ({s['pct_opt']:.0f}%)"
    )


# ---------------------------------------------------------------------------
#  Harness entry point
# ---------------------------------------------------------------------------


def _validate_subset_dp(n: int, seed: int) -> int:
    """Re-validate the subset DP against brute force (self-check).

    Returns the number of cases checked; raises if the DP ever
    disagrees with exhaustive enumeration.  Keeps the suite honest
    about the "optimal" it scores against.
    """
    checked = 0
    for tensors, sizes in random_nets(n, seed + 7):
        dp = net_optimal(tensors, sizes)
        brute = net_brute(tensors, sizes)
        assert abs(dp - brute) < _EPS, (
            f"subset DP {dp} != exhaustive {brute} on {tensors}"
        )
        checked += 1
    return checked


def run_bench(args: argparse.Namespace) -> Report:
    """Build the hard-instance set, score the ladder, return a report."""
    n_chains = int(getattr(args, "chains", None) or 60)
    n_nets = int(getattr(args, "nets", None) or 60)
    seed = int(getattr(args, "seed", None) or 0)
    repeats = int(getattr(args, "repeats", None) or 3)

    rules = all_rules()
    chain_players = _chain_players(rules)
    net_players = _net_players()
    runner = Runner(
        device="cpu",
        warmup=_RUNNER_WARMUP,
        min_run_time=_RUNNER_MIN_RUN_TIME,
    )

    checked = _validate_subset_dp(min(n_nets, 40), seed)

    cells: list[Cell] = []
    for i, dims in enumerate(random_chains(n_chains, seed)):
        cells.append(
            _cell(
                runner,
                name=f"chain-{i:02d}",
                family="chain",
                params={"n_mats": len(dims) - 1, "dims": str(dims)},
                players=chain_players,
                instance=dims,
                repeats=repeats,
            )
        )
    for i, (tensors, sizes) in enumerate(random_nets(n_nets, seed + 1)):
        cells.append(
            _cell(
                runner,
                name=f"net-{i:02d}",
                family="net",
                params={
                    "n_tensors": len(tensors),
                    "tensors": str(tensors),
                },
                players=net_players,
                instance=(tensors, sizes),
                repeats=repeats,
            )
        )

    print("== contraction-ordering ladder ==")
    print(
        f"  chains: {n_chains} (4-8 matrices, dims 1..40, seed {seed});"
        f" nets: {n_nets} (3-6 tensors, seed {seed + 1})"
    )
    print(
        f"  subset DP validated vs exhaustive enumeration on {checked}"
    )
    _print_ladder(cells, "chain", _CHAIN_ORDER)
    _print_ladder(cells, "net", _NET_ORDER)
    print()
    _print_ties(cells, "chain", "saturate")
    _print_ties(cells, "chain", "greedy-1step")
    _print_ties(cells, "chain", "greedy-episode")
    _print_ties(cells, "net", "greedy-classic")

    return Report(
        suite="contraction_ladder",
        title="Contraction-ordering ladder",
        summary=(
            "A seeded hard-instance set of 4-8 matrix chains and 3-6 "
            "tensor networks, scoring the non-learned contraction-"
            "ordering players (one-step oracle, greedy episode, "
            "saturation, classic greedy) by cost and rank against the "
            "exact DP optimum."
        ),
        findings=_findings(cells),
        cells=cells,
        env=collect_env("cpu"),
        provenance={
            "chains": n_chains,
            "nets": n_nets,
            "seed": seed,
            "repeats": repeats,
            "n_rules": len(rules),
            "subset_dp_validated": checked,
        },
    )


def main(argv: list[str] | None = None) -> None:
    """Direct entry point (``python -m bench.suites.evaluation...``)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chains", type=int, default=60)
    ap.add_argument("--nets", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        for key, val in QUICK.items():
            setattr(args, key, val)
    report = run_bench(args)
    if not args.no_artifacts:
        out = Path(args.out)
        report.to_json(out / "contraction_ladder.json")
        report.to_markdown(out / "contraction_ladder.md")
        report.to_html(out / "contraction_ladder.html")
        print(f"[artifacts] {out}/contraction_ladder.{{json,md,html}}")


if __name__ == "__main__":
    main()
