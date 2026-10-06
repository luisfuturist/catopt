"""The law-order board — does saturation order move the answer.

Plan 0021 depth question 1.  The contraction player won on its board;
the law-order board — the order laws are applied during e-graph
saturation + extraction — was unmeasured.  This module is the probe:
run one term to a (possibly bounded) fixed point under several rule
orderings and compare the three observables:

* **contents** — enode/class counts, per-rule fire counts, suspended
  budgets.  These CAN differ by order: instantiation resolves
  children through the *current* union-find (an early merge can
  collapse a later rule's RHS onto an existing enode), and per-rule
  enode budgets truncate the expansive closure at order-dependent
  subsets.
* **answer** — the extracted lowest-cost term: its cost, its
  spelling, and the priced top-``k`` frontier of the root class.
* **clock** — wall time to the run's stop reason.

Safety is not in question — a policy may only reorder the rules
:meth:`catopt_core.egraph.EGraph.run` already fires (ADR 0003
invariant 5), and every merge is law-witnessed.  ``run_board`` still
verifies each arm's certificate so the probe reports
order-robustness *of the proofs*, not just of the costs.

Reproduce the corpus numbers::

    python -m catopt_discovery.law_order --corpus models --regime prod
    python -m catopt_discovery.law_order --corpus zoo --regime prod
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import (
    CertificateVerificationError,
    EGraph,
    verify_certificate,
)
from catopt_core.ir import op_repr
from catopt_core.laws import DEFAULT, tags
from catopt_core.policies import GreedyPolicy, RandomPolicy

__all__ = [
    "BoardRun",
    "BoardVerdict",
    "compare_runs",
    "model_terms",
    "order_arms",
    "probe_term",
    "production_budgets",
    "report_lines",
    "run_board",
    "sweep",
    "zoo_terms",
]

#: One arm means one ordering: ``None`` is the engine's declared order.
Arm = tuple[str, Any]


@dataclass(frozen=True)
class BoardRun:
    """One saturated e-graph under one ordering — the observables."""

    arm: str
    iterations: int
    n_enodes: int
    n_classes: int
    n_proof_edges: int
    rule_fires: tuple[tuple[str, int], ...]
    suspended: tuple[str, ...]
    stop: str
    cost: float
    term_repr: str
    frontier: tuple[tuple[float, str], ...]
    wall_s: float
    certificate_ok: bool


@dataclass(frozen=True)
class BoardVerdict:
    """The order-sensitivity summary for one (term, regime) cell.

    ``contents_equal`` compares the materialized closure (enodes,
    proof edges, per-rule fire counts); ``partition_equal`` the
    quotient size (class count); ``answer_equal`` the extracted
    term's cost AND spelling; ``frontier_equal`` the priced top-k
    alternative surface of the root class.  ``wall_ratio`` is
    max/min wall-clock across arms.
    """

    n_arms: int
    contents_equal: bool
    partition_equal: bool
    answer_equal: bool
    frontier_equal: bool
    certificates_ok: bool
    cost: float
    wall_ratio: float


def production_budgets(
    rules: Any, symmetry_budget: int | None = 2048
) -> dict[str, int] | None:
    """Return the pipeline's bounded-saturation map (``EXPANSIVE``).

    Mirrors ``search()``'s ``symmetry_budget`` knob: every
    ``EXPANSIVE``-tagged rule may contribute at most that many new
    enodes over the run's lifetime.  ``None`` → unbounded.
    """
    if symmetry_budget is None:
        return None
    return {
        r.name: symmetry_budget for r in rules.tagged(tags.EXPANSIVE)
    }


def order_arms(
    rules: Any, seeds: tuple[int, ...] = (0, 7)
) -> tuple[Arm, ...]:
    """Return the standard ordering arms for one rule set.

    ``declared`` is the engine's own order (``policy=None``);
    ``reversed`` and the two ``expansive_*`` arms are greedy priority
    orderings (declaration order inverted; the closure-generating
    ``EXPANSIVE`` rules pushed last / first); ``random:<seed>`` is a
    seeded uniform reordering per iteration.  All arms use the
    shipped :mod:`catopt_core.policies` players — the probe changes
    the schedule, never the rule set.
    """
    rs = list(rules)
    arms: list[Arm] = [
        ("declared", None),
        (
            "reversed",
            GreedyPolicy({r.name: float(-i) for i, r in enumerate(rs)}),
        ),
    ]
    for seed in seeds:
        arms.append((f"random:{seed}", RandomPolicy(seed)))
    prio_last = {
        r.name: 20.0 if tags.EXPANSIVE in r.tags else 0.0 for r in rs
    }
    prio_first = {
        r.name: 0.0 if tags.EXPANSIVE in r.tags else 20.0 for r in rs
    }
    arms.append(("expansive_last", GreedyPolicy(prio_last)))
    arms.append(("expansive_first", GreedyPolicy(prio_first)))
    return tuple(arms)


def run_board(
    term: Any,
    rules: Any,
    *,
    arm: str = "declared",
    policy: Any = None,
    cost_fn: Any = None,
    top_k: int = 8,
    verify: bool = True,
    **run_kw: Any,
) -> BoardRun:
    """Saturate one term under one ordering; record the observables.

    A fresh :class:`EGraph` per arm — the board is the whole run,
    not a branch.  ``**run_kw`` forwards to :meth:`EGraph.run`
    (``max_iterations`` / ``max_nodes`` / ``rule_budgets`` /
    ``stop`` / ``patience``); ``cost_fn`` prices extraction and is
    always forwarded — ``stop="improving"`` needs it and the
    fixed-point loop ignores it.
    """
    cf = cost_fn if cost_fn is not None else flops_cost
    eg = EGraph()
    root = eg.add_term(term)
    t0 = time.perf_counter()
    stats = eg.run(rules, root, policy=policy, cost_fn=cf, **run_kw)
    wall = time.perf_counter() - t0
    best = eg.extract_best(root, cf)
    cost = dag_cost(best, cf) if best is not None else float("inf")
    frontier = tuple(
        (round(c, 9), op_repr(t))
        for c, t in eg.extract_alternatives(root, cf, top_k=top_k)
    )
    ok = False
    if verify and best is not None:
        cert = eg.certificate(term, best, root_eid=root, cost_fn=cf)
        try:
            replayed = verify_certificate(term, cert)
            ok = op_repr(replayed) == op_repr(best)
        except CertificateVerificationError:
            ok = False
    return BoardRun(
        arm=arm,
        iterations=stats["iterations"],
        n_enodes=stats["n_enodes"],
        n_classes=stats["n_classes"],
        n_proof_edges=stats["n_proof_edges"],
        rule_fires=tuple(sorted(eg.rule_fires.items())),
        suspended=tuple(stats["budget_suspended"]),
        stop=stats["stop"],
        cost=cost,
        term_repr=op_repr(best) if best is not None else "<none>",
        frontier=frontier,
        wall_s=wall,
        certificate_ok=ok,
    )


def probe_term(
    term: Any,
    rules: Any,
    arms: tuple[Arm, ...] | None = None,
    **kw: Any,
) -> tuple[BoardRun, ...]:
    """Run every ordering arm on one term; fresh e-graph per arm."""
    # Recursive walks (extraction, member resolution) descend the
    # e-class DAG, whose depth grows with the saturation closure —
    # same accommodation ``optimize.search`` makes.
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)
    if arms is None:
        arms = order_arms(rules)
    return tuple(
        run_board(term, rules, arm=name, policy=pol, **kw)
        for name, pol in arms
    )


def compare_runs(runs: tuple[BoardRun, ...]) -> BoardVerdict:
    """Fold one term's arm matrix into the order-sensitivity verdict."""
    first = runs[0]
    contents = [
        (r.n_enodes, r.n_proof_edges, r.rule_fires) for r in runs
    ]
    answers = [(r.cost, r.term_repr) for r in runs]
    walls = [r.wall_s for r in runs]
    return BoardVerdict(
        n_arms=len(runs),
        contents_equal=len(set(contents)) == 1,
        partition_equal=len({r.n_classes for r in runs}) == 1,
        answer_equal=len(set(answers)) == 1,
        frontier_equal=len({r.frontier for r in runs}) == 1,
        certificates_ok=all(r.certificate_ok for r in runs),
        cost=first.cost,
        wall_ratio=(max(walls) / min(walls) if min(walls) > 0 else 1.0),
    )


def zoo_terms(seed: int = 0) -> tuple[list[tuple[str, Any]], list[str]]:
    """Export the 22-model held-out zoo to terms; returns (terms, errors).

    Torch is imported lazily — the probe harness itself is torch-free;
    only corpus construction needs the export boundary.
    """
    import torch
    from catopt_torch.torch_bridge import export_to_ir

    from catopt_discovery.zoo import zoo

    torch.manual_seed(seed)
    out: list[tuple[str, Any]] = []
    errors: list[str] = []
    for w in zoo():
        try:
            model, x = w.build()
            feed = x if isinstance(x, tuple) else (x,)
            ir, _ = export_to_ir(model.eval().double(), feed)
        except Exception as e:  # honest per-case failure
            errors.append(f"{w.name}: {type(e).__name__}: {e}")
            continue
        out.append((w.name, ir.root))
    return out, errors


def model_terms() -> tuple[list[tuple[str, Any]], list[str]]:
    """Export the real-model corpus (``impact._model_cases``) to terms."""
    from catopt_discovery.impact import model_cases

    cases, errors = model_cases()
    return [(c.name, c.term) for c in cases], errors


def sweep(
    cases: list[tuple[str, Any]],
    rules: Any,
    regimes: dict[str, dict[str, Any]],
    *,
    arms: tuple[Arm, ...] | None = None,
    seeds: tuple[int, ...] = (0, 7),
    top_k: int = 8,
    cost_fn: Any = None,
) -> list[dict[str, Any]]:
    """Run the arm matrix over ``cases`` x ``regimes``.

    ``regimes`` maps a label to ``EGraph.run`` kwargs.  When ``arms``
    is unset each (case, regime) cell gets fresh arms from
    ``order_arms(rules, seeds)`` — seeded policies carry RNG state,
    so reusing one arm set across cells would sample a continued
    stream rather than a replayable ordering.  Rows carry the case
    name, the regime, the :class:`BoardVerdict`, and the declared
    arm's :class:`BoardRun` (``base``) for size/seed provenance.
    """
    rows: list[dict[str, Any]] = []
    for name, term in cases:
        for reg_name, regime_kw in regimes.items():
            cell_arms = (
                arms if arms is not None else order_arms(rules, seeds)
            )
            run_kw = dict(regime_kw)
            run_cf = run_kw.pop("cost_fn", None) or cost_fn
            runs = probe_term(
                term,
                rules,
                arms=cell_arms,
                cost_fn=run_cf,
                top_k=top_k,
                **run_kw,
            )
            rows.append(
                {
                    "name": name,
                    "regime": reg_name,
                    "verdict": compare_runs(runs),
                    "base": runs[0],
                    "runs": runs,
                }
            )
    return rows


def report_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Render the sweep table — one line per (case, regime) cell."""
    lines: list[str] = []
    for row in rows:
        v: BoardVerdict = row["verdict"]
        b: BoardRun = row["base"]
        flags = []
        if not v.contents_equal:
            flags.append("contents")
        if not v.partition_equal:
            flags.append("partition")
        if not v.answer_equal:
            flags.append("ANSWER")
        if not v.frontier_equal:
            flags.append("frontier")
        if not v.certificates_ok:
            flags.append("CERT-FAIL")
        lines.append(
            f"{row['name']:<24} {row['regime']:<10} "
            f"pe={b.n_proof_edges:<5} ne={b.n_enodes:<6} "
            f"it={b.iterations:<3} stop={b.stop:<12} "
            f"cost={b.cost:<12.1f} wall={b.wall_s * 1e3:8.1f}ms "
            f"wall-ratio={v.wall_ratio:5.1f}x "
            f"diffs={','.join(flags) if flags else '-'}"
        )
    return lines


def _default_regimes(
    rules: Any, cost_fn: Any
) -> dict[str, dict[str, Any]]:
    """Return the three saturation regimes the probe measures."""
    return {
        "prod": {
            "rule_budgets": production_budgets(rules, 2048),
            "max_nodes": 100_000,
        },
        "tight": {
            "rule_budgets": production_budgets(rules, 128),
            "max_nodes": 20_000,
        },
        "improving": {
            "rule_budgets": production_budgets(rules, 2048),
            "max_nodes": 60_000,
            "stop": "improving",
            "patience": 3,
            "cost_fn": cost_fn,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the board probe on a corpus; print the sweep table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        choices=("zoo", "models"),
        default="models",
        help="'zoo' is the 22-model holdout; 'models' the real "
        "corpus where the shipped laws actually fire",
    )
    p.add_argument(
        "--regime",
        choices=("prod", "tight", "improving", "all"),
        default="all",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=(0, 7),
        help="random-ordering seeds",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="first N cases only"
    )
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args(argv)

    if args.corpus == "zoo":
        cases, errors = zoo_terms()
    else:
        cases, errors = model_terms()
    if args.limit:
        cases = cases[: args.limit]
    for e in errors:
        print(f"  export error: {e}")  # stdout-compat
    rules = DEFAULT
    regimes = _default_regimes(rules, flops_cost)
    if args.regime != "all":
        regimes = {args.regime: regimes[args.regime]}
    rows = sweep(
        cases,
        rules,
        regimes,
        seeds=tuple(args.seeds),
        top_k=args.top_k,
        cost_fn=flops_cost,
    )
    n_arms = len(order_arms(rules, seeds=tuple(args.seeds)))
    print(  # stdout-compat
        f"== law-order board: {args.corpus}, {len(cases)} cases, "
        f"{n_arms} arms =="
    )
    n_answer = n_contents = n_cert_fail = 0
    for line in report_lines(rows):
        print(line)  # stdout-compat
    for row in rows:
        v = row["verdict"]
        n_answer += not v.answer_equal
        n_contents += not v.contents_equal
        n_cert_fail += not v.certificates_ok
    print(  # stdout-compat
        f"cells={len(rows)} answer-diff={n_answer} "
        f"contents-diff={n_contents} cert-fail={n_cert_fail}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
