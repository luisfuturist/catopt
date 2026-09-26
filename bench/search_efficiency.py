"""Search-efficiency benchmark: equivalent-program space vs e-graph size.

Quantifies the "finds it cheaply" half of the equality-saturation claim
on a family where the equivalent-program space grows combinatorially:

    a left/right-nested chain of ``k`` matmuls  W0 @ W1 @ ... @ W{k-1}

``assoc_matmul``/``assoc_matmul_rev`` are the only rules that fire
(matrix multiplication is associative but NOT commutative), so the
equivalence class of the chain is exactly the Catalan(k-1) bracketings
— ~4^(k-1) programs at k=24 (~3.4e11).  In the saturated e-graph the
closure compresses to one e-class per contiguous sub-product
(k(k+1)/2 classes) and one e-node per (interval, split-point) pair
((k^3-k)/6 + k live e-nodes) — polynomial in k.  The script verifies
the compression exactly: it counts the distinct programs encoded at
the root e-class by a DAG DP and checks it equals Catalan(k-1).

Two saturation policies are measured:

* ``exact``   — unbounded ``EGraph.run`` (the full closure).  Exact but
  expensive in this pure-Python engine past k ~ 11, because the generic
  matcher re-enumerates substitutions over fragmented intermediate
  classes.  Capped by ``--exact-max`` (default 11).
* ``bounded`` — the production policy from ``optimize_model``:
  ``rule_budgets`` caps each symmetry rule at ``--budget`` new e-nodes
  (bounded saturation).  Runs at every depth; the retained fragment
  still encodes an astronomically large set of programs, reported via
  the same term-count DP.

Also measured per depth:

* ``meta.canonicalize`` / ``stratified_run`` — the repo's engineered
  answer to symmetry blowup: coherent laws are *computed* (flatten +
  rebuild balanced) rather than searched, O(k) wall time.
* ``optimize_model`` end-to-end on an exported ``nn.Sequential`` of
  ``nn.Linear(bias=False)`` — the same assoc family reached through
  ``assoc_linear`` on the real export path (typing, attrs validation,
  pairing passes, extraction, lowering, verification).

Reporting goes through ``benchkit``: each (policy, k) cell is a
``Case`` carrying the sweep coordinates in ``params`` and every
measured quantity — space_log10, enodes, live e-nodes, e-classes,
iterations, rule fires, encode-DP count, wall/extract ms — in ``aux``
(the data channel; saturation is a one-shot stateful wall-clock run,
not a repeatable per-call kernel, so ``Timer`` autorange does not
apply).  Cells are collected into ``Report(suite="search_efficiency")``
which emits JSON + Markdown + grouped-bar PNGs under ``--out``
(default ``bench/results/``); on top of the generic artifacts the
script writes the two suite-specific plots: the money plot
(program-space size vs live e-nodes vs k, log-y) and wall_ms vs k.

Usage:

    .venv/bin/python bench/search_efficiency.py --device cpu
    .venv/bin/python bench/search_efficiency.py --depths 4,8,16 --out /tmp/se
    .venv/bin/python bench/search_efficiency.py --no-artifacts   # console only
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit import Case, Cell, Report, Runner, Variant, collect_env
from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, Param, TensorType
from catopt_core.meta import canonicalize, stratified_run
from catopt_core.laws import (
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    all_rules,
)

#: Hidden dims pattern for the chain (varied so bracketings genuinely
#: differ in FLOP cost — extraction solves the matrix-chain-ordering
#: problem, not a trivial all-equal pick).
_DIMS = (8, 512, 32, 256)

#: Rules classified as pure symmetry — the expansive closure generators
#: that ``optimize_model`` puts under an enode budget.
_ASSOC_RULES = (ASSOC_MATMUL, ASSOC_MATMUL_REV)

#: Runner settings for the marker variants.  The measured quantity of
#: every case here is a one-shot, stateful wall-clock run (build the
#: e-graph, saturate, count) — not a repeatable per-call kernel — so
#: ``blocked_autorange`` repetition does not apply.  ``Variant.stmt``
#: is therefore an instant marker; the variant's ``medians`` entry is
#: backfilled with the single measured wall time (N=1 → IQR 0) and the
#: full statistics live in ``Case.aux``.  warmup=0 + a tiny
#: min_run_time keep the marker timing itself negligible.
_RUNNER_WARMUP = 0
_RUNNER_MIN_RUN_TIME = 0.001

_ONESHOT_NOTE = (
    "one-shot wall-clock measurement; median backfilled from aux "
    "(wall_ms etc.), stats in aux"
)


def _marker() -> None:
    """Instant stand-in for a one-shot measurement (see module docstring)."""
    return None


def catalan(n: int) -> int:
    """Catalan(n) — binary bracketings of n+1 ordered leaves."""
    return math.comb(2 * n, n) // (n + 1)


def chain_dims(k: int) -> list[int]:
    return [_DIMS[i % len(_DIMS)] for i in range(k + 1)]


def matmul_chain(k: int) -> Any:
    """Right comb ``W0 @ (W1 @ (... @ W{k-1}))`` over k Param leaves.

    The right-nested seed matches ``assoc_matmul``'s LHS immediately;
    with both assoc directions the saturation closure is the full
    Tamari lattice of Catalan(k-1) bracketings.
    """
    dims = chain_dims(k)
    leaves = [
        Param(f"W{i}", TensorType((dims[i], dims[i + 1])))
        for i in range(k)
    ]
    term = leaves[-1]
    for leaf in reversed(leaves[:-1]):
        term = Op.make("matmul", leaf, term)
    return term


def live_enodes(eg: EGraph) -> int:
    """Member e-nodes actually held in e-classes, post-rebuild.

    ``eg.n_enodes`` is the hash-cons table size — it also retains
    pre-canonicalisation enodes minted mid-saturation.  This count is
    the e-graph's real information content (and, at the exact fixed
    point, equals the theoretical (k^3-k)/6 + k)."""
    return sum(len(ec.nodes) for ec in eg._classes.values())


def count_derivations(
    eg: EGraph, root_eid: int, cap: int = 10**25
) -> int:
    """Parse-derivation count of the root e-class, by DAG DP.

    count(C) = sum over member enodes of prod(count(child class));
    leaves contribute 1, cyclic branches contribute 0.  This is an
    UPPER BOUND on distinct programs: in a fragmented (partially
    merged) graph two enodes can derive the same surface term through
    different child classes.  At the exact fixed point the interval
    classes are canonical, derivations are injective, and the count
    provably equals Catalan(k-1).
    """
    memo: dict[int, int] = {}

    def cnt(eid: int) -> int:
        eid = eg.find(eid)
        if eid in memo:
            return memo[eid]
        memo[eid] = 0  # visiting marker: cyclic branches contribute 0
        total = 0
        for node in eg._classes[eid].nodes:
            if node.op == "leaf":
                total += 1
                continue
            prod = 1
            for child in node.children:
                if eg.find(child) == eid:
                    prod = 0
                    break
                prod *= cnt(child)
            total += prod
            if total > cap:
                return cap
        memo[eid] = total
        return total

    return cnt(root_eid)


def count_distinct_terms(
    eg: EGraph, root_eid: int, cap: int = 400_000
) -> tuple[int, bool]:
    """Distinct program terms encoded by the root e-class.

    Terms are interned by structure ((op, child-term-id, ...) tuples
    become a fresh int), so the per-class member set is a set of ints —
    the same DP as :func:`count_derivations` but with dedup across
    enodes/derivations.  Enumeration stops once a class's member set
    exceeds *cap* and returns ``(count, capped=True)`` — the count is
    then a certified LOWER bound.
    """
    intern: dict[Any, int] = {}
    memo: dict[int, frozenset[int]] = {}
    capped = False

    def term_id(key: Any) -> int:
        t = intern.get(key)
        if t is None:
            t = len(intern)
            intern[key] = t
        return t

    def go(eid: int) -> frozenset[int]:
        nonlocal capped
        eid = eg.find(eid)
        if eid in memo:
            return memo[eid]
        memo[eid] = frozenset()  # visiting marker: cyclic -> empty
        out: set[int] = set()
        for node in eg._classes[eid].nodes:
            if node.op == "leaf":
                out.add(term_id(("leaf", node.attrs[0][1])))
                continue
            if any(eg.find(c) == eid for c in node.children):
                continue
            child_sets = [go(c) for c in node.children]
            prod_size = 1
            for cs in child_sets:
                prod_size *= len(cs)
            if prod_size + len(out) > cap:
                # This enode alone would blow the cap: truncate the
                # member set — the count stays a certified lower bound.
                capped = True
                continue
            for combo in _product(child_sets):
                out.add(term_id((node.op, *combo)))
        memo[eid] = frozenset(out)
        return memo[eid]

    result = go(root_eid)
    return len(result), capped


def _product(sets: list[frozenset[int]]):
    """Iterate the cartesian product of child term-id sets lazily."""
    return itertools.product(*sets)


def log10n(n: int | float) -> float:
    """log10 that survives ints larger than float64 can hold."""
    if n <= 0:
        return float("-inf")
    s = str(int(n))
    if len(s) <= 15:
        return math.log10(int(s))
    return (len(s) - 15) + math.log10(int(s[:15]))


def measure_run(
    k: int,
    rules: list,
    rule_budgets: dict[str, int] | None,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """One saturation measurement at depth k."""
    eg = EGraph(truncation_level=1)  # pure quotient: bench measures
    # search space, not proof-tracking overhead.
    term = matmul_chain(k)
    root = eg.add_term(term)
    t0 = time.perf_counter()
    stats = eg.run(
        rules,
        root,
        max_iterations=max_iterations,
        rule_budgets=rule_budgets,
    )
    t1 = time.perf_counter()
    best = eg.extract_best(root, flops_cost)
    t2 = time.perf_counter()
    derivations = count_derivations(eg, root)
    terms, terms_capped = count_distinct_terms(eg, root)
    t3 = time.perf_counter()
    return {
        "k": k,
        "enodes": stats["n_enodes"],
        "live_enodes": live_enodes(eg),
        "eclasses": stats["n_classes"],
        "sat_iters": stats["iterations"],
        "wall_ms": (t1 - t0) * 1e3,
        "extract_ms": (t2 - t1) * 1e3,
        "count_ms": (t3 - t2) * 1e3,
        "rule_fires": dict(eg.rule_fires),
        "total_fires": sum(eg.rule_fires.values()),
        "derivations_at_root": derivations,
        "terms_at_root": terms,
        "terms_capped": terms_capped,
        "best_cost": flops_cost(best) if best is not None else None,
        "budget_spent": stats.get("rule_budgets", {}),
        "budget_suspended": stats.get("budget_suspended", []),
    }


def measure_stratified(k: int, rules: list) -> dict[str, Any]:
    """The engineered path: compute coherence, never store it.

    ``canonicalize`` flattens the chain and rebuilds the balanced
    (Blelloch-style) bracketing in O(k log k); ``stratified_run`` then
    saturates with the contentful rules only — on a pure matmul chain
    nothing fires, so this isolates the canonicalisation cost."""
    term = matmul_chain(k)
    t0 = time.perf_counter()
    canonicalize(term)
    t1 = time.perf_counter()
    eg = EGraph(truncation_level=1)
    out = stratified_run(
        eg, rules, term, cost_fn=flops_cost, extract=True
    )
    t2 = time.perf_counter()
    return {
        "k": k,
        "canon_ms": (t1 - t0) * 1e3,
        "stratified_ms": (t2 - t0) * 1e3,
        "enodes": out["stats"]["n_enodes"],
        "coherent_dropped": out["coherent_dropped"],
        "balanced_cost": (
            flops_cost(out["canonical_best"])
            if out.get("canonical_best") is not None
            else None
        ),
    }


def measure_e2e(k: int, device: str) -> dict[str, Any]:
    """End-to-end ``optimize_model`` on an exported Linear chain.

    torch.export lowers ``nn.Linear(bias=False)`` to 2-ary ``linear``
    ops; ``assoc_linear`` then composes stacked weights into a matmul
    product whose bracketing closure is the same Catalan family.  Runs
    under the production bounded-saturation budget
    (``symmetry_budget=2048``) — the same pipeline a user gets."""
    import torch
    from catopt_optimize.optimize import optimize_model

    dims = chain_dims(k)
    model = torch.nn.Sequential(
        *[
            torch.nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(k)
        ]
    ).to(device)
    x = torch.randn(4, dims[0], device=device)
    t0 = time.perf_counter()
    opt_mod, stats = optimize_model(model, x, verbose=False)
    wall = time.perf_counter() - t0
    with torch.no_grad():
        rel = (
            (model(x) - opt_mod(x)).abs().max()
            / model(x).abs().max().clamp_min(1e-12)
        ).item()
    return {
        "k": k,
        "e2e_ms": wall * 1e3,
        "enodes": stats.get("n_enodes"),
        "sat_iters": stats.get("iterations"),
        "total_fires": sum(stats.get("rule_fires", {}).values()),
        "verified_rel_diff": rel,
    }


# ---------------------------------------------------------------------------
#  benchkit adapters — Case construction, one-shot timing, plots
# ---------------------------------------------------------------------------


def _sat_aux(row: dict[str, Any], policy: str) -> dict[str, Any]:
    """Flatten a :func:`measure_run` row into the Case ``aux`` channel."""
    k = row["k"]
    cat = catalan(k - 1)
    return {
        "policy": policy,
        "space": cat,
        "space_terms": f"Catalan({k - 1})",
        "space_log10": log10n(cat),
        "live_theory": (k**3 - k) // 6 + k,
        "eclasses_theory": k * (k + 1) // 2,
        "enodes": row["enodes"],
        "live": row["live_enodes"],
        "eclasses": row["eclasses"],
        "iters": row["sat_iters"],
        "fires": row["total_fires"],
        "rule_fires": row["rule_fires"],
        "wall_ms": round(row["wall_ms"], 3),
        "extract_ms": round(row["extract_ms"], 3),
        "count_ms": round(row["count_ms"], 3),
        "encode_dp_count": row["terms_at_root"],
        "terms_capped": row["terms_capped"],
        "derivations": row["derivations_at_root"],
        "best_cost": row["best_cost"],
        "budget_spent": row["budget_spent"],
        "budget_suspended": row["budget_suspended"],
    }


def _strat_aux(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "canon_ms": round(row["canon_ms"], 3),
        "stratified_ms": round(row["stratified_ms"], 3),
        "enodes": row["enodes"],
        "coherent_dropped": row["coherent_dropped"],
        "balanced_cost": row["balanced_cost"],
    }


def _e2e_aux(row: dict[str, Any]) -> dict[str, Any]:
    if "error" in row:
        return {"error": row["error"]}
    return {
        "e2e_ms": round(row["e2e_ms"], 3),
        "enodes": row["enodes"],
        "iters": row["sat_iters"],
        "fires": row["total_fires"],
        "verified_rel_diff": row["verified_rel_diff"],
    }


def _oneshot_cell(
    runner: Runner,
    name: str,
    params: dict[str, Any],
    aux: dict[str, Any],
    wall_ms: dict[str, float],
) -> Cell:
    """Package an already-taken one-shot measurement as a ``Cell``.

    Saturation/export are single stateful wall-clock runs, so the
    variants carry marker statements (``_marker``) just to keep the
    case structurally uniform; ``run_case`` times them in ~µs and each
    real measured wall time is then written into ``medians`` — a
    one-shot measurement IS its own median, with IQR 0.  Variants with
    no measurement (e.g. a failed e2e) are dropped so reports render
    them as missing rather than as a spurious ~0 ms.
    """
    variants = [
        Variant(v, stmt=_marker, note=_ONESHOT_NOTE) for v in wall_ms
    ]
    case = Case(
        name=name, params=dict(params), variants=variants, aux=aux
    )
    cell = runner.run_case(case)
    for v, ms in wall_ms.items():
        if ms is None:
            cell.medians.pop(v, None)
            cell.iqr.pop(v, None)
        else:
            cell.medians[v] = ms / 1e3
            cell.iqr[v] = 0.0
    cell.aux.update(aux)
    return cell


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _cells_named(report: Report, prefix: str) -> list[Cell]:
    out = [
        c
        for c in report.cells
        if c.case.name.startswith(prefix) and "k" in c.case.params
    ]
    out.sort(key=lambda c: c.case.params["k"])
    return out


def plot_space_vs_enodes(report: Report, outdir: Path) -> Path | None:
    """The money plot: exponential program space vs polynomial e-graph.

    log-y curve of Catalan(k-1) — the size of the equivalent-program
    space the e-graph explores — against the live e-node count the
    saturated (exact, then bounded-fragment) graph actually stores,
    plus the (k^3-k)/6+k / k(k+1)/2 closed forms it provably attains.
    """
    exact = _cells_named(report, "exact_k")
    bounded = _cells_named(report, "bounded_k")
    if not exact and not bounded:
        return None
    ks = sorted(
        {
            c.case.params["k"]
            for c in exact + bounded
            if c.aux.get("space")
        }
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(
        ks,
        [catalan(k - 1) for k in ks],
        "k-",
        lw=1.6,
        label="program space  Catalan(k-1)",
    )
    ax.semilogy(
        ks,
        [(k**3 - k) // 6 + k for k in ks],
        ":",
        color="grey",
        label="live e-nodes theory  (k^3-k)/6+k",
    )
    ax.semilogy(
        ks,
        [k * (k + 1) // 2 for k in ks],
        "-.",
        color="grey",
        label="e-classes theory  k(k+1)/2",
    )
    if exact:
        ax.semilogy(
            [c.case.params["k"] for c in exact],
            [c.aux["live"] for c in exact],
            "o",
            color="tab:blue",
            label="exact saturation: live e-nodes",
        )
    if bounded:
        ax.semilogy(
            [c.case.params["k"] for c in bounded],
            [max(c.aux.get("live") or 1, 1) for c in bounded],
            "s",
            color="tab:orange",
            ms=4,
            label="bounded saturation: live e-nodes",
        )
    ax.set_xlabel("chain depth k (matmuls)")
    ax.set_ylabel("count (log)")
    ax.set_title(
        "search_efficiency: equivalent-program space vs e-graph size"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "search_efficiency_space_vs_enodes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_wall_ms(report: Report, outdir: Path) -> Path | None:
    """Saturation/pipeline wall time vs k (log-y; spans ms → seconds)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    drew = False
    for prefix, key, marker, label in (
        ("exact_k", "wall_ms", "o", "exact saturation"),
        ("bounded_k", "wall_ms", "s", "bounded saturation"),
        ("strat_k", "stratified_ms", "^", "canonicalize+stratified"),
        ("e2e_k", "e2e_ms", "v", "optimize_model e2e"),
    ):
        cells = [
            c
            for c in _cells_named(report, prefix)
            if c.aux.get(key) is not None
        ]
        if not cells:
            continue
        drew = True
        ax.semilogy(
            [c.case.params["k"] for c in cells],
            [c.aux[key] for c in cells],
            marker,
            label=label,
            ms=4,
        )
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("chain depth k (matmuls)")
    ax.set_ylabel("wall time, ms (log)")
    ax.set_title("search_efficiency: saturation wall time vs k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "search_efficiency_wall_ms.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def write_artifacts(report: Report, out: str | Path) -> list[Path]:
    """Emit the benchkit report (JSON + MD + bar plots) plus the two
    suite-specific PNGs under ``out``."""
    out = Path(out)
    plots_dir = out / "plots"
    paths: list[Path] = []

    j = out / "search_efficiency.json"
    report.to_json(j)
    paths.append(j)
    m = out / "search_efficiency.md"
    report.to_markdown(m)
    paths.append(m)

    try:
        paths += [Path(p) for p in report.to_plots(plots_dir)]
    except Exception as e:  # matplotlib optional — never sink the JSON
        print(f"[warn] benchkit plots skipped: {e}")
    try:
        for p in (
            plot_space_vs_enodes(report, plots_dir),
            plot_wall_ms(report, plots_dir),
        ):
            if p is not None:
                paths.append(p)
    except Exception as e:
        print(f"[warn] suite plots skipped: {e}")
    return paths


# ---------------------------------------------------------------------------
#  Console reporting (unchanged semantics)
# ---------------------------------------------------------------------------


def fmt_ms(ms: float) -> str:
    return f"{ms:10.1f}"


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    hdr = (
        f"{'k':>4} | {'space log10':>11} | {'terms log10':>11} | "
        f"{'enodes':>7} | {'live':>7} | {'ecls':>6} | {'iters':>5} | "
        f"{'wall_ms':>10} | {'extr_ms':>10} | {'fires':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        terms_col = (
            ">" if r["terms_capped"] else " "
        ) + f"{log10n(r['terms_at_root']):>10.2f}"
        print(
            f"{r['k']:>4} | "
            f"{log10n(catalan(r['k'] - 1)):>11.2f} | "
            f"{terms_col} | "
            f"{r['enodes']:>7} | "
            f"{r['live_enodes']:>7} | "
            f"{r['eclasses']:>6} | "
            f"{r['sat_iters']:>5} | "
            f"{fmt_ms(r['wall_ms'])} | "
            f"{fmt_ms(r['extract_ms'])} | "
            f"{r['total_fires']:>7}"
        )
    print(
        "       (space = Catalan(k-1) bracketings; terms = distinct "
        "programs the root e-class encodes, '>' = lower bound)"
    )
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Search-efficiency benchmark: equivalent-program space "
            "(Catalan(k-1) bracketings of a k-matmul chain) vs "
            "e-graph size and saturation wall-time."
        )
    )
    ap.add_argument(
        "--depths",
        default="4,8,12,16,20,24",
        help="comma-separated chain depths (matmul count k)",
    )
    ap.add_argument(
        "--device", default="cpu", help="torch device for the e2e path"
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results"),
        metavar="DIR",
        help="artifact directory for the benchkit report "
        "(search_efficiency.json/.md + plots/; default bench/results)",
    )
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="console output only — write no JSON/Markdown/PNG artifacts",
    )
    ap.add_argument(
        "--exact-max",
        type=int,
        default=11,
        help="largest k for unbounded exact saturation (default 11; "
        "k=12 takes ~3 min in this pure-Python engine)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="per-rule enode budget for bounded saturation "
        "(matches optimize_model's symmetry_budget default)",
    )
    ap.add_argument(
        "--assoc-only",
        action="store_true",
        help="run only [assoc_matmul, assoc_matmul_rev] instead of the "
        "full production ruleset (identical closure — nothing else "
        "matches a pure matmul chain — but the first whole-graph "
        "scan of the other ~50 rules is skipped)",
    )
    return ap.parse_args(argv)


def run_bench(args: argparse.Namespace) -> Report:
    """Run the sweep and return the populated benchkit ``Report``.

    Every measured cell becomes a ``Case`` — sweep coords in
    ``params``, all statistics in ``aux`` — run through ``Runner`` so
    the report's cells carry real one-shot wall medians (see
    :func:`_oneshot_cell`).
    """
    depths = sorted(
        {
            int(d)
            for d in (
                getattr(args, "depths", None) or "4,8,12,16,20,24"
            ).split(",")
            if d.strip()
        }
    )
    if not depths:
        raise SystemExit("--depths must contain at least one integer")
    rules = (
        list(_ASSOC_RULES)
        if getattr(args, "assoc_only", False)
        else all_rules()
    )
    budgets = {
        r.name: (getattr(args, "budget", None) or 2048)
        for r in _ASSOC_RULES
    }

    # Warm torch.export/dynamo caches once so the first e2e row isn't
    # dominated by one-time compile bookkeeping; also probes whether the
    # requested device exists.
    device = getattr(args, "device", None) or "cpu"
    if device != "cpu":
        try:
            import torch

            if (
                device.startswith("cuda")
                and not torch.cuda.is_available()
            ):
                print(f"[warn] {device} unavailable; e2e runs on cpu")
                device = "cpu"
        except ImportError:
            pass
    with contextlib.suppress(Exception):
        measure_e2e(2, device)

    runner = Runner(
        device=(getattr(args, "device", None) or "cpu"),
        warmup=_RUNNER_WARMUP,
        min_run_time=_RUNNER_MIN_RUN_TIME,
    )
    cells: list[Cell] = []
    exact_rows, bounded_rows, strat_rows, e2e_rows = [], [], [], []

    # -- exact saturation sweep: every k in [4, exact_max] -------------
    # The dense sweep matters here: it exposes the polynomial O(k^3)
    # e-node growth against the exponential Catalan(k-1) space at the
    # exact fixed point, verified by the term-count DP.
    exact_depths = [
        d
        for d in range(4, (getattr(args, "exact_max", None) or 11) + 1)
    ]
    for k in exact_depths:
        er = measure_run(k, rules, None)
        exact_rows.append(er)
        cells.append(
            _oneshot_cell(
                runner,
                name=f"exact_k{k}",
                params={"k": k},
                aux=_sat_aux(er, "exact"),
                wall_ms={"sat": er["wall_ms"]},
            )
        )
        live_theory = (k**3 - k) // 6 + k
        cls_theory = k * (k + 1) // 2
        cat = catalan(k - 1)
        assert er["live_enodes"] == live_theory, (
            f"k={k}: live enodes {er['live_enodes']} != "
            f"(k^3-k)/6+k = {live_theory}"
        )
        assert er["eclasses"] == cls_theory, (
            f"k={k}: eclasses {er['eclasses']} != k(k+1)/2 = {cls_theory}"
        )
        assert er["terms_at_root"] == cat and not er["terms_capped"], (
            f"k={k}: root encodes {er['terms_at_root']} terms != "
            f"Catalan({k - 1}) = {cat}"
        )
        assert er["derivations_at_root"] == cat, (
            f"k={k}: derivation count {er['derivations_at_root']} != "
            "Catalan — canonical closure expected injective derivations"
        )
        assert er["enodes"] <= k**6, (
            f"k={k}: hash-cons table {er['enodes']} exceeds "
            f"polynomial bound k^6"
        )
        print(
            f"k={k:>3} exact:   {er['enodes']} enodes "
            f"({er['live_enodes']} live == (k^3-k)/6+k, "
            f"{er['eclasses']} classes == k(k+1)/2) "
            f"in {er['wall_ms']:.0f} ms — encodes ALL "
            f"{cat:.3e} bracketings",
            flush=True,
        )
    skipped_exact = [
        d
        for d in depths
        if d > (getattr(args, "exact_max", None) or 11)
    ]
    if skipped_exact:
        print(
            f"exact saturation skipped at k={skipped_exact} "
            f"(--exact-max {(getattr(args, 'exact_max', None) or 11)}; k=12 already needs the "
            "generic matcher to enumerate ~4^12 substitutions, ~3 min)"
        )
    print()

    for k in depths:
        # -- bounded saturation: the production policy, at every k ------
        br = measure_run(k, rules, budgets)
        bounded_rows.append(br)
        cells.append(
            _oneshot_cell(
                runner,
                name=f"bounded_k{k}",
                params={
                    "k": k,
                    "budget": (getattr(args, "budget", None) or 2048),
                },
                aux=_sat_aux(br, "bounded"),
                wall_ms={"sat": br["wall_ms"]},
            )
        )
        assert (
            br["enodes"]
            <= k**5 + 4 * (getattr(args, "budget", None) or 2048) + 512
        ), (
            f"k={k}: bounded e-graph grew past its polynomial cap: "
            f"{br['enodes']}"
        )
        geq = ">" if br["terms_capped"] else "~"
        print(
            f"k={k:>3} bounded: {br['enodes']} enodes "
            f"({br['live_enodes']} live, {br['eclasses']} classes) "
            f"in {br['wall_ms']:.0f} ms — encodes "
            f"{geq}10^{log10n(br['terms_at_root']):.1f} distinct "
            f"programs of ~10^{log10n(catalan(k - 1)):.1f} possible",
            flush=True,
        )

        # -- engineered fast path --------------------------------------
        sr = measure_stratified(k, rules)
        strat_rows.append(sr)
        cells.append(
            _oneshot_cell(
                runner,
                name=f"strat_k{k}",
                params={"k": k},
                aux=_strat_aux(sr),
                wall_ms={
                    "canon": sr["canon_ms"],
                    "strat": sr["stratified_ms"],
                },
            )
        )

        # -- end-to-end pipeline ---------------------------------------
        try:
            e2e_rows.append(
                measure_e2e(k, (getattr(args, "device", None) or "cpu"))
            )
        except Exception as e:  # torch/export issues must not hide the
            # core measurement — report and continue.
            e2e_rows.append(
                {"k": k, "error": f"{type(e).__name__}: {e}"}
            )
            print(f"k={k:>3} e2e:     FAILED ({e})", flush=True)
        er2 = e2e_rows[-1]
        cells.append(
            _oneshot_cell(
                runner,
                name=f"e2e_k{k}",
                params={"k": k},
                aux=_e2e_aux(er2),
                wall_ms={"e2e": er2.get("e2e_ms")},
            )
        )

    print()
    print_table(
        "EXACT saturation (unbounded; stores the ENTIRE "
        "Catalan(k-1) closure):",
        exact_rows,
    )
    print_table(
        f"BOUNDED saturation (rule_budgets={(getattr(args, 'budget', None) or 2048)} per symmetry "
        "rule — the production policy):",
        bounded_rows,
    )

    # -- stratified / e2e summary --------------------------------------
    print("ENGINEERED PATH (meta.canonicalize + contentful-only run):")
    for sr in strat_rows:
        print(
            f"  k={sr['k']:>3}: canonicalize {sr['canon_ms']:7.2f} ms, "
            f"stratified total {sr['stratified_ms']:7.2f} ms, "
            f"{sr['enodes']} enodes, dropped coherent rules: "
            f"{sr['coherent_dropped']}"
        )
    print()
    print("END-TO-END (optimize_model on exported nn.Linear chain):")
    for r in e2e_rows:
        if "error" in r:
            print(f"  k={r['k']:>3}: error {r['error']}")
            continue
        print(
            f"  k={r['k']:>3}: {r['e2e_ms']:8.0f} ms total "
            f"(export+saturation+extract+lower+verify), "
            f"{r['enodes']} enodes, {r['sat_iters']} iters, "
            f"{r['total_fires']} rule fires, "
            f"verify rel diff {r['verified_rel_diff']:.2e}"
        )
    print()

    # -- the honest comparison ------------------------------------------
    kmax = max(depths)
    bmax = max(bounded_rows, key=lambda r: r["k"])
    print("HONEST COMPARISON")
    print(
        f"  * equivalent-program space at k={kmax}: "
        f"Catalan({kmax - 1}) ~ 10^{log10n(catalan(kmax - 1)):.1f} "
        f"programs (a comm+assoc add-chain would be "
        f"k!*Catalan ~ 10^{log10n(math.factorial(kmax) * catalan(kmax - 1)):.1f})."
    )
    print(
        f"  * saturated e-graph: {bmax['live_enodes']} live e-nodes, "
        f"{bmax['eclasses']} classes — O(k^3) and O(k^2); the interval "
        "graph stores every bracketing simultaneously via sharing."
    )
    bound = "at least" if bmax["terms_capped"] else ""
    print(
        f"  * bounded-saturation fragment at k={kmax} still encodes "
        f"{bound} ~10^{log10n(bmax['terms_at_root']):.1f} distinct "
        f"programs in {bmax['wall_ms']:.0f} ms; extraction prices the "
        f"whole class in {bmax['extract_ms']:.1f} ms — O(live enodes), "
        "the same work as a matrix-chain DP over the interval chart."
    )
    print(
        "  * a syntax-directed local-pattern optimizer (Inductor-style) "
        "commits to ONE rewrite at each of O(k) match sites per pass — "
        "it cannot hold the space; certifying the optimum by "
        "enumeration would require visiting Omega(Catalan(k-1)) "
        "concrete programs.  catopt's canonicalize/stratified_run go "
        "further: they compute the coherent normal form directly in "
        "O(k log k), never storing the symmetry closure at all."
    )
    best_exact = max(exact_rows, key=lambda r: r["k"], default=None)
    if best_exact is not None:
        print(
            f"  * verified: at k={best_exact['k']} the exact e-graph "
            f"encodes ALL Catalan({best_exact['k'] - 1}) = "
            f"{catalan(best_exact['k'] - 1):,} programs in "
            f"{best_exact['live_enodes']} live e-nodes "
            f"(== (k^3-k)/6 + k exactly)."
        )
    summary_bound = (
        f">10^{log10n(bmax['terms_at_root']):.0f}"
        if bmax["terms_capped"]
        else f"~10^{log10n(bmax['terms_at_root']):.0f}"
    )
    print(
        f"  SUMMARY: e-graph explored {summary_bound} "
        f"equivalent programs in {bmax['wall_ms']:.0f} ms "
        f"(bounded fragment of a ~10^{log10n(catalan(kmax - 1)):.0f} "
        "program space)."
    )

    return Report(
        suite="search_efficiency",
        cells=cells,
        env=collect_env(getattr(args, "device", None) or "cpu"),
    )


def main() -> None:
    args = parse_args()
    report = run_bench(args)
    if not getattr(args, "no_artifacts", False):
        paths = write_artifacts(
            report, (getattr(args, "out", None) or "bench/results")
        )
        print(
            f"\nartifacts written to {(getattr(args, 'out', None) or 'bench/results')}:"
        )
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
