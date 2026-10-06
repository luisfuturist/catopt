"""Law coherence, depth 2 — chains of derivations and mediating rules.

``catopt_discovery.coherence`` enumerated the *pair* layer: ``A ⇒ B``
direct derivations, derivable-vs-primitive verdicts, and the one-step
confluence probe.  Every relation it records is between two laws.
The open question this spike measures: **is there useful structure at
depth > 1, and does enumeration still suffice?**

Three probes, all on the same instance-level, budgeted machinery
(``catopt_discovery.verifier.verify_law`` — a fresh e-graph per question, so an
oracle call is one bounded saturation plus a replayable certificate):

1. **Derivation chains / stratification.**  The pair catalogue's
   "derivable" verdict means *the rest of the library* proves the law
   — but the proof may itself fire a *derivable* law.  Stratify: a
   derivable law has rank 1 when the *primitives alone* merge its
   instance sides, rank k when primitives plus rank-(<k) derivables
   do.  Laws that never stratify form self-supporting cycles (the
   inverse-pair classes: each twin's only proof fires the other).
   The measured **effective basis** is then ``#primitives +
   #unstratified-SCCs`` — one seed per cyclic class — checked by
   grounding: seed those SCCs and verify every remaining law derives.
   This answers "is the basis really 29, or do some derivations run
   through derivable intermediates (a 2-step chain ``A⇒D⇒C``)?"
2. **Reach under removal.**  Derivability is a saturated/bounded
   *equality* verdict; the pipeline's question is reach within a
   search budget.  For each law ``L``, saturate a small corpus of
   real model exports under ``ALL - {L}`` and compare e-node/e-class
   counts against the full library.  A *derivable* law whose removal
   shrinks reach is derivable-in-theory but load-bearing in practice
   (its substitute derivation costs more iterations than the budget
   buys); a primitive law whose removal changes nothing is
   inert on this corpus.
3. **The mediator table — the first triples.**  For every co-firing
   pair that fails to rejoin under ``{A, B}`` alone (the catalogue's
   ``lib-mediated`` / ``divergent`` rows), enumerate *which* third
   rules rejoin it: every ``C`` with ``{A, B, C}`` sufficient, which
   of them are *essential* (``universe - {C}`` fails to rejoin), and
   a minimal mediating subset.  "The coherence of ``A x B`` requires
   ``C``" is itself a derivable 3-cell — this is the catalogue beyond
   pairs.

And the honest cost accounting the question demands: every phase's
oracle calls and wall-time are counted, then projected onto the naive
all-triples enumeration (``C(71,3) ≈ 57.2 k`` triples) to show where
enumeration actually strains — and how much of the triple space the
pair results already prune away.

Run::

    .venv/bin/python -m catopt_discovery.coherence2            # ~1-2 min
    .venv/bin/python -m catopt_discovery.coherence2 --with-layout
    .venv/bin/python -m catopt_discovery.coherence2 --skip-reach --json o.json

CPU-only.  Torch is imported lazily and only for the corpus-reach
probe (model export); ``--skip-reach`` keeps the run torch-free.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.egraph import EGraph, Rewrite
from catopt_core.laws import (
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
)
from catopt_core.laws import tags as _tags
from catopt_core.meta import apply_rewrite_at

from catopt_discovery import coherence as lc
from catopt_discovery import verifier as lv

__all__ = ["MediatorRow", "main"]

#: Saturation budgets — the stratification/grounding checks reuse the
#: generous whole-library bound; the mediator probes reuse the
#: confluence bound (the reducts are one-step neighbours of a law
#: instance, no larger than the pair probe's terms).
_STRAT_ITERS = lv._MAX_ITERATIONS
_STRAT_NODES = lv._MAX_NODES
_MED_ITERS = lc._CONF_ITERS
_MED_NODES = lc._CONF_NODES

#: Corpus-reach budgets — smaller than law_impact's (this probe runs
#: ~380 saturations, not ~60).
_REACH_ITERS = 5
_REACH_NODES = 40_000
_EXPANSIVE_BUDGET = 600

#: The reach corpus — small real exports chosen to cover the
#: interesting law neighbourhoods: silu/swiglu, matmul chains,
#: linear folds, softmax/select, residual adds.
_CORPUS_MODELS = (
    "SwiGLU",
    "ResidualMLP",
    "GatedResidualBlock",
    "ManualSoftmaxAttention",
    "MatrixChain",
    "ParallelLinear",
    "NormLinear",
)


# ---------------------------------------------------------------------------
#  Oracle accounting — every verify_law call is timed under a phase label
# ---------------------------------------------------------------------------

_PHASE = {"name": "setup"}
_CALLS: dict[str, list[float]] = {}
_EGRAPH_RUNS = {"n": 0}

_ORIG_VERIFY = lv.verify_law


def _set_phase(name: str) -> None:
    """Label subsequent oracle calls with *name* for the cost table."""
    _PHASE["name"] = name


def _counted_verify(
    lhs: Any,
    rhs: Any,
    rules: Any,
    *,
    max_iterations: int = 30,
    max_nodes: int = 200000,
    rule_budgets: dict[str, int] | None = None,
) -> lv.LawResult:
    """Wrap ``lv.verify_law``; record one call + wall-time per phase."""
    t0 = time.perf_counter()
    try:
        return _ORIG_VERIFY(
            lhs,
            rhs,
            rules,
            max_iterations=max_iterations,
            max_nodes=max_nodes,
            rule_budgets=rule_budgets,
        )
    finally:
        rec = _CALLS.setdefault(_PHASE["name"], [0, 0.0])
        rec[0] += 1
        rec[1] += time.perf_counter() - t0


def _install_counter() -> None:
    """Route ``lv.verify_law`` through the accounting wrapper.

    The pair catalogue's calls resolve ``lv.verify_law`` at call
    time, so patching the module attribute counts them too.  The
    ``Any`` hop because a function value is not assignable to another
    function's module attribute under the checker (and ``setattr``
    would be rewritten back by the linter).
    """
    counted: Any = _counted_verify
    lv.verify_law = counted


def _cost_table() -> str:
    """Render the per-phase oracle-cost table plus the projection."""
    lines = [
        "Enumeration cost — oracle calls and wall-time per phase",
        "-" * 68,
        f"  {'phase':<18} {'calls':>7} {'seconds':>9} {'ms/call':>8}",
    ]
    total_calls = 0
    total_sec = 0.0
    for name, (calls, sec) in _CALLS.items():
        per = 1000.0 * sec / calls if calls else 0.0
        lines.append(f"  {name:<18} {calls:>7} {sec:>9.2f} {per:>8.1f}")
        total_calls += calls
        total_sec += sec
    lines.append(f"  {'TOTAL':<18} {total_calls:>7} {total_sec:>9.2f}")
    lines.append(
        f"  direct EGraph saturations (reach probe): "
        f"{_EGRAPH_RUNS['n']}"
    )
    # Projection: naive all-triples enumeration costs >= one oracle
    # call per triple even before mediators are sought inside it.
    n = len(ALL_RULES)
    triples = n * (n - 1) * (n - 2) // 6
    med_ms = 1000.0 * total_sec / total_calls if total_calls else 0.0
    est = triples * med_ms / 1000.0
    lines.append(
        f"  projection — naive C(n,3) triple enumeration: "
        f"{triples} triples x {med_ms:.1f} ms/call "
        f"= {est:.0f} s ({est / 60:.1f} min) at the observed "
        f"per-call cost, *before* per-triple mediator search"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Probe 1 — derivation graph + stratification ranks
# ---------------------------------------------------------------------------


def _premises(p: lc.LawProfile) -> set[str]:
    """Return every recorded premise of *p*.

    The union of witness, essential and direct-edge sources — a
    superset of any single derivation's needs.
    """
    return set(p.witness) | set(p.essential) | set(p.direct_from)


def _sccs(
    nodes: set[str], edges: dict[str, set[str]]
) -> list[list[str]]:
    """Tarjan SCCs of *edges* restricted to *nodes* (iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = [0]

    def visit(v0: str) -> None:
        work = [(v0, iter(sorted(edges.get(v0, set()) & nodes)))]
        index[v0] = low[v0] = counter[0]
        counter[0] += 1
        stack.append(v0)
        on_stack.add(v0)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append(
                        (w, iter(sorted(edges.get(w, set()) & nodes)))
                    )
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                out.append(sorted(scc))

    for v in sorted(nodes):
        if v not in index:
            visit(v)
    return out


def _stratify(
    rules: list[Rewrite],
    inst: dict[str, tuple[Any, Any]],
    profiles: dict[str, lc.LawProfile],
) -> dict[str, Any]:
    """Rank each derivable law by the least premise set that proves it.

    Rank 1 = provable from the *primitive* rules alone; rank k adds
    the rank-(<k) derivables to the premise set.  Laws that never
    stratify are cyclic — every recorded derivation fires another
    unstratified law (inverse-pair twins are the expected shape).
    """
    by_name = {r.name: r for r in rules}
    prim = [
        p.name
        for p in profiles.values()
        if p.verdict == "primitive" and p.name in inst
    ]
    remaining = {
        p.name for p in profiles.values() if p.verdict == "derivable"
    } & set(inst)
    rank: dict[str, int] = {}
    depth = 0
    while remaining:
        depth += 1
        s = [by_name[n] for n in prim] + [
            by_name[n] for n in sorted(rank)
        ]
        got = []
        for n in sorted(remaining):
            res = lv.verify_law(
                *inst[n],
                s,
                max_iterations=_STRAT_ITERS,
                max_nodes=_STRAT_NODES,
            )
            if res.derivable:
                got.append(n)
        if not got:
            break
        for n in got:
            rank[n] = depth
            remaining.discard(n)

    edges = {n: _premises(profiles[n]) for n in set(inst)}
    cyclic_sccs = [
        s
        for s in _sccs(set(remaining), edges)
        if len(s) > 1 or s[0] in (edges.get(s[0], set()) & remaining)
    ]
    return {
        "rank": rank,
        "remaining": sorted(remaining),
        "cyclic_sccs": cyclic_sccs,
        "n_primitive": len(prim),
        "prim_names": prim,
    }


def _grounding_check(
    rules: list[Rewrite],
    inst: dict[str, tuple[Any, Any]],
    strat: dict[str, Any],
) -> dict[str, Any]:
    """Seed each cyclic SCC once; check every unseeded law derives.

    The measured effective basis is ``#primitives + #cyclic-SCCs``
    iff every leftover law derives from that seed set within budget;
    laws that still fail are reported as extra required seeds.
    """
    by_name = {r.name: r for r in rules}
    seeds = set(strat["prim_names"])
    reps = set()
    for scc in strat["cyclic_sccs"]:
        reps.add(scc[0])
        seeds.add(scc[0])
    seeded_cycles = sorted(reps)
    uncovered: list[str] = []
    for n in sorted(strat["remaining"]):
        if n in seeds:
            continue
        res = lv.verify_law(
            *inst[n],
            [by_name[m] for m in sorted(seeds)],
            max_iterations=_STRAT_ITERS,
            max_nodes=_STRAT_NODES,
        )
        if not res.derivable:
            uncovered.append(n)
    return {
        "seeds": sorted(seeds),
        "seeded_cycles": seeded_cycles,
        "uncovered": uncovered,
        "effective_basis": len(seeds) + len(uncovered),
    }


def _derivation_graph_stats(
    profiles: dict[str, lc.LawProfile],
    strat: dict[str, Any],
) -> str:
    """Render the derivation-graph summary: edges, SCCs, ranks."""
    edges = {n: _premises(p) for n, p in profiles.items()}
    verdict = {n: p.verdict for n, p in profiles.items()}
    n_edges = sum(len(v) for v in edges.values())
    gen2 = sum(
        1
        for n, pre in edges.items()
        if verdict[n] == "derivable"
        for m in pre
        if verdict.get(m) == "derivable"
    )
    hist: dict[int, int] = {}
    for r in strat["rank"].values():
        hist[r] = hist.get(r, 0) + 1
    lines = [
        "Derivation graph — premise edges (witness U essential "
        "U direct)",
        "-" * 68,
        f"  premise edges: {n_edges} | through a DERIVABLE premise "
        f"(2nd-generation): {gen2}",
        "  stratification ranks (least premise set that proves "
        "the law):",
    ]
    for r in sorted(hist):
        names = sorted(n for n, k in strat["rank"].items() if k == r)
        lines.append(
            f"    rank {r}: {hist[r]} laws — {', '.join(names)}"
        )
    rem = strat["remaining"]
    lines.append(
        f"    unstratified (cyclic — every derivation fires an "
        f"unstratified law): {len(rem)}"
    )
    for scc in strat["cyclic_sccs"]:
        lines.append(f"      SCC {{ {', '.join(scc)} }}")
    singles = [
        n for n in rem if all(n not in s for s in strat["cyclic_sccs"])
    ]
    if singles:
        lines.append(
            "      downstream singletons (need a cyclic class, "
            "not self-cyclic):"
        )
        for n in singles:
            pre = sorted(edges.get(n, set()) & set(rem))
            lines.append(f"        {n}  <- {', '.join(pre) or '?'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Probe 2 — reach under removal on a small real corpus
# ---------------------------------------------------------------------------


def _corpus() -> list[tuple[str, Any]]:
    """Export the reach corpus; return ``(name, ir.root)`` pairs."""
    import torch  # lazy: only the reach probe needs the adapter
    from catopt_torch.adapters import TorchSource

    from catopt_discovery import impact as li

    wanted = set(_CORPUS_MODELS)
    src = TorchSource()
    out: list[tuple[str, Any]] = []
    for name, model, x in li._model_cases():
        if name not in wanted:
            continue
        torch.manual_seed(0)
        ir, _tensors = src.to_ir(model, x)
        out.append((name, ir.root))
    return out


def _reach_budgets(rules: list[Rewrite]) -> dict[str, int]:
    """Cap the EXPANSIVE closure generators (law_impact's policy)."""
    names = {r.name for r in rules}
    return {
        r.name: _EXPANSIVE_BUDGET
        for r in ALL_RULES_WITH_LAYOUT
        if r.name in names and _tags.EXPANSIVE in r.tags
    }


def _saturate_once(term: Any, rules: list[Rewrite]) -> tuple[int, int]:
    """One bounded saturation; return ``(n_enodes, n_classes)``."""
    _EGRAPH_RUNS["n"] += 1
    eg = EGraph()
    root = eg.add_term(term)
    stats = eg.run(
        rules,
        root,
        max_iterations=_REACH_ITERS,
        max_nodes=_REACH_NODES,
        rule_budgets=_reach_budgets(rules),
    )
    return stats["n_enodes"], stats["n_classes"]


def _reach_probe(
    rules: list[Rewrite],
    profiles: dict[str, lc.LawProfile],
) -> list[dict[str, Any]]:
    """Saturate the corpus under ``ALL - {L}`` for every law *L*."""
    corpus = _corpus()
    base: dict[str, tuple[int, int]] = {}
    for name, term in corpus:
        base[name] = _saturate_once(term, rules)
    rows: list[dict[str, Any]] = []
    for law in rules:
        minus = [r for r in rules if r.name != law.name]
        d_en = d_cl = affected = 0
        for name, term in corpus:
            en, cl = _saturate_once(term, minus)
            de, dc = base[name][0] - en, base[name][1] - cl
            d_en += de
            d_cl += dc
            affected += de != 0 or dc != 0
        rows.append(
            {
                "law": law.name,
                "verdict": profiles[law.name].verdict,
                "d_enodes": d_en,
                "d_classes": d_cl,
                "cases_affected": affected,
            }
        )
    return rows


def _reach_table(rows: list[dict[str, Any]]) -> str:
    """Render the removal-impact table, derivable laws flagged."""
    lines = [
        "Reach under removal — enode/class delta when L is dropped "
        "(corpus totals)",
        "-" * 68,
        f"  {'law':<34} {'verdict':<10} {'dEnodes':>8} {'dClasses':>9} "
        f"{'cases':>6}",
    ]
    shown = [r for r in rows if r["cases_affected"]]
    for r in sorted(shown, key=lambda r: -r["d_enodes"]):
        mark = " *" if r["verdict"] == "derivable" else "  "
        lines.append(
            f"{mark}{r['law']:<34} {r['verdict']:<10} "
            f"{r['d_enodes']:>8} {r['d_classes']:>9} "
            f"{r['cases_affected']:>6}"
        )
    inert = [r["law"] for r in rows if not r["cases_affected"]]
    n_der = sum(1 for r in shown if r["verdict"] == "derivable")
    lines.append(
        f"  (* = derivable law whose removal changed bounded reach: "
        f"{n_der})"
    )
    lines.append(
        f"  inert on this corpus (zero delta): {len(inert)} laws"
    )
    if inert:
        lines.append("    " + ", ".join(sorted(inert)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Probe 3 — the mediator table (triples that matter)
# ---------------------------------------------------------------------------


@dataclass
class MediatorRow:
    """One non-pair-confluent co-firing pair and its mediators."""

    pair: tuple[str, str]
    base: str
    status: str  # "lib-mediated" | "divergent"
    singles: tuple[str, ...] = ()  # C with {A,B,C} sufficient
    essential: tuple[str, ...] = ()  # universe - {C} fails to rejoin
    minimal: tuple[str, ...] = ()  # a minimal sufficient subset
    witness: tuple[str, ...] = ()  # the found derivation's rules
    note: str = ""


def _pair_reducts(
    a: Rewrite,
    b: Rewrite,
    inst: dict[str, tuple[Any, Any]],
) -> tuple[str, Any, Any] | None:
    """Recompute the confluence probe's two one-step reducts.

    Mirrors ``catopt_discovery.coherence._confluence_probe``: host on *b*'s LHS
    instance when *a* co-fires, else on *a*'s.  Returns
    ``(base_name, second_reduct, first_reduct)`` or ``None`` when the
    pair shares no firing instance.
    """
    for first, second in ((b, a), (a, b)):
        got = inst.get(first.name)
        if got is None:
            continue
        fired = lc._first_fire(second, got[0])
        if fired is None:
            continue
        _path, t_s = fired
        one = apply_rewrite_at(first, got[0], ())
        if one is None:
            continue
        return first.name, t_s, one
    return None


def _mediator_probe(
    a: Rewrite,
    b: Rewrite,
    inst: dict[str, tuple[Any, Any]],
    universe: list[Rewrite],
) -> MediatorRow | None:
    """Enumerate which third rules rejoin the *a* x *b* reducts.

    ``singles`` is the full census — every library rule that suffices
    alongside the pair.  ``essential`` is the load-bearing subset
    (removal breaks the library's join).  ``minimal`` is one minimal
    sufficient subset alongside ``{A, B}``, greedily reduced from the
    found witness.
    """
    rd = _pair_reducts(a, b, inst)
    if rd is None:
        return None
    base, t_s, one = rd
    pair = lv.verify_law(
        t_s,
        one,
        [a, b],
        max_iterations=_MED_ITERS,
        max_nodes=_MED_NODES,
    )
    if pair.derivable:
        return None  # pair-confluent: no triple needed
    whole = lv.verify_law(
        t_s,
        one,
        universe,
        max_iterations=_MED_ITERS,
        max_nodes=_MED_NODES,
    )
    by_name = {r.name: r for r in universe}
    if not whole.derivable:
        return MediatorRow(
            pair=(a.name, b.name),
            base=base,
            status="divergent",
            note=f"no rejoin under the universe (stop={whole.stop})",
        )
    witness = tuple(whole.witness_rules)
    # Full census: every single rule that mediates alongside {A, B}.
    singles: list[str] = []
    for c in universe:
        if c.name in (a.name, b.name):
            continue
        res = lv.verify_law(
            t_s,
            one,
            [a, b, c],
            max_iterations=_MED_ITERS,
            max_nodes=_MED_NODES,
        )
        if res.derivable:
            singles.append(c.name)
    # Essential: removing C breaks the whole-universe join.  Pair
    # members are candidates too — a rejoin can refire A or B (the
    # square_expand x square_to_pow join needs square_expand again).
    essential: list[str] = []
    cands = set(witness) | set(singles) | {a.name, b.name}
    for cn in sorted(cands):
        minus = [r for r in universe if r.name != cn]
        res = lv.verify_law(
            t_s,
            one,
            minus,
            max_iterations=_MED_ITERS,
            max_nodes=_MED_NODES,
        )
        if not res.derivable:
            essential.append(cn)
    # One minimal sufficient subset beside {A, B} — greedy from the
    # census when singles exist, else from the found witness minus
    # the pair (a pair needing a *combination* of mediators reports
    # that combination here — the first beyond-triple rows).
    cand = singles or [w for w in witness if w not in (a.name, b.name)]
    cur = list(cand)
    if cur:
        res = lv.verify_law(
            t_s,
            one,
            [a, b] + [by_name[m] for m in cur],
            max_iterations=_MED_ITERS,
            max_nodes=_MED_NODES,
        )
        if not res.derivable:
            cur = []  # found witness set itself fails — report empty
    for cn in list(cur):
        trial = [m for m in cur if m != cn]
        res = lv.verify_law(
            t_s,
            one,
            [a, b] + [by_name[m] for m in trial],
            max_iterations=_MED_ITERS,
            max_nodes=_MED_NODES,
        )
        if res.derivable:
            cur = trial
    return MediatorRow(
        pair=(a.name, b.name),
        base=base,
        status="lib-mediated",
        singles=tuple(sorted(singles)),
        essential=tuple(sorted(essential)),
        minimal=tuple(sorted(cur)),
        witness=witness,
        note=whole.note
        or ("" if whole.replayable else "witness not standalone"),
    )


def _mediator_table(
    rules: list[Rewrite],
    inst: dict[str, tuple[Any, Any]],
    conf: list[lc.ConfRow],
) -> list[MediatorRow]:
    """Probe every co-firing pair the catalogue marked non-confluent.

    The pair catalogue already found *which* pairs need a third rule;
    re-running only those rows keeps the triple census linear in the
    number of interesting pairs, not cubic in the library.
    """
    by_name = {r.name: r for r in rules}
    rows: list[MediatorRow] = []
    for row in conf:
        if row.join_pair:
            continue
        a, b = by_name[row.pair[0]], by_name[row.pair[1]]
        got = _mediator_probe(a, b, inst, rules)
        if got is not None:
            rows.append(got)
    return rows


def _mediator_section(rows: list[MediatorRow]) -> str:
    """Render the mediator table — the first 3-cell census."""
    lines = [
        "Mediator table — which third rules rejoin the non-pair-"
        "confluent pairs",
        "-" * 68,
        f"  {'pair':<44} {'status':<13} singles | essential",
    ]
    for r in rows:
        name = f"{r.pair[0]} x {r.pair[1]}"
        lines.append(f"  {name:<44} {r.status:<13}")
        lines.append(
            f"      singles: {', '.join(r.singles) or '(none)'}"
        )
        lines.append(
            f"      essential: {', '.join(r.essential) or '(none)'}"
            f"   minimal+pair: {{{', '.join(r.minimal)}}}"
        )
        lines.append(
            f"      witness: {', '.join(r.witness) or '(none)'}"
            f"   (on {r.base}'s instance) {r.note}"
        )
    if not rows:
        lines.append("  (no non-pair-confluent pairs)")
    hubs: dict[str, int] = {}
    for r in rows:
        for m in r.singles:
            hubs[m] = hubs.get(m, 0) + 1
    if hubs:
        top = sorted(hubs.items(), key=lambda kv: -kv[1])[:8]
        lines.append(
            "  mediator hubs (rules mediating the most pairs): "
            + ", ".join(f"{n} ({k})" for n, k in top)
        )
    multi = [
        r for r in rows if r.status == "lib-mediated" and not r.singles
    ]
    if multi:
        lines.append(
            "  pairs with NO single-rule mediator (need a mediator "
            "combination — beyond triples):"
        )
        for r in multi:
            lines.append(
                f"    {r.pair[0]} x {r.pair[1]}: minimal "
                f"{{{', '.join(r.minimal)}}} via witness "
                f"({', '.join(r.witness)})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Sanity — the depth-2 data must reproduce the known results
# ---------------------------------------------------------------------------


def _sanity(
    cat: dict[str, Any],
    med: list[MediatorRow],
    rules: list[Rewrite],
) -> str:
    """Check the new probes against the catalogue's known landmarks."""
    lines = [
        "Sanity — known results reproduced by the depth-2 probes",
        "-" * 68,
    ]
    matmul = {
        "distribute_matmul_over_add",
        "factor_matmul",
        "weight_distribute_matmul",
        "weight_factor_matmul",
    }
    ok1 = any(set(g) == matmul for g in cat["eq_classes"])
    lines.append(
        f"  [{'PASS' if ok1 else 'FAIL'}] 4-member matmul "
        f"equivalence class present"
    )
    by_pair = {r.pair: r for r in med}
    for pair, want in (
        (("silu_expand", "swiglu_fuse"), "silu_fold"),
        (("silu_mul_form", "swiglu_fuse"), "silu_fold"),
    ):
        row = by_pair.get(pair) or by_pair.get((pair[1], pair[0]))
        ok = row is not None and want in row.singles
        lines.append(
            f"  [{'PASS' if ok else 'FAIL'}] silu_fold mediates "
            f"{pair[0]} x {pair[1]}"
        )
    names = {r.name for r in rules}
    if "linear_to_matmul_t" in names:
        n_med = sum(1 for r in med if "linear_to_matmul_t" in r.singles)
        lines.append(
            f"  [{'PASS' if n_med else 'FAIL'}] linear_to_matmul_t "
            f"mediates {n_med} pairs (catalogue reported 15 "
            f"coherences)"
        )
    else:
        lines.append(
            "  [SKIP] linear_to_matmul_t (layout universe not "
            "active — rerun with --with-layout)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Driver + reporting
# ---------------------------------------------------------------------------


def _jsonable(
    strat: dict[str, Any],
    ground: dict[str, Any],
    med: list[MediatorRow],
    reach: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project the depth-2 results to a JSON-serializable dict."""
    return {
        "stratification": {
            "rank": strat["rank"],
            "remaining": strat["remaining"],
            "cyclic_sccs": strat["cyclic_sccs"],
            "n_primitive": strat["n_primitive"],
        },
        "grounding": ground,
        "mediators": [
            {
                "pair": list(r.pair),
                "base": r.base,
                "status": r.status,
                "singles": list(r.singles),
                "essential": list(r.essential),
                "minimal": list(r.minimal),
                "witness": list(r.witness),
                "note": r.note,
            }
            for r in med
        ],
        "reach": reach,
        "cost": {
            name: {"calls": c, "seconds": s}
            for name, (c, s) in _CALLS.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: pair catalogue + the three depth-2 probes."""
    ap = argparse.ArgumentParser(
        description="Depth-2 law coherence: derivation chains, "
        "reach under removal, and the mediator table."
    )
    ap.add_argument(
        "--with-layout",
        action="store_true",
        help="include the 77 opt-in LAYOUT_RULES (131-rule universe)",
    )
    ap.add_argument(
        "--skip-reach",
        action="store_true",
        help="skip the corpus reach-under-removal probe (torch-free)",
    )
    ap.add_argument(
        "--json", metavar="PATH", help="write the raw results as JSON"
    )
    args = ap.parse_args(argv)

    _install_counter()
    rules = list(
        ALL_RULES_WITH_LAYOUT if args.with_layout else ALL_RULES
    )
    by_name = {r.name: r for r in rules}

    _set_phase("catalogue")
    t0 = time.perf_counter()
    cat = lc.catalogue(rules)
    cat_seconds = time.perf_counter() - t0
    profiles: dict[str, lc.LawProfile] = cat["profiles"]
    inst = {n: lc._instance(by_name[n]) for n in cat["instanced"]}
    inst = {n: i for n, i in inst.items() if i is not None}

    print("=" * 68)  # stdout-compat
    print(  # stdout-compat
        "LAW COHERENCE — DEPTH 2 (chains, reach, mediators)"
    )
    print("=" * 68)  # stdout-compat
    print(  # stdout-compat
        f"universe: {len(rules)} rules | instanced: {len(inst)} | "
        f"pair catalogue: {cat_seconds:.1f} s"
    )
    print()  # stdout-compat

    # -- probe 1: derivation graph + stratification -------------------
    _set_phase("stratification")
    strat = _stratify(rules, inst, profiles)
    _set_phase("grounding")
    ground = _grounding_check(rules, inst, strat)
    print(_derivation_graph_stats(profiles, strat))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        "Effective basis — seeds needed to derive the library"
    )
    print("-" * 68)  # stdout-compat
    print(  # stdout-compat
        f"  primitives: {strat['n_primitive']} | cyclic SCCs "
        f"seeded: {len(ground['seeded_cycles'])} "
        f"({', '.join(ground['seeded_cycles']) or 'none'})"
    )
    print(  # stdout-compat
        f"  unstratified laws still underivable under the seeds: "
        f"{len(ground['uncovered'])}"
        + (
            f" — {', '.join(ground['uncovered'])}"
            if ground["uncovered"]
            else ""
        )
    )
    print(  # stdout-compat
        f"  measured effective basis: {ground['effective_basis']}"
    )
    print()  # stdout-compat

    # -- probe 2: reach under removal ---------------------------------
    reach: list[dict[str, Any]] = []
    if not args.skip_reach:
        _set_phase("reach")
        reach = _reach_probe(rules, profiles)
        print(_reach_table(reach))  # stdout-compat
        print()  # stdout-compat
    else:
        print("(reach probe skipped — --skip-reach)")  # stdout-compat
        print()  # stdout-compat

    # -- probe 3: mediator table --------------------------------------
    _set_phase("mediators")
    med = _mediator_table(rules, inst, cat["confluence"])
    print(_mediator_section(med))  # stdout-compat
    print()  # stdout-compat

    print(_sanity(cat, med, rules))  # stdout-compat
    print()  # stdout-compat
    print(_cost_table())  # stdout-compat

    if args.json:
        Path(args.json).write_text(
            json.dumps(_jsonable(strat, ground, med, reach), indent=1)
            + "\n"
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
