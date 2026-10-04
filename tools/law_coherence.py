"""Law coherence catalogue — the 3-cell layer, made explicit.

A rewrite law is a 2-cell ``lhs ~[2]~> rhs``; a relation *between*
two laws is a 3-cell (ADR 0002).  The pipeline has always detected
these implicitly — ``tools/law_pipeline.py``'s ``_relation`` flags
candidate proposals as ``duplicate`` / ``inverse`` of shipped rules
by alpha-normal key, and ``tools/law_verifier.py`` asks whether one
law is derivable from *the rest of the library*.  What has never
existed is the pairwise catalogue: for every ordered pair ``(A, B)``
of shipped laws, does ``A`` (possibly among the other rules) derive
``B``?  Do they commute?  Are they the same 2-cell twice?

This tool builds that catalogue over ``catopt_core.laws.ALL_RULES``
(54 rules; ``--with-layout`` adds the 77 opt-in ``LAYOUT_RULES``).
Every verdict is measured by the shipped machinery — ``EGraph``
saturation on a per-law concrete instance — never asserted:

* **direct** ``A ⇒ B`` — the single rule ``A`` alone merges ``B``'s
  two instance sides (``verify_law(inst_B, [A])``).  The strongest
  3-cell: ``B``'s equality is a one-rule consequence of ``A``.
  Inverse and duplicate rules always appear here, since the e-graph
  seeds both sides; the interesting edges are the *composite* ones.
* **derivable** — the library minus ``B`` proves ``B``
  (``law_verifier.rediscover``), with the replayable witness rules
  recorded.  A derivable law adds no basis element.
* **essential** — ``B`` is derivable, but removing *this* rule
  breaks the derivation (``verify_law(inst_B, ALL - {A, B})``
  fails).  ``A`` is a load-bearing premise of ``B``, not merely a
  participant.  Candidates are restricted to the found witness plus
  the direct-edge sources — an essential rule must occur in *every*
  derivation, hence in the one already found.
* **confluent** — on a term where both fire, apply ``A``-once and
  ``B``-once and ask whether the two one-step reducts rejoin under
  ``{A, B}`` (a bounded local-confluence / critical-pair probe).
  ``overlap`` marks whether ``A`` fires inside ``B``'s pattern
  skeleton (a genuine critical pair) or inside a bound metavariable
  (a parallel move).  Pairs with no common firing instance are
  reported ``no-cofire``, not guessed.
* **equivalent** — same alpha-normal ``(lhs, rhs)`` (duplicate) or
  mutually direct-derivable both ways: the same equality carried by
  two rule objects — "the same law two ways".

Honest limits, printed with the results: derivability is *on the
law's concrete instance* under bounded saturation, so "primitive"
means "no derivation found within budget", never "independent in
theory".  Confluence is instance-level, not a full critical-pair
analysis.  Both bounds are stated in the report.

Run::

    .venv/bin/python tools/law_coherence.py
    .venv/bin/python tools/law_coherence.py --with-layout
    .venv/bin/python tools/law_coherence.py --json /tmp/coh.json

CPU-only, ~30 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import TensorType, Var
from catopt_core.laws import ALL_RULES, ALL_RULES_WITH_LAYOUT
from catopt_core.meta import (
    _positions,
    apply_rewrite_at,
    instantiate_pattern,
    pattern_metavars,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import law_verifier as lv

__all__ = [
    "ConfRow",
    "LawProfile",
    "catalogue",
    "main",
]

#: Saturation budget for the pairwise direct-derivation probes.
_PAIR_ITERS = 8
_PAIR_NODES = 20_000

#: Saturation budget for the whole-library derivability checks —
#: the same generous bound ``law_verifier`` uses (law sides are tiny).
_DERIVE_ITERS = lv._MAX_ITERATIONS
_DERIVE_NODES = lv._MAX_NODES

#: Saturation budget for the local-confluence join under ``{A, B}``.
_CONF_ITERS = 10
_CONF_NODES = 30_000

#: Leaf shape for the generic layout-rule instance fallback.
_GENERIC_SHAPE = (4, 4)


# ---------------------------------------------------------------------------
#  Instances — every probe runs on a concrete, well-typed pair of terms
# ---------------------------------------------------------------------------


def _generic_instance(rule: Rewrite) -> tuple[Any, Any] | None:
    """Build an instance from *rule*'s own LHS pattern, generically.

    Fallback for the rules ``law_verifier.instance_of`` cannot cover
    (the ``LAYOUT_RULES`` family): each term metavariable becomes a
    ``(4, 4)`` ``Var`` — rank ≥ 2 so every ``transpose`` check hook
    sees a real axis pair — and each attr metavariable ``*0`` / ``*1``
    becomes ``0`` / ``1`` (distinct axes, and the last-two swap that
    the ``.mT`` guards require on rank 2).  The rule's own ``check``
    and ``derive`` hooks run on the substitution exactly as firing
    would; a veto or an unbound RHS metavar yields ``None``, never a
    guessed term.
    """
    subst: dict[str, Any] = {}
    for mv in sorted(pattern_metavars(rule.lhs)):
        if mv.startswith("$attr:"):
            n = mv[len("$attr:") :]
            if n.endswith("0"):
                subst[mv] = 0
            elif n.endswith("1"):
                subst[mv] = 1
            else:
                subst[mv] = 0
        else:
            subst[mv] = Var(mv, TensorType(_GENERIC_SHAPE))
    if rule.check is not None:
        try:
            if not rule.check(subst):
                return None
        except Exception:
            return None
    try:
        lhs = instantiate_pattern(rule.lhs, subst)
    except KeyError:
        return None
    inst = dict(subst)
    if rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            return None
        if extra is None:
            return None
        inst.update(extra)
    try:
        return lhs, instantiate_pattern(rule.rhs, inst)
    except KeyError:
        return None


def _instance(rule: Rewrite) -> tuple[Any, Any] | None:
    """Return a concrete ``(lhs, rhs)`` instance pair for *rule*.

    Prefers the bench-registry / SDPA-generic instances of
    ``law_verifier.instance_of``; falls back to
    :func:`_generic_instance` for the layout family.
    """
    return lv.instance_of(rule) or _generic_instance(rule)


# ---------------------------------------------------------------------------
#  Per-law profile and the measured relations
# ---------------------------------------------------------------------------


@dataclass
class LawProfile:
    """Everything measured about one law in the catalogue."""

    name: str
    instanced: bool = False
    verdict: str = "no-instance"  # primitive | derivable | no-instance
    witness: tuple[str, ...] = ()
    witness_steps: int = 0
    essential: tuple[str, ...] = ()
    direct_from: tuple[str, ...] = ()  # A with {A} ⇒ this law
    direct_to: tuple[str, ...] = ()  # laws this law directly derives


@dataclass(frozen=True)
class ConfRow:
    """One co-firing probe: do the two one-step reducts rejoin?."""

    pair: tuple[str, str]
    base: str  # which law's instance hosted the probe
    join_pair: bool  # rejoins under {A, B}
    join_lib: bool  # rejoins under the whole universe
    overlap: bool  # A fires inside B's pattern skeleton
    note: str = ""


def _in_skeleton(lhs: Any, path: tuple) -> bool:
    """Return whether *path* lands inside *lhs*'s pattern skeleton.

    Walks the pattern along *path*; descending through a metavariable
    (a ``str`` leaf) means the fire happened inside a *bound subterm*
    — a parallel move, not a critical overlap.  Reaching the end of
    the path while still on pattern ``Op`` structure is a genuine
    skeleton overlap.
    """
    node = lhs
    for i in path:
        if not hasattr(node, "args") or isinstance(node, str):
            return False
        if i >= len(node.args):
            return False
        node = node.args[i]
    return not isinstance(node, str)


def _first_fire(rule: Rewrite, term: Any) -> tuple[tuple, Any] | None:
    """Return ``(path, reduct)`` of *rule*'s first real firing on *term*."""
    for path, _sub in _positions(term):
        out = apply_rewrite_at(rule, term, path)
        if out is not None and out != term:
            return path, out
    return None


def _confluence_probe(
    a: Rewrite,
    b: Rewrite,
    inst: dict[str, tuple[Any, Any]],
    universe: list[Rewrite],
) -> ConfRow | None:
    """Probe local confluence of *a* and *b* on a shared instance.

    Hosts the probe on *b*'s LHS instance when *a* co-fires there,
    else on *a*'s LHS instance when *b* co-fires.  Returns ``None``
    when no shared firing instance exists (the honest ``no-cofire``).
    """
    for first, second in ((b, a), (a, b)):
        got = inst.get(first.name)
        if got is None:
            continue
        base_term = got[0]
        fired = _first_fire(second, base_term)
        if fired is None:
            continue
        path_s, t_s = fired
        one = apply_rewrite_at(first, base_term, ())
        if one is None:
            continue
        pair = lv.verify_law(
            t_s,
            one,
            [a, b],
            max_iterations=_CONF_ITERS,
            max_nodes=_CONF_NODES,
        )
        row = ConfRow(
            pair=(a.name, b.name),
            base=first.name,
            join_pair=pair.derivable,
            join_lib=False,
            overlap=_in_skeleton(first.lhs, path_s),
        )
        if not pair.derivable:
            # Library-mediated confluence: does the WHOLE universe —
            # the pair included — know how to rejoin the reducts?
            whole = lv.verify_law(
                t_s,
                one,
                universe,
                max_iterations=_CONF_ITERS,
                max_nodes=_CONF_NODES,
            )
            row = ConfRow(
                pair=row.pair,
                base=row.base,
                join_pair=False,
                join_lib=whole.derivable,
                overlap=row.overlap,
                note=""
                if whole.derivable
                else f"reducts did not rejoin (stop={whole.stop})",
            )
        return row
    return None


# ---------------------------------------------------------------------------
#  The catalogue
# ---------------------------------------------------------------------------


def catalogue(
    rules: list[Rewrite],
) -> dict[str, Any]:
    """Measure every pairwise relation over *rules*; return the data."""
    names = [r.name for r in rules]
    inst: dict[str, tuple[Any, Any]] = {}
    for r in rules:
        got = _instance(r)
        if got is not None:
            inst[r.name] = got

    structural = lv.structural_report(rules)

    profiles: dict[str, LawProfile] = {
        n: LawProfile(name=n, instanced=n in inst) for n in names
    }

    # -- derivability from the rest of the library --------------------
    others = {n: [r for r in rules if r.name != n] for n in names}
    for r in rules:
        got = inst.get(r.name)
        if got is None:
            continue
        res = lv.verify_law(
            *got,
            others[r.name],
            max_iterations=_DERIVE_ITERS,
            max_nodes=_DERIVE_NODES,
        )
        p = profiles[r.name]
        p.verdict = "derivable" if res.derivable else "primitive"
        p.witness = res.witness_rules
        p.witness_steps = res.witness_steps

    # -- direct pairwise edges {A} ⇒ B ---------------------------------
    direct_to: dict[str, set[str]] = {n: set() for n in names}
    direct_from: dict[str, set[str]] = {n: set() for n in names}
    for a in rules:
        for b in rules:
            if a.name == b.name or b.name not in inst:
                continue
            res = lv.verify_law(
                *inst[b.name],
                [a],
                max_iterations=_PAIR_ITERS,
                max_nodes=_PAIR_NODES,
            )
            if res.derivable:
                direct_to[a.name].add(b.name)
                direct_from[b.name].add(a.name)
    for n in names:
        profiles[n].direct_from = tuple(sorted(direct_from[n]))
        profiles[n].direct_to = tuple(sorted(direct_to[n]))

    # -- essential premises: removing A must break B's derivation -----
    # An essential rule occurs in every derivation of B, hence in the
    # one already found — candidates are the witness rules plus the
    # direct-edge sources.
    for b in rules:
        if profiles[b.name].verdict != "derivable":
            continue
        cands = set(profiles[b.name].witness) | direct_from[b.name]
        essential: list[str] = []
        for cn in sorted(cands):
            minus = [r for r in rules if r.name not in (b.name, cn)]
            res = lv.verify_law(
                *inst[b.name],
                minus,
                max_iterations=_DERIVE_ITERS,
                max_nodes=_DERIVE_NODES,
            )
            if not res.derivable:
                essential.append(cn)
        profiles[b.name].essential = tuple(essential)

    # -- equivalence classes: duplicates + mutual direct edges --------
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for group in structural["duplicates"]:
        for n in group[1:]:
            union(group[0], n)
    for a in rules:
        for bn in direct_to[a.name]:
            if a.name in direct_to[bn]:
                union(a.name, bn)
    classes: dict[str, list[str]] = {}
    for n in names:
        classes.setdefault(find(n), []).append(n)
    eq_classes = sorted(
        (sorted(v) for v in classes.values() if len(v) > 1),
        key=lambda g: (g[0], len(g)),
    )

    # -- confluence on shared firing instances ------------------------
    conf: list[ConfRow] = []
    for i, a in enumerate(rules):
        for b in rules[i + 1 :]:
            if a.name not in inst and b.name not in inst:
                continue
            row = _confluence_probe(a, b, inst, rules)
            if row is not None:
                conf.append(row)

    return {
        "rules": names,
        "instanced": sorted(inst),
        "structural": structural,
        "profiles": profiles,
        "eq_classes": eq_classes,
        "confluence": conf,
    }


# ---------------------------------------------------------------------------
#  Cluster skeleton — one hand-picked neighbourhood, relations drawn in
# ---------------------------------------------------------------------------

#: The select_mul / softmax_fold / transpose_push_mul neighbourhood:
#: the folds and the layout/pointwise naturality laws they touch.
_CLUSTER = (
    "select_mul",
    "softmax_fold",
    "sdpa_fold_add",
    "sdpa_fold_addmul",
    "comm_mul",
    "comm_add",
    "transpose_push_mul",
    "transpose_pull_mul",
    "transpose_push_add",
    "transpose_pull_add",
    "transpose_push_exp",
    "transpose_transpose_dd",
    "transpose_matmul",
)


def _cluster_section() -> str:
    """Print the pairwise relations inside the hand-picked cluster.

    Always runs against ``ALL_RULES_WITH_LAYOUT``: the cluster cut is
    fixed (it names layout laws), independent of the main universe.
    """
    universe = list(ALL_RULES_WITH_LAYOUT)
    by_name = {r.name: r for r in universe}
    members = [by_name[n] for n in _CLUSTER if n in by_name]
    lines = [
        "Cluster skeleton — select_mul / softmax_fold / "
        "transpose_push_mul neighbourhood",
        "-" * 68,
    ]
    if len(members) != len(_CLUSTER):
        missing = sorted(set(_CLUSTER) - {r.name for r in members})
        lines.append(
            f"  (absent from this universe: {', '.join(missing)})"
        )
    inst = {r.name: _instance(r) for r in members}
    inst = {n: i for n, i in inst.items() if i is not None}
    edges: list[str] = []
    for a in members:
        for b in members:
            if a.name == b.name or b.name not in inst:
                continue
            res = lv.verify_law(
                *inst[b.name],
                [a],
                max_iterations=_PAIR_ITERS,
                max_nodes=_PAIR_NODES,
            )
            if res.derivable:
                rel = lv.classify_relation(b, a)
                edges.append(f"  {a.name} ⇒ {b.name}   [{rel}]")
    lines.append("direct derivations inside the cluster:")
    lines.extend(edges or ["  (none)"])
    lines.append("co-firing / confluence inside the cluster:")
    conf_lines: list[str] = []
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            row = _confluence_probe(a, b, inst, universe)
            if row is None:
                continue
            verdict = (
                "confluent"
                if row.join_pair
                else ("lib-mediated" if row.join_lib else "DIVERGENT")
            )
            ov = "overlap" if row.overlap else "inside-metavar"
            conf_lines.append(
                f"  {a.name} x {b.name}: {verdict}"
                f" ({ov}, on {row.base}'s instance)"
            )
    lines.extend(conf_lines or ["  (no shared firing instances)"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _report(cat: dict[str, Any], rules: list[Rewrite]) -> str:
    """Render the catalogue as a printable report."""
    profiles: dict[str, LawProfile] = cat["profiles"]
    n = len(cat["rules"])
    instanced = len(cat["instanced"])
    derivable = [
        p for p in profiles.values() if p.verdict == "derivable"
    ]
    primitive = [
        p for p in profiles.values() if p.verdict == "primitive"
    ]
    out: list[str] = []
    out.append("=" * 68)
    out.append(
        "LAW COHERENCE CATALOGUE — 3-cells over the shipped laws"
    )
    out.append("=" * 68)
    out.append(
        f"universe: {n} rules | instanced: {instanced} | "
        f"budgets: pair {_PAIR_ITERS}it/{_PAIR_NODES}n, "
        f"derive {_DERIVE_ITERS}it/{_DERIVE_NODES}n"
    )
    out.append("")

    # -- per-law table -------------------------------------------------
    out.append("Per-law verdicts")
    out.append("-" * 68)
    out.append(f"{'law':<34} {'verdict':<11} {'witness':<30} essential")
    for r in rules:
        p = profiles[r.name]
        wit = ",".join(p.witness)[:29] or "-"
        ess = ",".join(p.essential) or "-"
        out.append(f"{r.name:<34} {p.verdict:<11} {wit:<30} {ess}")
    out.append("")

    # -- equivalence classes -------------------------------------------
    out.append("Equivalence classes — the same equality, two rules")
    out.append("-" * 68)
    dups = cat["structural"]["duplicates"]
    invs = cat["structural"]["inverses"]
    for g in cat["eq_classes"]:
        kinds = []
        for pair in dups:
            if set(pair) <= set(g):
                kinds.append("duplicate")
        for pair in invs:
            if set(pair) <= set(g):
                kinds.append("inverse")
        tag = f" ({', '.join(sorted(set(kinds)))})" if kinds else ""
        out.append(f"  {{ {', '.join(g)} }}{tag}")
    if not cat["eq_classes"]:
        out.append("  (none)")
    out.append("")

    # -- direct edges ---------------------------------------------------
    out.append("Direct derivations — {A} alone proves B's instance")
    out.append("-" * 68)
    by_name = {r.name: r for r in rules}
    genuine: list[str] = []
    for p in sorted(profiles.values(), key=lambda p: p.name):
        for a in p.direct_from:
            rel = lv.classify_relation(by_name[p.name], by_name[a])
            mark = "  *" if rel == "composite" else "  "
            line = f"{mark}{a} ⇒ {p.name}   [{rel}]"
            (genuine if rel == "composite" else out).append(line)
    out.extend(sorted(genuine))
    out.append(
        "  (* = composite: a genuine derivation, not a structural echo)"
    )
    out.append("")

    # -- confluence ------------------------------------------------------
    conf: list[ConfRow] = cat["confluence"]
    out.append("Confluence — one-step reducts rejoin under {A, B}")
    out.append("-" * 68)
    n_conf = sum(r.join_pair for r in conf)
    n_lib = sum(r.join_lib and not r.join_pair for r in conf)
    n_div = sum(not r.join_pair and not r.join_lib for r in conf)
    out.append(
        f"  co-firing pairs probed: {len(conf)} | "
        f"confluent: {n_conf} | library-mediated: {n_lib} | "
        f"divergent: {n_div}"
    )
    for r in conf:
        if not r.join_pair:
            ov = "overlap" if r.overlap else "inside-metavar"
            out.append(
                f"  {r.pair[0]} x {r.pair[1]}: "
                f"{'lib-mediated' if r.join_lib else 'DIVERGENT'}"
                f" ({ov}, on {r.base}) {r.note}"
            )
    out.append("")

    # -- the basis count ---------------------------------------------------
    out.append("Basis summary")
    out.append("-" * 68)
    out.append(f"  shipped rules:            {n}")
    out.append(f"  instanced:                {instanced}")
    out.append(f"  derivable (redundant):    {len(derivable)}")
    out.append(f"  primitive (independent):  {len(primitive)}")
    out.append(f"  duplicate groups:         {len(dups)}")
    out.append(f"  inverse pairs:            {len(invs)}")
    return "\n".join(out)


def _jsonable(cat: dict[str, Any]) -> dict[str, Any]:
    """Project the catalogue to a JSON-serializable dict."""
    profiles: dict[str, LawProfile] = cat["profiles"]
    return {
        "n_rules": len(cat["rules"]),
        "instanced": cat["instanced"],
        "structural": cat["structural"],
        "profiles": {
            n: {
                "verdict": p.verdict,
                "witness": list(p.witness),
                "witness_steps": p.witness_steps,
                "essential": list(p.essential),
                "direct_from": list(p.direct_from),
                "direct_to": list(p.direct_to),
            }
            for n, p in profiles.items()
        },
        "eq_classes": cat["eq_classes"],
        "confluence": [
            {
                "pair": list(r.pair),
                "base": r.base,
                "join_pair": r.join_pair,
                "join_lib": r.join_lib,
                "overlap": r.overlap,
                "note": r.note,
            }
            for r in cat["confluence"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: build the catalogue and print the report."""
    ap = argparse.ArgumentParser(
        description="Coherence catalogue over the shipped rewrite laws."
    )
    ap.add_argument(
        "--with-layout",
        action="store_true",
        help="include the 77 opt-in LAYOUT_RULES (131-rule universe)",
    )
    ap.add_argument(
        "--json", metavar="PATH", help="write the raw catalogue as JSON"
    )
    args = ap.parse_args(argv)

    rules = list(
        ALL_RULES_WITH_LAYOUT if args.with_layout else ALL_RULES
    )
    cat = catalogue(rules)
    print(_report(cat, rules))
    print()
    print(_cluster_section())
    if args.json:
        Path(args.json).write_text(
            json.dumps(_jsonable(cat), indent=1) + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
