r"""Lemma certificates — ``Rewrite.derivation`` as replayable proof data.

The axiom/lemma split annotated every non-kernel rule with
``derivation``: the names of shipped rules forming ONE measured
derivation of the rule's instance (``catopt_discovery.coherence
--emit-basis``, verified zero-drift).  Until now that field was
*provenance metadata* — a record that a derivation exists.  This tool
materializes it as a **certificate**: saturate the lemma's concrete
instance under the recorded premises alone, reconstruct the positional
derivation through the shipped proof machinery
(:meth:`EGraph.certificate`), and verify it by replaying
(:func:`verify_certificate`, strict — no trusted stubs).

The verdict ladder is honest about what saturation proves vs what
replays:

* **linear** — the certificate replays standalone: a genuine
  step-by-step derivation ``src -> dst`` on real terms.
* **enumerated** — ``certificate()``'s connect heuristic emits only
  e-graph-dependent stubs, but :meth:`EGraph.all_proofs` finds a
  linear derivation by enumerating the positional rewrite space (the
  merge is real and a linear proof exists; the reconstruction strategy
  just can't see it — ``_edge_path`` searches root-level rewrites
  only).
* **saturation-only** — the sides merge but no linear derivation is
  found within fuel.  Under pure rule saturation every merge has
  *some* linear witness in principle (each union IS a rule firing);
  this verdict means bounded search did not surface one.
* **gap** — the recorded premises fail to merge the instance sides at
  all: the ``derivation`` annotation has drifted from the library.

Direction matters: equality is symmetric, so a lemma proven *by* its
axiom replays in whichever direction the axiom fires forward.  For an
inverse twin (``silu_fold <- silu_expand``) the certificate runs
``rhs -> lhs`` — the axiom applied to the lemma's RHS instance *is*
the proof; the lemma's own LHS->RHS direction is unreachable by
forward premise rewriting, which is exactly why the twin is shipped.

Beyond the lemma table, ``--probe NAME`` runs the same verdict ladder
on an arbitrary rule against the universe minus itself — the boundary
probe.  ``right_factor_linear`` is the instructive case: primitive
under ``ALL_RULES``, derivable under ``--with-layout``, and its
``certificate()`` is e-graph-dependent — yet ``all_proofs`` surfaces
4-step linear derivations through the NT bridge
(``linear_to_matmul_t`` twice, ``right_factor_matmul``,
``linear_from_matmul_t``).  "Derivable-but-unreplayable" refines to
"derivable, reconstruction-hard".

Serialization: certificates are data — every step encodes through
:func:`catopt_core.egraph.cert_to_data` (terms via ``term_to_data``,
bindings tagged term/attr), rules referenced by name.  ``--demo NAME``
materializes one lemma's certificate, round-trips it through JSON, and
replays the reconstructed record with ``verify_certificate`` — the
full store loop: pattern + cond + derivation + replayable proof.

Run::

    .venv/bin/python -m catopt_discovery.lemma_cert
    .venv/bin/python -m catopt_discovery.lemma_cert --demo silu_mul_form
    .venv/bin/python -m catopt_discovery.lemma_cert --with-layout \\
        --probe right_factor_linear
    .venv/bin/python -m catopt_discovery.lemma_cert --json /tmp/certs.json

CPU-only, ~5 s (instances for bench-registered laws build through
torch, lazily).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.egraph import (
    Certificate,
    EGraph,
    Rewrite,
    cert_from_data,
    cert_to_data,
    verify_certificate,
)
from catopt_core.laws import ALL_RULES, ALL_RULES_WITH_LAYOUT

from catopt_discovery import coherence as lc
from catopt_discovery import verifier as lv

__all__ = [
    "LemmaCertRow",
    "main",
    "materialize",
    "probe",
    "serialize_demo",
]

#: Saturation budget — the same generous bound the derivability
#: checks use (law instances are tiny; every measured run stops at a
#: fixed point).
_DERIVE_ITERS = lv._MAX_ITERATIONS
_DERIVE_NODES = lv._MAX_NODES

#: Enumeration budget for the ``all_proofs`` fallback — reached only
#: when ``certificate()`` emits e-graph-dependent stubs.  Law-level
#: derivations are short; 8 paths / 10 steps / 200 k fuel is generous.
_PROOF_PATHS = 8
_PROOF_STEPS = 10
_PROOF_FUEL = 200_000


# ---------------------------------------------------------------------------
#  Materialization — run the recorded derivation, keep the certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LemmaCertRow:
    """One rule's certificate verdict plus the derivation summary."""

    name: str
    kind: str
    derivation: tuple[str, ...]
    verdict: str
    direction: str = ""
    n_steps: int = 0
    rules_used: tuple[str, ...] = ()
    premise_cover: bool = True
    n_enodes: int = 0
    note: str = ""


def _saturate(
    lhs: Any, rhs: Any, rules: list[Rewrite]
) -> tuple[EGraph, int, int, dict]:
    """Intern both sides in a fresh e-graph and saturate under *rules*."""
    eg = EGraph()
    r_lhs = eg.add_term(lhs)
    r_rhs = eg.add_term(rhs)
    stats = eg.run(
        rules,
        r_lhs,
        max_iterations=_DERIVE_ITERS,
        max_nodes=_DERIVE_NODES,
    )
    return eg, r_lhs, r_rhs, stats


def _replayable_cert(
    eg: EGraph,
    src: Any,
    dst: Any,
    eid: int,
) -> Certificate | None:
    """Return the replayable certificate ``src -> dst``, or None.

    Strict replay: a certificate carrying e-graph-dependent stubs is
    not a derivation, it is a merge witness — returned as ``None`` so
    the caller falls through to enumeration.
    """
    cert = eg.certificate(src, dst, root_eid=eid)
    if not cert.replayable:
        return None
    verify_certificate(src, cert, strict=True)
    return cert


def _enumerated_cert(
    eg: EGraph,
    src: Any,
    dst: Any,
    eid: int,
    by_name: dict[str, Rewrite],
) -> Certificate | None:
    """Find a linear derivation via ``all_proofs``; wrap as a certificate.

    Enumeration lives in the positional term-rewriting space, so every
    returned path is standalone-replayable by construction — the step
    objects already carry rule, path, and fired bindings.  The shortest
    path is verified strictly before being returned.
    """
    paths = eg.all_proofs(
        src,
        dst,
        root_eid=eid,
        max_paths=_PROOF_PATHS,
        max_steps=_PROOF_STEPS,
        fuel=_PROOF_FUEL,
    )
    for path in sorted(paths, key=len):
        cert = Certificate(
            src=src,
            dst=dst,
            root_eid=eid,
            steps=list(path),
            rules={s.rule: by_name[s.rule] for s in path},
            stats={"enumerated": True, "n_candidates": len(paths)},
        )
        try:
            verify_certificate(src, cert, strict=True)
        except Exception:
            continue
        return cert
    return None


def _run_cert(
    lhs: Any,
    rhs: Any,
    rules: list[Rewrite],
    by_name: dict[str, Rewrite],
) -> tuple[str, str, Certificate | None, dict]:
    """Saturate ``lhs``/``rhs`` under *rules*; run the verdict ladder.

    Returns ``(verdict, direction, certificate_or_None, stats)`` —
    ``linear`` when ``certificate()`` replays, ``enumerated`` when only
    ``all_proofs`` surfaces a derivation, ``saturation-only`` when the
    sides merge without a found linear proof, ``gap`` when they never
    merge.
    """
    eg, r_lhs, r_rhs, stats = _saturate(lhs, rhs, rules)
    if eg.find(r_lhs) != eg.find(r_rhs):
        return "gap", "", None, stats
    stub_note = ""
    for src, dst, eid, tag in (
        (lhs, rhs, r_lhs, "lhs->rhs"),
        (rhs, lhs, r_rhs, "rhs->lhs"),
    ):
        cert = _replayable_cert(eg, src, dst, eid)
        if cert is not None:
            return "linear", tag, cert, stats
        raw = eg.certificate(src, dst, root_eid=eid)
        stub_note = "; ".join(
            s.note for s in raw.steps if s.egraph_dependent
        )
    for src, dst, eid, tag in (
        (lhs, rhs, r_lhs, "lhs->rhs"),
        (rhs, lhs, r_rhs, "rhs->lhs"),
    ):
        cert = _enumerated_cert(eg, src, dst, eid, by_name)
        if cert is not None:
            return "enumerated", tag, cert, stats
    stats["stub_note"] = stub_note
    return "saturation-only", "", None, stats


def materialize(
    rule: Rewrite,
    universe: list[Rewrite],
    *,
    inst: tuple[Any, Any] | None = None,
) -> tuple[LemmaCertRow, Certificate | None]:
    """Materialize *rule*'s recorded ``derivation`` as a certificate.

    Saturates the rule's concrete instance under the *named premises
    only* — the annotation's actual claim is that those rules alone
    prove the law.  Returns the verdict row and the certificate (None
    for non-linear verdicts).
    """
    inst = lc._instance(rule) if inst is None else inst
    base: dict[str, Any] = {
        "name": rule.name,
        "kind": rule.kind,
        "derivation": rule.derivation,
    }
    if inst is None:
        return LemmaCertRow(verdict="no-instance", **base), None
    by_name = {r.name: r for r in universe}
    missing = [p for p in rule.derivation if p not in by_name]
    if missing:
        return (
            LemmaCertRow(
                verdict="bad-derivation",
                note=f"premise(s) not in universe: {', '.join(missing)}",
                **base,
            ),
            None,
        )
    premises = [by_name[p] for p in rule.derivation]
    lhs, rhs = inst
    verdict, direction, cert, stats = _run_cert(
        lhs, rhs, premises, by_name
    )
    used = tuple(cert.rules_used) if cert is not None else ()
    row = LemmaCertRow(
        verdict=verdict,
        direction=direction,
        n_steps=cert.n_steps if cert is not None else 0,
        rules_used=used,
        premise_cover=set(used) <= set(rule.derivation),
        n_enodes=stats["n_enodes"],
        note=stats.get("stub_note", ""),
        **base,
    )
    return row, cert


def probe(
    name: str,
    universe: list[Rewrite],
    *,
    inst: tuple[Any, Any] | None = None,
) -> tuple[LemmaCertRow, Certificate | None]:
    """Run the verdict ladder on *name* against ``universe - {name}``.

    The boundary probe: not restricted to annotated lemmas.  A rule
    derivable under the universe but carrying no ``derivation`` is
    either annotation drift or — like ``right_factor_linear`` under
    ``--with-layout`` — a derivation the annotation universe never
    claimed.
    """
    by_name = {r.name: r for r in universe}
    rule = by_name[name]
    others = [r for r in universe if r.name != name]
    inst = lc._instance(rule) if inst is None else inst
    base: dict[str, Any] = {
        "name": rule.name,
        "kind": rule.kind,
        "derivation": rule.derivation,
    }
    if inst is None:
        return LemmaCertRow(verdict="no-instance", **base), None
    lhs, rhs = inst
    verdict, direction, cert, stats = _run_cert(
        lhs, rhs, others, by_name
    )
    used = tuple(cert.rules_used) if cert is not None else ()
    row = LemmaCertRow(
        verdict=verdict,
        direction=direction,
        n_steps=cert.n_steps if cert is not None else 0,
        rules_used=used,
        premise_cover=set(used) <= set(rule.derivation or used),
        n_enodes=stats["n_enodes"],
        note=stats.get("stub_note", ""),
        **base,
    )
    return row, cert


def materialize_all(
    universe: list[Rewrite],
) -> tuple[list[LemmaCertRow], dict[str, Certificate]]:
    """Materialize every derivation-carrying rule's certificate."""
    rows: list[LemmaCertRow] = []
    certs: dict[str, Certificate] = {}
    for rule in universe:
        if not rule.derivation:
            continue
        row, cert = materialize(rule, universe)
        rows.append(row)
        if cert is not None:
            certs[rule.name] = cert
    return rows, certs


# ---------------------------------------------------------------------------
#  Serialization demo — the full store loop on one lemma
# ---------------------------------------------------------------------------


def serialize_demo(
    name: str, universe: list[Rewrite]
) -> tuple[dict[str, Any], Any]:
    """Materialize *name*'s cert, round-trip JSON, replay from data.

    Returns ``(record, replayed_term)`` — the record is the exact
    ``json.dumps``-safe dict that was reloaded, and the replayed term
    is ``verify_certificate``'s output on the *reconstructed*
    certificate (strict).  Raises ``ValueError`` when the rule has no
    linear/enumerated certificate to serialize.
    """
    by_name = {r.name: r for r in universe}
    row, cert = materialize(by_name[name], universe)
    if cert is None:
        raise ValueError(
            f"{name}: verdict {row.verdict} — no certificate "
            f"to serialize ({row.note or 'see table'})"
        )
    record = json.loads(json.dumps(cert_to_data(cert)))
    rebuilt = cert_from_data(record, by_name)
    replayed = verify_certificate(rebuilt.src, rebuilt, strict=True)
    return record, replayed


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _report(rows: list[LemmaCertRow], universe_name: str) -> str:
    """Render the lemma-certificate table plus the boundary summary."""
    out: list[str] = []
    out.append("=" * 68)
    out.append(
        "LEMMA CERTIFICATES — derivation as replayable proof data"
    )
    out.append("=" * 68)
    n_lemma = sum(r.kind == "lemma" for r in rows)
    n_red = sum(r.kind == "redundant" for r in rows)
    out.append(
        f"universe: {universe_name} | derivation-carrying: "
        f"{len(rows)} ({n_lemma} lemma, {n_red} redundant) | "
        f"budgets: {_DERIVE_ITERS}it/{_DERIVE_NODES}n, "
        f"enum {_PROOF_PATHS}p/{_PROOF_STEPS}s/{_PROOF_FUEL}f"
    )
    out.append("")
    out.append(
        f"  {'law':<30} {'kind':<10} {'verdict':<16} {'dir':<9} "
        f"{'steps':>5} cover"
    )
    out.append("  " + "-" * 78)
    for r in rows:
        cover = (
            "premise"
            if r.premise_cover
            else f"EXTRA: {sorted(set(r.rules_used) - set(r.derivation))}"
        )
        out.append(
            f"  {r.name:<30} {r.kind:<10} {r.verdict:<16} "
            f"{r.direction:<9} {r.n_steps:>5} {cover}"
        )
    out.append("")
    tally: dict[str, int] = {}
    for r in rows:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    out.append(
        "  verdicts: "
        + " | ".join(f"{k}: {v}" for k, v in sorted(tally.items()))
    )
    for r in rows:
        if r.note and r.verdict != "linear":
            out.append(f"    {r.name}: {r.note}")
    out.append("")
    out.append(
        "  direction key: the certificate replays in the direction the "
    )
    out.append(
        "  premise axiom fires forward — inverse twins prove rhs->lhs"
    )
    out.append(
        "  (the axiom applied to the lemma's RHS IS the proof; the"
    )
    out.append(
        "  lemma direction is unreachable by forward premise rewriting,"
    )
    out.append("  which is why the twin is shipped).")
    return "\n".join(out)


def _jsonable(
    rows: list[LemmaCertRow], certs: dict[str, Certificate]
) -> dict[str, Any]:
    """Project rows + materialized certificates to a JSON-safe dict."""
    return {
        "lemmas": [
            {
                "name": r.name,
                "kind": r.kind,
                "derivation": list(r.derivation),
                "verdict": r.verdict,
                "direction": r.direction,
                "n_steps": r.n_steps,
                "rules_used": list(r.rules_used),
                "premise_cover": r.premise_cover,
                "note": r.note,
            }
            for r in rows
        ],
        "certificates": {n: cert_to_data(c) for n, c in certs.items()},
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: materialize every lemma's derivation certificate."""
    ap = argparse.ArgumentParser(
        description="Materialize Rewrite.derivation as replayable "
        "certificates over the shipped laws."
    )
    ap.add_argument(
        "--with-layout",
        action="store_true",
        help="use ALL_RULES_WITH_LAYOUT (131-rule universe)",
    )
    ap.add_argument(
        "--json",
        metavar="PATH",
        help="write rows + certificates as JSON",
    )
    ap.add_argument(
        "--demo",
        metavar="RULE",
        help="serialize one rule's certificate to JSON and replay "
        "from data (default: silu_mul_form)",
        nargs="?",
        const="silu_mul_form",
    )
    ap.add_argument(
        "--probe",
        metavar="RULE",
        help="run the verdict ladder on RULE against the universe "
        "minus itself — the derivable-but-unannotated boundary probe",
    )
    args = ap.parse_args(argv)

    universe_name = (
        "ALL_RULES"
        if not args.with_layout
        else ("ALL_RULES_WITH_LAYOUT")
    )
    universe = list(
        ALL_RULES_WITH_LAYOUT if args.with_layout else ALL_RULES
    )
    rows, certs = materialize_all(universe)
    print(_report(rows, universe_name))  # stdout-compat

    if args.probe:
        row, cert = probe(args.probe, universe)
        print()  # stdout-compat
        print(  # stdout-compat
            f"Probe — {args.probe} vs (universe - self)"
        )
        print("-" * 68)  # stdout-compat
        print(  # stdout-compat
            f"  verdict={row.verdict} dir={row.direction} "
            f"steps={row.n_steps} rules={list(row.rules_used)}"
        )
        if cert is not None:
            for s in cert.steps:
                print(f"    {s.rule} @ {s.path}")  # stdout-compat
        elif row.note:
            print(f"  note: {row.note}")  # stdout-compat

    if args.demo:
        record, replayed = serialize_demo(args.demo, universe)
        print()  # stdout-compat
        print(f"Serialization demo — {args.demo}")  # stdout-compat
        print("-" * 68)  # stdout-compat
        print(json.dumps(record, indent=1))  # stdout-compat
        from catopt_core.ir import op_repr

        print(  # stdout-compat
            f"replayed from data -> {op_repr(replayed)}"
        )
        print("verify_certificate(strict=True): OK")  # stdout-compat

    if args.json:
        Path(args.json).write_text(
            json.dumps(_jsonable(rows, certs), indent=1) + "\n"
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
