"""Law rediscovery — re-deriving a rewrite law from the others.

An operational, falsifiable test of the "laws are emergent" claim.
A rewrite ``lhs -> rhs`` is a 2-cell; a law *about* rewrites is a
3-cell (ADR 0002).  Verifying a proposed law needs **no new
machinery**: put both sides in a fresh small
:class:`~catopt_core.egraph.EGraph`, saturate under the KNOWN laws,
and ask whether the two sides land in one e-class.  The e-graph *is*
the decision procedure for e-class membership.

Why this is cheap where full-program saturation is not: the
saturation wall (~n=12, Catalan e-class growth) is about saturating a
whole *program*.  A law's two sides are tiny, so the check reaches a
fixed point in milliseconds.

The verifier :func:`verify_law` reuses the shipped machinery only —
``EGraph``, ``find``, and the proof-carrying ``certificate`` /
``verify_certificate`` path.  On a merge it emits a positional
certificate and replays it on real terms, independent of the e-graph;
the certificate is tried in both directions (equality is symmetric)
and the replayable one is reported, so a witness is a genuine rule
derivation, never a trusted stub.

Three experiments ride on top:

* :func:`rediscover` — for each law in ``catopt_core.laws.ALL_RULES``,
  is it derivable from the *others*?  This measures how much of the
  library is redundant vs primitive.
* :func:`run_proposals` — a genuinely-new composite equality the
  verifier must CONFIRM, plus false equalities it must REJECT.  Both
  directions matter: a verifier that accepts everything is worthless.
* :func:`run_controls` — an empty rule set must never merge, and each
  law's own rule must always merge (no free merges, no false
  rejections).

Run::

    .venv/bin/python tools/law_verifier.py
    .venv/bin/python tools/law_verifier.py --json /tmp/law.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import Const, Op, TensorType, Var, op_repr
from catopt_core.laws import ALL_RULES
from catopt_core.meta import (
    _positions,
    _subterm,
    apply_rewrite_at,
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)

__all__ = [
    "LawResult",
    "LawRow",
    "ProposalRow",
    "classify_relation",
    "generic_instance",
    "instance_of",
    "main",
    "normalize_pattern",
    "rediscover",
    "run_controls",
    "run_proposals",
    "structural_report",
    "verify_law",
]

#: Saturation budget for a single law check.  Law sides are tiny, so
#: these are generous — every measured case stops at a fixed point.
_MAX_ITERATIONS = 30
_MAX_NODES = 200_000

#: Feature dim used to build the per-law synthetic instances.
_SIZE = 16


@dataclass(frozen=True)
class LawResult:
    """Outcome of asking whether *other* rules derive ``lhs == rhs``.

    ``derivable`` is the e-class verdict.  ``direction`` names the
    certificate that replayed (``"lhs->rhs"`` / ``"rhs->lhs"`` /
    ``""``); ``witness_rules`` are the rules its replayable steps used
    and ``witness_steps`` their count.  ``note`` carries the honest
    caveat when the merge is witnessed but no standalone derivation
    replayed.
    """

    derivable: bool
    stop: str = ""
    n_enodes: int = 0
    n_classes: int = 0
    direction: str = ""
    witness_rules: tuple[str, ...] = ()
    witness_steps: int = 0
    replayable: bool = False
    note: str = ""


@dataclass(frozen=True)
class LawRow:
    """One law's rediscovery row."""

    name: str
    category: str  # derivable | primitive | no-instance
    relation: str = ""  # duplicate | inverse | composite | ""
    lhs_repr: str = ""
    rhs_repr: str = ""
    result: LawResult | None = None


@dataclass(frozen=True)
class ProposalRow:
    """One proposal-battery row: a claimed equality + the verdict."""

    label: str
    expectation: str  # derivable | not-derivable
    lhs_repr: str
    rhs_repr: str
    result: LawResult = field(default_factory=LawResult)
    ok: bool = False


# ---------------------------------------------------------------------------
#  The verifier
# ---------------------------------------------------------------------------


def verify_law(
    lhs: Any,
    rhs: Any,
    rules: Any,
    *,
    max_iterations: int = _MAX_ITERATIONS,
    max_nodes: int = _MAX_NODES,
    rule_budgets: dict[str, int] | None = None,
) -> LawResult:
    """Decide whether *rules* derive the equality ``lhs == rhs``.

    Builds a fresh :class:`EGraph`, interns **both** sides, saturates
    under *rules*, and returns whether the two roots share an e-class.
    On a merge it builds a positional certificate through the shipped
    proof machinery and replays it with :func:`verify_certificate`,
    trying both directions and reporting the replayable one.
    """
    eg = EGraph()
    r_lhs = eg.add_term(lhs)
    r_rhs = eg.add_term(rhs)
    stats = eg.run(
        rules,
        r_lhs,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
        rule_budgets=rule_budgets,
    )
    base = {
        "stop": stats["stop"],
        "n_enodes": stats["n_enodes"],
        "n_classes": stats["n_classes"],
    }
    if eg.find(r_lhs) != eg.find(r_rhs):
        return LawResult(derivable=False, **base)
    # Equality is symmetric, so either direction witnesses the merge.
    # Prefer the one whose steps replay standalone; a rule only present
    # as its reverse makes the opposite direction an e-graph-dependent
    # stub, and we want the genuine derivation.
    attempts: list[LawResult] = []
    for src, dst, eid, tag in (
        (lhs, rhs, r_lhs, "lhs->rhs"),
        (rhs, lhs, r_rhs, "rhs->lhs"),
    ):
        cert = eg.certificate(src, dst, root_eid=eid)
        try:
            verify_certificate(src, cert)
        except Exception as exc:
            attempts.append(
                LawResult(
                    derivable=True,
                    direction=tag,
                    note=f"replay failed: {type(exc).__name__}: {exc}",
                    **base,
                )
            )
            continue
        attempts.append(
            LawResult(
                derivable=True,
                direction=tag,
                witness_rules=tuple(cert.rules_used),
                witness_steps=cert.n_steps,
                replayable=cert.replayable,
                note=""
                if cert.replayable
                else "e-graph-dependent witness",
                **base,
            )
        )
    for attempt in attempts:
        if attempt.replayable:
            return attempt
    return attempts[0]


# ---------------------------------------------------------------------------
#  Structural relation between two laws (duplicate / inverse)
# ---------------------------------------------------------------------------


def normalize_pattern(pat: Any, _mapping: dict | None = None) -> Any:
    """Return *pat* with metavariables renamed to ``v0, v1, …``.

    Two patterns are structurally identical iff their normal forms are
    ``op_repr``-equal, so this exposes exact duplicate and inverse
    rules independent of metavariable spelling.
    """
    mapping = {} if _mapping is None else _mapping
    if isinstance(pat, str):
        if pat not in mapping:
            mapping[pat] = f"v{len(mapping)}"
        return mapping[pat]
    if isinstance(pat, Op):
        args = [normalize_pattern(a, mapping) for a in pat.args]
        attrs = {
            k: normalize_pattern(v, mapping)
            if isinstance(v, str)
            else v
            for k, v in pat.attrs.items()
        }
        return Op.make(pat.op, *args, **attrs)
    return pat


def _key(rule: Any) -> tuple[str, str]:
    return op_repr(normalize_pattern(rule.lhs)), op_repr(
        normalize_pattern(rule.rhs)
    )


def classify_relation(target: Any, witness: Any) -> str:
    """Classify how *witness* relates to *target*.

    Returns ``"duplicate"`` (same lhs/rhs structure), ``"inverse"``
    (the swapped structure) or ``"composite"`` (anything else).
    """
    if target.name == witness.name:
        return "composite"
    tl, tr = _key(target)
    wl, wr = _key(witness)
    if (tl, tr) == (wl, wr):
        return "duplicate"
    if (tl, tr) == (wr, wl):
        return "inverse"
    return "composite"


# ---------------------------------------------------------------------------
#  Per-law synthetic instances (reused from the law_bench registry)
# ---------------------------------------------------------------------------


def _law_cases() -> dict:
    """Return bench's per-law ``LAW_CASES`` instance registry.

    Imported lazily so ``verify_law`` itself needs neither torch nor
    the bench package.  The repo root is put on ``sys.path`` because
    ``bench`` is a package, not an installed distribution.
    """
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from bench.suites.correctness.law_bench import LAW_CASES

    return LAW_CASES


def instance_of(rule: Any, size: int = _SIZE) -> tuple[Any, Any] | None:
    """Return ``(lhs_instance, rhs_instance)`` for *rule*, or None.

    Prefers the bench registry's synthetic term (a well-typed graph
    where the law's LHS applies), locates the first matching subterm,
    and rewrites it in place to obtain the RHS instance.  Falls back to
    :func:`generic_instance` (pattern-driven) for rules the registry
    does not cover — currently the SDPA-fold spelling variants.
    """
    build = _law_cases().get(rule.name)
    if build is None:
        return generic_instance(rule)
    import torch

    term, _env, _inputs = build(size, torch.device("cpu"))
    for path, sub in _positions(term):
        if match_pattern(rule.lhs, sub) is None:
            continue
        out = apply_rewrite_at(rule, term, path)
        if out is None:
            continue
        return sub, _subterm(out, path)
    return None


#: Leaf shapes for the SDPA-fold family's metavariables — the only
#: rules the bench registry does not cover.  ``(B,H,T,D)`` scores feed
#: ``(B,H,T,T)`` masks and a ``(B,H,T,D)`` value.
_LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "Q": (2, 4, 8, 4),
    "K": (2, 4, 8, 4),
    "V": (2, 4, 8, 4),
    "M": (2, 4, 8, 8),
    "MK": (2, 4, 8, 8),
}

#: Attr-metavariable defaults for the SDPA-fold family.
_ATTR_DEFAULTS: dict[str, Any] = {
    "TD1": -2,
    "TD2": -1,
    "SD": -1,
    "DP": 0.5,
    "DT": True,
}


def generic_instance(rule: Any) -> tuple[Any, Any] | None:
    """Build a well-typed instance from *rule*'s own LHS pattern.

    Substitutes the pattern's metavariables with shape-correct leaves
    (the SDPA-fold family's ``Q``/``K``/``V``/mask names and its
    transpose/softmax/dropout attrs), runs the ``check`` side condition
    and the ``derive`` hook, and returns ``(lhs, rhs)`` instances.
    Returns ``None`` for any metavariable outside the known table, or
    when the check vetoes — so an unbuildable instance is reported
    honestly, never guessed.
    """
    subst: dict[str, Any] = {}
    for mv in sorted(pattern_metavars(rule.lhs)):
        if mv.startswith("$attr:"):
            name = mv[len("$attr:") :]
            if name not in _ATTR_DEFAULTS:
                return None
            subst[mv] = _ATTR_DEFAULTS[name]
        elif mv in _LEAF_SHAPES:
            subst[mv] = Var(mv, TensorType(_LEAF_SHAPES[mv]))
        elif mv == "S":
            subst[mv] = Const(0.5)
        elif mv == "F":
            subst[mv] = Const(float("-inf"))
        else:
            return None
    if rule.check is not None:
        try:
            if not rule.check(subst):
                return None
        except Exception:
            return None
    lhs = instantiate_pattern(rule.lhs, subst)
    inst = dict(subst)
    if rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            return None
        if extra is None:
            return None
        inst.update(extra)
    return lhs, instantiate_pattern(rule.rhs, inst)


# ---------------------------------------------------------------------------
#  Experiment 1 — rediscovery
# ---------------------------------------------------------------------------


def rediscover(
    rules: list | None = None,
    *,
    max_nodes: int = _MAX_NODES,
    size: int = _SIZE,
) -> list[LawRow]:
    """Check each law's derivability from the *others*.

    For every rule ``L``, saturate ``L``'s two instance sides under
    the universe with ``L`` removed.  ``derivable`` means the rest of
    the library already proves ``L``; ``primitive`` means it does not;
    and ``no-instance`` records a law with no registered synthetic
    case.
    """
    universe = list(ALL_RULES if rules is None else rules)
    by_name = {r.name: r for r in universe}
    rows: list[LawRow] = []
    for rule in universe:
        inst = instance_of(rule, size)
        if inst is None:
            rows.append(LawRow(rule.name, "no-instance"))
            continue
        lhs, rhs = inst
        others = [r for r in universe if r.name != rule.name]
        res = verify_law(lhs, rhs, others, max_nodes=max_nodes)
        relation = ""
        if res.derivable and res.witness_rules:
            w = by_name.get(res.witness_rules[0])
            if w is not None:
                relation = classify_relation(rule, w)
        rows.append(
            LawRow(
                name=rule.name,
                category="derivable" if res.derivable else "primitive",
                relation=relation,
                lhs_repr=op_repr(lhs),
                rhs_repr=op_repr(rhs),
                result=res,
            )
        )
    return rows


# ---------------------------------------------------------------------------
#  Experiment 2 — proposal (confirm new equalities, reject false ones)
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _proposal_cases() -> list[tuple[str, str, Any, Any, Any]]:
    """Return the proposal battery as ``(label, expect, lhs, rhs, rules)``.

    ``expect`` is ``"derivable"`` or ``"not-derivable"``.  The
    accepted cases are equalities that are NOT a single library law —
    composites of distinct laws — plus one genuine rewrite discovered
    by composition.  The rejected cases are false equalities, plus one
    true-but-not-derivable conjecture under a restricted rule set.
    """
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    a, b = _v("a", 4, 4), _v("b", 4, 4)
    w = _v("W", 4, 4)
    no_id = [r for r in ALL_RULES if r.name != "id_add"]
    return [
        # -- confirm: genuinely new equalities (composites) ---------
        (
            "composite silu(neg(neg(x)))=x*sig(x)",
            "derivable",
            Op.make("silu", Op.make("neg", Op.make("neg", x))),
            Op.make("mul", x, Op.make("sigmoid", x)),
            ALL_RULES,
        ),
        (
            "composite sub(x,neg(y))=add(x,y)",
            "derivable",
            Op.make("sub", x, Op.make("neg", y)),
            Op.make("add", x, y),
            ALL_RULES,
        ),
        (
            "composite factor after comm",
            "derivable",
            Op.make(
                "add",
                Op.make("matmul", w, a),
                Op.make("matmul", w, b),
            ),
            Op.make("matmul", w, Op.make("add", b, a)),
            ALL_RULES,
        ),
        # -- reject: false equalities -------------------------------
        (
            "false add(x,y) = mul(x,y)",
            "not-derivable",
            Op.make("add", x, y),
            Op.make("mul", x, y),
            ALL_RULES,
        ),
        (
            "false matmul(A,B) = matmul(B,A)",
            "not-derivable",
            Op.make("matmul", a, b),
            Op.make("matmul", b, a),
            ALL_RULES,
        ),
        (
            "false neg(x) = x",
            "not-derivable",
            Op.make("neg", x),
            x,
            ALL_RULES,
        ),
        (
            "false sub(x,y) = sub(y,x)",
            "not-derivable",
            Op.make("sub", x, y),
            Op.make("sub", y, x),
            ALL_RULES,
        ),
        # -- reject: true but not derivable (a conjecture) ----------
        (
            "conjecture add(x,0)=x without id_add",
            "not-derivable",
            Op.make("add", x, Const(0)),
            x,
            no_id,
        ),
    ]


def run_proposals(*, max_nodes: int = _MAX_NODES) -> list[ProposalRow]:
    """Run the proposal battery; return one verdict row per case."""
    rows: list[ProposalRow] = []
    for label, expect, lhs, rhs, rules in _proposal_cases():
        res = verify_law(lhs, rhs, rules, max_nodes=max_nodes)
        got = "derivable" if res.derivable else "not-derivable"
        rows.append(
            ProposalRow(
                label=label,
                expectation=expect,
                lhs_repr=op_repr(lhs),
                rhs_repr=op_repr(rhs),
                result=res,
                ok=got == expect,
            )
        )
    return rows


# ---------------------------------------------------------------------------
#  Experiment 3 — soundness controls over every instanced law
# ---------------------------------------------------------------------------


def run_controls(
    rules: list | None = None, *, max_nodes: int = _MAX_NODES
) -> dict:
    """Run the empty-set / self-rule controls over every instanced law.

    The empty rule set must NEVER merge a law's two sides (the verifier
    is not a rubber stamp); the law's own rule alone MUST merge them
    (the verifier does not reject true equalities).  Returns the count
    of laws checked and the list of failures — empty means clean.
    """
    universe = list(ALL_RULES if rules is None else rules)
    failures: list[str] = []
    n = 0
    for rule in universe:
        inst = instance_of(rule)
        if inst is None:
            continue
        n += 1
        lhs, rhs = inst
        if verify_law(lhs, rhs, [], max_nodes=max_nodes).derivable:
            failures.append(f"{rule.name}: merged with NO rules")
        if not verify_law(
            lhs, rhs, [rule], max_nodes=max_nodes
        ).derivable:
            failures.append(f"{rule.name}: own rule did not derive it")
    return {"n": n, "failures": failures}


def structural_report(rules: list | None = None) -> dict:
    """List exact duplicate rules and structural inverse pairs.

    Independent of the verifier: two rules are duplicates iff their
    ``(lhs, rhs)`` normal forms coincide, inverses iff one is the
    other's swap.
    """
    universe = list(ALL_RULES if rules is None else rules)
    keys = {r.name: _key(r) for r in universe}
    dups: list[list[str]] = []
    seen: dict[tuple, list[str]] = {}
    for r in universe:
        seen.setdefault(keys[r.name], []).append(r.name)
    for names in seen.values():
        if len(names) > 1:
            dups.append(sorted(names))
    inverses: list[list[str]] = []
    for r in universe:
        swapped = (keys[r.name][1], keys[r.name][0])
        for name, key in keys.items():
            if name != r.name and key == swapped:
                pair = sorted([r.name, name])
                if pair not in inverses:
                    inverses.append(pair)
    return {"duplicates": sorted(dups), "inverses": sorted(inverses)}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _fmt_rows(rows: list[LawRow]) -> str:
    lines = [
        f"{'law':<34} {'category':<11} {'rel':<10} "
        f"{'witness':<26} {'steps':>5} {'enodes':>7}",
        "-" * 98,
    ]
    for row in rows:
        res = row.result
        if res is None:
            lines.append(f"{row.name:<34} {row.category:<11}")
            continue
        wit = ",".join(res.witness_rules) or "-"
        lines.append(
            f"{row.name:<34} {row.category:<11} "
            f"{row.relation or '-':<10} {wit:<26} "
            f"{res.witness_steps:>5} {res.n_enodes:>7}"
        )
    return "\n".join(lines)


def _fmt_proposals(rows: list[ProposalRow]) -> str:
    lines = [
        f"{'case':<40} {'expect':<15} {'got':<15} {'ok':<4} witness",
        "-" * 100,
    ]
    for row in rows:
        got = "derivable" if row.result.derivable else "not-derivable"
        wit = ",".join(row.result.witness_rules) or "-"
        lines.append(
            f"{row.label:<40} {row.expectation:<15} {got:<15} "
            f"{'yes' if row.ok else 'NO':<4} {wit}"
        )
    return "\n".join(lines)


def _summary(rows: list[LawRow]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
    rels: dict[str, int] = {}
    for row in rows:
        if row.category == "derivable":
            rels[row.relation or "?"] = (
                rels.get(row.relation or "?", 0) + 1
            )
    return {"counts": counts, "derivable_relations": rels}


def _dump_json(
    path: str,
    rows: list[LawRow],
    proposals: list[ProposalRow],
    controls: dict,
    struct: dict,
) -> None:
    payload = {
        "rediscovery": [
            {
                "name": r.name,
                "category": r.category,
                "relation": r.relation,
                "lhs": r.lhs_repr,
                "rhs": r.rhs_repr,
                "result": None
                if r.result is None
                else asdict(r.result),
            }
            for r in rows
        ],
        "proposals": [
            {
                "label": p.label,
                "expectation": p.expectation,
                "ok": p.ok,
                "lhs": p.lhs_repr,
                "rhs": p.rhs_repr,
                "result": asdict(p.result),
            }
            for p in proposals
        ],
        "controls": controls,
        "structural": struct,
        "summary": _summary(rows),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run both experiments and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=_MAX_NODES,
        help="per-check saturation node budget",
    )
    parser.add_argument("--size", type=int, default=_SIZE)
    args = parser.parse_args(argv)

    rows = rediscover(max_nodes=args.max_nodes, size=args.size)
    proposals = run_proposals(max_nodes=args.max_nodes)
    controls = run_controls(max_nodes=args.max_nodes)
    struct = structural_report()

    print(
        "== experiment 1: law rediscovery (derivable from the others?) =="
    )
    print(_fmt_rows(rows))
    print()
    print("summary:", _summary(rows))
    print()
    print("structural duplicates:", struct["duplicates"])
    print("structural inverse pairs:", len(struct["inverses"]))
    print()
    print("== experiment 2: proposal (confirm new, reject false) ==")
    print(_fmt_proposals(proposals))
    print()
    n_ok = sum(1 for p in proposals if p.ok)
    print(f"proposals correct: {n_ok}/{len(proposals)}")
    bad = [p.label for p in proposals if not p.ok]
    if bad:
        print("MISMATCHES:", bad)
    print()
    print("== experiment 3: soundness controls ==")
    print(
        f"laws checked: {controls['n']}; failures: "
        f"{controls['failures'] or 'none'}"
    )

    if args.json:
        _dump_json(args.json, rows, proposals, controls, struct)
        print(f"\nwrote {args.json}")
    return 0 if not bad and not controls["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
