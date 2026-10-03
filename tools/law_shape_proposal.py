"""Shape-aware law proposal — target the shapes real models contain.

The proposal retro (`project/retros/law-proposal.md`) enumerated an
*algebraic* grammar and found 11 true, cost-reducing laws; the impact
retro (`project/retros/law-impact.md`) then measured them on real
graphs and got **zero firings** — the ops were present, the *shapes*
were not.  Proposal was decoupled from reality: it optimised algebraic
novelty, not applicability.

This tool closes that loop.  It reads the shape census
(`tools/law_shape_census.py`) and, for the shapes real models actually
contain, asks the only question that matters: **is there a true,
cost-reducing equality whose LHS is (or contains) that shape?**

For each *schema* — a generic metavariable equality such as
``add(mul(A,B), mul(A,C)) -> mul(A, add(B,C))`` — it measures:

* **real matches** — subterms of the corpus (bench law cases + the 22
  exported ``catopt_torch.models`` blocks) that the schema's LHS
  matches *with its metavariable equalities enforced*.  A schema whose
  LHS never matches with the equalities is inapplicable, however true
  it is.
* **relaxed matches** — the same count with repeated metavariables
  relaxed to wildcards (the pure *shape*, no precondition).  The gap
  between the two is exactly "the shape is present, the equality is
  not" — the impact retro's finding, made per-schema.
* **true** — the instantiated LHS and RHS agree numerically on random
  fp64 tensors of the *real* leaf shapes (the numeric oracle
  ``law_proposal`` uses for new axioms).
* **derivable** — ``law_verifier.verify_law`` proves it from
  ``ALL_RULES`` (then it is a composite and cannot lower a cost).
* **new** — structurally not a duplicate or inverse of a library law.
* **useful** — saturating a real matched term under ``ALL_RULES`` vs
  ``ALL_RULES + {schema}`` strictly lowers the extracted cost.
* **fires / pays** — running the schema *alone* over every real model,
  how many fire and whether the extracted cost drops (verified through
  the pipeline's own ``_lower_extracted`` + ``sink.verify``).

The schemas are defined *inside this tool* — nothing is added to
``packages/`` or the shipped ``ALL_RULES``.  A firing table of zeros
is a decisive answer, not a bug.

Run::

    .venv/bin/python tools/law_shape_proposal.py
    .venv/bin/python tools/law_shape_proposal.py --json /tmp/shape.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.cost import dag_cost
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op
from catopt_core.laws import ALL_RULES

# Sibling tools: the census/corpus, the verifier, and the numeric
# oracle + saturation helpers the proposal retro already built.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_impact import (
    _bench_cases,
    _cost_fn,
    _iter_subterms,
    model_cases,
    new_laws,
    reach_row,
)
from law_proposal import (
    _library_keys,
    _numeric_true,
    _relation,
    _sat_cost,
)
from law_verifier import verify_law

__all__ = [
    "Schema",
    "SchemaOutcome",
    "main",
    "real_matches",
    "relaxed_matches",
    "run_proposal",
    "schemas",
]

#: Saturation budget for a cost-change check on a real term.
_COST_ITERS = 5
_COST_NODES = 40_000

#: Firing budget (one law alone over a model).
_FIRE_ITERS = 4
_FIRE_NODES = 40_000


# ---------------------------------------------------------------------------
#  Schema library — honest, generic identities over the real vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schema:
    """One generic equality ``lhs -> rhs`` over metavariable patterns."""

    name: str
    lhs: Any
    rhs: Any
    note: str = ""
    family: str = ""

    def as_rule(self) -> Rewrite:
        """Return the schema as a fireable ``Rewrite``."""
        return Rewrite(self.name, self.lhs, self.rhs)


def _op(name: str, *args: Any, **attrs: Any) -> Op:
    """Build an ``Op`` pattern node (metavariable leaves are ``str``)."""
    return Op.make(name, *args, **attrs)


def schemas() -> list[Schema]:
    """Return the schema library, grouped by the shape it targets.

    Every schema is a *true* algebraic identity over the real op
    vocabulary (elementwise arithmetic, the ``select``/``slice``
    naturality, layout).  Some are deliberately library duplicates
    (the ``*_factor`` controls) so the tool's duplicate detector is
    exercised, and some target shapes the census found frequent
    (``mul(select, select)``, ``add(mul, mul)``).
    """
    a, b, c, x = "A", "B", "C", "X"
    d, i = "D", "I"
    out: list[Schema] = []

    # -- elementwise factorization (targets add(mul,mul), sub(mul,mul)) --
    out.append(
        Schema(
            "factor_left",
            _op("add", _op("mul", a, b), _op("mul", a, c)),
            _op("mul", a, _op("add", b, c)),
            note="x*y + x*z = x*(y+z) (shared left factor).",
            family="elementwise-factor",
        )
    )
    out.append(
        Schema(
            "factor_right",
            _op("add", _op("mul", b, a), _op("mul", c, a)),
            _op("mul", _op("add", b, c), a),
            note="y*x + z*x = (y+z)*x (shared right factor).",
            family="elementwise-factor",
        )
    )
    out.append(
        Schema(
            "factor_mid",
            _op("add", _op("mul", a, b), _op("mul", c, b)),
            _op("mul", _op("add", a, c), b),
            note="x*y + z*y = (x+z)*y (shared right factor, y).",
            family="elementwise-factor",
        )
    )
    out.append(
        Schema(
            "factor_sub_left",
            _op("sub", _op("mul", a, b), _op("mul", a, c)),
            _op("mul", a, _op("sub", b, c)),
            note="x*y - x*z = x*(y-z).",
            family="elementwise-factor",
        )
    )
    out.append(
        Schema(
            "factor_sub_right",
            _op("sub", _op("mul", a, b), _op("mul", c, b)),
            _op("mul", _op("sub", a, c), b),
            note="x*y - z*y = (x-z)*y.",
            family="elementwise-factor",
        )
    )

    # -- select / slice naturality (targets mul(select,select)) --
    sel_a = _op("select", a, dim=d, index=i)
    sel_b = _op("select", b, dim=d, index=i)
    out.append(
        Schema(
            "select_mul",
            _op("mul", sel_a, sel_b),
            _op("select", _op("mul", a, b), dim=d, index=i),
            note="mul commutes with select: sel(x)*sel(y)=sel(x*y).",
            family="select-naturality",
        )
    )
    out.append(
        Schema(
            "select_add",
            _op(
                "add",
                _op("select", a, dim=d, index=i),
                _op("select", b, dim=d, index=i),
            ),
            _op("select", _op("add", a, b), dim=d, index=i),
            note="add commutes with select: sel(x)+sel(y)=sel(x+y).",
            family="select-naturality",
        )
    )
    out.append(
        Schema(
            "select_sub",
            _op(
                "sub",
                _op("select", a, dim=d, index=i),
                _op("select", b, dim=d, index=i),
            ),
            _op("select", _op("sub", a, b), dim=d, index=i),
            note="sub commutes with select: sel(x)-sel(y)=sel(x-y).",
            family="select-naturality",
        )
    )
    sl_a = _op("slice", a, dim=d, start="S0", end="E0")
    sl_b = _op("slice", b, dim=d, start="S0", end="E0")
    out.append(
        Schema(
            "slice_mul",
            _op("mul", sl_a, sl_b),
            _op("slice", _op("mul", a, b), dim=d, start="S0", end="E0"),
            note="mul commutes with slice (equal range).",
            family="slice-naturality",
        )
    )

    # -- layout (targets transpose(reshape), reshape(reshape)) --
    out.append(
        Schema(
            "reshape_reshape",
            _op(
                "reshape",
                _op("reshape", a, shape="S1"),
                shape="S2",
            ),
            _op("reshape", a, shape="S2"),
            note="consecutive reshapes fuse.",
            family="layout",
        )
    )
    out.append(
        Schema(
            "reshape_transpose",
            _op(
                "transpose",
                _op("reshape", a, shape="S"),
                dim0=d,
                dim1=i,
            ),
            _op(
                "reshape",
                _op("transpose", a, dim0=d, dim1=i),
                shape="S",
            ),
            note="conjecture — false in general (oracle must reject).",
            family="layout",
        )
    )

    # -- elementwise algebra (targets mul(silu,linear), neg/dist) --
    out.append(
        Schema(
            "neg_add",
            _op("add", _op("neg", a), _op("neg", b)),
            _op("neg", _op("add", a, b)),
            note="-x + -y = -(x+y).",
            family="elementwise-algebra",
        )
    )
    out.append(
        Schema(
            "sub_neg",
            _op("sub", a, _op("neg", b)),
            _op("add", a, b),
            note="x - (-y) = x + y.",
            family="elementwise-algebra",
        )
    )
    out.append(
        Schema(
            "mul_neg_left",
            _op("mul", _op("neg", a), b),
            _op("neg", _op("mul", a, b)),
            note="(-x)*y = -(x*y).",
            family="elementwise-algebra",
        )
    )
    out.append(
        Schema(
            "exp_add",
            _op("mul", _op("exp", a), _op("exp", b)),
            _op("exp", _op("add", a, b)),
            note="e^x * e^y = e^(x+y).",
            family="elementwise-algebra",
        )
    )
    out.append(
        Schema(
            "square_neg",
            _op("square", _op("neg", a)),
            _op("square", a),
            note="(-x)^2 = x^2.",
            family="elementwise-algebra",
        )
    )

    # -- linear/matmul factor (library controls; already shipped) --
    out.append(
        Schema(
            "linear_factor",
            _op("add", _op("linear", x, a), _op("linear", x, b)),
            _op("linear", x, _op("add", a, b)),
            note="dup of weight_factor_linear (control).",
            family="linear-control",
        )
    )
    out.append(
        Schema(
            "matmul_factor",
            _op("add", _op("matmul", x, a), _op("matmul", x, b)),
            _op("matmul", x, _op("add", a, b)),
            note="dup of weight_factor_matmul (control).",
            family="linear-control",
        )
    )

    # -- constants (annihilator / identity; targets are rare) --
    out.append(
        Schema(
            "mul_zero",
            _op("mul", a, Const(0)),
            Const(0),
            note="x*0 = 0.",
            family="annihilator",
        )
    )
    out.append(
        Schema(
            "id_mul_lit",
            _op("mul", a, Const(1)),
            a,
            note="x*1 = x (dup of id_mul).",
            family="annihilator",
        )
    )
    out.append(
        Schema(
            "sub_self",
            _op("sub", a, a),
            Const(0),
            note="x - x = 0.",
            family="annihilator",
        )
    )
    return out


# ---------------------------------------------------------------------------
#  Match counts — does the shape (and its equality) appear?
# ---------------------------------------------------------------------------


def _relax(lhs: Any) -> Any:
    """Rename each repeated metavariable to a fresh name (wildcards)."""
    counts: dict[str, int] = {}

    def walk(t: Any) -> Any:
        if isinstance(t, Op):
            return Op.make(
                t.op,
                *(walk(a) for a in t.args),
                **dict(t.attrs),
            )
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
            return t if counts[t] == 1 else f"{t}__{counts[t]}"
        return t

    return walk(lhs)


def real_matches(terms: list[Any], schema: Schema) -> list[Any]:
    """Return every real subterm the schema's LHS matches (equalities on).

    ``_term_match`` enforces that a repeated metavariable binds a
    structurally equal subterm, so this is the *applicable* count, not
    the shape count.  Subterms are deduped by identity (matching the
    census's ``_iter_subterms``), so a shared node is counted once, not
    once per path.
    """
    out: list[Any] = []
    for term in terms:
        for sub in _iter_subterms(term):
            if not isinstance(sub, Op):
                continue
            if _term_match(schema.lhs, sub) is not None:
                out.append(sub)
    return out


def relaxed_matches(terms: list[Any], schema: Schema) -> int:
    """Count real subterms matching the LHS with metavars relaxed.

    Repeated *argument* metavariables become wildcards, so this counts
    the pure shape (e.g. ``add(mul, mul)``); attribute metavariables are
    left shared, so a shape whose precondition is on attributes (the
    ``select`` dim/index) still enforces it.  Deduped by identity.
    """
    pat = _relax(schema.lhs)
    n = 0
    for term in terms:
        for sub in _iter_subterms(term):
            if (
                isinstance(sub, Op)
                and _term_match(pat, sub) is not None
            ):
                n += 1
    return n


# ---------------------------------------------------------------------------
#  Per-schema outcome
# ---------------------------------------------------------------------------


@dataclass
class SchemaOutcome:
    """One schema plus every measured verdict."""

    schema: Schema
    relaxed: int = 0
    matches: int = 0
    example: str = ""
    num_true: bool | None = None
    derivable: bool = False
    relation: str = "new"
    base_cost: float = 0.0
    cand_cost: float = 0.0
    model_fires: int = 0
    fire_cases: tuple[str, ...] = ()
    fire_changed: int = 0
    fire_paid: int = 0

    @property
    def new(self) -> bool:
        """True iff not a duplicate or inverse of a library law."""
        return self.relation == "new"

    @property
    def truth(self) -> bool:
        """True iff derivable, or numerically true on real shapes."""
        return self.derivable or self.num_true is True

    @property
    def useful(self) -> bool:
        """True iff a *true* rule strictly lowered a real term's cost."""
        return self.truth and self.cand_cost < self.base_cost


# ---------------------------------------------------------------------------
#  Truth / derivability / newness
# ---------------------------------------------------------------------------


def _instantiate(schema: Schema, sub: Any) -> Any | None:
    """Return the RHS instance of *schema* on the matched subterm *sub*."""
    subst = _term_match(schema.lhs, sub)
    if subst is None:
        return None
    return _term_instantiate(schema.rhs, subst)


# ---------------------------------------------------------------------------
#  Cost delta + firing on real graphs
# ---------------------------------------------------------------------------


def _cost_delta(
    schema: Schema, matches: list[Any], cost_fn: Any
) -> tuple:
    """Return ``(base, cand)`` costs over the schema's matched terms."""
    rule = schema.as_rule()
    base = float("inf")
    with_rule = float("inf")
    for term in matches[:8]:
        base = min(base, _sat_cost(term, list(ALL_RULES), cost_fn))
        with_rule = min(
            with_rule,
            _sat_cost(term, [*ALL_RULES, rule], cost_fn),
        )
    return base, with_rule


def _fires_on_models(
    schema: Schema, models: list[Any], cost_fn: Any
) -> tuple:
    """Run *schema* alone over every model; return firing aggregates."""
    rule = schema.as_rule()
    fires = 0
    cases: list[str] = []
    changed = 0
    paid = 0
    for c in models:
        eg = EGraph()
        root = eg.add_term(c.term)
        eg.run(
            [rule],
            root,
            max_iterations=_FIRE_ITERS,
            max_nodes=_FIRE_NODES,
        )
        n = eg.rule_fires.get(schema.name, 0)
        if not n:
            continue
        fires += n
        cases.append(c.name)
        best = eg.extract_best(root, cost_fn)
        if best is None or best == c.term:
            continue
        changed += 1
        if dag_cost(best, cost_fn) < dag_cost(c.term, cost_fn):
            paid += 1
    return fires, tuple(cases), changed, paid


_SINK: Any = None


def _sink() -> Any:
    """Return a shared ``TorchSink`` (built once)."""
    global _SINK
    if _SINK is None:
        from catopt_torch.adapters import TorchSink

        _SINK = TorchSink()
    return _SINK


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def run_proposal() -> dict:
    """Measure every schema over the real corpus; return the result."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    real = [c.term for c in [*bench, *models]]
    cost_fn = _cost_fn(_sink())
    lib = _library_keys()

    outcomes: list[SchemaOutcome] = []
    for schema in schemas():
        out = SchemaOutcome(schema=schema)
        out.relaxed = relaxed_matches(real, schema)
        matches = real_matches(real, schema)
        out.matches = len(matches)
        out.relation = _relation(schema.lhs, schema.rhs, lib)
        if matches:
            from catopt_core.ir import op_repr

            out.example = op_repr(matches[0])
            rhs = _instantiate(schema, matches[0])
            if rhs is not None:
                out.num_true = _numeric_true(matches[0], rhs)
                out.derivable = verify_law(
                    matches[0], rhs, ALL_RULES
                ).derivable
            out.base_cost, out.cand_cost = _cost_delta(
                schema, matches, cost_fn
            )
        (
            out.model_fires,
            out.fire_cases,
            out.fire_changed,
            out.fire_paid,
        ) = _fires_on_models(schema, models, cost_fn)
        outcomes.append(out)

    return {
        "n_bench": len(bench),
        "n_models": len(models),
        "outcomes": outcomes,
        "baseline": _baseline(models),
        "reach": _reach(outcomes, models, cost_fn),
    }


def _reach(
    outcomes: list[SchemaOutcome],
    models: list[Any],
    cost_fn: Any,
) -> list[dict]:
    """End-to-end reach for every schema that fires on a real model.

    Saturates each model under ``ALL_RULES`` vs
    ``ALL_RULES + {schema}`` and reports the extracted-cost delta and
    whether the certificate still replays — the same measurement the
    impact retro used, so a "pays" claim is end-to-end, not just on a
    matched subterm.
    """
    sink = _sink()
    rows: list[dict] = []
    for o in outcomes:
        if not o.model_fires:
            continue
        for c in models:
            r = reach_row(c, [o.schema.as_rule()], sink, cost_fn)
            if not r["new_fires"] and not r["changed"]:
                continue
            rows.append(
                {
                    "schema": o.schema.name,
                    "model": r["model"],
                    "base_cost": r["base_cost"],
                    "add_cost": r["add_cost"],
                    "changed": r["changed"],
                    "fires": sum(r["new_fires"].values()),
                    "cert": r["add_cert"],
                }
            )
    return rows


def _baseline(models: list[Any]) -> dict:
    """Measure the 11 earlier laws for comparison (the impact baseline)."""
    rules = new_laws()
    fired: dict[str, int] = {}
    for c in models:
        eg = EGraph()
        root = eg.add_term(c.term)
        eg.run(
            rules,
            root,
            max_iterations=_FIRE_ITERS,
            max_nodes=_FIRE_NODES,
        )
        for r in rules:
            n = eg.rule_fires.get(r.name, 0)
            if n:
                fired[r.name] = fired.get(r.name, 0) + n
    return {"laws": [r.name for r in rules], "fires": fired}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _match_table(outs: list[SchemaOutcome]) -> str:
    """Render the shape-match / truth / usefulness table."""
    head = (
        f"{'schema':<20} {'relax':>6} {'match':>6} {'true':>5} "
        f"{'deriv':>5} {'rel':>10} {'cost':>18} {'use':>4}"
    )
    lines = [head, "-" * len(head)]
    for o in outs:
        true = {True: "yes", False: "no", None: "-"}[o.num_true]
        cost = (
            f"{o.base_cost:.3g}->{o.cand_cost:.3g}"
            if o.matches
            else "-"
        )
        lines.append(
            f"{o.schema.name:<20} {o.relaxed:>6} {o.matches:>6} "
            f"{true:>5} {o.derivable!s:>5} {o.relation:>10} "
            f"{cost:>18} {'yes' if o.useful else 'no':>4}"
        )
    return "\n".join(lines)


def _fire_table(outs: list[SchemaOutcome]) -> str:
    """Render the per-schema firing table on the real models."""
    head = (
        f"{'schema':<20} {'fires':>6} {'changed':>8} {'paid':>5}  cases"
    )
    lines = [head, "-" * len(head)]
    for o in outs:
        cases = ",".join(o.fire_cases[:3])
        if len(o.fire_cases) > 3:
            cases += "…"
        lines.append(
            f"{o.schema.name:<20} {o.model_fires:>6} "
            f"{o.fire_changed:>8} {o.fire_paid:>5}  {cases}"
        )
    return "\n".join(lines)


def _reach_table(rows: list[dict]) -> str:
    """Render the end-to-end reach rows."""
    if not rows:
        return "  (no schema changed a real model end-to-end)"
    head = (
        f"{'schema':<18} {'model':<18} {'cost':>22} "
        f"{'fires':>6} {'cert':>5}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        cost = f"{r['base_cost']:.3g}->{r['add_cost']:.3g}"
        lines.append(
            f"{r['schema']:<18} {r['model']:<18} {cost:>22} "
            f"{r['fires']:>6} {r['cert']:>5}"
        )
    return "\n".join(lines)


def _verdict(result: dict) -> None:
    """Print the plain verdict the retro records."""
    outs: list[SchemaOutcome] = result["outcomes"]
    real = [o for o in outs if o.matches]
    true_new = [o for o in real if o.truth and o.new]
    useful = [o for o in outs if o.useful and o.new]
    fires = [o for o in outs if o.model_fires]
    pays = [o for o in outs if o.model_fires and o.fire_paid]
    true_pays = [o for o in pays if o.truth and o.new]
    false_pays = [o for o in pays if not (o.truth and o.new)]
    print("== verdict ==")
    print(
        f"  schemas: {len(outs)}; with a real (equality-enforced) "
        f"match: {len(real)}"
    )
    print(f"  true & new among those: {len(true_new)}")
    print(f"  useful (cost-lowering on a real term): {len(useful)}")
    print(f"  firing on a real model: {len(fires)}")
    print(f"  firing AND cost-lowering: {len(pays)}")
    print(f"    of which true & new: {len(true_pays)}")
    print(
        f"    of which false/duplicate (cost-only): {len(false_pays)}"
    )
    base = result["baseline"]
    print(
        f"  baseline (11 earlier laws): fires on models = "
        f"{sum(base['fires'].values())}"
    )
    if true_pays:
        print("  paying & true schemas:")
        for o in true_pays:
            print(
                f"    - {o.schema.name}: fires={o.model_fires}, "
                f"paid={o.fire_paid}, cases={o.fire_cases}"
            )
    if false_pays:
        print(
            "  cost-lowering but NOT true (oracle rejects — the "
            "reason a cost-only proposer is unsafe):"
        )
        for o in false_pays:
            print(
                f"    - {o.schema.name}: num_true={o.num_true}, "
                f"relation={o.relation}"
            )


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable result."""
    outs: list[SchemaOutcome] = result["outcomes"]
    payload = {
        "n_bench": result["n_bench"],
        "n_models": result["n_models"],
        "baseline": result["baseline"],
        "reach": result["reach"],
        "outcomes": [
            {
                "schema": o.schema.name,
                "family": o.schema.family,
                "note": o.schema.note,
                "relaxed": o.relaxed,
                "matches": o.matches,
                "example": o.example,
                "num_true": o.num_true,
                "derivable": o.derivable,
                "relation": o.relation,
                "base_cost": o.base_cost,
                "cand_cost": o.cand_cost,
                "useful": o.useful,
                "model_fires": o.model_fires,
                "fire_cases": list(o.fire_cases),
                "fire_changed": o.fire_changed,
                "fire_paid": o.fire_paid,
            }
            for o in outs
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the shape-aware proposal and print (or dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    result = run_proposal()
    outs: list[SchemaOutcome] = result["outcomes"]

    print("== law_shape_proposal — laws aimed at real shapes ==")
    print(
        f"   corpus: {result['n_bench']} bench + "
        f"{result['n_models']} models; {len(outs)} schemas"
    )
    print()
    print("-- match / truth / usefulness (relax = shape only) --")
    print(_match_table(outs))
    print()
    print("-- firing on real models (each schema run alone) --")
    print(_fire_table(outs))
    print()
    print("-- examples (first real match per applicable schema) --")
    for o in outs:
        if o.example:
            print(f"  {o.schema.name:<20} {o.example[:90]}")
    print()
    print("-- end-to-end reach (ALL_RULES vs ALL_RULES + schema) --")
    print(_reach_table(result["reach"]))
    print()
    _verdict(result)

    if args.json:
        _dump_json(args.json, result)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
