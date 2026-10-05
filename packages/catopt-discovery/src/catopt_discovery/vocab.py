"""Machine-derived op vocabulary — classify corpus ops by property.

The law-discovery pipeline (``catopt_discovery.pipeline``) derives the
*shapes* it proposes from the corpus (the census's frequent ``f(g, g)``
op-tuples), but takes the *op alphabet* — which ops are pointwise,
which are views — from two hand-written tuples (``_POINTWISE``,
``_VIEW_OPS``).  ``select_mul`` was found only because ``select``
happened to be listed; an op absent from the tables is invisible to
the generator.  This tool removes that last human ingredient: it mines
every op that actually occurs in the corpus (the 22 exported
``catopt_torch.models`` blocks plus the bench law cases) and classifies
it from a *test*, not a lookup.

Properties (each decided by evaluating the op, not by membership):

* **view** — arity 1, shape-changing, value-preserving: the output's
  elements are a re-indexing (permutation / subset / broadcast) of the
  input's, so the op computes nothing.  Test: ``v(x)`` on a random
  tensor changes shape for some occurrence, and every output value is
  an input value (``set(v(x)) ⊆ set(x)``).
* **pointwise** — shape-preserving *and* commuting with views: the
  output has the input's (broadcast) shape and ``f(v(x), …) =
  v(f(x, …))`` for every view ``v``.  Test: the numeric oracle
  (``catopt_discovery.proposal._allclose``) on both sides of the naturality.  The
  probe views are shape-agnostic (a flatten, an unsqueeze, a gather, a
  slice, a transpose), so the test applies to any op shape.
* **reduction** — arity ≥ 1 and the op combines elements: the output
  has fewer elements than the input and is not value-preserving.
* **attribute-carrying** — the op carries non-empty attrs in the
  corpus.

Validation (the honesty check): the derived ``pointwise`` / ``view``
sets are compared against the hand-written tables — the pipeline's
``_POINTWISE`` / ``_VIEW_OPS`` and the shipped
``catopt_core.cost._VIEW_OPS``,
``catopt_orchestrator.morphisms.signature._VIEW_OPS`` /
``_POINTWISE_OPS`` and ``catopt_core.laws.layout._POINTWISE_*``.
Agreement on the shared ops and every disagreement are reported; a
disagreement is either a bug in the derivation or a stale entry in the
human table, and the tool says which it believes it is.  The test is
**not** tuned to the tables.

``derive_vocabulary()`` is the seam the pipeline reads
(``law_pipeline --vocab derived``).

Run::

    .venv/bin/python -m catopt_discovery.vocab
    .venv/bin/python -m catopt_discovery.vocab --json /tmp/catopt_discovery.vocab.json

CPU-only, bounded to seconds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from catopt_core.ir import Op, TensorType, Var
from catopt_core.typing import _shape_of

from catopt_discovery.census import _iter_subterms

# ``law_impact`` / ``law_shape_census`` own the corpus; reuse them so
# the vocabulary is derived over exactly the graphs the pipeline reads.
from catopt_discovery.impact import _bench_cases, model_cases
from catopt_discovery.proposal import _allclose

__all__ = [
    "OpClass",
    "Vocabulary",
    "classify",
    "corpus_ops",
    "derive_vocabulary",
    "main",
    "validate",
]

#: Numeric-comparison tolerance (fp64).
_TOL = 1e-6

#: Sentinel for an op that could not be evaluated on a probe.
_ERR = object()


# ---------------------------------------------------------------------------
#  Evaluation helpers
# ---------------------------------------------------------------------------


def _rand(shape: tuple) -> Any:
    """Return a positive random fp64 tensor of *shape*.

    Positive values keep ``pow`` / ``rsqrt`` / ``log`` from producing
    ``nan`` (which would make a genuine pointwise op look false).
    """
    return torch.rand(tuple(shape), dtype=torch.float64) + 0.5


def _leaf(shape: tuple, name: str = "x") -> Var:
    """Return a fresh ``Var`` of *shape*."""
    return Var(name, TensorType(tuple(shape)))


def _numel(shape: tuple) -> int:
    """Return the element count of *shape*."""
    n = 1
    for d in shape:
        n *= d
    return n


def _eval(term: Any, env: dict) -> Any:
    """Evaluate *term* through the torch oracle, or ``_ERR``."""
    from catopt_discovery.proposal import _eval_backend

    try:
        return _eval_backend().eval_term(term, env)
    except Exception:
        return _ERR


def _concrete(shape: Any) -> bool:
    """Return whether *shape* is a concrete all-int tuple.

    ``()`` (a scalar) is concrete: it is a real operand shape (the
    ``const`` exponent of ``pow``, the divisor of ``div``), and a
    pointwise op must still commute with a view of its tensor operand.

    Non-positive dims are NOT concrete: ``_shape_of`` uses ``-1`` as
    its unknown-dim sentinel (e.g. ``expand``'s keep-dim ``-1`` attr
    leaks into the inferred shape), and a ``-1`` fed to ``torch.rand``
    is a crash, not a probe.
    """
    return isinstance(shape, tuple) and all(
        isinstance(d, int) and d >= 0 for d in shape
    )


# ---------------------------------------------------------------------------
#  Corpus op mining
# ---------------------------------------------------------------------------


def corpus_ops(terms: list) -> tuple[dict, dict]:
    """Return ``(occurrences, arity)`` for every op in the corpus.

    ``occurrences[op]`` is the list of ``(attrs, arg_shapes)`` seen,
    deduped by node identity (a shared node counts once).  ``arity[op]``
    is the operand count.
    """
    occ: dict[str, list] = {}
    arity: dict[str, int] = {}
    for t in terms:
        for s in _iter_subterms(t):
            if not isinstance(s, Op):
                continue
            arity[s.op] = len(s.args)
            occ.setdefault(s.op, []).append(
                (dict(s.attrs), [_shape_of(a) for a in s.args])
            )
    return occ, arity


# ---------------------------------------------------------------------------
#  Property tests
# ---------------------------------------------------------------------------


def _value_preserving(x: Any, y: Any) -> bool:
    """Return whether every value of *y* is a value of *x*.

    A view (permutation / subset / broadcast) only re-arranges the
    input's values; a computing op produces values not in the input.
    """
    xs = set(torch.unique(x).tolist())
    ys = set(torch.unique(y).tolist())
    return ys.issubset(xs)


def _is_view(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* is a view, or ``None`` if untestable.

    A view is arity-1, value-preserving on every evaluable occurrence,
    and shape-changing on at least one.
    """
    if arity != 1:
        return False
    good = 0
    tested = 0
    changed = False
    for attrs, shapes in occ[op]:
        sh = shapes[0]
        if not _concrete(sh):
            continue
        v = _leaf(sh)
        x = _rand(sh)
        y = _eval(Op.make(op, v, **attrs), {v: x})
        if y is _ERR or not isinstance(y, torch.Tensor):
            continue
        tested += 1
        if _value_preserving(x, y):
            good += 1
        if tuple(y.shape) != tuple(sh):
            changed = True
    if tested == 0:
        return None
    return good == tested and changed


def _rep_occurrence(op: str, occ: dict) -> tuple | None:
    """Return one ``(attrs, shapes)`` with concrete operand shapes."""
    for attrs, shapes in occ[op]:
        if shapes and all(_concrete(s) for s in shapes):
            return attrs, shapes
    return None


def _battery(shape: tuple) -> list[tuple]:
    """Return shape-agnostic view probes valid on *shape*.

    A flatten (reinterprets the whole element order — kills a reduction
    over any single axis), an unsqueeze, a single-element gather, a
    contiguous subset, and (rank ≥ 2) a first/last transpose.  A
    pointwise op commutes with every one; a view op or a reduction does
    not.
    """
    out = [
        ("reshape", {"shape": (_numel(shape),)}),
        ("unsqueeze", {"dim": 0}),
        ("select", {"dim": 0, "index": 0}),
        ("slice", {"dim": 0, "start": 0, "end": 1}),
    ]
    if len(shape) >= 2:
        out.append(("transpose", {"dim0": 0, "dim1": -1}))
    return out


def _op_output(op: str, attrs: dict, shapes: list) -> Any:
    """Return the output tensor of ``op(*args, **attrs)``, or ``None``."""
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    y = _eval(Op.make(op, *vs, **attrs), env)
    if y is _ERR or not isinstance(y, torch.Tensor):
        return None
    return y


def _commutes(
    op: str, attrs: dict, shapes: list, probe: tuple
) -> bool | None:
    """Return whether *op* commutes with *probe* (a view instance).

    ``f(v(x0), …) == v(f(x0, …))`` on random tensors; the view is
    applied to the full-size operands only (a scalar/broadcast operand
    is left alone).  ``None`` when the probe cannot be composed on
    either side (shape-incompatible), which is not evidence either way;
    a one-sided failure is ``False`` (a pointwise op never fails to
    compose).
    """
    vop, vattrs = probe
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    big = max(_numel(s) for s in shapes)

    def arg(v: Any, s: tuple) -> Any:
        if _numel(s) == big:
            return Op.make(vop, v, **vattrs)
        return v

    lhs = Op.make(
        op,
        *[arg(v, s) for v, s in zip(vs, shapes, strict=True)],
        **attrs,
    )
    rhs = Op.make(vop, Op.make(op, *vs, **attrs), **vattrs)
    a = _eval(lhs, env)
    b = _eval(rhs, env)
    if a is _ERR and b is _ERR:
        return None
    if a is _ERR or b is _ERR:
        return False
    return _allclose(a, b, _TOL)


def _is_pointwise(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* is pointwise, or ``None`` if untestable.

    Pointwise means: shape-preserving, and commuting with every view
    probe.  Arity > 2 is left undecided (the generator only consumes
    unary / binary ops, and a scalar or mask operand breaks the probe's
    composition).
    """
    if arity not in (1, 2):
        return None
    rep = _rep_occurrence(op, occ)
    if rep is None:
        return None
    attrs, shapes = rep
    out = _op_output(op, attrs, shapes)
    if out is None:
        return None
    big = max(shapes, key=_numel)
    if tuple(out.shape) != tuple(big):
        return False
    results = [_commutes(op, attrs, shapes, p) for p in _battery(big)]
    definite = [r for r in results if r is not None]
    if any(r is False for r in definite):
        return False
    if any(r is True for r in definite):
        return True
    return None


def _is_reduction(op: str, arity: int, occ: dict) -> bool | None:
    """Return whether *op* combines elements (fewer, non-preserved)."""
    rep = _rep_occurrence(op, occ)
    if rep is None:
        return None
    attrs, shapes = rep
    vs = [_leaf(s, f"x{i}") for i, s in enumerate(shapes)]
    env = {v: _rand(s) for v, s in zip(vs, shapes, strict=True)}
    x = env[vs[0]]
    y = _eval(Op.make(op, *vs, **attrs), env)
    if y is _ERR or not isinstance(y, torch.Tensor):
        return None
    if y.numel() >= x.numel():
        return False
    return not _value_preserving(x, y)


# ---------------------------------------------------------------------------
#  Classification record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpClass:
    """One corpus op with its property-derived class."""

    op: str
    arity: int
    pointwise: bool | None
    view: bool | None
    reduction: bool | None
    attribute_carrying: bool
    evidence: str = ""


@dataclass
class Vocabulary:
    """The property-derived op vocabulary over the corpus."""

    classes: tuple[OpClass, ...] = ()
    #: Diagnostics: the shape-agnostic probes the pointwise test used.
    probes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pointwise(self) -> tuple[str, ...]:
        """Arity-2 pointwise ops (the generator's ``_POINTWISE``)."""
        return tuple(
            c.op
            for c in self.classes
            if c.pointwise is True and c.arity == 2
        )

    @property
    def unary_pointwise(self) -> tuple[str, ...]:
        """Arity-1 pointwise ops."""
        return tuple(
            c.op
            for c in self.classes
            if c.pointwise is True and c.arity == 1
        )

    @property
    def views(self) -> tuple[str, ...]:
        """View ops (the generator's ``_VIEW_OPS``)."""
        return tuple(c.op for c in self.classes if c.view is True)


def classify(terms: list) -> Vocabulary:
    """Classify every op occurring in *terms* by property."""
    torch.manual_seed(0)
    occ, arity = corpus_ops(terms)
    classes: list[OpClass] = []
    for op in sorted(arity):
        n = arity[op]
        classes.append(
            OpClass(
                op=op,
                arity=n,
                pointwise=_is_pointwise(op, n, occ),
                view=_is_view(op, n, occ),
                reduction=_is_reduction(op, n, occ),
                attribute_carrying=any(a for a, _ in occ[op]),
            )
        )
    labels = tuple(
        f"{op}[{','.join(f'{k}={v}' for k, v in sorted(a.items()))}]"
        for op, a in _battery((2, 3, 4))
    )
    return Vocabulary(classes=tuple(classes), probes=labels)


def derive_vocabulary() -> Vocabulary:
    """Derive the vocabulary over the full corpus (the pipeline seam)."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    terms = [c.term for c in [*bench, *models]]
    return classify(terms)


# ---------------------------------------------------------------------------
#  Validation against the hand-written tables
# ---------------------------------------------------------------------------


def _hand_tables() -> list[tuple[str, str, frozenset]]:
    """Return ``(name, kind, ops)`` for every hand-written op table.

    ``kind`` is ``"view"``, ``"pointwise"`` (binary + unary),
    ``"binary"`` or ``"unary"`` — it selects which derived set the table
    is compared against.
    """
    from catopt_core.cost import _VIEW_OPS as cost_views
    from catopt_core.laws import layout
    from catopt_orchestrator.morphisms import signature

    from catopt_discovery import pipeline as lpl

    return [
        ("pipeline._POINTWISE", "binary", frozenset(lpl._POINTWISE)),
        ("pipeline._VIEW_OPS", "view", frozenset(lpl._VIEW_OPS)),
        ("cost._VIEW_OPS", "view", frozenset(cost_views)),
        (
            "signature._VIEW_OPS",
            "view",
            frozenset(signature._VIEW_OPS),
        ),
        (
            "signature._POINTWISE_OPS",
            "pointwise",
            frozenset(signature._POINTWISE_OPS),
        ),
        (
            "layout._POINTWISE_BINARY",
            "binary",
            frozenset(layout._POINTWISE_BINARY),
        ),
        (
            "layout._POINTWISE_UNARY",
            "unary",
            frozenset(layout._POINTWISE_UNARY),
        ),
    ]


def _derived_set(vocab: Vocabulary, kind: str) -> frozenset:
    """Return the derived op set matching a table *kind*."""
    if kind == "view":
        return frozenset(vocab.views)
    if kind == "binary":
        return frozenset(vocab.pointwise)
    if kind == "unary":
        return frozenset(vocab.unary_pointwise)
    return frozenset(vocab.pointwise) | frozenset(vocab.unary_pointwise)


def _compare(
    derived: frozenset, hand: frozenset, corpus: frozenset
) -> dict:
    """Compare a derived set with a hand table over the corpus ops."""
    d = derived & corpus
    h = hand & corpus
    return {
        "derived": sorted(d),
        "hand": sorted(h),
        "agree": sorted(d & h),
        "derived_only": sorted(d - h),
        "hand_only": sorted(h - d),
    }


def validate(vocab: Vocabulary) -> dict:
    """Compare the derived vocabulary against every hand table.

    Each derived-only op is tagged ``corroborated`` when it appears in
    some *other* hand table of the same kind — the machine-checkable
    signal that a "disagreement" is the hand table being incomplete
    rather than the derivation being wrong.
    """
    corpus = frozenset(c.op for c in vocab.classes)
    tables = _hand_tables()
    rows: list[dict] = []
    for name, kind, hand in tables:
        row = {"table": name, "kind": kind}
        row.update(_compare(_derived_set(vocab, kind), hand, corpus))
        corroborated = set()
        for other, other_kind, other_ops in tables:
            if other == name or other_kind != kind:
                continue
            corroborated |= other_ops
        row["corroborated"] = sorted(
            op for op in row["derived_only"] if op in corroborated
        )
        rows.append(row)
    return {"corpus_ops": sorted(corpus), "rows": rows}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _yn(v: bool | None) -> str:
    """Render a tri-state verdict."""
    return {True: "yes", False: "no", None: "?"}[v]


def _class_table(vocab: Vocabulary) -> str:
    """Render the per-op classification table."""
    head = (
        f"{'op':<14} {'ar':>2} {'point':>5} {'view':>5} "
        f"{'reduc':>5} {'attr':>4}"
    )
    lines = [head, "-" * len(head)]
    for c in vocab.classes:
        lines.append(
            f"{c.op:<14} {c.arity:>2} {_yn(c.pointwise):>5} "
            f"{_yn(c.view):>5} {_yn(c.reduction):>5} "
            f"{'yes' if c.attribute_carrying else '-':>4}"
        )
    return "\n".join(lines)


def _validation_table(val: dict) -> str:
    """Render the derived-vs-hand agreement table."""
    head = (
        f"{'hand table':<28} {'kind':<9} {'agree':>5} "
        f"{'derived-only':>12} {'hand-only':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in val["rows"]:
        lines.append(
            f"{r['table']:<28} {r['kind']:<9} {len(r['agree']):>5} "
            f"{len(r['derived_only']):>12} {len(r['hand_only']):>9}"
        )
    return "\n".join(lines)


def _print_report(vocab: Vocabulary, val: dict) -> None:
    """Print the full human-readable vocabulary report."""
    print(  # stdout-compat
        "== law_vocab — the op vocabulary by property =="
    )
    print(  # stdout-compat
        f"   corpus ops: {len(vocab.classes)}; "
        f"pointwise(binary)={len(vocab.pointwise)}, "
        f"pointwise(unary)={len(vocab.unary_pointwise)}, "
        f"views={len(vocab.views)}"
    )
    print()  # stdout-compat
    print("-- derived classification --")  # stdout-compat
    print(_class_table(vocab))  # stdout-compat
    print()  # stdout-compat
    print("-- derived sets --")  # stdout-compat
    print(  # stdout-compat
        f"   pointwise (arity 2): {', '.join(vocab.pointwise)}"
    )
    print(  # stdout-compat
        f"   pointwise (arity 1): {', '.join(vocab.unary_pointwise)}"
    )
    print(  # stdout-compat
        f"   views:               {', '.join(vocab.views)}"
    )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- validation: derived vs hand tables (over corpus ops) --"
    )
    print(_validation_table(val))  # stdout-compat
    for r in val["rows"]:
        if r["derived_only"]:
            tag = (
                " (corroborated by another hand table)"
                if r["corroborated"]
                else ""
            )
            print(  # stdout-compat
                f"   {r['table']}: derived-only "
                f"{', '.join(r['derived_only'])}{tag}"
            )
        if r["hand_only"]:
            print(  # stdout-compat
                f"   {r['table']}: hand-only "
                f"{', '.join(r['hand_only'])}"
            )
    print()  # stdout-compat


def _dump_json(path: str, vocab: Vocabulary, val: dict) -> None:
    """Write the machine-readable vocabulary result."""
    payload = {
        "classes": [
            {
                "op": c.op,
                "arity": c.arity,
                "pointwise": c.pointwise,
                "view": c.view,
                "reduction": c.reduction,
                "attribute_carrying": c.attribute_carrying,
            }
            for c in vocab.classes
        ],
        "pointwise": list(vocab.pointwise),
        "unary_pointwise": list(vocab.unary_pointwise),
        "views": list(vocab.views),
        "probes": list(vocab.probes),
        "validation": val,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Derive the vocabulary, validate it, and print (or dump) it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    vocab = derive_vocabulary()
    val = validate(vocab)
    _print_report(vocab, val)
    if args.json:
        _dump_json(args.json, vocab, val)
        print(f"wrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
