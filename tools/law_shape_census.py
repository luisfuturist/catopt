"""Shape census — which term shapes real models actually contain.

The proposal retro (``project/retros/law-proposal.md``) found laws by
enumerating an *algebraic* grammar; the impact retro
(``project/retros/law-impact.md``) then measured them on real graphs
and got **zero firings** — the ops were present, the *shapes* were
not.  The lesson: law proposal optimised algebraic novelty, not
applicability.  Before proposing anything shape-aware, measure the
distribution it must hit.

This tool mines the real corpus — every ``catopt_torch.models`` block
exported to IR, plus every ``bench`` law case — and reports the
**shape frequencies** of the subterms actually present:

* **op-tuple census** — for each op node, ``(op, child-op tuple)``;
  how often it occurs and in how many distinct terms.
* **shape census** — each distinct subterm abstracted to a shape
  (leaves become numbered placeholders, so a *repeated* leaf stays
  repeated); how often the shape occurs and in how many terms.
* **sharing census** — for the common binary shapes, whether the two
  operands share a subterm (the precondition a factoring/absorption
  law needs), and how many distinct leaves each shape touches.

The census is the deliverable that outlives the experiment: it is
what a shape-aware proposer must read, and it is the honest statement
of what the corpus contains.

Run::

    .venv/bin/python tools/law_shape_census.py
    .venv/bin/python tools/law_shape_census.py --json /tmp/census.json
    .venv/bin/python tools/law_shape_census.py --top 40

CPU-only, bounded (seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.ir import Const, Op

# ``law_impact`` owns the corpus builders (bench + real models); reuse
# them so the census reads exactly the graphs the impact tool probes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_impact import _bench_cases, model_cases

__all__ = [
    "CorpusTerm",
    "corpus",
    "main",
    "op_tuple_census",
    "shape_census",
    "sharing_census",
]

#: Default number of rows in each ranked table.
_DEFAULT_TOP = 30


@dataclass(frozen=True)
class CorpusTerm:
    """One exported graph, tagged with where it came from."""

    source: str  # "bench" | "model"
    name: str
    term: Any


# ---------------------------------------------------------------------------
#  Corpus
# ---------------------------------------------------------------------------


def corpus() -> list[CorpusTerm]:
    """Return every real term: bench law cases + exported models.

    Reuses ``law_impact``'s builders so the census and the impact
    probe read byte-identical graphs.  A builder that raised is
    already reported by ``law_impact``; here it is simply absent.
    """
    bench, _be = _bench_cases()
    models, _me = model_cases()
    out: list[CorpusTerm] = []
    for c in bench:
        out.append(CorpusTerm("bench", c.name, c.term))
    for c in models:
        out.append(CorpusTerm("model", c.name, c.term))
    return out


def _iter_subterms(term: Any) -> list[Any]:
    """Return every distinct subterm of *term* (deduped by identity)."""
    seen: set[int] = set()
    out: list[Any] = []
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return out


def _op_of(term: Any) -> str:
    """Return the op name of *term*, or a leaf/const marker."""
    if isinstance(term, Op):
        return term.op
    if isinstance(term, Const):
        return "const"
    return "·"


# ---------------------------------------------------------------------------
#  Shape abstraction
# ---------------------------------------------------------------------------


def _attr_key(attrs: dict) -> tuple:
    """Return a hashable, order-independent key for a node's attrs."""
    out = []
    for k, v in attrs.items():
        try:
            hash(v)
            out.append((k, v))
        except TypeError:
            out.append((k, repr(v)))
    return tuple(sorted(out))


def shape_key(term: Any, memo: dict | None = None) -> Any:
    """Abstract *term* to a leaf-numbered shape key.

    ``Op`` nodes keep op + attrs + the shape of each child; a ``Const``
    keeps its literal; every other leaf (``Var`` / ``Param``) becomes a
    numbered placeholder.  The numbering is by *object identity*, so a
    leaf that appears twice in one term stays the *same* number — the
    census sees the sharing a factoring law needs.
    """
    memo = {} if memo is None else memo
    if isinstance(term, Op):
        return (
            term.op,
            _attr_key(term.attrs),
            tuple(shape_key(a, memo) for a in term.args),
        )
    if isinstance(term, Const):
        return ("const", term.value)
    return ("leaf", memo.setdefault(id(term), len(memo)))


def shape_repr(key: Any) -> str:
    """Render a shape key as ``op(child, child)`` notation."""
    if not isinstance(key, tuple):
        return repr(key)
    head = key[0]
    if head == "leaf":
        return _letter(key[1])
    if head == "const":
        return repr(key[1])
    _op, attrs, kids = key
    inner = ", ".join(shape_repr(k) for k in kids)
    if attrs:
        a = ",".join(f"{k}={v}" for k, v in attrs)
        return f"{_op}[{a}]({inner})"
    return f"{_op}({inner})"


def _letter(i: int) -> str:
    """Return ``a, b, …`` for placeholder index *i*."""
    return chr(ord("a") + i) if i < 26 else f"v{i}"


def _leaf_ids(key: Any, out: list | None = None) -> list:
    """Return the placeholder indices of every leaf in a shape key."""
    out = [] if out is None else out
    if not isinstance(key, tuple):
        return out
    if key[0] == "leaf":
        out.append(key[1])
    elif key[0] not in ("const",) and len(key) == 3:
        for k in key[2]:
            _leaf_ids(k, out)
    return out


def _has_shared_leaf(key: Any) -> bool:
    """Return whether a placeholder repeats — the shape shares a subterm."""
    ids = _leaf_ids(key)
    return len(ids) != len(set(ids))


# ---------------------------------------------------------------------------
#  Census 1 — op-tuple frequencies
# ---------------------------------------------------------------------------


def op_tuple_census(
    terms: list[CorpusTerm],
) -> tuple[Counter, dict]:
    """Count ``(op, child-op tuple)`` nodes and the terms they span."""
    counts: Counter = Counter()
    terms_of: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op):
                continue
            key = (s.op, tuple(_op_of(a) for a in s.args))
            counts[key] += 1
            terms_of.setdefault(key, set()).add((ct.source, ct.name))
    return counts, terms_of


# ---------------------------------------------------------------------------
#  Census 2 — shape frequencies (leaf-abstracted)
# ---------------------------------------------------------------------------


def shape_census(
    terms: list[CorpusTerm],
) -> tuple[Counter, dict]:
    """Count distinct subterm shapes and the terms they span."""
    counts: Counter = Counter()
    terms_of: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op):
                continue
            key = shape_key(s, {})
            counts[key] += 1
            terms_of.setdefault(key, set()).add((ct.source, ct.name))
    return counts, terms_of


# ---------------------------------------------------------------------------
#  Census 3 — sharing structure of binary shapes
# ---------------------------------------------------------------------------


def sharing_census(terms: list[CorpusTerm]) -> dict:
    """Report the sharing structure of the corpus's binary op nodes.

    For every node whose op is a two-argument algebraic op, record
    whether its operands are structurally *equal* (the precondition a
    factoring / absorption law needs) and how many distinct leaves the
    node touches.  Aggregated by ``(op, child-op tuple)``.
    """
    binary = {"add", "sub", "mul", "div", "matmul"}
    agg: dict = {}
    for ct in terms:
        for s in _iter_subterms(ct.term):
            if not isinstance(s, Op) or s.op not in binary:
                continue
            if len(s.args) != 2:
                continue
            key = (s.op, tuple(_op_of(a) for a in s.args))
            rec = agg.setdefault(
                key,
                {"sites": 0, "shared_child": 0, "shared_leaf": 0},
            )
            rec["sites"] += 1
            a = shape_key(s.args[0], {})
            b = shape_key(s.args[1], {})
            if a == b:
                rec["shared_child"] += 1
            if _has_shared_leaf(shape_key(s, {})):
                rec["shared_leaf"] += 1
    return agg


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _op_tuple_table(counts: Counter, terms_of: dict, top: int) -> str:
    """Render the op-tuple frequency table."""
    head = f"{'count':>6} {'terms':>6}  op-tuple"
    lines = [head, "-" * len(head)]
    for key, n in counts.most_common(top):
        kids = ", ".join(key[1])
        lines.append(
            f"{n:>6} {len(terms_of[key]):>6}  {key[0]}({kids})"
        )
    return "\n".join(lines)


def _shape_table(counts: Counter, terms_of: dict, top: int) -> str:
    """Render the shape frequency table (shared shapes flagged)."""
    head = f"{'count':>6} {'terms':>6} {'share':>5}  shape"
    lines = [head, "-" * len(head)]
    for key, n in counts.most_common(top):
        shared = "yes" if _has_shared_leaf(key) else "-"
        lines.append(
            f"{n:>6} {len(terms_of[key]):>6} {shared:>5}  "
            f"{shape_repr(key)}"
        )
    return "\n".join(lines)


def _sharing_table(agg: dict) -> str:
    """Render the binary-shape sharing table."""
    head = (
        f"{'op-tuple':<24} {'sites':>6} {'same-kids':>9} "
        f"{'shared-leaf':>11}"
    )
    lines = [head, "-" * len(head)]
    for key, rec in sorted(agg.items(), key=lambda kv: -kv[1]["sites"]):
        kids = ", ".join(key[1])
        name = f"{key[0]}({kids})"
        lines.append(
            f"{name:<24} {rec['sites']:>6} "
            f"{rec['shared_child']:>9} {rec['shared_leaf']:>11}"
        )
    return "\n".join(lines)


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable census."""
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def run_census(top: int = _DEFAULT_TOP) -> dict:
    """Compute every census table; return a machine-readable result."""
    terms = corpus()
    op_counts, op_terms = op_tuple_census(terms)
    sh_counts, sh_terms = shape_census(terms)
    agg = sharing_census(terms)
    return {
        "n_terms": len(terms),
        "n_bench": sum(1 for t in terms if t.source == "bench"),
        "n_models": sum(1 for t in terms if t.source == "model"),
        "n_op_nodes": sum(op_counts.values()),
        "n_shapes": len(sh_counts),
        "op_tuples": [
            {
                "op": k[0],
                "children": list(k[1]),
                "count": n,
                "terms": len(op_terms[k]),
            }
            for k, n in op_counts.most_common(top)
        ],
        "shapes": [
            {
                "shape": shape_repr(k),
                "count": n,
                "terms": len(sh_terms[k]),
                "shared": _has_shared_leaf(k),
            }
            for k, n in sh_counts.most_common(top)
        ],
        "sharing": [
            {
                "op": k[0],
                "children": list(k[1]),
                **rec,
            }
            for k, rec in sorted(
                agg.items(), key=lambda kv: -kv[1]["sites"]
            )
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the census and print (or dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument("--top", type=int, default=_DEFAULT_TOP)
    args = parser.parse_args(argv)

    terms = corpus()
    op_counts, op_terms = op_tuple_census(terms)
    sh_counts, sh_terms = shape_census(terms)
    agg = sharing_census(terms)

    print(
        "== law_shape_census — what shapes do real graphs contain? =="
    )
    print(
        f"   corpus: {len(terms)} terms "
        f"({sum(1 for t in terms if t.source == 'bench')} bench, "
        f"{sum(1 for t in terms if t.source == 'model')} models)"
    )
    print(
        f"   {sum(op_counts.values())} op nodes, "
        f"{len(op_counts)} distinct op-tuples, "
        f"{len(sh_counts)} distinct shapes"
    )
    print()
    print(f"-- op-tuple census (top {args.top}) --")
    print(_op_tuple_table(op_counts, op_terms, args.top))
    print()
    print(f"-- shape census (top {args.top}; share = repeated leaf) --")
    print(_shape_table(sh_counts, sh_terms, args.top))
    print()
    print("-- sharing census (binary op nodes) --")
    print(_sharing_table(agg))
    print()

    if args.json:
        _dump_json(args.json, run_census(args.top))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
