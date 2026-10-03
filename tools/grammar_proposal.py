"""Schema-level search: proposing the grammar itself.

``tools/law_proposal.py`` closed the *propose* half of "the AI invents
laws" by *enumerating instances* of a hand-written schema grammar: 23
algebraic identities a human wrote down, of which 9 distinct are
genuinely-new and useful.  Its retro
(``project/retros/law-proposal.md``) names the honest caveat -- "the
schema grammar is human-authored; the machine enumerates, verifies and
prices what the grammar generates; the creativity is still the
grammar."

This tool attacks that caveat.  It treats a *schema* as data -- a shape
over the op algebra, with shared slots standing for repeated subterms
-- and searches the *shape space*: it mutates and recombines shapes
(swap operand positions, swap an op, change arity, change which
subterms are shared, vary the RHS), instantiates each shape into a
concrete law, and scores it with the same oracles ``law_proposal.py``
uses -- ``law_verifier.verify_law`` derivability, the numeric truth
oracle, and the cost-delta usefulness test.

Two questions are decided:

* does the search find a useful schema the *hand-written* grammar does
  not generate?
* does search beat enumeration -- useful laws per ``verify_law`` call?

Run::

    .venv/bin/python tools/grammar_proposal.py
    .venv/bin/python tools/grammar_proposal.py --json /tmp/gp.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.cost import flops_cost
from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, op_repr
from catopt_core.laws import ALL_RULES
from catopt_core.meta import canonicalize

# ``law_proposal`` is a sibling script; running this file puts
# ``tools/`` on ``sys.path``, so the import resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import law_proposal as lp

__all__ = [
    "Shape",
    "ShapeResult",
    "main",
    "search",
    "shape_of_term",
]

#: Binary ops the search may place at a two-child node.
_BINARY_OPS = ("add", "mul", "sub", "div", "pow", "matmul")

#: Unary ops the search may wrap a leaf in (and swap a unary node to).
_UNARY_OPS = (
    "neg",
    "exp",
    "square",
    "sqrt",
    "rsqrt",
    "sigmoid",
    "silu",
    "tanh",
)

#: Literal constants the search may substitute for a leaf.  Integer
#: spelling, matching the hand-written grammar (``Const(0)``, not
#: ``Const(0.0)``) -- the e-graph keys leaves by ``repr``, so a float
#: spelling would make a law miss its own grammar rule and look novel.
_LITERALS = (0, 1, 2)

#: Recombine (cross) LHS/RHS only for a frontier this small.
_CROSS_FRONTIER = 64

#: Node / iteration caps for the search's library-derivability probe
#: (see :func:`_evaluate`).
_SEARCH_MAX_NODES = 4000
_SEARCH_MAX_ITERATIONS = 8


# ---------------------------------------------------------------------------
#  Schema as data -- a shape over the op algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """A schema shape: an op over child shapes, or a leaf.

    A leaf is either a *slot* (``op is None``, ``slot`` an int) -- a
    shared metavariable standing for a repeated subterm, so two leaves
    with the same slot index denote the *same* concrete subterm when
    instantiated -- or a literal constant (``op is None``, ``const``
    set).  Op nodes carry the op name and their children.
    """

    op: str | None = None
    children: tuple[Shape, ...] = ()
    slot: int | None = None
    const: int | float | None = None


def _slot(index: int) -> Shape:
    """Return a slot leaf."""
    return Shape(slot=index)


def _lit(value: int | float) -> Shape:
    """Return a literal leaf."""
    return Shape(const=value)


def _node(op: str, *children: Shape) -> Shape:
    """Return an op node."""
    return Shape(op=op, children=tuple(children))


def _slots(shape: Shape, out: set[int] | None = None) -> set[int]:
    """Return every slot index used in *shape*."""
    out = set() if out is None else out
    if shape.op is None:
        if shape.slot is not None:
            out.add(shape.slot)
    else:
        for child in shape.children:
            _slots(child, out)
    return out


def _fresh_slot(shape: Shape) -> int:
    """Return an unused slot index for *shape*."""
    used = _slots(shape)
    return max(used) + 1 if used else 0


def _count_slot(shape: Shape, index: int) -> int:
    """Return how many leaves in *shape* carry slot *index*."""
    if shape.op is None:
        return 1 if shape.slot == index else 0
    return sum(_count_slot(c, index) for c in shape.children)


def _positions(shape: Shape, path: tuple = ()):
    """Yield ``(path, shape)`` pre-order (``path`` a child-index tuple)."""
    yield path, shape
    for i, child in enumerate(shape.children):
        yield from _positions(child, (*path, i))


def _replace(shape: Shape, path: tuple, new: Shape) -> Shape:
    """Return *shape* with the node at *path* replaced by *new*."""
    if not path:
        return new
    index = path[0]
    children = list(shape.children)
    children[index] = _replace(children[index], path[1:], new)
    return Shape(op=shape.op, children=tuple(children))


def _relabel(shape: Shape, old: int, new: int) -> Shape:
    """Return *shape* with slot *old* renamed to *new*."""
    if shape.op is None:
        return Shape(slot=new) if shape.slot == old else shape
    return Shape(
        op=shape.op,
        children=tuple(_relabel(c, old, new) for c in shape.children),
    )


def _arity_ops(arity: int) -> tuple[str, ...]:
    """Return the op vocabulary for a node of the given *arity*."""
    if arity == 1:
        return _UNARY_OPS
    if arity == 2:
        return _BINARY_OPS
    return ()


def _canon(shape: Shape, mapping: dict[int, int]) -> Any:
    """Canonical key of *shape* with slots renamed by first use."""
    if shape.op is None:
        if shape.slot is not None:
            index = mapping.setdefault(shape.slot, len(mapping))
            return ("s", index)
        return ("c", shape.const)
    return (
        shape.op,
        tuple(_canon(c, mapping) for c in shape.children),
    )


def _schema_key(lhs: Shape, rhs: Shape) -> tuple:
    """Alpha-normal ``(lhs, rhs)`` key of a schema (shared slots)."""
    mapping: dict[int, int] = {}
    return (_canon(lhs, mapping), _canon(rhs, mapping))


def shape_of_term(
    term: Any, slots: dict[str, int] | None = None
) -> Shape:
    """Abstract a concrete term to a shape.

    Distinct leaves (``Var`` / ``Param``, keyed by ``repr``) become
    distinct slots; ``Const`` stays literal.  Pass one shared *slots*
    dict across a schema's two sides so a leaf used on both sides maps
    to the same slot.
    """
    slots = {} if slots is None else slots
    if isinstance(term, Op):
        return Shape(
            op=term.op,
            children=tuple(shape_of_term(a, slots) for a in term.args),
        )
    if isinstance(term, Const):
        return Shape(const=term.value)
    key = repr(term)
    if key not in slots:
        slots[key] = len(slots)
    return Shape(slot=slots[key])


def _instantiate(shape: Shape, env: dict[int, Any]) -> Any:
    """Instantiate *shape* into a concrete term under *env*."""
    if shape.op is None:
        if shape.slot is not None:
            return env[shape.slot]
        return Const(shape.const)
    return Op.make(
        shape.op, *(_instantiate(c, env) for c in shape.children)
    )


def _to_pattern(shape: Shape) -> Any:
    """Render *shape* as a rule pattern (slots -> metavariables)."""
    if shape.op is None:
        if shape.slot is not None:
            return f"s{shape.slot}"
        return Const(shape.const)
    return Op.make(shape.op, *(_to_pattern(c) for c in shape.children))


def _leaf_order(term: Any) -> list:
    """Return the distinct non-``Const`` leaves in first-use order."""
    order: list = []
    seen: set = set()

    def go(t: Any) -> None:
        if isinstance(t, Op):
            for a in t.args:
                go(a)
        elif not isinstance(t, Const):
            key = repr(t)
            if key not in seen:
                seen.add(key)
                order.append(t)

    go(term)
    return order


def _renamed(term: Any, mapping: dict) -> Any:
    """Return *term* with each leaf replaced per *mapping*."""
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_renamed(a, mapping) for a in term.args),
            **dict(term.attrs),
        )
    if isinstance(term, Const):
        return term
    return mapping[repr(term)]


def _alpha_coherent_key(term: Any) -> Any:
    """Alpha- and coherence-invariant key of *term*.

    ``canonicalize`` collapses the library's coherent laws
    (comm/assoc/id/double_neg), but its child ordering is
    name-dependent, so it is not alpha-invariant on its own.
    Minimising the resulting key over *every* renaming of the term's
    leaves yields a key invariant under both alpha-renaming and
    coherence -- the equivalence the grammar's laws should be read up
    to.  Two terms share a key iff one is a coherent reordering of the
    other.  The leaf count is tiny (<= 4), so the permutation sweep is
    cheap.
    """
    order = _leaf_order(term)
    if not order:
        return lp._canon(canonicalize(term), {})
    best_text: str | None = None
    best_key: Any = None
    for perm in itertools.permutations(range(len(order))):
        mapping = {
            repr(order[i]): lp._v(f"z{perm[i]}", lp._D, lp._D)
            for i in range(len(order))
        }
        key = lp._canon(canonicalize(_renamed(term, mapping)), {})
        text = repr(key)
        if best_text is None or text < best_text:
            best_text, best_key = text, key
    return best_key


def _instantiate_schema(lhs: Shape, rhs: Shape) -> tuple[Any, Any]:
    """Instantiate a schema on shape-correct 4x4 leaves."""
    used = sorted(_slots(lhs) | _slots(rhs))
    env = {s: lp._v(f"x{s}", lp._D, lp._D) for s in used}
    return _instantiate(lhs, env), _instantiate(rhs, env)


# ---------------------------------------------------------------------------
#  Shape mutations -- the search operators
# ---------------------------------------------------------------------------


def _mutations(shape: Shape):
    """Yield the one-step shape neighbours of *shape* (deterministic).

    Operators, applied at every position: swap a binary node's operands;
    swap a node's op for another of the same arity; wrap a leaf in a
    unary op; unwrap a unary node; merge two slots (more sharing); split
    one occurrence of a repeated slot (less sharing); substitute a
    literal for a leaf; promote a literal to a fresh slot.
    """
    for path, node in _positions(shape):
        if node.op is None:
            yield from _leaf_mutations(shape, path, node)
            continue
        if len(node.children) == 2:
            swap = _node(node.op, node.children[1], node.children[0])
            yield _replace(shape, path, swap)
        for op in _arity_ops(len(node.children)):
            if op != node.op:
                new = _node(op, *node.children)
                yield _replace(shape, path, new)
        if len(node.children) == 1:
            yield _replace(shape, path, node.children[0])
    for path, node in _positions(shape):
        if (
            node.op is None
            and node.slot is not None
            and _count_slot(shape, node.slot) > 1
        ):
            split = _slot(_fresh_slot(shape))
            yield _replace(shape, path, split)
    used = sorted(_slots(shape))
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            yield _relabel(shape, used[j], used[i])


def _leaf_mutations(shape: Shape, path: tuple, node: Shape):
    """Yield the leaf-position mutations of *node* at *path*."""
    if node.slot is not None:
        for op in _UNARY_OPS:
            yield _replace(shape, path, _node(op, node))
        for value in _LITERALS:
            yield _replace(shape, path, _lit(value))
        return
    for value in _LITERALS:
        if value != node.const:
            yield _replace(shape, path, _lit(value))
    yield _replace(shape, path, _slot(_fresh_slot(shape)))


def _wellformed(lhs: Shape, rhs: Shape) -> bool:
    """Return whether the RHS introduces no slot absent from the LHS.

    A rewrite may not mint a fresh metavariable on its RHS (the
    ``meta._synthesizable`` condition).  A shape that violates it is
    not a law at all -- and, because the e-graph then merges the free
    RHS slot with anything, saturating it is pathological -- so the
    search never emits one.
    """
    return _slots(rhs) <= _slots(lhs)


def _schema_neighbors(lhs: Shape, rhs: Shape):
    """Yield the one-step schema neighbours (mutate one side)."""
    for new_lhs in _mutations(lhs):
        if _wellformed(new_lhs, rhs):
            yield new_lhs, rhs
    for new_rhs in _mutations(rhs):
        if _wellformed(lhs, new_rhs):
            yield lhs, new_rhs


def _crosses(schemas: list, cap: int):
    """Yield ``(a.lhs, b.rhs)`` recombinations across *schemas*."""
    n = 0
    for lhs, _ in schemas:
        for _, rhs in schemas:
            if not _wellformed(lhs, rhs):
                continue
            yield lhs, rhs
            n += 1
            if n >= cap:
                return


# ---------------------------------------------------------------------------
#  Search
# ---------------------------------------------------------------------------


def search(
    seeds: list,
    *,
    depth: int,
    level_cap: int,
    total_cap: int,
) -> list:
    """Bounded BFS over shape neighbours from *seeds*.

    Returns every ``(lhs, rhs)`` schema considered: the seeds first,
    then up to *level_cap* fresh neighbours per level for *depth*
    levels (or until *total_cap* schemas are reached).  Dedup is by
    alpha-normal schema key, so the visit order is deterministic.
    """
    evaluated = list(seeds)
    visited = {_schema_key(*s) for s in seeds}
    frontier = list(seeds)
    for _ in range(depth):
        nxt: list = []
        seen = set(visited)
        for lhs, rhs in frontier:
            for cand in _schema_neighbors(lhs, rhs):
                key = _schema_key(*cand)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(cand)
                if len(nxt) >= level_cap:
                    break
            if len(nxt) >= level_cap:
                break
        if len(frontier) <= _CROSS_FRONTIER:
            for cand in _crosses(frontier, level_cap):
                key = _schema_key(*cand)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(cand)
                if len(nxt) >= level_cap:
                    break
        visited = seen
        evaluated.extend(nxt)
        frontier = nxt
        if len(evaluated) >= total_cap:
            break
    return evaluated


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeResult:
    """One evaluated schema plus its verdicts."""

    lhs_repr: str
    rhs_repr: str
    law: str
    outcome: lp.Outcome
    new_to_grammar: bool
    closure_stop: str = ""


def _human_schemas() -> list:
    """Return the hand-written grammar abstracted to shapes."""
    out = []
    for cand in lp.schema_candidates():
        slots: dict[str, int] = {}
        out.append(
            (
                shape_of_term(cand.lhs, slots),
                shape_of_term(cand.rhs, slots),
            )
        )
    return out


def _coherent_key(term: Any) -> Any:
    """Alpha- and coherence-invariant key of *term* (see below)."""
    return _alpha_coherent_key(term)


def _equality_key(lhs: Any, rhs: Any) -> tuple:
    """Direction-free coherent key of the equality ``lhs = rhs``."""
    pair = (_coherent_key(lhs), _coherent_key(rhs))
    return tuple(sorted(pair, key=repr))


def _grammar_keys() -> set:
    """Coherent equality keys of every hand-written candidate.

    Two hand-written laws share a key iff one is a coherent reordering
    of the other (e.g. ``mul_factor`` and ``mul_factor_right``, or
    ``mul_zero`` and ``mul_zero_left``), so the 23 candidates collapse
    to 18 distinct equalities.
    """
    return {_equality_key(c.lhs, c.rhs) for c in lp.schema_candidates()}


#: Per-rule new-enode budget for the grammar-closure check.  Budgeting
#: *every* rule (not just the library's EXPANSIVE ones) is what keeps
#: the closure tractable: the hand-written annihilator laws
#: (``mul(x,0) -> 0``) merge a term with a subterm of itself, so an
#: unbudgeted saturation nests ``mul(x, mul(x, mul(x, ...)))`` without
#: bound.
_CLOSURE_RULE_BUDGET = 150

#: Node / iteration caps for the grammar-closure check.
_CLOSURE_MAX_NODES = 1500
_CLOSURE_MAX_ITERATIONS = 4


def _grammar_closure() -> tuple[list, dict[str, int]]:
    """Return ``ALL_RULES`` plus the hand-written schemas, budgeted.

    The hand-written schemas are abstracted to *patterns* (slots ->
    metavariables) so a candidate matches any instantiation -- a law
    that is the grammar's ``mul_zero`` applied to a subterm
    (``mul(neg x, 0) -> 0``) is a *generated instance*, not a new
    schema.  A search law is "generated by the grammar" when this rule
    set *derives* it, which covers reorderings (via the library's
    comm/assoc), instances, and composites of grammar laws.
    """
    grammar = [
        Rewrite(f"grammar:{i}", _to_pattern(lhs), _to_pattern(rhs))
        for i, (lhs, rhs) in enumerate(_human_schemas())
    ]
    rules = [*ALL_RULES, *grammar]
    budgets = {r.name: _CLOSURE_RULE_BUDGET for r in rules}
    return rules, budgets


def _evaluate(cands: list) -> list:
    """Score candidates with ``law_proposal``'s oracles, bounded.

    Computes the same four verdicts ``law_proposal.evaluate`` does
    (derivable / num_true / relation / cost delta) from the same
    functions, but the library-derivability probe runs under the
    pipeline's *bounded saturation* policy (``rule_budgets`` + a node
    cap).  The shipped ``verify_law`` default (200_000 nodes) lets a
    search term whose ``mul``/``add`` chain is comm-assoc-explosive
    saturate for minutes; bounding it keeps the search to seconds.  It
    cannot change usefulness: a derivable law can never lower a cost
    (its members are already reachable), so ``derivable`` is cosmetic
    for the verdict the search optimises.
    """
    lib = lp._library_keys()
    budgets = lp._rule_budgets(list(ALL_RULES))
    out = []
    for cand in cands:
        res = lp.verify_law(
            cand.lhs,
            cand.rhs,
            list(ALL_RULES),
            rule_budgets=budgets,
            max_iterations=_SEARCH_MAX_ITERATIONS,
            max_nodes=_SEARCH_MAX_NODES,
        )
        base, with_rule = lp._cost_delta(cand, [cand.lhs], flops_cost)
        out.append(
            lp.Outcome(
                candidate=cand,
                derivable=res.derivable,
                witness=tuple(res.witness_rules),
                num_true=lp._numeric_true(cand.lhs, cand.rhs),
                relation=lp._relation(cand.lhs, cand.rhs, lib),
                base_cost=base,
                cand_cost=with_rule,
            )
        )
    return out


def evaluate_schemas(schemas: list) -> list[ShapeResult]:
    """Instantiate and score every schema with the shipped oracles."""
    cands = []
    for lhs, rhs in schemas:
        term_l, term_r = _instantiate_schema(lhs, rhs)
        cands.append(
            lp.Candidate(
                "search",
                op_repr(term_l),
                term_l,
                term_r,
            )
        )
    outcomes = _evaluate(cands)
    rules, budgets = _grammar_closure()
    results = []
    for cand, outcome in zip(cands, outcomes, strict=True):
        # Only a *useful* law can be a "new schema": the rest are not
        # reported, and the closure check is the expensive step.
        novel = False
        stop = ""
        if outcome.useful:
            proof = lp.verify_law(
                cand.lhs,
                cand.rhs,
                rules,
                rule_budgets=budgets,
                max_iterations=_CLOSURE_MAX_ITERATIONS,
                max_nodes=_CLOSURE_MAX_NODES,
            )
            novel = not proof.derivable
            stop = proof.stop
        results.append(
            ShapeResult(
                lhs_repr=op_repr(cand.lhs),
                rhs_repr=op_repr(cand.rhs),
                law=f"{op_repr(cand.lhs)} -> {op_repr(cand.rhs)}",
                outcome=outcome,
                new_to_grammar=novel,
                closure_stop=stop,
            )
        )
    return results


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _counts(results: list[ShapeResult]) -> dict:
    """Aggregate the yield counts over a set of shape results."""
    useful = [r for r in results if r.outcome.useful]
    useful_new = [r for r in useful if r.outcome.new]
    novel = [r for r in useful_new if r.new_to_grammar]
    return {
        "evaluated": len(results),
        "useful": len(useful),
        "useful_new": len(useful_new),
        "useful_new_novel": len(novel),
        "novel": novel,
    }


def _fmt_ratio(hits: int, total: int) -> str:
    """Render a hits/total ratio (``0`` when *total* is 0)."""
    return f"{hits / total:.3f}" if total else "0.000"


def _fmt_report(human: dict, searched: dict, args) -> str:
    """Render the human-grammar vs search comparison."""
    # The seeds are the hand-written grammar, so the search's own
    # contribution is everything *beyond* them.
    marg_n = searched["evaluated"] - human["evaluated"]
    marg_hits = searched["useful_new"] - human["useful_new"]
    lines = [
        "== schema-level search vs the hand-written grammar ==",
        "",
        "human grammar (law_proposal.schema_candidates)",
        f"  candidates                {human['evaluated']:>6}",
        f"  distinct equalities       {human['distinct']:>6}",
        f"  useful & new (library)    {human['useful_new']:>6}",
        f"  useful per verification   "
        f"{_fmt_ratio(human['useful_new'], human['evaluated']):>6}",
        "",
        f"shape search (depth {args.depth}, level cap "
        f"{args.level_cap}, total cap {args.total_cap})",
        f"  shapes evaluated          {searched['evaluated']:>6}",
        f"  useful & new (library)    {searched['useful_new']:>6}",
        f"  useful & new & not in grammar "
        f"{searched['useful_new_novel']:>6}",
        f"  useful per verification   "
        f"{_fmt_ratio(searched['useful_new'], searched['evaluated']):>6}",
        "  search only (excludes the 23 seeds)",
        f"    verifications           {marg_n:>6}",
        f"    useful & new            {marg_hits:>6}",
        f"    useful per verification "
        f"{_fmt_ratio(marg_hits, marg_n):>6}",
        "",
        "== useful schemas the human grammar does NOT generate ==",
    ]
    novel = searched["novel"]
    if not novel:
        lines.append(
            "  (none -- the search found no useful law outside the "
            "hand-written grammar)"
        )
        return "\n".join(lines)
    for r in novel:
        o = r.outcome
        lines.append(f"  {r.law}")
        lines.append(
            f"      cost {o.base_cost:.0f} -> {o.cand_cost:.0f}"
            f"  (derivable={o.derivable}, num_true={o.num_true},"
            f" closure_stop={r.closure_stop})"
        )
    return "\n".join(lines)


def _dump_json(path: str, human: dict, searched: dict) -> None:
    """Write machine-readable results."""
    payload = {
        "human": {
            "evaluated": human["evaluated"],
            "useful_new": human["useful_new"],
        },
        "search": {
            "evaluated": searched["evaluated"],
            "useful_new": searched["useful_new"],
            "useful_new_novel": searched["useful_new_novel"],
        },
        "novel": [
            {
                "lhs": r.lhs_repr,
                "rhs": r.rhs_repr,
                "base_cost": r.outcome.base_cost,
                "cand_cost": r.outcome.cand_cost,
                "derivable": r.outcome.derivable,
                "num_true": r.outcome.num_true,
                "relation": r.outcome.relation,
                "closure_stop": r.closure_stop,
            }
            for r in searched["novel"]
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the schema search and print the comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument(
        "--depth", type=int, default=3, help="search levels"
    )
    parser.add_argument(
        "--level-cap",
        type=int,
        default=2000,
        help="fresh neighbours per level",
    )
    parser.add_argument(
        "--total-cap",
        type=int,
        default=6000,
        help="total schemas evaluated",
    )
    args = parser.parse_args(argv)

    seeds = _human_schemas()
    human_results = evaluate_schemas(seeds)
    human = _counts(human_results)
    human["distinct"] = len(_grammar_keys())

    schemas = search(
        seeds,
        depth=args.depth,
        level_cap=args.level_cap,
        total_cap=args.total_cap,
    )
    searched = _counts(evaluate_schemas(schemas))

    print(_fmt_report(human, searched, args))

    if args.json:
        _dump_json(args.json, human, searched)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
