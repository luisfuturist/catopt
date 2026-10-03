"""Candidate law proposal — the open half of "the AI invents laws".

``tools/law_verifier.py`` closed the *verify* half: a proposed
equality ``lhs = rhs`` is decided in milliseconds by putting both
sides in a fresh e-graph, saturating under the known laws, and asking
whether they land in one e-class.  The rediscovery retro
(``project/retros/law-rediscovery.md``) left the *propose* half open:
28 of 51 library laws are primitive axioms and the 23 "derivable" ones
are 22 construction artifacts plus one genuine composite — so
verification is free but nothing generates *good candidates*.

This tool builds and measures candidate-proposal strategies.  For
each strategy it generates candidates, then reports **yield**:

* **proposed** — candidates the strategy emitted;
* **derivable** — ``verify_law`` proved it from ``ALL_RULES`` (a
  composite of the library — the retro's ``derivable`` column);
* **num-true** — both sides agree numerically on random fp64 tensors
  (the truth proxy for a candidate ``verify_law`` *cannot* prove,
  because it is not a consequence of the library);
* **genuinely new** — structurally not a duplicate or an inverse of
  any library rule (the retro's own line);
* **useful** — adding the rule to ``ALL_RULES`` and re-saturating a
  real term drops the extracted cost under the shipped cost model
  (``flops_cost``).  A merely-true law that never changes a cost is
  *not* useful.

Three strategies are the ones the retro named; a fourth — algebraic
*schema enumeration* — is the one the retro did not try, and it is
the only one that produces a useful new law.  The decisive finding is
recorded in the companion retro.

Strategies:

1. :func:`near_miss_candidates` — saturate real terms, then propose
   the equality between structurally-close representatives from
   *different* e-classes (equal-after-one-more-step, but the step is
   not in the library).
2. :func:`composite_candidates` — fire one library law at a position
   of another law's instance; the composite equality is the candidate
   (the rediscovery retro found composites verify).
3. :func:`schema_candidates` — enumerate a small grammar of algebraic
   identities over the op vocabulary (distributivity, absorption,
   annihilators, involutions), instantiate on concrete shapes.
4. :func:`rank_pool` — score every candidate with a learned
   rule-value net, a cost-delta heuristic, and random, then compare
   top-k precision on the *useful* target.

Run::

    .venv/bin/python tools/law_proposal.py
    .venv/bin/python tools/law_proposal.py --json /tmp/law_proposal.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.egraph.types import _LeafRegistry
from catopt_core.features import compute_features
from catopt_core.game import Action
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import ALL_RULES
from catopt_core.laws import tags as _tags
from catopt_core.trajectories import rule_samples
from catopt_torch.learned_policy import LearnedPolicy, train_rule_value

# ``law_verifier`` is a sibling script; running this file puts
# ``tools/`` on ``sys.path``, so the import resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from law_verifier import (
    instance_of,
    verify_law,
)

__all__ = [
    "Candidate",
    "Outcome",
    "composite_candidates",
    "evaluate",
    "main",
    "near_miss_candidates",
    "rank_pool",
    "schema_candidates",
    "seed_terms",
    "yield_table",
]

#: Saturation budget for a cost-change check on a real term.
_MAX_ITERATIONS = 8
_MAX_NODES = 4_000

#: Largest program a cost-change check will saturate (bounds runtime).
_MAX_PROGRAM_SIZE = 10

#: Per-rule node budget for the EXPANSIVE closure generators
#: (comm/assoc/…), mirroring the pipeline's bounded saturation.  The
#: Catalan blow-up is the whole reason a raw ``ALL_RULES`` run on a
#: small term can still reach thousands of e-nodes.
_EXPANSIVE_BUDGET = 600

#: Tighter budget for the near-miss mining scan (many terms).
_SCAN_ITERATIONS = 5
_SCAN_NODES = 2_500

#: Numeric-comparison tolerance (fp64).
_TOL = 1e-6

#: Feature dim used to build the synthetic seed terms.
_D = 4


# ---------------------------------------------------------------------------
#  Candidate + outcome records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One proposed equality ``lhs = rhs`` with its provenance."""

    strategy: str
    label: str
    lhs: Any
    rhs: Any
    note: str = ""

    def as_rule(self) -> Rewrite:
        """Return the candidate as a fireable ``Rewrite``."""
        return Rewrite(
            f"{self.strategy}:{self.label}", self.lhs, self.rhs
        )


@dataclass
class Outcome:
    """A candidate plus every measured verdict."""

    candidate: Candidate
    derivable: bool = False
    witness: tuple[str, ...] = ()
    num_true: bool | None = None
    relation: str = "new"
    base_cost: float = 0.0
    cand_cost: float = 0.0

    @property
    def new(self) -> bool:
        """True iff not a tautology, duplicate, or inverse."""
        return self.relation == "new"

    @property
    def truth(self) -> bool:
        """True iff derivable, or numerically true on random fp64s."""
        return self.derivable or self.num_true is True

    @property
    def useful(self) -> bool:
        """True iff a *true* rule strictly lowered a program's cost."""
        return self.truth and self.cand_cost < self.base_cost


# ---------------------------------------------------------------------------
#  Structural key — duplicate / inverse detection (leaf-abstracted)
# ---------------------------------------------------------------------------


def _attr_canon(v: Any, mv: dict) -> Any:
    """Canonical form of one attr value under the metavar map *mv*."""
    if isinstance(v, str):
        if v.startswith("$attr:"):
            return ("av", v)
        return ("m", mv.setdefault(v, len(mv)))
    try:
        hash(v)
        return ("lit", v)
    except TypeError:
        return ("lit", repr(v))


def _canon(term: Any, mv: dict) -> Any:
    """Abstract a term to a leaf-renamed structural key.

    Distinct leaves (``Var`` / ``Param`` / a bare ``str`` metavariable)
    become shared metavar indices; ``Const`` stays literal; attrs are
    canonicalised.  Two terms are alpha-equal iff their keys coincide
    under a *shared* ``mv`` — the check the retro's
    ``classify_relation`` performs, extended to concrete leaves.
    """
    if isinstance(term, Op):
        args = tuple(_canon(a, mv) for a in term.args)
        attrs = tuple(
            sorted(
                (k, _attr_canon(v, mv)) for k, v in term.attrs.items()
            )
        )
        return (term.op, args, attrs)
    if isinstance(term, Const):
        return ("c", term.value)
    if isinstance(term, str):
        return ("m", mv.setdefault(term, len(mv)))
    return ("m", mv.setdefault(repr(term), len(mv)))


def _key(lhs: Any, rhs: Any) -> tuple[Any, Any]:
    """Return the alpha-normal ``(lhs, rhs)`` key of a proposed law."""
    mv: dict = {}
    return _canon(lhs, mv), _canon(rhs, mv)


def _library_keys() -> list[tuple[Any, Any]]:
    """Alpha-normal keys of every library rule."""
    return [_key(r.lhs, r.rhs) for r in ALL_RULES]


def _relation(lhs: Any, rhs: Any, lib: list[tuple[Any, Any]]) -> str:
    """Classify a candidate against the library.

    Returns ``"tautology"`` (``lhs == rhs``), ``"duplicate"`` (same
    structure as a rule), ``"inverse"`` (the swapped structure), or
    ``"new"``.
    """
    key = _key(lhs, rhs)
    if key[0] == key[1]:
        return "tautology"
    for rk in lib:
        if key == rk:
            return "duplicate"
        if key == (rk[1], rk[0]):
            return "inverse"
    return "new"


# ---------------------------------------------------------------------------
#  Numeric truth (the oracle for candidates verify_law cannot prove)
# ---------------------------------------------------------------------------


_EVAL: Any = None


def _eval_backend() -> Any:
    """Return the torch concrete-eval backend (registered lazily)."""
    global _EVAL
    if _EVAL is None:
        from catopt_torch.meta_eval import TorchConcreteEval

        _EVAL = TorchConcreteEval()
    return _EVAL


def _leaves(term: Any, out: set | None = None) -> set:
    """Return every non-``Const`` leaf of *term*."""
    out = set() if out is None else out
    if isinstance(term, Op):
        for a in term.args:
            _leaves(a, out)
    elif isinstance(term, (Var, Param)):
        out.add(term)
    return out


def _numeric_true(lhs: Any, rhs: Any) -> bool | None:
    """Return whether both sides agree on random fp64 tensors.

    ``None`` means "undecidable here" — an unbound op or an unknown
    leaf shape.  A genuine ``False`` is the oracle rejecting a false
    equality, which is what makes the numeric check falsifiable.
    """
    backend = _eval_backend()
    leaves = _leaves(lhs) | _leaves(rhs)
    env: dict = {}
    for leaf in leaves:
        typ = getattr(leaf, "typ", None)
        shape = tuple(typ.shape) if typ is not None else ()
        if any(d is None for d in shape):
            return None
        # ``torch.randn(())`` is a 0-dim tensor; ``torch.randn(*())``
        # would be a call with no size at all, so pass the tuple.
        env[leaf] = torch.randn(shape, dtype=torch.float64)
    try:
        a = backend.eval_term(lhs, env)
        b = backend.eval_term(rhs, env)
    except Exception:
        return None
    return _allclose(a, b, _TOL)


def _allclose(a: Any, b: Any, tol: float) -> bool:
    """Compare two values with tolerance, promoting to fp64.

    The shipped ``meta_eval`` comparison is strict about dtype (a
    ``Const`` leaf lowers to an integer tensor), so it rejects a true
    ``x*0 = 0``; promoting both sides to fp64 first is the fix.  A
    shape mismatch is a *false* equality — equal terms always share a
    shape — so it returns ``False`` rather than raising.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _allclose(x, y, tol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        try:
            return bool(
                torch.allclose(
                    a.to(torch.float64),
                    b.to(torch.float64),
                    atol=tol,
                    rtol=tol,
                )
            )
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
#  Usefulness — does the rule lower a real program's extracted cost?
# ---------------------------------------------------------------------------


def _sat_cost(term: Any, rules: list, cost_fn: Any) -> float:
    """Extract the cheapest member of *term* after saturating."""
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        rules,
        root,
        max_iterations=_MAX_ITERATIONS,
        max_nodes=_MAX_NODES,
        rule_budgets=_rule_budgets(rules),
    )
    best = eg.extract_best(root, cost_fn)
    return dag_cost(best, cost_fn) if best is not None else float("inf")


def _rule_budgets(rules: list) -> dict[str, int]:
    """Budget the EXPANSIVE rules present in *rules* (bounded saturation)."""
    names = {r.name for r in rules}
    return {
        r.name: _EXPANSIVE_BUDGET
        for r in ALL_RULES
        if r.name in names and _tags.EXPANSIVE in r.tags
    }


def _cost_delta(cand: Candidate, programs: list, cost_fn: Any) -> tuple:
    """Return ``(base, cand)`` costs over *programs* for *cand*.

    ``base`` saturates under ``ALL_RULES``; ``cand`` adds the candidate
    rule.  A derivable candidate cannot lower ``base`` (its members are
    already reachable), so this is the strict "does the library gain"
    test the retro asks for.
    """
    rule = cand.as_rule()
    base = float("inf")
    with_rule = float("inf")
    for term in programs:
        if _size(term) > _MAX_PROGRAM_SIZE:
            continue
        base = min(base, _sat_cost(term, list(ALL_RULES), cost_fn))
        with_rule = min(
            with_rule, _sat_cost(term, [*ALL_RULES, rule], cost_fn)
        )
    return base, with_rule


# ---------------------------------------------------------------------------
#  Seed terms — small, real, well-typed programs
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int) -> Var:
    """Return a named tensor variable."""
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape: int) -> Param:
    """Return a named parameter."""
    return Param(name, TensorType(tuple(shape)))


def seed_terms() -> list[Op]:
    """Return the curated real terms the strategies mine."""
    d = _D
    x, y, z = _v("x", d, d), _v("y", d, d), _v("z", d, d)
    a, b, c = _v("a", 3, 5), _v("b", 5, 2), _v("c", 2, 3)
    w, w1, w2 = _p("W", d, d), _p("W1", d, d), _p("W2", d, d)
    q = _v("q", 2, 8, 4)
    k = _v("k", 2, 8, 4)
    v = _v("v", 2, 8, 4)
    return [
        # matmul bracketing (associativity)
        Op.make("matmul", a, Op.make("matmul", b, c)),
        # elementwise duplication (CSE)
        Op.make(
            "add",
            Op.make("square", Op.make("mul", x, y)),
            Op.make("mul", Op.make("mul", x, y), Op.make("mul", x, y)),
        ),
        # weight-merge / distribute
        Op.make(
            "add", Op.make("matmul", x, w1), Op.make("matmul", x, w2)
        ),
        Op.make("matmul", w, Op.make("add", x, y)),
        # stacked linear
        Op.make("linear", Op.make("linear", x, w1), w2),
        # swiglu
        Op.make(
            "mul",
            Op.make("silu", Op.make("matmul", x, w1)),
            Op.make("matmul", x, w2),
        ),
        # attention path
        Op.make(
            "matmul",
            Op.make(
                "softmax",
                Op.make(
                    "matmul",
                    q,
                    Op.make("transpose", k, dim0=-2, dim1=-1),
                ),
                dim=-1,
            ),
            v,
        ),
        # elementwise algebra
        Op.make(
            "add",
            Op.make("mul", x, y),
            Op.make("mul", x, z),
        ),
        Op.make("add", Op.make("neg", x), Op.make("neg", y)),
        Op.make("mul", Op.make("exp", x), Op.make("exp", y)),
        Op.make("sub", x, Op.make("neg", y)),
        Op.make("square", Op.make("neg", x)),
        Op.make("add", x, x),
        Op.make("mul", x, Const(0)),
        Op.make("pow", x, Const(1)),
        Op.make("sub", x, x),
    ]


# ---------------------------------------------------------------------------
#  Strategy 1 — near-miss mining
# ---------------------------------------------------------------------------


def _size(term: Any) -> int:
    """Return the number of op nodes in *term*."""
    if not isinstance(term, Op):
        return 0
    return 1 + sum(_size(a) for a in term.args)


def _tree_dist(a: Any, b: Any) -> int:
    """Return a cheap structural distance between two terms.

    Same op / attrs recurse positionally; anything else charges the
    remaining subtree sizes.  ``0`` means identical, ``1`` means a
    single leaf or arity difference — the "one step away" region.
    """
    if isinstance(a, Op) and isinstance(b, Op):
        if a.op != b.op or a.attrs != b.attrs:
            return _size(a) + _size(b)
        dist = abs(len(a.args) - len(b.args))
        for ca, cb in zip(a.args, b.args, strict=False):
            dist += _tree_dist(ca, cb)
        return dist
    return 0 if a == b else 1


def _det_rep(
    eg: EGraph, eid: int, memo: dict, seen: frozenset = frozenset()
) -> Any:
    """Return a *deterministic* member term of an e-class.

    ``EGraph.any_term`` picks whichever member a Python ``set`` yields
    first — and an e-class's node set iterates in hash order, which the
    per-run string-hash seed perturbs, so the near-miss candidate set
    would wobble run to run.  This variant chooses the member with the
    lexicographically smallest ``op_repr`` (leaf first), recursing
    through the same rule, so the choice depends on the *term*, never
    on eid or set order.
    """
    eid = eg.find(eid)
    if eid in memo:
        return memo[eid]
    cls = eg._classes.get(eid)
    if cls is None:
        return None
    leaves = sorted(
        (n for n in cls.nodes if n.op == "leaf"),
        key=lambda n: repr(n.attrs),
    )
    if leaves:
        key = leaves[0].attrs[0][1] if leaves[0].attrs else "??"
        term = _LeafRegistry.decode(key)
        memo[eid] = term
        return term
    if eid in seen:
        return None
    seen = seen | {eid}
    best: Any = None
    best_key: str | None = None
    for node in cls.nodes:
        args = []
        ok = True
        for c in node.children:
            canon = eg.find(c)
            if canon == eid or canon in seen:
                ok = False
                break
            t = _det_rep(eg, canon, memo, seen)
            if t is None:
                ok = False
                break
            args.append(t)
        if not ok:
            continue
        term = Op.make(node.op, *args, **dict(node.attrs))
        r = op_repr(term)
        if best_key is None or r < best_key:
            best_key, best = r, term
    if best is not None:
        memo[eid] = best
    return best


def _class_reps(eg: EGraph, max_size: int) -> list[Op]:
    """One small representative per e-class of *eg* (deterministic)."""
    reps: list[Op] = []
    seen_eid: set[int] = set()
    seen_repr: set[str] = set()
    memo: dict = {}
    for eid in sorted(eg._classes):
        root = eg.find(eid)
        if root in seen_eid:
            continue
        seen_eid.add(root)
        term = _det_rep(eg, root, memo)
        if not isinstance(term, Op) or _size(term) > max_size:
            continue
        r = op_repr(term)
        if r in seen_repr:
            continue
        seen_repr.add(r)
        reps.append(term)
    reps.sort(key=op_repr)
    return reps


def near_miss_candidates(
    seeds: list,
    *,
    max_dist: int = 1,
    max_pairs: int = 120,
    max_size: int = 8,
) -> list[Candidate]:
    """Propose equalities between structurally-close e-class reps.

    Saturate each seed under ``ALL_RULES``, take one representative
    per e-class, and propose ``a = b`` whenever two reps from
    *different* classes sit within ``max_dist`` — they are "almost
    equal", but no library rule closes the gap (else they would share
    a class).  These are candidate *new axioms*, so ``verify_law``
    will (correctly) refuse them; the numeric oracle decides truth.
    """
    out: list[Candidate] = []
    seen: set[tuple[Any, Any]] = set()
    for term in seeds:
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(
            list(ALL_RULES),
            root,
            max_iterations=_SCAN_ITERATIONS,
            max_nodes=_SCAN_NODES,
            rule_budgets=_rule_budgets(list(ALL_RULES)),
        )
        reps = _class_reps(eg, max_size)
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a, b = reps[i], reps[j]
                if _tree_dist(a, b) > max_dist:
                    continue
                key = _key(a, b)
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                out.append(
                    Candidate(
                        "near-miss",
                        f"{op_repr(a)} = {op_repr(b)}",
                        a,
                        b,
                        note=f"dist<={max_dist}",
                    )
                )
                if len(out) >= max_pairs:
                    return out
    return out


# ---------------------------------------------------------------------------
#  Strategy 2 — composition of existing laws
# ---------------------------------------------------------------------------


def composite_candidates(
    rules: list | None = None, *, max_cands: int = 120
) -> list[Candidate]:
    """Compose two library laws into a composite equality.

    For each ordered pair ``(r1, r2)``, instantiate ``r1`` on a real
    term and fire ``r2`` at every position of the rewritten form; the
    pair ``(r1.lhs instance, r1.rhs with r2 applied)`` is the
    candidate.  It is derivable by construction — the retro's
    "composites verify" — so the question is whether any is *useful*.
    """
    universe = list(ALL_RULES if rules is None else rules)
    inst: dict[str, Any] = {}
    for r in universe:
        i = instance_of(r)
        if i is not None:
            inst[r.name] = i
    out: list[Candidate] = []
    seen: set[tuple[Any, Any]] = set()
    for r1 in universe:
        if r1.name not in inst:
            continue
        l1, m1 = inst[r1.name]
        for r2 in universe:
            if r2.name == r1.name:
                continue
            for path, _ in _positions(m1):
                out2 = _apply(r2, m1, path)
                if out2 is None or out2 == m1:
                    continue
                key = _key(l1, out2)
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                out.append(
                    Candidate(
                        "composite",
                        f"{r1.name}+{r2.name}",
                        l1,
                        out2,
                        note=f"{op_repr(l1)} -> {op_repr(out2)}",
                    )
                )
                if len(out) >= max_cands:
                    return out
    return out


def _positions(term: Any):
    """Yield ``(path, subterm)`` pre-order (thin meta re-export)."""
    from catopt_core.meta import _positions as _pos

    return _pos(term)


def _apply(rule: Rewrite, term: Any, path: tuple) -> Any:
    """Apply *rule* at *path* (``None`` when it does not fire)."""
    from catopt_core.meta import apply_rewrite_at

    return apply_rewrite_at(rule, term, path)


# ---------------------------------------------------------------------------
#  Strategy 3 — algebraic schema enumeration
# ---------------------------------------------------------------------------


def schema_candidates() -> list[Candidate]:
    """Enumerate a grammar of algebraic identities over the ops.

    Each schema is a mathematically-motivated equality (distributivity,
    factoring, absorption, annihilators, involutions, inverse
    elements) instantiated on concrete shapes.  The set deliberately
    mixes *true* identities with *false* ones (so the numeric oracle
    is exercised) and includes a library duplicate (so the
    structural classifier is exercised).  This is the strategy the
    retro did not try — and the only one that yields a useful new law.
    """
    d = _D
    x, y, z = _v("x", d, d), _v("y", d, d), _v("z", d, d)
    out: list[Candidate] = []

    def add(label: str, lhs: Any, rhs: Any, note: str = "") -> None:
        out.append(Candidate("schema", label, lhs, rhs, note))

    # -- true: elementwise distributivity / factoring -----------------
    add(
        "mul_factor",
        Op.make("add", Op.make("mul", x, y), Op.make("mul", x, z)),
        Op.make("mul", x, Op.make("add", y, z)),
        "x*y + x*z = x*(y+z)  (fewer nodes)",
    )
    add(
        "mul_distribute",
        Op.make("mul", x, Op.make("add", y, z)),
        Op.make("add", Op.make("mul", x, y), Op.make("mul", x, z)),
        "reverse (expands)",
    )
    add(
        "mul_factor_right",
        Op.make("add", Op.make("mul", y, x), Op.make("mul", z, x)),
        Op.make("mul", Op.make("add", y, z), x),
        "right-slot variant",
    )
    add(
        "neg_factor",
        Op.make("add", Op.make("neg", x), Op.make("neg", y)),
        Op.make("neg", Op.make("add", x, y)),
        "-x + -y = -(x+y)",
    )
    add(
        "neg_distribute",
        Op.make("neg", Op.make("add", x, y)),
        Op.make("add", Op.make("neg", x), Op.make("neg", y)),
        "reverse (expands)",
    )
    add(
        "sub_add_factor",
        Op.make("sub", Op.make("sub", x, y), z),
        Op.make("sub", x, Op.make("add", y, z)),
        "(x-y)-z = x-(y+z)",
    )
    add(
        "div_add",
        Op.make("div", Op.make("add", x, y), z),
        Op.make("add", Op.make("div", x, z), Op.make("div", y, z)),
        "(x+y)/z = x/z + y/z",
    )
    add(
        "square_neg",
        Op.make("square", Op.make("neg", x)),
        Op.make("square", x),
        "(-x)^2 = x^2",
    )
    add(
        "exp_factor",
        Op.make("mul", Op.make("exp", x), Op.make("exp", y)),
        Op.make("exp", Op.make("add", x, y)),
        "e^x e^y = e^(x+y)",
    )
    add(
        "exp_distribute",
        Op.make("exp", Op.make("add", x, y)),
        Op.make("mul", Op.make("exp", x), Op.make("exp", y)),
        "reverse (expands)",
    )
    add(
        "mul_neg",
        Op.make("mul", Op.make("neg", x), y),
        Op.make("neg", Op.make("mul", x, y)),
        "(-x)y = -(xy)",
    )
    add(
        "square_mul",
        Op.make("square", Op.make("mul", x, y)),
        Op.make("mul", Op.make("square", x), Op.make("square", y)),
        "(xy)^2 = x^2 y^2",
    )
    add(
        "sigmoid_neg",
        Op.make("sigmoid", Op.make("neg", x)),
        Op.make("sub", Const(1), Op.make("sigmoid", x)),
        "sigma(-x) = 1 - sigma(x)",
    )
    # -- true: annihilators / identity elements -----------------------
    add(
        "mul_zero",
        Op.make("mul", x, Const(0)),
        Const(0),
        "x*0 = 0",
    )
    add(
        "mul_zero_left",
        Op.make("mul", Const(0), x),
        Const(0),
        "0*x = 0",
    )
    add(
        "pow_one",
        Op.make("pow", x, Const(1)),
        x,
        "x^1 = x",
    )
    add(
        "sub_self",
        Op.make("sub", x, x),
        Const(0),
        "x - x = 0",
    )
    add(
        "add_inv",
        Op.make("add", x, Op.make("neg", x)),
        Const(0),
        "x + (-x) = 0",
    )
    add(
        "div_self",
        Op.make("div", x, x),
        Const(1),
        "x / x = 1",
    )
    # -- duplicate of a library law (classifier control) --------------
    add(
        "sub_to_add_dup",
        Op.make("sub", x, y),
        Op.make("add", x, Op.make("neg", y)),
        "already in the library",
    )
    # -- false identities (numeric-oracle controls) -------------------
    add(
        "FALSE_mul_factor",
        Op.make("add", Op.make("mul", x, y), Op.make("mul", x, z)),
        Op.make("mul", x, Op.make("add", x, z)),
        "false: x*y + x*z != x*(x+z)",
    )
    add(
        "FALSE_exp_add",
        Op.make("exp", Op.make("add", x, y)),
        Op.make("add", Op.make("exp", x), Op.make("exp", y)),
        "false: e^(x+y) != e^x + e^y",
    )
    add(
        "FALSE_square_add",
        Op.make("square", Op.make("add", x, y)),
        Op.make("add", Op.make("square", x), Op.make("square", y)),
        "false: (x+y)^2 != x^2 + y^2",
    )
    return out


# ---------------------------------------------------------------------------
#  Strategy 4 — rank a candidate pool (learned vs heuristic vs random)
# ---------------------------------------------------------------------------


def _pool_features(cand: Candidate) -> Any:
    """Return the static features of a candidate's LHS program."""
    return compute_features(cand.lhs)


def _heuristic_score(cand: Candidate) -> float:
    """Cost-delta heuristic: ``cost(lhs) - cost(rhs)`` (FLOPs)."""
    return dag_cost(cand.lhs, flops_cost) - dag_cost(
        cand.rhs, flops_cost
    )


def _train_scorer(seeds: list) -> Any:
    """Train a rule-value net on single-rule deltas over the seeds."""
    samples = [s for p in seeds for s in rule_samples(p, ALL_RULES)]
    return train_rule_value(
        samples, epochs=300, hidden=32, device="cpu", seed=0
    )


def rank_pool(
    pool: list[Candidate],
    outcomes: dict[int, Outcome],
    *,
    seeds: list | None = None,
) -> dict:
    """Compare learned / heuristic / random ranking on the pool.

    Target is ``useful`` (a true, non-derivable rule that lowers a
    cost).  Each scorer orders the pool and the metric is **average
    precision** — the mean of ``hits@rank / rank`` over the useful
    candidates — which is far more robust to pool composition than a
    fixed precision@k.  A uniformly random ordering has expected AP
    ``useful / pool``, reported as the ``random`` baseline.  ``{}``
    is returned when the pool carries no positive example (ranking is
    then undefined, not "random wins").
    """
    n_pos = sum(1 for o in outcomes.values() if o.useful)
    if n_pos == 0:
        return {}
    model = _train_scorer(seeds or [])
    policy = LearnedPolicy(
        model, {c.as_rule().name: c.as_rule() for c in pool}
    )
    scored: dict[str, list[float]] = {
        "learned": [
            policy.score(_pool_features(c), Action(c.as_rule().name))
            for c in pool
        ],
        "heuristic": [_heuristic_score(c) for c in pool],
    }
    result: dict[str, Any] = {
        "pool": len(pool),
        "useful": n_pos,
        "random_ap": n_pos / len(pool),
    }
    for name, scores in scored.items():
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        result[f"{name}_ap"] = _average_precision(order, outcomes)
    return result


def _average_precision(
    order: list[int], outcomes: dict[int, Outcome]
) -> float:
    """Return average precision of a best-first *order*."""
    hits = 0
    total = 0.0
    for rank, i in enumerate(order, 1):
        if outcomes[i].useful:
            hits += 1
            total += hits / rank
    n_pos = sum(1 for o in outcomes.values() if o.useful)
    return total / n_pos if n_pos else 0.0


# ---------------------------------------------------------------------------
#  Evaluation + yield
# ---------------------------------------------------------------------------


def evaluate(
    candidates: list[Candidate],
    *,
    cost_fn: Any = None,
) -> list[Outcome]:
    """Measure every candidate: derivable, num-true, relation, useful."""
    cost_fn = cost_fn or flops_cost
    lib = _library_keys()
    out: list[Outcome] = []
    for cand in candidates:
        res = verify_law(cand.lhs, cand.rhs, list(ALL_RULES))
        rel = _relation(cand.lhs, cand.rhs, lib)
        base, with_rule = _cost_delta(cand, [cand.lhs], cost_fn)
        out.append(
            Outcome(
                candidate=cand,
                derivable=res.derivable,
                witness=tuple(res.witness_rules),
                num_true=_numeric_true(cand.lhs, cand.rhs),
                relation=rel,
                base_cost=base,
                cand_cost=with_rule,
            )
        )
    return out


def yield_table(outcomes: list[Outcome]) -> dict:
    """Aggregate per-strategy yield counts."""
    table: dict[str, dict] = {}
    for o in outcomes:
        row = table.setdefault(
            o.candidate.strategy,
            {
                "proposed": 0,
                "derivable": 0,
                "num_true": 0,
                "new": 0,
                "useful": 0,
                "useful_new": 0,
            },
        )
        row["proposed"] += 1
        row["derivable"] += int(o.derivable)
        row["num_true"] += int(o.num_true is True)
        row["new"] += int(o.new)
        row["useful"] += int(o.useful)
        row["useful_new"] += int(o.useful and o.new)
    return table


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _fmt_yield(table: dict) -> str:
    """Render the per-strategy yield table."""
    head = (
        f"{'strategy':<12} {'prop':>5} {'deriv':>6} "
        f"{'ntrue':>6} {'new':>5} {'useful':>7} {'u_new':>6}"
    )
    lines = [head, "-" * len(head)]
    for name in sorted(table):
        r = table[name]
        lines.append(
            f"{name:<12} {r['proposed']:>5} {r['derivable']:>6} "
            f"{r['num_true']:>6} {r['new']:>5} {r['useful']:>7} "
            f"{r['useful_new']:>6}"
        )
    return "\n".join(lines)


def _fmt_useful(outcomes: list[Outcome]) -> str:
    """Render the genuinely-new useful laws found."""
    rows = [o for o in outcomes if o.useful and o.new]
    if not rows:
        return "  (none — no strategy produced a useful new law)"
    lines = []
    for o in rows:
        c = o.candidate
        lines.append(
            f"  [{c.strategy}] {op_repr(c.lhs)} -> {op_repr(c.rhs)}\n"
            f"      cost {o.base_cost:.0f} -> {o.cand_cost:.0f}"
            f"  (derivable={o.derivable}, num_true={o.num_true})"
        )
    return "\n".join(lines)


def _dump_json(
    path: str, outcomes: list[Outcome], table: dict, ranking: dict
) -> None:
    """Write machine-readable results."""
    payload = {
        "yield": table,
        "ranking": ranking,
        "outcomes": [
            {
                "strategy": o.candidate.strategy,
                "label": o.candidate.label,
                "lhs": op_repr(o.candidate.lhs),
                "rhs": op_repr(o.candidate.rhs),
                "derivable": o.derivable,
                "witness": list(o.witness),
                "num_true": o.num_true,
                "relation": o.relation,
                "base_cost": o.base_cost,
                "cand_cost": o.cand_cost,
                "useful": o.useful,
            }
            for o in outcomes
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _rank_on(pool: list, outcomes: list, seeds: list) -> dict:
    """Rank a (candidate, outcome) subset; ``{}`` when no positive."""
    if not any(o.useful for o in outcomes):
        return {}
    idx = {i: o for i, o in enumerate(outcomes)}
    return rank_pool(pool, idx, seeds=seeds)


def main(argv: list[str] | None = None) -> int:
    """Run every strategy, evaluate, and print the yield report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    seeds = seed_terms()
    near = near_miss_candidates(seeds)
    comp = composite_candidates()
    schemas = schema_candidates()
    pool = near + comp + schemas

    outcomes = evaluate(pool)
    table = yield_table(outcomes)

    print("== candidate law proposal — yield per strategy ==")
    print(_fmt_yield(table))
    print()
    print("== genuinely-new, useful laws found ==")
    print(_fmt_useful(outcomes))
    print()

    # A derivable candidate can never be useful (its members are
    # already reachable), so the value-ranking question is only
    # well-posed on the *non-derivable* sub-pool — that isolates
    # "which proposed new axiom is worth adopting" from "which
    # composite of the library happens to also lower cost".
    full = _rank_on(pool, outcomes, seeds)
    nd = [o for o in outcomes if not o.derivable]
    nd_pool = [o.candidate for o in nd]
    sub = _rank_on(nd_pool, nd, seeds)

    print("== strategy 4: learned vs heuristic vs random ranking ==")
    for label, r in (("full pool", full), ("non-derivable", sub)):
        if not r:
            print(f"  [{label}] no useful candidate — undefined")
            continue
        print(f"  [{label}] pool={r['pool']} useful={r['useful']}")
        print(
            f"    average precision  learned={r['learned_ap']:.2f}"
            f"  heuristic={r['heuristic_ap']:.2f}"
            f"  random={r['random_ap']:.2f}"
        )

    if args.json:
        _dump_json(
            args.json, outcomes, table, {"full": full, "sub": sub}
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
