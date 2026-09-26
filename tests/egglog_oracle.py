"""Test-only differential oracle: catopt's e-graph vs the ``egglog``
library (Phase 4).

This module is an **opt-in, test-only** oracle.  It is *not* a
production dependency and *not* an engine swap: ``catopt-core`` stays
pure-Python and zero-dependency, and nothing under ``packages/`` or
``catopt/`` imports ``egglog``.  It re-derives the essential parts of
the ``spike/egglog-engine`` prototype so the suite can *differentially
check* catopt's hand-rolled equality-saturation search against the Rust
``egglog`` engine on a small op/law subset.

The point of the oracle is **agreement**, not replacement:

* the same law subset is registered in both engines;
* on a handful of tiny graphs both must reach the *same* lowest-FLOPs
  term up to e-class representative choice (``comm_add``/``comm_mul``
  operand order is representative-only, so comparison canonicalises
  commutative operands);
* the original and both extracted terms must be numerically equal.

Usage (see ``tests/test_egglog_oracle.py`` and ``AGENTS.md``)::

    uv sync --group oracle
    .venv/bin/python -m pytest tests/test_egglog_oracle.py

``egglog`` is a compiled Rust extension; the test self-skips via
``pytest.importorskip("egglog")`` when the group is not installed, so
the rest of the suite stays green without it.

Design notes / deliberate limitations (ported from ``EGGLOG_SPIKE.md``):

* The egglog program is untyped-by-shape: the ``Term`` sort carries no
  tensor type.  Shapes are recovered Python-side from the original IR's
  leaf shapes plus catopt's ``typing._shape_of``.
* Rewrites catopt guards with a Python ``check`` callable (e.g.
  ``assoc_linear_bias``'s shape guard) are registered UNCONDITIONALLY
  here — egglog cannot call a Python predicate on a *matched* binding
  during saturation.  The oracle only feeds well-typed terms, so the
  guard is satisfied on every case it exercises.
* No proof/certificate replay: egglog has no first-class proof objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from catopt_core.cost import flops_cost
from catopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    op_repr,
)
from catopt_core.laws.tensor import (
    ASSOC_ADD,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    ASSOC_MUL,
    COMM_ADD,
    COMM_MUL,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
)
from catopt_core.typing import _shape_of
from egglog import (
    EGraph,
    Expr,
    StringLike,
    expr_parts,
    f64Like,
    function,
    i64Like,
    rewrite,
    ruleset,
    vars_,
)

__all__ = [
    "SPIKE_RULES",
    "SUPPORTED_OPS",
    "EgglogEngine",
    "Term",
    "build_program",
    "canon",
    "catopt_equivalent",
    "catopt_extract",
    "evaluate",
    "folded_flops",
    "spike_ruleset",
    "to_catopt",
]

#: The op subset the oracle covers.
SUPPORTED_OPS = frozenset(
    {"add", "mul", "neg", "matmul", "linear", "relu", "transpose"}
)

#: The small law subset BOTH engines run (catopt rewrites, by name):
#: comm/assoc add+mul, assoc_matmul (+rev), distribute/factor matmul,
#: assoc_linear (+ the biased 3-ary variant).  ``spike_ruleset`` mirrors
#: this set in egglog.
SPIKE_RULES = [
    COMM_ADD,
    ASSOC_ADD,
    COMM_MUL,
    ASSOC_MUL,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
]


# ---------------------------------------------------------------------------
#  egglog sorts: the Term algebra
# ---------------------------------------------------------------------------


class Term(Expr):
    """A catopt tensor term as an egglog sort (structure only, no type)."""


@function(egg_fn="TVar")
def tvar(name: StringLike) -> Term: ...


@function(egg_fn="TParam")
def tparam(name: StringLike) -> Term: ...


@function(egg_fn="TConst")
def tconst(value: f64Like) -> Term: ...


@function(egg_fn="Add")
def add(a: Term, b: Term) -> Term: ...


@function(egg_fn="Mul")
def mul(a: Term, b: Term) -> Term: ...


@function(egg_fn="Neg")
def neg(a: Term) -> Term: ...


@function(egg_fn="Matmul")
def matmul(a: Term, b: Term) -> Term: ...


@function(egg_fn="Linear")
def linear(x: Term, w: Term) -> Term: ...


@function(egg_fn="LinearB")
def linear_b(x: Term, w: Term, b: Term) -> Term: ...


@function(egg_fn="Relu")
def relu(a: Term) -> Term: ...


@function(egg_fn="Transpose")
def transpose(a: Term, _arg1: i64Like, _arg2: i64Like) -> Term: ...


# ---------------------------------------------------------------------------
#  IR -> egglog
# ---------------------------------------------------------------------------


def build_program(term: Any) -> Term:
    """Translate a catopt IR term into an egglog ``Term`` expression.

    Raises ``ValueError`` on ops outside :data:`SUPPORTED_OPS`.
    """
    if isinstance(term, Var):
        return tvar(term.name)
    if isinstance(term, Param):
        return tparam(term.name)
    if isinstance(term, Const):
        return tconst(float(term.value))
    if isinstance(term, Op):
        op = term.op
        if op not in SUPPORTED_OPS:
            raise ValueError(f"op {op!r} not in the egglog oracle subset")
        args = [build_program(a) for a in term.args]
        if op == "add":
            return add(args[0], args[1])
        if op == "mul":
            return mul(args[0], args[1])
        if op == "neg":
            return neg(args[0])
        if op == "matmul":
            return matmul(args[0], args[1])
        if op == "linear":
            if len(args) == 2:
                return linear(args[0], args[1])
            return linear_b(args[0], args[1], args[2])
        if op == "relu":
            return relu(args[0])
        if op == "transpose":
            a1 = int(term.attrs.get("arg1", -2))
            a2 = int(term.attrs.get("arg2", -1))
            return transpose(args[0], a1, a2)
    raise ValueError(f"cannot translate term: {op_repr(term)}")


# ---------------------------------------------------------------------------
#  egglog -> catopt (for the cost model + result readout)
# ---------------------------------------------------------------------------

#: egglog function ident name -> builder(args, env) -> catopt term.
#: egglog keys module-level ``@function``s by their Python name (not
#: ``egg_fn``), so both leaves and ops live in one registry.
_BUILDERS: dict[str, Callable[[list, dict], Any]] = {
    # leaves
    "tvar": lambda a, env: Var(
        a[0], TensorType(tuple(env["inputs"][a[0]]))
    ),
    "tparam": lambda a, env: Param(
        a[0], TensorType(tuple(env["params"][a[0]]))
    ),
    "tconst": lambda a, env: Const(a[0]),
    # ops
    "add": lambda a, env: Op.make("add", a[0], a[1]),
    "mul": lambda a, env: Op.make("mul", a[0], a[1]),
    "neg": lambda a, env: Op.make("neg", a[0]),
    "matmul": lambda a, env: Op.make("matmul", a[0], a[1]),
    "linear": lambda a, env: Op.make("linear", a[0], a[1]),
    "linear_b": lambda a, env: Op.make("linear", a[0], a[1], a[2]),
    "relu": lambda a, env: Op.make("relu", a[0]),
    "transpose": lambda a, env: Op.make(
        "transpose", a[0], arg1=a[1], arg2=a[2]
    ),
}


def to_catopt(expr: Term, env: dict) -> Any:
    """Reconstruct the catopt term a (representative) egglog expr denotes."""
    return _from_decl(expr_parts(expr), env)


def _from_decl(decl: Any, env: dict) -> Any:
    """Reconstruct from a ``TypedExprDecl`` / raw egglog decl node."""
    inner = decl.expr if hasattr(decl, "expr") else decl
    if not hasattr(inner, "callable"):
        return inner.value  # a literal (String / i64 / f64)
    callee = inner.callable
    args = [_from_decl(a, env) for a in inner.args]
    ident = getattr(callee, "ident", None)
    if ident is not None:
        return _BUILDERS[ident.name](args, env)
    return _BUILDERS[callee.method_name](args, env)


# ---------------------------------------------------------------------------
#  The law subset (mirrors the catopt rule subset)
# ---------------------------------------------------------------------------


def _spike_rules() -> list[tuple[str, Any]]:
    """The small law subset as ``(catopt_rule_name, egglog_rewrite)``."""
    a, b, c = vars_("a b c", Term)
    (w,) = vars_("w", Term)
    (x,) = vars_("x", Term)
    m, n, p = vars_("m n p", Term)
    q, r = vars_("q r", Term)
    b1, b2 = vars_("b1 b2", Term)
    return [
        ("comm_add", rewrite(add(a, b)).to(add(b, a))),
        (
            "assoc_add",
            rewrite(add(a, add(b, c))).to(add(add(a, b), c)),
        ),
        ("comm_mul", rewrite(mul(a, b)).to(mul(b, a))),
        (
            "assoc_mul",
            rewrite(mul(a, mul(b, c))).to(mul(mul(a, b), c)),
        ),
        (
            "assoc_matmul",
            rewrite(matmul(m, matmul(n, p))).to(
                matmul(matmul(m, n), p)
            ),
        ),
        (
            "assoc_matmul_rev",
            rewrite(matmul(matmul(m, n), p)).to(
                matmul(m, matmul(n, p))
            ),
        ),
        (
            "distribute_matmul_over_add",
            rewrite(matmul(w, add(a, b))).to(
                add(matmul(w, a), matmul(w, b))
            ),
        ),
        (
            "factor_matmul",
            rewrite(add(matmul(w, a), matmul(w, b))).to(
                matmul(w, add(a, b))
            ),
        ),
        (
            "assoc_linear",
            rewrite(linear(linear(x, q), r)).to(
                linear(x, matmul(r, q))
            ),
        ),
        (
            "assoc_linear_bias",
            rewrite(linear_b(linear_b(x, q, b1), r, b2)).to(
                add(linear_b(x, matmul(r, q), matmul(r, b1)), b2)
            ),
        ),
    ]


def spike_ruleset() -> Any:
    """The small law subset, as an egglog ruleset.

    Mirrors (by name) the catopt rewrites in :data:`SPIKE_RULES`.
    """
    return ruleset(*[rw for _, rw in _spike_rules()])


def _rule_name_map() -> dict[Any, str]:
    """Map an egglog rewrite decl (report key) -> catopt rule name."""
    return {rw.decl: name for name, rw in _spike_rules()}


# ---------------------------------------------------------------------------
#  The engine wrapper
# ---------------------------------------------------------------------------

#: egglog function ident name -> (catopt op name, # of Term-typed args).
_OP_SPEC: dict[str, tuple[str, int]] = {
    "add": ("add", 2),
    "mul": ("mul", 2),
    "neg": ("neg", 1),
    "matmul": ("matmul", 2),
    "linear": ("linear", 2),
    "linear_b": ("linear", 3),
    "relu": ("relu", 1),
    "transpose": ("transpose", 1),
}


@dataclass(frozen=True)
class FlopsShape:
    """A cost value carrying accumulated FLOPs, a term's shape, and
    whether the subtree is *param-only* (contains no data input).

    The shape is threaded through the tree-cost recursion so that each
    e-node's local FLOPs (which need the *children's* shapes) can be
    computed by catopt's own shape inference.  ``param_only`` replicates
    catopt's compile-time fold discount (a subtree with no ``Var`` leaf
    is materialised at lowering time and charged zero).  Ordering is by
    FLOPs only.
    """

    flops: float
    shape: Any = None
    param_only: bool = True

    def __add__(self, other: FlopsShape) -> FlopsShape:
        return FlopsShape(
            self.flops + other.flops,
            None,
            self.param_only and other.param_only,
        )

    def __lt__(self, other: FlopsShape) -> bool:
        return self.flops < other.flops

    def __le__(self, other: FlopsShape) -> bool:
        return self.flops <= other.flops

    def __gt__(self, other: FlopsShape) -> bool:
        return self.flops > other.flops

    def __ge__(self, other: FlopsShape) -> bool:
        return self.flops >= other.flops


def _node_name(expr: Term) -> tuple[str | None, Any]:
    """Return ``(egglog_fn_name, decl)`` — ``name`` is None for literals."""
    inner = expr_parts(expr).expr
    if not hasattr(inner, "callable"):
        return None, inner  # a literal
    callee = inner.callable
    ident = getattr(callee, "ident", None)
    name = ident.name if ident is not None else callee.method_name
    return name, inner


@dataclass
class EgglogEngine:
    """A thin driver over an egglog e-graph for the oracle op subset."""

    egraph: EGraph = field(default_factory=EGraph)
    env: dict = field(default_factory=lambda: {"inputs": {}, "params": {}})
    root: Term | None = None
    _cost_calls: int = 0

    @classmethod
    def from_ir(cls, ir: Any) -> EgglogEngine:
        """Build an engine seeded with a catopt ``IR``."""
        env = {
            "inputs": {v.name: tuple(v.typ.shape) for v in ir.inputs},
            "params": {
                p.name: tuple(p.typ.shape) for p in ir.params.values()
            },
        }
        eng = cls(env=env)
        eng.root = build_program(ir.root)
        eng.egraph.register(eng.root)
        return eng

    @classmethod
    def from_term(
        cls,
        term: Any,
        input_shapes: dict[str, tuple],
        param_shapes: dict[str, tuple],
    ) -> EgglogEngine:
        """Build an engine from a bare catopt term + leaf shapes."""
        env = {
            "inputs": dict(input_shapes),
            "params": dict(param_shapes),
        }
        eng = cls(env=env)
        eng.root = build_program(term)
        eng.egraph.register(eng.root)
        return eng

    def _tree_cost(
        self, egraph: EGraph, expr: Term, children_costs: list[FlopsShape]
    ) -> FlopsShape:
        """Cost of one subterm = Σ child FLOPs + this node's local FLOPs.

        The *shape* of this node is recovered from the children's carried
        shapes (an e-node's children arrive as opaque e-class values, so
        the term cannot be re-walked here) using catopt's shape rules.
        """
        self._cost_calls += 1
        name, inner = _node_name(expr)
        if name is None:
            return FlopsShape(0.0, None, True)  # a literal
        if name == "tvar":
            nm = _from_decl(inner.args[0], self.env)
            return FlopsShape(0.0, tuple(self.env["inputs"][nm]), False)
        if name == "tparam":
            nm = _from_decl(inner.args[0], self.env)
            return FlopsShape(0.0, tuple(self.env["params"][nm]), True)
        if name == "tconst":
            return FlopsShape(0.0, (), True)
        # an op node: rebuild a catopt Op from placeholder children that
        # carry the children's recovered shapes, then price it.
        op_name, _n = _OP_SPEC[name]
        term_shapes = [c.shape for c in children_costs if c.shape is not None]
        args = [
            Var(f"_c{i}", TensorType(tuple(s)))
            for i, s in enumerate(term_shapes)
        ]
        attrs: dict[str, Any] = {}
        if name == "transpose":
            attrs = {
                "arg1": _from_decl(inner.args[1], self.env),
                "arg2": _from_decl(inner.args[2], self.env),
            }
        node = Op.make(op_name, *args, **attrs)
        param_only = all(c.param_only for c in children_costs)
        # catopt's compile-time fold: a param-only subtree is materialised
        # at lowering, so it is charged zero runtime FLOPs.
        local = 0.0 if param_only else flops_cost(node)
        shape = _shape_of(node)
        total = sum(c.flops for c in children_costs) + local
        return FlopsShape(total, shape, param_only)

    def run(self, iterations: int = 8) -> dict[str, int]:
        """Saturate; return ``{catopt_rule_name: num_matches}``."""
        report = self.egraph.run(spike_ruleset() * iterations)
        names = _rule_name_map()
        fires: dict[str, int] = {}
        for key, count in (report.num_matches_per_rule or {}).items():
            if count:
                fires[names.get(key, str(key))] = count
        return fires

    def extract_best(self) -> tuple[Any, float]:
        """Extract the lowest-FLOPs term (tree cost, catopt FLOPs)."""
        assert self.root is not None
        expr, cost = self.egraph.extract(
            self.root,
            include_cost=True,
            cost_model=self._tree_cost,
            extractor="tree",
        )
        return to_catopt(expr, self.env), cost.flops

    def register(self, term: Any) -> Term:
        """Add an extra term to the e-graph (for equivalence checks)."""
        e = build_program(term)
        self.egraph.register(e)
        return e

    def check_equal(self, a: Term, b: Term) -> bool:
        """True if egglog proved ``a == b`` (both in the e-graph)."""
        try:
            self.egraph.check_bool(a == b)
        except Exception:
            return False
        return True


# ---------------------------------------------------------------------------
#  catopt side helpers (differential comparison)
# ---------------------------------------------------------------------------


def catopt_extract(term: Any, rules: list | None = None) -> tuple[Any, dict]:
    """Run catopt's e-graph with the shared law subset.

    Returns ``(extracted_best_term, rule_fires)``.
    """
    from catopt_core.egraph import EGraph as _EGraph

    eg = _EGraph()
    root = eg.add_term(term)
    eg.run(
        rules if rules is not None else SPIKE_RULES,
        root,
        max_iterations=10,
        max_nodes=200_000,
    )
    return eg.extract_best(root, flops_cost), dict(eg.rule_fires)


def catopt_equivalent(
    term: Any, other: Any, rules: list | None = None
) -> bool:
    """True if catopt's e-graph puts *term* and *other* in one e-class."""
    from catopt_core.egraph import EGraph as _EGraph

    eg = _EGraph()
    root = eg.add_term(term)
    eg.run(
        rules if rules is not None else SPIKE_RULES,
        root,
        max_iterations=10,
        max_nodes=200_000,
    )
    other_eid = eg.add_term(other)
    return eg.find(root) == eg.find(other_eid)


def folded_flops(term: Any) -> float:
    """catopt's *extraction* cost: FLOPs with param-only subtrees zeroed.

    Mirrors the discount in ``EGraph.extract_best`` (a subtree with no
    ``Var`` leaf folds into a materialised parameter at lowering time).
    """
    if isinstance(term, Var):
        return 0.0
    if isinstance(term, (Param, Const)):
        return 0.0
    if isinstance(term, Op):
        child_costs = [folded_flops(a) for a in term.args]
        param_only = not any(_has_var(a) for a in term.args)
        if param_only:
            return 0.0
        local = flops_cost(term) - sum(flops_cost(a) for a in term.args)
        return sum(child_costs) + max(local, 0.0)
    return 0.0


def _has_var(term: Any) -> bool:
    if isinstance(term, Var):
        return True
    if isinstance(term, Op):
        return any(_has_var(a) for a in term.args)
    return False


def canon(term: Any) -> Any:
    """Canonicalise a term modulo commutative-operand order.

    ``comm_add``/``comm_mul`` let the two engines pick either operand
    order for an ``add``/``mul`` node — an e-class *representative*
    difference with no semantic content.  Sorting those operands by
    their rendering makes the comparison representative-insensitive.
    """
    if isinstance(term, Op):
        args = [canon(a) for a in term.args]
        if term.op in ("add", "mul"):
            args = sorted(args, key=op_repr)
        return Op.make(term.op, *args, **term.attrs)
    return term


# ---------------------------------------------------------------------------
#  NumPy evaluator (numeric equivalence, no torch lowering)
# ---------------------------------------------------------------------------


def evaluate(
    term: Any,
    inputs: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a catopt term (oracle op subset) on concrete leaves."""
    if isinstance(term, Var):
        return inputs[term.name]
    if isinstance(term, Param):
        return params[term.name]
    if isinstance(term, Const):
        return np.asarray(term.value, dtype=np.float64)
    if isinstance(term, Op):
        op = term.op
        a = [evaluate(x, inputs, params) for x in term.args]
        if op == "add":
            return a[0] + a[1]
        if op == "mul":
            return a[0] * a[1]
        if op == "neg":
            return -a[0]
        if op == "matmul":
            return a[0] @ a[1]
        if op == "linear":
            out = a[0] @ np.swapaxes(a[1], -1, -2)
            if len(a) >= 3:
                out = out + a[2]
            return out
        if op == "relu":
            return np.maximum(a[0], 0.0)
        if op == "transpose":
            return np.swapaxes(
                a[0],
                int(term.attrs.get("arg1", -2)),
                int(term.attrs.get("arg2", -1)),
            )
    raise ValueError(f"cannot evaluate {term!r}")
