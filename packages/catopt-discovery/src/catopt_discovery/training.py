"""The reverse handler — backprop as a derived interpretation.

Sanada's reverse handler for arrows, made concrete over the term
universe: a forward program's *gradient program* is not a second
semantics stack — it is derived from the same data
(:data:`lawdata.REVERSE`, a per-op VJP table) and returns ordinary
``Op`` terms that the e-graph, laws, referee and extraction already
handle unchanged.  That is the load-bearing claim the module pins:
``backward(term)`` produces a legal citizen of the board.

``backward`` implements textbook reverse mode over the DAG:
topological order, one VJP row per op, cotangents accumulate by
``add`` at shared nodes (a term used twice gets the sum of its
contributions).  Ops absent from the table — reductions needing a
shaped broadcast, view ops — decline honestly: ``backward`` raises
``ValueError`` naming the op rather than silently dropping the
contribution.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from catopt_core.ir import Const, Op, Param, Var

from . import lawdata

__all__ = ["backward", "grad_program", "leaves_of"]


def _topo(term: Any, seen: set, out: list) -> None:
    """Post-order the term DAG into *out* (children first)."""
    if not isinstance(term, Op) or id(term) in seen:
        return
    seen.add(id(term))
    for a in term.args:
        _topo(a, seen, out)
    out.append(term)


def leaves_of(term: Any, out: dict | None = None) -> dict:
    """Collect the term's ``Var``/``Param`` leaves, name → leaf."""
    if out is None:
        out = {}
    if isinstance(term, (Var, Param)):
        out.setdefault(term.name, term)
    elif isinstance(term, Op):
        for a in term.args:
            leaves_of(a, out)
    return out


def _inst_grad(spec: Any, env: dict, attrs: dict) -> Any:
    """Instantiate one VJP spec against the node's *env*.

    ``"$g"``/``"$a<i>"`` look up the cotangent/forward args; a bare
    str is a literal leaf name; numbers become ``Const``; a tuple
    builds an ``Op`` with ``"$attrs"`` splicing the forward node's
    attrs.
    """
    if isinstance(spec, str):
        return env[spec] if spec.startswith("$") else spec
    if isinstance(spec, (int, float)):
        return Const(spec)
    op = spec[0]
    args: list = []
    node_attrs: dict = {}
    for e in spec[1:]:
        if e == "$attrs":
            node_attrs = dict(attrs)
        elif isinstance(e, dict):
            node_attrs = dict(e)
        else:
            args.append(_inst_grad(e, env, attrs))
    return Op.make(op, *args, **node_attrs)


def _accumulate(node: Any, rules: tuple, env: dict, grads: dict) -> None:
    """Route one node's cotangent to its children per *rules*."""
    for child, spec in zip(node.args, rules, strict=False):
        if spec is None:
            continue
        cg = _inst_grad(spec, env, node.attrs)
        grads[child] = (
            Op.make("add", grads[child], cg)
            if child in grads
            else cg
        )


def backward(
    term: Any,
    *,
    cotangent: Any = None,
    table: dict | None = None,
) -> dict[str, Any]:
    """Derive ``d(term)/d(leaf)`` for every leaf — as terms.

    *cotangent* is the incoming gradient at the root (default
    ``Const(1.0)`` — the scalar-loss case, broadcasting over
    elementwise parts; shaped outputs like a matmul result need a
    real cotangent term, e.g. a ``broadcast_to`` of ones).
    *table* overrides :data:`lawdata.REVERSE`.

    Returns ``{leaf_name: grad_term}`` — ``Var`` and ``Param``
    leaves alike (the params a training step actually optimizes).
    A leaf unreachable through VJP'd ops gets no entry; an op with
    no VJP row raises ``ValueError`` — a refused interpretation,
    not a silent zero.
    """
    vjp = lawdata.REVERSE if table is None else table
    g0 = cotangent if cotangent is not None else Const(1.0)
    order: list = []
    _topo(term, set(), order)
    grads: dict[Any, Any] = {term: g0}
    for node in reversed(order):
        g = grads.get(node)
        if g is None:
            continue
        rules = vjp.get(node.op)
        if rules is None:
            raise ValueError(
                f"backward: no VJP row for {node.op!r} — "
                "the reverse handler declines (add it to "
                "lawdata.REVERSE)"
            )
        env = {"$g": g}
        for i, a in enumerate(node.args):
            env[f"$a{i}"] = a
        _accumulate(node, rules, env, grads)
    out: dict[str, Any] = {}
    for name, leaf in leaves_of(term).items():
        if leaf in grads:
            out[name] = grads[leaf]
    return out


def grad_program(
    term: Any,
    names: Iterable[str] | None = None,
    *,
    cotangent: Any = None,
    table: dict | None = None,
) -> dict[str, Any]:
    """Alias with an explicit leaf subset — the training surface."""
    grads = backward(term, cotangent=cotangent, table=table)
    if names is None:
        return grads
    return {n: grads[n] for n in names if n in grads}
