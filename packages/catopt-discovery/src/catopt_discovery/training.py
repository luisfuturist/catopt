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

from catopt_core.ir import Const, Op, Param, Var, op_repr
from catopt_core.typing import _shape_of

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
    attrs, and a dict element splices an attr map whose ``"$…"``
    values resolve through *env* (``"$shape:N"`` & co.).
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
            node_attrs = {
                k: env[v]
                if isinstance(v, str) and v.startswith("$")
                else v
                for k, v in e.items()
            }
        else:
            args.append(_inst_grad(e, env, attrs))
    return Op.make(op, *args, **node_attrs)


def _row_for(node: Any, row: Any) -> dict | None:
    """Pick the first VJP row of *row* whose ``requires`` hold.

    ``requires`` entries are ``(attr, value)`` — ``"$ABSENT"`` means
    the attr must be missing, else the attr must equal *value*.  A
    plain tuple row normalises to one unconditional entry.
    """
    rows = row if isinstance(row, list) else [row]
    for r in rows:
        if not r:
            continue  # no row data = no VJP for this op
        r = r if isinstance(r, dict) else {"args": r}
        ok = True
        for attr, want in r.get("requires", ()):
            have = node.attrs.get(attr, "$ABSENT")
            if (want == "$ABSENT") != (have == "$ABSENT") or (
                want != "$ABSENT" and have != want
            ):
                ok = False
                break
        if ok:
            return r
    return None


def _node_env(node: Any, g: Any, memo: dict) -> dict:
    """Build the node's VJP env — args, shapes, and the splices.

    ``$shape:N`` is arg N's inferred shape (``_shape_of``); an op
    whose row references a shape an input lacks declines at
    instantiation.  ``$rcount`` counts the elements a ``mean``
    reduction divided by (full-reduce → numel, else the listed
    dims).  ``$invdims`` is a ``permute``'s inverse permutation.
    """
    env = {"$g": g}
    for i, a in enumerate(node.args):
        env[f"$a{i}"] = a
        shape = _shape_of(a, memo)
        if isinstance(shape, tuple):
            env[f"$shape:{i}"] = shape
    dims = node.attrs.get("dim", node.attrs.get("dims"))
    inshape = env.get("$shape:0")
    if isinstance(inshape, tuple):
        if dims is None:
            env["$rcount"] = Const(_numel(inshape))
        else:
            dd = dims if isinstance(dims, (tuple, list)) else (dims,)
            env["$rcount"] = Const(
                _numel(
                    tuple(inshape[int(d) % len(inshape)] for d in dd)
                )
            )
    if node.op == "permute" and isinstance(dims, (tuple, list)):
        env["$invdims"] = tuple(dims.index(i) for i in range(len(dims)))
    return env


def _numel(shape: tuple) -> int:
    """Product of a shape tuple — the reduction element count."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _accumulate(
    node: Any, rules: tuple, env: dict, grads: dict
) -> None:
    """Route one node's cotangent to its children per *rules*."""
    for child, spec in zip(node.args, rules, strict=False):
        if spec is None:
            continue
        cg = _inst_grad(spec, env, node.attrs)
        grads[child] = (
            Op.make("add", grads[child], cg) if child in grads else cg
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
    shape_memo: dict = {}
    order: list = []
    _topo(term, set(), order)
    grads: dict[Any, Any] = {term: g0}
    for node in reversed(order):
        g = grads.get(node)
        if g is None:
            continue
        row = _row_for(node, vjp.get(node.op, ()))
        if row is None:
            raise ValueError(
                f"backward: no applicable VJP row for "
                f"{op_repr(node)} — the reverse handler declines "
                "(add one to lawdata.REVERSE)"
            )
        env = _node_env(node, g, shape_memo)
        _accumulate(node, row["args"], env, grads)
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
