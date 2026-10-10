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
        env["$rcount"] = Const(_reduced_count(inshape, dims))
    if node.op == "permute" and isinstance(dims, (tuple, list)):
        env["$invdims"] = tuple(dims.index(i) for i in range(len(dims)))
    if node.op in ("expand", "broadcast_to"):
        env["$expdims"] = _expanded_dims(node, inshape)
    out_shape = _shape_of(node, memo)
    if isinstance(out_shape, tuple):
        env["$batchsum"] = _leading_sum(g, len(out_shape) - 1)
    return env


def _leading_sum(g: Any, ndim: int) -> Any:
    """``g`` summed over its first *ndim* dims (identity at ≤ 0)."""
    if ndim <= 0:
        return g
    return Op.make("sum", g, dim=tuple(range(ndim)), keepdim=False)


def _numel(shape: tuple) -> int:
    """Product of a shape tuple — the reduction element count."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _reduced_count(inshape: tuple, dims: Any) -> int:
    """Count the elements a ``mean`` reduction divided by."""
    if dims is None:
        return _numel(inshape)
    dd = dims if isinstance(dims, (tuple, list)) else (dims,)
    return _numel(tuple(inshape[int(d) % len(inshape)] for d in dd))


def _expanded_dims(node: Any, inshape: Any) -> tuple:
    """Dims an expand/broadcast grew (input 1-or-absent, target > 1)."""
    target = node.attrs.get("shape") or node.attrs.get("dim")
    if not isinstance(target, (tuple, list)) or not isinstance(
        inshape, tuple
    ):
        return ()
    pad = (1,) * (len(target) - len(inshape)) + tuple(inshape)
    return tuple(
        i
        for i, (t_, d_) in enumerate(zip(target, pad, strict=False))
        if d_ == 1 and t_ not in (1, -1)
    )


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


# ---------------------------------------------------------------------------
#  The training-graph probe — do the laws optimize backward programs?
# ---------------------------------------------------------------------------


def _demo_forwards() -> list:
    """Forward programs shaped like real model sites.

    Silu (SwiGLU's activation), a manual softmax
    (``exp(x)/sum(exp(x), keepdim)`` — attention's spine), and a
    matmul+bias+tanh linear block.  The keepdim spellings are the
    ones real lowering emits.
    """
    from catopt_core.ir import TensorType, Var

    x = Var("x", TensorType((4, 4)))
    w = Var("w", TensorType((4, 4)))
    b = Var("b", TensorType((4, 4)))
    return [
        ("silu_fwd", Op.make("mul", x, Op.make("sigmoid", x)), "x"),
        (
            "softmax_fwd",
            Op.make(
                "div",
                Op.make("exp", x),
                Op.make(
                    "sum",
                    Op.make("exp", x),
                    dim=1,
                    keepdim=True,
                ),
            ),
            "x",
        ),
        (
            "linear_fwd",
            Op.make(
                "tanh",
                Op.make("add", Op.make("matmul", x, w), b),
            ),
            "w",
        ),
    ]


def training_probe(
    *,
    cost_fn: Any = None,
    budget: int = 24,
    cotangents: dict | None = None,
) -> list[dict]:
    """Backward each demo forward; run the board on its gradient.

    Per case: the forward term's ``backward`` produces the gradient
    term; a ``MetaArena`` (ambient vocabulary, scripted
    saturate→extract) optimizes it.  Rows report the certified
    extraction cost vs the term's own baseline — the training-graph
    domain running on the *same* machinery, no new semantics.
    """
    from catopt_core.cost.basic import count_cost
    from catopt_core.cost.params import dag_cost

    from . import meta_arena as ma

    cf = cost_fn or count_cost
    rows: list[dict] = []
    for name, fwd, leaf in _demo_forwards():
        grads = backward(
            fwd,
            cotangent=(cotangents or {}).get(name),
        )
        if leaf not in grads:
            rows.append({"name": name, "declined": "no grad for leaf"})
            continue
        g = grads[leaf]
        try:
            arena = ma.MetaArena(
                g, supported=None, cost_fn=cf, max_specs=8
            )
        except Exception as exc:  # pragma: no cover — defensive
            rows.append({"name": name, "declined": str(exc)})
            continue
        base = ma._reported(dag_cost(g, arena.feasible_cost))
        arena.step(ma.Action.saturate(budget=512))
        _, rep = arena.step(ma.Action.extract())
        rows.append(
            {
                "name": name,
                "baseline": base,
                "cost": rep.cost if rep.applied else None,
                "cert": rep.certificate_ok if rep.applied else None,
                "n_enodes": arena.eg.n_enodes,
            }
        )
    return rows


def training_table(rows: list[dict]) -> str:
    """Render the training-graph probe rows."""
    head = f"{'forward':<14} {'baseline':>9} {'extract':>8} {'cert':>5} {'enodes':>7}"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "declined" in r:
            lines.append(f"{r['name']:<14} declined: {r['declined']}")
            continue
        cost = r["cost"]
        cc = f"{cost:>8.3f}" if isinstance(cost, float) else f"{'-':>8}"
        lines.append(
            f"{r['name']:<14} {r['baseline']:>9.3f} "
            f"{cc} {r['cert']!s:>5} {r['n_enodes']:>7}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  The joint probe — forward and backward on ONE e-graph
# ---------------------------------------------------------------------------


def _reachable(eg: Any, root: int) -> set[int]:
    """Canonical e-class ids reachable from *root*."""
    seen: set[int] = set()
    stack = [eg.find(root)]
    while stack:
        eid = stack.pop()
        if eid in seen:
            continue
        seen.add(eid)
        for n in eg.get_class(eid).nodes:
            stack.extend(eg.find(a) for a in n.children)
    return seen


def _join(terms: list) -> Any:
    """Wrap roots under one head for a shared-subtree ``dag_cost``."""
    return Op.make("joint", *terms, validate=False)


def _cotangent_for(fwd: Any) -> Any:
    """Shaped all-ones cotangent when the shape is known."""
    shape = _shape_of(fwd)
    if isinstance(shape, tuple) and shape:
        return Op.make("broadcast_to", Const(1.0), shape=shape)
    return Const(1.0)


def _saturate_extract(
    terms: list, rules: Any, cost_fn: Any, max_nodes: int
) -> tuple:
    """One e-graph, every term a root; saturate once, extract each."""
    from catopt_core.egraph import EGraph

    eg = EGraph()
    rids = [eg.add_term(t) for t in terms]
    eg.run(rules, rids[0], max_nodes=max_nodes)
    return eg, rids, [eg.extract_best(r, cost_fn) for r in rids]


def _joint_row(
    name: str, fwd: Any, cot: Any, cf: Any, rs: Any, max_nodes: int
) -> dict:
    """One forward: separate-pipeline vs joint-e-graph cost."""
    from catopt_core.cost.params import dag_cost

    try:
        grads = backward(fwd, cotangent=cot)
    except ValueError as exc:
        return {"name": name, "declined": str(exc)}
    g_terms = [grads[n] for n in sorted(grads)]
    if not g_terms:
        return {"name": name, "declined": "no leaf grads"}

    # separate: forward pipeline + gradient pipeline, billed apart
    eg_f, _, (f_ext,) = _saturate_extract([fwd], rs, cf, max_nodes)
    eg_g, _, g_ext = _saturate_extract(g_terms, rs, cf, max_nodes)
    sep = dag_cost(f_ext, cf) + dag_cost(_join(g_ext), cf)

    # joint: one e-graph over every root, one saturation
    eg, jrid, j_ext = _saturate_extract(
        [fwd, *g_terms], rs, cf, max_nodes
    )
    shared = len(
        _reachable(eg, jrid[0])
        & set().union(*(_reachable(eg, r) for r in jrid[1:]))
    )
    return {
        "name": name,
        "separate": sep,
        "joint": dag_cost(_join(j_ext), cf),
        "shared": shared,
        "enodes_sep": eg_f.n_enodes + eg_g.n_enodes,
        "enodes_joint": eg.n_enodes,
    }


def joint_probe(
    *,
    forwards: list | None = None,
    cost_fn: Any = None,
    rules: Any = None,
    cotangents: dict | None = None,
    max_nodes: int = 20_000,
) -> list[dict]:
    """Forward + gradients on ONE e-graph vs two — the boundary claim.

    Per case: ``backward`` derives every leaf's gradient term.  The
    *separate* arm is the autograd comparison — the forward saturates
    in one e-graph, the whole gradient program in a second, and the
    two extracted programs are billed independently (two pipelines
    cannot share work across the boundary).  The *joint* arm adds the
    forward and every gradient root to one e-graph and saturates
    once: identical subterms are the same e-class from the start,
    and any rewrite that makes a backward subterm equal to a forward
    one merges them.  ``dag_cost`` dedups shared subtrees within a
    program, so ``joint < separate`` is measured cross-boundary
    sharing — work autograd's two graphs must pay twice.
    """
    from catopt_core.cost.basic import count_cost
    from catopt_core.laws import DEFAULT

    cf = cost_fn or count_cost
    rs = DEFAULT if rules is None else rules
    cases = _demo_forwards() if forwards is None else forwards
    cots = cotangents or {}
    return [
        _joint_row(
            name,
            fwd,
            cots.get(name) or _cotangent_for(fwd),
            cf,
            rs,
            max_nodes,
        )
        for name, fwd, _leaf in cases
    ]


def joint_table(rows: list[dict]) -> str:
    """Render the joint-probe rows."""
    head = (
        f"{'forward':<14} {'separate':>9} {'joint':>8} {'shared':>7}"
        f" {'enodes s/j':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        if "declined" in r:
            lines.append(f"{r['name']:<14} declined: {r['declined']}")
            continue
        lines.append(
            f"{r['name']:<14} {r['separate']:>9.3f} {r['joint']:>8.3f} "
            f"{r['shared']:>7} "
            f"{r['enodes_sep']:>5}/{r['enodes_joint']:<5}"
        )
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    """Run the training-graph probe; print the table."""
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "backward() the demo forwards and optimize the "
            "derived gradient programs on the meta-arena board"
        )
    )
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument(
        "--joint",
        action="store_true",
        help="forward+gradients on ONE e-graph vs two pipelines",
    )
    args = ap.parse_args(argv)
    if args.joint:
        print(  # stdout-compat
            joint_table(joint_probe())
        )
        return 0
    print(  # stdout-compat
        training_table(training_probe(budget=args.budget))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
