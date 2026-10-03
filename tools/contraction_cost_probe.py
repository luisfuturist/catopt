"""Minimal probe for contraction-order pricing in the diagram space.

Retro ``project/retros/contraction-precondition.md`` §3a claims the
diagram contraction-move space carries **no cost signal** for a
matrix-chain ordering, because the composed weight product is a
*param-only* subtree and :func:`catopt_core.cost.dag_cost` discounts
param-only subtrees that fold to a parameter ("compile-time work —
charged 0").

This probe reproduces that minimally, in three steps, with no torch
and no diagram run:

1. **The zeroing path.**  Build the composed weight product
   ``matmul(W3, … matmul(W1, W0))`` directly — a pure ``Param`` term —
   bracket it several ways, and price each under ``flops_cost`` /
   ``launch_aware_cost`` / ``count_cost``.  Every bracketing prices at
   ``0`` because :func:`_folds_to_param` returns True and
   :func:`dag_cost` skips it.

2. **The reified chain term.**  Wrap each bracketing in the
   activation ``linear(x, product)`` — the shape the diagram window
   reify produces (:func:`catopt_orchestrator.morphisms.reify._reify`
   over ``linear`` blocks).  All bracketings price identically: the
   one activation GEMM.  The product's FLOPs are removed twice over —
   the subtractive DAG decomposition *and* the fold.

3. **The invariant.**  Show the same term under a *raw* additive model
   that does NOT fold (``flops_cost`` on the subtree directly, no
   ``dag_cost``) DOES differ by bracketing — i.e. the signal exists in
   the term, it is the fold-aware wrapper that erases it.

4. **Expressible but not priceable.**  Saturate the product in an
   :class:`~catopt_core.egraph.EGraph` with ``assoc_matmul`` /
   ``assoc_matmul_rev`` — every bracketing lands in the root e-class
   (the space *can express* the order) — then show the shipped models'
   :meth:`extract_best` picks **arbitrarily** (the fold zeroes every
   member) and is suboptimal on most chains.  An opt-in
   ``charges_param_only`` FLOP model — the "dedicated contraction cost
   model" lever — recovers the DP optimum.

Run::

    .venv/bin/python tools/contraction_cost_probe.py
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, cast

from catopt_core.cost import (
    _CostMarkers,
    _folds_to_param,
    count_cost,
    dag_cost,
    flops_cost,
    launch_aware_cost,
)
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, Param, TensorType, Var
from catopt_core.laws import FULL

__all__ = [
    "MODELS",
    "bracketings",
    "chain_optimal",
    "linear_chain",
    "main",
    "price",
    "product",
]


#: The shipped fold-aware models the diagram driver prices through.
MODELS: tuple[tuple[str, object], ...] = (
    ("flops_cost", flops_cost),
    ("launch_aware_cost", launch_aware_cost),
    ("count_cost", count_cost),
)


def _p(name: str, rows: int, cols: int) -> Param:
    """Return a 2-D weight parameter leaf."""
    return Param(name, TensorType((rows, cols)))


def product(dims: tuple[int, ...], bracket: str) -> Op:
    """Compose ``W_{n-1}·…·W_1·W_0`` over ``Param`` leaves.

    ``dims`` is a matrix-chain shape ``(d0, d1, …, dn)``; ``W_i`` has
    shape ``(d_{i+1}, d_i)``.  ``bracket`` selects the association:

    * ``"left"``  — ``(((W_{n-1}·W_{n-2})·…)·W_0)``
    * ``"right"`` — ``(W_{n-1}·(W_{n-2}·(…·W_0)))``
    * ``"balanced"`` — split the run in half at each level.
    """
    ws = [
        _p(f"W{i}", dims[i + 1], dims[i]) for i in range(len(dims) - 1)
    ]

    def left(seq: list) -> Op:
        # ((W_{n-1}·W_{n-2})·…)·W_0 — the product order is decreasing.
        acc = seq[-1]
        for w in reversed(seq[:-1]):
            acc = Op.make("matmul", acc, w)
        return acc

    def right(seq: list) -> Op:
        # W_{n-1}·(W_{n-2}·(…·W_0)) — decreasing, right-nested.
        acc = seq[0]
        for w in seq[1:]:
            acc = Op.make("matmul", w, acc)
        return acc

    def balanced(seq: list) -> Op:
        if len(seq) == 1:
            return seq[0]
        mid = len(seq) // 2
        return Op.make(
            "matmul", balanced(seq[mid:]), balanced(seq[:mid])
        )

    if bracket == "left":
        return left(ws)
    if bracket == "right":
        return right(ws)
    if bracket == "balanced":
        return balanced(ws)
    raise ValueError(f"unknown bracket {bracket!r}")


def bracketings(dims: tuple[int, ...]) -> dict[str, Op]:
    """Return the distinct bracketings of the composed weight product."""
    return {b: product(dims, b) for b in ("left", "right", "balanced")}


def linear_chain(dims: tuple[int, ...], bracket: str) -> Op:
    """Return the reified chain term ``linear(x, <product>)``.

    ``x`` is a ``Var`` activation of shape ``(1, d0)`` — the shape the
    diagram window reify lowers against (a single ``Var`` leaf, so the
    activation GEMM is the only runtime work).
    """
    x = Var("x", TensorType((1, dims[0])))
    return Op.make("linear", x, product(dims, bracket))


def price(term: Op, cost_fn: object) -> float:
    """Fold-aware DAG price — what the diagram driver charges."""
    return dag_cost(term, cost_fn)


def raw_price(term: Op, cost_fn: Callable[[Any], float]) -> float:
    """Raw additive price — the term's own tree, no fold discount."""
    return float(cost_fn(term))


def chain_optimal(dims: tuple[int, ...]) -> float:
    """Exact min-cost parenthesization (the O(n^3) interval DP)."""
    n = len(dims) - 1
    m = [[0.0] * (n + 1) for _ in range(n + 1)]
    for span in range(2, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            m[i][j] = min(
                m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                for k in range(i, j)
            )
    return m[1][n]


def _bill_flops(term: Any, memo: dict | None = None) -> float:
    """Price *term* billing the param-only fold — the opt-in lever.

    Sets ``charges_param_only`` — the marker both zeroing sites read —
    so the weight product's FLOPs are charged instead of discounted.
    This is the "dedicated contraction cost model" lever: extraction
    can then rank the bracketings.
    """
    return flops_cost(term, memo)


_bill_flops.__name__ = "bill_flops"
cast(_CostMarkers, _bill_flops).charges_param_only = True
cast(_CostMarkers, _bill_flops).dag_exact = True


def _sweep(n: int, seed: int) -> tuple[int, int, int, int, int]:
    """Count extraction outcomes over *n* random chains.

    Returns ``(n, members, suboptimal_shipped, suboptimal_optin,
    worst_ratio_x100)`` — the shipped model vs the opt-in billing
    model, both seeded with the diagram's natural (right-nested)
    product fold.
    """
    rng = random.Random(seed)
    rules = FULL.named("assoc_matmul", "assoc_matmul_rev")
    members = 0
    bad_shipped = 0
    bad_optin = 0
    worst = 0.0
    for _ in range(n):
        dims = tuple(
            rng.randint(1, 40) for _ in range(rng.randint(5, 8))
        )
        term = product(dims, "right")
        opt = 2.0 * chain_optimal(dims)
        eg = EGraph()
        eid = eg.add_term(term)
        eg.run(rules, eid, max_iterations=20, max_nodes=100_000)
        members = max(members, len(eg._classes[eg.find(eid)].nodes))
        shipped = flops_cost(eg.extract_best(eid, flops_cost))
        optin = flops_cost(eg.extract_best(eid, _bill_flops))
        if shipped > opt + 1e-6:
            bad_shipped += 1
            worst = max(worst, shipped / opt)
        if optin > opt + 1e-6:
            bad_optin += 1
    return (n, members, bad_shipped, bad_optin, int(worst * 100))


def main() -> int:
    """Print the zeroing path, the invariant, and the raw contrast."""
    dims = (2, 3, 14, 14, 3, 8)
    print(f"matrix-chain dims: {dims}")
    print()

    print("== 1. the composed weight product alone (param-only) ==")
    print(
        f"  {'bracket':<10} " + "  ".join(f"{n:>16}" for n, _ in MODELS)
    )
    for b, term in bracketings(dims).items():
        folds = _folds_to_param(term, None, {})
        cells = "  ".join(
            f"{price(term, fn):>16.1f}" for _, fn in MODELS
        )
        print(f"  {b:<10} {cells}   folds_to_param={folds}")
    print("  → every bracketing is a compile-time fold: charged 0.")
    print()

    print("== 2. the reified chain term linear(x, product) ==")
    print(
        f"  {'bracket':<10} " + "  ".join(f"{n:>16}" for n, _ in MODELS)
    )
    for b in ("left", "right", "balanced"):
        term = linear_chain(dims, b)
        cells = "  ".join(
            f"{price(term, fn):>16.1f}" for _, fn in MODELS
        )
        print(f"  {b:<10} {cells}")
    print("  → the product's FLOPs vanish; only the activation GEMM")
    print("    (2·1·d0·d_n) + one launch survives.  No order signal.")
    print()

    print("== 3. contrast — the SAME terms priced raw (no fold) ==")
    print(
        f"  {'bracket':<10} {'flops_cost(raw)':>18} "
        f"{'count_cost(raw)':>18}"
    )
    for b, term in bracketings(dims).items():
        print(
            f"  {b:<10} {raw_price(term, flops_cost):>18.1f} "
            f"{raw_price(term, count_cost):>18.1f}"
        )
    print("  → the ordering signal is IN the term; the fold-aware")
    print(
        "    dag_cost (and the param-only extraction discount) erases"
    )
    print("    it.  The diagram space cannot see it.")
    print()

    print("== 4. expressible but not priceable (40 random chains) ==")
    n, members, bad_s, bad_o, worst = _sweep(40, seed=0)
    print(f"  bracketings in the root e-class (max): {members}")
    print(
        f"  shipped models: suboptimal on {bad_s}/{n} chains "
        f"(worst {worst / 100:.1f}x the DP optimum)"
    )
    print(
        f"  opt-in charges_param_only FLOP model: suboptimal on "
        f"{bad_o}/{n}"
    )
    print("  → the space CAN express the order (all bracketings are in")
    print("    the e-graph) but the shipped models cannot PRICE it —")
    print(
        "    extraction is arbitrary and usually suboptimal.  Billing"
    )
    print("    the fold (charges_param_only) recovers the DP optimum,")
    print("    but prices compile-time weight materialisation, not")
    print("    runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
