"""Cost models for e-graph extraction.

The cost model assigns a scalar "cost" to a term, used by
EGraph.extract_best to find the minimum-cost representative.

Two models are provided:
* count_cost - counts the number of operations (simplest).
* flops_cost - estimates FLOPs using shape information.

For the "killer experiment", the FLOPs-based model matters: it rewards
the associativity / distributivity / naturality rewrites that produce
fewer total floating-point operations.
"""

from __future__ import annotations

from typing import Any

from catopt.ir import Op, Var, Const, Param


# ---------------------------------------------------------------------------
# Shape inference (lightweight)
# ---------------------------------------------------------------------------

def _shape_of(term: Any) -> tuple | None:
    """Best-effort shape inference for a term."""
    if isinstance(term, (Var, Param)):
        return term.typ.shape
    if isinstance(term, Const):
        return ()
    if isinstance(term, Op):
        return _infer_op_shape(term)
    return None


def _infer_op_shape(op: Op) -> tuple | None:
    shapes = [_shape_of(a) for a in op.args]
    if not shapes or any(s is None for s in shapes):
        if shapes and shapes[0] is not None:
            return shapes[0]
        return None
    match op.op:
        case "matmul":
            a, b = shapes[0], shapes[1]
            if len(a) >= 2 and len(b) >= 2:
                return a[:-1] + (b[-1],)
            return shapes[0]
        case "add" | "mul" | "sub":
            return shapes[0]
        case "square" | "neg" | "sigmoid" | "silu" | "tanh" | "gelu" | "rsqrt":
            return shapes[0]
        case "sum" | "mean":
            return ()
        case "transpose":
            return tuple(reversed(shapes[0])) if shapes[0] else shapes[0]
        case _:
            return shapes[0]


def _numel(shape) -> int:
    if shape is None:
        return 1
    result = 1
    for d in shape:
        result *= (d if d is not None else 1)
    return result


# ---------------------------------------------------------------------------
# Per-op FLOP weights
# ---------------------------------------------------------------------------

_OP_FLOPS: dict[str, int] = {
    "matmul": 2, "add": 1, "mul": 1, "sub": 1, "square": 1, "neg": 1,
    "sigmoid": 2, "silu": 3, "tanh": 2, "gelu": 3, "rsqrt": 2, "exp": 1,
    "sum": 1, "mean": 2, "max": 1,
    "transpose": 0, "reshape": 0, "broadcast": 0,
}


def flops_cost(term: Any) -> float:
    """Cost = estimated FLOPs using shape inference.

    For matmul, uses the standard 2*M*N*K formula.
    For element-wise ops, uses 1 FLOP per output element.
    """
    if isinstance(term, Op):
        shape = _infer_op_shape(term)
        n_out = _numel(shape)
        if term.op == "matmul":
            # Standard matmul: 2 * M * N * K
            shapes = [_shape_of(a) for a in term.args]
            if shapes and shapes[1] is not None:
                k_dim = shapes[1][-2] if len(shapes[1]) >= 2 else 1
                base = 2 * n_out * k_dim
            else:
                base = 2 * n_out
        else:
            base = _OP_FLOPS.get(term.op, 1) * n_out
        for arg in term.args:
            base += flops_cost(arg)
        return float(base)
    return 0.0


def count_cost(term: Any) -> float:
    """Cost = number of non-view operations in the term tree."""
    if isinstance(term, Op):
        n = 1 if term.op not in ("transpose", "reshape", "broadcast", "leaf") else 0
        for arg in term.args:
            n += count_cost(arg)
        return float(n)
    return 0.0


class CostModel:
    """A configurable cost model for EGraph.extract_best."""

    def __init__(
        self,
        op_weights: dict[str, float] | None = None,
        weight_coeff: float = 1.0,
        matmul_coeff: float = 2.0,
    ) -> None:
        self.op_weights = op_weights or _OP_FLOPS
        self.weight_coeff = weight_coeff
        self.matmul_coeff = matmul_coeff

    def __call__(self, term: Any) -> float:
        if isinstance(term, Op):
            shape = _infer_op_shape(term)
            n = _numel(shape)
            coeff = self.op_weights.get(term.op, self.weight_coeff)
            if term.op == "matmul":
                shapes = [_shape_of(a) for a in term.args]
                if shapes and shapes[1] is not None:
                    k_dim = shapes[1][-2] if len(shapes[1]) >= 2 else 1
                    base = 2 * n * k_dim
                else:
                    base = coeff * n
            else:
                base = coeff * n
            for arg in term.args:
                base += self(arg)
            return float(base)
        return 0.0
