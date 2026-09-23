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
        case "add" | "mul" | "sub" | "div":
            # Element-wise ops broadcast: result is the broadcast shape,
            # not simply the first operand's shape.
            return _broadcast(shapes[0], shapes[1] if len(shapes) > 1 else None)
        case (
            "square" | "sqrt" | "neg" | "sigmoid" | "silu" | "tanh"
            | "gelu" | "rsqrt" | "exp"
        ):
            return shapes[0]
        case "pow":
            return shapes[0] if shapes and shapes[0] is not None else ()
        case "linear":
            # F.linear(x[..., in], W[out, in]) -> [..., out]
            if len(shapes) >= 2 and shapes[0] is not None and shapes[1] is not None:
                w = shapes[1]
                if len(w) >= 1:
                    return tuple(shapes[0][:-1]) + (w[0],)
            return shapes[0]
        case "sum" | "mean":
            # Honor keepdim/dim when available, else reduce to scalar.
            dim = op.attrs.get("dim", op.attrs.get("axis", None))
            keep = bool(op.attrs.get("keepdim", False))
            base = shapes[0]
            if base is None:
                return ()
            if dim is None:
                return ()
            dims = dim if isinstance(dim, (tuple, list)) else (dim,)
            ndim = len(base)
            norm = {d % ndim for d in dims}
            if keep:
                return tuple(
                    (1 if i in norm else d) for i, d in enumerate(base)
                )
            return tuple(d for i, d in enumerate(base) if i not in norm)
        case "transpose":
            return tuple(reversed(shapes[0])) if shapes[0] else shapes[0]
        case "concat":
            a, b = shapes[0], shapes[1]
            if a is None or b is None:
                return a
            dim = op.attrs.get("dim", 0) % len(a)
            out = list(a)
            out[dim] = (a[dim] or 0) + (b[dim] or 0)
            return tuple(out)
        case "chunk":
            base = shapes[0]
            if base is None:
                return None
            dim = op.attrs.get("dim", -1) % len(base)
            n = op.attrs.get("chunks", 2)
            out = list(base)
            out[dim] = (base[dim] or 0) // n
            return tuple(out)
        case _:
            return shapes[0]


def _broadcast(a, b):
    """Broadcast two tensor shapes (numpy/PyTorch semantics)."""
    if a is None:
        return b
    if b is None:
        return a
    ndim = max(len(a), len(b))
    a_pad = (1,) * (ndim - len(a)) + tuple(a)
    b_pad = (1,) * (ndim - len(b)) + tuple(b)
    out = []
    for da, db in zip(a_pad, b_pad):
        if da is None or db is None:
            out.append(None)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        else:
            out.append(None)  # shape error; treat as unknown
    return tuple(out)


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
    "matmul": 2, "add": 1, "mul": 1, "sub": 1, "div": 4, "square": 1,
    "sqrt": 2, "neg": 1, "pow": 2,
    "sigmoid": 2, "silu": 3, "tanh": 2, "gelu": 3, "rsqrt": 2, "exp": 1,
    "sum": 1, "mean": 2, "max": 1,
    "transpose": 0, "reshape": 0, "broadcast": 0, "linear": 2,
    # concat/chunk are wire juxtaposition / projection: pure data
    # movement, zero FLOPs.  On weights they are compile-time work.
    "concat": 0, "chunk": 0,
}

#: Ops that produce no kernel — views or wire bookkeeping.  Exempt from
#: the launch penalty and from count_cost.
_VIEW_OPS = {"transpose", "reshape", "broadcast", "concat", "chunk", "leaf"}

#: Small per-op penalty modeling kernel-launch / scheduling overhead.
#: Two forms can have identical FLOPs yet differ in kernel count (e.g.
#: one fused GEMM vs two half-size GEMMs); the penalty breaks such ties
#: deterministically toward fewer launches.
_LAUNCH_PENALTY = 1.0


def _local_cost(term: Op, launch_penalty: float = 0.0) -> float:
    """Cost contribution of a single op node (excludes children).

    = op FLOPs on the inferred output shape + launch penalty.
    """
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
    elif term.op == "linear":
        # F.linear(x[..,in], W[out,in]) -> 2 * M * out * in
        shapes = [_shape_of(a) for a in term.args]
        if shapes and shapes[0] is not None and len(shapes[0]) >= 1:
            base = 2 * n_out * shapes[0][-1]
        else:
            base = 2 * n_out
    else:
        base = _OP_FLOPS.get(term.op, 1) * n_out
    if term.op not in _VIEW_OPS:
        base += launch_penalty
    return float(base)


def flops_cost(term: Any) -> float:
    """Cost = estimated FLOPs using shape inference.

    For matmul, uses the standard 2*M*N*K formula.
    For element-wise ops, uses 1 FLOP per output element.
    """
    if isinstance(term, Op):
        base = _local_cost(term)
        for arg in term.args:
            base += flops_cost(arg)
        return float(base)
    return 0.0


def launch_aware_cost(term: Any) -> float:
    """flops_cost + _LAUNCH_PENALTY per non-view op.

    Two equivalent forms can have identical FLOPs yet differ in kernel
    count (one fused GEMM vs two half-size GEMMs).  The penalty breaks
    such ties deterministically toward fewer launches.  This is the
    default extraction cost in :func:`catopt.optimize.optimize_model`.
    """
    if isinstance(term, Op):
        base = _local_cost(term, _LAUNCH_PENALTY)
        for arg in term.args:
            base += launch_aware_cost(arg)
        return float(base)
    return 0.0


def count_cost(term: Any) -> float:
    """Cost = number of non-view operations in the term tree."""
    if isinstance(term, Op):
        n = 0 if term.op in _VIEW_OPS else 1
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
            elif term.op == "linear":
                # F.linear(x[.., in], W[out, in]) -> 2 * M * out * in
                shapes = [_shape_of(a) for a in term.args]
                if shapes and shapes[0] is not None and len(shapes[0]) >= 1:
                    base = 2 * n * shapes[0][-1]
                else:
                    base = 2 * n
            else:
                base = coeff * n
            for arg in term.args:
                base += self(arg)
            return float(base)
        return 0.0
