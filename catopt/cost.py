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


def _infer_op_shape(op: Op):
    shapes = [_shape_of(a) for a in op.args]
    if any(s is _INVALID for s in shapes):
        return _INVALID
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
            base = shapes[0]
            if base is None:
                return None
            d0 = op.attrs.get("arg1", op.attrs.get("dim0", -2))
            d1 = op.attrs.get("arg2", op.attrs.get("dim1", -1))
            d0, d1 = d0 % len(base), d1 % len(base)
            out = list(base)
            out[d0], out[d1] = out[d1], out[d0]
            return tuple(out)
        case "reshape":
            shape = op.attrs.get("shape")
            if shape is not None:
                return tuple(shape)
            return shapes[0]
        case "contiguous":
            return shapes[0]
        case "sdpa":
            # out has q's shape (B, h, T, d)
            return shapes[0]
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


#: Sentinel returned by shape inference when two shapes are PROVABLY
#: incompatible (e.g. broadcasting (out,in) against (B,T,1)).  Distinct
#: from ``None`` (merely unknown): ill-typed terms are poisonous — the
#: cost model must never prefer them.
_INVALID = "__invalid_shape__"


def _broadcast(a, b):
    """Broadcast two tensor shapes (numpy/PyTorch semantics).

    Returns _INVALID on a hard mismatch — an ill-typed term, which the
    cost model prices as near-infinite so extraction never selects it.
    """
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
            return _INVALID  # provably ill-typed
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
    # contiguous copies memory (0 FLOPs but real bandwidth — the
    # roofline model prices it); sdpa ~2*T work per output element.
    "contiguous": 0, "sdpa": 2,
}

#: Ops that produce no kernel — views or wire bookkeeping.  Exempt from
#: the launch penalty and from count_cost.
_VIEW_OPS = {"transpose", "reshape", "broadcast", "concat", "chunk", "leaf"}

#: Small per-op penalty modeling kernel-launch / scheduling overhead.
#: Two forms can have identical FLOPs yet differ in kernel count (e.g.
#: one fused GEMM vs two half-size GEMMs); the penalty breaks such ties
#: deterministically toward fewer launches.
_LAUNCH_PENALTY = 1.0


#: Price of a provably ill-typed term.  Finite but so large it can
#: never win an extraction — the equivalence verifier is the last line
#: of defence, the cost model is the first.
_INVALID_COST = 1e15


def _flops_of(term: Op) -> float:
    """FLOP count of a single op node (excludes children)."""
    shape = _infer_op_shape(term)
    if shape is _INVALID:
        return _INVALID_COST
    n_out = _numel(shape)
    if term.op == "matmul":
        # Standard matmul: 2 * M * N * K
        shapes = [_shape_of(a) for a in term.args]
        if shapes and shapes[1] is not None and shapes[1] is not _INVALID:
            k_dim = shapes[1][-2] if len(shapes[1]) >= 2 else 1
            return float(2 * n_out * k_dim)
        return float(2 * n_out)
    if term.op == "linear":
        # F.linear(x[..,in], W[out,in]) -> 2 * M * out * in
        shapes = [_shape_of(a) for a in term.args]
        if shapes and shapes[0] is not None and len(shapes[0]) >= 1:
            return float(2 * n_out * shapes[0][-1])
        return float(2 * n_out)
    if term.op == "sdpa":
        # attention: ~2 * (T * d + T * T) per head ≈ 2*T*max(d,T)*B*h
        shapes = [_shape_of(a) for a in term.args]
        q = shapes[0] if shapes else None
        if q is not None and q is not _INVALID and len(q) >= 3:
            t_dim = q[-2]
            return float(4 * n_out * (t_dim or 1))
        return float(4 * n_out)
    return float(_OP_FLOPS.get(term.op, 1) * n_out)


def _local_cost(term: Op, launch_penalty: float = 0.0) -> float:
    """Cost contribution of a single op node (excludes children).

    = op FLOPs on the inferred output shape + launch penalty.
    """
    base = _flops_of(term)
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


# ---------------------------------------------------------------------------
#  Roofline cost model — item: layout/memory-aware costing
# ---------------------------------------------------------------------------
#
# flops_cost can only see arithmetic.  It cannot express why the fused
# SwiGLU form is a wash on CPU: the single wide GEMM saves a launch, but
# its chunk projections hand *strided* views to the elementwise kernels,
# which are memory-bound and pay for the wasted bandwidth.  The roofline
# model prices each op as
#
#     max(flops / PEAK_FLOPS, bytes / PEAK_BW) + launch_time
#
# which captures both regimes: GEMMs are compute-bound (flops term wins),
# elementwise and copy ops are bandwidth-bound (bytes term wins), and
# non-contiguous chunk views multiply the bytes a consumer must move.

_PEAK_FLOPS = 2.0e11    # ~200 GFLOP/s (CPU-class, order-of-magnitude)
_PEAK_BW = 4.0e10       # ~40 GB/s DRAM bandwidth
_LAUNCH_S = 5.0e-6      # ~5 µs kernel-launch / scheduling overhead
_STRIDE_PENALTY = 2.0   # strided reads waste ~half of each cache line


def _is_strided(term: Any) -> bool:
    """True if *term* is a view whose elements are not contiguous.

    chunk on the LAST dim splits each row — consumers read with a row
    stride of 2x the logical row.  chunk on any other dim yields
    contiguous blocks.
    """
    if not (isinstance(term, Op) and term.op == "chunk"):
        return False
    s = _infer_op_shape(term)
    if not isinstance(s, tuple) or not s:
        return False
    dim = term.attrs.get("dim", -1) % len(s)
    return dim == len(s) - 1


def _bytes_of(term: Op) -> float:
    """Bytes moved by a single op: inputs read + output written (fp32)."""
    in_bytes = 0.0
    for a in term.args:
        n = _numel(_shape_of(a))
        w = _STRIDE_PENALTY if _is_strided(a) else 1.0
        in_bytes += n * 4.0 * w
    # view ops share storage with their input — no output write
    out_bytes = 0.0 if term.op in _VIEW_OPS else _numel(
        _infer_op_shape(term)) * 4.0
    return in_bytes + out_bytes


def _local_roofline(term: Op) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch."""
    shape = _infer_op_shape(term)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term)
    if flops >= _INVALID_COST:
        return _INVALID_COST
    compute_s = flops / _PEAK_FLOPS
    memory_s = _bytes_of(term) / _PEAK_BW
    launch = 0.0 if term.op in _VIEW_OPS else _LAUNCH_S
    return (max(compute_s, memory_s) + launch) * 1e9


def roofline_cost(term: Any) -> float:
    """Roofline cost in estimated nanoseconds (per-op, additive).

    max(flops/PEAK_FLOPS, bytes/PEAK_BW) + launch per op; view ops are
    free except for the strided-read penalty they impose on consumers.
    This is the honest model for questions like "does the fused GEMM
    pay?" — it answers differently at different batch sizes, which is
    what the measurements show.
    """
    if isinstance(term, Op):
        base = _local_roofline(term)
        for arg in term.args:
            base += roofline_cost(arg)
        return float(base)
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
