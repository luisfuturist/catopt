"""Basic additive cost models: FLOPs, launches and op counts.

The extraction entry points that need only shape information and a
per-op FLOP weight — :func:`flops_cost`, :func:`launch_aware_cost`,
:func:`count_cost` — plus the per-op FLOP table (:data:`_OP_FLOPS`),
the view-op set (:data:`_VIEW_OPS`) and the per-node cost primitives
(:func:`_flops_of`, :func:`_local_cost`) every other cost module
builds on.  Also carries :class:`_CostMarkers`, the dynamic-marker
Protocol the ``*_for`` factories cast through.
"""
# ruff: noqa: RUF003 — math notation in comments

from __future__ import annotations

from typing import Any, Protocol, cast

from catopt_core.ir import Op
from catopt_core.typing import (
    _INVALID,
    _infer_op_shape,
    _numel,
    _shape_of,
)


class _CostMarkers(Protocol):
    """The dynamic per-model markers attached to a cost-fn object.

    ``EGraph.extract_best`` / ``dag_cost`` read ``charges_param_only``
    and ``dag_exact`` off the function via ``getattr``; the ``*_for``
    factories also hang a ``profile`` (and sometimes ``lowering`` /
    ``best_lowering``) off the closure for reporting.  These are extras
    on the function object, not :class:`~catopt_core.ports.CostFn`
    members, so the assignments below cast through this Protocol.
    """

    charges_param_only: bool
    dag_exact: bool
    profile: Any
    lowering: Any
    best_lowering: Any


# ---------------------------------------------------------------------------
# Per-op FLOP weights
# ---------------------------------------------------------------------------

_OP_FLOPS: dict[str, int] = {
    "matmul": 2,
    "add": 1,
    "mul": 1,
    "sub": 1,
    "div": 4,
    "square": 1,
    "sqrt": 2,
    "neg": 1,
    "pow": 2,
    "sigmoid": 2,
    "silu": 3,
    "tanh": 2,
    "gelu": 3,
    "rsqrt": 2,
    "exp": 1,
    "sum": 1,
    "mean": 2,
    "max": 1,
    "transpose": 0,
    "reshape": 0,
    "broadcast": 0,
    "linear": 2,
    # concat/chunk are wire juxtaposition / projection: pure data
    # movement, zero FLOPs.  On weights they are compile-time work.
    "concat": 0,
    "chunk": 0,
    "split": 0,
    # index_select is a gather: memory traffic, no arithmetic.
    "index_select": 0,
    # contiguous copies memory (0 FLOPs but real bandwidth — the
    # roofline model prices it); sdpa ~2*T work per output element.
    "contiguous": 0,
    "sdpa": 2,
    # Traced-monoidal ops (catopt_carriers.trace): wire juxtaposition and
    # constant morphisms carry no FLOPs; trace/inv are priced in
    # _flops_of directly (solve cost depends on usize).
    "bdiag": 0,
    "parl": 0,
    "eye": 0,
    "cswap": 0,
}

#: Ops that produce no kernel — true views or wire bookkeeping.
#: torch.split/chunk/transpose/reshape return views: no launch, no
#: memory traffic; their only runtime effect is the stride they leave
#: for consumers (priced via _STRIDE_PENALTY).  concat is NOT here: a
#: runtime cat() is a real copy kernel — it is only free when the whole
#: subtree is param-only (compile-time fold, handled by extraction).
_VIEW_OPS = {
    "transpose",
    "reshape",
    "broadcast",
    "chunk",
    "split",
    "leaf",
    "aff",
    "om",
    "aff_diag",
    # Constant morphisms (catopt_carriers.trace): zero-arg ops that
    # materialise a fixed matrix — compile-time constants,
    # like the carrier-packaging ops above.
    "eye",
    "cswap",
}

#: Small per-op penalty modeling kernel-launch / scheduling overhead.
#: Two forms can have identical FLOPs yet differ in kernel count (e.g.
#: one fused GEMM vs two half-size GEMMs); the penalty breaks such ties
#: deterministically toward fewer launches.
_LAUNCH_PENALTY = 1.0


#: Price of a provably ill-typed term.  Finite but so large it can
#: never win an extraction — the equivalence verifier is the last line
#: of defence, the cost model is the first.
_INVALID_COST = 1e15


def _flops_of(term: Op, memo: dict | None = None) -> float:
    """FLOP count of a single op node (excludes children)."""
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    n_out = _numel(shape)
    if term.op == "matmul":
        # Standard matmul: 2 * M * N * K
        shapes = [_shape_of(a, memo) for a in term.args]
        if (
            shapes
            and shapes[1] is not None
            and shapes[1] is not _INVALID
        ):
            k_dim = (
                shapes[1][-2]
                if len(shapes[1]) >= 2
                else (shapes[1][0] if len(shapes[1]) == 1 else 1)
            )
            return float(2 * n_out * k_dim)
        return float(2 * n_out)
    if term.op == "linear":
        # F.linear(x[..,in], W[out,in]) -> 2 * M * out * in
        shapes = [_shape_of(a, memo) for a in term.args]
        if shapes and shapes[0] is not None and len(shapes[0]) >= 1:
            return float(2 * n_out * shapes[0][-1])
        return float(2 * n_out)
    if term.op == "conv2d":
        # 2 * N * O * H' * W' * (C*kh*kw / groups)
        shapes = [_shape_of(a, memo) for a in term.args]
        w = shapes[1] if len(shapes) > 1 else None
        if (
            w is not None
            and w is not _INVALID
            and len(w) >= 4
            and all(isinstance(d, int) for d in w[1:4])
        ):
            w1 = cast("int", w[1])
            w2 = cast("int", w[2])
            w3 = cast("int", w[3])
            k = w1 * w2 * w3
            g = term.attrs.get("groups", 1)
            if isinstance(g, int) and g > 1:
                k //= g
            return float(2 * n_out * k)
        return float(2 * n_out)
    if term.op == "aff":
        # Packaging a pair — no runtime work.
        return 0.0
    if term.op == "aff_compose":
        # (A2,b2)∘(A1,b1) = (A2·A1, A2·b1 + b2): one d×d matmul,
        # one matvec, one add ≈ 2·d³ + O(d²) flops.
        shapes = [_shape_of(a, memo) for a in term.args]
        f = shapes[0] if shapes else None
        if (
            f is not None
            and f is not _INVALID
            and len(f) >= 2
            and isinstance(f[-1], int)
        ):
            return float(2 * n_out * f[-1])
        return float(2 * n_out)
    if term.op == "apply":
        # f·h + b: matvec + add ≈ 2·d² flops on a d-vector out.
        shapes = [_shape_of(a, memo) for a in term.args]
        f = shapes[0] if shapes else None
        if (
            f is not None
            and f is not _INVALID
            and len(f) >= 2
            and isinstance(f[-1], int)
        ):
            return float(2 * n_out * f[-1])
        return float(2 * n_out)
    if term.op == "aff_diag":
        # Packaging a diagonal pair — no runtime work.
        return 0.0
    if term.op == "affd_compose":
        # (a2,b2)∘(a1,b1) = (a2⊙a1, a2⊙b1 + b2): mul + mul + add, all
        # elementwise ≈ 3·n_out — O(d), not the dense carrier's O(d³).
        return float(3 * n_out)
    if term.op == "applyd":
        # f₀⊙h + f₁: mul + add ≈ 2·n_out.
        return float(2 * n_out)
    if term.op == "om":
        # Packaging the (m, l, a) triple — no runtime work.
        return 0.0
    if term.op == "om_elem":
        # rowmax + (s−m) + exp + rowsum ≈ 3·numel(s), plus the
        # exp(s−m) @ v GEMM at 2·n_out·K (K = scores' last dim).
        shapes = [_shape_of(a, memo) for a in term.args]
        s = shapes[0] if shapes else None
        k = (
            s[-1]
            if isinstance(s, tuple) and s and isinstance(s[-1], int)
            else 1
        )
        return float(2 * n_out * k + 3 * _numel(s))
    if term.op == "om_compose":
        # max + 2 rescale exps + 2 mul-adds per accumulator element.
        return float(6 * n_out)
    if term.op == "om_apply":
        # One division per output element.
        return float(n_out)
    if term.op == "sdpa":
        # attention: ~2 * (T * d + T * T) per head ≈ 2*T*max(d,T)*B*h
        shapes = [_shape_of(a, memo) for a in term.args]
        q = shapes[0] if shapes else None
        if q is not None and q is not _INVALID and len(q) >= 3:
            t_dim = q[-2]
            return float(4 * n_out * (t_dim or 1))
        return float(4 * n_out)
    if term.op == "trace":
        # LFT: one du×du solve (~2·du³) plus the resolvent projection
        # Q·(I−S)⁻¹·R on the dy×dx output (~2·du·n_out).
        u = term.attrs.get("usize", 0)
        du = sum(u) if isinstance(u, (list, tuple)) else u
        du = du if isinstance(du, int) else 0
        return float(2 * du * du * du + 2 * n_out * max(du, 1) + n_out)
    if term.op == "inv":
        # n×n inverse ≈ 2·n³ = 2·n_out^1.5 FLOPs.
        return float(2 * max(n_out, 1) ** 1.5)
    if term.op == "rms_norm":
        # aten.rms_norm(x, normalized_shape, weight, eps): the weight
        # operand is an in-kernel Hadamard gain — one extra
        # elementwise pass over the output vs the weightless form.
        return float((2 if len(term.args) > 1 else 1) * n_out)
    return float(_OP_FLOPS.get(term.op, 1) * n_out)


def _local_cost(
    term: Op, launch_penalty: float = 0.0, memo: dict | None = None
) -> float:
    """Cost contribution of a single op node (excludes children).

    = op FLOPs on the inferred output shape + launch penalty.
    """
    base = _flops_of(term, memo)
    if term.op not in _VIEW_OPS:
        base += launch_penalty
    return float(base)


def flops_cost(term: Any, memo: dict | None = None) -> float:
    """Cost = estimated FLOPs using shape inference.

    For matmul, uses the standard 2*M*N*K formula.
    For element-wise ops, uses 1 FLOP per output element.

    ``memo`` (content-keyed) makes repeated calls over a shared-subterm DAG
    linear instead of exponential; callers doing many evaluations
    (e.g. extraction) should pass a shared dict.
    """
    memo = {} if memo is None else memo
    key = term  # content-keyed: interned terms hash by structure
    ck = ("c", key)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_cost(term, memo=memo)
        for arg in term.args:
            base += flops_cost(arg, memo)
        memo[ck] = float(base)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0


def _memo_dispatch(cost_fn, memo: dict | None = None):
    """Bind *cost_fn* to a shared content-keyed memo, uniformly.

    Returns a one-argument ``term -> float`` callable conforming to the
    extraction CostFn convention: cost models that accept a ``memo``
    kwarg (the built-in models do) get the dict threaded in — a shared
    memo turns repeated evaluations over a shared-subterm DAG into a
    linear walk; models without one are called bare.  ``memo=None``
    creates a fresh dict kept inside the closure.

    This is the single place the ``inspect.signature`` probe lives —
    ``extract_best``, ``extract_paired`` and :func:`dag_cost` all wrap
    through here.
    """
    import inspect

    m = {} if memo is None else memo
    if "memo" in inspect.signature(cost_fn).parameters:
        return lambda t: cost_fn(t, memo=m)
    return lambda t: cost_fn(t)


def launch_aware_cost(term: Any, memo: dict | None = None) -> float:
    """flops_cost + _LAUNCH_PENALTY per non-view op.

    Two equivalent forms can have identical FLOPs yet differ in kernel
    count (one fused GEMM vs two half-size GEMMs).  The penalty breaks
    such ties deterministically toward fewer launches.  This is the
    default extraction cost in :func:`catopt_orchestrator.optimize.optimize_model`.
    """
    memo = {} if memo is None else memo
    ck = ("lc", term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_cost(term, _LAUNCH_PENALTY, memo=memo)
        for arg in term.args:
            base += launch_aware_cost(arg, memo)
        memo[ck] = float(base)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0


def count_cost(term: Any, memo: dict | None = None) -> float:
    """Cost = number of non-view operations in the term tree."""
    memo = {} if memo is None else memo
    ck = ("cc", term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        n = 0 if term.op in _VIEW_OPS else 1
        for arg in term.args:
            n += count_cost(arg, memo)
        memo[ck] = float(n)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0
