"""Cost models for e-graph extraction.

The cost model assigns a scalar "cost" to a term, used by
EGraph.extract_best to find the minimum-cost representative.

Models provided include:
* count_cost - counts the number of operations (simplest).
* flops_cost - estimates FLOPs using shape information.
* param_bytes_cost - counts stored parameter values (the storage
  axis; what lets extraction prefer weight-sharing members).
* executor_overhead / executor_cost_for - price the LOWERING:
  the serial evaluator's per-node dispatch+kernel time, the level-
  batched executors' leaf-gather/level-compose schedule, the compiled
  executor's fusion regions.
* fusion_regions - partition a term's op-DAG into Inductor-style
  pointwise-fusion regions (one region = one compiled kernel).
* fused_cost_for - the compiled lowering's price: each fusion region
  costs one launch + the max of its summed member FLOPs vs its
  external/boundary memory traffic; one dispatch per compiled graph.
* lowering_aware_cost_for - min over lowerings: extraction picks the
  term whose best lowering is cheapest.

For the "killer experiment", the FLOPs-based model matters: it rewards
the associativity / distributivity / naturality rewrites that produce
fewer total floating-point operations.
"""
# ruff: noqa: RUF002, RUF003 — math notation in comments

from __future__ import annotations

import math
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from catopt_core.ir import Const, Op, Param

if TYPE_CHECKING:
    from catopt_core.ports import CostFn

# compat: moved to catopt_core.typing — the shape/type-inference layer owns
# itself now; re-exported here so existing `from catopt_core.cost import
# _shape_of / _INVALID / _broadcast / _numel / _infer_op_shape` callers
# keep working.  All but ``_broadcast`` are also used internally below.
from catopt_core.typing import (  # noqa: F401
    _INVALID,
    _broadcast,
    _infer_op_shape,
    _numel,
    _shape_of,
    has_var_leaf,
)

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
            k = w[1] * w[2] * w[3]
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


def dag_cost(term: Any, cost_fn, memo: dict | None = None) -> float:
    """True DAG cost of an extracted term: shared subtrees charged once.

    ``cost_fn(term)`` counts shared subtrees once per *parent* (a tree
    walk); extracted terms can share Op objects when two e-class parents
    picked terms over the same e-class (e.g. one fused GEMM under two
    split views).  This sums each distinct node's local cost —
    ``cost_fn(node) - sum(cost_fn(children))`` — deduplicated by object
    identity.

    ``memo`` is forwarded to cost functions that accept it (the
    built-in models do), so children already costed are O(1) lookups.

    Cost models that price parameter *storage* (``param_bytes_cost``)
    opt out of the param-only discount by setting
    ``charges_param_only`` on the function: a folded subtree still
    stores its leaves' values.  Models that already index the whole
    term DAG (``param_bytes_cost`` again — by leaf name plus
    materialised-subtree identity) set ``dag_exact`` instead: the
    function's value on the root IS the DAG cost, so the subtractive
    per-node decomposition below is skipped — it would mis-bill them.
    """
    # getattr(..., "func", ...) unwraps functools.partial bindings.
    bill_params = getattr(
        getattr(cost_fn, "func", cost_fn), "charges_param_only", False
    )
    c = _memo_dispatch(cost_fn, memo)

    # Cost models that already index the whole term DAG — by leaf name
    # and by materialised-subtree identity (``param_bytes_cost``) —
    # compute the true DAG cost directly.  The per-node decomposition
    # ``local = c(t) − Σc(children)`` below would mis-bill them: a leaf
    # nested inside a folded subtree is subtracted at every ancestor
    # fold yet charged once as a leaf, erasing exactly the materialised
    # copies the model exists to count.
    if getattr(getattr(cost_fn, "func", cost_fn), "dag_exact", False):
        return c(term)

    seen: set[int] = set()
    total = 0.0

    var_memo: dict = {}

    def has_var(t: Any) -> bool:
        """True if the subtree reads a data input (Var leaf).

        Subtrees over only Param/Const leaves are compile-time work —
        lowering folds them into a materialised parameter — so they are
        charged 0, matching extract_best's param-only discount.
        Delegates to :func:`catopt_core.typing.has_var_leaf` (the shared
        implementation) with this DAG walk's private memo.
        """
        return has_var_leaf(t, var_memo)

    fold_memo: dict = {}

    def rec(t: Any) -> None:
        nonlocal total
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Op):
            if (
                not has_var(t)
                and not bill_params
                and _folds_to_param(t, None, fold_memo)
            ):
                return  # folds at compile time — free at runtime
            # Param-only but NOT foldable (e.g. a ``trace`` resolvent's
            # hidden linalg.solve) still runs every call — bill it.
            for a in t.args:
                rec(a)
            local = c(t) - sum(c(a) for a in t.args)
            total += max(local, 0.0)
        else:
            total += c(t)

    rec(term)
    return total


def launch_aware_cost(term: Any, memo: dict | None = None) -> float:
    """flops_cost + _LAUNCH_PENALTY per non-view op.

    Two equivalent forms can have identical FLOPs yet differ in kernel
    count (one fused GEMM vs two half-size GEMMs).  The penalty breaks
    such ties deterministically toward fewer launches.  This is the
    default extraction cost in :func:`catopt_optimize.optimize.optimize_model`.
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


def depth_cost(term: Any, memo: dict | None = None) -> float:
    """Critical-path cost: the longest dependency chain in seconds.

    Each op's latency is its roofline time (max(flops/peak, bytes/bw)
    + launch); the term's cost is local latency + max child depth.
    Work-preserving reassociations (parallel scans, balanced sums,
    repeated squaring) win here even when total FLOPs are identical —
    this is the axis on which a sequential recurrence and its
    log-depth Blelloch form differ."""
    memo = {} if memo is None else memo
    ck = ("dc", term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        local = _local_roofline(term, memo=memo)
        if local >= _INVALID_COST:
            # Unshapeable op: charge a launch, not a veto — depth is a
            # structural metric, not a soundness gate.
            local = _LAUNCH_S * 1e9
        child = max(
            (depth_cost(a, memo) for a in term.args), default=0.0
        )
        out = local + child
        memo[ck] = float(out)
        return out
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


# ---------------------------------------------------------------------------
#  Parameter-storage cost model — the memory axis's pricing side
# ---------------------------------------------------------------------------


def param_bytes_cost(
    term: Any,
    source_tensors: dict | None = None,
    memo: dict | None = None,
    by_bytes: bool = False,
) -> float:
    """Cost = stored parameter values — what the LOWERED module keeps.

    The storage axis the flop-based models cannot see: a shared
    (tied) weight or a deduplicated head-stack computes the same
    function from fewer stored scalars.  The unit is *values* (bytes
    at unit width — multiply by dtype size for true bytes); derived
    params registered by the sharing passes count normally, which is
    what lets extraction prefer the storage-cheaper member.

    Pricing mirrors ``IRModule._fold_weight_chains`` /
    ``_build_params`` (catopt_torch.torch_bridge), not the term's leaf list:

    * a ``Param`` leaf reachable after folding is one storage entry,
      deduplicated *by name* — a weight read by two consumers is
      stored once (two ``Param`` objects spelled identically are the
      same leaf in the e-graph anyway: ``repr(Param)`` is the name);
    * a param-only subtree the lowerer materialises (``matmul`` over
      stored weights, elementwise ops, ``concat`` — see
      ``_folds_to_param``) is billed at the fold's OUTPUT numel, once
      per subtree object.  Inside such a fold, occurrences are copies,
      not reads: ``concat(W, W)`` stores ``2·numel(W)`` and a folded
      ``matmul(B, A)`` stores ``numel(B@A)`` — billing the deduped
      leaf names would price phantom storage the weights file never
      had (and hide copies it does).

    A Param's numel comes from ``source_tensors[name]`` when the name
    resolves there — the actual tensor, authoritative for derived
    params — else from its ``TensorType`` (unknown dims
    count 1, matching ``_numel``'s best-effort convention).

    Follows the standard ``(term, memo=None)`` cost-fn convention;
    ``source_tensors`` is bound with :func:`param_bytes_cost_for` (or
    ``functools.partial(param_bytes_cost, source_tensors=...)``) for
    use in :meth:`EGraph.extract_best`.  The model sets
    ``charges_param_only`` so extraction does NOT apply the param-only
    discount — a compile-time-folded subtree still stores values —
    and ``dag_exact`` so :func:`dag_cost` returns this DAG-indexed sum
    verbatim instead of re-deriving it per node.
    """
    memo = {} if memo is None else memo
    return float(
        sum(_param_index(term, source_tensors, memo, by_bytes).values())
    )


# Markers read by EGraph.extract_best / dag_cost: storage pricing does
# not fold away at compile time, so param-only subtrees stay billed;
# and the leaf/fold index is already a true DAG cost, so dag_cost
# must not apply its subtractive per-node decomposition.
param_bytes_cost.charges_param_only = True
param_bytes_cost.dag_exact = True


def param_bytes_cost_for(
    source_tensors: dict | None = None, by_bytes: bool = False
):
    """Bind ``source_tensors`` and return a standard cost fn.

    Same closure convention as :func:`roofline_cost_for`: the result
    has signature ``fn(term, memo=None)`` — dropping straight into
    ``EGraph.extract_best`` / ``dag_cost`` — and carries the
    ``charges_param_only`` marker through to extraction.  ``by_bytes``
    weights each stored value by its dtype width — the axis under
    which quantized params price below fp32.
    """

    def cost(term: Any, memo: dict | None = None) -> float:
        return param_bytes_cost(term, source_tensors, memo, by_bytes)

    cost.__name__ = "param_bytes_cost_for"
    cost.charges_param_only = True
    cost.dag_exact = True
    return cost


def _param_numel(
    p: Param, source_tensors: dict | None, by_bytes: bool = False
) -> float:
    """Stored scalar count for one Param leaf.

    ``source_tensors`` (name -> tensor, e.g. from ``export_to_ir`` plus
    any derived params a pass injected) is authoritative when the
    name resolves there; else the declared ``TensorType``.  With
    ``by_bytes`` the count is weighted by the stored dtype's width
    (``numel * element_size``) — the axis under which int8 quantization
    prices 4× below fp32; unknown widths default to 4 bytes.
    """
    if source_tensors is not None:
        t = source_tensors.get(p.name)
        if t is not None:
            n = getattr(t, "numel", None)
            if callable(n):
                numel = float(n())
                if by_bytes:
                    esz = getattr(t, "element_size", None)
                    return numel * (esz() if callable(esz) else 4)
                return numel  # torch.Tensor / jax / etc.
            s = getattr(t, "size", None)
            if isinstance(s, (int, float)):
                return float(s)  # numpy .size
    shape = getattr(getattr(p, "typ", None), "shape", None)
    return float(_numel(shape))


#: Elementwise ops ``IRModule._fold_weight_chains`` (catopt_torch.torch_bridge)
#: folds eagerly through the ``_IR_TO_TORCH`` bindings when the whole
#: subtree is param-only.  ``matmul`` and ``concat`` are spelled out
#: separately in ``_folds_to_param`` because they fold under tighter arg
#: rules (real tensor operands, not Consts).  Keep in lock-step.
_FOLDABLE_ELEMWISE = frozenset(
    {
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "square",
        "sqrt",
        "sigmoid",
        "silu",
        "tanh",
        "gelu",
        "exp",
        "pow",
    }
)


def _has_var_leaf(term: Any, memo: dict) -> bool:
    """True iff the subtree reads a data input (Var leaf).

    Private alias kept for this module's fold walkers; delegates to
    :func:`catopt_core.typing.has_var_leaf` — the single implementation —
    which uses the same ``("hv", term)`` key convention, so the memo
    this shares with ``_shape_of``/``_folds_to_param`` keeps its exact
    key/value contents.
    """
    return has_var_leaf(term, memo)


def _param_resolves(p: Param, source_tensors: dict | None) -> bool:
    """Would ``p.name`` land in ``_param_values`` at lowering?

    ``optimize_model`` hands the whole ``source_tensors`` dict to the
    lowerer as ``param_values`` (the sharing passes register their
    derived names into it), so an unbound ``source_tensors`` —
    ``param_bytes_cost_for()`` — assumes every leaf resolves.  With a
    bound dict the check is exact: a leaf absent from it cannot fold
    (it is still stored — ``_build_params`` registers it regardless).
    """
    return source_tensors is None or p.name in source_tensors


def _folds_to_param(
    term: Any, source_tensors: dict | None, memo: dict
) -> bool:
    """True iff ``_fold_weight_chains`` rewrites *term* to a fused Param.

    Mirrors the lowerer bottom-up: a param-only subtree folds when
    every argument reduces to a stored parameter — a resolvable
    ``Param`` leaf or a subtree that itself folds (folded intermediates
    are registered into ``_param_values`` before their parents are
    considered).  ``Const`` operands are allowed only where the
    lowering accepts them: elementwise ops, but not the ``matmul``
    two-Param fold nor ``concat`` (``torch.cat`` has no scalar form).
    """
    if not isinstance(term, Op):
        return False
    k = ("pf", term)
    hit = memo.get(k)
    if hit is not None:
        return hit
    res = False
    if not _has_var_leaf(term, memo):

        def to_param(a: Any, allow_const: bool) -> bool:
            if isinstance(a, Param):
                return _param_resolves(a, source_tensors)
            if isinstance(a, Const):
                return allow_const
            return _folds_to_param(a, source_tensors, memo)

        if term.op in ("matmul", "concat") and len(term.args) >= 2:
            res = all(to_param(a, False) for a in term.args)
        elif term.op in _FOLDABLE_ELEMWISE:
            res = all(to_param(a, True) for a in term.args)
    memo[k] = res
    return res


def _fold_ewidth(
    term: Any, source_tensors: dict | None
) -> float | None:
    """Element width of a materialised fold — the widest resolvable
    leaf's dtype (the fused tensor inherits arg dtypes); ``None`` when
    no leaf carries one, letting the caller default to fp32."""
    if isinstance(term, Param):
        if source_tensors is not None:
            t = source_tensors.get(term.name)
            esz = getattr(t, "element_size", None)
            if callable(esz):
                return float(esz())
        return 4.0
    if isinstance(term, Op):
        ws = [
            w
            for a in term.args
            if (w := _fold_ewidth(a, source_tensors)) is not None
        ]
        return max(ws) if ws else None
    return None


def _fold_numel(
    term: Op, source_tensors: dict | None, memo: dict, by_bytes: bool
) -> float:
    """Stored size of the tensor a folding subtree materialises to —
    the OUTPUT numel: ``concat`` re-stores every argument's rows, a
    weight ``matmul`` stores the dense product.
    """
    n = float(_numel(_shape_of(term, memo)))
    if by_bytes:
        n *= _fold_ewidth(term, source_tensors) or 4.0
    return n


def _param_index(
    term: Any,
    source_tensors: dict | None,
    memo: dict,
    by_bytes: bool = False,
) -> dict[str, float]:
    """``{key: numel}`` for every *stored* parameter entry in a term DAG.

    Two kinds of entries, mirroring the lowered weights file:

    * ``{param_name: numel}`` — a Param leaf that survives folding;
      deduped by name, so a weight read by several consumers (or by
      several identically-spelled leaves) is stored once;
    * ``{"\\x00fold:<id>": out_numel}`` — a param-only subtree the
      lowerer materialises (``_folds_to_param``); keyed by subtree
      object identity, matching ``_fold_memo``/``_build_params``: the
      same object reached twice is one stored tensor, and each
      materialisation is billed at output size regardless of how the
      leaves underneath dedup.
    """
    key = ("pbi", term)
    hit = memo.get(key)
    if hit is not None:
        return hit
    if isinstance(term, Param):
        out = {term.name: _param_numel(term, source_tensors, by_bytes)}
    elif isinstance(term, Op):
        if _folds_to_param(term, source_tensors, memo):
            out = {
                ("\x00fold", term): _fold_numel(
                    term, source_tensors, memo, by_bytes
                )
            }
        else:
            out = {}
            for a in term.args:
                for n, v in _param_index(
                    a, source_tensors, memo, by_bytes
                ).items():
                    out.setdefault(n, v)
    else:
        out = {}
    memo[key] = out
    return out


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

# Calibrated on the dev GPU (RTX 2050 mobile, fp32): measured sustained
# matmul throughput ~2.5 TFLOPS, device copy bandwidth ~89 GB/s, eager
# kernel-launch overhead ~8.7 µs.  Raw strided copies measured ~1.0x
# (no penalty), so _STRIDE_PENALTY is set to 1.0 — the *real* cost of a
# strided view is not slower reads but forced materialisation when a
# layout-strict consumer (e.g. SDPA) needs contiguous input, priced in
# _local_roofline as an extra copy kernel.
_PEAK_FLOPS = 2.5e12  # measured: ~2.5 TFLOPS fp32 GEMM (RTX 2050)
_PEAK_BW = 8.9e10  # measured: ~89 GB/s copy bandwidth
_LAUNCH_S = 8.7e-6  # measured: ~8.7 µs eager launch overhead
_STRIDE_PENALTY = 1.0  # measured: strided copies ~1.0x on this GPU


def _is_strided(term: Any, memo: dict | None = None) -> bool:
    """True if *term* is a view whose elements are not contiguous.

    chunk on the LAST dim splits each row — consumers read with a row
    stride of 2x the logical row.  chunk on any other dim yields
    contiguous blocks.
    """
    if not (isinstance(term, Op) and term.op in ("chunk", "split")):
        return False
    s = _infer_op_shape(term, memo)
    if not isinstance(s, tuple) or not s:
        return False
    dim = term.attrs.get("dim", -1) % len(s)
    return dim == len(s) - 1


def _bytes_of(term: Op, memo: dict | None = None) -> float:
    """Bytes moved by a single op: inputs read + output written (fp32)."""
    in_bytes = 0.0
    for a in term.args:
        n = _numel(_shape_of(a, memo))
        w = _STRIDE_PENALTY if _is_strided(a, memo) else 1.0
        in_bytes += n * 4.0 * w
    # view ops share storage with their input — no output write
    out_bytes = (
        0.0
        if term.op in _VIEW_OPS
        else _numel(_infer_op_shape(term, memo)) * 4.0
    )
    return in_bytes + out_bytes


def _local_roofline(
    term: Op,
    memo: dict | None = None,
    *,
    peak_flops: float = _PEAK_FLOPS,
    peak_bw: float = _PEAK_BW,
    launch_s: float = _LAUNCH_S,
    kernel_ns=None,
) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch.

    ``kernel_ns`` (a ``_kernel_lookup`` callable) floors the estimate
    at the op's measured kernel time when the profile carries an
    ``op_kernel_ns`` table — see the "measured per-op kernel table"
    section above.
    """
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term, memo)
    if (
        flops >= _INVALID_COST
    ):  # pragma: no cover — INVALID shapes checked above
        return _INVALID_COST
    compute_s = flops / peak_flops
    memory_s = _bytes_of(term, memo) / peak_bw
    launch = 0.0 if term.op in _VIEW_OPS else launch_s
    # True views emit no kernel: no launch AND no memory traffic — the
    # read happens at the consumer, priced there via _STRIDE_PENALTY.
    if term.op in _VIEW_OPS:
        return 0.0
    base = (max(compute_s, memory_s) + launch) * 1e9
    if kernel_ns is not None:
        measured = kernel_ns(term, memo)
        if measured is not None:
            base = max(base, measured)
    return base


def _roofline_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    kernel_ns=None,
) -> float:
    """Shared traversal for roofline_cost and roofline_cost_for.

    The memo key carries the constants so two profiles can share a memo
    dict (e.g. inside dag_cost) without colliding.  When a measured
    kernel table is bound, the callable's identity joins the key —
    keeping the table-less ``("rc",pf,bw,ls,term)`` form intact for
    the extraction fast-path's pre-seeded entries (egraph/extract.py).
    """
    ck = (
        ("rc", peak_flops, peak_bw, launch_s, term)
        if kernel_ns is None
        else ("rc", peak_flops, peak_bw, launch_s, id(kernel_ns), term)
    )
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_roofline(
            term,
            memo,
            peak_flops=peak_flops,
            peak_bw=peak_bw,
            launch_s=launch_s,
            kernel_ns=kernel_ns,
        )
        for arg in term.args:
            base += _roofline_cost(
                arg, memo, peak_flops, peak_bw, launch_s, kernel_ns
            )
        memo[ck] = float(base)
        return memo[ck]
    memo[ck] = 0.0
    return 0.0


def roofline_cost(term: Any, memo: dict | None = None) -> float:
    """Roofline cost in estimated nanoseconds (per-op, additive).

    max(flops/PEAK_FLOPS, bytes/PEAK_BW) + launch per op; view ops are
    free except for the strided-read penalty they impose on consumers.
    This is the honest model for questions like "does the fused GEMM
    pay?" — it answers differently at different batch sizes, which is
    what the measurements show.

    The constants are the RTX 2050 profile hardcoded above; use
    :func:`roofline_cost_for` with a measured ``TargetProfile``
    (``catopt_optimize.calibrate.calibrate``) for other targets.
    """
    memo = {} if memo is None else memo
    return _roofline_cost(term, memo, _PEAK_FLOPS, _PEAK_BW, _LAUNCH_S)


def _profile_constants(profile: Any) -> tuple[float, float, float]:
    """(peak_flops, peak_bw, launch_s) from a TargetProfile-like object.

    Accepts anything with ``.tflops`` / ``.gbps`` / ``.launch_us``
    attributes (e.g. ``catopt_optimize.calibrate.TargetProfile``) or a dict with
    those keys; ``None`` yields the built-in RTX 2050 constants.
    """
    if profile is None:
        return _PEAK_FLOPS, _PEAK_BW, _LAUNCH_S
    if isinstance(profile, dict):
        get = profile.__getitem__
    else:
        get = lambda k: getattr(profile, k)  # noqa: E731
    return (
        float(get("tflops")) * 1e12,
        float(get("gbps")) * 1e9,
        float(get("launch_us")) * 1e-6,
    )


# ---------------------------------------------------------------------------
#  Measured per-op kernel table — the shape-dependent floor
# ---------------------------------------------------------------------------
#
# The roofline prices an op as max(flops/peak, bytes/bw) + launch —
# blind to how a real kernel's time scales with its shape (BLAS
# efficiency curves, cache-resident bandwidth, gather costs).  A
# calibrated profile carries ``op_kernel_ns``:
# ``{op_class: {shape_key: measured_ns}}`` where each value is the
# median wall time of one eager kernel call at that shape — launch,
# dispatch and kernel work inside (``catopt_optimize.calibrate``).
# A term whose op-class and shape signature lands near a measured
# bucket prices at ``max(roofline_ns, measured_ns)``: the measurement
# is a floor on the estimate — it can only raise the model toward the
# observed latency, never undercut it (the fidelity sweep's failure
# mode is under-prediction, so the asymmetric correction is the safe
# direction).

#: Reduction-style ops whose measured table class is ``"reduce"``,
#: bucketed by input element count (the traffic the kernel streams).
_MEASURED_REDUCE_OPS = frozenset(
    {"sum", "mean", "max", "min", "prod", "softmax"}
)

#: Gather ops whose measured table class is ``"index_select"``,
#: bucketed by output element count.
_MEASURED_GATHER_OPS = frozenset({"index_select", "embedding"})


def _mm_signature(
    term: Op, out_shape: tuple, memo: dict
) -> tuple[float, float, float] | None:
    """(M, K, N) signature for a matmul/linear op, or ``None``.

    ``M`` is the collapsed row count (``n_out / N`` — batch and row
    dims together), ``N`` the output's last dim, ``K`` the reduction
    dim of the weight argument (``w[-2]`` for matmul, ``w[-1]`` =
    in-features for linear).  Anything unshapeable — a rank-1 result
    (matvec/dot), a missing or dimensionless weight — yields ``None``
    and the op stays on the roofline.
    """
    if len(out_shape) < 2 or len(term.args) < 2:
        return None
    n_dim = out_shape[-1]
    w = _shape_of(term.args[1], memo)
    if (
        not isinstance(w, tuple)
        or len(w) < 2
        or not isinstance(n_dim, int)
        or n_dim <= 0
    ):
        return None
    k_dim = w[-1] if term.op == "linear" else w[-2]
    if not isinstance(k_dim, int) or k_dim <= 0:
        return None
    n_out = float(_numel(out_shape))
    return (n_out / n_dim, float(k_dim), float(n_dim))


def _kernel_signature(
    term: Op, memo: dict
) -> tuple[str, tuple[float, ...]] | None:
    """(op-class, shape signature) an ``op_kernel_ns`` bucket matches.

    The mapping mirrors the classes ``calibrate._measure_op_kernels``
    times: ``"matmul"`` (``"MxKxN"``) for matmul/linear, ``"reduce"``
    (input numel) for reductions, ``"concat"`` / ``"stack"`` /
    ``"index_select"`` and ``"pointwise"`` (output numel — for reduce,
    the input numel is what streams).  Ops without a measured class
    return ``None`` and keep the roofline price.
    """
    op = term.op
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID or not isinstance(shape, tuple):
        return None
    n_out = float(_numel(shape))
    if op in ("matmul", "linear"):
        sig = _mm_signature(term, shape, memo)
        return ("matmul", sig) if sig is not None else None
    if op in _MEASURED_REDUCE_OPS:
        # The streamed size is the reduction's INPUT (the output
        # shrinks); _numel tolerates None/() arg shapes (→ 1).
        return (
            "reduce",
            (float(_numel(_shape_of(term.args[0], memo))),),
        )
    if op == "concat":
        return ("concat", (n_out,))
    if op == "stack":
        return ("stack", (n_out,))
    if op in _MEASURED_GATHER_OPS:
        return ("index_select", (n_out,))
    if op in _FUSION_POINTWISE_OPS:
        return ("pointwise", (n_out,))
    return None


def _profile_kernel_table(profile: Any) -> dict | None:
    """The raw ``op_kernel_ns`` dict of a profile-like, or ``None``."""
    if profile is None:
        return None
    if isinstance(profile, dict):
        return profile.get("op_kernel_ns") or None
    return getattr(profile, "op_kernel_ns", None) or None


def _kernel_lookup(table: dict | None):
    """Build ``kns(term, memo) -> measured ns | None`` from a table.

    Parses the ``{op_class: {shape_key: ns}}`` profile dict once; the
    returned callable maps a term to the measured wall time of the
    NEAREST bucket in log-space — sum of ``|log2 ratio|`` over the
    signature dims — so shapes between measured points price at their
    closest probe.  ``None`` table (or one with no usable entries)
    yields ``None``, i.e. the pure roofline path.
    """
    if not table:
        return None
    parsed: dict[str, list[tuple[tuple[float, ...], float]]] = {}
    for cls, entries in table.items():
        if not isinstance(entries, dict):
            continue
        pts: list[tuple[tuple[float, ...], float]] = []
        for key, ns in entries.items():
            try:
                sig = tuple(float(v) for v in str(key).split("x"))
                pts.append((sig, float(ns)))
            except (TypeError, ValueError):
                continue
        if pts:
            parsed[cls] = pts
    if not parsed:
        return None

    def kns(term: Op, memo: dict | None) -> float | None:
        sig = _kernel_signature(term, memo)
        if sig is None:
            return None
        pts = parsed.get(sig[0])
        if pts is None:
            return None
        dims = sig[1]
        best_ns = None
        best_d = float("inf")
        for esig, ns in pts:
            if len(esig) != len(dims):
                continue
            d = 0.0
            for s, e in zip(dims, esig, strict=True):
                d += abs(math.log2(max(s, 1.0) / max(e, 1.0)))
            if d < best_d:
                best_d, best_ns = d, ns
        return best_ns

    return kns


def roofline_cost_for(
    profile: Any = None,
    *,
    peak_flops: float | None = None,
    peak_bw: float | None = None,
    launch_s: float | None = None,
):
    """Return a roofline cost fn calibrated to a measured target profile.

    ``profile`` is a ``catopt_optimize.calibrate.TargetProfile`` (or any object
    / dict with ``tflops``, ``gbps``, ``launch_us``); ``None`` plus
    keyword overrides gives a one-off calibration.  The returned
    closure has the standard cost-fn signature ``fn(term, memo=None)``
    and can be dropped into ``Regime(cost_fn=...)``,
    ``EGraph.extract_best``, or ``dag_cost``.

    When the profile carries an ``op_kernel_ns`` table (measured
    per-op-class kernel latencies — see
    :func:`catopt_optimize.calibrate.calibrate`), each op's estimate
    is floored at the measured time of its nearest shape bucket:
    measurements can only raise the price toward observed latency,
    never undercut the roofline.

    ``roofline_cost_for()`` (no args) is exactly ``roofline_cost``.
    """
    pf, bw, ls = _profile_constants(profile)
    if peak_flops is not None:
        pf = float(peak_flops)
    if peak_bw is not None:
        bw = float(peak_bw)
    if launch_s is not None:
        ls = float(launch_s)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        return _roofline_cost(term, memo, pf, bw, ls, kns)

    cost.__name__ = "roofline_cost_for"
    cost.profile = profile
    return cost


def _depth_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    kernel_ns=None,
) -> float:
    """Shared critical-path traversal for depth_cost_for and the
    ``base="depth"`` arm of :func:`executor_cost_for`.  The memo key
    carries the constants so two profiles can share a memo dict
    without colliding (the ``_roofline_cost`` convention)."""
    ck = (
        ("dc", peak_flops, peak_bw, launch_s, term)
        if kernel_ns is None
        else ("dc", peak_flops, peak_bw, launch_s, id(kernel_ns), term)
    )
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        local = _local_roofline(
            term,
            memo,
            peak_flops=peak_flops,
            peak_bw=peak_bw,
            launch_s=launch_s,
            kernel_ns=kernel_ns,
        )
        if local >= _INVALID_COST:
            local = launch_s * 1e9
        child = max(
            (
                _depth_cost(
                    a, memo, peak_flops, peak_bw, launch_s, kernel_ns
                )
                for a in term.args
            ),
            default=0.0,
        )
        out = local + child
        memo[ck] = float(out)
        return out
    memo[ck] = 0.0
    return 0.0


def depth_cost_for(profile: Any = None):
    """Return a critical-path cost fn calibrated to a target profile.

    Same closure convention as :func:`roofline_cost_for`, but the
    objective is depth (local roofline latency + max child depth) like
    :func:`depth_cost` — the axis on which a sequential recurrence and
    its log-depth scan differ.
    """
    pf, bw, ls = _profile_constants(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        return _depth_cost(term, memo, pf, bw, ls, kns)

    cost.__name__ = "depth_cost_for"
    cost.profile = profile
    return cost


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
                if (
                    shapes
                    and shapes[0] is not None
                    and len(shapes[0]) >= 1
                ):
                    base = 2 * n * shapes[0][-1]
                else:
                    base = 2 * n
            else:
                base = coeff * n
            for arg in term.args:
                base += self(arg)
            return float(base)
        return 0.0


# ---------------------------------------------------------------------------
#  Backend-relative pricing — the sink's supported-op bound
# ---------------------------------------------------------------------------


def _ops_supported(
    term: Any, allowed: frozenset[str], cache: dict[Any, bool]
) -> bool:
    """True iff every :class:`Op` in ``term``'s DAG names an op in
    ``allowed``; leaves (``Param`` / ``Const`` / ``Var``) are always
    supported.  ``cache`` memoizes the content-keyed verdict, so a
    shared-subterm DAG costs one linear walk across every extraction
    probe rather than one walk per probe."""
    hit = cache.get(term)
    if hit is not None:
        return hit
    ok = not isinstance(term, Op) or (
        term.op in allowed
        and all(_ops_supported(a, allowed, cache) for a in term.args)
    )
    cache[term] = ok
    return ok


def backend_cost(
    cost_fn: CostFn, supported_ops: Collection[str]
) -> CostFn:
    """Wrap ``cost_fn`` so members outside a backend's op set price at
    ``+inf`` — extraction is *backend-relative*.

    ``supported_ops`` is the op-name set a
    :class:`~catopt_core.ports.Sink` can lower (its ``supported_ops``).
    Any term using an op outside the set can never win extraction, so
    the optimizer commits only to forms the sink can execute: the
    reachable equivalence class is bounded by the backend's semantic
    language rather than discovered and then rejected at lowering.
    Leaves are always supported.

    The wrapper preserves the wrapped model's ``charges_param_only`` /
    ``dag_exact`` / ``profile`` markers, so :func:`dag_cost` and
    extraction bill a ``param_bytes_cost``-based model exactly as
    before.  It always declares ``memo`` (so the extraction memo is
    threaded in) but forwards it only to a ``cost_fn`` that accepts it,
    matching :func:`_memo_dispatch`'s adaptive call.
    """
    import inspect

    allowed = frozenset(supported_ops)
    cache: dict[Any, bool] = {}
    try:
        accepts_memo = "memo" in inspect.signature(cost_fn).parameters
    except (TypeError, ValueError):  # uninspectable callable
        accepts_memo = False

    def priced(term: Any, memo: dict | None = None) -> float:
        if not _ops_supported(term, allowed, cache):
            return float("inf")
        return cost_fn(term, memo) if accepts_memo else cost_fn(term)

    priced.__name__ = getattr(cost_fn, "__name__", "backend_cost")
    for marker in ("charges_param_only", "dag_exact", "profile"):
        if hasattr(cost_fn, marker):
            setattr(priced, marker, getattr(cost_fn, marker))
    return priced


# ---------------------------------------------------------------------------
#  Lowering-aware pricing — price the lowering, not just the term
# ---------------------------------------------------------------------------
#
# The fidelity study (bench/cost_fidelity.py) showed term-level cost is
# blind to the lowering: the same term prices identically whether the
# generic IRModule dispatches it node-by-node, a batched carrier
# executor level-schedules it, or a compiled kernel fuses it — while
# measured latency differs 6-30x.  The models below put the executor
# in the price: executor_overhead counts the dispatched units a
# lowering performs, executor_cost_for charges them at the profile's
# dispatch rate on top of a base model, fused_cost_for approximates
# Inductor-style pointwise fusion, and lowering_aware_cost_for prices
# each term at its cheapest lowering — extraction then picks the term
# whose best lowering is cheapest.

#: Executor kinds a term can be lowered through.
LOWERINGS: tuple = ("generic", "batched_scan", "compiled")

#: Root ops whose first argument is a carrier map tree the
#: level-batched executors lower as a scan plan: ``scan_lower``'s
#: ``apply``/``applyd``, ``om_lower``'s ``om_apply``, ``omd_lower``'s
#: ``omd_apply``/``omd_applym``.  Anything else falls back to the
#: serial per-node evaluator inside those modules too.
_SCAN_ROOT_OPS = frozenset(
    {"apply", "applyd", "om_apply", "omd_apply", "omd_applym"}
)

#: Carrier compose ops forming the balanced tree the batched
#: executors level-schedule — one batched op per tree *level* rather
#: than one dispatch per node.
_SCAN_COMPOSE_OPS = frozenset(
    {"aff_compose", "affd_compose", "om_compose", "omd_compose"}
)

#: Ops whose torch binding is pure POINTWISE tensor work — the unit a
#: compiled lowering (torch.compile / Inductor) fuses into a single
#: kernel.  The elementwise core set plus the rest of the pointwise
#: bindings in the core table (casts, comparisons, where/clone, the
#: cmask/fill/attnbias mask generators) and the carrier ops whose
#: bodies are ordinary elementwise arithmetic on the carried
#: components — ``applyd`` = f0⊙h + f1, ``affd_compose`` =
#: (f0⊙g0, f0⊙g1 + f1), ``om_compose``/``omd_compose`` = maximum +
#: where-rescaled mul-adds, ``om_apply`` = a/l, ``omd_apply`` =
#: (fa⊙h + fb)/l (torch_bridge._CORE_TORCH_BINDINGS,
#: catopt_carriers.xcarrier.TORCH_BINDINGS).  NOT here: the bindings
#: hiding a contraction or reduction — ``apply``/``aff_compose``/
#: ``omd_applym`` (matmul), ``om_elem``/``omd_elem``/``om_elem_aff``/
#: ``om_elem_affd`` (amax + GEMM) — those stay fusion boundaries.
_FUSION_POINTWISE_OPS: frozenset = _FOLDABLE_ELEMWISE | frozenset(
    {
        "relu",
        "rsqrt",
        "cos",
        "sin",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_not",
        "where",
        "masked_fill",
        "clone",
        "to",
        "type_as",
        "float",
        "cmask",
        "fill",
        "attnbias",
        "affd_compose",
        "applyd",
        "om_compose",
        "om_apply",
        "omd_compose",
        "omd_apply",
    }
)

#: Ops emitting NO kernel under a compiled lowering — fusion-
#: TRANSPARENT plumbing.  Pure views (Inductor folds their index
#: arithmetic into the consumer's kernel — the _VIEW_OPS convention;
#: the set adds the aten spellings torch.export emits: unbind/getitem/
#: select/slice/squeeze/unsqueeze/expand/flatten/alias/dropout) and
#: carrier *packaging*: ``aff``/``aff_diag``/``om``/``omd`` assemble
#: the carried pair/triple and ``affd_a``/``affd_b``/``aff_A``/``aff_b``
#: project a component — under dynamo tracing they are Python-level
#: plumbing that never reaches the graph.  A pointwise consumer unions
#: with a transparent node's pointwise DESCENDANTS: the plumbing does
#: not split the region.  Constant morphisms (``eye``/``cswap``)
#: materialise at compile time — free, and they take no args so
#: nothing forwards through them.
_FUSION_TRANSPARENT_OPS: frozenset = frozenset(
    {
        "transpose",
        "reshape",
        "broadcast",
        "chunk",
        "split",
        "unsqueeze",
        "squeeze",
        "select",
        "slice",
        "expand",
        "flatten",
        "getitem",
        "unbind",
        "alias",
        "dropout",
        "leaf",
        "aff",
        "aff_diag",
        "om",
        "omd",
        "affd_a",
        "affd_b",
        "aff_A",
        "aff_b",
        "eye",
        "cswap",
    }
)

#: Everything that can ride inside a fused region — pointwise members
#: plus transparent plumbing.  (Previously ``_VIEW_OPS |
#: _FOLDABLE_ELEMWISE``; the compiled-lowering model now also knows
#: the pointwise carrier bodies and the remaining aten views.)
_FUSIBLE_OPS = _FUSION_POINTWISE_OPS | _FUSION_TRANSPARENT_OPS

#: Ops whose binding hides a direct solver call (``linalg.solve`` /
#: inverse) — measured orders of magnitude beyond a dispatch
#: (the fidelity sweep caught a ``trace`` term priced like ~3
#: dispatches that measured 14.5 s: the resolvent's solve, not the
#: graph).  FLOP models already bill ``2·du³``; this surcharge carries
#: the dispatch-side constant until a measured ``solve_us`` profile
#: field lands.
_SOLVER_OPS = frozenset({"trace", "inv"})

#: A solver call ≈ this many generic dispatches — a conservative
#: floor (a 128×128 solve measures ~ms vs ~µs per dispatch).
_SOLVER_FACTOR = 10_000.0


def _generic_overhead(term: Any, memo: dict) -> float:
    """Per-node dispatch count the generic evaluator performs for
    *term* — one unit per op occurrence (+``_SOLVER_FACTOR`` for
    solver ops).

    Deliberately NOT DAG-deduplicated: ``EGraph.extract_best`` recovers
    a node's local cost as ``f(t) − Σf(children)``, which is exact
    only for additive functions — a shared child is already billed
    once at the e-class level, so deduplicating here would double-
    subtract and collapse locals to zero (non-additive cost fns are
    what made a ``trace`` resolvent term price like ~3 dispatches and
    win extraction — fidelity bench, ``cost_fidelity.py``).

    A param-only subtree ``_folds_to_param`` materialises at lowering
    contributes zero units — ``IRModule._fold_weight_chains`` rewrites
    it to a bound parameter before the first forward, so there is no
    runtime dispatch to price (a param-only subtree that does NOT fold
    — e.g. a ``trace`` resolvent — still evaluates every call and is
    counted).
    """
    ck = ("eo", "generic", term)
    if ck in memo:
        return memo[ck]
    # eo(t) = w(t) + Σ eo(children) — memoised PER NODE so dag_cost's
    # one-call-per-node pricing is linear in the DAG, not O(N²): the
    # previous flat stack walk repriced every node's whole subtree.
    # Iterative post-order — spine-depth chains blow the recursion
    # limit (T=2048).
    stack = [(term, 0)]
    while stack:
        t, phase = stack.pop()
        c2 = ("eo", "generic", t)
        if c2 in memo:
            continue
        if phase == 0:
            if not isinstance(t, Op) or _folds_to_param(t, None, memo):
                memo[c2] = 0.0
                continue
            stack.append((t, 1))
            for a in t.args:
                stack.append((a, 0))
        else:
            memo[c2] = (
                _SOLVER_FACTOR if t.op in _SOLVER_OPS else 1.0
            ) + sum(memo[("eo", "generic", a)] for a in t.args)
    return memo[ck]


def _outside_overhead(term: Op, spine: Op, memo: dict) -> float:
    """Distinct op nodes of *term*'s DAG outside the ``spine`` subtree.

    The batched executor still runs everything around the compose
    spine generically — the apply root itself, the carried-state
    argument, surrounding tensor terms.  Param-only foldable subtrees
    materialise at lowering (``_generic_overhead`` convention) and are
    not counted; nodes are deduplicated because the executor's eval
    memoizes shared subtrees.
    """
    n = 0.0
    seen: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if (
            not isinstance(t, Op)
            or t in seen
            or t == spine
            or _folds_to_param(t, None, memo)
        ):
            continue
        seen.add(t)
        n += 1.0
        stack.extend(t.args)
    return n


def _batched_scan_overhead(term: Op, memo: dict) -> float:
    """Dispatched units under the level-batched carrier lowering.

    The compose spine under an apply-family root collapses to one
    batched op per balanced-tree *level* — ``ceil(log2 n_leaves)``
    dispatches — plus the leaf materialisation: the executors stack
    uniform leaves into ONE batched evaluation per leaf *kind*
    (``leaf_a_shared``/``leaf_b_gather`` — a (T,·) gather, not T
    sequential evals), so distinct leaf op-families bill once each,
    weighted by their generic size.  Ops outside the spine — the
    apply root, the state argument — still dispatch generically and
    are counted as such.
    """
    spine = term.args[0]
    seen: set = set()
    leaves: list = []
    stack = [spine]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op) and t.op in _SCAN_COMPOSE_OPS:
            stack.extend(t.args)
        else:
            leaves.append(t)
    levels = math.ceil(math.log2(max(1, len(leaves))))
    total = float(levels) + _outside_overhead(term, spine, memo)
    # One batched leaf evaluation per leaf kind — the executor stacks
    # same-op leaves into a single gathered batch.
    for kind in {getattr(leaf, "op", "") for leaf in leaves}:
        rep = next(
            leaf for leaf in leaves if getattr(leaf, "op", "") == kind
        )
        total += max(1.0, _generic_overhead(rep, memo))
    return total


def executor_overhead(
    term: Any, lowering: str, memo: dict | None = None
) -> float:
    """Structural count of the executor work a *lowering* performs.

    Counts, not seconds — multiply by a per-dispatch cost (as
    :func:`executor_cost_for` does) to price it.  ``lowering`` is one
    of :data:`LOWERINGS`:

    * ``"generic"`` — the per-node IRModule dispatcher: every op
      occurrence is one dispatched evaluation (counted per-node, not
      deduplicated — additive, which ``extract_best``'s local-cost
      decomposition requires).  View ops count too — the generic
      evaluator still dispatches them even though they launch no
      kernel.  Solver ops (``trace``/``inv``) bill
      ``_SOLVER_FACTOR`` each.  Param-folded subtrees simply contain
      no ops to count: their params are bound at lowering, not
      evaluated.
    * ``"batched_scan"`` — the level-batched carrier executors
      (``BatchedScanModule`` / ``BatchedOMModule`` /
      ``BatchedOmdModule``): for an ``apply``/``applyd``/``om_apply``/
      ``omd_apply[m]``-rooted term, ``ceil(log2 n_leaves)`` batched
      compose levels + one evaluation per distinct compose-tree leaf;
      ops outside the spine dispatch generically.  Non-scan roots get
      the generic count — the modules' serial fallback.
    * ``"compiled"`` — ``len(fusion_regions(term))``: the compiled
      executor's unit of work is the KERNEL, so the count is the
      region count — a pointwise chain of any depth fuses to one
      region, not O(nodes).  Priced by :func:`fused_cost_for`; the
      count is a whole-DAG property (regions merge across siblings),
      i.e. NON-additive — fine for overhead reporting/min-over-
      lowerings, not for ``extract_best``'s subtractive local-cost
      decomposition (see ``_generic_overhead``).
    """
    memo = {} if memo is None else memo
    ck = ("eo", lowering, term)
    hit = memo.get(ck)
    if hit is not None:
        return hit
    if lowering == "compiled":
        out = float(len(fusion_regions(term, memo)))
    elif lowering == "generic":
        out = _generic_overhead(term, memo)
    elif lowering == "batched_scan":
        if (
            isinstance(term, Op)
            and term.op in _SCAN_ROOT_OPS
            and term.args
        ):
            out = _batched_scan_overhead(term, memo)
        else:
            # Not a scan shape — the batched modules run their serial
            # fallback, i.e. generic dispatch.
            out = _generic_overhead(term, memo)
    else:
        raise ValueError(
            f"unknown lowering {lowering!r} — "
            f"expected one of {LOWERINGS}"
        )
    memo[ck] = float(out)
    return memo[ck]


def _profile_dispatch_s(profile: Any) -> float:
    """Per-dispatch executor overhead in seconds, from a profile-like.

    Reads ``dispatch_us`` (attribute or dict key) — per-call
    dispatcher overhead *on top of* the kernel launch the roofline
    model already prices, e.g. Python-side op dispatch in the generic
    evaluator.  Absent a measurement it falls back to the built-in
    launch constant: dispatch ≈ launch.
    """
    if profile is None:
        return _LAUNCH_S
    if isinstance(profile, dict):
        us = profile.get("dispatch_us", _LAUNCH_S * 1e6)
    else:
        us = getattr(profile, "dispatch_us", _LAUNCH_S * 1e6)
    return float(us) * 1e-6


def _profile_graph_overhead_s(profile: Any) -> float:
    """Per-call overhead of a COMPILED graph in seconds.

    ``calibrate`` measures it as ``graph_overhead_us`` — the
    guards+graph-call boundary cost of an Inductor-compiled module,
    minus the launch constants already billed per kernel.  The fused
    price uses ``max(dispatch_s, this)`` for its one per-graph term —
    the measured residual the per-kernel table cannot see.  Fallback
    ``80us`` is conservative (measured ~6us CPU, ~50-150us with
    guards on real graphs).
    """
    fallback = 80.0
    if profile is None:
        return fallback * 1e-6
    if isinstance(profile, dict):
        us = profile.get("graph_overhead_us", fallback)
    else:
        us = getattr(profile, "graph_overhead_us", fallback)
    return float(us) * 1e-6


def _profile_leaf_eval_s(profile: Any) -> float:
    """Per-leaf scan-eval machinery overhead in seconds.

    Reads ``leaf_eval_us`` — one ``apply``/``applyd`` leaf-operand eval
    through the executor's ``eval_term`` machinery (gather/select plus
    elementwise combine) on top of its kernels, as measured by
    ``catopt_optimize.calibrate``.  Absent a measurement the fallback is
    conservative — a leaf eval is a handful of dispatches, so it
    prices at ``4 * dispatch_s``.
    """
    fallback = 4.0 * _profile_dispatch_s(profile)
    if profile is None:
        return fallback
    if isinstance(profile, dict):
        us = profile.get("leaf_eval_us", fallback * 1e6)
    else:
        us = getattr(profile, "leaf_eval_us", fallback * 1e6)
    return float(us) * 1e-6


def _generic_latency_ns(
    term: Any,
    memo: dict,
    pf: float,
    bw: float,
    ls: float,
    dsp: float,
    kernel_ns=None,
) -> float:
    """Whole-subtree latency of the serial per-node evaluator, ns.

    The ``executor_cost_for(lowering="generic")`` composition:
    per-op roofline (kernel work + launch, floored at measured kernel
    times when ``kernel_ns`` is bound) plus one ``dispatch_s``
    machinery overhead per dispatched op.  Used inside the batched
    model for the pieces the level executors still run generically —
    leaf operand evals and everything outside the compose spine.
    """
    return _roofline_cost(term, memo, pf, bw, ls, kernel_ns) + (
        _generic_overhead(term, memo) * dsp * 1e9
    )


def _leaf_shared_a(leaves: list) -> bool:
    """``leaf_a_shared``: every leaf's first arg is the same term.

    Mirrors ``catopt_carriers.scan_lower.build_scan_plan`` — same
    object, or identically-named ``Param`` leaves (LTI recurrence):
    the executor evaluates the transition once and ``expand``s a
    stride-0 batch view instead of stacking T copies.
    """
    if not leaves or not getattr(leaves[0], "args", None):
        return False
    a0 = leaves[0].args[0]
    for leaf in leaves[1:]:
        if not getattr(leaf, "args", None) or not leaf.args:
            return False
        a = leaf.args[0]
        if a is a0 or (
            isinstance(a, Param)
            and isinstance(a0, Param)
            and a.name == a0.name
        ):
            continue
        return False
    return True


def _leaf_gather_base(leaves: list) -> Any | None:
    """``leaf_b_gather``'s shared base term, or ``None``.

    Mirrors ``catopt_carriers.scan_lower._leaf_b_gather``: every
    leaf's *second* argument is ``select(base, dim, index)`` over the
    SAME base term and dim — one ``index_select`` (or the base itself
    for contiguous indices) replaces n tiny indexing evals.
    """
    base = None
    dim = None
    for leaf in leaves:
        args = getattr(leaf, "args", None)
        if not args or len(args) < 2:
            return None
        b = args[1]
        if not (isinstance(b, Op) and b.op == "select" and b.args):
            return None
        d = b.attrs.get("dim", 0)
        i = b.attrs.get("index")
        if not isinstance(d, int) or not isinstance(i, int):
            return None
        if base is None:
            base, dim = b.args[0], d
        elif b.args[0] is not base or d != dim:
            return None
    return base


def _batched_scan_latency(
    term: Op,
    memo: dict,
    pf: float,
    bw: float,
    ls: float,
    dsp: float,
    leaf_eval_s: float,
    kernel_ns=None,
) -> float:
    """Whole-term nanoseconds under the level-batched carrier lowering.

    The level executors (``BatchedScanModule`` and the om-family
    variants) do NOT walk the term node-by-node — the compose spine
    collapses to ~log₂(n) batched levels, so the price decomposes as
    the executor's own work:

    * **leaf materialisation** — ``leaf_a_shared`` (every leaf's
      transition is the same term) evaluates the a-part ONCE;
      ``leaf_b_gather`` (every leaf's input is ``base[i]``) evaluates
      the base once and pays one ``index_select``.  Otherwise each
      leaf operand evals through the eval machinery — ``leaf_eval_s``
      per leaf plus the arg subtrees' kernel time — and one
      ``torch.stack`` per side.
    * **compose levels** — ``ceil(log2 n)`` levels; level *l* emits
      ~``n/2^{l+1}`` carried elements.  The diagonal/elementwise path
      costs ~9 dispatched calls per level (4 slot gathers, mul, mul+add,
      2 cat); the dense affine path ~4 (2 gathers, bmm, cat).  Each
      level also moves the running carried state: gather reads,
      arithmetic writes, and the cat copies —
      ``s·(4·m_l + done)·4`` bytes where ``s`` is the carried
      element's scalar count.
    * **outside the spine** — the apply root, the carried-state
      argument and surrounding terms still evaluate generically:
      per-node roofline + dispatch, deduplicated (the executor's eval
      memoizes shared subtrees).

    Whole-spine and non-additive — the same reporting/frontier caveat
    as ``fused_cost_for`` (see :func:`fusion_regions`).  Callers
    restrict it to ``_SCAN_ROOT_OPS`` terms; anything else keeps the
    serial-fallback (generic) price.
    """
    spine = term.args[0]
    seen: set = set()
    leaves: list = []
    stack = [spine]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op) and t.op in _SCAN_COMPOSE_OPS:
            stack.extend(t.args)
        else:
            leaves.append(t)
    n = max(1, len(leaves))
    dispatch_ns = dsp * 1e9
    call_ns = (ls + dsp) * 1e9
    total = 0.0

    # ---- leaf materialisation -------------------------------------
    # one leaf-operand eval = leaf_eval_s eval-machinery overhead +
    # the arg subtree's kernel time (roofline only — dispatch is what
    # leaf_eval_us measures, so charging it again would double-count).
    def leaf_eval(subtree: Any) -> float:
        return leaf_eval_s * 1e9 + _roofline_cost(
            subtree, memo, pf, bw, ls, kernel_ns
        )

    two_part = bool(leaves) and all(
        isinstance(lf, Op) and len(lf.args) >= 2 for lf in leaves
    )
    stack_ns = call_ns  # one torch.stack per materialised side
    if two_part and _leaf_shared_a(leaves):
        # LTI: one eval + a stride-0 expand view (free).
        total += leaf_eval(leaves[0].args[0])
    elif two_part:
        a_terms = list({lf.args[0] for lf in leaves})
        total += n * leaf_eval_s * 1e9 + stack_ns
        total += sum(
            _roofline_cost(a, memo, pf, bw, ls, kernel_ns)
            for a in a_terms
        )
    if two_part:
        base = _leaf_gather_base(leaves)
        if base is not None:
            # One base eval + one index_select — not n indexing evals.
            total += leaf_eval(base)
            total += (
                call_ns + _numel(_shape_of(base, memo)) * 4.0 / bw * 1e9
            )
        else:
            b_terms = list({lf.args[1] for lf in leaves})
            total += n * leaf_eval_s * 1e9 + stack_ns
            total += sum(
                _roofline_cost(b, memo, pf, bw, ls, kernel_ns)
                for b in b_terms
            )
    else:
        # Non pair-carriers (om triples and friends): per-leaf evals.
        arg_terms = list(
            {a for lf in leaves for a in getattr(lf, "args", ())}
        )
        total += n * leaf_eval_s * 1e9 + stack_ns
        total += sum(
            _roofline_cost(a, memo, pf, bw, ls, kernel_ns)
            for a in arg_terms
        )

    # ---- compose levels ---------------------------------------------
    # carried element scalars: the leaf packaging's summed arg numel.
    s = 1.0
    if leaves and isinstance(leaves[0], Op):
        s = float(
            sum(_numel(_shape_of(a, memo)) for a in leaves[0].args)
        )
        s = max(s, 1.0)
    dense = term.op == "apply"  # dense affine: (d+1)x(d+1) bmm compose
    calls = 4.0 if dense else 9.0
    levels = math.ceil(math.log2(n))
    done = float(n)
    for lv in range(levels):
        m_l = max(1.0, math.ceil(n / 2 ** (lv + 1)))
        flops_lv = 2.0 * m_l * s**1.5 if dense else 3.0 * m_l * s
        # 2·m_l·s gather reads + m_l·s arith writes + (done+m_l)·s cat
        # copies of the running carried state.
        bytes_lv = (4.0 * m_l + done) * s * 4.0
        total += (
            calls * call_ns + max(flops_lv / pf, bytes_lv / bw) * 1e9
        )
        done += m_l

    # ---- outside the spine: generic eval, DAG-deduplicated ----------
    seen2: set = set()
    stack2 = [term]
    while stack2:
        t = stack2.pop()
        if (
            not isinstance(t, Op)
            or t in seen2
            or t == spine
            or _folds_to_param(t, None, memo)
        ):
            continue
        seen2.add(t)
        total += (
            _local_roofline(
                t,
                memo,
                peak_flops=pf,
                peak_bw=bw,
                launch_s=ls,
                kernel_ns=kernel_ns,
            )
            + dispatch_ns
        )
        stack2.extend(t.args)
    return float(total)


def executor_cost_for(
    profile: Any = None,
    *,
    lowering: str = "generic",
    base: str = "roofline",
) -> CostFn:
    """Cost fn = a base model's term cost + per-dispatch overhead.

    ``base`` picks the underlying term price — ``"roofline"`` (per-op
    roofline sum, like :func:`roofline_cost`), ``"depth"``
    (critical-path roofline, like :func:`depth_cost`), or ``"flops"``
    (:func:`flops_cost`).  On top the closure adds
    ``executor_overhead(term, lowering) * dispatch_s`` where
    ``dispatch_s`` is the profile's ``dispatch_us`` (seconds;
    :func:`_profile_dispatch_s` fallback applies).  When the profile
    carries an ``op_kernel_ns`` table, per-op kernel times inside the
    roofline/depth/batched pieces floor at the measured values
    (``_kernel_lookup`` — see ``roofline_cost_for``).

    Units follow the base model, matching this file's conventions:
    roofline and depth are nanoseconds (``_local_roofline`` returns
    ns), so the overhead term is ``overhead * dispatch_s * 1e9``.
    For ``"flops"`` each dispatched unit is billed ``dispatch_us``
    flop-equivalents — the ``launch_aware_cost`` convention scaled to
    microseconds.  That is an approximation, documented as such: a
    dispatch is time, not arithmetic; the honest conversion would be
    ``dispatch_s * peak_flops``, which swamps every real term.
    µs-as-flops keeps the penalty commensurate with the model's
    magnitudes.

    ``lowering="compiled"`` ignores ``base`` and delegates to
    :func:`fused_cost_for` — the compiled executor's price is its
    fusion structure, not a per-node dispatch count
    (``executor_overhead`` is 0 there).  ``lowering="batched_scan"``
    likewise ignores ``base`` on scan-apply roots: the level-batched
    executors don't run the term's nodes at all — they run
    :func:`_batched_scan_latency`'s leaf-gather / level-compose
    schedule, whose work is priced directly (a whole-spine property,
    so non-additive like ``fused_cost_for``; serial-fallback terms
    keep the additive generic price).  The returned closure has the
    standard ``fn(term, memo=None)`` signature.
    """
    if lowering not in LOWERINGS:
        raise ValueError(
            f"unknown lowering {lowering!r} — "
            f"expected one of {LOWERINGS}"
        )
    if base not in ("roofline", "depth", "flops"):
        raise ValueError(
            f"unknown base {base!r} — "
            "expected 'roofline', 'depth' or 'flops'"
        )
    if lowering == "compiled":
        return fused_cost_for(profile)
    pf, bw, ls = _profile_constants(profile)
    dispatch_s = _profile_dispatch_s(profile)
    leaf_eval_s = _profile_leaf_eval_s(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))
    # ns for the roofline/depth bases; flop-equivalents for flops.
    per = dispatch_s * (1e9 if base != "flops" else 1e6)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = (
            "ec",
            lowering,
            base,
            pf,
            bw,
            ls,
            dispatch_s,
            leaf_eval_s,
            id(kns),
            term,
        )
        if ck in memo:
            return memo[ck]
        if (
            lowering == "batched_scan"
            and isinstance(term, Op)
            and term.op in _SCAN_ROOT_OPS
            and term.args
        ):
            out = _batched_scan_latency(
                term, memo, pf, bw, ls, dispatch_s, leaf_eval_s, kns
            )
        else:
            if base == "roofline":
                out = _generic_latency_ns(
                    term, memo, pf, bw, ls, dispatch_s, kns
                )
            else:
                if base == "depth":
                    b = _depth_cost(term, memo, pf, bw, ls, kns)
                else:  # flops
                    b = flops_cost(term, memo)
                out = b + executor_overhead(term, lowering, memo) * per
        memo[ck] = float(out)
        return out

    cost.__name__ = "executor_cost_for"
    cost.profile = profile
    cost.lowering = lowering
    return cost


def fusion_regions(
    term: Any, memo: dict | None = None
) -> tuple[frozenset, ...]:
    """Partition *term*'s op-DAG into Inductor-style fusion regions —
    one entry per kernel the ``"compiled"`` lowering emits.

    Each frozenset is one kernel's member ops:

    * a maximal connected cluster of pointwise ops
      (``_FUSION_POINTWISE_OPS``) — Inductor streams the whole cluster
      in one kernel; or
    * a singleton ``{op}`` for every fusion BOUNDARY: contractions
      (``matmul``/``linear``/``conv2d``/``sdpa``), reductions
      (``sum``/``mean``/``max``/``min``/``softmax``/``*_norm``),
      materialising layout ops (``concat``/``stack``/``contiguous``,
      gathers ``index_select``/``embedding``, ``bdiag``/``parl``),
      solver ops (``trace``/``inv``), carrier ops whose binding hides
      a contraction (``apply``/``aff_compose``/``omd_applym``/
      ``om_elem``/``omd_elem``/``om_elem_aff*``) — and any unknown op,
      conservatively.

    ``_FUSION_TRANSPARENT_OPS`` plumbing (views and carrier packaging:
    ``aff``/``aff_diag``/``om``/``omd``/``affd_a``/...) emits no kernel
    and appears in no region — a pointwise consumer unions with a
    transparent node's pointwise DESCENDANTS, so the tuple/view
    plumbing does not split a region.  Param-only subtrees that
    ``_folds_to_param`` materialises at lowering contribute nothing:
    the compiled graph reads them as inputs (the same compile-time
    fold the extract_best param-only discount prices at 0).  Leaves
    are never members.

    ``len(fusion_regions(t))`` is the predicted kernel count — what
    :func:`executor_overhead` reports for ``lowering="compiled"``.
    The partition is profile-independent (shapes don't enter) and a
    WHOLE-DAG property, not additive per node: regions merge across
    siblings at a shared pointwise parent and a shared subterm fuses
    once.  Cost fns built on it (``fused_cost_for``) are therefore
    reporting/frontier models — under ``extract_best``/``dag_cost``'s
    subtractive ``local = c(t) − Σc(children)`` a sibling merge
    clamps to 0 and the merge is billed nowhere; fine for
    ``lowering_aware``/frontier comparisons, approximate inside
    extraction (the ``_generic_overhead`` docstring has the
    additivity contract).
    """
    memo = {} if memo is None else memo
    ck = ("fr", term)
    hit = memo.get(ck)
    if hit is not None:
        return hit
    out: list[frozenset] = []
    if isinstance(term, Op):
        # Collect the DAG's op nodes once.  A param-only subtree the
        # lowerer folds into a materialised Param is a kernel INPUT,
        # not a kernel — skip it without descending.
        nodes: list[Op] = []
        seen: set = set()
        stack = [term]
        while stack:
            t = stack.pop()
            if not isinstance(t, Op) or t in seen:
                continue
            seen.add(t)
            if _folds_to_param(t, None, memo):
                continue
            nodes.append(t)
            stack.extend(t.args)
        node_set = set(nodes)

        # Union-find: one region per connected pointwise cluster.
        rep = {t: t for t in nodes}

        def find(t: Op) -> Op:
            while rep[t] is not t:
                rep[t] = rep[rep[t]]
                t = rep[t]
            return t

        for t in nodes:
            if t.op not in _FUSION_POINTWISE_OPS:
                continue
            # Transparent args forward their own args (transparent ops
            # never fold, so every transparent arg was collected).
            eff: list[Any] = list(t.args)
            i = 0
            while i < len(eff):
                a = eff[i]
                if (
                    isinstance(a, Op)
                    and a.op in _FUSION_TRANSPARENT_OPS
                ):
                    eff[i : i + 1] = a.args
                else:
                    i += 1
            for a in eff:
                if (
                    isinstance(a, Op)
                    and a.op in _FUSION_POINTWISE_OPS
                    and a in node_set
                ):
                    ra, rb = find(t), find(a)
                    if ra is not rb:
                        rep[ra] = rb
        grouped: dict[Op, set] = {}
        for t in nodes:
            if t.op in _FUSION_POINTWISE_OPS:
                grouped.setdefault(find(t), set()).add(t)
        out = [frozenset(m) for m in grouped.values()]
        out += [
            frozenset({t}) for t in nodes if t.op not in _FUSIBLE_OPS
        ]
    memo[ck] = tuple(out)
    return memo[ck]


def _region_traffic(
    region: frozenset,
    parents: dict,
    root: Op,
    memo: dict,
) -> tuple[float, float]:
    """Bytes a fused kernel actually moves: (in, out).

    *in* — every effective input of a member that is NOT itself a
    member, deduplicated: leaves, other kernels' outputs, and
    materialised param folds, each read once.  Transparent plumbing
    forwards its own args, so a view never surfaces as an input
    (Inductor folds its index arithmetic into the kernel).

    *out* — every member value consumed by a non-member op (through
    any number of transparent forwards) or returned as the term root:
    the tensors the kernel must write.  Interior member→member values
    stay in registers and move nothing — the structural fix over
    dominant-member pricing, where a region whose members read
    DIFFERENT large externals was billed only the largest op's
    traffic.
    """
    in_b = 0.0
    out_b = 0.0
    ext: set = set()
    for m in region:
        # inputs: forward through transparent args until an opaque
        # producer (member → interior; anything else → an external
        # read).
        pending = list(m.args)
        while pending:
            a = pending.pop()
            if isinstance(a, Op) and a.op in _FUSION_TRANSPARENT_OPS:
                pending.extend(a.args)
                continue
            if a in region or a in ext:
                continue
            ext.add(a)
            in_b += _numel(_shape_of(a, memo)) * 4.0
        # outputs: m crosses the region boundary if a consumer chain
        # (forwarding through transparent parents) reaches an op
        # outside the region — or m is the term root.
        if m is root:
            out_b += _numel(_shape_of(m, memo)) * 4.0
            continue
        reach = [m]
        seen: set = {m}
        boundary = False
        while reach and not boundary:
            u = reach.pop()
            for p in parents.get(u, ()):
                if p in region:
                    continue
                if p.op in _FUSION_TRANSPARENT_OPS:
                    if p not in seen:
                        seen.add(p)
                        reach.append(p)
                    continue
                boundary = True
                break
        if boundary:
            out_b += _numel(_shape_of(m, memo)) * 4.0
    return in_b, out_b


def _fused_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
    dispatch_s: float,
    kernel_ns=None,
    graph_overhead_s: float = 0.0,
) -> float:
    """Inductor-approximation price of *term* (see fused_cost_for).

    One kernel per :func:`fusion_regions` entry.  A region costs
    ``max(total member FLOPs / peak, region traffic / bandwidth)`` +
    one ``launch_s``: the fused kernel runs the SUM of its members'
    arithmetic (fusion removes launches, not FLOPs) and streams
    :func:`_region_traffic`'s external reads + boundary writes, so
    intermediates never touch memory.  When the profile carries an
    ``op_kernel_ns`` table, the region's kernel time is additionally
    floored at the sum of its members' measured kernel work (each
    member's measured wall minus its solo launch — the region
    launches once).  The whole graph pays a single
    ``dispatch_s`` — compilation removes interior dispatch too; the
    surviving boundary is the compiled module's own call.  A region
    containing a solver op additionally bills ``_SOLVER_FACTOR``
    dispatches — the extern ``linalg.solve``/``inv`` call dominates
    any kernel math, the same floor the generic count carries.
    """
    if not isinstance(term, Op):
        return 0.0
    regions = fusion_regions(term, memo)
    if not regions:
        return 0.0
    # parent edges over the op-DAG (skipping folded param subtrees —
    # they are kernel inputs, not consumers) — used to find each
    # member's boundary-crossing outputs.
    parents: dict = {}
    seen_dag: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if not isinstance(t, Op) or t in seen_dag:
            continue
        seen_dag.add(t)
        if _folds_to_param(t, None, memo):
            continue
        for a in t.args:
            if isinstance(a, Op):
                parents.setdefault(a, []).append(t)
                stack.append(a)
    total = 0.0
    for region in regions:
        flops = 0.0
        solver = False
        for t in region:
            flops += _flops_of(t, memo)
            solver = solver or t.op in _SOLVER_OPS
        in_b, out_b = _region_traffic(region, parents, term, memo)
        kernel_s = max(flops / peak_flops, (in_b + out_b) / peak_bw)
        if kernel_ns is not None:
            # Measured kernel times floor the region: the fused kernel
            # cannot run faster than the sum of its members' measured
            # kernel work (each member's measured wall stripped of its
            # solo launch — the region launches once, priced below).
            work_ns = 0.0
            for t in region:
                m = kernel_ns(t, memo)
                if m is not None:
                    work_ns += max(0.0, m - launch_s * 1e9)
            kernel_s = max(kernel_s, work_ns / 1e9)
        total += (
            kernel_s * 1e9
            + launch_s * 1e9
            + (_SOLVER_FACTOR if solver else 0.0) * dispatch_s * 1e9
        )
    # One per-graph charge — the bigger of the serial dispatch and the
    # measured compiled-graph call overhead (guards + cudagraph-safe
    # entry): the per-kernel table cannot see it, and calibration
    # showed it dominates the compiled price's residual.
    total += max(dispatch_s, graph_overhead_s) * 1e9
    return float(total)


def fused_cost_for(profile: Any = None) -> CostFn:
    """Compiled-lowering cost — the fusion-region (Inductor) model.

    One predicted kernel per :func:`fusion_regions` region: a maximal
    connected pointwise cluster (``_FUSION_POINTWISE_OPS`` — elementwise
    bindings plus the pointwise carrier bodies ``affd_compose``/
    ``applyd``/``om[d]_compose``/``om[d]_apply``) priced at
    ``max(Σ member FLOPs / peak, region traffic / bandwidth)`` — the
    kernel runs every member's arithmetic (fusion removes launches,
    not FLOPs) and streams its external reads + boundary writes once,
    so intermediates never touch memory and interior launches vanish
    (:func:`_region_traffic` — the fix for dominant-member pricing
    undercharging a region whose members read different externals).
    Non-fusible ops (matmul/conv/sdpa, reductions, materialising
    layout ops, the solver ops ``trace``/``inv``, the
    contraction-bearing carrier ops) are region singletons priced the
    same way; solver singletons additionally bill ``_SOLVER_FACTOR``
    dispatches (the extern solve dwarfs launch overheads — the same
    floor the generic count carries).  Transparent plumbing (views,
    carrier packaging) emits no kernel; param-only folds are kernel
    inputs.  Each region pays one ``launch_s`` and the whole graph one
    ``dispatch_s`` (the profile's ``dispatch_us``, else the launch
    constant): compilation removes launches and interior dispatch —
    the surviving boundary is the compiled module's own call.  Without
    that floor a lone GEMM would always look cheaper compiled than
    generic, and it isn't; fusion cannot shrink one kernel.

    Non-additive: the region partition is a whole-DAG property (see
    :func:`fusion_regions`), so under ``extract_best``/``dag_cost``'s
    subtractive local-cost decomposition the model is approximate —
    prefer it for ``lowering_aware``/frontier reporting.

    Deliberately approximate: real fusion decisions are
    scheduler-dependent (rematerialise vs reuse, reduction splits,
    layout constraints), and a big fused kernel may also spill
    intermediates the register file can't hold — the traffic model
    bills boundary crossings only.  That is the honest part of the
    model — pointwise fusion is where the measured win lives.

    Units are nanoseconds, matching :func:`roofline_cost`; the closure
    has the standard ``fn(term, memo=None)`` signature.
    """
    pf, bw, ls = _profile_constants(profile)
    dispatch_s = _profile_dispatch_s(profile)
    goh = _profile_graph_overhead_s(profile)
    kns = _kernel_lookup(_profile_kernel_table(profile))

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = ("fc", pf, bw, ls, dispatch_s, goh, id(kns), term)
        if ck in memo:
            return memo[ck]
        out = _fused_cost(
            term, memo, pf, bw, ls, dispatch_s, kns, goh
        )
        memo[ck] = float(out)
        return out

    cost.__name__ = "fused_cost_for"
    cost.profile = profile
    return cost


def lowering_aware_cost_for(
    profile: Any = None,
    *,
    lowerings: tuple = LOWERINGS,
    base: str = "roofline",
) -> CostFn:
    """Min-over-lowerings cost: price each term at its cheapest executor.

    For each ``l`` in ``lowerings`` the term is priced by
    :func:`executor_cost_for` ``(lowering=l, base=base)`` —
    ``"compiled"`` routes to :func:`fused_cost_for` instead — and the
    term's cost is the minimum.  Extraction under this model
    implicitly picks the term whose *best* lowering is cheapest,
    rather than pricing every term as if the generic per-node
    dispatcher would run it: a balanced carrier tree credits its
    level-batched plan, a pointwise chain credits fusion.

    The ``"compiled"`` arm is non-additive (the region partition is a
    whole-DAG property — see :func:`fusion_regions`), so the minimum
    is too: as an ``extract_best`` cost_fn the model is approximate —
    a sibling merge can hide inside a clamped local.  Reporting and
    frontier comparison are its sound uses.

    The returned closure carries ``cost.best_lowering(term,
    memo=None) -> str`` — the argmin lowering (first in ``lowerings``
    order on ties), for reporting which executor the price
    corresponds to.
    """
    fns: dict[str, CostFn] = {}
    for lw in lowerings:
        if lw == "compiled":
            fns[lw] = fused_cost_for(profile)
        elif lw in LOWERINGS:
            fns[lw] = executor_cost_for(profile, lowering=lw, base=base)
        else:
            raise ValueError(
                f"unknown lowering {lw!r} — expected one of {LOWERINGS}"
            )
    if not fns:
        raise ValueError("lowerings must be non-empty")
    pf, bw, ls = _profile_constants(profile)
    dsp = _profile_dispatch_s(profile)
    kt = _profile_kernel_table(profile)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = (
            "lw",
            base,
            pf,
            bw,
            ls,
            dsp,
            id(kt),
            tuple(lowerings),
            term,
        )
        if ck in memo:
            return memo[ck]
        out = min(f(term, memo) for f in fns.values())
        memo[ck] = float(out)
        return out

    def best_lowering(term: Any, memo: dict | None = None) -> str:
        memo = {} if memo is None else memo
        return min(fns, key=lambda lw: fns[lw](term, memo))

    cost.__name__ = "lowering_aware_cost_for"
    cost.profile = profile
    cost.best_lowering = best_lowering
    return cost
