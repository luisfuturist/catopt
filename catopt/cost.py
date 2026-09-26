"""Cost models for e-graph extraction.

The cost model assigns a scalar "cost" to a term, used by
EGraph.extract_best to find the minimum-cost representative.

Models provided include:
* count_cost - counts the number of operations (simplest).
* flops_cost - estimates FLOPs using shape information.
* param_bytes_cost - counts stored parameter values (the storage
  axis; what lets extraction prefer certified compressed members).

For the "killer experiment", the FLOPs-based model matters: it rewards
the associativity / distributivity / naturality rewrites that produce
fewer total floating-point operations.
"""
# ruff: noqa: RUF002, RUF003 — math notation in comments

from __future__ import annotations

from typing import Any

from catopt.ir import Const, Op, Param, Var

# compat: moved to catopt.typing — the shape/type-inference layer owns
# itself now; re-exported here so existing `from catopt.cost import
# _shape_of / _INVALID / _broadcast / _numel / _infer_op_shape` callers
# keep working.  All but ``_broadcast`` are also used internally below.
from catopt.typing import (  # noqa: F401
    _INVALID,
    _broadcast,
    _infer_op_shape,
    _numel,
    _shape_of,
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
    # Traced-monoidal ops (catopt.trace): wire juxtaposition and
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
    # Constant morphisms (catopt.trace): zero-arg ops that
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
    import inspect

    memo = {} if memo is None else memo
    takes_memo = "memo" in inspect.signature(cost_fn).parameters
    # getattr(..., "func", ...) unwraps functools.partial bindings.
    bill_params = getattr(
        getattr(cost_fn, "func", cost_fn), "charges_param_only", False
    )

    def c(t: Any) -> float:
        return cost_fn(t, memo=memo) if takes_memo else cost_fn(t)

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

    var_memo: dict[int, bool] = {}

    def has_var(t: Any) -> bool:
        """True if the subtree reads a data input (Var leaf).

        Subtrees over only Param/Const leaves are compile-time work —
        lowering folds them into a materialised parameter — so they are
        charged 0, matching extract_best's param-only discount.
        """
        k = t  # content-keyed
        if k in var_memo:
            return var_memo[k]
        if isinstance(t, Var):
            out = True
        elif isinstance(t, Op):
            out = any(has_var(a) for a in t.args)
        else:
            out = False
        var_memo[k] = out
        return out

    def rec(t: Any) -> None:
        nonlocal total
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Op):
            if not has_var(t) and not bill_params:
                return  # folds at compile time — free at runtime
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
    default extraction cost in :func:`catopt.optimize.optimize_model`.
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
#  Parameter-storage cost model — the ε axis's pricing side
# ---------------------------------------------------------------------------


def param_bytes_cost(
    term: Any,
    source_tensors: dict | None = None,
    memo: dict | None = None,
    by_bytes: bool = False,
) -> float:
    """Cost = stored parameter values — what the LOWERED module keeps.

    The storage axis the flop-based models cannot see: a low-rank
    factorisation or a shared (tied) weight computes the same function
    from fewer stored scalars.  The unit is *values* (bytes at unit
    width — multiply by dtype size for true bytes); ``eps_*`` factor
    params introduced by :func:`catopt.eps.low_rank_params` count
    normally, which is what lets extraction prefer the certified
    compressed member.

    Pricing mirrors ``IRModule._fold_weight_chains`` /
    ``_build_params`` (catopt.torch_bridge), not the term's leaf list:

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
    ``eps_*`` params — else from its ``TensorType`` (unknown dims
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
    any ``eps_*`` factors a pass injected) is authoritative when the
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


#: Elementwise ops ``IRModule._fold_weight_chains`` (catopt.torch_bridge)
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

    Mirrors ``IRModule._uses_input``; content-keyed so the shared-subterm
    DAG stays a linear walk.
    """
    k = ("hv", term)
    hit = memo.get(k)
    if hit is not None:
        return hit
    if isinstance(term, Var):
        out = True
    elif isinstance(term, Op):
        out = any(_has_var_leaf(a, memo) for a in term.args)
    else:
        out = False
    memo[k] = out
    return out


def _param_resolves(p: Param, source_tensors: dict | None) -> bool:
    """Would ``p.name`` land in ``_param_values`` at lowering?

    ``optimize_model`` hands the whole ``source_tensors`` dict to the
    lowerer as ``param_values`` (sharing/eps passes register their
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
) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch."""
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term, memo)
    if flops >= _INVALID_COST:  # pragma: no cover — INVALID shapes checked above
        return _INVALID_COST
    compute_s = flops / peak_flops
    memory_s = _bytes_of(term, memo) / peak_bw
    launch = 0.0 if term.op in _VIEW_OPS else launch_s
    # True views emit no kernel: no launch AND no memory traffic — the
    # read happens at the consumer, priced there via _STRIDE_PENALTY.
    if term.op in _VIEW_OPS:
        return 0.0
    return (max(compute_s, memory_s) + launch) * 1e9


def _roofline_cost(
    term: Any,
    memo: dict,
    peak_flops: float,
    peak_bw: float,
    launch_s: float,
) -> float:
    """Shared traversal for roofline_cost and roofline_cost_for.

    The memo key carries the constants so two profiles can share a memo
    dict (e.g. inside dag_cost) without colliding.
    """
    ck = ("rc", peak_flops, peak_bw, launch_s, term)
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_roofline(
            term,
            memo,
            peak_flops=peak_flops,
            peak_bw=peak_bw,
            launch_s=launch_s,
        )
        for arg in term.args:
            base += _roofline_cost(
                arg, memo, peak_flops, peak_bw, launch_s
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
    (``catopt.calibrate.calibrate``) for other targets.
    """
    memo = {} if memo is None else memo
    return _roofline_cost(term, memo, _PEAK_FLOPS, _PEAK_BW, _LAUNCH_S)


def _profile_constants(profile: Any) -> tuple[float, float, float]:
    """(peak_flops, peak_bw, launch_s) from a TargetProfile-like object.

    Accepts anything with ``.tflops`` / ``.gbps`` / ``.launch_us``
    attributes (e.g. ``catopt.calibrate.TargetProfile``) or a dict with
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


def roofline_cost_for(
    profile: Any = None,
    *,
    peak_flops: float | None = None,
    peak_bw: float | None = None,
    launch_s: float | None = None,
):
    """Return a roofline cost fn calibrated to a measured target profile.

    ``profile`` is a ``catopt.calibrate.TargetProfile`` (or any object
    / dict with ``tflops``, ``gbps``, ``launch_us``); ``None`` plus
    keyword overrides gives a one-off calibration.  The returned
    closure has the standard cost-fn signature ``fn(term, memo=None)``
    and can be dropped into ``Regime(cost_fn=...)``,
    ``EGraph.extract_best``, or ``dag_cost``.

    ``roofline_cost_for()`` (no args) is exactly ``roofline_cost``.
    """
    pf, bw, ls = _profile_constants(profile)
    if peak_flops is not None:
        pf = float(peak_flops)
    if peak_bw is not None:
        bw = float(peak_bw)
    if launch_s is not None:
        ls = float(launch_s)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        return _roofline_cost(term, memo, pf, bw, ls)

    cost.__name__ = "roofline_cost_for"
    cost.profile = profile
    return cost


def depth_cost_for(profile: Any = None):
    """Return a critical-path cost fn calibrated to a target profile.

    Same closure convention as :func:`roofline_cost_for`, but the
    objective is depth (local roofline latency + max child depth) like
    :func:`depth_cost` — the axis on which a sequential recurrence and
    its log-depth scan differ.
    """
    pf, bw, ls = _profile_constants(profile)

    def cost(term: Any, memo: dict | None = None) -> float:
        memo = {} if memo is None else memo
        ck = ("dc", pf, bw, ls, term)
        if ck in memo:
            return memo[ck]
        if isinstance(term, Op):
            local = _local_roofline(
                term, memo, peak_flops=pf, peak_bw=bw, launch_s=ls
            )
            if local >= _INVALID_COST:
                local = ls * 1e9
            child = max((cost(a, memo) for a in term.args), default=0.0)
            out = local + child
            memo[ck] = float(out)
            return out
        memo[ck] = 0.0
        return 0.0

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
