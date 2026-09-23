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

def _shape_of(term: Any, memo: dict | None = None) -> tuple | None:
    """Best-effort shape inference for a term.

    ``memo`` is an optional ``id()``-keyed dict shared across a whole
    traversal: extracted terms share Op objects (DAG structure), so
    memoising turns an exponential tree walk into a linear DAG walk.
    """
    key = id(term)
    if memo is not None and key in memo:
        return memo[key]
    if isinstance(term, (Var, Param)):
        out = term.typ.shape
    elif isinstance(term, Const):
        out = ()
    elif isinstance(term, Op):
        out = _infer_op_shape(term, memo)
    else:
        out = None
    if memo is not None:
        memo[key] = out
    return out


def _infer_op_shape(op: Op, memo: dict | None = None):
    # Zero-argument constant morphisms (catopt.trace): their shapes are
    # fully determined by attributes, so they must be answered before
    # the empty-shapes early return below.
    if op.op == "eye":
        d = op.attrs.get("dim", op.attrs.get("d", 1))
        return (d, d) if isinstance(d, int) else (None, None)
    if op.op == "cswap":
        d1, d2 = op.attrs.get("d1", 0), op.attrs.get("d2", 0)
        if isinstance(d1, int) and isinstance(d2, int):
            return (d1 + d2, d1 + d2)
        return None
    shapes = [_shape_of(a, memo) for a in op.args]
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
        case "eq" | "ne" | "lt" | "le" | "gt" | "ge":
            return _broadcast(shapes[0], shapes[1] if len(shapes) > 1 else None)
        case "where":
            out = _broadcast(shapes[1], shapes[2] if len(shapes) > 2 else None)
            return _broadcast(out, shapes[0])
        case (
            "square" | "sqrt" | "neg" | "sigmoid" | "silu" | "tanh"
            | "gelu" | "rsqrt" | "exp" | "softmax" | "masked_fill"
            | "logical_not"
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
                shape = tuple(shape)
                # Exported graphs keep literal -1 dims ("infer from
                # numel").  Resolve them: a -1 dim equals
                # numel(input)/numel(known dims); if unresolvable,
                # treat as unknown rather than propagating a negative
                # dim that poisons downstream broadcasting (_INVALID).
                if -1 in shape:
                    base_n = _numel(shapes[0])
                    known = 1
                    for d in shape:
                        if d != -1:
                            known *= (d if d is not None and d > 0 else 1)
                    inferred = (base_n // known
                                if known and base_n % known == 0 else None)
                    shape = tuple(inferred if d == -1 else d
                                  for d in shape)
                return shape
            return shapes[0]
        case "unsqueeze":
            base = shapes[0]
            if base is None:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", -1))
            d = d % (len(base) + 1)
            return tuple(base[:d]) + (1,) + tuple(base[d:])
        case "squeeze":
            base = shapes[0]
            if base is None:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", -1)) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "expand":
            s = op.attrs.get("shape")
            return tuple(s) if s is not None else shapes[0]
        case "stack":
            # stack(ts, dim): all inputs share a shape; insert dim.
            base = shapes[0]
            if base is None:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", 0))
            d = d % (len(base) + 1)
            n = len(op.args)
            return tuple(base[:d]) + (n,) + tuple(base[d:])
        case "unbind":
            # Element shape: base with the unbound dim removed.  The
            # tuple arity lives in getitem/select consumers.
            base = shapes[0]
            if base is None or not base:
                return base
            d = op.attrs.get("arg1", op.attrs.get("dim", -1)) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "getitem":
            # After unbind the element shape is already the arg's shape.
            return shapes[0]
        case "select":
            base = shapes[0]
            if base is None or not base:
                return base
            d = op.attrs.get("arg1", op.attrs.get("dim", 0)) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "slice":
            base = shapes[0]
            if base is None:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", 0)) % len(base)
            lo = op.attrs.get("arg2", 0)
            hi = op.attrs.get("arg3")
            out = list(base)
            if isinstance(base[d], int) and isinstance(hi, int):
                out[d] = min(hi, base[d]) - (lo or 0)
            return tuple(out)
        case "flatten":
            base = shapes[0]
            if base is None:
                return None
            d0 = op.attrs.get("arg1", op.attrs.get("start_dim", 0))
            d1 = op.attrs.get("arg2", op.attrs.get("end_dim", -1))
            d0, d1 = d0 % len(base), d1 % len(base)
            merged = _numel(base[d0:d1 + 1])
            return tuple(base[:d0]) + (merged,) + tuple(base[d1 + 1:])
        case "contiguous" | "to" | "type_as" | "float" | "dropout" | "alias":
            return shapes[0]
        case "sdpa":
            # out has q's shape (B, h, T, d)
            return shapes[0]
        case "conv2d":
            # x (N,C,H,W) @ w (O,C,kh,kw) -> (N,O,H',W')
            x, w = shapes[0], shapes[1]
            if (x is None or w is None or len(x) < 4 or len(w) < 4
                    or isinstance(op.attrs.get("padding"), str)):
                return (x[0], w[0], None, None) if (
                    x is not None and w is not None
                    and len(x) >= 1 and len(w) >= 1) else x
            st = op.attrs.get("stride", 1)
            pd = op.attrs.get("padding", 0)
            dl = op.attrs.get("dilation", 1)
            st = st if isinstance(st, (tuple, list)) else (st, st)
            pd = pd if isinstance(pd, (tuple, list)) else (pd, pd)
            dl = dl if isinstance(dl, (tuple, list)) else (dl, dl)
            oh = ow = None
            if x[2] is not None and w[2] is not None:
                oh = (x[2] + 2 * pd[0] - dl[0] * (w[2] - 1) - 1) // st[0] + 1
            if x[3] is not None and w[3] is not None:
                ow = (x[3] + 2 * pd[1] - dl[1] * (w[3] - 1) - 1) // st[1] + 1
            return (x[0], w[0], oh, ow)
        case "aff":
            # The map h ↦ A·h + b is a pair value; its "shape" is the
            # linear part's — what consumers' costs are priced from.
            return shapes[0]
        case "aff_compose":
            # f∘g keeps the outer map's linear-part shape (d×d).
            return shapes[0]
        case "apply":
            # apply(f, h) evaluates back to tensor-land: h's shape.
            return shapes[1]
        case "aff_diag":
            # The diagonal map h ↦ a⊙h + b is a pair value; its "shape"
            # is the scale part's — what consumers' costs price from.
            return shapes[0]
        case "affd_compose":
            # f∘g keeps the outer map's diagonal shape.
            return shapes[0]
        case "applyd":
            # applyd(f, h) evaluates back to tensor-land: h's shape.
            return shapes[1]
        case "om":
            # The carrier triple (m, l, a); its "shape" is the
            # accumulator's — what consumers' costs are priced from.
            return shapes[2] if len(shapes) > 2 else shapes[0]
        case "om_elem":
            # elem(s[...,K], v[...,K,d]) reports the applied output
            # shape (...,T,d) — like `aff`, the carrier is priced as
            # the tensor it will become under om_apply.
            s, v = shapes[0], shapes[1]
            if (isinstance(s, tuple) and isinstance(v, tuple)
                    and len(s) >= 1 and len(v) >= 1):
                return tuple(s[:-1]) + (v[-1],)
            return s
        case "om_compose" | "om_apply":
            return shapes[0]
        case "trace":
            # Tr(f): drop the first `usize` rows/cols of the block
            # matrix f : U⊗X → U⊗Y, leaving the X → Y map.
            s = shapes[0]
            if not (isinstance(s, tuple) and len(s) == 2):
                return s
            u = op.attrs.get("usize", 0)
            du = sum(u) if isinstance(u, (list, tuple)) else u
            if not isinstance(du, int):
                return s
            r = s[0] - du if isinstance(s[0], int) else None
            c = s[1] - du if isinstance(s[1], int) else None
            return (r, c)
        case "bdiag" | "parl":
            # Both are total-dims-preserving matrix juxtaposition:
            # bdiag is literal block-diagonal; parl re-lays the same
            # blocks keeping feedback wires first (see catopt.trace).
            a, b = shapes[0], shapes[1]
            if (isinstance(a, tuple) and isinstance(b, tuple)
                    and len(a) == 2 and len(b) == 2
                    and all(isinstance(d, int) for d in (*a, *b))):
                return (a[0] + b[0], a[1] + b[1])
            return a
        case "inv":
            return shapes[0]
        case "concat":
            # Variadic cat: sum every operand along the cat axis.
            a = shapes[0]
            if a is None or len(a) == 0:
                return a
            dim = op.attrs.get("dim", op.attrs.get("arg1", 0)) % len(a)
            out = list(a)
            out[dim] = 0
            for s in shapes:
                if not isinstance(s, tuple) or len(s) != len(a):
                    return a
                out[dim] += (s[dim] or 0)
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
        case "split":
            base = shapes[0]
            if base is None:
                return None
            dim = op.attrs.get("dim", -1) % len(base)
            sizes = op.attrs.get("sizes", ())
            idx = op.attrs.get("index", 0)
            out = list(base)
            out[dim] = sizes[idx] if idx < len(sizes) else 0
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
        if da is not None and da < 0:
            da = None  # unresolved -1: treat as unknown, not a mismatch
        if db is not None and db < 0:
            db = None
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
    "concat": 0, "chunk": 0, "split": 0,
    # contiguous copies memory (0 FLOPs but real bandwidth — the
    # roofline model prices it); sdpa ~2*T work per output element.
    "contiguous": 0, "sdpa": 2,
    # Traced-monoidal ops (catopt.trace): wire juxtaposition and
    # constant morphisms carry no FLOPs; trace/inv are priced in
    # _flops_of directly (solve cost depends on usize).
    "bdiag": 0, "parl": 0, "eye": 0, "cswap": 0,
}

#: Ops that produce no kernel — true views or wire bookkeeping.
#: torch.split/chunk/transpose/reshape return views: no launch, no
#: memory traffic; their only runtime effect is the stride they leave
#: for consumers (priced via _STRIDE_PENALTY).  concat is NOT here: a
#: runtime cat() is a real copy kernel — it is only free when the whole
#: subtree is param-only (compile-time fold, handled by extraction).
_VIEW_OPS = {"transpose", "reshape", "broadcast", "chunk",
             "split", "leaf", "aff", "om", "aff_diag",
             # Constant morphisms (catopt.trace): zero-arg ops that
             # materialise a fixed matrix — compile-time constants,
             # like the carrier-packaging ops above.
             "eye", "cswap"}

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
        if shapes and shapes[1] is not None and shapes[1] is not _INVALID:
            k_dim = shapes[1][-2] if len(shapes[1]) >= 2 else 1
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
        if (w is not None and w is not _INVALID and len(w) >= 4
                and all(isinstance(d, int) for d in w[1:4])):
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
        if (f is not None and f is not _INVALID and len(f) >= 2
                and isinstance(f[-1], int)):
            return float(2 * n_out * f[-1])
        return float(2 * n_out)
    if term.op == "apply":
        # f·h + b: matvec + add ≈ 2·d² flops on a d-vector out.
        shapes = [_shape_of(a, memo) for a in term.args]
        f = shapes[0] if shapes else None
        if (f is not None and f is not _INVALID and len(f) >= 2
                and isinstance(f[-1], int)):
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
        k = (s[-1] if isinstance(s, tuple) and s
             and isinstance(s[-1], int) else 1)
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


def _local_cost(term: Op, launch_penalty: float = 0.0,
                memo: dict | None = None) -> float:
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

    ``memo`` (id-keyed) makes repeated calls over a shared-subterm DAG
    linear instead of exponential; callers doing many evaluations
    (e.g. extraction) should pass a shared dict.
    """
    memo = {} if memo is None else memo
    key = id(term)
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
    """
    import inspect
    memo = {} if memo is None else memo
    takes_memo = "memo" in inspect.signature(cost_fn).parameters

    def c(t: Any) -> float:
        return cost_fn(t, memo=memo) if takes_memo else cost_fn(t)

    seen: set[int] = set()
    total = 0.0

    var_memo: dict[int, bool] = {}

    def has_var(t: Any) -> bool:
        """True if the subtree reads a data input (Var leaf).

        Subtrees over only Param/Const leaves are compile-time work —
        lowering folds them into a materialised parameter — so they are
        charged 0, matching extract_best's param-only discount.
        """
        k = id(t)
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
        if id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, Op):
            if not has_var(t):
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
    ck = ("lc", id(term))
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
    ck = ("dc", id(term))
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        local = _local_roofline(term, memo=memo)
        if local >= _INVALID_COST:
            # Unshapeable op: charge a launch, not a veto — depth is a
            # structural metric, not a soundness gate.
            local = _LAUNCH_S * 1e9
        child = max((depth_cost(a, memo) for a in term.args), default=0.0)
        out = local + child
        memo[ck] = float(out)
        return out
    memo[ck] = 0.0
    return 0.0


def count_cost(term: Any, memo: dict | None = None) -> float:
    """Cost = number of non-view operations in the term tree."""
    memo = {} if memo is None else memo
    ck = ("cc", id(term))
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
_PEAK_FLOPS = 2.5e12    # measured: ~2.5 TFLOPS fp32 GEMM (RTX 2050)
_PEAK_BW = 8.9e10       # measured: ~89 GB/s copy bandwidth
_LAUNCH_S = 8.7e-6      # measured: ~8.7 µs eager launch overhead
_STRIDE_PENALTY = 1.0   # measured: strided copies ~1.0x on this GPU


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
    out_bytes = 0.0 if term.op in _VIEW_OPS else _numel(
        _infer_op_shape(term, memo)) * 4.0
    return in_bytes + out_bytes


def _local_roofline(term: Op, memo: dict | None = None) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch."""
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term, memo)
    if flops >= _INVALID_COST:
        return _INVALID_COST
    compute_s = flops / _PEAK_FLOPS
    memory_s = _bytes_of(term, memo) / _PEAK_BW
    launch = 0.0 if term.op in _VIEW_OPS else _LAUNCH_S
    # True views emit no kernel: no launch AND no memory traffic — the
    # read happens at the consumer, priced there via _STRIDE_PENALTY.
    if term.op in _VIEW_OPS:
        return 0.0
    return (max(compute_s, memory_s) + launch) * 1e9


def roofline_cost(term: Any, memo: dict | None = None) -> float:
    """Roofline cost in estimated nanoseconds (per-op, additive).

    max(flops/PEAK_FLOPS, bytes/PEAK_BW) + launch per op; view ops are
    free except for the strided-read penalty they impose on consumers.
    This is the honest model for questions like "does the fused GEMM
    pay?" — it answers differently at different batch sizes, which is
    what the measurements show.
    """
    memo = {} if memo is None else memo
    ck = ("rc", id(term))
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_roofline(term, memo)
        for arg in term.args:
            base += roofline_cost(arg, memo)
        memo[ck] = float(base)
        return memo[ck]
    memo[ck] = 0.0
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
