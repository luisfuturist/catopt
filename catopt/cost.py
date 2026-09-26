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

    Rank-0 (``()``) policy: a ``()``-shaped operand under an op that
    cannot be scalar — axis-indexing ops (``transpose``/``select``/
    ``slice``/``unbind``/``squeeze``/``concat``/``chunk``/``split``/
    ``flatten``/``index_select``, ``sum``/``mean`` with an explicit
    dim, ``matmul``/``linear``/``conv2d`` with a scalar operand) — or
    a ``()`` produced by a carrier op's *convention* shape (``aff``/
    ``apply``/``om`` family report map-part/state/accumulator slots)
    means a carrier-internal member is being read as a plain tensor.
    Its true value shape is unrecoverable at term level, so these
    report ``None`` (unknown) — which propagates harmlessly
    (``_broadcast`` treats it as wildcard, FLOP costs fall back to
    numel-1) — rather than a fabricated ``()`` which both lies and
    crashes ``d % len(base)`` divisions downstream.  Genuine scalar
    results (``Const``, full-reduce ``sum``/``mean``, vector dot
    ``matmul``, scalar elementwise) still report ``()``.
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
            # Rank-1 operands (torch.matmul semantics): matrix-vector
            # (m,k)@(k,) -> (m,), vector-matrix (k,)@(…,k,n) -> (…,n),
            # dot product (k,)@(k,) -> ().  Without these a matvec
            # infers the MATRIX's shape (o,h), which then poisons any
            # broadcast consumer to _INVALID.
            if len(a) >= 2 and len(b) == 1:
                return tuple(a[:-1])
            if len(a) == 1 and len(b) >= 2:
                return tuple(b[:-2]) + (b[-1],)
            if len(a) == 1 and len(b) == 1:
                return ()
            # A scalar operand under matmul is ill-typed — almost
            # always a carrier-internal member read as a tensor.
            return shapes[0] or None
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
            # A scalar x () has no in-features axis — fall through to
            # the unknown result rather than fabricating (w[0],).
            if (len(shapes) >= 2 and shapes[0] is not None
                    and shapes[0] and shapes[1] is not None):
                w = shapes[1]
                if len(w) >= 1:
                    out = tuple(shapes[0][:-1]) + (w[0],)
                    if len(shapes) >= 3:
                        b = shapes[2]
                        if len(b) >= 2 and b[-1] == 1:
                            # A column bias (o,1) is a rank-1 bias in
                            # disguise — prefer the squeezed (o,)
                            # broadcast so the bias lands on the
                            # output's last axis rather than spawning
                            # a phantom trailing one (vector-x case:
                            # (o,) broadcast against (o,1) would
                            # otherwise infer (o,o)).  Fall back to
                            # the raw broadcast when the squeeze is
                            # provably ill-typed — e.g. a real per-row
                            # column (B,1) on a (B,o) output.
                            out_b = _broadcast(out, b[:-1])
                            if out_b is _INVALID:
                                out_b = _broadcast(out, b)
                        else:
                            out_b = _broadcast(out, b)
                        return out_b
                    return out
            return shapes[0] or None
        case "sum" | "mean":
            # Honor keepdim/dim when available, else reduce to scalar.
            dim = op.attrs.get("dim", op.attrs.get("axis", None))
            keep = bool(op.attrs.get("keepdim", False))
            base = shapes[0]
            if base is None:
                return ()
            if dim is None:
                return ()
            if not base:
                # Explicit-dim reduce over a scalar operand — the
                # carrier-internal ``()`` case (see _shape_of docstring).
                return None
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
            if not base:
                # () has no axes to permute (also guards d % len(base)).
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
                # A reshape is a view: the declared shape MUST preserve
                # the input numel.  Returning the attr verbatim lets an
                # ill-typed member (e.g. one that halved the wrong axis)
                # report its own wrong shape downstream — slipping past
                # broadcast guards — and it is a free view, so extraction
                # can even prefer it.  Flag the mismatch instead so the
                # member costs _INVALID_COST and can never be picked.
                base = shapes[0]
                if (isinstance(base, tuple)
                        and all(isinstance(d, int) and d >= 0
                                for d in shape)
                        and all(isinstance(d, int) and d >= 0
                                for d in base)
                        and _numel(shape) != _numel(base)):
                    return _INVALID
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
            if not base:
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
            if not base:
                # () has no dim to unbind — carrier-internal operand.
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", -1)) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "getitem":
            # After unbind the element shape is already the arg's shape.
            return shapes[0]
        case "select":
            base = shapes[0]
            if not base:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", 0)) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "slice":
            # aten.slice(t, dim, start, end, step) — arg4 is the step
            # (torch.export spells it positionally).  Ignoring it makes
            # strided slices (x[..., ::2], RoPE) report the unsliced
            # shape and poisons every downstream broadcast as _INVALID.
            base = shapes[0]
            if not base:
                return None
            d = op.attrs.get("arg1", op.attrs.get("dim", 0)) % len(base)
            lo = op.attrs.get("arg2", 0) or 0
            hi = op.attrs.get("arg3")
            step = op.attrs.get("arg4", 1) or 1
            out = list(base)
            if isinstance(base[d], int):
                n = (min(hi, base[d]) if isinstance(hi, int) else base[d]) - lo
                out[d] = max(0, -(-n // step))
            return tuple(out)
        case "embedding":
            # Row gather: out = idx.shape + (d,) where W is (v, d).
            wsh, ishape = shapes[0], shapes[1]
            if wsh is None or len(wsh) != 2:
                return None
            return tuple(ishape or ()) + (wsh[1],)
        case "index_select":
            # Gather along dim: that axis resizes to len(index).
            base = shapes[0]
            if not base:
                return None
            d = op.attrs.get("dim", op.attrs.get("arg1", 0)) % len(base)
            idx = op.attrs.get("index", op.attrs.get("arg2"))
            out = list(base)
            out[d] = len(idx) if isinstance(idx, (tuple, list)) else None
            return tuple(out)
        case "flatten":
            base = shapes[0]
            if not base:
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
            return shapes[0] or None
        case "conv2d":
            # x (N,C,H,W) @ w (O,C,kh,kw) -> (N,O,H',W')
            x, w = shapes[0], shapes[1]
            if (x is None or w is None or len(x) < 4 or len(w) < 4
                    or isinstance(op.attrs.get("padding"), str)):
                return (x[0], w[0], None, None) if (
                    x is not None and w is not None
                    and len(x) >= 1 and len(w) >= 1) else (x or None)
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
            # A () slot means a carrier member is being read as a
            # tensor: unknown, not scalar (see _shape_of docstring).
            return shapes[0] or None
        case "aff_compose":
            # f∘g keeps the outer map's linear-part shape (d×d).
            return shapes[0] or None
        case "apply":
            # apply(f, h) evaluates back to tensor-land: h's shape.
            return shapes[1] or None
        case "aff_diag":
            # The diagonal map h ↦ a⊙h + b is a pair value; its "shape"
            # is the scale part's — what consumers' costs price from.
            return shapes[0] or None
        case "affd_compose":
            # f∘g keeps the outer map's diagonal shape.
            return shapes[0] or None
        case "applyd":
            # applyd(f, h) evaluates back to tensor-land: h's shape.
            return shapes[1] or None
        case "om":
            # The carrier triple (m, l, a); its "shape" is the
            # accumulator's — what consumers' costs are priced from.
            out = shapes[2] if len(shapes) > 2 else shapes[0]
            return out or None
        case "om_elem":
            # elem(s[...,K], v[...,K,d]) reports the applied output
            # shape (...,T,d) — like `aff`, the carrier is priced as
            # the tensor it will become under om_apply.
            s, v = shapes[0], shapes[1]
            if (isinstance(s, tuple) and isinstance(v, tuple)
                    and len(s) >= 1 and len(v) >= 1):
                return tuple(s[:-1]) + (v[-1],)
            # A ()-shaped operand is a carrier member read as a
            # tensor — unknown, not scalar.
            return None
        case "om_compose" | "om_apply":
            return shapes[0] or None
        case "trace":
            # Tr(f): drop the first `usize` rows/cols of the block
            # matrix f : U⊗X → U⊗Y, leaving the X → Y map.
            s = shapes[0]
            if not (isinstance(s, tuple) and len(s) == 2):
                return s or None
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
            return a or None
        case "inv":
            return shapes[0]
        case "concat":
            # Variadic cat: sum every operand along the cat axis.
            a = shapes[0]
            if not a:
                # () has no cat axis — carrier-internal operand.
                return None
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
            if not base:
                return None
            dim = op.attrs.get("dim", -1) % len(base)
            n = op.attrs.get("chunks", 2)
            out = list(base)
            out[dim] = (base[dim] or 0) // n
            return tuple(out)
        case "split":
            base = shapes[0]
            if not base:
                return None
            dim = op.attrs.get("dim", -1) % len(base)
            # Two attr spellings exist: pairing passes mint
            # ``sizes=(...)``/``index=i``; exported graphs carry
            # torch.split's ``arg1`` = per-section size (or a size
            # list) and ``arg3``/``index`` = the section index.
            sizes = op.attrs.get("sizes", op.attrs.get("arg1"))
            idx = op.attrs.get("index", op.attrs.get("arg3", 0)) or 0
            out = list(base)
            if isinstance(sizes, (tuple, list)):
                out[dim] = (sizes[idx] if idx < len(sizes)
                            else None)
            elif isinstance(sizes, int):
                out[dim] = sizes          # equal-size sections
            else:
                out[dim] = None
            return tuple(out)
        case _:
            # Unknown ops pass through the first operand's shape; a ()
            # result from that is a carrier-internal member read as a
            # tensor (omd_*/om_elem_aff*/attnbias/generator ops land
            # here) — report unknown, not scalar.
            return shapes[0] or None


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
    # index_select is a gather: memory traffic, no arithmetic.
    "index_select": 0,
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
            k_dim = (shapes[1][-2] if len(shapes[1]) >= 2
                     else (shapes[1][0] if len(shapes[1]) == 1 else 1))
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
    bill_params = getattr(getattr(cost_fn, "func", cost_fn),
                          "charges_param_only", False)

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
#  Parameter-storage cost model — the ε axis's pricing side
# ---------------------------------------------------------------------------

def param_bytes_cost(term: Any, source_tensors: dict | None = None,
                     memo: dict | None = None,
                     by_bytes: bool = False) -> float:
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
    return float(sum(_param_index(term, source_tensors, memo,
                                  by_bytes).values()))


# Markers read by EGraph.extract_best / dag_cost: storage pricing does
# not fold away at compile time, so param-only subtrees stay billed;
# and the leaf/fold index is already a true DAG cost, so dag_cost
# must not apply its subtractive per-node decomposition.
param_bytes_cost.charges_param_only = True
param_bytes_cost.dag_exact = True


def param_bytes_cost_for(source_tensors: dict | None = None,
                         by_bytes: bool = False):
    """Bind ``source_tensors`` and return a standard cost fn.

    Same closure convention as :func:`roofline_cost_for`: the result
    has signature ``fn(term, memo=None)`` — dropping straight into
    ``EGraph.extract_best`` / ``dag_cost`` — and carries the
    ``charges_param_only`` marker through to extraction.  ``by_bytes``
    weights each stored value by its dtype width — the axis under
    which quantized params price below fp32.
    """

    def cost(term: Any, memo: dict | None = None) -> float:
        return param_bytes_cost(term, source_tensors, memo,
                                by_bytes)

    cost.__name__ = "param_bytes_cost_for"
    cost.charges_param_only = True
    cost.dag_exact = True
    return cost


def _param_numel(p: Param, source_tensors: dict | None,
                 by_bytes: bool = False) -> float:
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
                return numel              # torch.Tensor / jax / etc.
            s = getattr(t, "size", None)
            if isinstance(s, (int, float)):
                return float(s)            # numpy .size
    shape = getattr(getattr(p, "typ", None), "shape", None)
    return float(_numel(shape))


#: Elementwise ops ``IRModule._fold_weight_chains`` (catopt.torch_bridge)
#: folds eagerly through the ``_IR_TO_TORCH`` bindings when the whole
#: subtree is param-only.  ``matmul`` and ``concat`` are spelled out
#: separately in ``_folds_to_param`` because they fold under tighter arg
#: rules (real tensor operands, not Consts).  Keep in lock-step.
_FOLDABLE_ELEMWISE = frozenset({
    "add", "mul", "sub", "div", "neg", "square", "sqrt",
    "sigmoid", "silu", "tanh", "gelu", "exp", "pow",
})


def _has_var_leaf(term: Any, memo: dict) -> bool:
    """True iff the subtree reads a data input (Var leaf).

    Mirrors ``IRModule._uses_input``; id-keyed so the shared-subterm
    DAG stays a linear walk.
    """
    k = ("hv", id(term))
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


def _folds_to_param(term: Any, source_tensors: dict | None,
                    memo: dict) -> bool:
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
    k = ("pf", id(term))
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


def _fold_ewidth(term: Any, source_tensors: dict | None) -> float | None:
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
        ws = [w for a in term.args
              if (w := _fold_ewidth(a, source_tensors)) is not None]
        return max(ws) if ws else None
    return None


def _fold_numel(term: Op, source_tensors: dict | None, memo: dict,
                by_bytes: bool) -> float:
    """Stored size of the tensor a folding subtree materialises to —
    the OUTPUT numel: ``concat`` re-stores every argument's rows, a
    weight ``matmul`` stores the dense product.
    """
    n = float(_numel(_shape_of(term, memo)))
    if by_bytes:
        n *= _fold_ewidth(term, source_tensors) or 4.0
    return n


def _param_index(term: Any, source_tensors: dict | None,
                 memo: dict, by_bytes: bool = False) -> dict[str, float]:
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
    key = ("pbi", id(term))
    hit = memo.get(key)
    if hit is not None:
        return hit
    if isinstance(term, Param):
        out = {term.name: _param_numel(term, source_tensors,
                                       by_bytes)}
    elif isinstance(term, Op):
        if _folds_to_param(term, source_tensors, memo):
            out = {f"\x00fold:{id(term)}": _fold_numel(
                term, source_tensors, memo, by_bytes)}
        else:
            out = {}
            for a in term.args:
                for n, v in _param_index(a, source_tensors, memo,
                                         by_bytes).items():
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


def _local_roofline(term: Op, memo: dict | None = None, *,
                    peak_flops: float = _PEAK_FLOPS,
                    peak_bw: float = _PEAK_BW,
                    launch_s: float = _LAUNCH_S) -> float:
    """Estimated nanoseconds for one op: max(compute, memory) + launch."""
    shape = _infer_op_shape(term, memo)
    if shape is _INVALID:
        return _INVALID_COST
    flops = _flops_of(term, memo)
    if flops >= _INVALID_COST:
        return _INVALID_COST
    compute_s = flops / peak_flops
    memory_s = _bytes_of(term, memo) / peak_bw
    launch = 0.0 if term.op in _VIEW_OPS else launch_s
    # True views emit no kernel: no launch AND no memory traffic — the
    # read happens at the consumer, priced there via _STRIDE_PENALTY.
    if term.op in _VIEW_OPS:
        return 0.0
    return (max(compute_s, memory_s) + launch) * 1e9


def _roofline_cost(term: Any, memo: dict, peak_flops: float,
                   peak_bw: float, launch_s: float) -> float:
    """Shared traversal for roofline_cost and roofline_cost_for.

    The memo key carries the constants so two profiles can share a memo
    dict (e.g. inside dag_cost) without colliding.
    """
    ck = ("rc", peak_flops, peak_bw, launch_s, id(term))
    if ck in memo:
        return memo[ck]
    if isinstance(term, Op):
        base = _local_roofline(term, memo, peak_flops=peak_flops,
                               peak_bw=peak_bw, launch_s=launch_s)
        for arg in term.args:
            base += _roofline_cost(arg, memo, peak_flops, peak_bw,
                                   launch_s)
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
    return (float(get("tflops")) * 1e12,
            float(get("gbps")) * 1e9,
            float(get("launch_us")) * 1e-6)


def roofline_cost_for(profile: Any = None, *,
                      peak_flops: float | None = None,
                      peak_bw: float | None = None,
                      launch_s: float | None = None):
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
        ck = ("dc", pf, bw, ls, id(term))
        if ck in memo:
            return memo[ck]
        if isinstance(term, Op):
            local = _local_roofline(term, memo, peak_flops=pf,
                                    peak_bw=bw, launch_s=ls)
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
