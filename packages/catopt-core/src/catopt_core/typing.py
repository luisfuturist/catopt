"""Term typing — best-effort shape inference for IR terms.

This layer is deliberately separate from pricing (``catopt_core.cost``):
costs ask "how expensive is this term?"; typing asks "what shape does
this term produce?".  Nothing here materialises tensors — shapes are
inferred structurally from ``catopt_core.ir`` terms and their attrs.

Contract
--------
A "shape" is one of:

* a ``tuple`` of dims — a believed shape; individual dims may be
  ``None`` (extent unknown);
* ``_INVALID`` — the term is *provably ill-typed* (e.g. broadcasting
  ``(out, in)`` against ``(B, T, 1)``).  Distinct from ``None``:
  ill-typed terms are poisonous — the cost model must never prefer
  them;
* ``None`` — merely *unknown*: propagates harmlessly (``_broadcast``
  treats it as a wildcard, numel-based costs fall back to 1);
* ``()`` — a *genuine scalar only* (``Const``, full-reduce
  ``sum``/``mean``, vector-dot ``matmul``, scalar elementwise).  A
  carrier-internal member read as a plain tensor — an ``om`` triple's
  accumulator slot, an ``aff`` pair's convention ``()`` — reports
  ``None`` (unknown), never ``()``: the true value shape is
  unrecoverable at term level, and a fabricated ``()`` both lies and
  crashes ``d % len(base)`` divisions downstream.  See the rank-0
  policy in :func:`_shape_of`.

Carrier ops report their shapes through registered handlers:
:func:`register_shape_rule` installs ``fn(op, shapes)`` in
``_SHAPE_RULES``.  ``_infer_op_shape`` consults the registry where the
op-dispatch match runs — after ``_INVALID`` propagation and the
unknown/empty early-returns — so a handler sees only non-``None``
operand shapes and must still map a carrier-internal ``()`` operand to
``None``.  Zero-argument ops (``eye``/``cswap`` constant morphisms)
have their handler called with ``shapes = []`` — their shapes are
fully determined by attrs.

The ``aff``/``apply``/``om``/``omd``/``trace``/``bdiag``/``parl``
carrier-family registrations live at the bottom of this module for
now; later refactor phases move each family next to its laws in its
own module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from catopt_core.attrs import attr_of, is_positional_attr
from catopt_core.ir import Const, Op, Param, Var

# ---------------------------------------------------------------------------
# Shape inference (lightweight)
# ---------------------------------------------------------------------------


def _shape_of(
    term: Any, memo: dict | None = None
) -> tuple | str | None:
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
    key = term  # content-keyed: interned terms hash by structure
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


#: Ops whose result shape is invariant under operand swap — used to
#: keep unknown-shape pricing symmetric for ``add``/``mul``/``eq``/``ne``.
_COMMUTATIVE_BROADCAST = frozenset({"add", "mul", "eq", "ne"})


def _infer_op_shape(op: Op, memo: dict | None = None):
    # Zero-argument ops with a registered rule (constant morphisms
    # like catopt_carriers.trace's ``eye``/``cswap``): their shapes are fully
    # determined by attributes, so they must be answered before the
    # empty-shapes early return below.
    rule = _SHAPE_RULES.get(op.op)
    if rule is not None and not op.args:
        return rule(op, [])
    shapes = [_shape_of(a, memo) for a in op.args]
    if any(s is _INVALID for s in shapes):
        return _INVALID
    if not shapes or any(s is None for s in shapes):
        if shapes and shapes[0] is not None:
            return shapes[0]
        # Commutative element-wise ops are operand-symmetric: with the
        # first shape unknown, fall back to whichever operand shape IS
        # known, so add(a, b) and add(b, a) price identically.
        if op.op in _COMMUTATIVE_BROADCAST:
            for s in shapes:
                if s is not None:
                    return s
        return None
    if rule is not None:
        return rule(op, shapes)
    match op.op:
        case "matmul":
            a = cast("tuple", shapes[0])
            b = cast("tuple", shapes[1])
            if len(a) >= 2 and len(b) >= 2:
                return (*a[:-1], b[-1])
            # Rank-1 operands (torch.matmul semantics): matrix-vector
            # (m,k)@(k,) -> (m,), vector-matrix (k,)@(…,k,n) -> (…,n),
            # dot product (k,)@(k,) -> ().  Without these a matvec
            # infers the MATRIX's shape (o,h), which then poisons any
            # broadcast consumer to _INVALID.
            if len(a) >= 2 and len(b) == 1:
                return tuple(a[:-1])
            if len(a) == 1 and len(b) >= 2:
                return (*tuple(b[:-2]), b[-1])
            if len(a) == 1 and len(b) == 1:
                return ()
            # A scalar operand under matmul is ill-typed — almost
            # always a carrier-internal member read as a tensor.
            return shapes[0] or None
        case "add" | "mul" | "sub" | "div":
            # Element-wise ops broadcast: result is the broadcast shape,
            # not simply the first operand's shape.
            return _broadcast(
                shapes[0], shapes[1] if len(shapes) > 1 else None
            )
        case (
            "maximum"
            | "minimum"
            | "fmax"
            | "fmin"
            | "fmod"
            | "remainder"
            | "xlogy"
            | "atan2"
            | "heaviside"
            | "isclose"
        ):
            return _broadcast(
                shapes[0], shapes[1] if len(shapes) > 1 else None
            )
        case "lerp" | "addcmul" | "addcdiv":
            # Ternary broadcast (the scalar ``value``/``weight`` arg,
            # when spelled as a Const operand, broadcasts as ()).
            out = shapes[0]
            for s in shapes[1:]:
                out = _broadcast(out, s)
                if out is _INVALID:
                    return _INVALID
            return out
        case "eq" | "ne" | "lt" | "le" | "gt" | "ge":
            return _broadcast(
                shapes[0], shapes[1] if len(shapes) > 1 else None
            )
        case (
            "logical_and"
            | "logical_or"
            | "logical_xor"
            | "bitwise_and"
            | "bitwise_or"
        ):
            return _broadcast(
                shapes[0], shapes[1] if len(shapes) > 1 else None
            )
        case "where":
            out = _broadcast(
                shapes[1], shapes[2] if len(shapes) > 2 else None
            )
            return _broadcast(out, shapes[0])
        case (
            "square"
            | "sqrt"
            | "neg"
            | "sigmoid"
            | "silu"
            | "tanh"
            | "gelu"
            | "rsqrt"
            | "exp"
            | "exp2"
            | "expm1"
            | "softmax"
            | "log_softmax"
            | "masked_fill"
            | "logical_not"
            | "relu"
            | "leaky_relu"
            | "elu"
            | "celu"
            | "softplus"
            | "softsign"
            | "hardswish"
            | "hardsigmoid"
            | "mish"
            | "relu6"
            | "prelu"
            | "hardtanh"
            | "abs"
            | "sign"
            | "floor"
            | "ceil"
            | "round"
            | "frac"
            | "trunc"
            | "log"
            | "log2"
            | "log10"
            | "log1p"
            | "erf"
            | "erfc"
            | "erfinv"
            | "gammaln"
            | "digamma"
            | "i0"
            | "sinc"
            | "reciprocal"
            | "nan_to_num"
            | "isnan"
            | "isinf"
            | "isfinite"
            | "isposinf"
            | "isneginf"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "detach"
            | "detach_"
            | "copy"
        ):
            return shapes[0]
        # Axis-anchored ops that keep the input rank: a ()-shaped
        # operand means a carrier member was read as a tensor — the
        # axis doesn't exist, so unknown (None), never scalar.
        case (
            "cumsum"
            | "cumprod"
            | "logcumsumexp"
            | "cummax"
            | "cummin"
            | "argsort"
            | "flip"
            | "roll"
            | "triu"
            | "tril"
            | "scatter"
            | "scatter_add"
            | "scatter_reduce"
            | "index_add"
            | "index_put"
            | "slice_scatter"
            | "select_scatter"
            | "batch_norm"
            | "group_norm"
            | "sort"
            | "topk"
            | "median"
            | "kthvalue"
            | "mode"
            | "_assert_tensor_metadata"
        ):
            return shapes[0] or None
        case "pow":
            return shapes[0] if shapes and shapes[0] is not None else ()
        case "linear":
            # F.linear(x[..., in], W[out, in]) -> [..., out]
            # A scalar x () has no in-features axis — fall through to
            # the unknown result rather than fabricating (w[0],).
            if (
                len(shapes) >= 2
                and shapes[0] is not None
                and shapes[0]
                and shapes[1] is not None
            ):
                w = shapes[1]
                if len(w) >= 1:
                    out = (*tuple(shapes[0][:-1]), w[0])
                    if len(shapes) >= 3:
                        b = cast("tuple", shapes[2])
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
        case (
            "sum"
            | "mean"
            | "prod"
            | "max"
            | "min"
            | "amax"
            | "amin"
            | "var_mean"
            | "std_mean"
            | "argmax"
            | "argmin"
            | "var"
            | "std"
            | "any"
            | "all"
            | "nansum"
            | "nanmean"
            | "count_nonzero"
            | "linalg_vector_norm"
        ):
            # Honor keepdim/dim when available, else reduce to scalar.
            dim = attr_of(op, "dim", "axis")
            keep = bool(op.attrs.get("keepdim", False))
            base = shapes[0]
            if (
                base is None
            ):  # pragma: no cover — dispatch filters None shapes
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
            d0 = attr_of(op, "dim0", default=-2)
            d1 = attr_of(op, "dim1", default=-1)
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
                            known *= d if d is not None and d > 0 else 1
                    inferred = (
                        base_n // known
                        if known and base_n % known == 0
                        else None
                    )
                    shape = tuple(
                        inferred if d == -1 else d for d in shape
                    )
                # A reshape is a view: the declared shape MUST preserve
                # the input numel.  Returning the attr verbatim lets an
                # ill-typed member (e.g. one that halved the wrong axis)
                # report its own wrong shape downstream — slipping past
                # broadcast guards — and it is a free view, so extraction
                # can even prefer it.  Flag the mismatch instead so the
                # member costs _INVALID_COST and can never be picked.
                base = shapes[0]
                if (
                    isinstance(base, tuple)
                    and all(
                        isinstance(d, int) and d >= 0 for d in shape
                    )
                    and all(isinstance(d, int) and d >= 0 for d in base)
                    and _numel(shape) != _numel(base)
                ):
                    return _INVALID
                return shape
            return shapes[0]
        case "unsqueeze":
            base = shapes[0]
            if (
                base is None
            ):  # pragma: no cover — dispatch filters None shapes
                return None
            d = attr_of(op, "dim", default=-1)
            d = d % (len(base) + 1)
            return (*tuple(base[:d]), 1, *tuple(base[d:]))
        case "squeeze":
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim")
            if d is None:
                # squeeze() with no dim drops every KNOWN size-1
                # axis; unknown extents stay (they might be 1).
                return tuple(x for x in base if x != 1)
            d = d % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "expand" | "broadcast_to":
            s = op.attrs.get("shape")
            return tuple(s) if s is not None else shapes[0]
        case "repeat":
            # repeat(*sizes): leading dims are new, each covered dim
            # multiplies its extent.  Not a view — materialises.
            rep = attr_of(op, "shape", "repeats")
            base = cast("tuple", shapes[0])
            if not isinstance(rep, (tuple, list)) or not rep:
                return base
            if len(rep) < len(base):
                # torch requires len(reps) >= ndim — a malformed term;
                # report the base rather than a fabricated shape.
                return base
            b_pad = (1,) * (len(rep) - len(base)) + tuple(base)
            return tuple(
                None if (b is None or r is None) else b * r
                for b, r in zip(b_pad, rep, strict=True)
            )
        case "permute":
            base = shapes[0]
            if not base:
                return None
            dims = attr_of(op, "dim", "dims", "order")
            if not isinstance(dims, (tuple, list)) or not dims:
                # permute() with no dims reverses all axes.
                return tuple(reversed(base))
            n = len(base)
            return tuple(base[int(d) % n] for d in dims)
        case "movedim":
            base = shapes[0]
            if not base:
                return None
            src = _ax_tuple(attr_of(op, "source"))
            dst = _ax_tuple(attr_of(op, "destination"))
            if not src or not dst or len(src) != len(dst):
                return base
            n = len(base)
            src_n = [s % n for s in src]
            order = [i for i in range(n) if i not in set(src_n)]
            for s, d in sorted(
                zip(src_n, (d % n for d in dst), strict=True),
                key=lambda p: p[1],
            ):
                order.insert(d, s)
            return tuple(base[i] for i in order)
        case "expand_as":
            return shapes[1] if len(shapes) > 1 else shapes[0]
        case "stack":
            # stack(ts, dim): all inputs share a shape; insert dim.
            base = shapes[0]
            if (
                base is None
            ):  # pragma: no cover — dispatch filters None shapes
                return None
            d = attr_of(op, "dim", default=0)
            d = d % (len(base) + 1)
            n = len(op.args)
            return (*tuple(base[:d]), n, *tuple(base[d:]))
        case "unbind":
            # Element shape: base with the unbound dim removed.  The
            # tuple arity lives in getitem/select consumers.
            base = shapes[0]
            if not base:
                # () has no dim to unbind — carrier-internal operand.
                return None
            d = attr_of(op, "dim", default=-1) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "getitem":
            # After unbind the element shape is already the arg's shape.
            return shapes[0]
        case "select":
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=0) % len(base)
            return tuple(x for i, x in enumerate(base) if i != d)
        case "slice":
            # aten.slice(t, dim, start, end, step) — step bounds the
            # stride (torch.export spells it positionally).  Ignoring
            # it makes strided slices (x[..., ::2], RoPE) report the
            # unsliced shape and poisons every downstream broadcast as
            # _INVALID.
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=0) % len(base)
            lo = op.attrs.get("start", 0) or 0
            hi = op.attrs.get("end")
            step = op.attrs.get("step", 1) or 1
            out = list(base)
            bd = base[d]
            if isinstance(bd, int):
                n = (min(hi, bd) if isinstance(hi, int) else bd) - lo
                out[d] = max(0, -(-n // step))
            return tuple(out)
        case "embedding":
            # Row gather: out = idx.shape + (d,) where W is (v, d).
            wsh, ishape = shapes[0], shapes[1]
            if wsh is None or len(wsh) != 2:
                return None
            return (*tuple(ishape or ()), wsh[1])
        case "index_select":
            # Gather along dim: that axis resizes to len(index).
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=0) % len(base)
            idx = attr_of(op, "index")
            out = list(base)
            out[d] = (
                len(idx) if isinstance(idx, (tuple, list)) else None
            )
            return tuple(out)
        case "gather" | "take_along_dim" | "searchsorted":
            # gather(t, dim, index) — the output IS the index shape.
            # searchsorted(sorted, values) reports the values shape.
            if len(shapes) < 2:
                return shapes[0] or None
            return shapes[1] or None
        case "index":
            return _index_shape(op, shapes)
        case "unflatten":
            # unflatten(t, dim, sizes): splice the size tuple in at dim.
            base = cast("tuple", shapes[0])
            if not base:
                return None
            sizes = attr_of(op, "sizes")
            if not isinstance(sizes, (tuple, list)) or not sizes:
                return base
            d = attr_of(op, "dim", default=0) % len(base)
            sz = list(sizes)
            if -1 in sz and isinstance(base[d], int):
                known = 1
                for s in sz:
                    if s != -1:
                        known *= (
                            s if isinstance(s, int) and s > 0 else 1
                        )
                inf = (
                    base[d] // known
                    if known and base[d] % known == 0
                    else None
                )
                sz = [inf if s == -1 else s for s in sz]
            return (*tuple(base[:d]), *tuple(sz), *tuple(base[d + 1 :]))
        case "narrow":
            # narrow(t, dim, start, length) = a length-bounded slice.
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=0) % len(base)
            out = list(base)
            out[d] = attr_of(op, "length")
            return tuple(out)
        case "pad":
            # F.pad constant: trailing dims get +left/+right extents.
            base = shapes[0]
            if not base:
                return None
            pads = attr_of(op, "pad")
            if not isinstance(pads, (tuple, list)) or not pads:
                return base
            out = list(base)
            for i in range(len(pads) // 2):
                ax = len(out) - 1 - i
                if ax < 0:
                    break
                lo, hi = pads[2 * i], pads[2 * i + 1]
                if out[ax] is not None and all(
                    isinstance(v, int) for v in (lo, hi)
                ):
                    out[ax] += lo + hi
                else:
                    out[ax] = None
            return tuple(out)
        case "glu":
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=-1) % len(base)
            out = list(base)
            out[d] = out[d] // 2 if isinstance(out[d], int) else None
            return tuple(out)
        case "nonzero":
            # nonzero(t): (nnz, ndim) — nnz is data-dependent.
            base = shapes[0]
            return (None, len(base) if isinstance(base, tuple) else 0)
        case "one_hot":
            base = shapes[0]
            if base is None:  # pragma: no cover — dispatch filters None
                return None
            nc = attr_of(op, "num_classes")
            return (
                *tuple(base),
                nc if isinstance(nc, int) and nc > 0 else None,
            )
        case "outer":
            # outer(u, v) — flatten each operand to 1-D, take products.
            return (_numel(shapes[0]), _numel(shapes[1]))
        case "diag_sum":
            # torch.trace — scalar sum of the diagonal (aten `trace`
            # maps here; the carrier ``trace`` is a different op).
            return ()
        case "item" | "numel":
            return ()
        case "einsum":
            # The equation string is opaque to the structural
            # walker — honest unknown rather than a wrong guess.
            return None
        case "addmm" | "addbmm" | "addbmm" | "baddbmm" | "addmv":
            # bias + a @ b (addmv: bias + a @ v): the GEMM part
            # shapes like matmul on args 1/2, then the bias broadcasts.
            a, b = cast("tuple", shapes[1]), cast("tuple", shapes[2])
            if len(a) >= 2 and len(b) >= 2:
                mm = (*tuple(a[:-1]), b[-1])
            elif len(a) >= 2 and len(b) == 1:
                mm = tuple(a[:-1])
            else:
                return shapes[0] or None
            return _broadcast(mm, shapes[0])
        case "conv1d":
            # x (N,C,L) @ w (O,C,k) -> (N,O,L')
            x, w = cast("tuple", shapes[0]), cast("tuple", shapes[1])
            if (
                x is None
                or w is None
                or len(x) < 3
                or len(w) < 3
                or isinstance(op.attrs.get("padding"), str)
            ):
                return (
                    (x[0], w[0], None)
                    if (
                        x is not None
                        and w is not None
                        and len(x) >= 1
                        and len(w) >= 1
                    )
                    else (x or None)
                )
            st = op.attrs.get("stride", 1)
            pd = op.attrs.get("padding", 0)
            dl = op.attrs.get("dilation", 1)
            st = st[0] if isinstance(st, (tuple, list)) else st
            pd = pd[0] if isinstance(pd, (tuple, list)) else pd
            dl = dl[0] if isinstance(dl, (tuple, list)) else dl
            ol = None
            if x[2] is not None and w[2] is not None:
                ol = (x[2] + 2 * pd - dl * (w[2] - 1) - 1) // st + 1
            return (x[0], w[0], ol)
        case "tensor_split":
            # Folded like split: ``sections`` is the count (or index
            # list), ``index`` the folded section index.
            base = cast("tuple", shapes[0])
            if not base:
                return None
            dim = attr_of(op, "dim", default=0) % len(base)
            sec = attr_of(op, "sections")
            idx = attr_of(op, "index", default=0) or 0
            out = list(base)
            if isinstance(sec, int):
                out[dim] = (
                    -(-base[dim] // sec)
                    if isinstance(base[dim], int)
                    else None
                )
            elif isinstance(sec, (tuple, list)) and isinstance(
                base[dim], int
            ):
                bounds = [0] + [int(s) for s in sec] + [base[dim]]
                out[dim] = (
                    bounds[idx + 1] - bounds[idx]
                    if idx + 1 < len(bounds)
                    else None
                )
            else:
                out[dim] = None
            return tuple(out)
        case "unfold":
            # unfold(dim, size, step): dim shrinks to sliding windows
            # and a new trailing axis of ``size`` appears.
            base = shapes[0]
            if not base:
                return None
            d = attr_of(op, "dim", default=0) % len(base)
            size = attr_of(op, "size")
            step = attr_of(op, "step", default=1) or 1
            out = list(base)
            if isinstance(out[d], int) and isinstance(size, int):
                out[d] = max(0, (out[d] - size) // step + 1)
            else:
                out[d] = None
            out.append(size if isinstance(size, int) else None)
            return tuple(out)
        case "pixel_shuffle":
            # (N, C*r^2, H, W) -> (N, C, H*r, W*r)
            base = cast("tuple", shapes[0])
            r = attr_of(op, "upscale_factor", default=1)
            if len(base) < 4 or not isinstance(r, int):
                return base
            c = base[1]
            return (
                base[0],
                c // (r * r) if isinstance(c, int) else None,
                base[2] * r if isinstance(base[2], int) else None,
                base[3] * r if isinstance(base[3], int) else None,
            )
        case "pixel_unshuffle":
            # (N, C, H, W) -> (N, C*r^2, H/r, W/r)
            base = cast("tuple", shapes[0])
            r = attr_of(op, "downscale_factor", default=1)
            if len(base) < 4 or not isinstance(r, int):
                return base
            c = base[1]
            return (
                base[0],
                c * r * r if isinstance(c, int) else None,
                base[2] // r if isinstance(base[2], int) else None,
                base[3] // r if isinstance(base[3], int) else None,
            )
        case "hstack" | "vstack":
            a = cast("tuple", shapes[0])
            if not a:
                return None
            if op.op == "hstack":
                dim = 0 if len(a) == 1 else 1
            else:
                # vstack promotes 1-D rows to (1, d) then cats dim 0.
                if len(a) < 2:
                    return (len(op.args), a[0] if a else None)
                dim = 0
            out = list(a)
            out[dim] = 0
            for s in shapes:
                if not isinstance(s, tuple) or len(s) != len(a):
                    return a
                out[dim] += s[dim] or 0
            return tuple(out)
        case "flatten":
            base = shapes[0]
            if not base:
                return None
            d0 = attr_of(op, "start_dim", default=0)
            d1 = attr_of(op, "end_dim", default=-1)
            d0, d1 = d0 % len(base), d1 % len(base)
            merged = _numel(base[d0 : d1 + 1])
            return (*tuple(base[:d0]), merged, *tuple(base[d1 + 1 :]))
        case (
            "contiguous"
            | "to"
            | "type_as"
            | "float"
            | "dropout"
            | "alias"
        ):
            return shapes[0]
        case "sdpa":
            # out has q's shape (B, h, T, d)
            return shapes[0] or None
        case "conv2d":
            # x (N,C,H,W) @ w (O,C,kh,kw) -> (N,O,H',W')
            x, w = shapes[0], shapes[1]
            if (
                not isinstance(x, tuple)
                or not isinstance(w, tuple)
                or len(x) < 4
                or len(w) < 4
                or isinstance(op.attrs.get("padding"), str)
            ):
                return (
                    (x[0], w[0], None, None)
                    if (
                        x is not None
                        and w is not None
                        and len(x) >= 1
                        and len(w) >= 1
                    )
                    else (x or None)
                )
            st = op.attrs.get("stride", 1)
            pd = op.attrs.get("padding", 0)
            dl = op.attrs.get("dilation", 1)
            st = st if isinstance(st, (tuple, list)) else (st, st)
            pd = pd if isinstance(pd, (tuple, list)) else (pd, pd)
            dl = dl if isinstance(dl, (tuple, list)) else (dl, dl)
            oh = ow = None
            xh, xw = x[2], x[3]
            wh, ww = w[2], w[3]
            if xh is not None and wh is not None:
                oh = (xh + 2 * pd[0] - dl[0] * (wh - 1) - 1) // st[
                    0
                ] + 1
            if xw is not None and ww is not None:
                ow = (xw + 2 * pd[1] - dl[1] * (ww - 1) - 1) // st[
                    1
                ] + 1
            return (x[0], w[0], oh, ow)
        case "inv":
            return shapes[0]
        case "concat":
            # Variadic cat: sum every operand along the cat axis.
            a = shapes[0]
            if not a:
                # () has no cat axis — carrier-internal operand.
                return None
            dim = attr_of(op, "dim", default=0) % len(a)
            out = list(a)
            out[dim] = 0
            for s in shapes:
                if not isinstance(s, tuple) or len(s) != len(a):
                    return a
                out[dim] += s[dim] or 0
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
            # ``sizes`` is the per-section size (or a size list);
            # ``index`` is the section index.
            sizes = attr_of(op, "sizes")
            idx = attr_of(op, "index", default=0) or 0
            out = list(base)
            if isinstance(sizes, (tuple, list)):
                out[dim] = sizes[idx] if idx < len(sizes) else None
            elif isinstance(sizes, int):
                out[dim] = sizes  # equal-size sections
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
#: cost model must never prefer them.  It is a ``str`` (a sentinel
#: value), so shape results are typed ``tuple | str | None``.
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
    for da, db in zip(a_pad, b_pad, strict=True):
        if da is not None and da < 0:
            da = None  # unresolved -1: treat as unknown, not a mismatch
        if db is not None and db < 0:
            db = None
        if da is None or db is None:
            out.append(None)
        elif da == 1:
            out.append(db)
        elif db == 1 or da == db:
            out.append(da)
        else:
            return _INVALID  # provably ill-typed
    return tuple(out)


def _dim_eq(x, y) -> bool:
    """Check dim equality for the matmul contraction check.

    A ``None`` (unknown) dim is a wildcard, matching
    ``_broadcast``'s convention.
    """
    return x is None or y is None or x == y


def _matmul_shape(a, b):
    """``torch.matmul`` output shape for operand shapes ``a``, ``b``.

    Full torch semantics: rank-1 promotion/demotion, batch-dim
    broadcasting for rank ≥ 2 operands.  Returns ``None`` when the
    contraction is invalid (inner-dim mismatch or a scalar operand)
    and the result tuple otherwise (which may carry ``None`` dims).
    """
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        return None
    if not a or not b:
        return None
    if len(a) == 1 and len(b) == 1:
        return () if _dim_eq(a[0], b[0]) else None
    if len(b) == 1:
        if len(a) < 2 or not _dim_eq(a[-1], b[0]):
            return None
        return tuple(a[:-1])
    if len(a) == 1:
        if not _dim_eq(a[0], b[-2]):
            return None
        return (*tuple(b[:-2]), b[-1])
    batch = _broadcast(tuple(a[:-2]), tuple(b[:-2]))
    if batch is _INVALID or not _dim_eq(a[-1], b[-2]):
        return None
    return (*batch, a[-2], b[-1])


def _numel(shape) -> int:
    if shape is None:
        return 1
    result = 1
    for d in shape:
        result *= d if d is not None else 1
    return result


# ---------------------------------------------------------------------------
#  Term predicates shared across cost / lowering / laws
# ---------------------------------------------------------------------------


def has_var_leaf(term: Any, memo: dict | None = None) -> bool:
    """Return True iff the subtree mentions a :class:`Var` leaf.

    Param-only subtrees fold at compile time — this predicate is the
    activation/weight distinction used by the cost model's DAG
    accounting and parameter-fold pricing (``cost.dag_cost`` /
    ``cost._folds_to_param``), the torch bridge's weight-chain
    folding, and the pairing pass.

    ``memo`` is a content-keyed dict threaded across a traversal so
    shared-subterm DAGs stay a linear walk (terms are interned
    content objects — safe dict keys, no ``id()``/GC hazards).  Keys
    are ``("hv", term)``-prefixed so the memo may be SHARED with other
    content-keyed helpers (``_shape_of`` keys on bare ``term``) — as
    ``param_bytes_cost`` does — without value collisions; a private
    per-call dict works identically.
    """
    memo = {} if memo is None else memo
    k = ("hv", term)
    hit = memo.get(k)
    if hit is not None:
        return hit
    if isinstance(term, Var):
        out = True
    elif isinstance(term, Op):
        out = any(has_var_leaf(a, memo) for a in term.args)
    else:
        out = False
    memo[k] = out
    return out


def _concrete(shape: Any) -> bool:
    """Return True iff *shape* is a tuple of concrete int dims."""
    return (
        isinstance(shape, tuple)
        and len(shape) > 0
        and all(isinstance(d, int) for d in shape)
    )


def _stack_dim(attrs: dict) -> int:
    """Return the concatenation/stack axis from an op's attrs.

    Uses the canonical ``dim`` spelling, defaulting to 0 on a non-int
    value.
    """
    d = attr_of(attrs, "dim", default=0)
    return d if isinstance(d, int) else 0


def _ax_tuple(v: Any) -> tuple | None:
    """Return an axis attr — an int, list, or tuple — as a tuple."""
    if isinstance(v, int):
        return (v,)
    if isinstance(v, (list, tuple)):
        return tuple(int(d) for d in v)
    return None


def _axis_pair(d0: Any, d1: Any, rank: int) -> tuple[int, int] | None:
    """Normalise a two-axis ``transpose`` pair against ``rank``.

    Returns ``(d0 % rank, d1 % rank)`` when both dims are ints in
    ``[-rank, rank)`` — the canonical per-operand form the axis readers
    above use.  ``None`` for a rank-0 operand (no axes to swap), a
    non-int dim, or a dim outside torch's accepted range.  Law
    check-hooks (:mod:`catopt_core.laws.layout`) use it to decide
    whether a bound ``transpose`` applies the axis swap their rewrite
    assumes.
    """
    if rank < 1:
        return None
    if not (isinstance(d0, int) and isinstance(d1, int)):
        return None
    if not (-rank <= d0 < rank and -rank <= d1 < rank):
        return None
    return (d0 % rank, d1 % rank)


def _index_shape(op: Op, shapes: list) -> tuple | str | None:
    """``index`` — numpy-style advanced indexing.

    The term records a ``layout`` attr: one entry per key position —
    ``True`` where an index operand sits, ``False`` for a full slice
    (exported ``aten.index`` spells ``x[:, i]`` as ``[None, i]`` and
    the Nones carry real semantic weight: without them the term
    ``x[:, i]`` would lower as ``x[i]``).  Numpy semantics apply:

    * indexed positions that are *contiguous* in the key splice the
      broadcast index shape in where they stood;
    * *separated* index positions move the broadcast index block to
      the front and every preserved axis follows in order;
    * axes beyond the key length pass through untouched.
    """
    base = shapes[0]
    if not base:
        return None
    idx_shapes = [s for s in shapes[1:]]
    ib: Any = idx_shapes[0] if idx_shapes else ()
    for s in idx_shapes[1:]:
        ib = _broadcast(ib, s)
        if ib is _INVALID:
            return _INVALID
    layout = attr_of(op, "layout")
    if not isinstance(layout, (tuple, list)) or not layout:
        # A term minted without layout treats its operands as
        # indexing the leading axes — the ``x[i]`` spelling.
        layout = (True,) * len(idx_shapes)
    consumed = min(len(layout), len(base))
    kept_before: list = []
    kept_after: list = []
    idx_pos = [i for i, flag in enumerate(layout) if flag]
    first_idx = idx_pos[0] if idx_pos else len(layout)
    last_idx = idx_pos[-1] if idx_pos else -1
    contiguous = idx_pos and last_idx - first_idx + 1 == len(idx_pos)
    trailing = list(base[consumed:])
    if contiguous:
        for i, flag in enumerate(layout[:consumed]):
            if flag:
                continue
            if i < first_idx:
                kept_before.append(base[i])
            else:
                kept_after.append(base[i])
        return (
            *kept_before,
            *(ib or ()),
            *kept_after,
            *trailing,
        )
    # Separated index positions: the index block moves to the front.
    kept = [
        base[i] for i, flag in enumerate(layout[:consumed]) if not flag
    ]
    return (*(ib or ()), *kept, *trailing)


# ---------------------------------------------------------------------------
# Per-op shape rules — the carrier ops' extensible dispatch
# ---------------------------------------------------------------------------
#
# A carrier op's convention-vs-value semantics is per-op DATA, not a
# shared match arm everyone edits: each family's resolver is registered
# into ``_SHAPE_RULES`` via :func:`register_shape_rule` and
# ``_infer_op_shape`` dispatches on ``op.op``.  Handlers run after
# ``_INVALID`` propagation and the unknown/empty early-returns, so they
# see only non-``None`` operand shapes; they still must map a
# carrier-internal ``()`` operand to ``None`` (unknown), never report
# a fabricated scalar.

_ShapeRule = Callable[[Op, list], tuple | str | None]

#: ``{op_name: fn(op, shapes) -> shape}`` — consulted by
#: ``_infer_op_shape`` after the ``_INVALID``/unknown early-returns
#: (or immediately for zero-argument ops, which get ``shapes = []``).
_SHAPE_RULES: dict[str, _ShapeRule] = {}


def register_shape_rule(op: str, fn: _ShapeRule) -> None:
    """Register ``fn(op, shapes)`` as the shape rule for ``op``.

    ``shapes`` is the list of inferred operand shapes — no ``None`` and
    no ``_INVALID`` members (those short-circuit in
    ``_infer_op_shape`` first); for a zero-argument op it is ``()``.
    The rule returns a shape ``tuple``, ``_INVALID``, or ``None``
    (unknown); a carrier-internal ``()`` operand must map to ``None``,
    never a fabricated scalar.
    """
    _SHAPE_RULES[op] = fn


# --- carrier-family registrations -------------------------------------
#
# These live in typing.py for now (Phase 1a): the registrations below
# are exactly the arms lifted out of the old ``_infer_op_shape`` match.
# Later phases move each family next to its laws in its own module
# (om.py, xcarrier.py, trace.py) — the mechanism stays.


def _carrier_shape(op: Op, shapes: list) -> tuple | str | None:
    """Convention shape = first member's, ``()`` -> unknown.

    Covers ``aff`` (the pair h |-> A·h + b is priced from its linear
    part), ``aff_compose`` (f∘g keeps the outer map's linear-part
    shape), ``aff_diag`` (the diagonal pair priced from its scale
    part), ``affd_compose``, and ``om_compose``/``om_apply``.  A ``()``
    slot means a carrier member is being read as a tensor: unknown,
    not scalar (see :func:`_shape_of`).
    """
    return shapes[0] or None


def _apply_shape(op: Op, shapes: list) -> tuple | str | None:
    """``apply``/``applyd``: evaluates back to tensor-land — h's shape."""
    return shapes[1] or None


def _om_shape(op: Op, shapes: list) -> tuple | str | None:
    """``om`` — the carrier triple (m, l, a).

    Its "shape" is the accumulator's — what consumers' costs are priced
    from.
    """
    out = shapes[2] if len(shapes) > 2 else shapes[0]
    return out or None


def _om_elem_shape(op: Op, shapes: list) -> tuple | str | None:
    """``om_elem`` — elem(s[...,K], v[...,K,d]).

    Reports the applied output shape (...,T,d) — like ``aff``, the
    carrier is priced as the tensor it will become under om_apply.
    """
    s, v = shapes[0], shapes[1]
    if (
        isinstance(s, tuple)
        and isinstance(v, tuple)
        and len(s) >= 1
        and len(v) >= 1
    ):
        return (*tuple(s[:-1]), v[-1])
    # A ()-shaped operand is a carrier member read as a tensor —
    # unknown, not scalar.
    return None


def _trace_shape(op: Op, shapes: list) -> tuple | str | None:
    """``trace`` — Tr(f).

    Drops the first ``usize`` rows/cols of the block matrix
    f : U⊗X → U⊗Y, leaving the X → Y map.
    """
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


def _bdiag_shape(op: Op, shapes: list) -> tuple | str | None:
    """``bdiag``/``parl`` — total-dims-preserving matrix juxtaposition.

    bdiag is literal block-diagonal; parl re-lays the same blocks
    keeping feedback wires first (see catopt_carriers.trace).
    """
    a, b = shapes[0], shapes[1]
    if (
        isinstance(a, tuple)
        and isinstance(b, tuple)
        and len(a) == 2
        and len(b) == 2
        and all(isinstance(d, int) for d in (*a, *b))
    ):
        return (a[0] + b[0], a[1] + b[1])
    return a or None


def _eye_shape(op: Op, shapes: list) -> tuple | str | None:
    """``eye`` — zero-arg identity morphism; attrs give the dim."""
    d = op.attrs.get("dim", op.attrs.get("d", 1))
    return (d, d) if isinstance(d, int) else (None, None)


def _cswap_shape(op: Op, shapes: list) -> tuple | str | None:
    """``cswap`` — zero-arg swap morphism on d1+d2 wires."""
    d1, d2 = op.attrs.get("d1", 0), op.attrs.get("d2", 0)
    if isinstance(d1, int) and isinstance(d2, int):
        return (d1 + d2, d1 + d2)
    return None


def _factory_shape(op: Op, shapes: list) -> tuple | str | None:
    """Infer the output shape of a tensor creator.

    ``zeros``/``ones``/``empty``/``full``/``randn``/``rand``/
    ``new_*``: the ``shape`` attr names the output.  ``full``/
    ``new_full`` carry a Const fill operand whose shape is ignored.
    """
    s = op.attrs.get("shape")
    return tuple(s) if isinstance(s, (list, tuple)) else None


def _arange_shape(op: Op, shapes: list) -> tuple | str | None:
    """``arange`` — a 1-D extent from the bound args.

    Const operands (minted spelling) or the exported
    ``arg0/arg1/arg2`` attrs.  ``arange(end)``, ``arange(start,
    end)``, ``arange(start, end, step)`` share the one spelling.
    """
    vals = [a.value for a in op.args if isinstance(a, Const)]
    if not vals:
        vals = [
            op.attrs[k]
            for k in sorted(
                (k for k in op.attrs if is_positional_attr(k)),
                key=lambda k: int(k[3:]),
            )
        ]
    if not vals:
        return (None,)
    if len(vals) == 1:
        start, end, step = 0, vals[0], 1
    elif len(vals) == 2:
        start, end, step = vals[0], vals[1], 1
    else:
        start, end, step = vals[0], vals[1], vals[2]
    if (
        not all(isinstance(v, (int, float)) for v in (start, end, step))
        or step == 0
    ):
        return (None,)  # pragma: no cover — Const values are numeric
    return (max(0, math.ceil((end - start) / step)),)


for _n in (
    "zeros",
    "ones",
    "empty",
    "randn",
    "rand",
    "full",
    "new_zeros",
    "new_ones",
    "new_empty",
    "new_full",
):
    register_shape_rule(_n, _factory_shape)
register_shape_rule("arange", _arange_shape)

for _n in (
    "aff",
    "aff_compose",
    "aff_diag",
    "affd_compose",
    "om_compose",
    "om_apply",
):
    register_shape_rule(_n, _carrier_shape)
register_shape_rule("apply", _apply_shape)
register_shape_rule("applyd", _apply_shape)
register_shape_rule("om", _om_shape)
register_shape_rule("om_elem", _om_elem_shape)
register_shape_rule("trace", _trace_shape)
register_shape_rule("bdiag", _bdiag_shape)
register_shape_rule("parl", _bdiag_shape)
register_shape_rule("eye", _eye_shape)
register_shape_rule("cswap", _cswap_shape)
