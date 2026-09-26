# ruff: noqa: RUF002, RUF003
"""Interval bound propagation (IBP) — tighter whole-model ε certificates.

``eps.model_bound`` propagates each certified site's bound to the output
through *global* per-op Lipschitz constants — spectral products.  Those
products are honest but loose: a ReLU whose pre-activations are all
negative still costs ×1.0, a weight-side edge is billed
``input_norm × whole-graph sensitivity`` instead of the *actual*
activation norm at that layer, and every sigmoid is charged its worst-
case slope 0.25 even when its inputs are saturated.

This module replaces the global constants with **intervals**:

* :func:`ibp_bound` — propagate ``[lo, hi]`` boxes through the IR ops
  that have sound interval rules (affine/elementwise/structural ops,
  interval matmul, a tight softmax rule).  Ops without a rule produce
  ±∞ boxes and are reported in ``unsupported`` — nothing is silently
  assumed tight.

* :func:`tight_model_bound` — the ``model_bound`` analogue: for every
  bound-carrying certificate step, locate its produced member in the
  extracted term, then walk the path to the root multiplying by
  *local* Lipschitz constants evaluated on the activation boxes (dead
  ReLU → 0, saturated sigmoid → its actual slope, weight-side edge →
  the measured max-row norm of the real activation, not a sensitivity
  product).  Sites whose path crosses an op with no local rule fall
  back to the spectral-path contribution of :func:`eps.model_bound`,
  so the result is never worse than spectral — and the result dict
  says so honestly when it isn't better.

Perturbations are tracked as **max-row-L2 bounds** (a "row" is the last
dimension).  A certificate bound ``b`` on a substituted tensor means
every row of the error is within ``b`` (spectral: row ≤ σ_max;
Frobenius: worst-case single row ≤ ‖·‖_F), and the output max-abs error
is within the row bound.  For sites sitting in a *weight slot*
(``arg1`` of ``linear``/``matmul``) the bound is carried as a spectral
bound on ΔW instead — ``y = x·ΔWᵀ`` gives ``‖δy_row‖ ≤ ‖x_row‖·‖ΔW‖₂``
directly, without paying a √m conversion.

Soundness caveat (documented, not hidden): local constants are evaluated
on activation boxes widened by the accumulated upstream perturbation
radii (an iterate-until-stable loop, ``max_iter`` rounds).  If the
iterate does not converge the reported bound is the last — largest —
iterate and ``converged`` is ``False`` in the result dict.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
from catopt_core.attrs import attr_of
from catopt_core.ir import IR, Const, Op, Param, Var
from catopt_core.typing import _shape_of

from catopt_eps.eps import (
    _find_subterms,
    _path_sensitivity,
    model_bound,
)

__all__ = ["Box", "ibp_bound", "tight_model_bound"]


class Box(NamedTuple):
    """An elementwise interval ``lo ≤ x ≤ hi``."""

    lo: torch.Tensor
    hi: torch.Tensor

    @property
    def width(self) -> torch.Tensor:
        return self.hi - self.lo

    @property
    def maxabs(self) -> torch.Tensor:
        return torch.maximum(self.lo.abs(), self.hi.abs())


# ---------------------------------------------------------------------------
#  Scalar bounds derived from a box
# ---------------------------------------------------------------------------


def _fro_bound(b: Box) -> float:
    """Upper bound on ‖x‖_F over the box."""
    return float(torch.linalg.norm(b.maxabs.double()))


def _maxrow_bound(b: Box) -> float:
    """Upper bound on the maximum row (last-dim) L2 norm over the box."""
    m = b.maxabs
    if m.ndim == 0:
        return float(m)
    rows = torch.linalg.norm(m.double(), dim=-1)
    return float(rows.max()) if rows.numel() else float(m)


def _spec_bound(b: Box) -> float:
    """Upper bound on σ_max(x) over the box (matrices / batches).

    For entrywise-nonnegative A ≤ B one has σ_max(A) ≤ σ_max(B), so the
    spectral norm of the elementwise max dominates every matrix in the
    box.  For 0/1-D boxes this is just max|x| (the right operator bound
    for elementwise use)."""
    m = b.maxabs.double()
    try:
        if m.ndim >= 2:
            return float(torch.linalg.matrix_norm(m, 2).max())
        return float(m.abs().max()) if m.numel() else 0.0
    except Exception:
        return float(torch.linalg.norm(m))


def _minabs(b: Box) -> float:
    """min |x| over the box — 0 when the box straddles zero."""
    lo, hi = b.lo, b.hi
    straddle = (lo <= 0) & (hi >= 0)
    m = torch.minimum(lo.abs(), hi.abs())
    m = torch.where(straddle, torch.zeros_like(m), m)
    return float(m.min()) if m.numel() else float(m)


# ---------------------------------------------------------------------------
#  Interval arithmetic primitives
# ---------------------------------------------------------------------------


def _pt(t: torch.Tensor) -> Box:
    return Box(t, t)


def _add(a: Box, b: Box) -> Box:
    return Box(a.lo + b.lo, a.hi + b.hi)


def _sub(a: Box, b: Box) -> Box:
    return Box(a.lo - b.hi, a.hi - b.lo)


def _neg(a: Box) -> Box:
    return Box(-a.hi, -a.lo)


def _mul(a: Box, b: Box) -> Box:
    ps = torch.stack(
        [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    )
    return Box(ps.amin(0), ps.amax(0))


def _div(a: Box, b: Box) -> Box | None:
    """Interval division — unbounded (None) when 0 ∈ b."""
    if _minabs(b) <= 0:
        return None
    rec = Box(1.0 / b.hi, 1.0 / b.lo)
    return _mul(a, rec)


def _matmul(a: Box, b: Box) -> Box:
    """Interval matrix product via centre/radius (Rump bound).

    For A = Ac ± Ar, B = Bc ± Br the product set is enclosed by
    Ac·Bc ± (Ar|Bc| + |Ac|Br + Ar·Br).  Sound for batched matmuls —
    torch.matmul broadcasts the leading dims."""
    dt = torch.promote_types(a.lo.dtype, b.lo.dtype)
    alo, ahi = a.lo.to(dt), a.hi.to(dt)
    blo, bhi = b.lo.to(dt), b.hi.to(dt)
    ac = (alo + ahi) / 2
    ar = (ahi - alo) / 2
    bc = (blo + bhi) / 2
    br = (bhi - blo) / 2
    c = torch.matmul(ac, bc)
    r = (
        torch.matmul(ar, bc.abs())
        + torch.matmul(ac.abs(), br)
        + torch.matmul(ar, br)
    )
    return Box(c - r, c + r)


def _softmax(a: Box, dim: int) -> Box:
    """Tight elementwise softmax rule: with S⁻ = Σeˡᵒ, S⁺ = Σeʰⁱ,
    softmax_i ∈ [eˡᵒ_i/(eˡᵒ_i + S⁺ − eʰⁱ_i), eʰⁱ_i/(eʰⁱ_i + S⁻ − eˡᵒ_i)]."""
    el, eh = a.lo.exp(), a.hi.exp()
    sl = el.sum(dim, keepdim=True)
    sh = eh.sum(dim, keepdim=True)
    lo = el / (el + (sh - eh))
    hi = eh / (eh + (sl - el))
    return Box(lo, hi)


#: Unary ops that are monotonically increasing everywhere.
_MONO_INC = {"relu", "sigmoid", "tanh", "exp"}
#: Unary ops with a single interior minimum at a known point.
_INTERIOR_MIN = {
    "silu": -1.2784645427610738,
    "gelu": -0.7517907364130968,
}


def _unary(b: Box, op: str, fn) -> Box | None:
    """Interval image of a scalar function: endpoints plus the known
    interior minimum for silu/gelu (single critical point)."""
    if op in _MONO_INC:
        return Box(fn(b.lo), fn(b.hi))
    if op == "neg":
        return _neg(b)
    if op == "sqrt":
        if float(b.lo.min()) < 0:
            return None  # NaN domain — unsupported
        return Box(fn(b.lo), fn(b.hi))
    if op == "rsqrt":
        if float(b.lo.min()) <= 0:
            return None
        return Box(fn(b.hi), fn(b.lo))  # decreasing
    if op in _INTERIOR_MIN:
        lo, hi = fn(b.lo), fn(b.hi)
        xs = _INTERIOR_MIN[op]
        mask = (b.lo < xs) & (b.hi > xs)
        if mask.any():
            ys = fn(torch.tensor(xs, dtype=b.lo.dtype))
            lo = torch.where(mask, torch.minimum(lo, ys), lo)
        return Box(torch.minimum(lo, hi), torch.maximum(lo, hi))
    if op == "square":
        m = b.maxabs
        lo = torch.where(
            (b.lo <= 0) & (b.hi >= 0),
            torch.zeros_like(m),
            torch.minimum(b.lo.abs(), b.hi.abs()) ** 2,
        )
        return Box(lo, m**2)
    return None


#: Structural ops — exact maps applied to both endpoints.  For
#: multi-operand ops the *non-first* operands are taken from ``.lo``
#: (they are indices/params — required to be point boxes by the caller).
_STRUCTURAL = {
    "reshape",
    "view",
    "transpose",
    "contiguous",
    "float",
    "to",
    "alias",
    "type_as",
    "clone",
    "dropout",
    "select",
    "slice",
    "unsqueeze",
    "squeeze",
    "flatten",
    "getitem",
    "chunk",
    "split",
    "unbind",
    "expand",
    "broadcast",
    "index_select",
}


def _eval_concrete(t: Any, env: dict, values: dict):
    """Evaluate a term to a tensor: Params from ``env``, Vars from
    ``values`` (concrete inputs), Consts by value, ops via the torch
    bindings.  ``None`` when anything is not concretely evaluatable.

    Delegates to :func:`catopt_torch.torch_bridge.eval_term` (plan 0002
    phase D) — permissive mode, and unlike ``optimize._eval_const``
    deliberately NOT ``tensor_only``: carrier bindings (``aff``,
    ``om``, ...) return tuples that are legitimate concrete values.
    """
    from catopt_torch.torch_bridge import _IR_TO_TORCH, eval_term

    return eval_term(
        t, var_env=values, param_env=env, bindings=_IR_TO_TORCH
    )


def _inf_box(t: Any) -> Box:
    """±∞ box with the term's inferred shape (scalar when unknown)."""
    shape = _shape_of(t)
    if (not isinstance(shape, tuple)
            or any(d is None for d in shape)):
        inf = torch.tensor(float("inf"))
        return Box(-inf, inf)
    lo = torch.full(tuple(int(d) for d in shape), float("-inf"))
    return Box(lo, -lo)


def _as_float(t: torch.Tensor) -> torch.Tensor:
    return (
        t if t.is_floating_point() else t.to(torch.get_default_dtype())
    )


def _ibp_eval(
    t: Any,
    env: dict,
    input_box: dict,
    values: dict,
    boxes: dict,
    path: tuple,
    unsupported: list,
    site_boxes: dict | None,
) -> Box:
    """Recursive interval evaluator; records every node's box in
    ``boxes`` (keyed by path) and flags rule-less ops in ``unsupported``."""
    from catopt_torch.torch_bridge import _IR_TO_TORCH

    box: Box | None = None
    if isinstance(t, Var):
        b = input_box.get(t.name)
        if b is not None:
            box = b
        elif t.name in values:
            v = _as_float(values[t.name])
            box = _pt(v)
        else:
            unsupported.append(f"var:{t.name}")
            box = _inf_box(t)
    elif isinstance(t, Param):
        w = env.get(t.name)
        if w is not None:
            wt = _as_float(w)
            box = _pt(wt)
        else:
            unsupported.append(f"param:{t.name}")
            box = _inf_box(t)
    elif isinstance(t, Const):
        box = _pt(torch.tensor(float(t.value)))
    elif isinstance(t, Op):
        argboxes = [
            _ibp_eval(
                a,
                env,
                input_box,
                values,
                boxes,
                (*path, i),
                unsupported,
                site_boxes,
            )
            for i, a in enumerate(t.args)
        ]
        op = t.op
        attrs = dict(t.attrs)
        fn = _IR_TO_TORCH.get(op)

        if op == "add":
            box = _add(argboxes[0], argboxes[1])
        elif op == "sub":
            box = _sub(argboxes[0], argboxes[1])
        elif op == "mul":
            box = _mul(argboxes[0], argboxes[1])
        elif op == "div":
            box = _div(argboxes[0], argboxes[1])
            if box is None:
                unsupported.append("div:0-in-divisor")
                box = _inf_box(t)
        elif op == "matmul":
            box = _matmul(argboxes[0], argboxes[1])
        elif op == "linear":
            w = argboxes[1]
            if w.lo.ndim >= 2:
                wt = Box(w.lo.transpose(-2, -1), w.hi.transpose(-2, -1))
                box = _matmul(argboxes[0], wt)
                if len(argboxes) >= 3:
                    box = _add(box, argboxes[2])
            else:
                unsupported.append("linear:1d-weight")
                box = _inf_box(t)
        elif op == "softmax":
            dim = int(attr_of(attrs, "arg1", "dim", default=-1))
            box = _softmax(argboxes[0], dim)
        elif op in ("sum", "mean") and fn is not None:
            # reductions are monotone in every input
            box = Box(
                fn(argboxes[0].lo, **attrs), fn(argboxes[0].hi, **attrs)
            )
        elif op in ("max", "min"):
            # no torch binding for these names — apply amax/amin on the
            # declared dim (whole tensor when absent); monotone in input
            a = argboxes[0]
            dim = attr_of(attrs, "dim", "arg1")
            keep = bool(attrs.get("keepdim", False))
            f = torch.amax if op == "max" else torch.amin
            if dim is None:
                box = Box(f(a.lo), f(a.hi))
            else:
                d = (
                    tuple(dim)
                    if isinstance(dim, (list, tuple))
                    else (int(dim),)
                )
                box = Box(
                    f(a.lo, dim=d, keepdim=keep),
                    f(a.hi, dim=d, keepdim=keep),
                )
        elif op == "pow":
            e = argboxes[1] if len(argboxes) > 1 else None
            if e is not None and torch.equal(e.lo, e.hi):
                ev = float(e.lo.flatten()[0])
                a = argboxes[0]
                if ev == 2:
                    box = _unary(a, "square", torch.square)
                elif ev >= 1 and float(a.lo.min()) >= 0:
                    box = Box(a.lo**ev, a.hi**ev)
                elif ev == 0:
                    box = _pt(torch.ones_like(a.lo))
                elif float(a.lo.min()) > 0:
                    box = (
                        Box(a.hi**ev, a.lo**ev)
                        if ev < 0
                        else Box(a.lo**ev, a.hi**ev)
                    )
            if box is None:
                unsupported.append("pow")
                box = _inf_box(t)
        elif (
            op in _MONO_INC
            or op in _INTERIOR_MIN
            or op in ("neg", "sqrt", "rsqrt", "square")
        ):
            if fn is not None:
                box = _unary(argboxes[0], op, fn)
            if box is None:
                unsupported.append(op)
                box = _inf_box(t)
        elif op == "embedding":
            idx = _eval_concrete(t.args[1], env, values)
            if idx is None:
                unsupported.append("embedding:idx")
                box = _inf_box(t)
            else:
                w = argboxes[0]
                box = Box(
                    torch.nn.functional.embedding(idx.long(), w.lo),
                    torch.nn.functional.embedding(idx.long(), w.hi),
                )
        elif op in ("concat", "stack", "cat"):
            if fn is not None:
                box = Box(
                    fn(*[b.lo for b in argboxes], **attrs),
                    fn(*[b.hi for b in argboxes], **attrs),
                )
        elif op in _STRUCTURAL and fn is not None:
            # exact map — apply to both endpoints; non-first operands
            # (indices, dims) are taken at their concrete value
            rest = [b.lo for b in argboxes[1:]]
            try:
                box = Box(
                    fn(argboxes[0].lo, *rest, **attrs),
                    fn(argboxes[0].hi, *rest, **attrs),
                )
            except Exception:
                box = None
        elif op == "where" and len(argboxes) == 3:
            c = _eval_concrete(t.args[0], env, values)
            if c is not None:
                box = Box(
                    torch.where(
                        c.bool(), argboxes[1].lo, argboxes[2].lo
                    ),
                    torch.where(
                        c.bool(), argboxes[1].hi, argboxes[2].hi
                    ),
                )
            else:
                # conservative hull of both branches
                box = Box(
                    torch.minimum(argboxes[1].lo, argboxes[2].lo),
                    torch.maximum(argboxes[1].hi, argboxes[2].hi),
                )
        elif op == "masked_fill" and len(argboxes) >= 2:
            m = _eval_concrete(t.args[1], env, values)
            v = (
                _eval_concrete(t.args[2], env, values)
                if len(t.args) > 2
                else None
            )
            try:
                vv = float(v) if v is not None else None
            except Exception:
                vv = None
            if m is not None and vv is not None:
                box = Box(
                    argboxes[0].lo.masked_fill(m.bool(), vv),
                    argboxes[0].hi.masked_fill(m.bool(), vv),
                )
        if box is None:
            # last resort: if every operand is a point box the op can be
            # evaluated exactly through its torch binding — this keeps
            # comparisons / exotic ops usable on point input boxes.
            if (
                fn is not None
                and argboxes
                and all(torch.equal(b.lo, b.hi) for b in argboxes)
            ):
                try:
                    out = fn(*[b.lo for b in argboxes], **attrs)
                    if isinstance(out, torch.Tensor):
                        box = _pt(_as_float(out))
                except Exception:
                    box = None
            if box is None:
                unsupported.append(op)
                box = _inf_box(t)
    else:
        unsupported.append(f"leaf:{type(t).__name__}")
        box = _inf_box(t)

    boxes[path] = box
    if site_boxes and path in site_boxes:
        r = site_boxes[path]
        box = Box(box.lo - r, box.hi + r)
        boxes[path] = box
    return box


# ---------------------------------------------------------------------------
#  Public: ibp_bound
# ---------------------------------------------------------------------------


def _collect_vars(
    t: Any, acc: list | None = None, seen: set | None = None
) -> list:
    if acc is None:
        acc, seen = [], set()
    if isinstance(t, Var):
        if t.name not in seen:
            seen.add(t.name)
            acc.append(t)
    elif isinstance(t, Op):
        for a in t.args:
            _collect_vars(a, acc, seen)
    return acc


def _norm_input_box(
    input_box: Any, term: Any, input_radius: float = 0.0
) -> dict:
    """Normalise the input box specification to ``{var_name: Box}``.

    Accepts: a dict ``{name: (lo,hi) | tensor}``, a single tensor
    (point box on the first Var), a ``(lo, hi)`` tensor pair, or a tuple
    of tensors mapped to the Vars in traversal order."""
    vars_ = _collect_vars(term)

    def mk(v):
        if isinstance(v, Box):
            return v
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lo, hi = (
                _as_float(torch.as_tensor(v[0])),
                _as_float(torch.as_tensor(v[1])),
            )
            return Box(lo, hi)
        t = _as_float(torch.as_tensor(v))
        return Box(t - input_radius, t + input_radius)

    if isinstance(input_box, dict):
        return {k: mk(v) for k, v in input_box.items()}
    if (
        isinstance(input_box, (tuple, list))
        and len(input_box) == 2
        and all(torch.is_tensor(v) for v in input_box)
        and vars_
        and torch.as_tensor(input_box[0]).shape
        == torch.as_tensor(input_box[1]).shape
        == vars_[0].typ.shape
    ):
        return {vars_[0].name: mk(input_box)}
    if isinstance(input_box, (tuple, list)):
        return {
            v.name: mk(t)
            for v, t in zip(vars_, input_box, strict=False)
        }
    # single tensor
    return {vars_[0].name: mk(input_box)} if vars_ else {}


def ibp_bound(
    term: Any,
    env: dict,
    input_box: Any,
    *,
    values: dict | None = None,
    site_boxes: dict | None = None,
    input_radius: float = 0.0,
) -> dict:
    """Propagate ``[lo, hi]`` boxes through ``term`` to the output.

    Parameters
    ----------
    term : IR term (Op/Var/Param/Const)
    env : dict
        ``source_tensors``-style parameter environment (Param name →
        tensor).  Eps-injected factor tensors live here too.
    input_box : tensor | (lo, hi) | dict
        Box on the input Var(s).  A bare tensor gives a point box
        (±``input_radius``); a dict maps Var names to boxes.
    values : dict, optional
        Concrete values for Var leaves — used to evaluate index
        operands (``embedding`` indices, ``where`` conditions).
    site_boxes : dict, optional
        ``{path: radius}`` — injected centre±radius boxes at certified
        ε sites (the mechanism the task describes: a site member is
        represented as its evaluated centre widened by its bound).

    Returns
    -------
    dict with ``lo``/``hi`` output bounds, max ``width``, the internal
    ``boxes`` map (path → Box), and the ``unsupported`` op list — ops
    with no sound interval rule produced ±∞ boxes rather than a guess.
    """
    boxes: dict = {}
    unsupported: list = []
    ib = _norm_input_box(input_box, term, input_radius)
    out = _ibp_eval(
        term, env, ib, values or {}, boxes, (), unsupported, site_boxes
    )
    width = out.hi - out.lo
    return {
        "lo": out.lo,
        "hi": out.hi,
        "width": float(width.abs().max())
        if width.numel()
        else float(width.abs()),
        "boxes": boxes,
        "unsupported": unsupported,
    }


# ---------------------------------------------------------------------------
#  Local (interval-derived) Lipschitz constants for the site→root walk
# ---------------------------------------------------------------------------

#: Ops that preserve any norm bound on the perturbation, whatever the
#: operand index (symmetric linear isometries on the perturbation).
_SPEC_PRESERVING_ANY_ARG = {"add", "sub", "concat", "cat", "stack"}

#: Ops preserving the bound only when the perturbation enters via
#: operand 0 (the data operand — index/mask/fill operands would not be
#: real-valued perturbations).
_SPEC_PRESERVING_ARG0 = {
    "neg",
    "float",
    "to",
    "alias",
    "type_as",
    "clone",
    "dropout",
    "broadcast",
    "masked_fill",
    "select",
    "slice",
    "getitem",
    "chunk",
    "split",
    "unbind",
    "index_select",
    "squeeze",
    "unsqueeze",
}


def _closest_to_zero(b: Box) -> torch.Tensor:
    """Element of [lo, hi] closest to 0 (elementwise)."""
    return torch.clamp(torch.zeros_like(b.lo), b.lo, b.hi)


def _grid_lip(b: Box, dfn, cands: tuple) -> float:
    """max |f'| over a box, sampling the endpoints plus interior
    candidate points where the derivative is known to peak."""
    best = 0.0
    pts = [b.lo, b.hi]
    for c in cands:
        pts.append(torch.full_like(b.lo, c))
    for p in pts:
        inside = (p >= b.lo) & (p <= b.hi)
        if inside.any():
            v = float(dfn(p)[inside].abs().max())
            best = max(best, v)
    return best


def _silu_der(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def _gelu_der(x: torch.Tensor) -> torch.Tensor:
    xd = x.double()
    phi = 0.5 * (1 + torch.erf(xd / math.sqrt(2)))
    pdf = torch.exp(-0.5 * xd * xd) / math.sqrt(2 * math.pi)
    return (phi + xd * pdf).to(x.dtype)


def _unary_lip(op: str, b: Box, attrs: dict) -> float | None:
    """Local Lipschitz constant of a unary elementwise op on box ``b``."""
    if op == "relu":
        return 0.0 if float(b.hi.max()) <= 0 else 1.0
    if op == "sigmoid":
        a = _closest_to_zero(b)
        s = torch.sigmoid(a)
        return float((s * (1 - s)).max())
    if op == "tanh":
        a = _closest_to_zero(b)
        t = torch.tanh(a)
        return float((1 - t * t).max())
    if op == "silu":
        return _grid_lip(b, _silu_der, (-4.0, -2.0, 0.0, 2.0, 4.0, 8.0))
    if op == "gelu":
        return _grid_lip(b, _gelu_der, (-3.0, -1.0, 0.0, 1.0, 3.0))
    if op == "exp":
        return float(b.hi.exp().max())
    if op == "neg":
        return 1.0
    if op == "sqrt":
        lo = float(b.lo.min())
        return 1.0 / (2 * math.sqrt(lo)) if lo > 0 else None
    if op == "rsqrt":
        lo = float(b.lo.min())
        return 0.5 * lo**-1.5 if lo > 0 else None
    if op == "square":
        return 2.0 * float(b.maxabs.max())
    if op == "pow":
        return None  # exponent handled via sibling box in _hop
    return None


def _softmax_lip(b: Box, attrs: dict) -> float:
    """‖J_softmax‖₂ ≤ max_i p_i (J = diag(p) − ppᵀ ≼ diag(p)); bounded
    by the interval softmax's upper corner."""
    dim = int(attr_of(attrs, "arg1", "dim", default=-1))
    return float(_softmax(b, dim).hi.max())


def _reduction_mult(op: str, in_box: Box, out_box: Box) -> float:
    n_in = in_box.lo.numel()
    n_out = max(1, out_box.lo.numel())
    n_r = max(1.0, n_in / n_out)
    if op == "sum":
        return math.sqrt(n_r)
    if op == "mean":
        return 1.0 / math.sqrt(n_r)
    return 1.0  # max/min: contraction


def _rearrange_mult(in_box: Box, out_box: Box) -> float:
    """Sound bound for arbitrary data rearrangement: an out-row's L2
    is within the whole input's Frobenius bound ≤ √rows_in · r.
    Exactly 1 when the last (row) dim is preserved."""
    d_in = in_box.lo.shape[-1] if in_box.lo.ndim >= 1 else 1
    d_out = out_box.lo.shape[-1] if out_box.lo.ndim >= 1 else 1
    if d_out == d_in:
        return 1.0
    return math.sqrt(in_box.lo.numel() / max(1, d_out))


def _hop(
    node: Op, i: int, argb: list, out_box: Box, kind: str
) -> tuple[float | None, str]:
    """Multiplier for a perturbation entering ``node`` via operand ``i``.

    ``kind`` is ``"row"`` (per-row L2 bound — the common currency) or
    ``"spec"`` (a spectral bound on a matrix perturbation, carried only
    while it may still meet a weight slot).  Returns (mult, out_kind);
    ``(None, …)`` = no local rule → caller falls back to spectral."""
    op = node.op

    if op in ("linear", "matmul"):
        if i == 0:
            s = _spec_bound(argb[1])
            return s, "row"
        if i == 1:
            # weight slot: out_row = a_row·ΔW — needs a *spectral* bound
            # on ΔW; a row bound converts at √m cost (m = ΔW rows).
            m = _maxrow_bound(argb[0])
            if kind == "row":
                w = argb[1].lo
                rows = w.shape[0] if w.ndim >= 2 else w.numel()
                m *= math.sqrt(rows)
            return m, "row"
        if i == 2 and op == "linear":  # bias: vector row bound
            return 1.0, "row"
        return None, kind

    if op == "conv2d":
        return None, kind

    if op == "embedding":
        # arg0 = table: out rows are selected table rows (row ≤ bound);
        # a spectral table bound also bounds each row.
        return (1.0, "row") if i == 0 else (None, kind)

    if op in _SPEC_PRESERVING_ANY_ARG:
        return 1.0, kind
    if op in _SPEC_PRESERVING_ARG0:
        return (1.0, kind) if i == 0 else (None, kind)

    if op in ("transpose",):
        # preserving spectral norm only if swapping the last two dims.
        # ``arg1``/``arg2`` are the canonical positional spelling for
        # transpose today; accept ``dim0``/``dim1`` as fallback like
        # other dual-spelling readers (torch_bridge, typing).
        nd = argb[0].lo.ndim
        a1 = int(attr_of(node, "arg1", "dim0", default=-2))
        a2 = int(attr_of(node, "arg2", "dim1", default=-1))
        a1, a2 = a1 % max(1, nd), a2 % max(1, nd)
        if {a1, a2} == {nd - 2, nd - 1}:
            return 1.0, kind
        return _rearrange_mult(argb[0], out_box), "row"

    if op in ("reshape", "view", "flatten", "expand"):
        return _rearrange_mult(argb[0], out_box), "row"

    if op == "mul":
        if len(argb) != 2:
            return None, kind
        sib = argb[1 - i]
        return float(sib.maxabs.max()), "row"

    if op == "div":
        if len(argb) != 2:
            return None, kind
        mn = _minabs(argb[1])
        if mn <= 0:
            return None, kind
        if i == 0:
            return 1.0 / mn, "row"
        return float(argb[0].maxabs.max()) / (mn * mn), "row"

    if op == "pow":
        if i != 0 or len(argb) != 2:
            return None, kind
        e = argb[1]
        if e is None or not torch.equal(e.lo, e.hi):
            return None, kind
        ev = float(e.lo.flatten()[0])
        a = argb[0]
        if ev == 0:
            return 0.0, "row"
        if ev == 1:
            return 1.0, "row"
        if ev > 1:
            if float(a.lo.min()) < 0 and ev != int(ev):
                return None, kind
            return abs(ev) * float(a.maxabs.max()) ** (ev - 1), "row"
        mn = _minabs(a)
        if mn <= 0:
            return None, kind
        return abs(ev) * mn ** (ev - 1), "row"

    if op == "softmax":
        return _softmax_lip(argb[0], node.attrs), "row"

    if op in ("sum", "mean", "max", "min"):
        return _reduction_mult(op, argb[0], out_box), "row"

    if op == "where":
        return (1.0, "row") if i > 0 else (None, kind)

    if op in (
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "gelu",
        "exp",
        "neg",
        "sqrt",
        "rsqrt",
        "square",
    ):
        return _unary_lip(op, argb[0], node.attrs), "row"

    # comparisons, sdpa, om/aff/trace carriers, logicals, unknown ops
    return None, kind


# ---------------------------------------------------------------------------
#  Site bookkeeping
# ---------------------------------------------------------------------------


def _subterm(term: Any, path: tuple) -> Any:
    t = term
    for i in path:
        t = t.args[i]
    return t


def _site_delta(site: dict, env: dict, values: dict):
    """Actual value difference lhs − rhs of the certified member —
    computable whenever both sides evaluate concretely (Var leaves
    resolved through ``values``).  This is a *tighter, still true* bound
    for this specific artifact: the certificate's error_bound is the
    a-priori radius; the realized difference is what actually flows."""
    lv = _eval_concrete(site["lhs"], env, values)
    rv = _eval_concrete(site["rhs"], env, values)
    if lv is None or rv is None:
        return None
    try:
        d = lv.detach().double() - rv.detach().double()
        return d if d.shape == lv.shape == rv.shape else None
    except Exception:
        return None


def _spectral_path_ok(site: dict, term: Any) -> bool:
    """Is model_bound's spectral contribution *sound* for this site?

    For Frobenius sites the declared bound is on the member tensor
    itself, and for spectral sites in weight slots / embedding tables /
    weight-program members the bound bounds every output row — both are
    propagated correctly.  But a spectral site at an *activation*
    position whose value is produced by the approximated matrix
    (``eps_lr``: ``linear(linear(x,V),U)``) has output error
    ``≤ σ_{r+1}·‖x‖`` — a factor model_bound's downstream-only walk
    misses.  For those, spectral can under-bound and must not be used
    as a cap or a fallback."""
    if (site.get("norm") or "frobenius") != "spectral":
        return True
    p = site["path"]
    if p:
        parent = _subterm(term, p[:-1])
        if (
            isinstance(parent, Op)
            and parent.op in ("linear", "matmul")
            and p[-1] == 1
        ):
            return True  # weight slot — sound
    s = _subterm(term, p) if p else term
    if isinstance(s, Op) and s.op == "linear":
        return False
    if isinstance(s, Op) and s.op == "matmul" and s.args:
        a0 = s.args[0]
        if isinstance(a0, Op) and a0.op == "embedding":
            return True  # per-row bound — sound
        # Param → bound on the member value; else activation matmul site
        return isinstance(a0, Param)
    return True


def _site_scalar(
    site: dict,
    term: Any,
    boxes: dict,
    env: dict,
    values: dict,
    actual: bool,
) -> tuple[float, str] | None:
    """The perturbation scalar injected at the site: (r0, kind).

    Weight-slot sites (site value feeds arg1 of linear/matmul) carry a
    spectral bound on ΔW; everything else carries a per-row L2 bound on
    the site value's perturbation.  ``actual`` selects the realized
    difference over the certificate radius."""
    p = site["path"]
    b = site["bound"]
    norm = site.get("norm") or "frobenius"
    delta = _site_delta(site, env, values) if actual else None

    # ---- weight slot? --------------------------------------------------
    if p:
        parent = _subterm(term, p[:-1])
        if (
            isinstance(parent, Op)
            and parent.op in ("linear", "matmul")
            and p[-1] == 1
        ):
            if delta is not None and delta.ndim >= 2:
                r = float(torch.linalg.matrix_norm(delta, 2).max())
                return (
                    (min(r, b), "spec")
                    if math.isfinite(r)
                    else (b, "spec")
                )
            if delta is not None and delta.ndim < 2:
                r = float(delta.abs().max())
                return (
                    (min(r, b), "spec")
                    if math.isfinite(r)
                    else (b, "spec")
                )
            # cert-radius fallback: both declared norms bound σ_max
            return b, "spec"

    # ---- activation (row-bound) site -----------------------------------
    if delta is not None:
        db = Box(delta, delta)
        r = _maxrow_bound(db)
        if math.isfinite(r):
            return r, "row"

    s = _subterm(term, p)
    if norm == "frobenius":
        return b, "row"  # worst case: one row holds it all
    # spectral: bound on σ_max — every row ≤ b; the only looser case is
    # a site whose *value* is an activation produced by the approximated
    # matrix (eps_lr: err_row ≤ σ_{r+1}·‖x_row‖).
    if isinstance(s, Op) and s.op == "linear":
        # eps_lr: site is linear(linear(x,V),U); output row err is
        # x_row·ΔWᵀ ≤ ‖x_row‖·b — x is the ORIGINAL site input, i.e.
        # the inner chain's operand at path p+(0,0).
        xpath = (*p, 0)
        inner = s.args[0]
        if isinstance(inner, Op) and inner.op == "linear":
            xpath = (*p, 0, 0)
        xb = boxes.get(xpath)
        if xb is not None:
            return b * _maxrow_bound(xb), "row"
        return None
    if isinstance(s, Op) and s.op == "matmul" and s.args:
        a0 = s.args[0]
        if isinstance(a0, Op) and a0.op == "embedding":
            return b, "row"  # eps_emb: per-row bound declared
        if isinstance(a0, Param):
            return b, "row"  # weight-program site: bound on value
        xb = boxes.get((*p, 0))
        if xb is not None:
            return b * _maxrow_bound(xb), "row"
        return None
    return b, "row"


def _walk_site(
    term: Any, path: tuple, r0: float, kind0: str, boxes: dict, R: dict
) -> float | None:
    """Multiply ``r0`` up the path site→root through local hop
    multipliers; accumulate per-node radii into ``R`` for the box
    widening pass.  ``None`` = a hop has no local rule (→ spectral)."""
    carried = r0
    kind = kind0
    R[path] = R.get(path, 0.0) + carried
    for d in range(len(path) - 1, -1, -1):
        node = _subterm(term, path[:d])
        if not isinstance(node, Op):
            return None
        argb = [
            boxes.get((*path[:d], j)) for j in range(len(node.args))
        ]
        outb = boxes.get(path[:d])
        if any(a is None for a in argb) or outb is None:
            return None
        mult, kind = _hop(node, path[d], argb, outb, kind)
        if mult is None or not math.isfinite(mult):
            return None
        carried *= mult
        R[path[:d]] = R.get(path[:d], 0.0) + carried
    return carried


# ---------------------------------------------------------------------------
#  Artifact propagation — the realized site difference as a tensor
# ---------------------------------------------------------------------------
#
# The certificate's error_bound is an a-priori *radius* (‖ΔW‖_F ≤ s/2·√n)
# covering every realization of the rounding.  The artifact that was
# actually extracted has a concrete difference ΔS = lhs − rhs, both
# sides evaluatable.  Propagating that tensor — exactly through linear
# and structural ops, elementwise-bounded through nonlinearities on the
# activation box — yields a bound that is still sound for this specific
# artifact and input, and is typically an order of magnitude tighter
# than the radius propagation.


def _elem_lip_map(op: str, b: Box) -> torch.Tensor | None:
    """Per-element Lipschitz bound tensor on box ``b`` — |f'(x)| ≤ map
    elementwise over the whole box (straddling elements take the
    worst-case slope in their interval)."""
    if op == "relu":
        return (b.hi > 0).to(b.lo.dtype)
    if op == "sigmoid":
        a = _closest_to_zero(b)
        s = torch.sigmoid(a)
        return s * (1 - s)
    if op == "tanh":
        a = _closest_to_zero(b)
        t = torch.tanh(a)
        return 1 - t * t
    if op == "silu":
        pts = [b.lo, b.hi] + [
            torch.full_like(b.lo, c)
            for c in (-4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
        ]
        m = _silu_der(b.lo).abs() * 0
        for p in pts:
            inside = (p >= b.lo) & (p <= b.hi)
            m = torch.where(
                inside, torch.maximum(m, _silu_der(p).abs()), m
            )
        return m
    if op == "gelu":
        pts = [b.lo, b.hi] + [
            torch.full_like(b.lo, c)
            for c in (-3.0, -1.0, 0.0, 1.0, 3.0)
        ]
        m = torch.zeros_like(b.lo)
        for p in pts:
            inside = (p >= b.lo) & (p <= b.hi)
            m = torch.where(
                inside, torch.maximum(m, _gelu_der(p).abs()), m
            )
        return m
    if op == "exp":
        return b.hi.exp()
    if op == "square":
        return 2 * b.maxabs
    if op == "sqrt":
        if float(b.lo.min()) <= 0:
            return None
        return 0.5 * b.lo**-0.5
    if op == "rsqrt":
        if float(b.lo.min()) <= 0:
            return None
        return 0.5 * b.lo**-1.5
    if op == "neg":
        return torch.ones_like(b.lo)
    return None


def _prop_delta(
    node: Op,
    i: int,
    cur: torch.Tensor,
    mag: bool,
    argb: list,
    outb: Box,
    env: dict,
    values: dict,
):
    """Propagate a perturbation TENSOR through one hop.

    ``cur`` is either the signed realized delta (``mag=False``) or a
    nonnegative elementwise magnitude bound (``mag=True`` — sign is
    lost at nonlinear hops, after which multiplicative ops must use
    |sibling|, never the signed value).  Returns ``(tensor, mag)`` or
    ``None`` = no tensor rule → caller degrades to the scalar walk."""
    from catopt_torch.torch_bridge import _IR_TO_TORCH

    op = node.op
    cur = cur.double()

    def sib(j):
        s = _eval_concrete(node.args[j], env, values)
        if s is not None and mag:
            s = s.double().abs()
        elif s is not None:
            s = s.double()
        return s

    if op == "linear":
        if i == 0:
            w = sib(1)
            return None if w is None else (cur @ w.T, mag)
        if i == 1:
            a = sib(0)
            return None if a is None else (a @ cur.T, mag)
        if i == 2:  # bias: broadcast the vector over every row
            shape = outb.lo.shape if outb is not None else cur.shape
            try:
                return torch.broadcast_to(cur, shape), mag
            except Exception:
                return cur, mag
        return None
    if op == "matmul":
        if i == 0:
            b = sib(1)
            return None if b is None else (cur @ b, mag)
        if i == 1:
            a = sib(0)
            return None if a is None else (a @ cur, mag)
        return None
    if op in ("add", "sub"):
        if mag:
            return cur, True  # magnitude bound passes both
        return (cur, False) if op == "add" or i == 0 else (-cur, False)
    if op in (
        "neg",
        "float",
        "to",
        "alias",
        "type_as",
        "clone",
        "dropout",
        "contiguous",
        "broadcast",
    ):
        if mag:
            return cur, True
        fn = _IR_TO_TORCH.get(op)
        if fn is None or op == "neg":
            return -cur, False
        try:
            return fn(cur, **dict(node.attrs)), False
        except Exception:
            return cur.double(), False
    if op in (
        "reshape",
        "view",
        "transpose",
        "select",
        "slice",
        "getitem",
        "chunk",
        "split",
        "unbind",
        "squeeze",
        "unsqueeze",
        "flatten",
        "expand",
    ):
        fn = _IR_TO_TORCH.get(op)
        if fn is None or i != 0:
            return None
        try:
            return fn(cur, **dict(node.attrs)), mag
        except Exception:
            return None
    if op in ("concat", "cat"):
        dim = int(attr_of(node, "dim", "arg1", default=0))
        parts = []
        for j, b in enumerate(argb):
            parts.append(
                cur if j == i else torch.zeros_like(b.lo.double())
            )
        return torch.cat(parts, dim=dim), mag
    if op == "stack":
        dim = int(attr_of(node, "dim", "arg1", default=0))
        parts = [
            cur if j == i else torch.zeros_like(b.lo.double())
            for j, b in enumerate(argb)
        ]
        return torch.stack(parts, dim=dim), mag
    if op == "index_select" and i == 0:
        fn = _IR_TO_TORCH.get("index_select")
        if fn is None:
            return None
        try:
            idx = (
                _eval_concrete(node.args[1], env, values)
                if len(node.args) > 1
                else None
            )
            return (
                (fn(cur, idx, **dict(node.attrs)), mag)
                if idx is not None
                else (fn(cur, **dict(node.attrs)), mag)
            )
        except Exception:
            return None
    if op == "embedding" and i == 0:
        idx = _eval_concrete(node.args[1], env, values)
        if idx is None:
            return None
        return torch.nn.functional.embedding(idx.long(), cur), mag
    if op == "mul":
        s = sib(1 - i)
        return None if s is None else (cur * s, mag)
    if op == "div":
        s = sib(1)
        if s is None or float(s.abs().min()) <= 0:
            return None
        if i == 0:
            return cur / s, mag
        a = sib(0)
        if a is None:
            return None
        # |δ_out| = |a·δ_b/b²| — always a magnitude bound
        return (a / (s * s)).abs() * cur.abs(), True
    if op in ("sum", "mean"):
        fn = _IR_TO_TORCH.get(op)
        try:
            return fn(cur, **dict(node.attrs)), mag
        except Exception:
            return None
    if op in (
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "gelu",
        "exp",
        "square",
        "sqrt",
        "rsqrt",
    ):
        lm = _elem_lip_map(op, argb[0])
        return None if lm is None else (cur.abs() * lm.double(), True)
    if op == "where" and i > 0:
        c = _eval_concrete(node.args[0], env, values)
        if c is None:
            return None
        z = torch.zeros_like(cur)
        return torch.where(
            c.bool(), cur if i == 1 else z, z if i == 1 else cur
        ), mag
    if op == "masked_fill" and i == 0:
        m = _eval_concrete(node.args[1], env, values)
        return (
            None if m is None else (cur.masked_fill(m.bool(), 0.0), mag)
        )
    return None


def _walk_site_artifact(
    term: Any,
    path: tuple,
    delta,
    kind0: str,
    boxes: dict,
    env: dict,
    values: dict,
):
    """Artifact walk: propagate the realized delta tensor hop-by-hop;
    degrade to the scalar row-bound walk at the first op with no
    tensor rule.  Returns a max-abs bound at the root, or None."""
    if delta is None:
        return None
    if not torch.is_tensor(delta):
        # scalar (r0, kind) injection — plain scalar walk
        return _walk_site(term, path, float(delta), kind0, boxes, {})
    cur = delta
    mag = False
    for d in range(len(path) - 1, -1, -1):
        node = _subterm(term, path[:d])
        if not isinstance(node, Op):
            return None
        argb = [
            boxes.get((*path[:d], j)) for j in range(len(node.args))
        ]
        outb = boxes.get(path[:d])
        res = _prop_delta(
            node, path[d], cur, mag, argb, outb, env, values
        )
        if res is None:
            # degrade: scalar row-bound walk on the remaining hops
            r = _maxrow_bound(Box(cur.abs(), cur.abs()))
            # continue the scalar walk from this depth upward
            carried, kind = r, "row"
            for dd in range(d, -1, -1):
                nd = _subterm(term, path[:dd])
                ab = [
                    boxes.get((*path[:dd], j))
                    for j in range(len(nd.args))
                ]
                ob = boxes.get(path[:dd])
                mult, kind = _hop(nd, path[dd], ab, ob, kind)
                if mult is None or not math.isfinite(mult):
                    return None
                carried *= mult
            return carried
        cur, mag = res
    # at the root the realized delta is concrete: max-abs is the
    # tightest sound reduction (row-L2 would overestimate by ≤ √d)
    return float(cur.abs().max())


def _widen(boxes: dict, R: dict) -> dict:
    """Widen every node box elementwise by its accumulated perturbation
    radius (|δ_ij| ≤ ‖δ_row‖ — sound)."""
    out = {}
    for p, b in boxes.items():
        r = R.get(p, 0.0)
        out[p] = Box(b.lo - r, b.hi + r) if r else b
    return out


def _collect_sites(term: Any, cert: Any) -> tuple[list, float]:
    """Locate every bound-carrying cert step in the extracted term.
    Returns (sites, unlocated_mass) mirroring model_bound's policy."""
    sites: list = []
    unlocated = 0.0
    for step in cert.steps:
        rule = cert.rules.get(step.rule)
        if rule is None or not rule.error_bound:
            continue
        paths = _find_subterms(term, step.rhs)
        if not paths:
            unlocated += rule.error_bound
            sites.append(
                {
                    "rule": step.rule,
                    "bound": rule.error_bound,
                    "norm": getattr(rule, "bound_norm", None),
                    "path": None,
                    "lhs": step.lhs,
                    "rhs": step.rhs,
                }
            )
            continue
        for p in paths:
            sites.append(
                {
                    "rule": step.rule,
                    "bound": rule.error_bound,
                    "norm": getattr(rule, "bound_norm", None),
                    "path": p,
                    "lhs": step.lhs,
                    "rhs": step.rhs,
                }
            )
    return sites, unlocated


def _vars_and_inputs(term: Any, example_input: Any):
    vars_ = _collect_vars(term)
    inputs = (
        list(example_input)
        if isinstance(example_input, (tuple, list))
        else [example_input]
    )
    values = {
        v.name: t for v, t in zip(vars_, inputs, strict=False)
    }
    return vars_, inputs, values


def _input_norm(inputs: list) -> float:
    best = 0.0
    for t in inputs:
        tt = torch.as_tensor(t)
        if tt.ndim >= 1 and tt.is_floating_point():
            rows = torch.linalg.norm(
                tt.double().reshape(-1, tt.shape[-1]), dim=-1
            )
            best = max(best, float(rows.max()))
        elif tt.ndim == 0:
            best = max(best, abs(float(tt)))
    return best or 1.0


def _measured_error(
    term: Any, cert: Any, src: dict, inputs: list, vars_: list
) -> float | None:
    """Run the extracted term and the certificate's source term through
    real torch modules and diff — the empirical error the bound covers."""
    try:
        from catopt_torch.torch_bridge import ir_to_torch_module

        mods = []
        for root in (cert.src, term):
            src_vars = _collect_vars(root)
            ir = IR(
                root=root,
                inputs=src_vars,
                input_names={v.name for v in src_vars},
                params={},
            )
            mods.append(ir_to_torch_module(ir, param_values=src))
        with torch.no_grad():
            o0 = mods[0](*[torch.as_tensor(t) for t in inputs])
            o1 = mods[1](*[torch.as_tensor(t) for t in inputs])
        return float((o0.double() - o1.double()).abs().max())
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Public: tight_model_bound
# ---------------------------------------------------------------------------


def tight_model_bound(
    term: Any,
    cert: Any,
    src: dict,
    example_input: Any,
    *,
    input_radius: float = 0.0,
    max_iter: int = 4,
) -> dict:
    """Whole-model error bound via interval propagation.

    For each bound-carrying certificate step the site's contribution is
    ``site perturbation × Π local Lipschitz constants`` along its path
    to the output — local meaning *evaluated on the actual activation
    boxes* from :func:`ibp_bound`, then widened by the accumulated
    upstream perturbations and iterated to a fixed point.

    Two totals are reported:

    * ``bound`` — cert-radius bound: each site contributes its declared
      ``error_bound`` propagated through local constants.  Never worse
      than ``spectral_bound`` (per-site fallback to
      :func:`eps._path_sensitivity` when no local rule exists).
    * ``artifact_bound`` — the same propagation applied to the
      *realized* site difference (lhs − rhs evaluated concretely).  A
      tighter, still sound bound for this specific artifact — the cert
      radius is a worst-case envelope over all quantized tensors; the
      artifact bound is for the one we actually built.

    ``measured_error`` diffs the extracted module against the source
    module on ``example_input`` when both lower successfully.
    """
    vars_, inputs, values = _vars_and_inputs(term, example_input)
    input_box = (
        _norm_input_box(
            {
                v.name: t
                for v, t in zip(vars_, inputs, strict=False)
            }
            if vars_
            else {},
            term,
            input_radius,
        )
        if vars_
        else {}
    )
    for v, t in zip(vars_, inputs, strict=False):
        input_box.setdefault(
            v.name,
            Box(
                _as_float(torch.as_tensor(t)) - input_radius,
                _as_float(torch.as_tensor(t)) + input_radius,
            ),
        )

    ib = ibp_bound(term, src, input_box, values=values)
    base_boxes = ib["boxes"]

    input_norm = _input_norm(inputs)
    spectral = model_bound(term, cert, src, input_norm)
    spec_contrib = {}
    for c in spectral["site_contributions"]:
        key = (
            c.get("rule"),
            tuple(c["path"]) if c.get("path") else None,
        )
        spec_contrib[key] = c.get("contribution")

    sites, unlocated = _collect_sites(term, cert)

    # ---- iterate: widen boxes by accumulated upstream radii -----------
    R: dict = {}
    total = total_a = 0.0
    contributions: list = []
    n_fallback = 0
    converged = False
    for _ in range(max_iter + 1):
        boxes = _widen(base_boxes, R)
        new_R: dict = {}
        total = unlocated
        total_a = unlocated
        contributions = []
        n_fallback = 0
        for s in sites:
            if s["path"] is None:
                contributions.append(
                    {
                        "rule": s["rule"],
                        "bound": s["bound"],
                        "unlocated": True,
                        "contribution": s["bound"],
                    }
                )
                continue
            c0 = _site_scalar(s, term, boxes, src, values, actual=False)
            if c0 is None:  # pragma: no cover — ibp_bound boxes every subterm; unreachable end-to-end
                c0 = (s["bound"], "row")
            cc = _walk_site(term, s["path"], c0[0], c0[1], boxes, new_R)
            # artifact variant: propagate the realized lhs−rhs delta
            # tensor (exact through linear hops, elementwise-bounded at
            # nonlinearities); fall back to the scalar walk.
            delta = _site_delta(s, src, values)
            ca = (
                _walk_site_artifact(
                    term, s["path"], delta, "row", boxes, src, values
                )
                if delta is not None
                else None
            )
            if ca is None:
                a0 = (
                    _site_scalar(
                        s, term, boxes, src, values, actual=True
                    )
                    or c0
                )
                ca = _walk_site(
                    term, s["path"], a0[0], a0[1], boxes, {}
                )
            fb = False
            spec_ok = _spectral_path_ok(s, term)
            sc = (
                spec_contrib.get((s["rule"], s["path"]))
                if spec_ok
                else None
            )
            if cc is None:
                fb = True
                n_fallback += 1
                cc = sc
                if cc is None:
                    # spectral fallback unusable for this site (or not
                    # recorded): the site's own scalar × the global
                    # Lipschitz product is still sound.
                    sens = _path_sensitivity(
                        term, s["path"], src, input_norm
                    )
                    cc = (
                        c0[0] * sens
                        if sens is not None
                        else float("inf")
                    )
                ca = cc
            else:
                # per-site min: widening margins can exceed spectral's
                # global estimate — never report worse than spectral
                # (only where spectral is itself sound for the site).
                if sc is not None and sc < cc:
                    fb = True
                    cc = sc
                if ca is None or ca > cc:
                    ca = cc
            total += cc
            total_a += ca
            contributions.append(
                {
                    "rule": s["rule"],
                    "bound": s["bound"],
                    "path": s["path"],
                    "contribution": cc,
                    "artifact_contribution": ca,
                    "fallback": fb,
                    "spectral_unsafe": not spec_ok,
                }
            )
        if all(
            new_R.get(k, 0.0) <= R.get(k, 0.0) + 1e-12 for k in new_R
        ):
            converged = True
            break
        for k, v in new_R.items():
            R[k] = max(R.get(k, 0.0), v)

    err = _measured_error(term, cert, src, inputs, vars_)

    spectral_bound = spectral["bound"]
    tighter = total < spectral_bound
    n_unsafe = sum(1 for c in contributions if c.get("spectral_unsafe"))
    notes = []
    if n_unsafe:
        notes.append(
            f"{n_unsafe} spectral site(s) sit at activation positions "
            "where model_bound's downstream-only walk misses the "
            "input-norm factor — the spectral figure may UNDER-bound "
            "there; IBP used the input-scaled local estimate instead"
        )
    if not tighter:
        notes.append(
            "IBP did not improve on the spectral path bound for this "
            "term (local constants were not smaller than global ones)"
        )
    if n_fallback:
        notes.append(
            f"{n_fallback} site(s) used the spectral path contribution "
            "(no interval rule on their path, or spectral was tighter "
            "after widening margins)"
        )
    if not converged:
        notes.append(
            f"widening iterate did not fully converge in {max_iter} "
            "rounds; reported bound is the last (largest) iterate"
        )
    if ib["unsupported"]:
        notes.append(
            "ops with no sound interval rule produced ±∞ boxes: "
            + ", ".join(sorted(set(ib["unsupported"])))
        )

    return {
        "bound": total,
        "artifact_bound": total_a,
        "spectral_bound": spectral_bound,
        "measured_error": err,
        "tighter": tighter,
        "improvement": (
            spectral_bound / total if total > 0 else float("inf")
        ),
        "cert_conservatism": (
            total / err if err and err > 0 else float("inf")
        ),
        "artifact_conservatism": (
            total_a / err if err and err > 0 else float("inf")
        ),
        "site_contributions": contributions,
        "n_bounded_steps": spectral["n_bounded_steps"],
        "n_fallback": n_fallback,
        "converged": converged,
        "unsupported_ops": sorted(set(ib["unsupported"])),
        "note": "; ".join(notes) if notes else "ok",
    }
