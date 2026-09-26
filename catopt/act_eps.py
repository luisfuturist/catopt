"""Activation-space ε: certified bounded-error rewrites on *activations*.

Optional toolkit, not part of the core optimizer — nothing here runs
unless this module is called directly (``optimize_model`` never
invokes it).

The ε machinery in :mod:`catopt.eps` certifies *weight* substitutions —
the tensor is known at compile time, so the bound is a constant.  This
module is the activation-side counterpart, the place where a norm bound
is genuinely the contract (KV-cache width, intermediate-activation
precision, certified deployment): the tensor is NOT known at compile
time, so the offer must be **dynamic**.

``act_quant`` rewrites every consumer edge of an activation value as

    f(…, x, …)  →  f(…, adequant(aquant(x)), …)

``aquant`` is a runtime op: per forward call it computes the per-tensor
symmetric scale ``s(x) = |x|max / (2^{bits-1} − 1)`` and returns the
``(q:int8, s)`` pair; ``adequant`` decodes ``q·s``.  The pair is
identity-preserving at runtime up to a *runtime-computable* bound —

    ‖x − dequant(quant(x))‖_F ≤ s(x)/2 · √n          (per call)

— and the witnessed rewrite carries the *calibrated* bound: the same
formula evaluated at the site's worst-case absmax over a calibration
batch (see :func:`calibrate`).  The certificate therefore certifies
"error ≤ bound whenever the runtime absmax stays within calibration" —
the honest contract for activation quantization, since no static bound
on ``s(x)`` exists without data.

The wrap is placed at *consumer* edges rather than inside the
activation's own e-class: merging ``adequant(aquant(c))`` into class
``c`` creates a cyclic member (``c`` reachable from itself) that
extraction must skip.  Rewriting ``f(x) → f(dequant(quant(x)))`` keeps
the member acyclic and models the real deployment — each consumer reads
a quantized copy of the activation, which is exactly what an int8 KV
cache or activation buffer does.

Both ops are registered through the existing extension points —
``ir.op_def`` for the generator and ``torch_bridge._IR_TO_TORCH`` for
the lowering — so this module is self-contained: no edits to ir.py,
torch_bridge.py, or eps.py.  (``eps._LIP_FREE`` additionally gains the
two op names so ``model_bound``'s Lipschitz walk treats the ≈identity
decode correctly.)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from catopt.egraph import EGraph, ENode, Rewrite
from catopt.ir import Const, Op, Param, TensorType, Var, op_def, op_repr

__all__ = [
    "ACT_EPS_OPS",
    "act_low_rank",
    "act_quant",
    "calibrate",
    "extract_with_offers",
]


# ---------------------------------------------------------------------------
#  IR ops + torch bindings — registered via the existing extension points
#  (``op_def`` writes the generator registry; ``_IR_TO_TORCH`` is the lowering
#  table ``IRModule._eval`` dispatches through).  No ir.py/torch_bridge.py
#  edits are needed.
# ---------------------------------------------------------------------------

ACT_EPS_OPS = ("aquant", "adequant")

op_def(
    "aquant",
    1,
    1,
    law="Per-call symmetric activation quantization: evaluates to the "
    "(q:int8, s) pair with s = |x|max/(2^{bits-1}-1).  Runtime "
    "contract: ‖x − dequant(quant(x))‖_F ≤ s(x)/2·√n per call.",
)
op_def(
    "adequant",
    1,
    1,
    law="Decode half of the activation quant pair: (q, s) ↦ q·s — the "
    "identity map up to the certified rounding bound.",
)


def _aquant_torch(x, *a, **kw):
    """Runtime per-call symmetric quantization → (q:int8, s).

    ``s`` stays in ``x``'s dtype so ``adequant`` restores it.  A zero
    tensor quantizes to itself (``s`` clamped to 1 — q is all zeros and
    q·s == x exactly)."""
    bits = int(kw.get("bits", 8))
    levels = 2 ** (bits - 1) - 1
    amax = x.detach().abs().max()
    s = amax / levels
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round(x / s).clamp(-levels - 1, levels).to(torch.int8)
    return (q, s)


def _adequant_torch(pair, *a, **kw):
    """(q, s) ↦ q·s — decode in the scale's (i.e. the input's) dtype."""
    q, s = pair
    return q.to(s.dtype) * s


def _register_extensions() -> None:
    from catopt.torch_bridge import _IR_TO_TORCH

    _IR_TO_TORCH.setdefault("aquant", _aquant_torch)
    _IR_TO_TORCH.setdefault("adequant", _adequant_torch)
    # The decode is ≈identity: let eps.model_bound's Lipschitz walk pass
    # through the pair (the substitution error is already carried by the
    # rewrite's error_bound).  Registry extension, not an eps.py edit.
    try:
        from catopt import eps as _eps

        _eps._LIP_FREE.update(("aquant", "adequant"))
    except Exception:
        pass


_register_extensions()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _has_var(t, _memo: dict | None = None) -> bool:
    """True when the term mentions a Var leaf (data-dependent subtree —
    the activation/weight distinction: param-only subtrees fold at
    compile time and belong to ``eps.quant_params``, not here)."""
    if _memo is None:
        _memo = {}
    key = id(t)
    if key in _memo:
        return _memo[key]
    if isinstance(t, Var):
        out = True
    elif isinstance(t, Op):
        out = any(_has_var(a, _memo) for a in t.args)
    else:
        out = False
    _memo[key] = out
    return out


def _site_absmax(calib, key: str):
    """Look up a site's calibrated absmax.

    ``calib`` may be (a) the dict returned by :func:`calibrate`
    (``{"per_site": {repr: amax}, "global": amax}``), (b) a plain
    ``{repr: amax}`` dict, or (c) a scalar applied to every site.
    ``None`` → uncalibrated."""
    if calib is None:
        return None
    if isinstance(calib, (int, float)):
        return float(calib)
    if isinstance(calib, dict):
        per = calib.get("per_site")
        if isinstance(per, dict):
            v = per.get(key)
            if v is not None:
                return float(v)
        elif key in calib:
            return float(calib[key])
        g = calib.get("global")
        return float(g) if g is not None else None
    return None


def _activation_classes(eg: EGraph) -> dict[int, Any]:
    """E-class id -> representative term for every *activation-producing*
    class: a class whose oldest member is a non-leaf Op that depends on
    an input Var.  Excludes Param leaves, input Vars, param-only
    computations, and our own quant members."""
    out: dict[int, object] = {}
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        if c in out:
            continue
        rep = eg._oldest_term(c) or eg.any_term(c)
        if (
            isinstance(rep, Op)
            and rep.op not in ACT_EPS_OPS
            and _has_var(rep)
        ):
            out[c] = rep
    return out


def _wrap_sites(
    eg: EGraph,
    act: dict[int, Any],
    wrap_factory,
    bound_of,
    tag: str,
    law_of,
    witness: bool,
) -> list[dict]:
    """For every consumer enode ``f(…, c, …)`` whose child ``c`` is an
    activation class, offer the member ``f(…, wrap(c), …)`` into the
    consumer's own e-class, witnessed by a bound-carrying Rewrite.

    ``wrap_factory(c)`` returns (creating lazily, memoized here) the
    wrapper e-class for activation class ``c`` — one shared wrapped-read
    per activation, every consumer quantizing the same read.
    ``bound_of``/``law_of`` give the certified bound and law text per
    wrapped class."""
    offers: list[dict] = []
    seen: set = set()
    wrap_cache: dict[int, int] = {}
    for cid in list(eg._classes.keys()):
        cc = eg.find(cid)
        ec = eg._classes.get(cc)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op == "leaf" or node.op in ACT_EPS_OPS:
                continue
            for i, ch in enumerate(node.children):
                ch_c = eg.find(ch)
                if ch_c not in act:
                    continue
                if ch_c not in wrap_cache:
                    wrap_cache[ch_c] = wrap_factory(ch_c)
                if wrap_cache[ch_c] is None:
                    continue
                dq = eg.find(wrap_cache[ch_c])
                children = tuple(
                    dq if j == i else eg.find(c2)
                    for j, c2 in enumerate(node.children)
                )
                meid = eg.add_enode(node.op, children, dict(node.attrs))
                member_en = ENode(node.op, children, node.attrs)
                key = (cc, member_en)
                if key in seen:
                    continue
                seen.add(key)
                bound, norm = bound_of(ch_c)
                src_term = eg._oldest_term(cc) or eg.any_term(cc)
                # Deterministic RHS: the member with each child resolved
                # to its class's OLDEST term — the derivation then
                # bridges original members, not arbitrary (possibly
                # bound-carrying) ones, keeping certificates tight.
                offer_term = eg._oldest_term(meid) or eg.any_term(meid)
                wit = None
                if (
                    witness
                    and src_term is not None
                    and offer_term is not None
                ):
                    wit = Rewrite(
                        name=f"{tag}#{meid}",
                        lhs=src_term,
                        rhs=offer_term,
                        law=law_of(ch_c, i, node, bound),
                        error_bound=bound,
                        bound_norm=norm,
                    )
                merged = eg.union(
                    cc,
                    meid,
                    witness=wit,
                    note=(
                        f"{tag}: wrap input {i} of {node.op} "
                        f"(activation class {ch_c}), ε={bound:.3e}"
                    ),
                )
                offers.append(
                    {
                        "site_eid": cc,
                        "quant_eid": ch_c,
                        "pos": i,
                        "op": node.op,
                        "member": member_en,
                        "member_eid": meid,
                        "bound": bound,
                        "bound_norm": norm,
                        "merged": merged,
                        "cyclic": cc == ch_c,
                    }
                )
    return offers


# ---------------------------------------------------------------------------
#  Calibration
# ---------------------------------------------------------------------------


def calibrate(
    model, inputs, source_tensors=None, *, keep_tensors: bool = False
) -> dict:
    """Run activations, collect absmax statistics per site.

    Two modes:

    * ``calibrate(ir, xs, source_tensors)`` — evaluate the IR term
      bottom-up with the torch lowerings and record ``|out|max`` of
      every op subterm, keyed by ``op_repr(subterm)``.  This is the key
      ``act_quant`` uses to bind an e-graph site (the class's oldest
      member) to its calibrated bound.
    * ``calibrate(nn_module, xs)`` — forward hooks on every submodule;
      absmax keyed by module name plus a ``global`` ceiling (usable as
      ``act_quant``'s fallback bound).

    ``inputs`` is a tensor, a tuple/list of tensors (positional, matched
    to ``ir.inputs``), or a ``{var_name: tensor}`` dict.

    Returns ``{"per_site": {key: absmax}, "global": max, "n_sites": n}``
    (plus ``"tensors": {key: tensor}`` when ``keep_tensors`` — the
    activation samples ``act_low_rank`` measures its residual on).
    """
    xs = inputs
    if isinstance(xs, torch.Tensor):
        xs = (xs,)
    elif isinstance(xs, dict):
        xs = dict(xs)

    per_site: dict[str, float] = {}
    tensors: dict[str, torch.Tensor] = {}

    if isinstance(model, torch.nn.Module):
        args = tuple(xs.values()) if isinstance(xs, dict) else tuple(xs)
        handles = []

        def hook(name):
            def h(_mod, _inp, out):
                if torch.is_tensor(out):
                    v = float(out.detach().abs().max())
                    per_site[name] = max(per_site.get(name, 0.0), v)
                    if keep_tensors:
                        tensors[name] = out.detach().clone()

            return h

        for name, mod in model.named_modules():
            handles.append(
                mod.register_forward_hook(hook(name or "model"))
            )
        try:
            with torch.no_grad():
                model(*args)
        finally:
            for h in handles:
                h.remove()
    else:
        # IR term evaluation — sites keyed by op_repr, matching
        # act_quant's e-class representative keys.
        from catopt.torch_bridge import _IR_TO_TORCH

        ir = model
        src = source_tensors or {}
        if isinstance(xs, dict):
            env = dict(xs)
        else:
            env = {v.name: t for v, t in zip(ir.inputs, xs)}
        memo: dict[int, object] = {}

        def ev(t):
            key = id(t)
            if key in memo:
                return memo[key]
            r = None
            if isinstance(t, Var):
                r = env.get(t.name)
            elif isinstance(t, Const):
                r = torch.tensor(t.value)
            elif isinstance(t, Param):
                r = src.get(t.name)
            elif isinstance(t, Op):
                fn = _IR_TO_TORCH.get(t.op)
                args = [ev(a) for a in t.args]
                if fn is not None and all(a is not None for a in args):
                    try:
                        r = fn(*args, **dict(t.attrs))
                    except Exception:
                        r = None
                if torch.is_tensor(r):
                    k = op_repr(t)
                    v = float(r.detach().abs().max())
                    per_site[k] = max(per_site.get(k, 0.0), v)
                    if keep_tensors:
                        tensors[k] = r.detach().clone()
            memo[key] = r
            return r

        with torch.no_grad():
            ev(ir.root)

    out = {
        "per_site": per_site,
        "global": max(per_site.values(), default=0.0),
        "n_sites": len(per_site),
    }
    if keep_tensors:
        out["tensors"] = tensors
    return out


# ---------------------------------------------------------------------------
#  act_quant — dynamic quantize/dequantize offers on activation edges
# ---------------------------------------------------------------------------


def act_quant(
    eg: EGraph,
    source_tensors: dict,
    *,
    bits: int = 8,
    calib=None,
    witness: bool = True,
) -> list[dict]:
    """Offer ``f(…, x, …) → f(…, adequant(aquant(x)), …)`` at every
    consumer edge whose child is an activation-producing e-class.

    ``aquant`` computes the per-call symmetric scale
    ``s(x) = |x|max / (2^{bits-1} − 1)`` at runtime and returns the
    ``(q:int8, s)`` pair — the *stored* object in a KV-cache-style
    contract.  The certified per-site bound is

        ‖x − q·s‖_F ≤ s(x)/2 · √n  ≤  bound

    where ``bound = (|x|max_cal / (2·levels)) · √n`` uses the worst-case
    absmax over the calibration batch (``calib``, from
    :func:`calibrate`).  The bound formula — per-call and
    runtime-computable — is recorded in the witness law text; the
    numeric ``error_bound`` is the calibration-supported worst case.
    Sites outside calibration (or with unknown element count) are still
    offered, carrying ``error_bound = inf`` — honestly uncertified
    rather than a fabricated constant.

    ``source_tensors`` is accepted for signature parity with the
    weight-space ε passes; activation quantization injects no params.

    Returns one dict per offered member: ``site_eid`` (the consumer
    class the member joined), ``quant_eid`` (the wrapped activation
    class), ``pos``, ``member``/``member_eid`` (for ``overrides``
    extraction), ``bound``, ``cyclic``.
    """
    levels = 2 ** (bits - 1) - 1
    act = _activation_classes(eg)
    from catopt.typing import _shape_of

    # Per-site data + shared wrap class are built lazily by the factory —
    # an activation nobody consumes (e.g. the model output) gets no
    # orphan quant classes.
    info: dict[int, tuple] = {}

    def wrap_factory(c):
        rep = act[c]
        aq = eg.add_enode("aquant", (c,), {"bits": bits})
        dq = eg.add_enode("adequant", (aq,), {})
        amax = _site_absmax(calib, op_repr(rep))
        shape = _shape_of(rep)
        n = (
            math.prod(shape)
            if isinstance(shape, tuple)
            and all(isinstance(d, int) for d in shape)
            else None
        )
        if amax is None:
            bound, norm = float("inf"), "frobenius"
        elif n is not None:
            bound = amax / (2.0 * levels) * math.sqrt(n)
            norm = "frobenius"
        else:
            # Element-count unknown: the bound that survives is the
            # elementwise one — |Δx_i| ≤ s/2, an L∞ certificate.
            bound, norm = amax / (2.0 * levels), "linf"
        info[c] = (bound, norm, amax, n)
        return dq

    def bound_of(c):
        return info[c][0], info[c][1]

    def law_of(c, i, node, bound):
        amax, n = info[c][2], info[c][3]
        formula = (
            f"‖Δx‖_F ≤ s(x)/2·√n per call, "
            f"s(x) = |x|max/{levels} (runtime-computable)"
        )
        if amax is None:
            return (
                f"dynamic int{bits} activation quant at input {i} "
                f"of {node.op}: {formula}; UNCALIBRATED — no static "
                f"bound without calibration data"
            )
        scale = amax / levels
        nn_ = f"n={n}" if n is not None else "n=? (L∞ bound)"
        return (
            f"dynamic int{bits} activation quant at input {i} of "
            f"{node.op}: {formula}; calibrated |x|max = {amax:.3e} "
            f"→ s ≤ {scale:.3e}, ε ≤ {bound:.3e} ({nn_})"
        )

    return _wrap_sites(
        eg, act, wrap_factory, bound_of, f"act_q{bits}", law_of, witness
    )


# ---------------------------------------------------------------------------
#  act_low_rank — random-projection bottleneck on activation edges (stretch)
# ---------------------------------------------------------------------------


def act_low_rank(
    eg: EGraph,
    source_tensors: dict,
    *,
    rank: int,
    calib=None,
    seed: int = 0,
    witness: bool = True,
) -> list[dict]:
    """Offer ``f(…, x, …) → f(…, matmul(matmul(x, P), Pᵀ), …)`` at every
    consumer edge of an activation class whose last dim ``d`` exceeds
    ``rank``.

    ``P ∈ ℝ^{d×r}`` has orthonormal columns (QR of a seeded random
    matrix), so ``PPᵀ`` is an orthogonal projection and the substitution
    is *contractive*:

        ‖x − x·PPᵀ‖_F ≤ ‖x‖_F ≤ √n · |x|max_cal

    — a certified bound (the residual can at worst discard everything
    outside ``ran(P)``).  It is honest but weak: when ``calib`` carries
    activation samples (``calibrate(..., keep_tensors=True)``) the
    *measured* residual ``max ‖x − xPPᵀ‖_F`` over the batch is recorded
    in the law text and offer dict — the informative number — while
    ``error_bound`` stays the certified contractive bound.  ``P`` and
    ``Pᵀ`` are injected into ``source_tensors`` as real params of the
    lowered module.
    """
    act = _activation_classes(eg)
    from catopt.typing import _shape_of

    gen = torch.Generator().manual_seed(seed)

    info: dict[int, tuple] = {}

    def wrap_factory(c):
        rep = act[c]
        shape = _shape_of(rep)
        if (
            not isinstance(shape, tuple)
            or not shape
            or not isinstance(shape[-1], int)
        ):
            return None
        d = shape[-1]
        if rank >= d:
            return None
        n = (
            math.prod(shape)
            if all(isinstance(dd, int) for dd in shape)
            else None
        )
        key = op_repr(rep)
        amax = _site_absmax(calib, key)
        P = (
            torch.linalg.qr(
                torch.randn(d, rank, generator=gen, dtype=torch.float64)
            )
            .Q[:, :rank]
            .contiguous()
        )
        # measured residual when calibration kept the activation samples
        resid = None
        samples = (
            calib.get("tensors", {}).get(key)
            if isinstance(calib, dict)
            else None
        )
        if torch.is_tensor(samples):
            xd = samples.detach().double()
            resid = float(torch.linalg.norm(xd - (xd @ P) @ P.T))
            P = P.to(samples.dtype)  # match the activation's dtype
        pname = f"act_lrP_{c}"
        ptname = f"act_lrPt_{c}"
        source_tensors[pname] = P
        source_tensors[ptname] = P.T.contiguous()
        p_eid = eg.add_term(Param(pname, TensorType((d, rank))))
        pt_eid = eg.add_term(Param(ptname, TensorType((rank, d))))
        inner = eg.add_enode("matmul", (c, p_eid), {})
        dq = eg.add_enode("matmul", (inner, pt_eid), {})
        bound = (
            amax * math.sqrt(n)
            if (amax is not None and n is not None)
            else float("inf")
        )
        info[c] = (bound, "frobenius", amax, n, d, resid)
        return dq

    def bound_of(c):
        return info[c][0], info[c][1]

    def law_of(c, i, node, bound):
        amax, n, d, resid = (
            info[c][2],
            info[c][3],
            info[c][4],
            info[c][5],
        )
        base = (
            f"rank-{rank} random projection of activation at input "
            f"{i} of {node.op}: x ↦ x·PPᵀ, P∈ℝ^{{{d}×{rank}}} "
            f"orthonormal ⇒ ‖Δx‖_F ≤ ‖x‖_F ≤ √n·|x|max"
        )
        if amax is not None:
            base += f" = {bound:.3e} (calibrated |x|max = {amax:.3e})"
        else:
            base += "; UNCALIBRATED"
        if resid is not None:
            base += (
                f"; measured residual {resid:.3e} on calibration batch"
            )
        return base

    return _wrap_sites(
        eg, act, wrap_factory, bound_of, "act_lr", law_of, witness
    )


# ---------------------------------------------------------------------------
#  Extraction helper — force the offered bounded members
# ---------------------------------------------------------------------------


def extract_with_offers(
    eg: EGraph,
    root_eid: int,
    offers: list[dict],
    *,
    cost_fn=None,
    quant_eids=None,
):
    """Extract a term forced through the offered bounded members.

    Activation quantization is a runtime *contract*, not a compute win —
    under any FLOP-style cost the exact member always wins, so selection
    is expressed as an extraction ``overrides`` (the same mechanism
    ``extract_paired`` uses for coordinated non-local choices): each
    chosen consumer class is pinned to its offered member.

    ``quant_eids`` optionally restricts which wrapped activation classes
    are forced (default: every offer).  Cyclic members (a class wrapped
    into itself) are skipped — extraction cannot realise them anyway.
    """
    if cost_fn is None:
        from catopt.cost import count_cost

        cost_fn = count_cost
    keep = set(quant_eids) if quant_eids is not None else None
    ov: dict[int, ENode] = {}
    for o in offers:
        if keep is not None and o["quant_eid"] not in keep:
            continue
        if o.get("cyclic"):
            continue
        ov.setdefault(eg.find(o["site_eid"]), o["member"])
    return eg.extract_best(root_eid, cost_fn, overrides=ov)
