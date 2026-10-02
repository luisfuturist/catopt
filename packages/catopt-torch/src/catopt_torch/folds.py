"""Compile-time const folds — the causal-mask specialization.

Moved from ``catopt_orchestrator.optimize`` (plan 0007): the fold
evaluates parameter-only subtrees to concrete tensors and rewrites
``sdpa(q,k,v, mask)`` → ``sdpa(q,k,v, is_causal=True)`` when the
materialised mask is exactly the causal lower triangle.  The
machinery is inherently backend machinery — it evaluates tensors in
the backend's own runtime — so it lives with the torch adapter and
the sink exposes it through the optional
:attr:`~catopt_core.ports.Capabilities` hook ``specialize_causal``.

The orchestrator calls it as
``sink.specialize_causal(term, params, memo) -> term``; ``memo["_hit"]``
records whether any rewrite fired (the search records
``stats["causal_specialized"]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from catopt_core.ir import Op

if TYPE_CHECKING:
    from catopt_core.ports import OpRegistry


def _eval_const(
    term: Any, params: dict, ops: OpRegistry | None = None
) -> torch.Tensor | None:
    """Evaluate a parameter-only subtree to a concrete tensor.

    Permissive compile-time fold — any un-evaluatable piece (Var,
    missing Param, missing binding, raising binding, non-tensor
    result) yields ``None``.  Delegates to
    :func:`catopt_torch.torch_bridge.eval_term` (plan 0002 phase D);
    ``tensor_only`` reproduces the per-level isinstance check.
    """
    from catopt_torch.torch_bridge import _IR_TO_TORCH, eval_term

    bindings = _IR_TO_TORCH if ops is None else ops.torch_bindings
    return eval_term(
        term,
        param_env=params,
        bindings=bindings,
        tensor_only=True,
    )


def _is_causal_keep_mask(mask_val: torch.Tensor, q_shape) -> bool:
    """Return True when a mask is exactly the causal lower triangle.

    ``(…, T, T)`` keeps the lower triangle and T matches q's sequence
    dim — i.e. the mask IS is_causal.
    """
    if not isinstance(q_shape, tuple) or len(q_shape) < 2:
        return False
    if (
        mask_val.ndim < 2
        or mask_val.shape[-1] != mask_val.shape[-2]
        or mask_val.shape[-1] != q_shape[-2]
    ):
        return False
    keep = (
        mask_val.bool()
        if mask_val.dtype == torch.bool
        else mask_val > -1e30
    )
    tril = torch.tril(
        torch.ones(
            mask_val.shape[-2],
            mask_val.shape[-1],
            dtype=torch.bool,
            device=mask_val.device,
        )
    )
    return bool((keep == tril).all())


def _specialize_causal(
    term: Any,
    params: dict,
    memo: dict | None = None,
    ops: OpRegistry | None = None,
) -> Any:
    """sdpa(q,k,v, mask) → sdpa(q,k,v, is_causal=True).

    Applies when mask is parameter-only and evaluates to a causal
    keep-mask.  Dropping the materialised mask unlocks the fused
    flash/mem-efficient kernels.
    """
    from catopt_core.typing import shape_of as _so

    if memo is None:
        memo = {}
    if not isinstance(term, Op):
        return term
    key = term  # content-keyed: interned terms hash by structure
    if key in memo:
        return memo[key]
    args = tuple(
        _specialize_causal(a, params, memo, ops) for a in term.args
    )
    attrs = dict(term.attrs)
    if term.op == "sdpa" and len(args) >= 4 and not attrs.get("arg5"):
        mv = _eval_const(args[3], params, ops)
        if mv is not None and _is_causal_keep_mask(mv, _so(args[0])):
            args = args[:3]
            attrs["arg5"] = True
            memo["_hit"] = True
    out = Op.make(term.op, *args, **attrs)
    memo[key] = out
    return out
