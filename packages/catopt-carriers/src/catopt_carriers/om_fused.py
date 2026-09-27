# ruff: noqa: RUF002
"""Canonical fused level-steps for the online-softmax carriers.

The om analogue of :mod:`~catopt_carriers.scan_fused`: after eqsat with
``OM_LAWS`` (or the ``xcarrier`` lifts for the deferred fiber), chunked
attention extracts as ``om_apply(<om_compose tree>)`` /
``omd_apply[m](<omd_compose tree>, h)``.  :mod:`om_lower`'s batched
executor already runs the extracted tree's levels; these helpers
*re-bracket* the leaf sequence into the canonical adjacent-pair
reduction instead — the monoid products are association-invariant
(om is commutative too), so the result is the same composed carrier,
but every level is a pure strided pointwise step on shrinking fresh
tensors: no ``index_select`` gathers, no growing-buffer ``cat``,
exactly the shape Inductor fuses to a handful of kernels under
``torch.compile(fullgraph=True)``.

Leaf layout: the caller stacks each leaf carrier's components into
``(n, ...)`` tensors in *product order* — one row per leaf
*occurrence* (:func:`~catopt_carriers.scan_fused.occurrence_slots`
walks the term generically — om/omd compose trees are binary
``.args`` trees to it — and returns the DAG-expanded slot list; the
compose order itself is immaterial for om since ⊕ commutes, but the
occurrence count is not).  Non-power-of-two ``n`` is padded with the
monoid identity — ``(m, l, a) = (−inf, 0, 0)`` — which is exact:
``maximum(m, −inf) = m``, ``exp(m − m) = 1``, ``0·e = 0``.

NaN semantics mirror the serial bindings exactly: the ``where``/
``isfinite`` guards that keep a fully-masked block (m = −inf, NaN
payloads) from contaminating the product are hoisted to a one-shot
leaf sanitisation — after it, ``non-finite m ⇒ l = a = 0`` is a
loop invariant, so the per-level combine collapses to plain products
(the same strength reduction ``om_lower._batched_compose`` documents).
A fully-masked ROW still produces ``l = 0`` and ``a/l = NaN`` — dense
softmax's own semantics, preserved.
"""

from __future__ import annotations

import torch

from catopt_carriers.scan_fused import _pad_to_pow2, occurrence_slots

__all__ = [
    "fused_om_levels",
    "fused_omd_levels",
    "fused_omdm_levels",
    "occurrence_slots",
]


def _om_combine(m1, l1, a1, m2, l2, a2):
    """One adjacent-pair ``om_compose`` level, batched over slot dim.

    Identical math to ``om_lower._batched_compose`` (which documents
    why the pre-sanitised operands let the serial binding's ``where``
    guards collapse to plain products): the rescale factors keep their
    own ``isfinite`` guard — ``exp(−inf − −inf)`` would be NaN.
    """
    mx = torch.maximum(m1, m2)
    e1 = torch.where(torch.isfinite(m1), torch.exp(m1 - mx), 0.0)
    e2 = torch.where(torch.isfinite(m2), torch.exp(m2 - mx), 0.0)
    return mx, l1 * e1 + l2 * e2, a1 * e1 + a2 * e2


def fused_om_levels(
    m_seq: torch.Tensor,
    l_seq: torch.Tensor,
    a_seq: torch.Tensor,
) -> torch.Tensor:
    """Apply the composed online-softmax carrier: ``a_root / l_root``.

    ``m_seq``/``l_seq`` are the leaf row-max and denominator stacks
    ``(n, ..., Tq, 1)`` and ``a_seq`` the numerator ``(n, ..., Tq, d)``,
    all in product (occurrence) order — exactly the triples
    ``om_elem``/``om`` leaves evaluate to.  Non-pow2 ``n`` is padded
    with the identity ``(−inf, 0, 0)``; returns the ``om_apply``
    readout.
    """
    m = _pad_to_pow2(m_seq, float("-inf"))
    l_ = _pad_to_pow2(l_seq, 0.0)
    a = _pad_to_pow2(a_seq, 0.0)
    # Leaf sanitisation hoisted out of the serial compose guards (see
    # module docstring): after this, non-finite m ⇒ zero payloads.
    fin = torch.isfinite(m)
    l_ = torch.where(fin, l_, 0.0)
    a = torch.where(fin, a, 0.0)
    while m.shape[0] > 1:
        m, l_, a = _om_combine(
            m[0::2],
            l_[0::2],
            a[0::2],
            m[1::2],
            l_[1::2],
            a[1::2],
        )
    return a[0] / l_[0]


def _fused_omd_reduce(
    m_seq: torch.Tensor,
    l_seq: torch.Tensor,
    fa_seq: torch.Tensor,
    fb_seq: torch.Tensor,
):
    """Adjacent-pair reduce an ``omd`` leaf stack to its root 4-tuple.

    Same schedule as :func:`fused_om_levels` over the deferred
    carrier ``(m, l, fa, fb)``: both numerator factors rescale by the
    same ``e_i`` — ``fa`` may carry extra trailing axes past e's
    ``(…,Tq,1)`` (the dense fiber's coefficient is ``(…,Tq,o,i)``), so
    the row weight gets trailing 1-dims to broadcast, verbatim from
    ``xcarrier._omd_compose``.
    """
    m = _pad_to_pow2(m_seq, float("-inf"))
    l_ = _pad_to_pow2(l_seq, 0.0)
    fa = _pad_to_pow2(fa_seq, 0.0)
    fb = _pad_to_pow2(fb_seq, 0.0)
    fin = torch.isfinite(m)
    l_ = torch.where(fin, l_, 0.0)
    fa = torch.where(
        fin.reshape(*fin.shape, *([1] * (fa.dim() - fin.dim()))),
        fa,
        0.0,
    )
    fb = torch.where(fin, fb, 0.0)
    while m.shape[0] > 1:
        m1, l1, fa1, fb1 = (
            m[0::2],
            l_[0::2],
            fa[0::2],
            fb[0::2],
        )
        m2, l2, fa2, fb2 = (
            m[1::2],
            l_[1::2],
            fa[1::2],
            fb[1::2],
        )
        mx = torch.maximum(m1, m2)
        e1 = torch.where(torch.isfinite(m1), torch.exp(m1 - mx), 0.0)
        e2 = torch.where(torch.isfinite(m2), torch.exp(m2 - mx), 0.0)
        e1f = e1.reshape(*e1.shape, *([1] * (fa1.dim() - e1.dim())))
        e2f = e2.reshape(*e2.shape, *([1] * (fa2.dim() - e2.dim())))
        m = mx
        l_ = l1 * e1 + l2 * e2
        fa = fa1 * e1f + fa2 * e2f
        fb = fb1 * e1 + fb2 * e2
    return m[0], l_[0], fa[0], fb[0]


def fused_omd_levels(
    m_seq: torch.Tensor,
    l_seq: torch.Tensor,
    fa_seq: torch.Tensor,
    fb_seq: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """``omd_apply`` readout on the reduced root: ``(fa⊙h + fb)/l``.

    Diagonal fiber: ``fa_seq`` is the stacked ``(n, ..., Tq, d)``
    diagonal coefficient — the ``omd_elem`` ``e@a`` component.
    """
    _, l_, fa, fb = _fused_omd_reduce(m_seq, l_seq, fa_seq, fb_seq)
    return (fa * h + fb) / l_


def fused_omdm_levels(
    m_seq: torch.Tensor,
    l_seq: torch.Tensor,
    fa_seq: torch.Tensor,
    fb_seq: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """``omd_applym`` readout on the reduced root: ``(fa@h + fb)/l``.

    Dense fiber: ``fa_seq``'s last axis is the map's input axis —
    stacked ``(n, ..., Tq, o, i)`` — contracting ``h`` gives the
    ``(…,Tq,o)`` numerator.
    """
    _, l_, fa, fb = _fused_omd_reduce(m_seq, l_seq, fa_seq, fb_seq)
    return (fa @ h + fb) / l_
