"""Canonical fused level-step for the batched scan executor.

:func:`~catopt_carriers.scan_lower.build_scan_plan` keeps the
*extracted* bracketing — arbitrary ``aff_compose``/``affd_compose``
tree shapes and DAG-shared subtrees — for scheduling metadata
(``levels``/``level_gather``/``n_levels``).  The fused executor
instead *re-brackets* the leaf sequence into the canonical
adjacent-pair reduction: the monoid product is association-invariant,
so any bracketing computes the same composed map, but adjacent pairs
turn every level into a pure strided pointwise step on a shrinking
fresh tensor — no ``index_select`` gathers, no growing-buffer
``cat`` — exactly the shape Inductor fuses to ~1 kernel per level
(and into a handful of kernels overall).  Measured on an RTX 2050
(retnet ``applyd`` term, fp32): torch.compile over this loop reaches
~1.4-1.7x of the single fused kernel Inductor emits for the unrolled
serial recurrence itself, and ~4x under it once captured in a CUDA
graph.

The module holds only pure-tensor helpers: occurrence-order analysis
(:func:`occurrence_slots`), the diagonal level body
(:func:`fused_diag_levels`), and the dense one
(:func:`fused_dense_levels`).  ``BatchedScanModule`` wires them in;
keeping them free functions makes them trivially ``torch.compile``-
wrappable and unit-testable.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "fused_dense_levels",
    "fused_diag_levels",
    "occurrence_slots",
]

#: Bound on leaf occurrences before the fused path is declined.  A
#: DAG-shared subtree is *recomputed* once per occurrence — for real
#: extracted terms sharing is rare, but a pathological DAG could make
#: occurrence expansion exponentially larger than the tree.  Falling
#: back to the standard level-gather schedule keeps those terms fast.
_MAX_OCC_FACTOR = 8


def occurrence_slots(
    f_term: Any, leaves: list[Any]
) -> list[int] | None:
    """Leaf-slot indices in product order, with multiplicity.

    Walks the (folded) map term depth-first — ``args[0]``'s subtree
    before ``args[1]``'s, the same order :func:`level_schedule`
    encounters leaves — and records each leaf *occurrence*'s slot in
    ``leaves``.  For a plain tree this is just ``range(n)``; a
    DAG-shared leaf or subtree yields a longer list (the fused
    reduction recomputes shared subtrees per occurrence, which stays
    correct: the product is what the term means).

    Returns ``None`` when occurrences exceed ``_MAX_OCC_FACTOR *
    len(leaves)`` — the caller then keeps the standard schedule.
    """
    slot = {id(lf): i for i, lf in enumerate(leaves)}
    occ: list[int] = []
    stack = [f_term]
    limit = _MAX_OCC_FACTOR * max(len(leaves), 1)
    while stack:
        t = stack.pop()
        s = slot.get(id(t))
        if s is not None:
            occ.append(s)
            if len(occ) > limit:
                return None
            continue
        args = getattr(t, "args", None)
        if not args:
            # Non-leaf, non-compose terminal inside the map tree —
            # plan shape guarantees this can't happen, but decline
            # rather than silently drop an operand.
            return None
        stack.extend(reversed(args))
    return occ


def _pad_to_pow2(t: torch.Tensor, fill: float) -> torch.Tensor:
    """Cat ``fill``-valued identity rows onto ``t`` up to a power of 2.

    The canonical reduction needs a power-of-two leaf count; identity
    affine maps (``a = 1, b = 0`` — or the identity matrix) pad the
    tail without changing the product.
    """
    n = t.shape[0]
    pad = (1 << max(n - 1, 0).bit_length()) - n
    if not pad:
        return t
    ones = t.new_full((pad, *t.shape[1:]), fill)
    return torch.cat([t, ones])


def fused_diag_levels(
    a_seq: torch.Tensor,
    b_seq: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Apply the composed diagonal scan map to ``h``.

    ``a_seq``/``b_seq`` hold the leaf ``(a, b)`` vectors in product
    order — ``(n, d)`` or batched ``(n, ..., d)``.  Each level pairs
    adjacent rows: ``out_a = a_f ⊙ a_g``, ``out_b = a_f ⊙ b_g + b_f``
    where ``f`` is the earlier (outer) operand.  Two pointwise ops per
    level on fresh tensors; returns ``a_root ⊙ h + b_root``.
    """
    a = _pad_to_pow2(a_seq, 1.0)
    b = _pad_to_pow2(b_seq, 0.0)
    while a.shape[0] > 1:
        fa, ga = a[0::2], a[1::2]
        fb, gb = b[0::2], b[1::2]
        a = fa * ga
        b = torch.addcmul(fb, fa, gb)
    return torch.addcmul(b[0], a[0], h)


def fused_dense_levels(
    m_seq: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Apply the composed dense affine map to ``h``.

    ``m_seq`` holds the leaf homogeneous matrices ``[[A, b], [0, 1]]``
    in product order — ``(n, d+1, d+1)``.  Adjacent-pair reduction by
    batched matmul; returns ``A_root h + b_root``.
    """
    n, d1 = m_seq.shape[0], m_seq.shape[-1]
    m = m_seq
    pad = (1 << max(n - 1, 0).bit_length()) - n
    if pad:
        eye = torch.eye(d1, dtype=m_seq.dtype, device=m_seq.device)
        m = torch.cat([m, eye.unsqueeze(0).expand(pad, d1, d1)])
    while m.shape[0] > 1:
        m = m[0::2] @ m[1::2]
    r = m[0]
    d = d1 - 1
    return r[:d, :d] @ h + r[:d, d]
