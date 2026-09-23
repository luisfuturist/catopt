"""Jamba-style hybrid blocks — a diagonal SSM and chunked attention in
one module, so the e-graph sees BOTH monoid carriers simultaneously.

Two carriers live side by side here:

* the *diagonal-affine* monoid (``aff_diag`` / ``affd_compose`` /
  ``applyd`` — see ``SCAN_DIAG_LAWS`` in :mod:`catopt.rules`) over the
  Mamba-faithful recurrence ``h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t``;
* the *online-softmax* monoid (``om_elem`` / ``om_compose`` /
  ``om_apply`` — see ``OM_LAWS`` in :mod:`catopt.om`) over the chunked
  attention ``softmax(q @ cat(k_i)ᵀ) @ cat(v_i)``.

The point of the exercise is the *seam*: the SSM emits a per-timestep
sequence ``y = stack(h_0..h_{T-1})`` which the attention layer then
reads through its Q/K/V projections, so the om carrier's score blocks
contain scan-produced e-classes.  Any rewrite that fuses across that
boundary — or fails to — is the finding.

Design notes:

* The attention is written in the decomposed (pre-flash) form —
  explicit ``q @ Kᵀ``, ``softmax``, ``@ V`` — with K and V spelled as
  ``torch.cat`` of ``chunk`` blocks so ``MATMUL_T_CONCAT`` and
  ``OM_SPLIT`` can see the block structure.
* The ``1/√d`` scale is folded into **q before the matmul**
  (``q * scale``), not onto the scores — a scalar ``mul`` *wrapping*
  the score concat would sit between ``om_elem`` and the ``concat``
  enode and block ``OM_SPLIT``'s homomorphism, exactly like a
  ``masked_fill``/additive causal mask would.  Kept unmasked for the
  same reason: this module exists to expose the carrier, not to be a
  production attention.
* ``torch.export`` emits ``cat`` with the axis positional
  (``arg1=-2``), while every rule-produced concat uses ``dim=``.  The
  om rule set speaks ``dim`` internally, so tests normalise the
  exported spelling before saturating — see tests/test_hybrid.py for
  the (honest) finding that the two spellings cannot recombine.

Everything is ``.double()``-compatible for fp64-exact verification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridBlock(nn.Module):
    """Diagonal-SSM recurrence feeding chunked self-attention.

    ``h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t`` with ``a_t = σ(decay_proj(x_t))``
    and ``b_t = B_proj(x_t)``, unrolled over ``steps`` steps; every
    intermediate state is collected into a sequence
    ``y = stack(h_0..h_{T-1})`` (T, d_inner).  Attention then reads y:

        q, k, v = q_proj(y)·s, k_proj(y), v_proj(y)
        K = cat(chunk(k, C, -2)),  V = cat(chunk(v, C, -2))
        out = softmax(q @ Kᵀ) @ V  →  out_proj

    Args:
        d_inner:  SSM state dimension (16–32 keeps the e-graph small).
        d_in:     input feature dimension.
        d_attn:   attention inner dimension.
        steps:    sequence length T — the loop is unrolled at export.
        n_chunks: number of key/value blocks the concat is built from.
    """

    def __init__(self, d_inner: int = 16, d_in: int = 16,
                 d_attn: int = 16, steps: int = 16,
                 n_chunks: int = 2) -> None:
        super().__init__()
        self.decay_proj = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj = nn.Linear(d_in, d_inner, bias=False)
        self.h0 = nn.Parameter(torch.zeros(d_inner))
        self.q_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.k_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.v_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_in, bias=False)
        self.steps = steps
        self.n_chunks = n_chunks
        self.scale = d_attn ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, d_in)
        a = torch.sigmoid(self.decay_proj(x))     # (T, d_inner) in (0,1)
        b = self.B_proj(x)                        # (T, d_inner)
        h = self.h0
        ys = []
        for t in range(self.steps):
            h = a[t] * h + b[t] * x[t]
            ys.append(h)
        y = torch.stack(ys, dim=0)                # (T, d_inner)

        q = self.q_proj(y) * self.scale           # scale on q, pre-matmul
        k = self.k_proj(y)
        v = self.v_proj(y)
        # Chunked K/V: cat of blocks so the om homomorphism can split.
        K = torch.cat(list(k.chunk(self.n_chunks, dim=-2)), dim=-2)
        V = torch.cat(list(v.chunk(self.n_chunks, dim=-2)), dim=-2)
        att = torch.softmax(q @ K.transpose(-2, -1), dim=-1)
        return self.out_proj(att @ V)


class TwoLayerHybrid(nn.Module):
    """SSM → attention → SSM: does the search compose across layers?

    Same carriers as :class:`HybridBlock`, stacked so that the second
    recurrence's inputs ``y2[t]`` are themselves reads of the attention
    output — the scan monoid's translation vectors then contain the
    om carrier (or the dense softmax form) inside ``select`` slices.
    Returns the second SSM's stacked state sequence, (T, d_inner).
    """

    def __init__(self, d_inner: int = 16, d_in: int = 16,
                 d_attn: int = 16, steps: int = 8,
                 n_chunks: int = 2) -> None:
        super().__init__()
        # Layer 1: diagonal SSM
        self.decay_proj1 = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj1 = nn.Linear(d_in, d_inner, bias=False)
        self.h0_1 = nn.Parameter(torch.zeros(d_inner))
        # Layer 2: chunked attention over the SSM sequence
        self.q_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.k_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.v_proj = nn.Linear(d_inner, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_in, bias=False)
        # Layer 3: second diagonal SSM reading the attention output
        self.decay_proj2 = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj2 = nn.Linear(d_in, d_inner, bias=False)
        self.h0_2 = nn.Parameter(torch.zeros(d_inner))
        self.steps = steps
        self.n_chunks = n_chunks
        self.scale = d_attn ** -0.5

    def _ssm(self, x: torch.Tensor, decay: nn.Linear,
             B: nn.Linear, h0: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(decay(x))
        b = B(x)
        h = h0
        ys = []
        for t in range(self.steps):
            h = a[t] * h + b[t] * x[t]
            ys.append(h)
        return torch.stack(ys, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, d_in)
        y1 = self._ssm(x, self.decay_proj1, self.B_proj1, self.h0_1)

        q = self.q_proj(y1) * self.scale
        k = self.k_proj(y1)
        v = self.v_proj(y1)
        K = torch.cat(list(k.chunk(self.n_chunks, dim=-2)), dim=-2)
        V = torch.cat(list(v.chunk(self.n_chunks, dim=-2)), dim=-2)
        att = torch.softmax(q @ K.transpose(-2, -1), dim=-1)
        y2 = self.out_proj(att @ V)               # (T, d_in)

        return self._ssm(y2, self.decay_proj2, self.B_proj2, self.h0_2)
