"""Selective state-space model (SSM) blocks — Mamba/S4-style input-dependent
dynamics, used as targets for the affine-map scan laws (``SCAN_LAWS``).

Two recurrence forms live here:

* :class:`SelectiveSSM` — a *dense* selective transition.  Each step is

      h_t = A_t h_{t-1} + u_t,
      A_t = I + Δ_t · A   (Euler discretisation),   u_t = B_t ⊙ x_t,

  with Δ_t = tanh(W_δ x_t) and B_t = W_B x_t both input-dependent.
  ``A_t @ h`` exports as ``matmul`` so the step matches the affine lift
  ``add(matmul(A, h), x) → apply(aff(A, x), h)`` — the per-step operator
  is an affine map even though the map itself is a function of the data.

  ``torch.exp`` is elementwise, so the ZOH-flavoured ``exp(Δ_t·A)``
  produces a near-ones (spectral radius ≈ d) matrix rather than a
  near-identity one — a footgun that makes the unrolled recurrence
  explode to ~1e16 in T steps.  The Euler form ``I + Δ_t·A`` is the
  correct first-order discretisation for a dense A and stays
  contractive for small ‖Δ_t·A‖ (‖Δ‖ < 1 via tanh, ‖A‖ small init).

* :class:`DiagonalSSM` — the *elementwise* (Mamba-faithful) form

      h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t,   a_t = σ(W_a x_t) ∈ (0,1),

  which exports as ``add(mul(a_t, h), mul(b_t, x_t))`` — NO ``matmul``.
  The affine lift cannot see it: ``h ↦ a⊙h + b`` is still an affine map
  (a diagonal one), but ``AFF_LIFT``'s LHS requires a literal ``matmul``
  node.  Full diagonal-SSM coverage needs a second lift rule such as
  ``add(mul(a, h), x) → apply(aff_diag(a, x), h)`` plus ``aff_diag``
  compose/apply lowering in the torch bridge — kept out of rules.py per
  the experiment's constraints and documented in test_ssm_scan.py.

Everything is ``.double()``-compatible for fp64-exact verification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectiveSSM(nn.Module):
    """Minimal selective SSM block with a dense input-dependent transition.

    ``h_t = A_t h_{t-1} + (B_t ⊙ x_t)`` unrolled over ``steps`` steps,
    where ``A_t = I + Δ_t ⊙ A`` (row-scaled Euler discretisation) and
    ``Δ_t = tanh(Δ_proj(x_t))``, ``B_t = B_proj(x_t)``.

    Args:
        d_inner: state dimension (16–32 keeps the e-graph small).
        d_in:    input feature dimension.
        steps:   sequence length T — the loop is unrolled at export time.
    """

    def __init__(
        self, d_inner: int = 16, d_in: int = 16, steps: int = 16
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_inner, d_inner) * 0.05)
        self.delta_proj = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj = nn.Linear(d_in, d_inner, bias=False)
        self.h0 = nn.Parameter(torch.zeros(d_inner))
        self.register_buffer("eye", torch.eye(d_inner))
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, d_in)
        delta = torch.tanh(self.delta_proj(x))  # (T, d_inner), |Δ| < 1
        B = self.B_proj(x)  # (T, d_inner)
        h = self.h0
        for t in range(self.steps):
            A_t = self.eye + delta[t].unsqueeze(-1) * self.A  # (d, d)
            h = A_t @ h + B[t] * x[t]
        return h


class DiagDenseSSM(nn.Module):
    """Diagonal selective decay materialised as a dense matmul.

    ``A_t = diag(a_t)`` with ``a_t = σ(decay_proj(x_t)) ∈ (0,1)`` —
    contractive by construction (Mamba-style decay) — but written
    ``(eye * a_t) @ h`` so the step still has the ``matmul`` shape the
    affine lift matches.  A correctness reference point: the e-graph
    reaches the same balanced scan as for :class:`SelectiveSSM`.
    """

    def __init__(
        self, d_inner: int = 16, d_in: int = 16, steps: int = 16
    ) -> None:
        super().__init__()
        self.decay_proj = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj = nn.Linear(d_in, d_inner, bias=False)
        self.h0 = nn.Parameter(torch.zeros(d_inner))
        self.register_buffer("eye", torch.eye(d_inner))
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.decay_proj(x))  # (T, d_inner) in (0,1)
        B = self.B_proj(x)
        h = self.h0
        for t in range(self.steps):
            A_t = self.eye * a[t].unsqueeze(-1)  # diag(a_t), dense
            h = A_t @ h + B[t] * x[t]
        return h


class DiagonalSSM(nn.Module):
    """Mamba-faithful elementwise selective scan: ``h_t = a_t ⊙ h_{t-1}
    + b_t ⊙ x_t``.

    Exported as ``add(mul(a_t, h), mul(b_t, x_t))`` — the affine step is
    present mathematically (a diagonal affine map) but ``SCAN_LAWS``
    cannot lift it because ``AFF_LIFT`` matches ``matmul``, not ``mul``.
    This is the honest negative result for elementwise SSMs.
    """

    def __init__(
        self, d_inner: int = 16, d_in: int = 16, steps: int = 16
    ) -> None:
        super().__init__()
        self.decay_proj = nn.Linear(d_in, d_inner, bias=False)
        self.B_proj = nn.Linear(d_in, d_inner, bias=False)
        self.h0 = nn.Parameter(torch.zeros(d_inner))
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.decay_proj(x))
        b = self.B_proj(x)
        h = self.h0
        for t in range(self.steps):
            h = a[t] * h + b[t] * x[t]
        return h
