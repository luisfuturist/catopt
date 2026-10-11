"""Held-out model zoo — architectures the intake corpus never saw.

``project/retros/real-corpus-yield.md`` measured ``shippable = 0``
on the real half of the intake corpus, and the remaining signal was
corpus-circular: every shipped rule's firing site is a purpose-built
spelling.  The question that retro could not answer is the honest
one — *does anything fire on architectures nobody targeted?*  This
module is the holdout set for that question: a registry of
``nn.Module`` workloads from architecture families the census has
never contained, kept deliberately OUT of
:func:`catopt_discovery.intake.candidates`
(``tests/test_discovery_zoo.py`` pins the disjointness).  Nothing
here was written to spell a known law's LHS or to fill a guard's
truth region — the selection criterion was *family coverage*:

* attention variants the corpus lacks — additive (Bahdanau),
  cosine-normalized (SwinV2-style), talking-heads mixing, and the
  softmax-free gated attention unit (GAU);
* conditioning spellings — FiLM and DiT's AdaLN (the real
  ``mul(unsqueeze, ·)`` sites, not the purpose-built corner);
* token mixers — MLP-Mixer, a Fourier neural operator block, GCN;
* routing — soft-MoE slot dispatch, expert-choice top-k, capsule
  routing;
* recurrence/scan — chunked RetNet retention, a deep-equilibrium
  unroll, WaveNet gated dilated conv;
* adapters/heads — LoRA, mixture-of-softmaxes, a VICReg-style
  covariance head, a KAN basis layer, Highway gates, RealNVP.

The registry reuses :class:`catopt_discovery.intake.Workload` so
:func:`catopt_discovery.intake.ingest` consumes it unchanged — the
holdout is a *provenance* property (the zoo is never registered in
``candidates()`` and never written to the ``tools/intake_corpus.json``
side-file), not a different ingestion path.  ``kind="zoo"`` keeps
the provenance visible in every report table and ``fire_cases`` name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from catopt_discovery.intake import Workload

__all__ = ["zoo"]


# ---------------------------------------------------------------------------
#  Attention variants the corpus lacks
# ---------------------------------------------------------------------------


class _GatedAttentionUnit(nn.Module):
    """GAU (FLASH) — softmax-free gated attention on squared scores.

    Hua et al. 2022: two cheap projections, a quadratic
    self-similarity score (``relu(u uᵀ)²``), and an output gate —
    no softmax anywhere.
    """

    def __init__(self, d: int) -> None:
        """Build the u/v projections and the output gate."""
        super().__init__()
        self.u = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sigmoid(gate(x)) ⊙ (relu(u uᵀ)² / s) v``."""
        u = torch.relu(self.u(x))
        a = (u @ u.transpose(-1, -2)) / (x.shape[-2] ** 0.5)
        return torch.sigmoid(self.gate(x)) * ((a * a) @ self.v(x))


class _AdditiveAttention(nn.Module):
    """Bahdanau additive attention — ``wᵀ tanh(Wq q + Wk k)`` scores.

    The original seq2seq attention: scores come from a learned MLP on
    the *sum* of projected query and key, broadcast over the (tq, tk)
    grid — the ``add(unsqueeze, unsqueeze)`` spelling, not a matmul.
    """

    def __init__(self, d: int) -> None:
        """Build the query/key/value projections and score vector."""
        super().__init__()
        self.qp = nn.Linear(d, d)
        self.kp = nn.Linear(d, d)
        self.vp = nn.Linear(d, d)
        self.score = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Score by the additive MLP, softmax over keys, mix values."""
        e = self.score(
            torch.tanh(
                self.qp(q).unsqueeze(2) + self.kp(k).unsqueeze(1)
            )
        ).squeeze(-1)
        a = torch.softmax(e, dim=-1)
        mixed = (a.unsqueeze(-1) * self.vp(v).unsqueeze(1)).sum(2)
        return self.out(mixed)


class _TalkingHeadsAttention(nn.Module):
    """Talking-heads attention — linear head-mixing around softmax.

    Shazeer 2020: per-head-pair weights mix the attention logits
    before the softmax and the probabilities after it, spelled as
    ``permute`` + ``matmul`` over the head axis.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the projections and the (heads x heads) mixers."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.pre = nn.Parameter(torch.randn(heads, heads) / heads)
        self.post = nn.Parameter(torch.randn(heads, heads) / heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend with head-mixed logits and head-mixed probabilities."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / (d // h) ** 0.5
        logits = (logits.permute(0, 2, 3, 1) @ self.pre).permute(
            0, 3, 1, 2
        )
        p = torch.softmax(logits, dim=-1)
        p = (p.permute(0, 2, 3, 1) @ self.post).permute(0, 3, 1, 2)
        return self.out((p @ v).transpose(1, 2).reshape(b, t, d))


class _CosineAttention(nn.Module):
    """Cosine attention — normalized q/k, learned logit scale.

    SwinV2-style: ``softmax(τ · q̂ k̂ᵀ)`` where q̂/k̂ are L2-normalized
    per head — the ``linalg_vector_norm`` spelling of attention.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the projections and the clamped logit scale."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend on cosine similarities scaled by a learned τ."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        logits = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-1, -2)
        ) * self.logit_scale.clamp(max=4.6)
        o = torch.softmax(logits, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(b, t, d))


# ---------------------------------------------------------------------------
#  Conditioning spellings — the real mul(unsqueeze, ·) sites
# ---------------------------------------------------------------------------


class _FiLMHead(nn.Module):
    """FiLM conditioning — ``gamma(c) ⊙ x + beta(c)`` feature-wise modulation.

    Perez et al. 2018: a conditioning input produces per-channel gain
    and bias, broadcast over the token axis — the canonical real
    ``mul(unsqueeze, ·)`` site (the purpose-built corpus only spelled
    the broadcast-trivial corner of this family).
    """

    def __init__(self, d: int, cdim: int) -> None:
        """Build the conditioning MLP and the feature norm."""
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.to_gb = nn.Linear(cdim, 2 * d)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Modulate normalized features by broadcast gain+shift."""
        gamma, beta = self.to_gb(c).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * gamma + beta


class _AdaLNBlock(nn.Module):
    """DiT adaptive layer-norm block — conditioning at block scale.

    Peebles & Xie 2022: ``LN(x) ⊙ (1 + scale(c)) + shift(c)`` feeds
    the block map and the residual is gated by ``gate(c)`` — three
    conditioning broadcasts in one real block.
    """

    def __init__(self, d: int, cdim: int) -> None:
        """Build the norm, the block map and the conditioning table."""
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        self.cond = nn.Linear(cdim, 3 * d)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Condition the block: scale+shift the norm, gate the skip."""
        scale, shift, gate = self.cond(c).unsqueeze(1).chunk(3, dim=-1)
        h = self.mlp(self.norm(x) * (1.0 + scale) + shift)
        return x + gate * h


class _LoRAAdapter(nn.Module):
    """LoRA adapter — frozen dense map plus a low-rank residual branch.

    Hu et al. 2021: ``W x + alpha·(B (A x))`` — the genuine production
    spelling of the shared-input factorization the purpose-built
    ``SharedFactor*`` workloads mimicked elementwise; here the shared
    factor is a matmul chain, what real adapters actually look like.
    """

    def __init__(self, d: int, rank: int) -> None:
        """Build the dense map and the (down, up) low-rank pair."""
        super().__init__()
        self.base = nn.Linear(d, d)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        self.scale = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dense map plus the scaled low-rank residual."""
        return self.base(x) + self.scale * self.up(self.down(x))


# ---------------------------------------------------------------------------
#  Token mixers / spectral / graphs
# ---------------------------------------------------------------------------


class _MLPMixerBlock(nn.Module):
    """MLP-Mixer block — token-mixing via transpose-sandwiched MLP.

    Tolstikhin 2021: mix along the token axis by transposing the
    feature map into a per-token linear — the transpose residuality
    family the corpus's Conformer block only touched by accident.
    """

    def __init__(self, d: int, t: int) -> None:
        """Build both norms and the token/channel mixing MLPs."""
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.token_mix = nn.Linear(t, t)
        self.channel_mix = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens (transposed linear) then channels (MLP)."""
        y = self.token_mix(self.norm1(x).transpose(1, 2))
        x = x + y.transpose(1, 2)
        return x + self.channel_mix(self.norm2(x))


class _FNOBlock(nn.Module):
    """Fourier neural operator layer — spectral filter plus bypass.

    Li et al. 2021: ``irfft2(scale · rfft2(x)) + conv1x1(x)`` with the
    top modes truncated.  The FFT ops are the zoo's honest
    binding-gap probe — this workload is expected to land census-only
    if the bridge lacks them.
    """

    def __init__(self, ch: int, modes: int) -> None:
        """Build the spectral scale and the bypass projection."""
        super().__init__()
        self.modes = modes
        self.scale = nn.Parameter(torch.randn(ch, modes) / modes)
        self.bypass = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter the lowest modes in frequency space, add bypass."""
        f = torch.fft.rfft2(x)
        lo = f[..., : self.modes] * self.scale[:, None, :]
        hi = torch.zeros_like(f[..., self.modes :])
        spec = torch.cat([lo, hi], dim=-1)
        return (
            self.bypass(x)
            + torch.fft.irfft2(spec, s=(x.shape[-2], x.shape[-1])).real
        )


class _GCNLayer(nn.Module):
    """Two-hop graph convolution — symmetric-normalized propagation.

    Kipf & Welling 2017: ``D^{-1/2} A D^{-1/2}`` built in-graph from
    the adjacency (two ``unsqueeze`` broadcast sites), then two
    propagation matmuls around the feature maps.
    """

    def __init__(self, d: int) -> None:
        """Build the two hop projections."""
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Propagate twice through the normalized adjacency."""
        d_inv = torch.rsqrt(adj.sum(-1).clamp_min(1e-6))
        norm = d_inv.unsqueeze(-1) * adj * d_inv.unsqueeze(-2)
        return torch.relu(norm @ self.w2(torch.relu(norm @ self.w1(x))))


class _AffineCoupling(nn.Module):
    """RealNVP affine coupling — split, scale-and-shift, recombine.

    Dinh et al. 2017: one half flows through unchanged and drives a
    ``(s, t)`` net that affinely transforms the other half —
    ``chunk`` + ``exp`` + ``cat`` spelling of a normalizing flow.
    """

    def __init__(self, d: int) -> None:
        """Build the conditioner producing per-feature (scale, shift)."""
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(d // 2, d), nn.ReLU(), nn.Linear(d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split channels, affine the second half, reconcatenate."""
        x1, x2 = x.chunk(2, dim=-1)
        s, t = self.cond(x1).chunk(2, dim=-1)
        y2 = x2 * torch.exp(s.clamp(-3.0, 3.0)) + t
        return torch.cat([x1, y2], dim=-1)


# ---------------------------------------------------------------------------
#  Routing / gating families
# ---------------------------------------------------------------------------


class _SoftSlotMoE(nn.Module):
    """Soft MoE — continuous slot dispatch instead of top-k routing.

    Puigcerver et al. 2023: every token is a convex combination over
    slots and each slot is a convex combination of tokens — dispatch
    and combine are softmaxes, not index selects.
    """

    def __init__(self, d: int, slots: int) -> None:
        """Build the slot logits and the per-slot MLP weights."""
        super().__init__()
        self.slot = nn.Linear(d, slots)
        self.w1 = nn.Parameter(torch.randn(slots, d, 2 * d) / d)
        self.w2 = nn.Parameter(torch.randn(slots, 2 * d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine tokens into slots, transform, dispatch back."""
        phi = self.slot(x)
        slots = torch.softmax(phi, dim=1).transpose(1, 2) @ x
        h = torch.einsum("bsd,sde->bse", slots, self.w1)
        h = torch.einsum("bse,sef->bsf", F.gelu(h), self.w2)
        return torch.softmax(phi, dim=-1) @ h


class _ExpertChoiceMoE(nn.Module):
    """Expert-choice routing — experts pick tokens, not vice versa.

    Zhou et al. 2022: per-expert top-k over the *token* axis, gather
    the chosen rows, apply each expert, scatter the results back —
    the gather/scatter_add spelling of MoE dispatch.
    """

    def __init__(self, d: int, n_exp: int, k: int) -> None:
        """Build the token scorer and the per-expert projections."""
        super().__init__()
        self.k = k
        self.score = nn.Linear(d, n_exp)
        self.w = nn.Parameter(torch.randn(n_exp, d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Each expert selects its top-k tokens and writes them back."""
        b, t, d = x.shape
        e = self.w.shape[0]
        w = torch.softmax(self.score(x), dim=-1).transpose(1, 2)
        gate, idx = torch.topk(w, self.k, dim=-1)
        sel = torch.gather(
            x.unsqueeze(1).expand(b, e, t, d),
            2,
            idx.unsqueeze(-1).expand(b, e, self.k, d),
        )
        y = torch.einsum("bekd,edf->bekf", sel, self.w)
        out = torch.zeros_like(x)
        flat_i = idx.reshape(b, e * self.k, 1).expand(b, e * self.k, d)
        flat_y = (y * gate.unsqueeze(-1)).reshape(b, e * self.k, d)
        return out.scatter_add(1, flat_i, flat_y)


class _CapsuleRouting(nn.Module):
    """Dynamic routing between capsule layers (Sabour et al. 2017).

    Two static routing iterations: agreement votes from a per-capsule
    transform, softmax coupling coefficients, squash nonlinearity.
    """

    def __init__(self, din: int, dout: int, n_out: int) -> None:
        """Build the vote transform ``(din, n_out, dout)``."""
        super().__init__()
        self.n_out = n_out
        self.w = nn.Parameter(torch.randn(din, n_out, dout) / din)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route lower capsules to output capsules, then squash."""
        votes = torch.einsum("bti,ioj->btoj", x, self.w)
        logits = torch.zeros(
            x.shape[0], x.shape[1], self.n_out, dtype=x.dtype
        )
        out = votes.sum(1)
        for _ in range(2):
            c = torch.softmax(logits, dim=-1)
            s = (c.unsqueeze(-1) * votes).sum(1)
            n2 = s.square().sum(-1, keepdim=True)
            out = s * n2 / ((1.0 + n2) * n2.sqrt().clamp_min(1e-6))
            logits = logits + (votes * out.unsqueeze(1)).sum(-1)
        return out


# ---------------------------------------------------------------------------
#  Recurrence / scan / conv families
# ---------------------------------------------------------------------------


class _ChunkedRetention(nn.Module):
    """RetNet inner-chunk retention — decay-masked quadratic attention.

    Sun et al. 2023: ``(QKᵀ ⊙ D) V`` where the decay matrix
    ``D[c,s] = gamma^{c-s}`` is built in-graph by ``tril`` over a power
    of a position difference — the same in-graph-bias family as
    ALiBi, but multiplicative rather than additive.
    """

    def __init__(self, d: int) -> None:
        """Build the q/k/v projections and record the decay rate."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gamma = 0.9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decay-masked attention within the chunk."""
        t = x.shape[-2]
        pos = torch.arange(t, dtype=x.dtype)
        decay = torch.tril(
            torch.pow(
                torch.tensor(self.gamma, dtype=x.dtype),
                (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0),
            )
        )
        s = self.q(x) @ self.k(x).transpose(-1, -2) / (t**0.5)
        return (s * decay) @ self.v(x)


class _DeepEquilibrium(nn.Module):
    """Deep-equilibrium block — an unrolled shared-weight fixed point.

    Bai et al. 2019: ``z ← tanh(Wz·z + Wx·x)`` iterated to a fixed
    point; unrolled three times it is a real program that reuses ONE
    parameter set across three stacked blocks — the spelling the
    param-sharing machinery exists for.
    """

    def __init__(self, d: int, iters: int) -> None:
        """Build the injection and the two fixed-point maps."""
        super().__init__()
        self.iters = iters
        self.inj = nn.Linear(d, d)
        self.fz = nn.Linear(d, d)
        self.fx = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iterate the fixed-point map a statically unrolled count."""
        z = torch.tanh(self.inj(x))
        for _ in range(self.iters):
            z = torch.tanh(self.fz(z) + self.fx(x))
        return z


class _WaveNetGate(nn.Module):
    """WaveNet gated dilated conv block — ``tanh(f) ⊙ sigmoid(g)`` + res.

    van den Oord et al. 2016: a dilated causal conv pair whose filter
    and gate branches multiply, then a 1x1 residual projection.
    """

    def __init__(self, ch: int) -> None:
        """Build the dilated filter/gate convs and the residual mix."""
        super().__init__()
        self.filter = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.gate = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.res = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated activation, cropped back to causal width."""
        f = torch.tanh(self.filter(x))[..., : x.shape[-1]]
        g = torch.sigmoid(self.gate(x))[..., : x.shape[-1]]
        return x + self.res(f * g)


class _TimeCondConv(nn.Module):
    """Diffusion residual block — sinusoidal time embedding + conv.

    The DDPM UNet piece: a sinusoidal timestep embedding projected
    into the conv block's channels and added as a broadcast bias —
    the 4-D ``unsqueeze`` conditioning site.
    """

    def __init__(self, ch: int, temb: int) -> None:
        """Build the time projection and the conv/norm stage."""
        super().__init__()
        self.temb = temb
        self.tproj = nn.Linear(2 * temb, ch)
        self.norm = nn.GroupNorm(4, ch)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add the sinusoidal time bias into the conv block."""
        freqs = torch.exp(
            torch.arange(self.temb, dtype=x.dtype)
            * (
                -torch.log(torch.tensor(10000.0, dtype=x.dtype))
                / self.temb
            )
        )
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = self.tproj(torch.cat([ang.sin(), ang.cos()], dim=-1))
        y = self.conv(F.silu(self.norm(x)))
        return x + y + emb[:, :, None, None]


# ---------------------------------------------------------------------------
#  Heads / objectives
# ---------------------------------------------------------------------------


class _MoSHead(nn.Module):
    """Mixture-of-softmaxes output head (Yang et al. 2018).

    ``Σ_k π_k(h) · softmax(W_k h)`` — a learned prior over several
    cheap softmaxes, the real alternative to a single full softmax.
    """

    def __init__(self, d: int, ncls: int, k: int) -> None:
        """Build the prior net and the stacked component logits."""
        super().__init__()
        self.prior = nn.Linear(d, k)
        self.w = nn.Parameter(torch.randn(k, d, ncls) / d)
        self.b = nn.Parameter(torch.zeros(k, ncls))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the prior-weighted mixture of component softmaxes."""
        pi = torch.softmax(self.prior(x), dim=-1)
        logits = torch.einsum("bd,kdn->bkn", x, self.w) + self.b
        probs = torch.softmax(logits, dim=-1)
        return (pi.unsqueeze(-1) * probs).sum(1)


class _CovarianceHead(nn.Module):
    """VICReg-style covariance statistic — centered Gram matrix.

    Bardes et al. 2022: ``C = Ẋᵀ Ẋ / (n-1)`` over the batch — a real
    self-supervised objective term spelling ``sub(mean)``,
    ``transpose`` and a contracted ``matmul`` with no labels anywhere.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature covariance of the batch."""
        c = x - x.mean(dim=0, keepdim=True)
        return c.transpose(0, 1) @ c / (x.shape[0] - 1)


class _SplineKAN(nn.Module):
    """KAN edge mixing over a fixed sinusoid/polynomial basis.

    Liu et al. 2024 (Kolmogorov-Arnold networks): each edge is a
    learned combination of basis functions of the input — spelled
    here as a ``stack([x, x², sin x])`` bank times an
    ``(in, out, basis)`` weight table.
    """

    def __init__(self, din: int, dout: int) -> None:
        """Build the per-edge basis weight table."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(din, dout, 3) / din)
        self.b = nn.Parameter(torch.zeros(dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis bank and mix it per output."""
        feats = torch.stack([x, x * x, torch.sin(x)], dim=-1)
        return torch.einsum("bif,iof->bo", feats, self.w) + self.b


class _HighwayGate(nn.Module):
    """Highway network block — learned transform/carry interpolation.

    Srivastava et al. 2015: ``t ⊙ h(x) + (1 - t) ⊙ x`` — the convex
    gate between a transformed and an identity path.
    """

    def __init__(self, d: int) -> None:
        """Build the transform and the carry-gate maps."""
        super().__init__()
        self.h = nn.Linear(d, d)
        self.t = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blend the transformed features with the identity."""
        t = torch.sigmoid(self.t(x))
        return t * torch.relu(self.h(x)) + (1.0 - t) * x


# ---------------------------------------------------------------------------
#  The registry — same Workload thunks as intake, disjoint names
# ---------------------------------------------------------------------------


def _r(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 CPU example input (intake's convention)."""
    return torch.randn(tuple(shape), dtype=torch.float64)


def _rand(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 uniform input (adjacency-like feeds)."""
    return torch.rand(tuple(shape), dtype=torch.float64)


def zoo(d: int = 16) -> list[Workload]:
    """Return the held-out registry — thunks like the intake's.

    Every workload builds a fresh module plus its example input under
    the caller's seed, exactly like ``intake.candidates`` — the same
    ``(model, feed)`` contract, ``kind="zoo"`` for the report tables,
    and ``purpose_built=False``: these are architectures chosen for
    family coverage, not spellings of a known law's region.  ``d``
    is the model width — the default keeps the honest small-scale
    probe; a bigger ``d`` checks the wins at sizes where pointwise
    tails carry real work.
    """
    return [
        # --- attention variants ----------------------------------
        Workload(
            "GatedAttentionUnit",
            lambda: (_GatedAttentionUnit(d), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "AdditiveAttention",
            lambda: (
                _AdditiveAttention(d),
                (_r(2, 4, d), _r(2, 8, d), _r(2, 8, d)),
            ),
            kind="zoo",
        ),
        Workload(
            "TalkingHeadsAttention",
            lambda: (_TalkingHeadsAttention(d, 4), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "CosineAttention",
            lambda: (_CosineAttention(d, 4), _r(2, 8, d)),
            kind="zoo",
        ),
        # --- conditioning ----------------------------------------
        Workload(
            "FiLMHead",
            lambda: (_FiLMHead(d, 8), (_r(2, 8, d), _r(2, 8))),
            kind="zoo",
        ),
        Workload(
            "AdaLNBlock",
            lambda: (_AdaLNBlock(d, 8), (_r(2, 8, d), _r(2, 8))),
            kind="zoo",
        ),
        Workload(
            "LoRAAdapter",
            lambda: (_LoRAAdapter(d, 4), _r(4, d)),
            kind="zoo",
        ),
        # --- token mixers / spectral / graphs --------------------
        Workload(
            "MLPMixerBlock",
            lambda: (_MLPMixerBlock(d, 8), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "FNOBlock",
            lambda: (_FNOBlock(8, 4), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "GCNLayer",
            lambda: (_GCNLayer(d), (_r(2, 6, d), _rand(2, 6, 6))),
            kind="zoo",
        ),
        Workload(
            "AffineCoupling",
            lambda: (_AffineCoupling(d), _r(4, d)),
            kind="zoo",
        ),
        # --- routing / gating ------------------------------------
        Workload(
            "SoftSlotMoE",
            lambda: (_SoftSlotMoE(d, 4), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "ExpertChoiceMoE",
            lambda: (_ExpertChoiceMoE(d, 4, 3), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "CapsuleRouting",
            lambda: (_CapsuleRouting(d, 8, 4), _r(2, 8, d)),
            kind="zoo",
        ),
        # --- recurrence / scan / conv ----------------------------
        Workload(
            "ChunkedRetention",
            lambda: (_ChunkedRetention(d), _r(2, 8, d)),
            kind="zoo",
        ),
        Workload(
            "DeepEquilibrium",
            lambda: (_DeepEquilibrium(d, 3), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "WaveNetGate",
            lambda: (_WaveNetGate(8), _r(1, 8, 16)),
            kind="zoo",
        ),
        Workload(
            "TimeCondConv",
            lambda: (_TimeCondConv(8, 4), (_r(1, 8, 8, 8), _r(1))),
            kind="zoo",
        ),
        # --- heads / objectives ----------------------------------
        Workload(
            "MoSHead",
            lambda: (_MoSHead(d, 10, 3), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "CovarianceHead",
            lambda: (_CovarianceHead(), _r(8, d)),
            kind="zoo",
        ),
        Workload(
            "SplineKAN",
            lambda: (_SplineKAN(d, 8), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "HighwayGate",
            lambda: (_HighwayGate(d), _r(4, d)),
            kind="zoo",
        ),
    ]
