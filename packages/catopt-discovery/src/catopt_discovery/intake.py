"""Workload intake — feed the census programs nobody wrote by hand.

The corpus is hand-curated: forty ``catopt_torch.models`` classes
registered in ``catopt_discovery.impact._model_cases``.  ``law-workload-gen.md``
proved resampling cannot mint new op-tuples and
``corpus-expansion-r2.md`` proved real architectures can — but every
expansion round meant writing a new builder by hand and registering
it in code.  The missing half of self-play is an **intake path**: a
tool that takes a ``torch.nn.Module`` + example input, runs the real
export boundary (``torch.export`` → ``export_to_ir``), and appends
whatever survives to the census — recording, just as honestly, what
the boundary rejects.

What counts as a candidate (``candidates()``):

* **torch-native modules** — ``nn.MultiheadAttention``,
  ``nn.TransformerEncoder`` stacks, ``nn.BatchNorm``,
  ``nn.Embedding``, the RNN family, pooling, losses, ….  Library
  code spells ops differently than our hand builders (``permute``,
  ``unflatten``, ``split_with_sizes``, ``feature_dropout``); every
  spelling the bridge cannot lower is a recorded finding, not a
  crash.
* **compound models** — hand-built but realistic assemblies (a
  shared-expert MoE, a ViT patch block, a bottleneck residual, a
  classifier loss head) at real depth.

Every candidate lands in exactly one class:

* **ingested** — exported, every op bound, and the lowered module
  verifies fp64 against the original.  These join the firing/reach
  probe under a node cap — deeper stacks still feed the census and
  the matchers.
* **census-only** — exported but at least one op has no torch
  binding, or the lowered module fails verification.  The term still
  feeds the census (a real program's shape is a real shape); the
  missing ops are the binding-gap backlog this tool exists to
  surface.
* **rejected** — ``torch.export`` / ``export_to_ir`` raised; the
  exception is recorded verbatim.

The census seam is a side-file, not a code edit: a run emits
``tools/intake_corpus.json`` (terms via ``catopt_core.ir.term_to_data``
plus per-workload metadata and the rejection backlog) and
``tools/intake_tensors.pt`` (the feeds/params ``torch.save``'d, so a
loaded term lowers and verifies exactly like an exported model).
``catopt_discovery.census.corpus()`` unions :func:`load_cases` when the
file exists — the file IS the appended census — and
``catopt_discovery.pipeline.run_pipeline`` reads the same file for its matcher
terms and (via :func:`probe_cases`) its firing probe.  Absent the
file, nothing changes: the baseline corpus is untouched.

Run::

    .venv/bin/python -m catopt_discovery.intake            # ingest + report
    .venv/bin/python -m catopt_discovery.intake --skip-pipeline
    .venv/bin/python -m catopt_discovery.intake --json /tmp/intake.json
    .venv/bin/python -m catopt_discovery.intake --holdout select_mul

CPU-only, a few minutes when the pipeline delta runs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import (
    Op,
    TensorType,
    Var,
    term_from_data,
    term_to_data,
)
from catopt_torch.adapters import TorchSink
from catopt_torch.torch_bridge import export_to_ir

from catopt_discovery import TOOLS
from catopt_discovery.impact import (
    TermCase,
    _bench_cases,
    _ir_of,
    _iter_subterms,
    model_cases,
)

__all__ = [
    "Rejection",
    "Workload",
    "candidates",
    "census_delta",
    "ingest",
    "load_cases",
    "load_records",
    "main",
    "probe_cases",
    "real_workloads",
    "write_intake",
]

#: The default side-file paths — the appended census + its tensors,
#: which stayed behind in ``tools/`` when the engine moved here.
_DEFAULT_JSON = TOOLS / "intake_corpus.json"
_DEFAULT_TENSORS = TOOLS / "intake_tensors.pt"

#: Verify tolerance for the lower-the-term-vs-original-model check.
_RTOL = 1e-4

#: Node cap for the firing/reach probe.  Deep stacks (the 105-node
#: TransformerEncoder(3), the 271-node full Transformer) still feed
#: the census and the matchers; the probe stays comparable in cost
#: to the corpus's own models (~80 nodes).
_PROBE_MAX_NODES = 120


# ---------------------------------------------------------------------------
#  Compound candidates — realistic assemblies, not one-op leaves
# ---------------------------------------------------------------------------


class _SharedExpertMoE(nn.Module):
    """DeepSeek-style routed MLP with a shared expert.

    An always-on shared expert plus a softmax-gated top-k dispatch
    over routed experts.
    """

    def __init__(self, d: int, hidden: int, n_exp: int) -> None:
        """Build the gate, the shared expert and the routed bank."""
        super().__init__()
        self.gate = nn.Linear(d, n_exp)
        self.shared = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d)
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, d),
            )
            for _ in range(n_exp)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the shared expert with weight-masked experts."""
        w = torch.softmax(self.gate(x), dim=-1)
        _, idx = torch.topk(w, 2, dim=-1)
        out = self.shared(x)
        for i, e in enumerate(self.experts):
            mask = (idx == i).any(dim=-1, keepdim=True).to(x.dtype)
            out = out + w[:, i : i + 1] * mask * e(x)
        return out


class _ViTPatchBlock(nn.Module):
    """ViT front end: strided-conv patchify, flatten, encoder layer."""

    def __init__(self, ch: int, d: int, heads: int) -> None:
        """Build the patch projection and the encoder layer."""
        super().__init__()
        self.patch = nn.Conv2d(ch, d, 4, 4)
        self.enc = nn.TransformerEncoderLayer(
            d, heads, 2 * d, batch_first=True, dropout=0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify then encode — the ``flatten``/``transpose`` spine."""
        t = self.patch(x)
        return self.enc(t.flatten(2).transpose(1, 2))


class _TinyCNN(nn.Module):
    """VGG-ish conv->bn->relu->pool stack at real depth."""

    def __init__(self, ch: int) -> None:
        """Build the two conv+norm stages and the linear head."""
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.head = nn.Linear(ch, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv-bn-relu-pool x2, adaptive pool, flatten, linear."""
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.b2(self.c2(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class _LossHead(nn.Module):
    """A training-step forward graph: body plus the CE loss itself."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the MLP body and the loss module."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, ncls)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss — a real two-input graph."""
        return self.loss(self.body(x), y)


class _ManualNLL(nn.Module):
    """``log_softmax`` + ``nll_loss`` spelled by hand."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the classifier head."""
        super().__init__()
        self.head = nn.Linear(d, ncls)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return NLL of the log-softmaxed logits."""
        return F.nll_loss(F.log_softmax(self.head(x), dim=-1), y)


class _Bottleneck(nn.Module):
    """ResNet bottleneck: 1x1 reduce, 3x3, 1x1 expand, plus skip."""

    def __init__(self, ch: int, mid: int) -> None:
        """Build the three conv+norm stages."""
        super().__init__()
        self.c1 = nn.Conv2d(ch, mid, 1)
        self.b1 = nn.BatchNorm2d(mid)
        self.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.b2 = nn.BatchNorm2d(mid)
        self.c3 = nn.Conv2d(mid, ch, 1)
        self.b3 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce-compute-expand with a residual add."""
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        return F.relu(self.b3(self.c3(y)) + x)


class _LogSoftmaxHead(nn.Module):
    """The canonical classifier tail: linear then ``log_softmax``."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the linear head."""
        super().__init__()
        self.head = nn.Linear(d, ncls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities."""
        return F.log_softmax(self.head(x), dim=-1)


# ---------------------------------------------------------------------------
#  Round-2 compound candidates — bound-but-uncovered op spellings
# ---------------------------------------------------------------------------


class _CircularPad(nn.Module):
    """Circular padding — a ``pad`` mode spelling conv builders use."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad circularly on the spatial axes."""
        return F.pad(x, (1, 1, 1, 1), mode="circular")


class _SpatialTransformer(nn.Module):
    """STN localization: ``affine_grid`` + ``grid_sample``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity-warp the input through a lifted theta."""
        theta = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=x.dtype
        ).expand(x.shape[0], 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class _ShiftedWindow(nn.Module):
    """Swin-style cyclic window shift — ``roll`` + rank-6 reshape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shift by half a window, partition, shift back."""
        y = torch.roll(x, shifts=(-2, -2), dims=(1, 2))
        b, h, w, c = y.shape
        y = (
            y.reshape(b, h // 4, 4, w // 4, 4, c)
            .transpose(2, 3)
            .reshape(b, h // 4, w // 4, 16, c)
        )
        return torch.roll(y, shifts=(2, 2), dims=(1, 2))


class _ALiBiAttention(nn.Module):
    """ALiBi attention: in-graph slope/distance bias feeding sdpa."""

    def __init__(self, heads: int) -> None:
        """Record the head count used for the slope schedule."""
        super().__init__()
        self.heads = heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Build the distance-penalty bias, then attend."""
        n = q.shape[-2]
        pos = torch.arange(n, dtype=q.dtype)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        slopes = torch.pow(
            torch.tensor(2.0, dtype=q.dtype),
            -torch.arange(self.heads, dtype=q.dtype) / self.heads,
        )
        bias = rel.unsqueeze(0) * slopes.reshape(-1, 1, 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class _GQAAttention(nn.Module):
    """Grouped-query attention — sdpa's ``enable_gqa`` spelling."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend with fewer kv heads than q heads."""
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class _EinsumAttention(nn.Module):
    """Attention spelled with ``einsum`` contractions."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Score, softmax, mix — all einsum."""
        s = torch.einsum("bhqd,bhkd->bhqk", q, k) / (q.shape[-1] ** 0.5)
        return torch.einsum(
            "bhqk,bhvd->bhqd", torch.softmax(s, dim=-1), v
        )


class _ManualAttention(nn.Module):
    """Manual attention: matmul + scale + softmax + matmul."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend without sdpa."""
        s = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return torch.softmax(s, -1) @ v


class _VarNorm(nn.Module):
    """Manual layer norm through ``var_mean``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last axis by its moments."""
        v, m = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        return (x - m) * torch.rsqrt(v + 1e-5)


class _StdNorm(nn.Module):
    """Manual normalisation through ``std_mean``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whitening spelled with std_mean."""
        s, m = torch.std_mean(x, dim=-1, keepdim=True)
        return (x - m) / (s + 1e-5)


class _VarStdHead(nn.Module):
    """Feature statistics: ``var`` + ``std`` reductions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return var+std summaries."""
        return x.var(dim=-1, keepdim=True) + x.std(dim=-1, keepdim=True)


class _FakeQuant(nn.Module):
    """Quantize-dequantize a layer: amax scale, clamp, round."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric int8-ish fake quantisation."""
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3) / 7.0
        return torch.round(x / s).clamp(-7, 7) * s


class _SincKernel(nn.Module):
    """Sinc interpolation kernel with a Kaiser-ish ``i0`` window."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale x by the summed windowed sinc."""
        t = torch.arange(16, dtype=x.dtype) - 7.5
        k = torch.sinc(t) * torch.i0(
            8.0 * (torch.ones_like(t) - (t / 8) ** 2).clamp_min(0)
        )
        return x * k.sum()


class _ScalarRsub(nn.Module):
    """``1 - sigmoid(x)`` — the ``rsub.Scalar`` spelling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the complement of a gate."""
        return 1.0 - torch.sigmoid(x)


class _CumsumScan(nn.Module):
    """Prefix computations: ``cumsum`` + ``logcumsumexp``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sum-prefix plus log-sum-exp-prefix."""
        return torch.cumsum(x, dim=1) + torch.logcumsumexp(x, dim=1)


class _CumProdGate(nn.Module):
    """Decay gate spelled as a cumulative product."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the running product of sigmoid gates."""
        return torch.cumprod(x.sigmoid(), dim=-1)


class _CumMaxMin(nn.Module):
    """Running range: ``cummax`` - ``cummin``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the running max-min envelope."""
        hi = torch.cummax(x, dim=-1).values
        lo = torch.cummin(x, dim=-1).values
        return hi - lo


class _IndexMoE(nn.Module):
    """Expert dispatch that scatters outputs back via ``index_add``."""

    def __init__(self, d: int) -> None:
        """Build the routed expert."""
        super().__init__()
        self.expert = nn.Linear(d, d)

    def forward(
        self, x: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Apply the expert then scatter rows back by index."""
        out = torch.zeros_like(x)
        return out.index_add(0, idx, self.expert(x))


class _ScatterAdd(nn.Module):
    """``scatter_add`` dispatch — bag-style index accumulation."""

    def forward(
        self, x: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate x's rows into a 4-row base by idx."""
        base = torch.zeros(4, 8, dtype=x.dtype)
        return base.scatter_add(0, idx.unsqueeze(-1).expand(-1, 8), x)


class _MedianPool(nn.Module):
    """Robust pooling — ``median`` over the feature axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the median per row."""
        return torch.median(x, dim=-1).values


class _SortSelect(nn.Module):
    """Rank selection through ``sort`` + slice."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the top-4 sorted values doubled."""
        v, _i = torch.sort(x, dim=-1, descending=True)
        return v[..., :4] * 2


class _ArgSort(nn.Module):
    """``argsort`` — rank indices as features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the argsort as floats."""
        return x.argsort(dim=-1).to(x.dtype)


class _KthMode(nn.Module):
    """Order statistics: ``kthvalue`` + ``mode``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return third-order statistic plus mode."""
        v, _i = torch.kthvalue(x, 3, dim=-1)
        m, _j = torch.mode(x, dim=-1)
        return v + m


class _TakeAlong(nn.Module):
    """``take_along_dim`` — gather by computed indices."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the top-4 elements by their argsort."""
        idx = x.argsort(dim=-1, descending=True)[..., :4]
        return x.take_along_dim(idx, dim=-1)


class _NormalizeHead(nn.Module):
    """``F.normalize`` — linalg_vector_norm + expand_as."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalise rows."""
        return F.normalize(x, p=2.0, dim=-1)


class _SlidingUnfold(nn.Module):
    """Sliding-window pooling via the ``unfold`` tensor method."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum non-overlapping-stride windows."""
        return x.unfold(-1, 4, 2).sum(-1)


class _MoveDimStack(nn.Module):
    """``movedim`` + ``vstack``/``hstack`` assembly."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Move axes, stack both spellings, combine."""
        moved = torch.vstack([x.movedim(1, 0), y.movedim(1, 0)]).sum(0)
        return moved + torch.hstack([x, y]).sum()


class _TensorSplit(nn.Module):
    """``tensor_split`` — uneven-division splitting."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split in two and re-multiply the halves."""
        a, b = torch.tensor_split(x, 2, dim=-1)
        return a * b


class _AngleLog(nn.Module):
    """``atan2`` + ``log1p`` — polar/log-domain head."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return angle plus soft log."""
        return torch.atan2(x, y) + torch.log1p(x * x)


class _Log10Expm1(nn.Module):
    """``log10``/``expm1`` — alternate log bases."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log10 of |x| plus expm1∘log1p of |x|."""
        return torch.log10(x.abs() + 1e-3) + torch.expm1(
            torch.log1p(x.abs())
        )


class _LogBase(nn.Module):
    """``log2``/``exp2`` — bit-exact log-domain features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log2 plus exp2."""
        return torch.log2(x.clamp_min(1e-3)) + torch.exp2(x)


class _NanGuard(nn.Module):
    """``where`` + ``isfinite`` — sanitize activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero non-finite entries."""
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


class _SanitizeHead(nn.Module):
    """``isnan``/``isinf``/``nan_to_num``/``logical_or`` cleanup."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp non-finite values and mask them out."""
        bad = torch.logical_or(torch.isnan(x), torch.isinf(x))
        fixed = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return fixed * torch.logical_not(bad).to(x.dtype)


class _NanStats(nn.Module):
    """``nanmean``/``nansum`` — NaN-tolerant reductions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return nan-tolerant mean plus sum."""
        return torch.nanmean(x, dim=-1) + torch.nansum(x, dim=-1)


class _MaskLogic(nn.Module):
    """Boolean mask composition — ``logical_and``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keep values in (0, 1)."""
        m = torch.logical_and(x > 0, x < 1)
        return x * m.to(x.dtype)


class _LogicCombo(nn.Module):
    """``logical_or``/``logical_xor``/``logical_not`` composition."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Combine predicates from two tensors."""
        m = torch.logical_or(x > 0, y > 0)
        m2 = torch.logical_xor(m, torch.logical_not(y > 1))
        return x * m2.to(x.dtype)


class _Bucketize(nn.Module):
    """``searchsorted`` — bucket assignment like FeatureProcessor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bucket x against quartile edges."""
        edges = torch.arange(4, dtype=x.dtype) / 4
        return torch.searchsorted(edges, x).to(x.dtype)


class _ChannelShuffle(nn.Module):
    """ShuffleNet channel shuffle — reshape/transpose spine."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shuffle four channel groups."""
        b, c, h, w = x.shape
        return (
            x.reshape(b, 4, c // 4, h, w)
            .transpose(1, 2)
            .reshape(b, c, h, w)
            .contiguous()
        )


class _FlipConv(nn.Module):
    """True convolution via a ``flip``ped kernel (corr→conv)."""

    def __init__(self) -> None:
        """Build the kernel."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv1d with the flipped kernel."""
        return F.conv1d(x, torch.flip(self.w, dims=[-1]))


class _GeoMeanPool(nn.Module):
    """Geometric-mean pooling — ``log``/``mean``/``exp``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the geometric mean per row."""
        return torch.exp(torch.log(x.abs() + 1e-3).mean(dim=-1))


class _LerpMix(nn.Module):
    """``lerp`` — gated linear interpolation."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Blend x and y by a learned-free sigmoid gate."""
        w = torch.sigmoid(x.mean(-1, keepdim=True))
        return torch.lerp(x, y, w)


class _AddcmulHead(nn.Module):
    """``addcmul``/``addcdiv`` — fused scaled-update spellings."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return x + 0.5*y*z + 0.25*y/z."""
        return x.addcmul(y, z, value=0.5) + x.addcdiv(
            y, z.clamp_min(1e-3), value=0.25
        )


class _TriuAttention(nn.Module):
    """Anti-causal attention — ``triu`` mask + ``masked_fill``."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend with an upper-triangular (backward) mask."""
        n = q.shape[-2]
        m = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        s = (q @ k.transpose(-1, -2)).masked_fill(m, float("-inf"))
        return torch.softmax(s, -1) @ v


class _TracePenalty(nn.Module):
    """Weight regularizer via the diagonal sum — aten ``trace``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return the readout plus tr(w)."""
        return (x @ w).sum() + torch.trace(w)


class _OuterPositional(nn.Module):
    """Positional matrix built with ``outer``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the outer-product position matrix."""
        n = x.shape[-1]
        pos = torch.arange(n, dtype=x.dtype)
        return x + torch.outer(pos, pos) / n


class _RepeatHead(nn.Module):
    """``repeat`` — tiled expansion then mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tile the batch, fold back by mean."""
        return x.repeat(2, 1).reshape(4, 2, 16).mean(1)


class _RemainderHead(nn.Module):
    """``remainder`` vs ``fmod`` — the two mod spellings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return their difference (sign handling)."""
        return x.remainder(2.0) - torch.fmod(x, 2.0)


class _ProdAminPool(nn.Module):
    """``prod`` + ``amin`` — product and min reductions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return row product plus min |x|."""
        return x.prod(dim=-1) + x.abs().amin(dim=-1)


class _MinFmax(nn.Module):
    """``min`` pair + ``fmax`` — elementwise bounds."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return row min plus clipped-y sum."""
        v, _i = torch.min(x, dim=-1)
        return v + torch.fmax(y, torch.zeros_like(y)).sum(-1)


class _BroadcastTo(nn.Module):
    """``broadcast_to`` — explicit broadcast then add."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Broadcast x into y's shape and add."""
        return torch.broadcast_to(x.unsqueeze(-1), y.shape) + y


class _CountHead(nn.Module):
    """``count_nonzero`` — a counting feature."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count positive entries per row."""
        return torch.count_nonzero(x > 0, dim=-1).to(x.dtype)


class _BooleanIndex(nn.Module):
    """Boolean-mask indexing — the ``index`` op spelling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the positive entries."""
        return x[x > 0].sum()


class _SignTruncFrac(nn.Module):
    """``sign``/``trunc``/``frac`` — integer-part decomposition."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sign + trunc + frac."""
        return torch.sign(x) + torch.trunc(x) + torch.frac(x)


class _CeilFloor(nn.Module):
    """``ceil``/``floor``/``reciprocal``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ceil - floor + clipped reciprocal."""
        return (
            torch.ceil(x) - torch.floor(x) + x.reciprocal().clamp(-4, 4)
        )


class _Heaviside(nn.Module):
    """``heaviside`` — a step gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Step at zero."""
        return torch.heaviside(x, torch.tensor(0.5, dtype=x.dtype))


class _SdpaBiasParam(nn.Module):
    """sdpa with a learned additive bias (attn_mask operand)."""

    def __init__(self, n: int) -> None:
        """Build the (n, n) bias parameter."""
        super().__init__()
        self.bias = nn.Parameter(torch.randn(n, n))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend with the learned bias."""
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.bias
        )


class _CDist(nn.Module):
    """``cdist`` — pairwise row distances (k-NN style)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return pairwise euclidean distances."""
        return torch.cdist(x, y)


class _EyeInit(nn.Module):
    """In-graph ``eye`` — identity-matrix residual."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the identity matrix."""
        return x + torch.eye(16, dtype=x.dtype)


class _UnbindHead(nn.Module):
    """``unbind`` — sequence-of-steps spelled per step."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the two unbound halves."""
        a, b = x.unbind(dim=0)
        return a + b


class _SliceFill(nn.Module):
    """``y[:, :4] = 0`` — ``fill_`` through a slice view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero the first four columns in place."""
        y = x.clone()
        y[:, :4] = 0.0
        return y


class _FillWhole(nn.Module):
    """``y.fill_(0.5)`` — whole-tensor scalar fill."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill the clone."""
        y = x.clone()
        y.fill_(0.5)
        return y


class _FillSelect(nn.Module):
    """``y[0].fill_(0)`` — ``fill_`` through a select view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero the first row in place."""
        y = x.clone()
        y[0].fill_(0.0)
        return y


class _ZeroInit(nn.Module):
    """``y.zero_()`` — the zeroing spelling of a fill."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero the clone."""
        y = x.clone()
        y.zero_()
        return y


class _MaskedFillInplace(nn.Module):
    """``y.masked_fill_(m, 0)`` — masked in-place write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask out positive entries in place."""
        y = x.clone()
        y.masked_fill_(x > 0, 0.0)
        return y


class _MaskedFillTensor(nn.Module):
    """``masked_fill_`` with a tensor fill value."""

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Mask positives to the given value."""
        y = x.clone()
        y.masked_fill_(x > 0, v)
        return y


class _MaskedFillView(nn.Module):
    """``masked_fill_`` through a slice view — scattered write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask positives in the first four columns only."""
        y = x.clone()
        y[:, :4].masked_fill_(x[:, :4] > 0, 0.0)
        return y


class _SelectWrite(nn.Module):
    """``y[0] = v`` — ``select_scatter`` write."""

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Write v into row zero."""
        y = x.clone()
        y[0] = v
        return y


class _IndexPut(nn.Module):
    """``y[idx] = v`` — ``index_put`` write."""

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Write v into rows 0 and 2."""
        y = x.clone()
        y[torch.tensor([0, 2], dtype=torch.long)] = v
        return y


class _NarrowHead(nn.Module):
    """``narrow`` — explicit windowing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a middle window."""
        return x.narrow(-1, 2, 8) * 2


class _NewCreators(nn.Module):
    """``new_ones``/``full_like`` — tensor-derived creators."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a ones and a half tensor shaped like x."""
        return x + x.new_ones(x.shape) + torch.full_like(x, 0.5)


class _TypeAs(nn.Module):
    """``type_as`` — dtype matching by operand."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cast x to y's dtype and add."""
        return x.type_as(y) + y


class _ExpandAs(nn.Module):
    """``expand_as`` — expand to another tensor's shape."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Expand x into y's trailing shape."""
        return x.unsqueeze(-1).expand_as(y) + y


class _IsNegPos(nn.Module):
    """``isneginf``/``isposinf`` — infinity bookkeeping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count infinities through a cast."""
        ninf = torch.isneginf(x).to(x.dtype)
        pinf = torch.isposinf(x).to(x.dtype)
        return x + ninf - pinf


class _TrigHead(nn.Module):
    """``atan``/``acos``/``asin`` — inverse trig features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the three inverse trig maps."""
        c = x.clamp(-0.9, 0.9)
        return torch.atan(x) + torch.acos(c) + torch.asin(c)


class _HyperbolicHead(nn.Module):
    """``sinh``/``cosh``/``asinh`` — hyperbolic features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a manual tanh plus asinh."""
        return torch.sinh(x) / torch.cosh(x) + torch.asinh(x)


class _CoshAcosh(nn.Module):
    """``cosh``/``acosh``/``atanh`` — the inverse family."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return cosh + acosh(|x|+1.1) + atanh(clamped)."""
        return (
            torch.cosh(x)
            + torch.acosh(x.abs() + 1.1)
            + torch.atanh(x.clamp(-0.9, 0.9))
        )


class _DigammaLn(nn.Module):
    """``digamma``/``gammaln`` — variational-style features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate digamma and gammaln at |x|+1."""
        a = x.abs() + 1.0
        return torch.digamma(a) + torch.special.gammaln(a)


class _XlogyKL(nn.Module):
    """``xlogy`` — KL divergence spelled in log-space."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return KL(p‖q) row-wise."""
        p = torch.softmax(x, -1)
        q = torch.softmax(y, -1)
        return (torch.xlogy(p, p) - torch.xlogy(p, q)).sum(-1)


class _RegluMlp(nn.Module):
    """ReGLU MLP — chunk + relu + mul gated feed-forward."""

    def __init__(self, d: int) -> None:
        """Build the up/down projections."""
        super().__init__()
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.relu(a) * b)


class _SwiGLU(nn.Module):
    """SwiGLU MLP — chunk + silu + mul gated feed-forward."""

    def __init__(self, d: int) -> None:
        """Build the up/down projections."""
        super().__init__()
        self.w = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        a, b = self.w(x).chunk(2, -1)
        return self.v(F.silu(a) * b)


class _OneHotEmbed(nn.Module):
    """``one_hot`` + ``to``-cast + matmul — bag-of-classes readout."""

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """One-hot then project through an identity matrix."""
        oh = F.one_hot(idx, 16).to(torch.float64)
        return oh @ torch.eye(16, dtype=torch.float64)


class _BitwiseHead(nn.Module):
    """Python ``&``/``|`` — the ``__and__``/``__or__`` spellings."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Combine bit patterns."""
        return ((x & y) | (x | y)).to(torch.float64)


class _RepeatInterleave(nn.Module):
    """``repeat_interleave`` — per-element duplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Duplicate every feature twice."""
        return x.repeat_interleave(2, dim=-1)


class _DepthwiseConv2d(nn.Module):
    """Depthwise + pointwise conv — ``conv2d`` with ``groups``."""

    def __init__(self) -> None:
        """Build the depthwise/pointwise pair."""
        super().__init__()
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw = nn.Conv2d(8, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise conv, relu, pointwise mix."""
        return self.pw(F.relu(self.dw(x)))


# ---------------------------------------------------------------------------
#  Round 3 — transformer variants, norm variants, conv/attn hybrids, scans
# ---------------------------------------------------------------------------


def _rope_freqs(
    t: int, d: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (cos, sin) rotary tables for *t* positions, *d* dims."""
    inv = torch.pow(
        torch.tensor(10000.0, dtype=dtype),
        -torch.arange(0, d, 2, dtype=dtype) / d,
    )
    freqs = torch.outer(torch.arange(t, dtype=dtype), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rope_apply(
    z: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate *z* by the (cos, sin) tables (interleaved-pair form)."""
    half = z[..., : z.shape[-1] // 2]
    rot = torch.cat([-z[..., z.shape[-1] // 2 :], half], dim=-1)
    return z * cos + rot * sin


class _RotaryEmbedding(nn.Module):
    """RoPE applied to a (B, H, T, D) tensor — the rotate-half spine."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate q/k-style features by their positions."""
        cos, sin = _rope_freqs(x.shape[-2], x.shape[-1], x.dtype)
        return _rope_apply(x, cos, sin)


class _RoPEAttention(nn.Module):
    """Multi-head attention with rotary q/k — the modern decoder block."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the qkv projection and the output projection."""
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate q/k, attend, project back."""
        b, t, d = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(b, t, 3, h, d // h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        cos, sin = _rope_freqs(t, d // h, x.dtype)
        q = _rope_apply(q, cos, sin)
        k = _rope_apply(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class _MQAAttention(nn.Module):
    """Multi-query attention — one shared kv head broadcast over q."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the q and packed kv projections."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * (d // heads))
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand the single kv head over all query heads, then attend."""
        b, t, d = x.shape
        h = self.heads
        hd = d // h
        q = self.q(x).reshape(b, t, h, hd).transpose(1, 2)
        kv = self.kv(x).reshape(b, t, 2, hd).transpose(1, 2)
        k, v = kv[:, 0], kv[:, 1]
        k = k.unsqueeze(1).expand(b, h, t, hd)
        v = v.unsqueeze(1).expand(b, h, t, hd)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, t, d)


class _RelativePositionBias(nn.Module):
    """T5-style bucketed relative position bias added to scores."""

    def __init__(self, heads: int, n: int) -> None:
        """Build the per-head bias table."""
        super().__init__()
        self.bias = nn.Parameter(torch.randn(heads, n))
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up the bucketed bias and add it to the scores."""
        t = x.shape[-2]
        idx = torch.arange(t)
        rel = (idx[None, :] - idx[:, None]).clamp(min=0)
        b = self.bias[:, rel]
        return x + b.unsqueeze(0)


class _MoEGate(nn.Module):
    """Top-2 softmax gating over a routed expert bank."""

    def __init__(self, d: int, n_exp: int) -> None:
        """Build the gate and the experts."""
        super().__init__()
        self.gate = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList(
            nn.Linear(d, d) for _ in range(n_exp)
        )
        self.n_exp = n_exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the top-2 experts by their softmax weights."""
        w = torch.softmax(self.gate(x), dim=-1)
        _, idx = torch.topk(w, 2, dim=-1)
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            sel = (idx == i).any(dim=-1, keepdim=True).to(x.dtype)
            out = out + sel * w[:, i : i + 1] * e(x)
        return out


class _PreNormBlock(nn.Module):
    """Pre-norm transformer block: norm before attention and MLP."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the two norms, attention and the MLP."""
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention then a pre-norm MLP."""
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + y
        return x + self.mlp(self.n2(x))


class _PostNormBlock(nn.Module):
    """Post-norm block: norm after each residual add."""

    def __init__(self, d: int, heads: int) -> None:
        """Build attention, norms and the MLP."""
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend then norm, MLP then norm."""
        y, _ = self.attn(x, x, x)
        x = self.n1(x + y)
        return self.n2(x + self.mlp(x))


class _SwiGLUBlock(nn.Module):
    """Pre-norm block with sdpa attention and a SwiGLU feed-forward."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the fused qkv, output, and gated MLP projections."""
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.w = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attend then apply the gated MLP."""
        b, t, d = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(b, t, 3, h, d // h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(attn)
        a, bb = self.w(self.n(x)).chunk(2, dim=-1)
        return x + self.v(F.silu(a) * bb)


class _CrossAttentionBlock(nn.Module):
    """Decoder cross-attention: q from x, kv from a context tensor."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the q, packed kv and output projections."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.heads = heads

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        """Attend x over ctx."""
        b, t, d = x.shape
        h = self.heads
        q = self.q(x).reshape(b, t, h, d // h).transpose(1, 2)
        kv = self.kv(ctx).reshape(b, ctx.shape[1], 2, h, d // h)
        k, v = kv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(b, t, d))


class _SlidingWindowMask(nn.Module):
    """Banded sliding-window attention mask over a local neighbourhood."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend within a window of ±2 positions."""
        n = q.shape[-2]
        idx = torch.arange(n)
        band = (idx[None, :] - idx[:, None]).abs() <= 2
        mask = torch.zeros(n, n, dtype=q.dtype).masked_fill(
            band.logical_not(), float("-inf")
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class _MultiLatentAttention(nn.Module):
    """MLA-style attention: kv compressed through a low-rank latent."""

    def __init__(self, d: int, heads: int, rank: int) -> None:
        """Build the q, down- and up-projection for the kv latent."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.dkv = nn.Linear(d, 2 * rank)
        self.ukv = nn.Linear(2 * rank, 2 * d)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress kv to the latent, expand back, attend."""
        b, t, d = x.shape
        h = self.heads
        q = self.q(x).reshape(b, t, h, d // h).transpose(1, 2)
        c = self.ukv(self.dkv(x))
        k, v = c.reshape(b, t, 2, h, d // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, t, d)


class _ManualRMSNorm(nn.Module):
    """RMSNorm spelled out: ``x * rsqrt(mean(x²)+eps) * g``."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        """Build the gain and record eps."""
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise by the root mean square."""
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.w


class _QKNorm(nn.Module):
    """Per-head RMS normalisation of q and k before attention."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Normalise q/k by their RMS, then attend."""

        def nrm(z: torch.Tensor) -> torch.Tensor:
            rms = z.pow(2).mean(dim=-1, keepdim=True)
            return z * torch.rsqrt(rms + 1e-6)

        return F.scaled_dot_product_attention(nrm(q), nrm(k), v)


class _DeepNorm(nn.Module):
    """DeepNorm residual: ``norm(alpha*x + res)``."""

    def __init__(self, d: int, alpha: float = 1.5) -> None:
        """Build the norm and record the residual scale."""
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.alpha = alpha

    def forward(
        self, x: torch.Tensor, res: torch.Tensor
    ) -> torch.Tensor:
        """Scale the branch, add the residual, normalise."""
        return self.n(self.alpha * x + res)


class _CenterNorm(nn.Module):
    """Mean-centering only — no variance rescaling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract the per-row mean."""
        return x - x.mean(dim=-1, keepdim=True)


class _ScaleNorm(nn.Module):
    """Scale-only norm: ``g * x / ||x||``."""

    def __init__(self, d: int) -> None:
        """Build the gain."""
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by the row norm then scale."""
        return self.g * x / x.norm(dim=-1, keepdim=True)


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7, LN, inverted bottleneck."""

    def __init__(self, ch: int) -> None:
        """Build the depthwise conv and the pointwise pair."""
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.n = nn.LayerNorm(ch, eps=1e-6)
        self.pw1 = nn.Linear(ch, 4 * ch)
        self.pw2 = nn.Linear(4 * ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise conv, channel-last norm, GELU MLP, residual."""
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.n(y)
        y = self.pw2(F.gelu(self.pw1(y))).permute(0, 3, 1, 2)
        return x + y


class _SqueezeExcitation(nn.Module):
    """SE block: global average pool, bottleneck gate, channel scale."""

    def __init__(self, ch: int, r: int = 4) -> None:
        """Build the squeeze/expand linears."""
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-weight channels by the pooled gate."""
        b, c, _h, _w = x.shape
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.reshape(b, c, 1, 1)


class _MBConvBlock(nn.Module):
    """MobileNet inverted-bottleneck block with squeeze-excitation."""

    def __init__(self, ch: int) -> None:
        """Build expand, depthwise, SE and project stages."""
        super().__init__()
        self.exp = nn.Conv2d(ch, 4 * ch, 1)
        self.dw = nn.Conv2d(4 * ch, 4 * ch, 3, padding=1, groups=4 * ch)
        self.se = _SqueezeExcitation(4 * ch)
        self.proj = nn.Conv2d(4 * ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand, depthwise, excite, project, residual."""
        y = F.silu(self.exp(x))
        y = self.se(F.silu(self.dw(y)))
        return x + self.proj(y)


class _GRN(nn.Module):
    """Global Response Norm — a conv-native spatial normaliser."""

    def __init__(self, ch: int) -> None:
        """Build the learned gain and bias."""
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate the spatial L2 norm and re-scale."""
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.g * (nx * x) + self.b + x


class _ConformerBlock(nn.Module):
    """Conformer block: FFN, self-attention and a depthwise conv module."""

    def __init__(self, d: int, heads: int) -> None:
        """Build the two FFNs, attention and the conv module."""
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        self.ff2 = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Half FFN, attention, conv, half FFN — all residual."""
        x = x + 0.5 * self.ff1(x)
        a, _ = self.attn(x, x, x)
        x = x + a
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + c
        return x + 0.5 * self.ff2(x)


class _AxialAttention(nn.Module):
    """Axial attention — rows then columns of a feature map."""

    def __init__(self, ch: int, heads: int) -> None:
        """Build the shared attention module."""
        super().__init__()
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend across width, then across height."""
        b, c, h, w = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        a, _ = self.attn(rows, rows, rows)
        a = (
            a.reshape(b, h, w, c)
            .permute(0, 2, 1, 3)
            .reshape(b * w, h, c)
        )
        a2, _ = self.attn(a, a, a)
        return a2.reshape(b, w, h, c).permute(0, 3, 2, 1)


class _PatchMerging(nn.Module):
    """ViT patch-merging: 2x2 reshape, concat, norm, linear reduce."""

    def __init__(self, d: int) -> None:
        """Build the norm and the reduction linear."""
        super().__init__()
        self.n = nn.LayerNorm(4 * d)
        self.red = nn.Linear(4 * d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge 2x2 patches."""
        b, n, c = x.shape
        s = int(n**0.5)
        y = x.reshape(b, s, s, c)
        y = torch.cat(
            [
                y[:, 0::2, 0::2],
                y[:, 1::2, 0::2],
                y[:, 0::2, 1::2],
                y[:, 1::2, 1::2],
            ],
            dim=-1,
        )
        return self.red(self.n(y.reshape(b, -1, 4 * c)))


class _SelectiveScan(nn.Module):
    """Mamba-style selective scan as a parallel cumulative decay."""

    def __init__(self, d: int) -> None:
        """Build the gate and input projections."""
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear recurrence through cumprod/cumsum."""
        g = torch.sigmoid(self.a(x))
        u = self.b(x)
        decay = torch.cumprod(g, dim=1)
        return decay * torch.cumsum(u / decay.clamp_min(1e-6), dim=1)


class _LinearAttention(nn.Module):
    """Linear attention via a cumsum over keys and values."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Feature-map attention with prefix sums."""
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        num = torch.einsum("btd,btdv->btv", q, kv)
        den = torch.cumsum(k, dim=1)
        den = torch.einsum("btd,btd->bt", q, den).unsqueeze(-1)
        return num / den.clamp_min(1e-6)


class _CumulativeGateRNN(nn.Module):
    """Elementwise recurrence spelled as a cumulative gate product."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale by the running product of sigmoid gates."""
        g = torch.sigmoid(x)
        return torch.cumprod(g, dim=1) * x


class _RunningNorm(nn.Module):
    """Running (prefix) normalisation through cumsum moments."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each step by the prefix mean/variance."""
        s = torch.cumsum(x, dim=1)
        ss = torch.cumsum(x * x, dim=1)
        n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)
        n = n.reshape(1, -1, 1)
        m = s / n
        var = ss / n - m * m
        return (x - m) / torch.sqrt(var.clamp_min(1e-6))


class _BroadcastPadLeft(nn.Module):
    """``u.unsqueeze(0) * v`` — broadcast-pad multiply (left)."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Multiply the front-padded operand into the batched one."""
        return u.unsqueeze(0) * v


class _BroadcastPadRight(nn.Module):
    """``u * v.unsqueeze(0)`` — broadcast-pad multiply (right)."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Multiply the batched operand by a front-padded one."""
        return u * v.unsqueeze(0)


class _BroadcastPadSub(nn.Module):
    """``u.unsqueeze(0) - v`` — broadcast-pad subtract."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Subtract the batched operand from the padded one."""
        return u.unsqueeze(0) - v


class _FullSliceScale(nn.Module):
    """``u[:, :n, :] * v`` — a full-extent slice in an elementwise op."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Scale by a slice covering the whole axis."""
        n = u.shape[1]
        return u[:, :n, :] * v


class _NoopTransposeScale(nn.Module):
    """``u.transpose(-1, -2) * v`` — a semantic-noop transpose."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Scale by the swapped trailing singleton axes."""
        return u.transpose(-1, -2) * v


class _NoopTransposeAdd(nn.Module):
    """``u.transpose(-1, -2) + v`` — the additive no-op transpose."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Add the swapped trailing singleton axes."""
        return u.transpose(-1, -2) + v


class _InertReshapeScale(nn.Module):
    """``u.reshape(1, h, w) * v`` — a broadcast-inert reshape."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Scale by the reshape that inserts a leading singleton."""
        return u.reshape(1, u.shape[0], u.shape[1]) * v


class _SingleChunkScale(nn.Module):
    """``u.chunk(1, -1)[0] * v`` — a one-chunk no-op split."""

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Scale by the sole chunk."""
        (a,) = u.chunk(1, dim=-1)
        return a * v


class _SquaredDistance(nn.Module):
    """``(a-b)*(a-b)`` — the ``mul(t, t)`` self-product spelling."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the per-row squared distance."""
        d = a - b
        return (d * d).sum(dim=-1)


class _ErfFeatures(nn.Module):
    """``erf``/``erfc``/``erfinv`` — error-function features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the three error-function maps."""
        return (
            torch.erf(x)
            + torch.erfc(x)
            + torch.erfinv(x.clamp(-0.9, 0.9))
        )


class _GeluErf(nn.Module):
    """GELU spelled exactly with ``erf`` and ``sqrt``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the exact (non-tanh) GELU."""
        c = torch.sqrt(torch.tensor(2.0, dtype=x.dtype))
        return 0.5 * x * (1.0 + torch.erf(x / c))


class _SoftsignHead(nn.Module):
    """``softsign`` — the bounded rational activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x / (1 + |x|)`` via the dedicated op."""
        return F.softsign(x)


class _Relu6Head(nn.Module):
    """``relu6`` — the dedicated bounded-ReLU op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``min(max(x, 0), 6)`` via the op spelling."""
        return F.relu6(x)


class _SqrtHead(nn.Module):
    """``sqrt`` — an explicit square root in the graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return two square roots."""
        a = x.abs()
        return torch.sqrt(a) + torch.sqrt(a + 1.0)


class _TraceNorm(nn.Module):
    """``trace`` — the diagonal sum of a square matrix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the trace of ``x @ x``."""
        return torch.trace(x @ x)


class _ScatterReduce(nn.Module):
    """``scatter_reduce`` — index accumulation with an explicit reduce."""

    def forward(
        self, x: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Scatter rows into a 4-row base, summing collisions."""
        base = torch.zeros(4, x.shape[1], dtype=x.dtype)
        j = idx.unsqueeze(-1).expand(-1, x.shape[1])
        return base.scatter_reduce(0, j, x, reduce="sum")


class _MinPair(nn.Module):
    """``minimum``/``fmin`` — the two elementwise-min spellings."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the NaN-aware and propagating minima."""
        return torch.minimum(x, y) + torch.fmin(x, y)


class _ClampMaxHead(nn.Module):
    """``clamp_max`` — the one-sided upper clamp spelling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp above and below through the two-sided ops."""
        return x.clamp_max(1.0) + x.clamp_min(-1.0)


class _ComparisonHead(nn.Module):
    """``isclose``/``le``/``ge``/``ne`` — the comparison-op family."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum the four boolean predicates as floats."""
        return (
            torch.isclose(x, y).to(x.dtype)
            + (x <= y).to(x.dtype)
            + (x >= y).to(x.dtype)
            + (x != y).to(x.dtype)
        )


class _AllReduce(nn.Module):
    """``all`` — a global boolean reduction."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask the row unless every element is positive."""
        m = torch.all(x > 0, dim=-1, keepdim=True).to(x.dtype)
        return x * m


class _BroadcastTensors(nn.Module):
    """``broadcast_tensors`` — an explicit multi-tensor broadcast."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Broadcast both operands to a common shape and add."""
        a, b = torch.broadcast_tensors(x.unsqueeze(-1), y)
        return a + b


class _MatrixInverse(nn.Module):
    """``linalg.inv`` — an explicit matrix inverse (preconditioner)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Invert the regularised batch of matrices."""
        d = x.shape[-1]
        reg = x + torch.eye(d, dtype=x.dtype)
        return torch.linalg.inv(reg)


# ---------------------------------------------------------------------------
#  Round 4 — the wrap / mirror spellings the inert guards need
# ---------------------------------------------------------------------------
#
# Round 3 fired the *left* view-identity guards (``mul_unsqueeze_l_id``
# &c.).  The auto-cond set's remaining inert guards are the *wrap*
# family ``f(view(u), v) -> view(f(u, v))`` — true exactly when ``v``
# broadcasts trivially into the view — plus the right-operand mirrors,
# the shared-factor algebra laws, the neg/exp/square grammar finds and
# the scalar-corner annihilators.  Each workload below spells one of
# those regions with a real broadcast / gate / residual idiom; the
# ``_w`` guards need ``v`` scalar or the same shape as the viewed
# operand, so every spelling is a genuine broadcast, not a synthetic
# term.


class _ChannelGateBroadcast(nn.Module):
    """Squeeze-excitation reweighting with a lifted batch axis.

    ``x.unsqueeze(0) * g`` with ``g`` the same shape as the feature
    map — the unsqueeze is broadcast-inert, the ``mul_unsqueeze_l_w``
    wrap region.
    """

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Reweight the lifted feature map by the channel gate."""
        return x.unsqueeze(0) * g


class _LiftedScalarScale(nn.Module):
    """A learned scalar output scale on a lifted feature map.

    ``x.unsqueeze(0) * s`` — the scalar broadcasts into the inserted
    axis, so the lift commutes (``mul_unsqueeze_l_w``).
    """

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Scale the lifted map by the scalar."""
        return x.unsqueeze(0) * s


class _HeadGateBroadcast(nn.Module):
    """Per-head gate broadcast under a leading head-axis lift.

    ``g * x.unsqueeze(0)`` — the gate carries the head shape and the
    leading unsqueeze is broadcast-inert (``mul_unsqueeze_r_w``).
    """

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Gate the head tensor through the leading lift."""
        return g * x.unsqueeze(0)


class _LiftedScalarScaleRight(nn.Module):
    """A scalar scale applied left of a trailing lift.

    ``s * x.unsqueeze(-1)`` — the right-operand mirror of
    :class:`_LiftedScalarScale` (``mul_unsqueeze_r_w``).
    """

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Scale the lifted tensor by the scalar."""
        return s * x.unsqueeze(-1)


class _LiftedScalarCenter(nn.Module):
    """Centre a lifted map by a scalar offset.

    ``x.unsqueeze(0) - s`` — the scalar broadcasts through the lift
    (``sub_unsqueeze_l_w``).
    """

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Subtract the scalar offset from the lifted map."""
        return x.unsqueeze(0) - s


class _ContrastiveCenter(nn.Module):
    """Centre a lifted map by a same-shape running mean.

    ``x.unsqueeze(0) - m`` — the mean broadcasts trivially, the
    ``sub_unsqueeze_l_w`` region.
    """

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Subtract the running mean from the lifted map."""
        return x.unsqueeze(0) - m


class _PairwiseSubLift(nn.Module):
    """Pairwise differences under an explicit batch-axis lift.

    ``x.unsqueeze(0) - y.unsqueeze(0)`` — both operands lifted the
    same way, the ``census:sub_unsqueeze`` region.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the lifted pairwise difference."""
        return x.unsqueeze(0) - y.unsqueeze(0)


class _ScaledChunkProjection(nn.Module):
    """A scalar-scaled half of a doubled projection.

    ``s * proj(x).chunk(2, -1)[0]`` — the scalar broadcasts into the
    chunk, the ``mul_chunk_r_w`` region.
    """

    def __init__(self, d: int) -> None:
        """Build the doubled projection."""
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Scale the first half of the doubled projection."""
        return s * self.proj(x).chunk(2, dim=-1)[0]


class _SingleChunkGate(nn.Module):
    """A gate on the sole chunk of a projection.

    ``g * proj(x).chunk(1, -1)[0]`` — the one-chunk split is a no-op,
    so both the ``mul_chunk_r_id`` (``chunks == 1``) and the
    ``mul_chunk_r_w`` guards fire.
    """

    def __init__(self, d: int) -> None:
        """Build the projection."""
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Gate the sole chunk of the projection."""
        return g * self.proj(x).chunk(1, dim=-1)[0]


class _ChunkHalfScale(nn.Module):
    """A scalar scale on the second half of a split projection.

    ``s * proj(x).chunk(2, -1)[1]`` — the ``mul_chunk_r_w`` region at
    chunk index 1.
    """

    def __init__(self, d: int) -> None:
        """Build the doubled projection."""
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Scale the second half of the doubled projection."""
        return s * self.proj(x).chunk(2, dim=-1)[1]


class _NoopTransposeResidual(nn.Module):
    """A residual add against a no-op-transposed singleton map.

    ``v + x.transpose(-1, -2)`` with ``x`` shaped ``(C, 1, 1)`` — the
    swapped axes are both size one, the ``add_transpose_r_id`` region.
    """

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the swapped singleton map into the residual."""
        return v + x.transpose(-1, -2)


class _ScalarTransposeBias(nn.Module):
    """A scalar bias added to a transposed map.

    ``s + x.transpose(-1, -2)`` — the scalar has rank 0, so the
    transpose commutes out (``add_transpose_r_w``).
    """

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the scalar bias to the transposed map."""
        return s + x.transpose(-1, -2)


class _Rank3TransposeAdd(nn.Module):
    """A rank-3 broadcast added to a no-op transposed map.

    ``v + x.transpose(-1, -2)`` with ``v`` shaped ``(1, C, 1, 1)`` —
    the ``rank(V) <= 1`` disjunct of ``add_transpose_r_w`` never holds,
    but ``axes-noop(V)`` does (both swapped axes are size one).
    """

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast the rank-3 map into the transposed residual."""
        return v + x.transpose(-1, -2)


class _SelectGateSum(nn.Module):
    """Sum two channel slices under a shared select.

    ``x[:, 0] + y[:, 0]`` — the select commutes with the add
    (``select_add``).
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add the first channel of both maps."""
        return x[:, 0] + y[:, 0]


class _SelectGateDiff(nn.Module):
    """Difference of two channel slices under a shared select.

    ``x[:, 0] - y[:, 0]`` — the ``select_sub`` naturality.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Subtract the first channel of the second map."""
        return x[:, 0] - y[:, 0]


class _SharedFactorMixture(nn.Module):
    """A two-expert elementwise mixture on a shared input.

    ``x * w1 + x * w2`` — the shared left factor, ``factor_left``.
    """

    def forward(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        """Sum the two multiplicative gates of the shared input."""
        return x * w1 + x * w2


class _SharedFactorContrast(nn.Module):
    """A two-expert elementwise contrast on a shared input.

    ``x * w1 - x * w2`` — the shared left factor under subtraction,
    ``factor_sub_left``.
    """

    def forward(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        """Subtract the second multiplicative gate of the shared input."""
        return x * w1 - x * w2


class _SharedFactorMixtureRight(nn.Module):
    """A two-expert mixture with the shared factor on the right.

    ``w1 * x + w2 * x`` — ``factor_right``.
    """

    def forward(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        """Sum the two right-hand multiplicative gates."""
        return w1 * x + w2 * x


class _SharedFactorContrastRight(nn.Module):
    """A two-expert contrast with the shared factor on the right.

    ``w1 * x - w2 * x`` — ``factor_sub_right``.
    """

    def forward(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        """Subtract the two right-hand multiplicative gates."""
        return w1 * x - w2 * x


class _QuadraticFeature(nn.Module):
    """A quadratic feature: self-product plus a cross term.

    ``d * d + d * e`` — the ``FALSE_mul_factor`` guard's ``term-eq``
    region (``M0 == M1``) and ``factor_left``.
    """

    def forward(self, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Return the squared plus cross feature."""
        return d * d + d * e


class _DoubleReshapeHead(nn.Module):
    """A head that flattens through two reshapes.

    ``x.reshape(a, b, c).reshape(a, b * c)`` — ``reshape_reshape``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the two trailing axes in two steps."""
        a, b, c = x.shape
        return x.reshape(a, b, c).reshape(a, b * c)


class _NegatedSum(nn.Module):
    """A sum of two negated operands.

    ``(-x) + (-y)`` — ``neg_add``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add the two negated operands."""
        return (-x) + (-y)


class _NegDistributeHead(nn.Module):
    """A negated sum spelled as a distribution.

    ``-(x + y)`` — ``grammar:neg_distribute``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negate the sum of the operands."""
        return -(x + y)


class _SubNegBias(nn.Module):
    """A subtract of a negated bias.

    ``x - (-y)`` — ``sub_neg``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Subtract the negated bias."""
        return x - (-y)


class _NegatedScale(nn.Module):
    """A scale by a negated factor.

    ``(-x) * y`` — ``mul_neg_left``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scale by the negated factor."""
        return (-x) * y


class _ExpProductHead(nn.Module):
    """A product of exponentials.

    ``exp(x) * exp(y)`` — ``exp_add``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the product of the exponentials."""
        return torch.exp(x) * torch.exp(y)


class _ExpSumHead(nn.Module):
    """An exponential of a sum.

    ``exp(x + y)`` — ``grammar:exp_distribute``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the exponential of the summed operands."""
        return torch.exp(x + y)


class _SquareNegHead(nn.Module):
    """A square of a negated operand.

    ``square(-x)`` — ``square_neg``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the square of the negated operand."""
        return torch.square(-x)


class _SquareMulHead(nn.Module):
    """A square of a product.

    ``square(x * y)`` — ``grammar:square_mul``.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the square of the product."""
        return torch.square(x * y)


class _SigmoidNegGate(nn.Module):
    """A sigmoid of a negated logit.

    ``sigmoid(-x)`` — ``grammar:sigmoid_neg`` (``= 1 - sigmoid(x)``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sigmoid of the negated logit."""
        return torch.sigmoid(-x)


class _PowOneHead(nn.Module):
    """A power-one passthrough.

    ``x ** 1`` — ``grammar:pow_one``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the power-one passthrough."""
        return x**1


class _SubAddFactorHead(nn.Module):
    """A doubly-subtracted reduction.

    ``(x - y) - z`` — ``grammar:sub_add_factor``.
    """

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Subtract the two operands in sequence."""
        return (x - y) - z


class _DivAddHead(nn.Module):
    """A divided sum.

    ``(x + y) / z`` — ``grammar:div_add``.
    """

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Divide the summed operands."""
        return (x + y) / z


class _ScalarAnnihilator(nn.Module):
    """A scalar gated to zero.

    ``s * 0`` on a rank-0 operand — ``mul_zero``'s ``rank(A) == 0``
    region (the only binding where the scalar-literal RHS is
    shape-equal).
    """

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Gate the scalar to zero."""
        return s * 0


class _ScalarSelfCancel(nn.Module):
    """A scalar centred against itself.

    ``s - s`` — ``sub_self``.
    """

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return the scalar self-cancellation."""
        return s - s


class _ScalarSelfRatio(nn.Module):
    """A scalar normalised by itself.

    ``s / s`` — ``div_self``.
    """

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return the scalar self-ratio."""
        return s / s


class _ScalarInverseSum(nn.Module):
    """A scalar added to its own negation.

    ``s + (-s)`` — ``add_inv``.
    """

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return the scalar inverse sum."""
        return s + (-s)


# ---------------------------------------------------------------------------
#  The candidate registry — thunks so import constructs nothing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """One intake candidate: a name and a ``(model, input)`` thunk.

    ``kind`` is the registry class (``"torch-native"`` / ``"compound"``);
    ``purpose_built`` is the provenance flag — true for a micro-module
    written to spell one specific pattern (a guard region) rather than
    to represent a real network.  :data:`_PURPOSE_BUILT` is the ledger;
    :func:`real_workloads` is the honest filter over it.
    """

    name: str
    build: Callable[[], tuple[nn.Module, Any]]
    kind: str = "torch-native"
    purpose_built: bool = False


#: The corpus-circularity ledger — the workload names written to spell
#: one specific *pattern* (an auto-cond guard's region), not to stand
#: for a real program.  Round 3's view-identity elementwise spellings
#: exist to fill the guards' empty regions ("Round 3 supplies the
#: no-op spellings directly"); every round-4 workload spells one
#: wrap / mirror / shared-factor / grammar / scalar-corner pattern.
#: Everything else — the torch-native ``nn.*`` modules and the
#: realistic compound assemblies — is what an honest "real-only"
#: measurement keeps.  This table is data, not code: :func:`candidates`
#: reads it to set ``Workload.purpose_built``, and :func:`real_workloads`
#: filters on the flag.
_PURPOSE_BUILT: frozenset[str] = frozenset(
    {
        # --- round 3 — view-identity elementwise spellings ----------
        "BroadcastPadLeft",
        "BroadcastPadRight",
        "BroadcastPadSub",
        "FullSliceScale",
        "NoopTransposeScale",
        "NoopTransposeAdd",
        "InertReshapeScale",
        "SingleChunkScale",
        "SquaredDistance",
        # --- round 4 — unsqueeze-wrap (gated broadcast) -------------
        "ChannelGateBroadcast",
        "LiftedScalarScale",
        "HeadGateBroadcast",
        "LiftedScalarScaleRight",
        "LiftedScalarCenter",
        "ContrastiveCenter",
        "PairwiseSubLift",
        # --- round 4 — chunked-projection mirrors -------------------
        "ScaledChunkProjection",
        "SingleChunkGate",
        "ChunkHalfScale",
        # --- round 4 — transposed-add mirrors -----------------------
        "NoopTransposeResidual",
        "ScalarTransposeBias",
        "Rank3TransposeAdd",
        # --- round 4 — select naturality ----------------------------
        "SelectGateSum",
        "SelectGateDiff",
        # --- round 4 — shared-factor algebra ------------------------
        "SharedFactorMixture",
        "SharedFactorContrast",
        "SharedFactorMixtureRight",
        "SharedFactorContrastRight",
        "QuadraticFeature",
        # --- round 4 — reshape / neg / exp / square grammar ---------
        "DoubleReshapeHead",
        "NegatedSum",
        "NegDistributeHead",
        "SubNegBias",
        "NegatedScale",
        "ExpProductHead",
        "ExpSumHead",
        "SquareNegHead",
        "SquareMulHead",
        "SigmoidNegGate",
        "PowOneHead",
        "SubAddFactorHead",
        "DivAddHead",
        # --- round 4 — scalar-corner annihilators -------------------
        "ScalarAnnihilator",
        "ScalarSelfCancel",
        "ScalarSelfRatio",
        "ScalarInverseSum",
    }
)


def _r(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 CPU example input.

    Called with no shape it yields a rank-0 (scalar) tensor — the
    binding the scalar-corner guards (``rank(A) == 0``) need.
    """
    return torch.randn(tuple(shape), dtype=torch.float64)


def _enc_layer(d: int) -> nn.TransformerEncoderLayer:
    """Return one batch-first encoder layer at the intake size."""
    return nn.TransformerEncoderLayer(
        d, 4, 2 * d, batch_first=True, dropout=0.0
    )


def candidates() -> list[Workload]:
    """Return the intake registry — torch natives + compound models.

    Thunks build a fresh module and example input per call, so the
    registry is cheap to import and every ingestion sees the same
    seeded weights the report records.
    """
    d = 16
    reg = [
        # --- torch-native: attention / transformer family -----------
        Workload(
            "nn.MultiheadAttention",
            lambda: (
                nn.MultiheadAttention(d, 4, batch_first=True),
                (_r(2, 8, d), _r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        Workload(
            "nn.TransformerEncoderLayer",
            lambda: (_enc_layer(d), _r(2, 8, d)),
        ),
        Workload(
            "nn.TransformerEncoder(d=3)",
            lambda: (
                nn.TransformerEncoder(_enc_layer(d), 3),
                _r(2, 8, d),
            ),
        ),
        Workload(
            "nn.TransformerDecoderLayer",
            lambda: (
                nn.TransformerDecoderLayer(
                    d, 4, 2 * d, batch_first=True, dropout=0.0
                ),
                (_r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        # --- torch-native: norms ------------------------------------
        Workload(
            "nn.BatchNorm1d", lambda: (nn.BatchNorm1d(d), _r(4, d))
        ),
        Workload(
            "nn.BatchNorm2d",
            lambda: (nn.BatchNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.GroupNorm", lambda: (nn.GroupNorm(4, 8), _r(1, 8, 8, 8))
        ),
        Workload("nn.LayerNorm", lambda: (nn.LayerNorm(d), _r(4, d))),
        Workload(
            "nn.InstanceNorm2d",
            lambda: (nn.InstanceNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.LocalResponseNorm",
            lambda: (nn.LocalResponseNorm(4), _r(1, 8, 8, 8)),
        ),
        # --- torch-native: conv / pooling / reshaping ---------------
        Workload(
            "nn.ConvTranspose2d",
            lambda: (nn.ConvTranspose2d(8, 8, 3), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Conv3d",
            lambda: (nn.Conv3d(4, 4, 3), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.MaxPool2d", lambda: (nn.MaxPool2d(2), _r(1, 8, 8, 8))
        ),
        Workload(
            "nn.AdaptiveAvgPool2d",
            lambda: (nn.AdaptiveAvgPool2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.PixelShuffle",
            lambda: (nn.PixelShuffle(2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(nearest)",
            lambda: (nn.Upsample(scale_factor=2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(bilinear)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload("nn.Unfold", lambda: (nn.Unfold(2), _r(1, 8, 8, 8))),
        Workload(
            "nn.Fold",
            lambda: (nn.Fold((8, 8), (2, 2)), _r(1, 8 * 4, 49)),
        ),
        Workload(
            "nn.Dropout2d", lambda: (nn.Dropout2d(0.3), _r(1, 8, 8, 8))
        ),
        # --- torch-native: embeddings / pairwise / losses -----------
        Workload(
            "nn.Embedding",
            lambda: (nn.Embedding(32, d), torch.randint(0, 32, (2, 8))),
        ),
        Workload(
            "nn.Bilinear",
            lambda: (nn.Bilinear(d, d, d), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CosineSimilarity",
            lambda: (nn.CosineSimilarity(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CrossEntropyLoss",
            lambda: (
                nn.CrossEntropyLoss(),
                (_r(4, 8), torch.randint(0, 8, (4,))),
            ),
        ),
        Workload(
            "nn.MSELoss",
            lambda: (nn.MSELoss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CTCLoss",
            lambda: (
                nn.CTCLoss(),
                (
                    F.log_softmax(_r(8, 2, 6), -1),
                    torch.randint(1, 6, (2, 3)),
                    torch.tensor([8, 8]),
                    torch.tensor([3, 3]),
                ),
            ),
        ),
        # --- torch-native: recurrent family -------------------------
        Workload(
            "nn.LSTM",
            lambda: (nn.LSTM(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.GRU",
            lambda: (nn.GRU(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.RNNCell",
            lambda: (nn.RNNCell(8, 8), (_r(2, 8), _r(2, 8))),
        ),
        # --- torch-native: assorted pointwise -----------------------
        Workload("nn.PReLU", lambda: (nn.PReLU(), _r(4, d))),
        Workload("nn.Mish", lambda: (nn.Mish(), _r(4, d))),
        Workload("nn.Hardswish", lambda: (nn.Hardswish(), _r(4, d))),
        # --- compound models ----------------------------------------
        Workload(
            "SharedExpertMoE",
            lambda: (_SharedExpertMoE(d, 8, 4), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ViTPatchBlock",
            lambda: (_ViTPatchBlock(8, d, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "TinyCNN",
            lambda: (_TinyCNN(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "Bottleneck",
            lambda: (_Bottleneck(8, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "LossHead",
            lambda: (
                _LossHead(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "ManualNLL",
            lambda: (
                _ManualNLL(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "LogSoftmaxHead",
            lambda: (_LogSoftmaxHead(d, 8), _r(4, d)),
            kind="compound",
        ),
        # --- round 2 — torch-native: activation spellings ----------
        Workload(
            "nn.LeakyReLU", lambda: (nn.LeakyReLU(0.05), _r(4, d))
        ),
        Workload("nn.CELU", lambda: (nn.CELU(), _r(4, d))),
        Workload("nn.SELU", lambda: (nn.SELU(), _r(4, d))),
        Workload("nn.Softplus", lambda: (nn.Softplus(), _r(4, d))),
        Workload("nn.Softsign", lambda: (nn.Softsign(), _r(4, d))),
        Workload(
            "nn.Hardsigmoid", lambda: (nn.Hardsigmoid(), _r(4, d))
        ),
        Workload("nn.Hardtanh", lambda: (nn.Hardtanh(-2, 2), _r(4, d))),
        Workload("nn.ReLU6", lambda: (nn.ReLU6(), _r(4, d))),
        Workload(
            "nn.RReLU", lambda: (nn.RReLU(0.1, 0.3).eval(), _r(4, d))
        ),
        Workload("nn.LogSigmoid", lambda: (nn.LogSigmoid(), _r(4, d))),
        Workload("nn.Softmin", lambda: (nn.Softmin(-1), _r(4, d))),
        Workload("nn.Tanhshrink", lambda: (nn.Tanhshrink(), _r(4, d))),
        Workload("nn.Softshrink", lambda: (nn.Softshrink(), _r(4, d))),
        # --- round 2 — pooling / upsampling / conv gaps ------------
        Workload(
            "nn.AvgPool1d", lambda: (nn.AvgPool1d(2), _r(1, 8, 16))
        ),
        Workload(
            "nn.AvgPool2d", lambda: (nn.AvgPool2d(2), _r(1, 8, 8, 8))
        ),
        Workload(
            "nn.AdaptiveMaxPool2d",
            lambda: (nn.AdaptiveMaxPool2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.MaxPool1d", lambda: (nn.MaxPool1d(2), _r(1, 8, 16))
        ),
        Workload(
            "nn.MaxPool3d",
            lambda: (nn.MaxPool3d(2), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.FractionalMaxPool2d",
            lambda: (
                nn.FractionalMaxPool2d(2, output_size=4),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(bicubic)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="bicubic", align_corners=False
                ),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(linear)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="linear", align_corners=False
                ),
                _r(1, 8, 16),
            ),
        ),
        Workload(
            "nn.Upsample(area)",
            lambda: (
                nn.Upsample(scale_factor=2, mode="area"),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.Upsample(nearest-exact)",
            lambda: (
                nn.Upsample(scale_factor=2, mode="nearest-exact"),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload(
            "nn.ConvTranspose1d",
            lambda: (nn.ConvTranspose1d(8, 8, 3), _r(1, 8, 16)),
        ),
        Workload(
            "nn.ConvTranspose3d",
            lambda: (nn.ConvTranspose3d(4, 4, 3), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.Conv1d(grouped)",
            lambda: (
                nn.Conv1d(8, 8, 3, padding=1, groups=4),
                _r(1, 8, 16),
            ),
        ),
        # --- round 2 — recurrent cells / embeddings / distances ----
        Workload(
            "nn.GRUCell",
            lambda: (nn.GRUCell(8, 8), (_r(2, 8), _r(2, 8))),
        ),
        Workload(
            "nn.LSTMCell",
            lambda: (
                nn.LSTMCell(8, 8),
                (_r(2, 8), (_r(2, 8), _r(2, 8))),
            ),
        ),
        Workload(
            "nn.EmbeddingBag",
            lambda: (
                nn.EmbeddingBag(32, d),
                torch.randint(0, 32, (2, 8)),
            ),
        ),
        Workload(
            "nn.PairwiseDistance",
            lambda: (nn.PairwiseDistance(), (_r(4, d), _r(4, d))),
        ),
        # --- round 2 — losses --------------------------------------
        Workload(
            "nn.L1Loss", lambda: (nn.L1Loss(), (_r(4, d), _r(4, d)))
        ),
        Workload(
            "nn.SmoothL1Loss",
            lambda: (nn.SmoothL1Loss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.HuberLoss",
            lambda: (nn.HuberLoss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.BCELoss",
            lambda: (
                nn.BCELoss(),
                (
                    torch.rand(4, 8).double(),
                    torch.rand(4, 8).double(),
                ),
            ),
        ),
        Workload(
            "nn.BCEWithLogitsLoss",
            lambda: (
                nn.BCEWithLogitsLoss(),
                (_r(4, 8), torch.rand(4, 8).double()),
            ),
        ),
        Workload(
            "nn.KLDivLoss",
            lambda: (
                nn.KLDivLoss(),
                (
                    F.log_softmax(_r(4, 8), -1),
                    F.softmax(_r(4, 8), -1),
                ),
            ),
        ),
        Workload(
            "nn.SoftMarginLoss",
            lambda: (
                nn.SoftMarginLoss(),
                (
                    _r(4, 8),
                    torch.randint(0, 2, (4, 8)).double() * 2 - 1,
                ),
            ),
        ),
        Workload(
            "nn.MarginRankingLoss",
            lambda: (
                nn.MarginRankingLoss(),
                (_r(4), _r(4), torch.ones(4)),
            ),
        ),
        Workload(
            "nn.TripletMarginLoss",
            lambda: (
                nn.TripletMarginLoss(),
                (_r(4, d), _r(4, d), _r(4, d)),
            ),
        ),
        Workload(
            "nn.PoissonNLLLoss",
            lambda: (
                nn.PoissonNLLLoss(),
                (_r(4, 8), torch.rand(4, 8).double() + 0.1),
            ),
        ),
        Workload(
            "nn.GaussianNLLLoss",
            lambda: (
                nn.GaussianNLLLoss(),
                (
                    _r(4, 8),
                    torch.rand(4, 8).double(),
                    torch.rand(4, 8).double() + 0.5,
                ),
            ),
        ),
        Workload(
            "nn.CosineEmbeddingLoss",
            lambda: (
                nn.CosineEmbeddingLoss(),
                (_r(4, d), _r(4, d), torch.ones(4)),
            ),
        ),
        # --- round 2 — padding modes / misc natives ----------------
        Workload(
            "nn.ReflectionPad2d",
            lambda: (nn.ReflectionPad2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.ReplicationPad2d",
            lambda: (nn.ReplicationPad2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.ConstantPad2d",
            lambda: (nn.ConstantPad2d(1, 0.5), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.PixelUnshuffle",
            lambda: (nn.PixelUnshuffle(2), _r(1, 8, 8, 8)),
        ),
        # --- round 2 — compound models ------------------------------
        Workload(
            "CircularPad",
            lambda: (_CircularPad(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "SpatialTransformer",
            lambda: (_SpatialTransformer(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "ShiftedWindowRoll",
            lambda: (_ShiftedWindow(), _r(1, 8, 8, d)),
            kind="compound",
        ),
        Workload(
            "ALiBiAttention",
            lambda: (
                _ALiBiAttention(4),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "GQA-sdpa",
            lambda: (
                _GQAAttention(),
                (_r(1, 8, 8, d), _r(1, 2, 8, d), _r(1, 2, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "EinsumAttention",
            lambda: (
                _EinsumAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "ManualAttention",
            lambda: (
                _ManualAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "VarNorm",
            lambda: (_VarNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "StdNorm",
            lambda: (_StdNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "VarStdHead",
            lambda: (_VarStdHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FakeQuant",
            lambda: (_FakeQuant(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SincKernel",
            lambda: (_SincKernel(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ScalarRsub",
            lambda: (_ScalarRsub(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CumsumScan",
            lambda: (_CumsumScan(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "CumProdGate",
            lambda: (_CumProdGate(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CumMaxMin",
            lambda: (_CumMaxMin(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "IndexAddMoE",
            lambda: (
                _IndexMoE(d),
                (_r(8, d), torch.randint(0, 8, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "ScatterAdd",
            lambda: (
                _ScatterAdd(),
                (_r(8, 8), torch.randint(0, 4, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "MedianPool",
            lambda: (_MedianPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SortSelect",
            lambda: (_SortSelect(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ArgSort",
            lambda: (_ArgSort(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "KthMode",
            lambda: (_KthMode(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TakeAlongGather",
            lambda: (_TakeAlong(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NormalizeHead",
            lambda: (_NormalizeHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SlidingUnfold",
            lambda: (_SlidingUnfold(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "MoveDimStack",
            lambda: (_MoveDimStack(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "TensorSplitHead",
            lambda: (_TensorSplit(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "AngleLog",
            lambda: (_AngleLog(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "Log10Expm1",
            lambda: (_Log10Expm1(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LogBase",
            lambda: (_LogBase(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NanGuard",
            lambda: (_NanGuard(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SanitizeHead",
            lambda: (_SanitizeHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NanStats",
            lambda: (_NanStats(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskLogic",
            lambda: (_MaskLogic(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LogicCombo",
            lambda: (_LogicCombo(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "Bucketize",
            lambda: (_Bucketize(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ChannelShuffle",
            lambda: (_ChannelShuffle(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "FlipConv",
            lambda: (_FlipConv(), _r(1, 4, 16)),
            kind="compound",
        ),
        Workload(
            "GeoMeanPool",
            lambda: (_GeoMeanPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "LerpMix",
            lambda: (_LerpMix(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "AddcmulHead",
            lambda: (_AddcmulHead(), (_r(4, d), _r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "TriuAttention",
            lambda: (
                _TriuAttention(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "TracePenalty",
            lambda: (_TracePenalty(), (_r(8, d), _r(d, d))),
            kind="compound",
        ),
        Workload(
            "OuterPositional",
            lambda: (_OuterPositional(), _r(2, d, d)),
            kind="compound",
        ),
        Workload(
            "RepeatHead",
            lambda: (_RepeatHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "RemainderHead",
            lambda: (_RemainderHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ProdAminPool",
            lambda: (_ProdAminPool(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MinFmax",
            lambda: (_MinFmax(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "BroadcastTo",
            lambda: (_BroadcastTo(), (_r(4, d), _r(4, d, 4))),
            kind="compound",
        ),
        Workload(
            "CountNonzero",
            lambda: (_CountHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "BooleanIndex",
            lambda: (_BooleanIndex(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SignTruncFrac",
            lambda: (_SignTruncFrac(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CeilFloor",
            lambda: (_CeilFloor(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "Heaviside",
            lambda: (_Heaviside(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SdpaBiasParam",
            lambda: (
                _SdpaBiasParam(8),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "Cdist",
            lambda: (_CDist(), (_r(4, 8), _r(6, 8))),
            kind="compound",
        ),
        Workload(
            "EyeInit",
            lambda: (_EyeInit(), _r(d, d)),
            kind="compound",
        ),
        Workload(
            "UnbindHead",
            lambda: (_UnbindHead(), _r(2, 8)),
            kind="compound",
        ),
        Workload(
            "SliceFill",
            lambda: (_SliceFill(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FillWhole",
            lambda: (_FillWhole(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "FillSelect",
            lambda: (_FillSelect(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ZeroInit",
            lambda: (_ZeroInit(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskedFillInplace",
            lambda: (_MaskedFillInplace(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "MaskedFillTensor",
            lambda: (
                _MaskedFillTensor(),
                (_r(4, d), torch.randn((), dtype=torch.float64)),
            ),
            kind="compound",
        ),
        Workload(
            "MaskedFillView",
            lambda: (_MaskedFillView(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SelectWrite",
            lambda: (_SelectWrite(), (_r(4, d), _r(d))),
            kind="compound",
        ),
        Workload(
            "IndexPut",
            lambda: (_IndexPut(), (_r(4, d), _r(2, d))),
            kind="compound",
        ),
        Workload(
            "NarrowHead",
            lambda: (_NarrowHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "NewCreators",
            lambda: (_NewCreators(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TypeAs",
            lambda: (_TypeAs(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpandAs",
            lambda: (_ExpandAs(), (_r(4, d), _r(4, d, 4))),
            kind="compound",
        ),
        Workload(
            "IsNegPos",
            lambda: (_IsNegPos(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "TrigHead",
            lambda: (_TrigHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "HyperbolicHead",
            lambda: (_HyperbolicHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "CoshAcosh",
            lambda: (_CoshAcosh(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "DigammaLn",
            lambda: (_DigammaLn(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "XlogyKL",
            lambda: (_XlogyKL(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "RegluMLP",
            lambda: (_RegluMlp(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SwiGLU",
            lambda: (_SwiGLU(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "OneHotEmbed",
            lambda: (_OneHotEmbed(), torch.randint(0, 16, (4, 8))),
            kind="compound",
        ),
        Workload(
            "BitwiseHead",
            lambda: (
                _BitwiseHead(),
                (
                    torch.randint(0, 8, (4, 8)),
                    torch.randint(0, 8, (4, 8)),
                ),
            ),
            kind="compound",
        ),
        Workload(
            "RepeatInterleave",
            lambda: (_RepeatInterleave(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "DepthwiseConv2d",
            lambda: (_DepthwiseConv2d(), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        # --- round 3 — torch-native norm gaps ----------------------
        Workload("nn.RMSNorm", lambda: (nn.RMSNorm(d), _r(4, d))),
        Workload(
            "nn.BatchNorm3d",
            lambda: (nn.BatchNorm3d(4), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.TransformerEncoderLayer(norm_first)",
            lambda: (
                nn.TransformerEncoderLayer(
                    d,
                    4,
                    2 * d,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                ),
                _r(2, 8, d),
            ),
        ),
        # --- round 3 — transformer variants ------------------------
        Workload(
            "RotaryEmbedding",
            lambda: (_RotaryEmbedding(), _r(1, 4, 8, d)),
            kind="compound",
        ),
        Workload(
            "RoPEAttention",
            lambda: (_RoPEAttention(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "MQAAttention",
            lambda: (_MQAAttention(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "RelativePositionBias",
            lambda: (_RelativePositionBias(4, 8), _r(2, 4, 8, 8)),
            kind="compound",
        ),
        Workload(
            "MoEGate",
            lambda: (_MoEGate(d, 4), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "PreNormBlock",
            lambda: (_PreNormBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "PostNormBlock",
            lambda: (_PostNormBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "SwiGLUBlock",
            lambda: (_SwiGLUBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "CrossAttentionBlock",
            lambda: (
                _CrossAttentionBlock(d, 4),
                (_r(2, 8, d), _r(2, 6, d)),
            ),
            kind="compound",
        ),
        Workload(
            "SlidingWindowMask",
            lambda: (
                _SlidingWindowMask(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "MultiLatentAttention",
            lambda: (_MultiLatentAttention(d, 4, 8), _r(2, 8, d)),
            kind="compound",
        ),
        # --- round 3 — normalisation variants ----------------------
        Workload(
            "ManualRMSNorm",
            lambda: (_ManualRMSNorm(d), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "QKNorm",
            lambda: (
                _QKNorm(),
                (_r(1, 4, 8, d), _r(1, 4, 8, d), _r(1, 4, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "DeepNorm",
            lambda: (_DeepNorm(d), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "CenterNorm",
            lambda: (_CenterNorm(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ScaleNorm",
            lambda: (_ScaleNorm(d), _r(4, d)),
            kind="compound",
        ),
        # --- round 3 — conv/attention hybrids ----------------------
        Workload(
            "ConvNeXtBlock",
            lambda: (_ConvNeXtBlock(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "MBConvBlock",
            lambda: (_MBConvBlock(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "GRN",
            lambda: (_GRN(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "ConformerBlock",
            lambda: (_ConformerBlock(d, 4), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "AxialAttention",
            lambda: (_AxialAttention(8, 2), _r(1, 8, 4, 4)),
            kind="compound",
        ),
        Workload(
            "PatchMerging",
            lambda: (_PatchMerging(d), _r(1, 4, d)),
            kind="compound",
        ),
        # --- round 3 — recurrent / scan-heavy ----------------------
        Workload(
            "SelectiveScan",
            lambda: (_SelectiveScan(d), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "LinearAttention",
            lambda: (
                _LinearAttention(),
                (_r(2, 8, d), _r(2, 8, d), _r(2, 8, d)),
            ),
            kind="compound",
        ),
        Workload(
            "CumulativeGateRNN",
            lambda: (_CumulativeGateRNN(), _r(2, 8, d)),
            kind="compound",
        ),
        Workload(
            "RunningNorm",
            lambda: (_RunningNorm(), _r(2, 8, d)),
            kind="compound",
        ),
        # --- round 3 — view-identity elementwise spellings ---------
        Workload(
            "BroadcastPadLeft",
            lambda: (_BroadcastPadLeft(), (_r(4, 8), _r(1, 4, 8))),
            kind="compound",
        ),
        Workload(
            "BroadcastPadRight",
            lambda: (_BroadcastPadRight(), (_r(1, 4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "BroadcastPadSub",
            lambda: (_BroadcastPadSub(), (_r(4, 8), _r(1, 4, 8))),
            kind="compound",
        ),
        Workload(
            "FullSliceScale",
            lambda: (_FullSliceScale(), (_r(2, 4, 8), _r(2, 4, 8))),
            kind="compound",
        ),
        Workload(
            "NoopTransposeScale",
            lambda: (_NoopTransposeScale(), (_r(8, 1, 1), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "NoopTransposeAdd",
            lambda: (_NoopTransposeAdd(), (_r(8, 1, 1), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "InertReshapeScale",
            lambda: (_InertReshapeScale(), (_r(8, 8), _r(1, 8, 8))),
            kind="compound",
        ),
        Workload(
            "SingleChunkScale",
            lambda: (_SingleChunkScale(), (_r(2, 8), _r(2, 8))),
            kind="compound",
        ),
        Workload(
            "SquaredDistance",
            lambda: (_SquaredDistance(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 3 — ops absent from the census ------------------
        Workload(
            "ErfFeatures",
            lambda: (_ErfFeatures(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "GeluErf",
            lambda: (_GeluErf(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SoftsignHead",
            lambda: (_SoftsignHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "Relu6Head",
            lambda: (_Relu6Head(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "SqrtHead",
            lambda: (_SqrtHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "TraceNorm",
            lambda: (_TraceNorm(), _r(8, 8)),
            kind="compound",
        ),
        Workload(
            "ScatterReduce",
            lambda: (
                _ScatterReduce(),
                (_r(8, 8), torch.randint(0, 4, (8,))),
            ),
            kind="compound",
        ),
        Workload(
            "MinPair",
            lambda: (_MinPair(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "ClampMaxHead",
            lambda: (_ClampMaxHead(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ComparisonHead",
            lambda: (_ComparisonHead(), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "AllReduce",
            lambda: (_AllReduce(), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "BroadcastTensors",
            lambda: (_BroadcastTensors(), (_r(4, 8), _r(4, 8, 3))),
            kind="compound",
        ),
        Workload(
            "MatrixInverse",
            lambda: (_MatrixInverse(), _r(2, 8, 8)),
            kind="compound",
        ),
        # --- round 4 — the unsqueeze-wrap (gated broadcast) family --
        Workload(
            "ChannelGateBroadcast",
            lambda: (
                _ChannelGateBroadcast(),
                (_r(8, 8, 8), _r(8, 8, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "LiftedScalarScale",
            lambda: (_LiftedScalarScale(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "HeadGateBroadcast",
            lambda: (_HeadGateBroadcast(), (_r(8, 8, 8), _r(8, 8, 8))),
            kind="compound",
        ),
        Workload(
            "LiftedScalarScaleRight",
            lambda: (_LiftedScalarScaleRight(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "LiftedScalarCenter",
            lambda: (_LiftedScalarCenter(), (_r(8, 8, 8), _r())),
            kind="compound",
        ),
        Workload(
            "ContrastiveCenter",
            lambda: (_ContrastiveCenter(), (_r(8, 8, 8), _r(8, 8, 8))),
            kind="compound",
        ),
        Workload(
            "PairwiseSubLift",
            lambda: (_PairwiseSubLift(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — chunked-projection mirrors -------------------
        Workload(
            "ScaledChunkProjection",
            lambda: (_ScaledChunkProjection(d), (_r(4, d), _r())),
            kind="compound",
        ),
        Workload(
            "SingleChunkGate",
            lambda: (_SingleChunkGate(d), (_r(4, d), _r(4, d))),
            kind="compound",
        ),
        Workload(
            "ChunkHalfScale",
            lambda: (_ChunkHalfScale(d), (_r(4, d), _r())),
            kind="compound",
        ),
        # --- round 4 — transposed-add mirrors -----------------------
        Workload(
            "NoopTransposeResidual",
            lambda: (
                _NoopTransposeResidual(),
                (_r(8, 1, 1), _r(8, 1, 1)),
            ),
            kind="compound",
        ),
        Workload(
            "ScalarTransposeBias",
            lambda: (_ScalarTransposeBias(), (_r(), _r(8, 1, 1))),
            kind="compound",
        ),
        Workload(
            "Rank3TransposeAdd",
            lambda: (
                _Rank3TransposeAdd(),
                (_r(1, 8, 1, 1), _r(8, 1, 1)),
            ),
            kind="compound",
        ),
        # --- round 4 — select naturality ----------------------------
        Workload(
            "SelectGateSum",
            lambda: (_SelectGateSum(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SelectGateDiff",
            lambda: (_SelectGateDiff(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — shared-factor algebra ------------------------
        Workload(
            "SharedFactorMixture",
            lambda: (
                _SharedFactorMixture(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorContrast",
            lambda: (
                _SharedFactorContrast(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorMixtureRight",
            lambda: (
                _SharedFactorMixtureRight(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "SharedFactorContrastRight",
            lambda: (
                _SharedFactorContrastRight(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "QuadraticFeature",
            lambda: (_QuadraticFeature(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — reshape / neg / exp / square grammar ---------
        Workload(
            "DoubleReshapeHead",
            lambda: (_DoubleReshapeHead(), _r(2, 3, 8)),
            kind="compound",
        ),
        Workload(
            "NegatedSum",
            lambda: (_NegatedSum(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "NegDistributeHead",
            lambda: (_NegDistributeHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SubNegBias",
            lambda: (_SubNegBias(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "NegatedScale",
            lambda: (_NegatedScale(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpProductHead",
            lambda: (_ExpProductHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "ExpSumHead",
            lambda: (_ExpSumHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SquareNegHead",
            lambda: (_SquareNegHead(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "SquareMulHead",
            lambda: (_SquareMulHead(), (_r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        Workload(
            "SigmoidNegGate",
            lambda: (_SigmoidNegGate(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "PowOneHead",
            lambda: (_PowOneHead(), _r(4, 8)),
            kind="compound",
        ),
        Workload(
            "SubAddFactorHead",
            lambda: (
                _SubAddFactorHead(),
                (_r(4, 8), _r(4, 8), _r(4, 8)),
            ),
            kind="compound",
        ),
        Workload(
            "DivAddHead",
            lambda: (_DivAddHead(), (_r(4, 8), _r(4, 8), _r(4, 8))),
            kind="compound",
        ),
        # --- round 4 — scalar-corner annihilators -------------------
        Workload(
            "ScalarAnnihilator",
            lambda: (_ScalarAnnihilator(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarSelfCancel",
            lambda: (_ScalarSelfCancel(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarSelfRatio",
            lambda: (_ScalarSelfRatio(), _r()),
            kind="compound",
        ),
        Workload(
            "ScalarInverseSum",
            lambda: (_ScalarInverseSum(), _r()),
            kind="compound",
        ),
    ]
    # Provenance is data: the ledger above marks the micro-modules
    # written to spell a pattern, so the honest "real-only" filter is
    # one predicate over the registry.
    return [
        replace(w, purpose_built=True)
        if w.name in _PURPOSE_BUILT
        else w
        for w in reg
    ]


def real_workloads(
    cands: list[Workload] | None = None,
) -> list[Workload]:
    """Return the registry minus the purpose-built spelling workloads.

    The honest subset for a full-vs-real corpus comparison: workloads
    that represent a real network — a torch-native ``nn.*`` module or
    a realistic compound assembly — not a micro-module written to
    spell one pattern.  ``purpose_built`` is the ledger
    (:data:`_PURPOSE_BUILT`); the filter excludes exactly those names.
    """
    src = candidates() if cands is None else cands
    return [w for w in src if not w.purpose_built]


# ---------------------------------------------------------------------------
#  Ingestion — export, classify, verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """A candidate the export boundary refused, with the reason."""

    name: str
    stage: str
    error: str


def _ops_of(term: Any) -> set[str]:
    """Return the distinct op names inside *term*."""
    return {s.op for s in _iter_subterms(term) if isinstance(s, Op)}


def _n_nodes(term: Any) -> int:
    """Return the op-node count of *term*."""
    return sum(1 for s in _iter_subterms(term) if isinstance(s, Op))


def _verify(
    model: nn.Module,
    ir: Any,
    tensors: dict,
    feed: tuple,
    sink: TorchSink,
) -> str:
    """Lower the exported IR and verify it against the module.

    Returns ``"pass"``, ``"FAIL"`` or an error tag — the same
    ``sink.lower`` + ``sink.verify`` path the corpus tests run.
    """
    try:
        lowered = sink.lower(
            _ir_of(ir.root, tuple(ir.inputs)), dict(tensors)
        )
        vr = sink.verify(model, lowered, feed, rtol=_RTOL)
    except Exception as e:
        return f"error:{type(e).__name__}: {e}"
    return "pass" if vr.passed else f"FAIL max_rel={vr.max_rel:.2e}"


def _ingest_one(
    cand: Workload, sink: TorchSink, supported: frozenset
) -> tuple[TermCase | None, dict | None, Rejection | None]:
    """Run one candidate through the boundary; classify the result."""
    try:
        model, x = cand.build()
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "build", f"{type(e).__name__}: {e}"),
        )
    feed = x if isinstance(x, tuple) else (x,)
    try:
        ir, tensors = export_to_ir(model.eval().double(), feed)
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "export", f"{type(e).__name__}: {e}"),
        )
    ops = _ops_of(ir.root)
    missing = sorted(ops - supported)
    rec: dict[str, Any] = {
        "name": f"intake:{cand.name}",
        "builder": cand.kind,
        "n_op_nodes": _n_nodes(ir.root),
        "ops": sorted(ops),
        "unsupported_ops": missing,
        "term": term_to_data(ir.root),
        "inputs": [
            {"name": v.name, "shape": list(v.typ.shape)}
            for v in ir.inputs
        ],
    }
    if missing:
        rec["status"] = "census-only"
        rec["verify"] = "skipped"
        rec["verify_note"] = f"unbound ops: {missing}"
    else:
        v = _verify(model, ir, tensors, feed, sink)
        rec["verify"] = v
        rec["status"] = "ingested" if v == "pass" else "verify-failed"
        rec["verify_note"] = "" if v == "pass" else v
    case = TermCase(
        source="intake",
        name=f"intake:{cand.name}",
        term=ir.root,
        inputs=tuple(ir.inputs),
        feed=tuple(feed),
        param_vals=dict(tensors),
    )
    return case, rec, None


def ingest(
    cands: list[Workload] | None = None,
    sink: TorchSink | None = None,
) -> tuple[list[TermCase], list[dict], list[Rejection]]:
    """Run every candidate through the export boundary.

    Returns ``(cases, records, rejections)``: the ``TermCase``s of
    every term that exported (any status), the per-workload metadata
    records the side-file persists, and the hard rejections.
    """
    sink = sink or TorchSink()
    supported = sink.supported_ops
    cases: list[TermCase] = []
    records: list[dict] = []
    rejections: list[Rejection] = []
    for cand in cands if cands is not None else candidates():
        case, rec, rej = _ingest_one(cand, sink, supported)
        if rej is not None:
            rejections.append(rej)
            continue
        # _ingest_one yields a rejection xor a (case, rec) pair.
        cases.append(cast("TermCase", case))
        records.append(cast("dict", rec))
    return cases, records, rejections


# ---------------------------------------------------------------------------
#  Persistence — the side-file census
# ---------------------------------------------------------------------------


def write_intake(
    records: list[dict],
    rejections: list[Rejection],
    tensors_of: dict[str, dict],
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> None:
    """Write the census contribution and the tensor blob.

    ``records`` carry the serialized terms; ``tensors_of`` maps each
    workload name to ``{"feed": [...], "params": {...}}`` — kept in a
    ``torch.save`` blob so the JSON stays diffable.
    """
    payload = {
        "format": 1,
        "tool": "law_intake",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "workloads": records,
        "rejections": [asdict(r) for r in rejections],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(tensors_of, tensors_path)


def load_records(path: Path = _DEFAULT_JSON) -> list[dict]:
    """Return the persisted workload records ([] when absent)."""
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("workloads", [])


def load_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> list[TermCase]:
    """Rebuild intake ``TermCase``s from the side-file census.

    Terms decode through ``term_from_data``; feeds/params ride the
    companion ``torch.save`` blob (absent blob → empty feed — the
    census only reads ``.term`` anyway, and :func:`probe_cases`
    requires a feed).
    """
    tensor_blob: dict = {}
    if tensors_path.exists():
        tensor_blob = torch.load(tensors_path, weights_only=True)
    cases: list[TermCase] = []
    for rec in load_records(path):
        inputs = tuple(
            Var(i["name"], TensorType(tuple(i["shape"])))
            for i in rec["inputs"]
        )
        blob = tensor_blob.get(rec["name"], {})
        cases.append(
            TermCase(
                source="intake",
                name=rec["name"],
                term=term_from_data(rec["term"]),
                inputs=inputs,
                feed=tuple(blob.get("feed", ())),
                param_vals=dict(blob.get("params", {})),
            )
        )
    return cases


def probe_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
    max_nodes: int = _PROBE_MAX_NODES,
) -> list[TermCase]:
    """Return the intake cases eligible for the firing/reach probe.

    Eligibility is the intake contract: status ``"ingested"`` (all
    ops bound, lowered module verified), a persisted feed, and under
    the node cap that keeps the probe's saturation cost comparable
    to the corpus's own models.
    """
    status = {r["name"]: r["status"] for r in load_records(path)}
    nodes = {r["name"]: r["n_op_nodes"] for r in load_records(path)}
    return [
        c
        for c in load_cases(path, tensors_path)
        if status.get(c.name) == "ingested"
        and c.feed
        and nodes.get(c.name, 0) <= max_nodes
    ]


def _tensors_of(cases: list[TermCase]) -> dict[str, dict]:
    """Collect each case's feed/param tensors for ``torch.save``."""
    return {
        c.name: {"feed": list(c.feed), "params": dict(c.param_vals)}
        for c in cases
    }


# ---------------------------------------------------------------------------
#  Census delta — what the intake adds that the corpus did not have
# ---------------------------------------------------------------------------


def census_delta(base: list[TermCase], intake: list[TermCase]) -> dict:
    """Diff the intake terms' census keys against the base corpus.

    Returns the new op-tuples, new shapes and previously-absent ops,
    plus a per-workload novelty table for the report.
    """
    from catopt_discovery.census import (
        CorpusTerm,
        op_tuple_census,
        shape_census,
    )

    base_ct = [CorpusTerm(c.source, c.name, c.term) for c in base]
    in_ct = [CorpusTerm(c.source, c.name, c.term) for c in intake]
    base_op, _ = op_tuple_census(base_ct)
    base_sh, _ = shape_census(base_ct)
    in_op, _ = op_tuple_census(in_ct)
    in_sh, _ = shape_census(in_ct)
    base_ops = set()
    for c in base:
        base_ops |= _ops_of(c.term)
    in_ops = set()
    for c in intake:
        in_ops |= _ops_of(c.term)
    per_case = {}
    for c in intake:
        ct = CorpusTerm(c.source, c.name, c.term)
        op_c, _ = op_tuple_census([ct])
        sh_c, _ = shape_census([ct])
        per_case[c.name] = (
            sorted(set(op_c) - set(base_op)),
            len(set(sh_c) - set(base_sh)),
        )
    return {
        "n_base_tuples": len(base_op),
        "n_base_shapes": len(base_sh),
        "new_op_tuples": sorted(set(in_op) - set(base_op)),
        "new_shapes": len(set(in_sh) - set(base_sh)),
        "new_ops": sorted(in_ops - base_ops),
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
#  Pipeline delta — mirror law_workload_gen's harness verbatim
# ---------------------------------------------------------------------------


def _run_delta(
    base_cases: list[TermCase],
    probe_base: list[TermCase],
    intake_cases: list[TermCase],
    probe_intake: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> dict:
    """Run the pipeline on the corpus, then corpus+intake.

    Reuses ``catopt_discovery.workload_gen._run_pipeline`` — the same census ->
    propose -> measure -> rank mirror that tool validated — with the
    intake cases as the corpus extension and the probe-eligible
    subset as the firing/reach additions.
    """
    from catopt_discovery import workload_gen as lwg

    base = lwg._run_pipeline(base_cases, probe_base, vocab, holdout)
    big = lwg._run_pipeline(
        [*base_cases, *intake_cases],
        [*probe_base, *probe_intake],
        vocab,
        holdout,
    )
    return {"baseline": base, "enlarged": big}


def _intake_fires(ev: Any) -> int:
    """Count an evidence row's firings on intake cases."""
    return sum(1 for c in ev.fire_cases if c.startswith("intake:"))


def _delta_table(res: dict) -> str:
    """Render baseline-vs-enlarged pipeline comparison."""
    b, e = res["baseline"], res["enlarged"]
    b_names = {ev.proposal.name for ev in b["ranked"]}
    e_names = {ev.proposal.name for ev in e["ranked"]}
    b_rank = {ev.proposal.name: ev for ev in b["ranked"]}
    e_rank = {ev.proposal.name: ev for ev in e["ranked"]}
    lines = [
        f"corpus: {b['n_terms']} -> {e['n_terms']} terms, "
        f"{b['n_tuples']} -> {e['n_tuples']} op-tuples",
        f"proposals: {len(b_names)} -> {len(e_names)}",
    ]
    new_props = sorted(e_names - b_names)
    lines.append(f"new proposals ({len(new_props)}):")
    for name in new_props:
        ev = e_rank[name]
        lines.append(
            f"  {name:<28} [{ev.proposal.family}] "
            f"match={ev.matches} fires={ev.fires} "
            f"(intake={_intake_fires(ev)}) paid={ev.paid} "
            f"ship={'SHIP' if ev.shippable else ev.no_ship_reason}"
        )
    if not new_props:
        lines.append("  (none)")
    lines.append("-- firing/paid deltas on shared proposals --")
    shown = 0
    for name in sorted(b_names & e_names):
        be, ee = b_rank[name], e_rank[name]
        extra = _intake_fires(ee)
        if not (
            extra or ee.fires != be.fires or ee.matches != be.matches
        ):
            continue
        lines.append(
            f"  {name:<28} match {be.matches}->{ee.matches} "
            f"fires {be.fires}->{ee.fires} (intake={extra}) "
            f"paid {be.paid}->{ee.paid} "
            f"ship: {'Y' if ee.shippable else ee.no_ship_reason}"
        )
        shown += 1
    if not shown:
        lines.append("  (no shared proposal changed)")
    b_ship = [ev.proposal.name for ev in b["ranked"] if ev.shippable]
    e_ship = [ev.proposal.name for ev in e["ranked"] if ev.shippable]
    lines.append(
        f"shippable: {len(b_ship)} -> {len(e_ship)}"
        + (
            f"  new: {[s for s in e_ship if s not in b_ship]}"
            if e_ship
            else ""
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Reporting + driver
# ---------------------------------------------------------------------------


def _status_table(records: list[dict], delta: dict) -> str:
    """Render the per-workload intake table."""
    head = (
        f"{'workload':<36} {'kind':<12} {'nodes':>6} {'status':<13} "
        f"{'+tup':>5} {'+shp':>5}  missing ops"
    )
    lines = [head, "-" * len(head)]
    for rec in records:
        nt, ns = delta["per_case"].get(rec["name"], ([], 0))
        lines.append(
            f"{rec['name']:<36} {rec['builder']:<12} "
            f"{rec['n_op_nodes']:>6} {rec['status']:<13} "
            f"{len(nt):>5} {ns:>5}  "
            f"{', '.join(rec['unsupported_ops']) or '-'}"
        )
    return "\n".join(lines)


def _rejection_table(rejections: list[Rejection]) -> str:
    """Render the export-boundary rejection backlog."""
    lines = [f"{'workload':<36} {'stage':<8} error"]
    lines.append("-" * 80)
    for r in rejections:
        lines.append(f"{r.name:<36} {r.stage:<8} {r.error}")
    return "\n".join(lines)


def _gap_table(records: list[dict]) -> str:
    """Aggregate unbound ops -> the workloads that need them."""
    gaps: dict[str, list[str]] = {}
    for rec in records:
        for op in rec["unsupported_ops"]:
            gaps.setdefault(op, []).append(rec["name"])
    if not gaps:
        return "  (no binding gaps)"
    lines = [f"{'missing op':<26} {'#':>3}  needed by"]
    lines.append("-" * 80)
    for op, names in sorted(gaps.items()):
        lines.append(f"{op:<26} {len(names):>3}  {', '.join(names)}")
    return "\n".join(lines)


def _eligible(cases: list[TermCase], records: list[dict]) -> list:
    """Return the in-memory probe-eligible cases (ingested + cap)."""
    meta = {r["name"]: (r["status"], r["n_op_nodes"]) for r in records}
    return [
        c
        for c in cases
        if meta.get(c.name, ("", 0))[0] == "ingested"
        and meta[c.name][1] <= _PROBE_MAX_NODES
    ]


def _report_pipeline(
    cases: list[TermCase],
    records: list[dict],
    args: argparse.Namespace,
    payload: dict,
) -> None:
    """Run the baseline-vs-enlarged pipeline delta and report it."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    probe = _eligible(cases, records)
    print(  # stdout-compat
        f"   probe-eligible intake cases: {len(probe)}"
    )
    res = _run_delta(
        [*bench, *models],
        models,
        cases,
        probe,
        args.vocab,
        args.holdout,
    )
    print(_delta_table(res))  # stdout-compat
    e = res["enlarged"]
    payload["pipeline"] = {
        "baseline_terms": res["baseline"]["n_terms"],
        "enlarged_terms": e["n_terms"],
        "baseline_tuples": res["baseline"]["n_tuples"],
        "enlarged_tuples": e["n_tuples"],
        "new_proposals": sorted(
            {ev.proposal.name for ev in e["ranked"]}
            - {ev.proposal.name for ev in res["baseline"]["ranked"]}
        ),
        "shippable": [
            ev.proposal.name for ev in e["ranked"] if ev.shippable
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Ingest candidates, measure the census delta, write the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_JSON,
        help="side-file census (default tools/intake_corpus.json)",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=_DEFAULT_TENSORS,
        help="feed/param tensor blob (tools/intake_tensors.pt)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run the measurement without writing the side files",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vocab", choices=("hand", "derived"), default="derived"
    )
    parser.add_argument("--holdout", help="pipeline holdout rules")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="ingest only real workloads (drop the purpose-built spellings)",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    print(  # stdout-compat
        "== law_intake — real workloads into the census =="
    )
    sink = TorchSink()
    cands = real_workloads() if args.real_only else None
    if cands is not None:
        print(  # stdout-compat
            f"   --real-only: {len(cands)} real workloads "
            f"(of {len(candidates())}; "
            f"{len(_PURPOSE_BUILT)} purpose-built dropped)"
        )
    cases, records, rejections = ingest(cands=cands, sink=sink)
    n_in = sum(1 for r in records if r["status"] == "ingested")
    n_co = sum(1 for r in records if r["status"] == "census-only")
    n_vf = sum(1 for r in records if r["status"] == "verify-failed")
    print(  # stdout-compat
        f"   {len(records)} exported: {n_in} ingested, "
        f"{n_co} census-only, {n_vf} verify-failed; "
        f"{len(rejections)} rejected"
    )

    bench, _be = _bench_cases()
    models, _me = model_cases()
    delta = census_delta([*bench, *models], cases)
    print()  # stdout-compat
    print("-- intake table --")  # stdout-compat
    print(_status_table(records, delta))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        f"-- census delta (intake vs {len(bench)} bench + "
        f"{len(models)} models) --"
    )
    print(  # stdout-compat
        f"   {delta['n_base_tuples']} -> "
        f"{delta['n_base_tuples'] + len(delta['new_op_tuples'])} "
        f"op-tuples, +{delta['new_shapes']} shapes, "
        f"{len(delta['new_ops'])} new ops: {delta['new_ops']}"
    )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- binding-gap backlog (exported, unbound) --"
    )
    print(_gap_table(records))  # stdout-compat
    print()  # stdout-compat
    print("-- export rejections --")  # stdout-compat
    print(_rejection_table(rejections))  # stdout-compat
    print()  # stdout-compat

    payload: dict[str, Any] = {
        "n_exported": len(records),
        "n_ingested": n_in,
        "n_census_only": n_co,
        "n_verify_failed": n_vf,
        "rejections": [asdict(r) for r in rejections],
        "census_delta": {
            "new_op_tuples": [
                f"{k[0]}({', '.join(k[1])})"
                for k in delta["new_op_tuples"]
            ],
            "new_shapes": delta["new_shapes"],
            "new_ops": delta["new_ops"],
        },
    }

    if not args.no_write:
        write_intake(
            records,
            rejections,
            _tensors_of(cases),
            args.out,
            args.tensors,
        )
        print(f"wrote {args.out} + {args.tensors}")  # stdout-compat
    else:
        print("(--no-write: side files not written)")  # stdout-compat

    if not args.skip_pipeline:
        print()  # stdout-compat
        print(  # stdout-compat
            "-- pipeline delta (baseline vs corpus+intake) --"
        )
        _report_pipeline(cases, records, args, payload)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
