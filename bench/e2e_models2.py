"""End-to-end bench #2 — small NON-decoder models, in-repo replicas.

``e2e_model`` covered a 3.3M llama-toy decoder; ``e2e_llm`` the
0.4B-class decoder with prefill + KV-cache decode.  This bench widens
the E2E story to the other shapes a deployment actually serves —
an embedder, a vision transformer, a conv stack, a mid-size llama
point, and a routed-MoE stack — as in-repo replicas of real
open-source architecture *shapes*: faithful structure, random
fixed-seed weights, NO checkpoint/network downloads.  Everything
fits the 4 GB RTX 2050 budget (fp16 where the size needs it).

Models (``--models`` subset):

* ``minilm``   — bge-small / MiniLM-L12-H384-shaped text embedder:
  30522 tok + learned pos embeddings → LN → 12 pre-LN encoder blocks
  (d=384, 12 heads, GELU FFN 1536, additive key-padding mask) →
  masked mean-pool → L2-normalized (B, 384) embedding.  ≈33 M params
  fp32.  Mechanism: q/k/v pairing into one GEMM per block (bias-free
  projections — the pre-fusion deployment form; pairing only sees
  ``linear(x, W)``).
* ``vit_ti``   — deit-tiny-shaped ViT: patch16 conv → 196 tokens +
  cls + learned pos-emb → 12 pre-LN blocks (d=192, 3 heads, FFN 768)
  → LN → cls head.  ≈5.7 M params fp32.  Same pairing mechanism,
  smaller GEMMs (a pairing-margin datapoint).
* ``conv_ti``  — ConvNeXt-tiny block stack: stem 4×4/s4 → stages
  [96,192,384,768] depths [1,1,3,1] of dwconv7×7 → NHWC LN →
  pw 4C → GELU → pw C → layer-scale residual blocks, LN+conv2/s2
  downsamples → global pool → LN → head.  ≈10 M fp32.  Mechanism
  probe: dwconv is unp pairable (grouped), LN-gain folds are
  layer_norm-atomic — expect parity; reported honestly either way.
* ``llama_mid``— ``e2e_model.TinyLlama`` (GQA + rotary + SwiGLU) at
  d=512/L=8/heads 8/kv 4/hidden 1408/vocab 16384 ≈40 M, fp16 — the
  mid-size point between e2e_model's 3.3 M and e2e_llm's ~0.4 B.
* ``moe_stack``— Mixtral-lite block stack: 4 blocks of rms→causal MHA
  then rms→routed MoE FFN (8 SwiGLU experts, top-2 router, gates
  renormalized; ALL experts evaluated and gate-summed — the
  dense-eval eager form deployment code starts from).  ≈14 M fp32.
  The question the per-block ``moe_sum`` soup-fold win raises: does
  anything survive E2E?  Here the honest expectation is partial —
  the 8 same-input w1 (resp. w3) projections can pair into grouped
  GEMMs, while Σ_e g_e·(x@W_e) must NOT fold (gates are
  input-dependent); the record shows which rules actually fired.

Variants per cell: ``eager`` | ``inductor`` (torch.compile,
SIGALRM-budgeted) | ``catopt`` (``optimize_compositional``, per-block
e-graphs grafted back) | ``catopt+inductor`` (torch.compile over the
recomposed module — the deployment path ``CompiledRunner`` encodes).

Protocol per cell: build → eager reference → optimize_compositional
(opt wall ``opt_s`` + per-block ledger: status/rel/pairing/fires/
lowering) → e2e verify → inductor + catopt+inductor compiles (walls
recorded separately) with verifies → CUDA peak-mem per variant →
benchkit timing (median + IQR, CUDA-synced).  Only verified variants
are timed; failures land in the record as strings.

Honesty conventions (e2e_llm's): fp16 cells verify at rel < 2e-3,
fp32 at 1e-4 (``--verify-tol`` overrides); the ``in_place`` graft
fallback (deepcopy OOM → original returned) is detected and the
catopt variants excluded rather than re-timing eager; throughput =
units/s (embeddings, images, or tokens) per variant.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-optimize/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/e2e_models2.py \
        --device cuda
    .venv/bin/python bench/e2e_models2.py --device cpu --quick
"""
# ruff: noqa: E402, RUF002, RUF003 -- ×, ·, → in strings/docstrings
# are deliberate math notation; sys.path setup must precede the
# benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.setrecursionlimit(400_000)

import torch
import torch.nn as nn
import torch.nn.functional as F
from benchkit import Case, Report, Runner, Variant, collect_env
from catopt.optimize import optimize_compositional
from catopt_torch.report import verify_equiv
from e2e_model import TinyLlama
from real_win_hunt import try_compile

# run_all.py picks these up for its --quick lane (~1 cell per model,
# embedder + the MoE fold question; ≤ ~60 s GPU incl. compiles).
QUICK = {
    "models": "minilm,moe_stack",
    "minilm_cells": "128x8",
    "moe_cells": "128x4",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "max_iterations": "8",
    "warmup": "2",
}

#: Per-model metadata: throughput unit, default cells, dtype policy,
#: and the mechanism sentence the report quotes.
SPECS: dict[str, dict] = {
    "minilm": dict(
        unit="emb",
        cells_flag="minilm_cells",
        default_cells="128x8,512x8,128x32,512x32",
        dtype="fp32",
        mech="q/k/v pairing into one GEMM per encoder block "
        "(pre-fusion bias-free form); LN-gain fold if layer_norm "
        "lowers to the gain form",
    ),
    "vit_ti": dict(
        unit="img",
        cells_flag="vit_cells",
        default_cells="8,32",
        dtype="fp32",
        mech="same pairing as minilm at d=192 — margin datapoint",
    ),
    "conv_ti": dict(
        unit="img",
        cells_flag="conv_cells",
        default_cells="8,32",
        dtype="fp32",
        mech="dwconv unpairable (grouped); LN folds behind atomic "
        "layer_norm — expect parity",
    ),
    "llama_mid": dict(
        unit="tok",
        cells_flag="llama_cells",
        default_cells="128x4,256x4",
        dtype="fp16_cuda",  # fp16 on CUDA, fp32 on CPU
        mech="e2e_model pairing/rotary/fold set at ~40M fp16",
    ),
    "moe_stack": dict(
        unit="tok",
        cells_flag="moe_cells",
        default_cells="128x4,256x4",
        dtype="fp32",
        mech="8 same-input expert up/gate projections pair into "
        "grouped GEMMs; gated expert-sum must NOT fold",
    ),
}
MODEL_ORDER = list(SPECS)


# ---------------------------------------------------------------------------
#  Model A/B — pre-LN transformer block (MiniLM encoder / ViT share it)
# ---------------------------------------------------------------------------


class PreLNBlock(nn.Module):
    """Pre-LN attention + GELU-FFN block, bias-free projections.

    ``forward(x, attn_bias=None)`` — one arg for ViT, two for the
    masked encoder; both trace fine (``optimize_compositional``
    captures positional args per call site).  The q/k/v trio reads
    ONE normed tensor — the pairing pass's canonical case.
    """

    def __init__(self, dim: int, n_heads: int, hidden: int) -> None:
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.ln_att = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ln_ffn = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(
        self, x: torch.Tensor, attn_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        n = self.ln_att(x)
        q = self.wq(n).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(n).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(n).view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, C))
        n = self.ln_ffn(x)
        return x + self.fc2(F.gelu(self.fc1(n)))


class MiniLMEmbedder(nn.Module):
    """bge-small / MiniLM-L12-H384-shaped sentence embedder.

    ``forward(idx, mask)`` → (B, dim) L2-normalized embeddings:
    token+position embedding LN → ``layers`` pre-LN blocks with an
    additive key-padding bias → final LN → masked mean-pool →
    ``F.normalize`` — the e5/bge serving form.
    """

    def __init__(
        self,
        dim: int = 384,
        n_heads: int = 12,
        hidden: int = 1536,
        layers: int = 12,
        vocab: int = 30522,
        max_seq: int = 512,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.norm_e = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            PreLNBlock(dim, n_heads, hidden) for _ in range(layers)
        )
        self.norm_f = nn.LayerNorm(dim)
        self.vocab, self.max_seq = vocab, max_seq

    def forward(
        self, idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.norm_e(self.tok_emb(idx) + self.pos_emb(pos))
        # Additive key-padding bias (B,1,1,T): 0 valid, -1e4 masked —
        # fp32/fp16-safe (exp(-1e4) = 0 in either).
        bias = (mask[:, None, None, :].to(h.dtype) - 1.0) * 1e4
        for b in self.blocks:
            h = b(h, bias)
        h = self.norm_f(h)
        m = mask.to(h.dtype).unsqueeze(-1)  # (B,T,1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


class ViTTiny(nn.Module):
    """deit-tiny shape: patch16 → 196 tokens + cls + pos-emb → 12
    pre-LN blocks (d=192, 3 heads) → LN → cls head."""

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 3,
        hidden: int = 768,
        layers: int = 12,
        image: int = 224,
        patch: int = 16,
        classes: int = 1000,
    ) -> None:
        super().__init__()
        n_tok = (image // patch) ** 2 + 1
        self.patch = nn.Conv2d(3, dim, patch, patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        self.blocks = nn.ModuleList(
            PreLNBlock(dim, n_heads, hidden) for _ in range(layers)
        )
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes, bias=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        t = self.patch(img).flatten(2).transpose(1, 2)  # (B,196,d)
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1)
        t = t + self.pos
        for b in self.blocks:
            t = b(t)
        return self.head(self.norm_f(t[:, 0]))


# ---------------------------------------------------------------------------
#  Model C — ConvNeXt-tiny block stack (channels-first dwconv blocks)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm as the real ConvNeXt writes it:
    permute to NHWC → F.layer_norm → permute back.  (A
    ``GroupNorm(1, C)`` shortcut would export to ``group_norm``,
    whose positional-attr schema the IR bridge doesn't cover.)"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CNBlock(nn.Module):
    """ConvNeXt block: dwconv7×7 → NHWC LN → pw 4C → GELU → pw C →
    layer-scale (gamma) → residual.  The dwconv is grouped (excluded
    from conv pairing by signature); the pw pair is sequential, not
    same-input — this model is the honest parity/loss probe."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim, 7, padding=3, groups=dim, bias=False
        )
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim, bias=False)
        self.pw2 = nn.Linear(4 * dim, dim, bias=False)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # real init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x).permute(0, 2, 3, 1)  # NHWC
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        y = y.permute(0, 3, 1, 2)
        return x + self.gamma[None, :, None, None] * y


class Stem(nn.Module):
    """Patchify stem: conv4/s4 → LN2d (ConvNeXt order)."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch, 4, 4, bias=False)
        self.norm = LayerNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class Downsample(nn.Module):
    """Between-stage LN2d → conv2/s2 (ConvNeXt order)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class ConvTiny(nn.Module):
    """4-stage ConvNeXt-tiny-shaped stack [96,192,384,768], depths
    [1,1,3,1].  Every x→x stage member sits in ONE flat ModuleList so
    ``optimize_compositional``'s default block_pred gives per-block
    granularity (stem, blocks, downsamples)."""

    def __init__(
        self,
        dims: tuple[int, ...] = (96, 192, 384, 768),
        depths: tuple[int, ...] = (1, 1, 3, 1),
        classes: int = 200,
    ) -> None:
        super().__init__()
        body: list[nn.Module] = [Stem(dims[0])]
        for i, (c, d) in enumerate(zip(dims, depths, strict=True)):
            body += [CNBlock(c) for _ in range(d)]
            if i + 1 < len(dims):
                body.append(Downsample(c, dims[i + 1]))
        self.body = nn.ModuleList(body)
        self.norm_f = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.body:
            x = m(x)
        return self.head(self.norm_f(x.mean((2, 3))))


# ---------------------------------------------------------------------------
#  Model E — Mixtral-lite routed-MoE block stack
# ---------------------------------------------------------------------------


class MoEBlock(nn.Module):
    """rms → causal MHA ; rms → routed MoE FFN (dense-eval form).

    Router: softmax → top-k → scatter-mask → renormalize; ALL experts
    run and the outputs gate-sum — the eager form deployment stacks
    start from before hand-writing grouped GEMMs.  Expert weights are
    stacked ``(E, out, in)`` Parameters indexed per expert, so the
    e-graph sees 8 ``linear(n, w1[e])`` nodes on the SAME input — the
    pairing question — plus a gate-weighted output sum that must not
    fold (the gates are input-dependent).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dff: int,
        n_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.E, self.k = n_experts, top_k
        self.rms_att = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rms_ffn = nn.Parameter(torch.ones(dim))
        self.router = nn.Linear(dim, n_experts, bias=False)
        g1, g2 = dim**-0.5, dff**-0.5
        self.w1 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w3 = nn.Parameter(torch.randn(n_experts, dff, dim) * g1)
        self.w2 = nn.Parameter(torch.randn(n_experts, dim, dff) * g2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n = F.rms_norm(x, (C,), self.rms_att, 1e-5)
        q = self.wq(n).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(n).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(n).view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, C))
        n = F.rms_norm(x, (C,), self.rms_ffn, 1e-5)
        w = F.softmax(self.router(n), dim=-1)  # (B,T,E)
        tv, ti = w.topk(self.k, dim=-1)
        g = torch.zeros_like(w).scatter(-1, ti, tv)
        g = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
        out = None
        for e in range(self.E):
            he = F.silu(F.linear(n, self.w1[e])) * F.linear(
                n, self.w3[e]
            )
            ye = F.linear(he, self.w2[e]) * g[..., e].unsqueeze(-1)
            out = ye if out is None else out + ye
        return x + out


class MoEStack(nn.Module):
    """Input token map (B,T,d) → ``layers`` MoEBlocks → final rms."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 4,
        dff: int = 512,
        n_experts: int = 8,
        top_k: int = 2,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            MoEBlock(dim, n_heads, dff, n_experts, top_k)
            for _ in range(layers)
        )
        self.rms_f = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return F.rms_norm(x, (x.shape[-1],), self.rms_f, 1e-5)


# ---------------------------------------------------------------------------
#  Cell registry
# ---------------------------------------------------------------------------


def _dtype_for(spec: dict, dev: torch.device) -> torch.dtype:
    d = spec["dtype"]
    if d == "fp16_cuda":
        return torch.float16 if dev.type == "cuda" else torch.float32
    return torch.float32


def _cells_for(name: str, args) -> list[dict]:
    """Parse ``"TxB"`` or plain ``"B"`` cell specs into dicts."""
    spec = (
        getattr(args, SPECS[name]["cells_flag"], None)
        or SPECS[name]["default_cells"]
    )
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if "x" in tok:
            t_s, b_s = tok.split("x")
            out.append({"B": int(b_s), "T": int(t_s)})
        else:
            out.append({"B": int(tok)})
    return out


def _build(name: str, cell: dict, dev: torch.device):
    """→ (model, args_tuple, params_dict, n_units)."""
    spec = SPECS[name]
    dtype = _dtype_for(spec, dev)
    g = torch.Generator().manual_seed(
        1234 + cell.get("B", 1) * 97 + cell.get("T", 0)
    )
    params = dict(cell)
    params["model"] = name
    if name == "minilm":
        T, B = cell["T"], cell["B"]
        m = MiniLMEmbedder(max_seq=max(512, T)).to(dtype).eval()
        idx = torch.randint(0, m.vocab, (B, T), generator=g)
        # Real serving batches are ragged: lengths in [T//2, T].
        lens = torch.randint(T // 2, T + 1, (B,), generator=g)
        mask = (torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)).to(
            dtype
        )
        return (
            m,
            (idx.to(dev), mask.to(dev)),
            params,
            B,
        )
    if name == "vit_ti":
        m = ViTTiny().to(dtype).eval()
        x = torch.randn(cell["B"], 3, 224, 224, generator=g).to(dtype)
        return m, (x.to(dev),), params, cell["B"]
    if name == "conv_ti":
        m = ConvTiny().to(dtype).eval()
        x = torch.randn(cell["B"], 3, 224, 224, generator=g).to(dtype)
        return m, (x.to(dev),), params, cell["B"]
    if name == "llama_mid":
        T, B = cell["T"], cell["B"]
        m = (
            TinyLlama(
                dim=512,
                n_heads=8,
                n_kv_heads=4,
                hidden=1408,
                layers=8,
                vocab=16384,
                max_seq=max(512, T),
            )
            .to(dtype)
            .eval()
        )
        idx = torch.randint(0, m.vocab, (B, T), generator=g)
        return m, (idx.to(dev),), params, B * T
    if name == "moe_stack":
        T, B = cell["T"], cell["B"]
        m = MoEStack().to(dtype).eval()
        x = torch.randn(B, T, m.dim, generator=g).to(dtype)
        return m, (x.to(dev),), params, B * T
    raise ValueError(name)


# ---------------------------------------------------------------------------
#  Helpers (e2e_llm conventions)
# ---------------------------------------------------------------------------


def _informative_fires(fires: dict) -> dict:
    """Drop the comm/assoc noise rules; keep the mechanism signal."""
    drop = ("comm_", "assoc_add", "id_add", "id_mul")
    return {
        k: v
        for k, v in (fires or {}).items()
        if v and not k.startswith(drop)
    }


def _fmt_rel(v) -> str:
    return f"{v:.1e}" if isinstance(v, float) else str(v or "—")


def _oom(e: BaseException) -> bool:
    return isinstance(
        e, (torch.cuda.OutOfMemoryError, MemoryError)
    ) or ("out of memory" in str(e).lower())


def _fwd_stmt(mod, args: tuple):
    def stmt() -> None:
        with torch.no_grad():
            mod(*args)

    return stmt


def _peak_mem(mod, args: tuple, calls: int = 6) -> float | None:
    """Peak CUDA allocator MiB over ``calls`` forwards (None on CPU)."""
    if not args[0].is_cuda:
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(calls):
            mod(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def _catopt(mod: nn.Module, example, args, rec: dict, tol: float):
    """``optimize_compositional`` + mechanism ledger → recomposed module.

    Returns ``None`` when nothing was actually delivered: an exception,
    or the ``in_place`` fallback — the driver grafts blocks into a
    ``deepcopy`` of the model; if that copy OOMs it silently returns
    the ORIGINAL module (``stats["in_place"]=True``, ``n_optimized=0``
    while blocks still report ``status="optimized"``).  Timing that
    module re-times eager, so it is excluded and recorded honestly.
    """
    max_iter = int(getattr(args, "max_iterations", None) or 12)
    max_en = int(getattr(args, "max_enodes", None) or 100_000)
    verbose = bool(getattr(args, "verbose", False))
    t0 = time.time()
    try:
        opt, rep = optimize_compositional(
            mod,
            example,
            verbose=verbose,
            max_iterations=max_iter,
            max_enodes=max_en,
            verify_tol=tol,
        )
    except Exception as e:
        rec["opt_s"] = round(time.time() - t0, 2)
        rec["opt_error"] = (
            "oom" if _oom(e) else f"{type(e).__name__}: {e}"
        )
        print(f"  catopt FAILED: {rec['opt_error']}", flush=True)
        return None
    rec["opt_s"] = round(time.time() - t0, 2)
    rec["n_blocks"] = rep["n_blocks"]
    rec["n_optimized"] = rep["n_optimized"]
    rec["n_failed"] = rep["n_failed"]
    rec["in_place"] = bool(rep.get("in_place"))
    fires: dict[str, int] = {}
    paired = 0
    groups = 0
    lowerings: dict[str, int] = {}
    blk = {}
    for bn, be in rep["blocks"].items():
        st = be.get("stats") or {}
        bf = _informative_fires(st.get("rule_fires", {}))
        for k, v in bf.items():
            fires[k] = fires.get(k, 0) + v
        paired += 1 if st.get("paired_extract") else 0
        groups += st.get("pairing_groups") or 0
        lw = st.get("lowering")
        if lw:
            lowerings[lw] = lowerings.get(lw, 0) + 1
        blk[bn] = {
            "status": be.get("status"),
            "time_s": round(be.get("time_s") or 0, 2),
            "rel": be.get("rel_diff"),
            "paired": st.get("paired_extract"),
            "groups": st.get("pairing_groups"),
            "lowering": lw,
            "fires": bf or None,
            "error": be.get("error"),
        }
    rec["blocks"] = blk
    rec["fires_total"] = fires or None
    rec["paired_blocks"] = paired
    rec["pairing_groups_total"] = groups
    rec["lowerings"] = lowerings or None
    e2e = rep.get("end_to_end") or {}
    rec["catopt_driver_e2e_rel"] = e2e.get("max_rel_diff")
    print(
        f"  catopt: {rec['n_optimized']}/{rec['n_blocks']} blocks "
        f"in {rec['opt_s']:.1f}s  paired={paired} "
        f"groups={groups} in_place={rec['in_place']}",
        flush=True,
    )
    if rec["in_place"] or rec["n_optimized"] == 0:
        rec["catopt_status"] = (
            "in_place fallback: graft deepcopy failed — returned "
            "original; catopt variants excluded"
            if rec["in_place"]
            else "no blocks optimized — recomposed is the original"
        )
        del opt
        return None
    return opt


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def run_cell(
    model_name: str, cell_spec: dict, args, dev: torch.device
) -> tuple[dict, Case]:
    """Optimize + compile + verify + pack one (model, cell) case."""
    coords = " ".join(f"{k}={v}" for k, v in cell_spec.items())
    print(f"\n=== {model_name} [{coords}] ===", flush=True)
    spec = SPECS[model_name]
    dtype = _dtype_for(spec, dev)
    tol = getattr(args, "verify_tol", None) or (
        2e-3 if dtype == torch.float16 else 1e-4
    )
    _cto = getattr(args, "compile_timeout", None)
    cto = 90.0 if _cto is None else float(_cto)  # 0 = disabled
    cuda = dev.type == "cuda"

    torch.manual_seed(0)
    model, args_t, params, n_units = _build(model_name, cell_spec, dev)
    model = model.to(dev).eval()
    rec: dict = {
        "name": f"{model_name}@{coords.replace(' ', '')}",
        "params": params,
        "n_units": n_units,
        "unit": spec["unit"],
        "mech": spec["mech"],
        "dtype": str(dtype).rsplit(".", 1)[-1],
        "verify_tol": tol,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with torch.no_grad():
        ref = model(*args_t)

    # -- catopt, block-wise --------------------------------------------
    opt_c = _catopt(
        model,
        args_t if len(args_t) > 1 else args_t[0],
        args,
        rec,
        tol,
    )
    if opt_c is not None:
        try:
            with torch.no_grad():
                vr = verify_equiv(ref, opt_c(*args_t), rtol=tol)
            rec["catopt_rel"] = vr.max_rel
            rec["catopt_verified"] = bool(vr.passed)
            print(f"  catopt e2e rel={vr.max_rel:.2e}", flush=True)
            if not vr.passed:
                opt_c = None
        except Exception as e:
            rec["catopt_status"] = (
                "oom" if _oom(e) else f"fwd:{type(e).__name__}: {e}"
            )
            opt_c = None

    # -- inductor + composed -------------------------------------------
    cm = oci = None
    if cto > 0:
        t0 = time.time()
        cm, rec["inductor_status"] = try_compile(model, args_t, cto)
        rec["inductor_compile_s"] = round(time.time() - t0, 2)
        print(
            f"  inductor: {rec['inductor_status']} "
            f"({rec['inductor_compile_s']:.1f}s)",
            flush=True,
        )
        if cm is not None:
            try:
                with torch.no_grad():
                    vr = verify_equiv(ref, cm(*args_t), rtol=tol)
                rec["inductor_rel"] = vr.max_rel
                rec["inductor_verified"] = bool(vr.passed)
                if not vr.passed:
                    cm = None
            except Exception as e:
                rec["inductor_rel"] = f"err:{type(e).__name__}"
                cm = None
        if opt_c is not None:
            t0 = time.time()
            oci, rec["catopt_ind_status"] = try_compile(
                opt_c, args_t, cto
            )
            rec["catopt_ind_compile_s"] = round(time.time() - t0, 2)
            print(
                f"  catopt+inductor: {rec['catopt_ind_status']} "
                f"({rec['catopt_ind_compile_s']:.1f}s)",
                flush=True,
            )
            if oci is not None:
                try:
                    with torch.no_grad():
                        vr = verify_equiv(ref, oci(*args_t), rtol=tol)
                    rec["catopt_ind_rel"] = vr.max_rel
                    rec["catopt_ind_verified"] = bool(vr.passed)
                    if not vr.passed:
                        oci = None
                except Exception as e:
                    rec["catopt_ind_rel"] = f"err:{type(e).__name__}"
                    oci = None
    else:
        rec["inductor_status"] = "disabled (--compile-timeout 0)"

    # -- peak memory (CUDA only) ----------------------------------------
    if cuda:
        for vn, m in (
            ("eager", model),
            ("inductor", cm),
            ("catopt", opt_c),
            ("catopt+inductor", oci),
        ):
            if m is not None:
                try:
                    rec[f"peak_mb_{vn}"] = round(
                        _peak_mem(m, args_t), 1
                    )
                except Exception as e:
                    rec[f"peak_mb_{vn}"] = f"err:{type(e).__name__}"

    # -- pack the benchkit case (verified variants only) -----------------
    ok = {
        "eager": True,
        "inductor": cm is not None and rec.get("inductor_verified"),
        "catopt": opt_c is not None and rec.get("catopt_verified"),
        "catopt+inductor": oci is not None
        and rec.get("catopt_ind_verified"),
    }
    variants = []
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("catopt", opt_c),
        ("catopt+inductor", oci),
    ):
        if m is not None and ok.get(vn):
            variants.append(Variant(vn, _fwd_stmt(m, args_t)))

    case = Case(
        name=rec["name"], params=params, variants=variants, aux=rec
    )
    return rec, case


# ---------------------------------------------------------------------------
#  Harness entry point (run_all.py convention)
# ---------------------------------------------------------------------------


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    min_run_time = float(getattr(args, "min_run_time", None) or 0.2)
    warmup = int(getattr(args, "warmup", None) or 3)

    model_names = getattr(args, "models", None) or ",".join(MODEL_ORDER)
    specs: list[tuple[str, list[dict]]] = []
    for name in (s.strip() for s in model_names.split(",")):
        if name in SPECS:
            specs.append((name, _cells_for(name, args)))
        elif name:
            print(f"  unknown model {name!r} — skipped", flush=True)

    print(
        f"e2e_models2 — device={dev} "
        f"cells={[(n, len(cs)) for n, cs in specs]}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cells: list = []
    for name, cells_spec in specs:
        for cell_spec in cells_spec:
            try:
                rec, case = run_cell(name, cell_spec, args, dev)
            except Exception as e:
                # A failed cell is recorded, not fatal to the sweep.
                rec = {
                    "name": f"{name}@err",
                    "params": dict(cell_spec) | {"model": name},
                    "status": "oom" if _oom(e) else f"err:{e}",
                    "unit": SPECS[name]["unit"],
                }
                case = Case(
                    name=rec["name"],
                    params=rec["params"],
                    variants=[],
                    aux=rec,
                )
                print(f"  cell FAILED: {rec['status']}", flush=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            recs.append(rec)
            # Time immediately: compiled artifacts are warm and
            # run_case releases each stmt (hence its module) — keeps
            # the 4 GB card clean across the sweep.
            cell = runner.run_case(case)
            print(
                "  times: "
                + "  ".join(
                    f"{n}={cell.medians[n] * 1e3:.3f}ms"
                    for n in cell.medians
                ),
                flush=True,
            )
            cells.append(cell)
            gc.collect()
            torch._dynamo.reset()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # Throughput per variant per cell.
    for rec, cell in zip(recs, cells, strict=True):
        unit, n_units = rec.get("unit", "unit"), rec.get("n_units", 0)
        for vn, med in cell.medians.items():
            cell.aux[f"{unit}_s_{vn}"] = round(n_units / med, 1)

    # -- console: per-model tables ---------------------------------------
    names = ["eager", "inductor", "catopt", "catopt+inductor"]
    for name, _ in specs:
        mrecs = [
            (r, c)
            for r, c in zip(recs, cells, strict=True)
            if r["params"].get("model") == name
        ]
        if not mrecs:
            continue
        r0 = mrecs[0][0]
        print(
            f"\n### {name} — {r0.get('n_params', 0) / 1e6:.1f}M "
            f"params {r0.get('dtype', '?')} — {r0.get('mech', '')}"
        )
        hdr = (
            f"{'cell':<18} | "
            + " | ".join(f"{n:>14}" for n in names)
            + f" | {'c+i/ind':>7} | {'ver':>3}"
        )
        print(hdr)
        print("-" * len(hdr))
        ratios = []
        for rec, cell in mrecs:
            ms = cell.medians
            row = f"{rec['name']:<18} | " + " | ".join(
                (
                    f"{ms[n] * 1e3:>8.3f} ms  "
                    if n in ms
                    else f"{'—':>14}"
                )
                for n in names
            )
            mi, mc = ms.get("inductor"), ms.get("catopt+inductor")
            ratio = mi / mc if mi and mc else None
            if ratio is not None:
                ratios.append(ratio)
            ver = "yes" if rec.get("catopt_verified") else "NO"
            print(
                f"{row} | {f'{ratio:.3f}' if ratio else '—':>7} | {ver:>3}"
            )
            unit = rec.get("unit", "unit")
            thr = "  ".join(
                f"{n}={cell.aux.get(f'{unit}_s_{n}', '—')}"
                for n in ("eager", "inductor", "catopt+inductor")
            )
            print(
                f"{'':<18} | {unit}/s {thr}  "
                f"opt={rec.get('opt_s', '—')}s  "
                f"ind={rec.get('inductor_status', '—')}",
                flush=True,
            )
        print("-" * len(hdr))
        if ratios:
            geo = 1.0
            for r in ratios:
                geo *= r
            geo **= 1.0 / len(ratios)
            wins = sum(1 for r in ratios if r > 1.0)
            print(
                f"  → {name}: c+i beats plain inductor in "
                f"{wins}/{len(ratios)} cells (geomean {geo:.3f}); "
                f"paired={mrecs[0][0].get('paired_blocks', '—')} "
                f"groups={mrecs[0][0].get('pairing_groups_total', '—')} "
                f"fires={mrecs[0][0].get('fires_total')}"
            )

    # -- console: compile/opt wall + memory + correctness ---------------
    print("\n--- wall clock (s) ---")
    print(
        f"{'cell':<18} | {'catopt opt':>10} | {'inductor':>10} | "
        f"{'c+i comp':>10}"
    )
    for rec in recs:
        print(
            f"{rec['name']:<18} | "
            f"{rec.get('opt_s', '—')!s:>10} | "
            f"{rec.get('inductor_compile_s', '—')!s:>10} | "
            f"{rec.get('catopt_ind_compile_s', '—')!s:>10}"
        )
    print("\n--- peak CUDA mem (MiB) ---")
    print(f"{'cell':<18} | " + " | ".join(f"{n:>10}" for n in names))
    for rec in recs:
        print(
            f"{rec['name']:<18} | "
            + " | ".join(
                f"{rec.get(f'peak_mb_{n}', '—')!s:>10}" for n in names
            )
        )
    print("\n--- correctness (max rel vs eager) ---")
    for rec in recs:
        print(
            f"{rec['name']:<18} | "
            f"catopt={_fmt_rel(rec.get('catopt_rel'))}  "
            f"ind={_fmt_rel(rec.get('inductor_rel'))}  "
            f"c+i={_fmt_rel(rec.get('catopt_ind_rel'))}  "
            f"(tol {rec.get('verify_tol', '—')})"
        )
    print(
        f"\n  c+i/ind = inductor_median / catopt+inductor_median "
        f"(>1 = composed wins).  total wall "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    env = collect_env(dev)
    env["notes"] = [
        "In-repo architecture replicas — fixed-seed random weights, "
        "no checkpoints.  Shapes: minilm ≈ bge-small/L12-H384, "
        "vit_ti ≈ deit-ti, conv_ti ≈ ConvNeXt-tiny stage stack, "
        "llama_mid ≈ stories-scale llama at d=512/L=8 fp16, "
        "moe_stack ≈ Mixtral-lite routed MoE (dense-eval gates).",
        "catopt = optimize_compositional per-block; catopt+inductor "
        "= torch.compile over the recomposed module (the "
        "CompiledRunner path).  opt_s and compile walls are reported "
        "separately from timed medians.",
        "verify gates: fp16 cells rel<2e-3, fp32 rel<1e-4; only "
        "verified variants are timed.  in_place=True records the "
        "graft-deepcopy OOM fallback (catopt variants excluded).",
        "minilm mask: per-row lengths in [T//2, T] — a real padded "
        "serving batch, not all-ones.",
    ]
    report = Report(suite="e2e_models2", cells=cells, env=env)
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"e2e_models2_{ts}.json"
        md_path = out_dir / f"e2e_models2_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "e2e bench #2: minilm embedder / vit-ti / convnext-ti "
            "stack / mid llama / routed MoE — eager vs inductor vs "
            "catopt vs catopt∘inductor"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma subset of " + ",".join(MODEL_ORDER),
    )
    ap.add_argument(
        "--minilm-cells",
        dest="minilm_cells",
        type=str,
        default=None,
        help="TxB list (default '128x8,512x8,128x32,512x32')",
    )
    ap.add_argument(
        "--vit-cells",
        dest="vit_cells",
        type=str,
        default=None,
        help="B list (default '8,32')",
    )
    ap.add_argument(
        "--conv-cells",
        dest="conv_cells",
        type=str,
        default=None,
        help="B list (default '8,32')",
    )
    ap.add_argument(
        "--llama-cells",
        dest="llama_cells",
        type=str,
        default=None,
        help="TxB list (default '128x4,256x4')",
    )
    ap.add_argument(
        "--moe-cells",
        dest="moe_cells",
        type=str,
        default=None,
        help="TxB list (default '128x4,256x4')",
    )
    ap.add_argument(
        "--verify-tol",
        dest="verify_tol",
        type=float,
        default=None,
        help="rel gate override (default 2e-3 fp16 / 1e-4 fp32)",
    )
    ap.add_argument(
        "--compile-timeout",
        dest="compile_timeout",
        type=float,
        default=90.0,
        help="torch.compile budget per variant per cell, seconds "
        "(0 disables compiled variants)",
    )
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--max-enodes", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument(
        "--min-run-time",
        dest="min_run_time",
        type=float,
        default=None,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()
    if args.quick:
        for k, v in QUICK.items():
            cur = getattr(args, k)
            setattr(args, k, str(v) if cur is None else type(cur)(v))
    run_bench(args)


if __name__ == "__main__":
    main()
