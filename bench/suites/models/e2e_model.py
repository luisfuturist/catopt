"""End-to-end in-repo model benchmark — a real-ish decoder, not cells.

``model_bench`` times complete models built from catopt's own model
zoo (``ParallelBlock``/``LinearAttnStack``).  This bench asks the
deployment question on a model that looks like real Llama code:
token embedding → N decoder blocks (RMSNorm → grouped-query causal
attention with rotary → SwiGLU MLP, separate q/k/v and w1/w3/w2
projections — the pre-fusion form deployment code actually ships) →
final RMSNorm → LM head.  Everything is defined here; no checkpoint,
no network.

Models:

* ``tiny_llama`` — d=256, L=4, 8 q-heads / 2 kv-heads (GQA), hidden
  704, vocab 1024.  Mechanisms the search should fire per block:
  product-law pairing (q/k/v → one GEMM, w1/w3 → one GEMM),
  rotary-concat reassociation, RMSNorm channel-gain folds into the
  following projections.
* ``conv_attn`` — 2-conv stem → token map → 2 non-causal attention
  blocks (same q/k/v + SwiGLU structure, no rotary) → pooled head.
  A cheap mixed conv/attention second model.

Protocol per sweep cell (one cell per (model, B, T)):

1.  ``optimize_compositional`` on the fp32 model — the block-wise
    path; per-block status/pairing/rule-fires/wall time are recorded
    and the delivered module is verified per-block (inside the
    driver) and end-to-end against eager.
2.  whole-graph ``optimize_model`` on a deepcopy (variant
    ``catopt_full``) — the monolithic search, SIGALRM-guarded; at
    L=4 it still fits, so the table shows what the compositional
    decomposition costs vs buys in search wall time.
3.  SIGALRM-guarded ``torch.compile`` of eager and of both delivered
    modules — ``inductor``, ``catopt+inductor``, ``catopt_full+ind``.
    ``optimize_compositional`` takes no ``runner=`` (it is a
    per-block driver), so the composed path is the honest equivalent:
    ``torch.compile(recomposed_module)`` — exactly what
    ``CompiledRunner`` would produce.
4.  benchkit ``Runner`` times every variant (median + IQR,
    CUDA-synced on GPU); each timed output is re-verified against the
    eager reference.  On CUDA, ``torch.cuda.max_memory_allocated``
    peaks are recorded per variant.

The reported number is whole-sequence forward (prefill) latency;
``tok/s`` = B·T / median.  There is no KV cache — an autoregressive
decode loop here would just re-run a growing prefill and recompile
per step, so forward latency is the honest unit.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/e2e_model.py \
        --device cuda
    .venv/bin/python bench/e2e_model.py --device cpu --quick
"""
# ruff: noqa: E402, RUF001, RUF003 -- ×, ·, → in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import copy
import gc
import signal
import sys
import time
from pathlib import Path

sys.setrecursionlimit(400_000)

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_orchestrator import Compositional, Optimizer
from catopt_torch.backend import TorchBackend
from catopt_torch.report import verify_equiv

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)
from bench.suites.algebra.real_win_hunt import try_compile

# run_all.py picks these up for its --quick lane.
QUICK = {
    "cells": "64x4,128x4",
    "models": "tiny_llama",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "full_timeout": "60.0",
    "max_iterations": "12",
}


class _CompileTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CompileTimeout()


# ---------------------------------------------------------------------------
#  Model 1 — Llama-style decoder (embedding + GQA + rotary + SwiGLU)
# ---------------------------------------------------------------------------


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotary (rotate-half form).  ``x``: (B, H, T, hd); ``cos``/``sin``:
    (T, hd//2)."""
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class GQAAttention(nn.Module):
    """Grouped-query causal attention with separate q/k/v projections
    and rotary on q/k — the real Llama pre-fusion form."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        self.nh, self.nkv = n_heads, n_kv_heads
        self.hd = dim // n_heads
        self.rms_w = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, dim, bias=False)

    def _expand_kv(self, t: torch.Tensor, B: int, T: int):
        # (B, nkv, T, hd) → (B, nh, T, hd): real GQA kv-head repeat.
        return (
            t[:, :, None, :, :]
            .expand(B, self.nkv, self.nh // self.nkv, T, self.hd)
            .reshape(B, self.nh, T, self.hd)
        )

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = h.shape
        n = F.rms_norm(h, (h.shape[-1],), self.rms_w, 1e-5)
        q = self.wq(n).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(n).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(n).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k, v = self._expand_kv(k, B, T), self._expand_kv(v, B, T)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class LlamaBlock(nn.Module):
    """``h + attn(rms h)``; ``h + w2(silu(w1 n)·w3 n)`` — the decoder
    block.  Two same-input projection sets per block (q/k/v reads one
    normed tensor, w1/w3 read another) — the pairing case."""

    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, hidden: int
    ) -> None:
        super().__init__()
        self.attn = GQAAttention(dim, n_heads, n_kv_heads)
        self.rms_ffn = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        h = h + self.attn(h, cos, sin)
        n = F.rms_norm(h, (h.shape[-1],), self.rms_ffn, 1e-5)
        return h + self.w2(F.silu(self.w1(n)) * self.w3(n))


class TinyLlama(nn.Module):
    """Embedding → ``layers`` LlamaBlocks → final RMSNorm → LM head.

    cos/sin live as model buffers (Llama convention); each block gets
    the ``[:T]`` slice positionally, so ``optimize_compositional`` sees
    each block as a three-input module and captures real tensors.
    """

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        hidden: int = 704,
        layers: int = 4,
        vocab: int = 1024,
        max_seq: int = 512,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            LlamaBlock(dim, n_heads, n_kv_heads, hidden)
            for _ in range(layers)
        )
        self.rms_final = nn.Parameter(torch.ones(dim))
        self.head = nn.Linear(dim, vocab, bias=False)
        hd = dim // n_heads
        freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(max_seq).float(), freqs)
        self.register_buffer("cos", fr.cos())  # (max_seq, hd//2)
        self.register_buffer("sin", fr.sin())
        self.vocab = vocab

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[-1]
        h = self.tok_emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            h = b(h, cos, sin)
        h = F.rms_norm(h, (h.shape[-1],), self.rms_final, 1e-5)
        return self.head(h)


# ---------------------------------------------------------------------------
#  Model 2 — conv stem + non-causal attention blocks + pooled head
# ---------------------------------------------------------------------------


class AttnBlock(nn.Module):
    """Non-causal MH attention + SwiGLU MLP on (B, T, C) tokens — the
    same pairing structure as LlamaBlock minus rotary/GQA."""

    def __init__(self, dim: int, n_heads: int, hidden: int) -> None:
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.rms_att = nn.Parameter(torch.ones(dim))
        self.rms_ffn = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n = F.rms_norm(x, (C,), self.rms_att, 1e-5)
        q = self.wq(n).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(n).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(n).view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, C))
        n = F.rms_norm(x, (C,), self.rms_ffn, 1e-5)
        return x + self.w2(F.silu(self.w1(n)) * self.w3(n))


class ConvAttnNet(nn.Module):
    """conv stem → token map → attention blocks → pooled head."""

    def __init__(
        self,
        in_ch: int = 3,
        ch: int = 64,
        n_heads: int = 4,
        hidden: int = 128,
        layers: int = 2,
        classes: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            AttnBlock(ch, n_heads, hidden) for _ in range(layers)
        )
        self.head = nn.Linear(ch, classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.stem(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        for b in self.blocks:
            t = b(t)
        return self.head(t.mean(dim=1))


# ---------------------------------------------------------------------------
#  Cell registry
# ---------------------------------------------------------------------------


def _llama_cells(args) -> list[dict]:
    spec = getattr(args, "cells", None) or "64x4,128x4,256x4,128x8"
    out = []
    for tok in spec.split(","):
        t_s, b_s = tok.strip().lower().split("x")
        out.append({"B": int(b_s), "T": int(t_s)})
    return out


def _conv_cells(args) -> list[dict]:
    spec = getattr(args, "conv_cells", None) or "16x4"
    out = []
    for tok in spec.split(","):
        hw_s, b_s = tok.strip().lower().split("x")
        out.append({"B": int(b_s), "HW": int(hw_s)})
    return out


def _build(name: str, args, cell: dict):
    """→ (model, args_tuple, params_dict, tokens_per_fwd)."""
    if name == "tiny_llama":
        m = TinyLlama(
            dim=int(getattr(args, "dim", None) or 256),
            n_heads=int(getattr(args, "heads", None) or 8),
            n_kv_heads=int(getattr(args, "kv_heads", None) or 2),
            hidden=int(getattr(args, "hidden", None) or 704),
            layers=int(getattr(args, "layers", None) or 4),
            vocab=int(getattr(args, "vocab", None) or 1024),
            max_seq=max(512, cell["T"]),
        ).eval()
        g = torch.Generator().manual_seed(
            1234 + cell["B"] * 97 + cell["T"]
        )
        idx = torch.randint(
            0, m.vocab, (cell["B"], cell["T"]), generator=g
        )
        params = dict(cell)
        params["model"] = name
        return m, (idx,), params, cell["B"] * cell["T"]
    if name == "conv_attn":
        m = ConvAttnNet(
            ch=int(getattr(args, "conv_ch", None) or 64),
            layers=int(getattr(args, "conv_layers", None) or 2),
        ).eval()
        g = torch.Generator().manual_seed(
            4321 + cell["B"] * 31 + cell["HW"]
        )
        x = torch.randn(
            cell["B"], 3, cell["HW"], cell["HW"], generator=g
        )
        params = dict(cell)
        params["model"] = name
        params["T"] = cell["HW"] ** 2  # token count after flatten
        return m, (x,), params, cell["B"] * cell["HW"] ** 2
    raise ValueError(name)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _fwd_stmt(mod, args: tuple):
    def stmt() -> None:
        with torch.no_grad():
            mod(*args)

    return stmt


def _peak_mem(mod, args: tuple, calls: int = 8) -> float | None:
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


def _informative_fires(fires: dict) -> dict:
    """Drop the comm/assoc noise rules; keep the mechanism signal."""
    drop = ("comm_", "assoc_add", "id_add", "id_mul")
    return {
        k: v
        for k, v in (fires or {}).items()
        if v and not k.startswith(drop)
    }


def _sig_guarded(fn, budget_s: float):
    """Run ``fn()`` under a SIGALRM wall-clock bound → (result, s, err)."""
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover — non-POSIX
        t0 = time.perf_counter()
        try:
            return fn(), time.perf_counter() - t0, None
        except Exception as e:
            return None, time.perf_counter() - t0, e
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    t0 = time.perf_counter()
    try:
        out = fn()
        return out, time.perf_counter() - t0, None
    except _CompileTimeout:
        return (
            None,
            time.perf_counter() - t0,
            TimeoutError(f"exceeded {budget_s:.0f}s budget"),
        )
    except Exception as e:
        return None, time.perf_counter() - t0, e
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def run_cell(
    model_name: str,
    cell_spec: dict,
    args,
    dev: torch.device,
) -> tuple[dict, Case]:
    """Optimize + compile + verify one (model, B, T) cell."""
    coords = " ".join(f"{k}={v}" for k, v in cell_spec.items())
    print(f"\n=== {model_name} [{coords}] ===", flush=True)
    torch.manual_seed(0)
    model, args_t, params, n_tokens = _build(
        model_name, args, cell_spec
    )
    model = model.to(dev).eval()
    args_t = tuple(a.to(dev) for a in args_t)
    ex = args_t if len(args_t) > 1 else args_t[0]
    rec: dict = {
        "name": f"{model_name}@{'x'.join(str(v) for v in cell_spec.values())}",
        "params": params,
        "n_tokens": n_tokens,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with torch.no_grad():
        ref = model(*args_t)

    max_iterations = int(getattr(args, "max_iterations", None) or 16)
    max_enodes = int(getattr(args, "max_enodes", None) or 200_000)
    compile_timeout = float(
        getattr(args, "compile_timeout", None) or 90.0
    )
    full_timeout = float(getattr(args, "full_timeout", None) or 120.0)
    # _LaxNamespace reads unset attrs as None — default flags True.
    fg = getattr(args, "full_graph", None)
    do_full = True if fg is None else bool(fg)
    fp64v = getattr(args, "fp64_verify", None)
    do_fp64 = True if fp64v is None else bool(fp64v)
    verbose = bool(getattr(args, "verbose", False))

    # -- catopt, block-wise (the compositional path) --------------------
    opt_c = None
    t0 = time.time()
    try:
        opt_c, rep = Optimizer(backend=TorchBackend()).optimize(model, ex, strategy=Compositional(), verbose=verbose, max_iterations=max_iterations, max_enodes=max_enodes)

        rec["opt_s"] = round(time.time() - t0, 2)
        rec["n_blocks"] = rep["n_blocks"]
        rec["n_optimized"] = rep["n_optimized"]
        rec["n_failed"] = rep["n_failed"]
        e2e = rep.get("end_to_end") or {}
        rec["catopt_e2e_rel"] = e2e.get("max_rel_diff")
        # Per-block mechanism ledger — which rewrites actually fired.
        blk = {}
        tot_groups = tot_lifts = n_paired = 0
        fires_all: dict[str, int] = {}
        for bn, be in rep["blocks"].items():
            st = be.get("stats") or {}
            fires = _informative_fires(st.get("rule_fires", {}))
            for k, v in fires.items():
                fires_all[k] = fires_all.get(k, 0) + v
            tot_groups += st.get("pairing_groups") or 0
            tot_lifts += st.get("nonlocal_lifts") or 0
            n_paired += 1 if st.get("paired_extract") else 0
            blk[bn] = {
                "status": be.get("status"),
                "time_s": round(be.get("time_s") or 0, 2),
                "rel": be.get("rel_diff"),
                "paired": st.get("paired_extract"),
                "groups": st.get("pairing_groups"),
                "lifts": st.get("nonlocal_lifts"),
                "lowering": st.get("lowering"),
                "fires": fires or None,
                "error": be.get("error"),
            }
        rec["blocks"] = blk
        rec["pairing_groups_total"] = tot_groups
        rec["paired_blocks"] = n_paired
        rec["nonlocal_lifts_total"] = tot_lifts
        rec["fires_total"] = fires_all
        # e2e verify (fp32) — the delivered module vs eager.
        with torch.no_grad():
            vr = verify_equiv(ref, opt_c(*args_t), rtol=1e-4)
        rec["catopt_rel"] = vr.max_rel
        rec["catopt_verified"] = bool(vr.passed)
        # fp64 verification pass: the delivered module deep-copies
        # poorly (IRModules hold non-leaf tensors), so the honest fp64
        # check is a second fp64 search + e2e compare — the same
        # convention model_bench uses (fp64 pipeline, fp32 delivery).
        # It also shows the search is dtype-stable.
        rec["catopt_fp64_rel"] = "skipped"
        if do_fp64:
            try:
                m64 = copy.deepcopy(model).double().eval()
                a64 = tuple(
                    a.double() if a.is_floating_point() else a
                    for a in args_t
                )
                ex64 = a64 if len(a64) > 1 else a64[0]
                t64 = time.time()
                opt64, rep64 = Optimizer(backend=TorchBackend()).optimize(m64, ex64, strategy=Compositional(), verbose=False, max_iterations=max_iterations, max_enodes=max_enodes)

                rec["opt64_s"] = round(time.time() - t64, 2)
                rec["n64_optimized"] = rep64["n_optimized"]
                with torch.no_grad():
                    rec["catopt_fp64_rel"] = verify_equiv(
                        m64(*a64), opt64(*a64), rtol=1e-4
                    ).max_rel
                del m64, opt64
            except Exception as e:
                rec["catopt_fp64_rel"] = f"err:{type(e).__name__}: {e}"
        print(
            f"  catopt: {rec['n_optimized']}/{rec['n_blocks']} blocks in "
            f"{rec['opt_s']:.1f}s  paired={n_paired} groups={tot_groups} "
            f"rel={vr.max_rel:.2e} fp64_rel={rec['catopt_fp64_rel']}",
            flush=True,
        )
    except Exception as e:
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        rec["opt_s"] = round(time.time() - t0, 2)
        print(f"  optimize_compositional FAILED: {e}", flush=True)

    # -- catopt, whole-graph (monolithic e-graph over all blocks) -------
    opt_f = None
    if do_full:

        def _full():
            return Optimizer(backend=TorchBackend()).optimize(copy.deepcopy(model), ex, max_iterations=max_iterations, max_enodes=max_enodes, verify=False, verbose=False)


        out, dt, err = _sig_guarded(_full, full_timeout)
        rec["opt_full_s"] = round(dt, 2)
        if err is None and out is not None:
            opt_f, st_f = out
            rec["full_pairing_groups"] = st_f.get("pairing_groups")
            rec["full_paired_extract"] = st_f.get("paired_extract")
            rec["full_lifts"] = st_f.get("nonlocal_lifts")
            rec["full_fires"] = _informative_fires(
                st_f.get("rule_fires", {})
            )
            try:
                with torch.no_grad():
                    vr = verify_equiv(ref, opt_f(*args_t), rtol=1e-4)
                rec["full_rel"] = vr.max_rel
                rec["full_verified"] = bool(vr.passed)
                if not vr.passed:
                    opt_f = None
                    rec["full_status"] = (
                        f"verify failed rel={vr.max_rel:.2e}"
                    )
                else:
                    rec["full_status"] = "ok"
            except Exception as e:
                opt_f = None
                rec["full_status"] = (
                    f"verify err: {type(e).__name__}: {e}"
                )
        else:
            rec["full_status"] = (
                f"{type(err).__name__}: {err}" if err else "no result"
            )
        print(
            f"  catopt_full: {rec['full_status']} ({rec['opt_full_s']:.1f}s)",
            flush=True,
        )

    # -- inductor baseline + composed variants --------------------------
    cm = oci = ofi = None
    if compile_timeout > 0:
        cm, rec["inductor_status"] = try_compile(
            copy.deepcopy(model), args_t, compile_timeout
        )
        print(f"  inductor: {rec['inductor_status']}", flush=True)
        if cm is not None:
            with torch.no_grad():
                vr = verify_equiv(ref, cm(*args_t), rtol=1e-4)
            rec["inductor_rel"] = vr.max_rel
            rec["inductor_verified"] = bool(vr.passed)
        if opt_c is not None and rec.get("catopt_verified"):
            oci, rec["catopt_ind_status"] = try_compile(
                opt_c, args_t, compile_timeout
            )
            print(
                f"  catopt+inductor: {rec['catopt_ind_status']}",
                flush=True,
            )
            if oci is not None:
                with torch.no_grad():
                    vr = verify_equiv(ref, oci(*args_t), rtol=1e-4)
                rec["catopt_ind_rel"] = vr.max_rel
                rec["catopt_ind_verified"] = bool(vr.passed)
                if not vr.passed:
                    oci = None
        if opt_f is not None:
            ofi, rec["full_ind_status"] = try_compile(
                opt_f, args_t, compile_timeout
            )
            if ofi is not None:
                with torch.no_grad():
                    vr = verify_equiv(ref, ofi(*args_t), rtol=1e-4)
                rec["full_ind_rel"] = vr.max_rel
                if not vr.passed:
                    ofi = None
    else:
        rec["inductor_status"] = "disabled (--compile-timeout 0)"

    # -- peak memory (CUDA only), before Runner timing ------------------
    if dev.type == "cuda":
        for vn, m in (
            ("eager", model),
            ("inductor", cm),
            ("catopt", opt_c),
            ("catopt+inductor", oci),
            ("catopt_full", opt_f),
            ("catopt_full+ind", ofi),
        ):
            if m is not None:
                try:
                    rec[f"peak_mb_{vn}"] = round(
                        _peak_mem(m, args_t), 1
                    )
                except Exception as e:
                    rec[f"peak_mb_{vn}"] = f"err:{type(e).__name__}"

    # -- pack the benchkit case ------------------------------------------
    # A wrong program is never benchmarked silently: only variants that
    # verified (rel < 1e-4 vs eager) get timed.
    ok = {
        "eager": True,
        "inductor": cm is not None and rec.get("inductor_verified"),
        "catopt": opt_c is not None and rec.get("catopt_verified"),
        "catopt+inductor": oci is not None
        and rec.get("catopt_ind_verified", True),
        "catopt_full": opt_f is not None,
        "catopt_full+ind": ofi is not None,
    }
    variants = []
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("catopt", opt_c),
        ("catopt+inductor", oci),
        ("catopt_full", opt_f),
        ("catopt_full+ind", ofi),
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

    model_names = (
        getattr(args, "models", None) or "tiny_llama,conv_attn"
    )
    specs: list[tuple[str, list[dict]]] = []
    for name in (s.strip() for s in model_names.split(",")):
        if name == "tiny_llama":
            specs.append((name, _llama_cells(args)))
        elif name == "conv_attn":
            specs.append((name, _conv_cells(args)))

    print(
        f"e2e_model — device={dev} "
        f"cells={[(n, len(cs)) for n, cs in specs]}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cells: list = []
    for name, specs_cells in specs:
        for cell_spec in specs_cells:
            rec, case = run_cell(name, cell_spec, args, dev)
            recs.append(rec)
            # Time immediately: the compiled variants were just built by
            # try_compile, so timing runs against warm artifacts; and
            # run_case releases each variant's stmt (hence its module)
            # afterwards, so per-cell deepcopies/compiled wrappers don't
            # pile up across the sweep on a 4 GB card.
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

    # Post-timing derived metrics: tokens/s per variant per cell.
    for rec, cell in zip(recs, cells, strict=True):
        tok = rec["n_tokens"]
        for vn, med in cell.medians.items():
            cell.aux[f"tok_s_{vn}"] = round(tok / med, 1)

    # -- console table ----------------------------------------------------
    names = [
        "eager",
        "inductor",
        "catopt",
        "catopt+inductor",
        "catopt_full",
        "catopt_full+ind",
    ]
    hdr = (
        f"{'cell':<22} | "
        + " | ".join(f"{n:>14}" for n in names)
        + f" | {'c+i/ind':>7} | {'ver':>3}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians
        row = f"{rec['name']:<22} | " + " | ".join(
            (f"{ms[n] * 1e3:>8.3f} ms  " if n in ms else f"{'—':>14}")
            for n in names
        )
        mi, mc = ms.get("inductor"), ms.get("catopt+inductor")
        ratio = f"{mi / mc:>7.3f}" if mi and mc else f"{'—':>7}"
        ver = "yes" if rec.get("catopt_verified") else "NO"
        print(f"{row} | {ratio} | {ver:>3}")
        print(
            f"{'':<22} | tok/s: "
            + "  ".join(
                f"{n}={cell.aux.get(f'tok_s_{n}', '—')}"
                for n in ("eager", "catopt+inductor")
            )
            + f"  opt={rec.get('opt_s', '—')}s"
            f"  ind={rec.get('inductor_status', '—')}"
        )
    print("-" * len(hdr))
    print(
        "  c+i/ind = inductor_median / catopt+inductor_median "
        "(>1 = catopt-composed wins).  ver = catopt e2e rel<1e-4."
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s", flush=True
    )

    scored = [
        (
            c.case.name,
            c.medians["inductor"] / c.medians["catopt+inductor"],
        )
        for c in cells
        if c.medians.get("inductor") and c.medians.get("catopt+inductor")
    ]
    best = max(scored, key=lambda t: t[1], default=None)
    n_verified = sum(1 for r in recs if r.get("catopt_verified"))
    findings = [
        Finding(
            claim=(
                "the composed catopt+Inductor module beats plain "
                "Inductor end-to-end"
            ),
            verdict=(
                Verdict.WIN
                if best and best[1] > 1.03
                else Verdict.PARITY
                if best and best[1] > 0.97
                else Verdict.REGRESSION
                if best
                else Verdict.NEGATIVE
            ),
            headline=(
                f"best {best[1]:.2f}× vs Inductor on {best[0]}"
                if best
                else "no cell measured"
            ),
            metric="inductor / catopt+inductor",
            value=best[1] if best else None,
        ),
        Finding(
            claim="the optimized module verifies against eager",
            verdict=(
                Verdict.WIN
                if recs and n_verified == len(recs)
                else Verdict.REGRESSION
            ),
            headline=f"{n_verified}/{len(recs)} cells verified",
            metric="verified cells",
            value=float(n_verified),
        ),
    ]
    report = Report(
        suite="e2e_model",
        title="MiniGPT end-to-end",
        summary=(
            "Whole-model optimize + verify: pairing fires per block and "
            "the composed module is timed against eager and Inductor."
        ),
        findings=findings,
        cells=cells,
        env=collect_env(dev),
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"e2e_model_{ts}.json"
        md_path = out_dir / f"e2e_model_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "end-to-end bench: in-repo Llama-style decoder — "
            "eager vs Inductor vs catopt (compositional + full-graph) "
            "vs catopt∘Inductor"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma subset: tiny_llama,conv_attn (default both)",
    )
    ap.add_argument(
        "--cells",
        type=str,
        default=None,
        help="tiny_llama cells as TxB list "
        "(default '64x4,128x4,256x4,128x8')",
    )
    ap.add_argument(
        "--conv-cells",
        type=str,
        default=None,
        help="conv_attn cells as HWxB list (default '16x4')",
    )
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=704)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--conv-ch", type=int, default=64)
    ap.add_argument("--conv-layers", type=int, default=2)
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=90.0,
        help="torch.compile budget per variant per cell, seconds "
        "(0 disables the compiled variants)",
    )
    ap.add_argument(
        "--full-graph",
        dest="full_graph",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-full-graph", dest="full_graph", action="store_false"
    )
    ap.add_argument(
        "--full-timeout",
        type=float,
        default=120.0,
        help="wall-clock budget for the whole-graph optimize_model "
        "variant per cell",
    )
    ap.add_argument(
        "--fp64-verify",
        dest="fp64_verify",
        action="store_true",
        default=True,
        help="second optimize_compositional pass on an fp64 copy of "
        "the model — fp64 end-to-end verify + dtype-stability check "
        "(default on; ~1-2 s per cell)",
    )
    ap.add_argument(
        "--no-fp64-verify", dest="fp64_verify", action="store_false"
    )
    ap.add_argument("--max-iterations", type=int, default=16)
    ap.add_argument("--max-enodes", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--out", default="bench/results", help="artifact dir"
    )
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()
    if args.quick:
        for k, v in QUICK.items():
            cur = getattr(args, k)
            setattr(args, k, str(v) if cur is None else type(cur)(v))
    run_bench(args)


if __name__ == "__main__":
    main()
