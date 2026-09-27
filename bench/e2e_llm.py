"""Reviewer-scale E2E bench — a ~0.4B Llama-shaped decoder, prefill
AND decode, fp16 on a 4 GB card.

``e2e_model`` answered the whole-sequence-forward question on a
256-dim toy.  This bench scales the same code shape to a deployment
plausible size and adds the two regimes a reviewer actually asks
about:

* **prefill** — one ``(1, T)`` forward, ``T ∈ {512, 2048, 8192}``
  (8192 is attempted and OOM-recorded, not assumed).
* **decode** — an autoregressive greedy loop of ``N`` new tokens at
  ``B ∈ {1, 4}`` over a REAL KV cache.  Each step is one ``(B, 1)``
  forward: query token attends a fixed-size history cache plus its
  own just-computed k/v, then the step driver writes the new k/v into
  slot ``pos`` (``index_copy_`` with a tensor position — no
  int-guard recompiles).

Why the decode blocks look unusual: ``optimize_compositional``
verifies every block with a single-tensor ``ref - out`` diff, and a
block that mutates its inputs or returns a tuple fails that gate.
So a decode block is *functional*: it takes the layer's whole
``(B, nkv, S, hd)`` history caches as inputs, attends
``cat([cache, k_new])`` under a caller-passed bool mask (fixed shape
``(1, 1, 1, S+1)`` for every step — the mask's *values* change, its
shape never does), and returns ``cat([h', k_new, v_new])`` packed
into one tensor that the ``step`` driver unpacks and writes back.
All four variants — eager, inductor, catopt, catopt+inductor — run
the identical step function, so the cache-append ``cat`` cost is
shared evenly and the comparison stays fair.

Variants per cell:

* ``eager``          — the fp16 model as built.
* ``inductor``       — ``torch.compile`` of the same module
  (SIGALRM-budgeted via ``real_win_hunt.try_compile``).
* ``catopt``         — ``optimize_compositional`` per-block, then the
  recomposed model, uncompiled.  Shape-specialized per cell: the
  optimizer runs once per (prefill T | decode B) — its wall time is
  reported as ``opt_s`` and is a real datapoint, not hidden.
* ``catopt+inductor``— ``torch.compile`` over the recomposed model —
  the realistic deployment path (catopt rewrites, inductor codegens).

Honesty conventions:

* fp16 end-to-end (4 GB card).  The parity gate is ``--verify-tol``
  (default 2e-3 — rel≈1e-3 is honest fp16); every variant's *actual*
  max-rel vs the fp16 eager reference is recorded either way.
* Only verified variants are timed; failures are recorded as
  strings (``"oom"``, ``"verify rel=..."``, compile status) rather
  than crashing the sweep.
* Decode cache init uses a real eager prefill (``prefill_caches``)
  snapshot once per cell and ``copy_``-restored before each variant,
  so every variant decodes from identical state.
* Config: ``half_b`` is d=1536/L=16/hidden 4160 (mult≈2.7), 16 q /
  4 kv heads, vocab 8192 — ≈0.43 B params.  (The brief's d=1024/L=16
  only reaches ≈0.19 B; d=1536 keeps the same shape multipliers and
  layer count while landing in the 0.4-0.6 B target band.)  On OOM
  at build/first-forward time the whole run falls back to
  ``quarter_b`` (d=1024, L=16, ≈0.19 B) and labels every record
  ``config=quarter_b (OOM fallback)``.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-optimize/src:." \\
        /tmp/catopt-cuda-venv/bin/python bench/e2e_llm.py --device cuda
"""
# ruff: noqa: E402, RUF003 -- ×, ·, → in strings are deliberate math
# notation; sys.path setup must precede benchkit/catopt imports.

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
from benchkit import Case, Cell, Report, Runner, Variant, collect_env
from catopt.optimize import optimize_compositional
from catopt_torch.report import verify_equiv
from real_win_hunt import try_compile
from torch.utils.benchmark import Timer

# run_all.py picks these up for its --quick lane.
QUICK = {
    "prefill_seq": "512",
    "decode": "1x64",
    "prompt_len": "128",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "max_iterations": "6",
    "decode_verify_steps": "8",
}

#: Model presets.  ``half_b`` is the reviewer's target (~0.4-0.6 B);
#: ``quarter_b`` is the OOM fallback; ``pico`` is a CPU-smoke config.
CONFIGS = {
    "half_b": dict(
        dim=1536,
        layers=16,
        n_heads=16,
        n_kv_heads=4,
        hidden=4160,
        vocab=8192,
    ),
    "quarter_b": dict(
        dim=1024,
        layers=16,
        n_heads=16,
        n_kv_heads=4,
        hidden=2752,
        vocab=8192,
    ),
    "pico": dict(
        dim=256,
        layers=4,
        n_heads=8,
        n_kv_heads=2,
        hidden=704,
        vocab=1024,
    ),
}


# ---------------------------------------------------------------------------
#  Model — LlamaHalfB: TinyLlama scaled, plus a functional KV-cache
#  decode path on the same block modules.
# ---------------------------------------------------------------------------


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotary (rotate-half).  ``x``: (B, H, T, hd); ``cos``/``sin``:
    (T, hd//2) — T=1 rows broadcast fine for decode steps."""
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class HalfBAttention(nn.Module):
    """GQA causal attention; ``forward`` is prefill, ``step``/``prefill_kv``
    are the decode-side paths (functional cache contract — see module
    docstring)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        self.nh, self.nkv = n_heads, n_kv_heads
        self.hd = dim // n_heads
        self.rms_w = nn.Parameter(torch.ones(dim))
        self.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, dim, bias=False)

    def _qkv(self, h: torch.Tensor):
        B, T, _ = h.shape
        n = F.rms_norm(h, (h.shape[-1],), self.rms_w, 1e-5)
        q = self.wq(n).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(n).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(n).view(B, T, self.nkv, self.hd).transpose(1, 2)
        return q, k, v

    def _expand_kv(self, t: torch.Tensor, B: int, T: int):
        return (
            t[:, :, None, :, :]
            .expand(B, self.nkv, self.nh // self.nkv, T, self.hd)
            .reshape(B, self.nh, T, self.hd)
        )

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Prefill: full causal attention over the (B, T) prompt."""
        B, T, _ = h.shape
        q, k, v = self._qkv(h)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q,
            self._expand_kv(k, B, T),
            self._expand_kv(v, B, T),
            is_causal=True,
        )
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))

    def prefill_kv(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
    ) -> torch.Tensor:
        """Eager-only cache initializer: writes slots ``[:T]`` of the
        layer's caches while computing the same prefill attention."""
        B, T, _ = h.shape
        q, k, v = self._qkv(h)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k_c[:, :, :T] = k
        v_c[:, :, :T] = v
        o = F.scaled_dot_product_attention(
            q,
            self._expand_kv(k, B, T),
            self._expand_kv(v, B, T),
            is_causal=True,
        )
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))

    def step(
        self,
        h: torch.Tensor,
        cos_row: torch.Tensor,
        sin_row: torch.Tensor,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        amask: torch.Tensor,
    ):
        """Decode one token.  ``h``: (B,1,d); ``k_c``/``v_c``:
        (B,nkv,S,hd) history; ``amask``: (1,1,1,S+1) bool (True =
        allowed).  Returns ``(attn_out, k_new, v_new)`` — the driver
        writes k_new/v_new into the caches afterwards."""
        B = h.shape[0]
        q, k, v = self._qkv(h)  # (B,nh/nkv,1,hd)
        q = _apply_rope(q, cos_row, sin_row)
        k = _apply_rope(k, cos_row, sin_row)
        S = k_c.shape[2]
        k_all = torch.cat([k_c, k], dim=2)  # (B,nkv,S+1,hd)
        v_all = torch.cat([v_c, v], dim=2)
        o = F.scaled_dot_product_attention(
            q,
            self._expand_kv(k_all, B, S + 1),
            self._expand_kv(v_all, B, S + 1),
            attn_mask=amask,
        )
        return (
            self.wo(o.transpose(1, 2).reshape(B, 1, -1)),
            k,
            v,
        )


class LlamaHalfBBlock(nn.Module):
    """Decoder block, dual-signature:

    * ``forward(h, cos, sin)`` — prefill, returns ``(B,T,d)``.
    * ``forward(h, cos_row, sin_row, k_c, v_c, amask)`` — decode step,
      returns ONE packed tensor ``(B, d + 2·nkv·hd)`` =
      ``cat([h', k_new, v_new])``.  Single-tensor out is what
      ``optimize_compositional``'s per-block verifier requires.
    """

    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, hidden: int
    ) -> None:
        super().__init__()
        self.dim = dim
        self.attn = HalfBAttention(dim, n_heads, n_kv_heads)
        self.rms_ffn = nn.Parameter(torch.ones(dim))
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        n = F.rms_norm(h, (self.dim,), self.rms_ffn, 1e-5)
        return h + self.w2(F.silu(self.w1(n)) * self.w3(n))

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_c: torch.Tensor | None = None,
        v_c: torch.Tensor | None = None,
        amask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if k_c is None:
            return self._mlp(h + self.attn(h, cos, sin))
        # Decode step — functional, packed single-tensor output.
        B = h.shape[0]
        attn_out, k_new, v_new = self.attn.step(
            h, cos, sin, k_c, v_c, amask
        )
        h2 = self._mlp(h + attn_out)
        return torch.cat(
            [
                h2.reshape(B, -1),
                k_new.reshape(B, -1),
                v_new.reshape(B, -1),
            ],
            dim=-1,
        )

    def prefill_kv(self, h, cos, sin, k_c, v_c):
        return self._mlp(
            h + self.attn.prefill_kv(h, cos, sin, k_c, v_c)
        )


class LlamaHalfB(nn.Module):
    """Embedding → L blocks → RMSNorm → LM head; plus ``step`` (one
    decode token over caller-owned KV caches) and ``prefill_caches``
    (eager cache initializer, not part of any timed path)."""

    def __init__(
        self,
        dim: int = 1536,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        hidden: int = 4160,
        layers: int = 16,
        vocab: int = 8192,
        max_seq: int = 8192,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.dim, self.nkv, self.hd, self.vocab = (
            dim,
            n_kv_heads,
            dim // n_heads,
            vocab,
        )
        self.tok_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            LlamaHalfBBlock(dim, n_heads, n_kv_heads, hidden)
            for _ in range(layers)
        )
        self.rms_final = nn.Parameter(torch.ones(dim))
        self.head = nn.Linear(dim, vocab, bias=False)
        hd = self.hd
        freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(max_seq).float(), freqs)
        self.register_buffer("cos", fr.cos())
        self.register_buffer("sin", fr.sin())
        self.to(dtype)

    # -- prefill ---------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[-1]
        h = self.tok_emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            h = b(h, cos, sin)
        h = F.rms_norm(h, (self.dim,), self.rms_final, 1e-5)
        return self.head(h)

    # -- decode ----------------------------------------------------------
    def step(
        self,
        tok: torch.Tensor,  # (B, 1) token ids
        pos_t: torch.Tensor,  # (1,) int64 — tensor, not int: no
        # dynamo value-guard recompiles across steps
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        """One autoregressive step → next-token logits (B,1,vocab)."""
        B = tok.shape[0]
        d, kvd = self.dim, self.nkv * self.hd
        S = k_caches[0].shape[2]
        h = self.tok_emb(tok)  # (B,1,d)
        cos_row = self.cos.index_select(0, pos_t)  # (1, hd//2)
        sin_row = self.sin.index_select(0, pos_t)
        valid = torch.cat(
            [
                torch.arange(S, device=tok.device) < pos_t,
                torch.ones(1, dtype=torch.bool, device=tok.device),
            ]
        )
        amask = valid.view(1, 1, 1, S + 1)
        for i, b in enumerate(self.blocks):
            packed = b(
                h, cos_row, sin_row, k_caches[i], v_caches[i], amask
            )
            h = packed[:, :d].view(B, 1, d)
            kn = packed[:, d : d + kvd].view(B, self.nkv, 1, self.hd)
            vn = packed[:, d + kvd :].view(B, self.nkv, 1, self.hd)
            # Driver-side cache write — outside the optimized block.
            k_caches[i].index_copy_(2, pos_t, kn)
            v_caches[i].index_copy_(2, pos_t, vn)
        h = F.rms_norm(h, (d,), self.rms_final, 1e-5)
        return self.head(h)

    @torch.no_grad()
    def prefill_caches(
        self,
        idx: torch.Tensor,
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        """Real-prompt cache init (eager path only — recomposed decode
        blocks can't run prefill, so all variants share this init)."""
        T = idx.shape[-1]
        h = self.tok_emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for i, b in enumerate(self.blocks):
            h = b.prefill_kv(h, cos, sin, k_caches[i], v_caches[i])
        h = F.rms_norm(h, (self.dim,), self.rms_final, 1e-5)
        return self.head(h)


class DecodeStep(nn.Module):
    """``model.step`` as an ``nn.Module.forward`` so
    ``optimize_compositional`` / ``torch.compile`` can drive it.
    Caches stay driver-owned list inputs (not parameters/buffers)."""

    def __init__(self, model: LlamaHalfB) -> None:
        super().__init__()
        self.m = model

    def forward(self, tok, pos_t, k_caches, v_caches):
        return self.m.step(tok, pos_t, k_caches, v_caches)


# ---------------------------------------------------------------------------
#  Harness helpers
# ---------------------------------------------------------------------------


def _informative_fires(fires: dict) -> dict:
    drop = ("comm_", "assoc_add", "id_add", "id_mul")
    return {
        k: v
        for k, v in (fires or {}).items()
        if v and not k.startswith(drop)
    }


def _peak_mem(fn, calls: int = 4) -> float:
    """Peak CUDA allocator MiB over ``calls`` invocations."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def _time_stmt(stmt, min_run_time: float, warmup: int, cuda: bool):
    """blocked_autorange on ``stmt`` → (median_s, iqr_s) or raises."""
    if cuda:

        def wrapped():
            stmt()
            torch.cuda.synchronize()
            gc.collect(0)

        fn = wrapped
    else:
        fn = stmt
    for _ in range(max(warmup, 0)):
        fn()
    meas = Timer(stmt="_fn()", globals={"_fn": fn}).blocked_autorange(
        min_run_time=min_run_time
    )
    return meas.median, meas.iqr


def _oom(e: BaseException) -> bool:
    return isinstance(
        e, (torch.cuda.OutOfMemoryError, MemoryError)
    ) or ("out of memory" in str(e).lower())


def _alloc_caches(model: LlamaHalfB, B: int, S: int, dev: torch.device):
    shape = (B, model.nkv, S, model.hd)
    dt = model.head.weight.dtype  # fp16 on cuda, fp32 on cpu
    ks = [
        torch.zeros(shape, device=dev, dtype=dt) for _ in model.blocks
    ]
    vs = [
        torch.zeros(shape, device=dev, dtype=dt) for _ in model.blocks
    ]
    return ks, vs


def _decode_loop(step_fn, tok0, pos_list, k_caches, v_caches):
    """Greedy autoregressive loop → last-step logits.  ``no_grad``
    lives INSIDE — the timed call sites would otherwise build a
    128-step autograd graph and OOM."""
    tok = tok0
    logits = None
    with torch.no_grad():
        for pos_t in pos_list:
            logits = step_fn(tok, pos_t, k_caches, v_caches)
            tok = logits[:, -1].argmax(-1, keepdim=True)
    return logits


def _decode_collect(step_fn, tok0, pos_list, k_caches, v_caches):
    """Free-running greedy loop → (per-step logits, per-step tokens)."""
    tok = tok0
    outs, toks = [], []
    with torch.no_grad():
        for pos_t in pos_list:
            logits = step_fn(tok, pos_t, k_caches, v_caches)
            tok = logits[:, -1].argmax(-1, keepdim=True)
            outs.append(logits[:, -1])
            toks.append(tok)
    return outs, toks


def _decode_teacher(
    step_fn, tok0, pos_list, forced, k_caches, v_caches
):
    """Teacher-forced probe — feeds the EAGER token stream so per-step
    logit diffs measure kernel parity, not chaotic argmax divergence
    on fp16 near-ties.  → per-step logits list."""
    tok = tok0
    outs = []
    with torch.no_grad():
        for i, pos_t in enumerate(pos_list):
            logits = step_fn(tok, pos_t, k_caches, v_caches)
            outs.append(logits[:, -1])
            tok = forced[i]
    return outs


def _restore(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
    for d, s in zip(dst, src, strict=True):
        d.copy_(s)


# ---------------------------------------------------------------------------
#  Cells
# ---------------------------------------------------------------------------


def _catopt(mod: nn.Module, example, args, rec: dict):
    """``optimize_compositional`` + mechanism ledger → recomposed module.

    Returns ``None`` when nothing was actually delivered: an exception,
    or the ``in_place`` fallback — ``optimize_compositional`` grafts
    optimized blocks into a ``deepcopy`` of the model; if that copy
    OOMs (the 4 GB card's whole budget is ~3.7 GB) the driver silently
    returns the ORIGINAL module with ``stats["in_place"]=True`` and
    ``n_optimized=0`` while every block still reports
    ``status="optimized"``.  Timing that module would just re-time
    eager, so it is excluded and recorded honestly instead.
    """
    max_iter = int(getattr(args, "max_iterations", None) or 8)
    max_en = int(getattr(args, "max_enodes", None) or 100_000)
    verify_tol = float(getattr(args, "verify_tol", None) or 2e-3)
    verbose = bool(getattr(args, "verbose", False))
    t0 = time.time()
    try:
        opt, rep = optimize_compositional(
            mod,
            example,
            verbose=verbose,
            max_iterations=max_iter,
            max_enodes=max_en,
            verify_tol=verify_tol,
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
    for be in rep["blocks"].values():
        st = be.get("stats") or {}
        for k, v in _informative_fires(
            st.get("rule_fires", {})
        ).items():
            fires[k] = fires.get(k, 0) + v
        paired += 1 if st.get("paired_extract") else 0
    rec["fires_total"] = fires or None
    rec["paired_blocks"] = paired
    e2e = rep.get("end_to_end") or {}
    rec["catopt_driver_e2e_rel"] = e2e.get("max_rel_diff")
    print(
        f"  catopt: {rec['n_optimized']}/{rec['n_blocks']} blocks "
        f"in {rec['opt_s']:.1f}s  paired={paired} "
        f"in_place={rec['in_place']}",
        flush=True,
    )
    if rec["in_place"] or rec["n_optimized"] == 0:
        # Nothing grafted — the returned module IS the original.
        rec["catopt_status"] = (
            "in_place fallback: graft deepcopy failed "
            "(CUDA OOM at this param count) — returned original; "
            "catopt variants excluded"
        )
        del opt
        return None
    return opt


def run_prefill_cell(
    model: LlamaHalfB, T: int, args, dev: torch.device, cuda: bool
) -> Cell:
    """One prefill cell: single (1, T) forward across all variants."""
    rec: dict = {
        "kind": "prefill",
        "T": T,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    print(f"\n=== prefill T={T} ===", flush=True)
    g = torch.Generator().manual_seed(1234 + T)
    idx = torch.randint(0, model.vocab, (1, T), generator=g).to(dev)
    verify_tol = float(getattr(args, "verify_tol", None) or 2e-3)
    cto = float(getattr(args, "compile_timeout", None) or 90.0)

    try:
        with torch.no_grad():
            ref = model(idx)
    except Exception as e:
        rec["status"] = "oom" if _oom(e) else f"err:{e}"
        rec["params"] = {"kind": "prefill", "T": T}
        print(f"  eager forward failed: {rec['status']}", flush=True)
        return _rec_to_cell(f"prefill_T{T}", rec)

    opt_c = _catopt(model, idx, args, rec)
    if opt_c is not None:
        with torch.no_grad():
            vr = verify_equiv(ref, opt_c(idx), rtol=verify_tol)
        rec["catopt_rel"] = vr.max_rel
        rec["catopt_verified"] = bool(vr.passed)
        print(f"  catopt e2e rel={vr.max_rel:.2e}", flush=True)

    cm = oci = None
    t0 = time.time()
    cm, rec["inductor_status"] = try_compile(model, (idx,), cto)
    rec["inductor_compile_s"] = round(time.time() - t0, 2)
    print(
        f"  inductor: {rec['inductor_status']} "
        f"({rec['inductor_compile_s']:.1f}s)",
        flush=True,
    )
    if cm is not None:
        try:
            with torch.no_grad():
                vr = verify_equiv(ref, cm(idx), rtol=verify_tol)
            rec["inductor_rel"] = vr.max_rel
            rec["inductor_verified"] = bool(vr.passed)
            if not vr.passed:
                cm = None
        except Exception as e:
            rec["inductor_rel"] = f"err:{type(e).__name__}"
            cm = None
    if opt_c is not None and rec.get("catopt_verified"):
        t0 = time.time()
        oci, rec["catopt+inductor_status"] = try_compile(
            opt_c, (idx,), cto
        )
        rec["catopt+inductor_compile_s"] = round(time.time() - t0, 2)
        print(
            f"  catopt+inductor: {rec['catopt+inductor_status']} "
            f"({rec['catopt+inductor_compile_s']:.1f}s)",
            flush=True,
        )
        if oci is not None:
            try:
                with torch.no_grad():
                    vr = verify_equiv(ref, oci(idx), rtol=verify_tol)
                rec["catopt+inductor_rel"] = vr.max_rel
                rec["catopt+inductor_verified"] = bool(vr.passed)
                if not vr.passed:
                    oci = None
            except Exception as e:
                rec["catopt+inductor_rel"] = f"err:{type(e).__name__}"
                oci = None
    if opt_c is not None and not rec.get("catopt_verified"):
        opt_c = None  # wrong program is never timed

    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    variants = [
        ("eager", model),
        ("inductor", cm),
        ("catopt", opt_c),
        ("catopt+inductor", oci),
    ]
    min_run = float(getattr(args, "min_run_time", None) or 0.2)
    warmup = int(getattr(args, "warmup", None) or 3)
    for name, m in variants:
        if m is None:
            rec[f"{name}_status"] = "absent"
            continue

        def stmt(m=m):
            with torch.no_grad():
                m(idx)

        try:
            rec[f"peak_mb_{name}"] = (
                round(_peak_mem(stmt), 1) if cuda else None
            )
            med, iqr = _time_stmt(stmt, min_run, warmup, cuda)
            medians[name] = med
            iqrs[name] = iqr
            rec[f"tok_s_{name}"] = round(T / med, 1)
        except Exception as e:
            rec[f"{name}_status"] = (
                "oom" if _oom(e) else f"err:{type(e).__name__}: {e}"
            )
            if cuda:
                torch.cuda.empty_cache()
    print(
        "  times: "
        + "  ".join(f"{n}={medians[n] * 1e3:.2f}ms" for n in medians),
        flush=True,
    )
    rec["params"] = {"kind": "prefill", "T": T}
    return _rec_to_cell(f"prefill_T{T}", rec, medians, iqrs)


def run_decode_cell(
    model: LlamaHalfB,
    B: int,
    N: int,
    args,
    dev: torch.device,
    cuda: bool,
) -> Cell:
    """One decode cell: N-token greedy loop at batch B, real KV cache."""
    P = int(getattr(args, "prompt_len", None) or 512)
    S = P + N
    rec: dict = {
        "kind": "decode",
        "B": B,
        "N": N,
        "prompt_len": P,
        "ctx": S,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    print(
        f"\n=== decode B={B} N={N} (prompt={P}, ctx={S}) ===",
        flush=True,
    )
    verify_tol = float(getattr(args, "verify_tol", None) or 2e-3)
    cto = float(getattr(args, "compile_timeout", None) or 90.0)
    kverify = int(getattr(args, "decode_verify_steps", None) or 16)

    g = torch.Generator().manual_seed(777 + B)
    prompt = torch.randint(0, model.vocab, (B, P), generator=g).to(dev)
    try:
        k_caches, v_caches = _alloc_caches(model, B, S, dev)
        with torch.no_grad():
            plogits = model.prefill_caches(prompt, k_caches, v_caches)
        tok0 = plogits[:, -1].argmax(-1, keepdim=True)
        # Snapshot post-prefill state — every variant restores it.
        k_snap = [c.clone() for c in k_caches]
        v_snap = [c.clone() for c in v_caches]
    except Exception as e:
        rec["status"] = "oom" if _oom(e) else f"err:{e}"
        rec["params"] = {"kind": "decode", "B": B, "N": N, "ctx": S}
        print(f"  cache init failed: {rec['status']}", flush=True)
        return _rec_to_cell(f"decode_B{B}x{N}", rec)

    pos_list = [
        torch.tensor([P + t], device=dev, dtype=torch.long)
        for t in range(N)
    ]
    pos0 = pos_list[0]
    wrapper = DecodeStep(model).eval()
    step_args = (tok0, pos0, k_caches, v_caches)

    # -- eager decode reference: free-running token stream + per-step
    #    logits, used for teacher-forced parity checks ------------------
    try:
        _restore(k_caches, k_snap)
        _restore(v_caches, v_snap)
        ref_outs, ref_toks = _decode_collect(
            wrapper, tok0, pos_list[:kverify], k_caches, v_caches
        )
    except Exception as e:
        rec["status"] = "oom" if _oom(e) else f"err:{e}"
        rec["params"] = {"kind": "decode", "B": B, "N": N, "ctx": S}
        print(f"  eager decode failed: {rec['status']}", flush=True)
        return _rec_to_cell(f"decode_B{B}x{N}", rec)

    def verify_decode(mod, name: str) -> bool:
        """Parity vs the eager decode.

        Gate = teacher-forced per-step logit rel < verify_tol (kernel
        parity on identical inputs — the honest fp16 check).  Two aux
        datapoints are recorded, not gated: ``argmax_mismatch`` (steps
        where the argmax would differ given identical history — fp16
        near-ties) and ``tok_mismatch`` (free-running divergence —
        chaotic once any argmax flips)."""
        try:
            _restore(k_caches, k_snap)
            _restore(v_caches, v_snap)
            outs = _decode_teacher(
                mod,
                tok0,
                pos_list[:kverify],
                ref_toks,
                k_caches,
                v_caches,
            )
            max_rel = 0.0
            amism = 0
            for o, r, rt in zip(outs, ref_outs, ref_toks, strict=True):
                max_rel = max(
                    max_rel,
                    verify_equiv(r, o, rtol=verify_tol).max_rel,
                )
                amism += int((o.argmax(-1, keepdim=True) != rt).sum())
            rec[f"{name}_rel"] = max_rel
            rec[f"{name}_argmax_mismatch@{kverify}"] = amism
            # Free-running divergence as an info datapoint.
            _restore(k_caches, k_snap)
            _restore(v_caches, v_snap)
            _, tks = _decode_collect(
                mod, tok0, pos_list[:kverify], k_caches, v_caches
            )
            rec[f"{name}_tok_mismatch@{kverify}"] = sum(
                int((a != b).sum())
                for a, b in zip(tks, ref_toks, strict=True)
            )
            ok = max_rel < verify_tol
            rec[f"{name}_verified"] = ok
            return ok
        except Exception as e:
            rec[f"{name}_status"] = (
                "oom" if _oom(e) else f"verify:{type(e).__name__}: {e}"
            )
            rec[f"{name}_verified"] = False
            if cuda:
                torch.cuda.empty_cache()
            return False

    # -- catopt: optimize_compositional over the step module ----------
    opt_w = _catopt(wrapper, step_args, args, rec)
    if opt_w is not None and not verify_decode(opt_w, "catopt"):
        opt_w = None

    # -- inductor + composed -------------------------------------------
    cm = oci = None
    _restore(k_caches, k_snap)
    _restore(v_caches, v_snap)
    t0 = time.time()
    cm, rec["inductor_status"] = try_compile(wrapper, step_args, cto)
    rec["inductor_compile_s"] = round(time.time() - t0, 2)
    print(
        f"  inductor: {rec['inductor_status']} "
        f"({rec['inductor_compile_s']:.1f}s)",
        flush=True,
    )
    if cm is not None and not verify_decode(cm, "inductor"):
        cm = None
    if opt_w is not None:
        _restore(k_caches, k_snap)
        _restore(v_caches, v_snap)
        t0 = time.time()
        oci, rec["catopt+inductor_status"] = try_compile(
            opt_w, step_args, cto
        )
        rec["catopt+inductor_compile_s"] = round(time.time() - t0, 2)
        print(
            f"  catopt+inductor: {rec['catopt+inductor_status']} "
            f"({rec['catopt+inductor_compile_s']:.1f}s)",
            flush=True,
        )
        if oci is not None and not verify_decode(
            oci, "catopt+inductor"
        ):
            oci = None

    # -- peak memory + timing: full N-token loop per variant ----------
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    variants = [
        ("eager", wrapper),
        ("inductor", cm),
        ("catopt", opt_w),
        ("catopt+inductor", oci),
    ]
    min_run = float(getattr(args, "min_run_time", None) or 0.2)
    warmup = int(getattr(args, "warmup", None) or 2)
    for name, m in variants:
        if m is None:
            rec.setdefault(f"{name}_status", "absent")
            continue

        def stmt(m=m):
            _decode_loop(m, tok0, pos_list, k_caches, v_caches)

        try:
            _restore(k_caches, k_snap)
            _restore(v_caches, v_snap)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            stmt()  # warm + fills caches
            if cuda:
                torch.cuda.synchronize()
                rec[f"peak_mb_{name}"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 1
                )
            med, iqr = _time_stmt(stmt, min_run, warmup, cuda)
            medians[name] = med
            iqrs[name] = iqr
            rec[f"tok_s_{name}"] = round(B * N / med, 1)
            rec[f"tok_ms_{name}"] = round(med / N * 1e3, 3)
        except Exception as e:
            rec[f"{name}_status"] = (
                "oom" if _oom(e) else f"err:{type(e).__name__}: {e}"
            )
            if cuda:
                torch.cuda.empty_cache()
    print(
        "  times: "
        + "  ".join(
            f"{n}={medians[n] * 1e3:.1f}ms/{N}tok" for n in medians
        ),
        flush=True,
    )
    rec["params"] = {"kind": "decode", "B": B, "N": N, "ctx": S}
    del wrapper, opt_w, cm, oci
    return _rec_to_cell(f"decode_B{B}x{N}", rec, medians, iqrs)


def _rec_to_cell(name: str, rec: dict, medians=None, iqrs=None) -> Cell:
    """Pack a record into a benchkit Cell (medians keyed per variant)."""
    case = Case(
        name=name,
        params=rec.pop("params", {}),
        variants=[
            Variant(n, Runner._released)  # stmt never stored
            for n in (medians or {})
        ],
        aux=rec,
    )
    return Cell(
        case=case,
        medians=dict(medians or {}),
        iqr=dict(iqrs or {}),
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    cuda = dev.type == "cuda"
    torch.manual_seed(0)

    # Default config is device-aware: half_b (0.43 B) targets the 4 GB
    # card (with the compositional preflight fallback to quarter_b);
    # pico keeps a CPU --quick lane a smoke test, not a marathon.
    cfg_name = getattr(args, "config", None) or (
        "pico" if dev.type == "cpu" else "half_b"
    )
    cfg = dict(CONFIGS[cfg_name])
    for k in (
        "dim",
        "layers",
        "n_heads",
        "n_kv_heads",
        "hidden",
        "vocab",
    ):
        ov = getattr(args, k, None)
        if ov is not None:
            cfg[k] = int(ov)
    fallback = ""

    def _build():
        m = LlamaHalfB(
            max_seq=8192,
            dtype=torch.float16 if cuda else torch.float32,
            **cfg,
        )
        return m.to(dev).eval()

    model = _build()
    n_params = sum(p.numel() for p in model.parameters())

    def _first_forward_ok() -> bool:
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model(
                    torch.randint(0, cfg["vocab"], (1, 16), device=dev)
                )
            torch.cuda.synchronize()
            return True
        except Exception as e:
            return _oom(e)

    def _compositional_fits() -> bool:
        """Cheap preflight: can optimize_compositional deliver at this
        size?  The graft deepcopy OOMs the driver into ``in_place`` on
        a 4 GB card once the model gets past ~0.35 B fp16 — probe it
        once here rather than discovering it inside the first cell."""
        probe = torch.randint(0, cfg["vocab"], (1, 64), device=dev)
        pre: dict = {}
        opt = _catopt(model, probe, args, pre)
        if opt is not None:
            del opt
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        ok = not pre.get("in_place") and pre.get("n_optimized", 0) > 0
        if not ok:
            print(
                "  preflight: compositional could not deliver at "
                f"{cfg_name} ({n_params / 1e6:.0f}M params, "
                f"in_place={pre.get('in_place')} — graft deepcopy "
                "OOMs past ~3.4 GB peak)",
                flush=True,
            )
        return ok

    if cuda and not _first_forward_ok():
        if cfg_name != "half_b":
            raise torch.cuda.OutOfMemoryError(
                f"cannot even forward {cfg_name} fp16"
            )
        print(
            "  half_b OOM on first forward — falling back to quarter_b",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        cfg_name, cfg = "quarter_b", dict(CONFIGS["quarter_b"])
        model = _build()
        n_params = sum(p.numel() for p in model.parameters())
        fallback = " (OOM fallback: half_b forward)"
    elif cuda and cfg_name == "half_b" and not _compositional_fits():
        del model
        gc.collect()
        torch.cuda.empty_cache()
        cfg_name, cfg = "quarter_b", dict(CONFIGS["quarter_b"])
        model = _build()
        n_params = sum(p.numel() for p in model.parameters())
        fallback = (
            " (OOM fallback: half_b compositional graft deepcopy "
            "exceeded 4 GB)"
        )
    print(
        f"e2e_llm — config={cfg_name}{fallback} "
        f"params={n_params / 1e6:.1f}M dtype="
        f"{'fp16' if cuda else 'fp32'} device={dev}",
        flush=True,
    )

    cells: list[Cell] = []
    t_all = time.perf_counter()
    prefill_seqs = [
        int(s)
        for s in (
            getattr(args, "prefill_seq", None) or "512,2048,8192"
        ).split(",")
        if s.strip()
    ]
    decode_cells = []
    for tok in (getattr(args, "decode", None) or "1x128,4x128").split(
        ","
    ):
        b_s, n_s = tok.strip().lower().split("x")
        decode_cells.append((int(b_s), int(n_s)))

    for T in prefill_seqs:
        cells.append(run_prefill_cell(model, T, args, dev, cuda))
        gc.collect()
        torch._dynamo.reset()
        if cuda:
            torch.cuda.empty_cache()
    for B, N in decode_cells:
        cells.append(run_decode_cell(model, B, N, args, dev, cuda))
        gc.collect()
        torch._dynamo.reset()
        if cuda:
            torch.cuda.empty_cache()

    # -- console summary --------------------------------------------------
    names = ["eager", "inductor", "catopt", "catopt+inductor"]
    hdr = (
        f"{'cell':<16} | "
        + " | ".join(f"{n:>14}" for n in names)
        + f" | {'c+i/ind':>7}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for cell in cells:
        ms = cell.medians
        unit = (
            "ms/fwd"
            if cell.aux.get("kind") == "prefill"
            else f"ms/{cell.aux.get('N', '?')}tok"
        )
        row = f"{cell.case.name:<16} | " + " | ".join(
            (
                f"{ms[n] * 1e3:>8.2f} {unit:<9}"
                if n in ms
                else f"{'—':>14}"
            )
            for n in names
        )
        mi, mc = ms.get("inductor"), ms.get("catopt+inductor")
        ratio = f"{mi / mc:>7.3f}" if mi and mc else f"{'—':>7}"
        print(f"{row} | {ratio}")
        a = cell.aux
        if a.get("status"):
            print(f"{'':<16} | STATUS: {a['status']}")
            continue
        toks = "  ".join(
            f"{n}={a.get(f'tok_s_{n}', '—')}"
            for n in ("eager", "catopt+inductor")
        )
        print(
            f"{'':<16} | tok/s {toks}  opt={a.get('opt_s', '—')}s"
            f"  ind_compile={a.get('inductor_compile_s', '—')}s"
            f"  rel: cat={_fmt_rel(a.get('catopt_rel'))}"
            f" ind={_fmt_rel(a.get('inductor_rel'))}"
            f" c+i={_fmt_rel(a.get('catopt+inductor_rel'))}",
            flush=True,
        )
    print("-" * len(hdr))
    print(
        "  c+i/ind > 1 = catopt-composed faster than plain inductor.  "
        "Reviewer bands: 1.05-1.3 typical, 1.3-1.7 favorable, "
        "~1.0 unfavorable."
    )
    print(
        f"  total wall time {time.perf_counter() - t_all:.1f}s",
        flush=True,
    )

    env = collect_env(dev)
    env["config"] = cfg_name + fallback
    env["n_params"] = n_params
    env["notes"] = [
        "decode = real KV cache; blocks are functional "
        "(cat[cache,k_new] attention + packed [h|k|v] output) so "
        "optimize_compositional's single-tensor verify applies; "
        "the driver does index_copy_ writes between block calls. "
        "The cat/pack overhead is identical across variants.",
        "decode verify gate = teacher-forced per-step logit rel "
        f"< {float(getattr(args, 'verify_tol', None) or 2e-3):.0e} "
        "(kernel parity); *_tok_mismatch is free-running argmax "
        "divergence (chaotic on fp16 near-ties) — reported, "
        "not gated.",
        "inductor compile times reuse the persistent inductor FX "
        "cache — reruns are warm (~1-2 s); cold compile measured "
        "~7-12 s per variant on this card.",
        "catopt(uncompiled) is the raw IRModule executor — its "
        "per-node eval overhead is included; catopt+inductor is the "
        "deployment path.",
    ]
    report = Report(suite="e2e_llm", cells=cells, env=env)
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"e2e_llm_{ts}.json"
        md_path = out_dir / f"e2e_llm_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def _fmt_rel(v) -> str:
    return f"{v:.1e}" if isinstance(v, float) else str(v or "—")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "reviewer-scale e2e: ~0.4B Llama decoder — prefill + "
            "KV-cache decode, eager vs inductor vs catopt vs composed"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--config",
        default=None,
        choices=list(CONFIGS),
        help="default: half_b on cuda (auto-fallback to quarter_b "
        "on OOM), pico on cpu",
    )
    for k in (
        "dim",
        "layers",
        "n_heads",
        "n_kv_heads",
        "hidden",
        "vocab",
    ):
        ap.add_argument(
            f"--{k.replace('_', '-')}", dest=k, type=int, default=None
        )
    ap.add_argument(
        "--prefill-seq",
        dest="prefill_seq",
        type=str,
        default=None,
        help="comma T list (default 512,2048,8192)",
    )
    ap.add_argument(
        "--decode",
        dest="decode",
        type=str,
        default=None,
        help="comma BxN list (default 1x128,4x128)",
    )
    ap.add_argument(
        "--prompt-len", dest="prompt_len", type=int, default=512
    )
    ap.add_argument(
        "--decode-verify-steps",
        dest="decode_verify_steps",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--verify-tol",
        dest="verify_tol",
        type=float,
        default=2e-3,
        help="fp16-honest rel gate (default 2e-3)",
    )
    ap.add_argument(
        "--compile-timeout",
        dest="compile_timeout",
        type=float,
        default=90.0,
    )
    ap.add_argument(
        "--max-iterations", dest="max_iterations", type=int, default=8
    )
    ap.add_argument(
        "--max-enodes", dest="max_enodes", type=int, default=100_000
    )
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument(
        "--min-run-time", dest="min_run_time", type=float, default=None
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
