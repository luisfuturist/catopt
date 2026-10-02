"""Whole-model benchmark — complete multi-block models, not cells.

``real_win_hunt`` asks where catopt wins on *representative blocks*.
This bench asks the deployment question: does the win survive when the
block is embedded in a COMPLETE model — stacked residual blocks whose
graphs share an accumulator, plus heads/stems — measured against eager
and TorchInductor on wall-clock, peak memory, compile/optimize time,
and correctness?

Models (all small but structurally real, torch nn code that exports
cleanly):

* ``tiny_decoder`` — a 4-block decoder: ``x + attn(rms(x))``;
  ``x + swiglu(rms(x))`` per block with *causal* multi-head SDPA
  (separate q/k/v projections — the pairing triple), a final RMSNorm,
  and a vocab head.  Mechanisms: product-law pairing (q/k/v → 1 GEMM,
  gate/up → 1 GEMM) and channel-gain norm folds into the following
  weights, ×4 blocks sharing one residual accumulator.
* ``tiny_ssm`` — a 2-block gated linear-attention stack
  (``real_linear_attn.LinearAttnStack``, retnet mode) on the
  UNBATCHED ``(T, d)`` scan form: block 1 emits a gated sequence,
  block 2 scans to a final state.  Mechanism: the affine-carrier
  scan lift → level-batched executor + weight-chain fold.  Known
  caveat (whole-graph saturation on multi-block emitted-seq graphs is
  the known cost cliff)
  on multi-block emitted-seq graphs is the known cost cliff — T is
  kept small and the search wall time recorded honestly, bounded by
  ``--max-enodes`` / ``--budget-s``.
* ``tiny_moe`` — a 3-block dense-MoE (model-soup) stack: each block
  computes ``x + down(silu(Σ_i gate_i(rms x)))`` over E=6 experts.
  Mechanism: ``weight_factor_linear`` folds each expert sum into ONE
  matmul — the 7-9× FLOP-cut win from ``moe_sum``, now per block with
  a nonlinearity and residual in the way.
* ``tiny_convnet`` — conv stem (4-branch 1×1 ``ParallelConv`` → conv
  pairing) + 2 PaLM ``ParallelBlock`` attention blocks on the
  flattened feature map + pooled head.  Mechanisms: conv pairing plus
  qkv/gate·up pairing — a mixed-domain model.

Protocol per model (same contract as ``real_win_hunt``):

1. ``optimize_model`` on the fp64 module → fp64 verify
   (``rel_to_max``); records search wall time, lowering, pairing
   groups, nonlocal lifts, and informative rule fires.
2. ``optimize_model_autotuned`` on the fp32 module — every lowering
   candidate (``eager,generic,batched,compiled,compiled_generic``,
   plus ``cuda_graph`` on CUDA) is rebuilt, ``sink.verify``-gated,
   and timed; ``catopt_best`` is the measured winner.
3. SIGALRM-guarded ``torch.compile`` baseline on a deepcopy (compile
   wall time recorded).
4. benchkit ``Runner`` times ``eager`` / ``inductor`` /
   ``catopt_best`` (median + IQR).  On CUDA each variant additionally
   gets a ``torch.cuda.max_memory_allocated`` peak after
   ``reset_peak_memory_stats``.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/model_bench.py \
        --device cuda
    .venv/bin/python bench/model_bench.py --device cpu --quick
"""
# ruff: noqa: E402 RUF001 RUF002 RUF003 -- ×, ·, Σ in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.setrecursionlimit(400_000)

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import (
    Autotuned,
    OptimizationResourceError,
)
from catopt_torch.autotune import TORCH_BUILDERS
from catopt_torch.backend import TorchBackend
from catopt_torch.models import ParallelBlock, ParallelConv

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)
from bench.suites.speedup.real_linear_attn import LinearAttnStack
from bench.suites.speedup.real_win_hunt import _rel_diff, try_compile

# run_all.py picks these up for its --quick lane.  The decoder cell is
# dropped from --quick: the paired-DAG ``dag_cost`` pass costs ~75s per
# search (×2: fp64 + fp32 autotune) regardless of --max-iterations.
QUICK = {
    "models": "tiny_ssm,tiny_moe,tiny_convnet",
    "calls": "12",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "max_iterations": "8",
    "budget_s": "150",
}


# ---------------------------------------------------------------------------
#  Models — complete multi-block stacks
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with SEPARATE q/k/v projections.

    Separate projections on the same normed input are what real
    decoder code looks like pre-fusion — and what the pairing pass
    folds into one GEMM.  ``is_causal=True`` keeps the block a real
    decoder layer (the bridge carries the flag through ``sdpa``).
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, C))


class DecoderBlock(nn.Module):
    """``x + attn(rms x)``; ``x + swiglu(rms x)`` — the GPT-2/RMSNorm
    decoder block.  Per block the pairing pass sees two same-input
    projection sets (q/k/v and gate/up) plus two foldable channel
    gains."""

    def __init__(
        self, dim: int, n_heads: int, hidden_mult: int = 2
    ) -> None:
        super().__init__()
        self.norm1_w = nn.Parameter(torch.ones(dim))
        self.norm2_w = nn.Parameter(torch.ones(dim))
        self.attn = CausalSelfAttention(dim, n_heads)
        h = dim * hidden_mult
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)

    @staticmethod
    def _rms(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
        return t * rms * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._rms(x, self.norm1_w))
        n = self._rms(x, self.norm2_w)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class TinyDecoder(nn.Module):
    """``layers`` decoder blocks + final RMSNorm + vocab head."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 4,
        layers: int = 4,
        hidden_mult: int = 2,
        vocab: int = 512,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DecoderBlock(dim, n_heads, hidden_mult)
            for _ in range(layers)
        )
        self.norm_f = nn.Parameter(torch.ones(dim))
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.head(x * rms * self.norm_f)


class MoEBlock(nn.Module):
    """Dense-MoE / model-soup block: E expert projections SUMMED.

    ``e = Σ_i gate_i(n)`` is the ``weight_factor_linear`` case — a
    real merge deployment tooling does by hand — sitting inside a
    residual block with a nonlinearity so the fold must interact with
    surrounding structure, not just fire on a bare sum.
    """

    def __init__(
        self, dim: int, hidden: int, n_experts: int = 6
    ) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(dim))
        self.experts = nn.ModuleList(
            nn.Linear(dim, hidden, bias=False) for _ in range(n_experts)
        )
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        n = x * rms * self.norm_w
        e = self.experts[0](n)
        for g in self.experts[1:]:
            e = e + g(n)
        return x + self.down(F.silu(e))


class TinyMoE(nn.Module):
    """``layers`` MoE blocks + pooled head — the soup-merge form."""

    def __init__(
        self,
        dim: int = 256,
        hidden: int = 512,
        n_experts: int = 6,
        layers: int = 3,
        vocab: int = 128,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MoEBlock(dim, hidden, n_experts) for _ in range(layers)
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.head(x.mean(dim=-2))


class TinyConvNet(nn.Module):
    """Multi-branch conv stem + attention blocks + pooled head.

    ``ParallelConv`` (4 same-input 1×1 convs summed) gives the conv
    pairing case; the flattened feature map then feeds ``ParallelBlock``
    attention blocks (the five-projection pairing case) — a mixed
    conv/attention model, the shape real vision-transformer stems
    take.
    """

    def __init__(
        self,
        ch: int = 64,
        branches: int = 4,
        n_heads: int = 4,
        layers: int = 2,
        hidden_mult: int = 2,
        vocab: int = 16,
    ) -> None:
        super().__init__()
        self.stem = ParallelConv(ch, ch, branches=branches, kernel=1)
        self.blocks = nn.ModuleList(
            ParallelBlock(ch, n_heads, hidden_mult)
            for _ in range(layers)
        )
        self.head = nn.Linear(ch, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H·W, C) token sequence for the blocks.
        t = self.stem(x).flatten(2).transpose(1, 2)
        for b in self.blocks:
            t = b(t)
        return self.head(t.mean(dim=1))


# ---------------------------------------------------------------------------
#  Cell registry
# ---------------------------------------------------------------------------


@dataclass
class ModelCell:
    """One bench cell: a whole-model factory plus its input shapes."""

    name: str
    build: callable  # () -> (model, args_tuple, params_dict)
    mech: str = ""  # mechanism under test (report prose)


def _cells(args) -> list[ModelCell]:
    """The model cells, sized for a 4 GB CUDA card."""
    cells: list[ModelCell] = []

    # -- 4-block causal decoder --------------------------------------
    def _decoder(B=4, T=128, d=256, H=4, L=4):
        return (
            TinyDecoder(d, H, L).eval(),
            (torch.randn(B, T, d),),
            {"B": B, "T": T, "d": d, "L": L},
        )

    cells.append(
        ModelCell(
            "tiny_decoder",
            _decoder,
            mech="qkv + gate/up pairing, norm-gain folds ×4 blocks",
        )
    )

    # -- 2-block gated linear-attention (retnet) stack ----------------
    # Unbatched (T, d) input — the canonical carrier form (batched
    # states lift too since the batched-state lift landed; this bench
    # keeps the unbatched shape for comparability).  L=2 keeps the
    # emitted-seq
    # intermediate block whose whole-graph saturation is the known
    # cost cliff; T=32 stays well under it.
    ssm_t = int(getattr(args, "ssm_t", None) or 32)

    def _ssm(T=ssm_t, d=128, L=2):
        return (
            LinearAttnStack(
                d, mode="retnet", k=1, n_blocks=L, seed=0
            ).eval(),
            (torch.randn(T, d),),
            {"T": T, "d": d, "L": L},
        )

    cells.append(
        ModelCell(
            "tiny_ssm",
            _ssm,
            mech="affine-carrier scan lift → level-batched executor",
        )
    )

    # -- 3-block expert-summed MoE ------------------------------------
    def _moe(B=8, T=64, d=256, h=512, E=6, L=3):
        return (
            TinyMoE(d, h, E, L).eval(),
            (torch.randn(B, T, d),),
            {"B": B, "T": T, "d": d, "E": E, "L": L},
        )

    cells.append(
        ModelCell(
            "tiny_moe",
            _moe,
            mech="weight_factor merge per block: Σ x@Wi → x@(ΣWi)",
        )
    )

    # -- conv stem + attention blocks ---------------------------------
    def _conv(B=8, ch=64, HW=16, br=4, L=2):
        return (
            TinyConvNet(ch, br, n_heads=4, layers=L).eval(),
            (torch.randn(B, ch, HW, HW),),
            {"B": B, "ch": ch, "HW": HW, "L": L},
        )

    cells.append(
        ModelCell(
            "tiny_convnet",
            _conv,
            mech="conv pairing (stem) + qkv/gate·up pairing (blocks)",
        )
    )

    only = getattr(args, "models", None)
    if only:
        keep = {s.strip() for s in only.split(",")}
        cells = [c for c in cells if c.name in keep]
    return cells


# ---------------------------------------------------------------------------
#  Timing helpers (real_win_hunt conventions)
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


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def run_cell(
    cell: ModelCell,
    dev: torch.device,
    *,
    candidates: tuple[str, ...],
    n_calls: int,
    at_warmup: int,
    compile_timeout: float,
    max_iterations: int,
    max_enodes: int,
    budget_s: float | None,
    verbose: bool,
) -> tuple[dict, Case]:
    """Optimize+autotune one whole model, pack a benchkit ``Case``."""
    print(f"\n=== {cell.name} ({cell.mech}) ===", flush=True)
    torch.manual_seed(0)
    model64, args64, params = cell.build()
    model64 = model64.to(torch.float64).to(dev).eval()
    args64 = tuple(a.to(dev).double() for a in args64)
    x64_or_t = args64 if len(args64) > 1 else args64[0]
    rec: dict = {"name": cell.name, "params": params}
    n_params = sum(p.numel() for p in model64.parameters())
    rec["n_params"] = n_params

    with torch.no_grad():
        ref64 = model64(*args64)

    # -- fp64 pipeline: the production search + verify -----------------
    t0 = time.time()
    try:
        del64, st64 = Optimizer(backend=TorchBackend()).optimize(
            model64,
            x64_or_t,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            verify=False,
            verbose=False,
        )

        del64 = del64.to(dev).eval()
        with torch.no_grad():
            d = _rel_diff(del64(*args64), ref64)
        rec["opt_s"] = round(time.time() - t0, 2)
        rec["fp64_rel"] = d["rel_to_max"]
        rec["lowering"] = st64.get("lowering")
        rec["pairing_groups"] = st64.get("pairing_groups")
        rec["paired_extract"] = st64.get("paired_extract")
        rec["nonlocal_lifts"] = st64.get("nonlocal_lifts")
        rec["fires"] = {
            k: v
            for k, v in st64.get("rule_fires", {}).items()
            if v and "comm" not in k and "assoc_add" not in k
        }
        print(
            f"  fp64: {rec['opt_s']:.1f}s lowering={rec['lowering']} "
            f"pairing={rec['pairing_groups']} "
            f"paired_extract={rec['paired_extract']} "
            f"lifts={rec['nonlocal_lifts']} "
            f"rel={d['rel_to_max']:.2e}",
            flush=True,
        )
    except OptimizationResourceError as e:
        rec["opt_error"] = f"resource: {e}"
        rec["opt_s"] = round(time.time() - t0, 2)
        print(
            f"  optimize_model resource-bound after "
            f"{rec['opt_s']:.1f}s: {e}",
            flush=True,
        )
    except Exception as e:
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        rec["opt_s"] = round(time.time() - t0, 2)
        print(f"  optimize_model FAILED: {e}", flush=True)

    # -- fp32 module + autotuned winner --------------------------------
    model32 = copy.deepcopy(model64).float().to(dev).eval()
    args32 = tuple(a.float() for a in args64)
    x32_or_t = args32 if len(args32) > 1 else args32[0]
    with torch.no_grad():
        ref32 = model32(*args32)
    # Scale-aware verify floor (real_win_hunt convention): outputs
    # span ~1e2-1e3 under reassociation, so the atol scales with the
    # output range — recorded in aux.
    atol_v = max(1e-5, 1e-4 * ref32.abs().max().item())
    rec["verify_atol"] = atol_v

    best = None
    try:
        best, st32 = Optimizer(backend=TorchBackend()).optimize(
            model32,
            x32_or_t,
            strategy=Autotuned(
                candidates,
                budget_s=budget_s,
                n_calls=n_calls,
                warmup=at_warmup,
                rtol=1e-4,
                atol=atol_v,
                verbose=verbose,
                builders=TORCH_BUILDERS,
            ),
            max_iterations=max_iterations,
            max_enodes=max_enodes,
        )

        at = st32["autotune"]
        rec["lowering32"] = st32.get("lowering")
        parts = []
        for cn, r in at["candidates"].items():
            if r.get("status") == "timed":
                parts.append(f"{cn}:{r['median_s'] * 1e3:.3f}ms")
            else:
                parts.append(f"{cn}:{r.get('status', '?')}")
        rec["autotune"] = " ".join(parts)
        rec["autotune_s"] = round(at["elapsed_s"], 2)
        rec["search32_s"] = round(at["search_s"], 2)
        if at["winner"] is not None:
            rec["pick"] = at["winner"]
            rec["verified"] = bool(
                at["candidates"][at["winner"]].get("verified")
            )
        else:
            rec["pick"] = "pipeline_fallback"
            rec["verified"] = bool(
                at["candidates"]
                .get("_pipeline_fallback", {})
                .get("verified")
            )
        print(
            f"  autotune: pick={rec['pick']} verified={rec['verified']} "
            f"({rec['autotune_s']:.1f}s) | {rec['autotune']}",
            flush=True,
        )
    except OptimizationResourceError as e:
        rec["autotune_error"] = f"resource: {e}"
        print(f"  autotune resource-bound: {e}", flush=True)
        best = None
    except Exception as e:
        rec["autotune_error"] = f"{type(e).__name__}: {e}"
        print(f"  autotune FAILED: {e}", flush=True)
        best = None

    # -- inductor baseline (deepcopy — compile rewrites forward) -------
    cm = None
    if compile_timeout and compile_timeout > 0:
        cm, status = try_compile(
            copy.deepcopy(model32), args32, compile_timeout
        )
        rec["inductor_status"] = status
        if cm is not None:
            with torch.no_grad():
                rec["inductor_fp32"] = _rel_diff(cm(*args32), ref32)
        print(f"  inductor: {status}", flush=True)

    # -- fp32 gates -----------------------------------------------------
    if best is not None:
        with torch.no_grad():
            rec["catopt_fp32"] = _rel_diff(best(*args32), ref32)

    # -- peak memory (CUDA only), before Runner timing ------------------
    if dev.type == "cuda":
        for name, m in (
            ("eager", model32),
            ("inductor", cm),
            ("catopt_best", best),
        ):
            if m is not None:
                try:
                    rec[f"peak_mb_{name}"] = round(
                        _peak_mem(m, args32), 1
                    )
                except Exception as e:
                    rec[f"peak_mb_{name}"] = f"err:{type(e).__name__}"

    # -- pack the benchkit case ----------------------------------------
    variants = [Variant("eager", _fwd_stmt(model32, args32))]
    if cm is not None:
        variants.append(
            Variant(
                "inductor",
                _fwd_stmt(cm, args32),
                note=rec.get("inductor_status", ""),
            )
        )
    if best is not None:
        variants.append(
            Variant(
                "catopt_best",
                _fwd_stmt(best, args32),
                note=f"pick={rec['pick']}",
            )
        )

    rec["mech"] = cell.mech
    case = Case(
        name=cell.name, params=params, variants=variants, aux=rec
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
    n_calls = int(getattr(args, "calls", None) or 25)
    at_warmup = int(getattr(args, "warmup", None) or 5)
    min_run_time = float(getattr(args, "min_run_time", None) or 0.2)
    max_iterations = int(getattr(args, "max_iterations", None) or 16)
    max_enodes = int(getattr(args, "max_enodes", None) or 300_000)
    _ct = getattr(args, "compile_timeout", None)
    compile_timeout = 60.0 if _ct is None else float(_ct)
    _bs = getattr(args, "budget_s", None)
    budget_s = None if _bs is None else float(_bs)
    verbose = bool(getattr(args, "verbose", False))

    cands = getattr(args, "candidates", None)
    cand_tuple = (
        tuple(s.strip() for s in cands.split(","))
        if cands
        else (
            "eager",
            "generic",
            "batched",
            "compiled",
            "compiled_generic",
        )
    )
    if dev.type == "cuda" and "cuda_graph" not in cand_tuple:
        cand_tuple = (*cand_tuple, "cuda_graph")

    cells_in = _cells(args)
    print(
        f"model_bench — device={dev} cells={[c.name for c in cells_in]} "
        f"candidates={cand_tuple}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(at_warmup, 3), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cases: list[Case] = []
    for cell in cells_in:
        rec, case = run_cell(
            cell,
            dev,
            candidates=cand_tuple,
            n_calls=n_calls,
            at_warmup=at_warmup,
            compile_timeout=compile_timeout,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            budget_s=budget_s,
            verbose=verbose,
        )
        recs.append(rec)
        cases.append(case)

    cells = runner.run(cases)

    # -- console table -------------------------------------------------
    hdr = (
        f"{'model':<14} | {'eager':>8} | {'inductor':>8} | "
        f"{'catopt':>8} | {'pick':<10} | {'ver':>4} | {'xE':>5} | "
        f"{'xI':>5} | {'peakMB e/i/c':>14}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians

        def g(n, _ms=ms):
            return f"{_ms[n] * 1e3:>8.3f}" if n in _ms else f"{'—':>8}"

        e, i, b = (
            ms.get("eager"),
            ms.get("inductor"),
            ms.get("catopt_best"),
        )
        xe = f"{e / b:>5.2f}" if e and b else f"{'—':>5}"
        xi = f"{i / b:>5.2f}" if i and b else f"{'—':>5}"
        peaks = "/".join(
            str(rec.get(f"peak_mb_{n}", "—"))
            for n in ("eager", "inductor", "catopt_best")
        )
        print(
            f"{rec['name']:<14} | {g('eager')} | {g('inductor')} | "
            f"{g('catopt_best')} | {rec.get('pick', '—')!s:<10} | "
            f"{'yes' if rec.get('verified') else 'NO':>4} | {xe} | "
            f"{xi} | {peaks:>14}"
        )
    print("-" * len(hdr))
    print(
        "  xE = catopt_best speedup vs eager · xI = vs inductor "
        "(>1 = catopt wins)"
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s", flush=True
    )

    scored = [
        (
            c.case.name,
            c.medians["inductor"] / c.medians["catopt_best"],
            c.medians["eager"] / c.medians["catopt_best"],
        )
        for c in cells
        if c.medians.get("inductor") and c.medians.get("catopt_best")
    ]
    best = max(scored, key=lambda t: t[1], default=None)
    n_verified = sum(1 for r in recs if r.get("verified"))
    findings = [
        Finding(
            claim=(
                "whole multi-block models beat Inductor under the "
                "autotuned lowering"
            ),
            verdict=(
                Verdict.WIN
                if best and best[1] > 1
                else Verdict.NEGATIVE
            ),
            headline=(
                f"best {best[1]:.2f}× vs Inductor ({best[2]:.2f}× vs "
                f"eager) on {best[0]}"
                if best
                else "no model beats Inductor"
            ),
            metric="catopt_best / inductor",
            value=best[1] if best else None,
            evidence={
                "picks": {r["name"]: r.get("pick") for r in recs}
            },
        ),
        Finding(
            claim="every optimized model verifies equivalent",
            verdict=(
                Verdict.WIN
                if recs and n_verified == len(recs)
                else Verdict.REGRESSION
            ),
            headline=f"{n_verified}/{len(recs)} models verified",
            metric="verified models",
            value=float(n_verified),
        ),
    ]
    report = Report(
        suite="model_bench",
        title="Whole-model optimize + verify",
        summary=(
            "Complete multi-block models: latency, peak memory and "
            "compile time for eager / Inductor / the autotuned catopt "
            "lowering, verified per model."
        ),
        findings=findings,
        cells=cells,
        env=collect_env(dev),
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"model_bench_{ts}.json"
        md_path = out_dir / f"model_bench_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "whole-model bench: complete multi-block models — "
            "autotuned catopt vs eager/Inductor"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma subset: tiny_decoder,tiny_ssm,tiny_moe,"
        "tiny_convnet (default all)",
    )
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="comma-separated autotune candidates (default "
        "eager,generic,batched,compiled,compiled_generic; "
        "cuda adds cuda_graph)",
    )
    ap.add_argument("--calls", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--ssm-t",
        type=int,
        default=32,
        help="sequence length for tiny_ssm (saturation cost grows "
        "with the unrolled horizon)",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=60.0,
        help="torch.compile budget for the inductor baseline "
        "(0 disables it)",
    )
    ap.add_argument(
        "--budget-s",
        type=float,
        default=None,
        help="autotune wall-clock budget per model (search included)",
    )
    ap.add_argument("--max-iterations", type=int, default=16)
    ap.add_argument("--max-enodes", type=int, default=300_000)
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
