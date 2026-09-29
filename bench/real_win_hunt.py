"""Real-ish win hunt — autotuned catopt vs eager/Inductor on CUDA.

Priority question: where does catopt produce *consistent wall-clock
wins* vs TorchInductor on representative model blocks — not synthetic
microbenchmarks?  Prior evidence: stories15M/110M transformer stacks
came out at parity; the k-deep weight chain (reassoc_scale) and the
chunked-decode + CUDA-graph form (decode_scan_bench, B=1) won.  This
bench hunts on five/six real-topology cells:

* ``linattn`` — a multi-head *unnormalized* attention block
  ``(Q Kᵀ) V`` (the linear-transformer / RetNet attention shape,
  no softmax): q/k/v/o projections plus the score path.  Two
  mechanisms in play: ``assoc_matmul`` reassociation (O(T²d) →
  O(Td²) — unreachable for Inductor, which has no matmul-chain
  reordering) and the product-law pairing of the three input
  projections into one GEMM.
* ``palm_stack`` — PaLM/GPT-J parallel blocks
  ``x + attn(norm x) + mlp(norm x)`` (``catopt_torch.models.ParallelBlock``
  ×2): five same-input projections the pairing pass can fold into
  one GEMM per block.  Same FLOPs, fewer launches.
* ``moe_sum`` — mixture-of-small-matmuls / model-soup merge:
  ``Σ_i x @ W_i`` over 8 experts — ``weight_factor_linear`` folds it
  to ONE matmul (a real FLOP cut deployment tools do by hand).
* ``conv_stem`` — ResNet-style 1×1 multi-branch stem
  (``catopt_torch.models.ParallelConv``): 4 same-input convs paired into
  one cuDNN call; Inductor does not fuse conv calls.
* ``decode_retnet`` — chunked streaming decode of a RetNet step,
  UNBATCHED vector state (the form the affine carrier lifts): the
  extracted ``applyd`` root routes to the level-batched scan
  executor; the ``cuda_graph`` candidate captures it so N/C chunk
  calls are N/C graph replays.  Timed unit = a full N-token decode.
* ``decode_flat_b8`` — the same decode at batch B=8 with the state
  kept flat (B·d).  Honest negative probe: the diagonal carrier
  lift does not fire on batched/flattened states, so this cell
  records what autotune falls back to — measured, not assumed.

Protocol per cell (one e-graph search per dtype, measured-honest
picks):

1. ``optimize_model`` on the fp64 module → fp64 verify
   (``rel_to_max``; reassociations land ~1e-7–1e-8, folds ~1e-14).
2. ``optimize_model_autotuned`` on the fp32 module → every lowering
   candidate (``eager,generic,batched,compiled,compiled_generic``,
   plus ``cuda_graph`` on CUDA) is rebuilt, ``sink.verify``-gated,
   and timed (``warmup`` + ``n_calls`` forwards, median).  The
   reported ``catopt_best`` is the measured winner — never a static
   cost-model claim.
3. ``torch.compile`` baseline on a deepcopy (SIGALRM-guarded).
4. benchkit ``Runner`` times ``eager`` / ``inductor`` /
   ``catopt_best``; decode cells time full N-token decode loops.

Measured outcome (RTX 2050 fp32, torch 2.14+cu130, two runs —
medians stable within ~10-20%):

* ``linattn`` — WIN ~1.9-2.1× vs Inductor.  The ``Q@(KᵀV)``
  reassociation cuts score-path FLOPs 12× (T=768, dh=64) and the
  qkv pairing merges three GEMMs; the ``compiled`` candidate wins.
* ``palm_stack`` — WIN ~1.1-1.3× vs Inductor.  12 pairing groups
  extracted (fused qkv+gate/up + channel-gain norm folds); modest
  because the work is already GEMM-dominated.
* ``moe_sum`` — WIN ~7-10× vs Inductor.  ``weight_factor_linear``
  folds 8 experts into ONE (512,512) matmul — an 8× runtime-FLOP
  cut Inductor cannot express.
* ``conv_stem`` — MARGINAL ~1.1× vs Inductor (run-to-run
  0.94-1.06× vs *eager* — the fused conv is roughly launch-parity).
* ``decode_retnet`` — WIN ~2.6-3.4× vs Inductor (5.5-7.7× vs
  eager).  Carrier lift → ``BatchedScanModule`` → ``cuda_graph``
  replay: 0.23 ms/chunk, the known decode-form win confirmed
  through the autotuned protocol.
* ``decode_flat_b8`` — LOSS ~0.5-0.6× vs Inductor.  The diagonal
  carrier does NOT lift the batched/flattened state (lowering stays
  ``generic``); Inductor's fused pointwise chain wins the unlifted
  form.  The B≥8-batch-flips-decode hypothesis is falsified at the
  lift, not the timing, layer.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/real_win_hunt.py \
        --device cuda
    .venv/bin/python bench/real_win_hunt.py --device cpu --quick
"""
# ruff: noqa: E402 RUF002 RUF003 -- ×, ·, Σ, ᵀ in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.setrecursionlimit(400_000)

import torch
import torch.nn as nn
from benchkit import Case, Report, Runner, Variant, collect_env
from catopt_torch.models import ParallelBlock, ParallelConv, ParallelLinear


from decode_scan_bench import ChunkedStep, DecodeStep
from catopt_orchestrator.optimize import Autotuned
from catopt_orchestrator import Optimizer

from catopt_torch.autotune import TORCH_BUILDERS
from catopt_torch.backend import TorchBackend

# run_all.py picks these up for its --quick lane.
QUICK = {
    "models": "linattn,moe_sum,decode_retnet",
    "calls": "20",
    "min_run_time": "0.1",
    "decode_n": "64",
    "compile_timeout": "60.0",
}


# ---------------------------------------------------------------------------
#  Models — real topologies, single rooted output, Param leaves
# ---------------------------------------------------------------------------


class UnnormAttnBlock(nn.Module):
    """Multi-head unnormalized attention: ``(Q Kᵀ) V``, no softmax.

    The linear-transformer identity: ``(Q Kᵀ) V`` is O(T²·dh) while
    ``Q (Kᵀ V)`` is O(T·dh²).  Export keeps both bracketings reachable
    for the e-graph; Inductor has no matmul-chain reordering pass, so
    any win here is structural, not kernel luck.  The q/k/v
    projections are also a same-input triple for the pairing pass.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) @ v
        return self.wo(att.transpose(1, 2).reshape(B, T, C))


class PalmStack(nn.Module):
    """``L`` PaLM parallel blocks (shared-norm attn + MLP branches)."""

    def __init__(self, d: int, layers: int, n_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelBlock(d, n_heads) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class FlatChunk(nn.Module):
    """Batched chunked decode with a FLAT state (B·d vector).

    Same math as ``decode_scan_bench.ChunkedStep`` but the carried
    state is kept flattened so the recurrence reads as a plain
    vector scan ``h = a ⊙ h + u`` over B·d elements.  Probe for the
    "does batching flip the decode result" question — the diagonal
    carrier either lifts this spine or the cell honestly reports
    generic lowering.
    """

    def __init__(
        self,
        d: int,
        B: int,
        mode: str = "retnet",
        C: int = 16,
        k: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.B, self.C, self.mode = d, B, C, mode
        self.wr = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.wr.weight.copy_(
                torch.randn(d, d, generator=g) * d**-0.5
            )
        if mode == "retnet":
            self.log_decay = nn.Parameter(
                torch.randn(d, generator=g) * 0.1 - 2.0
            )
        elif mode == "gla":
            self.wg = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.wg.weight.copy_(
                    torch.randn(d, d, generator=g) * d**-0.5
                )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.chain = nn.ParameterList(
            nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
            for _ in range(k)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, d = self.B, self.d
        hf = h.reshape(B * d)
        if self.mode == "retnet":
            a = (
                torch.sigmoid(self.log_decay)
                .unsqueeze(0)
                .expand(B, d)
                .reshape(B * d)
            )
        for t in range(self.C):
            xt = x[t]  # (B, d)
            u = xt
            for w in self.chain:
                u = u @ w
            u = torch.sigmoid(self.wr(xt)) * u
            if self.mode == "gla":
                a = torch.sigmoid(self.wg(xt)).reshape(B * d)
            hf = a * hf + u.reshape(B * d)
        return hf.reshape(B, d)


# ---------------------------------------------------------------------------
#  Cell registry
# ---------------------------------------------------------------------------


@dataclass
class HuntCell:
    """One bench cell: a module factory plus its input-shapes."""

    name: str
    build: callable  # (dev) -> (model, args_tuple, params_dict)
    decode_chunk: int = 0  # >0: timed unit is an N-token decode loop
    flops: dict | None = None  # variant name -> analytic FLOPs
    mech: str = ""  # mechanism under test (report prose)


def _cells(args) -> list[HuntCell]:
    """The hunt cells, sized for a 4 GB CUDA card."""
    cells: list[HuntCell] = []

    # -- unnormalized attention: reassoc + qkv pairing ---------------
    def _linattn(dev, B=4, T=768, d=256, H=4):
        return (
            UnnormAttnBlock(d, H).eval(),
            (torch.randn(B, T, d),),
            {"B": B, "T": T, "d": d},
        )

    B, T, d, H = 4, 768, 256, 4
    dh = d // H
    cells.append(
        HuntCell(
            "linattn",
            _linattn,
            flops={
                "eager": 4 * 2 * B * T * d * d + 2 * B * H * T * T * dh,
                "catopt": 4 * 2 * B * T * d * d
                + 2 * B * H * T * dh * dh,
            },
            mech="assoc_matmul reassoc + qkv pairing",
        )
    )

    # -- PaLM parallel-block stack ------------------------------------
    def _palm(dev, d=384, L=2, heads=8, B=4, T=96):
        return (
            PalmStack(d, L, heads).eval(),
            (torch.randn(B, T, d),),
            {"d": d, "L": L, "BT": B * T},
        )

    cells.append(
        HuntCell(
            "palm_stack",
            _palm,
            mech="product-law pairing (q/k/v/gate/up → 1 GEMM)",
        )
    )

    # -- mixture of small matmuls -------------------------------------
    def _moe(dev, d=512, E=8, R=4096):
        return (
            ParallelLinear(d, n_experts=E).eval(),
            (torch.randn(R, d),),
            {"d": d, "E": E, "R": R},
        )

    cells.append(
        HuntCell(
            "moe_sum",
            _moe,
            flops={
                "eager": 8 * 2 * 4096 * 512 * 512,
                "catopt": 2 * 4096 * 512 * 512,
            },
            mech="weight_factor merge: Σ x@Wi → x@(ΣWi)",
        )
    )

    # -- ResNet-style parallel 1x1 conv stem ---------------------------
    def _conv(dev, ch=128, branches=4, B=8, HW=28):
        return (
            ParallelConv(ch, ch, branches=branches, kernel=1).eval(),
            (torch.randn(B, ch, HW, HW),),
            {"ch": ch, "branches": branches, "B": B},
        )

    cells.append(
        HuntCell(
            "conv_stem",
            _conv,
            mech="conv pairing (4 same-input convs → 1 cuDNN call)",
        )
    )

    # -- chunked streaming decode, unbatched vector state --------------
    def _dec(dev, d=128, C=16):
        return (
            ChunkedStep(DecodeStep(d, mode="retnet", k=2), C).eval(),
            (torch.randn(C, d), torch.randn(d)),
            {"d": d, "C": C, "B": 1},
        )

    cells.append(
        HuntCell(
            "decode_retnet",
            _dec,
            decode_chunk=16,
            mech="carrier lift → batched scan → CUDA graph replay",
        )
    )

    # -- same decode batched B=8 (flat state — honest probe) -----------
    def _dec8(dev, d=128, C=16, B=8):
        return (
            FlatChunk(d, B, mode="retnet", C=C, k=2).eval(),
            (torch.randn(C, B, d), torch.randn(B, d)),
            {"d": d, "C": C, "B": B},
        )

    cells.append(
        HuntCell(
            "decode_flat_b8",
            _dec8,
            decode_chunk=16,
            mech="batched-state carrier probe (expect no lift)",
        )
    )

    only = getattr(args, "models", None)
    if only:
        keep = {s.strip() for s in only.split(",")}
        cells = [c for c in cells if c.name in keep]
    return cells


# ---------------------------------------------------------------------------
#  Compile guard + timing helpers (decode_scan_bench conventions)
# ---------------------------------------------------------------------------


class _CompileTimeout(Exception):
    pass


def _on_alarm(sig, frm):
    raise _CompileTimeout()


def try_compile(mod: nn.Module, args: tuple, budget_s: float):
    """``torch.compile`` a module on an args tuple, timeout-guarded."""
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover — non-POSIX
        try:
            cm = torch.compile(mod)
            with torch.no_grad():
                cm(*args)
            return cm, "compiled"
        except Exception as e:
            return None, f"compile failed: {type(e).__name__}: {e}"
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    t0 = time.perf_counter()
    try:
        cm = torch.compile(mod)
        with torch.no_grad():
            cm(*args)
            cm(*args)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        return cm, f"compiled in {time.perf_counter() - t0:.1f}s"
    except _CompileTimeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile TIMEOUT >{budget_s:.0f}s"
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile failed: {type(e).__name__}: {e}"


def _rel_diff(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Scale-relative diff (reassoc_scale convention)."""
    max_abs = (out - ref).abs().max().item()
    return {
        "max_abs": max_abs,
        "rel_to_max": max_abs / ref.abs().max().clamp_min(1e-30).item(),
    }


def _fwd_stmt(mod, args: tuple):
    def stmt() -> None:
        with torch.no_grad():
            mod(*args)

    return stmt


def _decode_stmt(mod, x: torch.Tensor, h0: torch.Tensor, C: int):
    """Timed unit: one full N-token decode, state fed back per chunk.

    A CUDA-graph replay returns the static output buffer — feeding it
    back is safe (the next call copies inputs into static buffers
    before replay) and is exactly what decode_scan_bench measured.
    """

    def stmt() -> None:
        with torch.no_grad():
            h = h0
            for t0 in range(0, x.shape[0], C):
                h = mod(x[t0 : t0 + C], h)

    return stmt


def _decode_final(mod, x, h0, C):
    with torch.no_grad():
        h = h0
        for t0 in range(0, x.shape[0], C):
            h = mod(x[t0 : t0 + C], h)
    return h


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def run_cell(
    cell: HuntCell,
    dev: torch.device,
    *,
    decode_n: int,
    candidates: tuple[str, ...],
    n_calls: int,
    at_warmup: int,
    compile_timeout: float,
    max_iterations: int,
    max_enodes: int,
    budget_s: float | None,
    verbose: bool,
) -> tuple[dict, Case]:
    """Optimize+autotune one cell, pack a benchkit ``Case``."""
    print(f"\n=== {cell.name} ({cell.mech}) ===", flush=True)
    torch.manual_seed(0)
    model64, args64, params = cell.build(dev)
    model64 = model64.to(torch.float64).to(dev).eval()
    args64 = tuple(a.to(dev).double() for a in args64)
    x64_or_t = args64 if len(args64) > 1 else args64[0]
    rec: dict = {"name": cell.name, "params": params}

    with torch.no_grad():
        ref64 = model64(*args64)

    # -- fp64 pipeline: the production search + verify -----------------
    t0 = time.time()
    try:
        del64, st64 = Optimizer(backend=TorchBackend()).optimize(model64, x64_or_t, max_iterations=max_iterations, max_enodes=max_enodes, verify=False, verbose=False)

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
            f"rel={d['rel_to_max']:.2e} fires={rec['fires']}",
            flush=True,
        )
    except Exception as e:
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  optimize_model FAILED: {e}", flush=True)

    # -- fp32 module + autotuned winner --------------------------------
    model32 = copy.deepcopy(model64).float().to(dev).eval()
    args32 = tuple(a.float() for a in args64)
    x32_or_t = args32 if len(args32) > 1 else args32[0]
    with torch.no_grad():
        ref32 = model32(*args32)
    # Scale-aware verify floor: unnormalized-attention outputs span
    # ~1e2-1e3, so a uniform atol=1e-5 fails honest reassociation
    # round-off (~1e-4 abs on near-zero elements).  ``1e-4·max|ref|``
    # is the rtol scale applied to the output range — recorded in aux.
    atol_v = max(1e-5, 1e-4 * ref32.abs().max().item())
    rec["verify_atol"] = atol_v

    best = None
    at = {}
    try:
        best, st32 = Optimizer(backend=TorchBackend()).optimize(model32, x32_or_t, strategy=Autotuned(candidates, budget_s=budget_s, n_calls=n_calls, warmup=at_warmup, rtol=1e-4, atol=atol_v, verbose=verbose, builders=TORCH_BUILDERS), max_iterations=max_iterations, max_enodes=max_enodes)

        at = st32["autotune"]
        rec["lowering32"] = st32.get("lowering")
        parts = []
        for cn, r in at["candidates"].items():
            if r.get("status") == "timed":
                parts.append(f"{cn}:{r['median_s'] * 1e3:.3f}ms")
            else:
                parts.append(f"{cn}:{r.get('status', '?')}")
        rec["autotune"] = " ".join(parts)
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
        rec["search32_s"] = round(at["search_s"], 2)
        print(
            f"  autotune: pick={rec['pick']} verified={rec['verified']} "
            f"| {rec['autotune']}",
            flush=True,
        )
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

    # -- fp32 gates + decode-loop verification --------------------------
    gate = dict(rtol=1e-4, atol=atol_v)
    if cell.decode_chunk:
        # The timed unit is a full decode_n-token loop — swap the
        # C-token chunk input for the full stream before verification
        # and timing (autotune already verified the single chunk).
        C = cell.decode_chunk
        x_full = torch.randn(decode_n, *args32[0].shape[1:], device=dev)
        args32 = (x_full, args32[1])
        h_ref = _decode_final(model32, *args32, C)
        if best is not None:
            h_best = _decode_final(best, *args32, C)
            rec["decode_fp32"] = _rel_diff(h_best, h_ref)
            rec["decode_ok"] = torch.allclose(h_best, h_ref, **gate)
            if cm is not None:
                h_ind = _decode_final(cm, *args32, C)
                rec["inductor_decode_fp32"] = _rel_diff(h_ind, h_ref)
                rec["inductor_decode_ok"] = torch.allclose(
                    h_ind, h_ref, **gate
                )
        else:
            rec["decode_ok"] = None

    # -- pack the benchkit case ----------------------------------------
    C = cell.decode_chunk
    if C:
        x_dec, h0 = args32

        def _stmt(m, _x=x_dec, _h=h0):
            return _decode_stmt(m, _x, _h, C)

        variants = [Variant("eager", _stmt(model32))]
        if cm is not None:
            variants.append(
                Variant(
                    "inductor",
                    _stmt(cm),
                    note=rec.get("inductor_status", ""),
                )
            )
        if best is not None:
            variants.append(
                Variant(
                    "catopt_best",
                    _stmt(best),
                    note=f"pick={rec['pick']}",
                )
            )
        params = {**params, "N": decode_n}
    else:
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
    if cell.flops:
        for v in variants:
            if v.name == "eager":
                v.flops = cell.flops.get("eager")
            elif v.name == "catopt_best":
                v.flops = cell.flops.get("catopt")

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
    n_calls = int(getattr(args, "calls", None) or 30)
    at_warmup = int(getattr(args, "warmup", None) or 5)
    min_run_time = float(getattr(args, "min_run_time", None) or 0.2)
    decode_n = int(getattr(args, "decode_n", None) or 128)
    max_iterations = int(getattr(args, "max_iterations", None) or 32)
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
        f"real_win_hunt — device={dev} cells={[c.name for c in cells_in]} "
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
            decode_n=decode_n,
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
        f"{'model':<15} | {'eager':>8} | {'inductor':>8} | "
        f"{'catopt':>8} | {'pick':<10} | {'ver':>4} | {'xE':>5} | "
        f"{'xI':>5}"
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
        print(
            f"{rec['name']:<15} | {g('eager')} | {g('inductor')} | "
            f"{g('catopt_best')} | {rec.get('pick', '—')!s:<10} | "
            f"{'yes' if rec.get('verified') else 'NO':>4} | {xe} | {xi}"
        )
    print("-" * len(hdr))
    print(
        "  xE = catopt_best speedup vs eager · xI = vs inductor "
        "(>1 = catopt wins)"
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s", flush=True
    )

    report = Report(
        suite="real_win_hunt", cells=cells, env=collect_env(dev)
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"real_win_hunt_{ts}.json"
        md_path = out_dir / f"real_win_hunt_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "real-ish win hunt: autotuned catopt vs eager/Inductor "
            "on representative blocks"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma subset: linattn,palm_stack,moe_sum,conv_stem,"
        "decode_retnet,decode_flat_b8 (default all)",
    )
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="comma-separated autotune candidates (default "
        "eager,generic,batched,compiled,compiled_generic; "
        "cuda adds cuda_graph)",
    )
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--decode-n",
        type=int,
        default=128,
        help="tokens per timed decode loop (decode cells)",
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
        help="autotune wall-clock budget (search included)",
    )
    ap.add_argument("--max-iterations", type=int, default=32)
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
