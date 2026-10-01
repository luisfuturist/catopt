"""Decode-mode scan bench — is the carrier the right *streaming* schedule?

``real_linear_attn`` answered the PREFILL question: unroll T steps,
lift the recurrence into the affine carrier, lower to
``BatchedScanModule`` — and on this CPU Inductor's fusion of the
unrolled chain usually wins anyway.  This bench asks the DECODE /
STREAMING question: tokens arrive one at a time, state ``h`` is
carried in Python, and there is **no loop for Inductor to fuse
across** — every step is one graph call.

The hypothesis under test: in decode the carrier form may be the
intrinsically right schedule.  A *chunked* decode step — unroll C
pending tokens inside the module, carry ``h`` in and out — exports as
``applyd(<compose tree>, h)`` with ``h`` an *input* leaf, so
``optimize_model`` routes it to the level-batched scan executor:
O(log C) batched compose levels per chunk call instead of C serial
step dispatches.  That is a schedule nothing else in the pipeline
produces (eager chunk = C serial steps; Inductor chunk = one fused
pointwise chain over C steps, still serial-depth C inside the
kernel).

The model (honest provenance — same family as real_linear_attn):

* ``step(x_t, h) -> h'`` — one SSM state update, single output (the
  catopt IR is single-rooted, so the carried state IS the output; a
  separate readout ``y_t = r_t ⊙ h_t`` would add identical per-token
  pointwise work to every schedule — the readout gate is folded into
  the value path as ``u = σ(W_r x_t) ⊙ (x_t W_1 … W_k)`` instead,
  which keeps a genuine gated emit *inside* the measured step).
* ``retnet`` — fixed decay ``γ = σ(log_decay)``, ``h' = γ⊙h + u``.
* ``gla`` — data-dependent decay ``a_t = σ(W_g x_t)`` (GLA/Mamba).
* ``delta`` — dense ``A_t = I − β_t k_t k_tᵀ``, ``h' = A_t h + u``
  (the honest dense-carrier negative case from real_linear_attn).

Timed unit = one full N-token decode (default N=256), so each
reported median divides by N for per-token latency / tokens-per-s —
the decode TPS number.  Variants per cell:

* ``eager`` — Python loop, N calls of the unmodified step.
* ``inductor`` — ``torch.compile(step)`` called per token (dynamo
  compiles the step once; it cannot fuse across Python iterations).
* ``inductor_ro`` — ``torch.compile(mode="reduce-overhead")`` —
  CUDA-graph-backed per-token calls (CUDA only).
* ``catopt_step`` — ``optimize_model`` output on the single step
  (generic IRModule; the pairing pass may fuse the shared-input
  linears ``W_g``/``W_r``/``W_1`` into one GEMM).
* ``catopt_step_ind`` — the optimized step under Inductor.
* ``eager_chunk`` — unoptimized ``ChunkedStep`` (C-step Python
  unroll inside the module) called N/C times — isolates the
  "fewer Python calls" effect from the carrier lowering.
* ``inductor_chunk`` — compiled unoptimized chunk.
* ``catopt_chunk`` — ``optimize_model`` on the chunk: the carrier
  lift fires (``stats["lowering"] == "batched"``), level-batched
  scan with ``h`` carried as an input, N/C calls.
* ``catopt_chunk_ind`` — the batched carrier under Inductor.
* ``catopt_chunk_graph`` — the batched carrier recorded into a
  manual CUDA graph (``capture_cuda_graph``), one replay per chunk
  (CUDA only).

Verification: every timed form is checked over a full N-step decode
in fp64 (final ``h``, ``rel_to_max``) and gated fp32
(``rtol=1e-4, atol=1e-5``) — the reassociation is exact to ~1e-14.
A cell whose optimized form fails the gate is reported, not timed.

Usage:
    .venv/bin/python bench/decode_scan_bench.py --device cpu --quick
    .venv/bin/python bench/decode_scan_bench.py --device cpu
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/decode_scan_bench.py \
        --device cuda
"""
# ruff: noqa: E402 RUF001 RUF002 RUF003 — ×, ·, −, ⊙, σ, γ in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import contextlib
import copy
import signal
import sys
import time
from pathlib import Path

sys.setrecursionlimit(400_000)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.nn as nn
from bench.benchkit import Case, Report, Runner, Variant, collect_env

from catopt_torch.torch_bridge import export_to_ir
from catopt_core.ir import IR
from catopt_orchestrator.optimize import _lower_extracted


from catopt_torch.adapters import TorchSink
from catopt_orchestrator import Optimizer

from catopt_torch.backend import TorchBackend

# run_all.py picks these up for its --quick lane.
QUICK = {
    "steps": 128,
    "chunks": "8",
    "families": "retnet,gla",
    "compile_timeout": 60.0,
}


# ---------------------------------------------------------------------------
#  Models — the decode step and its chunked wrapper
# ---------------------------------------------------------------------------


class DecodeStep(nn.Module):
    """One streaming step ``h' = a_t ⊙ h + u_t`` (or dense ``A_t h + u``).

    The value path is the gated map ``u = σ(W_r x_t) ⊙ (x_t W_1…W_k)``
    — the output gate of GLA folded onto the injected value, which
    keeps the emit work live inside a single-root graph (the catopt IR
    cannot represent a ``(y_t, h')`` tuple output).  ``h'`` doubles as
    the per-token output a readout head would consume.
    """

    def __init__(
        self, d: int, mode: str = "gla", k: int = 1, seed: int = 0
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mode = mode
        self.d = d
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
        elif mode == "delta":
            self.wb = nn.Linear(d, 1, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.wb.weight.copy_(
                    torch.randn(1, d, generator=g) * d**-0.5
                )
                self.wk.weight.copy_(
                    torch.randn(d, d, generator=g) * d**-0.5
                )
            self.register_buffer("eye", torch.eye(d))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.chain = nn.ParameterList(
            nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
            for _ in range(k)
        )

    def forward(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        u = x_t
        for w in self.chain:
            u = u @ w
        u = torch.sigmoid(self.wr(x_t)) * u
        if self.mode == "delta":
            beta = torch.sigmoid(self.wb(x_t))  # (1,)
            kn = self.wk(x_t)
            k = kn * torch.rsqrt((kn * kn).sum() + 1e-12)
            kt = k.unsqueeze(-1)  # (d, 1)
            A_t = self.eye - beta * (kt @ kt.transpose(-1, -2))
            return A_t @ h + u
        if self.mode == "gla":
            a = torch.sigmoid(self.wg(x_t))
        else:  # retnet
            a = torch.sigmoid(self.log_decay)
        return a * h + u


class ChunkedStep(nn.Module):
    """C pending tokens consumed in one call: ``(x_chunk, h) -> h'``.

    Unrolled C times over the shared step — ``optimize_model`` lifts
    the spine to ``apply[d](<compose tree>, h)`` with ``h`` a graph
    input, i.e. the carrier form *with state carry-in*.
    """

    def __init__(self, step: DecodeStep, chunk: int) -> None:
        super().__init__()
        self.step = step
        self.chunk = chunk

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for t in range(self.chunk):
            h = self.step(x[t], h)
        return h


# ---------------------------------------------------------------------------
#  Decode loops + compile guard
# ---------------------------------------------------------------------------


def _decode_loop(step_fn, x: torch.Tensor, h0: torch.Tensor) -> None:
    """One timed unit: N sequential steps, state carried in Python."""
    h = h0
    for t in range(x.shape[0]):
        h = step_fn(x[t], h)


def _decode_loop_ro(step_fn, x: torch.Tensor, h0: torch.Tensor) -> None:
    """Decode loop for ``mode="reduce-overhead"`` (cudagraph_trees).

    The carried ``h`` is a cudagraph-managed output buffer — feeding it
    back as the next call's input requires a per-step
    ``cudagraph_mark_step_begin()`` PLUS a ``.clone()`` of the returned
    state (torch's own guidance for outputs that persist across
    generations; the clone is honest per-token decode overhead).
    """
    h = h0
    for t in range(x.shape[0]):
        torch.compiler.cudagraph_mark_step_begin()
        h = step_fn(x[t], h).clone()


def _decode_chunked(
    chunk_fn, x: torch.Tensor, h0: torch.Tensor, C: int
) -> None:
    """One timed unit: N/C sequential chunk calls."""
    h = h0
    for t0 in range(0, x.shape[0], C):
        h = chunk_fn(x[t0 : t0 + C], h)


def _decode_h(
    step_fn, x: torch.Tensor, h0: torch.Tensor
) -> torch.Tensor:
    """Untimed reference: run the loop, return the final state."""
    h = h0
    for t in range(x.shape[0]):
        h = step_fn(x[t], h)
    return h


class _CompileTimeout(Exception):
    pass


def _on_alarm(sig, frm):
    raise _CompileTimeout()


def try_compile(callable_or_mod, args: tuple, budget_s: float, **kw):
    """``torch.compile`` guard — SIGALRM budgeted, first call inside.

    ``args`` is the positional-args tuple one call takes; the warmup
    call here is what forces dynamo to actually build the graph.
    Returns ``(callable_or_None, status_str)``.
    """
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover — non-POSIX
        try:
            cm = torch.compile(callable_or_mod, **kw)
            with torch.no_grad():
                cm(*args)
            return cm, "compiled"
        except Exception as e:
            return None, f"compile failed: {type(e).__name__}: {e}"
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    t0 = time.perf_counter()
    try:
        cm = torch.compile(callable_or_mod, **kw)
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


def _opt_fp32(opt64, ir64, src64, dev):
    """Rebuild the delivered lowering at fp32 (run_cell convention).

    Fused ``fused_*`` Param leaves materialised at fp64-lower time live
    in the built module's ``_param_map``, not in ``src64`` — pull them
    so the fp32 rebuild binds the same weights.
    """
    fp32_src = {n: v.float() for n, v in src64.items()}
    for n, p in opt64._param_map.items():
        if n not in fp32_src:
            fp32_src[n] = p.detach().float()
    # The delivered module's _root is the extracted (post-fold) term —
    # _lower_extracted lowers ir.root, so wrap it into a fresh IR that
    # carries the extracted term, not the original unrolled one.
    opt_ir = IR(
        root=opt64._root,
        inputs=ir64.inputs,
        input_names=ir64.input_names,
        params=ir64.params,
    )
    return (
        _lower_extracted(opt_ir.root, opt_ir, fp32_src, TorchSink())
        .to(dev)
        .eval()
    )


# ---------------------------------------------------------------------------
#  The sweep cell
# ---------------------------------------------------------------------------


def run_cell(
    mode: str,
    N: int,
    d: int,
    k: int,
    C: int,
    dev: torch.device,
    *,
    runner: Runner,
    compile_timeout: float,
    max_enodes: int,
    verbose: bool,
) -> tuple[dict, object | None]:
    """One (mode, N, d, k, C) cell → ``(record, benchkit.Cell)``."""
    torch.manual_seed(0)
    step64 = (
        DecodeStep(d, mode=mode, k=k).to(torch.float64).to(dev).eval()
    )
    x64 = torch.randn(N, d, device=dev, dtype=torch.float64)
    h0_64 = torch.randn(d, device=dev, dtype=torch.float64)
    tag = f"{mode} N={N} d={d} k={k} C={C}"
    cell: dict = {"mode": mode, "N": N, "d": d, "k": k, "C": C}
    print(f"\n=== {tag} ===", flush=True)
    if N % C:
        cell["verified"] = False
        cell["opt_error"] = f"N={N} not divisible by C={C}"
        return cell, None

    with torch.no_grad():
        ref64 = _decode_h(step64, x64, h0_64)

    # -- the unoptimized chunk (fewer calls, same math) ---------------
    chunk64 = ChunkedStep(step64, C).eval()
    with torch.no_grad():
        h_chk = h0_64
        for t0 in range(0, N, C):
            h_chk = chunk64(x64[t0 : t0 + C], h_chk)
        cell["chunk_fp64"] = _rel_diff(h_chk, ref64)

    # -- optimize the single step -------------------------------------
    step_opt64 = None
    step_ir64 = None
    step_src64 = None
    t0 = time.time()
    try:
        step_opt64, sstats = Optimizer(backend=TorchBackend()).optimize(step64, (x64[0], h0_64), max_iterations=32, max_enodes=max_enodes, verify=False, verbose=False)

        cell["step_opt_s"] = time.time() - t0
        cell["step_lowering"] = sstats.get("lowering")
        cell["step_pairing_groups"] = sstats.get("pairing_groups")
        cell["step_paired_extract"] = sstats.get("paired_extract")
        step_opt64 = step_opt64.to(dev).eval()
        with torch.no_grad():
            cell["step_opt_fp64"] = _rel_diff(
                step_opt64(x64[0], h0_64), step64(x64[0], h0_64)
            )
        step_ir64, step_src64 = export_to_ir(step64, (x64[0], h0_64))
        print(
            f"  step opt: {cell['step_opt_s']:.1f}s "
            f"lowering={cell['step_lowering']} "
            f"pairing={cell['step_pairing_groups']} "
            f"paired_extract={cell['step_paired_extract']} "
            f"rel={cell['step_opt_fp64']['rel_to_max']:.2e}",
            flush=True,
        )
    except Exception as e:
        cell["step_opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  step optimize_model FAILED: {e}", flush=True)

    # -- optimize the chunk -------------------------------------------
    chunk_opt64 = None
    chunk_ir64 = None
    chunk_src64 = None
    t0 = time.time()
    try:
        chunk_opt64, cstats = Optimizer(backend=TorchBackend()).optimize(chunk64, (x64[:C], h0_64), max_iterations=32, max_enodes=max_enodes, verify=False, verbose=False)

        cell["chunk_opt_s"] = time.time() - t0
        cell["chunk_lowering"] = cstats.get("lowering")
        cell["chunk_nonlocal_lifts"] = cstats.get("nonlocal_lifts")
        cell["chunk_pairing_groups"] = cstats.get("pairing_groups")
        cell["chunk_is_batched"] = getattr(
            chunk_opt64, "is_batched", False
        )
        cell["chunk_n_levels"] = getattr(chunk_opt64, "n_levels", 0)
        chunk_opt64 = chunk_opt64.to(dev).eval()
        with torch.no_grad():
            cell["chunk_opt_fp64"] = _rel_diff(
                chunk_opt64(x64[:C], h0_64), chunk64(x64[:C], h0_64)
            )
        chunk_ir64, chunk_src64 = export_to_ir(
            chunk64, (x64[:C], h0_64)
        )
        print(
            f"  chunk opt: {cell['chunk_opt_s']:.1f}s "
            f"lowering={cell['chunk_lowering']} "
            f"lifts={cell['chunk_nonlocal_lifts']} "
            f"batched={cell['chunk_is_batched']} "
            f"levels={cell['chunk_n_levels']} "
            f"rel={cell['chunk_opt_fp64']['rel_to_max']:.2e}",
            flush=True,
        )
    except Exception as e:
        cell["chunk_opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  chunk optimize_model FAILED: {e}", flush=True)

    # -- fp64 whole-decode verification of the lowered forms ----------
    gate_tol = dict(rtol=1e-4, atol=1e-5)
    checks: dict[str, bool] = {}
    with torch.no_grad():
        try:
            if step_opt64 is not None:
                o = _decode_h(step_opt64, x64, h0_64)
                cell["step_opt_decode_fp64"] = _rel_diff(o, ref64)
        except Exception as e:
            step_opt64 = None
            cell["step_opt_error"] = f"decode: {type(e).__name__}: {e}"
        try:
            if chunk_opt64 is not None:
                h_c = h0_64
                for t0_ in range(0, N, C):
                    h_c = chunk_opt64(x64[t0_ : t0_ + C], h_c)
                cell["chunk_opt_decode_fp64"] = _rel_diff(h_c, ref64)
        except Exception as e:
            chunk_opt64 = None
            cell["chunk_opt_error"] = f"decode: {type(e).__name__}: {e}"

    # -- fp32 timing modules ------------------------------------------
    step32 = copy.deepcopy(step64).float().to(dev).eval()
    chunk32 = copy.deepcopy(chunk64).float().to(dev).eval()
    x32 = x64.float()
    h0_32 = h0_64.float()
    with torch.no_grad():
        ref32 = _decode_h(step32, x32, h0_32)

    step_opt32 = chunk_opt32 = None
    try:
        if step_opt64 is not None:
            step_opt32 = _opt_fp32(
                step_opt64, step_ir64, step_src64, dev
            )
        if chunk_opt64 is not None:
            chunk_opt32 = _opt_fp32(
                chunk_opt64, chunk_ir64, chunk_src64, dev
            )
    except Exception as e:
        cell["opt32_error"] = f"{type(e).__name__}: {e}"

    # -- compiled variants ---------------------------------------------
    cstep32, status = try_compile(
        step32, (x32[0], h0_32), compile_timeout
    )
    cell["inductor_status"] = status
    cstep_ro32, status_ro = (
        try_compile(
            step32,
            (x32[0], h0_32),
            compile_timeout,
            mode="reduce-overhead",
        )
        if dev.type == "cuda"
        else (None, "skipped (cuda only)")
    )
    cell["inductor_ro_status"] = status_ro
    copt_step32, status = (
        try_compile(step_opt32, (x32[0], h0_32), compile_timeout)
        if step_opt32 is not None
        else (None, "skipped")
    )
    cell["opt_step_ind_status"] = status
    cchunk32, status = try_compile(
        chunk32, (x32[:C], h0_32), compile_timeout
    )
    cell["inductor_chunk_status"] = status
    copt_chunk32, status = (
        try_compile(chunk_opt32, (x32[:C], h0_32), compile_timeout)
        if chunk_opt32 is not None
        else (None, "skipped")
    )
    cell["opt_chunk_ind_status"] = status

    # -- manual CUDA graph on the batched carrier ---------------------
    # deepcopy can't clone a BatchedScanModule (its caches hold
    # non-leaf tensors) — rebuild a second instance for capture.
    g_chunk32 = None
    if (
        dev.type == "cuda"
        and chunk_opt32 is not None
        and getattr(chunk_opt32, "is_batched", False)
    ):
        try:
            g_chunk32 = _opt_fp32(
                chunk_opt64, chunk_ir64, chunk_src64, dev
            )
            g_chunk32.capture_cuda_graph(x32[:C], h0_32)
            cell["graph_captured"] = g_chunk32.is_graph_captured
        except Exception as e:
            g_chunk32 = None
            cell["graph_error"] = f"{type(e).__name__}: {e}"

    # -- fp32 whole-decode gate ----------------------------------------
    # Carrier forms are the claim under test — a failure fails the
    # cell.  Peripheral compiled variants that verify wrong (e.g.
    # reduce-overhead cudagraph aliasing on the carried state) are
    # dropped from timing and recorded, not silently benchmarked.
    def _gate_step(fn) -> bool:
        return torch.allclose(
            _decode_h(fn, x32, h0_32), ref32, **gate_tol
        )

    def _gate_step_ro(fn) -> bool:
        h = h0_32
        for t in range(N):
            torch.compiler.cudagraph_mark_step_begin()
            h = fn(x32[t], h).clone()
        return torch.allclose(h, ref32, **gate_tol)

    def _run_chunk(fn) -> torch.Tensor:
        h_c = h0_32
        for t0_ in range(0, N, C):
            out = fn(x32[t0_ : t0_ + C], h_c)
            # graph replay returns the static out buffer — clone
            # before it is overwritten by the next call.
            h_c = out.clone() if fn is g_chunk32 else out
        return h_c

    def _gate_chunk(fn) -> bool:
        return torch.allclose(_run_chunk(fn), ref32, **gate_tol)

    def _gate_chunk_carrier(fn) -> bool:
        """Gate for reassociating carriers (batched compose trees).

        The elementwise fp32-vs-eager32 gate demands two independent
        ~1e-5-noise fp32 paths coincide — below the noise floor of
        non-contracting recurrences (delta's ``I−βkkᵀ`` has
        eigenvalue 1: injected rounding never contracts and
        random-walks to ~1e-4 over a decode — measured).  Check the
        carrier's fp32 error vs fp64 TRUTH is at parity with eager's
        own fp32 error (×2 margin), which is the well-posed test of
        "noise at the recurrence's intrinsic floor".
        """
        h_c = _run_chunk(fn)
        ref64_32 = ref64.to(h_c.dtype)
        eager_err = (ref32 - ref64_32).abs().max()
        carrier_err = (h_c - ref64_32).abs().max()
        return bool(carrier_err <= 2.0 * eager_err)

    dropped: dict[str, str] = {}
    with torch.no_grad():
        try:
            if step_opt32 is not None:
                checks["catopt_step"] = _gate_step(step_opt32)
        except Exception as e:
            step_opt32 = None
            cell["step_opt_error"] = f"fp32: {type(e).__name__}: {e}"
        try:
            if chunk_opt32 is not None:
                checks["catopt_chunk"] = _gate_chunk_carrier(
                    chunk_opt32
                )
        except Exception as e:
            chunk_opt32 = None
            cell["chunk_opt_error"] = f"fp32: {type(e).__name__}: {e}"
        checks["eager_chunk"] = bool(
            cell["chunk_fp64"]["rel_to_max"] < 1e-8
        )
        for tag, fn, gate in (
            ("inductor", cstep32, _gate_step),
            ("inductor_ro", cstep_ro32, _gate_step_ro),
            ("catopt_step_ind", copt_step32, _gate_step),
            ("inductor_chunk", cchunk32, _gate_chunk),
            ("catopt_chunk_ind", copt_chunk32, _gate_chunk_carrier),
        ):
            if fn is None:
                continue
            try:
                if not gate(fn):
                    dropped[tag] = "fp32 decode mismatch"
            except Exception as e:
                dropped[tag] = f"{type(e).__name__}: {e}"
        if g_chunk32 is not None and g_chunk32.is_graph_captured:
            try:
                if not _gate_chunk(g_chunk32):
                    dropped["catopt_chunk_graph"] = (
                        "fp32 decode mismatch"
                    )
            except Exception as e:
                dropped["catopt_chunk_graph"] = (
                    f"{type(e).__name__}: {e}"
                )
    for tag in dropped:
        # Replace the failed callable with None so no variant is
        # emitted for it below.
        if tag == "inductor":
            cstep32 = None
        elif tag == "inductor_ro":
            cstep_ro32 = None
        elif tag == "catopt_step_ind":
            copt_step32 = None
        elif tag == "inductor_chunk":
            cchunk32 = None
        elif tag == "catopt_chunk_ind":
            copt_chunk32 = None
        elif tag == "catopt_chunk_graph":
            g_chunk32 = None
    cell["dropped_variants"] = dropped
    cell["gate_checks"] = checks
    cell["verified"] = bool(checks) and all(checks.values())
    if not cell["verified"]:
        cell["gate_failures"] = [
            n for n, ok in checks.items() if not ok
        ]
    print(
        f"  fp32 gate: {'✓' if cell['verified'] else '✗ FAIL'}"
        + (
            ""
            if cell["verified"]
            else f" ({', '.join(cell['gate_failures'])})"
        ),
        flush=True,
    )
    if not cell["verified"]:
        return cell, None

    # -- benchkit case --------------------------------------------------
    def dec_step(fn):
        def stmt() -> None:
            with torch.no_grad():
                _decode_loop(fn, x32, h0_32)

        return stmt

    def dec_chunk(fn):
        def stmt() -> None:
            with torch.no_grad():
                _decode_chunked(fn, x32, h0_32, C)

        return stmt

    variants = [
        Variant(
            name="eager",
            stmt=dec_step(step32),
            note="python loop, N step calls — O(N) serial dispatches",
        ),
        Variant(
            name="eager_chunk",
            stmt=dec_chunk(chunk32),
            note=f"unoptimized C-step unroll, N/C={N // C} calls",
        ),
    ]
    if cstep32 is not None:
        variants.append(
            Variant(
                name="inductor",
                stmt=dec_step(cstep32),
                note=f"compiled step, per-token — {cell['inductor_status']}",
            )
        )
    if cstep_ro32 is not None:

        def dec_step_ro(fn):
            def stmt() -> None:
                with torch.no_grad():
                    _decode_loop_ro(fn, x32, h0_32)

            return stmt

        variants.append(
            Variant(
                name="inductor_ro",
                stmt=dec_step_ro(cstep_ro32),
                note="compiled step, mode=reduce-overhead (cudagraphs)",
            )
        )
    if step_opt32 is not None:
        variants.append(
            Variant(
                name="catopt_step",
                stmt=dec_step(step_opt32),
                note=(
                    f"optimize_model(step): lowering={cell['step_lowering']}, "
                    f"pairing_groups={cell['step_pairing_groups']}"
                ),
            )
        )
    if copt_step32 is not None:
        variants.append(
            Variant(
                name="catopt_step_ind",
                stmt=dec_step(copt_step32),
                note="optimized step under Inductor, per-token",
            )
        )
    if cchunk32 is not None:
        variants.append(
            Variant(
                name="inductor_chunk",
                stmt=dec_chunk(cchunk32),
                note=(
                    "compiled unoptimized chunk — "
                    f"{cell['inductor_chunk_status']}"
                ),
            )
        )
    if chunk_opt32 is not None:
        variants.append(
            Variant(
                name="catopt_chunk",
                stmt=dec_chunk(chunk_opt32),
                note=(
                    f"optimize_model(chunk): lowering={cell['chunk_lowering']}, "
                    f"batched={cell['chunk_is_batched']}, "
                    f"levels={cell['chunk_n_levels']}, N/C={N // C} calls"
                ),
            )
        )
    if copt_chunk32 is not None:
        variants.append(
            Variant(
                name="catopt_chunk_ind",
                stmt=dec_chunk(copt_chunk32),
                note="batched carrier under Inductor",
            )
        )
    if g_chunk32 is not None and g_chunk32.is_graph_captured:
        variants.append(
            Variant(
                name="catopt_chunk_graph",
                stmt=dec_chunk(g_chunk32),
                note="batched carrier, manual CUDA graph replay",
            )
        )
    case = Case(
        name=f"{mode}_N{N}_d{d}_k{k}_C{C}",
        params={"mode": mode, "N": N, "d": d, "k": k, "C": C},
        variants=variants,
        aux={
            "verify": {
                "chunk_fp64": cell.get("chunk_fp64"),
                "step_opt_fp64": cell.get("step_opt_fp64"),
                "step_opt_decode_fp64": cell.get(
                    "step_opt_decode_fp64"
                ),
                "chunk_opt_fp64": cell.get("chunk_opt_fp64"),
                "chunk_opt_decode_fp64": cell.get(
                    "chunk_opt_decode_fp64"
                ),
                "allclose_fp32": cell["verified"],
            },
            "opt_stats": {
                "step_lowering": cell.get("step_lowering"),
                "step_pairing_groups": cell.get("step_pairing_groups"),
                "step_paired_extract": cell.get("step_paired_extract"),
                "step_pipeline_s": cell.get("step_opt_s"),
                "chunk_lowering": cell.get("chunk_lowering"),
                "chunk_nonlocal_lifts": cell.get(
                    "chunk_nonlocal_lifts"
                ),
                "chunk_pairing_groups": cell.get(
                    "chunk_pairing_groups"
                ),
                "chunk_is_batched": cell.get("chunk_is_batched"),
                "chunk_n_levels": cell.get("chunk_n_levels"),
                "chunk_pipeline_s": cell.get("chunk_opt_s"),
                "step_error": cell.get("step_opt_error"),
                "chunk_error": cell.get("chunk_opt_error"),
            },
            "compile_status": {
                "inductor": cell["inductor_status"],
                "inductor_ro": cell.get("inductor_ro_status"),
                "opt_step_ind": cell.get("opt_step_ind_status"),
                "inductor_chunk": cell.get("inductor_chunk_status"),
                "opt_chunk_ind": cell.get("opt_chunk_ind_status"),
                "chunk_graph_captured": cell.get("graph_captured"),
                "chunk_graph_error": cell.get("graph_error"),
            },
        },
    )
    ran = runner.run_case(case)
    cell["ms"] = {n: s * 1e3 for n, s in ran.medians.items()}
    # Per-token latency (the decode TPS number) and pairwise speedups.
    cell["us_per_token"] = {
        n: s * 1e6 / N for n, s in ran.medians.items()
    }
    cell["tps"] = {n: N / s for n, s in ran.medians.items()}
    for base in ("eager", "inductor", "inductor_chunk"):
        for n in ran.medians:
            if n != base and base in ran.medians:
                cell[f"x_{n}_vs_{base}"] = (
                    ran.medians[base] / ran.medians[n]
                )
    cell["_verdict"] = _cell_verdict(cell)
    return cell, ran


# The carrier claim: only forms whose schedule comes FROM the
# carrier lowering (BatchedScanModule, compiled or graph-replayed)
# count as "carrier" — inductor_chunk is the fused-unroll baseline
# the carrier must beat, not evidence for it.
_CARRIER_VARIANTS = (
    "catopt_chunk",
    "catopt_chunk_ind",
    "catopt_chunk_graph",
)
_NONCARRIER_VARIANTS = (
    "eager",
    "inductor",
    "inductor_ro",
    "catopt_step",
    "catopt_step_ind",
    "eager_chunk",
    "inductor_chunk",
)


def _cell_verdict(c: dict) -> str:
    """The headline comparison: best carrier schedule vs the best
    non-carrier schedule, on measured µs/token."""
    us = c.get("us_per_token") or {}
    car = [n for n in _CARRIER_VARIANTS if n in us]
    non = [n for n in _NONCARRIER_VARIANTS if n in us]
    if not car or not non:
        return "incomplete"
    bc = min(car, key=lambda n: us[n])
    bn = min(non, key=lambda n: us[n])
    ratio = us[bn] / us[bc]
    tag = (
        "carrier WIN"
        if ratio > 1.03
        else "carrier LOSS"
        if ratio < 0.97
        else "parity"
    )
    return (
        f"best non-carrier={bn} ({us[bn]:.2f}µs) vs "
        f"best carrier={bc} ({us[bc]:.2f}µs) → {tag} ({ratio:.2f}×)"
    )


def _parse_ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _parse_strs(s: str) -> list[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


# ---------------------------------------------------------------------------
#  Report supplement
# ---------------------------------------------------------------------------


def _md_supplement(results: list[dict]) -> str:
    """Markdown appended after ``Report.to_markdown``: per-token
    latency table and the honest verdict."""
    lines = [
        "",
        "## Per-token latency (µs/token · tokens/s)",
        "",
    ]
    ok = [c for c in results if c.get("verified")]
    names: list[str] = []
    for c in ok:
        for n in c.get("us_per_token", {}):
            if n not in names:
                names.append(n)
    if ok:
        hdr = "| case | " + " | ".join(names) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(names) + 1))
        for c in ok:
            row = [f"{c['mode']} N={c['N']} k={c['k']} C={c['C']}"]
            for n in names:
                us = c["us_per_token"].get(n)
                row.append(
                    f"{us:.2f} · {c['tps'][n]:.0f}"
                    if us is not None
                    else "—"
                )
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Summary", ""]
    if not ok:
        lines.append("no verified cells — honest negative result.")
        return "\n".join(lines) + "\n"
    # The headline question: does the chunked carrier beat the best
    # per-token schedule?  (verdicts computed in run_cell)
    for c in ok:
        lines.append(
            f"- {c['mode']} N={c['N']} k={c['k']} C={c['C']}: "
            f"{c.get('_verdict', '?')} — chunk lowering="
            f"{c.get('chunk_lowering')} batched={c.get('chunk_is_batched')} "
            f"levels={c.get('chunk_n_levels')}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def run_bench(args: argparse.Namespace) -> Report:
    """The full sweep → ``benchkit.Report`` (the run_all.py convention)."""
    dev = torch.device(getattr(args, "device", "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")

    warmup = getattr(args, "warmup", 5)
    min_run_time = getattr(args, "min_run_time", 0.4)
    verbose = getattr(args, "verbose", False)
    quick = getattr(args, "quick", False)
    compile_timeout = getattr(args, "compile_timeout", 90.0)
    max_enodes = getattr(args, "max_enodes", 300_000)

    steps = (
        _parse_ints(getattr(args, "steps", None) or "")
        if getattr(args, "steps", None)
        else ([128] if quick else [256])
    )
    dims = (
        _parse_ints(getattr(args, "dims", None) or "")
        if getattr(args, "dims", None)
        else [64]
    )
    chunks = (
        _parse_ints(getattr(args, "chunks", None) or "")
        if getattr(args, "chunks", None)
        else ([8] if quick else [8, 32])
    )
    depths = (
        _parse_ints(getattr(args, "depths", None) or "")
        if getattr(args, "depths", None)
        else [1]
    )
    families = (
        _parse_strs(getattr(args, "families", None) or "")
        if getattr(args, "families", None)
        else (
            ["retnet", "gla", "delta"]
            if not quick
            else ["retnet", "gla"]
        )
    )

    print("=" * 78)
    print(
        "  decode_scan: streaming SSM step — per-token loop vs "
        "carrier-lowered chunked decode"
    )
    print(
        f"  device={dev}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if dev.type == "cuda"
            else ""
        )
    )
    print(
        f"  families={families} N={steps} d={dims} k={depths} "
        f"C={chunks}"
    )
    print("=" * 78, flush=True)

    # Timing-loop hygiene: Timer.blocked_autorange toggles
    # torch.set_num_threads per measurement, which trips dynamo's
    # GLOBAL_STATE guard and can push a frame past recompile_limit —
    # after which compiled variants silently run EAGER.  Pin threads
    # to 1 (so the toggle is a no-op) and raise the recompile cap so
    # chunk modules of different C across cells each keep their own
    # compiled graph.
    if dev.type == "cpu":
        torch.set_num_threads(1)
    for attr, want in (
        ("recompile_limit", 128),
        ("accumulated_recompile_limit", 512),
        ("cache_size_limit", 128),
    ):
        if hasattr(torch._dynamo.config, attr):
            cur = getattr(torch._dynamo.config, attr)
            if cur < want:
                setattr(torch._dynamo.config, attr, want)

    runner = Runner(
        device=dev,
        warmup=warmup,
        min_run_time=min_run_time,
        num_threads=1 if dev.type == "cpu" else None,
    )
    results: list[dict] = []
    report_cells: list = []
    for mode in families:
        for N in steps:
            for d in dims:
                for k in depths:
                    for C in chunks:
                        c, ran = run_cell(
                            mode,
                            N,
                            d,
                            k,
                            C,
                            dev,
                            runner=runner,
                            compile_timeout=compile_timeout,
                            max_enodes=max_enodes,
                            verbose=verbose,
                        )
                        results.append(c)
                        if ran is not None:
                            report_cells.append(ran)
                        # Compiled guards are keyed per code object —
                        # reset between cells so one cell's chunk C
                        # cannot burn another cell's recompiles.
                        with contextlib.suppress(Exception):
                            torch._dynamo.reset()

    # -- console table -------------------------------------------------
    hdr = (
        f"{'mode':<7} {'N':>4} {'k':>2} {'C':>3} | "
        f"{'eager':>7} {'ind':>7} {'c_st':>7} {'e_chk':>7} "
        f"{'i_chk':>7} {'c_chk':>7} {'c_ci':>7} {'graph':>7} | "
        f"{'µs/tok':>7} {'tok/s':>7}"
    )
    print("\n" + "=" * 78)
    print(
        "  TIMING (median ms per FULL N-token decode) — the last two "
        "cols are the best carrier variant's per-token cost"
    )
    print("=" * 78)
    print(hdr)
    print("-" * len(hdr))
    for c in results:
        if not c.get("verified"):
            print(
                f"{c['mode']:<7} {c['N']:>4} {c['k']:>2} {c['C']:>3} | "
                "FAIL/SKIP — "
                f"{c.get('opt_error') or c.get('gate_failures')}"
            )
            continue
        ms = c["ms"]

        def g(n, _ms=ms) -> str:
            return f"{_ms[n]:>7.3f}" if n in _ms else f"{'—':>7}"

        car = [n for n in _CARRIER_VARIANTS if n in c["us_per_token"]]
        bc = (
            min(car, key=lambda n: c["us_per_token"][n])
            if car
            else None
        )
        cctok = c["us_per_token"].get(bc) if bc else None
        cctps = c["tps"].get(bc) if bc else None
        print(
            f"{c['mode']:<7} {c['N']:>4} {c['k']:>2} {c['C']:>3} | "
            f"{g('eager')} {g('inductor')} {g('catopt_step')} "
            f"{g('eager_chunk')} {g('inductor_chunk')} "
            f"{g('catopt_chunk')} {g('catopt_chunk_ind')} "
            f"{g('catopt_chunk_graph')} | "
            f"{cctok or 0:>7.2f} {cctps or 0:>7.0f}"
        )
        print(
            f"{'':<7} {'':>4} {'':>2} {'':>3} |   └ {c.get('_verdict', '')}"
        )
    print("-" * len(hdr))

    report = Report(
        suite="decode_scan",
        cells=report_cells,
        env=collect_env(dev),
    )

    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", "bench/results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"decode_scan_{ts}.json"
        md_path = out_dir / f"decode_scan_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        with md_path.open("a") as fh:
            fh.write(_md_supplement(results))
        try:
            report.to_plots(
                out_dir / "plots",
                x_param="C",
                speedup_vs="inductor",
                stem=f"decode_scan_{ts}",
            )
        except Exception as e:
            print(
                f"  note: Report.to_plots skipped "
                f"({type(e).__name__}: {e})"
            )
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "decode-mode scan bench — per-token SSM step loop vs "
            "carrier-lowered chunked decode"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="comma-separated decode lengths N (default 256; quick 128)",
    )
    ap.add_argument(
        "--dims",
        type=str,
        default=None,
        help="comma-separated state dims d (default 64)",
    )
    ap.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="comma-separated chunk sizes C (default 8,32; quick 8)",
    )
    ap.add_argument(
        "--depths",
        type=str,
        default=None,
        help="comma-separated value-chain depths k (default 1)",
    )
    ap.add_argument(
        "--families",
        type=str,
        default=None,
        help="comma-separated decay families: retnet,gla,delta "
        "(default all; quick: retnet,gla)",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.4,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=90.0,
        help="per-variant torch.compile budget, seconds",
    )
    ap.add_argument(
        "--max-enodes",
        type=int,
        default=300_000,
        help="e-graph enode bound for optimize_model",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="skip JSON/MD/plot emission",
    )
    ap.add_argument(
        "--out",
        default="bench/results",
        help="artifact directory (default bench/results)",
    )
    run_bench(ap.parse_args())


if __name__ == "__main__":
    main()
