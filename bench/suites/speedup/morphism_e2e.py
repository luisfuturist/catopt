"""Morphism-laws end-to-end bench — do TERM-level flops wins survive
delivery into measured wall time?

The README honest-numbers table reports the morphism laws' wins as
*term-level* flops (the e-graph cost model's count on the joint window
term): WindowCompose 4-block chains −92% (49,664→4,096),
ResidualReassoc ×4 streams −66% (50,176→17,024), KVLatentShare −62.5%
kv flops/bytes.  This bench converts those rows to runtime: it builds
the same fixtures the law tests use (``tests/test_morphism_windows.py``
+ ``tests/test_morphism_kv.py``), runs the real
``optimize_morphisms``/``MorphismSearch`` pipeline, and times the
delivered module.

Three flop counts per cell, deliberately distinguished:

* ``term_flops`` — the cost-model number the README cites
  (``cost_before``/``cost_after`` summed over grafted matches; for the
  KV fixture also the advertised ``kv_flops_*``/``kv_bytes_*`` pair).
* ``rt_flops`` — aten FLOPs the delivered module actually executes,
  measured with ``torch.profiler(with_flops=True)``.  The gap vs
  ``term_flops`` is real: slot-filler modules (identity/zero grafts)
  still run ops the cost model priced at zero.
* wall time — benchkit ``Runner`` medians.

The deliverable is the honest *conversion table*: which laws' term
wins translate to wall time, which don't, and the measured
``conv = (speedup − 1) / (flops_ratio − 1)`` factor — share of the
multiplicative headroom captured, negative on regression — with a
per-cell verdict
(launch/overhead-bound, partial, translates).  No marketing number —
at the test geometry (8×16) every variant is launch-bound and a −92%
flops cut is expected NOT to translate; the larger cells exist to find
the GEMM-bound regime where it should.

Variants per cell (only verified-equal ones are timed):

* ``eager``              — the fixture module, fp64 (test convention).
* ``inductor``           — ``torch.compile`` of a deepcopy (the
                           compiler's own shot at the same structure).
* ``morphism``           — the grafted module MorphismSearch delivers.
* ``morphism+inductor``  — ``torch.compile`` of the delivered module
                           (no deepcopy — IRModules hold non-leaf
                           tensors, so ``copy.deepcopy`` raises; this
                           is the same convention e2e_model uses).

Usage:
    .venv/bin/python bench/morphism_e2e.py --device cpu --quick
    .venv/bin/python bench/morphism_e2e.py --cells 8x16,1024x64,4096x128
"""
# ruff: noqa: E402, RUF001, RUF002, RUF003 -- ×, ·, →, − in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path

sys.setrecursionlimit(400_000)

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_orchestrator import MorphismSearch, Optimizer
from catopt_torch.backend import TorchBackend
from catopt_torch.models import DeepParallel
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
from bench.suites.speedup.real_win_hunt import try_compile

# run_all.py picks these up for its --quick lane.
QUICK = {
    "cells": "8x16",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
}


# ---------------------------------------------------------------------------
#  Fixtures — ported from test_morphism_windows.py / test_morphism_kv.py
# ---------------------------------------------------------------------------


class _ChainStack(nn.Module):
    """``x = b_i(x)`` — plain chain of DeepParallel blocks."""

    def __init__(self, dim: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _ResidualWrapped(nn.Module):
    """``x = x + b_i(x)`` — the residual-stream stack."""

    def __init__(self, dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DeepParallel(dim, dim, dim) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


def _latent(rank: int, dim: int, seed: int = 0) -> torch.Tensor:
    """A shared latent basis — fp64 rows of R^{dim}."""
    return torch.randn(
        rank,
        dim,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


class _KVBlock(nn.Module):
    """``sdpa(q,k,v) → out_proj`` on ``x``; k/v built through ``U``."""

    def __init__(
        self, dim: int, d_kv: int, U: torch.Tensor, seed: int
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q_proj = nn.Linear(dim, d_kv, bias=False)
        self.k_proj = nn.Linear(dim, d_kv, bias=False)
        self.v_proj = nn.Linear(dim, d_kv, bias=False)
        self.out_proj = nn.Linear(d_kv, dim, bias=False)
        self.double()
        dk = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        dv = torch.randn(
            d_kv, U.shape[0], generator=g, dtype=torch.float64
        )
        with torch.no_grad():
            self.k_proj.weight.copy_(dk @ U)
            self.v_proj.weight.copy_(dv @ U)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return self.out_proj(o)


class _SharedKVStack(nn.Module):
    """``y = Σ b_i(x)`` — parallel consumers of one input."""

    def __init__(
        self,
        dim: int = 16,
        d_kv: int = 8,
        rank: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        U = _latent(rank, dim)
        self.blocks = nn.ModuleList(
            _KVBlock(dim, d_kv, U, 10 + i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        for b in self.blocks[1:]:
            out = out + b(x)
        return out


def _kv(dim: int) -> _SharedKVStack:
    """Low-rank KV family at ``dim`` — d_kv = dim/2, rank = dim/4."""
    return _SharedKVStack(dim, max(4, dim // 2), max(2, dim // 4), 2)


#: Fixture registry — builder + the law set the test suite exercises.
_FIXTURES = {
    # WindowCompose needs ≥3 chain links; ×2 exercises OutInCompose.
    "chain_x2": {
        "build": lambda dim: _ChainStack(dim, 2),
        "laws": lambda: [M.WindowCompose(), M.OutInCompose()],
        "cells": "8x16,1024x64,4096x128",
    },
    "chain_x4": {
        "build": lambda dim: _ChainStack(dim, 4),
        "laws": lambda: [M.WindowCompose(), M.OutInCompose()],
        "cells": "8x16,1024x64,4096x128",
    },
    "resid_x3": {
        "build": lambda dim: _ResidualWrapped(dim, 3),
        "laws": lambda: [M.ResidualReassoc()],
        "cells": "8x16,1024x64,4096x128",
    },
    "resid_x4": {
        "build": lambda dim: _ResidualWrapped(dim, 4),
        "laws": lambda: [M.ResidualReassoc()],
        "cells": "8x16,1024x64,4096x128",
    },
    # SDPA dominates at larger T — the kv cell stays smaller so the
    # projection share of the window remains visible.
    "kv_x2": {
        "build": _kv,
        "laws": lambda: [K.KVLatentShare()],
        "cells": "8x16,512x64",
    },
}


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


def _rt_flops(mod, args: tuple) -> float | None:
    """Aten FLOPs actually executed — ``torch.profiler(with_flops)``.

    Counts what the delivered module *runs*, not what the cost model
    priced: slot-filler grafts (identity/zero IRModules) still execute
    their ops, so ``rt_flops`` ≥ ``term_flops_after`` in general.
    Inductor variants are skipped — compiled kernels don't attribute
    flops per aten op.
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        with (
            torch.no_grad(),
            profile(
                activities=[ProfilerActivity.CPU], with_flops=True
            ) as p,
        ):
            mod(*args)
        total = sum(
            e.flops
            for e in p.key_averages()
            if e.flops is not None and e.flops > 0
        )
        return float(total)
    except Exception:
        return None


def _param_bytes(mod: nn.Module) -> int:
    """Resident parameter+buffer bytes (fp64 here)."""
    return int(
        sum(p.numel() * p.element_size() for p in mod.parameters())
        + sum(b.numel() * b.element_size() for b in mod.buffers())
    )


def _verify(ref, mod, args: tuple, rtol: float = 1e-4):
    """→ (passed, max_rel); mod=None fails closed."""
    if mod is None:
        return False, None
    try:
        with torch.no_grad():
            vr = verify_equiv(ref, mod(*args), rtol=rtol)
        return bool(vr.passed), vr.max_rel
    except Exception:
        return False, None


def _parse_cells(spec: str) -> list[dict]:
    """``RxD`` list → [{'rows':…, 'dim':…}]."""
    out = []
    for tok in spec.split(","):
        r_s, d_s = tok.strip().lower().split("x")
        out.append({"rows": int(r_s), "dim": int(d_s)})
    return out


def _verdict(rec: dict, cell) -> str:
    """Classify the flops→runtime conversion for one timed cell."""
    if rec.get("n_grafted", 0) == 0:
        return "no graft — law declined"
    ms = cell.medians
    e, mo = ms.get("eager"), ms.get("morphism")
    if not e or not mo:
        return "unmeasured"
    ratio = rec.get("flops_ratio") or 0.0
    speedup = e / mo
    # Share of the multiplicative headroom captured: 100% would mean
    # the wall-time speedup equals the term-level flops ratio; a
    # negative share is a regression below eager parity.
    conv = (speedup - 1.0) / (ratio - 1.0) if ratio > 1 else None
    if speedup > 1.05 and conv is not None and conv >= 0.75:
        return f"translates ({conv:.0%} of headroom)"
    if speedup > 1.05:
        return (
            f"partial — {conv:.0%} of headroom, overhead floor"
            if conv is not None
            else "partial"
        )
    if speedup > 0.95:
        return "parity — launch/kernel-bound"
    return f"regression — {1 - speedup:.0%} slower despite flops win"


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def run_cell(
    fixture: str, cell_spec: dict, args, dev: torch.device
) -> tuple[dict, Case]:
    """Optimize + compile + verify one (fixture, rows×dim) cell."""
    spec = _FIXTURES[fixture]
    rows, dim = cell_spec["rows"], cell_spec["dim"]
    coords = f"rows={rows} dim={dim}"
    print(f"\n=== {fixture} [{coords}] ===", flush=True)
    torch.manual_seed(0)

    model = spec["build"](dim).eval().double().to(dev)
    g = torch.Generator().manual_seed(1234 + rows * 7 + dim)
    x = torch.randn(rows, dim, generator=g, dtype=torch.float64).to(dev)
    args_t = (x,)
    with torch.no_grad():
        ref = model(*args_t)

    rec: dict = {
        "name": f"{fixture}@{rows}x{dim}",
        "params": {"fixture": fixture, "rows": rows, "dim": dim},
        "param_bytes_eager": _param_bytes(model),
    }

    # -- morphism pipeline (the deliverable under test) ------------------
    opt_m = None
    t0 = time.time()
    try:
        opt_m, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            x,
            strategy=MorphismSearch(
                laws=spec["laws"](), optimize_rest=False
            ),
        )
        rec["opt_s"] = round(time.time() - t0, 2)
        rec["fires"] = stats.get("morphism_fires")
        rec["n_rewritten"] = stats.get("n_rewritten")
        e2e = stats.get("end_to_end") or {}
        rec["e2e_rel"] = e2e.get("max_rel_diff")
        # Per-match ledger + term-level flops over grafted windows.
        matches = {}
        c_b = c_a = 0.0
        kv_f = kv_b = kv_by_b = kv_by_a = None
        rec["n_grafted"] = 0
        for mk, mv in (stats.get("matches") or {}).items():
            entry = {
                k: mv.get(k)
                for k in (
                    "status",
                    "boundary",
                    "cost_before",
                    "cost_after",
                    "rel_diff",
                    "reason",
                    "latent_rank",
                    "n_kv_sites",
                    "kv_flops_before",
                    "kv_flops_after",
                    "kv_bytes_before",
                    "kv_bytes_after",
                )
                if mv.get(k) is not None
            }
            matches[mk] = entry
            if mv.get("status") == "grafted":
                rec["n_grafted"] += 1
                c_b += mv.get("cost_before") or 0.0
                c_a += mv.get("cost_after") or 0.0
                if mv.get("kv_flops_before") is not None:
                    kv_b = (kv_b or 0.0) + mv["kv_flops_before"]
                    kv_f = (kv_f or 0.0) + mv["kv_flops_after"]
                if mv.get("kv_bytes_before") is not None:
                    kv_by_b = (kv_by_b or 0.0) + mv["kv_bytes_before"]
                    kv_by_a = (kv_by_a or 0.0) + mv["kv_bytes_after"]
        rec["matches"] = matches
        if rec["n_grafted"]:
            rec["term_flops_before"] = c_b
            rec["term_flops_after"] = c_a
            rec["flops_ratio"] = c_b / c_a if c_a else None
        if kv_b is not None:
            rec["kv_flops_before"], rec["kv_flops_after"] = kv_b, kv_f
            rec["kv_flops_ratio"] = kv_b / kv_f if kv_f else None
        if kv_by_b is not None:
            rec["kv_bytes_before"], rec["kv_bytes_after"] = (
                kv_by_b,
                kv_by_a,
            )
        rec["param_bytes_opt"] = _param_bytes(opt_m)
        ok, rel = _verify(ref, opt_m, args_t)
        rec["morph_verified"], rec["morph_rel"] = ok, rel
        if not ok:
            opt_m = None
        print(
            f"  morphism: grafted={rec['n_grafted']} "
            f"fires={rec['fires']} "
            f"term {c_b:.3g}→{c_a:.3g} "
            f"rel={rec['morph_rel']}",
            flush=True,
        )
    except Exception as e:
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        rec["opt_s"] = round(time.time() - t0, 2)
        rec["n_grafted"] = 0
        print(f"  morphism optimize FAILED: {e}", flush=True)

    # -- runtime flops actually executed (aten, profiler) ----------------
    rec["rt_flops_eager"] = _rt_flops(model, args_t)
    rec["rt_flops_morphism"] = (
        _rt_flops(opt_m, args_t) if opt_m is not None else None
    )
    if rec["rt_flops_eager"] and rec["rt_flops_morphism"]:
        rec["rt_flops_ratio"] = (
            rec["rt_flops_eager"] / rec["rt_flops_morphism"]
        )
        print(
            f"  rt flops: {rec['rt_flops_eager']:.3g} → "
            f"{rec['rt_flops_morphism']:.3g} "
            f"({rec['rt_flops_ratio']:.2f}×)",
            flush=True,
        )

    # -- inductor baseline + composed variant ----------------------------
    ct = float(getattr(args, "compile_timeout", None) or 60.0)
    cm = oci = None
    if ct > 0:
        cm, rec["inductor_status"] = try_compile(
            copy.deepcopy(model), args_t, ct
        )
        ok, rel = _verify(ref, cm, args_t)
        rec["inductor_verified"], rec["inductor_rel"] = ok, rel
        if not ok:
            cm = None
        print(f"  inductor: {rec['inductor_status']}", flush=True)
        if opt_m is not None:
            # IRModules don't deepcopy (non-leaf tensors) — compile
            # the delivered module directly, e2e_model convention.
            oci, rec["morph_ind_status"] = try_compile(
                opt_m, args_t, ct
            )
            ok, rel = _verify(ref, oci, args_t)
            rec["morph_ind_verified"], rec["morph_ind_rel"] = ok, rel
            if not ok:
                oci = None
            print(
                f"  morphism+inductor: {rec['morph_ind_status']}",
                flush=True,
            )
    else:
        rec["inductor_status"] = "disabled (--compile-timeout 0)"

    # -- peak memory (CUDA only) ------------------------------------------
    if dev.type == "cuda":
        for vn, m in (
            ("eager", model),
            ("inductor", cm),
            ("morphism", opt_m),
            ("morphism+inductor", oci),
        ):
            if m is not None:
                try:
                    rec[f"peak_mb_{vn}"] = round(
                        _peak_mem(m, args_t), 1
                    )
                except Exception as e:
                    rec[f"peak_mb_{vn}"] = f"err:{type(e).__name__}"

    # -- pack the benchkit case -------------------------------------------
    variants = []
    flops_by = {
        "eager": rec.get("rt_flops_eager"),
        "morphism": rec.get("rt_flops_morphism"),
    }
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("morphism", opt_m),
        ("morphism+inductor", oci),
    ):
        if m is not None:
            variants.append(
                Variant(
                    vn, _fwd_stmt(m, args_t), flops=flops_by.get(vn)
                )
            )
    case = Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
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

    fixtures = (
        getattr(args, "fixtures", None) or ",".join(_FIXTURES)
    ).split(",")
    fixtures = [f.strip() for f in fixtures if f.strip()]
    cells_override = getattr(args, "cells", None)

    print(
        f"morphism_e2e — device={dev} fixtures={fixtures}", flush=True
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cells: list = []
    for fixture in fixtures:
        if fixture not in _FIXTURES:
            print(
                f"  unknown fixture {fixture!r} — skipped", flush=True
            )
            continue
        spec = _FIXTURES[fixture]
        cell_specs = _parse_cells(cells_override or spec["cells"])
        for cs in cell_specs:
            rec, case = run_cell(fixture, cs, args, dev)
            recs.append(rec)
            # Time immediately — compiled artifacts are warm; run_case
            # drops each variant's stmt (hence module) afterwards.
            cell = runner.run_case(case)
            cells.append(cell)
            print(
                "  times: "
                + "  ".join(
                    f"{n}={cell.medians[n] * 1e3:.3f}ms"
                    for n in cell.medians
                ),
                flush=True,
            )
            gc.collect()
            torch._dynamo.reset()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # -- post-timing conversion metrics ------------------------------------
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians
        e, mo = ms.get("eager"), ms.get("morphism")
        ind, mi = ms.get("inductor"), ms.get("morphism+inductor")
        if e and mo:
            rec["speedup_morph_vs_eager"] = e / mo
        if ind and mi:
            rec["speedup_morph_ind_vs_ind"] = ind / mi
        # Conversion = share of multiplicative headroom captured:
        # (speedup − 1) / (flops_ratio − 1); >0 translates, <0 regresses.
        if e and mo and (rec.get("flops_ratio") or 0) > 1:
            rec["conv_term"] = (e / mo - 1) / (rec["flops_ratio"] - 1)
        if e and mo and (rec.get("rt_flops_ratio") or 0) > 1:
            rec["conv_rt"] = (e / mo - 1) / (rec["rt_flops_ratio"] - 1)
        rec["verdict"] = _verdict(rec, cell)

    # -- console conversion table -------------------------------------------
    names = ["eager", "inductor", "morphism", "morphism+inductor"]
    hdr = (
        f"{'cell':<20} | {'law':<16} | {'flops t/r':>13} | "
        + " | ".join(f"{n:>14}" for n in names)
        + f" | {'e/m':>6} | {'conv':>5} | verdict"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians
        law = "+".join((rec.get("fires") or {}).keys()) or "—"
        tf = (
            f"{rec['flops_ratio']:.1f}×/"
            f"{(rec.get('rt_flops_ratio') or 0):.1f}×"
            if rec.get("flops_ratio")
            else "—"
        )
        row = (
            f"{rec['name']:<20} | {law:<16} | {tf:>13} | "
            + " | ".join(
                (
                    f"{ms[n] * 1e3:>8.3f} ms  "
                    if n in ms
                    else f"{'—':>14}"
                )
                for n in names
            )
        )
        sp = rec.get("speedup_morph_vs_eager")
        cv = rec.get("conv_term")
        print(
            f"{row} | "
            + (f"{sp:>5.2f}×" if sp else f"{'—':>6}")
            + " | "
            + (f"{cv:>4.0%}" if cv is not None else f"{'—':>5}")
            + f" | {rec['verdict']}"
        )
        extras = []
        if rec.get("kv_flops_ratio"):
            extras.append(
                f"kv_flops {rec['kv_flops_before']:.3g}→"
                f"{rec['kv_flops_after']:.3g} "
                f"({rec['kv_flops_ratio']:.2f}×)"
            )
        if rec.get("kv_bytes_before"):
            extras.append(
                f"kv_bytes {rec['kv_bytes_before']:.3g}→"
                f"{rec['kv_bytes_after']:.3g}"
            )
        pbe, pbo = (
            rec.get("param_bytes_eager"),
            rec.get("param_bytes_opt"),
        )
        if pbe and pbo and pbo != pbe:
            extras.append(f"params {pbe}→{pbo} B")
        peaks = {
            vn: rec.get(f"peak_mb_{vn}")
            for vn in ("eager", "morphism")
            if isinstance(rec.get(f"peak_mb_{vn}"), float)
        }
        if peaks:
            extras.append(
                "peakMB "
                + " ".join(f"{k}={v}" for k, v in peaks.items())
            )
        misp = rec.get("speedup_morph_ind_vs_ind")
        if misp:
            extras.append(f"m+i/ind {misp:.2f}×")
        extras.append(
            f"e2e_rel={rec.get('e2e_rel')} opt={rec.get('opt_s')}s"
        )
        print(f"{'':<20} | {'':<16} | {'':>13} | " + "  ".join(extras))
    print("-" * len(hdr))
    print(
        "  flops t/r = term-level ratio (cost model) / runtime ratio "
        "(aten profiler).  e/m = eager_ms/morphism_ms (>1 = faster).  "
        "conv = (e/m − 1)/(term ratio − 1) — share of the flops "
        "headroom that became wall time; negative = regression."
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    scored = [
        (r["name"], r["speedup_morph_vs_eager"])
        for r in recs
        if r.get("speedup_morph_vs_eager")
    ]
    best = max(scored, key=lambda t: t[1], default=None)
    n_verified = sum(1 for r in recs if r.get("morph_verified"))
    findings = [
        Finding(
            claim=(
                "morphism windows convert term-level flops headroom "
                "into measured wall time"
            ),
            verdict=(
                Verdict.WIN if best and best[1] > 1 else Verdict.NEGATIVE
            ),
            headline=(
                f"best {best[1]:.2f}× vs eager on {best[0]}"
                if best
                else "no cell converts to wall-time"
            ),
            metric="morphism / eager",
            value=best[1] if best else None,
            evidence={
                "term_flops_ratio": {
                    r["name"]: r.get("flops_ratio") for r in recs
                }
            },
        ),
        Finding(
            claim="every morphism rewrite verifies equivalent",
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
        suite="morphism_e2e",
        title="Morphism-window composition",
        summary=(
            "Term-flops → wall-time conversion for the morphism laws: "
            "which structural rewrites actually pay off at GEMM-bound "
            "sizes."
        ),
        findings=findings,
        cells=cells,
        env=collect_env(dev),
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "morphism_e2e.json"
        md_path = out_dir / "morphism_e2e.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "morphism-laws e2e: do term-level flops wins (WindowCompose"
            " / ResidualReassoc / KVLatentShare) convert to measured "
            "wall time?  eager vs inductor vs morphism vs "
            "morphism+inductor."
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="comma subset of " + ",".join(_FIXTURES),
    )
    ap.add_argument(
        "--cells",
        type=str,
        default=None,
        help="override every fixture's sweep — RxD list "
        "(per-fixture defaults: chains/resid "
        "'8x16,1024x64,4096x128', kv '8x16,512x64')",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=60.0,
        help="torch.compile budget per variant, seconds "
        "(0 disables compiled variants)",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument("--quick", action="store_true")
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
