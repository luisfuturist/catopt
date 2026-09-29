"""Law-effect bench — do the newly-wired law families pay at runtime?

Since the last e2e sweep, three law families entered the default
search paths:

* **LAYOUT_RULES** (``catopt_core.laws.layout``, in ``ALL_RULES``) —
  transpose migration through pointwise ops, product transposes, and
  the NT-GEMM bridge ``matmul(x, W.mT) ≡ linear(x, W)``.
* **DECODE_LAWS** (``catopt_carriers.decode_laws``, wired into
  ``CARRIER_LAWS`` in ``regime.py``) — ``cat(prefix_view(buf), new) ≡
  slice_scatter``, static ``index_select`` dedup, and the
  unsqueeze→expand→reshape head-repeat as an ``index_select``
  reindex.  Reachable through the carrier e-graph
  (``build_egraph`` / ``regime_dispatch``), NOT through
  ``optimize_model``'s ``all_rules()`` set — the bench records that
  difference explicitly.
* **Cross-block pair pass** (``max_cross_pairs=8`` in
  ``optimize_compositional``) — joint optimization of adjacent blocks
  across a chain/residual boundary.

One cell per family, each crafted so the family *should* pay:

* ``kv_append_decode`` — a GQA decode step that appends new K/V to a
  preallocated cache via ``cat(buf[:, :, :n], new)`` and repeats kv
  heads via the unsqueeze→expand→reshape chain.  DECODE_LAWS' target.
* ``cross_pair`` — two blocks where A ends in a wide Linear feeding
  B's two shared-input projections: the joint run can compose A's
  output projection into B's pair — invisible to per-block search.
* ``transpose_mlp`` — SwiGLU MLPs whose weights sit in ``(out, in)``
  storage and are consumed as ``x @ W.t()`` / ``x @ W.transpose(0,1)``
  — both NT-bridge spellings.

Per cell: eager / inductor / catopt(+inductor) variants, rule-fire
evidence, extracted-term op census, equivalence verification
(rel < 1e-4), benchkit timing.  Honest: a law that fires but isn't
picked, or is picked but doesn't pay at runtime, is reported as such.

Usage:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/laws_effect.py \
        --device cuda
    .venv/bin/python bench/laws_effect.py --device cpu --quick
"""
# ruff: noqa: E402, RUF003 -- ×, ·, → in strings/docstrings are
# deliberate math notation; sys.path setup must precede the
# benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import copy
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

from catopt_core.cost import flops_cost, launch_aware_cost
from catopt_core.ir import IR, Op
from catopt_orchestrator.regime import (
    Regime,
    build_egraph,
    regime_frontier,
)
from catopt_torch.report import verify_equiv
from catopt_torch.torch_bridge import ir_to_torch_module
from real_win_hunt import try_compile
from catopt_orchestrator import Compositional, Optimizer

from catopt_torch.backend import TorchBackend

# run_all.py picks these up for its --quick lane.
QUICK = {
    "min_run_time": "0.05",
    "compile_timeout": "30.0",
    "max_iterations": "10",
}

#: Decode-law prefixes — the fire-evidence filter for cell 1.
_DECODE_PREFIXES = (
    "kv_append_scatter",
    "scatter_to_cat",
    "index_select",
    "repeat_kv_as_gather",
)

#: Layout-law prefixes — the fire-evidence filter for cell 3.
_LAYOUT_PREFIXES = (
    "transpose_",
    "linear_from_matmul_t",
    "linear_to_matmul_t",
    "matmul_transpose_rev",
)


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------


def _term_census(term) -> dict[str, int]:
    """Op-name census of an IR term."""
    c: dict[str, int] = {}

    def rec(t) -> None:
        if isinstance(t, Op):
            c[t.op] = c.get(t.op, 0) + 1
            for ch in t.args:
                rec(ch)

    if term is not None:
        rec(term)
    return c


def _module_census(mod: nn.Module) -> dict[str, int]:
    """Sum the ``_root``-term census over every IRModule inside ``mod``.

    IRModule stores the delivered (weight-folded) term on ``_root``;
    compositional grafts (_FusedPair.inner etc.) hold IRModules, so
    walking submodules covers both delivery styles.
    """
    c: dict[str, int] = {}
    seen: set[int] = set()
    for m in mod.modules():
        t = getattr(m, "_root", None)
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            for k, v in _term_census(t).items():
                c[k] = c.get(k, 0) + v
    return c


def _fires_for(fires: dict, prefixes: tuple[str, ...]) -> dict:
    return {
        k: v
        for k, v in (fires or {}).items()
        if v and k.startswith(prefixes)
    }


def _fwd_stmt(mod, args: tuple):
    def stmt() -> None:
        with torch.no_grad():
            mod(*args)

    return stmt


def _verify(ref, mod, args: tuple, rtol: float = 1e-4):
    """→ (passed, max_rel); mod==None or eval failure → (False, inf)."""
    if mod is None:
        return False, float("inf")
    try:
        with torch.no_grad():
            vr = verify_equiv(ref, mod(*args), rtol=rtol)
        return bool(vr.passed), vr.max_rel
    except Exception:
        return False, float("inf")


# ---------------------------------------------------------------------------
#  Cell 1 — decode-ish KV-append step (DECODE_LAWS)
# ---------------------------------------------------------------------------


class DecodeStep(nn.Module):
    """One GQA decode step over a functional KV cache.

    ``x``: (B, 1, d) new token; ``kbuf``/``vbuf``: (B, nkv, ctx, hd)
    with ``ctx == n + 1`` — the tail IS the append slot, so
    ``cat(buf[:, :, :n], new)`` is exactly the ``kv_append_scatter``
    LHS.  The ``unsqueeze→expand→reshape`` GQA head copy is the
    ``repeat_kv_as_gather`` LHS.
    """

    def __init__(self, d: int = 256, nh: int = 8, nkv: int = 2) -> None:
        super().__init__()
        self.nh, self.nkv = nh, nkv
        self.hd = d // nh
        self.wq = nn.Linear(d, nh * self.hd, bias=False)
        self.wk = nn.Linear(d, nkv * self.hd, bias=False)
        self.wv = nn.Linear(d, nkv * self.hd, bias=False)
        self.wo = nn.Linear(nh * self.hd, d, bias=False)

    def forward(self, x, kbuf, vbuf):
        B = x.shape[0]
        ctx = kbuf.shape[2]
        n = ctx - 1
        q = self.wq(x).view(B, 1, self.nh, self.hd).transpose(1, 2)
        kn = self.wk(x).view(B, 1, self.nkv, self.hd).transpose(1, 2)
        vn = self.wv(x).view(B, 1, self.nkv, self.hd).transpose(1, 2)
        k = torch.cat([kbuf[:, :, :n], kn], dim=2)
        v = torch.cat([vbuf[:, :, :n], vn], dim=2)
        r = self.nh // self.nkv
        k = (
            k[:, :, None, :, :]
            .expand(B, self.nkv, r, ctx, self.hd)
            .reshape(B, self.nh, ctx, self.hd)
        )
        v = (
            v[:, :, None, :, :]
            .expand(B, self.nkv, r, ctx, self.hd)
            .reshape(B, self.nh, ctx, self.hd)
        )
        o = F.scaled_dot_product_attention(q, k, v)
        return self.wo(o.transpose(1, 2).reshape(B, 1, -1))


def _decode_cell(args, dev: torch.device) -> tuple[dict, Case]:
    d, nh, nkv, B, ctx = 256, 8, 2, 4, 1024
    torch.manual_seed(0)
    model = DecodeStep(d, nh, nkv).eval().to(dev)
    g = torch.Generator().manual_seed(77)
    x = torch.randn(B, 1, d, generator=g).to(dev)
    kb = torch.randn(B, nkv, ctx, d // nh, generator=g).to(dev)
    vb = torch.randn(B, nkv, ctx, d // nh, generator=g).to(dev)
    args_t = (x, kb, vb)
    max_it = int(getattr(args, "max_iterations", None) or 10)
    rec: dict = {
        "name": "kv_append_decode",
        "params": {"B": B, "ctx": ctx, "d": d, "nh": nh, "nkv": nkv},
    }
    with torch.no_grad():
        ref = model(*args_t)

    # -- carrier/regime path: CARRIER_LAWS carries DECODE_LAWS ---------
    t0 = time.time()
    eg, root, ir, src, st = build_egraph(
        model, args_t, max_iterations=max_it
    )
    rec["carrier_opt_s"] = round(time.time() - t0, 2)
    rec["carrier_enodes"] = st.get("n_enodes")
    rec["fires_decode"] = _fires_for(eg.rule_fires, _DECODE_PREFIXES)
    fr = regime_frontier(
        eg,
        root,
        [
            Regime(
                "decode",
                cost_fn=launch_aware_cost,
                executor="om_streaming",
            ),
            Regime("work", cost_fn=flops_cost, executor="generic"),
        ],
        ir=ir,
    )
    disp = fr.build(param_values=src)
    dec = fr["decode"]
    rec["decode_choice"] = {
        "executor": dec.executor,
        "degraded": dec.degraded,
        "forced": dec.forced,
        "label": dec.label,
        "census": _term_census(dec.term),
    }
    rec["work_census"] = _term_census(fr["work"].term)
    # Is the buffer-write spelling reachable? Count slice_scatter enodes.
    scatter_pins: dict[int, object] = {}
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        n = next((n for n in ec.nodes if n.op == "slice_scatter"), None)
        if n is not None:
            scatter_pins[c] = n
    rec["scatter_classes_reachable"] = len(scatter_pins)
    scatter_mod = None
    if scatter_pins:
        t_sc = eg.extract_best(
            eg.find(root), launch_aware_cost, overrides=scatter_pins
        )
        if t_sc is not None:
            rec["scatter_census"] = _term_census(t_sc)
            scatter_mod = ir_to_torch_module(
                IR(
                    root=t_sc,
                    inputs=ir.inputs,
                    input_names=ir.input_names,
                    params=ir.params,
                ),
                param_values=src,
            )

    # -- main path (optimize_model / ALL_RULES): does it see these? ----
    t0 = time.time()
    try:
        opt_m, st_m = Optimizer(backend=TorchBackend()).optimize(model, args_t, max_iterations=max_it, verify=False, verbose=False)

        rec["main_opt_s"] = round(time.time() - t0, 2)
        rec["fires_main_decode"] = _fires_for(
            st_m.get("rule_fires", {}), _DECODE_PREFIXES
        )
        rec["fires_main_all"] = {
            k: v for k, v in st_m.get("rule_fires", {}).items() if v
        }
        rec["main_census"] = _module_census(opt_m)
        ok, rel = _verify(ref, opt_m, args_t)
        rec["main_verified"], rec["main_rel"] = ok, rel
        if not ok:
            opt_m = None
    except Exception as e:
        opt_m = None
        rec["main_error"] = f"{type(e).__name__}: {e}"

    catopt_decode = disp.executor_module("decode")
    ok, rel = _verify(ref, catopt_decode, args_t)
    rec["decode_verified"], rec["decode_rel"] = ok, rel
    if not ok:
        catopt_decode = None
    ok, rel = _verify(ref, scatter_mod, args_t)
    rec["scatter_verified"], rec["scatter_rel"] = ok, rel
    if not ok:
        scatter_mod = None

    # -- compile -------------------------------------------------------
    ct = float(getattr(args, "compile_timeout", None) or 30.0)
    cm, rec["inductor_status"] = try_compile(
        copy.deepcopy(model), args_t, ct
    )
    ok, rel = _verify(ref, cm, args_t)
    rec["inductor_verified"], rec["inductor_rel"] = ok, rel
    if not ok:
        cm = None
    oci = None
    if catopt_decode is not None and ct > 0:
        oci, rec["decode_ind_status"] = try_compile(
            catopt_decode, args_t, ct
        )
        ok, rel = _verify(ref, oci, args_t)
        rec["decode_ind_verified"], rec["decode_ind_rel"] = ok, rel
        if not ok:
            oci = None

    variants = []
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("catopt_main", opt_m),
        ("catopt_decode", catopt_decode),
        ("catopt_decode+ind", oci),
        ("catopt_scatter", scatter_mod),
    ):
        if m is not None:
            variants.append(Variant(vn, _fwd_stmt(m, args_t)))
    return rec, Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Cell 2 — cross-block pair (max_cross_pairs joint pass)
# ---------------------------------------------------------------------------


class _WideUp(nn.Module):
    """Block A: narrow d → wide k, ENDING in a Linear — the boundary
    projection the joint pass can compose into B's projections."""

    def __init__(self, d: int = 64, k: int = 512) -> None:
        super().__init__()
        self.up = nn.Linear(d, k, bias=False)
        self.out = nn.Linear(k, k, bias=False)

    def forward(self, x):
        return self.out(F.gelu(self.up(x)))


class _WideDown(nn.Module):
    """Block B: SwiGLU on the wide stream → narrow d.  ``w1``/``w3``
    read A's output projection result — the cross-boundary share."""

    def __init__(self, k: int = 512, h: int = 256, d: int = 64) -> None:
        super().__init__()
        self.w1 = nn.Linear(k, h, bias=False)
        self.w3 = nn.Linear(k, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, u):
        return self.w2(F.silu(self.w1(u)) * self.w3(u))


class CrossPairNet(nn.Module):
    """Two chained blocks; B's input IS A's output (chain boundary)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_WideUp(), _WideDown()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _cross_cell(args, dev: torch.device) -> tuple[dict, Case]:
    B, T = 4, 256
    torch.manual_seed(0)
    model = CrossPairNet().eval().to(dev)
    g = torch.Generator().manual_seed(91)
    x = torch.randn(B, T, 64, generator=g).to(dev)
    args_t = (x,)
    max_it = int(getattr(args, "max_iterations", None) or 10)
    rec: dict = {
        "name": "cross_pair",
        "params": {"B": B, "T": T},
    }
    with torch.no_grad():
        ref = model(*args_t)

    # -- block-only pass (the pre-change behaviour) ---------------------
    t0 = time.time()
    opt_b, _rep_b = Optimizer(backend=TorchBackend()).optimize(model, x, strategy=Compositional(max_cross_pairs=0), verbose=False, max_iterations=max_it)

    rec["opt_blocks_s"] = round(time.time() - t0, 2)
    rec["blocks_census"] = _module_census(opt_b)
    ok, rel = _verify(ref, opt_b, args_t)
    rec["blocks_verified"], rec["blocks_rel"] = ok, rel
    if not ok:
        opt_b = None

    # -- with the cross-block pair pass --------------------------------
    t0 = time.time()
    opt_c, rep_c = Optimizer(backend=TorchBackend()).optimize(model, x, strategy=Compositional(max_cross_pairs=8), verbose=False, max_iterations=max_it)

    rec["opt_cross_s"] = round(time.time() - t0, 2)
    cp = (
        rep_c.get("cross_pairs")
        or (rep_c.get("extra") or {}).get("cross_pairs")
        or {}
    )
    rec["cross_pairs"] = {
        k: {
            kk: vv
            for kk, vv in v.items()
            if kk
            in (
                "status",
                "reason",
                "boundary",
                "joint_cost",
                "separate_cost",
                "time_s",
            )
        }
        for k, v in cp.items()
    }
    rec["cross_census"] = _module_census(opt_c)
    ok, rel = _verify(ref, opt_c, args_t)
    rec["cross_verified"], rec["cross_rel"] = ok, rel
    if not ok:
        opt_c = None

    # -- compile -------------------------------------------------------
    ct = float(getattr(args, "compile_timeout", None) or 30.0)
    cm, rec["inductor_status"] = try_compile(
        copy.deepcopy(model), args_t, ct
    )
    ok, rel = _verify(ref, cm, args_t)
    rec["inductor_verified"], rec["inductor_rel"] = ok, rel
    if not ok:
        cm = None
    obi = oci = None
    if opt_b is not None and ct > 0:
        obi, rec["blocks_ind_status"] = try_compile(opt_b, args_t, ct)
        ok, rel = _verify(ref, obi, args_t)
        rec["blocks_ind_verified"], rec["blocks_ind_rel"] = ok, rel
        if not ok:
            obi = None
    if opt_c is not None and ct > 0:
        oci, rec["cross_ind_status"] = try_compile(opt_c, args_t, ct)
        ok, rel = _verify(ref, oci, args_t)
        rec["cross_ind_verified"], rec["cross_ind_rel"] = ok, rel
        if not ok:
            oci = None

    variants = []
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("catopt_blocks", opt_b),
        ("catopt_blocks+ind", obi),
        ("catopt_cross", opt_c),
        ("catopt_cross+ind", oci),
    ):
        if m is not None:
            variants.append(Variant(vn, _fwd_stmt(m, args_t)))
    return rec, Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Cell 3 — transpose-heavy MLP (LAYOUT_RULES / NT-GEMM bridge)
# ---------------------------------------------------------------------------


class TmlpBlock(nn.Module):
    """SwiGLU MLP with weights in (out, in) storage consumed via an
    explicit transpose — ``x @ W.t()`` (bare) and ``x @
    W.transpose(0,1)`` (dimmed) — the two NT-bridge spellings."""

    def __init__(self, d: int = 256, h: int = 768) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(h, d) * 0.02)
        self.w3 = nn.Parameter(torch.randn(h, d) * 0.02)
        self.w2 = nn.Parameter(torch.randn(d, h) * 0.02)

    def forward(self, x):
        a = torch.matmul(x, self.w1.t())  # bare t() spelling
        b = torch.matmul(x, self.w3.transpose(0, 1))  # dimmed spelling
        return torch.matmul(F.silu(a) * b, self.w2.t())


class TmlpNet(nn.Module):
    def __init__(self, d: int = 256, h: int = 768, layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            TmlpBlock(d, h) for _ in range(layers)
        )

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


def _layout_cell(args, dev: torch.device) -> tuple[dict, Case]:
    B, T = 4, 256
    torch.manual_seed(0)
    model = TmlpNet().eval().to(dev)
    g = torch.Generator().manual_seed(55)
    x = torch.randn(B, T, 256, generator=g).to(dev)
    args_t = (x,)
    max_it = int(getattr(args, "max_iterations", None) or 10)
    rec: dict = {
        "name": "transpose_mlp",
        "params": {"B": B, "T": T},
    }
    with torch.no_grad():
        ref = model(*args_t)

    t0 = time.time()
    try:
        opt_m, st_m = Optimizer(backend=TorchBackend()).optimize(model, x, max_iterations=max_it, verify=False, verbose=False)

        rec["opt_s"] = round(time.time() - t0, 2)
        rec["fires_layout"] = _fires_for(
            st_m.get("rule_fires", {}), _LAYOUT_PREFIXES
        )
        rec["paired_extract"] = st_m.get("paired_extract")
        rec["pairing_groups"] = st_m.get("pairing_groups")
        rec["census_extracted"] = _module_census(opt_m)
        ok, rel = _verify(ref, opt_m, args_t)
        rec["catopt_verified"], rec["catopt_rel"] = ok, rel
        if not ok:
            opt_m = None
    except Exception as e:
        opt_m = None
        rec["opt_error"] = f"{type(e).__name__}: {e}"

    ct = float(getattr(args, "compile_timeout", None) or 30.0)
    cm, rec["inductor_status"] = try_compile(
        copy.deepcopy(model), args_t, ct
    )
    ok, rel = _verify(ref, cm, args_t)
    rec["inductor_verified"], rec["inductor_rel"] = ok, rel
    if not ok:
        cm = None
    oci = None
    if opt_m is not None and ct > 0:
        oci, rec["catopt_ind_status"] = try_compile(opt_m, args_t, ct)
        ok, rel = _verify(ref, oci, args_t)
        rec["catopt_ind_verified"], rec["catopt_ind_rel"] = ok, rel
        if not ok:
            oci = None

    variants = []
    for vn, m in (
        ("eager", model),
        ("inductor", cm),
        ("catopt", opt_m),
        ("catopt+inductor", oci),
    ):
        if m is not None:
            variants.append(Variant(vn, _fwd_stmt(m, args_t)))
    return rec, Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Harness entry point (run_all.py convention)
# ---------------------------------------------------------------------------

_CELLS = {
    "kv_append_decode": _decode_cell,
    "cross_pair": _cross_cell,
    "transpose_mlp": _layout_cell,
}


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    min_run_time = float(getattr(args, "min_run_time", None) or 0.1)
    warmup = int(getattr(args, "warmup", None) or 3)

    names = (getattr(args, "cells", None) or ",".join(_CELLS)).split(
        ","
    )
    names = [n.strip() for n in names if n.strip()]

    print(f"laws_effect — device={dev} cells={names}", flush=True)
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cells: list = []
    for name in names:
        fn = _CELLS[name]
        print(f"\n=== {name} ===", flush=True)
        rec, case = fn(args, dev)
        recs.append(rec)
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

    # -- console table ---------------------------------------------------
    all_names: list[str] = []
    for c in cells:
        for n in c.medians:
            if n not in all_names:
                all_names.append(n)
    hdr = (
        f"{'cell':<18} | "
        + " | ".join(f"{n:>16}" for n in all_names)
        + f" | {'ver':>3}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians
        row = f"{rec['name']:<18} | " + " | ".join(
            (
                f"{ms[n] * 1e3:>8.3f} ms     "
                if n in ms
                else f"{'—':>16}"
            )
            for n in all_names
        )
        ver = (
            "yes"
            if rec.get(
                "cross_verified",
                rec.get("decode_verified", rec.get("catopt_verified")),
            )
            else "NO"
        )
        print(f"{row} | {ver:>3}")
        if rec["name"] == "kv_append_decode":
            print(
                f"{'':<18} | decode fires: {rec.get('fires_decode')}  "
                f"| main-path decode fires: "
                f"{rec.get('fires_main_decode')}  | scatter classes "
                f"reachable: {rec.get('scatter_classes_reachable')}"
            )
            print(
                f"{'':<18} | decode pick census: "
                f"{rec.get('decode_choice', {}).get('census')}"
            )
        elif rec["name"] == "cross_pair":
            print(f"{'':<18} | cross_pairs: {rec.get('cross_pairs')}")
            print(
                f"{'':<18} | census blocks={rec.get('blocks_census')} "
                f"cross={rec.get('cross_census')}"
            )
        elif rec["name"] == "transpose_mlp":
            print(
                f"{'':<18} | layout fires: {rec.get('fires_layout')}  "
                f"| extracted census: {rec.get('census_extracted')}"
            )
    print("-" * len(hdr))
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    report = Report(
        suite="laws_effect", cells=cells, env=collect_env(dev)
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"laws_effect_{ts}.json"
        md_path = out_dir / f"laws_effect_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="inductor")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "law-effect bench: do the newly-default law families "
            "(LAYOUT_RULES / DECODE_LAWS / cross-block pairs) pay?"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--cells",
        type=str,
        default=None,
        help="comma subset of " + ",".join(_CELLS),
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=30.0,
        help="torch.compile budget per variant, seconds "
        "(0 disables compiled variants)",
    )
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-run-time", type=float, default=0.1)
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
