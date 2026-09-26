"""Cost-model fidelity — predicted cost vs measured latency.

Plan 0005's question: does the cost model's *ordering* of the
semantic-equivalence frontier track measured end-to-end latency?
Per cell (model, shape) this script

1. enumerates the frontier with ``discover_alternatives`` — the top-k
   cheapest *distinct* members of the equivalence class [G] under the
   extraction cost fn — plus the as-exported root, the canonical
   balanced carrier term (scan families), and the ``optimize_model``
   pick;
2. prices every candidate term under EVERY cost fn —
   ``flops_cost``, ``launch_aware_cost``, ``count_cost``,
   ``param_bytes_cost``, ``depth_cost``, the hardcoded RTX-2050
   roofline, and the ``calibrate()``-measured ``roofline`` / ``depth``
   closures — through ``dag_cost`` under ``backend_cost``, the same
   DAG-true backend-relative accounting extraction uses;
3. lowers every candidate through each applicable executor — the
   generic ``IRModule`` for every term, plus the level-batched carrier
   executor (``to_batched_scan_module``) for ``apply[d]`` scan terms —
   verifies it fp64 against the eager reference (``rel_to_max``
   reported; the fp32 ``allclose`` gate rtol=1e-4 / atol=1e-5 decides
   who gets timed — failures are recorded, not dropped);
4. measures each verified candidate with the benchkit ``Runner``
   (median via ``blocked_autorange``), alongside the anchors: the
   eager module, Inductor-compiled eager, and the ``optimize_model``
   output (eager + compiled);
5. reports, per cost fn per cell: **Spearman ρ**, **Kendall τ-b**,
   **pick accuracy** (does argmin predicted == argmin measured?), and
   the residual breakdown — every candidate whose predicted rank and
   measured rank differ by ≥3, the systematic mispredictions.

The candidate set is a (term × lowering) product, not just a term
list: two executors of one term share every predicted cost, so an
executor-side inversion (a batched scan measured slower than priced,
or Inductor's pointwise fusion measured faster) shows up directly as
a ρ drop and a rank residual — the question the plan actually asks.
When ``discover_alternatives`` returns near-identical members (the
top-k all of one form family — e.g. every chain alternative is some
weights-first bracketing) the record says so; the lowering axis is
what gives such cells discrimination.

Artifacts: benchkit ``Report`` → timestamped JSON + Markdown, a
sidecar ``cost_fidelity_<ts>.json`` with the full candidate table
(every predicted cost, verify diff, median ms, rank residual), and
plots: a per-cell predicted-vs-measured scatter (one log-log panel
per cost fn) plus a ρ-by-cost-fn bar chart across cells.

Usage:
    python bench/cost_fidelity.py --device cpu --quick
    python bench/cost_fidelity.py --device cpu --models chain,swiglu
    python bench/cost_fidelity.py --device cuda --sizes 128,512
"""
# ruff: noqa: E402 RUF001 RUF002 RUF003 — ρ, τ, ×, ≥, − in strings are
# deliberate math notation; sys.path setup must precede the
# benchkit/catopt imports (reassoc_scale / real_linear_attn convention).

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.setrecursionlimit(400_000)

import torch
from benchkit import Case, Report, Runner, Variant, collect_env
from catopt.adapters import TorchSink
from catopt.calibrate import calibrate
from catopt.cost import (
    backend_cost,
    count_cost,
    dag_cost,
    depth_cost,
    depth_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt.ir import IR, op_repr
from catopt.models import AttentionBlock, SwiGLU
from catopt.optimize import (
    OptimizationResourceError,
    discover_alternatives,
    optimize_model,
)
from catopt.scan_lower import is_scan_apply_term, to_batched_scan_module
from catopt.torch_bridge import ir_to_torch_module
from real_linear_attn import (
    LinearAttnStack,
    _canonical_scan_term,
    _rel_diff,
    try_compile,
)
from reassoc_scale import LinearAttnChain

# run_all.py picks these up for its --quick lane.
QUICK = {"models": "chain,retnet", "top_k": 4}

_SCAN_MODES = ("retnet", "gla", "delta")
_ALL_MODELS = ("chain", "retnet", "gla", "delta", "attn", "swiglu")
#: Per-model size axis when --sizes is not given: chain sweeps the
#: chain depth k, the scan/attention families sweep the horizon T,
#: swiglu sweeps the token count R.
_DEFAULT_SIZES = {
    "chain": [4, 8],
    "retnet": [128, 512],
    "gla": [128, 512],
    "delta": [128, 512],
    "attn": [128, 512],
    "swiglu": [4096, 16384],
}
_SIZE_LABEL = {"chain": "k", "swiglu": "R"}

#: Cost-fn registry order — also the scatter-panel order.
_COST_FN_NAMES = (
    "flops",
    "launch_aware",
    "count",
    "param_bytes",
    "depth",
    "roofline_rtx",
    "roofline_cal",
    "depth_cal",
)


def _cost_table(profile, source_tensors):
    """Raw cost fns keyed by report name (backend_cost wrap happens at
    pricing time so every number is the DAG-true backend-relative
    cost extraction would have used)."""
    return {
        "flops": flops_cost,
        "launch_aware": launch_aware_cost,
        "count": count_cost,
        "param_bytes": param_bytes_cost_for(source_tensors),
        "depth": depth_cost,
        "roofline_rtx": roofline_cost,
        "roofline_cal": roofline_cost_for(profile),
        "depth_cal": depth_cost_for(profile),
    }


# ---------------------------------------------------------------------------
#  Rank statistics (scipy is not a dependency)
# ---------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    """1-based ranks, average on ties."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    rk = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rk[order[k]] = r
        i = j + 1
    return rk


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return sxy / math.sqrt(sxx * syy)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman ρ — Pearson on average ranks."""
    if len(x) < 3:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall τ-b — concordant-minus-discordant pairs, tie-adjusted."""
    n = len(x)
    if n < 2:
        return float("nan")
    con = dis = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                con += 1
            else:
                dis += 1
    denom = math.sqrt((con + dis + tx) * (con + dis + ty))
    return (con - dis) / denom if denom else float("nan")


def _fidelity_metrics(candidates: list[dict], fn: str) -> dict:
    """ρ / τ-b / pick-accuracy / rank residuals for one cost fn.

    ``candidates`` carry ``pred`` (cost-fn → predicted) and ``ms``
    (median ms or None).  Only candidates with a finite prediction
    and a measured latency enter the ranking.
    """
    pts = [
        c
        for c in candidates
        if c.get("ms") is not None
        and isinstance(c["pred"].get(fn), float)
        and math.isfinite(c["pred"][fn])
    ]
    out = {"fn": fn, "n": len(pts)}
    if len(pts) < 2:
        return out
    xs = [c["pred"][fn] for c in pts]
    ys = [c["ms"] for c in pts]
    rx = _ranks(xs)
    ry = _ranks(ys)
    out["rho"] = _spearman(xs, ys)
    out["tau"] = _kendall_tau(xs, ys)
    i_pred = min(range(len(pts)), key=lambda i: xs[i])
    i_meas = min(range(len(pts)), key=lambda i: ys[i])
    out["pick_pred"] = pts[i_pred]["name"]
    out["pick_meas"] = pts[i_meas]["name"]
    out["pick_ok"] = i_pred == i_meas
    out["mispred"] = [
        {
            "name": c["name"],
            "pred_rank": rx[i],
            "meas_rank": ry[i],
            "pred": xs[i],
            "ms": ys[i],
        }
        for i, c in enumerate(pts)
        if abs(rx[i] - ry[i]) >= 3
    ]
    return out


# ---------------------------------------------------------------------------
#  Models — one builder per family
# ---------------------------------------------------------------------------


def _build_model(
    model_name: str, size: int, args, dev: torch.device
) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    """``(fp64 model, fp64 input, meta)`` for one (family, size) cell."""
    d = args.d
    if model_name == "chain":
        m = LinearAttnChain(args.chain_d, size, seed=0)
        x = torch.randn(args.chain_rows, args.chain_d)
        meta = {
            "size_name": "k",
            "d": args.chain_d,
            "rows": args.chain_rows,
        }
    elif model_name in _SCAN_MODES:
        m = LinearAttnStack(
            d, mode=model_name, k=args.value_depth, n_blocks=1, seed=0
        )
        x = torch.randn(size, d)
        meta = {"size_name": "T", "d": d, "k": args.value_depth}
    elif model_name == "attn":
        m = AttentionBlock(d, n_heads=args.heads)
        x = torch.randn(args.batch, size, d)
        meta = {"size_name": "T", "d": d, "B": args.batch}
    elif model_name == "swiglu":
        m = SwiGLU(d)
        x = torch.randn(size, d)
        meta = {"size_name": "R", "d": d}
    else:
        raise ValueError(f"unknown model {model_name!r}")
    m64 = m.to(torch.float64).to(dev).eval()
    return m64, x.to(device=dev, dtype=torch.float64), meta


# ---------------------------------------------------------------------------
#  Lowering / verification helpers
# ---------------------------------------------------------------------------


@contextmanager
def _default_dtype(dt: torch.dtype):
    """Eval context forcing ``torch.get_default_dtype`` — the ``eye`` /
    constant-morphism bindings read it when they materialise tensors."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _ir_of(term, ir: IR) -> IR:
    return IR(
        root=term,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )


def _fp32_params(src64: dict, param_map: dict) -> dict:
    """fp32 param_values for a re-lowering: every exported tensor plus
    any ``fused_*`` parameters the fp64 lowerer materialised (they live
    in the built module's ``_param_map``, not in ``src64``)."""
    out = {n: v.detach().to(torch.float32) for n, v in src64.items()}
    for n, p in param_map.items():
        out.setdefault(n, p.detach().to(torch.float32))
    return out


def _timed_stmt(fn, x: torch.Tensor):
    """Zero-arg variant callable: one inference under ``no_grad``."""

    def stmt() -> None:
        with torch.no_grad():
            fn(x)

    return stmt


def _verify(c: dict, mod64, mod32, x64, ref64, x32, ref32) -> bool:
    """fp64 rel + fp32 gate for one lowered candidate.  ``mod64`` may be
    None (compiled-fp32-only candidates): rel64 stays unset."""
    if mod64 is not None:
        with torch.no_grad(), _default_dtype(torch.float64):
            out64 = mod64(x64)
        c["rel64"] = _rel_diff(out64, ref64)["rel_to_max"]
    with torch.no_grad(), _default_dtype(torch.float32):
        out32 = mod32(x32)
    c["rel32"] = _rel_diff(out32, ref32)["rel_to_max"]
    c["gate"] = bool(torch.allclose(out32, ref32, rtol=1e-4, atol=1e-5))
    return c["gate"]


# ---------------------------------------------------------------------------
#  The sweep cell
# ---------------------------------------------------------------------------


def run_cell(
    model_name: str,
    size: int,
    args,
    dev: torch.device,
    *,
    runner: Runner,
    profile,
    supported: frozenset,
) -> tuple[dict, object | None]:
    """One (model, size) cell → ``(record, benchkit.Cell | None)``."""
    torch.manual_seed(0)
    m64, x64, meta = _build_model(model_name, size, args, dev)
    tag = f"{model_name} {meta['size_name']}={size} d={meta['d']}"
    cell: dict = {"model": model_name, "size": size, **meta}
    print(f"\n=== {tag} ===", flush=True)
    with torch.no_grad():
        ref64 = m64(x64)

    # -- 1. enumerate the frontier ------------------------------------
    t0 = time.time()
    try:
        disc = discover_alternatives(
            m64, x64, top_k=args.top_k, ruleset=args.ruleset
        )
    except Exception as e:
        cell["error"] = f"discover: {type(e).__name__}: {e}"
        print(f"  discover_alternatives FAILED: {e}", flush=True)
        return cell, None
    cell["discover_s"] = round(time.time() - t0, 2)
    ir, src = disc["ir"], disc["source_tensors"]
    alts = disc["alternatives"]
    st = disc["stats"]
    cell["n_alts"] = len(alts)
    cell["enodes"] = st.get("n_enodes")
    cell["n_classes"] = st.get("n_classes")
    cell["diverse_classes"] = len(disc["diverse_classes"])
    cell["rule_fires"] = dict(
        list(
            sorted(
                ((k, v) for k, v in disc["rule_fires"].items() if v),
                key=lambda kv: -kv[1],
            )
        )[:12]
    )

    # -- term table: orig + alternatives + canonical scan + opt -------
    terms: list[dict] = []
    seen_reprs: set[str] = set()

    def add_term(
        name: str, kind: str, term, sel_cost=None
    ) -> dict | None:
        r = op_repr(term)
        if r in seen_reprs:
            return None
        seen_reprs.add(r)
        e = {
            "name": name,
            "kind": kind,
            "term": term,
            "repr": r,
            "sel_cost": sel_cost,
            "scan": bool(is_scan_apply_term(term)),
        }
        terms.append(e)
        return e

    add_term("orig", "orig", ir.root)
    root_ops = set()
    for i, (c, t) in enumerate(alts):
        if isinstance(t, torch.Tensor):  # pragma: no cover — defensive
            continue
        root_ops.add(getattr(t, "op", type(t).__name__))
        add_term(f"alt{i}", "alt", t, sel_cost=float(c))
    cell["alt_root_ops"] = sorted(root_ops)
    if model_name in _SCAN_MODES:
        canon, cinfo = _canonical_scan_term(ir, model_name)
        cell["canon_info"] = cinfo
        if canon is not None and is_scan_apply_term(canon):
            add_term("canon", "canon", canon)
        else:
            cell["canon_note"] = "no canonical scan term"
    # One-family frontier → the lowering axis does the discriminating.
    cell["frontier_note"] = (
        "frontier members are near-identical — executor variants "
        "carry the discrimination"
        if len(root_ops) <= 2
        else f"{len(root_ops)} distinct root ops on the frontier"
    )
    print(
        f"  frontier: {len(alts)} alts ({cell['n_alts']} distinct), "
        f"root ops {cell['alt_root_ops']} | enodes={cell['enodes']} "
        f"| {cell['discover_s']:.1f}s",
        flush=True,
    )

    # -- optimize_model anchor -----------------------------------------
    opt64 = None
    opt_entry = None
    do_opt = model_name not in _SCAN_MODES or size <= args.opt_max_t
    if do_opt:
        t0 = time.time()
        try:
            opt64, _ostats = optimize_model(
                m64,
                x64,
                verbose=False,
                max_iterations=32,
                max_enodes=300_000,
            )
            cell["opt_s"] = round(time.time() - t0, 2)
            cell["opt_root"] = getattr(
                opt64._root, "op", type(opt64._root).__name__
            )
        except OptimizationResourceError as e:
            cell["opt_error"] = f"resource: {e}"
        except Exception as e:
            cell["opt_error"] = f"{type(e).__name__}: {e}"
        if opt64 is not None:
            # The folded root may repr-collide with a frontier term —
            # reuse that entry's pricing rather than duplicating it.
            r = op_repr(opt64._root)
            opt_entry = next((e for e in terms if e["repr"] == r), None)
            if opt_entry is None:
                opt_entry = add_term("opt", "opt", opt64._root)
            print(
                f"  optimize_model {cell['opt_s']:.1f}s "
                f"root={cell['opt_root']}",
                flush=True,
            )
        else:
            print(f"  optimize_model: {cell['opt_error']}", flush=True)
    else:
        cell["opt_error"] = f"skipped (size>{args.opt_max_t})"

    # -- 2. price every term under every cost fn -----------------------
    raw_fns = _cost_table(profile, src)
    fns = {n: backend_cost(f, supported) for n, f in raw_fns.items()}
    for e in terms:
        memo: dict = {}
        pred: dict[str, float] = {}
        errs: dict[str, str] = {}
        for fn_name, fn in fns.items():
            try:
                pred[fn_name] = float(dag_cost(e["term"], fn, memo))
            except Exception as exc:
                pred[fn_name] = float("nan")
                errs[fn_name] = f"{type(exc).__name__}"
        e["pred"] = pred
        if errs:
            e["pred_err"] = errs

    # -- 3. lower + verify every (term × lowering) candidate -----------
    m32 = copy.deepcopy(m64).float().to(dev).eval()
    x32 = x64.float()
    with torch.no_grad():
        ref32 = m32(x32)
    torch._dynamo.reset()

    orig = next(e for e in terms if e["name"] == "orig")
    cands: list[dict] = []

    def _cand(name: str, entry: dict, lowering: str) -> dict:
        c = {
            "name": name,
            "term": entry["name"],
            "kind": entry["kind"],
            "lowering": lowering,
            "pred": entry["pred"],
            "scan": entry["scan"],
            "status": "pending",
            "ms": None,
        }
        cands.append(c)
        return c

    def _lower_irmod(entry: dict, dtype: torch.dtype):
        vals = (
            src
            if dtype == torch.float64
            else _fp32_params(src, entry["_mod64"]._param_map)
        )
        return ir_to_torch_module(_ir_of(entry["term"], ir), vals)

    def _lower_scan(entry: dict, dtype: torch.dtype):
        vals = (
            src
            if dtype == torch.float64
            else _fp32_params(src, entry["_mod64"].eval_mod._param_map)
        )
        return to_batched_scan_module(_ir_of(entry["term"], ir), vals)

    # eager + inductor anchors on the original term.
    c_eager = _cand("eager", orig, "eager")
    c_eager["status"] = (
        "ok"
        if _verify(c_eager, m64, m32, x64, ref64, x32, ref32)
        else "verify_fail"
    )
    c_eager["mod"] = m32

    c_ind = _cand("inductor", orig, "inductor")
    cm32, status = try_compile(m32, x32, args.compile_timeout)
    cell["inductor_status"] = status
    if cm32 is not None:
        try:
            c_ind["status"] = (
                "ok"
                if _verify(c_ind, None, cm32, x64, ref64, x32, ref32)
                else "verify_fail"
            )
        except Exception as e:
            c_ind["status"] = f"eval_fail: {type(e).__name__}: {e}"
            cm32 = None
        if c_ind["status"] == "ok":
            c_ind["mod"] = cm32
    else:
        c_ind["status"] = f"compile_fail: {status}"
    print(
        f"  inductor: {status} (candidate: {c_ind['status']})",
        flush=True,
    )

    # optimize_model output — generic executor + compiled.
    if opt64 is not None and opt_entry is not None:
        c_opt = _cand("opt_irmod", opt_entry, "irmod")
        try:
            opt_ir = _ir_of(opt64._root, ir)
            opt32 = (
                ir_to_torch_module(
                    opt_ir, _fp32_params(src, opt64._param_map)
                )
                .to(dev)
                .eval()
            )
            c_opt["status"] = (
                "ok"
                if _verify(c_opt, opt64, opt32, x64, ref64, x32, ref32)
                else "verify_fail"
            )
            if c_opt["status"] == "ok":
                c_opt["mod"] = opt32
        except Exception as e:
            c_opt["status"] = f"lower_fail: {type(e).__name__}: {e}"
        if c_opt["status"] == "ok":
            c_oi = _cand("opt_ind", opt_entry, "inductor")
            co32, status = try_compile(
                c_opt["mod"], x32, args.compile_timeout
            )
            cell["opt_ind_status"] = status
            if co32 is not None:
                try:
                    c_oi["status"] = (
                        "ok"
                        if _verify(
                            c_oi, None, co32, x64, ref64, x32, ref32
                        )
                        else "verify_fail"
                    )
                    if c_oi["status"] == "ok":
                        c_oi["mod"] = co32
                except Exception as e:
                    c_oi["status"] = (
                        f"eval_fail: {type(e).__name__}: {e}"
                    )
            else:
                c_oi["status"] = f"compile_fail: {status}"

    # frontier terms × lowerings.
    for e in terms:
        if e is opt_entry:
            continue  # lowered above from opt64 itself
        for lowering in ["irmod", "scan"] if e["scan"] else ["irmod"]:
            c = _cand(f"{e['name']}_{lowering}", e, lowering)
            lower = _lower_scan if lowering == "scan" else _lower_irmod
            try:
                with _default_dtype(torch.float64):
                    e["_mod64"] = lower(e, torch.float64)
                if lowering == "scan":
                    c["is_batched"] = bool(e["_mod64"].is_batched)
                    c["n_levels"] = int(e["_mod64"].n_levels)
                mod32 = lower(e, torch.float32).to(dev).eval()
                ok = _verify(
                    c, e["_mod64"], mod32, x64, ref64, x32, ref32
                )
                c["status"] = "ok" if ok else "verify_fail"
                if ok:
                    c["mod"] = mod32
                    if args.compile_alts:
                        ci, cstat = try_compile(
                            mod32, x32, args.compile_timeout
                        )
                        cc = _cand(f"{c['name']}_ind", e, "inductor")
                        if ci is not None:
                            try:
                                okc = _verify(
                                    cc, None, ci, x64, ref64, x32, ref32
                                )
                                cc["status"] = (
                                    "ok" if okc else "verify_fail"
                                )
                                if okc:
                                    cc["mod"] = ci
                            except Exception as exc:
                                cc["status"] = (
                                    f"eval_fail: {type(exc).__name__}: {exc}"
                                )
                        else:
                            cc["status"] = f"compile_fail: {cstat}"
            except Exception as exc:
                c["status"] = f"lower_fail: {type(exc).__name__}: {exc}"
            e.pop("_mod64", None)  # drop the fp64 module — done with it

    cell["verify_failures"] = [
        c["name"] for c in cands if c["status"] != "ok"
    ]
    n_ok = sum(1 for c in cands if c["status"] == "ok")
    print(
        f"  candidates: {len(cands)} ({n_ok} verified) — "
        f"failures: {cell['verify_failures'] or 'none'}",
        flush=True,
    )

    # -- 4. benchkit timing ---------------------------------------------
    # Pre-flight: one no_grad call per verified candidate.  Candidates
    # slower than --slow-cap are recorded as an honest ONE-SHOT latency
    # and excluded from autorange — e.g. the trace-carrier members take
    # ~15 s/call (a giant linalg.solve the per-op models price like a
    # GEMM); running them through warmup+autorange would dominate the
    # sweep without changing the conclusion.  The pre-flight call is
    # also free warmup for everyone else.
    timed = [
        c
        for c in cands
        if c["status"] == "ok" and c.get("mod") is not None
    ]
    for c in timed:
        with torch.no_grad():
            t0 = time.perf_counter()
            c["mod"](x32)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
        if dt > args.slow_cap:
            c["ms"] = dt * 1e3
            c["one_shot"] = True
            print(
                f"  {c['name']}: {dt:.1f}s one-shot >{args.slow_cap:.0f}s "
                f"cap — recorded, excluded from autorange",
                flush=True,
            )
    variants = [
        Variant(
            name=c["name"],
            stmt=_timed_stmt(c["mod"], x32),
            note=(
                f"{c['kind']}:{c['term']} via {c['lowering']}"
                + (" [scan-apply]" if c["scan"] else "")
            ),
        )
        for c in timed
        if not c.get("one_shot")
    ]
    if not variants and not any(c.get("one_shot") for c in timed):
        cell["error"] = "no verified candidates to time"
        return cell, None
    case = Case(
        name=f"{model_name}_{meta['size_name']}{size}",
        params={
            "model": model_name,
            meta["size_name"]: size,
            "d": meta["d"],
        },
        variants=variants,
        aux={},
    )
    ran = runner.run_case(case)
    for c in cands:
        if c["name"] in ran.medians:
            c["ms"] = float(ran.medians[c["name"]]) * 1e3
    cell["ms"] = {
        c["name"]: c["ms"] for c in cands if c["ms"] is not None
    }

    # -- 5. per-cost-fn fidelity ----------------------------------------
    cell["metrics"] = {
        fn: _fidelity_metrics(cands, fn) for fn in _COST_FN_NAMES
    }
    print("  fidelity (predicted cost vs measured ms):", flush=True)
    print(
        f"    {'cost fn':<14} {'n':>3} {'rho':>7} {'tau':>7}  pick",
        flush=True,
    )
    for fn in _COST_FN_NAMES:
        mt = cell["metrics"][fn]
        if mt.get("n", 0) < 2:
            print(f"    {fn:<14} {mt.get('n', 0):>3} {'—':>7} {'—':>7}")
            continue
        pick = (
            f"{mt['pick_pred']} {'==' if mt['pick_ok'] else '!='} "
            f"{mt['pick_meas']}"
        )
        print(
            f"    {fn:<14} {mt['n']:>3} {mt['rho']:>7.3f} "
            f"{mt['tau']:>7.3f}  {pick}",
            flush=True,
        )
        for mp in mt["mispred"]:
            print(
                f"      mispred {mp['name']}: pred rank {mp['pred_rank']:.0f} "
                f"vs measured {mp['meas_rank']:.0f} "
                f"({mp['pred']:.3g} → {mp['ms']:.3f} ms)",
                flush=True,
            )

    # slim candidate records for the report/JSON
    cell["candidates"] = [
        {
            k: c[k]
            for k in (
                "name",
                "term",
                "kind",
                "lowering",
                "scan",
                "status",
                "rel64",
                "rel32",
                "ms",
                "one_shot",
                "is_batched",
                "n_levels",
            )
            if k in c
        }
        | {"pred": c["pred"]}
        for c in cands
    ]
    cell["term_reprs"] = {e["name"]: e["repr"][:200] for e in terms}
    ran.aux = {
        "verify": {
            "gate_rtol": 1e-4,
            "gate_atol": 1e-5,
            "failures": cell["verify_failures"],
        },
        "frontier": {
            "n_alts": cell["n_alts"],
            "alt_root_ops": cell["alt_root_ops"],
            "diverse_classes": cell["diverse_classes"],
            "note": cell["frontier_note"],
        },
        "rho": {
            fn: round(cell["metrics"][fn]["rho"], 4)
            for fn in _COST_FN_NAMES
            if math.isfinite(
                cell["metrics"][fn].get("rho", float("nan"))
            )
        },
    }
    return cell, ran


# ---------------------------------------------------------------------------
#  Report supplement + plots
# ---------------------------------------------------------------------------


def _md_supplement(results: list[dict]) -> str:
    """Markdown appended after ``Report.to_markdown``: the fidelity
    tables — ρ/τ/pick per cost fn per cell plus the flagged rank
    residuals."""
    lines = ["", "## Cost-model fidelity (per cell)", ""]
    for c in results:
        tag = f"{c['model']} {c.get('size_name', '?')}={c['size']}"
        if "error" in c and "metrics" not in c:
            lines.append(f"### {tag}\n\n- ERROR `{c['error']}`")
            continue
        lines += [
            f"### {tag}",
            "",
            f"- frontier: {c.get('n_alts')} alts, root ops "
            f"`{c.get('alt_root_ops')}` — {c.get('frontier_note')}",
            f"- verify failures: {c.get('verify_failures') or 'none'}",
            "",
            "| cost fn | n | ρ | τ | pick predicted | pick measured | ok |",
            "|---|---|---|---|---|---|---|",
        ]
        for fn in _COST_FN_NAMES:
            mt = c["metrics"].get(fn, {})
            if mt.get("n", 0) < 2:
                lines.append(
                    f"| {fn} | {mt.get('n', 0)} | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| {fn} | {mt['n']} | {mt['rho']:.3f} | {mt['tau']:.3f} "
                f"| {mt['pick_pred']} | {mt['pick_meas']} "
                f"| {'✓' if mt['pick_ok'] else '✗'} |"
            )
        bad = [
            (fn, mp)
            for fn in _COST_FN_NAMES
            for mp in c["metrics"].get(fn, {}).get("mispred", [])
        ]
        if bad:
            lines += ["", "systematic mispredictions (rank gap ≥3):"]
            for fn, mp in bad:
                lines.append(
                    f"- `{mp['name']}` under **{fn}**: predicted rank "
                    f"{mp['pred_rank']:.0f} vs measured {mp['meas_rank']:.0f} "
                    f"(pred {mp['pred']:.3g} → {mp['ms']:.3f} ms)"
                )
        lines.append("")
    lines += ["", "## Summary", ""]
    ok = [c for c in results if c.get("metrics")]
    if not ok:
        lines.append("no measured cells — honest negative result.")
        return "\n".join(lines) + "\n"
    for fn in _COST_FN_NAMES:
        rhos = [
            c["metrics"][fn]["rho"]
            for c in ok
            if math.isfinite(c["metrics"][fn].get("rho", float("nan")))
        ]
        picks = sum(1 for c in ok if c["metrics"][fn].get("pick_ok"))
        if rhos:
            lines.append(
                f"- **{fn}**: median ρ {sorted(rhos)[len(rhos) // 2]:.3f} "
                f"over {len(rhos)} cell(s); pick accuracy "
                f"{picks}/{len(ok)}"
            )
    return "\n".join(lines) + "\n"


def _emit_plots(
    results: list[dict], plots_dir: Path, ts: str
) -> list[Path]:
    """Per-cell predicted-vs-measured scatter (one panel per cost fn,
    log-log) + a ρ-by-cost-fn grouped bar chart across cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    ok = [c for c in results if c.get("metrics")]
    for c in ok:
        tag = f"{c['model']}_{c.get('size_name', '?')}{c['size']}"
        ncols = 4
        nrows = math.ceil(len(_COST_FN_NAMES) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.6 * nrows),
            squeeze=False,
        )
        for ax, fn in zip(axes.flat, _COST_FN_NAMES, strict=False):
            pts = [
                (cand["pred"][fn], cand["ms"], cand["name"])
                for cand in c["candidates"]
                if cand.get("ms") is not None
                and math.isfinite(cand["pred"].get(fn, float("nan")))
                and cand["pred"][fn] > 0
            ]
            if pts:
                ax.scatter(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    s=18,
                )
                for px, py, nm in pts:
                    ax.annotate(nm, (px, py), fontsize=5, alpha=0.75)
            mt = c["metrics"].get(fn, {})
            rho = mt.get("rho")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{fn}  ρ={rho:.2f}"
                if math.isfinite(rho or float("nan"))
                else f"{fn}  ρ=n/a",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(_COST_FN_NAMES) :]:
            ax.axis("off")
        fig.suptitle(
            f"{tag}: predicted cost vs measured ms", fontsize=11
        )
        fig.tight_layout()
        p = plots_dir / f"cost_fidelity_{ts}_scatter_{tag}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    # ρ summary: one group of bars per cost fn, one bar per cell.
    fig, ax = plt.subplots(
        figsize=(max(7.0, 1.6 * len(_COST_FN_NAMES)), 4.5)
    )
    n_cells = len(ok)
    width = 0.8 / max(n_cells, 1)
    xs = list(range(len(_COST_FN_NAMES)))
    for ci, c in enumerate(ok):
        tag = f"{c['model']}_{c.get('size_name', '?')}{c['size']}"
        ys = [
            c["metrics"].get(fn, {}).get("rho", float("nan"))
            for fn in _COST_FN_NAMES
        ]
        ax.bar(
            [x + (ci - (n_cells - 1) / 2) * width for x in xs],
            [0.0 if math.isnan(y) else y for y in ys],
            width * 0.9,
            label=tag,
        )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(_COST_FN_NAMES), fontsize=8, rotation=20)
    ax.set_ylabel("Spearman ρ (predicted cost vs measured ms)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("cost_fidelity — rank correlation per cost fn")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = plots_dir / f"cost_fidelity_{ts}_rho.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)
    return written


def _parse_ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _parse_strs(s: str) -> list[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def run_bench(args: argparse.Namespace) -> Report:
    """The full sweep → ``benchkit.Report`` (the run_all.py convention)."""
    dev = torch.device(getattr(args, "device", "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")

    warmup = getattr(args, "warmup", None) or 5
    min_run_time = getattr(args, "min_run_time", None) or 0.2
    quick = bool(getattr(args, "quick", False))
    top_k = getattr(args, "top_k", None) or (4 if quick else 8)
    args.top_k = top_k
    args.compile_timeout = (
        getattr(args, "compile_timeout", None) or 60.0
    )
    args.opt_max_t = getattr(args, "opt_max_t", None) or 512
    args.d = getattr(args, "d", None) or 64
    args.chain_d = getattr(args, "chain_d", None) or 128
    args.chain_rows = getattr(args, "chain_rows", None) or 2048
    args.value_depth = getattr(args, "value_depth", None) or 2
    args.heads = getattr(args, "heads", None) or 4
    args.batch = getattr(args, "batch", None) or 2
    args.compile_alts = bool(getattr(args, "compile_alts", False))
    args.slow_cap = getattr(args, "slow_cap", None) or (
        10.0 if quick else 30.0
    )
    args.ruleset = getattr(args, "ruleset", None) or "all"

    models = (
        _parse_strs(args.models)
        if getattr(args, "models", None)
        else (["chain", "retnet"] if quick else list(_ALL_MODELS))
    )
    sizes_arg = getattr(args, "sizes", None)
    cells_spec: list[tuple[str, int]] = []
    for m in models:
        if m not in _ALL_MODELS:
            raise SystemExit(
                f"unknown model {m!r} — pick from {_ALL_MODELS}"
            )
        sizes = (
            _parse_ints(sizes_arg)
            if sizes_arg
            else _DEFAULT_SIZES[m][
                : 1 if quick else len(_DEFAULT_SIZES[m])
            ]
        )
        for s in sizes:
            cells_spec.append((m, s))

    print("=" * 78)
    print("  cost_fidelity: predicted cost vs measured latency")
    print(
        f"  device={dev}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if dev.type == "cuda"
            else ""
        )
    )
    print(
        f"  models={models} cells={cells_spec} top_k={top_k} "
        f"compile_alts={args.compile_alts}"
    )
    print("=" * 78, flush=True)

    # Calibrate the target once — the roofline/depth _cal fns use it.
    t0 = time.time()
    profile = calibrate(device=dev, quick=quick)
    print(
        f"  calibrated {profile.name}: {profile.tflops:.3f} TFLOPS, "
        f"{profile.gbps:.1f} GB/s, {profile.launch_us:.2f} µs launch "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    supported = TorchSink().supported_ops

    runner = Runner(
        device=dev, warmup=warmup, min_run_time=min_run_time
    )
    results: list[dict] = []
    report_cells: list = []
    for m, s in cells_spec:
        c, ran = run_cell(
            m,
            s,
            args,
            dev,
            runner=runner,
            profile=profile,
            supported=supported,
        )
        results.append(c)
        if ran is not None:
            report_cells.append(ran)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    report = Report(
        suite="cost_fidelity", cells=report_cells, env=collect_env(dev)
    )

    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", "bench/results"))
        plots_dir = Path(
            getattr(args, "plots", None) or out_dir / "plots"
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"cost_fidelity_{ts}.json"
        md_path = out_dir / f"cost_fidelity_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="eager")
        with md_path.open("a") as fh:
            fh.write(_md_supplement(results))
        sidecar = out_dir / f"cost_fidelity_{ts}_detail.json"

        def _default(o):
            if isinstance(o, float):
                return o if math.isfinite(o) else str(o)
            return str(o)

        sidecar.write_text(
            json.dumps(
                {
                    "suite": "cost_fidelity",
                    "env": report.env,
                    "profile": profile.to_json(),
                    "cost_fns": list(_COST_FN_NAMES),
                    "cells": results,
                },
                indent=2,
                default=_default,
            )
            + "\n"
        )
        written: list[Path] = []
        try:
            written = _emit_plots(results, plots_dir, ts)
        except Exception as e:
            print(
                f"  note: fidelity plots skipped "
                f"({type(e).__name__}: {e})"
            )
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
        print(f"            → {sidecar}")
        for p in written:
            print(f"            → {p}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "cost-model fidelity: does predicted cost order match "
            "measured latency across the equivalence frontier?"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated families: chain,retnet,gla,delta,attn,"
        "swiglu (default all; quick: chain,retnet)",
    )
    ap.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="comma-separated size axis applied to every selected "
        "model — chain depth k for 'chain', horizon T for "
        "retnet/gla/delta/attn, token count R for swiglu",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="frontier size for discover_alternatives (default 8; "
        "quick: 4)",
    )
    ap.add_argument(
        "--ruleset",
        type=str,
        default=None,
        choices=["all", "categorical", "simpl"],
        help="rewrite-ruleset for the frontier search (default 'all' "
        "— the same space optimize_model explores)",
    )
    ap.add_argument(
        "--d",
        type=int,
        default=None,
        help="model dim for scan/attn/swiglu families (default 64)",
    )
    ap.add_argument(
        "--chain-d",
        type=int,
        default=None,
        help="weight dim for the chain cell (default 128)",
    )
    ap.add_argument(
        "--chain-rows",
        type=int,
        default=None,
        help="activation rows for the chain cell (default 2048)",
    )
    ap.add_argument(
        "--value-depth",
        type=int,
        default=None,
        help="value-chain depth k for scan families (default 2)",
    )
    ap.add_argument(
        "--heads", type=int, default=None, help="attn heads (default 4)"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help="batch for the attn cell (default 2)",
    )
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=None,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=None,
        help="per-module torch.compile budget, seconds (default 60)",
    )
    ap.add_argument(
        "--opt-max-t",
        type=int,
        default=None,
        help="skip optimize_model on scan cells above this size "
        "(default 512; non-scan cells always run it)",
    )
    ap.add_argument(
        "--slow-cap",
        type=float,
        default=None,
        help="per-candidate wall-clock cap in seconds — slower verified "
        "candidates are recorded as a one-shot latency instead of an "
        "autoranged median (default 30; quick: 10)",
    )
    ap.add_argument(
        "--compile-alts",
        action="store_true",
        help="also torch.compile every verified candidate lowering "
        "(off by default — dynamo compile dominates wall-clock and "
        "routinely fails to trace the generic evaluators; the eager "
        "and optimize_model anchors are always compiled)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="2 cells (chain k=4, retnet T=128), top_k=4",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="bench/results",
        help="artifact dir (default bench/results)",
    )
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="skip JSON/Markdown/plot emission for quick runs",
    )
    run_bench(ap.parse_args())


if __name__ == "__main__":
    main()
