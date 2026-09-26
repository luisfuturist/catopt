"""Reassociation-at-scale — the flagship head-to-head.

A linear-attention-style weight chain

    y = x @ W_q @ W_kv1 @ ... @ W_kv{k-1}

with ``x`` a tall activation (R = B·T token rows, flattened, by d) and
k weight matrices ``(d, d)`` is written left-associative — the only
thing PyTorch/Inductor can do with it is k sequential ``(R,d)@(d,d)``
GEMMs, O(k·R·d²) FLOPs.  Matmul is associative, so the equivalent
weights-first form ``x @ (W_q @ W_kv1 @ ...)`` costs one runtime GEMM
plus a one-time compile-time fold: O(R·d²) + O(k·d³).

TorchInductor has no matmul-reassociation pass, so this transform is
unreachable for it — the script *proves* that by capturing the
post-grad FX graph and checking all k ``mm`` nodes still sit on the
data spine (no weight×weight product).  CatOpt reaches it via
``assoc_matmul`` / ``assoc_matmul_rev`` in the e-graph and folds the
weight product at lowering time (``IRModule._fold_weight_chains``); the
extracted term is asserted to nest weights-first.

Per cell (k, d, R):

* eager fp32, Inductor-compiled original, a hand-built folded
  reference (the reachable optimum), the catopt-optimized module, and
  the catopt module compiled with the same Inductor backend — all
  timed with ``torch.utils.benchmark`` (median via blocked_autorange).
* correctness gate: ``torch.allclose`` fp32 vs the original eager
  output for every variant; a failed gate marks the cell FAIL.

Usage:
    python bench/reassoc_scale.py --device cpu
    python bench/reassoc_scale.py --device cuda --dims 256,512,1024
    python bench/reassoc_scale.py --device cpu --depths 4,8 --rows 16384
"""
# ruff: noqa: RUF001 RUF002 RUF003 -- ×, ·, −, ² in strings/docstrings
# are deliberate math notation; same convention as catopt_core.laws.

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from catopt.cost import launch_aware_cost
from catopt.egraph import EGraph
from catopt.ir import Op, Param, Var, op_repr
from catopt.optimize import _EXPANSIVE_RULES, optimize_model
from catopt.rules import (
    all_rules,
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from catopt.torch_bridge import export_to_ir
from catopt.typing import has_var_leaf
from catopt_carriers.trace_lift import lift_scan_to_trace
from catopt_carriers.xcarrier import (
    gather_apply_stack,
    gather_applyd_stack,
    omd_tree_lift,
)
from torch.utils.benchmark import Timer

#: Rules subsumed inside ``optimize_model`` by the non-local pairing
#: pass — mirrored so the inspection e-graph below is the same search
#: space ``optimize_model`` explores.  None can fire on a pure chain.
_SUBSUMED = {
    "swiglu_fuse",
    "parallel_mul_fuse",
    "qkv_fuse",
    "qkv_fuse_asym",
}

_MM_OPS = {"mm", "bmm", "addmm", "matmul"}


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------


class LinearAttnChain(nn.Module):
    """Linear-attention-style chain: ``y = x @ W_q @ W_kv1 @ ...``.

    ``x`` is ``(R, d)`` — R = B·T token rows flattened into one axis
    (per-position maps commute with batch/time, so the flat row view is
    equivalent to the usual ``(B, T, d)``).  Every weight is ``(d, d)``:
    a query projection followed by k−1 kv/state-transition maps.

    Written left-associative, as PyTorch evaluates it: k sequential
    ``(R,d)@(d,d)`` GEMMs = O(k·R·d²) FLOPs.  The weights-first
    reassociation folds the k matrices into one ``(d,d)`` product at
    compile time → one runtime GEMM.
    """

    def __init__(self, d: int, k: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # entries ~ N(0, 1/d): variance-preserving, so the depth-k
        # product stays O(1) and fp32 reassociation noise stays small.
        self.wq = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
        self.wkv = nn.ParameterList(
            nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
            for _ in range(k - 1)
        )
        self.k = k
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.wq
        for w in self.wkv:
            h = h @ w
        return h


class FoldedChain(nn.Module):
    """Manual weights-first reference — the transform done by hand.

    Precomputes ``W_eff = W_q @ W_kv1 @ ...`` once at construction and
    runs a single GEMM at runtime: the reachable optimum the compiler
    should find on its own.
    """

    def __init__(self, chain: LinearAttnChain) -> None:
        super().__init__()
        with torch.no_grad():
            w = chain.wq.detach().clone()
            for wi in chain.wkv:
                w = w @ wi
        self.register_buffer("W_eff", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_eff


# ---------------------------------------------------------------------------
#  CatOpt pipeline (proof artifact: the extracted term, pre-fold)
# ---------------------------------------------------------------------------


def saturate_and_extract(
    model: nn.Module, x: torch.Tensor, max_iterations: int = 64
) -> tuple[Op, dict]:
    """``optimize_model``'s search half, instrumented.

    Export → EGraph → bounded saturation with the same rule set and the
    same per-rule symmetry budgets as ``optimize_model`` → the same
    non-local passes (pairing, carrier lifts, param sharing — all no-ops
    on a pure matmul chain, included for fidelity) → ``extract_best``
    under the default ``launch_aware_cost``.  Returns the extracted term
    so the caller can assert the weights-first structure before
    ``IRModule`` folds it.
    """
    ir, source_tensors = export_to_ir(model, x)
    eg = EGraph()
    root_eid = eg.add_term(ir.root)
    rules = [r for r in all_rules() if r.name not in _SUBSUMED]
    budgets = {n: 2048 for n in _EXPANSIVE_RULES}
    cap = 200_000
    stats = eg.run(
        rules,
        root_eid,
        max_iterations=max_iterations,
        max_nodes=cap,
        rule_budgets=budgets,
    )
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    if groups:
        eg.rebuild()
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=cap,
            rule_budgets=budgets,
        )
    lifts = (
        lift_scan_to_trace(eg)
        + gather_applyd_stack(eg)
        + gather_apply_stack(eg)
        + omd_tree_lift(eg)
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
    )
    if lifts:
        eg.rebuild()
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=cap,
            rule_budgets=budgets,
        )
    term = eg.extract_best(root_eid, launch_aware_cost)
    stats["rule_fires"] = dict(eg.rule_fires)
    return term, stats


def check_weights_first(term, k: int) -> tuple[bool, str]:
    """Assert ``term`` is ``matmul(x, <param-only product of k weights>)``.

    The extracted form must nest the weights together — a single data
    matmul against a parameter-only subtree of k−1 matmuls over k
    ``Param`` leaves.  That subtree materialises as one ``fused_*``
    parameter when ``IRModule`` lowers it.
    """
    if not (
        isinstance(term, Op)
        and term.op == "matmul"
        and len(term.args) == 2
    ):
        return (
            False,
            f"root is {getattr(term, 'op', type(term).__name__)}",
        )
    data, w = term.args
    if not isinstance(data, Var):
        return (
            False,
            f"left operand is {type(data).__name__}, not input Var",
        )
    if has_var_leaf(w, {}):
        return False, "right operand still reads the data input"
    n_par, n_mm = 0, 0
    seen: set = set()

    def rec(t):
        nonlocal n_par, n_mm
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Param):
            n_par += 1
        elif isinstance(t, Op):
            n_mm += t.op == "matmul"
            for a in t.args:
                rec(a)

    rec(w)
    if k > 1 and (n_par != k or n_mm != k - 1):
        return (
            False,
            f"weight subtree: {n_par} params / {n_mm} matmuls "
            f"(expected {k} / {k - 1})",
        )
    return True, f"matmul(x, W_q@W_kv1@…@W_kv{k - 1}) — weights-first"


# ---------------------------------------------------------------------------
#  Inductor post-grad graph: capture + left-association check
# ---------------------------------------------------------------------------


def _postgrad_logger() -> logging.Logger:
    """The logger ``torch._inductor.compile_fx`` emits the post-grad FX
    graph artifact on (``<module>.__post_grad_graphs``)."""
    try:
        import torch._logging._internal as li

        for q in li.log_registry.get_artifact_log_qnames():
            if q.endswith("__post_grad_graphs"):
                return logging.getLogger(q)
    except Exception:
        pass
    return logging.getLogger(
        "torch._inductor.compile_fx.__post_grad_graphs"
    )


def compile_with_postgrad(
    model: nn.Module, x: torch.Tensor
) -> tuple[nn.Module, str]:
    """``torch.compile`` the module, capturing the post-grad FX graph.

    The ``post_grad_graphs`` artifact emits on
    ``torch._inductor.compile_fx.__post_grad_graphs`` and propagates to
    ``torch._inductor``'s ``_StderrHandler`` (bound to the real stderr
    at ``set_logs`` time — a ``sys.stderr`` swap does not redirect it).
    So: attach our handler directly to the artifact logger, and detach
    the ``torch._inductor`` handlers for the capture window — its
    ``propagate=False`` caps the chain there, so nothing else prints.
    """
    lg = _postgrad_logger()
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(h)
    torch._logging.set_logs(post_grad_graphs=True)
    # ``set_logs`` → ``_init_logs`` (re)creates the ``_StderrHandler``s
    # and flips artifact propagate back on — so both silencing moves
    # must come *after* it.
    lg.propagate = False
    anc = logging.getLogger("torch._inductor")
    anc_handlers = anc.handlers[:]
    for h0 in anc_handlers:
        anc.removeHandler(h0)
    try:
        cm = torch.compile(model)
        with torch.no_grad():
            cm(x)
    finally:
        torch._logging.set_logs(post_grad_graphs=False)
        lg.propagate = True
        lg.removeHandler(h)
        for h0 in anc_handlers:
            if h0 not in anc.handlers:
                anc.addHandler(h0)
    return cm, buf.getvalue()


_STMT_RE = re.compile(
    r"(\w+):\s*\"[^\"]*\"\s*=\s*torch\.ops\.aten\.(\w+)(?:\.\w+)*\(([^)]*)\)"
)
_PH_RE = re.compile(r"(\w+):\s*\"[^\"]*?\[([\d,\s]+)\]")


def _postgrad_blocks(text: str) -> list[str]:
    """Split the captured log into per-graph chunks (one per
    ``AFTER POST GRAD`` marker); a graph break would emit several."""
    blocks = re.split(r"={3,}\s*AFTER POST GRAD\s*={3,}", text)
    return [b for b in blocks[1:] if "torch.ops.aten" in b]


def _analyze_block(block: str, x_shape: tuple[int, int]) -> dict | None:
    """Per-graph-block analysis: producer map + weight×weight count.

    Returns ``None`` for blocks without aten statements."""
    producers: dict[str, tuple[str, list[str]]] = {}
    placeholders: dict[str, tuple[int, ...]] = {}
    mm_lines = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("def forward"):
            for nm, dims in _PH_RE.findall(line):
                placeholders[nm] = tuple(
                    int(s) for s in dims.split(",") if s.strip()
                )
        m = _STMT_RE.match(line)
        if not m:
            continue
        name, op, argstr = m.groups()
        producers[name] = (op, re.findall(r"\b\w+\b", argstr))
        if op in _MM_OPS:
            mm_lines.append(line)
    if not producers:
        return None
    # The data operand is the placeholder shaped (rows, d); weights are
    # (d, d).  rows > d in every sweep cell.
    x_names = {nm for nm, sh in placeholders.items() if sh == x_shape}
    names = set(producers) | set(placeholders)

    def reaches_x(name: str, memo: dict, stack: frozenset) -> bool:
        if name in x_names:
            return True
        if name not in producers or name in stack:
            return False
        if name in memo:
            return memo[name]
        memo[name] = any(
            reaches_x(a, memo, stack | {name})
            for a in producers[name][1]
            if a in names
        )
        return memo[name]

    memo: dict = {}
    mm_nodes = [n for n, (op, _) in producers.items() if op in _MM_OPS]
    return {
        "n_mm": len(mm_nodes),
        "n_weight_only_mm": sum(
            1 for n in mm_nodes if not reaches_x(n, memo, frozenset())
        ),
        "mm_lines": mm_lines,
        "x_names": sorted(x_names),
    }


def analyze_postgrad(text: str, k: int, rows: int, d: int) -> dict:
    """Prove the Inductor graph is still the left-associative chain.

    For each block: collect every aten statement into a producer map,
    identify the data placeholder ``x`` by its ``(rows, d)`` annotation,
    then ask two things of every matmul node:

    * does its operand-DAG reach ``x``?  A weight×weight product mm —
      the signature of ANY weights-first form — does not.
    * is the total mm count still k (no fused/short-circuited matmul)?

    Left-assoc chain ⟹ mm count == k and every mm reaches ``x``.
    """
    result = {
        "n_mm": 0,
        "n_weight_only_mm": 0,
        "mm_lines": [],
        "left_assoc": False,
        "graph_found": False,
    }
    best = None
    for block in _postgrad_blocks(text):
        cand = _analyze_block(block, (rows, d))
        if cand and (best is None or cand["n_mm"] > best["n_mm"]):
            best = cand
    if best is None:
        return result
    result.update(best)
    result["graph_found"] = True
    # Left-assoc ⟺ exactly k mms AND none is a weight×weight product.
    # (k mms, each transitively consuming x, with only (R,d)-vs-(d,d)
    # operand shapes available, can only be arranged as the data spine.)
    result["left_assoc"] = (
        best["n_mm"] == k and best["n_weight_only_mm"] == 0
    )
    result["k_expected"] = k
    return result


# ---------------------------------------------------------------------------
#  Timing
# ---------------------------------------------------------------------------


def time_ms(
    fn,
    x: torch.Tensor,
    dev: torch.device,
    warmup: int,
    min_run_time: float,
) -> float:
    """Median ms/call via ``torch.utils.benchmark.Timer`` —
    explicit warmup (≥3) then blocked_autorange medians."""
    sync = dev.type == "cuda"
    stmt = "fn(x); torch.cuda.synchronize()" if sync else "fn(x)"
    timer = Timer(stmt=stmt, globals={"fn": fn, "x": x, "torch": torch})
    with torch.no_grad():
        for _ in range(max(warmup, 3)):
            fn(x)
        if sync:
            torch.cuda.synchronize()
        meas = timer.blocked_autorange(min_run_time=min_run_time)
    return meas.median * 1e3


# ---------------------------------------------------------------------------
#  Sweep cell
# ---------------------------------------------------------------------------


def run_cell(
    k: int,
    d: int,
    rows: int,
    dev: torch.device,
    *,
    warmup: int,
    min_run_time: float,
    verbose: bool,
) -> dict:
    torch.manual_seed(0)
    m = LinearAttnChain(d, k).to(dev).eval()
    x = torch.randn(rows, d, device=dev)

    cell = {
        "k": k,
        "d": d,
        "rows": rows,
        "flops_left": 2 * rows * d * d * k,
        "flops_right_runtime": 2 * rows * d * d,
        "flops_right_fold": 2 * (k - 1) * d * d * d,
    }
    print(f"\n=== k={k}  d={d}  B·T={rows} ===")
    print(
        f"  modelled FLOPs: left-assoc {cell['flops_left'] / 1e9:.2f} G | "
        f"weights-first {cell['flops_right_runtime'] / 1e9:.2f} G runtime"
        f" (+{cell['flops_right_fold'] / 1e9:.2f} G one-time fold)"
        f" → {cell['flops_left'] / cell['flops_right_runtime']:.1f}× runtime"
    )

    with torch.no_grad():
        orig = m(x)

    # -- manual right-assoc reference (reachable optimum) ---------------
    ref = FoldedChain(m).to(dev).eval()

    # -- catopt ----------------------------------------------------------
    t0 = time.time()
    try:
        opt, stats = optimize_model(
            m,
            x,
            verbose=False,
            max_iterations=64,
            max_enodes=200_000,
        )
    except Exception as e:  # honest failure path — do not fake the win
        print(f"  catopt FAILED: {type(e).__name__}: {e}")
        cell["error"] = f"{type(e).__name__}: {e}"
        return cell
    opt = opt.to(dev).eval()
    cell["pipeline_s"] = time.time() - t0
    fires = {
        n: c
        for n, c in stats.get("rule_fires", {}).items()
        if "assoc" in n
    }
    cell["assoc_rule_fires"] = fires

    # The extracted term itself — the proof the e-graph found the
    # weights-first form (same rules, budgets and cost model as
    # optimize_model; the pairing/lift passes are no-ops on this IR).
    term, _estats = saturate_and_extract(m, x)
    ok_term, term_msg = check_weights_first(term, k)
    cell["term_weights_first"] = ok_term
    if verbose or not ok_term:
        print(f"  extracted term: {op_repr(term)[:400]}")
    print(
        f"  catopt: extracted {'weights-FIRST ✓' if ok_term else 'UNEXPECTED — ' + term_msg}"
        f" | lowered root: {op_repr(opt._root)[:80]}"
        f" | assoc fires: {fires}"
    )

    # -- Inductor non-reachability proof ----------------------------------
    torch._dynamo.reset()
    cm, pg = compile_with_postgrad(m, x)
    pa = analyze_postgrad(pg, k, rows, d)
    cell["inductor_mm"] = pa["n_mm"]
    cell["inductor_left_assoc"] = pa["left_assoc"]
    if pa["graph_found"]:
        print(
            f"  inductor post-grad: {pa['n_mm']} mm nodes, "
            f"{pa['n_weight_only_mm']} weight×weight "
            f"→ {'LEFT-ASSOC chain (unreached)' if pa['left_assoc'] else 'NOT a pure left chain!'}"
        )
        for ln in pa["mm_lines"][:k]:
            print(f"      {ln[:110]}")
    else:
        print("  inductor post-grad: no graph captured (!)")

    copt, pg2 = compile_with_postgrad(opt, x)
    pa2 = analyze_postgrad(pg2, k, rows, d)
    cell["catopt_inductor_mm"] = pa2["n_mm"]
    print(
        f"  catopt+inductor post-grad: {pa2['n_mm']} mm nodes "
        f"(weight product folded into a param)"
    )

    # -- correctness gate --------------------------------------------------
    with torch.no_grad():
        opt_out = opt(x)
        ref_out = ref(x)
        ind_out = cm(x)
        ci_out = copt(x)
    cell["max_abs"] = (opt_out - orig).abs().max().item()
    # Scale-relative diff: max|opt-orig| / max|orig| — elementwise
    # rel err on near-zero outputs is noise, not signal.
    cell["max_rel"] = (
        cell["max_abs"] / orig.abs().max().clamp_min(1e-12).item()
    )
    ok = torch.allclose(opt_out, orig, rtol=1e-4, atol=1e-5)
    ok &= torch.allclose(ref_out, orig, rtol=1e-4, atol=1e-5)
    ok &= torch.allclose(ind_out, orig, rtol=1e-4, atol=1e-5)
    ok &= torch.allclose(ci_out, orig, rtol=1e-4, atol=1e-5)
    cell["verified"] = bool(ok)
    print(
        f"  verify: opt max_abs={cell['max_abs']:.2e} "
        f"rel_to_max={cell['max_rel']:.2e} | allclose fp32 "
        f"{'✓' if ok else '✗ FAIL'}"
    )
    if not ok:
        return cell

    # -- timing ---------------------------------------------------------
    cell["eager_ms"] = time_ms(m, x, dev, warmup, min_run_time)
    cell["inductor_ms"] = time_ms(cm, x, dev, warmup, min_run_time)
    cell["ref_ms"] = time_ms(ref, x, dev, warmup, min_run_time)
    cell["catopt_ms"] = time_ms(opt, x, dev, warmup, min_run_time)
    cell["catopt_inductor_ms"] = time_ms(
        copt, x, dev, warmup, min_run_time
    )
    cell["x_vs_eager"] = cell["eager_ms"] / cell["catopt_ms"]
    cell["x_vs_inductor"] = (
        cell["inductor_ms"] / cell["catopt_inductor_ms"]
    )
    return cell


def _parse_ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="matmul-chain reassociation: catopt vs Inductor"
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--depths",
        type=str,
        default=None,
        help="comma-separated chain depths k (default 2,4,8,16)",
    )
    ap.add_argument(
        "--dims",
        type=str,
        default=None,
        help="comma-separated dims d (default 512 on cpu, "
        "256,512,1024 on cuda)",
    )
    ap.add_argument(
        "--rows",
        type=str,
        default=None,
        help="comma-separated B·T row counts (default "
        "4096,16384,65536)",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.4,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="print extracted terms"
    )
    ap.add_argument(
        "--json", type=str, default=None, help="write results to PATH"
    )
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")

    # Post-grad capture needs a fresh compile each cell — a served FX
    # cache hit skips the pass pipeline and emits no graph.
    try:
        import torch._inductor.config as _icfg

        _icfg.fx_graph_cache = False
    except Exception:
        pass

    depths = _parse_ints(args.depths) if args.depths else [2, 4, 8, 16]
    if args.dims:
        dims = _parse_ints(args.dims)
    else:
        # CPU-safe default: one dim keeps the sweep quick; Inductor
        # compile time dominates, not the matmuls themselves.
        dims = [256, 512, 1024] if dev.type == "cuda" else [512]
    rows = _parse_ints(args.rows) if args.rows else [4096, 16384, 65536]
    cells = [(k, d, r) for d in dims for k in depths for r in rows]
    for _k, _d, _r in cells:
        if _r <= _d:
            print(
                f"warning: rows={_r} <= d={_d} — the crossover needs "
                f"B·T > k·d; results may still favour reassociation "
                f"(fewer launches) but the FLOP model won't."
            )

    print("=" * 78)
    print("  reassoc_scale: x @ W_q @ W_kv1 @ … @ W_kv{k-1} chain")
    print(
        f"  device={dev}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if dev.type == "cuda"
            else ""
        )
    )
    print(
        f"  depths={depths} dims={dims} rows={rows} "
        f"({len(cells)} cells)"
    )
    print("=" * 78)

    results = []
    for k, d, r in cells:
        results.append(
            run_cell(
                k,
                d,
                r,
                dev,
                warmup=args.warmup,
                min_run_time=args.min_run_time,
                verbose=args.verbose,
            )
        )

    # -- clean table ----------------------------------------------------
    hdr = (
        f"{'k':>3} {'d':>5} {'B·T':>6} | {'GF(L)':>7} {'GF(R)':>7} "
        f"{'GF(fold)':>8} | {'eager':>8} {'induct':>8} {'ref':>8} "
        f"{'catopt':>8} {'cat+ind':>8} | {'xE':>5} {'xI':>5} | gate"
    )
    print("\n" + "=" * 78)
    print(
        "  TIMING (median ms/call) — eager fp32 / inductor / manual "
        "right-assoc / catopt / catopt+inductor"
    )
    print("=" * 78)
    print(hdr)
    print("-" * len(hdr))
    best = None
    for c in results:
        if not c.get("verified"):
            print(
                f"{c['k']:>3} {c['d']:>5} {c['rows']:>6} | "
                f"{'':>7} {'':>7} {'':>8} | "
                f"{'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} | "
                f"{'':>5} {'':>5} | "
                f"{'FAIL' if 'verified' in c else 'ERROR'}"
            )
            continue
        print(
            f"{c['k']:>3} {c['d']:>5} {c['rows']:>6} | "
            f"{c['flops_left'] / 1e9:>7.2f} "
            f"{c['flops_right_runtime'] / 1e9:>7.2f} "
            f"{c['flops_right_fold'] / 1e9:>8.2f} | "
            f"{c['eager_ms']:>8.2f} {c['inductor_ms']:>8.2f} "
            f"{c['ref_ms']:>8.2f} {c['catopt_ms']:>8.2f} "
            f"{c['catopt_inductor_ms']:>8.2f} | "
            f"{c['x_vs_eager']:>5.2f} {c['x_vs_inductor']:>5.2f} | "
            f"{'OK' if c['inductor_left_assoc'] else 'proof?'}"
        )
        if (
            c["inductor_left_assoc"]
            and c["term_weights_first"]
            and (
                best is None
                or c["x_vs_inductor"] > best["x_vs_inductor"]
            )
        ):
            best = c

    print("-" * len(hdr))
    if best is not None:
        print(
            f"\n  catopt finds a transform Inductor cannot express: "
            f"{best['x_vs_inductor']:.2f}× vs Inductor on "
            f"(k,d,B·T)=({best['k']},{best['d']},{best['rows']}) "
            f"({best['x_vs_eager']:.2f}× vs eager; Inductor emitted "
            f"{best['inductor_mm']} left-assoc mms, catopt runs "
            f"{best['catopt_inductor_mm']})"
        )
    else:
        print(
            "\n  no verified cell where the proof held — see per-cell "
            "output (honest negative result)"
        )

    # Notes on honesty: CPU torch.compile overhead can dominate small
    # cells, and the one-time fold cost is compile-time only.
    if dev.type == "cpu":
        print(
            "  note: CPU cell — Inductor CPU compile+runtime overheads "
            "are real; 'cat+ind' vs 'catopt' shows what the backend "
            "adds to a single-GEMM module."
        )
    print(
        "  FLOPs: GF(L)=k·2·R·d² left-assoc; GF(R)=2·R·d² runtime "
        "weights-first; GF(fold)=2·(k−1)·d³ one-time compile-time "
        "fold (not in the timed loop)."
    )

    if args.json:
        payload = {
            "device": str(dev),
            "depths": depths,
            "dims": dims,
            "rows": rows,
            "cells": results,
            "best_vs_inductor": (
                {
                    "k": best["k"],
                    "d": best["d"],
                    "rows": best["rows"],
                    "speedup": best["x_vs_inductor"],
                }
                if best
                else None
            ),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"  results → {args.json}")


if __name__ == "__main__":
    main()
