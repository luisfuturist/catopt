"""Killer demo — measured autotuning over catopt lowering paths.

The pitch this bench backs: ``optimize_model`` picks ONE lowering up
front (serial ``IRModule``, level-batched carrier executor, or
``torch.compile``-wrapped) — but which is fastest is device-dependent
and only measurable, not derivable.  ``optimize_model_autotuned``
runs the search once, re-lowers the same extracted term through each
candidate executor, verifies every candidate against the original
model, times the survivors, and returns the measured winner.

Per model cell, timed through the shared ``benchkit.Runner``:

* ``eager``       — the original module;
* ``inductor``    — ``torch.compile`` of the original module
  (SIGALRM-guarded; a timeout/failure is recorded, not hidden);
* ``catopt_best`` — the autotuned winner, whichever lowering it is.

The autotune harness's own per-candidate medians land in the cell's
``aux`` (``autotune = generic:0.31ms · batched:0.12ms · …``), plus
``pick`` (the winning lowering), ``verified`` (the winner passed the
equivalence gate), and ``search_s`` (the one paid e-graph search).

Usage:
    python bench/killer_demo.py --device cpu --quick
    python bench/killer_demo.py --device cuda

    # CUDA dev venv:
    PYTHONPATH="packages/catopt-core/src:packages/catopt-torch/src:\
packages/catopt-carriers/src:packages/catopt-orchestrator/src:." \
        /tmp/catopt-cuda-venv/bin/python bench/killer_demo.py \
        --device cuda
"""
# ruff: noqa: E402, RUF001 -- sys.path setup must precede the benchkit/catopt
# imports (bench_omd2 / real_linear_attn convention).

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.setrecursionlimit(400_000)

import torch
import torch.nn as nn
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import Autotuned
from catopt_torch.autotune import TORCH_BUILDERS
from catopt_torch.backend import TorchBackend
from catopt_torch.models import (
    LinearRecurrence,
    MatrixChain,
    ResidualMLP,
)

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)
from bench.suites.algebra.real_linear_attn import (
    LinearAttnStack,
    try_compile,
)

# run_all.py picks these up for its --quick lane.
QUICK = {
    "retnet_t": "32",
    "calls": "20",
    "min_run_time": "0.1",
}


# ---------------------------------------------------------------------------
#  Model cells
# ---------------------------------------------------------------------------


def _models(retnet_t: int) -> list[tuple[str, nn.Module, torch.Tensor]]:
    """The demo cells: a scan (the carrier win), an MLP block, a
    weight-chain fold, and a real gated linear-attention stack."""
    cells: list[tuple[str, nn.Module, torch.Tensor]] = []
    torch.manual_seed(0)
    cells.append(
        (
            "linear_recurrence",
            LinearRecurrence(dim=32, steps=8).eval(),
            torch.randn(8, 32),
        )
    )
    torch.manual_seed(0)
    cells.append(
        (
            "residual_mlp",
            ResidualMLP(dim=64).eval(),
            torch.randn(16, 64),
        )
    )
    torch.manual_seed(0)
    cells.append(
        (
            "matrix_chain",
            MatrixChain(128, 64, 32, 8).eval(),
            torch.randn(32, 128),
        )
    )
    torch.manual_seed(0)
    cells.append(
        (
            "retnet_stack",
            LinearAttnStack(
                d=32, mode="retnet", k=2, n_blocks=2
            ).eval(),
            torch.randn(retnet_t, 32),
        )
    )
    return cells


# ---------------------------------------------------------------------------
#  One cell
# ---------------------------------------------------------------------------


def _timed_stmt(mod, x):
    """Zero-arg variant callable: one inference under ``no_grad``."""

    def stmt() -> None:
        with torch.no_grad():
            mod(x)

    return stmt


def run_cell(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    candidates: tuple[str, ...],
    n_calls: int,
    at_warmup: int,
    compile_timeout: float,
    verbose: bool,
) -> tuple[dict, Case]:
    """Autotune one model and pack the timing cell."""
    model = model.to(device).eval()
    x = x.to(device)

    opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(model, x, strategy=Autotuned(candidates, n_calls=n_calls, warmup=at_warmup, verbose=verbose, builders=TORCH_BUILDERS))

    at = stats["autotune"]
    cand_recs = at["candidates"]

    # Winner's verification status (fallback = the pipeline module).
    if at["winner"] is not None:
        verified = bool(cand_recs[at["winner"]].get("verified"))
    else:
        verified = bool(
            cand_recs.get("_pipeline_fallback", {}).get("verified")
        )

    # Compact per-candidate summary for the report's aux section.
    parts = []
    for cn, rec in cand_recs.items():
        if rec.get("status") == "timed":
            parts.append(f"{cn}:{rec['median_s'] * 1e3:.3f}ms")
        else:
            parts.append(f"{cn}:{rec.get('status', '?')}")
    autotune_summary = " ".join(parts)

    # -- inductor baseline on a COPY of the model ---------------------
    # torch.compile(m) rewrites m.forward (dynamo dispatch wrapper) —
    # compiling the timed model would contaminate the eager variant.
    cm = None
    ind_status = "skipped"
    if compile_timeout and compile_timeout > 0:
        cm, ind_status = try_compile(
            copy.deepcopy(model), x, compile_timeout
        )
    print(f"  inductor: {ind_status}", flush=True)

    variants: list[Variant] = [
        Variant(name="eager", stmt=_timed_stmt(model, x)),
    ]
    if cm is not None:
        variants.append(
            Variant(
                name="inductor",
                stmt=_timed_stmt(cm, x),
                note=f"torch.compile eager — {ind_status}",
            )
        )
    variants.append(
        Variant(
            name="catopt_best",
            stmt=_timed_stmt(opt_mod, x),
            note=f"pick={at['winner'] or 'pipeline fallback'}",
        )
    )

    aux = {
        "pick": at["winner"] or "pipeline_fallback",
        "verified": verified,
        "lowering": stats.get("lowering", "?"),
        "search_s": round(at["search_s"], 2),
        "autotune": autotune_summary,
        "inductor_status": ind_status,
    }
    case = Case(
        name=name,
        params={"device": str(device)},
        variants=variants,
        aux=aux,
    )
    return {"name": name, "aux": aux}, case


# ---------------------------------------------------------------------------
#  Harness entry point (run_all.py convention)
# ---------------------------------------------------------------------------


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    retnet_t = int(getattr(args, "retnet_t", None) or 32)
    n_calls = int(getattr(args, "calls", None) or 30)
    at_warmup = int(getattr(args, "warmup", None) or 5)
    warmup = max(at_warmup, 3)
    min_run_time = float(getattr(args, "min_run_time", None) or 0.2)
    # explicit 0 disables the inductor baseline — the `or`-fallback
    # used elsewhere would silently restore the default.
    _ct = getattr(args, "compile_timeout", None)
    compile_timeout = 60.0 if _ct is None else float(_ct)
    verbose = bool(getattr(args, "verbose", False))

    candidates = getattr(args, "candidates", None)
    cand_tuple = (
        tuple(s.strip() for s in candidates.split(","))
        if candidates
        else ("eager", "generic", "batched", "compiled")
    )
    if dev.type == "cuda" and "cuda_graph" not in cand_tuple:
        cand_tuple = (*cand_tuple, "cuda_graph")

    only = getattr(args, "models", None)
    cells_in = _models(retnet_t)
    if only:
        keep = {s.strip() for s in only.split(",")}
        cells_in = [c for c in cells_in if c[0] in keep]

    print(
        f"killer_demo — device={dev} candidates={cand_tuple} "
        f"retnet_T={retnet_t}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=warmup, min_run_time=min_run_time
    )

    results: list[dict] = []
    report_cells: list[Case] = []
    for name, model, x in cells_in:
        print(f"\n=== {name} ===", flush=True)
        res, case = run_cell(
            name,
            model,
            x,
            device=dev,
            candidates=cand_tuple,
            n_calls=n_calls,
            at_warmup=at_warmup,
            compile_timeout=compile_timeout,
            verbose=verbose,
        )
        results.append(res)
        report_cells.append(case)

    cells = runner.run(report_cells)

    # -- console table ------------------------------------------------
    hdr = (
        f"{'model':<18} | {'eager':>8} | {'compile':>8} | "
        f"{'catopt':>8} | {'pick':<12} | {'verif':>5} | {'xE':>6}"
    )
    print("\n" + "=" * len(hdr))
    print(
        "  model | eager ms | torch.compile ms | catopt best ms | "
        "catopt pick | verified? | speedup-vs-eager"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for res, cell in zip(results, cells, strict=True):
        ms = cell.medians
        eager = ms.get("eager")
        best = ms.get("catopt_best")

        def g(n, _ms=ms):
            return f"{_ms[n] * 1e3:>8.3f}" if n in _ms else f"{'—':>8}"

        speedup = f"{eager / best:>6.2f}" if eager and best else "—"
        print(
            f"{res['name']:<18} | {g('eager')} | {g('inductor')} | "
            f"{g('catopt_best')} | {res['aux']['pick']:<12} | "
            f"{'yes' if res['aux']['verified'] else 'NO':>5} | "
            f"{speedup}"
        )
    print("-" * len(hdr))
    print(
        "  xE = catopt_best speedup vs eager; pick = the lowering "
        "autotune measured fastest"
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    scored = [
        (c.case.name, c.medians["eager"] / c.medians["catopt_best"])
        for c in cells
        if c.medians.get("eager") and c.medians.get("catopt_best")
    ]
    best = max(scored, key=lambda t: t[1], default=None)
    n_verified = sum(1 for r in results if r["aux"]["verified"])
    findings = [
        Finding(
            claim=(
                "the Autotuned lowering beats eager on at least one "
                "model"
            ),
            verdict=(
                Verdict.WIN if best and best[1] > 1 else Verdict.NEGATIVE
            ),
            headline=(
                f"best {best[1]:.2f}× vs eager on {best[0]}"
                if best
                else "no model beats eager"
            ),
            metric="catopt_best / eager",
            value=best[1] if best else None,
            evidence={
                "picks": {r["name"]: r["aux"]["pick"] for r in results}
            },
        ),
        Finding(
            claim="every model's optimized lowering verifies",
            verdict=(
                Verdict.WIN
                if n_verified == len(results)
                else Verdict.REGRESSION
            ),
            headline=f"{n_verified}/{len(results)} models verified",
            metric="verified models",
            value=float(n_verified),
        ),
    ]
    report = Report(
        suite="killer_demo",
        title="Compositional pairing demo",
        summary=(
            "Per-model lowering autotune (eager / generic / batched / "
            "compiled): the fastest verified lowering is reported, "
            "including when eager wins."
        ),
        findings=findings,
        cells=cells,
        env=collect_env(dev),
    )

    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"killer_demo_{ts}.json"
        md_path = out_dir / f"killer_demo_{ts}.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="eager")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
        try:
            written = report.to_plots(
                out_dir / "plots",
                speedup_vs="eager",
                stem=f"killer_demo_{ts}",
            )
            for p in written:
                print(f"            → {p}")
        except Exception as e:
            print(f"  note: plots skipped ({type(e).__name__}: {e})")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "killer demo: measured autotuning over catopt lowering "
            "paths vs eager / Inductor"
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated cell subset: linear_recurrence,"
        "residual_mlp,matrix_chain,retnet_stack (default all)",
    )
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="comma-separated autotune candidates (default "
        "eager,generic,batched,compiled; cuda adds cuda_graph)",
    )
    ap.add_argument(
        "--retnet-t",
        type=int,
        default=32,
        help="horizon for the retnet_stack cell (default 32)",
    )
    ap.add_argument(
        "--calls",
        type=int,
        default=30,
        help="timed calls per autotune candidate (default 30)",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=60.0,
        help="torch.compile budget for the inductor baseline "
        "(0 disables it)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="print optimizer detail"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small smoke run (run_all convention)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="bench/results",
        help="artifact dir for benchkit JSON/Markdown/plots "
        "(default bench/results)",
    )
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="skip JSON/Markdown/plot emission",
    )
    args = ap.parse_args()
    if args.quick:
        # QUICK overrides: same dict run_all applies.
        for k, v in QUICK.items():
            setattr(args, k, type(getattr(args, k))(v))
    run_bench(args)


if __name__ == "__main__":
    main()
