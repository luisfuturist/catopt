"""Bench orchestrator — run harnessed suites, emit JSON + MD + plots.

Each suite is a module in ``bench/`` exposing the harness entry point

    def run_bench(args) -> benchkit.Report

where ``args`` is the same namespace the module's own ``main()``
would parse — at minimum ``device``, ``warmup``, ``min_run_time``,
``quick``, ``out``, ``plots`` — filled here with a *lax* namespace:
attributes the orchestrator doesn't set read as ``None``, so
suite-specific flags (``--depths``, ``--rows``, …) fall back to the
module's own defaults.  A module may additionally declare a
``QUICK`` dict of attribute overrides applied under ``--quick``
(e.g. ``QUICK = {"depths": "2,4", "rows": "4096"}``).

Modules without ``run_bench`` are legacy ad-hoc scripts — they are
skipped and marked ``ad-hoc, not harnessed`` in the index.

Outputs land under ``bench/results/`` (gitignored).  The orchestrator
always emits the canonical set and links it in the index; a suite's
``run_bench`` may emit further suite-specific artifacts into
``args.out`` / ``args.plots`` alongside.

    <suite>_<UTCts>.json / .md       — per-suite Report emissions
    plots/<suite>_<UTCts>_*.png      — bar grids (+ speedup curve when
                                       --x-param/--speedup-vs given)
    REPORT_<UTCts>.md                — header index across suites

Usage:

    .venv/bin/python bench/run_all.py --device cpu --quick
    .venv/bin/python bench/run_all.py --suites reassoc_scale,search_efficiency
    .venv/bin/python bench/run_all.py --suites reassoc_scale \\
        --x-param k --speedup-vs inductor --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit

#: Default suite order — harnessed modules first.
DEFAULT_SUITES = [
    "reassoc_scale",
    "search_efficiency",
    "real_linear_attn",
    "bench_e2e",
]


class _LaxNamespace(argparse.Namespace):
    """Namespace whose *missing* attributes read as ``None``.

    Suite ``run_bench`` implementations typically direct-access the
    same flags their ``main()`` parser defines; handing them a lax
    namespace lets every unset suite-specific knob fall through to
    the module's own defaults instead of AttributeError-ing.
    """

    def __getattr__(self, name: str) -> None:
        return None


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_suite(name: str):
    """Import ``bench/<name>.py``; accepts ``name``/``bench.name``/``name.py``."""
    modname = name.removesuffix(".py").rsplit(".", 1)[-1]
    return importlib.import_module(modname)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="run harnessed bench suites, emit reports"
    )
    ap.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="comma-separated bench module names "
        f"(default: {','.join(DEFAULT_SUITES)})",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results"),
        help="output dir (default bench/results)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="short timings: warmup=2, min_run_time=0.05; modules may "
        "shrink their sweep via a module-level QUICK dict",
    )
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--min-run-time", type=float, default=None)
    ap.add_argument(
        "--x-param",
        default=None,
        help="sweep-coord name for the speedup-curve x axis",
    )
    ap.add_argument(
        "--speedup-vs",
        default=None,
        help="baseline variant name for the speedup curve "
        "(e.g. inductor)",
    )
    args = ap.parse_args()

    warmup = (
        args.warmup
        if args.warmup is not None
        else (2 if args.quick else 5)
    )
    min_run_time = (
        args.min_run_time
        if args.min_run_time is not None
        else (0.05 if args.quick else 0.2)
    )

    outdir = Path(args.out)
    plotsdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotsdir.mkdir(parents=True, exist_ok=True)
    ts = _stamp()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    rows = []  # (suite, status, artifact links, detail)
    env = benchkit.collect_env(args.device)

    for suite in suites:
        print(f"=== {suite} ===", flush=True)
        try:
            mod = _load_suite(suite)
        except Exception as e:
            print(f"  import failed: {e}", flush=True)
            rows.append((suite, "import failed", "—", str(e)))
            continue

        run_bench = getattr(mod, "run_bench", None)
        if run_bench is None:
            print("  ad-hoc script — not harnessed", flush=True)
            rows.append((suite, "ad-hoc, not harnessed", "—", ""))
            continue

        bench_args = _LaxNamespace(
            device=args.device,
            quick=args.quick,
            warmup=warmup,
            min_run_time=min_run_time,
            out=outdir,
            plots=plotsdir,
        )
        # Optional per-suite quick-sweep overrides.
        if args.quick:
            for k, v in (getattr(mod, "QUICK", None) or {}).items():
                setattr(bench_args, k, v)

        try:
            report = run_bench(bench_args)
        except Exception:
            traceback.print_exc()
            rows.append((suite, "crashed", "—", "see stderr"))
            continue
        if not isinstance(report, benchkit.Report):
            rows.append(
                (
                    suite,
                    "bad return",
                    "—",
                    f"run_bench returned {type(report).__name__}, "
                    "not a benchkit.Report",
                )
            )
            continue

        stem = f"{suite}_{ts}"
        jpath = outdir / f"{stem}.json"
        mpath = outdir / f"{stem}.md"
        try:
            report.to_json(jpath)
            report.to_markdown(mpath, speedup_vs=args.speedup_vs)
            pngs = report.to_plots(
                plotsdir,
                x_param=args.x_param,
                speedup_vs=args.speedup_vs,
                stem=stem,
            )
        except Exception:
            traceback.print_exc()
            rows.append((suite, "emit failed", "—", "see stderr"))
            continue

        links = f"[json]({jpath.name}) · [md]({mpath.name})" + (
            f" · {len(pngs)} plots" if pngs else ""
        )
        rows.append(
            (suite, f"ok ({len(report.cells)} cells)", links, "")
        )

    # Combined header index.
    ipath = outdir / f"REPORT_{ts}.md"
    lines = [
        f"# catopt bench run — {ts}",
        "",
        f"- **device**: `{env.get('device', '?')}`"
        f" — {env.get('device_name', '?')}",
        f"- **torch**: {env.get('torch', '?')}"
        f" · **cuda**: {env.get('torch_cuda', '?')}"
        f" · **python**: {env.get('python', '?')}",
        f"- **git**: `{env.get('git_sha', '?')}`"
        + (" dirty" if env.get("git_dirty") else " clean")
        + f" · **quick**: {args.quick}",
        "",
        "| suite | status | artifacts | note |",
        "|---|---|---|---|",
    ]
    for suite, status, links, note in rows:
        lines.append(f"| {suite} | {status} | {links} | {note} |")
    ipath.write_text("\n".join(lines) + "\n")

    print(f"\nindex: {ipath}", flush=True)
    failed = (
        "import failed",
        "crashed",
        "emit failed",
        "bad return",
    )
    n_bad = sum(1 for _, s, _, _ in rows if s in failed)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
