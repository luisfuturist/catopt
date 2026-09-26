"""Shared bench harness — timing, statistics, and report emission.

Every harnessed bench builds ``Case`` objects — one per sweep cell:
a display name, the sweep coordinates (``params``), and the
``Variant`` callables to time — runs them through ``Runner``
(``torch.utils.benchmark.Timer.blocked_autorange``; median + IQR),
and packs the resulting ``Cell``s into a ``Report`` carrying
environment provenance from ``collect_env()``.

``Report`` emits:

* ``to_json(path)``     — machine-readable cells + env.
* ``to_markdown(path)`` — GitHub-flavored results table + env preamble.
* ``to_plots(outdir)``  — grouped bar charts (median ms per variant,
  log-y when the spread exceeds 50x) and, when given an ``x_param``
  sweep axis and a ``speedup_vs`` baseline variant name, a
  speedup-vs-baseline curve via ``plot_cells``.

Typical use (also the convention ``run_all.py`` drives):

    from benchkit import Case, Report, Runner, Variant, collect_env

    runner = Runner(device="cpu", warmup=5, min_run_time=0.2)
    cells = runner.run(cases)
    report = Report(suite="my_bench", cells=cells,
                    env=collect_env("cpu"))
    report.to_json("results/my_bench.json")
    report.to_markdown("results/my_bench.md")
    report.to_plots("results/plots", x_param="k", speedup_vs="inductor")

Self-test — times a trivial case end-to-end and writes JSON + MD +
PNGs under ``/tmp/benchkit_smoke`` (or ``--out``):

    .venv/bin/python bench/benchkit.py
    .venv/bin/python bench/benchkit.py --out /tmp/benchkit_smoke
"""
# ruff: noqa: RUF001 RUF003 -- ×, ·, — in strings are deliberate
# math notation; same convention as catopt_core.laws.

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.benchmark import Timer

# Repo root — bench/ sits directly under it.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """One timed implementation inside a ``Case`` cell."""

    name: str  # "eager" | "inductor" | "catopt" | ...
    stmt: Callable[[], object]  # one forward call, CPU/GPU-synced
    flops: float | None = None  # analytic runtime-FLOPs, for report
    note: str = ""  # provenance notes (e.g. "weight-folded")


@dataclass
class Case:
    """One sweep cell: display name, coordinates, variants to time."""

    name: str  # display name
    params: dict  # sweep coords — ordered cols in report
    variants: list[Variant]
    aux: dict = field(default_factory=dict)  # verify diffs, proof notes


@dataclass
class Cell:
    """Timing outcome for one ``Case``."""

    case: Case
    medians: dict[str, float]  # variant -> median seconds
    iqr: dict[str, float]  # interquartile range seconds
    aux: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


class Runner:
    """Times ``Case`` variants with ``Timer.blocked_autorange``.

    ``stmt`` callables are invoked through ``Timer(stmt="_fn()",
    globals={...})`` — torch's Timer only accepts string statements.
    On CUDA devices each timed call is wrapped so it ends in
    ``torch.cuda.synchronize()``, i.e. the measured time includes the
    GPU tail rather than just kernel-launch overhead.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        warmup: int = 5,
        min_run_time: float = 0.2,
        num_threads: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.warmup = warmup
        self.min_run_time = min_run_time
        self.num_threads = num_threads

    def _wrap(self, stmt: Callable[[], object]) -> Callable[[], object]:
        if self.device.type != "cuda":
            return stmt

        def synced() -> object:
            out = stmt()
            torch.cuda.synchronize()
            return out

        return synced

    def run_case(self, case: Case) -> Cell:
        medians: dict[str, float] = {}
        iqrs: dict[str, float] = {}
        for v in case.variants:
            stmt = self._wrap(v.stmt)
            for _ in range(max(self.warmup, 0)):
                stmt()
            kwargs = {}
            if self.num_threads is not None:
                kwargs["num_threads"] = self.num_threads
            timer = Timer(stmt="_fn()", globals={"_fn": stmt}, **kwargs)
            meas = timer.blocked_autorange(
                min_run_time=self.min_run_time
            )
            medians[v.name] = meas.median
            iqrs[v.name] = meas.iqr
        return Cell(
            case=case, medians=medians, iqr=iqrs, aux=dict(case.aux)
        )

    def run(self, cases: list[Case]) -> list[Cell]:
        """Run every case, printing a one-line summary per cell."""
        cells = []
        for case in cases:
            cell = self.run_case(case)
            cells.append(cell)
            coords = " ".join(
                f"{k}={v}" for k, v in case.params.items()
            )
            times = "  ".join(
                f"{n}={cell.medians[n] * 1e3:.3f}ms"
                for n in cell.medians
            )
            print(f"  {case.name} [{coords}]  {times}", flush=True)
        return cells


# ---------------------------------------------------------------------------
#  Environment provenance
# ---------------------------------------------------------------------------


def _git_state() -> tuple[str, bool]:
    """``(short HEAD sha, dirty)`` for the repo, or ``("unknown", ...)``."""
    try:
        sha = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", False


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def collect_env(device: str | torch.device | None = None) -> dict:
    """Provenance for a report: versions, device, git state, UTC time."""
    if device is None:
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        dev = torch.device(device)
    device_name = (
        torch.cuda.get_device_name(dev)
        if dev.type == "cuda" and torch.cuda.is_available()
        else _cpu_model()
    )
    sha, dirty = _git_state()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "none",
        "device": str(dev),
        "device_name": device_name,
        "cpu_count": os.cpu_count(),
        "git_sha": sha,
        "git_dirty": dirty,
    }


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


def _json_default(o: object) -> object:
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float | int | bool | str):
        return o
    return str(o)


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:.4g}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _variant_names(cells: list[Cell]) -> list[str]:
    """Union of variant names, ordered by first appearance."""
    names: list[str] = []
    for cell in cells:
        for name in cell.medians:
            if name not in names:
                names.append(name)
    return names


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class Report:
    """A named suite's results + environment provenance."""

    suite: str
    cells: list[Cell]
    env: dict = field(default_factory=collect_env)

    # -- serialization -----------------------------------------------------

    def _cell_record(self, cell: Cell) -> dict:
        return {
            "name": cell.case.name,
            "params": dict(cell.case.params),
            "variants": [
                {"name": v.name, "flops": v.flops, "note": v.note}
                for v in cell.case.variants
            ],
            "median_s": dict(cell.medians),
            "iqr_s": dict(cell.iqr),
            "aux": dict(cell.aux),
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite": self.suite,
            "env": self.env,
            "units": {"median_s": "seconds", "iqr_s": "seconds"},
            "cells": [self._cell_record(c) for c in self.cells],
        }
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )

    # -- markdown -----------------------------------------------------------

    def to_markdown(
        self, path: str | Path, speedup_vs: str | None = None
    ) -> None:
        """GitHub-flavored results table with an env preamble.

        Cells show ``median ± IQR`` in milliseconds.  When
        ``speedup_vs`` names a baseline variant, each non-baseline
        variant gets an extra ``<name>/vs-<base>`` column with the
        speedup ratio (baseline median / variant median; >1 = faster).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        env = self.env
        names = _variant_names(self.cells)
        param_keys: list[str] = []
        for cell in self.cells:
            for k in cell.case.params:
                if k not in param_keys:
                    param_keys.append(k)

        lines = [
            f"# {self.suite}",
            "",
            f"- **when**: {env.get('timestamp_utc', '?')}"
            f" (git `{env.get('git_sha', '?')}`"
            + (" dirty" if env.get("git_dirty") else " clean")
            + ")",
            f"- **device**: `{env.get('device', '?')}`"
            f" — {env.get('device_name', '?')}",
            f"- **torch**: {env.get('torch', '?')}"
            f" · **cuda**: {env.get('torch_cuda', '?')}"
            f" · **python**: {env.get('python', '?')}",
            "",
        ]

        cols = ["case", *param_keys]
        cols += [f"{n} ms" for n in names]
        if speedup_vs:
            cols += [
                f"{n}/vs-{speedup_vs}" for n in names if n != speedup_vs
            ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))

        for cell in self.cells:
            row = [cell.case.name]
            row += [
                str(cell.case.params.get(k, "—")) for k in param_keys
            ]
            for n in names:
                if n in cell.medians:
                    med = _fmt_ms(cell.medians[n])
                    iqr = _fmt_ms(cell.iqr.get(n, 0.0))
                    row.append(f"{med} ± {iqr}")
                else:
                    row.append("—")
            if speedup_vs:
                base = cell.medians.get(speedup_vs)
                for n in names:
                    if n == speedup_vs:
                        continue
                    if base and n in cell.medians:
                        row.append(f"{base / cell.medians[n]:.3f}×")
                    else:
                        row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        aux_cells = [c for c in self.cells if c.aux]
        if aux_cells:
            lines += ["", "## aux", ""]
            for cell in aux_cells:
                items = " · ".join(
                    f"`{k}`={v}" for k, v in cell.aux.items()
                )
                lines.append(f"- **{cell.case.name}**: {items}")

        flops_cells = [
            c
            for c in self.cells
            if any(v.flops for v in c.case.variants)
        ]
        if flops_cells:
            lines += ["", "## analytic runtime-FLOPs", ""]
            for cell in flops_cells:
                items = " · ".join(
                    f"`{v.name}`={v.flops:.3g}"
                    for v in cell.case.variants
                    if v.flops
                )
                lines.append(f"- **{cell.case.name}**: {items}")

        path.write_text("\n".join(lines) + "\n")

    def to_dataframe(self):
        """Flatten cells to a ``pandas.DataFrame`` (import guarded)."""
        import pandas as pd

        rows = []
        for cell in self.cells:
            row = {"case": cell.case.name, **cell.case.params}
            for n, s in cell.medians.items():
                row[f"{n}_median_ms"] = s * 1e3
            for n, s in cell.iqr.items():
                row[f"{n}_iqr_ms"] = s * 1e3
            rows.append(row)
        return pd.DataFrame(rows)

    # -- plots ---------------------------------------------------------------

    def to_plots(
        self,
        outdir: str | Path,
        x_param: str | None = None,
        speedup_vs: str | None = None,
        stem: str | None = None,
    ) -> list[str]:
        """Emit PNGs under ``outdir``; returns the paths written.

        One grouped bar chart per param-key grid (cases sharing the
        same sweep-coordinate names) — median ms per variant, IQR
        whiskers, log-y when the median spread exceeds 50x.  When
        ``x_param`` and ``speedup_vs`` are given, also emits a
        speedup-vs-baseline curve via ``plot_cells``.
        """
        plt = _plt()
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or self.suite
        paths: list[str] = []

        # Group cells by their sweep-coordinate signature.
        grids: dict[tuple, list[Cell]] = {}
        for cell in self.cells:
            key = tuple(cell.case.params.keys())
            grids.setdefault(key, []).append(cell)

        for gi, (key, cells) in enumerate(grids.items()):
            names = _variant_names(cells)
            if not names:
                continue
            labels = []
            for c in cells:
                coords = ",".join(
                    f"{k}={v}" for k, v in c.case.params.items()
                )
                labels.append(
                    f"{c.case.name}\n{coords}"
                    if coords
                    else c.case.name
                )
            n_x, n_v = len(cells), len(names)
            width = 0.8 / max(n_v, 1)
            fig, ax = plt.subplots(figsize=(max(6, n_x * 1.4), 4.5))
            for vi, name in enumerate(names):
                xs = [
                    i + (vi - (n_v - 1) / 2) * width for i in range(n_x)
                ]
                ys = [c.medians.get(name) for c in cells]
                es = [c.iqr.get(name, 0.0) for c in cells]
                xs = [
                    x
                    for x, y in zip(xs, ys, strict=True)
                    if y is not None
                ]
                es = [
                    e * 1e3
                    for e, y in zip(es, ys, strict=True)
                    if y is not None
                ]
                ys = [y * 1e3 for y in ys if y is not None]
                ax.bar(
                    xs,
                    ys,
                    width=width * 0.9,
                    yerr=es,
                    capsize=2,
                    label=name,
                )
            meds = [
                m for c in cells for m in c.medians.values() if m > 0
            ]
            if meds and max(meds) / min(meds) > 50:
                ax.set_yscale("log")
            ax.set_xticks(range(n_x))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("median ms/call (IQR whiskers)")
            suffix = f" ({', '.join(key)})" if key else ""
            ax.set_title(f"{self.suite}{suffix}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = outdir / f"{_slug(stem)}_grid{gi}.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(str(p))

        if x_param and speedup_vs:
            paths.append(
                self.plot_cells(
                    x_param,
                    speedup_vs=speedup_vs,
                    outdir=outdir,
                    stem=stem,
                )
            )
        return paths

    def plot_cells(
        self,
        x_param: str,
        speedup_vs: str,
        outdir: str | Path,
        stem: str | None = None,
    ) -> str:
        """Speedup-vs-``speedup_vs`` curve over the ``x_param`` axis.

        For every variant present alongside the baseline, plots
        ``baseline_median / variant_median`` (>1 = faster than the
        baseline) against the sorted ``x_param`` sweep values.  Cells
        lacking ``x_param`` or the baseline are skipped.  Returns the
        PNG path.
        """
        plt = _plt()
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or self.suite

        usable = [
            c
            for c in self.cells
            if x_param in c.case.params and speedup_vs in c.medians
        ]

        def _xval(c: Cell):
            v = c.case.params[x_param]
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v))

        usable.sort(key=_xval)
        xs = [c.case.params[x_param] for c in usable]
        names = [n for n in _variant_names(usable) if n != speedup_vs]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in names:
            pts = [
                (x, c.medians[speedup_vs] / c.medians[name])
                for x, c in zip(xs, usable, strict=True)
                if name in c.medians
            ]
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    label=name,
                )
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(x_param)
        ax.set_ylabel(
            f"speedup vs {speedup_vs} "
            f"(median {speedup_vs} / median variant; >1 = faster)"
        )
        ax.set_title(f"{self.suite}: speedup vs {speedup_vs}")
        if len(set(map(str, xs))) > 4:
            try:
                vals = [float(v) for v in xs]
                if min(vals) > 0 and max(vals) / min(vals) > 20:
                    ax.set_xscale("log")
            except (TypeError, ValueError):
                pass
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / f"{_slug(stem)}_speedup_vs_{_slug(speedup_vs)}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        return str(p)


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------


def _smoke(out: Path) -> None:
    """Time a trivial case end-to-end; write JSON + MD + PNGs."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 256, generator=g)
    w = torch.randn(256, 256, generator=g)
    w2 = torch.randn(256, 256, generator=g)
    xb = torch.randn(512, 512, generator=g)
    wb = torch.randn(512, 512, generator=g)
    cases = [
        Case(
            name="matmul",
            params={"n": 256},
            variants=[
                Variant(
                    "one_mm",
                    lambda: x @ w,
                    flops=2 * 256**3,
                    note="single GEMM",
                ),
                Variant(
                    "two_mm",
                    lambda: (x @ w) @ w2,
                    flops=4 * 256**3,
                    note="two chained GEMMs",
                ),
            ],
            aux={"smoke": True},
        ),
        Case(
            name="matmul_big",
            params={"n": 512},
            variants=[
                Variant(
                    "one_mm",
                    lambda: xb @ wb,
                    flops=2 * 512**3,
                ),
            ],
        ),
    ]
    runner = Runner(device="cpu", warmup=2, min_run_time=0.05)
    report = Report(
        suite="benchkit_smoke",
        cells=runner.run(cases),
        env=collect_env("cpu"),
    )
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "smoke.json", out / "smoke.md"
    report.to_json(j)
    report.to_markdown(m, speedup_vs="two_mm")
    pngs = report.to_plots(
        out / "plots", x_param="n", speedup_vs="one_mm"
    )
    print(f"wrote {j}\nwrote {m}")
    for p in pngs:
        print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="benchkit self-test")
    ap.add_argument(
        "--out",
        default="/tmp/benchkit_smoke",
        help="output dir (default /tmp/benchkit_smoke)",
    )
    _args = ap.parse_args()
    _smoke(Path(_args.out))
