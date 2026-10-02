"""``python -m bench`` — the unified benchmark CLI.

    python -m bench list
    python -m bench catalog --check
    python -m bench run reassoc_scale --device cpu --quick
    python -m bench run-all --quick
    python -m bench report bench/results/reassoc_scale.json
    python -m bench compare reassoc_scale --baseline bench/baselines
    python -m bench gate

Command parsing is ``tyro`` (dataclasses -> typed flags), console
output is ``rich``.
"""

# ruff: noqa: RUF001 -- the multiplication sign in tables is notation.

from __future__ import annotations

import importlib
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from bench import registry
from bench.benchkit import Report
from bench.benchkit import ledger as _ledger


class _Lax:
    """Namespace whose missing attributes read as ``None``.

    Suite ``run_bench`` implementations direct-access the same flags
    their own parser defined; handing them a lax namespace lets every
    unset suite-specific knob fall through to the module's default.
    """

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)

    def __getattr__(self, name: str):
        return None


@dataclass
class ListConfig:
    """List the benchmark catalog."""

    intent: str | None = None
    mechanism: str | None = None


@dataclass
class CatalogConfig:
    """Emit, sync, or check the registry-generated suite catalog."""

    out: Path | None = None
    mechanisms: bool = False
    write: bool = False
    check: bool = False


@dataclass
class RunConfig:
    """Run one harnessed suite and emit its report."""

    suite: Annotated[str, tyro.conf.Positional]
    device: str = "cpu"
    quick: bool = False
    warmup: int = 5
    min_run_time: float = 0.2
    out: Path = Path("bench/results")
    speedup_vs: str | None = None
    x_param: str | None = None
    plots: bool = True
    html: bool = True
    slidev: bool = False
    ledger: bool = True


@dataclass
class RunAllConfig:
    """Run every harnessed suite in catalog order."""

    device: str = "cpu"
    quick: bool = False
    warmup: int = 5
    min_run_time: float = 0.2
    out: Path = Path("bench/results")
    suites: str = ""
    speedup_vs: str | None = None
    x_param: str | None = None
    ledger: bool = True


@dataclass
class ReportConfig:
    """Regenerate the presentation surfaces from a report JSON."""

    json: Annotated[Path, tyro.conf.Positional]
    out: Path = Path("bench/results")
    plots: bool = True
    html: bool = True
    quarto: bool = False
    slidev: bool = False
    speedup_vs: str | None = None
    x_param: str | None = None
    ledger: bool = False


@dataclass
class DashboardConfig:
    """Build the cross-suite HTML dashboard from report JSON."""

    results: Path = Path("bench/results")
    out: Path | None = None
    title: str = "catopt benchmarks"


@dataclass
class ResultsConfig:
    """Render the pinned baselines into one results document."""

    baselines: Path = Path("bench/baselines")
    out: Path = Path("docs/results.md")


@dataclass
class CompareConfig:
    """Compare a suite's latest run against its pinned baseline."""

    suite: Annotated[str, tyro.conf.Positional]
    baselines: Path = Path("bench/baselines")
    threshold: float = 0.05


@dataclass
class GateConfig:
    """Fail if any suite regressed beyond its baseline."""

    baselines: Path = Path("bench/baselines")
    threshold: float = 0.05
    out: Path = Path("bench/results")


def _console():
    from rich.console import Console

    return Console()


def _print_table(
    title: str, columns: list[str], rows: list[list[str]]
) -> None:
    from rich.table import Table

    table = Table(title=title, header_style="bold")
    for c in columns:
        table.add_column(c)
    for row in rows:
        table.add_row(*row)
    _console().print(table)


def _findings_summary(report: Report) -> None:
    if not report.findings:
        return
    _print_table(
        f"findings — {report.title or report.suite}",
        ["verdict", "claim", "headline"],
        [
            [f.verdict.badge, f.claim, f.headline or "—"]
            for f in report.findings
        ],
    )


def _emit(report: Report, cfg) -> list[str]:
    """Write every requested surface; returns the artifact paths."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = report.suite
    written: list[str] = []

    jpath = out / f"{stem}.json"
    report.to_json(jpath)
    written.append(str(jpath))

    mpath = out / f"{stem}.md"
    report.to_markdown(mpath, speedup_vs=cfg.speedup_vs)
    written.append(str(mpath))

    if getattr(cfg, "html", False):
        hpath = out / f"{stem}.html"
        report.to_html(hpath)
        written.append(str(hpath))

    figures: dict[str, str] = {}
    if getattr(cfg, "plots", False):
        paths = report.to_plots(
            out / "plots",
            x_param=cfg.x_param,
            speedup_vs=cfg.speedup_vs,
            stem=stem,
        )
        written += paths
        figures = {Path(p).stem: p for p in paths if p.endswith(".svg")}

    if getattr(cfg, "quarto", False):
        qpath = out / f"{stem}.qmd"
        report.to_quarto(qpath, figures)
        written.append(str(qpath))

    if getattr(cfg, "slidev", False):
        assets = report.to_slidev(out / "slidev", figures)
        written += list(assets.values())

    if getattr(cfg, "ledger", False):
        written.append(
            str(_ledger.append(report, out / "ledger.jsonl"))
        )

    return written


def _run_one(
    name: str,
    device: str,
    quick: bool,
    warmup: int,
    min_run_time: float,
    out: Path,
) -> Report:
    spec = registry.get(name)
    mod = importlib.import_module(spec.module)
    run_bench = getattr(mod, "run_bench", None)
    if run_bench is None:
        raise RuntimeError(
            f"{name} is not harnessed (status={spec.status})"
        )
    ns = _Lax(
        device=device,
        quick=quick,
        warmup=warmup,
        min_run_time=min_run_time,
        out=Path(out),
        plots=Path(out) / "plots",
        no_artifacts=True,  # the CLI owns emission
    )
    if quick:
        for k, v in (
            spec.quick or getattr(mod, "QUICK", None) or {}
        ).items():
            setattr(ns, k, v)
    report = run_bench(ns)
    if not isinstance(report, Report):
        raise RuntimeError(
            f"{name}.run_bench returned {type(report).__name__}, "
            "not a benchkit.Report"
        )
    # Flag quick runs so `gate`/`compare` never treat them as canonical.
    report.provenance["quick"] = bool(quick)
    return report


def cmd_list(cfg: ListConfig) -> int:
    """Print the catalog, grouped by intent."""
    rows = []
    for s in registry.SUITES:
        if cfg.intent and s.intent != cfg.intent:
            continue
        if cfg.mechanism and cfg.mechanism not in s.mechanisms:
            continue
        rows.append(
            [
                s.intent,
                s.name,
                s.tier,
                s.status,
                "cuda" if s.needs_cuda else "",
                ", ".join(s.mechanisms),
            ]
        )
    _print_table(
        "catopt bench suites (by intent)",
        ["intent", "suite", "tier", "status", "needs", "mechanisms"],
        rows,
    )
    return 0


def cmd_catalog(cfg: CatalogConfig) -> int:
    """Emit, sync, or check the registry-generated catalog Markdown."""
    from bench.benchkit.render import catalog as _catalog

    if cfg.check:
        if _catalog.readme_synced():
            _console().print("[green]catalog in sync[/green]")
            return 0
        _console().print(
            "[red]catalog drift[/red] — run "
            "`python -m bench catalog --write`"
        )
        return 1
    if cfg.write:
        path = _catalog.write_readme()
        _console().print(f"  [dim]→[/dim] {path}")
        return 0
    md = _catalog.render_catalog_markdown()
    if cfg.mechanisms:
        md += (
            "\n## By mechanism\n\n"
            + _catalog.render_mechanisms_markdown()
        )
    if cfg.out:
        Path(cfg.out).write_text(md)
        _console().print(f"  [dim]→[/dim] {cfg.out}")
    else:
        print(md)
    return 0


def cmd_run(cfg: RunConfig) -> int:
    """Run one suite."""
    try:
        report = _run_one(
            cfg.suite,
            cfg.device,
            cfg.quick,
            cfg.warmup,
            cfg.min_run_time,
            cfg.out,
        )
    except Exception as e:
        traceback.print_exc()
        _console().print(f"[red]run failed:[/red] {e}")
        return 1
    _findings_summary(report)
    written = _emit(report, cfg)
    for p in written:
        _console().print(f"  [dim]→[/dim] {p}")
    return 0


def cmd_run_all(cfg: RunAllConfig) -> int:
    """Run every harnessed suite."""
    names = (
        [s.strip() for s in cfg.suites.split(",") if s.strip()]
        if cfg.suites
        else [s.name for s in registry.harnessed()]
    )
    rows = []
    for name in names:
        _console().rule(f"[bold]{name}[/bold]")
        try:
            report = _run_one(
                name,
                cfg.device,
                cfg.quick,
                cfg.warmup,
                cfg.min_run_time,
                cfg.out,
            )
        except Exception as e:
            traceback.print_exc()
            rows.append([name, "crashed", str(e)[:60]])
            continue
        _findings_summary(report)
        written = _emit(report, cfg)
        rows.append(
            [
                name,
                f"ok ({len(report.cells)} cells)",
                f"{len(written)} files",
            ]
        )
    _print_table(
        "run-all summary", ["suite", "status", "artifacts"], rows
    )
    return 1 if any(r[1] == "crashed" for r in rows) else 0


def cmd_report(cfg: ReportConfig) -> int:
    """Rebuild surfaces from a report JSON."""
    report = Report.from_json(cfg.json)
    written = _emit(report, cfg)
    for p in written:
        _console().print(f"  [dim]→[/dim] {p}")
    return 0


def cmd_dashboard(cfg: DashboardConfig) -> int:
    """Build the cross-suite HTML index from report JSON."""
    from bench.benchkit.render.dashboard import build_dashboard

    index = build_dashboard(cfg.results, cfg.out, cfg.title)
    _console().print(f"  [dim]→[/dim] {index}")
    return 0


def cmd_results(cfg: ResultsConfig) -> int:
    """Render the pinned baselines into one results document."""
    from bench.benchkit.render.results import render_results_doc

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_results_doc(cfg.baselines))
    _console().print(f"  [dim]→[/dim] {out}")
    return 0


def cmd_compare(cfg: CompareConfig) -> int:
    """Compare the latest ledger run against the pinned baseline."""
    from bench.benchkit.compare import compare_baseline

    base = Path(cfg.baselines) / f"{cfg.suite}.json"
    if not base.exists():
        _console().print(f"[red]no baseline[/red] at {base}")
        return 1
    rec = _ledger.latest(cfg.suite)
    if rec is None:
        _console().print(f"[red]no ledger entry[/red] for {cfg.suite}")
        return 1
    if rec.get("provenance", {}).get("quick"):
        _console().print(
            f"[yellow]{cfg.suite}: latest ledger run was --quick[/yellow] "
            "— not comparable to a canonical baseline"
        )
        return 1
    regressions = compare_baseline(rec, base, cfg.threshold)
    if not regressions:
        _console().print(f"[green]{cfg.suite}: no regression[/green]")
        return 0
    _print_table(
        f"{cfg.suite} — regressions (>{cfg.threshold:.0%})",
        ["case", "variant", "baseline ms", "now ms", "ratio"],
        [
            [
                r.case,
                r.variant,
                f"{r.base_ms:.4g}",
                f"{r.now_ms:.4g}",
                f"{r.ratio:.3f}×",
            ]
            for r in regressions
        ],
    )
    return 1


def cmd_gate(cfg: GateConfig) -> int:
    """Fail if any baselined suite regressed; never pass vacuously."""
    from bench.benchkit.compare import (
        compare_baseline,
        expectation_gaps,
    )

    baselines = sorted(Path(cfg.baselines).glob("*.json"))
    if not baselines:
        _console().print(
            "[yellow]no baselines pinned[/yellow] — nothing to gate"
        )
        return 0

    bad = 0
    checked = 0
    skipped: list[str] = []
    quick: list[str] = []
    for base in baselines:
        rec = _ledger.latest(base.stem, cfg.out / "ledger.jsonl")
        if rec is None:
            skipped.append(base.stem)
            continue
        if rec.get("provenance", {}).get("quick"):
            quick.append(base.stem)
            continue
        checked += 1
        regressions = compare_baseline(rec, base, cfg.threshold)
        if regressions:
            bad += 1
            _console().print(
                f"[red]{base.stem}[/red]: {len(regressions)} regressions"
            )
    if skipped:
        _console().print(
            "[yellow]no ledger run to compare:[/yellow] "
            + ", ".join(skipped)
        )
    if quick:
        _console().print(
            "[yellow]latest run was --quick (not comparable):[/yellow] "
            + ", ".join(quick)
        )
    for gap in expectation_gaps(cfg.baselines):
        _console().print(f"[yellow]expectation gap:[/yellow] {gap}")
    if checked == 0:
        _console().print(
            "[yellow]gate: nothing compared[/yellow] — no baselined "
            "suite has a canonical (non-quick) ledger run"
        )
        return 0
    if bad:
        _console().print(
            f"[red]gate failed[/red] — {bad}/{checked} suite(s) regressed"
        )
        return 1
    _console().print(
        f"[green]gate passed[/green] — {checked} suite(s) compared"
    )
    return 0


#: The subcommand union — tyro turns each into a flat-flag subcommand.
Command = (
    Annotated[ListConfig, tyro.conf.subcommand("list")]
    | Annotated[CatalogConfig, tyro.conf.subcommand("catalog")]
    | Annotated[RunConfig, tyro.conf.subcommand("run")]
    | Annotated[RunAllConfig, tyro.conf.subcommand("run-all")]
    | Annotated[ReportConfig, tyro.conf.subcommand("report")]
    | Annotated[DashboardConfig, tyro.conf.subcommand("dashboard")]
    | Annotated[ResultsConfig, tyro.conf.subcommand("results")]
    | Annotated[CompareConfig, tyro.conf.subcommand("compare")]
    | Annotated[GateConfig, tyro.conf.subcommand("gate")]
)

_DISPATCH = {
    ListConfig: cmd_list,
    CatalogConfig: cmd_catalog,
    RunConfig: cmd_run,
    RunAllConfig: cmd_run_all,
    ReportConfig: cmd_report,
    DashboardConfig: cmd_dashboard,
    ResultsConfig: cmd_results,
    CompareConfig: cmd_compare,
    GateConfig: cmd_gate,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point — tyro builds the subcommands from the dataclasses."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _console().print(
            "usage: python -m bench {list,catalog,run,run-all,report,"
            "dashboard,results,compare,gate}"
        )
        return 2
    cmd = tyro.cli(Command, args=argv)
    return int(_DISPATCH[type(cmd)](cmd) or 0)
