"""Results document — the pinned baselines, rendered into one page.

`docs/results.md` is generated from `bench/baselines/*.json` so the
README can quote *findings* while every number keeps its provenance
(device, git sha, timestamp) and links back to the committed baseline.
"""

from __future__ import annotations

from pathlib import Path


def _provenance(env: dict, name: str) -> str:
    """One line naming the machine a baseline was measured on."""
    when = env.get("timestamp_utc", "?")
    dev = env.get("device_name", "?")
    sha = env.get("git_sha", "?")
    dirty = " (dirty)" if env.get("git_dirty") else ""
    return (
        f"Measured {when} on {dev} (git `{sha}`{dirty}) — "
        f"baseline [`bench/baselines/{name}`]"
        f"(../bench/baselines/{name})."
    )


def render_results_doc(
    baselines_dir: str | Path,
    title: str = "catopt — measured results",
) -> str:
    """Render every pinned baseline as one Markdown document."""
    from bench import registry
    from bench.benchkit.report import Report

    lines = [
        f"# {title}",
        "",
        "Generated from the pinned baselines in `bench/baselines/` —",
        "regenerate with `python -m bench results --out docs/results.md`.",
        "Each entry states what the suite asked, what it expected, what",
        "it measured, and the machine it measured on.  The full",
        "per-suite reports (timings, plots, raw metrics) are produced by",
        "`python -m bench report <baseline.json>`.",
        "",
    ]
    for path in sorted(Path(baselines_dir).glob("*.json")):
        try:
            spec = registry.get(path.stem)
        except KeyError:
            continue
        report = Report.from_json(path)
        lines += [f"## {spec.title}", "", f"*{spec.question}*", ""]
        lines += [f"**Expected** — {spec.expects}", ""]
        if report.findings:
            lines += [
                "| verdict | finding | headline |",
                "|---|---|---|",
            ]
            for f in report.findings:
                lines.append(
                    f"| **{f.verdict.badge}** | {f.claim} | {f.headline} |"
                )
            lines.append("")
        else:
            lines += [
                "> no findings recorded — baseline is not golden",
                "",
            ]
        lines += [_provenance(report.env, path.name), ""]
    return "\n".join(lines) + "\n"
