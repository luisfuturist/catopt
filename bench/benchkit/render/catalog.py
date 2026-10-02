"""Catalog renderer — the docs, generated from the registry.

`bench/registry.py` is the single source of truth for what a suite
asks, what it expects, and how it is categorized; the README tables and
the dashboard grouping are rendered from it so they cannot drift.
"""

from __future__ import annotations


def render_catalog_markdown() -> str:
    """The suite catalog as Markdown, grouped by intent."""
    from bench import registry

    lines: list[str] = []
    for intent in registry.INTENTS:
        suites = [s for s in registry.SUITES if s.intent == intent]
        if not suites:
            continue
        lines += [
            f"### {intent}",
            "",
            "| suite | tier | question | expected |",
            "|---|---|---|---|",
        ]
        for s in suites:
            flags = []
            if s.status != "harnessed":
                flags.append(s.status)
            if s.needs_cuda:
                flags.append("cuda")
            suffix = f" _({', '.join(flags)})_" if flags else ""
            lines.append(
                f"| `{s.name}`{suffix} | {s.tier} | {s.question} "
                f"| {s.expects} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_mechanisms_markdown() -> str:
    """A suite-by-mechanism index (controlled vocabulary)."""
    from bench import registry

    lines = ["| mechanism | suites |", "|---|---|"]
    for mech in registry.MECHANISMS:
        names = ", ".join(
            f"`{s.name}`" for s in registry.by_mechanism(mech)
        )
        if names:
            lines.append(f"| {mech} | {names} |")
    return "\n".join(lines) + "\n"
