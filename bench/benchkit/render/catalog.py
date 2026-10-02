"""Catalog renderer — the docs, generated from the registry.

`bench/registry.py` is the single source of truth for what a suite
asks, what it expects, and how it is categorized; the README tables and
the dashboard grouping are rendered from it so they cannot drift.

The README's catalog lives between the ``BEGIN``/``END`` markers below;
`readme_synced` / `write_readme` keep that block in lockstep with the
registry, and `tests/test_benchkit.py` fails if it drifts.
"""

from __future__ import annotations

from pathlib import Path

#: The README markers the generated block lives between.
BEGIN = "<!-- BEGIN GENERATED CATALOG -->"
END = "<!-- END GENERATED CATALOG -->"

#: The README that carries the generated block.
DEFAULT_README = Path("bench/README.md")


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


def render_catalog_block() -> str:
    """The full generated block, including the mechanism index."""
    return (
        render_catalog_markdown()
        + "\n## By mechanism\n\n"
        + render_mechanisms_markdown()
    )


def _spliced(text: str, block: str) -> str:
    """Replace the marker-delimited block inside ``text``."""
    before, sep, rest = text.partition(BEGIN)
    if not sep:
        raise ValueError(f"missing {BEGIN!r} marker")
    _, sep2, after = rest.partition(END)
    if not sep2:
        raise ValueError(f"missing {END!r} marker")
    return f"{before}{BEGIN}\n{block}{END}{after}"


def readme_synced(path: str | Path = DEFAULT_README) -> bool:
    """True when the README's generated block matches the registry."""
    text = Path(path).read_text()
    return text == _spliced(text, render_catalog_block())


def write_readme(path: str | Path = DEFAULT_README) -> Path:
    """Rewrite the README's generated block in place; returns the path."""
    path = Path(path)
    path.write_text(_spliced(path.read_text(), render_catalog_block()))
    return path
