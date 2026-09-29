"""Cyclomatic-complexity ratchet (radon).

The engine is math-heavy: several rewrite/planning routines are
legitimately complex, so a hard ceiling would fail the tree at HEAD.
Instead this is a *ratchet* (same spirit as the ty/vulture/coverage
gates): every function's complexity is pinned in
``tools/complexity_baseline.json`` and may not increase; functions not
in the baseline must stay at or below ``THRESHOLD`` (rank C).

Run ``.venv/bin/python tools/radon_ratchet.py`` to check, or
``--update`` to regenerate the baseline after an intentional change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["packages"]
BASELINE = Path(__file__).resolve().parent / "complexity_baseline.json"
# Rank-C boundary: a brand-new function may be at most this complex
# before it must be justified and added to the baseline explicitly.
THRESHOLD = 11


def _iter_blocks(blocks):
    """Yield every block, descending into classes and closures."""
    for block in blocks:
        yield block
        if isinstance(block, Class):
            yield from _iter_blocks(block.methods)
        yield from _iter_blocks(getattr(block, "closures", []))


def scan() -> dict[str, int]:
    """Map ``<path>::<qualname>`` to its cyclomatic complexity."""
    out: dict[str, int] = {}
    for target in TARGETS:
        for path in sorted((ROOT / target).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            for block in _iter_blocks(cc_visit(path.read_text())):
                out[f"{rel}::{block.fullname}"] = block.complexity
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the ratchet check (or ``--update`` the baseline)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the current tree",
    )
    args = parser.parse_args(argv)

    current = scan()
    if args.update:
        BASELINE.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {len(current)} entries to {BASELINE.name}")
        return 0

    baseline = json.loads(BASELINE.read_text())
    problems: list[str] = []
    for key, cc in sorted(current.items()):
        recorded = baseline.get(key)
        if recorded is None:
            if cc > THRESHOLD:
                problems.append(
                    f"new function {key} is complexity {cc} (> {THRESHOLD})"
                )
        elif cc > recorded:
            problems.append(f"{key} rose to {cc} (baseline {recorded})")

    if problems:
        print("complexity ratchet failed:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"complexity ratchet ok ({len(current)} functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
