"""catopt benchmarks — systematic, presentable measurement of the engine.

Layout:

* ``bench.benchkit`` — the harness: timing, provenance, findings, and
  the renderers (Markdown / HTML dashboard / Quarto / Slidev).
* ``bench.suites`` — the benchmarks themselves, grouped by intent
  (``correctness`` / ``speedup`` / ``e2e`` / …; the directory is the
  intent).
* ``bench.registry`` — the suite catalog the CLI drives.
* ``bench.cli`` — ``python -m bench``.

Run ``python -m bench list`` to see the catalog.
"""

from __future__ import annotations

__all__: list[str] = []
