"""Compatibility entry point — the engine moved to ``catopt_discovery``.

``python tools/law_shape_census.py …`` keeps working; the canonical
invocation is ``python -m catopt_discovery.census …``.
"""

from catopt_discovery.census import main

if __name__ == "__main__":
    raise SystemExit(main())
