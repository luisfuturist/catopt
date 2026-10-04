"""Compatibility entry point — the engine moved to ``catopt_discovery``.

``python tools/law_pipeline.py …`` keeps working; the canonical
invocation is ``python -m catopt_discovery.pipeline …``.
"""

from catopt_discovery.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
