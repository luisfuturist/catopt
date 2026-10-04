"""Compatibility entry point — the engine moved to ``catopt_discovery``.

``python tools/law_intake.py …`` keeps working; the canonical
invocation is ``python -m catopt_discovery.intake …``.
"""

from catopt_discovery.intake import main

if __name__ == "__main__":
    raise SystemExit(main())
