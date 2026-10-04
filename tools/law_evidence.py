"""Compatibility entry point — the engine moved to ``catopt_discovery``.

``python tools/law_evidence.py …`` keeps working; the canonical
invocation is ``python -m catopt_discovery.evidence …``.
"""

from catopt_discovery.evidence import main

if __name__ == "__main__":
    raise SystemExit(main())
