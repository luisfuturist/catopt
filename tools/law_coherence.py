"""Compatibility entry point — the engine moved to ``catopt_discovery``.

``python tools/law_coherence.py …`` keeps working; the canonical
invocation is ``python -m catopt_discovery.coherence …``.
"""

from catopt_discovery.coherence import main

if __name__ == "__main__":
    raise SystemExit(main())
