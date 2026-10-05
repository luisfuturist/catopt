"""Law-discovery engine for catopt.

The meta-level toolkit that finds, verifies and ranks candidate
rewrite laws over the core IR: shape census (:mod:`.census`),
proposal generators (:mod:`.proposal`, :mod:`.shape_proposal`,
:mod:`.grammar`), oracles (:mod:`.verifier`, :mod:`.oracle`), the
impact/verdict probes (:mod:`.impact`, :mod:`.pipeline`), the sqlite
evidence store (:mod:`.evidence`), the coherence catalogue
(:mod:`.coherence`, :mod:`.coherence2`), workload intake
(:mod:`.intake`, :mod:`.workload_gen`), and the emit/vocab/meta-game
helpers, and the object constructor (:mod:`.object_synthesis` — the
ADR-0004 "construction operations" that *build* declared objects for
the evidence store to admit).  Formerly the ``tools/law_*.py``
scripts; every module keeps its CLI — invoke as
``python -m catopt_discovery.<module>``.

.. data:: REPO_ROOT

   Absolute path of the repository root, derived from this editable
   file location (``packages/catopt-discovery/src/catopt_discovery``
   is four levels below the root).  Used to reach ``bench`` (a
   repo-root package, not an installed distribution) and the
   ``tools/`` artifacts that stayed behind.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS = REPO_ROOT / "tools"
