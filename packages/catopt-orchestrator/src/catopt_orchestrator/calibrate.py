"""Compatibility shim — the calibration module moved (plan 0007).

The profile *data* half lives in torch-free
:mod:`catopt_core.profile`; the measurement half (``calibrate()`` and
the torch micro-benchmarks) lives in :mod:`catopt_torch.calibrate`.
Importing this module resolves to the torch module — the historical
``catopt_optimize.calibrate`` / ``catopt.calibrate`` attribute paths
(and private probe names the tests patch) stay working — while the
neutral orchestrator itself imports nothing backend-coupled.
"""

import importlib
import sys as _sys

#: Replace this module object with the torch implementation: every
#: attribute — public API and private probe alike — resolves there,
#: and ``monkeypatch.setattr(catopt_optimize.calibrate, ...)`` lands on
#: the real module dict.
_sys.modules[__name__] = importlib.import_module(
    "catopt_torch.calibrate"
)
