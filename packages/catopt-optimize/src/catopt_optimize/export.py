"""Compatibility shim — the export module moved (plan 0007).

Production export of optimized models is torch machinery; it lives in
:mod:`catopt_torch.export`.  Importing this module resolves to the
torch module — the historical ``catopt_optimize.export`` /
``catopt.export`` attribute paths (and private helpers like
``_diff``/``_read_safetensors`` the tests exercise) stay working —
while the neutral orchestrator itself imports nothing
backend-coupled.
"""

import importlib
import sys as _sys

#: Replace this module object with the torch implementation.
_sys.modules[__name__] = importlib.import_module("catopt_torch.export")
