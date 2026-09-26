"""catopt-optimize — the optimizer orchestrators.

Pipeline entry points (:func:`optimize_model`,
:func:`optimize_compositional`, `discover_alternatives` in
:mod:`~catopt_optimize.optimize`), the executor/regime dispatch
(:mod:`~catopt_optimize.regime`), and per-device cost calibration
(:mod:`~catopt_optimize.calibrate`).
"""

from catopt_optimize.optimize import (
    OptimizationResourceError,
    optimize_compositional,
    optimize_model,
)

__all__ = [
    "OptimizationResourceError",
    "optimize_compositional",
    "optimize_model",
]
