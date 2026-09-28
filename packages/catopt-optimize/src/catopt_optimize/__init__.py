"""catopt-optimize — the optimizer orchestrators.

Pipeline entry points (:func:`optimize_model`,
:func:`optimize_compositional`, `discover_alternatives` in
:mod:`~catopt_optimize.optimize`), the executor/regime dispatch
(:mod:`~catopt_optimize.regime`), and per-device cost calibration
(:mod:`~catopt_optimize.calibrate`).
"""

from catopt_optimize.autotune import optimize_model_autotuned
from catopt_optimize.criteria import (
    Blend,
    CompiledCriterion,
    Criteria,
    Criterion,
    DepthCriterion,
    FlopsCriterion,
    LatencyCriterion,
    MemoryCriterion,
    criteria_cost,
    peak_bytes_cost,
)
from catopt_optimize.export import (
    ExportError,
    export_optimized,
    load_optimized,
    save_optimized,
)
from catopt_optimize.optimize import (
    OptimizationResourceError,
    optimize_compositional,
    optimize_model,
)
from catopt_optimize.runners import (
    ChainedRunner,
    CompiledRunner,
    CudaGraphRunner,
    GenericRunner,
    Runner,
    runner_candidate,
)

__all__ = [
    "Blend",
    "ChainedRunner",
    "CompiledCriterion",
    "CompiledRunner",
    "Criteria",
    "Criterion",
    "CudaGraphRunner",
    "DepthCriterion",
    "ExportError",
    "FlopsCriterion",
    "GenericRunner",
    "LatencyCriterion",
    "MemoryCriterion",
    "OptimizationResourceError",
    "Runner",
    "criteria_cost",
    "export_optimized",
    "load_optimized",
    "optimize_compositional",
    "optimize_model",
    "optimize_model_autotuned",
    "peak_bytes_cost",
    "runner_candidate",
    "save_optimized",
]
