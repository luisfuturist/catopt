"""catopt-optimize — the optimizer orchestrators.

The public seam (plan 0006): :class:`Optimizer` — the configured entry
point with required ``source``/``sink`` ports — the phase verbs
:func:`search` / :func:`lower` and their result objects
(:class:`SearchResult` / :class:`LowerResult`), and the
:class:`~catopt_core.ports.Strategy` implementations
:class:`Monolithic` / :class:`Compositional` / :class:`Autotuned`.
The historical entry points (:func:`optimize_model`,
:func:`optimize_compositional`, :func:`optimize_model_autotuned`,
:func:`discover_alternatives` in :mod:`~catopt_optimize.optimize` /
:mod:`~catopt_optimize.autotune`) remain as wrappers; plus the
executor/regime dispatch (:mod:`~catopt_optimize.regime`) and
per-device cost calibration (:mod:`~catopt_optimize.calibrate`).
"""

from catopt_core.pipeline import LowerResult, SearchResult

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
    Autotuned,
    Compositional,
    Monolithic,
    OptimizationResourceError,
    Optimizer,
    discover_alternatives,
    lower,
    optimize_compositional,
    optimize_model,
    search,
)
from catopt_optimize.runners import (
    ChainedRunner,
    CudaGraphRunner,
    IdentityRunner,
    Runner,
    TorchCompileRunner,
    runner_candidate,
)

__all__ = [
    "Autotuned",
    "Blend",
    "ChainedRunner",
    "CompiledCriterion",
    "Compositional",
    "Criteria",
    "Criterion",
    "CudaGraphRunner",
    "DepthCriterion",
    "ExportError",
    "FlopsCriterion",
    "IdentityRunner",
    "LatencyCriterion",
    "LowerResult",
    "MemoryCriterion",
    "Monolithic",
    "OptimizationResourceError",
    "Optimizer",
    "Runner",
    "SearchResult",
    "TorchCompileRunner",
    "criteria_cost",
    "discover_alternatives",
    "export_optimized",
    "load_optimized",
    "lower",
    "optimize_compositional",
    "optimize_model",
    "optimize_model_autotuned",
    "peak_bytes_cost",
    "runner_candidate",
    "save_optimized",
    "search",
]
