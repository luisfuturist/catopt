"""catopt-optimize — the backend-neutral optimizer orchestrator.

The public seam (plan 0006/0007): :class:`Optimizer` — the configured
entry point over an explicit
:class:`~catopt_core.pipeline.Backend` (or explicit
``source``/``sink``/``composer``/``meter`` ports — there is no
default backend) — the phase verbs :func:`search` / :func:`lower`
and their result objects (:class:`SearchResult` /
:class:`LowerResult`), and the
:class:`~catopt_core.ports.Strategy` implementations
:class:`Monolithic` / :class:`Compositional` / :class:`Autotuned`.

This package imports **no backend**: no torch, no
``catopt_torch``, no ``catopt_cuda``.  The deprecated
``optimize_*`` wrappers with their torch defaults moved to
``catopt_torch.api``; they resolve lazily through module
``__getattr__`` — the historical names still work on a torch
install (``catopt.optimize_model`` and friends), but importing this
package alone never loads a tensor library.  The
executor/dispatch (:mod:`catopt_optimize.regime`), runner
(:mod:`catopt_optimize.runners`), calibrate and export module paths
are likewise lazy shims.
"""

import importlib
from typing import Any

from catopt_core.pipeline import Backend, LowerResult, SearchResult
from catopt_core.ports import Criterion, Runner

from catopt_optimize.criteria import (
    Blend,
    CompiledCriterion,
    Criteria,
    DepthCriterion,
    FlopsCriterion,
    LatencyCriterion,
    MemoryCriterion,
    criteria_cost,
    peak_bytes_cost,
)
from catopt_optimize.optimize import (
    Autotuned,
    Compositional,
    Monolithic,
    OptimizationResourceError,
    Optimizer,
    discover_alternatives,
    lower,
    search,
)
from catopt_optimize.runners import (
    ChainedRunner,
    IdentityRunner,
    runner_candidate,
)

__all__ = [
    "Autotuned",
    "Backend",
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

# ---------------------------------------------------------------------------
# Compatibility delegation — moved torch-facing names (lazy)
# ---------------------------------------------------------------------------
#
# The deprecated ``optimize_*`` entry points resolve torch defaults,
# so they live in ``catopt_torch.api``; the runner implementations in
# ``catopt_torch.runners`` / ``catopt_cuda``; the export helpers in
# ``catopt_torch.export``.  These resolve lazily so importing the
# orchestrator never loads a backend, while
# ``from catopt_optimize import optimize_model`` still works on a
# torch install.

_DELEGATED = {
    # torch-defaulted wrappers
    "optimize_model": "catopt_torch.api",
    "optimize_compositional": "catopt_torch.api",
    "optimize_model_autotuned": "catopt_torch.api",
    # torch delivery runners
    "TorchCompileRunner": "catopt_torch.runners",
    "CudaGraphRunner": "catopt_cuda",
    # torch production export
    "ExportError": "catopt_torch.export",
    "export_optimized": "catopt_torch.export",
    "load_optimized": "catopt_torch.export",
    "save_optimized": "catopt_torch.export",
}


def __getattr__(name: str) -> Any:
    """Resolve the moved torch-facing names lazily."""
    mod = _DELEGATED.get(name)
    if mod is not None:
        return getattr(importlib.import_module(mod), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
