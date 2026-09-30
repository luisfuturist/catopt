"""catopt-orchestrator — the backend-neutral optimizer orchestrator.

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
``catopt_torch``, no ``catopt_cuda``.  Backend specifics arrive
through the ports the caller passes — e.g.
``Optimizer(backend=TorchBackend())`` with
``catopt_torch.backend.TorchBackend``.  The delivery runners live
with their backends (:class:`catopt_torch.runners.TorchCompileRunner`,
:class:`catopt_cuda.CudaGraphRunner`); the neutral
:class:`IdentityRunner` / :class:`ChainedRunner` /
:func:`runner_candidate` live in :mod:`catopt_orchestrator.runners`.
"""

from catopt_core.pipeline import Backend, LowerResult, SearchResult
from catopt_core.ports import Criterion, Runner

from catopt_orchestrator.criteria import (
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
from catopt_orchestrator.morphisms import (
    DEFAULT_MORPHISM_LAWS,
    BlockSig,
    CrossBlockCSE,
    KVLatentShare,
    MorphismGraph,
    MorphismLaw,
    MorphismMatch,
    MorphismSearch,
    NormSig,
    ReifySpec,
    WeightRef,
    Wire,
    block_signature,
    lift_graph,
    optimize_morphisms,
)
from catopt_orchestrator.optimize import (
    Autotuned,
    Compositional,
    Monolithic,
    OptimizationResourceError,
    Optimizer,
    discover_alternatives,
    lower,
    search,
    structural_key,
)
from catopt_orchestrator.runners import (
    ChainedRunner,
    IdentityRunner,
    runner_candidate,
)

__all__ = [
    "DEFAULT_MORPHISM_LAWS",
    "DEFAULT_RULES",
    "Autotuned",
    "Backend",
    "Blend",
    "BlockSig",
    "ChainedRunner",
    "CompiledCriterion",
    "Compositional",
    "Criteria",
    "Criterion",
    "CrossBlockCSE",
    "DepthCriterion",
    "FlopsCriterion",
    "IdentityRunner",
    "KVLatentShare",
    "LatencyCriterion",
    "LowerResult",
    "MemoryCriterion",
    "Monolithic",
    "MorphismGraph",
    "MorphismLaw",
    "MorphismMatch",
    "MorphismSearch",
    "NormSig",
    "OptimizationResourceError",
    "Optimizer",
    "ReifySpec",
    "Runner",
    "SearchResult",
    "WeightRef",
    "Wire",
    "block_signature",
    "criteria_cost",
    "default_rules",
    "discover_alternatives",
    "lift_graph",
    "lower",
    "optimize_morphisms",
    "peak_bytes_cost",
    "runner_candidate",
    "search",
    "structural_key",
]


def __getattr__(name: str):
    """Lazily resolve the backend-coupled surfaces.

    ``DEFAULT_RULES`` composes ``catopt_core.laws.DEFAULT`` with the
    carrier-package ``CARRIERS`` preset on first access, and
    ``default_rules`` is its accessor — both keep
    ``import catopt_orchestrator`` free of the (torch-coupled)
    carriers package.
    """
    if name in ("DEFAULT_RULES", "default_rules"):
        from catopt_orchestrator import optimize

        value = optimize.default_rules()
        resolved = {
            "DEFAULT_RULES": value,
            "default_rules": optimize.default_rules,
        }
        globals().update(resolved)
        return resolved[name]
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
