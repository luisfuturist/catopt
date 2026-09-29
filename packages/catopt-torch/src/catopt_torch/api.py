"""The torch-facing pipeline entry points (plan 0007).

The deprecated ``optimize_*`` compatibility wrappers live here — the
ONLY place torch defaults appear.  They assemble the ports a
:class:`~catopt_core.pipeline.Backend` needs (via
:func:`~catopt_torch.backend.TorchBackend`), then delegate to the
backend-neutral orchestrator in :mod:`catopt_optimize`:

* :func:`optimize_model` — ``Optimizer(...).optimize(...)`` —
  :func:`~catopt_optimize.optimize.lower` ∘
  :func:`~catopt_optimize.optimize.search`, torch ports by default.
* :func:`optimize_compositional` — the same under the
  :class:`~catopt_optimize.optimize.Compositional` strategy.
* :func:`optimize_model_autotuned` — under
  :class:`~catopt_optimize.optimize.Autotuned`, with the torch
  candidate builders (:data:`TORCH_BUILDERS`: ``"torch_compile"``,
  ``"torch_compile_generic"``, ``"cuda_graph"``) resolving the
  built-in names.
* :func:`param_report` / :func:`save_optimized_weights` — the
  optimized weights-file audit helpers.

New code should call ``Optimizer(backend=TorchBackend()).optimize``
directly (or assemble the ports explicitly); these names retire with
the compatibility façade in plan 0008.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
from catopt_core.ops import OpTable
from catopt_core.ports import (
    Composer,
    CostFn,
    Executor,
    Meter,
    Runner,
    Sink,
    Source,
)
from catopt_optimize.autotune import (
    AutotuneContext,
    CandidateBuilder,
    CandidateUnavailableError,
)
from catopt_optimize.criteria import Criteria, Criterion
from catopt_optimize.optimize import (
    Autotuned,
    Compositional,
    Optimizer,
    _lower_extracted,
    _oom_to_resource_error,
    discover_alternatives,
    lower,
    search,
)
from catopt_optimize.runners import IdentityRunner

from catopt_torch.backend import TorchBackend
from catopt_torch.composer import param_report

__all__ = [
    "TORCH_BUILDERS",
    "discover_alternatives",
    "lower",
    "optimize_compositional",
    "optimize_model",
    "optimize_model_autotuned",
    "param_report",
    "save_optimized_weights",
    "search",
]


@_oom_to_resource_error
def optimize_model(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    cost_fn: CostFn | None = None,
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None,
    fusion_epsilon: float = 0.0,
    symmetry_budget: int | None = 2048,
    ops: OpTable | None = None,
    source: Source | None = None,
    sink: Sink | None = None,
    runner: Runner | None = None,
    verbose: bool = True,
) -> tuple[Executor, dict[str, Any]]:
    """End-to-end categorical optimization — the legacy entry point.

    A one-line wrapper: it resolves the historical
    ``source``/``sink``/``ops``/``runner`` defaults — torch IS the
    default here, that is what the compat name always meant — then
    runs ``Optimizer(...).optimize(...)``, i.e.
    :func:`~catopt_optimize.optimize.lower` ∘
    :func:`~catopt_optimize.optimize.search`.  New code should use
    the verbs (or :class:`Optimizer`) directly; this name retires in
    0008.

    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize.
    example_input : torch.Tensor
        An example input for tracing.
    ruleset : str
        Which rewrite rules to use: ``"all"``, ``"simpl"``, or ``"categorical"``.
    max_iterations : int
        Maximum equality-saturation iterations.
    max_enodes : int, optional
        E-node bound on the e-graph.  Enforced once per saturation
        iteration inside ``EGraph.run`` (plus per-match for budgeted
        rules) and re-checked at each phase boundary; reaching it
        raises :class:`OptimizationResourceError` rather than
        extracting from a truncated graph.  ``None`` disables the
        bound (and the run-loop watermark).
    max_memory_mb : float, optional
        Process memory bound in MiB — host RSS plus CUDA-allocated
        bytes — checked at phase boundaries; crossing it raises
        :class:`OptimizationResourceError`.  ``None`` (default)
        disables the check.
    cost_fn : CostFn | None
        Cost function for term extraction — the
        :class:`catopt_core.ports.CostFn` port.  Defaults to
        :func:`executor_cost_for` with ``lowering="generic"`` —
        roofline plus per-node dispatch overhead (solver ops
        surcharged).  Carrier-apply selections are routed to their
        level-batched executors at lowering time rather than priced
        in, because batched cost is non-additive over the spine and
        ``extract_best``'s local-cost decomposition can't see it.
    criteria : dict, Criterion, Criteria, or sequence, optional
        Selection axes blended into the extraction model — a
        ``{axis: weight}`` dict over the named axes
        (:data:`~catopt_optimize.criteria.AXES`), a single
        :class:`~catopt_optimize.criteria.Criterion`, a
        ``Criteria``/``Blend`` composition (e.g.
        ``LatencyCriterion() * 0.7 + MemoryCriterion("peak") * 0.3``),
        or a list of criteria / ``(criterion, weight)`` pairs; see
        :func:`catopt_optimize.criteria.criteria_cost`.  Consulted only
        when ``cost_fn`` is ``None`` — precedence is explicit
        ``cost_fn`` > ``criteria`` > the default model.
        ``stats["criteria"]`` records the normalised axes priced.
    fusion_epsilon : float, default 0.0
        Fusion-preferred near-tie band for extraction — forwarded to
        :meth:`EGraph.extract_best`: members priced within this
        relative band of the class minimum compete on
        :func:`catopt_core.cost.fusion_member_key` (predicted kernel
        count, then root fusibility) instead of structural size.
        Arm it when the delivered module will be ``torch.compile``d
        (``TorchCompileRunner`` / autotune's ``"torch_compile"``
        candidate — ``0.05`` is the validated band); ``0`` disables
        and selection is byte-identical to before.
    symmetry_budget : int, optional
        Per-rule enode budget for the expansive rules in
        ``_EXPANSIVE_RULES`` (monoid symmetries and scale hoists) —
        bounded saturation.  The reordering closure these rules
        generate grows Catalan-fast on stacked blocks (the residual
        accumulator's bracketings), which is what pushed monolithic
        eqsat past ~2 blocks.  ``None`` restores unbounded
        saturation.  The bound can only *miss* optimizations, never
        introduce wrong ones — every recorded merge is still a real
        equality.
    ops : OpTable, optional
        The op table the optimized term is lowered through (plan 0001
        phase 2c) — used to build the default :class:`TorchSink` and
        for the causal-mask constant evaluation.  ``None`` (default)
        resolves to ``OpTable.full()`` — the ambient ``_IR_TO_TORCH``
        table.  Ignored when ``sink`` is given.
    source : Source, optional
        The graph-source port (``model -> (IR, leaves)``), defaulting
        to :class:`TorchSource` (``torch.export``).  Pass a different
        source to optimize a non-torch frontend — the core search never
        imports torch itself.
    sink : Sink, optional
        The graph-sink port (``IR -> runnable``, plus its op set and
        equivalence gate), defaulting to ``TorchSink(ops=ops)``.
        Extraction is priced against ``sink.supported_ops``
        (:func:`catopt_core.cost.backend_cost`), so the optimizer only
        commits to forms the sink can lower.  Takes precedence over
        ``ops``.
    runner : Runner, optional
        Delivery-stage object deciding HOW the lowered executor is
        executed — see :mod:`catopt_optimize.runners`
        (:class:`IdentityRunner` identity, :class:`TorchCompileRunner`
        ``torch.compile``, :class:`CudaGraphRunner` CUDA-graph
        capture, :class:`ChainedRunner` left-to-right composition —
        e.g. ``ChainedRunner([TorchCompileRunner(), CudaGraphRunner()])``
        compiles first, then defers capture to the compile's
        outcome).  Applied once to the routed executor;
        ``stats["runner"]`` records its name (a list of member names
        for a chain) and the runner writes its own outcome keys
        (``stats["compiled"]``, ``stats["cuda_graph"]``).  ``None``
        delivers the module as lowered (:class:`IdentityRunner`).
        Duck-typed — any object with ``name`` and
        ``apply(module, example_input, stats)`` conforms.
    verbose : bool
        Print progress.  Also gates the equivalence check, exactly as
        before — the wrapper verifies iff ``verbose``.

    Returns
    -------
    (optimized_module, stats)
        The optimized ``torch.nn.Module`` and a dictionary of e-graph stats.

    Raises
    ------
    OptimizationResourceError
        When a resource bound is crossed (``max_enodes``,
        ``max_memory_mb``) or an allocator failure —
        ``torch.cuda.OutOfMemoryError``, ``MemoryError``, or the
        equivalent ``RuntimeError`` — is raised anywhere in the
        pipeline.  ``optimize_compositional`` treats this as a normal
        per-block fallback (status ``"failed"``,
        ``reason == "resource_limit"``).

    """
    backend = TorchBackend(ops=ops)
    lr = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        criteria=criteria,
        runner=runner if runner is not None else IdentityRunner(),
    ).optimize(
        model,
        example_input,
        ruleset=ruleset,
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        max_memory_mb=max_memory_mb,
        cost_fn=cost_fn,
        fusion_epsilon=fusion_epsilon,
        symmetry_budget=symmetry_budget,
        verify=verbose,
        verbose=verbose,
    )
    return cast(torch.nn.Module, lr.module), lr.stats


def optimize_compositional(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple,
    *,
    block_pred: Callable[[torch.nn.Module, str, torch.nn.Module], bool]
    | None = None,
    cost_fn: CostFn | None = None,
    ruleset: str = "all",
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verify_tol: float = 1e-4,
    ops: OpTable | None = None,
    max_cross_pairs: int = 8,
    source: Source | None = None,
    sink: Sink | None = None,
    composer: Composer | None = None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Optimize a stacked/multi-block model one block at a time.

    A compatibility wrapper: resolves the historical
    ``source``/``sink``/``ops`` defaults — torch IS the default here —
    then runs ``Optimizer(...).optimize(..., strategy=Compositional(...))``.
    The pipeline is the backend-neutral
    :func:`catopt_optimize.optimize._optimize_compositional` driven
    through the :class:`~catopt_core.ports.Composer` port; the
    signature is unchanged plus the (additive) ``source``/``sink``/
    ``composer`` ports.

    Returns ``(recomposed_module, stats)`` where ``stats["blocks"]`` maps
    each block's dotted name to ``{"status", "stats", "param_report",
    "time_s", ...}`` and ``stats["param_report"]`` aggregates the
    per-block parameter diffs (eliminated/derived names are prefixed by
    block name for auditability).  ``stats["cross_pairs"]`` maps each
    attempted pair ``"a+b"`` to ``{"status", "boundary", ...}`` —
    ``"grafted"`` / ``"declined"`` / ``"skipped"``.  ``stats["shared_
    params"]`` is True when the recomposed module shares the original's
    tensor storage (the normal path); ``stats["in_place"]`` True means
    cloning failed and the input module was returned unmodified.
    """
    backend = TorchBackend(ops=ops)
    lr = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        composer=composer,
    ).optimize(
        model,
        example_input,
        strategy=Compositional(
            block_pred=block_pred,
            verify_tol=verify_tol,
            max_cross_pairs=max_cross_pairs,
        ),
        cost_fn=cost_fn,
        ruleset=ruleset,
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        max_memory_mb=max_memory_mb,
        verbose=verbose,
    )
    return cast(torch.nn.Module, lr.module), lr.stats


def optimize_model_autotuned(
    model: torch.nn.Module,
    example_input: Any,  # tensor or positional-args tuple
    *,
    candidates: Iterable[str | tuple[str, CandidateBuilder]] = (
        "generic",
        "batched",
        "torch_compile",
    ),
    budget_s: float | None = None,
    n_calls: int = 30,
    warmup: int = 5,
    rtol: float = 1e-4,
    atol: float | None = None,
    source: Source | None = None,
    sink: Sink | None = None,
    meter: Meter | None = None,
    profile: Any = None,
    verbose: bool = False,
    **optimize_kwargs: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Optimize ``model``, then autotune over lowering paths.

    A compatibility wrapper: resolves the historical
    ``source``/``sink``/``ops``/``meter`` defaults — torch IS the
    default here — then runs ``Optimizer(...).optimize(...,
    strategy=Autotuned(..., builders=TORCH_BUILDERS))``;
    :func:`catopt_optimize.autotune._autotuned_impl` is the pipeline.
    Retires in 0008.

    Steps: run :func:`optimize_model` once (uncompiled — compilation
    and capture are candidates, not presets) for the extracted term
    and the pipeline's own lowered module; re-lower that term through
    each requested candidate (the backend's executor names plus
    :data:`TORCH_BUILDERS`); ``sink.verify`` each candidate against
    ``model`` on ``example_input`` (a candidate is never timed
    unverified); time each survivor through the ``meter`` port;
    return the measured-fastest.

    Returns ``(module, stats)`` — ``stats`` is the
    ``optimize_model`` stats dict plus ``stats["autotune"]``:
    ``winner``, ``winner_median_s``, per-candidate records,
    ``fallback``, ``predicted_*``/``measured_ns``/``profile``
    (with ``profile=``), ``search_s``, ``elapsed_s``.
    """
    backend = TorchBackend(ops=optimize_kwargs.pop("ops", None))
    lr = Optimizer(
        backend=backend,
        source=source,
        sink=sink,
        meter=meter,
    ).optimize(
        model,
        example_input,
        strategy=Autotuned(
            candidates,
            budget_s=budget_s,
            n_calls=n_calls,
            warmup=warmup,
            rtol=rtol,
            atol=atol,
            profile=profile,
            verbose=verbose,
            builders=TORCH_BUILDERS,
        ),
        **optimize_kwargs,
    )
    return cast(torch.nn.Module, lr.module), lr.stats


def save_optimized_weights(
    optimized_module: torch.nn.Module, path: str
) -> None:
    """Emit the optimized weights file.

    Only the parameters the certified form actually needs (folded
    derived tensors included).
    """
    torch.save(optimized_module.state_dict(), path)


# ---------------------------------------------------------------------------
#  Torch candidate builders — the torch-side lowering paths
# ---------------------------------------------------------------------------
#
# The built-in *neutral* candidates (``"generic"`` / ``"batched"`` /
# ``"eager"`` plus every name in ``sink.executors``) live in
# ``catopt_optimize.autotune.CANDIDATE_BUILDERS``.  The entries below
# are the torch-coupled paths — compilation and CUDA-graph capture —
# which the wrapper supplies through ``Autotuned(builders=…)``.


def _build_compiled(ctx: AutotuneContext) -> Any:
    """``torch.compile`` over the routed executor.

    The ``optimize_model(runner=TorchCompileRunner())`` delivery.

    Always a FRESH module: ``torch.compile`` rewrites the module's
    ``forward`` attribute (dynamo dispatch), so compiling
    ``ctx.delivered`` would contaminate the ``batched`` candidate —
    they are the same object when the pipeline routed there.
    """
    if ctx.ir is None:
        raise CandidateUnavailableError(
            "delivered module did not expose its extracted IR"
        )
    return torch.compile(
        _lower_extracted(ctx.term, ctx.ir, ctx.param_values, ctx.sink)
    )


def _build_compiled_generic(ctx: AutotuneContext) -> Any:
    """``torch.compile`` over a FRESH serial ``IRModule``.

    Fresh for the same ``forward``-mutation reason as ``compiled``.
    """
    if ctx.ir is None:
        raise CandidateUnavailableError(
            "delivered module did not expose its extracted IR"
        )
    return torch.compile(
        cast(
            Any,
            ctx.sink.lower(ctx.ir, ctx.param_values),
        )
    )


def _capture_routed(  # pragma: no cover — CUDA-only body
    ctx: AutotuneContext,
) -> Any:
    """Fresh routed executor captured into a CUDA graph.

    Fresh because ``capture_cuda_graph`` mutates the module —
    capturing ``ctx.delivered`` would silently upgrade the
    ``batched`` candidate too.
    """
    if ctx.ir is None:
        raise CandidateUnavailableError(
            "delivered module did not expose its extracted IR"
        )
    mod = _lower_extracted(ctx.term, ctx.ir, ctx.param_values, ctx.sink)
    capture = getattr(mod, "capture_cuda_graph", None)
    if capture is None:
        raise CandidateUnavailableError(
            "routed executor has no capture_cuda_graph"
        )
    args = (
        ctx.example_input
        if isinstance(ctx.example_input, tuple)
        else (ctx.example_input,)
    )
    capture(*args)
    return mod


def _input_is_cuda(example_input: Any) -> bool:
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    return any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)


def _build_cuda_graph(ctx: AutotuneContext) -> Any:
    """Build the ``cuda_graph`` candidate.

    ``optimize_model(runner=CudaGraphRunner())`` semantics on a
    fresh module (see :func:`_capture_routed`).
    """
    if not _input_is_cuda(ctx.example_input):
        raise CandidateUnavailableError(
            "cuda_graph needs a CUDA example input"
        )
    return _capture_routed(ctx)  # pragma: no cover — CUDA-only


#: The torch-side candidate builders — compiled and CUDA-graph
#: lowering paths (the neutral names live in the orchestrator's
#: ``CANDIDATE_BUILDERS``; the wrapper passes this map as
#: ``Autotuned(builders=…)``).
TORCH_BUILDERS: dict[str, CandidateBuilder] = {
    "torch_compile": _build_compiled,
    "torch_compile_generic": _build_compiled_generic,
    "cuda_graph": _build_cuda_graph,
}
