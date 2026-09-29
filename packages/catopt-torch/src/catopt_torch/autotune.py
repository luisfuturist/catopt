"""The torch-coupled autotune candidate builders.

The built-in *neutral* candidates (``"generic"`` / ``"batched"`` /
``"eager"`` plus every name in ``sink.executors``) live in
``catopt_orchestrator.autotune.CANDIDATE_BUILDERS``.  The entries in
:data:`TORCH_BUILDERS` are the torch-coupled lowering paths —
compilation and CUDA-graph capture — which an
:class:`~catopt_orchestrator.optimize.Autotuned` strategy receives
through its ``builders`` map::

    from catopt_orchestrator import Autotuned, Optimizer
    from catopt_torch.autotune import TORCH_BUILDERS
    from catopt_torch.backend import TorchBackend

    opt = Optimizer(backend=TorchBackend())
    mod, stats = opt.optimize(
        model, x, strategy=Autotuned(builders=TORCH_BUILDERS)
    )
"""

from __future__ import annotations

from typing import Any, cast

import torch
from catopt_orchestrator.autotune import (
    AutotuneContext,
    CandidateBuilder,
    CandidateUnavailableError,
)
from catopt_orchestrator.optimize import _lower_extracted

__all__ = ["TORCH_BUILDERS"]


def _build_compiled(ctx: AutotuneContext) -> Any:
    """``torch.compile`` over the routed executor.

    The ``runner=TorchCompileRunner()`` delivery.

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

    ``runner=CudaGraphRunner()`` semantics on a fresh module (see
    :func:`_capture_routed`).
    """
    if not _input_is_cuda(ctx.example_input):
        raise CandidateUnavailableError(
            "cuda_graph needs a CUDA example input"
        )
    return _capture_routed(ctx)  # pragma: no cover — CUDA-only


#: The torch-side candidate builders — compiled and CUDA-graph
#: lowering paths (the neutral names live in the orchestrator's
#: ``CANDIDATE_BUILDERS``; an ``Autotuned`` strategy receives this
#: map as ``Autotuned(builders=…)``).
TORCH_BUILDERS: dict[str, CandidateBuilder] = {
    "torch_compile": _build_compiled,
    "torch_compile_generic": _build_compiled_generic,
    "cuda_graph": _build_cuda_graph,
}
