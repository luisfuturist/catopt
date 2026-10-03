"""CUDA-graph delivery runner.

Moved verbatim from ``catopt_orchestrator.runners`` (plan 0007): the
capture mechanism is torch-coupled *and* CUDA-device-coupled, so it
sits in its own package — the orchestrator imports no torch and the
torch package imports no CUDA-specific machinery.

:class:`CudaGraphRunner` conforms to
:class:`catopt_core.ports.Runner`: ``apply(module, example_input,
stats) -> module``.  Only batched executors expose
``capture_cuda_graph``; generic ``IRModule``s and compiled wrappers
degrade quietly.

A capture that raises still degrades gracefully — the module drops its
partial graph and delivery continues — but the reason is no longer
swallowed (plan 0016 stage 3): the exception is classified through
:func:`catopt_core.failures.classify` and recorded under
``stats["cuda_graph_failure"]``, so an OOM capture is distinguishable
from a kernel fault or a missing device.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from catopt_core.failures import FailureClass, classify

__all__ = ["CudaGraphRunner"]


def _capture_failure_class(exc: BaseException) -> FailureClass:
    """Classify a failed CUDA-graph capture into a ``FailureClass``."""
    return classify(exc)


class CudaGraphRunner:
    """Capture the delivered executor into a CUDA graph.

    Collapses the carrier's per-level launches into one replayable
    graph (measured ~2.4x on the batched scan).  Only batched
    executors expose ``capture_cuda_graph``; generic ``IRModule``s
    and compiled wrappers degrade quietly.  Shape is baked at
    capture — mismatched calls fall back to eager internally.

    Compiled delivery wins: when ``stats["compiled"]`` is set the
    module is already a dynamo wrapper (which also fuses launches),
    so capture is skipped entirely — ``stats["cuda_graph"]`` stays
    unset, matching the historical flag behaviour.
    """

    name = "cuda_graph"

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Capture *module* into a CUDA graph when the input is CUDA."""
        if stats.get("compiled"):
            return module
        stats["cuda_graph"] = False
        ex = (
            example_input[0]
            if isinstance(example_input, (tuple, list))
            else example_input
        )
        capture = getattr(module, "capture_cuda_graph", None)
        if (
            isinstance(ex, torch.Tensor)
            and ex.is_cuda
            and capture is not None
        ):
            try:  # pragma: no cover — CUDA-only body; the
                # requires_cuda test exercises it on GPU.
                xs = (
                    tuple(example_input)
                    if isinstance(example_input, (tuple, list))
                    else (example_input,)
                )
                capture(*xs)
                stats["cuda_graph"] = True
            except Exception as exc:  # pragma: no cover — CUDA-only
                stats["cuda_graph_failure"] = _capture_failure_class(
                    exc
                )
                cast(Any, module).drop_cuda_graph()
        return module
