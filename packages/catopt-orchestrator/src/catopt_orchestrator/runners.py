"""Delivery runners — the backend-neutral half (plan 0007).

A runner decides HOW the lowered executor is executed, applied once
to the routed module after ``sink``/executor dispatch and before
``sink.verify``.  The :class:`~catopt_core.ports.Runner` protocol
lives in core (promoted in plan 0007) and is re-exported here; the
backend-neutral members — :class:`IdentityRunner`,
:class:`ChainedRunner`, :func:`runner_candidate` — live here; the
torch-coupled :class:`TorchCompileRunner` lives in
``catopt_torch.runners`` and the CUDA-device-coupled
:class:`CudaGraphRunner` in ``catopt_cuda``.  Both resolve through
module-level ``__getattr__`` so the historical
``catopt_optimize.runners`` / ``catopt.runners`` attribute paths keep
working on a torch install.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any

from catopt_core.ports import Runner

__all__ = [
    "ChainedRunner",
    "IdentityRunner",
    "Runner",
    "runner_candidate",
]


class IdentityRunner:
    """No transform — deliver the module as lowered."""

    name = "identity"

    def apply(
        self, module: Any, example_input: Any, stats: dict[str, Any]
    ) -> Any:
        """Return *module* unchanged."""
        return module


class ChainedRunner:
    """Compose runners left-to-right — each sees the previous output.

    ``ChainedRunner([TorchCompileRunner(), CudaGraphRunner()])`` compiles
    first, then captures — and because ``CudaGraphRunner``
    defers to a successful compile, the "compiled wins" precedence
    is preserved inside the chain itself.

    ``stats["runner"]`` records :attr:`names` (a list) rather than
    the joined :attr:`name`.
    """

    def __init__(self, runners: Iterable[Runner]) -> None:
        """Store the ordered *runners*."""
        self.runners = list(runners)

    @property
    def name(self) -> str:
        """The ``+``-joined member names."""
        return "+".join(r.name for r in self.runners) or "identity"

    @property
    def names(self) -> list[str]:
        """The ordered member names."""
        return [r.name for r in self.runners]

    @property
    def delivers_compiled(self) -> bool:
        """True when any member marks a compiled delivery.

        The fusion-region pricing propagates through the chain.
        """
        return any(
            getattr(r, "delivers_compiled", False) for r in self.runners
        )

    def apply(
        self,
        module: Any,
        example_input: Any,
        stats: dict[str, Any],
    ) -> Any:
        """Apply each member runner in order."""
        for r in self.runners:
            module = r.apply(module, example_input, stats)
        return module


def runner_candidate(runner: Runner) -> Callable[[Any], Any]:
    """Adapt a :class:`Runner` into an autotune ``CandidateBuilder``.

    The hook that makes runners usable inside
    :func:`catopt_optimize.autotune.optimize_model_autotuned` without
    touching its machinery: ``candidates=[("my_runner",
    runner_candidate(MyRunner()))]``.  The builder applies the
    runner to a FRESH routed executor — the same freshness rule the
    ``torch_compile`` candidate documents, since ``torch.compile`` and
    ``capture_cuda_graph`` mutate the module they wrap.  The
    runner's stats writes go to a scratch dict; per-candidate
    outcomes are recorded under ``stats["autotune"]["candidates"]``
    as usual.
    """

    def build(ctx: Any) -> Any:
        from catopt_optimize.autotune import CandidateUnavailableError
        from catopt_optimize.optimize import _lower_extracted

        if ctx.ir is None:
            raise CandidateUnavailableError(
                "delivered module did not expose its extracted IR"
            )
        mod = _lower_extracted(
            ctx.term, ctx.ir, ctx.param_values, ctx.sink
        )
        return runner.apply(mod, ctx.example_input, {})

    return build


# ---------------------------------------------------------------------------
# Compatibility delegation — the backend-coupled runners (lazy)
# ---------------------------------------------------------------------------
#
# ``TorchCompileRunner`` (torch.compile) lives in
# ``catopt_torch.runners``; ``CudaGraphRunner`` (CUDA capture) in
# ``catopt_cuda``.  Both resolve lazily: importing this module never
# loads a backend, but the historical attribute paths keep working on
# an install that has them.

_DELEGATED = {
    "TorchCompileRunner": "catopt_torch.runners",
    "CudaGraphRunner": "catopt_cuda",
}


def __getattr__(name: str) -> Any:
    """Resolve the backend-coupled runner names lazily."""
    mod = _DELEGATED.get(name)
    if mod is not None:
        return getattr(importlib.import_module(mod), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
