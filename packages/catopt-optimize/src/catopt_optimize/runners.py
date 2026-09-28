"""Delivery runners — how the optimized module is executed.

The pipeline's phase-3 lowering picks WHICH executor runs the
extracted term (``_lower_extracted``: level-batched carrier or
generic ``IRModule``).  A :class:`Runner` is the composable,
first-class object deciding what happens to that executor before it
is returned to the caller — wrap it in ``torch.compile``, capture it
into a CUDA graph, compose several transforms left-to-right, or ship
it untouched::

    optimize_model(m, x, runner=CompiledRunner())
    optimize_model(
        m, x,
        runner=ChainedRunner([CompiledRunner(), CudaGraphRunner()]),
    )

The ``runner`` parameter of :func:`optimize_model` is the only
execution control — ``None`` ships the executor as lowered; pass a
:class:`ChainedRunner` to compose deliveries (put
:class:`CompiledRunner` before :class:`CudaGraphRunner` so the
compiled delivery keeps precedence and capture is skipped, the
ordering the retired ``compile`` / ``cuda_graph`` flags had).

A runner is duck-typed — any object with a ``name`` and
``apply(module, example_input, stats) -> module`` conforms to the
:class:`Runner` protocol; ``stats`` is the live stats dict, so a
runner records what it did (``stats["compiled"]``,
``stats["cuda_graph"]``) and may read what earlier runners did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast, runtime_checkable

import torch

__all__ = [
    "ChainedRunner",
    "CompiledRunner",
    "CudaGraphRunner",
    "GenericRunner",
    "Runner",
    "runner_candidate",
]


@runtime_checkable
class Runner(Protocol):
    """The delivery-stage contract.

    ``apply`` receives the module the lowering produced plus the
    pipeline's ``example_input`` (tensor or positional-args tuple)
    and returns the module to deliver — possibly a wrapped or
    mutated version of the input.  ``stats`` is the same dict
    :func:`optimize_model` returns, so runners record their outcome
    there (and earlier runners in a chain can gate later ones — the
    ``stats["compiled"]`` → skip-capture precedence lives in
    :class:`CudaGraphRunner` itself).
    """

    name: str

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Return the module to deliver, recording into *stats*."""
        ...


class GenericRunner:
    """Identity runner — deliver the executor as lowered."""

    name = "generic"

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Return *module* unchanged — the executor as lowered."""
        return module


class CompiledRunner:
    """Wrap the delivered executor in ``torch.compile``.

    Compile failures surface at first call — the module is invoked
    once here so a broken backend keeps the uncompiled executor
    rather than shipping a wrapper that fails later.
    ``stats["compiled"]`` records which ran.
    """

    name = "compiled"
    #: Marker ``optimize_model`` reads to price the fusion-region
    #: cost model during carrier selection — a compiled delivery
    #: bills every term under ``lowering="compiled"``.
    delivers_compiled = True

    def __init__(self, **compile_kwargs: Any) -> None:
        """Record the ``torch.compile`` keyword arguments."""
        self.compile_kwargs = dict(compile_kwargs)

    def apply(
        self,
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
        """Wrap *module* in ``torch.compile``, recording the outcome."""
        try:
            compiled = torch.compile(module, **self.compile_kwargs)
            compiled(example_input)
            stats["compiled"] = True
            return cast(torch.nn.Module, compiled)
        except Exception:
            stats["compiled"] = False
            return module


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
            except Exception:  # pragma: no cover — CUDA-only
                cast(Any, module).drop_cuda_graph()
        return module


class ChainedRunner:
    """Compose runners left-to-right — each sees the previous output.

    ``ChainedRunner([CompiledRunner(), CudaGraphRunner()])`` compiles
    first, then captures — and because :class:`CudaGraphRunner`
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
        module: torch.nn.Module,
        example_input: Any,
        stats: dict[str, Any],
    ) -> torch.nn.Module:
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
    ``compiled`` candidate documents, since ``torch.compile`` and
    ``capture_cuda_graph`` mutate the module they wrap.  The
    runner's stats writes go to a scratch dict; per-candidate
    outcomes are recorded under ``stats["autotune"]["candidates"]``
    as usual.
    """

    def build(ctx: Any) -> Any:
        # Local imports: autotune imports optimize, which imports
        # this module — deferring keeps the import graph acyclic.
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
