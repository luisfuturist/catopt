"""Torch delivery runners — the torch-coupled half of the Runner seam.

The delivery runners split by backend (plan 0007): the backend-neutral
:class:`Runner` protocol lives in :mod:`catopt_core.ports`;
:class:`IdentityRunner` / :class:`ChainedRunner` /
:func:`runner_candidate` live in the orchestrator
(:mod:`catopt_orchestrator.runners`); the torch-coupled
:class:`TorchCompileRunner` lives here; and the CUDA-device-coupled
:class:`CudaGraphRunner` lives in :mod:`catopt_cuda`.

This module re-exports the neutral runner surface so the historical
``catopt_orchestrator.runners`` path (and ``from catopt_torch.runners
import …``) resolves it.  :class:`CudaGraphRunner` is deliberately NOT
re-exported: it lives in :mod:`catopt_cuda`, and importing it here
would make ``catopt_torch`` depend on ``catopt_cuda`` (which in turn
depends on ``catopt_torch``), a cycle.  Import it from its home::

    from catopt_cuda import CudaGraphRunner
"""

from __future__ import annotations

from typing import Any, cast

import torch
from catopt_core.ports import Runner
from catopt_orchestrator.runners import (
    ChainedRunner,
    IdentityRunner,
    runner_candidate,
)

__all__ = [
    "ChainedRunner",
    "IdentityRunner",
    "Runner",
    "TorchCompileRunner",
    "runner_candidate",
]


class TorchCompileRunner:
    """Wrap the delivered executor in ``torch.compile``.

    Compile failures surface at first call — the module is invoked
    once here so a broken backend keeps the uncompiled executor
    rather than shipping a wrapper that fails later.
    ``stats["compiled"]`` records which ran.
    """

    name = "torch_compile"
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
