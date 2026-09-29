"""catopt-torch — the PyTorch integration adapters + backend.

``torch.export`` → IR (:mod:`~catopt_torch.torch_bridge`), the
executable ``IRModule`` + weight-chain folding, the batched executor
machinery (:mod:`~catopt_torch.executors`), the small model zoo
(:mod:`~catopt_torch.models`), and the typed reports + equivalence
gate (:mod:`~catopt_torch.report`).

Plan 0007 adds the backend ports — :func:`~catopt_torch.backend.TorchBackend`
(the :class:`~catopt_core.pipeline.Backend` value:
:class:`TorchSource` + :class:`TorchSink` +
:class:`~catopt_torch.composer.TorchComposer` +
:class:`~catopt_torch.meter.TorchMeter`), the torch delivery runner
(:class:`~catopt_torch.runners.TorchCompileRunner`), the torch halves
of the regime machinery (:mod:`~catopt_torch.regime`) and calibration
(:mod:`~catopt_torch.calibrate`), the production export helpers
(:mod:`~catopt_torch.export`), and the deprecated ``optimize_*``
wrappers (:mod:`~catopt_torch.api`).  All resolve lazily — importing
``catopt_torch`` pulls no pipeline machinery until the names are
touched.
"""

import importlib
from typing import Any

# Importing the concrete-eval backend registers it with catopt-core
# (:func:`catopt_core.meta.register_concrete_eval`): any import of the
# torch adapter makes core's numeric candidate check work, and core
# itself imports neither ``torch`` nor ``catopt_torch``.
from catopt_torch import meta_eval as _meta_eval  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy public surface — names resolve on first attribute access.
# ---------------------------------------------------------------------------

_DELEGATED = {
    "TorchBackend": "catopt_torch.backend",
    "TorchComposer": "catopt_torch.composer",
    "TorchMeter": "catopt_torch.meter",
    "TorchCompileRunner": "catopt_torch.runners",
    "optimize_model": "catopt_torch.api",
    "optimize_compositional": "catopt_torch.api",
    "optimize_model_autotuned": "catopt_torch.api",
    "RegimeDispatch": "catopt_torch.regime",
    "regime_dispatch": "catopt_torch.regime",
    "calibrate": "catopt_torch.calibrate",
    "TargetProfile": "catopt_torch.calibrate",
    "load_profile": "catopt_torch.calibrate",
}


def __getattr__(name: str) -> Any:
    """Resolve the adapter-surface names lazily."""
    mod = _DELEGATED.get(name)
    if mod is not None:
        return getattr(importlib.import_module(mod), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
