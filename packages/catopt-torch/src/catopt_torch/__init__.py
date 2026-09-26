"""catopt-torch — the PyTorch integration adapters.

``torch.export`` → IR (:mod:`~catopt_torch.torch_bridge`), the
executable ``IRModule`` + weight-chain folding, the batched executor
machinery (:mod:`~catopt_torch.executors`), the small model zoo
(:mod:`~catopt_torch.models`), and the typed reports + equivalence
gate (:mod:`~catopt_torch.report`).
"""

# Importing the concrete-eval backend registers it with catopt-core
# (:func:`catopt_core.meta.register_concrete_eval`): any import of the
# torch adapter makes core's numeric candidate check work, and core
# itself imports neither ``torch`` nor ``catopt_torch``.
from catopt_torch import meta_eval as _meta_eval  # noqa: F401
