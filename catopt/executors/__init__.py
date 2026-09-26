"""Executor mixins — shared machinery of the batched adapters.

``base.py`` — :class:`BatchedExecutorBase`, the CUDA-graph capture /
eval-plumbing mixin shared by ``BatchedScanModule`` (scan_lower),
``BatchedOMModule`` (om_lower), and ``BatchedOmdModule`` (omd_lower).
"""

from catopt.executors.base import BatchedExecutorBase

__all__ = ["BatchedExecutorBase"]
