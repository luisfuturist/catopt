"""Executor mixins — shared machinery of the batched adapters.

``base.py`` — :class:`BatchedExecutorBase`, the CUDA-graph capture /
eval-plumbing mixin shared by ``BatchedScanModule`` (scan_lower),
``BatchedOMModule`` (om_lower), and ``BatchedOmdModule`` (omd_lower),
plus the plan-time level-schedule skeleton (:func:`level_schedule`,
:func:`slot_gathers`) those lowerers' plan builders share.
"""

from catopt_torch.executors.base import (
    BatchedExecutorBase,
    level_schedule,
    slot_gathers,
)

__all__ = ["BatchedExecutorBase", "level_schedule", "slot_gathers"]
