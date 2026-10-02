"""Analytical performance model — features to predicted runtime.

The first :class:`~catopt_core.ports.PerformanceModel` (ADR 0003,
plan 0016 stage 8): a roofline prediction from static program
``features`` and a target's peak throughput / bandwidth.  It *ranks*;
it never prunes the semantic space.

Analytical first, learned later: a learned model is one more
conforming value behind the same port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AnalyticalPerformanceModel"]

_TERA = 1e12
_GIGA = 1e9
_MICRO = 1e-6


@dataclass(frozen=True)
class AnalyticalPerformanceModel:
    """Roofline prediction: ``max(compute, memory) + launch overhead``.

    ``tflops`` / ``gbps`` / ``launch_us`` default to modest CPU-ish
    numbers; pass a ``TargetProfile``-like ``hardware`` to
    :meth:`predict` to override them per target.  Missing attributes
    fall back to the model's own defaults.
    """

    name: str = "analytical"
    tflops: float = 1.0
    gbps: float = 100.0
    launch_us: float = 5.0

    def predict(self, features: Any, hardware: Any = None) -> float:
        """Return the predicted runtime (seconds) of ``features``."""
        tflops, gbps, launch_us = self.tflops, self.gbps, self.launch_us
        if hardware is not None:
            tflops = float(getattr(hardware, "tflops", tflops))
            gbps = float(getattr(hardware, "gbps", gbps))
            launch_us = float(getattr(hardware, "launch_us", launch_us))
        compute_s = features.flops / (tflops * _TERA)
        mem_s = (features.bytes_read + features.bytes_written) / (
            gbps * _GIGA
        )
        overhead_s = launch_us * _MICRO * features.operations
        return max(compute_s, mem_s) + overhead_s
