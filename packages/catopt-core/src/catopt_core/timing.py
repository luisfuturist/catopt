"""The one timing contract — warmup, call count, median and IQR.

Every measurement path in catopt reduces a list of wall-clock
samples to the same two numbers under the same rules.  This module
*is* that contract, so the paths share one definition instead of each
inventing its own:

* :class:`catopt_torch.meter.TorchMeter` — the production
  :class:`~catopt_core.ports.Meter` port;
* :mod:`catopt_torch.calibrate` — its micro-benchmark probes;
* :class:`bench.benchkit.runner.Runner` — the benchmark harness.

The contract
------------

1. **Warmup.**  A measurement runs ``warmup`` untimed calls first —
   autotune, cache fill, lazy init — before the timed block.  A
   negative ``warmup`` clamps to zero; warmup calls never enter the
   samples.
2. **Call count.**  The timed block runs ``n_calls`` calls; a value
   below one clamps to one, because a measurement with no call
   measures nothing.  ``n_calls`` is the count of *intended* calls,
   not the count that completed.
3. **Median.**  The reported statistic is the median of the timed
   samples — robust to the GC pause or scheduler blip a mean would
   absorb.
4. **IQR.**  The reported spread is the interquartile range
   (``q3 - q1``) of the samples.  Fewer than
   :data:`IQR_MIN_SAMPLES` samples leave the quartiles ill-defined,
   so :func:`iqr` reports exactly ``0.0`` there.

Device synchronisation (``torch.cuda.synchronize`` on CUDA inputs),
the clock, and the timed loop itself stay the adapter's business:
this contract is pure Python and names no tensor library.
"""

from __future__ import annotations

import statistics

__all__ = [
    "IQR_MIN_SAMPLES",
    "iqr",
    "median",
    "warmup_calls",
]

#: Fewest samples for which the IQR is defined.  Below this the
#: quartiles interpolate over too few points, so :func:`iqr` reports
#: ``0.0`` rather than a number the data cannot support.
IQR_MIN_SAMPLES = 4


def warmup_calls(warmup: int, n_calls: int) -> tuple[int, int]:
    """Resolve the contract's ``(warmup, calls)`` counts.

    ``warmup`` clamps to zero and ``n_calls`` to at least one
    (contract rules 1 and 2), so a caller may pass unvalidated counts
    and still run a defined measurement.
    """
    return max(warmup, 0), max(n_calls, 1)


def median(samples: list[float]) -> float:
    """Return the median of ``samples`` — contract rule 3.

    The robust centre every path reports; ``samples`` must hold at
    least one value.
    """
    return statistics.median(samples)


def iqr(samples: list[float]) -> float:
    """Return the interquartile range of ``samples`` — contract rule 4.

    ``q3 - q1`` by :func:`statistics.quantiles` with ``n=4``; exactly
    ``0.0`` when ``samples`` holds fewer than
    :data:`IQR_MIN_SAMPLES` entries, where the quartiles are not
    defined.
    """
    if len(samples) < IQR_MIN_SAMPLES:
        return 0.0
    q1, _, q3 = statistics.quantiles(samples, n=4)
    return q3 - q1
