"""TorchMeter — the torch :class:`~catopt_core.ports.Meter` port.

The autotuned strategy's wall-clock timing, moved verbatim from
``catopt_optimize.autotune`` (plan 0007): ``warmup`` untimed calls
then ``n_calls`` timed forwards, median + IQR of wall seconds.
CUDA inputs end every timed call in ``torch.cuda.synchronize`` so the
measured time includes the GPU tail.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from catopt_core.ports import TimingResult

__all__ = ["TorchMeter"]


def _input_is_cuda(example_input: Any) -> bool:
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    return any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)


class TorchMeter:
    """Time a runnable's forward on the real input — the torch Meter.

    ``time(runnable, inputs, *, warmup, n_calls)`` runs ``warmup``
    untimed calls first (inductor autotune, cache fill), then
    ``max(n_calls, 1)`` timed forwards; the median decides, the IQR
    reports spread.
    """

    def time(
        self,
        runnable: Any,
        inputs: Any,
        *,
        warmup: int = 5,
        n_calls: int = 30,
    ) -> TimingResult:
        """Median + IQR wall seconds of one ``runnable(*inputs)`` call."""
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        is_cuda = _input_is_cuda(inputs)

        def call() -> None:
            with torch.no_grad():
                runnable(*args)

        for _ in range(max(warmup, 0)):
            call()
        if is_cuda:
            torch.cuda.synchronize()  # pragma: no cover — CUDA-only
        calls = max(n_calls, 1)
        times: list[float] = []
        for _ in range(calls):
            t0 = time.perf_counter()
            call()
            if is_cuda:
                torch.cuda.synchronize()  # pragma: no cover — CUDA-only
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        if len(times) >= 4:
            q1, _, q3 = statistics.quantiles(times, n=4)
            iqr = q3 - q1
        else:
            iqr = 0.0
        return TimingResult(median_s=med, iqr_s=iqr, n_calls=calls)
