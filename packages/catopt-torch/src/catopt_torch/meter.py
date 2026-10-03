"""TorchMeter — the torch :class:`~catopt_core.ports.Meter` port.

The autotuned strategy's wall-clock timing, moved verbatim from
``catopt_orchestrator.autotune`` (plan 0007): ``warmup`` untimed calls
then ``n_calls`` timed forwards, median + IQR of wall seconds.
CUDA inputs end every timed call in ``torch.cuda.synchronize`` so the
measured time includes the GPU tail.

The warmup/call-count semantics and the median/IQR reduction are
:mod:`catopt_core.timing`'s one contract — the meter only owns the
device-specific timed loop.

Plan 0016 stage 3 gives the measurement provenance and a classified
outcome: the returned :class:`~catopt_core.ports.TimingResult` carries
the input ``device`` and the ``warmup`` count, and a call that raises
(or an expired ``timeout_s`` budget) is classified through
:func:`catopt_core.failures.classify` into ``failure`` — with
``median_s`` set to ``NaN`` — instead of escaping as a bare exception.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from catopt_core.failures import FailureClass, classify
from catopt_core.ports import TimingResult
from catopt_core.timing import iqr, median, warmup_calls

__all__ = ["TorchMeter"]


def _input_args(example_input: Any) -> tuple:
    """Normalise an example input into a positional-args tuple."""
    return (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )


def _input_device(example_input: Any) -> str | None:
    """Return the first tensor operand's device, or ``None``."""
    for a in _input_args(example_input):
        if isinstance(a, torch.Tensor):
            return str(a.device)
    return None


def _input_is_cuda(example_input: Any) -> bool:
    """Whether any operand is a CUDA tensor."""
    return any(
        isinstance(a, torch.Tensor) and a.is_cuda
        for a in _input_args(example_input)
    )


class TorchMeter:
    """Time a runnable's forward on the real input — the torch Meter.

    ``time(runnable, inputs, *, warmup, n_calls, timeout_s)`` runs
    ``warmup`` untimed calls first (inductor autotune, cache fill),
    then ``max(n_calls, 1)`` timed forwards; the median decides, the
    IQR reports spread.

    A call that raises is caught and classified
    (:func:`catopt_core.failures.classify`): the result carries the
    bucket in ``failure`` with ``median_s`` set to ``NaN`` — the
    companion ``is_nonfinite`` check a consumer gates on.  A
    ``timeout_s`` wall-clock budget is checked between timed calls and
    expires as :attr:`~catopt_core.failures.FailureClass.TIMEOUT` (it
    bounds a slow loop; it cannot interrupt a single blocking kernel
    call).
    """

    def time(
        self,
        runnable: Any,
        inputs: Any,
        *,
        warmup: int = 5,
        n_calls: int = 30,
        timeout_s: float | None = None,
    ) -> TimingResult:
        """Median + IQR wall seconds of one ``runnable(*inputs)`` call."""
        args = _input_args(inputs)
        is_cuda = _input_is_cuda(inputs)
        device = _input_device(inputs)
        n_warm, calls = warmup_calls(warmup, n_calls)

        def call() -> None:
            with torch.no_grad():
                runnable(*args)

        times: list[float] = []
        try:
            for _ in range(n_warm):
                call()
            if is_cuda:
                torch.cuda.synchronize()  # pragma: no cover — CUDA-only
            deadline = (
                time.perf_counter() + timeout_s
                if timeout_s is not None
                else None
            )
            for _ in range(calls):
                t0 = time.perf_counter()
                call()
                if is_cuda:
                    torch.cuda.synchronize()  # pragma: no cover
                times.append(time.perf_counter() - t0)
                if (
                    deadline is not None
                    and time.perf_counter() > deadline
                ):
                    raise TimeoutError(
                        f"measurement exceeded {timeout_s}s"
                    )
        except Exception as exc:
            return TimingResult(
                median_s=float("nan"),
                iqr_s=0.0,
                n_calls=len(times),
                device=device,
                warmup=n_warm,
                failure=classify(exc),
            )
        return TimingResult(
            median_s=median(times),
            iqr_s=iqr(times),
            n_calls=calls,
            device=device,
            warmup=n_warm,
            failure=FailureClass.OK,
        )
