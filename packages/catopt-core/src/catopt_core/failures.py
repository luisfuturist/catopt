"""Failure taxonomy — classify execution/measurement failures.

ADR 0003 / plan 0016 stage 3.  An optimizer *executes* candidate
graphs on the deployment target, and those runs fail in ways that
need naming rather than a bare traceback: an out-of-memory is a
budget verdict, a timeout a wall-clock one, a kernel fault a backend
bug, a NaN a broken measurement, a missing device a feasibility one.
:class:`FailureClass` is that vocabulary and :func:`classify` maps a
raised exception onto it by exception *type* and message text alone —
no backend import, so the torch-free core stays zero-dependency.
:func:`is_nonfinite` is the companion check for a *measured* result:
a NaN or infinite latency is a broken measurement, not a fast one.

Classification is total: every exception lands in exactly one
:class:`FailureClass`, except the process-control signals
(``KeyboardInterrupt`` / ``SystemExit``), which are re-raised so an
interrupt is never swallowed as an ``UNKNOWN`` failure.  The
heuristics read ``str(exc)`` case-insensitively and are deliberately
coarse — a taxonomy, not a parser.
"""

from __future__ import annotations

import math
from enum import StrEnum

__all__ = [
    "FailureClass",
    "classify",
    "is_nonfinite",
]


class FailureClass(StrEnum):
    """One bucket an execution/measurement failure lands in.

    ``OK`` is the no-failure sentinel (a run that completed);
    :func:`classify` only ever returns a genuine failure bucket.
    """

    OK = "ok"
    OOM = "oom"
    TIMEOUT = "timeout"
    KERNEL = "kernel"
    NONFINITE = "nonfinite"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


def _msg_oom(msg: str) -> bool:
    """Whether a lowercased message reads as out-of-memory."""
    return "out of memory" in msg or "oom" in msg


def _msg_timeout(msg: str) -> bool:
    """Whether a lowercased message reads as a timeout."""
    return "timed out" in msg or "timeout" in msg


def _msg_unavailable(msg: str) -> bool:
    """Whether a lowercased message names a device as unavailable."""
    return "device" in msg and (
        "not available" in msg
        or "unavailable" in msg
        or "no cuda" in msg
    )


def _msg_kernel(msg: str) -> bool:
    """Whether a lowercased message reads as a kernel/CUDA fault."""
    return "cuda" in msg or "device-side" in msg or "kernel" in msg


def classify(exc: BaseException) -> FailureClass:
    """Classify *exc* into exactly one :class:`FailureClass`.

    Maps by exception *type* and message text only — the core imports
    no tensor library, so a ``torch`` OOM reaches here as its
    ``RuntimeError`` text.  Precedence, first match wins:

    * ``KeyboardInterrupt`` / ``SystemExit`` — re-raised, never
      classified (a process-control signal is not a failure bucket);
    * ``MemoryError``, or an ``"out of memory"`` / ``"OOM"`` message —
      :attr:`FailureClass.OOM`;
    * ``TimeoutError``, or a ``"timed out"`` / ``"timeout"`` message —
      :attr:`FailureClass.TIMEOUT`;
    * a ``"device"`` message naming it unavailable —
      :attr:`FailureClass.UNAVAILABLE`;
    * a ``"CUDA"`` / ``"device-side"`` / ``"kernel"`` message —
      :attr:`FailureClass.KERNEL`;
    * anything else — :attr:`FailureClass.UNKNOWN`.

    ``UNAVAILABLE`` outranks ``KERNEL`` so a ``"no CUDA device"``
    message reads as a feasibility gap, not a kernel fault.

    Raises:
        BaseException: *exc* itself, when it is ``KeyboardInterrupt``
            or ``SystemExit``.

    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    msg = str(exc).lower()
    if isinstance(exc, MemoryError) or _msg_oom(msg):
        return FailureClass.OOM
    if isinstance(exc, TimeoutError) or _msg_timeout(msg):
        return FailureClass.TIMEOUT
    if _msg_unavailable(msg):
        return FailureClass.UNAVAILABLE
    if _msg_kernel(msg):
        return FailureClass.KERNEL
    return FailureClass.UNKNOWN


def is_nonfinite(value: float) -> bool:
    """Return whether *value* is NaN or ±inf.

    The NONFINITE check for a *measured* result: a NaN or infinite
    latency/throughput is a broken measurement, not a fast one, so a
    consumer gating on it should record
    :attr:`FailureClass.NONFINITE` rather than trust the number.
    """
    return not math.isfinite(value)
