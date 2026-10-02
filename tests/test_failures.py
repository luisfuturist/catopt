"""Contract tests for :mod:`catopt_core.failures`.

Plan 0016 stage 3: execution/measurement failures are *classified*
(OOM / timeout / kernel / nonfinite / unavailable / unknown), not
crashes.  These tests pin the total taxonomy — every exception lands
in exactly one :class:`FailureClass` — plus the process-control
re-raise and the measured-result nonfinite check.
"""

from __future__ import annotations

import math

import pytest
from catopt_core.failures import FailureClass, classify, is_nonfinite

# ---------------------------------------------------------------------------
#  The taxonomy — membership and str-Enum identity
# ---------------------------------------------------------------------------


def test_failure_class_members():
    """The seven buckets are present with their string values."""
    assert [c.name for c in FailureClass] == [
        "OK",
        "OOM",
        "TIMEOUT",
        "KERNEL",
        "NONFINITE",
        "UNAVAILABLE",
        "UNKNOWN",
    ]
    assert FailureClass.OK.value == "ok"
    assert FailureClass.OOM.value == "oom"
    assert FailureClass.TIMEOUT.value == "timeout"
    assert FailureClass.KERNEL.value == "kernel"
    assert FailureClass.NONFINITE.value == "nonfinite"
    assert FailureClass.UNAVAILABLE.value == "unavailable"
    assert FailureClass.UNKNOWN.value == "unknown"


def test_failure_class_is_str_enum():
    """Members compare equal to their plain string values."""
    assert isinstance(FailureClass.OOM, str)
    assert FailureClass.OOM == "oom"
    assert classify(RuntimeError("plain failure")) == "unknown"


# ---------------------------------------------------------------------------
#  classify — process-control signals are re-raised, never classified
# ---------------------------------------------------------------------------


def test_classify_reraise_keyboard_interrupt():
    """A KeyboardInterrupt is re-raised, not bucketed."""
    with pytest.raises(KeyboardInterrupt):
        classify(KeyboardInterrupt())


def test_classify_reraise_system_exit():
    """A SystemExit is re-raised, not bucketed."""
    with pytest.raises(SystemExit):
        classify(SystemExit())


# ---------------------------------------------------------------------------
#  classify — OOM by type and by message
# ---------------------------------------------------------------------------


def test_classify_memory_error():
    """A MemoryError classifies to OOM regardless of message."""
    assert classify(MemoryError()) is FailureClass.OOM


def test_classify_oom_by_message():
    """A torch-style 'out of memory' message classifies to OOM."""
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    assert classify(exc) is FailureClass.OOM


def test_classify_oom_by_oom_token():
    """A bare 'OOM' token (case-insensitive) classifies to OOM."""
    assert classify(RuntimeError("OOM")) is FailureClass.OOM


# ---------------------------------------------------------------------------
#  classify — timeout by type and by message
# ---------------------------------------------------------------------------


def test_classify_timeout_error():
    """A TimeoutError classifies to TIMEOUT by type."""
    assert classify(TimeoutError()) is FailureClass.TIMEOUT


def test_classify_timeout_by_message():
    """'timed out' / 'timeout' messages classify to TIMEOUT."""
    assert (
        classify(RuntimeError("op timed out")) is FailureClass.TIMEOUT
    )
    assert (
        classify(RuntimeError("timeout exceeded"))
        is FailureClass.TIMEOUT
    )


# ---------------------------------------------------------------------------
#  classify — device-unavailable outranks kernel
# ---------------------------------------------------------------------------


def test_classify_unavailable_by_message():
    """A 'device' message naming it unavailable is UNAVAILABLE."""
    assert (
        classify(RuntimeError("device not available"))
        is FailureClass.UNAVAILABLE
    )
    assert (
        classify(RuntimeError("device is unavailable"))
        is FailureClass.UNAVAILABLE
    )


def test_classify_unavailable_no_cuda():
    """'no CUDA device' reads as a feasibility gap, not a kernel."""
    assert (
        classify(RuntimeError("no CUDA device present"))
        is FailureClass.UNAVAILABLE
    )


# ---------------------------------------------------------------------------
#  classify — kernel faults
# ---------------------------------------------------------------------------


def test_classify_kernel_by_message():
    """CUDA / device-side / kernel messages classify to KERNEL."""
    assert (
        classify(RuntimeError("CUDA driver error"))
        is FailureClass.KERNEL
    )
    assert (
        classify(RuntimeError("device-side assert triggered"))
        is FailureClass.KERNEL
    )
    assert (
        classify(RuntimeError("kernel launch failed"))
        is FailureClass.KERNEL
    )


# ---------------------------------------------------------------------------
#  classify — fallthrough and the 'device' guard's false arm
# ---------------------------------------------------------------------------


def test_classify_device_without_availability_is_unknown():
    """A 'device' message with no availability/kernel cue is UNKNOWN."""
    assert (
        classify(RuntimeError("device memory report"))
        is FailureClass.UNKNOWN
    )


def test_classify_unknown_fallthrough():
    """Anything unrecognised classifies to UNKNOWN."""
    assert classify(ValueError("bad shape")) is FailureClass.UNKNOWN
    assert classify(RuntimeError("")) is FailureClass.UNKNOWN


# ---------------------------------------------------------------------------
#  is_nonfinite — the measured-result check
# ---------------------------------------------------------------------------


def test_is_nonfinite_true():
    """NaN and ±inf are nonfinite."""
    assert is_nonfinite(float("nan"))
    assert is_nonfinite(float("inf"))
    assert is_nonfinite(float("-inf"))


def test_is_nonfinite_false():
    """Finite values (including zero) are finite."""
    assert not is_nonfinite(0.0)
    assert not is_nonfinite(-0.0)
    assert not is_nonfinite(1.5)
    assert not is_nonfinite(math.pi)
