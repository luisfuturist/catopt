"""Tests for :mod:`catopt_torch.meter` and the ``TimingResult``
provenance contract (plan 0016 stage 3).

The torch ``Meter`` port measures a runnable's forward, and — since
stage 3 — reports *provenance* (``device`` / ``warmup``) and a
*classified* outcome: a raised call, or an expired ``timeout_s``
budget, lands in ``TimingResult.failure`` as a
:class:`~catopt_core.failures.FailureClass` with ``median_s`` set to
``NaN`` rather than escaping as a bare traceback.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from catopt_core.failures import FailureClass, is_nonfinite
from catopt_core.ports import Meter, TimingResult
from catopt_torch.meter import (
    TorchMeter,
    _input_args,
    _input_device,
    _input_is_cuda,
)


# ---------------------------------------------------------------------------
#  TimingResult — optional provenance with backward-compatible defaults
# ---------------------------------------------------------------------------


def test_timing_result_defaults():
    """A bare three-field measurement carries no provenance."""
    r = TimingResult(median_s=0.5, iqr_s=0.1, n_calls=10)
    assert r.device is None
    assert r.warmup == 0
    assert r.failure is FailureClass.OK


def test_timing_result_is_frozen():
    r = TimingResult(0.5, 0.1, 10)
    with pytest.raises(Exception):
        r.device = "cpu"


def test_torch_meter_conforms_to_meter_port():
    assert isinstance(TorchMeter(), Meter)


# ---------------------------------------------------------------------------
#  Input-provenance helpers
# ---------------------------------------------------------------------------


def test_input_args_wraps_single_and_passes_tuple():
    x = torch.zeros(2)
    assert _input_args(x) == (x,)
    assert _input_args((x, x)) == (x, x)


def test_input_device_first_tensor():
    x = torch.zeros(2)
    # a leading non-tensor operand is skipped
    assert _input_device(("scalar", x)) == "cpu"
    assert _input_device(x) == "cpu"


def test_input_device_none_without_tensor():
    assert _input_device("no-tensor") is None
    assert _input_device((1, 2.0)) is None


def test_input_is_cuda_false_on_cpu():
    assert _input_is_cuda(torch.zeros(2)) is False
    assert _input_is_cuda("no-tensor") is False


# ---------------------------------------------------------------------------
#  time — success path: provenance + OK
# ---------------------------------------------------------------------------


def test_time_success_records_provenance():
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    r = TorchMeter().time(mod, x, warmup=2, n_calls=6)
    assert r.failure is FailureClass.OK
    assert r.device == "cpu"
    assert r.warmup == 2
    assert r.n_calls == 6
    assert r.median_s > 0.0
    # >= 4 samples → a real IQR is computed
    assert r.iqr_s >= 0.0


def test_time_few_calls_zero_iqr():
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    r = TorchMeter().time(mod, x, warmup=1, n_calls=2)
    assert r.iqr_s == 0.0
    assert r.n_calls == 2


def test_time_negative_warmup_clamped():
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    r = TorchMeter().time(mod, x, warmup=-5, n_calls=1)
    assert r.warmup == 0
    assert r.failure is FailureClass.OK


def test_time_non_tensor_input_has_no_device():
    r = TorchMeter().time(lambda *a: None, "not-a-tensor", n_calls=1)
    assert r.device is None
    assert r.failure is FailureClass.OK


# ---------------------------------------------------------------------------
#  time — failure path: classified, NaN median
# ---------------------------------------------------------------------------


def test_time_classifies_raised_call():
    def boom(*a):
        raise RuntimeError("bad shape")

    r = TorchMeter().time(boom, torch.zeros(2), warmup=1, n_calls=3)
    assert r.failure is FailureClass.UNKNOWN
    assert is_nonfinite(r.median_s)
    assert r.n_calls == 0  # nothing completed
    assert r.device == "cpu"


def test_time_classifies_oom_message():
    def oom(*a):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    r = TorchMeter().time(oom, torch.zeros(2), n_calls=1)
    assert r.failure is FailureClass.OOM
    assert is_nonfinite(r.median_s)


def test_time_failure_after_some_calls_counts_completed():
    state = {"n": 0}

    def flaky(*a):
        state["n"] += 1
        if state["n"] > 2:  # two warmup calls pass, timing explodes
            raise RuntimeError("device-side assert triggered")

    r = TorchMeter().time(flaky, torch.zeros(2), warmup=2, n_calls=4)
    assert r.failure is FailureClass.KERNEL
    assert r.n_calls == 0
    assert math.isnan(r.median_s)


def test_time_timeout_budget_classifies_timeout():
    """A wall-clock budget already in the past expires on the first
    timed call — deterministically, without a slow loop."""
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    r = TorchMeter().time(mod, x, n_calls=100, timeout_s=-1.0)
    assert r.failure is FailureClass.TIMEOUT
    assert is_nonfinite(r.median_s)
    # one timed call completed before the budget check fired
    assert r.n_calls == 1
