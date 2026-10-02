"""Analytical performance model (:mod:`catopt_core.perf_model`)."""

import pytest
from catopt_core.features import ProgramFeatures
from catopt_core.perf_model import AnalyticalPerformanceModel
from catopt_core.ports import PerformanceModel


def _f(
    flops=0.0, read=0.0, written=0.0, operations=1
) -> ProgramFeatures:
    return ProgramFeatures(
        flops, read, written, 0.0, 1, operations, 0, 0.0, 1.0
    )


def test_model_conforms_and_names_itself():
    m = AnalyticalPerformanceModel()
    assert isinstance(m, PerformanceModel)
    assert m.name == "analytical"


def test_compute_bound_prediction():
    m = AnalyticalPerformanceModel(
        tflops=1.0, gbps=100.0, launch_us=0.0
    )
    assert m.predict(_f(flops=1e12)) == pytest.approx(1.0)


def test_memory_bound_prediction():
    m = AnalyticalPerformanceModel(
        tflops=1e9, gbps=100.0, launch_us=0.0
    )
    assert m.predict(_f(read=1e9, written=1e9)) == pytest.approx(0.02)


def test_launch_overhead_scales_with_operations():
    m = AnalyticalPerformanceModel(
        tflops=1.0, gbps=100.0, launch_us=5.0
    )
    assert m.predict(_f(operations=4)) == pytest.approx(4 * 5e-6)


def test_hardware_override():
    class _HW:
        tflops = 2.0
        gbps = 50.0
        launch_us = 1.0

    m = AnalyticalPerformanceModel()
    got = m.predict(_f(flops=2e12, operations=2), _HW())
    assert got == pytest.approx(1.0 + 2e-6)


def test_hardware_missing_attrs_fall_back():
    class _Partial:
        tflops = 2.0

    m = AnalyticalPerformanceModel(
        tflops=1.0, gbps=100.0, launch_us=0.0
    )
    assert m.predict(_f(flops=2e12), _Partial()) == pytest.approx(1.0)
