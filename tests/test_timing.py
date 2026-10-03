"""Tests for :mod:`catopt_core.timing` — the one timing contract.

Every measurement path (the ``Meter`` port, ``calibrate``'s probes,
the bench ``Runner``) reduces its samples through these three
functions, so the semantics are pinned here: the warmup/call-count
clamp and the median/IQR reduction, with the IQR exactly ``0.0``
below :data:`IQR_MIN_SAMPLES`.
"""

from __future__ import annotations

import statistics

import pytest
from catopt_core.timing import (
    IQR_MIN_SAMPLES,
    iqr,
    median,
    warmup_calls,
)


# ---------------------------------------------------------------------------
#  warmup / n_calls contract
# ---------------------------------------------------------------------------


def test_warmup_calls_passes_valid_counts():
    assert warmup_calls(5, 30) == (5, 30)
    assert warmup_calls(0, 1) == (0, 1)


def test_warmup_calls_clamps_negative_and_zero():
    # rule 1: a negative warmup clamps to zero
    assert warmup_calls(-5, 10) == (0, 10)
    # rule 2: fewer than one call clamps to one
    assert warmup_calls(3, 0) == (3, 1)
    assert warmup_calls(3, -1) == (3, 1)


# ---------------------------------------------------------------------------
#  median
# ---------------------------------------------------------------------------


def test_median_matches_statistics():
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert median([7.0]) == 7.0


def test_median_needs_a_sample():
    with pytest.raises(statistics.StatisticsError):
        median([])


# ---------------------------------------------------------------------------
#  IQR
# ---------------------------------------------------------------------------


def test_iqr_min_samples_is_four():
    assert IQR_MIN_SAMPLES == 4


@pytest.mark.parametrize(
    "samples", [[], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0]]
)
def test_iqr_zero_below_min_samples(samples):
    """Fewer than four samples → the quartiles are undefined → 0.0."""
    assert iqr(samples) == 0.0


def test_iqr_is_the_quartile_spread_at_four():
    # exclusive quantiles: q1=1.25, q3=3.75 over [1, 2, 3, 4]
    assert iqr([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_iqr_is_the_quartile_spread_above_four():
    # exclusive quantiles: q1=2.25, q3=6.75 over [1..8]
    assert iqr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) == (
        pytest.approx(4.5)
    )


def test_iqr_agrees_with_statistics_quantiles():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    q1, _, q3 = statistics.quantiles(samples, n=4)
    assert iqr(samples) == pytest.approx(q3 - q1)
