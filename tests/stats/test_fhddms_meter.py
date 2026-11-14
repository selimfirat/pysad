import math

import numpy as np
import pytest

from pysad.statistics import FHDDMSMeter


def _bits(probabilities, length, seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.rand(length) < p for p in probabilities]).astype(float)


def _run_meter(meter, values):
    drifts = []
    for value in values:
        meter.update(value)
        drifts.append(meter.drift_detected)

    return drifts


def test_short_window_uses_the_paper_threshold_at_equality():
    # The short-window accuracy falls by 0.5. Its Hoeffding bound is also 0.5.
    meter = FHDDMSMeter(
        short_window_size=2,
        long_window_size=4,
        delta=math.exp(-1.0),
    )

    drifts = _run_meter(meter, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    assert drifts == [False, False, False, False, True, False]
    assert meter.get() == pytest.approx(0.0)


def test_long_window_detects_a_change_that_the_short_window_misses():
    # The long-window accuracy falls by 0.5, above its bound of about 0.318.
    # The short-window accuracy falls by the same amount, below its bound of about 0.634.
    meter = FHDDMSMeter(
        short_window_size=2,
        long_window_size=4,
        delta=0.2,
    )

    drifts = _run_meter(meter, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    assert drifts == [False, False, False, False, False, False, True]


def test_stationary_stream_has_no_drift():
    values = _bits([0.3], length=2000, seed=0)
    meter = FHDDMSMeter()

    assert not any(_run_meter(meter, values))
    assert meter.num_items == 2000
    assert meter.get() == pytest.approx(values[-100:].mean())


def test_rise_of_the_error_rate_is_detected():
    values = _bits([0.2, 0.5], length=1000, seed=0)

    drifts = _run_meter(FHDDMSMeter(), values)

    first = drifts.index(True)
    assert 1000 <= first < 1200


def test_fall_of_the_error_rate_is_not_a_drift():
    values = _bits([0.5, 0.1], length=1000, seed=0)

    assert not any(_run_meter(FHDDMSMeter(), values))


def test_update_after_a_drift_forgets_earlier_values():
    meter = FHDDMSMeter(short_window_size=2, long_window_size=4, delta=0.2)
    for value in [0.0] * 4 + [1.0, 0.0, 1.0]:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(1.0)

    assert not meter.drift_detected
    assert meter.num_items == 1
    assert meter.get() == pytest.approx(1.0)


@pytest.mark.parametrize("short_window_size", [0, -1])
def test_nonpositive_short_window_size_is_rejected(short_window_size):
    with pytest.raises(ValueError, match="short_window_size"):
        FHDDMSMeter(short_window_size=short_window_size)


@pytest.mark.parametrize(
    ("short_window_size", "long_window_size"),
    [(2, 0), (2, 2), (25, 25)],
)
def test_long_window_must_be_larger_than_the_short_window(
    short_window_size,
    long_window_size,
):
    with pytest.raises(ValueError, match="long_window_size"):
        FHDDMSMeter(
            short_window_size=short_window_size,
            long_window_size=long_window_size,
        )


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_delta_out_of_range_is_rejected(delta):
    with pytest.raises(ValueError, match="delta"):
        FHDDMSMeter(delta=delta)


def test_smallest_positive_delta_keeps_the_bounds_finite():
    meter = FHDDMSMeter(delta=np.nextafter(0.0, 1.0))

    assert math.isfinite(meter.epsilon_short)
    assert math.isfinite(meter.epsilon_long)


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = FHDDMSMeter()
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(FHDDMSMeter, window_size=10)
