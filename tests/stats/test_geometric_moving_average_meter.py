import math

import numpy as np
import pytest

from pysad.statistics import GeometricMovingAverageMeter


def test_empty_meter_returns_zero():
    assert GeometricMovingAverageMeter().get() == 0.0


def test_first_update_seeds_the_value():
    meter = GeometricMovingAverageMeter(alpha=0.5)

    assert meter.update(4.0).get() == 4.0


def test_update_follows_the_recurrence():
    meter = GeometricMovingAverageMeter(alpha=0.5)

    weight_sum = 0.0
    weighted_log_sum = 0.0
    for value in [1.0, 2.0, 4.0]:
        weight_sum = 0.5 + 0.5 * weight_sum
        weighted_log_sum = 0.5 * math.log(value) + 0.5 * weighted_log_sum
        expected = math.exp(weighted_log_sum / weight_sum)
        assert meter.update(value).get() == pytest.approx(expected)


def test_matches_a_manual_recurrence_on_a_random_stream():
    from pysad.utils import fix_seed

    fix_seed(61)
    values = np.random.rand(200) + 0.1
    alpha = 0.3
    meter = GeometricMovingAverageMeter(alpha=alpha)

    weight_sum = 0.0
    weighted_log_sum = 0.0
    for value in values:
        weight_sum = alpha + (1.0 - alpha) * weight_sum
        weighted_log_sum = alpha * math.log(value) + (1.0 - alpha) * weighted_log_sum
        expected = math.exp(weighted_log_sum / weight_sum)
        meter.update(float(value))
        assert meter.get() == pytest.approx(expected)


def test_constant_stream_has_no_drift():
    meter = GeometricMovingAverageMeter(alpha=0.5, threshold=1.0)

    for _ in range(50):
        meter.update(7.0)
        assert not meter.drift_detected
        assert meter.get() == pytest.approx(7.0)


def test_step_change_reports_drift():
    meter = GeometricMovingAverageMeter(alpha=0.5, threshold=0.5)

    for _ in range(10):
        meter.update(1.0)
        assert not meter.drift_detected

    assert meter.update(100.0).drift_detected


def test_update_returns_self():
    meter = GeometricMovingAverageMeter(alpha=0.5)

    assert meter.update(3.0) is meter


def test_zero_observation_drives_the_mean_to_zero():
    meter = GeometricMovingAverageMeter(alpha=0.5)
    assert meter.update(8.0).get() == pytest.approx(8.0)
    assert meter.update(0.0).get() == 0.0
    # The zero keeps a non-zero weight under exponential decay, so the mean stays zero.
    assert meter.update(8.0).get() == 0.0


def test_negative_observation_raises():
    meter = GeometricMovingAverageMeter(alpha=0.5)
    with pytest.raises(ValueError):
        meter.update(-1.0)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.5}, {"threshold": -1.0}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        GeometricMovingAverageMeter(**kwargs)


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = GeometricMovingAverageMeter(alpha=0.5)
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(GeometricMovingAverageMeter, window_size=10, alpha=0.5)
