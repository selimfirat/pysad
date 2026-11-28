import math

import pytest


def test_ewma_control_chart_starts_at_the_target_mean():
    from pysad.statistics import EWMAControlChartMeter

    meter = EWMAControlChartMeter(mean=10.0)
    assert meter.get() == 10.0
    assert meter.limit_ == 0.0
    assert not meter.drift_detected


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_ewma_control_chart_detects_a_shift_from_the_target_mean(direction):
    from pysad.statistics import EWMAControlChartMeter

    # alpha=0.5: the limit is 3 * 2 * sqrt(1/3 * (1 - 0.25 ** t)).
    meter = EWMAControlChartMeter(mean=10.0, std=2.0, alpha=0.5, threshold=3.0)
    for value, ewma, limit, drift in [
        (10.0, 10.0, 3.0, False),
        (10.0 + direction * 6.0, 10.0 + direction * 3.0, 6.0 * math.sqrt(0.3125), False),
        (10.0 + direction * 6.0, 10.0 + direction * 4.5, 6.0 * math.sqrt(0.328125), True),
    ]:
        assert meter.update(value) is meter
        assert meter.get() == pytest.approx(ewma)
        assert meter.limit_ == pytest.approx(limit)
        assert meter.drift_detected is drift


def test_ewma_control_chart_limits_widen_to_their_steady_state():
    from pysad.statistics import EWMAControlChartMeter

    meter = EWMAControlChartMeter(std=2.0, alpha=0.2, threshold=3.0)
    limits = [meter.update(0.0).limit_ for _ in range(100)]
    assert limits == sorted(limits)
    assert limits[-1] == pytest.approx(3.0 * 2.0 * math.sqrt(0.2 / 1.8))


def test_ewma_control_chart_with_full_weight_is_a_shewhart_chart():
    from pysad.statistics import EWMAControlChartMeter

    meter = EWMAControlChartMeter(mean=0.0, std=1.0, alpha=1.0, threshold=3.0)
    meter.update(3.0)
    assert meter.limit_ == pytest.approx(3.0)
    assert not meter.drift_detected
    meter.update(-3.5)
    assert meter.get() == pytest.approx(-3.5)
    assert meter.drift_detected


def test_ewma_control_chart_keeps_running_after_an_alarm():
    from pysad.statistics import EWMAControlChartMeter

    meter = EWMAControlChartMeter(mean=0.0, std=1.0, alpha=0.5, threshold=3.0)
    meter.update(10.0)
    assert meter.drift_detected
    for _ in range(10):
        meter.update(0.0)
    assert meter.get() == pytest.approx(10.0 * 0.5**11)
    assert not meter.drift_detected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"std": 0.0},
        {"std": -1.0},
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"threshold": 0.0},
    ],
)
def test_ewma_control_chart_rejects_invalid_parameters(kwargs):
    from pysad.statistics import EWMAControlChartMeter

    with pytest.raises(ValueError):
        EWMAControlChartMeter(**kwargs)
