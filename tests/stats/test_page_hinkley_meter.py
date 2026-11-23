import pytest

from pysad.statistics import PageHinkleyMeter


def _run_meter(meter, values):
    for value in values:
        meter.update(value)

    return meter.get()


def test_constant_stream_stays_at_zero():
    assert _run_meter(PageHinkleyMeter(), [7.0] * 100) == 0.0


def test_upward_shift_accumulates_hand_computed_value():
    # Running mean includes each new observation, so the deviations are
    # 0, 0, 0, 6.75, 5.4, 4.5 and the upward sum is 16.65.
    assert _run_meter(PageHinkleyMeter(), [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]) == pytest.approx(16.65)


def test_downward_shift_accumulates_symmetrically():
    assert _run_meter(PageHinkleyMeter(), [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]) == pytest.approx(16.65)


def test_delta_absorbs_small_fluctuations():
    # With delta=1.0 the deviations 6.75, 5.4, 4.5 shrink to 5.75, 4.4, 3.5.
    assert _run_meter(
        PageHinkleyMeter(delta=1.0), [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    ) == pytest.approx(13.65)


def test_statistic_bleeds_off_at_the_running_mean():
    # After the shift the running mean is 5.5 and the delta=1.0 statistic is
    # 13.65. Values at the mean contribute -delta per step, so the sum drains
    # to zero instead of lingering.
    meter = PageHinkleyMeter(delta=1.0)
    _run_meter(meter, [1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    assert meter.get() == pytest.approx(13.65)

    for _ in range(20):
        meter.update(5.5)

    assert meter.get() == 0.0


def test_shift_crosses_a_threshold_but_baseline_does_not():
    baseline = _run_meter(PageHinkleyMeter(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    shifted = _run_meter(PageHinkleyMeter(), [1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

    assert baseline < 5.0 < shifted


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError, match="delta must be non-negative"):
        PageHinkleyMeter(delta=-0.5)


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold must be positive"):
        PageHinkleyMeter(threshold=-1.0)


def test_zero_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold must be positive"):
        PageHinkleyMeter(threshold=0.0)


def test_min_instances_below_one_is_rejected():
    with pytest.raises(ValueError, match="min_instances must be a positive integer"):
        PageHinkleyMeter(min_instances=0)


def test_drift_detected_flag_when_statistic_exceeds_threshold():
    meter = PageHinkleyMeter(threshold=5.0, min_instances=1)
    # Values below threshold
    for _ in range(5):
        meter.update(1.0)
        assert not meter.drift_detected
    # Shift that exceeds threshold
    meter.update(100.0)
    assert meter.drift_detected


def test_drift_resets_on_next_update():
    meter = PageHinkleyMeter(threshold=5.0, min_instances=1)
    meter.update(1.0)
    meter.update(100.0)
    assert meter.drift_detected
    assert meter.get() > 5.0

    # Next update resets sums
    meter.update(1.0)
    assert not meter.drift_detected
    assert meter.get() == 0.0


def test_min_instances_delays_testing():
    meter = PageHinkleyMeter(threshold=5.0, min_instances=10)
    # First 9 values establish a baseline
    for _ in range(9):
        meter.update(1.0)
        assert not meter.drift_detected
    # 10th value is a large shift
    meter.update(100.0)
    assert meter.drift_detected


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = PageHinkleyMeter()
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(PageHinkleyMeter, window_size=10)


def test_parameters_are_attributes():
    meter = PageHinkleyMeter(delta=0.5, threshold=25.0, min_instances=20)
    assert meter.delta == 0.5
    assert meter.threshold == 25.0
    assert meter.min_instances == 20
