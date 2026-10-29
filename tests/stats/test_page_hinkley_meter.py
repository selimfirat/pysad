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


def test_remove_is_not_supported():
    meter = PageHinkleyMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
