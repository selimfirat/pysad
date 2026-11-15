import pytest


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_cusum_detects_a_shift_from_a_fixed_mean(direction):
    from pysad.statistics import CUSUMMeter

    meter = CUSUMMeter(mean=10.0, delta=0.5, threshold=3.0)
    assert meter.get() == 0.0
    assert not meter.drift_detected
    for value, expected, drift in [
        (10.0, 0.0, False),
        (10.0 + direction * 2.0, 1.5, False),
        (10.0 + direction * 2.0, 3.0, False),
        (10.0 + direction * 2.0, 4.5, True),
    ]:
        assert meter.update(value) is meter
        assert meter.get() == pytest.approx(expected)
        assert meter.g_plus_ == pytest.approx(expected if direction > 0 else 0.0)
        assert meter.g_minus_ == pytest.approx(-expected if direction < 0 else 0.0)
        assert meter.drift_detected is drift


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_cusum_restarts_on_the_update_after_an_alarm(direction):
    from pysad.statistics import CUSUMMeter

    meter = CUSUMMeter(mean=10.0, delta=0.5, threshold=2.0)
    meter.update(10.0 + direction * 4.0)
    assert meter.get() == pytest.approx(3.5)
    assert meter.drift_detected

    meter.update(10.0 - direction * 2.0)
    assert meter.get() == pytest.approx(1.5)
    assert not meter.drift_detected
    meter.update(10.0 - direction * 2.0)
    assert meter.get() == pytest.approx(3.0)
    assert meter.drift_detected


def test_cusum_deviations_decay_and_change_direction():
    from pysad.statistics import CUSUMMeter

    meter = CUSUMMeter(mean=10.0, delta=0.5, threshold=100.0)
    for value, upward, downward in [
        (12.0, 1.5, 0.0),
        (10.0, 1.0, 0.0),
        (8.0, 0.0, -1.5),
        (10.0, 0.0, -1.0),
        (10.0, 0.0, -0.5),
        (10.0, 0.0, 0.0),
    ]:
        meter.update(value)
        assert meter.g_plus_ == pytest.approx(upward)
        assert meter.g_minus_ == pytest.approx(downward)
        assert meter.get() == pytest.approx(max(upward, -downward))
        assert not meter.drift_detected


@pytest.mark.parametrize("delta", [0.0, 0.5])
def test_cusum_absorbs_fluctuations_within_delta(delta):
    from pysad.statistics import CUSUMMeter

    meter = CUSUMMeter(mean=10.0, delta=delta)
    for value in [10.0 - delta, 10.0, 10.0 + delta] * 100:
        meter.update(value)
        assert meter.get() == 0.0
        assert not meter.drift_detected


@pytest.mark.parametrize("kwargs", [{"delta": -0.5}, {"threshold": 0.0}, {"threshold": -1.0}])
def test_cusum_rejects_parameters_that_invalidate_the_alarm_rule(kwargs):
    from pysad.statistics import CUSUMMeter

    with pytest.raises(ValueError):
        CUSUMMeter(**kwargs)
