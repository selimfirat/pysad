import pickle

import numpy as np
import pytest

from pysad.statistics import ExponentialMovingAverageMeter


def test_empty_meter_returns_zero():
    assert ExponentialMovingAverageMeter(alpha=0.5).get() == 0.0


def test_first_update_seeds_the_value():
    meter = ExponentialMovingAverageMeter(alpha=0.5)

    assert meter.update(4.0).get() == 4.0


def test_update_follows_the_recurrence():
    meter = ExponentialMovingAverageMeter(alpha=0.5)

    for value, expected in [(0.0, 0.0), (10.0, 5.0), (10.0, 7.5), (0.0, 3.75)]:
        assert meter.update(value).get() == expected


def test_matches_a_manual_recurrence_on_a_random_stream():
    from pysad.utils import fix_seed

    fix_seed(61)
    values = np.random.rand(200)
    alpha = 0.3
    meter = ExponentialMovingAverageMeter(alpha=alpha)

    expected = float(values[0])
    meter.update(expected)
    assert meter.get() == pytest.approx(expected)
    for value in values[1:]:
        meter.update(float(value))
        expected = alpha * float(value) + (1.0 - alpha) * expected
        assert meter.get() == pytest.approx(expected)


def test_half_life_matches_the_matching_alpha():
    values = [1.0, 2.0, 4.0, 8.0, 3.0, 5.0]
    alpha_meter = ExponentialMovingAverageMeter(alpha=0.5)
    half_life_meter = ExponentialMovingAverageMeter(half_life=1.0)

    for value in values:
        assert alpha_meter.update(value).get() == half_life_meter.update(value).get()


def test_constant_stream_returns_the_constant():
    meter = ExponentialMovingAverageMeter(alpha=0.25)

    for _ in range(100):
        assert meter.update(7.0).get() == 7.0


def test_update_returns_self():
    meter = ExponentialMovingAverageMeter(alpha=0.5)

    assert meter.update(3.0) is meter


def test_remove_is_not_supported():
    with pytest.raises(NotImplementedError, match="cannot remove observations"):
        ExponentialMovingAverageMeter(alpha=0.5).remove(3.0)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.5}, {"half_life": 0.0}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        ExponentialMovingAverageMeter(**kwargs)


def test_both_alpha_and_half_life_raise():
    with pytest.raises(ValueError, match="only one"):
        ExponentialMovingAverageMeter(alpha=0.5, half_life=2.0)


def test_pickle_round_trip():
    meter = ExponentialMovingAverageMeter(alpha=0.4)
    for value in [1.0, 2.0, 3.0]:
        meter.update(value)

    restored = pickle.loads(pickle.dumps(meter))

    assert type(restored) is type(meter)
    for value in [5.0, 4.0]:
        assert restored.update(value).get() == meter.update(value).get()
