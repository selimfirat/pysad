import numpy as np
import pytest

from pysad.statistics import OnlinePeriodicityMeter


def test_periodic_pattern_returns_its_period():
    """The repeating pattern [1, 2, 3, 4] has period 4 and the strongest autocorrelation at lag 4."""
    meter = OnlinePeriodicityMeter(max_lag=20, alpha=0.1)
    pattern = [1.0, 2.0, 3.0, 4.0]
    for i in range(2000):
        meter.update(pattern[i % 4])

    assert meter.get() == 4.0


def test_sine_wave_returns_its_period():
    """A sine wave with period 24 has its strongest autocorrelation at lag 24."""
    meter = OnlinePeriodicityMeter(max_lag=60, alpha=0.05)
    period = 24
    for t in range(5000):
        meter.update(np.sin(2 * np.pi * t / period))

    assert meter.get() == period


def test_constant_stream_has_zero_period():
    """A constant stream has zero variance, so no lag has a positive autocorrelation."""
    meter = OnlinePeriodicityMeter(max_lag=20, alpha=0.1)
    for _ in range(500):
        meter.update(3.0)

    assert meter.get() == 0.0


def test_get_is_zero_before_enough_data():
    """With fewer than two values no lag is computable, so the period is zero."""
    meter = OnlinePeriodicityMeter(max_lag=10)

    assert meter.get() == 0.0

    meter.update(1.0)
    assert meter.get() == 0.0


def test_period_is_bounded_by_max_lag():
    """The estimated period is never larger than max_lag and never negative."""
    meter = OnlinePeriodicityMeter(max_lag=8, alpha=0.2)
    pattern = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for i in range(3000):
        meter.update(pattern[i % 8])

    period = meter.get()
    assert 0.0 <= period <= 8.0


def test_validation():
    with pytest.raises(ValueError):
        OnlinePeriodicityMeter(max_lag=0)

    with pytest.raises(ValueError):
        OnlinePeriodicityMeter(max_lag=-1)

    with pytest.raises(ValueError):
        OnlinePeriodicityMeter(alpha=0.0)

    with pytest.raises(ValueError):
        OnlinePeriodicityMeter(alpha=1.5)


def test_remove_raises():
    """The meter keeps exponentially weighted summaries, so it cannot remove observations."""
    meter = OnlinePeriodicityMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)


def test_get_returns_float():
    meter = OnlinePeriodicityMeter(max_lag=10, alpha=0.5)
    meter.update(1.0)
    meter.update(2.0)

    assert isinstance(meter.get(), float)


def test_update_is_fast():
    """One update stays well under 5 ms even after many points."""
    import time

    meter = OnlinePeriodicityMeter(max_lag=200, alpha=0.01)
    rng = np.random.default_rng(0)
    values = rng.normal(size=20000)

    for value in values:
        meter.update(value)

    start = time.perf_counter()
    for value in values:
        meter.update(value)
    elapsed = time.perf_counter() - start

    assert elapsed / values.shape[0] < 0.005
