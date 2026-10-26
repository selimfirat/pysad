import numpy as np
import pytest

from pysad.statistics import QuantileMeter

SEED = 61
NUM_POINTS = 20_000


def _uniform_stream():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_POINTS)


def _normal_stream():
    rng = np.random.RandomState(SEED)
    return rng.randn(NUM_POINTS)


def _run_meter(meter, values):
    for value in values:
        meter.update(float(value))

    return meter.get()


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
def test_tracks_quantile_of_uniform_stream(q):
    values = _uniform_stream()

    estimate = _run_meter(QuantileMeter(q=q), values)
    exact = np.quantile(values, q)

    assert abs(estimate - exact) < 0.01


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_tracks_quantile_of_normal_stream(q):
    values = _normal_stream()

    estimate = _run_meter(QuantileMeter(q=q), values)
    exact = np.quantile(values, q)

    assert abs(estimate - exact) < 0.05


@pytest.mark.parametrize("num_values", [1, 2, 3, 4])
def test_startup_interpolates_exactly(num_values):
    values = [0.37, 0.11, 0.83, 0.59][:num_values]
    meter = QuantileMeter(q=0.7)

    for value in values:
        meter.update(value)

    assert meter.get() == pytest.approx(np.quantile(values, 0.7), rel=1e-10)


def test_switch_to_markers_at_five_observations():
    meter = QuantileMeter(q=0.5)

    for value in [1.0, 5.0, 2.0, 4.0, 3.0]:
        meter.update(value)

    # The central marker starts at the third order statistic.
    assert meter.get() == 3.0


def test_constant_stream_returns_the_constant():
    meter = QuantileMeter(q=0.95)

    for _ in range(100):
        meter.update(7.0)

    assert meter.get() == 7.0


def test_higher_quantiles_do_not_cross():
    values = _uniform_stream()
    meters = [QuantileMeter(q=q) for q in (0.25, 0.5, 0.75)]

    for value in values:
        for meter in meters:
            meter.update(float(value))

    assert meter is meters[-1]
    assert meters[0].get() <= meters[1].get() <= meters[2].get()


def test_same_stream_gives_the_same_estimate():
    values = _uniform_stream()

    first = _run_meter(QuantileMeter(q=0.9), values)
    second = _run_meter(QuantileMeter(q=0.9), values)

    assert first == second


def test_empty_meter_returns_zero():
    assert QuantileMeter(q=0.5).get() == 0.0


def test_update_returns_self():
    meter = QuantileMeter(q=0.5)

    assert meter.update(3.0) is meter


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_invalid_quantile_raises(q):
    with pytest.raises(ValueError, match="open interval"):
        QuantileMeter(q=q)


def test_remove_is_not_supported():
    with pytest.raises(NotImplementedError, match="cannot remove observations"):
        QuantileMeter(q=0.5).remove(3.0)
