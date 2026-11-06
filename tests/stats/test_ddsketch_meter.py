import numpy as np
import pytest

from pysad.statistics import DDSketchMeter, RunningStatistic

SEED = 61
NUM_POINTS = 10_000
QUANTILES = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]


def _streams():
    rng = np.random.RandomState(SEED)
    return {
        "lognormal": rng.lognormal(sigma=2.0, size=NUM_POINTS),
        "normal": rng.randn(NUM_POINTS),
        "with_zeros": np.concatenate([rng.randn(NUM_POINTS), np.zeros(NUM_POINTS // 4)]),
    }


def _run_meter(meter, values):
    for value in values:
        meter.update(float(value))

    return meter.get()


@pytest.mark.parametrize("stream", ["lognormal", "normal", "with_zeros"])
@pytest.mark.parametrize("q", QUANTILES)
def test_estimate_is_within_relative_accuracy_of_lower_quantile(stream, q):
    values = _streams()[stream]

    estimate = _run_meter(DDSketchMeter(q=q, relative_accuracy=0.01), values)
    exact = np.quantile(values, q, method="lower")

    assert abs(estimate - exact) <= 0.01 * abs(exact) + 1e-12


def test_bounded_buckets_keep_high_quantiles_accurate():
    values = np.random.RandomState(SEED).lognormal(sigma=3.0, size=NUM_POINTS)
    meter = DDSketchMeter(q=0.99, relative_accuracy=0.01, max_buckets=250)

    estimate = _run_meter(meter, values)
    exact = np.quantile(values, 0.99, method="lower")

    # The stream spans about 1,000 buckets, so the lowest ones are merged. The guarantee
    # holds for quantiles at most a factor gamma ** 249 below the maximum.
    assert values.max() <= exact * (1.01 / 0.99) ** 249
    assert abs(estimate - exact) <= 0.01 * exact


def test_remove_restores_the_estimate():
    values = _streams()["normal"]
    meter = DDSketchMeter(q=0.9)
    for value in values[:1000]:
        meter.update(float(value))
    before = meter.get()

    for value in values[1000:2000]:
        meter.update(float(value))
    for value in values[1000:2000]:
        meter.remove(float(value))

    assert meter.get() == before
    assert meter.num_items == 1000


def test_remove_after_merged_buckets_empties_the_meter():
    values = np.random.RandomState(SEED).lognormal(sigma=3.0, size=2000)
    meter = DDSketchMeter(max_buckets=20)

    for value in values:
        meter.update(float(value))
    for value in values:
        meter.remove(float(value))

    assert meter.num_items == 0
    assert meter.get() == 0.0


def test_remove_finds_merged_counts_below_a_later_bucket():
    meter = DDSketchMeter(max_buckets=5)
    for exponent in range(1, 12):
        meter.update(10.0**exponent)
    for exponent in range(6, 12):
        meter.remove(10.0**exponent)
    meter.update(1e-3)

    meter.remove(10.0)

    assert meter.num_items == 5


@pytest.mark.parametrize("seed", range(5))
def test_windowed_quantile_with_merged_buckets_on_heavy_tailed_stream(seed):
    values = np.random.RandomState(seed).lognormal(sigma=4.0, size=2000)
    meter = RunningStatistic(DDSketchMeter, window_size=200, max_buckets=50)

    for value in values:
        meter.update(float(value))

    assert meter.get() > 0.0


def test_windowed_quantile_follows_the_window():
    values = np.concatenate([np.full(500, 1.0), np.full(500, 100.0)])
    meter = RunningStatistic(DDSketchMeter, window_size=100, q=0.5)

    for value in values:
        meter.update(float(value))

    assert meter.get() == pytest.approx(100.0, rel=0.01)


def test_remove_of_absent_value_raises():
    meter = DDSketchMeter()
    meter.update(1.0)

    with pytest.raises(ValueError, match="not in DDSketchMeter"):
        meter.remove(5.0)
    with pytest.raises(ValueError, match="not in DDSketchMeter"):
        meter.remove(0.0)


def test_empty_meter_returns_zero():
    assert DDSketchMeter().get() == 0.0


def test_update_returns_self():
    meter = DDSketchMeter()

    assert meter.update(3.0) is meter
    assert meter.remove(3.0) is meter


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_invalid_quantile_raises(q):
    with pytest.raises(ValueError, match="q must be"):
        DDSketchMeter(q=q)


@pytest.mark.parametrize("relative_accuracy", [0.0, 1.0, -0.1])
def test_invalid_relative_accuracy_raises(relative_accuracy):
    with pytest.raises(ValueError, match="relative_accuracy must be"):
        DDSketchMeter(relative_accuracy=relative_accuracy)
