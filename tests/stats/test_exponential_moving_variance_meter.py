import numpy as np
import pytest


@pytest.mark.parametrize(
    ("kwargs", "alpha"),
    [({}, 0.1), ({"alpha": 0.3}, 0.3), ({"alpha": 1.0}, 1.0), ({"half_life": 2.0}, 1 - 0.5**0.5)],
)
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_prefixes_match_weighted_population_variance(kwargs, alpha, offset):
    from pysad.statistics import ExponentialMovingVarianceMeter

    values = np.random.RandomState(61).normal(size=200) + offset
    meter = ExponentialMovingVarianceMeter(**kwargs)

    for size, value in enumerate(values, start=1):
        assert meter.update(float(value)) is meter
        weights = (1 - alpha) ** np.arange(size - 1, -1, -1, dtype=float)
        weights[1:] *= alpha
        # Center the data first to keep the independent calculation stable.
        centered = values[:size] - values[0]
        mean = np.average(centered, weights=weights)
        expected = np.average((centered - mean) ** 2, weights=weights)
        assert meter.get() == pytest.approx(expected, rel=1e-6, abs=1e-6)
        assert meter.num_items == size


def test_empty_and_constant_stream_have_zero_variance():
    from pysad.statistics import ExponentialMovingVarianceMeter

    meter = ExponentialMovingVarianceMeter()
    assert meter.get() == 0.0
    for _ in range(100):
        assert meter.update(7.0).get() == 0.0


def test_alpha_one_keeps_zero_variance_for_large_finite_inputs():
    from pysad.statistics import ExponentialMovingVarianceMeter

    meter = ExponentialMovingVarianceMeter(alpha=1.0)
    values = [0.0, 1e200, -1e200, 1e308, -1e308, 3.0]

    variances = [meter.update(value).get() for value in values]

    assert variances == [0.0] * len(values)
    assert meter.num_items == len(values)
