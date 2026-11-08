import numpy as np
import pytest


def _seasonal_series(seed, num_values=2000, period=50):
    rng = np.random.default_rng(seed)
    t = np.arange(num_values)
    return 0.5 + 0.2 * np.sin(2 * np.pi * t / period) + rng.normal(0.0, 0.01, num_values)


@pytest.mark.parametrize("seed", range(3))
def test_holt_winters_flags_a_level_shift_as_a_failure(seed):
    from pysad.models import HoltWinters

    X = _seasonal_series(seed)
    X[1500:1520] += 0.2

    scores = HoltWinters(period=50, alpha=0.1, beta=0.001, gamma=0.1).fit_score(X.reshape(-1, 1))

    # The 7th value of the shift completes 7 violations among the last 9 values.
    assert np.flatnonzero(scores)[0] == 1506
    assert np.all(scores[1506:1515] >= 7)


def test_holt_winters_flags_each_spike_as_a_violation():
    from pysad.models import HoltWinters

    X = _seasonal_series(0, num_values=1000)
    X[[700, 703]] += 1.0

    scores = HoltWinters(period=50, window_size=1, threshold=1).fit_score(X.reshape(-1, 1))

    assert scores[700] == 1.0
    assert scores[703] == 1.0


@pytest.mark.parametrize("threshold", [2, 7, 9])
def test_holt_winters_failure_is_a_violation_count_of_at_least_threshold(threshold):
    from pysad.models import HoltWinters

    X = _seasonal_series(0, num_values=1000).reshape(-1, 1)

    counts = HoltWinters(period=50, threshold=1).fit_score(X)
    scores = HoltWinters(period=50, threshold=threshold).fit_score(X)

    assert counts.max() <= 9
    assert np.array_equal(scores, np.where(counts >= threshold, counts, 0.0))


def test_holt_winters_forecasts_with_the_update_equations_of_the_paper():
    from pysad.models import HoltWinters

    alpha, beta, gamma, period = 0.3, 0.2, 0.4, 3
    X = np.random.default_rng(0).normal(size=20)
    model = HoltWinters(period=period, alpha=alpha, beta=beta, gamma=gamma).fit(X.reshape(-1, 1))

    # The cold start: the first period sets the seasonal coefficients against the first value.
    a, b, c = X[0], 0.0, list(X[:period] - X[0])
    for y in X[period:]:
        a_next = alpha * (y - c[0]) + (1 - alpha) * (a + b)
        b = beta * (a_next - a) + (1 - beta) * b
        a = a_next
        c = [*c[1:], gamma * (y - a) + (1 - gamma) * c[0]]

    assert model.intercept_ == pytest.approx(a)
    assert model.slope_ == pytest.approx(b)
    # 20 values end 2 points into the cycle, so the next point of the cycle is the third.
    assert np.roll(model.seasonal_, -(X.size % period)) == pytest.approx(c)


def test_holt_winters_smooths_the_seasonal_cycle_at_the_end_of_a_period():
    from pysad.models import HoltWinters

    X = np.random.default_rng(0).normal(size=40)

    model = HoltWinters(period=40).fit(X.reshape(-1, 1))

    # A period of 40 smooths over 2 * floor(0.025 * 40) + 1 = 3 points of the cycle.
    seasonal = X - X[0]
    expected = (np.roll(seasonal, 1) + seasonal + np.roll(seasonal, -1)) / 3
    assert model.seasonal_ == pytest.approx(expected)


def test_holt_winters_scores_nothing_before_it_knows_the_deviations():
    from pysad.models import HoltWinters

    X = np.random.default_rng(0).normal(size=(30, 1))

    scores = HoltWinters(period=5, window_size=1, threshold=1).fit_score(X)

    assert np.all(scores[:10] == 0.0)
    assert np.any(scores[10:] > 0.0)


def test_holt_winters_scores_a_constant_series_zero():
    from pysad.models import HoltWinters

    scores = HoltWinters(period=48, threshold=1).fit_score(np.full((500, 1), 3.0))

    assert np.all(scores == 0.0)


def test_holt_winters_rejects_multivariate_instances():
    from pysad.models import HoltWinters

    with pytest.raises(ValueError, match="univariate"):
        HoltWinters(period=4).fit_partial(np.zeros(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": 0},
        {"alpha": 0.0},
        {"beta": 1.0},
        {"gamma": 1.5},
        {"delta": 0.0},
        {"window_size": 0},
        {"threshold": 0},
        {"window_size": 5, "threshold": 6},
    ],
)
def test_holt_winters_rejects_invalid_parameters(kwargs):
    from pysad.models import HoltWinters

    with pytest.raises(ValueError):
        HoltWinters(**{"period": 4, **kwargs})
