import numpy as np
import pytest


@pytest.mark.parametrize("seed", range(3))
def test_change_finder_scores_a_level_shift_highest(seed):
    from pysad.models import ChangeFinder

    rng = np.random.default_rng(seed)
    t = np.arange(1000)
    X = 0.5 + 0.2 * np.sin(2 * np.pi * t / 50) + rng.normal(0.0, 0.02, t.size)
    X[600:] += 0.3

    scores = ChangeFinder().fit_score(X.reshape(-1, 1))

    # The first points only move the AR models away from their initial parameters.
    assert 600 <= scores[100:].argmax() + 100 <= 610


def test_change_finder_outlier_score_is_the_log_loss_under_the_learned_ar_model():
    from pysad.models import ChangeFinder

    # An AR(1) series with coefficient 0.9 and noise variance 0.01.
    rng = np.random.default_rng(0)
    X = np.zeros(20_000)
    for i in range(1, X.size):
        X[i] = 0.9 * X[i - 1] + rng.normal(0.0, 0.1)

    scores = ChangeFinder(change_point=False).fit_score(X.reshape(-1, 1))

    # The expected log loss of the true model is the entropy of its noise. A model without the AR
    # coefficient has the variance 0.01 / (1 - 0.81) and an expected log loss of about -0.05.
    entropy = 0.5 * np.log(2 * np.pi * np.e * 0.01)
    assert scores[5000:].mean() == pytest.approx(entropy, abs=0.02)


def test_change_finder_scores_the_first_value_under_the_initial_parameters():
    from pysad.models import ChangeFinder

    model = ChangeFinder(order=1, change_point=False).fit(np.array([[0.3]]))

    variance = 1.0 + 1e-6
    expected = 0.5 * (np.log(2 * np.pi * variance) + 0.7**2 / variance)
    assert model.score_partial(np.array([0.7])) == pytest.approx(expected)


@pytest.mark.parametrize(("order", "window_size"), [(1, 1), (1, 5), (3, 4)])
def test_change_finder_warmup(order, window_size):
    from pysad.models import ChangeFinder

    X = np.random.default_rng(0).normal(size=(50, 1))

    outlier_scores = ChangeFinder(order, window_size=window_size, change_point=False).fit_score(X)
    change_scores = ChangeFinder(order, window_size=window_size).fit_score(X)

    assert np.all(outlier_scores[:order] == 0.0)
    assert np.all(outlier_scores[order:] != 0.0)
    num_zeros = 2 * (order + window_size - 1)
    assert np.all(change_scores[:num_zeros] == 0.0)
    assert np.all(change_scores[num_zeros:] != 0.0)


def test_change_finder_rejects_multivariate_instances():
    from pysad.models import ChangeFinder

    with pytest.raises(ValueError, match="univariate"):
        ChangeFinder().fit_partial(np.zeros(2))


@pytest.mark.parametrize(
    "kwargs",
    [{"order": 0}, {"window_size": 0}, {"discount": 0.0}, {"discount": 1.0}],
)
def test_change_finder_rejects_invalid_parameters(kwargs):
    from pysad.models import ChangeFinder

    with pytest.raises(ValueError):
        ChangeFinder(**kwargs)
