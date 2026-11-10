import pytest


def test_pewma_scores_with_the_exact_mean_and_variance_during_the_training_period():
    import numpy as np

    from pysad.models import PEWMA

    x = np.random.RandomState(0).normal(loc=5.0, scale=2.0, size=30)
    scores = PEWMA(training_period=30).fit_score(x.reshape(-1, 1))

    # fit_score_partial scores each value with the statistics of the values before it.
    expected = [abs(x[t] - x[:t].mean()) / x[:t].std() for t in range(2, len(x))]
    np.testing.assert_allclose(scores[2:], expected)


def test_pewma_with_zero_beta_is_an_ewma_with_weight_alpha():
    import numpy as np

    from pysad.models import PEWMA

    x = np.random.RandomState(1).normal(size=100)
    alpha, training_period = 0.9, 10
    scores = PEWMA(alpha=alpha, beta=0.0, training_period=training_period).fit_score(
        x.reshape(-1, 1)
    )

    s1, s2 = x[:training_period].mean(), (x[:training_period] ** 2).mean()
    for t in range(training_period, len(x)):
        np.testing.assert_allclose(scores[t], abs(x[t] - s1) / np.sqrt(s2 - s1**2))
        s1 = alpha * s1 + (1 - alpha) * x[t]
        s2 = alpha * s2 + (1 - alpha) * x[t] ** 2


def test_pewma_weights_each_value_by_its_probability_after_the_training_period():
    import math

    import numpy as np

    from pysad.models import PEWMA

    model = PEWMA(alpha=0.5, beta=1.0, training_period=2).fit(np.array([[0.0], [2.0]]))
    # The training period gives the mean 1 and the standard deviation 1.
    assert model.score_partial(np.array([4.0])) == pytest.approx(3.0)

    model.fit_partial(np.array([4.0]))
    p = math.exp(-(3.0**2) / 2) / math.sqrt(2 * math.pi)
    weight = (1 - p) * 0.5
    s1 = weight * 1.0 + (1 - weight) * 4.0
    s2 = weight * 2.0 + (1 - weight) * 16.0
    assert model.score_partial(np.array([0.0])) == pytest.approx(s1 / math.sqrt(s2 - s1**2))


def test_pewma_forgets_a_spike_sooner_than_an_ewma():
    import numpy as np

    from pysad.models import PEWMA

    x = np.random.RandomState(2).normal(size=200)
    x[150] = 50.0
    X = x.reshape(-1, 1)

    pewma_scores = PEWMA(beta=1.0).fit_score(X)
    ewma_scores = PEWMA(beta=0.0).fit_score(X)

    # Both models give the spike the weight alpha.
    # PEWMA gives the likely values after it a smaller weight.
    # Its standard deviation shrinks back sooner, so these values score higher.
    assert pewma_scores[150] > 10.0 and ewma_scores[150] > 10.0
    assert pewma_scores[151:].mean() > ewma_scores[151:].mean()


def test_pewma_scores_zero_without_spread():
    import numpy as np

    from pysad.models import PEWMA

    model = PEWMA()
    assert model.score_partial(np.array([1.0])) == 0.0

    # One value, or a constant stream, gives a standard deviation of 0.
    model.fit_partial(np.array([1.0]))
    assert model.score_partial(np.array([3.0])) == 0.0

    scores = PEWMA().fit_score(np.full((100, 1), 0.1))
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": -0.1}, {"alpha": 1.1}, {"beta": -0.1}, {"beta": 1.1}, {"training_period": 0}],
)
def test_pewma_rejects_parameters_outside_the_paper_range(kwargs):
    from pysad.models import PEWMA

    with pytest.raises(ValueError):
        PEWMA(**kwargs)
