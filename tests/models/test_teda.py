def test_teda_recursive_statistics_match_numpy():
    import numpy as np

    from pysad.models import TEDA

    X = np.random.RandomState(0).normal(size=(50, 3))
    model = TEDA().fit(X)

    np.testing.assert_allclose(model.mean_, X.mean(axis=0))
    # The paper's variance is the mean squared Euclidean distance to the mean.
    np.testing.assert_allclose(model.variance_, ((X - X.mean(axis=0)) ** 2).sum(axis=1).mean())


def test_teda_scores_match_the_paper_eccentricity_rule():
    import numpy as np

    from pysad.models import TEDA

    X = np.random.RandomState(1).normal(size=(40, 2))
    X[30] += 10.0
    scores = TEDA().fit_score(X)

    m = 3.0
    for k in range(2, len(X) + 1):
        seen = X[:k]  # the statistics include the scored point
        mean = seen.mean(axis=0)
        variance = ((seen - mean) ** 2).sum(axis=1).mean()
        sq_dist = ((seen[-1] - mean) ** 2).sum()
        np.testing.assert_allclose(scores[k - 1], sq_dist / variance)

        normalized_eccentricity = (1 / k + sq_dist / (k * variance)) / 2
        assert (normalized_eccentricity > (m**2 + 1) / (2 * k)) == (scores[k - 1] > m**2)

    assert scores[30] > m**2


def test_teda_far_point_scores_above_nine_and_most_of_a_gaussian_cloud_below():
    import numpy as np

    from pysad.models import TEDA

    X = np.random.RandomState(2).normal(size=(500, 2))
    model = TEDA()
    scores = model.fit_score(X)

    assert np.mean(scores[10:] < 9.0) > 0.95
    assert model.score_partial(np.array([20.0, 20.0])) > 9.0


def test_teda_scores_zero_for_the_first_point_and_a_constant_stream():
    import numpy as np

    from pysad.models import TEDA

    scores = TEDA().fit_score(np.full((10, 2), 0.1))

    np.testing.assert_array_equal(scores, np.zeros(10))
