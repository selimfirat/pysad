import numpy as np
import pytest


def test_rs_forest_estimates_the_density_of_the_previous_window():
    from scipy.stats import norm

    from pysad.models import RSForest

    window_size = 4096
    X = np.random.default_rng(1).normal(size=(window_size, 1))
    model = RSForest(
        window_size=window_size, num_trees=100, max_depth=8, node_size_limit=0, random_state=0
    ).fit(X)

    # The score is the negative density relative to the volume of the estimated range: the
    # shortest interval that holds 90% of the first window, widened by three standard deviations.
    values = np.sort(X[:, 0])
    m = window_size * 9 // 10
    start = (values[m:] - values[: window_size - m]).argmin()
    volume = values[start + m] - values[start] + 6.0 * X.std()

    points = np.array([-1.0, 0.0, 1.0])
    densities = [-model.score_partial(np.array([x])) / volume for x in points]
    np.testing.assert_allclose(densities, norm.pdf(points), rtol=0.05)


def test_rs_forest_scores_zero_until_the_first_window_is_full():
    from pysad.models import RSForest

    X = np.random.default_rng(0).normal(size=(40, 2))
    scores = RSForest(window_size=20, num_trees=5, max_depth=6, random_state=0).fit_score(X)

    np.testing.assert_array_equal(scores[:20], 0.0)
    assert (scores[20:] < 0.0).all()


@pytest.mark.parametrize(("label", "dense"), [(None, True), (1, False)])
def test_rs_forest_learns_the_latest_window_without_labeled_anomalies(label, dense):
    from pysad.models import RSForest

    window_size = 50
    rng = np.random.default_rng(0)
    model = RSForest(window_size=window_size, num_trees=10, max_depth=6, random_state=0)
    model.fit(rng.normal(size=(window_size, 2)))
    # A window of points at a new place becomes the reference, unless each one is labeled an anomaly.
    shifted = 2.0 + 0.1 * rng.normal(size=(window_size, 2))
    for x in shifted:
        model.fit_partial(x, label)

    assert (model.score_partial(np.array([2.0, 2.0])) < 0.0) == dense


def test_rs_forest_rejects_a_zero_window_size():
    from pysad.models import RSForest

    with pytest.raises(ValueError, match="window_size"):
        RSForest(window_size=0)
