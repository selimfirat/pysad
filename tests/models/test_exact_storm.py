def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outlier_idx] = 1

    return X, y


def test_exact_storm_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import ExactStorm

    X, y = generate_stream()

    model = ExactStorm(window_size=500, max_radius=0.5)
    scores = model.fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.95


def test_exact_storm_outlier_scores_higher():
    import numpy as np

    from pysad.models import ExactStorm

    X, _ = generate_stream()

    model = ExactStorm(window_size=500, max_radius=0.5).fit(X)
    inlier_score, outlier_score = model.score(np.array([[0.0, 0.0], [10.0, 10.0]]))

    assert outlier_score == 1.0
    assert inlier_score < outlier_score


def test_exact_storm_neighbor_counting():
    import numpy as np
    from numpy.testing import assert_almost_equal

    from pysad.models import ExactStorm

    model = ExactStorm(window_size=5, max_radius=1.0)
    model.fit(np.array([[0.0], [1.0], [5.0], [6.0]]))

    # Neighbors are at distance not greater than max_radius, and every instance in the window counts while the window is not full.
    assert_almost_equal(model.score_partial(np.array([0.0])), 0.5)
    assert_almost_equal(model.score_partial(np.array([20.0])), 1.0)

    model.fit_partial(np.array([10.0]))

    # The fitted instance is not its own neighbor: after 0.0 expires, only 5.0 and 6.0 of the other four instances in the window are neighbors of itself.
    assert_almost_equal(model.fit_score_partial(np.array([5.5])), 0.5)


def test_exact_storm_empty_window():
    import numpy as np

    from pysad.models import ExactStorm

    # With no instances to be neighbors of, an instance has fewer than k neighbors for any k, so it gets the maximum score.
    assert ExactStorm().score_partial(np.array([0.0, 0.0])) == 1.0
    assert ExactStorm().fit_score_partial(np.array([0.0, 0.0])) == 1.0


def test_exact_storm_score_partial_skips_the_expiring_instance():
    import copy

    import numpy as np

    from pysad.models import ExactStorm

    model = ExactStorm(window_size=3, max_radius=0.5).fit(np.array([[0.0], [5.0], [10.0]]))

    # When 0.0 arrives again, the first 0.0 expires, so it is not a neighbor.
    assert model.score_partial(np.array([0.0])) == 1.0
    assert copy.deepcopy(model).fit_score_partial(np.array([0.0])) == 1.0
    assert model.score_partial(np.array([5.0])) == 0.5


def test_exact_storm_score_partial_matches_fit_score_partial():
    import copy

    import numpy as np

    from pysad.models import ExactStorm

    rng = np.random.RandomState(61)
    X = rng.normal(size=(100, 2))
    model = ExactStorm(window_size=20, max_radius=0.5)

    for x in X:
        assert model.score_partial(x) == copy.deepcopy(model).fit_score_partial(x)
        model.fit_partial(x)


def test_exact_storm_window_of_one():
    import numpy as np

    from pysad.models import ExactStorm

    # A window of one holds only the scored instance, so no other instance can be its neighbor.
    model = ExactStorm(window_size=1, max_radius=1.0).fit(np.array([[0.0]]))

    assert model.score_partial(np.array([0.0])) == 1.0
    assert model.fit_score_partial(np.array([0.0])) == 1.0
