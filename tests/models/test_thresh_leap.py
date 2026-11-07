def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outlier_idx] = 1

    return X, y


def test_thresh_leap_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import ThreshLEAP

    X, y = generate_stream()

    scores = ThreshLEAP(window_size=500, max_radius=0.5, num_neighbors=10).fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.95


def test_thresh_leap_scores_the_missing_neighbors_of_definition_1():
    """The score is 1 - min(n, k) / k for the n neighbors among the window_size - 1 previous instances."""
    import numpy as np

    from pysad.models import ThreshLEAP

    X = np.random.RandomState(3).normal(size=(400, 2))
    window_size, radius, k = 60, 0.5, 5

    scores = ThreshLEAP(window_size=window_size, max_radius=radius, num_neighbors=k).fit_score(X)

    for t, x in enumerate(X):
        previous = X[max(0, t - window_size + 1) : t]
        n = (np.linalg.norm(previous - x, axis=1) <= radius).sum()
        assert scores[t] == 1.0 - min(n, k) / k


def test_thresh_leap_ranks_outliers_by_their_neighbors():
    import numpy as np

    from pysad.models import ThreshLEAP

    model = ThreshLEAP(window_size=10, max_radius=0.5, num_neighbors=4)
    model.fit(np.array([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1]]))

    assert model.score_partial(np.array([0.15])) == 0.0  # an inlier with 4 neighbors
    assert model.score_partial(np.array([2.05])) == 0.5  # an outlier with 2 neighbors
    assert model.score_partial(np.array([9.0])) == 1.0  # an outlier with no neighbors


def test_thresh_leap_score_partial_skips_the_expiring_instance():
    import numpy as np

    from pysad.models import ThreshLEAP

    model = ThreshLEAP(window_size=3, max_radius=0.5, num_neighbors=2)
    model.fit(np.array([[0.0], [0.1], [5.0]]))

    # The neighbor 0.0 expires when the next instance arrives.
    assert model.score_partial(np.array([0.05])) == 0.5
    assert model.fit_score_partial(np.array([0.05])) == 0.5


def test_thresh_leap_without_a_radius_or_a_window():
    import numpy as np

    from pysad.models import ThreshLEAP

    assert ThreshLEAP().score_partial(np.zeros(2)) == 1.0
    assert ThreshLEAP(max_radius=1.0).score_partial(np.zeros(2)) == 1.0


def test_thresh_leap_sets_the_radius_from_the_first_instances():
    import numpy as np

    from pysad.models import ThreshLEAP

    X = np.random.RandomState(0).normal(size=(60, 2))
    model = ThreshLEAP(calibration_size=40).fit(X)

    nearest = np.sort(np.linalg.norm(X[:40, None] - X[None, :40], axis=2), axis=1)[:, 1]
    assert model.radius_ == np.quantile(nearest, 0.95)
