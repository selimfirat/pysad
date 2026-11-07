def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outlier_idx] = 1

    return X, y


def test_approx_storm_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import ApproxStorm

    X, y = generate_stream()

    model = ApproxStorm(
        window_size=500, max_radius=0.5, num_neighbors=2, sample_fraction=0.2, random_state=0
    )
    scores = model.fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.95


def test_approx_storm_estimates_the_exact_storm_score():
    """With half of the window as the sample, the scores stay close to the exact fractions of non-neighbors."""
    import numpy as np

    from pysad.models import ApproxStorm, ExactStorm

    X, _ = generate_stream()

    approx = ApproxStorm(
        window_size=500, max_radius=0.5, num_neighbors=2, sample_fraction=0.5, random_state=0
    )
    exact = ExactStorm(window_size=500, max_radius=0.5)
    errors = np.abs(approx.fit_score(X) - exact.fit_score(X))[500:]

    assert errors.mean() < 0.02


def test_approx_storm_counts_safe_inliers_only():
    """fract_before is the fraction of the kept safe inliers that are neighbors (Section 4.2 of the paper)."""
    import numpy as np

    from pysad.models import ApproxStorm

    # With k = 3, an instance is a safe inlier once it has two later neighbors: 0.0 is, 0.1 needs one more.
    model = ApproxStorm(window_size=10, max_radius=0.5, num_neighbors=3, sample_fraction=0.5)
    model.fit(np.array([[0.0], [0.1], [0.2], [5.0]]))

    assert (
        model.score_partial(np.array([0.15])) == 0.0
    )  # 0.1 becomes a safe inlier, both are neighbors
    assert (
        model.score_partial(np.array([0.6])) == 0.5
    )  # neighbors 0.1 and 0.2, safe inliers 0.0 and 0.1
    assert model.score_partial(np.array([5.0])) == 1.0  # the neighbor 5.0 is not a safe inlier


def test_approx_storm_keeps_at_most_rho_w_safe_inliers():
    import numpy as np

    from pysad.models import ApproxStorm

    model = ApproxStorm(
        window_size=100, max_radius=10.0, num_neighbors=2, sample_fraction=0.1, random_state=0
    )
    model.fit(np.zeros((100, 1)))

    # All 100 instances are neighbors, the newest one is not a safe inlier yet, and 10 safe inliers stay.
    assert model.kept_.sum() == 11


def test_approx_storm_without_safe_inliers():
    import numpy as np

    from pysad.models import ApproxStorm

    assert ApproxStorm().score_partial(np.array([0.0, 0.0])) == 1.0
    assert ApproxStorm().fit_score_partial(np.array([0.0, 0.0])) == 1.0
    model = ApproxStorm(max_radius=1.0, num_neighbors=4).fit(np.zeros((2, 2)))
    assert model.score_partial(np.zeros(2)) == 1.0


def test_approx_storm_score_partial_skips_the_expiring_instance():
    import numpy as np

    from pysad.models import ApproxStorm

    model = ApproxStorm(window_size=3, max_radius=0.5, num_neighbors=2, sample_fraction=1.0)
    model.fit(np.array([[0.0], [0.1], [5.0]]))

    # The only safe inlier 0.0 expires when the next instance arrives.
    assert model.score_partial(np.array([-0.45])) == 1.0
    assert model.fit_score_partial(np.array([-0.45])) == 1.0


def test_approx_storm_score_partial_matches_fit_score_partial():
    import numpy as np

    from pysad.models import ApproxStorm

    X = np.random.RandomState(3).normal(size=(300, 2))
    model = ApproxStorm(
        window_size=50, max_radius=0.8, num_neighbors=4, sample_fraction=0.2, random_state=0
    )
    for x in X:
        expected = model.score_partial(x)
        assert model.fit_score_partial(x) == expected


def test_approx_storm_sets_the_radius_from_the_first_instances():
    import numpy as np

    from pysad.models import ApproxStorm

    X = np.random.RandomState(0).normal(size=(60, 2))
    model = ApproxStorm(calibration_size=40, random_state=0).fit(X)

    nearest = np.sort(np.linalg.norm(X[:40, None] - X[None, :40], axis=2), axis=1)[:, 1]
    assert model.radius_ == np.quantile(nearest, 0.95)


def test_approx_storm_rejects_an_empty_sample():
    import pytest

    from pysad.models import ApproxStorm

    with pytest.raises(ValueError, match="sample_fraction"):
        ApproxStorm(window_size=10, sample_fraction=0.05)
