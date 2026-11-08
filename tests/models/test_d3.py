def test_d3_auroc():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from pysad.models import D3

    rng = np.random.RandomState(61)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)
    outliers = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outliers] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outliers] = 1

    scores = D3(window_size=500, max_radius=0.5, random_state=0).fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.95


def test_d3_estimates_the_number_of_window_values_within_the_radius():
    import numpy as np

    from pysad.models import D3

    rng = np.random.RandomState(0)
    X = rng.normal(size=(3000, 2))
    model = D3(window_size=1000, sample_fraction=0.5, max_radius=0.25, random_state=0).fit(X)

    points = rng.uniform(-2.0, 2.0, size=(50, 2))
    estimates = -model.score(points)
    # The neighborhood is the box of half width max_radius around the point.
    counts = (np.abs(X[-1000:, None] - points) <= 0.25).all(axis=2).sum(axis=0)

    np.testing.assert_allclose(estimates.mean(), counts.mean(), rtol=0.1)
    assert np.corrcoef(estimates, counts)[0, 1] > 0.9


def test_d3_forgets_values_that_left_the_window():
    import numpy as np

    from pysad.models import D3

    rng = np.random.RandomState(0)
    model = D3(window_size=100, max_radius=0.5, random_state=0)
    model.fit(rng.normal(0.0, 1.0, size=(100, 1)))
    assert model.score_partial(np.zeros(1)) < -10.0

    model.fit(rng.normal(50.0, 1.0, size=(100, 1)))
    assert model.score_partial(np.zeros(1)) == 0.0
    assert model.score_partial(np.full(1, 50.0)) < -10.0


def test_d3_counts_equal_values_of_a_constant_feature():
    import numpy as np

    from pysad.models import D3

    rng = np.random.RandomState(0)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    model = D3(window_size=100, max_radius=0.5, random_state=0).fit(X)

    inside, outside = model.score(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert np.isfinite(inside) and inside < -10.0
    assert outside == 0.0


def test_d3_scores_0_before_any_fit():
    import numpy as np

    from pysad.models import D3

    assert D3().score_partial(np.ones(3)) == 0.0
