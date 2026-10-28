import pytest


@pytest.mark.parametrize("kwargs", [{"num_trees": 0}, {"max_leaf_samples": 0}, {"window_size": 32}])
def test_invalid_parameters(kwargs):
    from pysad.models import OnlineIsolationForest

    with pytest.raises(ValueError):
        OnlineIsolationForest(**kwargs)


def test_random_state_reproducibility():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 3))

    model1 = OnlineIsolationForest(num_trees=16, random_state=123)
    model2 = OnlineIsolationForest(num_trees=16, random_state=123)
    model3 = OnlineIsolationForest(num_trees=16, random_state=456)

    scores1 = model1.fit_score(X)
    scores2 = model2.fit_score(X)
    scores3 = model3.fit_score(X)

    np.testing.assert_array_equal(scores1, scores2)
    assert not np.array_equal(scores1, scores3)


def test_global_seed_reproducibility():
    import numpy as np

    from pysad.models import OnlineIsolationForest
    from pysad.utils import fix_seed

    rng = np.random.RandomState(42)
    X = rng.normal(size=(50, 3))

    fix_seed(999)
    model1 = OnlineIsolationForest(num_trees=16, random_state=None)
    scores1 = model1.fit_score(X)

    fix_seed(999)
    model2 = OnlineIsolationForest(num_trees=16, random_state=None)
    scores2 = model2.fit_score(X)

    np.testing.assert_array_equal(scores1, scores2)


def test_anomalies_score_higher_than_inliers():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(0)
    inliers = rng.normal(loc=0.0, scale=0.5, size=(300, 3))
    outlier = np.array([10.0, 10.0, 10.0])

    model = OnlineIsolationForest(
        num_trees=32, max_leaf_samples=16, window_size=500, random_state=0
    )
    for x in inliers:
        model.fit_partial(x)

    inlier_score = model.score_partial(inliers[0])
    outlier_score = model.score_partial(outlier)

    assert outlier_score > inlier_score
    assert 0.0 <= inlier_score <= 1.0
    assert 0.0 <= outlier_score <= 1.0


def test_window_forgetting():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    window_size = 100
    model = OnlineIsolationForest(
        num_trees=16, max_leaf_samples=8, window_size=window_size, random_state=0
    )

    rng = np.random.RandomState(0)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(window_size, 2))
    cluster_b = rng.normal(loc=10.0, scale=0.2, size=(window_size, 2))

    # Fit cluster A
    for x in cluster_a:
        model.fit_partial(x)

    # Point in cluster A is normal, point in cluster B is anomalous
    score_a_initial = model.score_partial(np.array([0.0, 0.0]))
    score_b_initial = model.score_partial(np.array([10.0, 10.0]))
    assert score_b_initial > score_a_initial

    # Now stream cluster B for full window size, forgetting cluster A
    for x in cluster_b:
        model.fit_partial(x)

    # Now cluster B is normal and cluster A is anomalous
    score_a_after = model.score_partial(np.array([0.0, 0.0]))
    score_b_after = model.score_partial(np.array([10.0, 10.0]))
    assert score_a_after > score_b_after


def test_pickling_preserves_scoring():
    import pickle

    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 3))

    model = OnlineIsolationForest(num_trees=16, max_leaf_samples=8, window_size=50, random_state=42)
    for x in X[:40]:
        model.fit_partial(x)

    pickled = pickle.loads(pickle.dumps(model))

    scores_orig = [model.fit_score_partial(x) for x in X[40:]]
    scores_pickled = [pickled.fit_score_partial(x) for x in X[40:]]

    np.testing.assert_allclose(scores_orig, scores_pickled)


def test_scores_use_the_papers_window_normalization():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=1, max_leaf_samples=2, window_size=8, random_state=0)
    model.fit(np.array([[0.0], [2.0]]))

    # The root splits at two observations. Both leaves have depth one, and
    # c(window_size, max_leaf_samples) = log2(8 / 2) = 2.
    np.testing.assert_allclose(model.score(np.array([[0.0], [2.0]])), 2.0 ** (-1.0 / 2.0))


def test_reusing_an_input_buffer_keeps_the_same_scores():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    X = np.random.RandomState(0).normal(size=(300, 2))
    params = dict(num_trees=4, max_leaf_samples=4, window_size=20, random_state=0)
    expected = OnlineIsolationForest(**params).fit_score(X)
    model = OnlineIsolationForest(**params)
    buffer = np.empty(2)
    actual = []
    for x in X:
        buffer[:] = x
        actual.append(model.fit_score_partial(buffer))

    np.testing.assert_array_equal(actual, expected)


def test_unfitted_model_scores_zero():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    assert OnlineIsolationForest().score_partial(np.array([1.0, 2.0])) == 0.0


def test_single_observation_can_trigger_a_split():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=1, max_leaf_samples=1, window_size=4, random_state=0)
    model.fit_partial(np.array([1.0]))
    assert model.score_partial(np.array([1.0])) == pytest.approx(2.0 ** (-1.0 / 2.0))


def test_memory_stays_bounded_over_a_long_stream():
    import tracemalloc

    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=2, max_leaf_samples=4, window_size=20, random_state=0)
    rng = np.random.RandomState(0)
    tracemalloc.start()
    try:
        for _ in range(2000):
            model.fit_score_partial(rng.rand(3))
        initial = tracemalloc.get_traced_memory()[0]
        for _ in range(10000):
            model.fit_score_partial(rng.rand(3))
        final = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert final - initial < 64 * 1024
