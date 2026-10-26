import pytest


def test_invalid_parameters():
    from pysad.models import OnlineIsolationForest

    with pytest.raises(ValueError, match="num_trees"):
        OnlineIsolationForest(num_trees=0)
    with pytest.raises(ValueError, match="num_trees"):
        OnlineIsolationForest(num_trees=True)

    with pytest.raises(ValueError, match="max_leaf_samples"):
        OnlineIsolationForest(max_leaf_samples=0)
    with pytest.raises(ValueError, match="max_leaf_samples"):
        OnlineIsolationForest(max_leaf_samples=False)

    with pytest.raises(ValueError, match="window_size"):
        OnlineIsolationForest(window_size=0)
    with pytest.raises(ValueError, match="window_size"):
        OnlineIsolationForest(window_size=True)

    with pytest.raises(ValueError, match="growth_criterion"):
        OnlineIsolationForest(growth_criterion="invalid")

    with pytest.raises(ValueError, match="subsample"):
        OnlineIsolationForest(subsample=0.0)
    with pytest.raises(ValueError, match="subsample"):
        OnlineIsolationForest(subsample=1.5)
    with pytest.raises(ValueError, match="subsample"):
        OnlineIsolationForest(subsample=True)

    with pytest.raises(ValueError, match="branching_factor"):
        OnlineIsolationForest(branching_factor=1)

    with pytest.raises(ValueError, match="split"):
        OnlineIsolationForest(split="oblique")


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


def test_memory_bounded_by_window_size():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    window_size = 50
    model = OnlineIsolationForest(
        num_trees=8, max_leaf_samples=8, window_size=window_size, random_state=0
    )
    rng = np.random.RandomState(0)

    for _ in range(150):
        x = rng.normal(size=(3,))
        model.fit_partial(x)
        assert len(model.data_window) <= window_size
        assert model.data_size <= window_size


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


def test_growth_criteria():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    for criterion in ("adaptive", "fixed"):
        model = OnlineIsolationForest(
            num_trees=8, max_leaf_samples=8, growth_criterion=criterion, random_state=0
        )
        rng = np.random.RandomState(0)
        X = rng.normal(size=(60, 2))
        scores = model.fit_score(X)
        assert len(scores) == 60
        assert np.all((scores >= 0.0) & (scores <= 1.0))


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


def test_branching_factor_greater_than_two():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(42)
    X = rng.normal(size=(80, 2))
    model = OnlineIsolationForest(
        num_trees=4,
        max_leaf_samples=4,
        branching_factor=4,
        window_size=20,
        random_state=42,
    )
    for x in X:
        score = model.fit_score_partial(x)
        assert 0.0 <= score <= 1.0


def test_subsample():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(42)
    X = rng.normal(size=(60, 2))
    model = OnlineIsolationForest(
        num_trees=4,
        max_leaf_samples=4,
        subsample=0.5,
        window_size=20,
        random_state=42,
    )
    for x in X:
        model.fit_partial(x)
        score = model.score_partial(x)
        assert 0.0 <= score <= 1.0


def test_empty_tree_predict():
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=8, branching_factor=2, growth_criterion="adaptive", subsample=1.0
    )
    assert tree.predict_one(np.array([1.0, 2.0])) == 0.0
