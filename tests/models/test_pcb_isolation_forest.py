import time

import numpy as np
import pytest

from pysad.evaluation import AUROCMetric
from pysad.utils import fix_seed


def _make_stream(num_instances, num_features, seed):
    fix_seed(seed)
    return np.random.rand(num_instances, num_features)


def test_unfitted_model_scores_zero():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(num_trees=10, window_size=20, stat_size=5)
    x = np.zeros(3)
    assert model.score_partial(x) == 0.0
    assert model.fit_score_partial(x) == 0.0
    assert model.score_partial(x) == 0.0


def test_window_fills_and_forest_is_built():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(num_trees=10, max_leaf_samples=4, window_size=10, stat_size=3)
    X = _make_stream(40, 2, 0)
    scores = model.fit_score(X)

    assert type(scores) is np.ndarray
    assert scores.shape == (40,)
    assert scores.dtype == np.float64
    assert len(model.trees_) == 10
    # The first window_size points score 0.0, the rest are learned scores.
    assert (scores[:10] == 0.0).all()
    assert scores[10:].min() > 0.0 or scores[10:].max() > 0.0


def test_anomalies_score_higher_than_inliers():
    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    inliers = rng.normal(loc=0.0, scale=0.5, size=(200, 3))
    outlier = np.array([10.0, 10.0, 10.0])

    model = PCBIsolationForest(
        num_trees=20, max_leaf_samples=8, window_size=100, stat_size=20, random_state=0
    )
    for x in inliers:
        model.fit_partial(x)

    inlier_score = model.score_partial(inliers[0])
    outlier_score = model.score_partial(outlier)

    assert outlier_score > inlier_score
    assert 0.0 <= inlier_score <= 1.0
    assert 0.0 <= outlier_score <= 1.0


def test_strong_shift_triggers_drift_and_refit():
    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(200, 2))
    cluster_b = rng.normal(loc=8.0, scale=0.2, size=(200, 2))

    model = PCBIsolationForest(
        num_trees=10, max_leaf_samples=4, window_size=40, stat_size=10, random_state=0
    )
    refit_called = []
    original_refit = model._partial_refit

    def mark_refit():
        refit_called.append(True)
        return original_refit()

    model._partial_refit = mark_refit

    # A uniform stream fills the window but causes no drift and no refit.
    for x in cluster_a[:40]:
        model.fit_partial(x)
    assert not refit_called

    # A strong distribution shift triggers the NDKSWIN drift test and a refit.
    for x in cluster_b[:80]:
        model.fit_partial(x)
    assert refit_called


def test_score_partial_does_not_change_state():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(
        num_trees=10, max_leaf_samples=4, window_size=10, stat_size=3, random_state=0
    )
    X = _make_stream(30, 2, 1)
    model.fit(X)

    counters_before = model.performance_counters_.copy()
    trees_before = len(model.trees_)
    for x in X:
        model.score_partial(x)

    assert len(model.trees_) == trees_before
    np.testing.assert_array_equal(model.performance_counters_, counters_before)


def test_fit_partial_updates_performance_counters():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(
        num_trees=10, max_leaf_samples=4, window_size=10, stat_size=3, random_state=0
    )
    X = _make_stream(30, 2, 2)
    model.fit(X)  # builds the forest

    counters_before = model.performance_counters_.copy()
    model.fit_partial(X[0])
    assert not np.array_equal(model.performance_counters_, counters_before)


def test_parameter_validation():
    from pysad.models import PCBIsolationForest

    invalid = [
        {"num_trees": 0},
        {"max_leaf_samples": 0},
        {"stat_size": 0},
        {"window_size": 1, "stat_size": 1},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"growth_criterion": "invalid"},
        {"subsample": 0.0},
        {"subsample": 1.5},
        {"branching_factor": 1},
    ]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            PCBIsolationForest(**kwargs)


def test_pickling_preserves_scoring():
    import pickle

    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 3))

    model = PCBIsolationForest(
        num_trees=10,
        max_leaf_samples=8,
        window_size=32,
        stat_size=10,
        growth_criterion="fixed",
        branching_factor=4,
        subsample=0.5,
        random_state=42,
    )
    for x in X[:40]:
        model.fit_partial(x)

    restored = pickle.loads(pickle.dumps(model))

    scores_orig = [model.fit_score_partial(x) for x in X[40:]]
    scores_restored = [restored.fit_score_partial(x) for x in X[40:]]

    np.testing.assert_allclose(scores_orig, scores_restored)


def test_same_seed_gives_same_scores():
    from pysad.models import PCBIsolationForest

    X = _make_stream(80, 3, 0)

    model_a = PCBIsolationForest(random_state=5).fit_score(X)
    model_b = PCBIsolationForest(random_state=5).fit_score(X)

    np.testing.assert_array_equal(model_a, model_b)


def test_different_seeds_give_different_scores():
    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 3))

    scores_a = PCBIsolationForest(
        num_trees=20, max_leaf_samples=4, window_size=32, stat_size=10, random_state=5
    ).fit_score(X)
    scores_b = PCBIsolationForest(
        num_trees=20, max_leaf_samples=4, window_size=32, stat_size=10, random_state=6
    ).fit_score(X)

    assert not np.array_equal(scores_a, scores_b)


def test_auroc_above_chance_on_a_test_stream():
    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    normal = rng.normal(loc=0.0, scale=0.5, size=(400, 3))
    anomaly = rng.normal(loc=5.0, scale=0.5, size=(100, 3))
    X = np.vstack([normal, anomaly])
    y = np.array([0] * 400 + [1] * 100)

    model = PCBIsolationForest(num_trees=20, max_leaf_samples=8, window_size=100, random_state=0)
    metric = AUROCMetric()
    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    assert metric.get() > 0.5


def test_score_stays_under_five_ms_per_instance():
    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    model = PCBIsolationForest(num_trees=50, max_leaf_samples=8, window_size=200, random_state=0)
    X = rng.normal(size=(1500, 5))
    for x in X[:500]:
        model.fit_partial(x)

    start = time.perf_counter()
    for x in X[500:]:
        model.score_partial(x)
    elapsed = time.perf_counter() - start

    num_points = len(X) - 500
    ms_per_point = (elapsed / num_points) * 1000
    assert ms_per_point < 5.0


def test_memory_stays_bounded_over_a_long_stream():
    import tracemalloc

    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(
        num_trees=2, max_leaf_samples=4, window_size=20, stat_size=5, random_state=0
    )
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
