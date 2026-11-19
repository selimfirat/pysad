import time

import numpy as np
import pytest


def test_initial_window_scores_zero():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(num_trees=3, window_size=8, stat_size=2, random_state=0)
    scores = model.fit_score(np.arange(12.0).reshape(-1, 1))
    np.testing.assert_array_equal(scores[:8], 0.0)
    assert (scores[8:] > 0.0).all()


def test_algorithm_one_scores_and_counters_across_partial_replacements():
    from pysad.models import PCBIsolationForest

    # Independent scikit-learn tree paths and SciPy KS distances give these values.
    # The stream replaces only tree 0 at row 13 and only tree 2 at row 16.
    expected = {
        8: (0.44129165461730385, [-1, 1, 1]),
        9: (0.5736508771881734, [0, 2, 2]),
        13: (0.5815233990425025, [0, 0, 0]),
        14: (0.4587266145157976, [1, -1, 1]),
        15: (0.47666947203590376, [2, 0, 0]),
        16: (0.47666947203590376, [0, 0, 0]),
        17: (0.44129165461730385, [0, 0, 0]),
        18: (0.43515769108599534, [1, 1, -1]),
        20: (0.401518431990416, [0, 0, 0]),
        39: (0.5338776545612856, [1, 1, -1]),
        40: (0.6151626581380819, [2, 2, 0]),
        60: (0.6151626581380819, [0, 0, 0]),
        79: (0.6151626581380819, [1, 1, 1]),
    }
    X = np.random.RandomState(7).normal(size=(80, 2))
    X[40:] += 3
    model = PCBIsolationForest(num_trees=3, window_size=8, stat_size=2, alpha=0.5, random_state=42)
    for i, x in enumerate(X):
        score = model.fit_score_partial(x)
        if i in expected:
            value, counters = expected[i]
            assert score == pytest.approx(value)
            np.testing.assert_array_equal(model.performance_counters_, counters)


def test_score_stays_under_five_ms_per_instance():
    from pysad.models import PCBIsolationForest

    X = np.random.RandomState(0).normal(size=(1200, 5))
    model = PCBIsolationForest(random_state=0).fit(X[:200])
    start = time.perf_counter()
    for x in X[200:]:
        model.score_partial(x)
    assert (time.perf_counter() - start) / 1000 < 0.005


def test_auroc_above_chance_on_a_test_stream():
    from sklearn.metrics import roc_auc_score

    from pysad.models import PCBIsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(scale=0.5, size=(600, 3))
    labels = np.zeros(600, dtype=int)
    labels[250::10] = 1
    X[labels == 1] += 5
    model = PCBIsolationForest(num_trees=20, window_size=100, random_state=0)
    scores = model.fit_score(X)
    assert roc_auc_score(labels[100:], scores[100:]) > 0.8


def test_memory_stays_bounded_over_a_long_stream():
    import pickle

    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(num_trees=3, window_size=20, stat_size=5, random_state=0)
    rng = np.random.RandomState(0)
    model.fit(rng.normal(size=(100, 3)))
    initial_size = len(pickle.dumps(model))
    model.fit(rng.normal(size=(2000, 3)))
    assert len(pickle.dumps(model)) <= initial_size + 1024


def test_duplicate_leaf_path_correction():
    from pysad.models import PCBIsolationForest

    # Two leaves hold two points each. Each path has length 1 + c(2) = 2.
    model = PCBIsolationForest(num_trees=3, window_size=4, stat_size=2, random_state=0)
    model.fit(np.array([[0.0], [0.0], [1.0], [1.0]]))
    correction = 2 * (np.log(3) + np.euler_gamma) - 1.5
    expected = 2 ** (-2 / correction)
    assert model.score_partial(np.array([0.0])) == pytest.approx(expected)


def test_constant_window_scores_half_and_threshold_ties_lose_credit():
    from pysad.models import PCBIsolationForest

    model = PCBIsolationForest(num_trees=3, window_size=8, stat_size=2, random_state=0)
    model.fit(np.ones((8, 1)))
    assert model.score_partial(np.ones(1)) == 0.5
    model.fit_partial(np.ones(1))
    np.testing.assert_array_equal(model.performance_counters_, [-1, -1, -1])
