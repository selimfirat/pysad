def _verify_initial_window_reproducibility(window_transform):
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    fix_seed(42)
    initial_window_X = np.random.rand(10, 2)
    test_X = np.random.rand(5, 2)

    fix_seed(42)
    model_with_window = HalfSpaceTrees(feature_mins=[0, 0], feature_maxes=[1, 1],
                                       num_trees=5, max_depth=5,
                                       initial_window_X=window_transform(initial_window_X))

    fix_seed(42)
    model_without_window = HalfSpaceTrees(feature_mins=[0, 0], feature_maxes=[1, 1],
                                          num_trees=5, max_depth=5)
    model_without_window.fit(initial_window_X)

    scores_with_window = np.array([model_with_window.score_partial(x) for x in test_X])
    scores_without_window = np.array([model_without_window.score_partial(x) for x in test_X])

    np.testing.assert_array_equal(scores_with_window, scores_without_window)
    # A fitted model must score differently from an unfitted one, otherwise this test would pass trivially.
    assert np.any(scores_with_window != 0.0)


def test_half_space_trees_with_numpy_initial_window():
    _verify_initial_window_reproducibility(lambda x: x)


def test_half_space_trees_with_list_initial_window():
    _verify_initial_window_reproducibility(lambda x: x.tolist())


def test_half_space_trees_scores_outlier_that_closes_a_window():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    X = np.random.default_rng(0).normal(0, 1, (1000, 2))

    def rank_of_outlier(position):
        """Share of the other points that score lower than a far outlier placed at `position`."""
        X_ = X.copy()
        X_[position] = [6.0, 6.0]
        fix_seed(0)
        scores = HalfSpaceTrees(feature_mins=[-7, -7], feature_maxes=[7, 7], window_size=100).fit_score(X_)
        return np.mean(np.delete(scores, position) < scores[position])

    # Both the point right before a window closes and the point that closes it must be
    # ranked as highly anomalous; before the fix, the latter scored as normal (rank 0.12).
    assert rank_of_outlier(498) >= 0.95
    assert rank_of_outlier(499) >= 0.95


def test_half_space_trees_first_window_scores_are_the_minimum():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20
    num_trees = 5
    max_depth = 4

    fix_seed(0)
    model = HalfSpaceTrees(
        feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
        window_size=window_size, num_trees=num_trees, max_depth=max_depth)

    expected_min_score = -num_trees * window_size * (2 ** (max_depth + 1) - 1)
    assert model._min_score == expected_min_score

    X = np.random.uniform(size=(window_size, 2))
    scores = model.fit_score(X)

    # Algorithm 3 does not score the first window at all; every instance in it gets the
    # documented lowest possible score, including the instance that closes the window.
    assert all(score == expected_min_score for score in scores)
    assert model.is_first_window is False


def test_half_space_trees_score_then_fit_matches_fit_score_partial():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    fix_seed(7)
    X = np.random.uniform(size=(250, 3))

    fix_seed(123)
    model_a = HalfSpaceTrees(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3, window_size=50)
    scores_a = []
    for x in X:
        scores_a.append(model_a.score_partial(x))
        model_a.fit_partial(x)

    fix_seed(123)
    model_b = HalfSpaceTrees(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3, window_size=50)
    scores_b = [model_b.fit_score_partial(x) for x in X]

    np.testing.assert_allclose(scores_a, scores_b)
