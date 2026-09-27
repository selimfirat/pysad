def _score_against(model, reference_X, x):
    """Scores `x` with the model's trees as if their mass profile held exactly the instances in `reference_X`.

    The masses are counted here from the tree splits alone, so the result does not depend on the
    masses the model has recorded.
    """
    score = 0.0
    for root in model.roots:
        node, in_node = root, list(reference_X)
        while node is not None:
            score += len(in_node) * 2 ** node.k
            goes_right = x[node.split_att] > node.split_value
            in_node = [r for r in in_node if (r[node.split_att] > node.split_value) == goes_right]
            node = node.right if goes_right else node.left

    return -score


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


def test_half_space_trees_first_window_scores_against_the_instances_before_it():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(window_size, 2))
    model = HalfSpaceTrees(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
                           window_size=window_size, num_trees=5, max_depth=6)
    scores = model.fit_score(X)

    # Nothing has been recorded when the very first instance arrives.
    assert scores[0] == 0.0
    # Every later instance of the first window, including the one that closes it, is scored
    # against the instances before it, without its own mass.
    for i in range(1, window_size):
        assert scores[i] == _score_against(model, X[:i], X[i])
    assert model.is_first_window is False


def test_half_space_trees_scores_a_batch_fitted_on_less_than_a_window():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (60, 2))
    X_test = rng.normal(0, 1, (40, 2))
    X_test[7] = [6.0, 6.0]

    fix_seed(0)
    model = HalfSpaceTrees(feature_mins=[-7, -7], feature_maxes=[7, 7], window_size=100, num_trees=5, max_depth=6)
    scores = model.fit(X_train).score(X_test)

    # The first window is still open, so the test batch is scored against the 60 instances fitted so far.
    assert model.is_first_window is True
    for x, score in zip(X_test, scores):
        assert score == _score_against(model, X_train, x)
    assert np.all(np.delete(scores, 7) < scores[7])


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
