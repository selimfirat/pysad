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

    window_size = 10

    fix_seed(42)
    initial_window_X = np.random.rand(window_size, 2)
    test_X = np.random.rand(5, 2)

    def new_model(**kwargs):
        fix_seed(42)
        return HalfSpaceTrees(feature_mins=[0, 0], feature_maxes=[1, 1],
                              window_size=window_size, num_trees=5, max_depth=5, **kwargs)

    model_with_window = new_model(initial_window_X=window_transform(initial_window_X))
    model_without_window = new_model().fit(initial_window_X)
    unfitted_model = new_model()

    # An initial window of window_size instances is the first window, so it becomes the reference.
    assert model_with_window.is_first_window is False

    scores_with_window = np.array([model_with_window.score_partial(x) for x in test_X])
    scores_without_window = np.array([model_without_window.score_partial(x) for x in test_X])
    scores_unfitted = np.array([unfitted_model.score_partial(x) for x in test_X])

    np.testing.assert_array_equal(scores_with_window, scores_without_window)
    # The initial window must have been fitted, otherwise the models above would match trivially.
    assert np.all(scores_with_window != scores_unfitted)


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
        scores = HalfSpaceTrees(feature_mins=[-7, -7], feature_maxes=[7, 7], window_size=100,
                                num_trees=10, max_depth=12).fit_score(X_)
        return np.mean(np.delete(scores, position) < scores[position])

    # Both the point right before a window closes and the point that closes it must be
    # ranked as highly anomalous; before the fix, the latter scored as normal (rank 0.59 with
    # these trees, 0.12 with the default 25 trees of depth 15).
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


def test_half_space_trees_scores_against_the_first_window_right_after_it_closes():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(2 * window_size, 2))
    model = HalfSpaceTrees(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
                           window_size=window_size, num_trees=5, max_depth=6)
    scores = model.fit_score(X)

    # From the instance right after the first window on, the reference is the whole first window.
    for x, score in zip(X[window_size:], scores[window_size:]):
        assert score == _score_against(model, X[:window_size], x)


def test_half_space_trees_reference_profile_is_fixed_within_a_window():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(2 * window_size, 2))
    probe = np.array([0.3, 0.6])
    model = HalfSpaceTrees(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
                           window_size=window_size, num_trees=5, max_depth=6)
    model.fit(X[:window_size])
    first_reference_score = model.score_partial(probe)

    # Recording the instances of the second window must not change the reference until that window closes.
    for x in X[window_size:-1]:
        model.fit_partial(x)
        assert model.score_partial(probe) == first_reference_score

    model.fit_partial(X[-1])
    assert model.score_partial(probe) != first_reference_score


def test_half_space_trees_window_swap_replaces_the_reference_with_the_last_window():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(4 * window_size, 2))
    model = HalfSpaceTrees(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
                           window_size=window_size, num_trees=5, max_depth=6)

    # The instance that closes a window belongs to the reference that window becomes.
    model.fit(X[:window_size])
    assert all(root.r_mass == window_size and root.l_mass == 0 for root in model.roots)

    # Older windows are forgotten: the reference holds only the last window.
    model.fit(X[window_size:3 * window_size])
    assert all(root.r_mass == window_size and root.l_mass == 0 for root in model.roots)
    scores = model.fit_score(X[3 * window_size:])
    for x, score in zip(X[3 * window_size:], scores):
        assert score == _score_against(model, X[2 * window_size:3 * window_size], x)


def test_half_space_trees_score_partial_does_not_record_the_instance():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(3 * window_size, 2))
    extra_X = np.random.uniform(size=(10, 2))

    def later_scores(num_fitted, score_extra):
        fix_seed(1)
        model = HalfSpaceTrees(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0],
                               window_size=window_size, num_trees=5, max_depth=6)
        model.fit(X[:num_fitted])
        if score_extra:
            model.score(extra_X)
        return model.fit_score(X[num_fitted:])

    # Scoring extra instances, during the first window or after it, must leave every later score unchanged.
    for num_fitted in [window_size // 2, window_size + window_size // 2]:
        np.testing.assert_array_equal(later_scores(num_fitted, score_extra=True),
                                      later_scores(num_fitted, score_extra=False))


def test_half_space_trees_fit_score_matches_scoring_then_fitting_each_instance():
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    fix_seed(7)
    X = np.random.uniform(size=(250, 3))

    def new_model():
        fix_seed(123)
        return HalfSpaceTrees(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3,
                              window_size=50, num_trees=5, max_depth=6)

    model = new_model()
    expected_scores = []
    for x in X:
        expected_scores.append(model.score_partial(x))
        model.fit_partial(x)

    np.testing.assert_array_equal(new_model().fit_score(X), expected_scores)
