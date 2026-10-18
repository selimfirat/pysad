import pytest


def _score_against(model, reference_X, x):
    """Scores `x` with the model's trees as if their mass profile held exactly the instances in `reference_X`.

    The masses are counted here from the tree splits alone, so the result does not depend on the
    masses the model has recorded.
    """
    score = 0.0
    for atts, values in zip(model.split_atts, model.split_values, strict=True):
        node, in_node = 0, list(reference_X)
        for k in range(model.max_depth + 1):
            score += len(in_node) * 2**k
            if k == model.max_depth:
                break
            att, value = atts[node], values[node]
            goes_right = x[att] > value
            in_node = [r for r in in_node if (r[att] > value) == goes_right]
            node = 2 * node + 1 + goes_right

    return -score


def _new_hst(**kwargs):
    from pysad.models import HalfSpaceTrees

    params = dict(
        feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0], window_size=10, num_trees=3, max_depth=4
    )
    params.update(kwargs)
    return HalfSpaceTrees(**params)


def _verify_initial_window_reproducibility(window_transform):
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 10

    fix_seed(42)
    initial_window_X = np.random.rand(window_size, 2)
    test_X = np.random.rand(5, 2)

    def new_model(**kwargs):
        fix_seed(42)
        return _new_hst(window_size=window_size, num_trees=5, max_depth=5, **kwargs)

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
        scores = HalfSpaceTrees(
            feature_mins=[-7, -7], feature_maxes=[7, 7], window_size=100, num_trees=10, max_depth=12
        ).fit_score(X_)
        return np.mean(np.delete(scores, position) < scores[position])

    # Both the point right before a window closes and the point that closes it must be
    # ranked as highly anomalous; before the fix, the latter scored as normal (rank 0.59 with
    # these trees, 0.12 with the default 25 trees of depth 15).
    assert rank_of_outlier(498) >= 0.95
    assert rank_of_outlier(499) >= 0.95


def test_half_space_trees_first_window_scores_against_the_instances_before_it():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(window_size, 2))
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
    scores = model.fit_score(X)

    # Nothing has been recorded when the very first instance arrives.
    assert scores[0] == 0.0
    # Every later instance of the first window, including the one that closes it, is scored
    # against the instances before it, without its own mass, rescaled to a full window.
    for i in range(1, window_size):
        assert scores[i] == _score_against(model, X[:i], X[i]) * (window_size / i)
    assert model.is_first_window is False


def test_half_space_trees_first_window_scores_match_the_scale_of_a_full_window():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    # On a constant stream, a partial profile of n copies rescaled to a full window equals the
    # reference profile of a full window, so every score after the very first is the same.
    X = np.full((3 * window_size, 2), 0.3)
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
    scores = model.fit_score(X)

    assert scores[0] == 0.0
    assert np.all(scores[1:] == scores[window_size])
    assert scores[window_size] == _score_against(model, X[:window_size], X[0])


def test_half_space_trees_scores_a_batch_fitted_on_less_than_a_window():
    import numpy as np

    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (60, 2))
    X_test = rng.normal(0, 1, (40, 2))
    X_test[7] = [6.0, 6.0]

    fix_seed(0)
    model = HalfSpaceTrees(
        feature_mins=[-7, -7], feature_maxes=[7, 7], window_size=100, num_trees=5, max_depth=6
    )
    scores = model.fit(X_train).score(X_test)

    # The first window is still open, so the test batch is scored against the 60 instances fitted so far,
    # rescaled to a full window of 100.
    assert model.is_first_window is True
    for x, score in zip(X_test, scores, strict=True):
        assert score == _score_against(model, X_train, x) * (100 / 60)
    assert np.all(np.delete(scores, 7) < scores[7])


def test_half_space_trees_scores_against_the_first_window_right_after_it_closes():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(2 * window_size, 2))
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
    scores = model.fit_score(X)

    # From the instance right after the first window on, the reference is the whole first window.
    for x, score in zip(X[window_size:], scores[window_size:], strict=True):
        assert score == _score_against(model, X[:window_size], x)


def test_half_space_trees_reference_profile_is_fixed_within_a_window():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(2 * window_size, 2))
    probe = np.array([0.3, 0.6])
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
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

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(4 * window_size, 2))
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)

    def root_masses():
        return list(zip(model.r_mass[:, 0], model.l_mass[:, 0], strict=True))

    # The instance that closes a window belongs to the reference that window becomes.
    model.fit(X[:window_size])
    assert all(masses == (window_size, 0) for masses in root_masses())

    # Older windows are forgotten: the reference holds only the last window.
    model.fit(X[window_size : 3 * window_size])
    assert all(masses == (window_size, 0) for masses in root_masses())
    scores = model.fit_score(X[3 * window_size :])
    for x, score in zip(X[3 * window_size :], scores, strict=True):
        assert score == _score_against(model, X[2 * window_size : 3 * window_size], x)


def test_half_space_trees_score_partial_does_not_record_the_instance():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 20

    fix_seed(0)
    X = np.random.uniform(size=(3 * window_size, 2))
    extra_X = np.random.uniform(size=(10, 2))

    def later_scores(num_fitted, score_extra):
        fix_seed(1)
        model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
        model.fit(X[:num_fitted])
        if score_extra:
            model.score(extra_X)
        return model.fit_score(X[num_fitted:])

    # Scoring extra instances, during the first window or after it, must leave every later score unchanged.
    for num_fitted in [window_size // 2, window_size + window_size // 2]:
        np.testing.assert_array_equal(
            later_scores(num_fitted, score_extra=True), later_scores(num_fitted, score_extra=False)
        )


def test_half_space_trees_fit_score_matches_scoring_then_fitting_each_instance():
    import numpy as np

    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    fix_seed(7)
    X = np.random.uniform(size=(250, 3))

    def new_model():
        fix_seed(123)
        return HalfSpaceTrees(
            feature_mins=[0.0] * 3,
            feature_maxes=[1.0] * 3,
            window_size=50,
            num_trees=5,
            max_depth=6,
        )

    model = new_model()
    expected_scores = []
    for x in X:
        expected_scores.append(model.score_partial(x))
        model.fit_partial(x)

    np.testing.assert_array_equal(new_model().fit_score(X), expected_scores)


@pytest.mark.parametrize("name", ["window_size", "num_trees", "max_depth"])
@pytest.mark.parametrize("value", [0, -1, 2.5, 3.0, "3", None, True])
def test_half_space_trees_rejects_invalid_hyperparameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        _new_hst(**{name: value})


def test_half_space_trees_accepts_numpy_integer_hyperparameters():
    import numpy as np

    model = _new_hst(window_size=np.int64(10), num_trees=np.int32(3), max_depth=np.uint8(1))

    assert model.split_values.shape == (3, 1)
    assert model.fit_score(np.random.uniform(size=(25, 2))).shape == (25,)


@pytest.mark.parametrize(
    "feature_mins, feature_maxes",
    [
        ([0.0, 0.0], [1.0]),
        ([], []),
        ([[0.0, 0.0]], [[1.0, 1.0]]),
        ([0.0, 2.0], [1.0, 1.0]),
        ([0.0, float("-inf")], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, float("nan")]),
    ],
)
def test_half_space_trees_rejects_invalid_feature_bounds(feature_mins, feature_maxes):
    with pytest.raises(ValueError, match="feature_mins"):
        _new_hst(feature_mins=feature_mins, feature_maxes=feature_maxes)


def test_half_space_trees_integer_bounds_build_the_same_trees_as_float_bounds():
    import numpy as np

    from pysad.utils import fix_seed

    def split_values(feature_mins, feature_maxes):
        fix_seed(0)
        model = _new_hst(
            feature_mins=feature_mins, feature_maxes=feature_maxes, num_trees=5, max_depth=6
        )
        return model.split_values.tolist()

    # Integer arrays used to truncate every split value written back into them.
    float_splits = split_values(np.array([0.0, -3.0]), np.array([10.0, 3.0]))
    assert split_values(np.array([0, -3]), np.array([10, 3])) == float_splits


def test_half_space_trees_work_spaces_cover_the_feature_ranges():
    import numpy as np

    from pysad.utils import fix_seed

    fix_seed(0)
    feature_mins, feature_maxes = np.array([0.0, -5.0, 2.0]), np.array([1.0, 5.0, 2.0])
    model = _new_hst(feature_mins=feature_mins, feature_maxes=feature_maxes)
    width = feature_maxes - feature_mins
    rng = np.random.RandomState(0)

    for _ in range(100):
        mins, maxes = model._work_space(rng)
        center = (mins + maxes) / 2
        # Tan et al. (IJCAI 2011): s in [min, max] and a work range s +- 2 * max(s - min, max - s),
        # so it holds the whole feature range and is 2 to 4 times as wide.
        assert np.all((feature_mins <= center) & (center <= feature_maxes))
        assert np.all((mins <= feature_mins) & (feature_maxes <= maxes))
        np.testing.assert_allclose(
            maxes - center, 2 * np.maximum(center - feature_mins, feature_maxes - center)
        )
        assert np.all((2 * width <= maxes - mins) & (maxes - mins <= 4 * width))


def test_half_space_trees_differ_on_one_dimensional_streams():
    from pysad.utils import fix_seed

    fix_seed(0)
    model = _new_hst(feature_mins=[0.0], feature_maxes=[1.0], num_trees=10, max_depth=5)

    # Without a work space per tree, every tree halved [0, 1] at the same points (0.5, 0.25, 0.75, ...).
    assert len({tuple(values) for values in model.split_values}) == 10
    assert len(set(model.split_values[:, 0])) == 10


def test_half_space_trees_without_random_work_space_split_the_feature_ranges():
    import numpy as np

    from pysad.utils import fix_seed

    fix_seed(0)
    model = _new_hst(feature_mins=[0, 2], feature_maxes=[1, 6], random_work_space=False)
    mins, maxes = model._work_space(np.random.RandomState(0))

    np.testing.assert_array_equal(mins, [0.0, 2.0])
    np.testing.assert_array_equal(maxes, [1.0, 6.0])
    assert set(model.split_values[:, 0]) <= {0.5, 4.0}


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_half_space_trees_rejects_a_non_bool_random_work_space(value):
    with pytest.raises(ValueError, match="random_work_space must be a bool"):
        _new_hst(random_work_space=value)


def test_half_space_trees_forgets_regions_left_empty_for_a_whole_window():
    import numpy as np

    from pysad.utils import fix_seed

    window_size = 10

    fix_seed(0)
    low = np.random.uniform(0.0, 0.2, size=(window_size, 2))
    high = np.random.uniform(0.8, 1.0, size=(2 * window_size, 2))
    model = _new_hst(window_size=window_size, num_trees=5, max_depth=6)
    model.fit(low)
    model.fit(high)

    # The nodes under the first window's region are not visited during the next two windows, so their
    # deferred swap must not carry the first window's mass into the reference.
    for x in np.vstack([low[:3], high[:3]]):
        assert model.score_partial(x) == _score_against(model, high[window_size:], x)


def test_half_space_trees_match_a_recursive_build():
    import numpy as np

    from pysad.utils import fix_seed

    def recursive_build(model, mins, maxes, depth, rng):
        """Builds a tree the straightforward way: draw a feature, halve its range, recurse left then right."""
        if depth == model.max_depth:
            return []
        q = rng.randint(model.num_dimensions)
        p = (maxes[q] + mins[q]) / 2.0
        upper, maxes[q] = maxes[q], p
        left = recursive_build(model, mins, maxes, depth + 1, rng)
        maxes[q], lower, mins[q] = upper, mins[q], p
        right = recursive_build(model, mins, maxes, depth + 1, rng)
        mins[q] = lower
        # Interleave the levels of both subtrees, so that the splits come out in heap order.
        levels = [[(q, p)]]
        for k in range(model.max_depth - depth - 1):
            levels.append(left[2**k - 1 : 2 ** (k + 1) - 1] + right[2**k - 1 : 2 ** (k + 1) - 1])
        return [split for level in levels for split in level]

    for num_features, random_work_space in [(1, True), (3, True), (3, False)]:
        fix_seed(0)
        model = _new_hst(
            feature_mins=[0.0] * num_features,
            feature_maxes=[1.0] * num_features,
            num_trees=4,
            max_depth=7,
            random_work_space=random_work_space,
        )
        fix_seed(0)
        rng = np.random.mtrand._rand
        for atts, values in zip(model.split_atts, model.split_values, strict=True):
            expected = recursive_build(model, *model._work_space(rng), 0, rng)
            assert list(zip(atts.tolist(), values.tolist(), strict=True)) == expected


def test_half_space_trees_default_trees_take_about_11_mb():
    import pickle

    import numpy as np

    from pysad.models import HalfSpaceTrees

    model = HalfSpaceTrees(feature_mins=np.zeros(21), feature_maxes=np.ones(21), random_state=0)

    # 25 trees of depth 15: one byte and one float64 for each of the 32,767 splits per tree, and two
    # one-byte masses for each of the 65,535 nodes per tree, since a node holds at most 250 instances.
    assert model.split_atts.dtype == model.l_mass.dtype == model.r_mass.dtype == np.uint8
    assert len(pickle.dumps(model)) < 11_000_000
