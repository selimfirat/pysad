import copy

import numpy as np
import pytest

from pysad.transform.preprocessing import ShingleTransformer


def test_shingles_hold_the_last_points_oldest_first():
    X = np.arange(1.0, 6.0).reshape(-1, 1)

    shingles = ShingleTransformer(3).fit_transform(X)

    # The first points have too few earlier points, so the oldest point fills the gap.
    np.testing.assert_array_equal(shingles, [[1, 1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def test_shingles_of_points_with_several_features():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    shingles = ShingleTransformer(2).fit_transform(X)

    np.testing.assert_array_equal(shingles, [[1, 10, 1, 10], [1, 10, 2, 20], [2, 20, 3, 30]])


def test_fit_transform_partial_transforms_before_it_fits():
    transformer = ShingleTransformer(2).fit(np.array([[1.0], [2.0]]))
    before = copy.deepcopy(transformer)

    # transform_partial leaves the history as it was.
    np.testing.assert_array_equal(transformer.transform_partial(np.array([3.0])), [2, 3])
    assert list(map(list, transformer.recent)) == list(map(list, before.recent))

    np.testing.assert_array_equal(transformer.fit_transform_partial(np.array([3.0])), [2, 3])
    np.testing.assert_array_equal(transformer.transform_partial(np.array([4.0])), [3, 4])


def test_shingle_size_one_gives_each_point():
    X = np.random.default_rng(0).random((5, 3))

    np.testing.assert_array_equal(ShingleTransformer(1).fit_transform(X), X)


@pytest.mark.parametrize(
    "size, error", [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)]
)
def test_invalid_shingle_size_raises(size, error):
    with pytest.raises(error):
        ShingleTransformer(size)


def test_a_point_with_other_features_raises():
    transformer = ShingleTransformer(3).fit_partial(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="features"):
        transformer.transform_partial(np.array([1.0, 2.0, 3.0]))


def test_a_point_with_non_finite_values_raises_and_is_not_fitted():
    transformer = ShingleTransformer(3)

    for bad in (np.array([1.0, np.nan]), np.array([np.inf]), np.array([1.0, -np.inf])):
        with pytest.raises(ValueError, match="non-finite"):
            transformer.transform_partial(bad)
        with pytest.raises(ValueError, match="non-finite"):
            transformer.fit_partial(bad)

    # None of the rejected points reached the history: a good point still
    # builds its shingle from copies of itself alone.
    np.testing.assert_array_equal(transformer.transform_partial(np.array([1.0])), [1, 1, 1])


def test_a_point_that_is_not_1d_raises():
    transformer = ShingleTransformer(3)

    with pytest.raises(ValueError, match="1-D"):
        transformer.transform_partial(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="1-D"):
        transformer.transform_partial(np.array([]))


def test_a_scalar_point_reads_as_one_feature():
    transformer = ShingleTransformer(2)

    np.testing.assert_array_equal(transformer.fit_transform_partial(1.0), [1.0, 1.0])


def test_a_batch_with_a_wrong_shape_raises_clearly():
    with pytest.raises(ValueError, match="1-D"):
        ShingleTransformer(2).fit_transform(np.zeros((2, 2, 2)))


def test_a_detector_scores_the_shingles_of_a_univariate_stream():
    from pysad.models import LODA

    x = np.sin(np.arange(300) * 2 * np.pi / 24)
    x[200] += 3.0  # a spike
    transformer, model = ShingleTransformer(8), LODA(random_state=0)

    scores = [model.fit_score_partial(transformer.fit_transform_partial(np.array([v]))) for v in x]

    assert np.argmax(scores[100:]) + 100 in range(200, 208)  # a shingle that holds the spike


def test_full_tells_when_a_shingle_needs_no_copies():
    transformer = ShingleTransformer(3)

    assert not transformer.full
    transformer.fit_partial(np.array([1.0]))
    assert not transformer.full
    transformer.fit_partial(np.array([2.0]))
    assert transformer.full
    assert ShingleTransformer(1).full
