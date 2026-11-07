import numpy as np
import pytest

LENGTH = 25


def _sine(num_values, period, rng):
    return np.sin(2 * np.pi * np.arange(num_values) / period) + rng.normal(0.0, 0.05, num_values)


def _znorm(x):
    return (x - x.mean()) / x.std()


def test_sand_scores_zero_until_the_first_batch_completes():
    from pysad.models import SAND

    X = _sine(199, 25, np.random.default_rng(0)).reshape(-1, 1)
    scores = SAND(subsequence_length=LENGTH, batch_size=200, random_state=0).fit_score(X)

    np.testing.assert_array_equal(scores, 0.0)


def test_sand_score_is_the_normalized_weighted_distance_to_the_centroids():
    from pysad.models import SAND

    rng = np.random.default_rng(0)
    x = _sine(1000, 25, rng)
    model = SAND(subsequence_length=LENGTH, batch_size=400, random_state=0).fit(x[:-1, None])

    # Eq. 5: the weighted sum of the smallest z-normalized distances to the windows of each centroid.
    query = _znorm(x[-LENGTH:])
    d = 0.0
    for weight, centroid in zip(model.weights_, model.centroids_, strict=True):
        windows = np.lib.stride_tricks.sliding_window_view(centroid, LENGTH)
        d += weight * min(np.linalg.norm(query - _znorm(w)) for w in windows)

    expected = (d - model.mean_) / model.std_
    np.testing.assert_allclose(model.score_partial(x[-1:]), expected)
    np.testing.assert_allclose(model.fit_score_partial(x[-1:]), expected)


def test_sand_ranks_subsequences_of_another_shape_above_the_normal_ones():
    from sklearn.metrics import roc_auc_score

    from pysad.models import SAND

    rng = np.random.default_rng(0)
    x, y = _sine(4000, 25, rng), np.zeros(4000, dtype=int)
    for start in range(1200, 4000, 400):
        x[start : start + LENGTH] = np.sin(2 * np.pi * np.arange(LENGTH) / 10)
        # The model scores the subsequence that ends with each value.
        y[start + LENGTH - 1 : start + 2 * LENGTH - 1] = 1

    scores = SAND(subsequence_length=LENGTH, batch_size=1000, random_state=0).fit_score(x[:, None])

    assert roc_auc_score(y[1000:], scores[1000:]) > 0.9


def test_sand_learns_a_new_normal_behavior():
    from pysad.models import SAND

    rng = np.random.default_rng(0)
    x = np.concatenate([_sine(3000, 25, rng), _sine(5000, 40, rng)])
    model = SAND(subsequence_length=LENGTH, batch_size=1000, random_state=0)
    scores = model.fit_score(x[:, None])

    # Once the new behavior fills batches, its clusters gain weight and its scores fall. With
    # alpha=0, the weights, mu and sigma stay those of the first batch, and the scores stay high.
    assert np.median(scores[7000:]) < 0.1 * np.median(scores[3000:4000])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"subsequence_length": 1}, "subsequence_length must be at least 2"),
        ({"subsequence_length": 10, "centroid_length": 10}, "centroid_length must be at least 11"),
        ({"subsequence_length": 10, "batch_size": 44}, "batch_size must be at least 45"),
        ({"subsequence_length": 10, "alpha": 1.5}, r"alpha must be in \[0, 1\]"),
        ({"subsequence_length": 10, "min_weight": -0.1}, r"min_weight must be in \[0, 1\)"),
        ({"subsequence_length": 10, "min_weight": 1.0}, r"min_weight must be in \[0, 1\)"),
    ],
)
def test_sand_rejects_invalid_parameters(params, message):
    from pysad.models import SAND

    with pytest.raises(ValueError, match=message):
        SAND(**params)


def test_sand_is_univariate():
    from pysad.models import SAND

    with pytest.raises(ValueError, match="SAND is univariate"):
        SAND(subsequence_length=4).fit_partial(np.zeros(2))


def test_sand_min_weight_bounds_the_clusters_on_a_drifting_stream():
    from pysad.models import SAND

    batches = np.random.default_rng(0).normal(size=8000).cumsum().reshape(20, 400, 1)
    default = SAND(subsequence_length=LENGTH, batch_size=400, random_state=0)
    bounded = SAND(subsequence_length=LENGTH, batch_size=400, min_weight=0.01, random_state=0)
    default_sizes = [len(default.fit(batch).weights_) for batch in batches]
    bounded_sizes = [len(bounded.fit(batch).weights_) for batch in batches]

    # Without min_weight, the walk keeps adding clusters. With it, old clusters leave.
    assert default_sizes[-1] > default_sizes[9]
    assert max(bounded_sizes[10:]) <= max(bounded_sizes[:10])
    assert bounded_sizes[-1] < default_sizes[-1] / 4
    np.testing.assert_allclose(bounded.weights_.sum(), 1.0)
    assert np.isfinite(bounded.score(batches[-1])).all()


def test_sand_min_weight_none_keeps_every_cluster():
    from pysad.models import SAND

    x = np.random.default_rng(0).normal(size=3000).cumsum()[:, None]
    scores = SAND(subsequence_length=LENGTH, batch_size=400, random_state=0).fit_score(x)
    explicit = SAND(subsequence_length=LENGTH, batch_size=400, min_weight=None, random_state=0)

    np.testing.assert_array_equal(explicit.fit_score(x), scores)
