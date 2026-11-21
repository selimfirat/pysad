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

    length = 15
    rng = np.random.default_rng(0)
    x, y = _sine(2000, length, rng), np.zeros(2000, dtype=int)
    for start in range(500, 2000, 300):
        x[start : start + length] = np.sin(2 * np.pi * np.arange(length) / 6)
        # The model scores the subsequence that ends with each value.
        y[start + length - 1 : start + 2 * length - 1] = 1

    scores = SAND(subsequence_length=length, batch_size=400, random_state=0).fit_score(x[:, None])

    assert roc_auc_score(y[400:], scores[400:]) > 0.9


def test_sand_learns_a_new_normal_behavior():
    from pysad.models import SAND

    length = 15
    rng = np.random.default_rng(0)
    x = np.concatenate([_sine(1200, length, rng), _sine(2400, 28, rng)])
    model = SAND(subsequence_length=length, batch_size=400, random_state=0)
    scores = model.fit_score(x[:, None])

    # Once the new behavior fills batches, its clusters gain weight and its scores fall. With
    # alpha=0, the weights, mu and sigma stay those of the first batch, and the scores stay high.
    assert np.median(scores[3000:]) < 0.1 * np.median(scores[1200:1600])


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

    batches = np.random.default_rng(0).normal(size=2100).cumsum().reshape(14, 150, 1)
    min_weight = 0.25
    bounded = SAND(
        subsequence_length=10,
        batch_size=150,
        min_weight=min_weight,
        random_state=0,
    )
    max_clusters = int(1 / min_weight)

    for batch in batches:
        bounded.fit(batch)

        # The retained clusters must obey the weight threshold and the cluster bound.
        assert len(bounded.weights_) <= max_clusters
        assert np.all(bounded.weights_ >= min_weight)
        np.testing.assert_allclose(bounded.weights_.sum(), 1.0)

    assert np.isfinite(bounded.score(batches[-1])).all()


def test_sand_min_weight_none_keeps_every_cluster():
    from pysad.models import SAND

    x = np.random.default_rng(0).normal(size=1200).cumsum()[:, None]
    scores = SAND(subsequence_length=10, batch_size=200, random_state=0).fit_score(x)
    explicit = SAND(subsequence_length=10, batch_size=200, min_weight=None, random_state=0)

    np.testing.assert_array_equal(explicit.fit_score(x), scores)
