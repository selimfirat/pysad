def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outlier_idx] = 1

    return X, y


def test_stream_km_plusplus_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import StreamKMPlusPlus

    X, y = generate_stream()

    scores = StreamKMPlusPlus(num_clusters=3, coreset_size=100, random_state=0).fit_score(X)

    assert scores.shape == (X.shape[0],)
    # The paper-faithful detector clusters only the size-m coreset, so its AUROC is lower and
    # more seed-sensitive than the previous union clustering; across seeds 0-4 it scores
    # 0.87-0.95 on this stream.
    assert roc_auc_score(y, scores) > 0.85


def test_stream_km_plusplus_ranks_far_points_higher():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    model = StreamKMPlusPlus(num_clusters=2, coreset_size=10, random_state=0)
    model.fit(np.array([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1]]))

    assert model.score_partial(np.array([0.15])) < model.score_partial(np.array([9.0]))
    assert model.score_partial(np.array([0.15])) < 1e-9  # a fitted center sits at 0.15
    assert model.score_partial(np.array([9.0])) > 5.0


def test_stream_km_plusplus_coreset_weights_follow_tree_partition():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    rng = np.random.RandomState(7)
    X = np.sort(rng.normal(0.0, 3.0, size=(12, 1)), axis=0)

    sampled, weights = StreamKMPlusPlus(num_clusters=2, coreset_size=4, random_state=3)._coreset(
        X, np.ones(X.shape[0])
    )

    # The weights are the sizes of the tree leaf partition (paper sections 4.2-4.3), not the
    # global nearest-sample assignment: the point 1.80 falls in the leaf of 0.01, even though
    # 3.05 is its nearest sampled point.
    order = np.argsort(sampled.ravel())
    np.testing.assert_allclose(
        sampled.ravel()[order],
        [-5.2641729190362625, 0.00619671871784439, 3.0529740169904795, 5.071577111401068],
    )
    np.testing.assert_allclose(weights[order], [1.0, 9.0, 1.0, 1.0])
    assert weights.sum() == X.shape[0]
    nearest = ((X[:, None, :] - sampled[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    nearest_weights = np.zeros(len(sampled))
    np.add.at(nearest_weights, nearest, np.ones(X.shape[0]))
    np.testing.assert_allclose(nearest_weights[order], [1.0, 8.0, 2.0, 1.0])
    assert not np.array_equal(weights, nearest_weights)


def test_stream_km_plusplus_final_coreset_has_size_m():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    model = StreamKMPlusPlus(num_clusters=3, coreset_size=10, random_state=0)
    X = np.random.RandomState(0).rand(500, 2)
    model.fit(X)

    # The final reduction (paper section 5.1) reduces the union of the buffer and all level
    # coresets to one coreset of size m, preserving the total weight.
    points, weights = model._union()
    coreset_points, coreset_weights = model._coreset(points, weights)

    assert coreset_points.shape == (10, 2)
    assert coreset_weights.sum() == 500


def test_stream_km_plusplus_five_restarts_select_lowest_cost():
    import numpy as np
    import pytest

    from pysad.models import StreamKMPlusPlus

    points = np.array(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [5.0], [5.1], [5.2], [9.0], [9.1]]
    )
    weights = np.ones(points.shape[0])

    model = StreamKMPlusPlus(num_clusters=3, coreset_size=12, random_state=1)
    once = model._kmeans_once(points, weights)
    best = StreamKMPlusPlus(num_clusters=3, coreset_size=12, random_state=1)._kmeans(
        points, weights
    )

    # One restart lands in a poor local optimum; the best of five runs (paper section 5) finds
    # the three groups.
    assert model._weighted_cost(points, weights, once) == pytest.approx(18.818, abs=0.01)
    assert model._weighted_cost(points, weights, best) == pytest.approx(0.305, abs=0.01)
    assert model._weighted_cost(points, weights, best) < model._weighted_cost(points, weights, once)


def test_stream_km_plusplus_seeded_centers_are_stable():
    import numpy as np
    import pytest

    from pysad.models import StreamKMPlusPlus

    rng = np.random.RandomState(11)
    X = np.concatenate([rng.normal(0.0, 0.5, size=(30, 2)), rng.normal(5.0, 0.5, size=(30, 2))])

    model = StreamKMPlusPlus(num_clusters=2, coreset_size=10, random_state=0)
    model.fit(X)

    # The full pipeline (tree sampling, size-m reduction, five restarts) is deterministic for a
    # fixed seed; pin the centers so a regression in any step shows up here.
    np.testing.assert_allclose(
        model.centers_[np.argsort(model.centers_[:, 0])],
        [[0.18431411278012824, 0.009146313317602569], [5.17787311674871, 4.998233196624181]],
        rtol=1e-12,
        atol=1e-12,
    )
    assert model.score_partial(np.array([0.0, 0.0])) == pytest.approx(0.1845409093324014, rel=1e-9)
    assert model.score_partial(np.array([10.0, 10.0])) == pytest.approx(6.947703133664186, rel=1e-9)

    refit = StreamKMPlusPlus(num_clusters=2, coreset_size=10, random_state=0)
    refit.fit(X)
    assert np.array_equal(model.centers_, refit.centers_)


def test_stream_km_plusplus_memory_stays_bounded():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    # With n_max=100 the level cap is ceil(log2(100)) = 7, so at most (7 + 1) * 10 points are
    # stored no matter how long the stream runs.
    bound = (7 + 1) * 10
    for num_points in (200, 1000, 2000):
        X = np.random.RandomState(3).normal(size=(num_points, 2))
        model = StreamKMPlusPlus(num_clusters=3, coreset_size=10, n_max=100, random_state=0)
        model.fit(X)

        stored = model._num_buffered + sum(
            coreset[0].shape[0] for coreset in model._levels if coreset is not None
        )
        stored_weight = float(model._num_buffered) + sum(
            float(coreset[1].sum()) for coreset in model._levels if coreset is not None
        )
        assert len(model._levels) == 7
        assert stored <= bound
        assert stored_weight == num_points  # the cap folds merges, it never drops points
        assert model.centers_.shape == (3, 2)
        assert np.isfinite(model.score_partial(np.zeros(2)))
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    assert StreamKMPlusPlus().score_partial(np.zeros(2)) == 0.0


def test_stream_km_plusplus_scores_before_the_first_fit():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    assert StreamKMPlusPlus().score_partial(np.zeros(2)) == 0.0


def test_stream_km_plusplus_reduces_small_buffers():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    model = StreamKMPlusPlus(num_clusters=3, coreset_size=10, random_state=0)
    X = np.random.RandomState(0).rand(500, 2)
    scores = model.fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()
    assert model.centers_.shape == (3, 2)


def test_stream_km_plusplus_scores_constant_streams():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    scores = StreamKMPlusPlus(num_clusters=3, coreset_size=5, random_state=0).fit_score(
        np.ones((50, 2))
    )

    assert np.isfinite(scores).all()
    assert scores.max() == 0.0


def test_stream_km_plusplus_rejects_bad_parameters():
    import pytest

    from pysad.models import StreamKMPlusPlus

    with pytest.raises(ValueError):
        StreamKMPlusPlus(num_clusters=0)
    with pytest.raises(ValueError):
        StreamKMPlusPlus(coreset_size=0)
    with pytest.raises(ValueError):
        StreamKMPlusPlus(num_iterations=0)
    with pytest.raises(ValueError):
        StreamKMPlusPlus(n_max=0)


def test_stream_km_plusplus_first_center_is_uniform_over_non_dummy_points():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    # With four clusters on four points, Lloyd iterations leave the centers as the
    # points themselves, so centers[0] is the first k-means++ pick. The reference
    # picks it uniformly from non-dummy points (paper section 5), so a point with
    # weight 1 must be picked about as often as the point with weight 100, and a
    # zero-weight point must never be picked. Weight-proportional picking would
    # pick the weight-1 point about 1% of the time.
    points = np.array([[0.0], [1.0], [2.0], [3.0]])
    weights = np.array([100.0, 1.0, 1.0, 0.0])
    picks = []
    for seed in range(200):
        model = StreamKMPlusPlus(num_clusters=4, coreset_size=4, random_state=seed)
        centers = model._kmeans_once(points, weights)
        picks.append(float(centers[0, 0]))
    picks = np.array(picks)
    assert (picks == 3.0).sum() == 0
    assert (picks == 1.0).mean() > 0.10


def test_stream_km_plusplus_pickle_round_trip_keeps_rng():
    import pickle

    import numpy as np

    from pysad.models import StreamKMPlusPlus

    X = np.random.RandomState(0).rand(50, 2)

    # An integer seed keeps its stream position across a pickle round trip.
    model = StreamKMPlusPlus(num_clusters=3, coreset_size=10, random_state=0)
    model.fit(X[:25])
    before = model.score_partial(X[25])
    restored = pickle.loads(pickle.dumps(model))
    assert restored.score_partial(X[25]) == before

    # A None seed draws from NumPy's global state and must survive a round trip
    # even if the global RandomState object has been replaced since pickling.
    np.random.seed(1)
    model = StreamKMPlusPlus(num_clusters=3, coreset_size=10, random_state=None)
    model.fit(X[:25])
    pickled = pickle.dumps(model)
    old_rand = np.random.mtrand._rand
    try:
        np.random.mtrand._rand = np.random.RandomState(99)
        restored = pickle.loads(pickled)
        assert np.isfinite(restored.score_partial(X[25]))
    finally:
        np.random.mtrand._rand = old_rand
