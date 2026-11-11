import numpy as np
import pytest


def test_scores_equal_incremental_lof_until_the_window_is_full():
    from pysad.models import IncrementalLOF, MiLOF

    X = np.random.RandomState(0).normal(size=(99, 3))

    # With fewer points than the window, no summarization runs, so MiLOF equals IncrementalLOF.
    scores = MiLOF(num_neighbors=7, window_size=100, num_summaries=10).fit_score(X)
    reference = IncrementalLOF(num_neighbors=7, window_size=100).fit_score(X)

    np.testing.assert_allclose(scores, reference)


def test_summarization_keeps_the_newest_half_and_summarizes_the_oldest_half():
    from pysad.models import MiLOF

    X = np.random.RandomState(0).normal(size=(40, 2))
    model = MiLOF(num_neighbors=3, window_size=40, num_summaries=5).fit(X)

    assert model.num_instances_ == 20
    np.testing.assert_array_equal(model.X_[:20], X[20:])
    assert model.centers_.shape == (5, 2)
    assert model.center_weights_.shape == (5,)
    assert model.center_weights_.sum() == 20


def test_window_stays_bounded():
    from pysad.models import MiLOF

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = MiLOF(num_neighbors=5, window_size=40, num_summaries=5)

    for x in X:
        model.fit_partial(x)
        assert model.num_instances_ < 40


def test_scores_zero_until_the_memory_holds_more_than_num_neighbors():
    from pysad.models import MiLOF

    X = np.random.RandomState(0).normal(size=(10, 2))
    scores = MiLOF(num_neighbors=5, window_size=20, num_summaries=5).fit_score(X)

    # No cluster centers yet, so the score is 0 while at most num_neighbors points are held.
    np.testing.assert_array_equal(scores[:6], 0.0)
    assert (scores[6:] > 0.0).all()


def test_outlier_scores_higher():
    from pysad.models import MiLOF

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = MiLOF(window_size=100).fit(X)

    assert model.score_partial(np.array([6.0, 6.0])) > model.score_partial(np.zeros(2))


def test_rejects_invalid_parameters():
    from pysad.models import MiLOF

    with pytest.raises(ValueError, match="window_size"):
        MiLOF(num_neighbors=10, window_size=21)
    with pytest.raises(ValueError, match="window_size"):
        MiLOF(num_neighbors=10, window_size=22, num_summaries=12)
    with pytest.raises(ValueError, match="num_summaries"):
        MiLOF(num_neighbors=10, window_size=22, num_summaries=0)


def test_kept_points_use_centers_as_neighbors_after_merge():
    from scipy.spatial.distance import cdist

    from pysad.models import MiLOF

    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 2))
    model = MiLOF(num_neighbors=5, window_size=40, num_summaries=4, random_state=0).fit(X)

    # One summarization ran and no further insert followed, so the kept points' k-distances and
    # lrd values reflect the cluster centers as candidate neighbors (Algorithm 2, line 23).
    half = model.window_size // 2
    raw = model.X_[:half]
    centers = model.centers_
    cand = np.vstack([raw, centers])
    k = model.num_neighbors

    # k-distances are independent, so compute them all first; lrd values then use the finalized
    # k-distances of the neighbors (a center's k-distance comes from center_k_dists_).
    expected_k_dists = np.empty(half)
    for i in range(half):
        d = cdist(raw[i : i + 1], cand)[0]
        d[i] = np.inf  # a point is not its own neighbor
        nb = np.argpartition(d, k - 1)[:k]
        expected_k_dists[i] = d[nb].max()

    expected_lrds = np.empty(half)
    for i in range(half):
        d = cdist(raw[i : i + 1], cand)[0]
        d[i] = np.inf
        nb = np.argpartition(d, k - 1)[:k]
        nd = d[nb]
        is_center = nb >= half
        k_dist_nb = np.where(
            is_center,
            model.center_k_dists_[np.clip(nb - half, 0, centers.shape[0] - 1)],
            expected_k_dists[np.clip(nb, 0, half - 1)],
        )
        reach = np.maximum(k_dist_nb, nd)
        expected_lrds[i] = 1.0 / (reach.mean() + 1e-10)

    np.testing.assert_allclose(model.k_dists_[:half], expected_k_dists)
    np.testing.assert_allclose(model.lrds_[:half], expected_lrds)


def test_fixed_random_state_is_reproducible():
    from pysad.models import MiLOF

    X = np.random.RandomState(0).normal(size=(200, 3))
    probes = np.random.RandomState(1).normal(size=(10, 3))

    first = MiLOF(num_neighbors=5, window_size=40, num_summaries=4, random_state=3)
    second = MiLOF(num_neighbors=5, window_size=40, num_summaries=4, random_state=3)

    np.testing.assert_allclose(first.fit_score(X), second.fit_score(X))
    np.testing.assert_allclose(first.score(probes), second.score(probes))
