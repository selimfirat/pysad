def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 3))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-6.0, 6.0], size=(30, 3))
    y[outlier_idx] = 1

    return X, y


def test_anyout_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import AnyOut

    X, y = generate_stream()

    for params in [{}, {"outlier_score": "density"}, {"depth": 2}, {"max_height": 3}]:
        scores = AnyOut(**params).fit_score(X)
        assert scores.shape == (X.shape[0],)
        assert roc_auc_score(y, scores) > 0.9, params


def _three_clusters():
    """One instance of each of three distant clusters, then the rest in random order."""
    import numpy as np

    rng = np.random.RandomState(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    labels = np.r_[0, 1, 2, rng.randint(3, size=60)]
    return centers[labels] + rng.normal(0.0, [0.5, 2.0], size=(63, 2)), labels


def test_anyout_scores_against_the_closest_cluster_of_a_one_level_tree():
    """A one-level tree merges its closest entries, so it holds one entry per distant cluster."""
    import numpy as np
    from scipy.stats import norm

    from pysad.models import AnyOut

    X, labels = _three_clusters()
    x = np.array([9.0, 3.0])
    cluster = X[labels == 1]

    mean_model = AnyOut(max_height=1, decay=0.0).fit(X)
    density_model = AnyOut(outlier_score="density", max_height=1, decay=0.0).fit(X)

    np.testing.assert_allclose(mean_model.score_partial(x), np.linalg.norm(x - cluster.mean(0)))
    expected = -norm.logpdf(x, loc=cluster.mean(0), scale=cluster.std(0)).sum()
    np.testing.assert_allclose(density_model.score_partial(x), expected)


def test_anyout_reads_finer_clusters_at_deeper_levels():
    from pysad.models import AnyOut

    X, _ = _three_clusters()

    # The root entries hold whole clusters, and the leaves hold single instances.
    root_scores = AnyOut(depth=1, decay=0.0).fit(X).score(X)
    leaf_scores = AnyOut(decay=0.0).fit(X).score(X)
    assert leaf_scores.mean() < 0.5 * root_scores.mean()


def test_anyout_forgets_a_region_once_its_entries_are_outdated():
    import numpy as np

    from pysad.models import AnyOut

    rng = np.random.RandomState(0)
    old, new = rng.normal(0.0, 0.1, size=(30, 2)), rng.normal(5.0, 0.1, size=(300, 2))
    X = np.vstack([old, new])

    remembering = AnyOut(max_height=2, decay=0.0).fit(X)
    forgetting = AnyOut(max_height=2, decay=0.05, snapshot_interval=20).fit(X)

    assert remembering.score_partial(np.zeros(2)) < 1.0
    assert forgetting.score_partial(np.zeros(2)) > 4.0


def test_anyout_keeps_at_most_3_to_the_max_height_leaf_entries():
    import numpy as np

    from pysad.models import AnyOut

    model = AnyOut(max_height=2, decay=0.0).fit(np.random.RandomState(0).normal(size=(500, 2)))

    assert model.height_ == 2
    assert sum(leaf.size for leaf in model.root_.children) <= 9


def test_anyout_scores_0_before_the_first_fit():
    import numpy as np

    from pysad.models import AnyOut

    assert AnyOut().score_partial(np.ones(3)) == 0.0


def test_anyout_rejects_an_unknown_outlier_score():
    import pytest

    from pysad.models import AnyOut

    with pytest.raises(ValueError, match="outlier_score"):
        AnyOut(outlier_score="median")


def test_anyout_scores_do_not_change_when_a_feature_gets_a_constant_offset():
    """Adding a constant to a feature leaves every distance and variance of the tree unchanged."""
    import numpy as np

    from pysad.models import AnyOut

    X, _ = generate_stream()
    shifted = X.copy()
    shifted[:, 0] += 1e6

    for params in [
        {},
        {"outlier_score": "density", "depth": 1},
        {"outlier_score": "density", "depth": 2},
    ]:
        scores = AnyOut(**params).fit_score(X)
        shifted_scores = AnyOut(**params).fit_score(shifted)
        np.testing.assert_allclose(
            shifted_scores, scores, rtol=1e-7, atol=1e-9, err_msg=str(params)
        )
