def generate_stream(seed=61, scale=1.0):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 5))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] = rng.uniform(6.0, 8.0, size=(30, 5)) * rng.choice([-1, 1], size=(30, 5))
    y[outlier_idx] = 1

    return X * scale, y


def test_rs_hash_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    for scale in [1.0, 100.0]:
        fix_seed(61)
        X, y = generate_stream(scale=scale)

        model = RSHash(feature_mins=X.min(axis=0), feature_maxes=X.max(axis=0))
        scores = model.fit_score(X)

        assert scores.shape == (X.shape[0],)
        assert roc_auc_score(y, scores) > 0.95


def test_rs_hash_outlier_scores_higher():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.normal(0.0, 1.0, size=(500, 3))
    outlier = np.array([10.0, -10.0, 10.0])

    model = RSHash(feature_mins=[-10.0] * 3, feature_maxes=[10.0] * 3)
    inlier_scores = np.array([model.fit_score_partial(x) for x in X])
    outlier_score = model.fit_score_partial(outlier)

    assert outlier_score > np.max(inlier_scores[-100:])
    assert outlier_score > np.mean(inlier_scores)


def test_rs_hash_score_partial_scores_given_instance():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(0)
    rng = np.random.default_rng(0)
    model = RSHash(feature_mins=np.zeros(3), feature_maxes=np.ones(3)).fit(rng.random((500, 3)))

    X_test = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.4], [9.0, 9.0, 9.0]])
    scores = model.score(X_test)

    # The far outlier falls outside every trained grid cell, so it must not share a score with either inlier.
    assert len(set(scores)) == 3
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_rs_hash_score_partial_has_no_side_effects():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(42)
    model = RSHash(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3)
    model.fit(np.random.uniform(size=(50, 3)))

    x = np.array([0.3, 0.6, 0.9])
    timestamps_before = model.sketch_timestamps.copy()
    counts_before = model.sketch_counts.copy()
    index_before = model.index

    score1 = model.score_partial(x)
    score2 = model.score_partial(x)

    assert score1 == score2
    np.testing.assert_array_equal(model.sketch_timestamps, timestamps_before)
    np.testing.assert_array_equal(model.sketch_counts, counts_before)
    assert model.index == index_before


def test_rs_hash_sketch_arrays_have_fixed_shape_and_do_not_grow():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(0)
    rng = np.random.default_rng(0)
    X = rng.random((20000, 5)) + np.linspace(0, 10, 20000)[:, None]

    model = RSHash(
        feature_mins=np.zeros(5), feature_maxes=np.full(5, 11.0), num_hash_fns=3, hash_range=97
    )

    expected_shape = (3, 97)
    assert model.sketch_timestamps.shape == expected_shape
    assert model.sketch_counts.shape == expected_shape

    for i, x in enumerate(X, 1):
        model.fit_score_partial(x)
        if i in (1000, 5000, 20000):
            assert model.sketch_timestamps.shape == expected_shape
            assert model.sketch_counts.shape == expected_shape


def test_rs_hash_num_hash_fns_changes_scores_with_small_hash_range():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    rng = np.random.default_rng(0)
    X = rng.random((2000, 5)) + np.linspace(0, 10, 2000)[:, None]

    def run(num_hash_fns):
        fix_seed(0)
        model = RSHash(
            feature_mins=np.zeros(5),
            feature_maxes=np.full(5, 11.0),
            num_hash_fns=num_hash_fns,
            hash_range=17,
        )
        scores = np.array([model.fit_score_partial(x) for x in X])
        return model, scores

    model_w1, scores_w1 = run(1)
    model_w3, scores_w3 = run(3)

    # num_hash_fns must not perturb the sampled subspaces (V) or shifts (alpha) under a fixed
    # seed, or a difference below could come from a different grid instead of from the sketch
    # actually reading more tables.
    assert len(model_w1.V) == len(model_w3.V)
    for v1, v3 in zip(model_w1.V, model_w3.V):
        np.testing.assert_array_equal(v1, v3)
    for a1, a3 in zip(model_w1.alpha, model_w3.alpha):
        np.testing.assert_array_equal(a1, a3)

    assert not np.array_equal(scores_w1, scores_w3)


def test_rs_hash_large_hash_range_matches_exact_count_behavior():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 3))

    fix_seed(123)
    model = RSHash(
        feature_mins=[0.0] * 3,
        feature_maxes=[1.0] * 3,
        num_components=5,
        num_hash_fns=2,
        hash_range=2_000_000,
    )

    # Reference: an exact, unbounded count-min sketch (a plain dict per key, as pysad used before
    # #122), which is what a fixed-size sketch degenerates to when its hash_range is large enough
    # that this short stream causes no slot collisions.
    exact_counts = {}
    index = 1
    exact_scores = []
    for x in X:
        keys = model._cell_keys(x)

        score_instance = 0.0
        for key in keys:
            tstamp, wt = exact_counts.get(key, (index, 0.0))
            decayed = wt * np.power(2, -model.decay * (index - tstamp))
            score_instance += np.log2(1 + decayed)
        exact_scores.append(-score_instance / model.m)

        for key in keys:
            tstamp, wt = exact_counts.get(key, (index, 0.0))
            decayed = wt * np.power(2, -model.decay * (index - tstamp))
            exact_counts[key] = (index, decayed + 1)
        index += 1

    model_scores = np.array([model.fit_score_partial(x) for x in X])

    np.testing.assert_allclose(model_scores, np.array(exact_scores))


def test_rs_hash_sampling_points_warns_and_has_no_effect():
    import warnings

    import numpy as np
    import pytest

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    X = np.random.default_rng(0).random((200, 5))

    fix_seed(0)
    with pytest.warns(FutureWarning, match="sampling_points") as record:
        model_with = RSHash(feature_mins=np.zeros(5), feature_maxes=np.ones(5), sampling_points=10)
    scores_with = model_with.fit_score(X)

    # stacklevel=2 attributes the warning to the code that passed sampling_points, not to rs_hash.py.
    assert record[0].filename == __file__

    fix_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model_without = RSHash(feature_mins=np.zeros(5), feature_maxes=np.ones(5))
    scores_without = model_without.fit_score(X)

    np.testing.assert_array_equal(scores_with, scores_without)


def test_rs_hash_score_then_fit_matches_fit_score_partial():
    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(7)
    X = np.random.uniform(size=(200, 3))

    fix_seed(123)
    model_a = RSHash(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3)
    scores_a = []
    for x in X:
        scores_a.append(model_a.score_partial(x))
        model_a.fit_partial(x)

    fix_seed(123)
    model_b = RSHash(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3)
    scores_b = [model_b.fit_score_partial(x) for x in X]

    np.testing.assert_allclose(scores_a, scores_b)
