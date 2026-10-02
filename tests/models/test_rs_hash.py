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
    import copy

    import numpy as np

    from pysad.models import RSHash
    from pysad.utils import fix_seed

    fix_seed(42)
    model = RSHash(feature_mins=[0.0] * 3, feature_maxes=[1.0] * 3)
    model.fit(np.random.uniform(size=(50, 3)))

    x = np.array([0.3, 0.6, 0.9])
    sketches_before = copy.deepcopy(model.cmsketches)
    index_before = model.index

    score1 = model.score_partial(x)
    score2 = model.score_partial(x)

    assert score1 == score2
    assert model.cmsketches == sketches_before
    assert model.index == index_before


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


def test_rs_hash_accepts_integer_decay():
    import numpy as np

    from pysad.models import RSHash

    feature_mins = np.zeros(2)
    feature_maxes = np.ones(2)
    X = np.array([[0.2, 0.3], [0.2, 0.3], [0.8, 0.7]])

    for decay in [1, np.int64(1)]:
        model_int = RSHash(
            feature_mins,
            feature_maxes,
            decay=decay,
            random_state=0,
        )
        model_float = RSHash(
            feature_mins,
            feature_maxes,
            decay=1.0,
            random_state=0,
        )

        scores_int = model_int.fit_score(X)
        scores_float = model_float.fit_score(X)

        np.testing.assert_allclose(scores_int, scores_float)


def test_rs_hash_rejects_non_positive_decay():
    import numpy as np
    import pytest

    from pysad.models import RSHash

    feature_mins = np.zeros(2)
    feature_maxes = np.ones(2)

    for decay in [0, 0.0, np.int64(0), -0.1]:
        with pytest.raises(ValueError, match="decay.*positive"):
            RSHash(feature_mins, feature_maxes, decay=decay)
