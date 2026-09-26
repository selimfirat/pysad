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
    from pysad.models import RSHash
    from pysad.utils import fix_seed
    from sklearn.metrics import roc_auc_score

    for scale in [1.0, 100.0]:
        fix_seed(61)
        X, y = generate_stream(scale=scale)

        model = RSHash(feature_mins=X.min(axis=0), feature_maxes=X.max(axis=0))
        scores = model.fit_score(X)

        assert scores.shape == (X.shape[0],)
        assert roc_auc_score(y, scores) > 0.95


def test_rs_hash_outlier_scores_higher():
    from pysad.models import RSHash
    from pysad.utils import fix_seed
    import numpy as np

    fix_seed(61)
    X = np.random.normal(0.0, 1.0, size=(500, 3))
    outlier = np.array([10.0, -10.0, 10.0])

    model = RSHash(feature_mins=[-10.0] * 3, feature_maxes=[10.0] * 3)
    inlier_scores = np.array([model.fit_score_partial(x) for x in X])
    outlier_score = model.fit_score_partial(outlier)

    assert outlier_score > np.max(inlier_scores[-100:])
    assert outlier_score > np.mean(inlier_scores)
