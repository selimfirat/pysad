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
    assert roc_auc_score(y, scores) > 0.95


def test_stream_km_plusplus_ranks_far_points_higher():
    import numpy as np

    from pysad.models import StreamKMPlusPlus

    model = StreamKMPlusPlus(num_clusters=2, coreset_size=10, random_state=0)
    model.fit(np.array([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1]]))

    assert model.score_partial(np.array([0.15])) < model.score_partial(np.array([9.0]))
    assert model.score_partial(np.array([0.15])) < 1e-9  # a fitted center sits at 0.15
    assert model.score_partial(np.array([9.0])) > 5.0


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
