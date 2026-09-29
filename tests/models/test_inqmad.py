import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("jax")

from pysad.models.inqmad import Inqmad


def test_score_partial_follows_later_training():
    """Regression test for #120: scoring after a second fit_partial/fit
    must reflect the updated model, not the model as it stood at the
    first score_partial call (which used to get baked into a stale
    jit-compiled trace).
    """
    rng = np.random.default_rng(0)
    a = rng.random((50, 3))
    b = rng.random((50, 3)) + 5.0
    q = np.array([5.5, 5.5, 5.5])

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0)
    model.fit(a)
    model.score_partial(q)  # primes the jit cache, as in the issue's repro
    model.fit(b)
    score_after_b = model.score_partial(q)

    fresh = Inqmad(input_shape=3, dim_x=32, gamma=1.0)
    fresh.fit(a)
    fresh.fit(b)
    expected = fresh.score_partial(q)

    assert score_after_b == pytest.approx(expected)


def test_score_orders_far_outliers_above_inliers():
    """Regression test for #120: scores must be anomaly scores (higher
    is more anomalous), not raw density (higher is more normal).
    """
    rng = np.random.default_rng(0)
    train = rng.random((200, 3))
    X = np.vstack([rng.random((20, 3)), rng.random((20, 3)) + 5.0])
    y = np.r_[np.zeros(20), np.ones(20)]

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit(train)
    scores = model.score(X)

    assert roc_auc_score(y, scores) > 0.9
