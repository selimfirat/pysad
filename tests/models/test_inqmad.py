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
    score_after_a = model.score_partial(q)  # primes the jit cache, as in the issue's repro
    model.fit(b)
    score_after_b = model.score_partial(q)

    fresh = Inqmad(input_shape=3, dim_x=32, gamma=1.0)
    fresh.fit(a)
    fresh.fit(b)
    expected = fresh.score_partial(q)

    assert score_after_b != pytest.approx(score_after_a)
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


def test_fit_score_ranks_interleaved_far_outliers_above_inliers():
    """Regression test for #120: on the streaming path each score must
    depend on the data seen so far, not only on how many instances were
    fitted (a replacing update used to give exactly -1/t^2).
    """
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.zeros(300)
    outliers = rng.choice(300, size=30, replace=False)
    X[outliers] += 5.0
    y[outliers] = 1

    scores = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit_score(X)

    assert not np.allclose(scores, -1.0 / np.arange(1, 301) ** 2)
    assert roc_auc_score(y, scores) > 0.9


def test_score_partial_depends_on_whole_fitted_history():
    """Regression test for #120: the model must remember every fitted
    instance, not only the last one, so two histories that end with the
    same row give different scores.
    """
    rng = np.random.default_rng(0)
    history = rng.random((199, 3))
    last = rng.random((1, 3))
    q = np.array([0.5, 0.5, 0.5])

    near = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit(np.vstack([history, last]))
    far = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit(np.vstack([history + 5.0, last]))

    assert near.score_partial(q) < far.score_partial(q)


def test_far_point_fitted_last_scores_above_inliers():
    """Regression test for #120: fitting one far point after many inliers
    must not make it the model's whole memory.
    """
    rng = np.random.default_rng(0)
    far_point = np.array([5.0, 5.0, 5.0])
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit(
        np.vstack([rng.random((199, 3)), far_point])
    )

    inlier_scores = model.score(rng.random((20, 3)))

    assert model.score_partial(far_point) > inlier_scores.max()


def test_density_matrix_is_mean_of_fitted_states():
    """Regression test for #120: per-instance fits and the batches of a
    single multi-row fit_partial call both add to the density matrix.
    """
    rng = np.random.default_rng(0)
    a = rng.random((7, 3))
    b = rng.random((5, 3)) + 5.0

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, batch_size=2)
    model.fit(a)
    model.fit_partial(b)  # three batches in one update

    states = np.asarray(model.inqmad.fm_x(np.vstack([a, b])), dtype=np.float64)
    rho = np.asarray(model.inqmad.rho_res, dtype=np.float64) / model.inqmad.num_samples

    np.testing.assert_allclose(rho, np.einsum("ni,nj->ij", states, states) / 12, atol=1e-6)


def test_score_partial_is_negated_paper_density():
    """Regression test for #120: the score is the negated density
    estimate psi^T rho psi of the paper's Eq. 3, with rho normalised by
    the number of fitted instances.
    """
    rng = np.random.default_rng(0)
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0).fit(rng.random((50, 3)))
    q = rng.random(3)

    psi = np.asarray(model.inqmad.fm_x(q[None, :]), dtype=np.float64)[0]
    rho = np.asarray(model.inqmad.rho_res, dtype=np.float64) / model.inqmad.num_samples

    assert model.score_partial(q) == pytest.approx(-np.einsum("i,ij,j->", psi, rho, psi), rel=1e-5)
