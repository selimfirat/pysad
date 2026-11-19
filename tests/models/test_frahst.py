import numpy as np
import pytest


def test_rank_alarms_follow_energy_thresholds():
    from pysad.models import FRAHST

    model = FRAHST(random_state=0)
    # The first direction fills the model. A later orthogonal direction needs another variable.
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([0.0, 10.0, 0.0])
    assert model.fit_score_partial(first) == 0.0
    model.fit(np.tile(first, (100, 1)))
    assert model.rank_ == 1
    assert model.score_partial(first) == 0.0
    assert model.score_partial(second) == 1.0
    assert model.fit_score_partial(second) == 1.0
    assert model.rank_ == 2
    # Figure 2 suppresses consecutive rank-increase alarms.
    third = np.array([0.0, 0.0, 100.0])
    assert model.fit_score_partial(third) == 0.0
    assert model.rank_ == 3
    model.fit(np.tile(first, (500, 1)))
    assert model.rank_ == 1


def test_basis_tracks_a_rotating_line():
    from pysad.models import FRAHST

    model = FRAHST(random_state=0)
    for angle in np.linspace(0.0, 0.5, 1000):
        model.fit_partial(np.array([np.cos(angle), np.sin(angle), 0.0]))
    assert model.rank_ == 1
    direction = np.array([np.cos(0.5), np.sin(0.5), 0.0])
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(model.rank_), atol=1e-10)
    assert np.linalg.norm(direction - model.basis_ @ (model.basis_.T @ direction)) < 0.02


@pytest.mark.parametrize("value", [0.0, 1.0, 1e-10])
def test_constant_stream_stays_finite(value):
    from pysad.models import FRAHST

    model = FRAHST(random_state=0)
    scores = model.fit_score(np.full((500, 3), value))
    assert np.isfinite(scores).all()
    assert model.rank_ == 1
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(1), atol=1e-10)


def test_first_basis_matches_one_covariance_iteration():
    from pysad.models import FRAHST

    # Equation 5 gives an independent rank-one orthogonal iteration.
    initial = np.array([1.764052345967664, 0.4001572083672233])
    initial /= np.linalg.norm(initial)
    x = np.array([1.0, 0.0])
    covariance = 0.96e-4 * np.eye(2) + np.outer(x, x)
    expected = covariance @ initial
    expected /= np.linalg.norm(expected)
    model = FRAHST(random_state=0).fit_partial(x)
    np.testing.assert_allclose(model.basis_.ravel(), expected, atol=1e-14)
