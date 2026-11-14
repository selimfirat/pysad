from __future__ import annotations

import numpy as np
import pytest

from pysad.models.absad import (
    ABSAD,
    _calculate_control_limit,
    _calculate_feature_relevance,
    _calculate_local_outlier_score,
    _normalize_window_and_point,
    _select_reference_set,
)


def test_absad_default_initialization() -> None:
    model = ABSAD()
    assert model.window_size == 750
    assert model.num_neighbors == 100
    assert model.num_references == 100
    assert model.theta == 0.4
    assert model.confidence == 0.99
    assert model.control_limit_ is None
    assert len(model._window) == 0


def test_absad_custom_initialization() -> None:
    model = ABSAD(
        window_size=200,
        num_neighbors=30,
        num_references=20,
        theta=0.5,
        confidence=0.95,
    )
    assert model.window_size == 200
    assert model.num_neighbors == 30
    assert model.num_references == 20
    assert model.theta == 0.5
    assert model.confidence == 0.95


@pytest.mark.parametrize(
    ("kwargs", "expected_msg"),
    [
        ({"window_size": 0}, "window_size must be positive"),
        ({"window_size": -10}, "window_size must be positive"),
        (
            {"window_size": 100, "num_neighbors": 0},
            "num_neighbors must be between 1 and window_size",
        ),
        (
            {"window_size": 100, "num_neighbors": 101},
            "num_neighbors must be between 1 and window_size",
        ),
        (
            {"window_size": 100, "num_neighbors": 50, "num_references": 0},
            "num_references must be between 1 and num_neighbors",
        ),
        (
            {"window_size": 100, "num_neighbors": 50, "num_references": 51},
            "num_references must be between 1 and num_neighbors",
        ),
        ({"theta": -0.1}, "theta must be non-negative"),
        ({"confidence": 0.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.5}, r"confidence must be in \(0, 1\)"),
    ],
)
def test_absad_invalid_parameters(kwargs: dict[str, float | int], expected_msg: str) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        ABSAD(**kwargs)


def test_normalization_constant_features() -> None:
    rng = np.random.RandomState(42)
    # Window with 2 random features and 1 constant feature
    W = rng.randn(20, 3)
    W[:, 1] = 5.0
    x = np.array([1.0, 5.0, -1.0])

    W_norm, x_norm = _normalize_window_and_point(W, x)
    assert W_norm.shape == W.shape
    assert x_norm.shape == x.shape
    # Constant feature should be normalized to 0.0 without NaNs or Infs
    assert np.all(W_norm[:, 1] == 0.0)
    assert x_norm[1] == 0.0
    assert not np.isnan(W_norm).any()
    assert not np.isnan(x_norm).any()


def test_select_reference_set() -> None:
    rng = np.random.RandomState(42)
    W = rng.randn(30, 4)
    x = rng.randn(4)

    W_norm, x_norm = _normalize_window_and_point(W, x)
    ref_set, indices = _select_reference_set(W_norm, x_norm, k=10, s=5)

    assert ref_set.shape == (5, 4)
    assert len(indices) == 5
    assert len(set(indices)) == 5
    assert np.all(indices >= 0) and np.all(indices < 30)


def test_calculate_feature_relevance() -> None:
    ref_set = np.zeros((10, 5))
    # Query point has significant deviation only along features 0 and 2
    x = np.array([10.0, 0.0, 8.0, 0.0, 0.0])

    relevance, selected = _calculate_feature_relevance(ref_set, x, theta=0.4)
    assert relevance.shape == (5,)
    assert 0 in selected
    assert 2 in selected
    assert 1 not in selected
    assert 3 not in selected
    assert 4 not in selected


def test_calculate_feature_relevance_fallback() -> None:
    ref_set = np.zeros((10, 3))
    # Point at the center of the reference set
    x = np.zeros(3)

    relevance, selected = _calculate_feature_relevance(ref_set, x, theta=0.4)
    assert len(selected) == 3
    assert np.array_equal(selected, np.arange(3))


def test_calculate_local_outlier_score_singular_covariance() -> None:
    # All reference points identical -> zero covariance
    ref_set = np.ones((5, 3))
    x = np.array([2.0, 2.0, 2.0])
    subspace = np.array([0, 1, 2])

    score = _calculate_local_outlier_score(ref_set, x, subspace)
    assert np.isfinite(score)
    assert score > 0.0


def test_score_partial_discrimination() -> None:
    rng = np.random.RandomState(42)
    model = ABSAD(window_size=30, num_neighbors=15, num_references=10)

    # Fit 30 normal samples centered at 0
    normal_data = rng.randn(30, 4) * 0.5
    for sample in normal_data:
        model.fit_partial(sample)

    normal_query = rng.randn(4) * 0.5
    anomalous_query = np.array([10.0, 10.0, -10.0, 8.0])

    normal_score = model.score_partial(normal_query)
    anom_score = model.score_partial(anomalous_query)

    assert anom_score > normal_score
    assert np.isfinite(normal_score)
    assert np.isfinite(anom_score)


def test_score_partial_preserves_state() -> None:
    rng = np.random.RandomState(42)
    model = ABSAD(window_size=20, num_neighbors=10, num_references=8)

    for sample in rng.randn(20, 3):
        model.fit_partial(sample)

    window_before = model._window.to_array()
    query = rng.randn(3)

    score_1 = model.score_partial(query)
    score_2 = model.score_partial(query)

    assert score_1 == score_2
    assert len(model._window) == 20
    assert np.array_equal(window_before, model._window.to_array())


def test_calculate_control_limit() -> None:
    rng = np.random.RandomState(42)
    # Simulated exponential or chi-squared like LOS scores
    los_history = rng.exponential(scale=2.0, size=100)

    cl_95 = _calculate_control_limit(los_history, confidence=0.95)
    cl_99 = _calculate_control_limit(los_history, confidence=0.99)

    assert np.isfinite(cl_95)
    assert np.isfinite(cl_99)
    assert cl_99 > cl_95
    # The control limit should be higher than most historical LOS values
    assert np.mean(los_history <= cl_99) >= 0.90


def test_calculate_control_limit_degenerate() -> None:
    # All LOS values constant
    los_history = np.full(50, 2.5)
    cl = _calculate_control_limit(los_history, confidence=0.99)
    assert np.isfinite(cl)
    assert cl >= 2.5


def test_selective_window_update_and_rejection() -> None:
    rng = np.random.RandomState(42)
    window_size = 20
    model = ABSAD(window_size=window_size, num_neighbors=10, num_references=8, confidence=0.95)

    # 1. Warm-up
    normal_samples = rng.randn(window_size, 3) * 0.2
    for s in normal_samples:
        model.fit_partial(s)

    assert len(model._window) == window_size
    assert model.control_limit_ is not None

    # 2. Feeding a normal point within control limit should be accepted
    normal_candidate = rng.randn(3) * 0.1
    score_normal = model.score_partial(normal_candidate)
    assert score_normal <= model.control_limit_

    model.fit_partial(normal_candidate)
    assert len(model._window) == window_size
    assert np.allclose(model._window.to_array()[-1], normal_candidate)

    # 3. Feeding an extreme anomaly should be rejected
    anom_candidate = np.array([50.0, -50.0, 50.0])
    score_anom = model.score_partial(anom_candidate)
    assert score_anom > model.control_limit_

    window_before_anom = model._window.to_array()
    model.fit_partial(anom_candidate)

    # Window must not have accepted the anomaly!
    assert len(model._window) == window_size
    assert np.array_equal(window_before_anom, model._window.to_array())


def test_absad_high_dimensional_subspace_anomaly() -> None:
    """Validates that ABSAD detects subtle subspace anomalies in high-dimensional streams.

    Scenario:
    - 50-dimensional feature space.
    - Anomaly deviates significantly in only 2 features (features 0 and 1).
    - Its full-space Euclidean distance to normal data is ~sqrt(50) ~= 7.07, identical to
      an average normal observation, so full-space distance-based detectors cannot distinguish it.
    - In the relevant 2D subspace, it exhibits an extreme anomaly signal.
    - ABSAD must select the anomalous subspace and assign it a significantly higher LOS.
    """
    rng = np.random.RandomState(42)
    n_dim = 50
    window_size = 80
    model = ABSAD(
        window_size=window_size,
        num_neighbors=30,
        num_references=20,
        theta=0.4,
        confidence=0.99,
    )

    # Fit normal observations
    normal_window = rng.randn(window_size, n_dim)
    for sample in normal_window:
        model.fit_partial(sample)

    normal_test = rng.randn(n_dim)

    # Subspace anomaly: deviates in features 0 and 1, normal in remaining 48 features
    anom_subspace = np.zeros(n_dim)
    anom_subspace[0] = 5.0
    anom_subspace[1] = 5.0

    # Ensure full-space Euclidean norm of anom is comparable to normal test point
    full_space_dist_normal = np.linalg.norm(normal_test)
    full_space_dist_anom = np.linalg.norm(anom_subspace)
    assert abs(full_space_dist_anom - full_space_dist_normal) < 2.0

    score_normal = model.score_partial(normal_test)
    score_anom = model.score_partial(anom_subspace)

    assert score_anom > score_normal


def test_absad_window_eviction_bounded_memory() -> None:
    window_size = 15
    model = ABSAD(window_size=window_size, num_neighbors=8, num_references=5)
    rng = np.random.RandomState(42)

    # Feed 60 streaming instances
    for _ in range(60):
        model.fit_partial(rng.randn(4))

    # Memory must stay bounded at window_size
    assert len(model._window) == window_size
    assert len(model._window_los) == window_size


def test_absad_duplicate_observations() -> None:
    model = ABSAD(window_size=20, num_neighbors=10, num_references=8)
    # Stream identical duplicate observations
    identical_point = np.array([3.0, -1.0, 2.0])
    for _ in range(20):
        model.fit_partial(identical_point)

    score = model.score_partial(identical_point)
    assert np.isfinite(score)
    assert score >= 0.0


def test_absad_warm_up_score_partial_is_zero() -> None:
    model = ABSAD(window_size=50, num_neighbors=20, num_references=10)
    rng = np.random.RandomState(42)

    # Only 5 samples arrived (less than num_references = 10)
    for _ in range(5):
        model.fit_partial(rng.randn(3))

    assert model.score_partial(rng.randn(3)) == 0.0


def test_absad_fit_score_batch() -> None:
    rng = np.random.RandomState(42)
    model = ABSAD(window_size=30, num_neighbors=15, num_references=10)
    X = rng.randn(40, 3)

    scores = model.fit_score(X)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (40,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
