from __future__ import annotations

import pytest

from pysad.models.absad import ABSAD


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
        ({"window_size": 100, "num_neighbors": 0}, "num_neighbors must be between 1 and window_size"),
        ({"window_size": 100, "num_neighbors": 101}, "num_neighbors must be between 1 and window_size"),
        ({"window_size": 100, "num_neighbors": 50, "num_references": 0}, "num_references must be between 1 and num_neighbors"),
        ({"window_size": 100, "num_neighbors": 50, "num_references": 51}, "num_references must be between 1 and num_neighbors"),
        ({"theta": -0.1}, "theta must be non-negative"),
        ({"confidence": 0.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.5}, r"confidence must be in \(0, 1\)"),
    ],
)
def test_absad_invalid_parameters(kwargs: dict[str, float | int], expected_msg: str) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        ABSAD(**kwargs)


import numpy as np

from pysad.models.absad import (
    _calculate_feature_relevance,
    _normalize_window_and_point,
    _select_reference_set,
)


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
