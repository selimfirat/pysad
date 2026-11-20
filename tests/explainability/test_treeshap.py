"""Tests for the TreeSHAP explainability module."""

import numpy as np

from pysad.explainability import treeshap_ensemble, treeshap_ensemble_weighted


def _simple_tree():
    """A simple binary tree for testing.

    Structure:
         0 (feature 0, threshold 0.5)
        / \
       1   2 (leaf, value=1.0) (leaf, value=0.0)

    Features: [0, 1]
    """
    features = np.array([0, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 1, 2], dtype=np.intp)
    right = np.array([2, 1, 2], dtype=np.intp)
    values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return features, thresholds, left, right, values


def test_treeshap_simple_tree():
    """Test TreeSHAP on a simple tree with known SHAP values."""
    features, thresholds, left, right, values = _simple_tree()

    # Instance x = [0.3, 0.7] goes left at root (feature 0 < 0.5)
    # Expected: feature 0 gets all the credit since it determines the path
    x = np.array([0.3, 0.7], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # Feature 0 should have positive SHAP (it pushed towards value 1.0)
    # Feature 1 should have 0 SHAP (not used in tree)
    assert phi.shape == (2,)
    assert phi[0] > 0.0  # Feature 0 contributes
    assert abs(phi[1]) < 1e-10  # Feature 1 doesn't contribute


def test_treeshap_tree_right_branch():
    """Test TreeSHAP when instance goes right."""
    features, thresholds, left, right, values = _simple_tree()

    # Instance x = [0.7, 0.3] goes right at root (feature 0 >= 0.5)
    x = np.array([0.7, 0.3], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # Feature 0 should have negative SHAP (it pushed towards value 0.0)
    assert phi.shape == (2,)
    assert phi[0] < 0.0
    assert abs(phi[1]) < 1e-10


def test_treeshap_ensemble_averaging():
    """Test that ensemble averages SHAP values across trees."""
    features, thresholds, left, right, values = _simple_tree()

    # Two identical trees
    x = np.array([0.3, 0.7], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features, features],
        [thresholds, thresholds],
        [left, left],
        [right, right],
        [values, values],
        num_features=2,
    )

    # Should be same as single tree
    phi_single = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    np.testing.assert_allclose(phi, phi_single)


def test_treeshap_weighted_ensemble():
    """Test weighted ensemble SHAP."""
    features, thresholds, left, right, values = _simple_tree()

    x = np.array([0.3, 0.7], dtype=np.float64)

    # Weight first tree 2x, second tree 1x
    phi = treeshap_ensemble_weighted(
        x,
        [features, features],
        [thresholds, thresholds],
        [left, left],
        [right, right],
        [values, values],
        np.array([2.0, 1.0]),
        num_features=2,
    )

    # Should equal single tree (since trees are identical)
    phi_single = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    np.testing.assert_allclose(phi, phi_single)


def test_treeshap_zero_weight():
    """Test that zero weight trees are ignored."""
    features, thresholds, left, right, values = _simple_tree()

    x = np.array([0.3, 0.7], dtype=np.float64)

    # First tree has weight 0, second has weight 1
    phi = treeshap_ensemble_weighted(
        x,
        [features, features],
        [thresholds, thresholds],
        [left, left],
        [right, right],
        [values, values],
        np.array([0.0, 1.0]),
        num_features=2,
    )

    # Should equal single tree
    phi_single = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    np.testing.assert_allclose(phi, phi_single)


def test_treeshap_empty_tree():
    """Test TreeSHAP with empty tree (single leaf)."""
    features = np.array([-1], dtype=np.intp)
    thresholds = np.array([0.0], dtype=np.float64)
    left = np.array([0], dtype=np.intp)
    right = np.array([0], dtype=np.intp)
    values = np.array([0.5], dtype=np.float64)

    x = np.array([0.3, 0.7], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # No features used, all SHAP should be 0
    assert np.all(phi == 0.0)


def test_treeshap_multiple_features():
    """Test TreeSHAP with a tree using multiple features."""
    # Tree where both features genuinely contribute:
    #       0 (feature 0, thresh 0.5)
    #      / \
    #     1   2 (feature 1, thresh 0.5)
    #    / \  / \
    #   3  4 5   6
    #  1.0 0.0 0.0 1.0
    # Left branch: feature 1 -> left=1.0, right=0.0
    # Right branch: feature 1 -> left=0.0, right=1.0
    features = np.array([0, 1, 1, -1, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 5, 3, 4, 5, 6], dtype=np.intp)
    right = np.array([2, 4, 6, 3, 4, 5, 6], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # x = [0.3, 0.3] -> goes left at root, then left at feature 1 -> leaf 3 (value 1.0)
    x = np.array([0.3, 0.3], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # Both features contribute to reaching leaf 3
    assert phi.shape == (2,)
    assert phi[0] > 0.0  # Feature 0 pushed left
    assert phi[1] > 0.0  # Feature 1 pushed left


def test_treeshap_reproducibility():
    """Test that TreeSHAP gives reproducible results."""
    features, thresholds, left, right, values = _simple_tree()
    x = np.array([0.3, 0.7], dtype=np.float64)

    phi1 = treeshap_ensemble(x, [features], [thresholds], [left], [right], [values], num_features=2)
    phi2 = treeshap_ensemble(x, [features], [thresholds], [left], [right], [values], num_features=2)

    np.testing.assert_array_equal(phi1, phi2)


def test_treeshap_repeated_feature_on_path():
    """Test TreeSHAP with a feature repeated on the decision path.

    Isolation forests often split on the same feature multiple times along a path.
    This test ensures each occurrence is treated as a separate position and SHAP
    values are correctly accumulated.
    """
    # Tree where feature 0 appears twice on the left path:
    #       0 (feature 0, thresh 0.5)
    #      /
    #     1 (feature 0, thresh 0.25)
    #    / \
    #   3   4
    #  1.0  0.0
    # Right branch is a leaf with value 0.0
    # x = [0.1, 0.5] -> left at root (0 < 0.5), left at node 1 (0 < 0.25) -> leaf 3 (value 1.0)
    features = np.array([0, 0, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 1, 3, 4], dtype=np.intp)
    right = np.array([2, 4, 1, 3, 4], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    x = np.array([0.1, 0.5], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # Feature 0 appears twice on the path and should get all the SHAP value
    # Feature 1 is not used and should get 0
    assert phi.shape == (2,)
    assert phi[0] > 0.0  # Feature 0 contributes (both positions)
    assert abs(phi[1]) < 1e-10  # Feature 1 doesn't contribute


def test_treeshap_repeated_feature_different_branches():
    """Test TreeSHAP with a feature used on different branches."""
    # Tree where feature 0 is used at root, and feature 1 is used on both branches:
    #       0 (feature 0, thresh 0.5)
    #      / \
    #     1   2 (feature 1, thresh 0.5)
    #    / \  / \
    #   3  4 5   6
    #  1.0 0.0 0.0 1.0
    # Both branches use feature 1, but with different outcomes.
    # x = [0.3, 0.3] -> left at root (feature 0), left at node 1 (feature 1) -> leaf 3 (value 1.0)
    features = np.array([0, 1, 1, -1, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 5, 3, 4, 5, 6], dtype=np.intp)
    right = np.array([2, 4, 6, 3, 4, 5, 6], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

    x = np.array([0.3, 0.3], dtype=np.float64)

    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    # Both features contribute to reaching leaf 3
    assert phi.shape == (2,)
    assert phi[0] > 0.0  # Feature 0 pushed left
    assert phi[1] > 0.0  # Feature 1 pushed left
