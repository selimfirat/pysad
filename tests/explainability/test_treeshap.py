"""Tests for the TreeSHAP explainability module."""

import math
from itertools import combinations

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
    """Test TreeSHAP when a feature appears multiple times along a decision path.

    Algorithm 2 unwinds previous occurrences of a repeated feature so that each
    unique feature acts as a single Shapley player with exact conditional expectation
    weights.
    """
    # Tree where feature 0 appears twice on the left path:
    #       0 (feature 0, thresh 0.5)
    #      / \
    #     1   2 (leaf, value 0.0)
    #    / \
    #   3   4
    #  1.0  0.0
    # x = [0.1, 0.5] -> left at root (0.1 < 0.5), left at node 1 (0.1 < 0.25) -> leaf 3 (value 1.0)
    features = np.array([0, 0, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 2, 3, 4], dtype=np.intp)
    right = np.array([2, 4, 2, 3, 4], dtype=np.intp)
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

    # Exact expectation is 0.5 * 0.5 * 1.0 = 0.25, so phi[0] = 1.0 - 0.25 = 0.75.
    np.testing.assert_allclose(phi, [0.75, 0.0])


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


def _brute_force_treeshap(
    x,
    features,
    thresholds,
    left,
    right,
    values,
    num_features,
    left_fractions=None,
):
    """Exact Algorithm 1 Shapley reference over all 2^|U| subsets of unique features."""
    left_fracs = (
        np.full(len(features), 0.5, dtype=np.float64) if left_fractions is None else left_fractions
    )
    used_features = sorted({int(f) for f in features if f >= 0})
    m = len(used_features)
    phi = np.zeros(num_features, dtype=np.float64)
    if m == 0:
        return phi

    def _eval_subset(subset):
        def _walk(node):
            feat = int(features[node])
            if feat < 0:
                return float(values[node])
            if feat in subset:
                nxt = int(left[node]) if x[feat] < thresholds[node] else int(right[node])
                return _walk(nxt)
            r_left = float(left_fracs[node])
            res = 0.0
            if r_left != 0.0:
                res += r_left * _walk(int(left[node]))
            if 1.0 - r_left != 0.0:
                res += (1.0 - r_left) * _walk(int(right[node]))
            return res

        return _walk(0)

    m_fact = math.factorial(m)
    exp_cache = {}
    for size in range(m + 1):
        for comb in combinations(used_features, size):
            subset = frozenset(comb)
            exp_cache[subset] = _eval_subset(subset)

    for feat in used_features:
        others = [f for f in used_features if f != feat]
        contrib = 0.0
        for size in range(m):
            weight = math.factorial(size) * math.factorial(m - 1 - size) / m_fact
            for comb in combinations(others, size):
                subset = frozenset(comb)
                contrib += weight * (exp_cache[subset | {feat}] - exp_cache[subset])
        phi[feat] = contrib
    return phi


def test_treeshap_dummy_feature_returns_zero():
    """A dummy split feature whose branches compute identical subtrees gets zero SHAP."""
    # f(x0, x1) = 10 if x1 < 0 else 0, even though root splits on x0 < 0.
    features = np.array([0, 1, 1, -1, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 5, 3, 4, 5, 6], dtype=np.intp)
    right = np.array([2, 4, 6, 3, 4, 5, 6], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0], dtype=np.float64)

    phi = treeshap_ensemble(
        np.array([-1.0, -1.0], dtype=np.float64),
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    np.testing.assert_allclose(phi, [0.0, 5.0], atol=1e-12)


def test_treeshap_matches_brute_force_on_random_trees():
    """Algorithm 2 matches brute-force Algorithm 1 and satisfies local accuracy."""
    rng = np.random.RandomState(42)
    num_features = 4

    def _build_random_tree(max_depth):
        feat_list = []
        thr_list = []
        left_list = []
        right_list = []
        val_list = []
        frac_list = []

        def _add_node(depth):
            idx = len(feat_list)
            feat_list.append(-1)
            thr_list.append(0.0)
            left_list.append(idx)
            right_list.append(idx)
            val_list.append(0.0)
            frac_list.append(0.5)

            if depth >= max_depth or (depth > 0 and rng.rand() < 0.3):
                val_list[idx] = float(rng.uniform(-5.0, 5.0))
                return idx

            feat_list[idx] = int(rng.randint(0, num_features))
            thr_list[idx] = float(rng.uniform(-1.0, 1.0))
            frac_list[idx] = float(rng.uniform(0.1, 0.9))
            left_list[idx] = _add_node(depth + 1)
            right_list[idx] = _add_node(depth + 1)
            return idx

        _add_node(0)
        return (
            np.array(feat_list, dtype=np.intp),
            np.array(thr_list, dtype=np.float64),
            np.array(left_list, dtype=np.intp),
            np.array(right_list, dtype=np.intp),
            np.array(val_list, dtype=np.float64),
            np.array(frac_list, dtype=np.float64),
        )

    def _expected_tree_value(node, features, left, right, values, left_fracs):
        if features[node] < 0:
            return float(values[node])
        r = 0.5 if left_fracs is None else float(left_fracs[node])
        return r * _expected_tree_value(
            int(left[node]), features, left, right, values, left_fracs
        ) + (1.0 - r) * _expected_tree_value(
            int(right[node]), features, left, right, values, left_fracs
        )

    for _ in range(10):
        features, thresholds, left, right, values, left_fracs = _build_random_tree(max_depth=5)
        x = rng.uniform(-1.5, 1.5, size=num_features)

        for pass_fracs in (None, left_fracs):
            fracs_arg = [pass_fracs] if pass_fracs is not None else None
            phi = treeshap_ensemble(
                x,
                [features],
                [thresholds],
                [left],
                [right],
                [values],
                num_features=num_features,
                trees_left_fractions=fracs_arg,
            )
            expected = _brute_force_treeshap(
                x,
                features,
                thresholds,
                left,
                right,
                values,
                num_features,
                pass_fracs,
            )
            np.testing.assert_allclose(phi, expected, atol=1e-10)

            # Check local accuracy against tree output minus expected output.
            node = 0
            while features[node] >= 0:
                node = left[node] if x[features[node]] < thresholds[node] else right[node]
            pred = float(values[node])
            expected_val = _expected_tree_value(0, features, left, right, values, pass_fracs)
            np.testing.assert_allclose(phi.sum(), pred - expected_val, atol=1e-10)


def test_treeshap_repeated_feature_symmetry():
    """Redundant repeated splits preserve symmetry between equally contributing features."""
    # Symmetric AND function 1[x0 < 0.5 and x1 < 0.5] with a redundant split on x0 < 0.25.
    # Node 0: x0 < 0.5 -> (1, 2)
    # Node 1: x1 < 0.5 -> (3, 4=leaf 0.0)
    # Node 2: x1 < 0.5 -> (5=leaf 0.0, 6=leaf 0.0)
    # Node 3: x0 < 0.25 -> (7=leaf 1.0, 8=leaf 1.0)
    features = np.array([0, 1, 1, 0, -1, -1, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.5, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 5, 7, 4, 5, 6, 7, 8], dtype=np.intp)
    right = np.array([2, 4, 6, 8, 4, 5, 6, 7, 8], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    phi = treeshap_ensemble(
        np.array([0.1, 0.1], dtype=np.float64),
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=2,
    )

    np.testing.assert_allclose(phi, [0.375, 0.375], atol=1e-12)


def test_treeshap_zero_and_one_left_fractions():
    """Zero and one cover fractions execute safely and match brute-force Algorithm 1."""
    # Tree with repeated feature 0 and feature 1:
    # Node 0: x0 < 0.5 -> (1, 2)
    # Node 1: x1 < 0.5 -> (3, 4=leaf 2.0)
    # Node 2: x1 < 0.5 -> (5=leaf 4.0, 6=leaf 8.0)
    # Node 3: x0 < 0.2 -> (7=leaf 10.0, 8=leaf 6.0)
    features = np.array([0, 1, 1, 0, -1, -1, -1, -1, -1], dtype=np.intp)
    thresholds = np.array([0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([1, 3, 5, 7, 4, 5, 6, 7, 8], dtype=np.intp)
    right = np.array([2, 4, 6, 8, 4, 5, 6, 7, 8], dtype=np.intp)
    values = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 8.0, 10.0, 6.0], dtype=np.float64)

    fraction_cases = [
        np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        np.array([0.4, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    ]
    points = [
        np.array([0.1, 0.1], dtype=np.float64),
        np.array([0.1, 0.9], dtype=np.float64),
        np.array([0.8, 0.1], dtype=np.float64),
        np.array([0.8, 0.9], dtype=np.float64),
    ]

    for left_fracs in fraction_cases:
        for x in points:
            phi = treeshap_ensemble(
                x,
                [features],
                [thresholds],
                [left],
                [right],
                [values],
                num_features=2,
                trees_left_fractions=[left_fracs],
            )
            expected = _brute_force_treeshap(
                x, features, thresholds, left, right, values, 2, left_fracs
            )
            assert np.all(np.isfinite(phi))
            np.testing.assert_allclose(phi, expected, atol=1e-10)


def test_treeshap_depth_20_chain_tree():
    """Algorithm 2 handles depth-20 trees efficiently with exact symmetry and accuracy."""
    depth = 20
    num_nodes = 2 * depth + 1
    features = np.full(num_nodes, -1, dtype=np.intp)
    thresholds = np.zeros(num_nodes, dtype=np.float64)
    left = np.arange(num_nodes, dtype=np.intp)
    right = np.arange(num_nodes, dtype=np.intp)
    values = np.zeros(num_nodes, dtype=np.float64)

    for i in range(depth):
        features[i] = i
        thresholds[i] = 0.5
        left[i] = i + 1
        right[i] = depth + 1 + i
    values[depth] = 1.0

    x = np.zeros(depth, dtype=np.float64)
    phi = treeshap_ensemble(
        x,
        [features],
        [thresholds],
        [left],
        [right],
        [values],
        num_features=depth,
    )

    expected_each = (1.0 - 2.0**-depth) / depth
    np.testing.assert_allclose(phi, np.full(depth, expected_each), atol=1e-10)
