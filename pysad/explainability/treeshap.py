"""TreeSHAP explainability for tree ensembles.

Implements the TreeSHAP algorithm from Lundberg et al. (2020) "Consistent Individualized
Feature Attribution for Tree Ensembles", Nature MI. Computes exact SHAP values for
tree ensemble models in streaming anomaly detection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _treeshap_tree(
    x: NDArray[np.float64],
    features: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    left_children: NDArray[np.intp],
    right_children: NDArray[np.intp],
    values: NDArray[np.float64],
    num_features: int,
    left_fractions: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute SHAP values for a single tree using the TreeSHAP algorithm (Algorithm 2).

    Args:
        x: Instance to explain, shape (num_features,).
        features: Split feature for each node, shape (num_nodes,). Leaf nodes have feature = -1.
        thresholds: Split threshold for each node, shape (num_nodes,).
        left_children: Left child index for each node, shape (num_nodes,).
        right_children: Right child index for each node, shape (num_nodes,).
        values: Output value for each node, shape (num_nodes,). Only leaf nodes have non-zero values.
        num_features: Number of features.
        left_fractions: Fraction of samples going left at each node, shape (num_nodes,).
            If None, assumes 0.5 for all internal nodes.

    Returns:
        SHAP values for each feature, shape (num_features,).
    """
    num_nodes = len(features)
    phi = np.zeros(num_features, dtype=np.float64)

    if left_fractions is None:
        left_fractions = np.full(num_nodes, 0.5, dtype=np.float64)
    right_fractions = 1.0 - left_fractions

    # Step 1: Compute unconditional expected values for all nodes (bottom-up).
    # unconditional[node] = expected value if we continue from this node with all features unknown.
    unconditional = np.zeros(num_nodes, dtype=np.float64)

    # Process nodes in reverse order (post-order).
    # Since trees are typically stored in DFS order, we can process in reverse.
    for node in range(num_nodes - 1, -1, -1):
        feat = features[node]
        if feat == -1:  # leaf
            unconditional[node] = values[node]
        else:
            left = left_children[node]
            right = right_children[node]
            unconditional[node] = (
                left_fractions[node] * unconditional[left]
                + right_fractions[node] * unconditional[right]
            )

    # Step 2: Traverse the tree for instance x, computing SHAP contributions.
    # We use the TreeSHAP recursive algorithm (Algorithm 2).
    # The algorithm maintains a path of (node, fraction, on_path) and computes
    # contributions when a feature is encountered on the path.

    def traverse(node: int, fraction: float, on_path: bool) -> float:
        """Returns the expected value from this node given the path so far.

        Also updates phi for features on the path.
        """
        feat = features[node]
        if feat == -1:  # leaf
            return float(values[node])

        threshold = thresholds[node]
        left = left_children[node]
        right = right_children[node]

        x_val = x[feat]
        goes_right = x_val >= threshold

        if on_path:
            # On the path of x: feature is known.
            # Contribution = fraction * (value_if_known - value_if_unknown)
            # value_if_known: follow x's path
            known_child = right if goes_right else left
            known_val = traverse(known_child, fraction, True)

            # value_if_unknown: weighted average of both children (using unconditional values)
            unknown_val = (
                left_fractions[node] * unconditional[left]
                + right_fractions[node] * unconditional[right]
            )

            # Update SHAP for this feature
            phi[feat] += fraction * (known_val - unknown_val)

            return known_val
        else:
            # Off the path: feature is unknown, use unconditional value directly.
            return float(unconditional[node])

    traverse(0, 1.0, True)

    return phi


def treeshap_ensemble(
    x: NDArray[np.float64],
    trees_features: list[NDArray[np.intp]],
    trees_thresholds: list[NDArray[np.float64]],
    trees_left: list[NDArray[np.intp]],
    trees_right: list[NDArray[np.intp]],
    trees_values: list[NDArray[np.float64]],
    num_features: int,
    trees_left_fractions: list[NDArray[np.float64] | None] | None = None,
) -> NDArray[np.float64]:
    """Compute SHAP values for a tree ensemble by averaging over trees.

    Args:
        x: Instance to explain, shape (num_features,).
        trees_features: List of feature arrays for each tree.
        trees_thresholds: List of threshold arrays for each tree.
        trees_left: List of left child arrays for each tree.
        trees_right: List of right child arrays for each tree.
        trees_values: List of value arrays for each tree.
        num_features: Number of features.
        trees_left_fractions: Optional list of left fraction arrays for each tree.

    Returns:
        SHAP values for each feature, shape (num_features,).
    """
    total_phi = np.zeros(num_features, dtype=np.float64)
    num_trees = len(trees_features)
    if num_trees == 0:
        return total_phi

    if trees_left_fractions is None:
        trees_left_fractions = [None] * num_trees

    for features, thresholds, left, right, values, left_fracs in zip(
        trees_features,
        trees_thresholds,
        trees_left,
        trees_right,
        trees_values,
        trees_left_fractions,
        strict=True,
    ):
        if len(features) == 0:
            continue
        phi = _treeshap_tree(x, features, thresholds, left, right, values, num_features, left_fracs)
        total_phi += phi

    return total_phi / num_trees


def treeshap_ensemble_weighted(
    x: NDArray[np.float64],
    trees_features: list[NDArray[np.intp]],
    trees_thresholds: list[NDArray[np.float64]],
    trees_left: list[NDArray[np.intp]],
    trees_right: list[NDArray[np.intp]],
    trees_values: list[NDArray[np.float64]],
    tree_weights: NDArray[np.float64],
    num_features: int,
    trees_left_fractions: list[NDArray[np.float64] | None] | None = None,
) -> NDArray[np.float64]:
    """Compute weighted SHAP values for a tree ensemble.

    Args:
        x: Instance to explain, shape (num_features,).
        trees_features: List of feature arrays for each tree.
        trees_thresholds: List of threshold arrays for each tree.
        trees_left: List of left child arrays for each tree.
        trees_right: List of right child arrays for each tree.
        trees_values: List of value arrays for each tree.
        tree_weights: Weight for each tree, shape (num_trees,).
        num_features: Number of features.
        trees_left_fractions: Optional list of left fraction arrays for each tree.

    Returns:
        SHAP values for each feature, shape (num_features,).
    """
    total_phi = np.zeros(num_features, dtype=np.float64)
    total_weight = float(tree_weights.sum())
    if total_weight == 0.0:
        return total_phi

    if trees_left_fractions is None:
        trees_left_fractions = [None] * len(trees_features)

    for features, thresholds, left, right, values, weight, left_fracs in zip(
        trees_features,
        trees_thresholds,
        trees_left,
        trees_right,
        trees_values,
        tree_weights,
        trees_left_fractions,
        strict=True,
    ):
        if len(features) == 0 or weight == 0.0:
            continue
        phi = _treeshap_tree(x, features, thresholds, left, right, values, num_features, left_fracs)
        total_phi += phi * weight

    return total_phi / total_weight
