"""TreeSHAP explainability for tree ensembles.

Implements the TreeSHAP algorithm from Lundberg et al. (2020) "Consistent Individualized
Feature Attribution for Tree Ensembles", Nature MI. Computes exact SHAP values for
tree ensemble models in streaming anomaly detection.

The implementation follows Algorithm 2 from the paper, which computes exact Shapley
values by considering all subsets of features on the decision path. For a tree of
depth M, this requires evaluating 2^M subsets. Since isolation forest trees have
bounded depth (typically < 10 for default parameters), the exact exponential
algorithm is efficient and numerically equivalent to the polynomial-time EXTEND/UNWIND
formulation.

Deviations from the paper:
- The paper's Algorithm 2 describes a polynomial-time EXTEND/UNWIND dynamic
  programming approach. This implementation uses explicit subset enumeration which
  is exact and equivalent for the bounded tree depths in streaming isolation forests.
- The `left_fractions` parameter (cover fractions) defaults to 0.5 for all internal
  nodes. In the paper, these are computed from training data cover statistics.
  For OnlineIsolationForest, cover fractions are not tracked during streaming
  learning. Users can provide custom `left_fractions` if available.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

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
    """Compute SHAP values for a single tree using exact TreeSHAP (Algorithm 2).

    This implements the exact TreeSHAP algorithm from Lundberg et al. (2020) by
    enumerating all subsets of features on the decision path. For a path with M
    internal nodes, there are 2^M subsets. The Shapley value for each feature is
    computed as the weighted average of its marginal contributions across all
    subsets of the other features.

    Args:
        x: Instance to explain, shape (num_features,).
        features: Split feature for each node, shape (num_nodes,). Leaf nodes have feature = -1.
        thresholds: Split threshold for each node, shape (num_nodes,).
        left_children: Left child index for each node, shape (num_nodes,).
        right_children: Right child index for each node, shape (num_nodes,).
        values: Output value for each node, shape (num_nodes,). Only leaf nodes have non-zero values.
        num_features: Number of features.
        left_fractions: Fraction of samples going left at each node, shape (num_nodes,).
            If None, assumes 0.5 for all internal nodes. These are the "cover fractions"
            from the paper and should ideally be computed from training data statistics.

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

    # Step 2: Find the decision path for instance x.
    path: list[int] = []
    node = 0
    while features[node] != -1:
        path.append(node)
        if x[features[node]] < thresholds[node]:
            node = left_children[node]
        else:
            node = right_children[node]
    path.append(node)  # leaf node

    M = len(path) - 1  # number of internal nodes (features) on the path
    if M == 0:
        return phi

    path_features = [features[n] for n in path[:-1]]

    # Step 3: Exact TreeSHAP via explicit subset enumeration.
    # For each feature i on the path, compute its Shapley value:
    # phi_i = sum_{S subset of F\{i}} weight(|S|) * (f(S U {i}) - f(S))
    # where F is the set of features on the path, weight(|S|) = 1 / (M * C(M-1, |S|))
    # and f(S) is the expected value when features in S are known (follow x's path)
    # and features not in S are unknown (use cover-weighted average).

    # Precompute expected values for all 2^M subsets.
    # subset_values[mask] = expected value when features in mask are known.
    # mask is an M-bit integer where bit k corresponds to path_features[k].
    num_subsets = 1 << M
    subset_values = np.zeros(num_subsets, dtype=np.float64)

    for mask in range(num_subsets):
        # Compute expected value for this subset of known features.
        # Start at root, traverse tree: known features follow x's path, unknown use cover fractions.
        val = 0.0
        stack = [(0, 1.0)]  # (node, weight)
        while stack:
            current_node, weight = stack.pop()
            feat = features[current_node]
            if feat == -1:  # leaf
                val += weight * values[current_node]
                continue

            # Determine if this feature is on the path and if it's known in this subset
            on_path = False
            path_idx = -1
            for idx, pf in enumerate(path_features):
                if pf == feat:
                    on_path = True
                    path_idx = idx
                    break

            if on_path and (mask & (1 << path_idx)):
                # Feature is known: follow x's actual path
                x_val = x[feat]
                goes_right = x_val >= thresholds[current_node]
                child = right_children[current_node] if goes_right else left_children[current_node]
                stack.append((child, weight))
            else:
                # Feature is unknown: use cover-weighted average
                left = left_children[current_node]
                right = right_children[current_node]
                stack.append((left, weight * left_fractions[current_node]))
                stack.append((right, weight * right_fractions[current_node]))

        subset_values[mask] = val

    # Step 4: Compute Shapley values for each feature on the path.
    for i in range(M):
        feat = path_features[i]
        total_contribution = 0.0

        # Sum over all subsets S of the other M-1 features
        other_indices = [j for j in range(M) if j != i]
        for k in range(M):  # subset size k
            weight = 1.0 / (M * math.comb(M - 1, k))
            for S in combinations(other_indices, k):
                mask_without = 0
                for j in S:
                    mask_without |= 1 << j
                mask_with = mask_without | (1 << i)

                val_with = subset_values[mask_with]
                val_without = subset_values[mask_without]
                total_contribution += weight * (val_with - val_without)

        phi[feat] += total_contribution

    return phi


def treeshap_ensemble(
    x: NDArray[np.float64],
    trees_features: Sequence[NDArray[np.intp]],
    trees_thresholds: Sequence[NDArray[np.float64]],
    trees_left: Sequence[NDArray[np.intp]],
    trees_right: Sequence[NDArray[np.intp]],
    trees_values: Sequence[NDArray[np.float64]],
    num_features: int,
    trees_left_fractions: Sequence[NDArray[np.float64] | None] | None = None,
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
    trees_features: Sequence[NDArray[np.intp]],
    trees_thresholds: Sequence[NDArray[np.float64]],
    trees_left: Sequence[NDArray[np.intp]],
    trees_right: Sequence[NDArray[np.intp]],
    trees_values: Sequence[NDArray[np.float64]],
    tree_weights: NDArray[np.float64],
    num_features: int,
    trees_left_fractions: Sequence[NDArray[np.float64] | None] | None = None,
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
