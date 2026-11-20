"""TreeSHAP explainability for tree ensembles.

Implements the TreeSHAP algorithm from Lundberg et al. (2020) "From local explanations
to global understanding with explainable AI for trees", Nature Machine Intelligence
:cite:`lundberg2020consistent`. Computes exact SHAP values for tree ensemble models
in streaming anomaly detection.

The implementation computes exact Shapley values by explicit enumeration of all
subsets of features on the decision path (exponential in path length, equivalent
to the paper's Algorithm 1 approach). For a tree of depth M, this requires
evaluating 2^M subsets. Since isolation forest trees have bounded depth (typically
< 10 for default parameters), the exact exponential algorithm is efficient.

Deviations from the paper:
- The paper's Algorithm 2 describes a polynomial-time EXTEND/UNWIND dynamic
  programming approach. This implementation uses explicit subset enumeration
  (exponential in path length) which is exact and equivalent for the bounded
  tree depths in streaming isolation forests.
- The `left_fractions` parameter (cover fractions) defaults to 0.5 for all internal
  nodes. In the paper, these are computed from training data cover statistics.
  For OnlineIsolationForest, cover fractions are not tracked during streaming
  learning. Users can provide custom `left_fractions` if available.
- Each occurrence of a feature on the decision path is treated as a separate
  position for Shapley computation. The final SHAP values are the sum over all
  positions of that feature. This correctly handles repeated features on a path
  (common in isolation forests).
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
    """Compute SHAP values for a single tree using exact TreeSHAP via explicit subset enumeration.

    This implements exact Shapley value computation by enumerating all subsets of
    positions on the decision path. Each internal node on the path is a position.
    For a path with M internal nodes, there are 2^M subsets. The Shapley value for
    each position is computed as the weighted average of its marginal contributions
    across all subsets of the other positions. Final SHAP values per feature are
    the sum over all positions of that feature on the path.

    This handles repeated features correctly: each occurrence of a feature on the
    decision path is a separate position with its own Shapley value.

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
    # path_nodes holds the internal node indices on the path (excluding the leaf).
    path_nodes: list[int] = []
    node = 0
    while features[node] != -1:
        path_nodes.append(node)
        if x[features[node]] < thresholds[node]:
            node = left_children[node]
        else:
            node = right_children[node]
    # path_nodes now contains M internal nodes; the leaf is not included.

    M = len(path_nodes)
    if M == 0:
        return phi

    # Step 3: Exact TreeSHAP via explicit subset enumeration over path positions.
    # Each position i (0 <= i < M) corresponds to path_nodes[i].
    # For each position i, compute its Shapley value:
    # phi_i = sum_{S subset of P\{i}} weight(|S|) * (f(S U {i}) - f(S))
    # where P is the set of M positions, weight(|S|) = 1 / (M * C(M-1, |S|))
    # and f(S) is the expected value when positions in S are known (follow x's path)
    # and positions not in S are unknown (use cover-weighted average).

    # Precompute expected values for all 2^M subsets.
    # subset_values[mask] = expected value when positions in mask are known.
    # mask is an M-bit integer where bit k corresponds to position k (path_nodes[k]).
    num_subsets = 1 << M
    subset_values = np.zeros(num_subsets, dtype=np.float64)

    for mask in range(num_subsets):
        # Compute expected value for this subset of known positions.
        # Start at root, traverse tree: known positions follow x's path, unknown use cover fractions.
        val = 0.0
        stack = [(0, 1.0)]  # (node, weight)
        while stack:
            current_node, weight = stack.pop()
            feat = features[current_node]
            if feat == -1:  # leaf
                val += weight * values[current_node]
                continue

            # Determine if this node is on the path and if its position is known in this subset
            on_path = False
            path_pos = -1
            for pos, pn in enumerate(path_nodes):
                if pn == current_node:
                    on_path = True
                    path_pos = pos
                    break

            if on_path and (mask & (1 << path_pos)):
                # Position is known: follow x's actual path
                x_val = x[feat]
                goes_right = x_val >= thresholds[current_node]
                child = right_children[current_node] if goes_right else left_children[current_node]
                stack.append((child, weight))
            else:
                # Position is unknown (or node not on path): use cover-weighted average
                left = left_children[current_node]
                right = right_children[current_node]
                stack.append((left, weight * left_fractions[current_node]))
                stack.append((right, weight * right_fractions[current_node]))

        subset_values[mask] = val

    # Step 4: Compute Shapley values for each position on the path, accumulate by feature.
    for i in range(M):
        node_i = path_nodes[i]
        feat_i = features[node_i]
        total_contribution = 0.0

        # Sum over all subsets S of the other M-1 positions
        other_positions = [j for j in range(M) if j != i]
        for k in range(M):  # subset size k
            weight = 1.0 / (M * math.comb(M - 1, k))
            for S in combinations(other_positions, k):
                mask_without = 0
                for j in S:
                    mask_without |= 1 << j
                mask_with = mask_without | (1 << i)

                val_with = subset_values[mask_with]
                val_without = subset_values[mask_without]
                total_contribution += weight * (val_with - val_without)

        phi[feat_i] += total_contribution

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
