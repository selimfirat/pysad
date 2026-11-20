"""TreeSHAP explainability for tree ensembles.

Implements the polynomial-time TreeSHAP Algorithm 2 from Lundberg et al. (2020),
"From local explanations to global understanding with explainable AI for trees",
Nature Machine Intelligence :cite:`lundberg2020consistent`. Computes exact Shapley
feature attributions in O(T L D^2) time across an ensemble of T trees with up to L
leaves and depth D.

Deviations from the paper:
- `left_fractions` defaults to 0.5 for all internal nodes when cover fractions are
  not provided. `OnlineIsolationForest.explain_partial` supplies live per-node
  child sample-count fractions computed from streaming `data_size` statistics.
- Split routing uses `x[feature] < threshold` for the left branch and
  `x[feature] >= threshold` for the right branch to match `OnlineIsolationForest`.
- Zero-cover branches where both zero and one path fractions vanish are pruned to
  avoid division by zero during unwinding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _extend(
    m_d: list[int],
    m_z: list[float],
    m_o: list[float],
    m_w: list[float],
    unique_depth: int,
    zero_fraction: float,
    one_fraction: float,
    feature_index: int,
) -> None:
    """Extend the subset-weight polynomial by one feature entry."""
    m_d[unique_depth] = feature_index
    m_z[unique_depth] = zero_fraction
    m_o[unique_depth] = one_fraction
    m_w[unique_depth] = 1.0 if unique_depth == 0 else 0.0

    for i in range(unique_depth - 1, -1, -1):
        m_w[i + 1] += one_fraction * m_w[i] * (i + 1) / (unique_depth + 1)
        m_w[i] = zero_fraction * m_w[i] * (unique_depth - i) / (unique_depth + 1)


def _unwind(
    m_d: list[int],
    m_z: list[float],
    m_o: list[float],
    m_w: list[float],
    unique_depth: int,
    path_index: int,
) -> None:
    """Remove a feature entry at `path_index` from the path polynomial in place."""
    one_fraction = m_o[path_index]
    zero_fraction = m_z[path_index]

    if one_fraction != 0.0:
        next_one = m_w[unique_depth]
        for i in range(unique_depth - 1, -1, -1):
            orig_w = m_w[i]
            unwound_w = next_one * (unique_depth + 1) / (one_fraction * (i + 1))
            m_w[i] = unwound_w
            next_one = orig_w - unwound_w * zero_fraction * (unique_depth - i) / (unique_depth + 1)
    else:
        for i in range(unique_depth - 1, -1, -1):
            m_w[i] = m_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        m_d[i] = m_d[i + 1]
        m_z[i] = m_z[i + 1]
        m_o[i] = m_o[i + 1]


def _unwound_sum(
    m_z: list[float],
    m_o: list[float],
    m_w: list[float],
    unique_depth: int,
    path_index: int,
) -> float:
    """Compute the sum of subset weights obtained by unwinding `path_index`."""
    one_fraction = m_o[path_index]
    zero_fraction = m_z[path_index]
    total = 0.0

    if one_fraction != 0.0:
        next_one = m_w[unique_depth]
        for i in range(unique_depth - 1, -1, -1):
            unwound_w = next_one * (unique_depth + 1) / (one_fraction * (i + 1))
            total += unwound_w
            next_one = m_w[i] - unwound_w * zero_fraction * (unique_depth - i) / (unique_depth + 1)
    else:
        for i in range(unique_depth - 1, -1, -1):
            total += m_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))

    return total


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
    """Compute exact SHAP values for one tree using TreeSHAP Algorithm 2.

    Args:
        x: Instance to explain, shape (num_features,).
        features: Feature index per node (`-1` for leaves).
        thresholds: Split threshold per node.
        left_children: Left child node index per node.
        right_children: Right child node index per node.
        values: Leaf output value per node.
        num_features: Total number of features.
        left_fractions: Optional fraction of samples routed left at each node.
            Defaults to 0.5 for every node when omitted.

    Returns:
        SHAP values for each feature, shape (num_features,).
    """
    phi = np.zeros(num_features, dtype=np.float64)
    if len(features) == 0:
        return phi

    left_fracs = (
        np.full(len(features), 0.5, dtype=np.float64) if left_fractions is None else left_fractions
    )

    max_path_len = min(len(features), num_features) + 2
    m_d = [-1] * max_path_len
    m_z = [0.0] * max_path_len
    m_o = [0.0] * max_path_len
    m_w = [0.0] * max_path_len

    def _recurse(
        node: int,
        unique_depth: int,
        parent_zero_fraction: float,
        parent_one_fraction: float,
        parent_feature_index: int,
    ) -> None:
        _extend(
            m_d,
            m_z,
            m_o,
            m_w,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        )

        split_index = int(features[node])
        if split_index < 0:
            leaf_value = float(values[node])
            for i in range(1, unique_depth + 1):
                weight = _unwound_sum(m_z, m_o, m_w, unique_depth, i)
                phi[m_d[i]] += weight * (m_o[i] - m_z[i]) * leaf_value
            return

        threshold = float(thresholds[node])
        left_frac = float(left_fracs[node])
        if float(x[split_index]) < threshold:
            hot_child = int(left_children[node])
            cold_child = int(right_children[node])
            hot_zero = left_frac
            cold_zero = 1.0 - left_frac
        else:
            hot_child = int(right_children[node])
            cold_child = int(left_children[node])
            hot_zero = 1.0 - left_frac
            cold_zero = left_frac

        hot_one = 1.0
        incoming_depth = unique_depth
        incoming_index = -1
        for k in range(1, unique_depth + 1):
            if m_d[k] == split_index:
                incoming_index = k
                break

        if incoming_index >= 0:
            hot_zero *= m_z[incoming_index]
            hot_one *= m_o[incoming_index]
            cold_zero *= m_z[incoming_index]
            _unwind(m_d, m_z, m_o, m_w, unique_depth, incoming_index)
            incoming_depth -= 1

        visit_hot = hot_zero != 0.0 or hot_one != 0.0
        visit_cold = cold_zero != 0.0
        next_depth = incoming_depth + 1

        if visit_hot and visit_cold:
            saved_len = next_depth
            saved_d = m_d[:saved_len]
            saved_z = m_z[:saved_len]
            saved_o = m_o[:saved_len]
            saved_w = m_w[:saved_len]
            _recurse(hot_child, next_depth, hot_zero, hot_one, split_index)
            m_d[:saved_len] = saved_d
            m_z[:saved_len] = saved_z
            m_o[:saved_len] = saved_o
            m_w[:saved_len] = saved_w
            _recurse(cold_child, next_depth, cold_zero, 0.0, split_index)
        elif visit_hot:
            _recurse(hot_child, next_depth, hot_zero, hot_one, split_index)
        elif visit_cold:
            _recurse(cold_child, next_depth, cold_zero, 0.0, split_index)

    _recurse(0, 0, 1.0, 1.0, -1)
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
