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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class _PathState:
    """Subset-weight polynomial buffers tracked along one root-to-leaf path."""

    feature_index: list[int]
    zero_fraction: list[float]
    one_fraction: list[float]
    weight: list[float]

    @classmethod
    def allocate(cls, max_path_len: int) -> _PathState:
        """Allocate pre-sized buffers for a path of up to `max_path_len` entries."""
        return cls(
            feature_index=[-1] * max_path_len,
            zero_fraction=[0.0] * max_path_len,
            one_fraction=[0.0] * max_path_len,
            weight=[0.0] * max_path_len,
        )

    def extend(
        self,
        unique_depth: int,
        zero_fraction: float,
        one_fraction: float,
        feature_index: int,
    ) -> None:
        """Extend the subset-weight polynomial by one feature entry."""
        self.feature_index[unique_depth] = feature_index
        self.zero_fraction[unique_depth] = zero_fraction
        self.one_fraction[unique_depth] = one_fraction
        self.weight[unique_depth] = 1.0 if unique_depth == 0 else 0.0

        m_w = self.weight
        for i in range(unique_depth - 1, -1, -1):
            m_w[i + 1] += one_fraction * m_w[i] * (i + 1) / (unique_depth + 1)
            m_w[i] = zero_fraction * m_w[i] * (unique_depth - i) / (unique_depth + 1)

    def unwind(self, unique_depth: int, path_index: int) -> None:
        """Remove a feature entry at `path_index` from the path polynomial in place."""
        one_fraction = self.one_fraction[path_index]
        zero_fraction = self.zero_fraction[path_index]
        m_w = self.weight

        if one_fraction != 0.0:
            next_one = m_w[unique_depth]
            for i in range(unique_depth - 1, -1, -1):
                orig_w = m_w[i]
                unwound_w = next_one * (unique_depth + 1) / (one_fraction * (i + 1))
                m_w[i] = unwound_w
                next_one = orig_w - unwound_w * zero_fraction * (unique_depth - i) / (
                    unique_depth + 1
                )
        else:
            for i in range(unique_depth - 1, -1, -1):
                m_w[i] = m_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))

        for i in range(path_index, unique_depth):
            self.feature_index[i] = self.feature_index[i + 1]
            self.zero_fraction[i] = self.zero_fraction[i + 1]
            self.one_fraction[i] = self.one_fraction[i + 1]

    def unwound_sum(self, unique_depth: int, path_index: int) -> float:
        """Compute the sum of subset weights obtained by unwinding `path_index`."""
        one_fraction = self.one_fraction[path_index]
        zero_fraction = self.zero_fraction[path_index]
        m_w = self.weight
        total = 0.0

        if one_fraction != 0.0:
            next_one = m_w[unique_depth]
            for i in range(unique_depth - 1, -1, -1):
                unwound_w = next_one * (unique_depth + 1) / (one_fraction * (i + 1))
                total += unwound_w
                next_one = m_w[i] - unwound_w * zero_fraction * (unique_depth - i) / (
                    unique_depth + 1
                )
            return total

        for i in range(unique_depth - 1, -1, -1):
            total += m_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))
        return total

    def find_feature(self, split_index: int, unique_depth: int) -> int:
        """Return the path slot matching `split_index`, or `-1` if absent."""
        for k in range(1, unique_depth + 1):
            if self.feature_index[k] == split_index:
                return k
        return -1

    def snapshot(self, length: int) -> tuple[list[int], list[float], list[float], list[float]]:
        """Copy the first `length` polynomial entries before visiting two children."""
        return (
            self.feature_index[:length],
            self.zero_fraction[:length],
            self.one_fraction[:length],
            self.weight[:length],
        )

    def restore(
        self, length: int, saved: tuple[list[int], list[float], list[float], list[float]]
    ) -> None:
        """Restore the first `length` polynomial entries after visiting the hot child."""
        self.feature_index[:length] = saved[0]
        self.zero_fraction[:length] = saved[1]
        self.one_fraction[:length] = saved[2]
        self.weight[:length] = saved[3]


@dataclass(frozen=True)
class _TreeArrays:
    """Packed arrays describing a single decision tree for TreeSHAP."""

    features: NDArray[np.intp]
    thresholds: NDArray[np.float64]
    left_children: NDArray[np.intp]
    right_children: NDArray[np.intp]
    values: NDArray[np.float64]
    left_fractions: NDArray[np.float64]


def _accumulate_leaf(
    state: _PathState,
    phi: NDArray[np.float64],
    unique_depth: int,
    leaf_value: float,
) -> None:
    """Accumulate Shapley contributions at a leaf node across active path features."""
    for i in range(1, unique_depth + 1):
        weight = state.unwound_sum(unique_depth, i)
        phi[state.feature_index[i]] += (
            weight * (state.one_fraction[i] - state.zero_fraction[i]) * leaf_value
        )


def _recurse_tree(
    x: NDArray[np.float64],
    tree: _TreeArrays,
    state: _PathState,
    phi: NDArray[np.float64],
    node: int,
    unique_depth: int,
) -> None:
    """Descend one node in TreeSHAP Algorithm 2 after path extension."""
    split_index = int(tree.features[node])
    if split_index < 0:
        _accumulate_leaf(state, phi, unique_depth, float(tree.values[node]))
        return

    left_frac = float(tree.left_fractions[node])
    go_left = float(x[split_index]) < float(tree.thresholds[node])
    hot_child = int(tree.left_children[node] if go_left else tree.right_children[node])
    cold_child = int(tree.right_children[node] if go_left else tree.left_children[node])
    hot_zero = left_frac if go_left else 1.0 - left_frac
    cold_zero = 1.0 - left_frac if go_left else left_frac

    hot_one = 1.0
    incoming_depth = unique_depth
    incoming_index = state.find_feature(split_index, unique_depth)
    if incoming_index >= 0:
        hot_zero *= state.zero_fraction[incoming_index]
        hot_one *= state.one_fraction[incoming_index]
        cold_zero *= state.zero_fraction[incoming_index]
        state.unwind(unique_depth, incoming_index)
        incoming_depth -= 1

    visit_hot = hot_zero != 0.0 or hot_one != 0.0
    visit_cold = cold_zero != 0.0
    next_depth = incoming_depth + 1

    if visit_hot and visit_cold:
        saved = state.snapshot(next_depth)
        state.extend(next_depth, hot_zero, hot_one, split_index)
        _recurse_tree(x, tree, state, phi, hot_child, next_depth)
        state.restore(next_depth, saved)
        state.extend(next_depth, cold_zero, 0.0, split_index)
        _recurse_tree(x, tree, state, phi, cold_child, next_depth)
    elif visit_hot:
        state.extend(next_depth, hot_zero, hot_one, split_index)
        _recurse_tree(x, tree, state, phi, hot_child, next_depth)
    elif visit_cold:
        state.extend(next_depth, cold_zero, 0.0, split_index)
        _recurse_tree(x, tree, state, phi, cold_child, next_depth)


def _treeshap_tree(
    x: NDArray[np.float64],
    tree: _TreeArrays,
    num_features: int,
) -> NDArray[np.float64]:
    """Compute exact SHAP values for one tree using TreeSHAP Algorithm 2."""
    phi = np.zeros(num_features, dtype=np.float64)
    if len(tree.features) == 0:
        return phi

    state = _PathState.allocate(min(len(tree.features), num_features) + 2)
    state.extend(0, 1.0, 1.0, -1)
    _recurse_tree(x, tree, state, phi, 0, 0)
    return phi


def _make_tree_arrays(
    features: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    values: NDArray[np.float64],
    left_fracs: NDArray[np.float64] | None,
) -> _TreeArrays:
    """Wrap raw per-tree arrays and fill default cover fractions when omitted."""
    resolved_fracs = (
        np.full(len(features), 0.5, dtype=np.float64) if left_fracs is None else left_fracs
    )
    return _TreeArrays(
        features=features,
        thresholds=thresholds,
        left_children=left,
        right_children=right,
        values=values,
        left_fractions=resolved_fracs,
    )


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
        tree = _make_tree_arrays(features, thresholds, left, right, values, left_fracs)
        total_phi += _treeshap_tree(x, tree, num_features)

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
        tree = _make_tree_arrays(features, thresholds, left, right, values, left_fracs)
        total_phi += _treeshap_tree(x, tree, num_features) * weight

    return total_phi / total_weight
