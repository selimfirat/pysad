from __future__ import annotations

import numbers
from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


def _get_random_path_length(
    branching_factor: int, max_leaf_samples: int, num_samples: float
) -> float:
    if num_samples < max_leaf_samples:
        return 0.0
    return float(np.log(num_samples / max_leaf_samples) / np.log(2 * branching_factor))


class _OnlineIsolationNode:
    __slots__ = (
        "children",
        "data_size",
        "depth",
        "dim",
        "max_values",
        "min_values",
        "split_values",
    )

    def __init__(
        self,
        data_size: int,
        depth: int,
        min_values: np.ndarray,
        max_values: np.ndarray,
        children: list[_OnlineIsolationNode] | None = None,
        dim: int | None = None,
        split_values: np.ndarray | None = None,
    ) -> None:
        self.data_size = data_size
        self.depth = depth
        self.min_values = min_values
        self.max_values = max_values
        self.children = children
        self.dim = dim
        self.split_values = split_values


class _OnlineIsolationTree:
    def __init__(
        self,
        max_leaf_samples: int,
        growth_criterion: str,
        subsample: float,
        branching_factor: int,
    ) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.growth_criterion = growth_criterion
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.data_size = 0
        self.depth_limit = 0.0
        self.root: _OnlineIsolationNode | None = None

    def _build_leaf(self, X: np.ndarray, depth: int) -> _OnlineIsolationNode:
        return _OnlineIsolationNode(
            data_size=1,
            depth=depth,
            min_values=X.copy(),
            max_values=X.copy(),
        )

    def _recursive_build(
        self, data: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        multiplier = 2**depth if self.growth_criterion == "adaptive" else 1
        if len(data) < self.max_leaf_samples * multiplier or depth >= self.depth_limit:
            if len(data) == 0:
                mins = np.full(data.shape[1], np.inf)
                maxs = np.full(data.shape[1], -np.inf)
            else:
                mins = data.min(axis=0)
                maxs = data.max(axis=0)
            return _OnlineIsolationNode(
                data_size=len(data),
                depth=depth,
                min_values=mins,
                max_values=maxs,
            )

        num_features = data.shape[1]
        dim = int(rng.randint(0, num_features))
        col = data[:, dim]
        p_min, p_max = float(col.min()), float(col.max())

        if p_min == p_max:
            split_values = np.full(self.branching_factor - 1, p_min)
        else:
            split_values = np.sort(rng.uniform(p_min, p_max, size=self.branching_factor - 1))

        indices = np.searchsorted(split_values, col)
        children = []
        for i in range(self.branching_factor):
            child_node = self._recursive_build(data[indices == i], depth + 1, rng)
            children.append(child_node)

        return _OnlineIsolationNode(
            data_size=len(data),
            depth=depth,
            min_values=data.min(axis=0),
            max_values=data.max(axis=0),
            children=children,
            dim=dim,
            split_values=split_values,
        )

    def learn_one(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        if self.subsample < 1.0 and rng.uniform(0.0, 1.0) >= self.subsample:
            return
        self.data_size += 1
        self.depth_limit = _get_random_path_length(
            self.branching_factor, self.max_leaf_samples, self.data_size
        )
        if self.root is None:
            self.root = self._build_leaf(X, 0)
        else:
            self.root = self._learn_one_node(self.root, X, rng)

    def _learn_one_node(
        self, node: _OnlineIsolationNode, X: np.ndarray, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        node.data_size += 1
        node.min_values = np.minimum(node.min_values, X)
        node.max_values = np.maximum(node.max_values, X)

        if node.children is None:
            multiplier = 2**node.depth if self.growth_criterion == "adaptive" else 1
            if (
                node.data_size >= self.max_leaf_samples * multiplier
                and node.depth < self.depth_limit
            ):
                data_sampled = rng.uniform(
                    node.min_values, node.max_values, size=(node.data_size, len(X))
                )
                return self._recursive_build(data_sampled, node.depth, rng)
            return node

        assert node.split_values is not None and node.dim is not None
        if len(node.split_values) == 1:
            child_idx = 0 if X[node.dim] <= node.split_values[0] else 1
        else:
            child_idx = int(np.searchsorted(node.split_values, X[node.dim]))

        assert node.children is not None
        node.children[child_idx] = self._learn_one_node(node.children[child_idx], X, rng)
        return node

    def unlearn_one(self, old_x: np.ndarray, rng: np.random.RandomState) -> None:
        if self.subsample < 1.0 and rng.uniform(0.0, 1.0) >= self.subsample:
            return
        self.data_size -= 1
        self.depth_limit = _get_random_path_length(
            self.branching_factor, self.max_leaf_samples, self.data_size
        )
        if self.root is not None:
            self.root = self._unlearn_one_node(self.root, old_x)

    def _unlearn_one_node(
        self, node: _OnlineIsolationNode, old_x: np.ndarray
    ) -> _OnlineIsolationNode:
        node.data_size -= 1
        if node.children is None:
            return node

        multiplier = 2**node.depth if self.growth_criterion == "adaptive" else 1
        if node.data_size < self.max_leaf_samples * multiplier:
            return self._recursive_unbuild(node)

        assert node.split_values is not None and node.dim is not None
        if len(node.split_values) == 1:
            child_idx = 0 if old_x[node.dim] <= node.split_values[0] else 1
        else:
            child_idx = int(np.searchsorted(node.split_values, old_x[node.dim]))

        assert node.children is not None
        node.children[child_idx] = self._unlearn_one_node(node.children[child_idx], old_x)

        active_children = [c for c in node.children if c.data_size > 0]
        if active_children:
            node.min_values = np.min([c.min_values for c in active_children], axis=0)
            node.max_values = np.max([c.max_values for c in active_children], axis=0)
        return node

    def _recursive_unbuild(self, node: _OnlineIsolationNode) -> _OnlineIsolationNode:
        if node.children is None:
            return node
        for i in range(len(node.children)):
            node.children[i] = self._recursive_unbuild(node.children[i])
        active_children = [c for c in node.children if c.data_size > 0]
        if active_children:
            node.min_values = np.min([c.min_values for c in active_children], axis=0)
            node.max_values = np.max([c.max_values for c in active_children], axis=0)
        node.children = None
        node.dim = None
        node.split_values = None
        return node

    def predict_one(self, X: np.ndarray) -> float:
        if self.root is None:
            return 0.0
        node = self.root
        while node.children is not None:
            assert node.split_values is not None and node.dim is not None
            if len(node.split_values) == 1:
                child_idx = 0 if X[node.dim] <= node.split_values[0] else 1
            else:
                child_idx = int(np.searchsorted(node.split_values, X[node.dim]))
            node = node.children[child_idx]
        return float(
            node.depth
            + _get_random_path_length(self.branching_factor, self.max_leaf_samples, node.data_size)
        )


class OnlineIsolationForest(BaseModel):
    """Scores each instance with an online isolation forest :cite:`leveni2024online`.

    Online-iForest updates an ensemble of online isolation trees instance by instance.
    The trees partition the feature space into multi-resolution bounding boxes. When a leaf
    accumulates enough points and the depth limit allows, it splits by drawing synthetic
    points uniformly from its bounding box. As older instances exceed ``window_size``, they
    are unlearned and underpopulated nodes are pruned back to leaves, allowing the forest
    to adapt to concept drift without periodic refits.

    Anomaly scores are normalized between 0 and 1 using the expected path length:

    .. math::

        s(x) = 2^{-\\frac{\\bar{h}(x)}{c(\\omega) + \\epsilon}}

    where :math:`\\bar{h}(x)` is the mean path length across trees and :math:`c(\\omega)` is the
    random path length for the active window size :math:`\\omega`. Higher scores indicate greater
    anomalousness.

    Args:
        num_trees (int): Number of online isolation trees in the forest (Default=32).
        max_leaf_samples (int): Minimum number of samples in a leaf before splitting (Default=32).
        window_size (int): Size of the sliding window for unlearning old instances (Default=2048).
        growth_criterion (str): Tree growth schedule, either 'adaptive' or 'fixed'. Under
            'adaptive', the split threshold doubles with tree depth (Default='adaptive').
        subsample (float): Fraction of instances sampled per tree in (0, 1] (Default=1.0).
        branching_factor (int): Branching factor of each node (Default=2).
        split (str): Splitting strategy. Currently only 'axisparallel' is supported (Default='axisparallel').
        random_state (int, np.random.RandomState, or None): Seed or random number generator. None
            draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If any parameter violates its constraints.
    """

    num_trees: int
    max_leaf_samples: int
    window_size: int
    growth_criterion: str
    subsample: float
    branching_factor: int
    split: str
    random_state: int | np.random.RandomState | None
    rng: np.random.RandomState
    trees: list[_OnlineIsolationTree]
    data_window: deque[np.ndarray]
    data_size: int
    normalization_factor: float

    def __init__(
        self,
        num_trees: int = 32,
        max_leaf_samples: int = 32,
        window_size: int = 2048,
        growth_criterion: str = "adaptive",
        subsample: float = 1.0,
        branching_factor: int = 2,
        split: str = "axisparallel",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if (
            isinstance(num_trees, bool)
            or not isinstance(num_trees, numbers.Integral)
            or num_trees < 1
        ):
            raise ValueError(f"num_trees must be a positive integer, got {num_trees!r}.")
        if (
            isinstance(max_leaf_samples, bool)
            or not isinstance(max_leaf_samples, numbers.Integral)
            or max_leaf_samples < 1
        ):
            raise ValueError(
                f"max_leaf_samples must be a positive integer, got {max_leaf_samples!r}."
            )
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, numbers.Integral)
            or window_size < 1
        ):
            raise ValueError(f"window_size must be a positive integer, got {window_size!r}.")
        if growth_criterion not in ("adaptive", "fixed"):
            raise ValueError(
                f"growth_criterion must be 'adaptive' or 'fixed', got {growth_criterion!r}."
            )
        if (
            isinstance(subsample, bool)
            or not isinstance(subsample, numbers.Real)
            or not 0.0 < float(subsample) <= 1.0
        ):
            raise ValueError(f"subsample must be a real number in (0, 1], got {subsample!r}.")
        if (
            isinstance(branching_factor, bool)
            or not isinstance(branching_factor, numbers.Integral)
            or branching_factor < 2
        ):
            raise ValueError(f"branching_factor must be an integer >= 2, got {branching_factor!r}.")
        if split != "axisparallel":
            raise ValueError(f"split must be 'axisparallel', got {split!r}.")

        self.num_trees = int(num_trees)
        self.max_leaf_samples = int(max_leaf_samples)
        self.window_size = int(window_size)
        self.growth_criterion = growth_criterion
        self.subsample = float(subsample)
        self.branching_factor = int(branching_factor)
        self.split = split
        self.random_state = random_state

        self.rng = check_random_state(random_state)
        self.trees = [
            _OnlineIsolationTree(
                max_leaf_samples=self.max_leaf_samples,
                growth_criterion=self.growth_criterion,
                subsample=self.subsample,
                branching_factor=self.branching_factor,
            )
            for _ in range(self.num_trees)
        ]
        self.data_window: deque[np.ndarray] = deque()
        self.data_size = 0
        self.normalization_factor = 0.0

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.rng is np.random.mtrand._rand:
            state["rng"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.rng = check_random_state(self.rng)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OnlineIsolationForest:
        """Updates the model with the next instance. ``y`` is accepted for API consistency but ignored.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): Ignored since the model is unsupervised (Default=None).

        Returns:
            OnlineIsolationForest: self.
        """
        x = np.asarray(X, dtype=np.float64)

        self.data_size += 1
        self.normalization_factor = _get_random_path_length(
            self.branching_factor, self.max_leaf_samples, self.data_size * self.subsample
        )
        for tree in self.trees:
            tree.learn_one(x, self.rng)

        self.data_window.append(x)
        if len(self.data_window) > self.window_size:
            old_x = self.data_window.popleft()
            self.data_size -= 1
            self.normalization_factor = _get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size * self.subsample
            )
            for tree in self.trees:
                tree.unlearn_one(old_x, self.rng)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: Anomaly score in [0, 1]. Higher indicates more anomalous.
        """
        if self.data_size == 0:
            return 0.0

        x = np.asarray(X, dtype=np.float64)
        depths = [tree.predict_one(x) for tree in self.trees]
        mean_depth = float(np.mean(depths))

        return float(2.0 ** (-mean_depth / (self.normalization_factor + np.finfo(float).eps)))
