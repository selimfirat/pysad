from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


def _path_length(num_samples: int, max_leaf_samples: int) -> float:
    if num_samples <= max_leaf_samples:
        return 0.0
    return float(np.log2(num_samples / max_leaf_samples))


class _OnlineIsolationNode:
    __slots__ = (
        "children",
        "data_size",
        "depth",
        "dim",
        "max_values",
        "min_values",
        "split_value",
    )

    def __init__(self, data: np.ndarray, depth: int) -> None:
        self.data_size = len(data)
        self.depth = depth
        self.min_values = data.min(axis=0, initial=np.inf)
        self.max_values = data.max(axis=0, initial=-np.inf)
        self.children: list[_OnlineIsolationNode] | None = None
        self.dim = 0
        self.split_value = 0.0


class _OnlineIsolationTree:
    def __init__(self, max_leaf_samples: int, depth_limit: float) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.depth_limit = depth_limit
        self.root: _OnlineIsolationNode | None = None

    def learn_one(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        if self.root is None:
            self.root = _OnlineIsolationNode(np.empty((0, len(X))), depth=0)
        node = self.root
        while True:
            node.data_size += 1
            np.minimum(node.min_values, X, out=node.min_values)
            np.maximum(node.max_values, X, out=node.max_values)
            if node.children is None:
                if (
                    node.data_size >= self.max_leaf_samples * 2**node.depth
                    and node.depth < self.depth_limit
                ):
                    self._split(node, rng)
                return
            node = node.children[int(X[node.dim] >= node.split_value)]

    def _split(self, node: _OnlineIsolationNode, rng: np.random.RandomState) -> None:
        # Algorithm 2 draws the split before the synthetic points and keeps the
        # parent's observed support. Only the children use synthetic bounds.
        node.dim = int(rng.randint(len(node.min_values)))
        node.split_value = float(rng.uniform(node.min_values[node.dim], node.max_values[node.dim]))
        samples = rng.uniform(
            low=node.min_values,
            high=node.max_values,
            size=(node.data_size, len(node.min_values)),
        )
        left = samples[:, node.dim] < node.split_value
        node.children = [
            _OnlineIsolationNode(samples[left], depth=node.depth + 1),
            _OnlineIsolationNode(samples[~left], depth=node.depth + 1),
        ]

    def unlearn_one(self, X: np.ndarray) -> None:
        node = self.root
        assert node is not None
        while True:
            node.data_size -= 1
            if node.children is None:
                return
            if node.data_size < self.max_leaf_samples * 2**node.depth:
                # Algorithm 3 merges the immediate children's supports when
                # removing a split, including their historical bounds.
                left, right = node.children
                np.minimum(left.min_values, right.min_values, out=node.min_values)
                np.maximum(left.max_values, right.max_values, out=node.max_values)
                node.children = None
                return
            node = node.children[int(X[node.dim] >= node.split_value)]

    def predict_one(self, X: np.ndarray) -> float:
        node = self.root
        if node is None:
            return 0.0
        while node.children is not None:
            node = node.children[int(X[node.dim] >= node.split_value)]
        return node.depth + _path_length(node.data_size, self.max_leaf_samples)


class OnlineIsolationForest(BaseModel):
    r"""Scores instances with Online-iForest :cite:`leveni2024online`.

    Implements the binary trees, adaptive split threshold, and sliding-window
    forgetting of Algorithms 1--4. Each instance updates every tree. A leaf at
    depth ``k`` splits after ``max_leaf_samples * 2**k`` observations, up to the
    fixed depth limit ``log2(window_size / max_leaf_samples)``. Child counts and
    bounds are initialized from uniform samples of the parent's bounding box.
    Underpopulated internal nodes merge their children when old points expire.

    The anomaly score is ``2 ** (-mean_depth / c)``, where
    ``c = log2(window_size / max_leaf_samples)``. Higher scores are more anomalous.
    The window and the depth limit bound memory and per-instance work independently
    of stream length.

    Deviations from the paper: an unfitted model scores 0.0. The leaf depth
    correction is zero for counts below ``max_leaf_samples``, avoiding negative
    depths during initialization and forgetting. As throughout PySAD,
    ``fit_score_partial`` scores before learning; the paper's Algorithm 1 scores
    after learning. For that order, call ``fit_partial`` then ``score_partial``.
    The authors' reference code uses a different path-length denominator and a
    growing depth limit; this implementation follows the paper's base-2 logarithm
    and fixed window instead. The reference's independent subsampling of learned
    and forgotten points, fixed-growth mode, and nonbinary splits are omitted.

    Args:
        num_trees (int): Number of trees (Default=32, as in the paper).
        max_leaf_samples (int): Base split threshold (Default=32, as in the paper).
        window_size (int): Number of recent instances retained for forgetting.
            Must exceed ``max_leaf_samples`` (Default=2048, as in the paper).
        random_state (int, np.random.RandomState, or None): Seed or random number
            generator. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        num_trees: int = 32,
        max_leaf_samples: int = 32,
        window_size: int = 2048,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if num_trees < 1:
            raise ValueError("num_trees must be positive.")
        if max_leaf_samples < 1:
            raise ValueError("max_leaf_samples must be positive.")
        if window_size <= max_leaf_samples:
            raise ValueError("window_size must exceed max_leaf_samples.")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.window_size = window_size
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.normalization_ = _path_length(window_size, max_leaf_samples)
        self.trees_ = [
            _OnlineIsolationTree(
                max_leaf_samples=max_leaf_samples,
                depth_limit=self.normalization_,
            )
            for _ in range(num_trees)
        ]
        self.window_: deque[np.ndarray] = deque()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.rng_ is np.random.mtrand._rand:
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.rng_ = check_random_state(self.rng_)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OnlineIsolationForest:
        """Learns one instance, then forgets the oldest if the window is full.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to learn.
            y (int or None): Ignored (Default=None).

        Returns:
            OnlineIsolationForest: self.
        """
        x = np.array(X, dtype=np.float64, copy=True)
        for tree in self.trees_:
            tree.learn_one(x, self.rng_)
        self.window_.append(x)
        if len(self.window_) > self.window_size:
            old_x = self.window_.popleft()
            for tree in self.trees_:
                tree.unlearn_one(old_x)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores an instance without learning it.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to score.

        Returns:
            float: Anomaly score in [0, 1], or 0.0 before learning.
        """
        if not self.window_:
            return 0.0
        depth = sum(tree.predict_one(X) for tree in self.trees_) / self.num_trees
        return float(2.0 ** (-depth / self.normalization_))
