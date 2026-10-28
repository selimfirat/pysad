# Adapted from ineveLoppiliF/Online-Isolation-Forest (revision e528c86).
# MIT License
#
# Copyright (c) 2025 Filippo Leveni
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


def _path_length(num_samples: int, max_leaf_samples: int) -> float:
    if num_samples <= max_leaf_samples:
        return 0.0
    return float(np.log(num_samples / max_leaf_samples) / np.log(4.0))


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
    def __init__(self, max_leaf_samples: int) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.data_size = 0
        self.depth_limit = 0.0
        self.root: _OnlineIsolationNode | None = None

    def learn_one(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        # Retain the reference's sampling draw with subsample=1 for seeded parity.
        rng.random_sample()
        self.data_size += 1
        self.depth_limit = _path_length(self.data_size, self.max_leaf_samples)
        if self.root is None:
            self.root = self._build(X[None, :], depth=0, rng=rng)
            return
        self.root = self._learn(self.root, X, rng)

    def _learn(
        self, node: _OnlineIsolationNode, X: np.ndarray | None, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        if X is not None:
            node.data_size += 1
            np.minimum(node.min_values, X, out=node.min_values)
            np.maximum(node.max_values, X, out=node.max_values)
        if node.children is None:
            if (
                node.data_size >= self.max_leaf_samples * 2**node.depth
                and node.depth < self.depth_limit
            ):
                samples = rng.uniform(
                    low=node.min_values,
                    high=node.max_values,
                    size=(node.data_size, len(node.min_values)),
                )
                return self._build(samples, node.depth, rng)
            return node

        goes_right = X is not None and X[node.dim] >= node.split_value
        # The reference revisits empty branches too: a growing depth limit can
        # allow an existing leaf to split even without receiving this point.
        for i, child in enumerate(node.children):
            node.children[i] = self._learn(child, X if i == goes_right else None, rng)
        return node

    def _build(
        self, samples: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        node = _OnlineIsolationNode(samples, depth)
        if node.data_size < self.max_leaf_samples * 2**depth or depth >= self.depth_limit:
            return node
        node.dim = int(rng.choice(samples.shape[1]))
        node.split_value = float(rng.uniform(node.min_values[node.dim], node.max_values[node.dim]))
        left = samples[:, node.dim] < node.split_value
        node.children = [
            self._build(samples[left], depth=depth + 1, rng=rng),
            self._build(samples[~left], depth=depth + 1, rng=rng),
        ]
        return node

    def unlearn_one(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        rng.random_sample()
        self.data_size -= 1
        self.depth_limit = _path_length(self.data_size, self.max_leaf_samples)
        assert self.root is not None
        self._unlearn(self.root, X)

    def _unlearn(self, node: _OnlineIsolationNode, X: np.ndarray | None) -> None:
        if X is not None:
            node.data_size -= 1
        if node.children is None:
            return
        if node.data_size < self.max_leaf_samples * 2**node.depth:
            self._unbuild(node)
            return
        goes_right = X is not None and X[node.dim] >= node.split_value
        for i, child in enumerate(node.children):
            self._unlearn(child, X if i == goes_right else None)
        left, right = node.children
        np.minimum(left.min_values, right.min_values, out=node.min_values)
        np.maximum(left.max_values, right.max_values, out=node.max_values)

    def _unbuild(self, node: _OnlineIsolationNode) -> None:
        if node.children is None:
            return
        for child in node.children:
            self._unbuild(child)
        left, right = node.children
        np.minimum(left.min_values, right.min_values, out=node.min_values)
        np.maximum(left.max_values, right.max_values, out=node.max_values)
        node.children = None

    def predict_one(self, X: np.ndarray) -> float:
        node = self.root
        if node is None:
            return 0.0
        while node.children is not None:
            node = node.children[int(X[node.dim] >= node.split_value)]
        return node.depth + _path_length(node.data_size, self.max_leaf_samples)


class OnlineIsolationForest(BaseModel):
    r"""Scores instances with Online-iForest :cite:`leveni2024online`.

    Follows the binary, adaptive, non-subsampled variant of the
    `authors' reference implementation
    <https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/e528c86b77219791d026d732f618efb0c7bde3c1>`_.
    Trees learn each new point and forget the oldest point after ``window_size``.
    A leaf at depth ``k`` splits after ``max_leaf_samples * 2**k`` observations,
    subject to the current depth limit. Scores approach one for more anomalous
    instances. Memory and per-instance work stay bounded as the stream grows.

    Differences from the paper, following the reference implementation:

    * The path correction is ``log(n / max_leaf_samples) / log(4)`` for binary
      trees, rather than the paper's base-2 logarithm. It is zero for counts
      below ``max_leaf_samples``.
    * The depth limit and score normalization use the active sample count, up
      to ``window_size``, rather than the paper's fixed window size. Learning
      precedes forgetting, so a tree can briefly hold ``window_size + 1`` points.
      Machine epsilon is added to the score denominator during initialization.
    * Synthetic points are drawn before choosing the split. Rebuilding replaces
      the parent's bounds with synthetic bounds and can create several levels
      at once. All branches are revisited as the depth limit grows, including
      branches that did not receive the current point.
    * Forgetting refreshes internal bounds from the children, and removing a
      split recursively merges descendant supports. Algorithm 3 only updates
      support when removing a split and uses the immediate child supports.

    PySAD adaptations: ``fit_score_partial`` scores before learning. To follow
    Algorithm 1's order, call ``fit_partial`` then ``score_partial``. An unfitted
    model scores 1.0, following the
    `CapyMOA port <https://github.com/adaptive-machine-learning/CapyMOA/blob/3e255b1/src/capymoa/anomaly/_online_isolation_forest.py>`_;
    the original reference cannot score an unfitted forest.
    ``random_state`` controls one shared NumPy ``RandomState``,
    preserving the original reference's random draw order for single-point
    updates. Unlike CapyMOA, it does not create separate generators for each tree.
    Binary adaptive trees use every point; the reference-only fixed growth,
    nonbinary splits, independent subsampling, and parallel batch processing are
    not exposed. Incoming points are copied so callers can reuse their buffers.

    Args:
        num_trees (int): Number of trees (Default=32, as in the paper and
            CapyMOA; the original reference defaults to 100).
        max_leaf_samples (int): Base split threshold (Default=32).
        window_size (int): Number of recent instances retained for forgetting
            (Default=2048).
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
        if window_size < 1:
            raise ValueError("window_size must be positive.")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.window_size = window_size
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.normalization_ = 0.0
        self.trees_ = [
            _OnlineIsolationTree(
                max_leaf_samples=max_leaf_samples,
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
                tree.unlearn_one(old_x, self.rng_)
        self.normalization_ = _path_length(len(self.window_), self.max_leaf_samples)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores an instance without learning it.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to score.

        Returns:
            float: Anomaly score in [0, 1], or 1.0 before learning.
        """
        if not self.window_:
            return 1.0
        depth = sum(tree.predict_one(X) for tree in self.trees_) / self.num_trees
        return float(2.0 ** (-depth / (self.normalization_ + np.finfo(float).eps)))
