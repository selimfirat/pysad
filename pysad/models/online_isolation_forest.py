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


def _path_length(num_samples: float, max_leaf_samples: int, branching_factor: int) -> float:
    if num_samples <= max_leaf_samples:
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

    def __init__(self, data: np.ndarray, depth: int) -> None:
        self.data_size = len(data)
        self.depth = depth
        self.min_values = data.min(axis=0, initial=np.inf)
        self.max_values = data.max(axis=0, initial=-np.inf)
        self.children: list[_OnlineIsolationNode] | None = None
        self.dim = 0
        self.split_values: np.ndarray | None = None

    def child_index(self, X: np.ndarray) -> int:
        assert self.split_values is not None
        if len(self.split_values) == 1:
            return int(X[self.dim] >= self.split_values[0])
        return int(np.searchsorted(self.split_values, X[self.dim], side="right"))


class _OnlineIsolationTree:
    def __init__(
        self, max_leaf_samples: int, growth_criterion: str, subsample: float, branching_factor: int
    ) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.growth_criterion = growth_criterion
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.data_size = 0
        self.depth_limit = 0.0
        self.root: _OnlineIsolationNode | None = None

    def _threshold(self, depth: int) -> int:
        return self.max_leaf_samples * (2**depth if self.growth_criterion == "adaptive" else 1)

    def learn_one(self, X: np.ndarray, rng: np.random.RandomState) -> bool:
        if rng.random_sample() >= self.subsample:
            return False
        self.data_size += 1
        self.depth_limit = _path_length(
            num_samples=self.data_size,
            max_leaf_samples=self.max_leaf_samples,
            branching_factor=self.branching_factor,
        )
        if self.root is None:
            self.root = self._build(X[None, :], depth=0, rng=rng)
            return True
        self.root = self._learn(self.root, X, rng)
        return True

    def _learn(
        self, node: _OnlineIsolationNode, X: np.ndarray | None, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        if X is not None:
            node.data_size += 1
            np.minimum(node.min_values, X, out=node.min_values)
            np.maximum(node.max_values, X, out=node.max_values)
        if node.children is None:
            if node.data_size >= self._threshold(node.depth) and node.depth < self.depth_limit:
                samples = rng.uniform(
                    low=node.min_values,
                    high=node.max_values,
                    size=(node.data_size, len(node.min_values)),
                )
                return self._build(samples, node.depth, rng)
            return node

        index = node.child_index(X) if X is not None else None
        # The reference revisits empty branches too: a growing depth limit can
        # allow an existing leaf to split even without receiving this point.
        for i, child in enumerate(node.children):
            node.children[i] = self._learn(child, X if i == index else None, rng)
        return node

    def _build(
        self, samples: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        node = _OnlineIsolationNode(samples, depth)
        if node.data_size < self._threshold(depth) or depth >= self.depth_limit:
            return node
        node.dim = int(rng.choice(samples.shape[1]))
        node.split_values = np.sort(
            rng.uniform(
                low=node.min_values[node.dim],
                high=node.max_values[node.dim],
                size=self.branching_factor - 1,
            )
        )
        indices = np.searchsorted(node.split_values, samples[:, node.dim], side="right")
        node.children = [
            self._build(samples[indices == i], depth=depth + 1, rng=rng)
            for i in range(self.branching_factor)
        ]
        return node

    def unlearn_one(self, X: np.ndarray, rng: np.random.RandomState, selected: bool) -> None:
        # Keep the reference's draw order, but expire the original selection.
        rng.random_sample()
        if not selected:
            return
        self.data_size -= 1
        self.depth_limit = _path_length(
            num_samples=self.data_size,
            max_leaf_samples=self.max_leaf_samples,
            branching_factor=self.branching_factor,
        )
        assert self.root is not None
        self._unlearn(self.root, X)

    def _unlearn(self, node: _OnlineIsolationNode, X: np.ndarray | None) -> None:
        if X is not None:
            node.data_size -= 1
        if node.children is None:
            return
        if node.data_size < self._threshold(node.depth):
            self._unbuild(node)
            return
        index = node.child_index(X) if X is not None else None
        for i, child in enumerate(node.children):
            self._unlearn(child, X if i == index else None)
        self._refresh_bounds(node)

    def _refresh_bounds(self, node: _OnlineIsolationNode) -> None:
        assert node.children is not None
        np.minimum.reduce([c.min_values for c in node.children], out=node.min_values)
        np.maximum.reduce([c.max_values for c in node.children], out=node.max_values)

    def _unbuild(self, node: _OnlineIsolationNode) -> None:
        if node.children is None:
            return
        for child in node.children:
            self._unbuild(child)
        self._refresh_bounds(node)
        node.children = None

    def predict_one(self, X: np.ndarray) -> float:
        node = self.root
        if node is None:
            return 0.0
        while node.children is not None:
            node = node.children[node.child_index(X)]
        return node.depth + _path_length(
            num_samples=node.data_size,
            max_leaf_samples=self.max_leaf_samples,
            branching_factor=self.branching_factor,
        )


class OnlineIsolationForest(BaseModel):
    r"""Scores instances with Online-iForest :cite:`leveni2024online`.

    Follows the `authors' reference implementation
    <https://github.com/ineveLoppiliF/Online-Isolation-Forest/tree/e528c86b77219791d026d732f618efb0c7bde3c1>`_,
    with fixed or adaptive growth, subsampling and multiway axis-parallel splits.
    Trees learn new points and forget expired points from a bounded window.
    With adaptive growth, a node at depth ``k`` splits after
    ``max_leaf_samples * 2**k`` observations; fixed growth uses
    ``max_leaf_samples`` at every depth. Higher scores are more anomalous.

    Differences from the paper, following the reference implementation:

    * The path correction is ``log(n / max_leaf_samples) / log(2 * branching_factor)``.
      With binary trees this is a base-4 logarithm, rather than the paper's
      base-2 logarithm. It is zero for counts below ``max_leaf_samples``.
      Fixed growth, nonbinary splits and subsampling are reference-code options
      outside the binary adaptive algorithm presented in the paper.
    * The depth limit uses each tree's active sample count. Score normalization
      uses the active window count times ``subsample``, rather than the paper's
      fixed window size. Learning precedes forgetting, so a tree can briefly
      hold ``window_size + 1`` points. Machine epsilon is added to the score
      denominator to handle initialization.
    * Rebuilding samples the node's current count ``h``, which can exceed the
      paper's split threshold. It draws those synthetic points before choosing
      the split, replaces the parent's bounds with synthetic bounds and can
      create several levels at once. The paper draws threshold-many samples
      after choosing the split and retains the parent's observed bounds.
    * All branches are revisited, including branches without the current point.
      Updates therefore visit all existing nodes, rather than only the point's
      path as in the paper; its path-only update complexity does not apply.
    * Forgetting refreshes internal bounds from all children. Removing a split
      recursively merges descendant supports. Algorithm 3 only updates support
      when removing a split and uses the immediate child supports.

    Scoring order:
        ``fit_score_partial(X)`` scores before learning, following PySAD's
        prequential convention. For the paper's learn-then-score order, call
        ``fit_partial(X)`` followed by ``score_partial(X)``. Both orders leave
        scoring itself free of updates. An unfitted forest returns 0.0; the
        original reference cannot score an unfitted forest. A tree without any
        selected points contributes zero depth instead of raising an error.

    Subsampling correction:
        The reference independently resamples expired points, which can remove
        observations a tree never learned and corrupt its count. This version
        stores each point's original per-tree selection and uses it when the
        point expires. It retains the reference's forgetting RNG draw for draw
        order consistency. Scores can therefore differ from the reference after
        expiration when ``subsample < 1``. With ``subsample=1``, trained scores
        follow the reference's single-point updates up to floating-point rounding.

    Input ownership:
        Every fitted point is copied into a separate float64 array. The model
        owns that copy until it expires, so callers may overwrite or reuse their
        input buffers without changing the history used for forgetting. The
        original reference may retain array views into caller-owned data, so
        later mutations can change what it forgets. With subsampling, the window
        also retains the point's original selection flags until expiration.

    Batch behavior:
        Inherited ``fit`` and ``fit_score`` apply one update per row. The
        reference's ``learn_batch`` processes the supplied rows together before
        expiring old rows, so one bulk update can build different trees from
        repeated single-point updates. Numerical reference parity is restricted
        to one-point batches, not arbitrary bulk batches or the paper's batched
        experiments. Parallel batch execution is not exposed. ``window_size``
        must be finite and positive; the reference's unbounded-window mode is
        not supported. Memory and per-instance work stay bounded as the stream grows.

    ``random_state`` controls one shared NumPy ``RandomState``. None uses NumPy's
    global state, as in the original reference. Unlike CapyMOA, this model does
    not create independent generators for each tree.

    Args:
        num_trees (int): Number of trees (Default=100, as in the reference code).
        max_leaf_samples (int): Base split threshold (Default=32).
        window_size (int): Number of recent instances retained for forgetting
            (Default=2048).
        growth_criterion (str): 'adaptive' doubles the split threshold per depth;
            'fixed' keeps it constant. Corresponds to the reference's ``type``
            parameter (Default='adaptive').
        subsample (float): Probability of including a point in each tree, in
            (0, 1] (Default=1.0).
        branching_factor (int): Number of children per split, at least two
            (Default=2).
        split (str): Splitting strategy. Only 'axisparallel' is supported, as for
            the reference's ``metric`` parameter (Default='axisparallel').
        random_state (int, np.random.RandomState, or None): Seed or random number
            generator. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_leaf_samples: int = 32,
        window_size: int = 2048,
        growth_criterion: str = "adaptive",
        subsample: float = 1.0,
        branching_factor: int = 2,
        split: str = "axisparallel",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if num_trees < 1:
            raise ValueError("num_trees must be positive.")
        if max_leaf_samples < 1:
            raise ValueError("max_leaf_samples must be positive.")
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        if growth_criterion not in {"adaptive", "fixed"}:
            raise ValueError("growth_criterion must be 'adaptive' or 'fixed'.")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")
        if branching_factor < 2:
            raise ValueError("branching_factor must be at least two.")
        if split != "axisparallel":
            raise ValueError("split must be 'axisparallel'.")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.window_size = window_size
        self.growth_criterion = growth_criterion
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.split = split
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.normalization_ = 0.0
        self.trees_ = [
            _OnlineIsolationTree(
                max_leaf_samples=max_leaf_samples,
                growth_criterion=growth_criterion,
                subsample=subsample,
                branching_factor=branching_factor,
            )
            for _ in range(num_trees)
        ]
        self.window_: deque[tuple[np.ndarray, np.ndarray | None]] = deque()

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
        selected = np.empty(self.num_trees, dtype=bool) if self.subsample < 1.0 else None
        for i, tree in enumerate(self.trees_):
            learned = tree.learn_one(x, self.rng_)
            if selected is not None:
                selected[i] = learned
        self.window_.append((x, selected))
        if len(self.window_) > self.window_size:
            old_x, old_selected = self.window_.popleft()
            for i, tree in enumerate(self.trees_):
                tree.unlearn_one(
                    X=old_x,
                    rng=self.rng_,
                    selected=bool(old_selected[i]) if old_selected is not None else True,
                )
        self.normalization_ = _path_length(
            num_samples=len(self.window_) * self.subsample,
            max_leaf_samples=self.max_leaf_samples,
            branching_factor=self.branching_factor,
        )
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
        return float(2.0 ** (-depth / (self.normalization_ + np.finfo(float).eps)))
