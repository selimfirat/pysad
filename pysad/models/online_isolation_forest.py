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

import math
from collections import deque
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


def _path_length(num_samples: float, max_leaf_samples: int, inv_log: float) -> float:
    if num_samples <= max_leaf_samples:
        return 0.0
    return math.log(num_samples / max_leaf_samples) * inv_log


class _OnlineIsolationNode:
    __slots__ = (
        "children",
        "data_size",
        "depth",
        "dim",
        "max_values",
        "min_values",
        "slot",
        "split_val",
        "split_values",
        "threshold",
    )

    def __init__(self, data: np.ndarray, depth: int, threshold: int) -> None:
        self.data_size = len(data)
        self.depth = depth
        self.threshold = threshold
        self.min_values = data.min(axis=0, initial=np.inf)
        self.max_values = data.max(axis=0, initial=-np.inf)
        self.children: list[_OnlineIsolationNode] | None = None
        self.dim = 0
        self.split_val = 0.0
        self.split_values: np.ndarray | None = None
        self.slot = 0

    def child_index(self, X: np.ndarray) -> int:
        if self.split_values is None or len(self.split_values) == 1:
            return 1 if X[self.dim] >= self.split_val else 0
        return int(np.searchsorted(self.split_values, X[self.dim], side="right"))


class _OnlineIsolationTree:
    def __init__(
        self, max_leaf_samples: int, growth_criterion: str, subsample: float, branching_factor: int
    ) -> None:
        self.max_leaf_samples = max_leaf_samples
        self.growth_criterion = growth_criterion
        self._adaptive = growth_criterion == "adaptive"
        self.subsample = subsample
        self.branching_factor = branching_factor
        self._inv_log = 1.0 / math.log(2 * branching_factor)
        self.data_size = 0
        self.depth_limit = 0.0
        self._settled_limit = 0
        self.root: _OnlineIsolationNode | None = None
        self.max_depth = 0
        self.node_count = 0

    def _threshold(self, depth: int) -> int:
        return self.max_leaf_samples << depth if self._adaptive else self.max_leaf_samples

    def learn_one(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> tuple[bool, bool, _OnlineIsolationNode | None]:
        """Returns (selected, structure_changed, touched_leaf)."""
        if rng.random_sample() >= self.subsample:
            return False, False, None
        self.data_size += 1
        self.depth_limit = _path_length(
            num_samples=self.data_size,
            max_leaf_samples=self.max_leaf_samples,
            inv_log=self._inv_log,
        )
        if self.root is None:
            self.root = self._build(X[None, :], depth=0, rng=rng)
            self._settled_limit = math.ceil(self.depth_limit)
            return True, True, None

        ceil_limit = math.ceil(self.depth_limit)
        if ceil_limit > self._settled_limit:
            self.root, changed = self._learn_all(self.root, X, rng)
            self._settled_limit = ceil_limit
            if changed:
                return True, True, None
            # Find the leaf that X reached so its packed leaf_pl updates cleanly.
            curr = self.root
            while curr.children is not None:
                curr = curr.children[curr.child_index(X)]
            return True, False, curr

        return self._learn_path(X, rng)

    def _learn_path(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> tuple[bool, bool, _OnlineIsolationNode | None]:
        assert self.root is not None
        curr = self.root
        parent: _OnlineIsolationNode | None = None
        child_idx = 0
        binary = self.branching_factor == 2
        while curr.children is not None:
            curr.data_size += 1
            np.minimum(curr.min_values, X, out=curr.min_values)
            np.maximum(curr.max_values, X, out=curr.max_values)
            parent = curr
            if binary:
                child_idx = 1 if X[curr.dim] >= curr.split_val else 0
            else:
                assert curr.split_values is not None
                child_idx = int(np.searchsorted(curr.split_values, X[curr.dim], side="right"))
            curr = curr.children[child_idx]

        curr.data_size += 1
        np.minimum(curr.min_values, X, out=curr.min_values)
        np.maximum(curr.max_values, X, out=curr.max_values)
        if curr.data_size >= curr.threshold and curr.depth < self.depth_limit:
            samples = rng.uniform(
                low=curr.min_values,
                high=curr.max_values,
                size=(curr.data_size, len(curr.min_values)),
            )
            rebuilt = self._build(samples, curr.depth, rng)
            if parent is None:
                self.root = rebuilt
            else:
                assert parent.children is not None
                parent.children[child_idx] = rebuilt
            return True, True, None
        return True, False, curr

    def _learn_all(
        self, node: _OnlineIsolationNode, X: np.ndarray | None, rng: np.random.RandomState
    ) -> tuple[_OnlineIsolationNode, bool]:
        if X is not None:
            node.data_size += 1
            np.minimum(node.min_values, X, out=node.min_values)
            np.maximum(node.max_values, X, out=node.max_values)
        if node.children is None:
            if node.data_size >= node.threshold and node.depth < self.depth_limit:
                samples = rng.uniform(
                    low=node.min_values,
                    high=node.max_values,
                    size=(node.data_size, len(node.min_values)),
                )
                return self._build(samples, node.depth, rng), True
            return node, False

        index = node.child_index(X) if X is not None else -1
        # The reference revisits empty branches too: a growing depth limit can
        # allow an existing leaf to split even without receiving this point.
        changed = False
        for i, child in enumerate(node.children):
            new_child, child_changed = self._learn_all(child, X if i == index else None, rng)
            node.children[i] = new_child
            if child_changed:
                changed = True
        return node, changed

    def _build(
        self, samples: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> _OnlineIsolationNode:
        threshold = self._threshold(depth)
        node = _OnlineIsolationNode(samples, depth, threshold)
        if node.data_size < threshold or depth >= self.depth_limit:
            return node
        node.dim = int(rng.choice(samples.shape[1]))
        if self.branching_factor == 2:
            val = float(
                rng.uniform(
                    low=node.min_values[node.dim],
                    high=node.max_values[node.dim],
                    size=1,
                )[0]
            )
            node.split_val = val
            col = samples[:, node.dim]
            right_mask = col >= val
            node.children = [
                self._build(samples[~right_mask], depth=depth + 1, rng=rng),
                self._build(samples[right_mask], depth=depth + 1, rng=rng),
            ]
        else:
            splits = np.sort(
                rng.uniform(
                    low=node.min_values[node.dim],
                    high=node.max_values[node.dim],
                    size=self.branching_factor - 1,
                )
            )
            node.split_values = splits
            node.split_val = float(splits[0])
            indices = np.searchsorted(splits, samples[:, node.dim], side="right")
            node.children = [
                self._build(samples[indices == i], depth=depth + 1, rng=rng)
                for i in range(self.branching_factor)
            ]
        return node

    def unlearn_one(
        self, X: np.ndarray, rng: np.random.RandomState, selected: bool
    ) -> tuple[bool, _OnlineIsolationNode | None]:
        """Returns (structure_changed, touched_leaf)."""
        # Keep the reference's draw order, but expire the original selection.
        rng.random_sample()
        if not selected:
            return False, None
        self.data_size -= 1
        self.depth_limit = _path_length(
            num_samples=self.data_size,
            max_leaf_samples=self.max_leaf_samples,
            inv_log=self._inv_log,
        )
        self._settled_limit = min(self._settled_limit, math.ceil(self.depth_limit))
        assert self.root is not None
        return self._unlearn(self.root, X)

    def _unlearn(
        self, node: _OnlineIsolationNode, X: np.ndarray | None
    ) -> tuple[bool, _OnlineIsolationNode | None]:
        if X is not None:
            node.data_size -= 1
        if node.children is None:
            return False, node if X is not None else None
        if node.data_size < node.threshold:
            self._unbuild(node)
            return True, None
        index = node.child_index(X) if X is not None else -1
        changed = False
        touched: _OnlineIsolationNode | None = None
        for i, child in enumerate(node.children):
            c_changed, c_leaf = self._unlearn(child, X if i == index else None)
            if c_changed:
                changed = True
            if c_leaf is not None:
                touched = c_leaf
        self._refresh_bounds(node)
        return changed, touched

    def _refresh_bounds(self, node: _OnlineIsolationNode) -> None:
        assert node.children is not None
        if len(node.children) == 2:
            c0, c1 = node.children[0], node.children[1]
            np.minimum(c0.min_values, c1.min_values, out=node.min_values)
            np.maximum(c0.max_values, c1.max_values, out=node.max_values)
        else:
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
        binary = self.branching_factor == 2
        while (children := node.children) is not None:
            if binary:
                node = children[1 if X[node.dim] >= node.split_val else 0]
            else:
                assert node.split_values is not None
                idx = int(np.searchsorted(node.split_values, X[node.dim], side="right"))
                node = children[idx]
        return node.depth + _path_length(
            num_samples=node.data_size,
            max_leaf_samples=self.max_leaf_samples,
            inv_log=self._inv_log,
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
        self._inv_log = 1.0 / math.log(2 * branching_factor)
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

        # Packed 2D arrays across trees for fast vectorized binary-tree scoring.
        cap = 16
        self._tree_idx = np.arange(num_trees, dtype=np.intp)
        self._p_cap = cap
        self._p_dim = np.zeros((num_trees, cap), dtype=np.intp)
        self._p_split = np.zeros((num_trees, cap), dtype=np.float64)
        self._p_left = np.zeros((num_trees, cap), dtype=np.intp)
        self._p_right = np.zeros((num_trees, cap), dtype=np.intp)
        self._p_leaf_pl = np.zeros((num_trees, cap), dtype=np.float64)
        self._p_max_depth = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.rng_ is np.random.mtrand._rand:
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng_ = check_random_state(self.rng_)

    def _ensure_packed_cap(self, needed: int) -> None:
        if needed <= self._p_cap:
            return
        new_cap = self._p_cap
        while new_cap < needed:
            new_cap *= 2
        n_trees = self.num_trees
        old_cap = self._p_cap
        for attr in ("_p_dim", "_p_left", "_p_right"):
            old = getattr(self, attr)
            arr = np.zeros((n_trees, new_cap), dtype=np.intp)
            arr[:, :old_cap] = old
            setattr(self, attr, arr)
        for attr in ("_p_split", "_p_leaf_pl"):
            old = getattr(self, attr)
            arr = np.zeros((n_trees, new_cap), dtype=np.float64)
            arr[:, :old_cap] = old
            setattr(self, attr, arr)
        self._p_cap = new_cap

    def _pack_tree(self, t_idx: int) -> None:
        tree = self.trees_[t_idx]
        root = tree.root
        if root is None:
            tree.max_depth = 0
            tree.node_count = 0
            self._p_dim[t_idx, 0] = 0
            self._p_split[t_idx, 0] = 0.0
            self._p_left[t_idx, 0] = 0
            self._p_right[t_idx, 0] = 0
            self._p_leaf_pl[t_idx, 0] = 0.0
            return

        # Collect nodes in BFS/DFS order to assign slots.
        nodes: list[_OnlineIsolationNode] = []
        stack = [root]
        max_d = 0
        while stack:
            u = stack.pop()
            u.slot = len(nodes)
            nodes.append(u)
            if u.depth > max_d:
                max_d = u.depth
            if u.children is not None:
                stack.append(u.children[1])
                stack.append(u.children[0])

        nc = len(nodes)
        tree.max_depth = max_d
        tree.node_count = nc
        self._ensure_packed_cap(nc)

        p_dim = self._p_dim[t_idx]
        p_split = self._p_split[t_idx]
        p_left = self._p_left[t_idx]
        p_right = self._p_right[t_idx]
        p_leaf_pl = self._p_leaf_pl[t_idx]
        mls = self.max_leaf_samples
        inv_log = self._inv_log

        for u in nodes:
            s = u.slot
            if u.children is None:
                p_dim[s] = 0
                p_split[s] = 0.0
                p_left[s] = s
                p_right[s] = s
                ns = u.data_size
                p_leaf_pl[s] = u.depth + (0.0 if ns <= mls else math.log(ns / mls) * inv_log)
            else:
                p_dim[s] = u.dim
                p_split[s] = u.split_val
                p_left[s] = u.children[0].slot
                p_right[s] = u.children[1].slot
                p_leaf_pl[s] = 0.0

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
        binary = self.branching_factor == 2
        mls = self.max_leaf_samples
        inv_log = self._inv_log
        any_repacked = False

        for i, tree in enumerate(self.trees_):
            learned, changed, leaf = tree.learn_one(x, self.rng_)
            if selected is not None:
                selected[i] = learned
            if binary and learned:
                if changed:
                    self._pack_tree(i)
                    any_repacked = True
                elif leaf is not None:
                    ns = leaf.data_size
                    self._p_leaf_pl[i, leaf.slot] = leaf.depth + (
                        0.0 if ns <= mls else math.log(ns / mls) * inv_log
                    )

        self.window_.append((x, selected))
        if len(self.window_) > self.window_size:
            old_x, old_selected = self.window_.popleft()
            for i, tree in enumerate(self.trees_):
                was_sel = bool(old_selected[i]) if old_selected is not None else True
                changed, leaf = tree.unlearn_one(
                    X=old_x,
                    rng=self.rng_,
                    selected=was_sel,
                )
                if binary and was_sel:
                    if changed:
                        self._pack_tree(i)
                        any_repacked = True
                    elif leaf is not None:
                        ns = leaf.data_size
                        self._p_leaf_pl[i, leaf.slot] = leaf.depth + (
                            0.0 if ns <= mls else math.log(ns / mls) * inv_log
                        )

        if any_repacked:
            self._p_max_depth = max(t.max_depth for t in self.trees_)

        self.normalization_ = _path_length(
            num_samples=len(self.window_) * self.subsample,
            max_leaf_samples=mls,
            inv_log=inv_log,
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
        if self.branching_factor == 2:
            x = np.asarray(X, dtype=np.float64)
            max_d = self._p_max_depth
            if max_d == 0:
                depth = float(self._p_leaf_pl[:, 0].sum()) / self.num_trees
            else:
                nodes = np.where(
                    x[self._p_dim[:, 0]] >= self._p_split[:, 0],
                    self._p_right[:, 0],
                    self._p_left[:, 0],
                )
                t_idx = self._tree_idx
                p_dim = self._p_dim
                p_split = self._p_split
                p_left = self._p_left
                p_right = self._p_right
                for _ in range(max_d - 1):
                    nodes = np.where(
                        x[p_dim[t_idx, nodes]] >= p_split[t_idx, nodes],
                        p_right[t_idx, nodes],
                        p_left[t_idx, nodes],
                    )
                depth = float(self._p_leaf_pl[t_idx, nodes].sum()) / self.num_trees
            return float(2.0 ** (-depth / (self.normalization_ + np.finfo(float).eps)))

        depth = sum(tree.predict_one(X) for tree in self.trees_) / self.num_trees
        return float(2.0 ** (-depth / (self.normalization_ + np.finfo(float).eps)))
