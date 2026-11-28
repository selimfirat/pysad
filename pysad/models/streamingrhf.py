from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel

__bibtex__ = r"""
@inproceedings{nesic2022streamrhf,
  title={StreamRHF: Tree-based unsupervised anomaly detection for data streams},
  author={Stefan Nesic and Andrian Putina and Maroua Bahri and Alexis Huet and Jose Manuel Navarro and Dario Rossi and Mauro Sozio},
  booktitle={2022 IEEE/ACS 19th International Conference on Computer Systems and Applications (AICCSA 2022)},
  pages={1--8},
  year={2022},
  publisher={IEEE}
}
"""


class StreamingRHF(BaseModel):
    """Scores each instance with a streaming random histogram forest :cite:`nesic2022streamrhf`.

    The forest is an ensemble of random histogram trees. To split a node, this class computes the
    kurtosis (fourth standardized moment) of every feature on the points that reach the node, turns
    the kurtosis values into a probability distribution with ``log(kurtosis + 1)``, and draws the
    split feature from that distribution using a fixed per-node seed. The split value is drawn
    uniformly in the observed range of that feature, again from a fixed per-node seed. Instances
    that end up in small leaves are more likely to be anomalies, so the score of an instance is the
    sum over the trees of ``log(n / leaf_size)`` where ``leaf_size`` is the number of stream points
    in the leaf the instance routes to.

    The model is parameter-free beyond the window, the tree count and the height. It builds the
    initial forest on the first ``window_size`` instances. Every following instance is routed down
    every tree; if inserting it would change the split attribute of a node, that subtree is rebuilt
    from the points it holds plus the new instance. Every ``window_size`` insertions the forest is
    rebuilt from scratch on the most recent ``window_size`` instances, so old information is
    forgotten and concept drift is handled while memory stays bounded.

    Deviations from :cite:`nesic2022streamrhf`:

    - The paper inserts an instance and then scores it with the instance included in its leaf.
      PySAD scores an instance against the forest built from the earlier instances only (the
      current instance is not yet recorded), which is the prequential convention of :class:`BaseModel`.
      The paper scores the first ``window_size`` instances after building the forest on them; PySAD
      returns ``0.0`` for those instances because no forest exists yet.
    - The reference implementation recomputes the kurtosis of a node incrementally as the new
      instance passes through it. This class recomputes the kurtosis of a node from the points it
      currently holds, which gives the same split decision while keeping the code simpler.

    Higher scores mean more anomalous instances. The output is a continuous score; pipe the model
    into :class:`~pysad.transform.postprocessing.ThresholdFilter` or
    :class:`~pysad.transform.postprocessing.QuantileFilter` to obtain binary decisions.

    Args:
        window_size (int): The size of the sliding/reference window. The initial forest is built on
            the first ``window_size`` instances and the forest is rebuilt every ``window_size``
            insertions. Must be a positive integer (Default=50).
        num_trees (int): The number of random histogram trees in the forest. Must be a positive
            integer (Default=50).
        height (int): The maximum height of every tree. A tree of height ``h`` has ``2**h - 1``
            internal nodes and ``2**h`` leaves. Must be a positive integer (Default=5).
        random_state (int, np.random.RandomState or None): The seed or random number generator used
            to draw the fixed per-node split seeds. None draws from NumPy's global random state,
            which `pysad.utils.fix_seed` seeds (Default=None).
    """

    _output_kind = "score"
    _has_decision_rule = False

    def __init__(
        self,
        window_size: int = 50,
        num_trees: int = 50,
        height: int = 5,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        for name, value in (
            ("window_size", window_size),
            ("num_trees", num_trees),
            ("height", height),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        self.window_size = int(window_size)
        self.num_trees = int(num_trees)
        self.height = int(height)
        self.random_state = random_state

        self.num_splits = 2**self.height - 1
        self.num_leaves = 2**self.height
        self.buffer_size = 2 * self.window_size

        rng = check_random_state(random_state)
        # One fixed seed pair per internal node and per tree, kept for the whole stream. Both seeds
        # are drawn in (0, 1): the first selects the split attribute, the second picks the split
        # value inside the observed feature range. Drawing them once makes the model deterministic
        # for a given random_state and avoids consuming randomness during fit or score.
        self.r_seeds_ = rng.uniform(0.0, 1.0, size=(self.num_trees, self.num_splits, 2))
        zero_mask = self.r_seeds_ <= 0.0
        if bool(zero_mask.any()):
            self.r_seeds_[zero_mask] = rng.uniform(0.0, 1.0, size=int(zero_mask.sum()))

        # The per-tree structure does not depend on the feature dimension, so it is allocated here.
        # ``window_buffer_`` needs the dimension and is allocated on the first fit_partial.
        self.split_atts_: NDArray[np.int64] = np.full(
            (self.num_trees, self.num_splits), -1, dtype=np.int64
        )
        self.split_values_: NDArray[np.float64] = np.zeros(
            (self.num_trees, self.num_splits), dtype=np.float64
        )
        self.leaf_counts_: NDArray[np.int64] = np.zeros(
            (self.num_trees, self.num_leaves), dtype=np.int64
        )
        self.leaf_members_: NDArray[np.int64] = np.full(
            (self.num_trees, self.num_leaves, self.buffer_size), -1, dtype=np.int64
        )
        self.window_buffer_: NDArray[np.float64] | None = None

        self.point_count_ = 0
        self.n_active_ = 0
        self.forest_built_ = False
        self.num_features_ = 0

    def _leaf_index(self, node_id: int, depth: int) -> int:
        """Maps a heap-indexed node to its leaf bucket in ``leaf_members_``."""
        if node_id >= self.num_splits:
            raw = node_id
        else:
            raw = (2 ** (self.height - depth)) * (node_id + 1) - 1
        return int(raw - self.num_splits)

    def _require_window_buffer(self) -> NDArray[np.float64]:
        """Return the allocated window buffer or raise if called before initialization."""
        if self.window_buffer_ is None:
            raise RuntimeError("StreamingRHF window buffer is not initialized.")
        return self.window_buffer_

    def _allocate_window_buffer(self, num_features: int) -> None:
        """Allocates the rolling buffer of point values once the feature dimension is known."""
        self.num_features_ = int(num_features)
        self.window_buffer_ = np.empty((self.buffer_size, self.num_features_), dtype=np.float64)

    def _feature_log_kurtosis(self, vals: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns ``log(kurtosis + 1)`` of every feature of `vals` (n, d).

        A constant feature has kurtosis zero and contributes ``log(1) = 0``, so it is never drawn
        as the split feature when the total mass is positive.
        """
        n = vals.shape[0]
        mean = vals.mean(axis=0)
        centered = vals - mean
        m2 = (centered * centered).sum(axis=0)
        m4 = (centered * centered * centered * centered).sum(axis=0)
        kurtosis = np.zeros(self.num_features_, dtype=np.float64)
        nonzero = m2 > 0.0
        kurtosis[nonzero] = (n * m4[nonzero]) / (m2[nonzero] * m2[nonzero])
        return np.log(kurtosis + 1.0)

    def _select_split(
        self,
        tree: int,
        node_id: int,
        vals: NDArray[np.float64],
        log_kurt: NDArray[np.float64],
        ks: float,
    ) -> tuple[int, float]:
        """Picks the split feature and value of a node using its fixed per-node seeds."""
        num_features = self.num_features_
        seed0, seed1 = self.r_seeds_[tree, node_id]
        r_a = seed0 * ks
        feature = 0
        cumul = 0.0
        for f in range(num_features):
            cumul += log_kurt[f]
            if r_a <= cumul:
                feature = f
                break
        col = vals[:, feature]
        a_min = float(col.min())
        a_max = float(col.max())
        if a_max <= a_min:
            # A constant feature cannot be split on; the caller turns the node into a leaf.
            return feature, a_min
        split_value = a_min + seed1 * (a_max - a_min)
        return int(feature), float(split_value)

    def _fill_leaf(
        self,
        tree: int,
        node_id: int,
        depth: int,
        slots: NDArray[np.int64],
    ) -> None:
        """Records `slots` as the members of the leaf that `node_id` represents at `depth`."""
        leaf_index = self._leaf_index(node_id, depth)
        count = int(slots.shape[0])
        self.leaf_counts_[tree, leaf_index] = count
        self.leaf_members_[tree, leaf_index, :count] = slots

    def _collect_subtree_slots(self, tree: int, node_id: int, depth: int) -> NDArray[np.int64]:
        """Returns the buffer slots of every point held in the subtree rooted at `node_id`."""
        split_atts = self.split_atts_[tree]
        if node_id >= self.num_splits or split_atts[node_id] == -1:
            leaf_index = self._leaf_index(node_id, depth)
            count = int(self.leaf_counts_[tree, leaf_index])
            return self.leaf_members_[tree, leaf_index, :count].copy()
        left = self._collect_subtree_slots(tree, 2 * node_id + 1, depth + 1)
        right = self._collect_subtree_slots(tree, 2 * node_id + 2, depth + 1)
        return np.concatenate([left, right])

    def _rht(self, tree: int, node_id: int, depth: int, slots: NDArray[np.int64]) -> None:
        """Builds (or rebuilds) the subtree rooted at `node_id` from the points in `slots`.

        This is the streaming counterpart of the batch RHT build. It sets the split attribute and
        value of `node_id`, partitions `slots` between the two children, and recurses. A node
        becomes a leaf when it holds a single point, reaches the maximum height, or every feature
        has zero kurtosis.
        """
        vals = self._require_window_buffer()[slots]
        npts = int(vals.shape[0])
        split_atts = self.split_atts_[tree]
        split_values = self.split_values_[tree]
        internal = node_id < self.num_splits

        if npts == 1 or depth == self.height:
            if internal:
                split_atts[node_id] = -1
            self._fill_leaf(tree, node_id, depth, slots)
            return

        log_kurt = self._feature_log_kurtosis(vals)
        ks = float(log_kurt.sum())
        if ks <= 0.0:
            if internal:
                split_atts[node_id] = -1
            self._fill_leaf(tree, node_id, depth, slots)
            return

        feature, split_value = self._select_split(tree, node_id, vals, log_kurt, ks)
        col = vals[:, feature]
        a_min = float(col.min())
        a_max = float(col.max())
        if not a_min < split_value < a_max:
            # Degenerate split (constant feature): keep the node a leaf.
            if internal:
                split_atts[node_id] = -1
            self._fill_leaf(tree, node_id, depth, slots)
            return

        split_atts[node_id] = feature
        split_values[node_id] = split_value
        left_mask = col <= split_value
        left_slots = slots[left_mask]
        right_slots = slots[~left_mask]
        if left_slots.size == 0 or right_slots.size == 0:
            split_atts[node_id] = -1
            self._fill_leaf(tree, node_id, depth, slots)
            return
        self._rht(tree, 2 * node_id + 1, depth + 1, left_slots)
        self._rht(tree, 2 * node_id + 2, depth + 1, right_slots)

    def _build_forest(self) -> None:
        """Builds the initial forest on the first ``window_size`` stream points."""
        initial_slots = np.arange(self.window_size, dtype=np.int64)
        for tree in range(self.num_trees):
            self._rht(tree, 0, 0, initial_slots)
        self.n_active_ = self.window_size
        self.forest_built_ = True

    def _reset_forest(self) -> None:
        """Rebuilds the whole forest from the most recent ``window_size`` points.

        This forgets the oldest information, handles concept drift and keeps memory bounded.
        """
        start = self.point_count_ - self.window_size
        slots = np.arange(start, self.point_count_, dtype=np.int64) % self.buffer_size
        for tree in range(self.num_trees):
            self.split_atts_[tree].fill(-1)
            self.leaf_counts_[tree].fill(0)
            self._rht(tree, 0, 0, slots)
        self.n_active_ = self.window_size

    def _updated_split_feature(
        self, tree: int, node_id: int, all_slots: NDArray[np.int64], fallback: int
    ) -> int:
        """Recompute a node's split feature after adding a point into `all_slots`."""
        vals = self._require_window_buffer()[all_slots]
        log_kurt = self._feature_log_kurtosis(vals)
        ks = float(log_kurt.sum())
        if ks <= 0.0:
            return fallback
        r_a = self.r_seeds_[tree, node_id, 0] * ks
        cumul = 0.0
        for f in range(self.num_features_):
            cumul += log_kurt[f]
            if r_a <= cumul:
                return f
        return fallback

    def _insert(self, tree: int, x: NDArray[np.float64], slot: int) -> None:
        """Inserts one instance into one tree, rebuilding subtrees whose split attribute changes."""
        node_id = 0
        depth = 0
        split_atts = self.split_atts_[tree]
        split_values = self.split_values_[tree]

        while node_id < self.num_splits and split_atts[node_id] != -1:
            feature = int(split_atts[node_id])
            split_value = float(split_values[node_id])
            all_slots = np.append(self._collect_subtree_slots(tree, node_id, depth), slot)
            if self._updated_split_feature(tree, node_id, all_slots, feature) != feature:
                self._rht(tree, node_id, depth, all_slots)
                return

            node_id = 2 * node_id + (1 if x[feature] <= split_value else 2)
            depth += 1

        leaf_index = self._leaf_index(node_id, depth)
        count = int(self.leaf_counts_[tree, leaf_index])
        old_slots = self.leaf_members_[tree, leaf_index, :count]
        all_slots = np.append(old_slots, slot)
        if depth < self.height:
            self._rht(tree, node_id, depth, all_slots)
        else:
            self.leaf_counts_[tree, leaf_index] = count + 1
            self.leaf_members_[tree, leaf_index, count] = slot

    def fit_partial(self, X: NDArray[Any], y: int | None = None) -> StreamingRHF:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            StreamingRHF: The model itself.
        """
        x = np.asarray(X, dtype=np.float64).reshape(-1)

        if not self.forest_built_:
            if self.window_buffer_ is None:
                self._allocate_window_buffer(x.shape[0])
            self._require_window_buffer()[self.point_count_] = x
            self.point_count_ += 1
            if self.point_count_ == self.window_size:
                self._build_forest()
            return self

        slot = self.point_count_ % self.buffer_size
        self._require_window_buffer()[slot] = x
        self.point_count_ += 1
        self.n_active_ += 1
        for tree in range(self.num_trees):
            self._insert(tree, x, slot)
        if self.n_active_ == 2 * self.window_size:
            self._reset_forest()
        return self

    def score_partial(self, X: NDArray[Any]) -> float:
        """Scores the anomalousness of the next instance without recording it.

        The instance is routed down every tree to the leaf it would fall into, and its score is the
        sum over the trees of ``log(n_active / leaf_size)`` where ``leaf_size`` is the number of
        stream points already held in that leaf. Small leaves give high scores, so isolated
        instances score higher. Before the initial forest is built the score is ``0.0``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores
                represent more anomalous instances whereas lower scores correspond to more normal
                instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        x = np.asarray(X, dtype=np.float64).reshape(-1)
        if not self.forest_built_ or self.n_active_ == 0:
            return 0.0
        n = self.n_active_
        total = 0.0
        for tree in range(self.num_trees):
            node_id = 0
            depth = 0
            split_atts = self.split_atts_[tree]
            split_values = self.split_values_[tree]
            while node_id < self.num_splits and split_atts[node_id] != -1:
                feature = int(split_atts[node_id])
                split_value = float(split_values[node_id])
                if x[feature] <= split_value:
                    node_id = 2 * node_id + 1
                else:
                    node_id = 2 * node_id + 2
                depth += 1
            leaf_index = self._leaf_index(node_id, depth)
            count = int(self.leaf_counts_[tree, leaf_index])
            if count <= 0:
                count = 1
            total += float(np.log(n / count))
        return float(total)
