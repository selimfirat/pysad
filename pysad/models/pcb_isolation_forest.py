from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class PCBIsolationForest(BaseModel):
    r"""Score instances with PCB-iForest :cite:`heigl2021improvement`.

    The first full window builds the forest. Each tree gets a performance
    counter. A counter gains one when the tree agrees with the ensemble at
    threshold 0.5. Otherwise, it loses one. NDKSWIN tests the latest window
    for drift. After drift, the forest replaces only trees with negative
    counters and resets all counters. Other trees keep their splits.

    This model follows Algorithm 1 and Section 4.1 of the paper. The ensemble
    score is the mean of the tree scores. Higher scores indicate more anomalous
    instances. Each tree uses ``2 ** (-path_length / c(window_size))``.
    The path includes the standard isolation forest correction for leaf size.
    For ``n > 2``, ``c(n) = 2 * (log(n - 1) + gamma) - 2 * (n - 1) / n``.
    For one or two points, ``c(1) = 0`` and ``c(2) = 1``.

    Base learner:
        Section 4.3 permits different base learners. This model uses classic
        axis-parallel isolation trees through scikit-learn's BSD-licensed
        ``IsolationForest``. Each tree uses the full window without bootstrap
        sampling and stops at depth ``ceil(log2(window_size))``. This model
        exposes neither the EIF splits nor the IBFS feature scores from the
        paper's two experimental variants. Scikit-learn converts training
        values to float32. Scoring uses the same conversion.

    Paper and reference choices:
        Algorithm 1 uses threshold 0.5 for counter updates. This model follows
        that rule. Section 4.2 also mentions comparisons with the ensemble
        score. A tree score of exactly 0.5 loses one counter unit.

        NDKSWIN :cite:`togbe2021anomalies` compares the latest ``stat_size``
        rows with equally many older rows drawn without replacement.
        Each feature uses its own sample. Drift occurs if any feature's KS
        distance exceeds ``sqrt(-log(alpha) / stat_size)``, as Section 4.1
        specifies. The window keeps all recent rows after drift for tree fitting.

        The `MIT-licensed NDKSWIN reference from Elmecio
        <https://github.com/Elmecio/IForestASD_based_methods_in_scikit_Multiflow/blob/main/source/ndkswin.py>`_
        uses sampled dimensions, sampled arrivals, sampling with replacement,
        and a p-value test.
        It also truncates its own window after drift. This model follows
        Section 4.1 instead. Thus, drift times can differ from that reference.

    PySAD behavior:
        ``fit_score_partial`` scores before learning. ``score_partial`` never
        updates counters or random state. ``fit_partial`` updates counters
        before testing for drift. Scores are 0.0 until the first window fills.
        The model copies fitted inputs. Memory and per-instance work stay
        bounded as the stream grows. Tree rebuilding costs depend on the
        fixed window size and tree count.

    Args:
        num_trees (int): Number of trees (Default=100).
        window_size (int): Number of recent instances. This must be at least
            ``2 * stat_size`` (Default=200).
        stat_size (int): Number of recent rows in each KS comparison
            (Default=30).
        alpha (float): KS significance level in (0, 1) (Default=0.01).
        random_state (int, np.random.RandomState, or None): Random seed or
            generator. None uses NumPy's global state (Default=None).

    Attributes:
        performance_counters_ (np.int64 array of shape (num_trees,)): Tree
            agreement counts since initialization or the latest drift.
    """

    def __init__(
        self,
        num_trees: int = 100,
        window_size: int = 200,
        stat_size: int = 30,
        alpha: float = 0.01,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if num_trees < 1:
            raise ValueError("num_trees must be positive.")
        if window_size < 2 * stat_size:
            raise ValueError("window_size must be at least 2 * stat_size.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.num_trees = num_trees
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.performance_counters_ = np.zeros(num_trees, dtype=np.int64)
        self.window_: np.ndarray | None = None
        self._count = 0
        self._threshold = math.sqrt(-math.log(alpha) / stat_size)
        self._depth_limit = math.ceil(math.log2(window_size))
        # A binary tree with w leaves has at most 2w - 1 nodes.
        shape = (num_trees, 2 * window_size - 1)
        self._features_ = np.zeros(shape, dtype=np.intp)
        self._left_ = np.zeros(shape, dtype=np.intp)
        self._right_ = np.zeros(shape, dtype=np.intp)
        self._splits_ = np.zeros(shape)
        self._paths_ = np.zeros(shape)
        self._indices = np.arange(num_trees)
        counts = np.arange(window_size + 1, dtype=float)
        self._correction = np.zeros(window_size + 1)
        self._correction[2] = 1.0
        self._correction[3:] = (
            2 * (np.log(counts[3:] - 1) + np.euler_gamma) - 2 * (counts[3:] - 1) / counts[3:]
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.rng_ is np.random.mtrand._rand:
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng_ = check_random_state(self.rng_)

    def _build(self, indices: np.ndarray) -> None:
        if not len(indices):
            return
        forest = IsolationForest(
            n_estimators=len(indices),
            max_samples=self.window_size,
            random_state=self.rng_,
        ).fit(self.window_)
        for i, estimator in zip(indices, forest.estimators_, strict=True):
            tree = estimator.tree_
            size = tree.node_count
            self._features_[i, :size] = np.maximum(tree.feature, 0)
            self._left_[i, :size] = tree.children_left
            self._right_[i, :size] = tree.children_right
            self._splits_[i, :size] = tree.threshold
            self._paths_[i, :size] = (
                tree.compute_node_depths() - 1 + self._correction[tree.n_node_samples]
            )

    def _drift(self) -> bool:
        assert self.window_ is not None
        recent = np.sort(self.window_[-self.stat_size :], axis=0)
        for j in range(self.window_.shape[1]):
            rows = self.rng_.choice(
                self.window_size - self.stat_size, size=self.stat_size, replace=False
            )
            older = np.sort(self.window_[rows, j])
            points = np.concatenate((older, recent[:, j]))
            gaps = older.searchsorted(points, side="right") - recent[:, j].searchsorted(
                points, side="right"
            )
            if np.abs(gaps).max() / self.stat_size > self._threshold:
                return True
        return False

    def _scores(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float32)
        nodes: np.ndarray = np.zeros(self.num_trees, dtype=np.intp)
        # Track one node per tree. Leaves keep their index.
        for _ in range(self._depth_limit):
            left = self._left_[self._indices, nodes]
            branch = np.where(
                x[self._features_[self._indices, nodes]] <= self._splits_[self._indices, nodes],
                left,
                self._right_[self._indices, nodes],
            )
            nodes = np.where(left == -1, nodes, branch)
        return np.exp2(-self._paths_[self._indices, nodes] / self._correction[self.window_size])

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> PCBIsolationForest:
        """Learn one instance and replace weak trees after drift.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to learn.
            y (int or None): Ignored (Default=None).

        Returns:
            PCBIsolationForest: self.
        """
        if self.window_ is None:
            self.window_ = np.empty((self.window_size, len(X)))
        if self._count < self.window_size:
            self.window_[self._count] = X
            self._count += 1
            if self._count == self.window_size:
                self._build(self._indices)
            return self
        scores = self._scores(X)
        agree = scores > 0.5 if scores.mean() > 0.5 else scores < 0.5
        self.performance_counters_ += np.where(agree, 1, -1)
        self.window_[:-1] = self.window_[1:]
        self.window_[-1] = X
        if self._drift():
            self._build(np.flatnonzero(self.performance_counters_ < 0))
            self.performance_counters_.fill(0)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Score an instance without changing the forest.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to score.

        Returns:
            float: Mean tree score in [0, 1], or 0.0 before initialization.
        """
        if self._count < self.window_size:
            return 0.0
        return float(self._scores(X).mean())
