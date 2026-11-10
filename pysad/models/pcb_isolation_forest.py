from __future__ import annotations

import math

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.models.online_isolation_forest import _OnlineIsolationTree, _path_length

ANOMALY_THRESHOLD = 0.5


class PCBIsolationForest(BaseModel):
    r"""Scores instances with PCB-iForest :cite:`heigl2021improvement`.

    PCB-iForest (Performance Counter-Based isolation forest) keeps a sliding
    window of the latest window_size instances. Once the window is full, it
    builds an isolation forest from it. Each arriving instance is scored by the
    forest. Each tree gets a performance counter (PC). The PC grows by 1 when
    the tree agrees with the ensemble about whether the instance is anomalous,
    and shrinks by 1 otherwise. Agreement is measured against the anomaly
    threshold 0.5, as in Algorithm 1 of the paper.

    An NDKSWIN drift test :cite:`togbe2021anomalies` runs on the raw window
    values: one Kolmogorov-Smirnov test per feature, as in KSWIN
    :cite:`raab2020reactive`. A drift fires when any feature exceeds the test
    threshold. When a drift fires, the trees with negative PCs are replaced by
    new trees built from the current window, and all PCs are reset to zero.

    The trees are the trees of :class:`~pysad.models.OnlineIsolationForest`,
    rebuilt from the whole window at initialization and after each drift. The
    score of an instance is ``2 ** (-mean_path_length / c)``. A tree scores an
    instance as ``2 ** (-path_length / c)``. Higher scores are more anomalous.

    Deviations and choices where the paper leaves room:

    * The paper's Algorithm 1 updates each PC against the anomaly threshold 0.5.
      The paper's prose also mentions comparing each tree score with the
      ensemble score. This model follows Algorithm 1 and uses the threshold 0.5.
    * NDKSWIN draws one shared set of r older window rows for all features,
      instead of one sample per feature. After a drift the window keeps sliding,
      because the same window feeds the partial update; KSWIN alone resets its
      window to the recent part.
    * The drift test starts after the instance that fills the window. That
      instance builds the initial forest and sets all PCs to zero, as the
      paper's initialization phase does.
    * PySAD scores before fitting in ``fit_score_partial``, so the PC update of
      an instance happens in ``fit_partial``, not in ``score_partial``. Scoring
      never changes the model state.
    * An unfitted model (a window that is not full yet) returns 0.0.
    * ``window_size`` must be positive and at least ``2 * stat_size``, so that
      the older sample can be drawn without replacement.

    ``random_state`` controls one shared NumPy ``RandomState`` for the tree
    splits, the subsampling and the drift test. None uses NumPy's global state.

    Args:
        num_trees (int): Number of trees in the forest (Default=100, as in the
            paper).
        max_leaf_samples (int): Base split threshold of each tree (Default=32).
        window_size (int): Number of latest instances in the sliding window W,
            at least ``2 * stat_size`` (Default=200, the window size of the
            paper's experiments).
        stat_size (int): Number r of latest window values that the KS test
            compares with r older values drawn without replacement, per feature
            (Default=30, the NDKSWIN default used in the paper).
        alpha (float): Significance level of the KS test, in (0, 1). A feature
            detects a drift if its KS distance is above
            ``sqrt(-ln(alpha) / r)`` (Default=0.01, the NDKSWIN default used
            in the paper).
        growth_criterion (str): 'adaptive' doubles the split threshold per
            depth; 'fixed' keeps it constant (Default='adaptive').
        subsample (float): Probability of including a window point in each
            tree, in (0, 1] (Default=1.0).
        branching_factor (int): Number of children per split, at least two
            (Default=2).
        random_state (int, np.random.RandomState, or None): Seed or random
            number generator. None draws from NumPy's global random state
            (Default=None).

    Attributes:
        performance_counters_ (np.int64 array of shape (num_trees,)): The
            performance counter (PC) of each tree.
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_leaf_samples: int = 32,
        window_size: int = 200,
        stat_size: int = 30,
        alpha: float = 0.01,
        growth_criterion: str = "adaptive",
        subsample: float = 1.0,
        branching_factor: int = 2,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if num_trees < 1:
            raise ValueError("num_trees must be positive.")
        if max_leaf_samples < 1:
            raise ValueError("max_leaf_samples must be positive.")
        if stat_size < 1:
            raise ValueError("stat_size must be positive.")
        # The older sample of r rows is drawn without replacement from the rest.
        if window_size < 2 * stat_size:
            raise ValueError("window_size must be at least 2 * stat_size.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if growth_criterion not in {"adaptive", "fixed"}:
            raise ValueError("growth_criterion must be 'adaptive' or 'fixed'.")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")
        if branching_factor < 2:
            raise ValueError("branching_factor must be at least two.")

        self.num_trees = num_trees
        self.max_leaf_samples = max_leaf_samples
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.growth_criterion = growth_criterion
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.trees_: list[_OnlineIsolationTree] = []
        self.performance_counters_ = np.zeros(num_trees, dtype=np.int64)
        self.window_: np.ndarray | None = None
        self._window_count = 0
        self.normalization_ = 0.0
        self._threshold = math.sqrt(-math.log(alpha) / stat_size)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.rng_ is np.random.mtrand._rand:
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.rng_ = check_random_state(self.rng_)

    def _new_tree(self) -> _OnlineIsolationTree:
        return _OnlineIsolationTree(
            max_leaf_samples=self.max_leaf_samples,
            growth_criterion=self.growth_criterion,
            subsample=self.subsample,
            branching_factor=self.branching_factor,
        )

    def _fit_tree_on_window(self, tree: _OnlineIsolationTree) -> None:
        assert self.window_ is not None
        for row in self.window_[: self._window_count]:
            tree.learn_one(row, self.rng_)

    def _init_forest(self) -> None:
        """Builds every tree from the full window and resets the counters."""
        self.trees_ = [self._new_tree() for _ in range(self.num_trees)]
        for tree in self.trees_:
            self._fit_tree_on_window(tree)
        self.performance_counters_ = np.zeros(self.num_trees, dtype=np.int64)
        self.normalization_ = _path_length(
            num_samples=self._window_count * self.subsample,
            max_leaf_samples=self.max_leaf_samples,
            branching_factor=self.branching_factor,
        )

    def _partial_refit(self) -> None:
        """Replaces the trees with negative counters and resets all counters."""
        for i in np.flatnonzero(self.performance_counters_ < 0):
            tree = self._new_tree()
            self._fit_tree_on_window(tree)
            self.trees_[i] = tree
        self.performance_counters_[:] = 0

    def _ndkswin_drift(self) -> bool:
        """Runs one KS test per feature on the full window, as in NDKSWIN."""
        assert self.window_ is not None
        n, r = self.window_size, self.stat_size
        window = self.window_
        # R holds the r latest rows, W holds r older rows drawn without replacement.
        recent = np.sort(window[n - r :], axis=0)
        older = np.sort(window[self.rng_.choice(n - r, size=r, replace=False)], axis=0)
        for j in range(window.shape[1]):
            points = np.concatenate((older[:, j], recent[:, j]))
            gaps = older[:, j].searchsorted(points, side="right") - recent[:, j].searchsorted(
                points, side="right"
            )
            if np.abs(gaps).max() / r > self._threshold:
                return True
        return False

    def _tree_scores(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns the score of each tree and the ensemble score for one instance."""
        depths = np.fromiter(
            (tree.predict_one(x) for tree in self.trees_), dtype=np.float64, count=self.num_trees
        )
        denom = self.normalization_ + np.finfo(float).eps
        return 2.0 ** (-depths / denom), float(2.0 ** (-depths.mean() / denom))

    def _update_counters(self, scores: np.ndarray, ensemble_score: float) -> None:
        """Updates each performance counter as in Algorithm 1 of the paper.

        When the ensemble scores an instance as anomalous, each tree whose own
        score is above the threshold 0.5 is rewarded, and every other tree is
        penalized. When the ensemble scores an instance as normal, each tree
        whose own score is below 0.5 is rewarded, and every other tree is
        penalized.
        """
        if ensemble_score > ANOMALY_THRESHOLD:
            self.performance_counters_ += np.where(scores > ANOMALY_THRESHOLD, 1, -1)
        else:
            self.performance_counters_ += np.where(scores < ANOMALY_THRESHOLD, 1, -1)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> PCBIsolationForest:
        """Learns one instance, updates the counters, and tests the window for drift.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to learn.
            y (int or None): Ignored (Default=None).

        Returns:
            PCBIsolationForest: self.
        """
        x = np.array(X, dtype=np.float64, copy=True)
        if self.window_ is None:
            self.window_ = np.empty((self.window_size, x.shape[0]), dtype=np.float64)
        if self.trees_:
            scores, ensemble_score = self._tree_scores(x)
            self._update_counters(scores, ensemble_score)
        if self._window_count < self.window_size:
            self.window_[self._window_count] = x
            self._window_count += 1
        else:
            self.window_[:-1] = self.window_[1:]
            self.window_[-1] = x
        if self._window_count == self.window_size:
            if not self.trees_:
                self._init_forest()
            elif self._ndkswin_drift():
                self._partial_refit()
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores an instance without learning it.

        Args:
            X (np.float64 array of shape (num_features,)): Instance to score.

        Returns:
            float: Anomaly score in [0, 1], or 0.0 before the window is full.
        """
        if not self.trees_:
            return 0.0
        depth = sum(tree.predict_one(X) for tree in self.trees_) / self.num_trees
        return float(2.0 ** (-depth / (self.normalization_ + np.finfo(float).eps)))
