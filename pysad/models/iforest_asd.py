from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import IsolationForest

from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pysad.utils import _iterate


def _check_drift_threshold(drift_threshold: float | None) -> None:
    """Rejects a drift threshold outside the anomaly-rate range."""
    if drift_threshold is None:
        return

    if (
        isinstance(drift_threshold, bool)
        or not isinstance(drift_threshold, numbers.Real)
        or not 0.0 <= float(drift_threshold) <= 1.0
    ):
        raise ValueError(
            f"drift_threshold must be None or a real number in [0, 1], got {drift_threshold!r}."
        )


def _average_path_length(n_samples_leaf: np.ndarray) -> np.ndarray:
    """Average path length of an unsuccessful BST search, matching scikit-learn."""
    n = np.asarray(n_samples_leaf, dtype=np.float64)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    mask = n > 2
    nm = n[mask]
    out[mask] = 2.0 * (np.log(nm - 1.0) + np.euler_gamma) - 2.0 * (nm - 1.0) / nm
    return out


class _IsolationForest(IsolationForest):
    def fit(
        self,
        X: ArrayLike,
        y: Any = None,
        sample_weight: ArrayLike | None = None,
    ) -> _IsolationForest:
        super().fit(X, y=y, sample_weight=sample_weight)
        n_trees = len(self.estimators_)
        max_nodes = max(est.tree_.node_count for est in self.estimators_)
        self._max_depth = max(est.tree_.max_depth for est in self.estimators_)

        feat = np.zeros((n_trees, max_nodes), dtype=np.intp)
        left = np.zeros((n_trees, max_nodes), dtype=np.intp)
        right = np.zeros((n_trees, max_nodes), dtype=np.intp)
        thresh = np.zeros((n_trees, max_nodes), dtype=np.float64)
        leaf_depth = np.zeros((n_trees, max_nodes), dtype=np.float64)

        for i, (est, est_feats) in enumerate(
            zip(self.estimators_, self.estimators_features_, strict=True)
        ):
            tr = est.tree_
            nc = tr.node_count
            cl = tr.children_left
            cr = tr.children_right
            is_leaf = cl < 0
            f = tr.feature
            feat[i, :nc] = np.where(is_leaf, 0, est_feats[np.maximum(f, 0)])
            idx = np.arange(nc, dtype=np.intp)
            left[i, :nc] = np.where(is_leaf, idx, cl)
            right[i, :nc] = np.where(is_leaf, idx, cr)
            thresh[i, :nc] = tr.threshold
            depths = tr.compute_node_depths()
            leaf_depth[i, :nc] = depths + _average_path_length(tr.n_node_samples) - 1.0

        self._feat = feat
        self._left = left
        self._right = right
        self._thresh = thresh
        self._leaf_depth = leaf_depth
        self._root_feat = feat[:, 0].copy()
        self._root_left = left[:, 0].copy()
        self._root_right = right[:, 0].copy()
        self._root_thresh = thresh[:, 0].copy()
        self._root_depth_sum = math.fsum(leaf_depth[:, 0].tolist())
        self._tree_idx = np.arange(n_trees, dtype=np.intp)
        self._denom = n_trees * float(_average_path_length(np.array([self.max_samples_]))[0])
        return self

    def score_one(self, X: ArrayLike) -> float:
        x32 = np.asarray(X, dtype=np.float32).reshape(-1)
        if self._denom == 0.0:
            raw = 0.5
        elif self._max_depth == 0:
            raw = 2.0 ** (-self._root_depth_sum / self._denom)
        else:
            nodes = np.where(
                x32[self._root_feat] <= self._root_thresh,
                self._root_left,
                self._root_right,
            )
            tree_idx = self._tree_idx
            feat = self._feat
            thresh = self._thresh
            left = self._left
            right = self._right
            for _ in range(self._max_depth - 1):
                nodes = np.where(
                    x32[feat[tree_idx, nodes]] <= thresh[tree_idx, nodes],
                    left[tree_idx, nodes],
                    right[tree_idx, nodes],
                )
            d_sum = math.fsum(self._leaf_depth[tree_idx, nodes].tolist())
            raw = 2.0 ** (-d_sum / self._denom)
        return float(-((-raw) - self.offset_))

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        X32 = np.asarray(X, dtype=np.float32)
        if X32.ndim == 1:
            X32 = X32.reshape(1, -1)
        n_samples = len(X32)
        if n_samples == 0:
            return np.empty(0, dtype=np.float64)
        if n_samples == 1:
            return np.array([self.score_one(X32[0])], dtype=np.float64)
        if self._denom == 0.0:
            return np.full(n_samples, -((-0.5) - self.offset_), dtype=np.float64)
        if self._max_depth == 0:
            raw_val = 2.0 ** (-self._root_depth_sum / self._denom)
            return np.full(n_samples, -((-raw_val) - self.offset_), dtype=np.float64)

        nodes = np.where(
            X32[:, self._root_feat] <= self._root_thresh,
            self._root_left,
            self._root_right,
        )
        tree_row = self._tree_idx[None, :]
        row_col = np.arange(n_samples, dtype=np.intp)[:, None]
        feat = self._feat
        thresh = self._thresh
        left = self._left
        right = self._right
        for _ in range(self._max_depth - 1):
            nodes = np.where(
                X32[row_col, feat[tree_row, nodes]] <= thresh[tree_row, nodes],
                left[tree_row, nodes],
                right[tree_row, nodes],
            )
        denom = self._denom
        leaf_rows = self._leaf_depth[tree_row, nodes].tolist()
        raw = np.fromiter(
            (2.0 ** (-math.fsum(row) / denom) for row in leaf_rows),
            dtype=np.float64,
            count=n_samples,
        )
        return -((-raw) - self.offset_)

    def predict(self, X: ArrayLike) -> np.ndarray:
        # 1 for anomalies, 0 for inliers.
        return (self.decision_function(X) > 0).astype(int)


class IForestASD(ReferenceWindowModel):
    """Scores each instance with an isolation forest that it retrains on a sliding reference window (IForestASD) :cite:`ding2013anomaly`.

    The model follows Algorithm 2 of the paper. After the first forest has been fit, the current forest scores the instances of each completed window. The forest is rebuilt from that window only when the window anomaly rate is at least ``drift_threshold`` (:math:`\\text{rate} \\ge u`). Otherwise, the existing forest is kept and the window is discarded. In their empirical evaluation on real-world benchmark datasets, the authors set :math:`u` to match each dataset's expected anomaly (contamination) rate.

    With ``drift_threshold=None`` (the default), the concept drift check is skipped and the forest is retrained on every window. The score is the anomaly score of the paper minus the ``contamination`` quantile of the training scores of the current forest. The model is unsupervised and ignores ``y``.

    Args:
        initial_window_X (np.float64 array of shape (num_initial_instances, num_features)): The initial window to fit for the initial calibration period. The model fits the forest to these instances (Default=None).
        window_size (int): The size of the reference window and its sliding (Default=2048).
        drift_threshold (float or None): Algorithm 2's anomaly-rate threshold :math:`u`, in ``[0, 1]``. The forest is rebuilt only when the fraction of instances labeled as anomalies in a full window is greater than or equal to this value. In their empirical evaluation on real-world benchmark datasets, the authors set :math:`u` to match each dataset's expected anomaly (contamination) rate. ``None`` retrains on every window (Default=None).
        **kwargs: Keyword arguments passed to scikit-learn's ``IsolationForest``, for example ``n_estimators``. ``contamination`` defaults to 0.1.
    """

    def __init__(
        self,
        initial_window_X: np.ndarray | None = None,
        window_size: int = 2048,
        drift_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        _check_drift_threshold(drift_threshold)
        kwargs = {"contamination": 0.1, **kwargs}
        super().__init__(_IsolationForest, window_size, window_size, initial_window_X, **kwargs)
        self.drift_threshold = drift_threshold
        self._drift_check_active = self.initial_ref_window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> IForestASD:
        """Updates the model with the next instance. ``y`` is accepted for API consistency but always
        ignored, since the model is unsupervised.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self.drift_threshold is None:
            return super().fit_partial(X, None)

        self.cur_window_X.append(np.array(X))

        if (
            not self.initial_ref_window
            and len(self.cur_window_X) < self.window_size
            and (self.reference_window_X is None or len(self.reference_window_X) < self.window_size)
        ):
            if len(self.cur_window_X) & (len(self.cur_window_X) - 1) == 0:
                self.reference_window_X = self.cur_window_X.copy()
                self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self._complete_window()

        return self

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Applies ``fit_score_partial`` to each instance in order and returns the scores.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): Ignored since the model is unsupervised (Default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if self.drift_threshold is None:
            return super().fit_score(X, None)

        y_pred = np.empty(len(X), dtype=np.float64)
        start = 0
        for i, (xi, _) in enumerate(_iterate(X)):
            next_len = len(self.cur_window_X) + 1
            if (
                not self.initial_ref_window
                and next_len < self.window_size
                and (
                    self.reference_window_X is None
                    or len(self.reference_window_X) < self.window_size
                )
            ):
                refit = (next_len & (next_len - 1)) == 0
            else:
                refit = next_len % self.sliding_size == 0

            if refit:
                y_pred[start : i + 1] = self.score(X[start : i + 1])
                start = i + 1

            self.fit_partial(xi, None)

        if start < len(X):
            y_pred[start:] = self.score(X[start:])

        return y_pred

    def _complete_window(self) -> None:
        """Applies Algorithm 2 to a completed window."""
        window = self.cur_window_X
        if self.drift_threshold is not None and self._drift_check_active:
            labels = np.ravel(self.model.predict(np.asarray(window)))
            anomaly_rate = float(np.mean(labels == 1))
            if anomaly_rate < self.drift_threshold:
                self.cur_window_X = []
                self.cur_window_y = []
                return

        self.reference_window_X = np.asarray(self.cur_window_X)
        self.cur_window_X = []
        self.cur_window_y = []
        self._fit_model()
        self._drift_check_active = True
