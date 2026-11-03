from __future__ import annotations

import numpy as np
import scipy

from pysad.core.base_model import BaseModel

# Labels in cluster_: an empty slot and an instance in no micro-cluster (the set PD of the paper).
_EMPTY = -2
_PD = -1


class MCOD(BaseModel):
    """Micro-cluster-based Continuous Outlier Detection (MCOD) :cite:`kontaki2011continuous`, with the score of :class:`~pysad.models.ExactStorm`.

    An instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius` (R). MCOD groups the window into micro-clusters. A micro-cluster has a fixed center and at least k + 1 instances within R/2 of the center, where k is ``num_neighbors``. All other instances form the set PD of the paper. As in the paper, a new instance joins the micro-cluster with the closest center within R/2. Otherwise, it forms a new micro-cluster with itself as the center if more than k instances of PD are within R/2 of it (theta = 1, as in the experiments of the paper). If an expired instance leaves a micro-cluster with fewer than k + 1 instances, the micro-cluster is dissolved, and each of its instances is processed again as a new instance.

    The paper reports an instance as an outlier if it has fewer than k neighbors. This method gives the score of :class:`~pysad.models.ExactStorm` instead: the fraction of window instances that are not neighbors of the scored instance. Thus the two models give the same scores, and k changes only the speed. The score counts every instance of a micro-cluster whose center is within R/2 of the scored instance, because each of them is within R by the triangle inequality. It computes the distances only to PD and to the instances of micro-clusters whose centers are within 3R/2, as the range queries of the paper do (Lemma 4). This method does not keep the neighbor lists and the event queue of the paper, because they only report the outliers for a fixed k.

    Until the window is full, the fraction uses the instances that the window holds so far. An instance scored against an empty window gets the maximum score of 1. If ``max_radius`` is None, the radius comes from the first ``calibration_size`` instances, as in :class:`~pysad.models.ExactStorm`. Until the radius is fixed, all instances stay in PD.

    Args:
        window_size (int): The number of instances in the window to score (Default=10000).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        num_neighbors (int): The number of neighbors k. A micro-cluster has at least k + 1 instances (Default=10, as in the paper).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
    """

    def __init__(
        self,
        window_size: int = 10000,
        max_radius: float | None = None,
        num_neighbors: int = 10,
        calibration_size: int = 256,
        outlier_fraction: float = 0.05,
    ) -> None:
        self.window_size = window_size
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius_: float | None = max_radius
        self._calibration: list[np.ndarray] = []  # the first instances, while they set the radius
        self._nearest = np.empty(0)  # the distance from each of them to its nearest other one
        self.points_ = np.empty((0, 0))  # the window as a ring buffer, (window_size, num_features)
        self.cluster_ = np.full(window_size, _EMPTY)  # the micro-cluster of each slot, or a label
        self.centers_ = np.empty((0, 0))  # (max micro-clusters, num_features)
        self.sizes_ = np.zeros(window_size // (num_neighbors + 1) + 1, dtype=np.int64)  # 0 is free
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MCOD:
        """Updates the model with the next instance. The oldest instance of a full window expires.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        X = np.array(X, dtype=np.float64)
        if self.points_.size == 0:
            self.points_ = np.empty((self.window_size, len(X)))
            self.centers_ = np.empty((len(self.sizes_), len(X)))

        i = self._next
        self._expire(i)
        self.points_[i] = X
        self.cluster_[i] = _PD
        self._next = (i + 1) % self.window_size

        if self.max_radius is None and len(self._calibration) < self.calibration_size:
            self._calibrate(X)
            if len(self._calibration) == self.calibration_size and self.radius_ is not None:
                # The radius is fixed from now on, so the micro-clusters can form.
                for j in np.flatnonzero(self.cluster_ == _PD).tolist():
                    if self.cluster_[j] == _PD:
                        self._insert(j)
        elif self.radius_ is not None:
            self._insert(i)

        return self

    def _calibrate(self, X: np.ndarray) -> None:
        dists = np.linalg.norm(np.array(self._calibration) - X, axis=1) if self._calibration else []
        self._nearest = np.append(np.minimum(self._nearest, dists), np.min(dists, initial=np.inf))
        self._calibration.append(X)
        if len(self._calibration) >= 2:
            self.radius_ = float(np.quantile(self._nearest, 1 - self.outlier_fraction))

    def _expire(self, i: int) -> None:
        """Step 1 and Step 4 of the paper: removes slot i and dissolves a micro-cluster that is too small."""
        c = self.cluster_[i]
        self.cluster_[i] = _EMPTY
        if c < 0:
            return

        self.sizes_[c] -= 1
        if self.sizes_[c] > self.num_neighbors:
            return

        self.sizes_[c] = 0
        members = np.flatnonzero(self.cluster_ == c)
        self.cluster_[members] = _PD
        for j in members.tolist():
            if self.cluster_[j] == _PD:
                self._insert(j)

    def _insert(self, i: int) -> None:
        """Steps 2 and 3 of the paper for the instance in slot i, which is in PD."""
        assert self.radius_ is not None
        active = np.flatnonzero(self.sizes_)
        if len(active) > 0:
            dists = np.linalg.norm(self.centers_[active] - self.points_[i], axis=1)
            closest = dists.argmin()
            if dists[closest] <= self.radius_ / 2:
                self.cluster_[i] = active[closest]
                self.sizes_[active[closest]] += 1
                return

        self._form(i)

    def _form(self, i: int) -> None:
        """Step 3b of the paper: a new micro-cluster centered at slot i if more than k instances of PD are within R/2."""
        assert self.radius_ is not None
        pd = np.flatnonzero(self.cluster_ == _PD)
        pd = pd[pd != i]
        close = pd[np.linalg.norm(self.points_[pd] - self.points_[i], axis=1) <= self.radius_ / 2]
        if len(close) <= self.num_neighbors:
            return

        c = int(np.flatnonzero(self.sizes_ == 0)[0])
        self.centers_[c] = self.points_[i]
        self.cluster_[close] = c
        self.cluster_[i] = c
        self.sizes_[c] = len(close) + 1

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = np.asarray(X, dtype=np.float64)
        cluster = self.cluster_.copy()
        cluster[self._next] = _EMPTY  # It expires when X arrives.
        num_window = int((cluster != _EMPTY).sum())
        if num_window == 0 or self.radius_ is None:
            return 1.0

        # Whether to count every instance (inside) or to compute the distances (near), by label + 2.
        inside = np.zeros(len(self.sizes_) + 2, dtype=bool)
        near = np.zeros(len(self.sizes_) + 2, dtype=bool)
        near[_PD + 2] = True
        active = np.flatnonzero(self.sizes_)
        if len(active) > 0:
            dists = np.linalg.norm(self.centers_[active] - X, axis=1)
            inside[active + 2] = dists <= self.radius_ / 2
            near[active + 2] = (dists > self.radius_ / 2) & (dists <= 1.5 * self.radius_)

        labels = cluster + 2
        candidates = self.points_[near[labels]]  # (num_candidates, num_features)
        num_neighbors = (
            inside[labels].sum()
            + (scipy.spatial.distance.cdist(candidates, [X]) <= self.radius_).sum()
        )

        return float(1.0 - num_neighbors / num_window)
