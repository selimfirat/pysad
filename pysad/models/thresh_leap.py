from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel


class ThreshLEAP(BaseModel):
    """Distance-threshold outlier detection with the lifespan-aware probing operation (Thresh_LEAP) :cite:`cao2014scalable`.

    An instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius` (R). By Definition 1 of the paper, an instance is an outlier if it has fewer than k neighbors, where k is ``num_neighbors``. Thus k neighbors are the minimal evidence that proves an inlier (MESI). The LEAP operation of the paper (Algorithm 2) follows two principles. Minimal probing tests the window instances one by one and stops as soon as it finds k neighbors, instead of a full range query. Lifespan-aware prioritization tests the instances that expire last first: the succeeding instances, and then the preceding slides from the newest to the oldest.

    Each instance gets its score when it arrives, so the slide holds one instance, and no succeeding instance exists yet. The probe tests the window from the newest to the oldest instance and stops when it finds k neighbors. The score is 1 - min(n, k) / k, where n is the number of neighbors that the probe finds. Thus the score is above 0 exactly when the arriving instance is an outlier of the paper, and an outlier with fewer neighbors gets a higher score. Every inlier gets the score 0, because the probe stops at k neighbors and does not count the others. The probe tests the instances in blocks, with k instances in the first block and twice as many in each next block, so it tests at most about twice as many instances as the probe of the paper. The paper keeps the neighbor count of each probed slide, a triggered list for each slide, and probes an instance again when its evidence expires. These steps only update the outlier status of instances that already got their score, so this method does not use them.

    Until the window holds k instances, every instance is an outlier, as in Definition 1. An instance scored against an empty window gets the maximum score of 1. The paper sets the radius for each query. If ``max_radius`` is None, the radius comes from the first ``calibration_size`` instances, as in :class:`~pysad.models.ExactStorm`.

    Args:
        window_size (int): The number of instances in the window to score (Default=10000).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        num_neighbors (int): The number of neighbors k that proves an inlier (Default=50).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
    """

    def __init__(
        self,
        window_size: int = 10000,
        max_radius: float | None = None,
        num_neighbors: int = 50,
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
        self._num = 0  # the number of instances in the window
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ThreshLEAP:
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

        self.points_[self._next] = X
        self._next = (self._next + 1) % self.window_size
        self._num = min(self._num + 1, self.window_size)

        if self.max_radius is None and len(self._calibration) < self.calibration_size:
            self._calibrate(X)

        return self

    def _calibrate(self, X: np.ndarray) -> None:
        dists = np.linalg.norm(np.array(self._calibration) - X, axis=1) if self._calibration else []
        self._nearest = np.append(np.minimum(self._nearest, dists), np.min(dists, initial=np.inf))
        self._calibration.append(X)
        if len(self._calibration) >= 2:
            self.radius_ = float(np.quantile(self._nearest, 1 - self.outlier_fraction))

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.radius_ is None:
            return 1.0

        X = np.asarray(X, dtype=np.float64)
        k = self.num_neighbors
        num_window = self._num - (self._num == self.window_size)  # The oldest one expires.
        found = 0
        start, size = 0, k
        while found < k and start < num_window:
            stop = min(start + size, num_window)
            slots = (self._next - 1 - np.arange(start, stop)) % self.window_size  # newest first
            found += int((np.linalg.norm(self.points_[slots] - X, axis=1) <= self.radius_).sum())
            start, size = stop, 2 * size

        return 1.0 - min(found, k) / k
