from __future__ import annotations

import numpy as np
import scipy

from pysad.core.base_model import BaseModel
from pysad.models._window import _RadiusCalibrator


class CPOD(BaseModel):
    """Core point-based outlier detection (CPOD) :cite:`tran2020cpod`, with the score of :class:`~pysad.models.ExactStorm`.

    An instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius` (R). As in the paper, CPOD indexes the window with core points. A core point c keeps the window instances in four rings, E_k(c) = {p | kR/2 < d(c, p) <= (k + 1)R/2} for k = 0, 1, 2, 3 (Definition 3.1). An instance is linked to c if it is in E_0(c) or E_1(c). A new instance is linked to the nearest core point within R. If there is none, the instance becomes a new core point, so any two core points are more than R apart (Definition 3.2). The rings of a new core point come from the instances linked to the core points within 3R of it (Theorem 3.7). The scored instance q uses its nearest core point c. If d(q, c) <= R/2, every instance of E_0(c) is a neighbor (Theorem 3.3), and the other neighbors are in E_1(c) and E_2(c) (Theorem 3.4). If d(q, c) <= R, the neighbors are in the rings of c (Theorem 3.5). If d(q, c) <= 2R, the neighbors are in E_0 and E_1 of the core points within 2R of q. Otherwise q has no neighbors (Theorem 3.6). The score computes the distances only to these candidates.

    The paper reports an instance as an outlier if it has fewer than k neighbors. This method gives the score of :class:`~pysad.models.ExactStorm` instead: the fraction of window instances that are not neighbors of the scored instance. Thus the two models give the same scores. The fast inlier test (more than k instances in E_0 of the close core point), the minimal probing that stops the neighbor search at k neighbors, and the split into preceding and succeeding neighbors only decide the outlier status for a fixed k, so this method does not use k or these steps. In the paper, the window moves by a slide of S instances, each slide has its own core points, and the rings are kept per slide. Here each instance gets its score when it arrives, so the slide holds one instance. Thus the rings hold the whole window, and the core points of all slides in the window form one supporting core point set, because a new core point is more than R from all existing core points (Algorithm 2). The theorems hold for any set of instances, so they hold for the whole window. A core point is removed when no window instance is linked to it. The paper finds the core points with an M-tree and keeps the rings as lists. This method computes the distances to all core points instead, and it keeps the ring of each window instance for each core point in one array, so the rings take one byte per core point and window instance. One instance can be linked to several core points, so the number of core points can exceed `window_size`. In the 2R case, the authors' code (tranvanluan2/cpod) also skips E_0 of a core point that is more than 3R/2 from q. This method does not use that pruning.

    Until the window is full, the fraction uses the instances that the window holds so far. An instance scored against an empty window gets the maximum score of 1. If ``max_radius`` is None, the radius comes from the first ``calibration_size`` instances, as in :class:`~pysad.models.ExactStorm`, and the core points are built again each time the radius changes, as the paper does when R changes online.

    Args:
        window_size (int): The number of instances in the window to score (Default=10000).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
    """

    def __init__(
        self,
        window_size: int = 10000,
        max_radius: float | None = None,
        calibration_size: int = 256,
        outlier_fraction: float = 0.05,
    ) -> None:
        self.window_size = window_size
        self.max_radius = max_radius
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius_: float | None = max_radius
        self._calibrator = _RadiusCalibrator(max_radius, calibration_size, outlier_fraction)
        self.points_ = np.empty((0, 0))  # the window as a ring buffer, (window_size, num_features)
        self.cores_: np.ndarray = np.empty((0, 0))  # the core points, (capacity, num_features)
        # The ring of each window slot for each core point, 4 if none, (capacity, window_size).
        self.rings_: np.ndarray = np.empty((0, window_size), dtype=np.int8)
        self.half_: np.ndarray = np.zeros(0, dtype=np.int64)  # the size of E_0 of each core point
        # The number of instances linked to each core point, the size of E_0 and E_1. A row with 0 is free.
        self.linked_: np.ndarray = np.zeros(0, dtype=np.int64)
        self._num = 0  # the number of instances in the window
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> CPOD:
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
            self.cores_ = np.empty((0, len(X)))

        i = self._next
        if self._num == self.window_size:
            self._remove(i)
        else:
            self._num += 1
        self.points_[i] = X
        self._next = (i + 1) % self.window_size

        if self._calibrator.active:
            self.radius_ = self._calibrator.update(X)
            if self.radius_ is not None:
                self._build()
        elif self.radius_ is not None:
            self._add(i)

        return self

    def _ring(self, dists: np.ndarray) -> np.ndarray:
        """The ring k of each distance, with k = 4 for distances above 2R."""
        assert self.radius_ is not None
        return np.searchsorted(np.array([0.5, 1.0, 1.5, 2.0]) * self.radius_, dists).astype(np.int8)

    def _build(self) -> None:
        """Builds the core points of the window with the current radius."""
        self.rings_[:] = 4
        self.half_[:] = 0
        self.linked_[:] = 0
        for i in range(self._num):
            self._add(i)

    def _add(self, i: int) -> None:
        """Algorithm 2 of the paper for the instance in slot i."""
        assert self.radius_ is not None
        active = np.flatnonzero(self.linked_)
        dists = scipy.spatial.distance.cdist(self.cores_[active], self.points_[i : i + 1]).ravel()
        rings = self._ring(dists)
        self.rings_[active, i] = rings
        self.half_[active[rings == 0]] += 1
        self.linked_[active[rings <= 1]] += 1
        if (rings <= 1).any():
            return

        # A new core point at the instance in a free row.
        free = np.flatnonzero(self.linked_ == 0)
        if len(free) == 0:
            num = max(len(self.linked_), 1)  # doubles the capacity
            free = np.arange(len(self.linked_), len(self.linked_) + num)
            self.cores_ = np.concatenate([self.cores_, np.empty((num, self.cores_.shape[1]))])
            self.rings_ = np.concatenate(
                [self.rings_, np.full((num, self.window_size), 4, dtype=np.int8)]
            )
            self.half_ = np.concatenate([self.half_, np.zeros(num, dtype=np.int64)])
            self.linked_ = np.concatenate([self.linked_, np.zeros(num, dtype=np.int64)])
        c = free[0]
        self.cores_[c] = self.points_[i]

        # Its rings are within the instances linked to the core points within 3R of it (Theorem 3.7).
        linked = np.flatnonzero((self.rings_[active[dists <= 3 * self.radius_]] <= 1).any(axis=0))
        rings = self._ring(
            scipy.spatial.distance.cdist(self.points_[linked], self.cores_[c : c + 1]).ravel()
        )
        self.rings_[c] = 4
        self.rings_[c, linked] = rings
        self.rings_[c, i] = 0
        self.half_[c] = (rings == 0).sum() + 1
        self.linked_[c] = (rings <= 1).sum() + 1

    def _remove(self, i: int) -> None:
        """Removes the expired instance in slot i. A core point with no linked window instance is removed."""
        rings = self.rings_[:, i]  # a view
        self.half_[rings == 0] -= 1
        self.linked_[rings <= 1] -= 1
        rings[:] = 4

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = np.asarray(X, dtype=np.float64)
        expiring = self._num == self.window_size
        num_window = self._num - expiring
        if num_window == 0 or self.radius_ is None:
            return 1.0

        active = np.flatnonzero(self.linked_)
        dists = scipy.spatial.distance.cdist(self.cores_[active], [X]).ravel()
        nearest = dists.argmin()
        c = active[nearest]
        num_neighbors = 0
        if dists[nearest] <= self.radius_ / 2:  # Theorems 3.3 and 3.4
            num_neighbors = self.half_[c] - (expiring and self.rings_[c, self._next] == 0)
            candidates = np.flatnonzero((self.rings_[c] == 1) | (self.rings_[c] == 2))
        elif dists[nearest] <= self.radius_:  # Theorem 3.5
            candidates = np.flatnonzero(self.rings_[c] <= 3)
        else:  # Theorem 3.6, with no candidates beyond 2R
            candidates = np.flatnonzero(
                (self.rings_[active[dists <= 2 * self.radius_]] <= 1).any(axis=0)
            )
        if expiring:  # The oldest instance expires when X arrives.
            candidates = candidates[candidates != self._next]
        num_neighbors += (
            scipy.spatial.distance.cdist(self.points_[candidates], [X]) <= self.radius_
        ).sum()

        return float(1.0 - num_neighbors / num_window)
