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
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.max_radius = max_radius
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius_: float | None = None
        self._ring_thresholds: np.ndarray | None = None
        self._set_radius(max_radius)
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

    def _set_radius(self, radius: float | None) -> None:
        self.radius_ = radius
        self._ring_thresholds = (
            None if radius is None else np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64) * radius
        )

    @staticmethod
    def _dists_to_point(points: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty(0, dtype=np.float64)
        diff = points - x
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    def _grow_cores(self) -> np.ndarray:
        num = max(len(self.linked_), 1)
        start = len(self.linked_)
        self.cores_ = np.concatenate([self.cores_, np.empty((num, self.cores_.shape[1]))])
        self.rings_ = np.concatenate(
            [self.rings_, np.full((num, self.window_size), 4, dtype=np.int8)]
        )
        self.half_ = np.concatenate([self.half_, np.zeros(num, dtype=np.int64)])
        self.linked_ = np.concatenate([self.linked_, np.zeros(num, dtype=np.int64)])
        return np.arange(start, start + num)

    def _fit_with_dists(
        self,
        X: np.ndarray,
        active: np.ndarray | None = None,
        dists: np.ndarray | None = None,
    ) -> None:
        if self.points_.size == 0:
            self.points_ = np.empty((self.window_size, len(X)))
            self.cores_ = np.empty((0, len(X)))

        i = self._next
        if self._num == self.window_size:
            self._remove(i)
            if active is not None and dists is not None:
                keep = self.linked_[active] > 0
                active = active[keep]
                dists = dists[keep]
        else:
            self._num += 1
        self.points_[i] = X
        self._next = (i + 1) % self.window_size

        if self._calibrator.active:
            prev_radius = self.radius_
            self._set_radius(self._calibrator.update(X))
            if self.radius_ is not None:
                if self.radius_ != prev_radius:
                    self._build()
                else:
                    self._add(i, active, dists)
        elif self.radius_ is not None:
            self._add(i, active, dists)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> CPOD:
        """Updates the model with the next instance. The oldest instance of a full window expires.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        self._fit_with_dists(np.array(X, dtype=np.float64))
        return self

    def _ring(self, dists: np.ndarray) -> np.ndarray:
        """The ring k of each distance, with k = 4 for distances above 2R."""
        assert self._ring_thresholds is not None
        return np.searchsorted(self._ring_thresholds, dists).astype(np.int8)

    def _build(self) -> None:
        """Builds the core points of the window with the current radius."""
        m = self._num
        self.rings_[:, :m] = 4
        self.half_[:] = 0
        self.linked_[:] = 0
        if m == 0 or self.radius_ is None:
            return

        pts = self.points_[:m]
        pair_dists = scipy.spatial.distance.cdist(pts, pts)
        pair_rings = self._ring(pair_dists)
        core_pts: list[int] = []
        for i in range(m):
            if core_pts:
                c_count = len(core_pts)
                rings_i = pair_rings[core_pts, i]
                self.rings_[:c_count, i] = rings_i
                self.half_[:c_count] += rings_i == 0
                self.linked_[:c_count] += rings_i <= 1
                if (rings_i <= 1).any():
                    continue

            c = len(core_pts)
            if c == len(self.linked_):
                self._grow_cores()
            self.cores_[c] = pts[i]
            self.rings_[c, :m] = 4
            if c > 0:
                near3 = np.flatnonzero(pair_dists[core_pts, i] <= 3.0 * self.radius_)
                if near3.size == 1:
                    linked = np.flatnonzero(self.rings_[near3[0], :i] <= 1)
                elif near3.size > 1:
                    linked = np.flatnonzero((self.rings_[near3, :i] <= 1).any(axis=0))
                else:
                    linked = np.empty(0, dtype=np.intp)
                if linked.size:
                    r_linked = pair_rings[i, linked]
                    self.rings_[c, linked] = r_linked
                    self.half_[c] = int((r_linked == 0).sum()) + 1
                    self.linked_[c] = int((r_linked <= 1).sum()) + 1
                else:
                    self.half_[c] = 1
                    self.linked_[c] = 1
            else:
                self.half_[c] = 1
                self.linked_[c] = 1
            self.rings_[c, i] = 0
            core_pts.append(i)

    def _add(
        self,
        i: int,
        active: np.ndarray | None = None,
        dists: np.ndarray | None = None,
    ) -> None:
        """Algorithm 2 of the paper for the instance in slot i."""
        assert self.radius_ is not None
        pt = self.points_[i]
        if active is None or dists is None:
            active = np.flatnonzero(self.linked_)
            dists = self._dists_to_point(self.cores_[active], pt)
        rings = self._ring(dists)
        self.rings_[active, i] = rings
        self.half_[active[rings == 0]] += 1
        self.linked_[active[rings <= 1]] += 1
        if (rings <= 1).any():
            return

        # A new core point at the instance in a free row.
        free = np.flatnonzero(self.linked_ == 0)
        if len(free) == 0:
            free = self._grow_cores()
        c = int(free[0])
        self.cores_[c] = pt

        # Its rings are within the instances linked to the core points within 3R of it (Theorem 3.7).
        n = self._num
        sub3 = active[dists <= 3.0 * self.radius_]
        if sub3.size == 1:
            linked = np.flatnonzero(self.rings_[sub3[0], :n] <= 1)
        elif sub3.size > 1:
            linked = np.flatnonzero((self.rings_[sub3, :n] <= 1).any(axis=0))
        else:
            linked = np.empty(0, dtype=np.intp)

        self.rings_[c, :n] = 4
        if linked.size:
            rings_linked = self._ring(self._dists_to_point(self.points_[linked], pt))
            self.rings_[c, linked] = rings_linked
            self.half_[c] = int((rings_linked == 0).sum()) + 1
            self.linked_[c] = int((rings_linked <= 1).sum()) + 1
        else:
            self.half_[c] = 1
            self.linked_[c] = 1
        self.rings_[c, i] = 0

    def _remove(self, i: int) -> None:
        """Removes the expired instance in slot i. A core point with no linked window instance is removed."""
        rings = self.rings_[:, i]  # a view
        self.half_[rings == 0] -= 1
        self.linked_[rings <= 1] -= 1
        rings[:] = 4

    def _score_with_dists(
        self, X: np.ndarray
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        expiring = self._num == self.window_size
        num_window = self._num - expiring
        if num_window == 0 or self.radius_ is None:
            return 1.0, None, None

        active = np.flatnonzero(self.linked_)
        dists = self._dists_to_point(self.cores_[active], X)
        nearest = int(dists.argmin())
        min_dist = float(dists[nearest])
        if min_dist > 2.0 * self.radius_:  # Theorem 3.6: no candidates beyond 2R
            return 1.0, active, dists

        c = int(active[nearest])
        n = self._num
        num_neighbors = 0
        if min_dist <= 0.5 * self.radius_:  # Theorems 3.3 and 3.4
            num_neighbors = int(self.half_[c]) - int(expiring and self.rings_[c, self._next] == 0)
            row = self.rings_[c, :n]
            candidates = np.flatnonzero((row == 1) | (row == 2))
        elif min_dist <= self.radius_:  # Theorem 3.5
            candidates = np.flatnonzero(self.rings_[c, :n] <= 3)
        else:  # Theorem 3.6, within 2R
            sub2 = active[dists <= 2.0 * self.radius_]
            if sub2.size == 1:
                candidates = np.flatnonzero(self.rings_[sub2[0], :n] <= 1)
            else:
                candidates = np.flatnonzero((self.rings_[sub2, :n] <= 1).any(axis=0))

        if expiring:  # The oldest instance expires when X arrives.
            candidates = candidates[candidates != self._next]
        if candidates.size:
            num_neighbors += int(
                (self._dists_to_point(self.points_[candidates], X) <= self.radius_).sum()
            )

        return float(1.0 - num_neighbors / num_window), active, dists

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self._score_with_dists(np.asarray(X, dtype=np.float64))[0]

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance and adds it to the window, reusing core distances across both steps.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        x = np.array(X, dtype=np.float64)
        score, active, dists = self._score_with_dists(x)
        self._fit_with_dists(x, active, dists)
        return score
