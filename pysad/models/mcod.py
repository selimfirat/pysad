from __future__ import annotations

import numpy as np
import scipy

from pysad.core.base_model import BaseModel
from pysad.models._window import _RadiusCalibrator

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
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius_: float | None = max_radius
        self._calibrator = _RadiusCalibrator(max_radius, calibration_size, outlier_fraction)
        max_clusters = window_size // (num_neighbors + 1) + 1
        self.points_ = np.empty((0, 0), dtype=np.float64)  # (window_size, num_features)
        self.cluster_ = np.full(window_size, _EMPTY, dtype=np.intp)
        self.centers_ = np.empty((0, 0), dtype=np.float64)  # (max micro-clusters, num_features)
        self.sizes_ = np.zeros(max_clusters, dtype=np.int64)
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window
        self._filled = 0
        self._pd_slots = np.empty(window_size, dtype=np.intp)
        self._num_pd = 0
        self._pos = np.empty(window_size, dtype=np.intp)
        self._members: list[list[int]] = [[] for _ in range(max_clusters)]
        self._active_clusters = np.empty(max_clusters, dtype=np.intp)
        self._active_pos = np.empty(max_clusters, dtype=np.intp)
        self._num_active = 0
        self._free_clusters: list[int] = list(range(max_clusters - 1, -1, -1))
        self._cand_slots = np.empty(window_size, dtype=np.intp)

    def _add_pd(self, i: int) -> None:
        self.cluster_[i] = _PD
        pos = self._num_pd
        self._pd_slots[pos] = i
        self._pos[i] = pos
        self._num_pd = pos + 1

    def _remove_pd(self, i: int) -> None:
        pos = int(self._pos[i])
        last = self._num_pd - 1
        if pos != last:
            moved = int(self._pd_slots[last])
            self._pd_slots[pos] = moved
            self._pos[moved] = pos
        self._num_pd = last

    def _add_to_cluster(self, c: int, i: int) -> None:
        members = self._members[c]
        self._pos[i] = len(members)
        members.append(i)
        self.cluster_[i] = c
        self.sizes_[c] += 1

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
            self.points_ = np.empty((self.window_size, len(X)), dtype=np.float64)
            self.centers_ = np.empty((len(self.sizes_), len(X)), dtype=np.float64)

        i = self._next
        if self._filled == self.window_size:
            self._expire(i)
        else:
            self._filled += 1
        self.points_[i] = X
        self._add_pd(i)
        self._next = (i + 1) % self.window_size

        if self._calibrator.active:
            self.radius_ = self._calibrator.update(X)
            if not self._calibrator.active and self.radius_ is not None:
                # The radius is fixed from now on, so the micro-clusters can form.
                for j in sorted(self._pd_slots[: self._num_pd].tolist()):
                    if self.cluster_[j] == _PD:
                        self._insert(j)
        elif self.radius_ is not None:
            self._insert(i)

        return self

    def _expire(self, i: int) -> None:
        """Step 1 and Step 4 of the paper: removes slot i and dissolves a micro-cluster that is too small."""
        c = int(self.cluster_[i])
        self.cluster_[i] = _EMPTY
        if c < 0:
            if c == _PD:
                self._remove_pd(i)
            return

        members = self._members[c]
        pos = int(self._pos[i])
        last_member = members.pop()
        if pos < len(members):
            members[pos] = last_member
            self._pos[last_member] = pos

        self.sizes_[c] -= 1
        if self.sizes_[c] > self.num_neighbors:
            return

        self.sizes_[c] = 0
        apos = int(self._active_pos[c])
        last_active = self._num_active - 1
        if apos != last_active:
            moved_c = int(self._active_clusters[last_active])
            self._active_clusters[apos] = moved_c
            self._active_pos[moved_c] = apos
        self._num_active = last_active
        self._free_clusters.append(c)

        dissolved = sorted(members)
        members.clear()
        for j in dissolved:
            self._add_pd(j)
        for j in dissolved:
            if self.cluster_[j] == _PD:
                self._insert(j)

    def _insert(self, i: int) -> None:
        """Steps 2 and 3 of the paper for the instance in slot i, which is in PD."""
        assert self.radius_ is not None
        if self._num_active > 0:
            active = self._active_clusters[: self._num_active]
            dists = scipy.spatial.distance.cdist(
                self.centers_[active], self.points_[i : i + 1]
            ).ravel()
            closest = int(dists.argmin())
            if dists[closest] <= self.radius_ / 2:
                self._remove_pd(i)
                self._add_to_cluster(int(active[closest]), i)
                return

        self._form(i)

    def _form(self, i: int) -> None:
        """Step 3b of the paper: a new micro-cluster centered at slot i if more than k instances of PD are within R/2."""
        assert self.radius_ is not None
        if self._num_pd < self.num_neighbors + 2:
            return
        pd = self._pd_slots[: self._num_pd]
        dists = scipy.spatial.distance.cdist(self.points_[pd], self.points_[i : i + 1]).ravel()
        close = pd[(pd != i) & (dists <= self.radius_ / 2)]
        if len(close) <= self.num_neighbors:
            return

        c = self._free_clusters.pop()
        apos = self._num_active
        self._active_clusters[apos] = c
        self._active_pos[c] = apos
        self._num_active = apos + 1
        self.centers_[c] = self.points_[i]
        for j in close.tolist():
            self._remove_pd(j)
            self._add_to_cluster(c, j)
        self._remove_pd(i)
        self._add_to_cluster(c, i)

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = np.asarray(X, dtype=np.float64)
        expiring = self._next if self._filled == self.window_size else -1
        num_window = self._filled - (1 if expiring >= 0 else 0)
        if num_window == 0 or self.radius_ is None:
            return 1.0

        query = X.reshape(1, -1)
        r = self.radius_
        expiring_cluster = int(self.cluster_[expiring]) if expiring >= 0 else _EMPTY
        num_neighbors = 0
        expiring_in_cands = expiring_cluster == _PD

        offset = self._num_pd
        if offset > 0:
            self._cand_slots[:offset] = self._pd_slots[:offset]

        if self._num_active > 0:
            active = self._active_clusters[: self._num_active]
            dists = scipy.spatial.distance.cdist(self.centers_[active], query).ravel()
            half_r = r / 2
            inside_mask = dists <= half_r
            if inside_mask.any():
                num_neighbors = int(self.sizes_[active[inside_mask]].sum())
                if expiring_cluster >= 0 and inside_mask[self._active_pos[expiring_cluster]]:
                    num_neighbors -= 1
            near_mask = (dists > half_r) & (dists <= 1.5 * r)
            if near_mask.any():
                if expiring_cluster >= 0 and near_mask[self._active_pos[expiring_cluster]]:
                    expiring_in_cands = True
                for c in active[near_mask].tolist():
                    members = self._members[c]
                    m_len = len(members)
                    self._cand_slots[offset : offset + m_len] = members
                    offset += m_len

        if offset > 0:
            candidates = self.points_[self._cand_slots[:offset]]
            num_neighbors += int((scipy.spatial.distance.cdist(candidates, query) <= r).sum())
            if expiring_in_cands and (
                scipy.spatial.distance.cdist(self.points_[expiring : expiring + 1], query)[0, 0]
                <= r
            ):
                num_neighbors -= 1

        return float(1.0 - num_neighbors / num_window)
