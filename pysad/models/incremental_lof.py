from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from pysad.core.base_model import BaseModel

# The lrd adds this fraction of the largest mean reachability distance among an instance and its
# neighbors to its mean reachability distance, so that duplicate instances get a finite density.
_EPS = 1e-10


def _lof(reach: np.ndarray, neighbor_reach: np.ndarray) -> np.ndarray:
    """The LOF of instances from their mean reachability distances, of shape (m,), and those of their neighbors, of shape (m, k)."""
    # A floor relative to the distances keeps the LOF free of the unit of the values; an absolute
    # one exceeds the distances of values in small units, which pulls each LOF towards 1.
    largest = np.maximum(reach, neighbor_reach.max(axis=1))
    # If every distance is 0, the instance and its neighbors are duplicates, with an LOF of 1.
    floor = np.where(largest > 0.0, _EPS * largest, 1.0)
    return (reach + floor) * (1.0 / (neighbor_reach + floor[:, None])).mean(axis=1)


class IncrementalLOF(BaseModel):
    """Scores each instance by its local outlier factor (LOF) in a sliding window that the incremental LOF algorithm updates :cite:`pokrajac2007incremental`.

    The LOF of an instance is the mean local reachability density (lrd) of its `num_neighbors` nearest neighbors divided by its own lrd, so instances in sparser regions than their neighbors get higher scores. The paper shows that the insertion of an instance changes the k-distances, the reachability distances and the lrd values of only a few instances near it, and gives the update rules for insertion and for deletion. This method applies the insertion rule to each fitted instance, and applies the deletion rule to the oldest instance once the window holds `window_size` instances, so that the window slides by one instance. The k-distances and lrd values of the window are thus equal to those of static LOF on the window.

    `score_partial` returns the LOF of the instance against the current window, without inserting it. This equals scikit-learn's `LocalOutlierFactor(novelty=True)` fitted on the window, with the sign flipped. Until the window holds `num_neighbors + 1` instances, the score is 0.

    Deviations from the paper: each instance has exactly `num_neighbors` neighbors, as in scikit-learn, while under ties the k-distance neighborhood of the paper can hold more. The lrd adds 1e-10 times the largest mean reachability distance among the instance and its neighbors to its mean reachability distance, so that duplicate instances get a finite density, and an instance whose neighbors and itself all have a mean reachability distance of 0 has an LOF of 1. scikit-learn adds an absolute 1e-10, which exceeds the distances of values in small units and pulls each LOF towards 1. Both give the same scores up to a relative difference of about 1e-10, unless a mean reachability distance is 0. The model does not keep the LOF values of the window instances, which the last update step of the paper maintains, because only arriving instances are scored. In Fig. 4 of the paper, the LOF of an arriving instance is computed after its insertion, so the k-distances and lrd values of its reverse neighbors are already updated. `score_partial` scores the instance before insertion, with the old k-distances and lrd values of its neighbors, so that scoring does not change the model state. The two scores differ when the instance enters the k-nearest neighbors of one of its neighbors. The paper has no sliding window and no window size. It deletes blocks of obsolete records when needed (Fig. 6), and its experiments only insert. The deletion of the oldest instance once the window holds `window_size` instances, and the default `window_size` of 1000, are choices of this implementation.

    Args:
        num_neighbors (int): The number of nearest neighbors, k in the paper (Default=10).
        window_size (int): The number of most recent instances that the model keeps. Must be at least `num_neighbors + 2` (Default=1000).
    """

    def __init__(self, num_neighbors: int = 10, window_size: int = 1000) -> None:
        if window_size < num_neighbors + 2:
            raise ValueError(
                f"window_size must be at least num_neighbors + 2 = {num_neighbors + 2}, got {window_size}."
            )
        self.num_neighbors = num_neighbors
        self.window_size = window_size
        self.num_instances_ = 0  # the number of instances in the window
        # The slot of the next instance, which holds the oldest instance once the window is full.
        self._next = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> IncrementalLOF:
        """Inserts the instance into the window, deleting the oldest instance if the window is full.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        k = self.num_neighbors
        if self.num_instances_ == 0:
            self.X_ = np.empty((self.window_size, len(X)))
            self.neighbors_ = np.empty((self.window_size, k), dtype=np.intp)
            self.neighbor_dists_ = np.empty((self.window_size, k))
            self.k_dists_ = np.empty(self.window_size)
            self.reach_dists_ = np.empty(self.window_size)  # the mean reachability distances

        p = self._next
        if self.num_instances_ == self.window_size:
            self._delete(p)
        self.X_[p] = X
        self._next = (p + 1) % self.window_size
        if self.num_instances_ < self.window_size:
            self.num_instances_ += 1

        n = self.num_instances_
        if n == k + 1:
            # The first instances that all have k neighbors: static LOF on them.
            self._update_neighbors(np.arange(n), exclude=None)
            self._update_reach_dists(np.arange(n))
        elif n > k + 1:
            self._insert(p)

        return self

    def _insert(self, p: int) -> None:
        # Insertion of the instance in slot p (Fig. 4 of the paper).
        dists = cdist(self.X_[p : p + 1], self.X_[: self.num_instances_])[0]  # (n,)
        dists[p] = np.inf
        # The instances whose k-distance shrinks: p replaces their farthest neighbor.
        updated = np.flatnonzero(dists < self.k_dists_[: self.num_instances_])
        farthest = self.neighbor_dists_[updated].argmax(axis=1)
        self.neighbors_[updated, farthest] = p
        self.neighbor_dists_[updated, farthest] = dists[updated]
        self.k_dists_[updated] = self.neighbor_dists_[updated].max(axis=1)

        neighbors = np.argpartition(dists, self.num_neighbors - 1)[: self.num_neighbors]
        self.neighbors_[p] = neighbors
        self.neighbor_dists_[p] = dists[neighbors]
        self.k_dists_[p] = dists[neighbors].max()

        self._update_reach_dists(np.concatenate([self._reverse_neighbors(updated), [p]]))

    def _delete(self, p: int) -> None:
        # Deletion of the instance in slot p, from a full window.
        updated = np.flatnonzero((self.neighbors_ == p).any(axis=1))
        updated = updated[updated != p]
        self._update_neighbors(updated, exclude=p)
        lrds = self._reverse_neighbors(updated)
        self._update_reach_dists(lrds[lrds != p])

    def _reverse_neighbors(self, updated: np.ndarray) -> np.ndarray:
        # The instances whose lrd changes when the k-distances of `updated` change: `updated`, and
        # each neighbor q of an instance o in `updated` that has o as a neighbor too (Fig. 4).
        candidates = self.neighbors_[updated]  # (num_updated, k)
        has_o = (self.neighbors_[candidates] == updated[:, None, None]).any(axis=2)
        return np.union1d(updated, candidates[has_o])

    def _update_neighbors(self, idx: np.ndarray, exclude: int | None) -> None:
        # Recomputes the k nearest neighbors of the instances in slots idx.
        n = self.num_instances_
        dists = cdist(self.X_[idx], self.X_[:n])  # (len(idx), n)
        dists[np.arange(len(idx)), idx] = np.inf
        if exclude is not None:
            dists[:, exclude] = np.inf
        neighbors = np.argpartition(dists, self.num_neighbors - 1, axis=1)[:, : self.num_neighbors]
        self.neighbors_[idx] = neighbors
        self.neighbor_dists_[idx] = np.take_along_axis(dists, neighbors, axis=1)
        self.k_dists_[idx] = self.neighbor_dists_[idx].max(axis=1)

    def _update_reach_dists(self, idx: np.ndarray) -> None:
        # reach-dist(q, o) = max(k-distance(o), d(q, o)) for each neighbor o of q.
        reach = np.maximum(self.k_dists_[self.neighbors_[idx]], self.neighbor_dists_[idx])
        self.reach_dists_[idx] = reach.mean(axis=1)

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance by its LOF against the current window, without inserting it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The LOF of the instance. Higher scores represent more anomalous instances.
        """
        n = self.num_instances_
        if n <= self.num_neighbors:
            return 0.0

        dists = cdist(np.atleast_2d(X), self.X_[:n])[0]  # (n,)
        neighbors = np.argpartition(dists, self.num_neighbors - 1)[: self.num_neighbors]
        reach = np.maximum(self.k_dists_[neighbors], dists[neighbors])

        return float(_lof(reach.mean(keepdims=True), self.reach_dists_[neighbors][None])[0])
