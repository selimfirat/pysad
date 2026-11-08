from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class StreamKMPlusPlus(BaseModel):
    """Distance to the nearest center of a StreamKM++ streaming coreset clustering :cite:`ackermann2012streamkmpp`.

    StreamKM++ maintains a small weighted coreset of the stream with the merge-and-reduce technique. Points accumulate in a buffer of ``coreset_size`` points. When the buffer is full, the coreset construction reduces it to a weighted coreset of ``coreset_size`` points: it samples points with the k-means++ seeding procedure, where each next point is picked with probability proportional to its weight times its squared distance to the nearest point already picked, and it weights each sampled point by the total weight of the points closest to it. The coreset enters level 0 of the merge-and-reduce levels. Whenever two coresets represent the same number of stream points, they merge into one set and the coreset construction reduces it back to ``coreset_size`` points, which moves up to the next level. Level ``i`` therefore holds a coreset of at most ``coreset_size`` points that represents ``2**i * coreset_size`` stream points.

    The centers come from weighted k-means on the union of the buffer and all level coresets: k-means++ seeding followed by Lloyd iterations, each center the weighted mean of the points closest to it. The score of an instance is its Euclidean distance to the nearest center, so instances far from every center score high.

    The paper computes the clustering once, after the stream ends. This detector recomputes the centers from the current coreset union after every instance, so each instance can be scored while the stream runs. The paper speeds up the sampling with a coreset tree; this detector samples by a direct scan over the points, which is fast for the small coreset sizes used here. The paper inserts points into the coreset tree one by one; this detector accumulates ``coreset_size`` points in a buffer and reduces the full buffer, which is the same step the tree takes when a leaf fills. Instances with non-finite values are not fitted, since they would poison the squared distances. Before any instance is fitted the score is 0.0, and with fewer points than ``num_clusters`` the centers are the points seen so far.

    Args:
        num_clusters (int): The number of centers k (Default=5).
        coreset_size (int): The number of points m in each coreset and in the buffer (Default=200).
        num_iterations (int): The number of Lloyd iterations for the centers (Default=10).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the coreset sampling and the k-means++ seeding. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_clusters: int = 5,
        coreset_size: int = 200,
        num_iterations: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive.")
        if coreset_size < 1:
            raise ValueError("coreset_size must be positive.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive.")

        self.num_clusters = num_clusters
        self.coreset_size = coreset_size
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.centers_ = np.empty((0, 0))  # the current centers, (num_clusters, num_features)
        self._buffer = np.empty(
            (0, 0)
        )  # the points since the last reduce, (coreset_size, num_features)
        self._num_buffered = 0  # the number of points in the buffer
        self._levels: list[
            tuple[np.ndarray, np.ndarray] | None
        ] = []  # one coreset per level, or None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.rng_ is np.random.mtrand._rand:
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.rng_ = check_random_state(self.rng_)

    def _coreset(self, points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reduces `points` with `weights` to a weighted coreset of at most `coreset_size` points.

        The first point is picked with probability proportional to its weight, and each next point with probability proportional to its weight times its squared distance to the nearest point already picked. Each picked point is weighted by the total weight of the points closest to it. A point already picked, or identical to one, has distance zero and is never picked again. When every remaining point coincides with a picked one, the coreset holds the points picked so far.

        Args:
            points (np.float64 array of shape (num_points, num_features)): The points to reduce.
            weights (np.float64 array of shape (num_points,)): The weight of each point.

        Returns:
            tuple: The sampled points of shape (num_sampled, num_features) and their weights of shape (num_sampled,), with the weights summing to the sum of `weights`.
        """
        num_points = points.shape[0]
        num_sampled = min(self.coreset_size, num_points)
        min_dists = np.full(num_points, np.inf)  # squared distance to the nearest picked point
        sampled: list[int] = []
        for j in range(num_sampled):
            if j == 0:
                probs = weights / weights.sum()
            else:
                probs = weights * min_dists
                total = probs.sum()
                if not np.isfinite(total) or total <= 0.0:
                    break
                probs = probs / total
            idx = int(self.rng_.choice(num_points, p=probs))
            sampled.append(idx)
            min_dists = np.minimum(min_dists, ((points - points[idx]) ** 2).sum(axis=1))

        sampled_points = points[sampled]
        nearest = (
            ((points[:, None, :] - sampled_points[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        )
        sampled_weights = np.zeros(len(sampled))
        np.add.at(sampled_weights, nearest, weights)

        return sampled_points, sampled_weights

    def _insert(self, coreset: tuple[np.ndarray, np.ndarray]) -> None:
        """Inserts `coreset` into the merge-and-reduce levels, merging and reducing upwards."""
        level = 0
        current = coreset
        while True:
            if level == len(self._levels):
                self._levels.append(current)
                return
            existing = self._levels[level]
            if existing is None:
                self._levels[level] = current
                return
            merged_points = np.concatenate([existing[0], current[0]])
            merged_weights = np.concatenate([existing[1], current[1]])
            self._levels[level] = None
            current = self._coreset(merged_points, merged_weights)
            level += 1

    def _union(self) -> tuple[np.ndarray, np.ndarray]:
        """The buffer and all level coresets as one weighted point set."""
        points = [self._buffer[: self._num_buffered]] if self._num_buffered else []
        weights = [np.ones(self._num_buffered)] if self._num_buffered else []
        for level in self._levels:
            if level is not None:
                points.append(level[0])
                weights.append(level[1])
        if not points:
            return np.empty((0, 0)), np.empty(0)

        return np.concatenate(points), np.concatenate(weights)

    def _kmeans(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted k-means++ seeding followed by Lloyd iterations on `points` with `weights`.

        Args:
            points (np.float64 array of shape (num_points, num_features)): The points to cluster.
            weights (np.float64 array of shape (num_points,)): The weight of each point.

        Returns:
            np.float64 array of shape (num_centers, num_features): The centers, with num_centers the smaller of `num_clusters` and the number of points.
        """
        num_points, num_features = points.shape
        num_centers = min(self.num_clusters, num_points)
        centers = np.repeat(points[:1], num_centers, axis=0)
        min_dists = np.full(num_points, np.inf)  # squared distance to the nearest center
        for j in range(num_centers):
            if j == 0:
                probs = weights / weights.sum()
            else:
                probs = weights * min_dists
                total = probs.sum()
                if not np.isfinite(total) or total <= 0.0:
                    break
                probs = probs / total
            idx = int(self.rng_.choice(num_points, p=probs))
            centers[j] = points[idx]
            min_dists = np.minimum(min_dists, ((points - points[idx]) ** 2).sum(axis=1))

        for _ in range(self.num_iterations):
            nearest = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            new_centers = centers.copy()
            for j in range(num_centers):
                members = nearest == j
                if members.any():
                    new_centers[j] = np.average(points[members], axis=0, weights=weights[members])
            if np.array_equal(new_centers, centers):
                break
            centers = new_centers

        return centers

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> StreamKMPlusPlus:
        """Updates the model with the next instance and recomputes the centers.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        x = np.array(X, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(x)):  # Would poison the squared distances.
            return self
        if self._buffer.size == 0:
            self._buffer = np.empty((self.coreset_size, x.shape[0]))
            self.centers_ = np.empty((0, x.shape[0]))

        self._buffer[self._num_buffered] = x
        self._num_buffered += 1
        if self._num_buffered == self.coreset_size:
            self._insert(self._coreset(self._buffer, np.ones(self.coreset_size)))
            self._num_buffered = 0

        points, weights = self._union()
        self.centers_ = self._kmeans(points, weights)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance by its distance to the nearest center.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.centers_.shape[0] == 0:
            return 0.0

        x = np.asarray(X, dtype=np.float64)
        return float(np.linalg.norm(self.centers_ - x, axis=1).min())
