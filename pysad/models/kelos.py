from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from pysad.core.base_model import BaseModel

_MIN_BANDWIDTH = 1e-12  # keeps the kernel finite when the k nearest centers coincide


class KELOS(BaseModel):
    """Kernel density estimation-based local outlier detection over sliding windows (KELOS) :cite:`qin2019scalable`.

    KELOS summarizes the window of the last `window_size` instances with the micro-clusters of Algorithm 2 (Section 7). An arriving instance joins its nearest cluster if the distance to the centroid is less than `radius` (theta), and starts a new cluster otherwise. The centroid of each cluster, weighted by the number of its instances, is an abstract kernel center (Section 5). The density at a point is the weighted balloon kernel estimate of Eq. 6 over its k nearest abstract kernel centers, with k = `num_neighbors` and the weights of Eq. 7. The bandwidth of each estimate follows the rule of thumb of Eq. 8, h = 1.06 sigma k^(-1/(d+1)). The KLOME score of Definition 3.1 is the z-score of the density at an instance among the densities at its k nearest abstract kernel centers. A low KLOME means an outlier, so this class returns -KLOME.

    Each instance gets its score when it arrives, in the window that it enters. Thus the oldest instance of a full window expires, and the arriving instance joins a cluster, before the score is computed. This is the pane-based window of Section 7 with a slide of one instance. The paper returns the N instances with the lowest KLOME in each window, and it prunes the clusters whose KLOME bounds (Lemma 6.1, Eqs. 11 to 13) show that they hold no top-N outlier. These bounds only skip the instances that are not in the top N, and this class scores every instance, so it does not use them. The deviations from the paper:

    - The paper uses the product kernel of Eqs. 5 and 6, with one bandwidth h_l per dimension from the weighted standard deviation sigma_l of Eq. 9. This class uses one Gaussian kernel of Eq. 2 of the Euclidean distance u, with the factor 1 / (sqrt(2 pi) h) of the univariate density, and one bandwidth h from sigma = sqrt(sum_l sigma_l^2). Thus it also drops the normalization h^(-d) of a d-variate kernel. The product of the d factors 1 / (sqrt(2 pi) h_l) spreads the densities over many orders of magnitude in high dimensions, and the z-score then gives almost the same score to every instance. On the first 2,000 points of the 17 ODDS streams, the product kernel gives a mean AUROC of 0.61 with k = 100 and 0.50 with k = 10, and this kernel gives 0.79 with k = 100.
    - Eq. 10 divides the weighted sum of the centers by k. The weights already sum to 1 and the paper calls sigma_l the weighted standard deviation, so this class uses the weighted mean of the centers.
    - The paper suggests to adapt theta while the stream evolves, but it gives no rule, so theta stays fixed. The default theta fits features scaled to the range from 0 to 1. The paper sets theta for each dataset.
    - The default k = 100 is a choice of this class. The efficiency experiments of Section 8.2 use k = 10, and in Section 8.3 the accuracy of KELOS on HTTP peaks for k from 80 to 100.
    - The z-score of Definition 3.1 is undefined if the densities at the k centers are equal, for example while the window holds a single cluster. Such an instance gets the score 0.

    Args:
        window_size (int): The number of instances in the window (Default=1000).
        radius (float): The radius threshold theta of the micro-clusters (Default=0.5).
        num_neighbors (int): The number of nearest abstract kernel centers k of each density estimate (Default=100).
    """

    def __init__(
        self, window_size: int = 1000, radius: float = 0.5, num_neighbors: int = 100
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.points_ = np.empty((0, 0))  # the window as a ring buffer, (window_size, num_features)
        self.labels_ = np.zeros(window_size, dtype=np.intp)  # the cluster of each window instance
        # The size of each cluster, 0 for a free slot. A window of n instances has at most n clusters.
        self.counts_ = np.zeros(window_size, dtype=np.int64)
        self.sums_ = np.empty((0, 0))  # the linear sum of each cluster, (window_size, num_features)
        self._num = 0  # the number of instances in the window
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def _insert(self, X: np.ndarray, counts: np.ndarray, sums: np.ndarray) -> int:
        """Moves the window of the clusters `counts` and `sums` to `X` in place and returns the cluster of `X`."""
        if self._num == self.window_size:
            oldest = self.labels_[self._next]
            counts[oldest] -= 1
            sums[oldest] = sums[oldest] - self.points_[self._next] if counts[oldest] else 0.0

        active = np.flatnonzero(counts)
        cluster = int(np.argmin(counts))  # a free slot, since the window has room for X
        if active.size:
            dists = np.linalg.norm(sums[active] / counts[active, None] - X, axis=1)
            nearest = int(dists.argmin())
            if dists[nearest] < self.radius:
                cluster = int(active[nearest])
        counts[cluster] += 1
        sums[cluster] += X
        return cluster

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> KELOS:
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
            self.sums_ = np.zeros((self.window_size, len(X)))

        self.labels_[self._next] = self._insert(X, self.counts_, self.sums_)
        self.points_[self._next] = X
        self._next = (self._next + 1) % self.window_size
        self._num = min(self._num + 1, self.window_size)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance in the window that it enters, without changing the model.

        The oldest instance of a full window expires and the instance joins a cluster, as in `fit_partial`, so this gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = np.asarray(X, dtype=np.float64)
        if self.points_.size == 0:
            return 0.0

        counts, sums = self.counts_.copy(), self.sums_.copy()
        self._insert(X, counts, sums)
        active = counts > 0
        weights = counts[active].astype(np.float64)  # (m,)
        centers = (
            sums[active] / weights[:, None] - X
        )  # (m, d), relative to the instance at the origin
        num_centers, num_features = centers.shape
        k = min(self.num_neighbors, num_centers)

        # The instance, then its k nearest centers, are the points where the density is estimated.
        sq_norms = np.einsum("ij,ij->i", centers, centers)  # (m,)
        points = np.vstack(
            [np.zeros(num_features), centers[np.argpartition(sq_norms, k - 1)[:k]]]
        )  # (k+1, d)
        sq_dists = np.maximum(
            sq_norms - 2 * points @ centers.T + np.einsum("ij,ij->i", points, points)[:, None], 0.0
        )  # (k+1, m)
        nearest = np.argpartition(sq_dists, k - 1, axis=1)[:, :k]  # (k+1, k)
        nearest_sq_dists = np.take_along_axis(sq_dists, nearest, axis=1)  # (k+1, k)
        omega = weights[nearest]
        omega /= omega.sum(axis=1, keepdims=True)  # Eq. 7, (k+1, k)

        # sigma^2 = sum_l sigma_l^2 (Eq. 9) from the squared distances to the point p: sum(omega |cc - p|^2) - |mu - p|^2.
        omega_rows = np.zeros((k + 1, num_centers))
        np.put_along_axis(omega_rows, nearest, omega, axis=1)
        offsets = omega_rows @ centers - points  # mu - p, (k+1, d)
        variance = np.einsum("ij,ij->i", omega, nearest_sq_dists) - np.einsum(
            "ij,ij->i", offsets, offsets
        )
        sigma = np.sqrt(np.maximum(variance, 0.0))  # (k+1,)
        bandwidth = np.maximum(
            1.06 * sigma * k ** (-1 / (num_features + 1)), _MIN_BANDWIDTH
        )  # Eq. 8
        log_densities = logsumexp(
            -nearest_sq_dists / (2 * bandwidth[:, None] ** 2), b=omega, axis=1
        ) - np.log(np.sqrt(2 * np.pi) * bandwidth)  # Eqs. 2 and 6, (k+1,)

        # A z-score is unchanged when all densities scale by one factor, so scale the largest to 1.
        densities = np.exp(log_densities - log_densities.max())
        spread = densities[1:].std()
        if spread == 0.0:
            return 0.0
        return float((densities[1:].mean() - densities[0]) / spread)
