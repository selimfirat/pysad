from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from pysad.models.incremental_lof import _EPS, IncrementalLOF

# The step size, its decay per iteration, the number of iterations and the weight of the size
# constraint in the summarization, as in the experiments of the paper and its reference code.
_LEARNING_RATE = 0.3
_DECAY = 0.95
_NUM_ITERATIONS = 20
_LAMBDA = 0.001
# exp(LOF) overflows for the very high LOF values of duplicate instances.
_MAX_EXP_LOF = 50.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class DILOF(IncrementalLOF):
    """Scores each instance by its local outlier factor (LOF) in a window that DILOF summarizes when it is full :cite:`na2018dilof`.

    DILOF has two phases. The last outlier-aware detection (LOD) phase inserts each fitted instance into the window with the incremental LOF algorithm, as :class:`IncrementalLOF` does. The window keeps every instance until it holds `window_size` instances. Then the nonparametric density summarization (NDS) phase replaces the oldest half of the window with a quarter of the window, and keeps the newest half. NDS selects the summary with a binary decision variable for each old instance. It relaxes the variables to real values and runs gradient descent on a loss with four terms: an estimate of the Rényi divergence between the old instances and the summary from their k-distances, a shape term that prefers instances with a high LOF, which lie on the cluster boundaries, a penalty that keeps the variables between 0 and 1, and a penalty that keeps the summary size at a quarter of the window. The instances with the largest variables form the summary. The LOF values of the window are then computed again on the remaining instances.

    The skipping scheme of LOD targets a long sequence of outliers, which would otherwise fill the window and become a dense region. If the LOF of an inserted instance is above `threshold`, the instance is the last outlier. If the distance of the next fitted instance to the last outlier is below the mean distance of the window instances to their nearest neighbors, the model does not insert it, and it becomes the last outlier. Otherwise, skipping stops until the next outlier.

    `score_partial` returns the LOF of the instance against the current window, without inserting it, as :class:`IncrementalLOF` does. Until the window holds `num_neighbors + 1` instances, the score is 0.

    Deviations from the paper: the paper outputs a binary label for each instance, LOF above `threshold`. This model outputs the LOF itself, and `threshold` only controls the skipping. The paper sweeps the threshold to compute its AUROC and has no default; the default of 1.5 is a choice of this implementation. The details of NDS follow the reference code of the authors. These details are the 20 iterations from an initial value of 0.5, the step size of 0.3 multiplied by 0.95 before each iteration, and the quadratic penalties outside [0, 1]. They also include the approximation of the k-distance of an old instance in the summary: the term for the instance itself moves its k-distance towards its largest distance by the share of exp(sigmoid(LOF)) that its neighbors hold, and the term for the instances that have it as their k-th neighbor uses the first instance farther than their k-distance and nearer than 2 sigmoid(LOF) times it, for each instance whose neighbors hold more than the mean share. The k-distances of NDS are taken among the old instances, as in the loss of the paper, while the reference code takes them in the whole window. The divisions by a k-distance add 1e-10 to it, and the shape term caps the LOF at 50 in exp(LOF), so that duplicate instances do not give infinite values. In the reference code, the skipping scheme does not move the last outlier to a skipped instance, and an insertion does not update the neighbors of the window instances. This model follows the paper on both.

    Args:
        num_neighbors (int): The number of nearest neighbors, k in the paper (Default=8).
        window_size (int): The number of instances in the window that starts a summarization, W in the paper. Must be at least `2 * num_neighbors + 2` (Default=400).
        threshold (float or None): The LOF above which an inserted instance starts the skipping scheme, θ in the paper. None turns the skipping scheme off (Default=1.5).
    """

    def __init__(
        self,
        num_neighbors: int = 8,
        window_size: int = 400,
        threshold: float | None = 1.5,
    ) -> None:
        if window_size < 2 * num_neighbors + 2:
            raise ValueError(
                f"window_size must be at least 2 * num_neighbors + 2 = {2 * num_neighbors + 2}, got {window_size}."
            )
        super().__init__(num_neighbors=num_neighbors, window_size=window_size)
        self.threshold = threshold
        self._last_outlier: np.ndarray | None = None

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> DILOF:
        """Inserts the instance into the window unless the skipping scheme skips it, and summarizes the window when it is full.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        n = self.num_instances_
        if self._last_outlier is not None:
            mean_nn_dist = self.neighbor_dists_[:n].min(axis=1).mean()
            if np.linalg.norm(X - self._last_outlier) < mean_nn_dist:
                self._last_outlier = np.array(X, dtype=np.float64)
                return self
            self._last_outlier = None

        p = self._next
        super().fit_partial(X)
        n = self.num_instances_
        if self.threshold is not None and n > self.num_neighbors:
            lof = self.lrds_[self.neighbors_[p]].mean() / self.lrds_[p]
            if lof > self.threshold:
                self._last_outlier = self.X_[p].copy()

        if n == self.window_size:
            self._summarize()

        return self

    def _summarize(self) -> None:
        # NDS: selects a quarter of the window from its oldest half, slots [0, half).
        k = self.num_neighbors
        half = self.window_size // 2
        size = half // 2
        lofs = self.lrds_[self.neighbors_[:half]].mean(axis=1) / self.lrds_[:half]  # (half,)

        dists = cdist(self.X_[:half], self.X_[:half])  # (half, half)
        max_dists = dists.max(axis=1)
        np.fill_diagonal(dists, np.inf)
        neighbors = np.argpartition(dists, k - 1, axis=1)[:, :k]  # (half, k)
        neighbor_dists = np.take_along_axis(dists, neighbors, axis=1)
        kth = neighbors[np.arange(half), neighbor_dists.argmax(axis=1)]
        k_dists = neighbor_dists.max(axis=1)

        # The k-distance of each instance in the summary over its k-distance now, approximated.
        sig = _sigmoid(lofs)
        shares = np.exp(sig)
        neighbor_shares = shares[neighbors].sum(axis=1)
        self_ratios = (k_dists + neighbor_shares / shares.sum() * (max_dists - k_dists)) / (
            k_dists + _EPS
        )
        # The k-th neighbor in the summary of the instances whose neighbors are likely removed.
        band = (dists > k_dists[:, None]) & (dists < 2.0 * sig[:, None] * k_dists[:, None])
        moved = (neighbor_shares > neighbor_shares.mean()) & band.any(axis=1)
        kth[moved] = band[moved].argmax(axis=1)
        kth_ratios = dists[np.arange(half), kth] / (k_dists + _EPS)
        shape = np.exp(np.minimum(lofs, _MAX_EXP_LOF))

        y = np.full(half, 0.5)
        learning_rate = _LEARNING_RATE
        for _ in range(_NUM_ITERATIONS):
            learning_rate *= _DECAY
            gradients = np.bincount(kth, weights=y * kth_ratios, minlength=half)
            gradients += self_ratios - shape + _LAMBDA * (y.sum() - size)
            gradients += 2.0 * (np.maximum(y - 1.0, 0.0) + np.minimum(y, 0.0))
            y -= learning_rate * gradients

        summary = np.sort(np.argsort(-y, kind="stable")[:size])
        kept = np.concatenate([summary, np.arange(half, self.window_size)])
        n = len(kept)
        self.X_[:n] = self.X_[kept]
        self.num_instances_ = n
        self._next = n
        self._update_neighbors(np.arange(n), exclude=None)
        self._update_lrds(np.arange(n))
