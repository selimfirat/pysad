from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from pysad.models.incremental_lof import _EPS, IncrementalLOF, _lof

# The step size, its decay per iteration and the weight of the size constraint of the
# summarization come from the paper. The number of iterations comes from its reference code.
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

    The skipping scheme of LOD targets a long sequence of outliers, which would otherwise fill the window and become a dense region. If the LOF of an inserted instance is above `threshold`, the instance is the last outlier. If the distance of the next fitted instance to the last outlier is below the mean distance of the window instances to their nearest neighbors, the model does not insert it, and it becomes the last outlier. Otherwise, skipping stops until the next outlier. The paper adds an inserted instance to its set of outliers if its LOF is above `threshold` (Algorithm 4, lines 27 to 28), and adds each skipped instance to that set whatever its LOF (Algorithm 5, line 5).

    `score_partial` returns the LOF of the instance against the current window, without inserting it, as :class:`IncrementalLOF` does. If the skipping scheme would skip the instance, the score is at least the LOF of the last inserted outlier, which is above `threshold`. Until the window holds `num_neighbors + 1` instances, the score is 0.

    Deviations from the paper: the paper returns the LOF of an inserted instance and -1 for a skipped instance (Algorithm 4, lines 1 to 5). This model returns at least the LOF of the last inserted outlier for a skipped instance, as above, so that a higher score stays more anomalous. The model keeps no set of outliers, so `threshold` only controls the skipping. The paper measures accuracy with AUC and gives no default for `threshold`; the default of 1.5 is a choice of this implementation. The paper gives the initial value of 0.5, the step size of 0.3 multiplied by 0.95 before each iteration, and the quadratic penalties outside [0, 1]. The paper does not give the number of iterations, and this model uses the 20 iterations of the reference code of the authors. The approximation of the k-distance of an old instance in the summary follows the paper. The k-distance of the instance itself moves towards its largest distance by the share of exp(sigmoid(LOF)) that its neighbors hold (Eqs. 11 to 13). For each instance whose neighbors hold more than the mean share, the instances that have it as their k-th neighbor use the first instance farther than their k-distance and nearer than 2 exp(sigmoid(LOF)) times it (Algorithm 3, line 10), while the reference code uses 2 sigmoid(LOF). If no instance lies in that range, the model keeps the exact k-th neighbor, a case that the paper does not cover. For each old instance x_n, the gradient sums y_i d(x_i, x_n) / ν_k(x_i) over the instances x_i that have x_n as their approximated k-th neighbor, where y_i is the variable of x_i and ν_k(x_i) its k-distance. This term deviates from both forms in the paper: Algorithm 2, line 8 sums only y_i over the same instances, and Eq. 10 sums ρ_k(x_i) / ν_k(x_i). This model uses the term of the reference code because it is the exact derivative of the loss in Eq. 8 with the k-distance approximation of Eq. 9. The k-distances of NDS are taken among the old instances, as in the loss of the paper, while the reference code takes them in the whole window. The divisions by a k-distance add 1e-10 times the largest distance of the instance to the other old instances to it, so that the ratios do not depend on the unit of the values, and the shape term caps the LOF at 50 in exp(LOF), so that duplicate instances do not give infinite values. In the reference code, the skipping scheme does not move the last outlier to a skipped instance, the k-th neighbor bound is 2 sigmoid(LOF) times the k-distance, and an insertion does not update the neighbors of the window instances. This model follows the paper on all three.

    As in :class:`IncrementalLOF`, each instance has exactly `num_neighbors` neighbors, including under distance ties; the paper's k-distance neighborhood can contain more. The C++ reference initializes on half a window, while this model starts scoring once it has `num_neighbors + 1` instances. The reference code used for comparison is preserved at https://github.com/LiDan456/DILOF (commit 5ea0d019a1fefc10f2568572bc898ca73ab9d3bd). That mirror has no license; no code is copied from it.

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
        self._last_outlier_lof = 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> DILOF:
        """Inserts the instance into the window unless the skipping scheme skips it, and summarizes the window when it is full.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self._skips(X):
            self._last_outlier = np.array(X, dtype=np.float64)
            return self
        self._last_outlier = None

        p = self._next
        super().fit_partial(X)
        n = self.num_instances_
        if self.threshold is not None and n > self.num_neighbors:
            lof = _lof(self.reach_dists_[p : p + 1], self.reach_dists_[self.neighbors_[p]][None])[0]
            if lof > self.threshold:
                self._last_outlier = self.X_[p].copy()
                self._last_outlier_lof = float(lof)

        if n == self.window_size:
            self._summarize()

        return self

    def _skips(self, X: np.ndarray) -> bool:
        # Whether the skipping scheme skips the instance: it is closer to the last outlier than
        # the mean nearest-neighbor distance of the window.
        if self._last_outlier is None:
            return False
        mean_nn_dist = self.neighbor_dists_[: self.num_instances_].min(axis=1).mean()
        return bool(np.linalg.norm(X - self._last_outlier) < mean_nn_dist)

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance by its LOF against the current window, without inserting it.

        If the skipping scheme would skip the instance, the score is at least the LOF of the last inserted outlier.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The score of the instance. Higher scores represent more anomalous instances.
        """
        score = super().score_partial(X)
        if self._skips(X):
            return max(score, self._last_outlier_lof)
        return score

    def _summarize(self) -> None:
        # NDS: selects a quarter of the window from its oldest half, slots [0, half).
        k = self.num_neighbors
        half = self.window_size // 2
        size = half // 2
        lofs = _lof(self.reach_dists_[:half], self.reach_dists_[self.neighbors_[:half]])  # (half,)

        dists = cdist(self.X_[:half], self.X_[:half])  # (half, half)
        max_dists = dists.max(axis=1)
        np.fill_diagonal(dists, np.inf)
        neighbors = np.argpartition(dists, k - 1, axis=1)[:, :k]  # (half, k)
        neighbor_dists = np.take_along_axis(dists, neighbors, axis=1)
        kth = neighbors[np.arange(half), neighbor_dists.argmax(axis=1)]
        k_dists = neighbor_dists.max(axis=1)
        # The k-distances with a floor relative to the largest distance, so that the ratios do not depend on the unit.
        floored_k_dists = k_dists + np.where(max_dists > 0.0, _EPS * max_dists, 1.0)

        # The k-distance of each instance in the summary over its k-distance now, approximated.
        shares = np.exp(_sigmoid(lofs))
        neighbor_shares = shares[neighbors].sum(axis=1)
        self_ratios = (
            k_dists + neighbor_shares / shares.sum() * (max_dists - k_dists)
        ) / floored_k_dists
        # The k-th neighbor in the summary of the instances whose neighbors are likely removed.
        band = (dists > k_dists[:, None]) & (dists < 2.0 * (shares * k_dists)[:, None])
        moved = (neighbor_shares > neighbor_shares.mean()) & band.any(axis=1)
        kth[moved] = band[moved].argmax(axis=1)
        kth_ratios = dists[np.arange(half), kth] / floored_k_dists
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
        self._update_reach_dists(np.arange(n))
