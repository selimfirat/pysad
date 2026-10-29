from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class LODA(BaseModel):
    """Scores each point by its density in histograms of sparse random projections :cite:`pevny2016loda`.

    The implementation is adapted to the streaming framework from the `PyOD framework <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/loda.html#LODA>`_.

    The model keeps ``num_random_cuts`` sparse random projections and a one-dimensional histogram on each of them. The projections are drawn once, when the first instance arrives: each has ``int(sqrt(num_features))`` non-zero components drawn from N(0, 1) and zeros elsewhere.

    Each histogram has ``num_bins`` equi-width bins and is updated incrementally with every instance. Since the range of the stream is not known in advance, the bins are grown on demand: when a projected value falls outside the current range, the bin width is doubled and adjacent pairs of bins are merged, which extends the range towards the new value, until the value fits. Merging pairs of bins keeps the counts exact, so no instance is ever redistributed or forgotten. The initial bin width is set from the first two distinct values seen on a projection.

    The score of an instance is the mean over the projections of the negative log of the estimated density ``(count + 1) / ((num_seen + num_bins) * bin_width)`` of its bin, so higher scores are more anomalous. Values outside the current range are treated as falling into an empty bin. Instances with non-finite values are not fitted.

    As in the paper, ``fit_score_partial`` scores an instance before it updates the histograms with it.

    Args:
        num_bins (int): The number of bins of each histogram (Default=10).
        num_random_cuts (int): The number of random projections, i.e. histograms (Default=100).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the projections. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_bins: int = 10,
        num_random_cuts: int = 100,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.to_init = True
        self.n_bins = num_bins
        self.n_random_cuts = num_random_cuts
        self.random_state = random_state

    def _init_model(self, num_features: int) -> None:
        self.num_features = num_features
        n_nonzero_components = max(1, int(np.sqrt(self.num_features)))
        rng = check_random_state(self.random_state)

        self.projections_: np.ndarray = np.zeros((self.n_random_cuts, self.num_features))
        for i in range(self.n_random_cuts):
            nonzero = rng.permutation(self.num_features)[:n_nonzero_components]
            self.projections_[i, nonzero] = rng.randn(n_nonzero_components)

        self.histograms_: np.ndarray = np.zeros((self.n_random_cuts, self.n_bins))
        # Left edge and width of the bins of each histogram. A width of 0 means that the projection has only seen a single distinct value so far, which is kept in ``bin_lows_``.
        self.bin_lows_: np.ndarray = np.zeros(self.n_random_cuts)
        self.bin_widths_: np.ndarray = np.zeros(self.n_random_cuts)
        self.num_seen_ = 0

        self.to_init = False

    def _bin_indices(self, projected: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.floor((projected - self.bin_lows_) / self.bin_widths_)

    def _extend(self, i: int, value: float) -> int:
        """Doubles the bin width of histogram ``i`` towards ``value`` until it covers ``value``."""
        while True:
            ind = np.floor((value - self.bin_lows_[i]) / self.bin_widths_[i])
            if 0 <= ind < self.n_bins:
                return int(ind)

            padded = np.zeros(2 * self.n_bins)
            if ind < 0:  # Extend to the left, the old bins become the right half.
                padded[self.n_bins :] = self.histograms_[i]
                self.bin_lows_[i] -= self.n_bins * self.bin_widths_[i]
            else:  # Extend to the right, the old bins become the left half.
                padded[: self.n_bins] = self.histograms_[i]
            self.histograms_[i] = padded.reshape(self.n_bins, 2).sum(axis=1)
            self.bin_widths_[i] *= 2.0

    def _fit_projected(self, projected: np.ndarray, inds: np.ndarray) -> None:
        if self.num_seen_ == 0:
            self.bin_lows_[:] = projected
            self.histograms_[:, 0] = 1.0
            self.num_seen_ = 1
            return

        in_range = (self.bin_widths_ > 0.0) & (inds >= 0) & (inds < self.n_bins)
        self.histograms_[in_range, inds[in_range].astype(np.intp)] += 1.0
        for i in np.flatnonzero(~in_range).tolist():
            if self.bin_widths_[i] == 0.0:  # Only a single distinct value seen so far.
                seen = self.bin_lows_[i]
                if projected[i] == seen:
                    self.histograms_[i, 0] += 1.0
                    continue
                # The smaller value starts the first bin and the larger one falls in the middle of the last bin.
                self.bin_widths_[i] = abs(projected[i] - seen) / (self.n_bins - 0.5)
                self.bin_lows_[i] = min(projected[i], seen)
                count = self.histograms_[i, 0]
                self.histograms_[i, 0] = 0.0
                self.histograms_[i, self._extend(i, seen)] = count
                self.histograms_[i, self._extend(i, projected[i])] += 1.0
            else:
                self.histograms_[i, self._extend(i, projected[i])] += 1.0

        self.num_seen_ += 1

    def _score_projected(self, inds: np.ndarray) -> float:
        ready = self.bin_widths_ > 0.0
        in_range = ready & (inds >= 0) & (inds < self.n_bins)

        counts = np.zeros(self.n_random_cuts)
        counts[in_range] = self.histograms_[in_range, inds[in_range].astype(np.intp)]

        neg_log_densities = np.zeros(self.n_random_cuts)
        neg_log_densities[ready] = -np.log(
            (counts[ready] + 1.0) / ((self.num_seen_ + self.n_bins) * self.bin_widths_[ready])
        )

        return float(neg_log_densities.mean())

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> LODA:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        if not np.all(np.isfinite(X)):  # Would stretch the bins without bound.
            return self

        projected = self.projections_.dot(X.reshape(-1))
        self._fit_projected(projected, self._bin_indices(projected))

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        projected = self.projections_.dot(X.reshape(-1))
        return self._score_projected(self._bin_indices(projected))

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance, and then updates the model with it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        projected = self.projections_.dot(X.reshape(-1))
        inds = self._bin_indices(projected)
        score = self._score_projected(inds)
        if np.all(np.isfinite(X)):  # Non-finite values would stretch the bins without bound.
            self._fit_projected(projected, inds)

        return score
