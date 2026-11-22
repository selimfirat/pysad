from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class LODA(BaseModel):
    """Scores each point by its density in histograms of sparse random projections :cite:`pevny2016loda`.

    The implementation is adapted to the streaming framework from the `PyOD framework <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/loda.html#LODA>`_.

    The model keeps ``num_random_cuts`` sparse random projections and a one-dimensional histogram on each of them. The projections are drawn once, when the first instance arrives: each has ``int(sqrt(num_features))`` non-zero components drawn from N(0, 1) and zeros elsewhere.

    Each histogram has ``num_bins`` equi-width bins and is updated incrementally with every instance. Since the range of the stream is not known in advance, the bins are grown on demand: when a projected value falls outside the current range, the bin width is doubled and adjacent pairs of bins are merged, which extends the range towards the new value, until the value fits. Merging pairs of bins keeps the counts exact within a window.

    Following the paper (§3.4 and §4.2), two histograms per projection alternate: one histogram is updated for ``window_size`` points while the other is used for scoring, then they swap and the new active histogram is reset. Old windows are forgotten, so a single extreme point can only affect scores for a bounded number of updates instead of degrading the model forever. The default ``window_size=256`` follows the paper.

    The score of an instance is the mean over the projections of the negative log of the estimated density ``(count + 1) / ((num_seen + num_bins) * bin_width)`` of its bin, so higher scores are more anomalous. Values outside the current range are treated as falling into an empty bin. Instances with non-finite values are not fitted.

    ``explain_partial`` ranks the features of an instance by the one-tailed two-sample t statistic of Section 3.3 of the paper. The variances in the statistic are sample variances, as in Welch's t-test. The paper does not say which variance it uses.     The `anlearn <https://github.com/gaussalgo/anlearn>`_ implementation of the statistic uses population variances instead, which gives larger absolute values, largest for features that few projections use.

    As in the paper, ``fit_score_partial`` scores an instance before it updates the histograms with it.

    Args:
        num_bins (int): The number of bins of each histogram (Default=10).
        num_random_cuts (int): The number of random projections, i.e. histograms (Default=100).
        window_size (int): Number of points after which the active and scoring histograms swap. The paper uses 256 (Default=256).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the projections. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    __bibtex__ = """
@article{pevny2016loda,
  title={Loda: Lightweight on-line detector of anomalies},
  author={Pevn\'{y}, Tom\'{a}\v{s}},
  journal={Machine Learning},
  volume={102},
  pages={275--304},
  year={2016},
  publisher={Springer}
}
"""

    def __init__(
        self,
        num_bins: int = 10,
        num_random_cuts: int = 100,
        window_size: int = 256,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.to_init = True
        self.n_bins = num_bins
        self.n_random_cuts = num_random_cuts
        self.window_size = window_size
        self.random_state = random_state

    def _init_model(self, num_features: int) -> None:
        self.num_features = num_features
        n_nonzero_components = max(1, int(np.sqrt(self.num_features)))
        rng = check_random_state(self.random_state)

        self.projections_: NDArray[np.float64] = np.zeros((self.n_random_cuts, self.num_features))
        for i in range(self.n_random_cuts):
            nonzero = rng.permutation(self.num_features)[:n_nonzero_components]
            self.projections_[i, nonzero] = rng.randn(n_nonzero_components)

        # Two sets of histograms that alternate: [active, scoring]
        self.histograms_: NDArray[np.float64] = np.zeros((2, self.n_random_cuts, self.n_bins))
        self.bin_lows_: NDArray[np.float64] = np.zeros((2, self.n_random_cuts))
        self.bin_widths_: NDArray[np.float64] = np.zeros((2, self.n_random_cuts))
        self.num_seen_: NDArray[np.int64] = np.zeros(2, dtype=np.int64)

        self.active_idx_ = 0  # 0 or 1, which histogram is being updated
        self.points_in_active_ = 0

        self.to_init = False

    def _active(self) -> int:
        return self.active_idx_

    def _scoring(self) -> int:
        scoring = 1 - self.active_idx_
        # During warm-up, the scoring histogram is empty; use the active one instead.
        if self.num_seen_[scoring] == 0:
            return self.active_idx_
        return scoring

    def _bin_indices(self, projected: NDArray[np.float64], hist_idx: int) -> NDArray[np.float64]:
        with np.errstate(divide="ignore", invalid="ignore"):
            indices: NDArray[np.float64] = np.floor((projected - self.bin_lows_[hist_idx]) / self.bin_widths_[hist_idx])
            return indices

    def _extend(self, hist_idx: int, i: int, value: float) -> int:
        """Doubles the bin width of histogram ``i`` towards ``value`` until it covers ``value``."""
        while True:
            ind = np.floor((value - self.bin_lows_[hist_idx, i]) / self.bin_widths_[hist_idx, i])
            if 0 <= ind < self.n_bins:
                return int(ind)

            padded = np.zeros(2 * self.n_bins)
            if ind < 0:  # Extend to the left, the old bins become the right half.
                padded[self.n_bins:] = self.histograms_[hist_idx, i]
                self.bin_lows_[hist_idx, i] -= self.n_bins * self.bin_widths_[hist_idx, i]
            else:  # Extend to the right, the old bins become the left half.
                padded[: self.n_bins] = self.histograms_[hist_idx, i]
            self.histograms_[hist_idx, i] = padded.reshape(self.n_bins, 2).sum(axis=1)
            self.bin_widths_[hist_idx, i] *= 2.0

    def _reset_histogram(self, hist_idx: int) -> None:
        """Resets histogram ``hist_idx`` to empty state."""
        self.histograms_[hist_idx].fill(0.0)
        self.bin_lows_[hist_idx].fill(0.0)
        self.bin_widths_[hist_idx].fill(0.0)
        self.num_seen_[hist_idx] = 0

    def _fit_projected(self, projected: NDArray[np.float64], inds: NDArray[np.float64], hist_idx: int) -> None:
        if self.num_seen_[hist_idx] == 0:
            self.bin_lows_[hist_idx] = projected
            self.histograms_[hist_idx, :, 0] = 1.0
            self.num_seen_[hist_idx] = 1
            return

        in_range = (self.bin_widths_[hist_idx] > 0.0) & (inds >= 0) & (inds < self.n_bins)
        self.histograms_[hist_idx, in_range, inds[in_range].astype(np.intp)] += 1.0
        for i in np.flatnonzero(~in_range).tolist():
            if self.bin_widths_[hist_idx, i] == 0.0:  # Only a single distinct value seen so far.
                seen = self.bin_lows_[hist_idx, i]
                if projected[i] == seen:
                    self.histograms_[hist_idx, i, 0] += 1.0
                    continue
                width = abs(projected[i] - seen) / (self.n_bins - 0.5)
                if width == 0.0:
                    self.histograms_[hist_idx, i, 0] += 1.0
                    continue
                # The smaller value starts the first bin and the larger one falls in the middle of the last bin.
                self.bin_widths_[hist_idx, i] = width
                self.bin_lows_[hist_idx, i] = min(projected[i], seen)
                count = self.histograms_[hist_idx, i, 0]
                self.histograms_[hist_idx, i, 0] = 0.0
                self.histograms_[hist_idx, i, self._extend(hist_idx, i, seen)] = count
                self.histograms_[hist_idx, i, self._extend(hist_idx, i, projected[i])] += 1.0
            else:
                self.histograms_[hist_idx, i, self._extend(hist_idx, i, projected[i])] += 1.0

        self.num_seen_[hist_idx] += 1

    def _neg_log_densities(self, inds: NDArray[np.float64], hist_idx: int) -> NDArray[np.float64]:
        ready = self.bin_widths_[hist_idx] > 0.0
        in_range = ready & (inds >= 0) & (inds < self.n_bins)

        counts = np.zeros(self.n_random_cuts)
        counts[in_range] = self.histograms_[hist_idx, in_range, inds[in_range].astype(np.intp)]

        neg_log_densities = np.zeros(self.n_random_cuts)
        neg_log_densities[ready] = -np.log(
            (counts[ready] + 1.0) / ((self.num_seen_[hist_idx] + self.n_bins) * self.bin_widths_[hist_idx, ready])
        )

        return neg_log_densities

    def _maybe_swap_histograms(self) -> None:
        self.points_in_active_ += 1
        if self.points_in_active_ >= self.window_size:
            # Swap active and scoring histograms
            self.active_idx_ = 1 - self.active_idx_
            self.points_in_active_ = 0
            # Reset the new active histogram
            self._reset_histogram(self.active_idx_)

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> LODA:
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
        active = self._active()
        self._fit_projected(projected, self._bin_indices(projected, active), active)
        self._maybe_swap_histograms()

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        projected = self.projections_.dot(X.reshape(-1))
        scoring = self._scoring()
        return float(self._neg_log_densities(self._bin_indices(projected, scoring), scoring).mean())

    def explain_partial(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ranks the features of the next instance by their contribution to its score, with Eq. (4) of the paper.

        For each feature, the one-tailed two-sample t statistic compares the negative log densities of the projections that use the feature with those of the projections that do not. A higher value means that the feature contributes more to the anomalousness of the instance. Like ``score_partial``, it does not update the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to explain.

        Returns:
            np.float64 array of shape (num_features,): The t statistic of each feature. It is NaN for a feature that fewer than two projections use, or that fewer than two projections leave out, for example when the instance has a single feature.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        projected = self.projections_.dot(X.reshape(-1))
        scoring = self._scoring()
        # (num_random_cuts, 1)
        neg_log = self._neg_log_densities(self._bin_indices(projected, scoring), scoring)[:, None]
        uses = self.projections_ != 0.0  # (num_random_cuts, num_features)
        n_with = uses.sum(axis=0)
        n_without = self.n_random_cuts - n_with

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_with = (neg_log * uses).sum(axis=0) / n_with
            mean_without = (neg_log * ~uses).sum(axis=0) / n_without
            var_with = ((neg_log - mean_with) ** 2 * uses).sum(axis=0) / (n_with - 1)
            var_without = ((neg_log - mean_without) ** 2 * ~uses).sum(axis=0) / (n_without - 1)
            t_stats: NDArray[np.float64] = (mean_with - mean_without) / np.sqrt(
                var_with / n_with + var_without / n_without
            )
            return t_stats

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
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
        scoring = self._scoring()
        inds = self._bin_indices(projected, scoring)
        score = float(self._neg_log_densities(inds, scoring).mean())
        if np.all(np.isfinite(X)):  # Non-finite values would stretch the bins without bound.
            active = self._active()
            self._fit_projected(projected, self._bin_indices(projected, active), active)
            self._maybe_swap_histograms()

        return score

    def score_explain_partial(self, X: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Returns the anomaly score and per-feature t-statistic explanation for the next instance.

        The explanation uses the one-tailed two-sample t statistic from Section 3.3 of the paper
        (Eq. 4), comparing the negative log densities of projections that use each feature
        versus those that do not. Higher t-statistic values indicate features that contribute
        more to the anomalousness of the instance.

        Like ``score_partial``, it does not update the model or record the instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and explain.

        Returns:
            tuple: (score, t_stats) where
                score (float): The anomalousness score of the input instance.
                t_stats (np.float64 array of shape (num_features,)): t-statistic for each
                feature. Higher values mean the feature contributes more to anomalousness.
                NaN for features with insufficient projections (fewer than 2 using or not using).

        Raises:
            ValueError: If `X` holds NaN.
        """
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        score = self.score_partial(X)
        explanation = self.explain_partial(X)
        return score, explanation
