from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.models._window import _RadiusCalibrator


class ApproxStorm(BaseModel):
    """Scores each instance by the estimated fraction of a sliding window that are not its neighbors (Approx-STORM) :cite:`angiulli2007detecting`.

    An instance is a neighbor of the scored instance if their distance is not greater than `max_radius` (R). As in Definition 3.1 and Figure 2 of the paper, each instance counts itself and its later neighbors in the window of length `window_size` (W). It becomes a safe inlier when this count reaches k, where k is ``num_neighbors``. A safe inlier stays an inlier until it expires. As in the paper, the model keeps every window instance that is not a safe inlier, but at most ``sample_fraction`` times W safe inliers (rho W). If more safe inliers form, the model removes randomly selected safe inliers, so that the kept safe inliers are a uniform sample of the window. This class stores all W window instances and runs the range query on the kept instances only, so the sample reduces the work per instance, not the memory.

    For each arriving instance, the paper stores fract_before: the number of kept safe inliers that are its neighbors, divided by the number of kept safe inliers. The paper estimates the number of neighbors that precede the instance in the window as fract_before times W. This method gives the score 1 - fract_before, which estimates the fraction of window instances that are not neighbors of the instance, the score of :class:`~pysad.models.ExactStorm`. The paper reports an instance as an outlier if its estimated number of neighbors is less than k. This method does not use this threshold, because it gives the same score to most of the stream. Each instance gets its score once, when it arrives, so the later count of its neighbors does not change its score.

    The Figure 2 pseudo-code of the paper divides the number of all kept neighbors by the number of kept safe inliers. The text and the error bounds of the paper count the safe inliers only, and so does this method. The pseudo-code removes a safe inlier during the range query, each time the safe inliers exceed rho W. This method computes fract_before from the range query first, and then removes the excess safe inliers, so that ``score_partial`` gives the same score as ``fit_score_partial`` without random draws. Until a safe inlier exists, fract_before is 0 and the score is 1.

    The paper sets the radius for each dataset. If ``max_radius`` is None, the radius comes from the first ``calibration_size`` instances, as in :class:`~pysad.models.ExactStorm`. Until the radius is fixed, each neighbor count uses the radius at the arrival of the later instance.

    Args:
        window_size (int): The number of instances in the window W (Default=10000, as in the paper).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        num_neighbors (int): The number of neighbors k that makes an instance a safe inlier, counting the instance itself (Default=50, as in the paper).
        sample_fraction (float): The fraction rho of the window that the kept safe inliers can fill. The paper uses 0.01, 0.05 and 0.1 (Default=0.05).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the removed safe inliers. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        window_size: int = 10000,
        max_radius: float | None = None,
        num_neighbors: int = 50,
        sample_fraction: float = 0.05,
        calibration_size: int = 256,
        outlier_fraction: float = 0.05,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_safe_inliers = int(sample_fraction * window_size)
        if self.num_safe_inliers < 1:
            raise ValueError(
                "sample_fraction * window_size must be at least 1, so that a safe inlier can be kept."
            )
        self.window_size = window_size
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.sample_fraction = sample_fraction
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )
        self.radius_: float | None = max_radius
        self._calibrator = _RadiusCalibrator(max_radius, calibration_size, outlier_fraction)
        self.points_ = np.empty((0, 0))  # the window as a ring buffer, (window_size, num_features)
        self.kept_ = np.zeros(window_size, dtype=bool)  # whether each slot holds a kept instance
        # The number of the instance and its later neighbors, for each slot.
        self.count_after_ = np.zeros(window_size, dtype=np.int64)
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ApproxStorm:
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

        i = self._next
        if self.radius_ is not None:
            kept, neighbors = self._neighbors(X)
            self.count_after_[kept[neighbors]] += 1

        self.points_[i] = X
        self.kept_[i] = True
        self.count_after_[i] = 1
        self._next = (i + 1) % self.window_size

        safe = np.flatnonzero(self.kept_ & (self.count_after_ >= self.num_neighbors))
        if len(safe) > self.num_safe_inliers:
            rng = self._rng if self._rng is not None else check_random_state(None)
            removed = rng.choice(safe, size=len(safe) - self.num_safe_inliers, replace=False)
            self.kept_[removed] = False

        if self._calibrator.active:
            self.radius_ = self._calibrator.update(X)

        return self

    def _neighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the kept slots, except the slot of the next instance, and which of them are neighbors of X."""
        kept = np.flatnonzero(self.kept_)
        kept = kept[kept != self._next]
        return kept, np.linalg.norm(self.points_[kept] - X, axis=1) <= self.radius_

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without updating the model.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored without it. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.radius_ is None or self.points_.size == 0:
            return 1.0

        kept, neighbors = self._neighbors(np.asarray(X, dtype=np.float64))
        safe = self.count_after_[kept] + neighbors >= self.num_neighbors  # after the count of X
        num_safe = safe.sum()
        if num_safe == 0:
            return 1.0

        return float(1.0 - (safe & neighbors).sum() / num_safe)
