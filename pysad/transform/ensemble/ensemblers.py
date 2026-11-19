from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from pysad.core.base_score_ensembler import BaseScoreEnsembler


class PYODScoreEnsembler(BaseScoreEnsembler):
    """Abstract base class for stateless score-combination ensemblers :cite:`zhao2020combo`."""

    _stateless = True

    @abstractmethod
    def _combine(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Combines a 2-D matrix of detector scores into 1-D combined scores.

        Args:
            scores (np.float64 array of shape (num_instances, num_anomaly_detectors)): Scores from multiple anomaly detectors.

        Returns:
            np.float64 array of shape (num_instances,): Combined anomaly scores.
        """
        pass

    def fit_partial(self, scores: ArrayLike) -> PYODScoreEnsembler:
        """Fits one timestep's vector of detector scores (no-op for stateless ensemblers).

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors,)): Scores from multiple anomaly detectors.

        Returns:
            PYODScoreEnsembler: ``self``.
        """
        return self

    def transform_partial(self, scores: ArrayLike) -> float:
        """Combines anomaly scores from multiple anomaly detectors for one timestep.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors,)): Scores from multiple anomaly detectors.

        Returns:
            float: Combined anomaly score.
        """
        score_array = np.asarray(scores, dtype=np.float64).reshape(1, -1)
        return float(self._combine(score_array)[0])

    def fit(self, scores: np.ndarray) -> PYODScoreEnsembler:
        """Returns the ensembler unchanged without iterating the batch."""
        return self


class _DeterministicScoreEnsembler(PYODScoreEnsembler):
    """Base class for deterministic score ensemblers that combine a full batch in one call."""

    def transform(self, scores: np.ndarray) -> np.ndarray:
        arr = self._as_2d_scores(scores)
        if len(arr) == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self._combine(arr), dtype=np.float64).reshape(-1)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.transform(scores)


class AverageScoreEnsembler(_DeterministicScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by taking their (weighted) average :cite:`zhao2020combo`.

    Args:
        estimator_weights (np.float64 array of shape (num_anomaly_detectors,) or (1, num_anomaly_detectors) or None): Detector weights. If None, all detectors receive uniform weights (Default=None).
    """

    def __init__(self, estimator_weights: np.ndarray | None = None) -> None:
        self.estimator_weights = estimator_weights

    def _combine(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.estimator_weights is None:
            return np.asarray(scores.mean(axis=1), dtype=np.float64)
        weights = np.asarray(self.estimator_weights, dtype=np.float64)
        n_detectors = scores.shape[1]
        if weights.shape not in ((n_detectors,), (1, n_detectors)):
            raise ValueError(
                f"Bad input shape of estimator_weights: expected ({n_detectors},) or (1, {n_detectors}), "
                f"got {weights.shape}."
            )
        w = weights.ravel()
        return np.asarray((scores @ w) / w.sum(), dtype=np.float64)


class MaximumScoreEnsembler(_DeterministicScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by taking their maximum :cite:`zhao2020combo`."""

    def _combine(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(scores.max(axis=1), dtype=np.float64)


class MedianScoreEnsembler(_DeterministicScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by taking their median :cite:`zhao2020combo`."""

    def _combine(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(np.median(scores, axis=1), dtype=np.float64)


class _BucketScoreEnsembler(PYODScoreEnsembler):
    """Base class for score ensemblers that partition detectors into random buckets :cite:`aggarwal2015theoretical`.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, builds buckets randomly with a dynamic bucket size in ``[2, n_estimators // 2)`` (Default='static').
        bootstrap_estimators (bool): Whether buckets are drawn independently with replacement across buckets (Default=False).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the bucket assignment. None draws from NumPy's global random state, which ``pysad.utils.fix_seed`` seeds (Default=None).
    """

    _aom: bool

    def __init__(
        self,
        n_buckets: int = 5,
        method: str = "static",
        bootstrap_estimators: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.method = method
        self.n_buckets = n_buckets
        self.bootstrap_estimators = bootstrap_estimators
        self.random_state = random_state

    def _combine(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        n_estimators = scores.shape[1]
        if not 2 <= self.n_buckets <= n_estimators:
            raise ValueError(
                f"n_buckets must be between 2 and {n_estimators}, got {self.n_buckets}."
            )
        rng = check_random_state(self.random_state)

        if self.method == "static":
            if n_estimators % self.n_buckets != 0:
                raise ValueError(
                    "n_estimators / n_buckets has a remainder. Not allowed in static mode."
                )
            bucket_size = n_estimators // self.n_buckets
            if not self.bootstrap_estimators:
                idx = rng.permutation(n_estimators).reshape(self.n_buckets, bucket_size)
            else:
                idx = np.array(
                    [
                        rng.choice(n_estimators, size=bucket_size, replace=False)
                        for _ in range(self.n_buckets)
                    ]
                )
            grouped = scores[:, idx]  # (num_instances, n_buckets, bucket_size)
            if self._aom:
                return np.asarray(grouped.max(axis=2).mean(axis=1), dtype=np.float64)
            return np.asarray(grouped.mean(axis=2).max(axis=1), dtype=np.float64)

        if self.method == "dynamic":
            high = n_estimators // 2
            bucket_scores = np.empty((scores.shape[0], self.n_buckets), dtype=np.float64)
            for b in range(self.n_buckets):
                size = int(rng.randint(2, high))
                ind = rng.choice(n_estimators, size=size, replace=False)
                bucket_scores[:, b] = (
                    scores[:, ind].max(axis=1) if self._aom else scores[:, ind].mean(axis=1)
                )
            return np.asarray(
                bucket_scores.mean(axis=1) if self._aom else bucket_scores.max(axis=1),
                dtype=np.float64,
            )

        raise NotImplementedError(f"{self.method} is not implemented")


class AverageOfMaximumScoreEnsembler(_BucketScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by the average of the bucket maxima (AOM) :cite:`aggarwal2015theoretical,zhao2020combo`.

    The ensembler divides the detector scores into buckets, takes the maximum score in each bucket, and outputs the average across buckets.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, the ensembler builds buckets randomly with a dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement across buckets (Default=False).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the bucket assignment. None draws from NumPy's global random state, which ``pysad.utils.fix_seed`` seeds (Default=None).
    """

    _aom = True

    def __init__(
        self,
        n_buckets: int = 5,
        method: str = "static",
        bootstrap_estimators: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            n_buckets=n_buckets,
            method=method,
            bootstrap_estimators=bootstrap_estimators,
            random_state=random_state,
        )


class MaximumOfAverageScoreEnsembler(_BucketScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by the maximum of the bucket averages (MOA) :cite:`aggarwal2015theoretical,zhao2020combo`.

    The ensembler divides the detector scores into buckets, averages the scores in each bucket, and outputs the maximum across buckets.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, the ensembler builds buckets randomly with a dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement across buckets (Default=False).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the bucket assignment. None draws from NumPy's global random state, which ``pysad.utils.fix_seed`` seeds (Default=None).
    """

    _aom = False

    def __init__(
        self,
        n_buckets: int = 5,
        method: str = "static",
        bootstrap_estimators: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            n_buckets=n_buckets,
            method=method,
            bootstrap_estimators=bootstrap_estimators,
            random_state=random_state,
        )
