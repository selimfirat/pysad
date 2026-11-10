from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class OnlineKMeans(BaseModel):
    """Scores each instance by its squared Euclidean distance to the nearest center of the online k-means clustering of Liberty et al. :cite:`liberty2016online`.

    The model follows Algorithm 2 of the paper, the fully online algorithm, which needs no estimate of the stream length or of the optimal cost. The first ``num_clusters + 1`` distinct instances become centers. Half of the smallest squared distance between two of them is the lower bound ``w*`` on the optimal cost, and the facility cost starts at ``f = w* / num_clusters``. Every later instance ``v`` opens a new center at ``v`` with probability ``min(D(v, C)^2 / f, 1)``, where ``D(v, C)`` is the distance of ``v`` to the nearest center. When ``3 * num_clusters * (1 + log(n))`` centers have opened since the last doubling, where ``n`` is the number of instances so far, the facility cost doubles and the count restarts.

    The score of an instance is ``D(v, C)^2``. Before the first instance is fitted, the score is 0. ``fit_score_partial`` scores an instance before it updates the centers with it. The model keeps at most ``max_centers`` centers. When it opens a center while it already has ``max_centers``, it removes the oldest center.

    Deviations from the paper: the paper never removes a center, so its number of centers grows with the stream, ``O(num_clusters * log(n) * log(gamma * n))`` in expectation by Theorem 3, where ``gamma`` is the ratio of the largest to the smallest distance between two instances. The limit ``max_centers`` and the removal of the oldest center are not in the paper; they keep the memory and the cost of each call fixed. The facility cost and the count of opened centers do not change when a center is removed. The paper does not give the base of the logarithm; this model uses the natural logarithm. The paper gives no default for ``k``; the default ``num_clusters=10`` is a choice of this implementation. The practical variant of Section 4.1 (Algorithm 3) is not implemented.

    Args:
        num_clusters (int): The number of clusters k of the paper. The model opens more centers than k by design (Default=10).
        max_centers (int): The largest number of centers that the model keeps. It must be greater than ``num_clusters``. When it is reached, a new center replaces the oldest one (Default=100).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the decisions to open a center. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_clusters: int = 10,
        max_centers: int = 100,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if max_centers <= num_clusters:
            # The first num_clusters + 1 distinct instances must fit before the facility cost is set.
            raise ValueError(
                f"max_centers ({max_centers}) must be greater than num_clusters ({num_clusters})."
            )
        self.num_clusters = num_clusters
        self.max_centers = max_centers
        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )
        self._centers: np.ndarray | None = None
        self._num_added = 0
        self.num_centers_ = 0
        self.num_seen_ = 0
        self.facility_cost_ = 0.0
        self.num_opened_ = 0

    def _squared_distances(self, X: np.ndarray) -> np.ndarray:
        assert self._centers is not None
        return ((self._centers[: self.num_centers_] - X) ** 2).sum(axis=1)

    def _add_center(self, X: np.ndarray) -> None:
        assert self._centers is not None
        # The centers form a ring buffer, so a new center overwrites the oldest one once it is full.
        self._centers[self._num_added % self.max_centers] = X
        self._num_added += 1
        self.num_centers_ = min(self._num_added, self.max_centers)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OnlineKMeans:
        """Updates the centers with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self._centers is None:
            self._centers = np.empty((self.max_centers, X.shape[0]))
            self._add_center(X)
            return self

        nearest = self._squared_distances(X).min()

        # The first num_clusters + 1 distinct instances are the initial centers.
        if self.num_centers_ <= self.num_clusters:
            if nearest > 0.0:
                self._add_center(X)
                if self.num_centers_ == self.num_clusters + 1:
                    self._init_facility_cost()
            return self

        self.num_seen_ += 1
        rng = self._rng if self._rng is not None else check_random_state(None)
        # The draw is below 1, so this opens a center with probability min(nearest / f, 1).
        if rng.random_sample() < nearest / self.facility_cost_:
            self._add_center(X)
            self.num_opened_ += 1
            if self.num_opened_ >= 3 * self.num_clusters * (1 + np.log(self.num_seen_)):
                self.num_opened_ = 0
                self.facility_cost_ *= 2.0

        return self

    def _init_facility_cost(self) -> None:
        assert self._centers is not None
        initial = self._centers[: self.num_centers_]
        # (num_clusters + 1, num_clusters + 1) squared distances between the initial centers.
        squared = ((initial[:, None, :] - initial[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(squared, np.inf)
        self.facility_cost_ = float(squared.min()) / 2.0 / self.num_clusters
        self.num_seen_ = self.num_clusters + 1

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The squared Euclidean distance of the instance to the nearest center.
        """
        if self.num_centers_ == 0:
            return 0.0

        return float(self._squared_distances(X).min())
