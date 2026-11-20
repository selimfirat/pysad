from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class ReservoirLifelongDetector(BaseModel):
    """Detects anomalies with a bounded reservoir memory and k-nearest-neighbor scoring.

    Maintains a reservoir sample of the stream using Algorithm R (Vitter, 1985) so that every
    instance seen so far has equal probability of being in the reservoir. Anomaly scores are
    the mean Euclidean distance to the ``num_neighbors`` nearest reservoir samples.

    This adapts the episodic memory idea of Gradient Episodic Memory (GEM) by Lopez-Paz and
    Ranzato (NeurIPS 2017) to unsupervised streaming anomaly detection: the reservoir replays a
    representative subset of past instances, providing continual-learning stability without storing
    the full stream.

    Until the reservoir is full, scores are zero. The reservoir is updated after scoring,
    so each instance is scored against the reservoir from earlier instances (prequential order).

    Memory is O(reservoir_size * num_features). Each call costs O(reservoir_size * num_features),
    independent of stream length.

    Args:
        reservoir_size (int): Maximum number of instances stored in the reservoir (Default=256).
        num_neighbors (int): Number of nearest reservoir samples used for scoring; at most reservoir_size (Default=5).
        random_state (int, np.random.RandomState or None): Seed or generator for reservoir sampling.
            None uses NumPy's global generator (Default=None).

    References:
        .. bibliography::
           :filter: docname in docnames
    """

    def __init__(
        self,
        reservoir_size: int = 256,
        num_neighbors: int = 5,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if reservoir_size < 1:
            raise ValueError("reservoir_size must be a positive integer.")
        if not 1 <= num_neighbors <= reservoir_size:
            raise ValueError("num_neighbors must be between 1 and reservoir_size.")
        self.reservoir_size = reservoir_size
        self.num_neighbors = num_neighbors
        self.random_state = random_state
        self.num_seen_ = 0
        self.reservoir_: np.ndarray | None = None
        self.rng: np.random.RandomState = check_random_state(random_state)

    def __getstate__(self) -> dict[str, Any]:
        # NumPy's global random state is not pickled, so that the restored model keeps drawing from it.
        state = self.__dict__.copy()
        if self.rng is np.random.mtrand._rand:
            state["rng"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = check_random_state(self.rng)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ReservoirLifelongDetector:
        """Updates the reservoir with the next instance using reservoir sampling.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if self.num_seen_ == 0:
            self.reservoir_ = np.empty((self.reservoir_size, len(X)), dtype=np.float64)

        assert self.reservoir_ is not None
        reservoir = self.reservoir_
        if self.num_seen_ < self.reservoir_size:
            reservoir[self.num_seen_] = X
        else:
            j = self.rng.randint(0, self.num_seen_ + 1)
            if j < self.reservoir_size:
                reservoir[j] = X

        self.num_seen_ += 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Returns the mean Euclidean distance to the k nearest reservoir samples.

        Returns zero while the reservoir is not yet full.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.
                Higher scores represent more anomalous instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.num_seen_ < self.reservoir_size or self.reservoir_ is None:
            return 0.0

        # Use only the filled portion of the reservoir
        filled = self.reservoir_[: self.reservoir_size]
        distances = np.linalg.norm(filled - X, axis=1)
        k = min(self.num_neighbors, len(filled))
        nearest = np.partition(distances, k - 1)[:k]
        return float(nearest.mean())
