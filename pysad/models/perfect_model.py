from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel


class PerfectModel(BaseModel):
    """Returns the ground truth label of each instance as its score, for tests and pipeline checks."""

    def __init__(self) -> None:
        self.labels: list[int] = []

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> PerfectModel:
        """Stores the ground truth label of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.
            y (int): The true label.

        Returns:
            object: The model itself.
        """
        if y is None:
            raise ValueError("y should be the true score")

        self.labels.append(y)

        return self

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Stores the ground truth label of the next instance, and then returns it as the score.

        The model needs the label before it can score, so it fits first, unlike the default of :class:`~pysad.core.BaseModel`.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.
            y (int): The true label.

        Returns:
            float: The true label.
        """
        return float(self.fit_partial(X, y).score_partial(X))

    def score_partial(self, X: np.ndarray) -> float:
        """Returns the oldest stored label that the model has not returned yet.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.

        Returns:
            float: The true label.
        """
        score = self.labels[0]
        self.labels = self.labels[1:]

        return score
