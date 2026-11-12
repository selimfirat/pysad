from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class NullModel(BaseModel):
    """Returns 0.5 for every instance. Use it to test a pipeline."""

    _is_detector: ClassVar[bool] = False

    def fit_partial(self, X: Any, y: Any = None) -> NullModel:
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: The model itself.
        """
        return self

    def score_partial(self, X: Any) -> float:
        """Directly returns 0.5.

        Args:
            X: any (Ignored)

        Returns:
            float: Equal to `0.5`.
        """
        return 0.5


class PerfectModel(BaseModel):
    """Returns the ground truth label of each instance as its score, for tests and pipeline checks."""

    _is_detector: ClassVar[bool] = False

    def __init__(self) -> None:
        self.labels: deque[int] = deque()

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
        return self.labels.popleft()


class RandomModel(BaseModel):
    """Returns a random score between 0 and 1 for every instance. Use it as a baseline.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for the scores. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    _is_detector: ClassVar[bool] = False

    def __init__(self, random_state: int | np.random.RandomState | None = None) -> None:
        self.random_state = random_state
        # None is resolved on every draw instead, so that a pickled model keeps drawing from the global state rather than from a frozen copy of it.
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )

    def fit_partial(self, X: Any, y: Any = None) -> RandomModel:
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: The model itself.
        """
        return self

    def score_partial(self, X: Any) -> float:
        """Randomly outputs a score from the uniform distribution.

        Args:
            X: any (Ignored)

        Returns:
            float: Uniform random between [0,1).
        """
        rng = self._rng if self._rng is not None else check_random_state(None)

        return rng.uniform()
