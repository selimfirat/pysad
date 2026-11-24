from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class NullModel(BaseModel):
    """Returns 0.5 for every instance. Use it as a neutral baseline in a pipeline.

    Higher scores represent more anomalous, lower scores represent more normal.
    The score is constantly 0.5 regardless of input, so it has no memory and
    does not change state.
    """

    _is_detector: ClassVar[bool] = False

    def fit_partial(self, X: Any, y: Any = None) -> NullModel:
        """This method is ignored. Added for convenience.

        The model does not learn from instances, so ``X`` and ``y`` are accepted
        but have no effect.

        Args:
            X: any
                The instance to fit (ignored).

            y: any
                The label (ignored).

        Returns:
            object: The model itself.
        """
        return self

    def score_partial(self, X: Any) -> float:
        """Directly returns 0.5.

        The model does not observe ``X``, so the score is constant.

        Args:
            X: any (Ignored)
                The instance to score.

        Returns:
            float: Equal to ``0.5``. This baseline model returns a constant score for every
                instance, used as a reference for pipeline testing.
        """
        return 0.5


class PerfectModel(BaseModel):
    """Returns the ground truth label of each instance as its score, for tests and pipeline checks.

    Higher scores represent more anomalous if the true label is 1, lower scores
    represent more normal if the label is 0. The model stores labels in a first-in
    first-out buffer with bounded memory; once the buffer is full the oldest labels
    are dropped.

    During warm-up (before any label is fitted) the score is 0.0.
    """

    _is_detector: ClassVar[bool] = False

    def __init__(self) -> None:
        self.labels: deque[int] = deque()

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> PerfectModel:
        """Stores the ground truth label of the next instance.

        The model stores labels in a first-in-first-out buffer. The buffer has
        bounded memory: when too many labels are stored the oldest are dropped so
        the number of kept labels never grows unbounded.

        Args:
            X (np.float64 array of shape (num_features,)):
                The instance to fit (ignored).

            y (int):
                The true label of the instance.

        Returns:
            object: The model itself.
        """
        if y is None:
            raise ValueError("y should be the true score")

        self.labels.append(y)

        return self

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
        """Stores the ground truth label of the next instance, and then returns it as the score.

        The model needs the label before it can score, so it fits first, unlike the
        default of :class:`~pysad.core.BaseModel`. This order is prequential evaluation,
        also called test-then-train. The model never scores an instance that it learned
        from.

        The bounded-memory FIFO buffer ensures the number of stored labels never
        grows unbounded; old labels are dropped as new ones are added.

        Args:
            X (np.float64 array of shape (num_features,)):
                The instance to fit and score (ignored).

            y (int):
                The true label of the instance.

        Returns:
            float: The true label.
        """
        return float(self.fit_partial(X, y).score_partial(X))

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Returns the oldest stored label that the model has not returned yet, or ``0.0`` if none is stored.

        During warm-up (before any label is fitted via :meth:`fit_partial`) the score
        is 0.0. The bounded-memory FIFO buffer ensures the number of returned labels
        never exceeds the number of labels that were fitted.

        Args:
            X (np.float64 array of shape (num_features,)):
                The instance to score (ignored).

        Returns:
            float: The true label of the instance, or ``0.0`` before any label is fitted.
                This model returns ground truth labels as scores, used for pipeline testing
                and evaluation.
        """
        if not self.labels:
            return 0.0
        return float(self.labels.popleft())


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
            float: Uniform random score from ``[0, 1)``. This baseline model returns
                random scores for every instance, used as a chance reference in pipeline
                testing and evaluation.
        """
        rng = self._rng if self._rng is not None else check_random_state(None)

        return rng.uniform()
