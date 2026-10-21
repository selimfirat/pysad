from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class RandomModel(BaseModel):
    """Returns a random score between 0 and 1 for every instance. Use it as a baseline.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for the scores. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

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
