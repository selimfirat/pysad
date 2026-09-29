from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class RandomModel(BaseModel):
    """Random scorer that chooses a score between 0 and 1 ignoring the input.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for the scores. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        # None is resolved on every draw instead, so that a pickled model keeps drawing from the global state rather than from a frozen copy of it.
        self._rng = None if random_state is None else check_random_state(random_state)

    def fit_partial(self, X, y=None):
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: Returns the self.
        """
        return self

    def score_partial(self, X):
        """Randomly outputs a score from the uniform distribution.

        Args:
            X: any (Ignored)

        Returns:
            float: Uniform random between [0,1).
        """
        rng = self._rng if self._rng is not None else check_random_state(None)

        return rng.uniform()
