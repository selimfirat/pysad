from pysad.core.base_model import BaseModel
import numpy as np


class RandomModel(BaseModel):
    """Random scorer that chooses a score between 0 and 1 ignoring the input.

    """

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
        return np.random.uniform()
