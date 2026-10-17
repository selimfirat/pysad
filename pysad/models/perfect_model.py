from pysad.core.base_model import BaseModel


class PerfectModel(BaseModel):
    """Returns the ground truth label of each instance as its score, for tests and pipeline checks."""

    def __init__(self):
        self.labels = []

    def fit_partial(self, X, y):
        """Stores the ground truth label of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.
            y (int): The true label.

        Returns:
            object: Returns the self.
        """
        if y is None:
            raise ValueError("y should be the true score")

        self.labels.append(y)

        return self

    def score_partial(self, X):
        """Returns the oldest stored label that the model has not returned yet.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.

        Returns:
            float: The true label.
        """
        score = self.labels[0]
        self.labels = self.labels[1:]

        return score
