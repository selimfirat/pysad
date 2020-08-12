from pysad.core.base_model import BaseModel


class PerfectModel(BaseModel):
    """This model directly outputs the ground truth labels. This method is added for testing and pipelining purposes.
    """

    def __init__(self):
        self.labels = []

    def fit_partial(self, X, y):
        """Fits the model to the ground truth label. Adds the label to the self.label queue.

        Args:
            X: any (Ignored)
            y (int): The true label.

        Returns:
            object: Returns the self.
        """
        if y is None:
            raise ValueError("y should be the true score")

        self.labels.append(y)

        return self

    def score_partial(self, X):
        """Pops the score from the self.label queue.

        Args:
            X: any (Ignored)

        Returns:
            float: The true label.
        """
        score = self.labels[0]
        self.labels = self.labels[1:]

        return score
