from pysad.core.base_model import BaseModel


class NullModel(BaseModel):
    """Returns 0.5 for every instance. Use it to test a pipeline."""

    def __init__(self):
        self.labels = []

    def fit_partial(self, X, y=None):
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: The model itself.
        """
        return self

    def score_partial(self, X):
        """Directly returns 0.5.

        Args:
            X: any (Ignored)

        Returns:
            float: Equal to `0.5`.
        """
        return 0.5
