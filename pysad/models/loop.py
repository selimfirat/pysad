from pysad.core.base_model import BaseModel


class StreamLocalOutlierProbability(BaseModel):

    def __init__(self, initial_X, num_neighbors=10, extent=3):
        from PyNomaly import loop

        self.model = loop.LocalOutlierProbability(data=initial_X, extent=extent, n_neighbors=num_neighbors)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                Ignored since the model is unsupervised.

        Returns:
            self: object
                Returns the self.
        """
        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score: float
                The anomalousness score of the input instance.
        """
        return self.model.stream(X)
