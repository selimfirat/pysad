from pysad.core.base_model import BaseModel


class StreamLocalOutlierProbability(BaseModel):

    def __init__(self, initial_X, num_neighbors=10, extent=3):
        """The implementation of streaming Local Outlier Probabilities method :cite:`kriegel2009loop`, which uses the implementation of PyNomaly library :cite:`constantinou2018pynomaly`.

        Args:
            initial_X: Initial training data to calibrate the model.
            num_neighbors:
            extent: int
            an integer value that controls the statistical
            extent, e.g. lambda times the standard deviation from the mean (optional,
            default 3)
            :param n_neighbors: the total number of neighbors to consider w.r.t. each
            sample (optional, default 10)
        """
        from PyNomaly import loop

        self.model = loop.LocalOutlierProbability(initial_X, extent=extent, n_neighbors=num_neighbors).fit()

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
