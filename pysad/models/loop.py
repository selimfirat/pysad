import numpy as np

from pysad.core.base_model import BaseModel


class LocalOutlierProbability(BaseModel):
    """Scores each instance by its local outlier probability against an initial reference set :cite:`kriegel2009loop`.

    The model uses the incremental and approximate version of LoOP by Hamlet et al. :cite:`hamlet2017incremental`, as the PyNomaly library :cite:`constantinou2018pynomaly` implements it. The model fits LoOP once on ``initial_X``. Then it scores each instance with values from that fit, for example a global expected value of the probabilistic distance. The model does not learn from the stream, so ``fit_partial`` does nothing.

    Args:
        initial_X (np.float64 array of shape (num_instances, num_features)): Initial training data to calibrate the model.
        num_neighbors (int): The total number of neighbors to consider w.r.t. each sample (Default=10).
        extent (int): The statistical extent lambda, the number of standard deviations from the mean that the probabilistic distance covers. PyNomaly accepts 1, 2 or 3, and warns for other values (Default=3).
    """

    def __init__(self, initial_X, num_neighbors=10, extent=3):
        from PyNomaly import loop

        self.model = loop.LocalOutlierProbability(
            initial_X, extent=extent, n_neighbors=num_neighbors
        ).fit()

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        # When the reference PLOFs have no spread (e.g. identical initial instances), PyNomaly
        # divides by a zero normalization; the score is then its limit, 1 for any instance with a
        # positive PLOF, so the warning is noise.
        with np.errstate(divide="ignore"):
            return self.model.stream(X)
