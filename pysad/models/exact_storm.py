import scipy
import numpy as np
from pysad.core.base_model import BaseModel
from pysad.utils.window import Window


class ExactStorm(BaseModel):
    """The Exact-STORM method :cite:`angiulli2007detecting`. This method assigns anomaly score that is the mean of distances to the instances in window of length `window_size` with distnaces less than `max_radius`. Note that the decision making in :cite:`angiulli2007detecting` is not implemented.

            Args:
                window_size: int
                    The number of instances in the window to score.
                max_radius: float
                    Maximum radius for the near instance selection.
    """

    def __init__(self, window_size=10000, max_radius=0.1):
        self.max_radius = max_radius
        self.window = Window(window_size=window_size)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance. Simply, adds the instance to the window.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                The label of the instance (Optional for unsupervised models)
        """
        self.window.update(X)

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
        window = self.window.get()[:-1]
        if len(window) == 0:
            return 0.0

        dists = scipy.spatial.distance.cdist(window, [X])

        return np.mean(dists < self.max_radius)
