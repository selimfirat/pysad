import scipy
import numpy as np
from pysad.core.base_model import BaseModel
from pysad.utils.window import Window


class ExactStorm(BaseModel):
    """The Exact-STORM method :cite:`angiulli2007detecting`. Following the paper, an instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius`, and the scored instance is never its own neighbor. In the paper, an instance is an outlier if it has fewer than k neighbors. This method assigns an anomaly score that is the fraction of window instances that are not neighbors of the scored instance, so instances with fewer neighbors get higher scores. An instance scored against an empty window has no neighbors and gets the maximum score of 1. Note that the decision making with a fixed k in :cite:`angiulli2007detecting` is not implemented.

            Args:
                window_size : int (Default=10000)
                    The number of instances in the window to score.
                max_radius : float (Default=0.1)
                    Maximum radius for the near instance selection.
    """

    def __init__(self, window_size=10000, max_radius=0.1):
        self.max_radius = max_radius
        self.window = Window(window_size=window_size)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance. Simply, adds the instance to the window.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        self.window.update(X)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance against all instances in the window.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self._score(self.window.get(), X)

    def fit_score_partial(self, X, y=None):
        """Adds the instance to the window and scores it against the other instances in the window, so that the instance is not counted as its own neighbor.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        self.fit_partial(X, y)

        return self._score(self.window.get()[:-1], X)

    def _score(self, window, X):
        if len(window) == 0:
            return 1.0

        dists = scipy.spatial.distance.cdist(window, [X])

        return 1.0 - np.mean(dists <= self.max_radius)
