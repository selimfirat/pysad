import numpy as np
import scipy

from pysad.core.base_model import BaseModel
from pysad.utils.window import Window


class ExactStorm(BaseModel):
    """Scores each instance by the fraction of instances in a sliding window that are not its neighbors (Exact-STORM) :cite:`angiulli2007detecting`.

    Following the paper, an instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius`, and the scored instance is never its own neighbor. In the paper, an instance is an outlier if it has fewer than k neighbors. This method assigns an anomaly score that is the fraction of window instances that are not neighbors of the scored instance, so instances with fewer neighbors get higher scores. An instance scored against an empty window has no neighbors and gets the maximum score of 1. This method does not use the fixed k of the paper. A score capped at k gives every instance with at least k neighbors the same score of 0, so it cannot rank most of the stream. Until the window is full, the fraction uses the instances that the window holds so far. A later instance cannot change the score of an earlier one, because each instance gets its score once, when it arrives.

    The paper sets the radius for each dataset, so that a few percent of the window are outliers. If ``max_radius`` is None, this method sets the radius in the same way from the first ``calibration_size`` instances. The radius is the ``1 - outlier_fraction`` quantile of the distance from each of these instances to its nearest other instance. Thus an ``outlier_fraction`` of them have no neighbor. This is the outlier rule of the paper with k = 1. The paper uses k = 0.003 times the window size, which is about 1 for 256 instances. Each instance is scored with the radius from the instances before it, so the radius changes until ``calibration_size`` instances arrive. Then it stays fixed. Until two instances arrive, there is no radius, and the score is 1.

    Args:
        window_size (int): The number of instances in the window to score (Default=10000).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
    """

    def __init__(
        self, window_size=10000, max_radius=None, calibration_size=256, outlier_fraction=0.05
    ):
        self.max_radius = max_radius
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.window = Window(window_size=window_size)
        self.radius_ = max_radius
        self._calibration = []  # the first instances, while they set the radius
        self._nearest = np.empty(0)  # the distance from each of them to its nearest other one

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance by adding it to the window.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        X = np.array(X, dtype=np.float64)
        self.window.update(X)
        if self.max_radius is None and len(self._calibration) < self.calibration_size:
            self._calibrate(X)

        return self

    def _calibrate(self, X):
        dists = np.linalg.norm(np.array(self._calibration) - X, axis=1) if self._calibration else []
        self._nearest = np.append(np.minimum(self._nearest, dists), np.min(dists, initial=np.inf))
        self._calibration.append(X)
        if len(self._calibration) >= 2:
            self.radius_ = np.quantile(self._nearest, 1 - self.outlier_fraction)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        window = self.window.get()

        return self._score(window[max(len(window) - self.window.window_size + 1, 0) :], X)

    def _score(self, window, X):
        if len(window) == 0 or self.radius_ is None:
            return 1.0

        dists = scipy.spatial.distance.cdist(window, [X])

        return 1.0 - np.mean(dists <= self.radius_)
