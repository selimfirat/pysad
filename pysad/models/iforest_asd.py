from sklearn.ensemble import IsolationForest

from pysad.models.integrations.reference_window_model import ReferenceWindowModel


class _IsolationForest(IsolationForest):
    def decision_function(self, X):
        # scikit-learn gives lower scores to anomalies.
        return -super().decision_function(X)


class IForestASD(ReferenceWindowModel):
    """Scores each instance with an isolation forest that it retrains on a sliding reference window (IForestASD) :cite:`ding2013anomaly`.

    The model follows Algorithm 2 of the paper. The paper retrains the forest only when the anomaly rate of a window shows a concept drift. This model does not implement the concept drift check. It retrains the forest on the reference window after every ``window_size`` instances. The score is the anomaly score of the paper minus the ``contamination`` quantile of the training scores of the current forest. The model is unsupervised and ignores ``y``.

    Args:
        initial_window_X (np.float64 array of shape (num_initial_instances, num_features)): The initial window to fit for the initial calibration period. The model fits the forest to these instances (Default=None).
        window_size (int): The size of the reference window and its sliding (Default=2048).
        **kwargs: Keyword arguments passed to scikit-learn's ``IsolationForest``, for example ``n_estimators``. ``contamination`` defaults to 0.1.
    """

    def __init__(self, initial_window_X=None, window_size=2048, **kwargs):
        kwargs = {"contamination": 0.1, **kwargs}
        super().__init__(_IsolationForest, window_size, window_size, initial_window_X, **kwargs)
        # TODO: implement concept drift method

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance. `y` is accepted for API consistency but always
        ignored, since the model is unsupervised.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        return super().fit_partial(X, None)
