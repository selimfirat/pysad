import numbers

import numpy as np
from sklearn.ensemble import IsolationForest

from pysad.models.integrations.reference_window_model import ReferenceWindowModel


def _check_drift_threshold(drift_threshold: float | None) -> None:
    """Rejects a drift threshold outside the anomaly-rate range."""
    if drift_threshold is None:
        return

    if (
        isinstance(drift_threshold, bool)
        or not isinstance(drift_threshold, numbers.Real)
        or not 0.0 <= float(drift_threshold) <= 1.0
    ):
        raise ValueError(
            f"drift_threshold must be None or a real number in [0, 1], got {drift_threshold!r}."
        )


class _IsolationForest(IsolationForest):
    def decision_function(self, X):
        # scikit-learn gives lower scores to anomalies.
        return -super().decision_function(X)

    def predict(self, X):
        # 1 for anomalies, 0 for inliers.
        return (self.decision_function(X) > 0).astype(int)


class IForestASD(ReferenceWindowModel):
    """Scores each instance with an isolation forest that it retrains on a sliding reference window (IForestASD) :cite:`ding2013anomaly`.

    The model follows Algorithm 2 of the paper. After the first forest has been fit, the current forest scores the instances of each completed window. The forest is rebuilt from that window only when the window anomaly rate is at least ``drift_threshold`` (:math:`\\text{rate} \\ge u`). Otherwise, the existing forest is kept and the window is discarded. In their empirical evaluation on real-world benchmark datasets, the authors set :math:`u` to match each dataset's expected anomaly (contamination) rate.

    With ``drift_threshold=None`` (the default), the concept drift check is skipped and the forest is retrained on every window. The score is the anomaly score of the paper minus the ``contamination`` quantile of the training scores of the current forest. The model is unsupervised and ignores ``y``.

    Args:
        initial_window_X (np.float64 array of shape (num_initial_instances, num_features)): The initial window to fit for the initial calibration period. The model fits the forest to these instances (Default=None).
        window_size (int): The size of the reference window and its sliding (Default=2048).
        drift_threshold (float or None): Algorithm 2's anomaly-rate threshold :math:`u`, in ``[0, 1]``. The forest is rebuilt only when the fraction of instances labeled as anomalies in a full window is greater than or equal to this value. In their empirical evaluation on real-world benchmark datasets, the authors set :math:`u` to match each dataset's expected anomaly (contamination) rate. ``None`` retrains on every window (Default=None).
        **kwargs: Keyword arguments passed to scikit-learn's ``IsolationForest``, for example ``n_estimators``. ``contamination`` defaults to 0.1.
    """

    def __init__(
        self,
        initial_window_X=None,
        window_size=2048,
        drift_threshold: float | None = None,
        **kwargs,
    ):
        _check_drift_threshold(drift_threshold)
        kwargs = {"contamination": 0.1, **kwargs}
        super().__init__(_IsolationForest, window_size, window_size, initial_window_X, **kwargs)
        self.drift_threshold = drift_threshold
        self._drift_check_active = self.initial_ref_window

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance. ``y`` is accepted for API consistency but always
        ignored, since the model is unsupervised.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self.drift_threshold is None:
            return super().fit_partial(X, None)

        self.cur_window_X.append(X)

        if (
            not self.initial_ref_window
            and len(self.cur_window_X) < self.window_size
            and (self.reference_window_X is None or len(self.reference_window_X) < self.window_size)
        ):
            if len(self.cur_window_X) & (len(self.cur_window_X) - 1) == 0:
                self.reference_window_X = self.cur_window_X.copy()
                self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self._complete_window()

        return self

    def _complete_window(self):
        """Applies Algorithm 2 to a completed window."""
        window = self.cur_window_X
        if self._drift_check_active:
            labels = np.ravel(self.model.predict(np.asarray(window)))
            anomaly_rate = float(np.mean(labels == 1))
            if anomaly_rate < self.drift_threshold:
                self.cur_window_X = []
                self.cur_window_y = []
                return

        self.reference_window_X = np.asarray(self.cur_window_X)
        self.cur_window_X = []
        self.cur_window_y = []
        self._fit_model()
        self._drift_check_active = True
