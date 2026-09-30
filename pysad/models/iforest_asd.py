import numbers

import numpy as np
from pyod.models.iforest import IForest

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


class IForestASD(ReferenceWindowModel):
    """Isolation Forest on sliding windows for streaming data :cite:`ding2013anomaly`.

    The detector is PyOD's unsupervised ``IForest``. ``y`` is ignored. Each window has ``window_size`` instances and the next window starts where the previous one ended.

    Concept drift follows Algorithm 2 of Ding and Fei (IFAC 2013). In that algorithm ``u`` is the predefined anomaly-rate threshold. After the first forest has been fit, the current forest labels every instance of the newest window. The window anomaly rate is the fraction of those instances labeled as anomalies. The forest is deleted and rebuilt from that window only when the rate is greater than ``u``. Otherwise the forest is kept and the window is discarded. ``drift_threshold`` is that ``u``.

    With ``drift_threshold=None`` the test is skipped and the forest is retrained on every window, which is the historical behavior of this class. The first forest is always fit, on ``initial_window_X`` when it is given and otherwise on the first window, before any drift test. An instance counts as an anomaly when PyOD's ``IForest.predict`` returns 1, so the rate uses the forest's contamination.

    Args:
        initial_window_X (np.float64 array of shape (num_initial_instances, num_features)): Instances used to fit the first forest. When omitted, the first window of the stream is used (Default=None).
        window_size (int): The number of instances in each window (Default=2048).
        drift_threshold (float or None): Algorithm 2's anomaly-rate threshold ``u``, in ``[0, 1]``. The current forest is rebuilt from a new window only when the fraction of instances it labels as anomalies is strictly greater than this value. ``None`` retrains on every window (Default=None).
        **kwargs: Keyword arguments passed to PyOD's ``IForest``, including ``contamination`` and ``random_state``.

    Raises:
        ValueError: If ``drift_threshold`` is a bool, a non-numeric value, or a number outside ``[0, 1]``.
    """

    def __init__(
        self,
        initial_window_X=None,
        window_size=2048,
        drift_threshold: float | None = None,
        **kwargs,
    ):
        _check_drift_threshold(drift_threshold)
        super().__init__(IForest, window_size, window_size, initial_window_X, **kwargs)
        self.drift_threshold = drift_threshold
        # An initial window is the first fit, so the next full window is tested.
        self._drift_check_active = self.initial_ref_window

    def fit_partial(self, X, y=None):
        """Fits the model to next instance. ``y`` is accepted for API consistency but always ignored, since the wrapped ``IForest`` is unsupervised and PyOD warns whenever a non-``None`` ``y`` reaches its ``fit``.

        When ``drift_threshold`` is set, a completed window is scored with the current forest and replaces it only if the window anomaly rate is greater than ``drift_threshold`` (Algorithm 2). The first window, or ``initial_window_X``, is always fit.

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
            self.reference_window_X = self.cur_window_X.copy()
            self.reference_window_y = None
            self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self._complete_window()

        return self

    def _complete_window(self):
        """Applies Algorithm 2 to a just-completed window.

        The newest window is discarded, and the current forest kept, unless this is the first fit or the window anomaly rate exceeds ``drift_threshold``. A rebuild uses the same reference-window update as ``ReferenceWindowModel``: with ``sliding_size == window_size`` that update is exactly the newest window.
        """
        window = self.cur_window_X
        if self._drift_check_active:
            labels = np.ravel(self.model.predict(np.asarray(window)))
            anomaly_rate = float(np.mean(labels == 1))
            if anomaly_rate <= self.drift_threshold:
                self.cur_window_X = []
                self.cur_window_y = []
                return

        self.reference_window_X = np.concatenate([self.reference_window_X, window], axis=0)
        self.reference_window_X = self.reference_window_X[
            max(0, len(self.reference_window_X) - self.window_size) :
        ]
        self.cur_window_X = []
        self.cur_window_y = []
        self._fit_model()
        self._drift_check_active = True
