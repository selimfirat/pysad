from __future__ import annotations

from typing import Any

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate


class PredictiveAnomalyDetector(BaseModel):
    """Scores each value of a univariate stream by its deviation from the one-step forecast of a `River <https://riverml.xyz/>`_ forecaster, the plug-in method of EGADS :cite:`laptev2015generic`.

    The forecaster is the time-series modeling module of the paper: a black box that gives the expected value ``u_t`` of the next value ``x_t``. River forecasters include ``river.time_series.SNARIMAX`` and ``river.time_series.HoltWinters``. ``SNARIMAX(p=k, d=0, q=0, regressor=...)`` predicts each value from the last ``k`` values with any River regressor. The score of ``x_t`` is the deviation metric of the paper: the absolute prediction error ``|x_t - u_t|`` by default, or the absolute relative error ``|x_t - u_t| / |u_t|`` of Eq. (1) with ``relative=True``. The paper notes that the prediction error suits a regular series with an accurate model, and that the relative error removes the dependence on the magnitude of the expected value. The absolute prediction error is also the error ``e(t)`` of Telemanom :cite:`hundman2018detecting` (Sec. 3.2).

    The model returns the deviation itself, so you can threshold it with a postprocessor. The ``K sigma`` threshold of the paper (Sec. 4.1) corresponds to :class:`~pysad.transform.postprocessing.ZScorePostprocessor`, and the nonparametric dynamic threshold of Telemanom is :class:`~pysad.transform.postprocessing.NonparametricDynamicThresholdPostprocessor`.

    The model differs from the paper as follows:

    - EGADS trains its models in batch and applies them in real time (Sec. 2.1). Here the forecaster learns each value after the model scores it, so River updates its parameters online. River forecasters cannot condition on a new value without learning from it.
    - Eq. (1) is the signed relative error. The model takes its absolute value, so that a value far below its forecast also gets a high score.
    - The first ``probationary_period`` values get the score 0, a case that the paper does not discuss. Some forecasters cannot forecast before they see enough values. For example, ``HoltWinters`` needs ``max(2, seasonality)`` values.

    River needs Python 3.11 or newer. Install it with ``pip install river``.

    Args:
        model (object): A River forecaster. The detector trains this object in place.
        relative (bool): Whether to divide the absolute prediction error by the absolute forecast. A forecast of exactly 0 then raises ``ZeroDivisionError`` (Default=False).
        probationary_period (int): The number of first values that get the score 0 (Default=0).

    Raises:
        TypeError: If probationary_period is not an int.
        ValueError: If probationary_period is negative.
    """

    def __init__(self, model: Any, relative: bool = False, probationary_period: int = 0) -> None:
        self.model = model
        self.relative = relative
        self.probationary_period = _int_at_least(
            value=probationary_period,
            name="probationary_period",
            minimum=0,
        )
        self.num_values_ = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> PredictiveAnomalyDetector:
        """Updates the forecaster with the next value.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        self.model.learn_one(y=check_univariate(X, "PredictiveAnomalyDetector"))
        self.num_values_ += 1

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next value by its deviation from the forecast of the forecaster.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "PredictiveAnomalyDetector")
        if self.num_values_ < self.probationary_period:
            return 0.0

        forecast = self.model.forecast(horizon=1)[0]
        error = abs(value - forecast)

        return error / abs(forecast) if self.relative else error
