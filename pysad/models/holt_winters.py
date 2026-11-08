from __future__ import annotations

from collections import deque

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate

# The width of the moving average that smooths the seasonal coefficients and deviations within a
# period, as a fraction of the period.
SMOOTHING_WINDOW = 0.05


def _smooth(values: np.ndarray, offset: int) -> np.ndarray:
    """The circular, centered, equal-weight moving average of ``values`` over ``2 * offset + 1`` points."""
    padded = np.concatenate((values[-offset:], values, values[:offset]))  # (period + 2 * offset,)
    return np.convolve(padded, np.full(2 * offset + 1, 1.0 / (2 * offset + 1)), mode="valid")


class HoltWinters(BaseModel):
    """Detects aberrant behavior in a seasonal univariate stream with Holt-Winters forecasting, a confidence band of seasonal deviations and a moving window of violations :cite:`brutlag2000aberrant`.

    Holt-Winters forecasting splits the series into a baseline ``a``, a linear trend ``b`` and a seasonal coefficient ``c`` for each of the ``period`` points of the seasonal cycle. Each value ``y_t`` updates them with exponential smoothing:

    - ``a_t = alpha (y_t - c_{t-m}) + (1 - alpha)(a_{t-1} + b_{t-1})``
    - ``b_t = beta (a_t - a_{t-1}) + (1 - beta) b_{t-1}``
    - ``c_t = gamma (y_t - a_t) + (1 - gamma) c_{t-m}``

    The forecast of ``y_t`` is ``y_hat_t = a_{t-1} + b_{t-1} + c_{t-m}``, where ``m`` is the period. The predicted deviation ``d_t = gamma |y_t - y_hat_t| + (1 - gamma) d_{t-m}`` measures the variability at each point of the cycle. A value is a violation if it falls outside the confidence band ``(y_hat_t - delta d_{t-m}, y_hat_t + delta d_{t-m})``. A failure is at least ``threshold`` violations among the last ``window_size`` values.

    The score of a value is the number of violations among the last ``window_size`` values, the value included, if they make a failure, else 0.0. Higher scores are more anomalous. With ``threshold=1``, the score is the number of violations in the window.

    The model starts cold, as RRDtool does in the paper. The baseline is the first value and the trend is 0. During the first period, the seasonal coefficient of each point is the value minus the first value, and the baseline and the trend do not change. During the second period, the model starts to forecast, and the first error at each point of the cycle is its deviation. The model finds violations from the third period on, so the score is 0.0 for the first ``2 * period`` values. At the end of each period, an equal-weight moving average over ``2 * floor(0.025 * period) + 1`` points of the cycle smooths the seasonal coefficients and the deviations, as the paper does with a window of ``0.05 * period``. There is no smoothing for a period below 40.

    The implementation differs from the paper as follows:

    - The band is symmetric, with one ``delta``. The paper also allows different factors above and below the forecast.
    - The paper does not say when the model smooths the seasonal cycle. This model smooths it at the end of each period. RRDtool also moves the mean of the seasonal coefficients into the baseline when it smooths them, which the paper does not describe and this model does not do.
    - The model rejects NaN values. The paper forecasts several steps ahead to fill missing values.
    - The paper does not give default parameters. The defaults ``alpha=0.1``, ``beta=0.0035`` and ``gamma=0.1`` are those of the paper's RRDtool example, for a daily period of 288 values. The defaults ``delta=2.0``, ``window_size=9`` and ``threshold=7`` are the RRDtool defaults that the paper gives.
    - The paper's failure is a binary indicator. This model returns the number of violations in the window for a failure, so that the scores rank failures.

    Args:
        period (int): The number of values ``m`` in one seasonal cycle.
        alpha (float): The adaptation parameter of the baseline, in ``(0, 1)`` (Default=0.1).
        beta (float): The adaptation parameter of the trend, in ``(0, 1)`` (Default=0.0035).
        gamma (float): The adaptation parameter of the seasonal coefficients and the deviations, in ``(0, 1)`` (Default=0.1).
        delta (float): The width of the confidence band in predicted deviations, greater than 0. The paper suggests values between 2 and 3 (Default=2.0).
        window_size (int): The number of recent values in which the model counts violations (Default=9).
        threshold (int): The number of violations in the window that makes a failure, at most ``window_size`` (Default=7).

    Attributes:
        intercept_ (float): The baseline ``a``, or None before the first value.
        slope_ (float): The linear trend ``b``.
        seasonal_ (np.float64 array of shape (period,)): The seasonal coefficient of each point of the cycle, NaN until the model sees the point.
        deviation_ (np.float64 array of shape (period,)): The predicted deviation of each point of the cycle, NaN until the model forecasts the point.

    Raises:
        ValueError: If ``alpha``, ``beta`` or ``gamma`` is not in ``(0, 1)``, ``delta`` is not greater than 0, ``period``, ``window_size`` or ``threshold`` is less than 1, or ``threshold`` is greater than ``window_size``.
    """

    def __init__(
        self,
        period: int,
        alpha: float = 0.1,
        beta: float = 0.0035,
        gamma: float = 0.1,
        delta: float = 2.0,
        window_size: int = 9,
        threshold: int = 7,
    ) -> None:
        self.period = _int_at_least(period, "period", 1)
        self.alpha = _finite_float(alpha, "alpha")
        self.beta = _finite_float(beta, "beta")
        self.gamma = _finite_float(gamma, "gamma")
        for name, value in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}.")
        self.delta = _finite_float(delta, "delta")
        if self.delta <= 0.0:
            raise ValueError(f"delta must be greater than 0, got {self.delta}.")
        self.window_size = _int_at_least(window_size, "window_size", 1)
        self.threshold = _int_at_least(threshold, "threshold", 1)
        if self.threshold > self.window_size:
            raise ValueError(
                f"threshold must be at most window_size ({self.window_size}), got {self.threshold}."
            )

        self.intercept_: float | None = None
        self.slope_ = 0.0
        self.seasonal_ = np.full(self.period, np.nan)
        self.deviation_ = np.full(self.period, np.nan)
        self._num_values = 0
        # Whether each of the last window_size - 1 values was a violation.
        self._violations: deque[bool] = deque(maxlen=self.window_size - 1)

    def _forecast(self, x: float) -> tuple[float | None, bool]:
        """Returns the forecast of ``x``, None until the model forecasts, and whether ``x`` is a violation."""
        seasonal = self.seasonal_[self._num_values % self.period]
        if self.intercept_ is None or np.isnan(seasonal):
            return None, False

        forecast = self.intercept_ + self.slope_ + seasonal
        deviation = self.deviation_[self._num_values % self.period]
        # A NaN deviation, before the model forecasts this point once, makes no violation.
        return forecast, bool(abs(x - forecast) > self.delta * deviation)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> HoltWinters:
        """Updates the baseline, the trend, and the seasonal coefficient and deviation of the point of the cycle with the next value.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "HoltWinters")
        if self.intercept_ is None:
            self.intercept_ = x
        forecast, violation = self._forecast(x)
        k = self._num_values % self.period
        if forecast is None:
            self.seasonal_[k] = x - self.intercept_
        else:
            previous = self.intercept_
            self.intercept_ = self.alpha * (x - self.seasonal_[k]) + (1.0 - self.alpha) * (
                previous + self.slope_
            )
            self.slope_ = self.beta * (self.intercept_ - previous) + (1.0 - self.beta) * self.slope_
            self.seasonal_[k] = self.gamma * (x - self.intercept_) + (1.0 - self.gamma) * self.seasonal_[k]
            error = abs(x - forecast)
            if np.isnan(self.deviation_[k]):
                self.deviation_[k] = error
            else:
                self.deviation_[k] = self.gamma * error + (1.0 - self.gamma) * self.deviation_[k]

        self._violations.append(violation)
        self._num_values += 1
        offset = int(SMOOTHING_WINDOW / 2 * self.period)
        if offset > 0 and self._num_values % self.period == 0:
            # The seasonal coefficients are all set after the first period, the deviations after the second.
            self.seasonal_ = _smooth(self.seasonal_, offset)
            if not np.isnan(self.deviation_).any():
                self.deviation_ = _smooth(self.deviation_, offset)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next value with the failure rule over the last ``window_size`` values.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value. Higher scores represent more anomalous values.

        Returns:
            float: The number of violations among the last ``window_size`` values, ``X`` included, if it is at least ``threshold``, else 0.0.
        """
        _, violation = self._forecast(check_univariate(X, "HoltWinters"))
        num_violations = sum(self._violations) + violation
        return float(num_violations) if num_violations >= self.threshold else 0.0
